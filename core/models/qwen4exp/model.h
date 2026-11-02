// qwen4exp on one or two cards: every layer split between the cards (docs/DESIGN.md, section 5; ADR-0023), the
// experts the cards do not cache computed by the CPU. Each card records its step once per window width, at load.
// The cards exchange partial sums twice per layer (ADR-0030); card 0 hands each layer's misses to the CPU through
// flags in host memory (ADR-0011). The host's work per step is the embedding, the n-gram rows, the misses and the
// sampling.
#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "api/model.h"
#include "cpu/experts.h"
#include "load/direct.h"
#include "load/gguf.h"
#include "models/qwen4exp/layout.h"
#include "models/qwen4exp/mrope.h"
#include "models/qwen4exp/ngram.h"
#include "models/qwen4exp/shape.h"
#include "quant.h"
#include "step.h"
#include "util/parallel.h"
#include "ze/library.h"
#include "ze/ze.h"

namespace engine::qwen4exp {

using engine::Candidate;
using engine::Image;
using engine::ImageShape;
using engine::kNoToken;
using engine::LoadInfo;
using engine::NgramStats;
using engine::Options;
using engine::StepIn;
using engine::StepStats;
using engine::TraceRow;

// The image and video tokens, and 16-pixel patches of 768 bytes merged 2 x 2 (docs/spec/MODEL_VISION.md, sections 2,
// 4.1 and 6).
constexpr ImageShape kImageShape{248056, 248057, 768, 2};
// The drafts a round makes by default: the draft layer's; DFlash 2 drafts a whole window in one pass.
constexpr uint32_t kDraftMaxLayer = 3, kDraftMaxCap = STEP_MAX_ROWS - 1;

class Model final : public engine::Model {
 public:
  Model(const std::string &path, const Options &opt);
  ~Model() override;
  Model(const Model &) = delete;
  Model &operator=(const Model &) = delete;

  // Zeroes the slot's recurrent state too.
  void reset_slot(uint32_t slot) override;

  // vision.cpp. The images set the positions of every cell from the first image on (docs/spec/MODEL_VISION.md,
  // section 4.2).
  void set_images(uint32_t slot, const Image *images, uint32_t n) override;
  // A cell's positions (time, height, width): (cell, cell, cell) for text before any image.
  std::array<uint32_t, 3> position(uint32_t slot, uint32_t cell) const;
  // The most tokens an image may have: 0 without Options::vision_path.
  uint32_t image_tokens_max() const override { return vmax_tokens_; }
  ImageShape image_shape() const override { return kImageShape; }
  // An image's rows of the text model's input (vision.cpp), image_tokens rows of 2,560 into out: what the prompt path
  // reads at the image's block. For tests; prompt() runs it itself.
  void encode_image(const Image &im, float *out);

  // One token of one sequence at position pos; p1 and p2 are the two tokens before it (kNoToken when absent). Its
  // KV and state go into the slot. Returns the largest logits, highest first.
  std::vector<Candidate> step(uint32_t slot, uint32_t pos, int32_t token, int32_t p1, int32_t p2);

  // With window, the recurrent state is what waits for commit; the KV is written in place.
  std::vector<std::vector<Candidate>> step(const StepIn *rows, uint32_t n, bool window = false,
                                           bool prompt = false) override;
  // Draft-layer passes run beside a commit's work on the cards.
  void commit(uint32_t slot, uint32_t pos0, uint32_t n) override;
  void commit_runs(const CommitRun *runs, uint32_t n) override;

  // The draft layer (mtp.cpp).
  bool drafting() const override { return mtp_on_ || df_on_; }
  std::vector<std::vector<Candidate>> mtp_catchup(const StepIn *rows, uint32_t n, const std::vector<uint32_t> &keep,
                                                  const uint32_t *from = nullptr) override;
  std::vector<Candidate> mtp_draft(uint32_t slot, uint32_t pos, int32_t token, uint32_t src_row) override;
  std::vector<std::vector<Candidate>> mtp_drafts(const StepIn *rows, uint32_t n, const uint32_t *src_rows) override;
  std::vector<std::vector<Candidate>> mtp_catchup_prompt(const StepIn *rows, uint32_t n, uint32_t end,
                                                         const std::vector<uint32_t> &keep) override;
  // The slot's first cell the draft layer attends to (earlier ones were never computed).
  void mtp_start(uint32_t slot, uint32_t first);

  // The DFlash 2 drafter (dflash.cpp), in place of the draft layer when Options::mtp_dir holds its files. Context
  // rows put the features of the decode step's rows into the slot's ring; a block's rows after its token are mask
  // tokens, and attend to the ring.
  bool dflash() const override { return df_on_; }
  std::vector<std::vector<std::vector<Candidate>>> df_pass(const StepIn *ctx, uint32_t nctx, const DraftBlock *blocks,
                                                           uint32_t nblocks) override;
  void df_context_prompt(const StepIn *rows, uint32_t n, uint32_t end) override;
  // mtp_start, or DFlash 2's.
  void draft_start(uint32_t slot, uint32_t first) override;
  uint32_t draft_max_default() const override { return df_on_ ? kDraftMaxCap : kDraftMaxLayer; }
  uint32_t draft_max_cap() const override { return kDraftMaxCap; }
  // For tests: card 0's features of the last decode step's row r (DF_FEAT bf16 values), and the last block pass's
  // final normed rows (STEP_MAX_ROWS x Q_EMBD).
  std::vector<uint16_t> df_features(uint32_t row);
  std::vector<float> df_hidden();
  // For tests: the last decode step's logits of row `row`, all the vocabulary's, from every card's share of the head.
  std::vector<float> logits(uint32_t row);

  // n prompt tokens of a sequence, from position pos0, in one pass (prompt.cpp); p1 and p2 are the two tokens
  // before pos0 (kNoToken when absent). n is at most prompt_chunk(). Returns the last token's largest logits. next:
  // the tokens of the chunk expected to follow (next_n of them, from pos0 + n), or null; their n-gram rows are read
  // while this chunk runs, and the next call takes them if its chunk is that one.
  std::vector<Candidate> prompt(uint32_t slot, uint32_t pos0, const int32_t *tokens, uint32_t n, int32_t p1,
                                int32_t p2, const int32_t *next = nullptr, uint32_t next_n = 0);
  // The experts are staged once for all the runs' rows. Each run's state and KV are as prompt gives them, bit for
  // bit. next and next_n as for prompt.
  std::vector<std::vector<Candidate>> prompt_runs(const PromptRun *runs, uint32_t nruns, const int32_t *next = nullptr,
                                                  uint32_t next_n = 0) override;
  uint32_t prompt_chunk() const override { return tmax_; }
  uint32_t fetch_rows() const override;
  // About a decode token's time over a prompt row's on this model's kernels. A constant for now; it should one day be
  // derived from the GGUF, since the time ratio follows the file's layout and quantization
  // (docs/measurements/MEASUREMENTS.md#2026-11-01-piece-rule-with-a-slow-split).
  uint32_t decode_weight() const override { return 30; }
  // With Options::trace: the last prompt chunk's (or decode step's) time on a card per kernel, largest first.
  std::vector<TraceRow> prompt_trace(uint32_t card) const;
  std::vector<TraceRow> step_trace(uint32_t card) const;
  std::vector<TraceRow> df_trace(uint32_t card) const;  // the last DFlash 2 block pass
  // With Options::profile_routing: rows routed to each expert, [layer][expert], by the prompts read so far.
  std::vector<uint32_t> routing_counts() const;

  // A slot's fixed state (state.cpp): the DeltaNet layers' and the PLE history.
  uint64_t slot_state_bytes() override;
  void save_slot(uint32_t slot, uint8_t *dst) override;
  void load_slot(uint32_t slot, const uint8_t *src) override;
  void *alloc_pinned(size_t bytes) override;
  void free_pinned(void *p) override;

  // wake loads the cards as the constructor did, with the experts cached at the sleep.
  void sleep() override;
  void wake() override;
  bool asleep() const { return asleep_; }

  // The pool (layout.h) is a multiple of the cards. At load, slot s's logical page i is page s pages_per_slot() + i
  // while that is in the pool, and page 0 past it. copy_cells copies every attention layer on every card (state.cpp).
  // Neither set_pages nor copy_cells waits for a window's commit, which reads no KV.
  uint32_t kv_pages() const override { return kv_pages_; }
  uint32_t pages_per_slot() const override { return npp_; }
  void set_pages(uint32_t slot, uint32_t first, const uint32_t *pages, uint32_t n) override;
  void copy_cells(uint32_t from, uint32_t to, uint32_t cells) override;
  // With Options::share_kv (kv_share.cpp) the expert cache may take the memory of the pool's pages from in_use up,
  // and kv_need takes it back when it must. Streamed (Options::kv_resident), the ring of resident K and V pages holds
  // at most `resident` pages (up to its size) once the steps have claimed them; kvs_drop is for an idle slot.
  void kv_need(uint32_t in_use, uint32_t resident = UINT32_MAX) override;
  void kv_ahead() override;
  void kvs_drop(uint32_t slot, uint32_t pages) override;

  // Adaptive swaps (swap.cpp): a round makes the clearest swaps first.
  void note_prompt_routing() override;
  uint32_t swap_round(uint32_t max_swaps = 64, bool decay = true) override;
  // The swap and move threads are done.
  bool swaps_ready() const override { return swap_busy_.load() == 0; }
  // No swaps, refills or evictions. After swaps_ready.
  bool swaps_settled() const override { return swap_next_.empty() && evict_next_.empty(); }
  uint64_t swaps() const { return swaps_; }

  const LoadInfo &info() const override { return info_; }
  const StepStats &last_stats() const override { return stats_; }
  // With Options::share_kv the cache gives blocks to the KV pool and takes them back (kv_share.cpp). On the engine
  // thread.
  uint32_t experts_cached() const override;
  uint32_t experts_floor() const override { return nslots_cached_; }
  // ngram.h. On the engine thread.
  NgramStats ngram_stats() const override { return ngram_ ? ngram_->stats() : NgramStats{}; }
  const Options &options() const { return opt_; }

 public:
  struct Card;  // the model's state on one card (model_impl.h)
  static uint32_t ncards_of(const Card &card);

 private:
  struct Mat {
    uint32_t type = 0, rows = 0, K = 0, bpr = 0;
    uint8_t *p = nullptr;
  };
  struct Layer;
  struct HostLayer;
  struct MtpCard;
  struct DfCard;
  struct PromptCard;
  struct VisionCard;

  void wait_commit();  // until the last commit has run
  // rope_[slot] for a step whose first row of slot is at cell; check: refuse a cell of an image.
  void set_rope(uint32_t slot, uint32_t cell, bool check);
  void load_cards(bool first);  // everything on the cards; first: also the arena, at the constructor
  void load_card(Card &card);
  // A tensor's bytes for the load, read with O_DIRECT (load/direct.h), never through the mapping; held until
  // unstage().
  const GgufTensor &src(const std::string &name);
  void unstage() { staged_.clear(); }
  void load_experts(bool first);
  std::unique_ptr<HostLayer> host_layer(uint32_t l, uint64_t n, const std::vector<uint64_t> &profile,
                                        uint64_t stg_bytes, uint32_t positions);
  void record(Card &card, uint32_t width, bool kvs = true);  // kvs: with the streamed pool's claim
  void record_head(ze::List &L, Card &card, const StepPlan *plan, uint32_t width);
  void record_commit(Card &card);
  uint64_t mtp_bytes() const;
  void load_mtp(Card &card);
  void record_mtp(Card &card, uint32_t width, uint32_t rest, bool chain);
  std::vector<std::vector<Candidate>> run_mtp(const StepIn *rows, uint32_t n, uint32_t nkeep, bool chain,
                                              uint32_t src_rows);
  std::vector<std::vector<Candidate>> catchup(const StepIn *rows, uint32_t n, const std::vector<uint32_t> &keep,
                                              const uint32_t *from);
  bool init_dflash();  // DFlash 2's host buffers, when Options::mtp_dir holds its files
  void load_dflash(Card &card);
  void record_df(Card &card, uint32_t width, bool block);
  void load_prompt(Card &card);
  // The image encoder: first, its host side (the weights' stages, pixels, encoding); always, each card's side.
  void load_vision(bool first);
  void vision_slots();
  // Each image block of the slot's prompt that meets cells pos0 .. pos0 + n - 1 into pembed_, from row row0
  // (vision.cpp).
  void place_images(uint32_t slot, uint32_t pos0, uint32_t n, uint32_t row0);
  void record_prompt_copy(Card &card);
  using Staging = std::vector<std::vector<ExpertWeights>>;  // each layer's staged experts' matrices, by position
  Staging staging_now() const;
  Staging staging_after() const;  // after the chosen round's refills and planned eviction
  std::unique_ptr<ze::List> prompt_copies(Card &card, const Staging &st, uint32_t &copies);
  // A short chunk's staging on card c, on its own thread: each layer's uncached experts in order (hottest first)
  // until the router's counts are in, then the rest it routes to.
  void fetch_experts(uint32_t c, const std::vector<std::vector<uint32_t>> &order);
  ze::List &prompt_list(Card &card, uint32_t T, uint32_t runs);
  uint32_t pev(uint32_t kind, uint32_t layer, uint32_t c) const;
  Mat mat(Card &card, const GgufTensor &t, const std::vector<std::pair<uint64_t, uint64_t>> &rows,
          const std::vector<std::pair<uint64_t, uint64_t>> &cols = {});
  template <class T>
  T *upload(Card &card, const void *src, size_t bytes);
  void matvec(ze::List &list, Card &card, const Mat &m, const uint8_t *act, float *y, uint32_t ntok,
              ze_event_handle_t signal = nullptr);
  ze_kernel_handle_t kernel(uint32_t card, const std::string &module, const std::string &name);
  uint32_t ev(uint32_t layer, uint32_t which) const;
  void note_routing(uint32_t layer, const int32_t *ids, uint32_t n, float weight);
  struct SwapPlan {
    uint32_t l, in, out;  // out kNoExpert: in fills the empty slot `slot` (a refill, kv_share.cpp)
    uint8_t *piece;  // a spare piece the swap thread fills with the outgoing expert
    int32_t slot = -1;
  };
  static constexpr uint32_t kNoExpert = UINT32_MAX;
  void swap_maps(const SwapPlan &w);
  uint32_t apply_round();  // the last round's choice; returns the swaps and refills made
  void start_round(std::vector<float> use = {}, uint32_t max_swaps = 0);  // a thread readies the choice behind the steps
  struct SwapMove {
    uint32_t l, e;  // an outgoing expert, copied from the piece the swap thread filled into in's arena piece
    uint8_t *to, *from;
  };
  void choose_swaps(const std::vector<float> &use, uint32_t max_swaps);
  void settle_moves();
  // Sharing the cards' memory between the KV pool and the expert cache (kv_share.cpp).
  struct Region {  // a reserved range of addresses: its first `fixed` bytes backed for good, then blocks as needed
    uint8_t *va = nullptr;
    uint64_t fixed = 0;
    std::vector<ze_physical_mem_handle_t> blocks;
  };
  struct Evict {  // a layer's part in giving cache memory to the KV
    uint32_t l, take;              // the layer, and the blocks of each card it gives up
    uint32_t first;                // the staging position of the first victim
    std::vector<uint32_t> victims; // the experts that leave the cards
  };
  uint8_t *kv_region(Card &card, uint32_t x, uint64_t unit, uint32_t div, uint32_t cap = 0, uint32_t min_units = 0);
  uint32_t kv_pages_of(uint32_t x, size_t blocks) const;
  size_t kv_blocks_for(uint32_t x, uint32_t pages) const;
  uint32_t kv_level(uint32_t x, uint32_t pool, uint32_t ring) const { return kv_ring_[x] ? ring : pool; }
  void ring_resize(uint32_t R);
  void ring_sync();
  void own_levels();
  uint32_t cache_slots(uint32_t l, size_t blocks) const;
  void share_cards(bool first, uint64_t budget, const std::vector<uint64_t> &stg_bytes, std::vector<uint32_t> &count);
  void map_cache(Card &card, uint32_t l, size_t blocks);
  void grow_kv(uint32_t pages, uint32_t ring);
  void shrink_kv(uint32_t pages, uint32_t ring);
  void plan_evict(uint32_t pages, uint32_t ring, bool beside);
  void fill_evict(Pool &pool);
  void apply_evict(std::chrono::steady_clock::time_point t0, bool beside);
  // The blocks' maps and unmaps that map_cache, grow_kv and shrink_kv queued: made now, or on a thread beside the
  // next steps when no list reads the memory they touch. finish_remap waits for that thread; reap_remap only if
  // it is done.
  struct VmOp {
    uint8_t *va;
    ze_physical_mem_handle_t block;  // null: unmap the block at va
  };
  void run_vm(bool beside);
  void finish_remap();
  void reap_remap();
  void plan_refills(uint32_t max);
  struct PieceMove {
    uint8_t *to;
    const uint8_t *from;
    uint64_t bytes;
  };
  void refill_maps(const SwapPlan &w, std::vector<PieceMove> &moves);
  void install_copies();
  void log_share(const char *why, std::chrono::steady_clock::time_point t0 = {});
  void fill_piece(uint32_t l, uint32_t e, uint8_t *piece);  // expert e's matrices from the file, as the arena holds them
  void release_expert(uint32_t l, uint32_t e);               // its pages of the file back to the host
  uint8_t *spare_piece(uint64_t need);
  uint8_t *arena_piece(uint64_t need);
  double measure_host_read();
  unsigned loader_threads();
  double measure_copy(uint32_t card);
  void settle_arena();
  uint32_t ngram_ev() const;  // the event the host signals when a decode step's n-gram rows are in
  uint32_t idx_ev(uint32_t attn, uint32_t c) const;  // card c's indexer candidates of attention layer attn are out

  Options opt_;
  GgufModel gguf_;
  Shape shape_;  // the file's sizes; any layout.h still holds are equal to it once the constructor has checked them
  std::string layer_module_, prompt_module_;  // the kernels built for shape_.experts
  std::unique_ptr<ze::Runtime> rt_;
  std::unique_ptr<ze::Library> lib_;
  std::unique_ptr<ze::EventPool> events_;
  std::unique_ptr<Pool> pool_;
  std::unique_ptr<CpuExperts> cpu_;
  std::unique_ptr<Ngram> ngram_;
  std::vector<std::unique_ptr<Card>> cards_;
  std::vector<std::unique_ptr<HostLayer>> host_layers_;
  uint32_t ncards_ = 1;
  uint32_t npp_ = 0;       // pages per slot's row of the table: the context's, rounded up to a multiple of the cards
  uint32_t kv_pages_ = 0;  // pages in the pool
  uint32_t kv_res_ = 0;    // streamed (Options::kv_resident): the ring of resident K and V pages a layer on each
                           // card, at most; 0: every page resident
  std::vector<uint8_t *> kv_host_;  // the host copies of K and V, [card][attention layer][K, V]; kept across sleeps
  std::vector<uint32_t> kvs_to_;    // [slot] the position its pages are claimed through (kvs_claim), 0: not settled
  uint32_t ring_level_ = 0;         // the ring's size on the cards (kst[4])
  uint32_t nslots_cached_ = 0;
  bool asleep_ = false;
  struct StagedTensor {
    GgufTensor t;
    AlignedBuffer buf;
  };
  std::unique_ptr<DirectReader> reader_;  // during a load
  std::unordered_map<std::string, StagedTensor> staged_;
  LoadInfo info_;
  StepStats stats_;

  // Host memory every card reads or writes.
  StepPlan *plan_ = nullptr;
  StepPlan *head_plan_ = nullptr;  // one row: the prompt path's head
  CommitPlan *cplan_ = nullptr;    // a window's commit
  SlotRope *rope_ = nullptr;       // [slot] the decode step's rotation (layout.h)
  std::vector<std::vector<Image>> images_;        // [slot]
  std::vector<std::vector<ImageBlock>> blocks_;  // [slot] their blocks
  bool commit_pending_ = false;    // the last commit was executed and not yet waited for
  // The draft layer's plan, inputs, exchanges, events and candidates.
  bool mtp_on_ = false;
  uint32_t mtp_type_ = QT_IQ4_NL;  // the draft layer's experts' type
  StepPlan *mplan_ = nullptr, *mrest_ = nullptr;  // a pass's rows, and the rows that run the whole layer
  float *membed_ = nullptr, *mex1_ = nullptr, *mex2_ = nullptr;
  uint32_t *mtop_idx_ = nullptr, *mtop_n_ = nullptr;
  float *mtop_val_ = nullptr;
  std::unique_ptr<ze::EventPool> mevents_;
  // DFlash 2's plans, block embeddings, exchange events and candidates (dflash.cpp).
  bool df_on_ = false;
  DraftPlan *dfcplan_ = nullptr, *dfbplan_ = nullptr;
  float *dfembed_ = nullptr;
  float *dfex_ = nullptr;  // the exchanges' partial sums, [2 DF_LAYERS][card][STEP_MAX_ROWS][Q_EMBD]
  uint32_t *dtop_idx_ = nullptr, *dtop_n_ = nullptr;
  float *dtop_val_ = nullptr;
  std::unique_ptr<ze::EventPool> devents_;
  uint16_t *pfeat_ = nullptr;      // [run][STEP_MAX_ROWS][DF_FEAT]: each prompt run's last rows' features
  std::vector<uint32_t> dffirst_;  // [slot] the first position its ring holds
  float *embed_ = nullptr;   // the token's embedding row
  float *ngram_u_ = nullptr; // its n-gram rows
  float *ex1_ = nullptr, *ex2_ = nullptr;  // [layer][card][2560] partial sums
  MissJob *jobs_ = nullptr;      // [layer] card 0's record of the misses for the CPU (ADR-0011)
  uint32_t *ans_seq_ = nullptr;  // [layer], 64 bytes apart: the step the CPU answered last
  uint32_t *stuck_ = nullptr;    // card 0 gave up waiting for an answer
  float *cpu_y_ = nullptr;   // [layer][2560] the CPU's sum for the misses
  uint32_t *top_idx_ = nullptr;  // [card][candidates]
  float *top_val_ = nullptr;
  uint32_t *top_n_ = nullptr;
  uint32_t *icand_ = nullptr;  // [attention layer][card][row][kIdxCand]: the indexer's candidates
  std::vector<float> use_;  // [layer][expert] decayed routing counts, with Options::adapt_swaps
  uint64_t swaps_ = 0;
  std::vector<SwapPlan> swap_next_;  // chosen by the last round, applied by the next
  std::map<uint64_t, std::vector<uint8_t *>> spares_;  // spare arena pieces by size
  std::thread swap_bg_;
  std::atomic<int> swap_busy_{0};  // swap_bg_ and move_bg_ while they run
  std::vector<SwapMove> moves_;  // the last round's, until settle_moves
  std::thread move_bg_;
  std::unique_ptr<Pool> swap_pool_;
  // Sharing (kv_share.cpp). Each card holds the same regions and block counts: Card::kv, 4 a + k for attention
  // layer a's K, V, raw keys and pooled keys, and Card::cache per layer.
  bool share_ = false;
  uint64_t blk_ = 0;              // a block's bytes
  uint32_t kv_own_ = 0;           // pages [0, kv_own_) backed on every card once the queued maps are made
  uint32_t kv_mapped_ = 0;        // pages [0, kv_mapped_) backed on every card now
  uint32_t kv_want_ = 0;          // the pages the last plan had in use
  uint32_t kv_margin_ = 0;        // pages backed ahead of use
  // The ring regions (streamed: K and V) are counted in ring pages, as the pool regions in pool pages.
  uint32_t ring_own_ = 0, ring_mapped_ = 0, ring_want_ = 0;
  std::chrono::steady_clock::time_point ring_over_{};  // since when the ring holds more than wanted (swap_round)
  static constexpr std::chrono::seconds kRingIdle{300};
  std::vector<uint64_t> kv_unit_; // per KV region: bytes a page, or a local page (kv_div_ = the cards)
  std::vector<uint32_t> kv_div_;
  std::vector<uint32_t> kv_cap_;  // per KV region: its pages at most (the pool's, or the ring's)
  std::vector<bool> kv_ring_;     // per KV region: a ring region
  std::vector<Evict> evict_next_;
  uint32_t evict_to_ = 0;         // the pages evict_next_ backs
  uint32_t ring_to_ = 0;          // and the ring pages
  std::vector<VmOp> vm_ops_;      // queued, in order
  std::thread remap_bg_;          // making a batch of them beside the steps
  std::atomic<bool> remap_end_{false};
  std::string remap_err_;
  uint32_t remap_to_ = 0;         // kv_own_ when the batch began: kv_mapped_ once it ends
  uint32_t ring_remap_to_ = 0;    // and ring_own_
  uint32_t copy_count_ = 0;  // the copies of the prompt path's staging list
  std::vector<std::unique_ptr<ze::List>> next_copies_;  // a card each: recorded behind a round, for the staging after it
  uint32_t next_copy_count_ = 0;
  std::vector<std::unique_ptr<ze::List>> retired_;  // copy lists no longer used, freed behind the next round
  bool copies_dirty_ = false;     // the prompt path's copy list no longer matches the uncached experts
  uint8_t *swap_buf_ = nullptr;  // pinned host memory: swap_next_'s halves, card-minor
  uint64_t swap_buf_bytes_ = 0;

  // The prompt path (prompt.cpp): the chunk's plan, embeddings and n-gram rows, its events.
  static constexpr uint64_t kArenaPiece = 1ull << 30;
  uint8_t *arena_at_ = nullptr;  // the arena's current piece, and what is left of it
  uint64_t arena_left_ = 0;
  struct AnonPiece {
    void *map;         // the mapping, unmapped at the end
    size_t map_bytes;
    uint8_t *at;       // the piece: 2 MiB aligned, kArenaPiece bytes
    uint64_t used;     // bytes of it arena_piece has handed out
  };
  std::vector<AnonPiece> anon_pieces_;  // Options::anon_arena
  int arena_lock_errno_ = 0;            // why mlock2 failed for a piece; 0: every piece is locked
  uint32_t tmax_ = 0;
  uint32_t last_width_ = 1;  // the width of the last decode step
  PromptPlan *pplan_ = nullptr;  // the chunk's plan, then its runs' (PROMPT_MAX_RUNS, n 0 when unused)
  uint32_t *prows_ = nullptr;  // each card's count of the chunk's rows when the hyper-connections split them
  uint32_t *ppos_ = nullptr;  // the runs' positions (step.h), ppos_run() values apart
  uint32_t ppos_run() const { return (tmax_ + Q_IDX_BLOCK) * 4; }
  float *pembed_ = nullptr, *pngram_ = nullptr;
  uint16_t *pngram16_ = nullptr;  // the n-gram rows as bf16, for the cards
  // The next chunk's n-gram rows, read during this chunk (prompt's next), and which chunk they are for.
  struct NgramAhead {
    bool valid = false;
    uint32_t slot = 0, pos0 = 0;
    int32_t p1 = 0, p2 = 0;
    std::vector<int32_t> tokens;
    std::vector<float> rows;
  } ahead_;
  std::unique_ptr<ze::EventPool> pevents_;

  // The image encoder (vision.cpp).
  std::unique_ptr<GgufModel> vgguf_;
  uint32_t vmax_tokens_ = 0;
  uint8_t *vw_ = nullptr;  // pinned host memory: the layers' and the merger's 16-bit weights, stage by stage
  std::vector<uint64_t> vstage_off_, vstage_len_;
  uint64_t vstage_bytes_ = 0;  // the largest stage
  uint8_t *vpix_ = nullptr;    // pinned host memory: an image's pixels on their way to the cards
  uint32_t *vcount_ = nullptr; // host memory: per card, its rows and its windows (the products' ntok)
  float *venc_ = nullptr;      // host memory: the last image encoded, its rows of 2,560
  // And whose: the slot, its images' generation and the image's index. set_images gives a slot's images a new
  // generation (from vgen_), so an encoding is never taken for another prompt's image.
  int64_t venc_slot_ = -1;
  uint64_t venc_gen_ = 0;
  size_t venc_index_ = 0;
  uint64_t vgen_ = 0;
  std::vector<uint64_t> slot_gen_;  // [slot]
  std::unique_ptr<ze::EventPool> vevents_;
};

}  // namespace engine::qwen4exp
