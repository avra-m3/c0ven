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

#include "cpu/experts.h"
#include "load/direct.h"
#include "load/gguf.h"
#include "mrope.h"
#include "ngram.h"
#include "layout.h"
#include "quant.h"
#include "step.h"
#include "util/parallel.h"
#include "ze/library.h"
#include "ze/ze.h"

namespace engine::qwen4exp {

struct Options {
  uint32_t slots = 1;                       // sequences resident at once
  uint32_t max_ctx = 32768;                 // positions per sequence, a multiple of 4
  uint32_t kv_pages = 0;                    // the KV cache's pool, in pages of 4 cells (layout.h); 0: every slot's
                                            // whole context
  uint32_t kv_resident = 0;                 // K and V cells per slot kept in VRAM, the rest in host memory (ADR-0008
                                            // item 3, kv_share.cpp's note); 0: every cell
  uint32_t max_cards = 2;                   // 1 runs the whole model on card 0
  uint64_t reserve_bytes = 512ull << 20;    // device memory left free on each card, beyond what the engine counts
  int32_t experts_per_layer = -1;           // experts cached per layer; -1: as many as fit
  unsigned cpu_threads = 16;                // the CPU's expert workers, with the host thread
  uint32_t candidates = 64;                 // logits returned per step, the largest
  uint32_t prompt_chunk = 4352;             // tokens per prompt pass at most, a multiple of 64; 0: one token per
                                            // step
  bool trace = false;                       // the lists time every kernel (prompt_trace, step_trace, df_trace)
  bool profile_routing = false;             // the prompt path counts each expert's rows (routing_counts)
  bool adapt_swaps = false;                 // count the routing, for swap_round (swap.cpp)
  // The pages of the KV pool not in use lend their memory to the expert cache (kv_share.cpp); needs adapt_swaps.
  bool share_kv = false;
  // The arena in the process's own memory, which the copy engines read in place (ADR-0035). false, or a card without
  // shared system USM: the driver's buffers, which xe moves out of GTT under reclaim and so stalls the CPU's experts
  // (docs/measurements/MEASUREMENTS.md#2026-10-17-decode-stalls-eviction). options() gives the value the engine took.
  bool anon_arena = true;
  bool ngram_in_ram = false;                // the whole n-gram table in RAM (ngram.h)
  // A profile to choose the cached experts by (tools/route_profile.cpp); empty: the first experts of each layer.
  std::string routing_profile = ENGINE_ROUTING_PROFILE;
  // The draft (MTP) layer's tensors (docs/MODEL_FILES.md, "The MTP layer"); empty: no drafting.
  std::string mtp_dir;
  // The image encoder's file (docs/spec/MODEL_VISION.md, section 1); empty: no images. Needs prompt_chunk.
  std::string vision_path;
};

// One kernel's share of a traced list's last run.
struct TraceRow {
  std::string kernel;
  uint32_t launches = 0;
  double ms = 0;  // from the end of the launch before it, so waits on events count to the kernel after them
};

struct Candidate {
  int32_t token;
  float logit;
};

struct LoadInfo {
  uint32_t cards = 0;
  uint32_t experts_per_layer = 0;
  uint32_t vocab = 0;
  uint64_t device_bytes[2] = {0, 0};
  uint64_t arena_bytes = 0;  // host memory holding the experts not on the cards
  bool arena_anon = false;   // the arena is the process's own memory (Options::anon_arena)
  double seconds = 0;
  double host_read_bytes_per_s = 0;  // the CPU workers' read of the arena at the first load; 0: not measured
  unsigned loader_threads = 0;       // the loader's threads, from DirectReader::probe_threads
  double loader_bytes_per_s = 0;     // their read rate in the probe
  bool ngram_in_ram = false;
  double ngram_read_us = 0;          // one 4 KiB row read from the file, when the table is not in RAM
  double copy_bytes_per_s[2] = {0, 0};  // each card's copy engine reading pinned host memory
};

// One row of a decode step: a token of a sequence at its position; p1 and p2 are the two tokens before it (kNoToken
// when absent).
struct StepIn {
  uint32_t slot = 0, pos = 0;
  int32_t token = 0, p1 = -1, p2 = -1;
};

// An image in a slot's prompt (docs/spec/MODEL_VISION.md): its block of h w / 4 placeholder tokens starts at prompt
// index first; grid_h and grid_w are its size in patches (both even). frames is 1 for a still image, 2 for a pair of
// video frames (section 6). pixels holds its grid_h x grid_w patches in the encoder's order (section 2.2), each
// frames x 768 bytes of 8-bit RGB, frame, channel, row, column; null when only the positions are wanted.
struct Image {
  uint32_t first = 0;
  uint32_t frames = 1, grid_h = 0, grid_w = 0;
  const uint8_t *pixels = nullptr;
};

struct StepStats {
  uint32_t misses = 0;          // routed experts the CPU computed
  uint32_t layers_missed = 0;   // layers with at least one miss
};

class Model {
 public:
  Model(const std::string &path, const Options &opt);
  ~Model();
  Model(const Model &) = delete;
  Model &operator=(const Model &) = delete;

  // Zeroes a slot's recurrent state, before a new sequence, and forgets its images.
  void reset_slot(uint32_t slot);

  // The images of the prompt a slot reads (vision.cpp), in order, their blocks apart; they set the positions of every
  // cell from the first image on (docs/spec/MODEL_VISION.md, section 4.2). The pixels are read when the first prompt
  // chunk that holds part of an image's block runs, so they must live until then.
  void set_images(uint32_t slot, const Image *images, uint32_t n);
  // A cell's positions (time, height, width): (cell, cell, cell) for text before any image.
  std::array<uint32_t, 3> position(uint32_t slot, uint32_t cell) const;
  // The most tokens an image may have: 0 without Options::vision_path.
  uint32_t image_tokens_max() const { return vmax_tokens_; }
  // An image's rows of the text model's input (vision.cpp), image_tokens rows of 2,560 into out: what the prompt path
  // reads at the image's block. For tests; prompt() runs it itself.
  void encode_image(const Image &im, float *out);

  // One token of one sequence at position pos; p1 and p2 are the two tokens before it (kNoToken when absent). Its
  // KV and state go into the slot. Returns the largest logits, highest first.
  std::vector<Candidate> step(uint32_t slot, uint32_t pos, int32_t token, int32_t p1, int32_t p2);

  // One step of n rows (1 to STEP_MAX_ROWS): rows of several sequences, or consecutive positions of one, in order.
  // Returns each row's largest logits, highest first. With window, the rows are runs of consecutive positions, one run
  // a sequence, and their recurrent state waits for commit, which keeps each run's first rows' (the accepted drafts
  // and the row before them) and forgets the rest. Their KV is overwritten when their positions come again. With
  // prompt, the rows are prompt tokens, counted for adaptive swaps as the prompt path counts them.
  std::vector<std::vector<Candidate>> step(const StepIn *rows, uint32_t n, bool window = false, bool prompt = false);
  // Returns before the cards finish; the next call on a slot's state waits for it (draft-layer passes run beside it).
  // commit: one run, the window's first n rows. commit_runs: n runs of the window, each its own sequence's.
  void commit(uint32_t slot, uint32_t pos0, uint32_t n);
  void commit_runs(const CommitRun *runs, uint32_t n);

  // The draft layer (mtp.cpp). mtp_catchup: rows r = (slot, position i, the token at i + 1), the hidden input the
  // main model's final residual of the last step's row r (row from[r] with from). Every row writes its cell; only the
  // rows in keep run the rest of the layer, and it returns their draft candidates (for i + 2) in keep's order. A draft
  // from keep[j] reads src_row j. mtp_draft: one more draft for such a row, the hidden input the draft layer's own
  // residual of row src_row of its last pass (whose cell was i - 1).
  bool drafting() const { return mtp_on_ || df_on_; }
  std::vector<std::vector<Candidate>> mtp_catchup(const StepIn *rows, uint32_t n, const std::vector<uint32_t> &keep,
                                                  const uint32_t *from = nullptr);
  std::vector<Candidate> mtp_draft(uint32_t slot, uint32_t pos, int32_t token, uint32_t src_row);
  // mtp_drafts: mtp_draft for rows of distinct sequences in one pass, row r's input row src_rows[r] of the last pass.
  std::vector<std::vector<Candidate>> mtp_drafts(const StepIn *rows, uint32_t n, const uint32_t *src_rows);
  // mtp_catchup_prompt: mtp_catchup over the n rows before row `end` of the last prompt chunk (a run's last rows),
  // from the prompt path's residuals. mtp_start: the slot's first cell the draft layer attends to (earlier ones were
  // never computed).
  std::vector<std::vector<Candidate>> mtp_catchup_prompt(const StepIn *rows, uint32_t n, uint32_t end,
                                                         const std::vector<uint32_t> &keep);
  void mtp_start(uint32_t slot, uint32_t first);

  // The DFlash 2 drafter (dflash.cpp), in place of the draft layer when Options::mtp_dir holds its files. A pass takes
  // context rows first: row r = (slot, position), with the features of the last decode step's row r, into the slot's
  // ring. Then each block: n rows from position pos of a slot (its token there, then mask tokens) that attend to the
  // slot's ring from the position draft_start set. Returns each block's candidates for positions pos + 1 .. pos + n - 1.
  // Blocks of distinct slots, at most STEP_MAX_ROWS rows in all; a pass may have no context rows or no blocks.
  struct DraftBlock {
    uint32_t slot, pos;
    int32_t token;
    uint32_t n;
  };
  bool dflash() const { return df_on_; }
  std::vector<std::vector<std::vector<Candidate>>> df_pass(const StepIn *ctx, uint32_t nctx, const DraftBlock *blocks,
                                                           uint32_t nblocks);
  // df_pass's context rows only, from the n rows before row end of the last prompt chunk (a run's last rows).
  void df_context_prompt(const StepIn *rows, uint32_t n, uint32_t end);
  // The slot's first position the drafter reads (earlier ones were never computed): mtp_start, or DFlash 2's.
  void draft_start(uint32_t slot, uint32_t first);
  // For tests: card 0's features of the last decode step's row r (DF_FEAT bf16 values), and the last block pass's
  // final normed rows (STEP_MAX_ROWS x Q_EMBD).
  std::vector<uint16_t> df_features(uint32_t row);
  std::vector<float> df_hidden();
  // For tests: the last decode step's logits of row `row`, all Q_VOCAB of them, from every card's share of the head.
  std::vector<float> logits(uint32_t row);

  // n prompt tokens of a sequence, from position pos0, in one pass (prompt.cpp); p1 and p2 are the two tokens
  // before pos0 (kNoToken when absent). n is at most prompt_chunk(). Returns the last token's largest logits. next:
  // the tokens of the chunk expected to follow (next_n of them, from pos0 + n), or null; their n-gram rows are read
  // while this chunk runs, and the next call takes them if its chunk is that one.
  std::vector<Candidate> prompt(uint32_t slot, uint32_t pos0, const int32_t *tokens, uint32_t n, int32_t p1,
                                int32_t p2, const int32_t *next = nullptr, uint32_t next_n = 0);
  // One sequence's run of a prompt chunk: n tokens from position pos0, p1 and p2 as for prompt.
  struct PromptRun {
    uint32_t slot, pos0;
    const int32_t *tokens;
    uint32_t n;
    int32_t p1, p2;
  };
  // Runs of 1 to PROMPT_MAX_RUNS sequences, each of them once, in one pass: the experts are staged once for all
  // their rows, at most prompt_chunk() of them. Each run's state and KV are as prompt gives them, bit for bit. Returns
  // each run's last token's largest logits. next and next_n as for prompt, with one run.
  std::vector<std::vector<Candidate>> prompt_runs(const PromptRun *runs, uint32_t nruns, const int32_t *next = nullptr,
                                                  uint32_t next_n = 0);
  uint32_t prompt_chunk() const { return tmax_; }
  // With Options::trace: the last prompt chunk's (or decode step's) time on a card per kernel, largest first.
  std::vector<TraceRow> prompt_trace(uint32_t card) const;
  std::vector<TraceRow> step_trace(uint32_t card) const;
  std::vector<TraceRow> df_trace(uint32_t card) const;  // the last DFlash 2 block pass
  // With Options::profile_routing: rows routed to each expert, [layer][expert], by the prompts read so far.
  std::vector<uint32_t> routing_counts() const;

  // A slot's fixed state as one block (state.cpp): the DeltaNet layers' and the PLE history, not the KV, which stays in
  // its pages. Its size, and copies to and from host memory (the block must be host memory every card reaches:
  // alloc_pinned).
  uint64_t slot_state_bytes();
  void save_slot(uint32_t slot, uint8_t *dst);
  void load_slot(uint32_t slot, const uint8_t *src);
  void *alloc_pinned(size_t bytes);
  void free_pinned(void *p);

  // Sleep (ADR-0032): sleep frees all the model holds on the cards and keeps the runtime, kernels, host buffers and
  // arena. wake loads the cards as the constructor did, with the experts cached at the sleep, and throws, leaving
  // the model asleep, when they no longer fit. Slots are empty after a wake. Neither runs during a step or a prompt
  // chunk, and while asleep no other call may run.
  void sleep();
  void wake();
  bool asleep() const { return asleep_; }

  // The KV cache's pages (layout.h). The pool holds kv_pages() pages, a multiple of the cards; a slot's row of the page
  // table has pages_per_slot() entries. At load, slot s's logical page i is page s pages_per_slot() + i while that is
  // in the pool, and page 0 past it. set_pages maps n logical pages of a slot from first; copy_cells copies the first
  // cells cells (fewer than a page) of page from into page to, every attention layer on every card (state.cpp).
  // Between steps: neither waits for a window's commit, which reads no KV.
  uint32_t kv_pages() const { return kv_pages_; }
  uint32_t pages_per_slot() const { return npp_; }
  void set_pages(uint32_t slot, uint32_t first, const uint32_t *pages, uint32_t n);
  void copy_cells(uint32_t from, uint32_t to, uint32_t cells);
  // With Options::share_kv (kv_share.cpp), the pool's pages from in_use up hold nothing to read again, and the
  // expert cache may take their memory. kv_need backs every page below in_use before it returns, taking memory back
  // from the cache if it must; kv_ahead chooses what the cache gives up for the next pages, behind the steps. Call
  // both between steps. Streamed (Options::kv_resident), the ring of resident K and V pages is sized the same way:
  // resident is the pages the slots in use read, and the ring holds at most that many (up to its size) once the
  // steps have claimed them. kvs_drop takes a slot's first `pages` logical pages out of the ring: an idle slot's,
  // so they no longer count. Between steps.
  void kv_need(uint32_t in_use, uint32_t resident = UINT32_MAX);
  void kv_ahead();
  void kvs_drop(uint32_t slot, uint32_t pages);

  // Adaptive swaps (swap.cpp): adds the last prompt chunk's routing to the counts, and makes up to max_swaps swaps
  // where the counts say so, the clearest first. Returns the swaps made. Call between steps.
  void note_prompt_routing();
  uint32_t swap_round(uint32_t max_swaps = 64);
  uint64_t swaps() const { return swaps_; }

  const LoadInfo &info() const { return info_; }
  const StepStats &last_stats() const { return stats_; }
  // Experts cached per layer now, the mean over the layers as LoadInfo counts it: with Options::share_kv the cache
  // gives blocks to the KV pool and takes them back. Between steps, on the engine thread. experts_floor is the
  // count the load keeps room for whatever the pool holds (kv_share.cpp).
  uint32_t experts_cached() const;
  uint32_t experts_floor() const { return nslots_cached_; }
  // The n-gram table's counters (ngram.h); zeros without the table. Between steps, on the engine thread.
  NgramStats ngram_stats() const { return ngram_ ? ngram_->stats() : NgramStats{}; }
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
  SlotRope *rope_ = nullptr;       // [slot] the decode step's rotation (step.h)
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
