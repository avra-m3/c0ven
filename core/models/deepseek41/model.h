// deepseek41's decode step on two cards (ADR-0074): each card runs one recorded list a step, the two exchange their
// partial sums through each other's memory, and the CPU answers the routed experts the cards do not hold. Prompts are
// read as decode steps; the model reads no images. With Options::mtp_dir, the path of the file's drafter (a second
// GGUF), the cards also hold the drafter: its layers follow the model's in every card's list of layers, and all of
// its experts are on the cards.
#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <string>
#include <vector>

#include "api/model.h"
#include "load/direct.h"
#include "load/gguf.h"
#include "load/runs.h"
#include "models/deepseek41/engram.h"
#include "models/deepseek41/layout.h"
#include "models/deepseek41/tiers.h"
#include "ze/library.h"

namespace engine::deepseek41 {

// The step as a card reads it, copied from host memory when its list starts; the copy zeroes stuck and start.
struct StepWords {
  uint32_t real;  // rows that hold a token; the list's other rows are padded
  uint32_t step;
  uint32_t stuck;  // a wait gave up (ADR-0074 item 5)
  uint32_t pad;
  uint64_t start;  // ds_wait's clock at the list's first polling wait
  uint64_t pad2;
  D41Row rows[STEP_MAX_ROWS];
};

// Where the host thread's time in the last step went, ms.
struct StepTimes {
  double before = 0;   // the embeddings, before the lists start
  double engram = 0;   // gathering the Engram rows, while the lists run
  double records = 0;  // waiting for the layers' routing records: the cards' work up to each router
  double answers = 0;  // the CPU's experts
  double after = 0;    // from the last answer to the lists' end
  std::array<float, D41_LAYERS> layer_records{}, layer_answers{};  // records and answers by layer
};

// What the tests ask for beyond Options.
struct Debug {
  bool probes = false;  // card 0 copies the streams out after every sublayer, and what a step stores for the drafter
};

// One draft of a block: the token chosen, the drafter's own chance, 0 to 1, that the model keeps it, and the candidates
// it was chosen from, largest logit first, the bias of the token before it added.
struct CardDraft {
  int32_t token = 0;
  float keep = 0;
  std::vector<Candidate> candidates;
};
// Picks draft i's token from its candidates. Not given, a draft is the first candidate.
using DraftPick = std::function<int32_t(uint32_t i, const std::vector<Candidate> &candidates)>;

class Model final : public engine::Model {
 public:
  Model(const std::string &path, const Options &opt, const Debug &debug = {});
  ~Model() override;

  void reset_slot(uint32_t) override {}
  void set_images(uint32_t slot, const Image *images, uint32_t n) override;
  uint32_t image_tokens_max() const override { return 0; }
  ImageShape image_shape() const override { return {}; }

  std::vector<std::vector<Candidate>> step(const StepIn *rows, uint32_t n, bool window = false,
                                           bool prompt = false) override;
  // Every cache is indexed by position, so a rejected row's cells are written again and nothing waits for a commit.
  void commit(uint32_t, uint32_t, uint32_t) override {}
  void commit_runs(const CommitRun *, uint32_t) override {}

  bool drafting() const override { return dsp_ != nullptr; }
  bool dflash() const override { return dsp_ != nullptr; }
  uint32_t draft_max_default() const override { return dsp_ ? D41_DSP_BLOCK : 0; }
  uint32_t draft_max_cap() const override { return dsp_ ? D41_DSP_BLOCK : 0; }
  std::vector<std::vector<Candidate>> mtp_catchup(const StepIn *, uint32_t, const std::vector<uint32_t> &,
                                                  const uint32_t *) override;
  std::vector<Candidate> mtp_draft(uint32_t, uint32_t, int32_t, uint32_t) override;
  std::vector<std::vector<Candidate>> mtp_drafts(const StepIn *, uint32_t, const uint32_t *) override;
  std::vector<std::vector<Candidate>> mtp_catchup_prompt(const StepIn *, uint32_t, uint32_t,
                                                         const std::vector<uint32_t> &) override;
  // One block a pass, of 1 to D41_DSP_BLOCK rows, row i for position pos + 1 + i. Returns the first row's candidates
  // only: a later row's depend on the draft chosen before it, and drafts gives them. A step stores every row's keys
  // for the drafter, so the context rows need nothing here.
  std::vector<std::vector<std::vector<Candidate>>> df_pass(const StepIn *, uint32_t, const DraftBlock *,
                                                           uint32_t) override;
  // The last df_pass's block as drafts, chosen in turn: draft i from row i's logits plus the bias of the token before
  // it, which is the block's token, then the drafts chosen. The cards add the bias over the whole vocabulary, a round
  // a draft after the first.
  std::vector<CardDraft> drafts(const DraftPick &pick = nullptr);
  int32_t noise_token() const;
  void df_context_prompt(const StepIn *, uint32_t, uint32_t) override;
  void draft_start(uint32_t, uint32_t) override {}

  std::vector<std::vector<Candidate>> prompt_runs(const PromptRun *, uint32_t, const int32_t *, uint32_t) override;
  uint32_t prompt_chunk() const override { return 0; }
  uint32_t fetch_rows() const override { return 0; }
  uint32_t decode_weight() const override { return 1; }

  uint64_t slot_state_bytes() override;
  void save_slot(uint32_t slot, uint8_t *dst) override;
  void load_slot(uint32_t slot, const uint8_t *src) override;
  void *alloc_pinned(size_t bytes) override { return rt_->alloc_host(bytes); }
  void free_pinned(void *p) override { rt_->free(p); }

  void sleep() override;
  void wake() override;

  uint32_t kv_pages() const override { return kv_pages_; }
  uint32_t pages_per_slot() const override { return npp_; }
  void set_pages(uint32_t slot, uint32_t first, const uint32_t *pages, uint32_t n) override;
  void copy_cells(uint32_t from, uint32_t to, uint32_t cells) override;
  void kv_need(uint32_t, uint32_t) override {}
  void kv_ahead() override {}
  void kvs_drop(uint32_t, uint32_t) override {}

  void note_prompt_routing() override {}
  uint32_t swap_round(uint32_t, bool) override { return 0; }
  bool swaps_ready() const override { return true; }
  bool swaps_settled() const override { return true; }

  const LoadInfo &info() const override { return info_; }
  const StepStats &last_stats() const override { return stats_; }
  uint32_t experts_cached() const override { return info_.experts_per_layer; }
  uint32_t experts_floor() const override { return 0; }
  NgramStats ngram_stats() const override { return engram_->stats(); }

  // For tests. logits: a row of the last step or block, every token's. probe: the streams of the last step's or
  // block's rows after layer's attention (ffn false) or its MoE block, D41_HC_DIM values a row, the drafter's layers
  // numbered from D41_LAYERS; needs Debug::probes.
  std::vector<float> logits(uint32_t row);
  const float *probe(uint32_t layer, bool ffn) const;
  // What the last step stored for the drafter, with Debug::probes: each row's input (D41_EMBD values a row), and the
  // key of a drafter layer (D41_HD a row). And a row of the last block before the output norm, D41_EMBD values.
  const float *draft_input() const;
  const float *draft_key(uint32_t layer) const;
  const float *draft_out(uint32_t row) const;
  // With Debug::probes: the experts a row of the last block routed to at a drafter layer, D41_DSP_USED ids.
  const int32_t *draft_route(uint32_t layer, uint32_t row) const;
  static constexpr uint32_t kAllLayers = D41_LAYERS + D41_DSP_LAYERS;
  // What the CPU answered in the last step, summed over its layers.
  const MissCounts &miss_counts() const { return counts_; }
  const StepTimes &step_times() const { return times_; }
  uint64_t swaps_done() const { return swaps_done_; }
  // The experts the last step's row routed to at layer, D41_USED ids.
  const int32_t *routed(uint32_t layer, uint32_t row) const { return jobs_[(size_t)layer * STEP_MAX_ROWS + row].all; }
  // A card's kernels in the last step, with Options::trace: each name's launches and time, in the list's order.
  std::vector<TraceRow> trace(uint32_t card) const;

 private:
  // A card's share of a matrix, in the runs layout.
  struct Mat {
    uint8_t *p = nullptr;
    uint32_t type = 0, rows = 0, K = 0, bpr = 0;
  };
  struct Layer {
    float *attn_fn, *attn_scale, *attn_base, *ffn_fn, *ffn_scale, *ffn_base;
    float *attn_norm, *ffn_norm, *q_norm, *kv_norm, *sinks, *bias;
    Mat q_a, kv, q_b, out_a, out_b, sh_gu, sh_down;
    uint32_t sh_ff = 0;  // the card's rows of the shared expert
    uint16_t *gate_inp;
    uint16_t *comp_kv = nullptr, *comp_gate = nullptr, *attn_k = nullptr, *idx_q_b = nullptr, *idx_proj = nullptr;
    float *comp_norm = nullptr, *k_norm = nullptr;
    Mat engram_kv;
    float *engram_gate = nullptr;  // the card's two streams' rows
    uint16_t *ring;                // the window, [slot][D41_RING][D41_HD] f16
    uint16_t *pool = nullptr;      // a KV source's compressed rows, by page
    float *cr = nullptr;           // a ratio-2 source's compressor ring
    // The routed experts the cards hold: each has part A in a slot on one card and part B on the other. A pair below
    // split has the part of class flip here.
    uint8_t *cache = nullptr;
    D41Class *classes = nullptr;
    int32_t *slot_map = nullptr;   // expert to its pair, -1 when the cards do not hold it
    uint32_t split = 0x7fffffffu, flip = 0;
  };
  struct Card {
    uint32_t c = 0;
    std::unique_ptr<ze::List> up;
    std::mutex up_mu;  // for the load's threads
    std::unique_ptr<ze::List> swap_up;  // the swap thread's copies
    uint8_t *swap_slot = nullptr;       // and the slot it builds, in pinned memory
    uint8_t *grids;
    StepWords *plan;
    uint32_t *pt;
    float *e;
    uint8_t *eng_a8[kEngramTables];
    float *X, *X1, *ss, *m;
    D41HcMix *amix, *fmix;
    uint8_t *xa;
    uint16_t *xb;
    float *qr, *kv, *q, *ckv, *csc, *iq, *iw, *score, *part, *y;
    uint8_t *qa, *oa, *aa;
    uint16_t *qb;
    uint32_t *idx;
    float *in[2];  // the other card's partial sums, even and odd exchanges
    float *ekv;
    D41EngramSend *eown[kEngramTables], *ein[kEngramTables];
    float *rlogits, *w, *sgu, *shared, *y6, *out;
    int32_t *ids, *slots;
    uint8_t *ha;  // the routed experts' hidden rows as a8, one an assignment
    uint32_t *nmiss, *ngroups;
    D41Group *groups;
    uint8_t *sha;
    float *out_norm, *logits;
    Mat head;
    uint32_t vocab0 = 0, vocab_n = 0;
    // The drafter's: its product with the features, the norms around its layers, and a step's rows on their way to
    // its keys.
    Mat fc;
    float *enc_norm = nullptr, *draft_norm = nullptr;
    uint8_t *fa = nullptr, *dxa = nullptr;
    float *dmain = nullptr, *dkv = nullptr, *dxf = nullptr, *dkvf = nullptr;
    // A draft's choice: markov_w2's rows of the card's tokens, the row of the token before, the bias and the row's
    // logits with it, and the words {row, 1}.
    uint16_t *w2 = nullptr, *rb = nullptr;
    float *bias = nullptr, *biased = nullptr;
    uint32_t *pick = nullptr;
    std::unique_ptr<ze::List> pick_list;
    std::vector<Layer> layers;  // the model's, then the drafter's
    std::map<uint32_t, std::unique_ptr<ze::List>> lists, draft_lists;  // by width
    std::map<uint32_t, uint64_t *> trace_ts, draft_trace_ts;           // a traced list's timestamps
  };
  // The drafter's file and what the host keeps of it.
  struct Drafter {
    explicit Drafter(const std::string &path) : gguf(path), reader(gguf.files()) {}
    GgufModel gguf;
    DirectReader reader;
    std::vector<LayerExperts> experts;
    int32_t noise = 0;
    std::vector<float> conf;       // conf_proj: D41_EMBD weights of a row's output, then D41_DSP_RANK of the token before
    AlignedBuffer w1_buf;          // markov_w1, [token][D41_DSP_RANK] bf16; markov_w2 is on the cards
    const uint16_t *w1 = nullptr;
    uint16_t *r = nullptr;         // for the cards: the row of the token before the draft being chosen
    uint32_t *words = nullptr;     // and {the draft's row, 1}
    float *x = nullptr;            // the last block's streams, [row][D41_HC_DIM], and its last mixes
    D41HcMix *mix = nullptr;
    float *probe_x = nullptr, *probe_kv = nullptr;
    int32_t *probe_ids = nullptr;  // the last block's routed experts, [layer][row][D41_DSP_USED]
    // The last block.
    uint32_t n = 0;
    int32_t token = 0;
    std::vector<Candidate> first;  // the first draft's candidates
    std::vector<float> out;        // [row][D41_EMBD], before the output norm
  };

  template <class T>
  T *dev(Card &card, size_t n);
  template <class T>
  T *upload(Card &card, const T *src, size_t n);
  float *upload_floats(Card &card, const std::string &name, std::initializer_list<uint64_t> shape, size_t first = 0,
                       size_t n = 0);
  uint16_t *upload_bf16(Card &card, const std::string &name, std::initializer_list<uint64_t> shape);
  const GgufModel &source() const { return dsp_loading_ ? dsp_->gguf : gguf_; }
  DirectReader &reader() { return dsp_loading_ ? dsp_->reader : ssd_->direct(); }
  const LayerExperts &layer_experts(uint32_t l) const {
    return l < D41_LAYERS ? experts_[l] : dsp_->experts[l - D41_LAYERS];
  }
  void load_drafter();
  void record_feed(ze::List &L, Card &card, uint32_t W);
  void record_draft(Card &card, uint32_t W);
  void record_pick(ze::List &L, Card &card);
  void set_pick(uint32_t row, int32_t prev);
  std::vector<Candidate> picked() const;
  Mat mat(Card &card, uint32_t type, uint32_t K, const std::vector<SliceSpec> &specs);
  const GgufTensor &tensor(const std::string &name, std::initializer_list<uint64_t> shape) const;
  const uint8_t *held(const GgufTensor &t);
  void load_layer(uint32_t l);
  void load_experts();
  void load_cache(uint32_t l, const std::vector<uint32_t> &ids);
  void make_slot(uint32_t l, uint32_t c, uint32_t pair, const uint8_t *gate, const uint8_t *up, const uint8_t *down,
                 const uint8_t *down_b, uint8_t *slot) const;
  uint8_t *slot_at(uint32_t l, uint32_t c, uint32_t pair) const;
  const D41Class &pair_class(uint32_t l, uint32_t c, uint32_t pair) const;
  void map_expert(uint32_t l, uint32_t expert, int32_t pair);
  void advance_swaps();
  void count_routing(uint32_t n);
  void swap_work();
  void alloc_buffers(Card &card);

  ze_kernel_handle_t kernel(Card &card, const char *module, const char *name);
  void matvec(ze::List &L, Card &card, const char *name, const Mat &m, const uint8_t *act, float *y, uint32_t W,
              uint32_t lanes, uint32_t ks = 0);
  void product_bf16(ze::List &L, Card &card, const char *name, const uint16_t *w, uint32_t N, uint32_t K,
                    const uint16_t *xb, float *y);
  void wait(ze::List &L, Card &card, const char *name, uint32_t *raise, const uint32_t *flag,
            const uint32_t *nmiss = nullptr, uint32_t W = 0);
  void record(Card &card, uint32_t W);
  void record_engram(ze::List &L, Card &card, uint32_t l, uint32_t W);
  void record_attention(ze::List &L, Card &card, uint32_t l, uint32_t W);
  void record_ffn(ze::List &L, Card &card, uint32_t l, uint32_t W);
  uint32_t *flag(uint32_t card, uint32_t k) const { return flags_ + ((size_t)card * kFlags + k) * 16; }

  // An exchange a sublayer, the drafter's after the model's, then one for each Engram layer.
  static constexpr uint32_t kFlags = 2 * kAllLayers + kEngramTables;

  Options opt_;
  Debug debug_;
  GgufModel gguf_;
  std::unique_ptr<ze::Runtime> rt_;
  std::unique_ptr<ze::Library> lib_;
  std::unique_ptr<Pool> pool_;
  std::unique_ptr<ExpertArena> arena_;
  std::unique_ptr<SsdReader> ssd_;
  std::unique_ptr<MissJobs> miss_;
  std::unique_ptr<EngramHash> hash_;
  std::unique_ptr<EngramReader> engram_;
  std::vector<LayerExperts> experts_;
  std::unique_ptr<Drafter> dsp_;
  bool dsp_loading_ = false;  // tensors are the drafter's file's
  bool last_draft_ = false;   // the last list run was a block's
  std::unique_ptr<Card> cards_[2];
  std::map<std::pair<uint32_t, uint64_t>, AlignedBuffer> held_;  // a tensor's bytes while its layer loads

  uint32_t vocab_ = 0, npp_ = 0, kv_pages_ = 0, step_ = 0, last_width_ = 0;
  StepWords *plan_ = nullptr;
  float *embed_ = nullptr;
  uint8_t *eng_a8_[kEngramTables] = {};
  D41MissJob *jobs_ = nullptr;   // [layer][row]
  float *cpu_y_ = nullptr;       // [layer][row][D41_EMBD]
  uint32_t *ans_flag_ = nullptr;  // a 64-byte line a layer
  uint32_t *eng_flag_ = nullptr;  // a line an Engram table: its rows of this step are in eng_a8_
  // The Engram rows' thread: it gathers a step's rows from the file while the lists run their first layers.
  std::thread engram_thread_;
  std::mutex engram_mu_;
  std::condition_variable engram_cv_;
  const StepIn *engram_rows_ = nullptr;
  uint32_t engram_n_ = 0;
  uint64_t engram_asked_ = 0;
  std::atomic<uint64_t> engram_done_{0};
  std::exception_ptr engram_failed_;
  bool engram_stop_ = false;
  void engram_work();
  void engram_wait();
  uint32_t *flags_ = nullptr;     // [card][exchange], a 64-byte line each
  uint32_t *stuck_ = nullptr;     // a line a card
  uint32_t *top_idx_ = nullptr, *top_n_ = nullptr;
  float *top_val_ = nullptr;
  float *probe_ = nullptr;        // [sublayer][layer][row][D41_HC_DIM]
  std::vector<uint32_t> pt_;      // [slot][page]
  std::vector<float> eng_f_;
  LoadInfo info_;
  StepStats stats_;
  MissCounts counts_;
  StepTimes times_;

  // The cards' pairs by layer: the expert each holds (kNoExpert while it changes), the split, the slot classes by card.
  static constexpr uint32_t kNoExpert = 0xffffffffu;
  std::vector<std::vector<uint32_t>> pairs_;
  std::vector<uint32_t> split_;
  std::vector<std::array<std::array<D41Class, 2>, 2>> tables_;
  // Options::adapt_swaps: an expert the CPU keeps answering takes the pair of the layer's least used one. The step
  // thread chooses and maps between steps; the swap thread loads the slots, then refills the arena places they freed.
  // With in kNoExpert no pair changes: out comes from the file into place, which an arena expert used less gave up.
  struct Swap {
    uint32_t l, pair, in, out;
    uint8_t *place;  // in's place in the arena, which out takes; null when in came from the file
  };
  enum class SwapState { idle, loading, loaded, refilling, refilled, failed };
  std::vector<float> heat_;  // [layer][expert]: routings, the older ones counting less
  float heat_unit_ = 1.0f;
  std::vector<Swap> swaps_;
  std::vector<std::pair<uint32_t, uint32_t>> file_seen_;  // (layer, expert) the file gave since the last choice
  std::vector<std::vector<uint32_t>> missed_;             // by layer: the experts missed since the last choice
  uint32_t coldest_in_arena(uint32_t l) const;
  std::atomic<SwapState> swap_state_{SwapState::idle};
  std::mutex swap_mu_;
  std::condition_variable swap_cv_;
  bool swap_stop_ = false;
  std::exception_ptr swap_failed_;
  std::thread swapper_;
  uint64_t swaps_done_ = 0;
};

}  // namespace engine::deepseek41
