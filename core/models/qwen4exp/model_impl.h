// The model's internal state: shared by model.cpp (loading and the decode step) and prompt.cpp (the prompt path).
#pragma once

#include <map>
#include <cstring>
#include <string>

#include "ggml.h"
#include "layout.h"
#include "model.h"
#include "moe_tile.h"
#include "quant.h"

namespace engine::qwen4exp {

namespace detail {

using Ranges = std::vector<std::pair<uint64_t, uint64_t>>;  // [start, start + count)

inline constexpr uint32_t kRecFloats = 2 * Q_USED + Q_EMBD;  // the router's record: ids, weights, x
inline constexpr uint32_t kGdn = 36, kAttn = 12;
inline constexpr uint32_t kSgSize = 16;  // the kernels' SG_SIZE
inline constexpr uint32_t kAttXmxSubGroups = 16;  // p_attn_xmx's sub-groups a work-group (prompt_attn.cl)
inline constexpr uint32_t kIdxScoreRows = 8;  // prompt.cl's IDX_SCORE_ROWS
// The fewest rows of a prompt chunk at which the two cards split the hyper-connections
// (docs/measurements/MEASUREMENTS.md#2026-10-17-hc-split).
inline constexpr uint32_t kHcSplitRows = 2048;
// The largest prompt chunk whose uncached experts are fetched as its router picks them (prompt.cpp, fetch_experts)
// instead of all staged by the recorded copy list.
inline constexpr uint32_t kFetchRows = 1024;
// p_idx_score_xmx's work-group (prompt_attn.cl): IDX_XMX_ROWS rows by IDX_XMX_SG sub-groups of 16 IDX_XMX_TB blocks.
inline constexpr uint32_t kIdxXmxRows = 16, kIdxXmxSg = 4, kIdxXmxBlocks = 16 * 2 * kIdxXmxSg;
inline constexpr uint32_t kAttSplits = (Q_IDX_CELLS + 63) / 64;  // attn_split's ATT_SPLITS
inline constexpr uint32_t kIdxCand = 2 * (Q_IDX_BLOCKS + 1) + 2;  // words per row of the indexer's candidates
// The streamed pool (kvs_claim): the logical pages a claim takes past each row's, so the steps that open them need
// no claim; the pages a step's rows can claim at most, its miss list; and what a width's list without the claim adds
// to its key (Model::record).
inline constexpr uint32_t kKvsAhead = 32;
inline constexpr uint32_t kKvsMaxMiss = STEP_MAX_ROWS * (Q_IDX_BLOCKS + 2 + kKvsAhead);
inline constexpr uint32_t kKvsSkipList = 16;
inline constexpr uint32_t kMvKsRows = 512;  // dense products of up to this many rows split K (matvec_ks)
// The decode step's widths, each with its own lists: a step runs the smallest of at least its rows, and its products
// and expert kernels compute that many.
inline constexpr uint32_t kStepWidths[] = {1, 2, 3, 4, 6, 8};
inline uint32_t step_width(uint32_t n) {
  for (uint32_t W : kStepWidths)
    if (W >= n) return W;
  return STEP_MAX_ROWS;
}
// A prompt token's weight in the swap counts against a written token's 1: there are many, and decode is what the
// cache serves.
inline constexpr float kPromptRouting = 0.05f;
// The main layers whose outputs DFlash 2 reads, in the order of its features (zero-based; the model's
// target_layer_ids).
inline constexpr uint32_t kDfTargetLayers[DF_LAYERS] = {5, 15, 25, 35, 45};

inline uint64_t a8_bytes(uint64_t K) { return K + K / 8; }
inline uint64_t align256(uint64_t x) { return (x + 255) & ~255ull; }

inline bool is_raw(uint32_t type) { return type == QT_F32 || type == QT_F16 || type == QT_BF16; }

// Lanes per row for a dense product: 16 for long rows of the i-quants, 8 for the K-quants and IQ4_XS, 4 for short
// rows, the fastest of each (docs/measurements/MEASUREMENTS.md#2026-10-11-kernel-spike).
inline uint32_t lanes_for(uint32_t type, uint32_t K) {
  if (K / 32 < 40) return 4;
  if (type == QT_Q3_K || type == QT_Q4_K || type == QT_Q5_K || type == QT_Q6_K || type == QT_IQ4_XS) return 8;
  return 16;
}

// f to bf16 as the kernels' f32_to_bf16 rounds it: to nearest, ties to even; NaN stays NaN.
inline uint16_t bf16_bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 64);
  return (uint16_t)((u + (0x7fffu + ((u >> 16) & 1))) >> 16);
}

// Lanes per row for matvec_bf16 (common.cl): 8 values per lane per read, so short rows share a sub-group.
inline uint32_t bf16_lanes(uint32_t K) { return K >= 1024 ? 16 : K >= 256 ? 4 : 2; }

inline std::string mv_module(uint32_t type, uint32_t lanes) {
  std::string s = std::string("matvec_") + ggml_type_label(type) + "_l" + std::to_string(lanes);
  for (auto &c : s) c = (char)std::tolower(c);
  return s;
}

// kernels/quant/moe.cl's module for an expert type and groups of up to W assignments (W: kStepWidths), or "" when it
// has no kernel for the type (matvec.cl's then).
inline std::string moe_module(uint32_t type, uint32_t W, bool down) {
  bool has = down ? type == QT_IQ4_NL || type == QT_Q2_0 || type == QT_Q8_0
                  : type == QT_IQ2_S || type == QT_IQ3_XXS || type == QT_IQ3_S || type == QT_IQ4_XS ||
                        type == QT_IQ2_XXS || type == QT_IQ2_XS || type == QT_IQ1_M;
  if (!has) return "";
  std::string s = std::string("moe_") + ggml_type_label(type) + "_g" + std::to_string(W);
  for (auto &c : s) c = (char)std::tolower(c);
  return s;
}

inline std::string blk(uint32_t l, const char *name) { return "blk." + std::to_string(l) + "." + name; }

// An expert's down matrix (rows of row_bytes, as the file stores it) into dst as `parts` regions, one after the other:
// region p holds part p of every row's blocks (ExpertWeights::down_parts).
inline void write_down(uint8_t *dst, const uint8_t *src, uint64_t rows, uint64_t row_bytes, uint32_t parts) {
  if (parts == 1) {
    std::memcpy(dst, src, rows * row_bytes);
    return;
  }
  const uint64_t pb = row_bytes / parts;
  for (uint32_t p = 0; p < parts; p++)
    for (uint64_t r = 0; r < rows; r++) std::memcpy(dst + (p * rows + r) * pb, src + r * row_bytes + p * pb, pb);
}

}  // namespace detail

// One layer's weights and state on one card.
struct Model::Layer {
  bool attn = false;
  uint32_t idx = 0;  // index among the layers of its kind
  // Hyper-connections, whole on every card: [0] before the mixer, [1] before the MoE.
  float *hc_norm[2] = {};
  uint16_t *hc_down[2] = {}, *hc_up[2] = {}, *hc_inj[2] = {};
  // Gated DeltaNet, this card's heads.
  Mat qkv, z, out;
  uint16_t *ab = nullptr;  // alpha rows then beta rows
  float *conv = nullptr, *A = nullptr, *dt = nullptr, *ssm_norm = nullptr;
  // Attention, this card's heads; the indexer whole.
  Mat q, k, v, o;
  float *qn = nullptr, *kn = nullptr, *iqn = nullptr, *ikn = nullptr;
  uint16_t *idx_w = nullptr;  // indexer query (512 rows) then key (128 rows)
  // MoE.
  uint16_t *router = nullptr;  // 512 router rows, then the shared expert's gate row
  Mat sh_gate, sh_up, sh_down;
  uint32_t gu_type = 0, down_type = 0;
  uint8_t *cache = nullptr;
  uint64_t slot_bytes = 0, up_off = 0, down_off = 0;
  int32_t *slot_map = nullptr;
  // Per-slot state.
  float *conv_state = nullptr, *S = nullptr;
  float *wu = nullptr, *wgb = nullptr, *wm = nullptr;  // a step's u, gates and m, kept for a window's commit
  uint64_t conv_stride = 0, S_stride = 0;  // floats per slot
  // The KV cache's pool of pages (layout.h): K and V of every page; the indexer's raw and pooled keys of this card's
  // pages only, at local page p / C.
  uint8_t *kc = nullptr, *vc = nullptr, *raw = nullptr;
  float *pooled = nullptr;
  // Streamed (Options::kv_resident, layer.cl's kvs_claim): kc and vc hold a ring of up to kv_res_ resident pages,
  // kst[4] its size now; khost and vhost in host memory hold every page; where, spage, ref, mark, kst and miss are
  // the residency's state.
  uint8_t *khost = nullptr, *vhost = nullptr;
  uint32_t *where = nullptr, *spage = nullptr, *ref = nullptr, *mark = nullptr, *kst = nullptr, *miss = nullptr;
};

// The draft (MTP) layer on one card (mtp.cpp).
struct Model::MtpCard {
  // The input: the embedding's and the residual's norms and projections.
  float *ge = nullptr, *gh = nullptr;
  uint16_t *fc_e = nullptr, *fc_h = nullptr;
  // The decoder layer, as a main attention layer, BF16.
  float *hc_norm[2] = {};
  uint16_t *hc_down[2] = {}, *hc_up[2] = {};  // down: 320 rows then inject's 4
  uint16_t *q = nullptr, *k = nullptr, *v = nullptr, *o = nullptr;
  float *qn = nullptr, *kn = nullptr;
  uint16_t *router = nullptr;  // 512 rows, then the shared expert's gate
  uint16_t *sh_gate = nullptr, *sh_up = nullptr, *sh_down = nullptr;
  uint8_t *cache = nullptr;
  uint64_t slot_bytes = 0, up_off = 0, down_off = 0;
  int32_t *slot_map = nullptr;
  uint8_t *kc = nullptr, *vc = nullptr;
  uint64_t kv_stride = 0;
  // The final mixer.
  float *mx_norm = nullptr;
  uint16_t *mx_down = nullptr, *mx_up = nullptr;
  // Buffers, STEP_MAX_ROWS rows (hn and hs: 4 streams a row).
  float *R = nullptr, *e = nullptr, *en = nullptr, *hn = nullptr, *hs = nullptr;
  uint32_t *first = nullptr;  // per slot: its first cell
  StepPlan *plan = nullptr;     // the draft step's plan, copied from host memory by each list's first commands
  StepPlan *rest = nullptr;     // the rows of it that run the whole layer, first in the plan
  uint32_t *counts = nullptr;   // counts[i] = i, for row counts that are not the plan's
  std::map<uint32_t, std::unique_ptr<ze::List>> lists;  // catch-up, per width * 16 + the rest's width (0 to width)
  std::map<uint32_t, std::unique_ptr<ze::List>> chains; // one draft step from the draft before, per width
};

// The DFlash 2 drafter on one card (dflash.cpp): BF16 weights, this card's query and KV heads and MLP rows; fc, the
// norms and the convolutions whole.
struct Model::DfCard {
  uint32_t nq = 0, nkv = 0, ff = 0;
  uint16_t *fc = nullptr;
  float *hnorm = nullptr, *norm = nullptr;
  struct Layer {
    float *in_norm = nullptr, *post_norm = nullptr, *qn = nullptr, *kn = nullptr;
    uint16_t *q = nullptr, *k = nullptr, *v = nullptr, *o = nullptr, *gate = nullptr, *up = nullptr,
             *down = nullptr;
    float *abase = nullptr, *mbase = nullptr;  // the convolutions' fixed taps, [set][tap][Q_EMBD]
    uint16_t *akp = nullptr, *mkp = nullptr;   // their dynamic taps' projections, DF_KP rows
    float *rk = nullptr, *rv = nullptr;        // the ring: [slot][DF_RING][nkv][DF_D], a cell at position % DF_RING
  };
  Layer layers[DF_LAYERS];
  // The decode step's features of its rows, bf16 [STEP_MAX_ROWS][DF_FEAT] (Model::record writes them).
  uint16_t *feat = nullptr;
  // Buffers, STEP_MAX_ROWS rows; the bf16 ones are read whole by mv_xmx_bf16.
  float *ctx = nullptr, *h = nullptr, *x = nullptr, *xc = nullptr, *dyn = nullptr, *q = nullptr, *k = nullptr,
        *v = nullptr, *part = nullptr, *o = nullptr, *g = nullptr, *u = nullptr, *hn = nullptr;
  uint16_t *ctxb = nullptr, *xb = nullptr, *xcb = nullptr, *ob = nullptr, *gub = nullptr;
  DraftPlan *cplan = nullptr, *bplan = nullptr;  // copied from host memory by each list's first command
  std::map<uint32_t, std::unique_ptr<ze::List>> ctx_lists, block_lists;  // per width
  std::map<uint32_t, uint64_t *> trace;                                    // the block lists' kernel times
  uint32_t last = 0;                                                       // the last block pass's width
};

struct Model::Card {
  uint32_t c = 0, C = 1;  // this card, and how many
  uint32_t nkh = 0, nvh = 0, nch = 0, nq = 0, nkv = 0, ff = 0;
  uint32_t vocab0 = 0, vocab_n = 0;
  std::vector<Layer> layers;
  float *out_norm = nullptr;
  uint16_t *out_down = nullptr, *out_up = nullptr;
  Mat head;
  uint32_t *ptab = nullptr;  // the page table, [slot][pages_per_slot()] (layout.h)
  uint16_t *ple_k = nullptr, *ple_v = nullptr;
  float *ple_gk = nullptr, *ple_gq = nullptr, *ple_gc = nullptr, *ple_conv = nullptr, *ple_hist = nullptr;
  uint64_t ple_stride = 0;
  uint8_t *grids = nullptr;
  // Scratch.
  float *R, *n, *lr, *a, *inj[2], *x, *m, *z, *ab, *gb, *u, *o, *Q, *Kt, *Vt, *idxq, *scores, *bscores, *att;
  uint16_t *xb16 = nullptr;  // up to 8 rows of Q_HC_DIM bf16 activations, for mv_xmx_bf16
  uint16_t *lrb16 = nullptr;  // up to 8 rows of Q_HC_LR bf16, for hc_up_mix_xmx
  uint16_t *xbf16 = nullptr;  // up to 8 rows of x as bf16 (hc_up_mix_xmx writes them), for the BF16 products
  uint16_t *ngb16 = nullptr;  // up to 8 n-gram rows as bf16, for the PLE module's products
  uint32_t *blocks, *cells, *ncells;
  float *rl, *w, *h, *y, *shg, *shu, *shh, *s, *pk, *pv, *gates, *N, *logits;
  int32_t *ids, *slots;
  float *merged;  // a row's merged indexer candidates' scores
  uint32_t *icand;  // [row][kIdxCand]: this card's indexer candidates, before one copy takes them to icand_
  uint32_t *nmiss;  // [layer][row] experts the row's token routes to that the cards do not hold
  MoeGroup *groups;  // the step's cached experts grouped, and how many groups
  uint32_t *ngroups;
  uint8_t *xa8, *oa8, *ha8, *sha8;
  float *yo = nullptr;  // a window's out product (ssm_out, attn_output), before one copy takes it to ex1_
  // A window's partial sums with two cards (ADR-0030): [exchange 0 mixer, 1 MoE][card][STEP_MAX_ROWS][Q_EMBD]. Each
  // card writes its own block here and one copy places it in the other card's.
  float *xp = nullptr;
  StepPlan *plan = nullptr;  // the step plan, copied from host memory by the decode step's first command
  std::unique_ptr<ze::List> up;
  std::map<uint32_t, std::unique_ptr<ze::List>> lists;  // the decode step, per width (kStepWidths)
  std::map<uint32_t, uint64_t *> trace;                 // their timestamps, with Options::trace
  std::unique_ptr<ze::List> commit;                     // a window's commit (CommitPlan)
  float *Sw = nullptr;                                  // a window's DeltaNet state past its first row
  std::unique_ptr<PromptCard> pc;
  std::unique_ptr<MtpCard> mtp;  // the draft layer (mtp.cpp)
  std::unique_ptr<DfCard> df;    // the DFlash 2 drafter (dflash.cpp)
  std::unique_ptr<VisionCard> vc;  // the image encoder (vision.cpp)
  // With Options::share_kv (kv_share.cpp): the KV pool's buffers and the layers' caches as regions, and the blocks
  // mapped in none.
  std::vector<Region> kv, cache;
  std::vector<ze_physical_mem_handle_t> spare;
};

struct Model::HostLayer {
  ExpertShape shape;
  std::vector<int32_t> slot_of;              // expert -> slot on the cards, or -1
  std::vector<int32_t> stage_of;             // expert -> its slot in the prompt path's staging, or -1
  std::vector<uint32_t> uncached;            // the experts not on the cards, in staging order
  std::vector<ExpertWeights> weights;        // expert -> its matrices in the arena (misses only)
  std::vector<uint8_t *> arena;              // one host allocation per expert not cached: gate, up, down
  std::vector<uint8_t *> pieces;             // staging position -> its piece, for every expert the cards may leave out
  uint64_t gate_bytes = 0, down_bytes = 0;   // one expert's gate (or up) matrix and its down matrix in the file
  uint64_t gate_row_bytes = 0;
  uint32_t down_parts = 1;                   // down in the arena as one region per card (write_down)
  // An expert in the staging buffer of a card: its half of gate's rows, then of up's, then its region of down.
  uint64_t stg_up_off = 0, stg_down_off = 0, stg_bytes = 0;
};

// The prompt path's buffers on one card (prompt.cpp). Rows are tokens, up to the largest chunk.
struct Model::PromptCard {
  float *R = nullptr, *inj[2] = {}, *lr = nullptr, *ab = nullptr, *gb = nullptr;
  uint16_t *lrb = nullptr, *x16 = nullptr, *xb = nullptr, *o16 = nullptr;
  // The hyper-connection mix, the experts and the PLE module use one region in turn.
  float *pk = nullptr, *N = nullptr;
  uint16_t *a = nullptr, *nb = nullptr, *xg = nullptr, *hg = nullptr, *yg = nullptr;
  float *rr = nullptr;  // the streams' 1 / rms, per row
  // So do DeltaNet and attention.
  float *m = nullptr, *z = nullptr, *u = nullptr, *raw = nullptr;
  float *Q = nullptr, *K = nullptr, *V = nullptr, *iq = nullptr, *scores = nullptr, *bscores = nullptr;
  uint32_t *blocks = nullptr, *cells = nullptr, *ncells = nullptr;
  uint32_t *cand = nullptr, *cand_peer = nullptr;  // the indexer's candidates: this card's, and the other card's
  float *merged = nullptr;
  uint16_t *kh = nullptr, *vh = nullptr;  // the layer's K and V as fp16, every position
  uint16_t *iqh = nullptr, *iql = nullptr;  // the chunk's indexer queries as two fp16 terms (p_idx_qsplit)
  uint16_t *ikh = nullptr, *ikl = nullptr;  // the pooled keys as two fp16 terms, at their local page (p_idx_ksplit)
  uint8_t *kvst = nullptr;  // streamed: p_attn_prep's new K then V cells, [row][nkv] (p_kv_flush)
  float *apart = nullptr;                 // the short runs' attention splits (p_attn_rows)
  uint16_t *qt = nullptr;                 // the queries as fp16, dimension-major (p_attn_xmx)
  uint32_t hp = 0;                        // query slots per dimension in qt (nkv Q_ATT_SLOTS)
  // The experts' routing and the shared expert.
  float *rl = nullptr, *w = nullptr, *shg = nullptr, *shu = nullptr, *s = nullptr;
  uint16_t *part = nullptr;     // this card's fp16 partial sum, before one copy takes it to the other card
  uint16_t *recv[2] = {};       // the other card's, copied in by that card: exchange e into recv[e % 2]
  int32_t *ids = nullptr;
  uint32_t *cnt = nullptr, *base = nullptr, *cursor = nullptr, *inv = nullptr;
  uint32_t *ntiles_c = nullptr, *ntiles_s = nullptr;
  MoeTile *tiles_c = nullptr, *tiles_s = nullptr;
  uint16_t *shh = nullptr;
  // PLE.
  uint16_t *ub = nullptr;
  float *pv = nullptr, *gates = nullptr;
  uint16_t *w16 = nullptr;  // a dense weight dequantised to fp16
  uint32_t *prof = nullptr;  // [layer][expert] rows routed, with Options::profile_routing (card 0)
  // The experts the cards do not cache, streamed in on the copy engine: two layers' worth, used in turn.
  uint8_t *stage = nullptr;
  uint64_t stage_slot = 0;
  std::vector<int32_t *> stage_map;  // per layer: expert -> staging slot, or -1
  std::unique_ptr<ze::List> copy;
  // Chunks of kFetchRows and under: the immediate list the host fetches experts on, and each layer's routing counts
  // (p_moe_count's), copied to host memory as the router finishes: [layer][expert].
  std::unique_ptr<ze::List> fetch;
  uint32_t *routed = nullptr;
  std::map<uint32_t, std::unique_ptr<ze::List>> lists;  // per chunk size recorded
  std::map<uint32_t, uint64_t *> trace;                 // per chunk size, with Options::trace
  uint32_t last = 0;                                     // the chunk size run last
  // Regions A and B, free between chunks: the image encoder's activations live there (vision.cpp).
  uint8_t *reg[2] = {};
  uint64_t reg_bytes[2] = {};
};

// The image encoder on one card (vision.cpp). The small f32 tensors stay on the card; the 16-bit weights come from
// host memory a stage at a time (a layer, or a matrix of the merger) into two slots used in turn.
struct Model::VisionCard {
  float *wt = nullptr, *pbias = nullptr, *pos = nullptr;  // the patch embedding, W0 + W1 transposed; bias; table
  float *wt2 = nullptr;                                   // W0 transposed, then W1: a pair of video frames
  uint16_t *gelu = nullptr;                               // the GELU's fp16 table (vision.cpp, gelu_table)
  struct Layer {
    float *ln1w, *ln1b, *bqkv, *bout, *ln2w, *ln2b, *bup, *bdown;
  };
  std::vector<Layer> layers;
  float *post_w = nullptr, *post_b = nullptr, *b0 = nullptr, *b2 = nullptr;
  uint8_t *slot[2] = {};
  // Activations, in the prompt path's regions: rows of this card's patches, and K and V of every patch.
  float *x = nullptr, *y = nullptr;
  uint16_t *a = nullptr, *q = nullptr, *k = nullptr, *v = nullptr, *o = nullptr, *h = nullptr;
  uint8_t *pix = nullptr;
  std::unique_ptr<ze::List> list, copy;
};

// A traced list's last run, summed per kernel name.
std::vector<TraceRow> trace_rows(const ze::List &L, const uint64_t *ts, const ze::Card &card);

}  // namespace engine::qwen4exp
