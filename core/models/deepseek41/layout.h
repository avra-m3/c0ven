// deepseek41's sizes, and the layout of what the core and the kernels share: the one definition, read by both (C++20
// and OpenCL C). The model and its file are ADR-0074's; the layers are DeepSeek's inference/model.py for
// DeepSeek-V4.1-Flash-1017.
#ifndef DEEPSEEK41_LAYOUT_H
#define DEEPSEEK41_LAYOUT_H

#include "step.h"
#include "types.h"

#define D41_EMBD 5120
#define D41_LAYERS 40
#define D41_HC 4                       // residual streams
#define D41_HC_DIM (D41_HC * D41_EMBD)  // 20,480
#define D41_MIX ((2 + D41_HC) * D41_HC)  // 24 mixes a sublayer: pre 4, post 4, comb 4 x 4
#define D41_SINKHORN 20                // Sinkhorn iterations of comb
#define D41_HC_EPS 1e-6f
#define D41_EPS 1e-20f                 // every RMS norm's epsilon, and the router's

#define D41_H 64                       // query heads; one KV head, K = V
#define D41_HD 512                     // head size
#define D41_ROT 64                     // rotated dimensions, the last of a head: adjacent pairs (2i, 2i + 1)
#define D41_NOPE (D41_HD - D41_ROT)    // 448
#define D41_QLR 1280                   // q's low rank
#define D41_OG 8                       // output groups of 8 heads
#define D41_OLR 1024                   // each group's low rank
#define D41_WIN 128                    // a query sees positions p - 127 .. p of the window
#define D41_RING (D41_WIN + STEP_MAX_ROWS)  // window cells a slot and layer: a step's rows write past the window
#define D41_CRING 16                   // a ratio-2 compressor's entries a slot, at position % 16

#define D41_IDX_H 32                   // indexer heads
#define D41_IDX_D 128                  // indexer head size; its last D41_ROT dimensions rotate
#define D41_TOPK 512                   // compressed rows a query attends to at most
#define D41_CAND_BLOCK 8               // the candidate filter's block of rows (layer 20)
#define D41_CAND_BLOCKS 2048           // blocks it keeps
#define D41_CTX_CAP 16384              // positions a slot, until the candidate filter exists; rows of a score
#define D41_IDX_WG_ROWS 64             // ds_idx_score's rows a work-group, 16 a sub-group
#define D41_IDX_WGS (D41_CTX_CAP / D41_IDX_WG_ROWS)  // ds_idx_score's work-groups a row of the step

#define D41_EXPERTS 384
#define D41_USED 6
#define D41_FF 2304                    // expert and shared-expert width
#define D41_ROUTE_SCALE 1.5f
#define D41_SWIGLU_LIMIT 10.0f         // min(g, L) and clamp(u, -L, L), routed and shared experts

#define D41_VOCAB 129280

// The drafter (the file's own draft model, a second GGUF): D41_DSP_LAYERS layers shaped as the window-only ones, with
// fewer routed experts. A draft is one pass over a block: the last chosen token, then noise tokens.
#define D41_DSP_LAYERS 3
#define D41_DSP_EXPERTS 128
#define D41_DSP_USED 3
#define D41_DSP_BLOCK 5                // tokens a draft holds at most
#define D41_DSP_RANK 256               // the previous token's row that biases a draft's logits
#define D41_DSP_TARGETS 3              // model layers whose inputs feed it
#define D41_DSP_TARGET0 37             // the first of them; the others follow
#define D41_DSP_FEATURES (D41_DSP_TARGETS * D41_EMBD)
#define D41_ENG_NGRAM 4                // a token and the three before it
#define D41_ENG_HEADS 8                // heads an n-gram size
#define D41_ENG_COLS ((D41_ENG_NGRAM - 1) * D41_ENG_HEADS)  // 24 rows a token and Engram layer
#define D41_ENG_ROW 256                // values a row: 8 IQ4_NL blocks of 32
#define D41_ENG_ROW_BYTES 144          // a row's bytes in the file
#define D41_ENG_G (D41_ENG_COLS * D41_ENG_ROW)  // 6,144: a token's gathered rows
#define D41_ENG_KV ((D41_HC + 1) * D41_EMBD)    // 25,600: a key per stream, then the shared value
#define D41_ENG_CLAMP 1e-6f

// Rotary bases: base 10000 with no YaRN at the window-only layers (0, 1); base 160000 with YaRN at the others
// (factor 16 over an original 65,536, beta_fast 32, beta_slow 1, no magnitude scale). rope.h holds the angles.
#define D41_ROPE_BASE 10000.0f
#define D41_ROPE_BASE_C 160000.0f
#define D41_YARN_FACTOR 16.0f
#define D41_YARN_ORIG 65536
#define D41_YARN_BETA_FAST 32.0f
#define D41_YARN_BETA_SLOW 1.0f

// Layer kinds. Ratio 0: the window only (0, 1); 2: a compressed row per 2 positions (2-19); 1: one per position
// (20-39). A layer reads the compressed rows and index keys of its KV source and the top-k of its index source.
#define D41_RATIO(l) ((l) < 2 ? 0 : (l) < 20 ? 2 : 1)
#define D41_KV_SRC(l) ((l) < 2 ? -1 : (l) < 20 ? 2 + 6 * (((l) - 2) / 6) : 20)
#define D41_IDX_SRC(l) ((l) < 20 ? D41_KV_SRC(l) : 20 + 4 * (((l) - 20) / 4))
#define D41_ENGRAM(l) ((l) == 1 ? 0 : (l) == 14 ? 1 : -1)  // the layer's Engram table, or -1

// A compressed row: the rotated latent (D41_HD f16), then its index key (D41_IDX_D f16), 1,280 B. Row j stands for
// positions j r .. j r + r - 1 and is rotated at position j r.
#define D41_COMP_ROW (D41_HD + D41_IDX_D)
#define D41_COMP_ROW_BYTES (2 * D41_COMP_ROW)
#define D41_COMP_ROWF (D41_COMP_ROW + D41_HD)  // ds_compress's f32 output for tests: the row, then the unrotated latent

// The compressed pool is paged as the KV cache is: a page is KV_PAGE_CELLS positions, so it holds 2 rows at ratio 2
// and 4 at ratio 1. A row's index in its source's pool, from the page holding position j r.
SHARED_ASSERT(KV_PAGE_CELLS == 4, "a page holds whole compressed rows at ratios 1 and 2");
SHARED_FN u64 d41_comp_index(u32 page, u32 j, u32 r) {
  return (u64)page * (KV_PAGE_CELLS / r) + j % (KV_PAGE_CELLS / r);
}
#define D41_NO_PAGE 0xffffffffu

#ifdef __OPENCL_C_VERSION__
// The page holding a slot's position: from the slot's row of the table (npp entries), or with no table the slot's own
// run of npp pages.
static inline uint d41_page(__global const uint *pt, uint npp, uint slot, uint pos) {
  uint i = slot * npp + pos / KV_PAGE_CELLS;
  return pt ? pt[i] : i;
}
#endif

// One row of a step, as the kernels read it from the plan's copy on the card. A padded row has slot D41_NO_ROW and
// writes no cache.
#define D41_NO_ROW 0xffffffffu
typedef struct {
  u32 slot;
  u32 pos;
  u32 ring;  // pos % D41_RING, the window cell it writes
  u32 prev;  // the row of this step holding the slot's position pos - 1, or D41_NO_ROW
} D41Row;
SHARED_ASSERT(sizeof(D41Row) == 16, "D41Row layout");

// ds_qkv_norm's sub-groups a work-group: qr has 5 chunks of 64 in each, kv 2.
#define D41_QKV_SGS 4
#define D41_QR_A8_ROW (D41_QLR + D41_QLR / 8)  // bytes of a row of qr as a8 (kernels/include/a8.h)

// ds_attn_split's grid: (first dimension) groups of D41_ATT_HEADS heads, (second) rows, (third) splits of keys, split 0
// the window and splits 1 to 4 up to D41_ATT_KEYS compressed rows each. 16 sub-groups share a work-group's products,
// faster than 8 (docs/measurements/MEASUREMENTS.md#2026-11-02-deepseek41-attn-split).
#define D41_ATT_KEYS 128
#define D41_ATT_SPLITS (1 + D41_TOPK / D41_ATT_KEYS)
#define D41_ATT_HEADS 8
#define D41_ATT_SGS 16
#define D41_ATT_WG (D41_ATT_SGS * 16)
// What ds_attn_split leaves for ds_attn_merge, for each row and head D41_ATT_PART f32: each split's sum of
// exp(s - m) v (D41_HD), then (m, sum of exp(s - m)) for each split at D41_ATT_ML, padded so a head's end is 64-aligned.
#define D41_ATT_ML (D41_ATT_SPLITS * D41_HD)
#define D41_ATT_PART (D41_ATT_ML + 16)

// An output group: its heads one after the other as one a8 row, the input of attn_output_a's D41_OLR rows of that group.
#define D41_OG_HEADS (D41_H / D41_OG)
#define D41_OG_K (D41_OG_HEADS * D41_HD)            // 4,096
#define D41_OG_A8_ROW (D41_OG_K + D41_OG_K / 8)     // bytes of a group's a8 row
#define D41_OG_WG_ROWS 32                           // ds_output_groups' rows a work-group: an a8 unit of its output
// ds_output_groups' lanes a row: the shortest time with cold weights
// (docs/measurements/MEASUREMENTS.md#2026-11-02-deepseek41-output-q5k).
#define D41_OG_LANES 4

// A ratio-2 source's compressor ring: [slot][D41_CRING] cells of kv' then sc (D41_HD f32 each).
#define D41_CRING_CELL (2 * D41_HD)

// The decode step's doorbell with the CPU (ADR-0011): one work-item polls with a system-scope acquire fence, then a
// volatile load, and gives up after 2.4e9 device-clock ticks, setting the stuck flag (ADR-0074 item 5). The posting
// card writes one record a row in host memory, its step last; a padded row's nmiss is 0. The CPU computes the misses
// from the arena or the SSD staging ring and answers by its own flag. A layer with no miss is skipped by both.
typedef struct {
  u32 seq;                 // the step, written last, behind a release fence
  u32 nmiss;               // experts not on the cards
  i32 ids[D41_USED];       // the misses' ids, then their weights
  float w[D41_USED];
  i32 all[D41_USED];       // every routed expert's id, for the routing counts
  u32 pad[12];
  float x[D41_EMBD];       // the MoE block's input, normed
} D41MissJob;
SHARED_ASSERT(sizeof(D41MissJob) == 128 + 4 * D41_EMBD, "D41MissJob layout");

// A cached expert's assignments in a step (row * D41_USED + rank), so its weights are decoded once for all of them. slot
// is its place in its class, cls the class's row in the layer's table: 0 part A or whole, 1 part B. A padded row and a
// missed expert have none. The first three fields are MoeGroup's (shared/moe_tile.h).
typedef struct {
  i32 slot;
  u32 n;                  // assignments, 1 .. STEP_MAX_ROWS
  u32 asg[STEP_MAX_ROWS];
  u32 cls;
  u32 pad;
} D41Group;
SHARED_ASSERT(sizeof(D41Group) == 48, "D41Group layout");

// An expert's cut: part A is gate and up rows 0-1,279 and down's blocks 0-4, part B the rest (ADR-0074 item 2).
#define D41_PART_A 1280
#define D41_PART_B 1024

// Lanes a row and rows a sub-group of kernels/quant/moe.cl's class kernels, by type t and largest group a: the shortest
// time on cold parts in a launch of 6 a mixed parts
// (docs/measurements/MEASUREMENTS.md#2026-11-02-deepseek41-moe-step-sweep).
#define D41_GU_LANES 16
#define D41_GU_ROWS(a) ((a) == 1 ? 2 : 1)
#define D41_DOWN_LANES(t, a) ((t) == QT_IQ1_S && (a) == 1 ? 16 : 8)
#define D41_DOWN_ROWS 4

// A slot class of a layer on a card: where its slots lie in the card's expert cache and which part they hold, read by
// kernels/quant/moe.cl for each group. The layer's table has a row for each class D41Group.cls names.
typedef struct {
  u64 base;         // the first slot, in bytes from the cache's start
  u64 slot_bytes;
  u64 up_off;       // up and down, in bytes from a slot's start; gate is at the start
  u64 down_off;
  u32 rows;         // gate's and up's rows, and down's K
  u32 pad[3];
} D41Class;
SHARED_ASSERT(sizeof(D41Class) == 48, "D41Class layout");
#define D41_NO_STEP 0xffffffffu  // a MissJob's seq before its step is raised

// ds_wait's bounds, whichever comes first: device-clock ticks, which are core cycles, and polls (ADR-0074 item 5).
#define D41_WAIT_TICKS 2400000000u
#define D41_WAIT_POLLS (1u << 30)
// A list's bound on its waits, from its first polling wait: the ticks of 4.5 s at 1,200 MHz, so inside xe's 5 s job
// timeout from that clock up (ADR-0074 item 5).
#define D41_LIST_TICKS 5400000000ul
// ds_exch_send's sub-groups a work-group; each stores D41_EMBD / D41_EX_SGS values a row, a multiple of 64. The fewest
// that reach the send's shortest time (docs/measurements/MEASUREMENTS.md#2026-11-02-deepseek41-exchange).
#define D41_EX_SGS 16

// Sub-groups a work-group of ds_swiglu_clamp_a8 and ds_moe_combine and a row of ds_router, whose scoring is bound by its
// threads' instructions: the fewest that reach each one's shortest time
// (docs/measurements/MEASUREMENTS.md#2026-11-02-deepseek41-router).
#define D41_RT_SGS(W) ((W) <= 2 ? 24 : (W) <= 4 ? 12 : 6)
// ffn_gate_inp . x by mv_xmx_bf16 on the stream's bf16 form, whole on both cards: K split over this many sub-groups, the
// shortest time with cold weights (docs/measurements/MEASUREMENTS.md#2026-11-02-deepseek41-router-product).
#define D41_RT_KS 8
#define D41_SWIGLU_SGS 20
#define D41_CMB_SGS(W) ((W) <= 6 ? 16 : 8)

// The stream kernels' work-group: one row, in D41_HC_SGS sub-groups of 16, each with D41_EMBD / D41_HC_SGS values in
// chunks of 64. An a8 unit is 32 values, so a sub-group must hold whole units, and 64 sub-groups (1,024) would not.
#define D41_HC_SGS 40
// ds_hc_mix's sub-groups a work-group: its threads wait on reads, so the most that the local sums allow.
#define D41_HC_KSPLIT 32
#define D41_A8_ROW (D41_EMBD + D41_EMBD / 8)  // bytes of a row of x as a8 (kernels/include/a8.h)

// One row's mixes for one sublayer. comb[src * D41_HC + dst] weighs stream src into stream dst.
typedef struct {
  float pre[D41_HC];
  float post[D41_HC];
  float comb[D41_HC * D41_HC];
} D41HcMix;
SHARED_ASSERT(sizeof(D41HcMix) == 4 * D41_MIX, "D41HcMix layout");

// What a card sends the other for one row at an Engram layer, in the other card's memory or pinned host memory; a
// card also keeps its own gates in one. Each gate fills its 64 bytes, since every store across PCIe is that wide.
// value is indexed by value row; only the sender's rows are written.
typedef struct {
  float gate[D41_HC][16];
  float value[D41_EMBD];
} D41EngramSend;
SHARED_ASSERT(sizeof(D41EngramSend) == 256 + 4 * D41_EMBD, "D41EngramSend layout");

#endif
