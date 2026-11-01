// deepseek41's sizes, and the layout of what the core and the kernels share: the one definition, read by both (C++20
// and OpenCL C). The model and its file are ADR-0062's; the layers are DeepSeek's inference/model.py for
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

#define D41_EXPERTS 384
#define D41_USED 6
#define D41_FF 2304                    // expert and shared-expert width
#define D41_ROUTE_SCALE 1.5f
#define D41_SWIGLU_LIMIT 10.0f         // min(g, L) and clamp(u, -L, L), routed and shared experts

#define D41_VOCAB 129280
#define D41_ENG_NGRAM 4                // a token and the three before it
#define D41_ENG_HEADS 8                // heads an n-gram size
#define D41_ENG_COLS ((D41_ENG_NGRAM - 1) * D41_ENG_HEADS)  // 24 rows a token and Engram layer
#define D41_ENG_ROW 256                // values a row: one Q5_K block, 176 B
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

// The decode step's doorbell with the CPU (ADR-0011): one work-item polls with a system-scope acquire fence, then a
// volatile load, and gives up after 2.4e9 device-clock ticks, setting the stuck flag (ADR-0062 item 5). The posting
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

#endif
