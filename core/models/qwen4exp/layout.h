// qwen4exp's sizes, and the layout of its per-sequence state and KV cache: the one definition, read by the core and
// by the kernels (C++20 and OpenCL C). The model is docs/spec/MODEL_LAYER.md and MODEL_IO.md; how a layer is split
// between the cards is docs/DESIGN.md, section 5.
#ifndef QWEN4EXP_LAYOUT_H
#define QWEN4EXP_LAYOUT_H

#include "step.h"
#include "types.h"

// The kernels see the sizes in kernel_sizes.h. Model code in the core does not, and reads them from Shape (ADR-0064).
#ifdef __OPENCL_C_VERSION__
#include "kernel_sizes.h"
#endif

#define Q_EMBD 2560
#define Q_HC_DIM (Q_HC * Q_EMBD)    // 10,240
#define Q_EPS 1e-6f

#define Q_GDN_D 128                 // head size
#define Q_GDN_CONV 4

#define Q_ATT_H 24                  // query heads
#define Q_ATT_KV 2                  // KV heads; query head h reads KV head h / 12
#define Q_ATT_D 256
#define Q_ATT_SLOTS 16  // p_attn_xmx's query slots a KV head: its 12 heads, rounded up to 8 (prompt_attn.cl)
#define Q_ROT 64                    // rotated dimensions per head, NeoX pairs (i, i + 32)
#define Q_ROPE_BASE 10000000.0f

#define Q_IDX_H 4                   // indexer query heads
#define Q_IDX_D 128
#define Q_IDX_BLOCK 4               // positions per pooled key
#define Q_IDX_TOPK 2048
#define Q_IDX_CELLS (Q_IDX_TOPK + Q_IDX_BLOCK - 1)  // llama.cpp keeps min(n_kv, 2,051) cells
#define Q_IDX_BLOCKS (Q_IDX_TOPK / Q_IDX_BLOCK)       // 512

// Q_EXPERTS is a build option of the modules that route experts (kernels/models/qwen4exp/CMakeLists.txt, ADR-0063).

#define Q_PLE_DIL 3                 // the PLE conv's dilation (the n-gram size)
#define Q_PLE_HIST (Q_PLE_DIL * (Q_GDN_CONV - 1))  // 9 positions of history

// The DFlash 2 drafter (docs/MODEL_FILES.md, "The DFlash 2 drafter"): Qwen3-shaped layers that read the main
// model's hidden states at five layers. Its norms take no +1.
#define DF_LAYERS 5
#define DF_FEAT (DF_LAYERS * Q_EMBD)  // 12,800: the five layers' means of the streams, one after another
#define DF_H 32                       // query heads
#define DF_KV 8                       // KV heads; query head h reads KV head h / 4
#define DF_D 128
#define DF_FF 7680
#define DF_GROUP 16                   // channels sharing a convolution's dynamic weight
#define DF_KP 640                     // a convolution's dynamic weights: 2 sets x 2 taps x 160 groups
#define DF_WINDOW 2048                // a row attends to positions less than this far before it
#define DF_RING 2112                  // K and V cells per slot: the window and a pass's rows past it
#define DF_ROPE_BASE 10000.0f
#define DF_EPS 1e-6f
#define DF_MASK 248077                // the token of a block's rows after its first

// The decode step's doorbell with the CPU (ADR-0011): the reader does a system-scope acquire fence, then a volatile
// load. Card 0 writes one record per layer in host memory, its step last; the CPU computes the misses and answers by
// its own flag. A layer with no miss is skipped by both.
#define MISS_USED 10  // the experts a record holds: its layout does not follow the file
#ifdef __OPENCL_C_VERSION__
#if Q_USED > MISS_USED
#error "a MissJob holds fewer experts than the kernels route"
#endif
#endif
typedef struct {
  u32 seq;                 // the step, written last, behind a release fence
  u32 nmiss;               // experts not on the cards
  i32 ids[MISS_USED];      // the misses' ids, then their weights
  float w[MISS_USED];
  i32 all[MISS_USED];      // every routed expert's id, for the swaps' counts
  float x[Q_EMBD];         // the MoE block's input
} MissJob;
SHARED_ASSERT(sizeof(MissJob) == 8 + 12 * MISS_USED + 4 * Q_EMBD, "MissJob layout");

// A slot's rotation in a decode step (docs/spec/MODEL_VISION.md, section 4.2). With images in the prompt a position is
// three numbers (time, height, width) and no longer the cell: a decode row at cell c turns by c - delta on all three.
// The indexer's block that starts at cell blk turns by (t, h, w), the positions of that cell, which may lie in an
// image; a later block turns by its first cell less delta. Without images delta is 0 and (t, h, w) = (blk, blk, blk).
typedef struct {
  u32 delta;
  u32 blk;
  u32 t, h, w;
  u32 pad[3];
} SlotRope;
SHARED_ASSERT(sizeof(SlotRope) == 32, "SlotRope layout");

// A prompt chunk's positions are (t, h, w, 0) per cell in host memory, from the first cell of the chunk's first block
// (pos0 rounded down to a multiple of 4) to its last row.

// The 8-bit KV cache: Q8_0 blocks of 32 values, 34 bytes (docs/spec/QUANT_FORMATS.md, Q8_0). One head of 256 values
// is 8 blocks, 272 bytes; one indexer key of 128 values is 4 blocks, 136 bytes.
#define Q8_BLOCK 34
#define KV_HEAD_BYTES ((Q_ATT_D / 32) * Q8_BLOCK)
#define IDX_KEY_BYTES ((Q_IDX_D / 32) * Q8_BLOCK)

// The KV cache in pages (docs/DESIGN.md, section 4). A page is 4 cells, one indexer block. Each attention layer of a
// card has one pool of pages; cell 4p + j of the pool is cell j of page p, with K and V for the card's KV heads. A
// slot's row of the page table maps its logical page i (cells 4i .. 4i + 3) to a page of the pool, the same id on
// both cards; Python keeps the table and decides it. The indexer's keys are split by position: with C cards, page p's
// raw and pooled keys live on card p % C at local page p / C, so logical page i maps only to a page with
// p % C == i % C. The draft layer's KV is not paged: its kernels get no table, and slot s's logical page i is page
// s npp + i of its own pool.
SHARED_ASSERT(KV_PAGE_CELLS == Q_IDX_BLOCK, "a page is one indexer block");
#define KV_NO_PAGE 0xffffffffu

// A cell's index in the pool, from the page holding it and its position in the sequence.
SHARED_FN u64 kv_cell(u32 page, u32 pos) { return (u64)page * KV_PAGE_CELLS + pos % KV_PAGE_CELLS; }

#ifdef __OPENCL_C_VERSION__
// The page holding a slot's position: from the slot's row of the table (npp entries), or with no table the slot's own
// run of npp pages.
static inline uint kv_page(__global const uint *pt, uint npp, uint slot, uint pos) {
  uint i = slot * npp + pos / KV_PAGE_CELLS;
  return pt ? pt[i] : i;
}
#endif

#endif
