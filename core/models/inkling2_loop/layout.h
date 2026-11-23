// Card slices, KV page addressing, SWA ring capacity, and short-convolution state geometry.
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include "models/inkling2_loop/kernel_sizes.h"
#include "models/inkling2_loop/shape.h"
#include "step.h"

namespace engine::inkling2_loop {

#define KV_NO_PAGE UINT32_MAX

struct Partition {
  uint32_t first = 0;
  uint32_t count = 0;
};

inline Partition partition(uint32_t total, uint32_t ncards, uint32_t card) {
  const uint32_t base = total / ncards;
  const uint32_t rem = total % ncards;
  return {card * base + std::min(card, rem), base + (card < rem ? 1u : 0u)};
}

struct CardSlice {
  uint32_t c = 0;
  uint32_t C = 1;
  uint32_t kv0 = 0;
  uint32_t nkv = 0;
  uint32_t q0 = 0;
  uint32_t nq = 0;
  uint32_t ff0 = 0;
  uint32_t ff_n = 0;
  uint32_t vocab0 = 0;
  uint32_t vocab_n = 0;

  uint32_t q_n(const Shape &s) const { return nq * s.head_d; }
  uint32_t kv_n(const Shape &s) const { return nkv * s.head_d; }
  uint32_t rel_n(const Shape &s) const { return nq * s.rel_d; }
};

inline std::vector<CardSlice> card_slices(const Shape &s, uint32_t ncards, uint32_t ff_align = 32) {
  std::vector<CardSlice> out(ncards);
  const uint32_t gqa = s.gqa_ratio();
  const uint32_t ff_blocks = s.ff / ff_align;
  for (uint32_t c = 0; c < ncards; c++) {
    const Partition kv = partition(s.head_kv, ncards, c);
    const Partition ffb = partition(ff_blocks, ncards, c);
    const Partition voc = partition(s.vocab_unpadded, ncards, c);
    out[c] = {c, ncards, kv.first, kv.count, kv.first * gqa, kv.count * gqa, ffb.first * ff_align, ffb.count * ff_align,
              voc.first, voc.count};
  }
  return out;
}

inline uint64_t kv_cell(uint32_t page, uint32_t cell_in_page) {
  return (uint64_t)page * KV_PAGE_CELLS + (cell_in_page % KV_PAGE_CELLS);
}

// Ring capacity for SWA layers: enough headroom over sliding_window for a speculative window pass without overwriting
// live context before commit (mirrors SWA_RING_EXTRA in engine/models/inkling2_loop/memory.py).
inline uint32_t swa_ring_cap(const Shape &s) { return s.sliding_window + 64; }

// Short-convolution state channels kept per logical layer slot on a card: this card's K and V heads, then the
// post-attention and post-FFN residual projections (replicated).
inline uint32_t conv_channels(const Shape &s, uint32_t nkv) { return 2 * nkv * s.head_d + 2 * s.embd; }

inline uint32_t conv_hist(const Shape &s) { return s.conv - 1; }

// Bytes of K or V per KV head per cell (FP16).
inline uint64_t kv_head_bytes(const Shape &s) { return (uint64_t)s.head_d * 2; }

}  // namespace engine::inkling2_loop
