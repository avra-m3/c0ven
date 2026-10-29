// A slot's fixed state as one block of bytes, for checkpoints (docs/DESIGN.md, section 4): per card, the DeltaNet
// layers' convolution rings and states, and the PLE history. The KV stays in its pages, which the caller keeps for
// the checkpoint and maps into the slot it restores to; the caller also keeps the token history. Then the writes to
// the page table, and copies between pages.
#include <cstring>
#include <stdexcept>
#include <string>

#include "model_impl.h"

namespace engine::qwen4exp {

using namespace detail;

// Calls f(device pointer, bytes) for each region of the slot's fixed state on a card, in a fixed order.
template <class F>
static void for_regions(Model::Card &card, uint32_t slot, F &&f) {
  for (auto &L : card.layers) {
    if (L.attn) continue;
    f(reinterpret_cast<uint8_t *>(L.conv_state + slot * L.conv_stride), L.conv_stride * 4);
    f(reinterpret_cast<uint8_t *>(L.S + slot * L.S_stride), L.S_stride * 4);
  }
  f(reinterpret_cast<uint8_t *>(card.ple_hist + slot * card.ple_stride), card.ple_stride * 4);
}

uint64_t Model::slot_state_bytes() {
  uint64_t n = 0;
  for (auto &card : cards_) for_regions(*card, 0, [&](uint8_t *, uint64_t b) { n += b; });
  return n;
}

void Model::save_slot(uint32_t slot, uint8_t *dst) {
  if (slot >= opt_.slots) throw std::runtime_error("slot out of range");
  wait_commit();
  for (auto &card : cards_) {
    for_regions(*card, slot, [&](uint8_t *p, uint64_t b) {
      card->up->copy(dst, p, b);
      dst += b;
    });
    card->up->synchronize();
  }
}

void Model::load_slot(uint32_t slot, const uint8_t *src) {
  if (slot >= opt_.slots) throw std::runtime_error("slot out of range");
  wait_commit();
  for (auto &card : cards_) {
    for_regions(*card, slot, [&](uint8_t *p, uint64_t b) {
      card->up->copy(p, src, b);
      src += b;
    });
    card->up->synchronize();
  }
}

void *Model::alloc_pinned(size_t bytes) { return rt_->alloc_host(bytes); }
void Model::free_pinned(void *p) { rt_->free(p); }

void Model::set_pages(uint32_t slot, uint32_t first, const uint32_t *pages, uint32_t n) {
  if (slot >= opt_.slots || first > npp_ || n > npp_ - first) throw std::runtime_error("pages out of the slot's table");
  for (uint32_t i = 0; i < n; i++)
    if (pages[i] >= kv_pages_ || pages[i] % ncards_ != (first + i) % ncards_)
      throw std::runtime_error("page " + std::to_string(pages[i]) + " cannot hold logical page " +
                               std::to_string(first + i));
  if (n == 0) return;
  // The immediate list copies from the caller's memory before it returns.
  for (auto &card : cards_) {
    card->up->copy(card->ptab + (size_t)slot * npp_ + first, pages, (size_t)n * 4);
    card->up->synchronize();
  }
}

void Model::copy_cells(uint32_t from, uint32_t to, uint32_t cells) {
  if (from >= kv_pages_ || to >= kv_pages_ || from % ncards_ != to % ncards_ || cells >= KV_PAGE_CELLS)
    throw std::runtime_error("copy between pages out of range");
  if (cells == 0 || from == to) return;
  const uint32_t none = KV_NO_PAGE;
  for (auto &card : cards_) {
    const uint32_t C = card->C;
    const uint64_t kvb = (uint64_t)card->nkv * KV_HEAD_BYTES;
    for (auto &L : card->layers) {
      if (!L.attn) continue;
      if (kv_res_) {  // streamed: the host copies; the resident copy of `to`, if any, is stale from here
        card->up->copy(L.khost + kv_cell(to, 0) * kvb, L.khost + kv_cell(from, 0) * kvb, cells * kvb);
        card->up->copy(L.vhost + kv_cell(to, 0) * kvb, L.vhost + kv_cell(from, 0) * kvb, cells * kvb);
        card->up->copy(L.where + to, &none, 4);
      } else {
        card->up->copy(L.kc + kv_cell(to, 0) * kvb, L.kc + kv_cell(from, 0) * kvb, cells * kvb);
        card->up->copy(L.vc + kv_cell(to, 0) * kvb, L.vc + kv_cell(from, 0) * kvb, cells * kvb);
      }
      // The raw indexer keys live on the page's card. A part of a page has no pooled key.
      if (from % C == card->c)
        card->up->copy(L.raw + kv_cell(to / C, 0) * IDX_KEY_BYTES, L.raw + kv_cell(from / C, 0) * IDX_KEY_BYTES,
                       cells * IDX_KEY_BYTES);
    }
    card->up->synchronize();
  }
}

}  // namespace engine::qwen4exp
