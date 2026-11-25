// A slot's fixed state as one block of bytes, for checkpoints (docs/DESIGN.md, section 4): per card, the DeltaNet
// layers' convolution rings and states, and the PLE history. The KV stays in its pages, which the caller keeps for
// the checkpoint and maps into the slot it restores to; the caller also keeps the token history. Then the writes to
// the page table, copies between pages, and whole pages to and from host memory for the disk tier.
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

#include "models/qwen4exp/model_impl.h"

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
  if (kv_res_) kvs_to_[slot] = 0;  // the slot's new pages have no resident copy
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
  if (kv_res_) std::fill(kvs_to_.begin(), kvs_to_.end(), 0);  // `to` loses its resident copy
  const uint32_t none = KV_NO_PAGE;
  for (auto &card : cards_) {
    const uint32_t C = card->C;
    const uint64_t kvb = (uint64_t)card->nkv * shape_.kv_head_bytes();
    const uint64_t ikb = shape_.idx_key_bytes();
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
        card->up->copy(L.raw + kv_cell(to / C, 0) * ikb, L.raw + kv_cell(from / C, 0) * ikb, cells * ikb);
    }
    card->up->synchronize();
  }
}

// Calls f(pointer, bytes, host) for each run of pages of the pool on a card, in the order of a page file's group
// (engine_save_pages): per attention layer, K then V of every page, then the raw and pooled indexer keys of the
// card's own pages (p % C == c, at local page p / C). `host` is true when the run lives in host memory (streamed K/V).
template <class F>
static void for_page_runs(Model::Card &card, const Shape &shape, uint32_t C, bool streamed, const uint32_t *pages,
                          uint32_t n, F &&f) {
  const uint64_t kvb = (uint64_t)KV_PAGE_CELLS * card.nkv * shape.kv_head_bytes();
  const uint64_t rawb = (uint64_t)KV_PAGE_CELLS * shape.idx_key_bytes(), poolb = (uint64_t)shape.idx_d * 4;
  std::vector<uint32_t> own;
  for (uint32_t i = 0; i < n; i++)
    if (pages[i] % C == card.c) own.push_back(pages[i] / C);
  // f over runs of ids in ids[0..m), each id a unit of `unit` bytes at base.
  auto runs = [&](uint8_t *base, uint64_t unit, const uint32_t *ids, size_t m, bool host) {
    for (size_t i = 0; i < m;) {
      size_t j = i + 1;
      while (j < m && ids[j] == ids[j - 1] + 1) j++;
      f(base + ids[i] * unit, (j - i) * unit, host);
      i = j;
    }
  };
  for (auto &L : card.layers) {
    if (!L.attn) continue;
    runs(streamed ? L.khost : L.kc, kvb, pages, n, streamed);
    runs(streamed ? L.vhost : L.vc, kvb, pages, n, streamed);
    runs(L.raw, rawb, own.data(), own.size(), false);
    runs(reinterpret_cast<uint8_t *>(L.pooled), poolb, own.data(), own.size(), false);
  }
}

void Model::kv_back(uint32_t pages) {
  if (share_ && pages > kv_want_) kv_need(pages, ring_want_);
}

uint64_t Model::kv_page_bytes() const {
  uint64_t n = 0;
  for (auto &card : cards_)
    for (auto &L : card->layers)
      if (L.attn) n += 2ull * KV_PAGE_CELLS * card->nkv * shape_.kv_head_bytes();
  // Each page's indexer keys are on one card.
  return n + (uint64_t)shape_.attn_layers() * (KV_PAGE_CELLS * shape_.idx_key_bytes() + shape_.idx_d * 4ull);
}

void Model::read_pages(const uint32_t *pages, uint32_t n, uint8_t *dst) {
  for (uint32_t i = 0; i < n; i++)
    if (pages[i] >= kv_pages_) throw std::runtime_error("page " + std::to_string(pages[i]) + " out of the pool");
  if (n == 0) return;
  wait_commit();
  std::vector<bool> queued(cards_.size(), false);
  for (size_t c = 0; c < cards_.size(); c++) {
    Card &card = *cards_[c];
    ze::List &list = *card.page_copy;
    list.reset_list();
    for_page_runs(card, shape_, ncards_, kv_res_ != 0, pages, n, [&](uint8_t *p, uint64_t b, bool host) {
      if (host) {
        std::memcpy(dst, p, b);
      } else {
        list.copy(dst, p, b);
        queued[c] = true;
      }
      dst += b;
    });
    if (queued[c]) {
      list.close();
      list.execute();
    }
  }
  for (size_t c = 0; c < cards_.size(); c++)
    if (queued[c]) cards_[c]->page_copy->synchronize();
}

void Model::write_pages(const uint32_t *pages, uint32_t n, const uint8_t *src) {
  uint32_t top = 0;
  for (uint32_t i = 0; i < n; i++) {
    if (pages[i] >= kv_pages_) throw std::runtime_error("page " + std::to_string(pages[i]) + " out of the pool");
    top = std::max(top, pages[i] + 1);
  }
  if (n == 0) return;
  wait_commit();
  kv_back(top);
  if (kv_res_) std::fill(kvs_to_.begin(), kvs_to_.end(), 0);  // the pages lose their resident copies
  static_assert(KV_NO_PAGE == 0xffffffffu);
  const uint8_t none = 0xff;
  std::vector<bool> queued(cards_.size(), false);
  for (size_t c = 0; c < cards_.size(); c++) {
    Card &card = *cards_[c];
    ze::List &list = *card.page_copy;
    list.reset_list();
    for_page_runs(card, shape_, ncards_, kv_res_ != 0, pages, n, [&](uint8_t *p, uint64_t b, bool host) {
      if (host) {
        std::memcpy(p, src, b);
      } else {
        list.copy(p, src, b);
        queued[c] = true;
      }
      src += b;
    });
    if (kv_res_)
      for (auto &L : card.layers) {
        if (!L.attn) continue;
        for (uint32_t i = 0; i < n;) {
          uint32_t j = i + 1;
          while (j < n && pages[j] == pages[j - 1] + 1) j++;
          list.fill(L.where + pages[i], &none, 1, (size_t)(j - i) * 4);
          queued[c] = true;
          i = j;
        }
      }
    if (queued[c]) {
      list.close();
      list.execute();
    }
  }
  for (size_t c = 0; c < cards_.size(); c++)
    if (queued[c]) cards_[c]->page_copy->synchronize();
}

}  // namespace engine::qwen4exp
