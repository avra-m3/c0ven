// The KV pool and the expert cache share each card's memory (Options::share_kv, ADR-0045). The pool's buffers and
// each layer's cache are ranges of addresses that never move, backed by 2 MB blocks of device memory that do. The
// caller takes the lowest free page first; the blocks of the pages above those in use serve the cache. A page that
// needs its block takes one from the layers whose cached experts are used least; a freed block refills the cache
// with the most used left out, in the swap rounds. The cards always keep room for the whole pool. Each group's cards
// share their own memory between the KV of the group's layers and the group's caches.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "model_impl.h"

namespace engine::qwen4exp {

using namespace detail;

namespace {

uint64_t cdiv(uint64_t x, uint64_t a) { return (x + a - 1) / a; }

}  // namespace

// Region x of the KV pool on a card: addresses for the whole pool, its fixed part backed (load_cards' warm-up runs a
// prompt chunk in slot 0's first pages). share_cards backs the rest as needed. unit: bytes a page, or a local page
// of the card's own pages when div is the cards.
uint8_t *Model::kv_region(Card &card, uint32_t x, uint64_t unit, uint32_t div) {
  if (kv_unit_.size() <= x) {
    kv_unit_.resize(x + 1);
    kv_div_.resize(x + 1);
  }
  if (kv_g_.size() <= x) kv_g_.resize(x + 1);
  kv_unit_[x] = unit;
  kv_div_[x] = div;
  kv_g_[x] = card.g;
  if (card.kv.size() != x) throw std::logic_error("kv_region: regions out of order");
  const uint64_t whole = cdiv(cdiv(kv_pages_, div) * unit, blk_) * blk_;
  Region r;
  r.va = static_cast<uint8_t *>(rt_->vm_reserve(card.dev, whole));
  r.fixed = std::min(whole, cdiv(cdiv(tmax_ / KV_PAGE_CELLS + 1, div) * unit, blk_) * blk_);
  rt_->vm_map(r.va, r.fixed, rt_->vm_create(card.dev, r.fixed));
  card.kv.push_back(std::move(r));
  return card.kv.back().va;
}

// Region x on the first card of the group that holds it.
const Model::Region &Model::kv_home(uint32_t x) const { return group_card(kv_g_[x], 0).kv[x]; }

// The pages region x backs with its fixed part and `blocks` blocks.
uint32_t Model::kv_pages_of(uint32_t x, size_t blocks) const {
  const uint64_t bytes = kv_home(x).fixed + blocks * blk_;
  return (uint32_t)std::min<uint64_t>(kv_pages_, bytes / kv_unit_[x] * kv_div_[x]);
}

// The blocks region x needs past its fixed part for the pages [0, pages).
size_t Model::kv_blocks_for(uint32_t x, uint32_t pages) const {
  const uint64_t bytes = cdiv(pages, kv_div_[x]) * kv_unit_[x], fixed = kv_home(x).fixed;
  return bytes > fixed ? cdiv(bytes - fixed, blk_) : 0;
}

// The slots layer l's cache holds with its fixed part and `blocks` blocks.
uint32_t Model::cache_slots(uint32_t l, size_t blocks) const {
  return (uint32_t)std::min<uint64_t>(Q_EXPERTS,
                                      (home(l).cache[l].fixed + blocks * blk_) / home(l).layers[l].slot_bytes);
}

// The load's split of each card. Fixed parts: the pool's (kv_region) and n_min experts a layer, n_min the most that
// leave every group room for the rest of its pool and the staging of the experts they leave out; the other blocks
// move. first: the cache takes every block the pool's fixed part leaves; after a sleep, the cards hold the experts and
// back the pages they did. budget: each group's room. count: each layer's cached experts, in slots 0 .. count - 1.
void Model::share_cards(bool first, const std::vector<uint64_t> &budget, const std::vector<uint64_t> &stg_bytes,
                        std::vector<uint32_t> &count) {
  std::vector<uint64_t> sb(Q_LAYERS);
  for (uint32_t l = 0; l < Q_LAYERS; l++) sb[l] = cards_[0]->layers[l].slot_bytes;
  const uint32_t nx = (uint32_t)kv_unit_.size();
  auto mine = [&](uint32_t l, uint32_t g) { return home(l).g == g; };
  auto staging = [&](uint32_t g, uint64_t n) {
    uint64_t mx = 0;
    for (uint32_t l = 0; tmax_ && l < Q_LAYERS; l++)
      if (mine(l, g)) mx = std::max(mx, (Q_EXPERTS - n) * stg_bytes[l]);
    return 2 * mx;
  };
  auto kv_blocks = [&](uint32_t g, uint32_t pages) {
    size_t b = 0;
    for (uint32_t x = 0; x < nx; x++)
      if (kv_g_[x] == g) b += kv_blocks_for(x, pages);
    return b;
  };
  auto fits = [&](uint64_t n) {
    for (uint32_t g = 0; g < ngroups_; g++) {
      uint64_t b = kv_blocks(g, kv_pages_);
      for (uint32_t l = 0; l < Q_LAYERS; l++)
        if (mine(l, g)) b += cdiv(n * sb[l], blk_);
      if (b * blk_ + staging(g, n) > budget[g]) return false;
    }
    return true;
  };
  uint64_t n_min = nslots_cached_;
  if (first)
    while (n_min < Q_EXPERTS && fits(n_min + 1)) n_min++;
  if (n_min == 0 || !fits(n_min))
    throw std::runtime_error(first ? "the cards have no room for the KV pool and an expert cache"
                                   : "the cards no longer have room for the KV pool and the cache they held before the sleep");
  nslots_cached_ = (uint32_t)n_min;
  std::vector<uint64_t> fixed(Q_LAYERS);
  std::vector<int64_t> free(ngroups_);
  for (uint32_t g = 0; g < ngroups_; g++) free[g] = (int64_t)((budget[g] - staging(g, n_min)) / blk_);
  for (uint32_t l = 0; l < Q_LAYERS; l++) {
    fixed[l] = cdiv(n_min * sb[l], blk_) * blk_;
    free[home(l).g] -= (int64_t)(fixed[l] / blk_);
  }
  auto blocks_for = [&](uint32_t l, uint64_t n) -> size_t {
    return n * sb[l] > fixed[l] ? cdiv(n * sb[l] - fixed[l], blk_) : 0;
  };
  if (first) {
    kv_own_ = kv_pages_;
    for (uint32_t x = 0; x < nx; x++) kv_own_ = std::min(kv_own_, kv_pages_of(x, 0));
  }
  std::vector<size_t> dyn(Q_LAYERS);
  for (uint32_t g = 0; g < ngroups_; g++) {
    const size_t kv_now = kv_blocks(g, kv_own_);
    if (first) {
      // Every layer of the group the same count, the most the blocks give.
      auto need = [&](uint64_t n) {
        size_t b = kv_now;
        for (uint32_t l = 0; l < Q_LAYERS; l++)
          if (mine(l, g)) b += blocks_for(l, n);
        return b;
      };
      uint64_t n = n_min;
      while (n < Q_EXPERTS && (int64_t)need(n + 1) <= free[g]) n++;
      for (uint32_t l = 0; l < Q_LAYERS; l++)
        if (mine(l, g)) dyn[l] = blocks_for(l, n);
    } else {
      for (uint32_t l = 0; l < Q_LAYERS; l++)
        if (mine(l, g)) dyn[l] = blocks_for(l, count[l]);
    }
    int64_t spare = free[g] - (int64_t)kv_now;
    for (uint32_t l = 0; l < Q_LAYERS; l++)
      if (mine(l, g)) spare -= (int64_t)dyn[l];
    if (spare < 0)
      throw std::runtime_error("the cards no longer have room for the KV pool and the cache they held before the sleep");
  }
  for (auto &card : cards_) {
    for (int64_t i = 0; i < free[card->g]; i++) card->spare.push_back(rt_->vm_create(card->dev, blk_));
    card->cache.resize(Q_LAYERS);
    for (uint32_t l = card->l0; l < card->l1; l++) {
      Region &r = card->cache[l];
      r.va = static_cast<uint8_t *>(rt_->vm_reserve(card->dev, cdiv(Q_EXPERTS * sb[l], blk_) * blk_));
      r.fixed = fixed[l];
      rt_->vm_map(r.va, r.fixed, rt_->vm_create(card->dev, r.fixed));
      card->layers[l].cache = r.va;
      map_cache(*card, l, dyn[l]);
    }
  }
  grow_kv(kv_own_);
  for (uint32_t l = 0; l < Q_LAYERS; l++) {
    const uint32_t s = cache_slots(l, dyn[l]);
    count[l] = first ? s : std::min(count[l], s);
  }
}

// Layer l's cache on a card at `blocks` blocks past its fixed part: blocks from the spare ones, or back to them.
void Model::map_cache(Card &card, uint32_t l, size_t blocks) {
  Region &r = card.cache[l];
  while (r.blocks.size() < blocks) {
    if (card.spare.empty()) throw std::logic_error("map_cache: no spare block");
    r.blocks.push_back(card.spare.back());
    card.spare.pop_back();
    rt_->vm_map(r.va + r.fixed + (r.blocks.size() - 1) * blk_, blk_, r.blocks.back());
  }
  while (r.blocks.size() > blocks) {
    rt_->vm_unmap(r.va + r.fixed + (r.blocks.size() - 1) * blk_, blk_);
    card.spare.push_back(r.blocks.back());
    r.blocks.pop_back();
  }
}

// The pages [0, pages) backed on every card, from the spare blocks.
void Model::grow_kv(uint32_t pages) {
  pages = std::min(pages, kv_pages_);
  for (auto &card : cards_)
    for (uint32_t x = 0; x < card->kv.size(); x++) {
      Region &r = card->kv[x];
      if (!r.va) continue;  // another group's
      while (r.blocks.size() < kv_blocks_for(x, pages)) {
        if (card->spare.empty()) throw std::logic_error("grow_kv: no spare block");
        r.blocks.push_back(card->spare.back());
        card->spare.pop_back();
        rt_->vm_map(r.va + r.fixed + (r.blocks.size() - 1) * blk_, blk_, r.blocks.back());
      }
    }
  kv_own_ = kv_pages_;
  for (uint32_t x = 0; x < kv_unit_.size(); x++) kv_own_ = std::min(kv_own_, kv_pages_of(x, kv_home(x).blocks.size()));
}

// The blocks of the pages from `pages` up back to the spare ones: the caller has nothing in them.
void Model::shrink_kv(uint32_t pages) {
  const auto t0 = std::chrono::steady_clock::now();
  pages = std::min(pages, kv_pages_);
  for (auto &card : cards_)
    for (uint32_t x = 0; x < card->kv.size(); x++) {
      Region &r = card->kv[x];
      if (!r.va) continue;
      while (r.blocks.size() > kv_blocks_for(x, pages)) {
        rt_->vm_unmap(r.va + r.fixed + (r.blocks.size() - 1) * blk_, blk_);
        card->spare.push_back(r.blocks.back());
        r.blocks.pop_back();
      }
    }
  kv_own_ = kv_pages_;
  for (uint32_t x = 0; x < kv_unit_.size(); x++) kv_own_ = std::min(kv_own_, kv_pages_of(x, kv_home(x).blocks.size()));
  log_share("the KV gave memory back", t0);
}

// The blocks to back the pages [0, pages): the spare ones, then each next block from the layer whose experts it
// would put out are the coldest, by the routing counts, each group from its own layers. The victims' pieces are
// filled before the blocks move (start_round, or kv_need at once); apply_evict moves them. With enough spare blocks,
// the pool grows now.
void Model::plan_evict(uint32_t pages) {
  pages = std::min(pages, kv_pages_);
  std::vector<size_t> need(ngroups_, 0), want(ngroups_, 0);
  for (uint32_t x = 0; x < kv_unit_.size(); x++) {
    const size_t w = kv_blocks_for(x, pages), have = kv_home(x).blocks.size();
    need[kv_g_[x]] += w > have ? w - have : 0;
  }
  bool short_ = false;
  for (uint32_t g = 0; g < ngroups_; g++) {
    const size_t spare = group_card(g, 0).spare.size();
    want[g] = need[g] > spare ? need[g] - spare : 0;
    short_ |= want[g] > 0;
  }
  if (!short_) {
    grow_kv(pages);
    return;
  }
  struct Side {
    std::vector<uint32_t> cold;  // cached experts, coldest first
    size_t blocks = 0, take = 0;
  };
  std::vector<Side> side(Q_LAYERS);
  for (uint32_t l = 0; l < Q_LAYERS; l++) {
    const HostLayer &hl = *host_layers_[l];
    const float *u = &use_[(size_t)l * Q_EXPERTS];
    for (uint32_t e = 0; e < Q_EXPERTS; e++)
      if (hl.slot_of[e] >= 0) side[l].cold.push_back(e);
    std::sort(side[l].cold.begin(), side[l].cold.end(),
              [&](uint32_t a, uint32_t b) { return u[a] != u[b] ? u[a] < u[b] : a > b; });
    side[l].blocks = home(l).cache[l].blocks.size();
  }
  // Experts layer l puts out when it gives up t blocks: those its slots no longer hold.
  auto out = [&](uint32_t l, size_t t) -> size_t {
    const size_t c = side[l].cold.size(), s = cache_slots(l, side[l].blocks - t);
    return c > s ? c - s : 0;
  };
  // Each group's blocks from its own layers.
  for (uint32_t g = 0; g < ngroups_; g++) {
    for (; want[g] > 0; want[g]--) {
      int best = -1;
      float best_cost = 0;
      for (uint32_t l = 0; l < Q_LAYERS; l++) {
        Side &sd = side[l];
        if (home(l).g != g || sd.take == sd.blocks) continue;
        float cost = 0;
        for (size_t i = out(l, sd.take); i < out(l, sd.take + 1); i++) cost += use_[(size_t)l * Q_EXPERTS + sd.cold[i]];
        // The cheapest; between equals, the layer with the most experts on the cards.
        if (best < 0 || cost < best_cost ||
            (cost == best_cost &&
             sd.cold.size() - out(l, sd.take) > side[best].cold.size() - out(best, side[best].take)))
          best = (int)l, best_cost = cost;
      }
      if (best < 0) throw std::logic_error("plan_evict: the caches have no blocks left");
      side[best].take++;
    }
  }
  for (uint32_t l = 0; l < Q_LAYERS; l++) {
    if (!side[l].take) continue;
    const size_t k = out(l, side[l].take);
    const HostLayer &hl = *host_layers_[l];
    if (hl.uncached.size() + k > hl.pieces.size()) throw std::logic_error("plan_evict: no staging position left");
    evict_next_.push_back({l, (uint32_t)side[l].take, (uint32_t)hl.uncached.size(),
                           std::vector<uint32_t>(side[l].cold.begin(), side[l].cold.begin() + (long)k)});
  }
  evict_to_ = pages;
}

void Model::fill_evict(Pool &pool) {
  std::vector<std::pair<const Evict *, uint32_t>> v;
  for (const Evict &ev : evict_next_)
    for (uint32_t k = 0; k < ev.victims.size(); k++) v.push_back({&ev, k});
  pool.run(v.size(), [&](uint64_t i) {
    const Evict &ev = *v[i].first;
    fill_piece(ev.l, ev.victims[v[i].second], host_layers_[ev.l]->pieces[ev.first + v[i].second]);
  });
  for (const Evict &ev : evict_next_)
    for (uint32_t e : ev.victims) release_expert(ev.l, e);
}

// The planned eviction, its pieces filled: the experts in the slots a layer gives up move into the slots below that
// its victims or nothing hold, the victims join the staging, the blocks go to the pool.
void Model::apply_evict(std::chrono::steady_clock::time_point t0) {
  for (const Evict &ev : evict_next_) {
    HostLayer &hl = *host_layers_[ev.l];
    const uint32_t g = home(ev.l).g;
    const size_t blocks = home(ev.l).cache[ev.l].blocks.size();
    const uint32_t old = cache_slots(ev.l, blocks), now = cache_slots(ev.l, blocks - ev.take);
    std::vector<int32_t> at(old, -1);
    for (uint32_t e = 0; e < Q_EXPERTS; e++)
      if (hl.slot_of[e] >= 0) at[hl.slot_of[e]] = (int32_t)e;
    std::vector<bool> victim(Q_EXPERTS, false);
    for (uint32_t e : ev.victims) victim[e] = true;
    std::vector<uint32_t> holes;
    for (uint32_t s = 0; s < now; s++)
      if (at[s] < 0 || victim[at[s]]) holes.push_back(s);
    std::vector<std::pair<uint32_t, uint32_t>> moves;  // slot from, slot to
    for (uint32_t s = now, h = 0; s < old; s++)
      if (at[s] >= 0 && !victim[at[s]]) moves.push_back({s, holes.at(h++)});
    for (uint32_t r = 0; r < width_; r++) {
      Card &card = group_card(g, r);
      Layer &L = card.layers[ev.l];
      for (auto [from, to] : moves)
        card.up->copy(L.cache + (uint64_t)to * L.slot_bytes, L.cache + (uint64_t)from * L.slot_bytes, L.slot_bytes);
    }
    for (uint32_t r = 0; r < width_; r++) group_card(g, r).up->synchronize();
    for (auto [from, to] : moves) hl.slot_of[at[from]] = (int32_t)to;
    for (uint32_t k = 0; k < ev.victims.size(); k++) {
      const uint32_t e = ev.victims[k], pos = ev.first + k;
      if (pos != hl.uncached.size()) throw std::logic_error("apply_evict: staging out of step");
      uint8_t *p = hl.pieces[pos];
      hl.slot_of[e] = -1;
      hl.stage_of[e] = (int32_t)pos;
      hl.uncached.push_back(e);
      hl.arena[e] = p;
      hl.weights[e] = {p, p + hl.gate_bytes, p + 2 * hl.gate_bytes, hl.down_parts};
    }
    for (uint32_t r = 0; r < width_; r++) {
      Card &card = group_card(g, r);
      card.up->copy(card.layers[ev.l].slot_map, hl.slot_of.data(), Q_EXPERTS * 4);
      if (card.pc && !card.pc->stage_map.empty())
        card.up->copy(card.pc->stage_map[ev.l], hl.stage_of.data(), Q_EXPERTS * 4);
    }
    for (uint32_t r = 0; r < width_; r++) {
      Card &card = group_card(g, r);
      card.up->synchronize();
      map_cache(card, ev.l, blocks - ev.take);
    }
  }
  if (!evict_next_.empty()) copies_dirty_ = true;
  evict_next_.clear();
  grow_kv(evict_to_);
  log_share("the cache gave memory to the KV", t0);
}

// Up to max experts back into the caches, the hottest left out first: into a layer's empty slots, or the slots of a
// spare block. apply_round maps the blocks and copies them in, as for swaps.
void Model::plan_refills(uint32_t max) {
  std::vector<size_t> spare(ngroups_);  // each group's
  for (uint32_t g = 0; g < ngroups_; g++) spare[g] = group_card(g, 0).spare.size();
  struct Side {
    std::vector<uint32_t> hot;   // uncached experts, hottest first
    std::vector<int32_t> empty;  // empty slots, the lowest last
    size_t blocks = 0, next = 0;
  };
  std::vector<Side> side(Q_LAYERS);
  for (uint32_t l = 0; l < Q_LAYERS; l++) {
    const HostLayer &hl = *host_layers_[l];
    const float *u = &use_[(size_t)l * Q_EXPERTS];
    Side &sd = side[l];
    sd.hot = hl.uncached;
    std::sort(sd.hot.begin(), sd.hot.end(), [&](uint32_t a, uint32_t b) { return u[a] != u[b] ? u[a] > u[b] : a < b; });
    sd.blocks = home(l).cache[l].blocks.size();
    const uint32_t n = cache_slots(l, sd.blocks);
    std::vector<bool> used(n, false);
    for (uint32_t e = 0; e < Q_EXPERTS; e++)
      if (hl.slot_of[e] >= 0) used[hl.slot_of[e]] = true;
    for (uint32_t s = n; s-- > 0;)
      if (!used[s]) sd.empty.push_back((int32_t)s);
  }
  for (uint32_t done = 0; done < max; done++) {
    int best = -1;
    for (uint32_t l = 0; l < Q_LAYERS; l++) {
      const Side &sd = side[l];
      if (sd.next == sd.hot.size()) continue;
      if (sd.empty.empty() && (spare[home(l).g] == 0 || cache_slots(l, sd.blocks + 1) == cache_slots(l, sd.blocks)))
        continue;
      if (best < 0 || use_[(size_t)l * Q_EXPERTS + sd.hot[sd.next]] >
                          use_[(size_t)best * Q_EXPERTS + side[best].hot[side[best].next]])
        best = (int)l;
    }
    if (best < 0) break;
    Side &sd = side[best];
    if (sd.empty.empty()) {
      const uint32_t s0 = cache_slots(best, sd.blocks), s1 = cache_slots(best, sd.blocks + 1);
      sd.blocks++;
      spare[home(best).g]--;
      for (uint32_t s = s1; s-- > s0;) sd.empty.push_back((int32_t)s);
    }
    swap_next_.push_back({(uint32_t)best, sd.hot[sd.next++], kNoExpert, nullptr, sd.empty.back()});
    sd.empty.pop_back();
  }
}

// The host's maps after a refill: in takes its slot and leaves the staging; the expert at the last staging position
// takes in's position, its weights moved into in's piece (moves), so the positions stay 0 .. uncached - 1.
void Model::refill_maps(const SwapPlan &w, std::vector<PieceMove> &moves) {
  HostLayer &hl = *host_layers_[w.l];
  const uint32_t pos = (uint32_t)hl.stage_of[w.in], last = (uint32_t)hl.uncached.size() - 1;
  if (pos != last) {
    const uint32_t x = hl.uncached[last];
    uint8_t *p = hl.pieces[pos];
    moves.push_back({p, hl.pieces[last], 2 * hl.gate_bytes + hl.down_bytes});
    hl.arena[x] = p;
    hl.weights[x] = {p, p + hl.gate_bytes, p + 2 * hl.gate_bytes, hl.down_parts};
    hl.stage_of[x] = (int32_t)pos;
    hl.uncached[pos] = x;
  }
  hl.uncached.pop_back();
  hl.stage_of[w.in] = -1;
  hl.arena[w.in] = nullptr;
  hl.weights[w.in] = {};
  hl.slot_of[w.in] = w.slot;
  copies_dirty_ = true;
}

// The staging after the chosen round: its refills as refill_maps leaves it, then its eviction's victims.
Model::Staging Model::staging_after() const {
  Staging st = staging_now();
  auto at = [&](uint32_t l, uint32_t pos) {
    const HostLayer &hl = *host_layers_[l];
    uint8_t *p = hl.pieces[pos];
    return ExpertWeights{p, p + hl.gate_bytes, p + 2 * hl.gate_bytes, hl.down_parts};
  };
  std::vector<std::pair<uint32_t, uint32_t>> refills;  // layer, staging position: the highest first, as apply_round
  for (const SwapPlan &w : swap_next_)
    if (w.out == kNoExpert) refills.push_back({w.l, (uint32_t)host_layers_[w.l]->stage_of[w.in]});
  std::sort(refills.begin(), refills.end(), [](const auto &a, const auto &b) { return a.second > b.second; });
  for (auto [l, pos] : refills) {
    if (pos + 1 != st[l].size()) st[l][pos] = at(l, pos);
    st[l].pop_back();
  }
  for (const Evict &ev : evict_next_)
    for (uint32_t k = 0; k < ev.victims.size(); k++) st[ev.l].push_back(at(ev.l, ev.first + k));
  return st;
}

// After a round that changed the staging: the copy list recorded behind it in place of the one before.
void Model::install_copies() {
  if (next_copies_.empty()) return;
  for (size_t c = 0; copies_dirty_ && c < cards_.size(); c++) std::swap(cards_[c]->pc->copy, next_copies_[c]);
  if (copies_dirty_) copy_count_ = next_copy_count_;
  copies_dirty_ = false;
  for (auto &list : next_copies_) retired_.push_back(std::move(list));
  next_copies_.clear();
}

void Model::kv_need(uint32_t in_use) {
  if (!share_) return;
  kv_want_ = std::min(in_use, kv_pages_);
  if (kv_want_ <= kv_own_) return;
  if (swap_bg_.joinable()) swap_bg_.join();
  apply_round();
  if (kv_want_ <= kv_own_) return;
  const auto t0 = std::chrono::steady_clock::now();
  plan_evict(std::min(kv_pages_, kv_want_ + kv_margin_));
  if (!evict_next_.empty()) {
    fill_evict(*pool_);
    apply_evict(t0);
  }
  if (kv_want_ > kv_own_) throw std::logic_error("kv_need: the pages are not backed");
}

void Model::kv_ahead() {
  if (!share_ || !evict_next_.empty() || kv_own_ >= std::min(kv_pages_, kv_want_ + kv_margin_)) return;
  if (swap_bg_.joinable()) swap_bg_.join();
  if (!swap_next_.empty()) apply_round();
  plan_evict(std::min(kv_pages_, kv_want_ + 2 * kv_margin_));
  if (!evict_next_.empty()) start_round();  // the eviction only: no swaps chosen with it
}

uint32_t Model::experts_cached() const {
  uint64_t sum = 0;
  for (auto &hl : host_layers_) sum += Q_EXPERTS - (uint32_t)hl->uncached.size();
  return (uint32_t)(sum / Q_LAYERS);
}

// A change of the split, on stderr: t0, when given, the time it took from.
void Model::log_share(const char *why, std::chrono::steady_clock::time_point t0) {
  uint32_t lo = Q_EXPERTS, hi = 0;
  uint64_t sum = 0;
  for (auto &hl : host_layers_) {
    const uint32_t n = Q_EXPERTS - (uint32_t)hl->uncached.size();
    lo = std::min(lo, n), hi = std::max(hi, n), sum += n;
  }
  size_t spare = 0;
  for (uint32_t g = 0; g < ngroups_; g++) spare += group_card(g, 0).spare.size();
  char took[32] = "";
  if (t0 != std::chrono::steady_clock::time_point{})
    std::snprintf(took, sizeof took, " in %.1f ms",
                  std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  std::fprintf(stderr,
               "engine: %s%s: KV pages %u backed, %u in use; experts cached a layer %.1f (%u-%u, at least %u); %zu "
               "blocks spare\n",
               why, took, kv_own_, kv_want_, (double)sum / Q_LAYERS, lo, hi, nslots_cached_, spare);
}

}  // namespace engine::qwen4exp
