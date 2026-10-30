// Adaptive swaps (Strata's --adapt-swaps, docs/spec/STRATA_MEMORY.md): the cards' expert cache follows what the
// requests route to. Decode and prompt routing go into decayed per-layer counts; a swap round, between steps, lets
// the uncached experts used most replace the cached ones used least. A swap changes which experts the CPU computes,
// and so their rounding: runs that compare tokens turn swaps off (Options::adapt_swaps, ADR-0016's switches).
#include <algorithm>
#include <chrono>
#include <cstring>
#include <thread>
#include <utility>

#include "load/runs.h"
#include "model_impl.h"

namespace engine::qwen4exp {

using namespace detail;

void Model::note_routing(uint32_t layer, const int32_t *ids, uint32_t n, float weight) {
  float *u = &use_[(size_t)layer * Q_EXPERTS];
  for (uint32_t k = 0; k < n; k++)
    if (ids[k] >= 0 && ids[k] < (int32_t)Q_EXPERTS) u[ids[k]] += weight;
}

// The prompt path's counts of the last chunk (P.prof, kept when swaps are on) into the decayed counts, then zeroed.
void Model::note_prompt_routing() {
  const PromptCard *P = cards_[0]->pc.get();
  if (!P || !P->prof) return;
  std::vector<uint32_t> v(Q_LAYERS * Q_EXPERTS);
  cards_[0]->up->copy(v.data(), P->prof, v.size() * 4);
  const uint32_t z = 0;
  cards_[0]->up->fill(P->prof, &z, 4, v.size() * 4);
  cards_[0]->up->synchronize();
  for (size_t i = 0; i < v.size(); i++) use_[i] += kPromptRouting * (float)v[i];
}

// A round applies the previous round's choice: the incoming experts' halves into their cache slots and the maps,
// waited for once. Behind the next steps, a thread copies the outgoing experts into the arena pieces the incoming ones
// left (settle_moves), and the swap thread chooses the next round's swaps from the counts as they were at this round,
// builds the incoming experts' halves from the arena and reads the outgoing ones from the file into spare pieces.
// With Options::share_kv the round first moves memory between the KV pool and the cache (kv_share.cpp): while the KV
// needs memory or the cache can refill, that is the round's choice instead of swaps, made on the engine thread.
uint32_t Model::swap_round(uint32_t max_swaps) {
  if (!share_ && (nslots_cached_ == 0 || nslots_cached_ >= Q_EXPERTS)) return 0;
  if (swap_bg_.joinable()) swap_bg_.join();
  const uint32_t applied = apply_round();
  bool chosen = false;
  if (share_) {
    const uint32_t m = kv_margin_;
    if (kv_own_ < std::min(kv_pages_, kv_want_ + m) || (kv_res_ && ring_own_ < std::min(kv_res_, ring_want_ + m))) {
      plan_evict(std::min(kv_pages_, kv_want_ + 2 * m), std::min(kv_res_, ring_want_ + 2 * m), true);
      chosen = true;
    } else {
      // The ring gives memory back only once it has been over for kRingIdle: the next request may read as much, and
      // the experts the memory refills would leave again (the idle slots' pages left it already, kvs_drop).
      uint32_t pool = kv_own_, ring = ring_own_;
      if (kv_own_ > kv_want_ + 4 * m) pool = kv_want_ + 2 * m;
      const auto now = std::chrono::steady_clock::now();
      if (kv_res_ && ring_own_ > ring_want_ + 4 * m) {
        if (ring_over_ == std::chrono::steady_clock::time_point{}) ring_over_ = now;
        if (now - ring_over_ >= kRingIdle) ring = ring_want_ + 2 * m;
      } else {
        ring_over_ = {};
      }
      if (pool != kv_own_ || ring != ring_own_) shrink_kv(pool, ring);
      plan_refills(max_swaps);
      chosen = !swap_next_.empty();
    }
  }
  std::vector<float> use;
  if (!chosen) use = use_;
  for (auto &x : use_) x *= 0.8f;  // the counts decay each round
  start_round(std::move(use), chosen ? 0 : max_swaps);
  return applied;
}

// In each layer, the uncached experts used most against the cached ones used least, while a newcomer's count is
// clearly higher; then the round's best max_swaps by margin, each with a spare piece for its outgoing expert. On the
// swap thread: the engine thread changes the maps and the spares only after joining it.
void Model::choose_swaps(const std::vector<float> &use, uint32_t max_swaps) {
  std::vector<std::pair<float, SwapPlan>> all;
  for (uint32_t l = 0; l < Q_LAYERS; l++) {
    HostLayer &hl = *host_layers_[l];
    const float *u = &use[(size_t)l * Q_EXPERTS];
    std::vector<uint32_t> in, out;
    for (uint32_t e = 0; e < Q_EXPERTS; e++) (hl.slot_of[e] < 0 ? in : out).push_back(e);
    std::sort(in.begin(), in.end(), [&](uint32_t a, uint32_t b) { return u[a] != u[b] ? u[a] > u[b] : a < b; });
    std::sort(out.begin(), out.end(), [&](uint32_t a, uint32_t b) { return u[a] != u[b] ? u[a] < u[b] : a < b; });
    for (uint32_t n = 0; n < in.size() && n < out.size() && n < max_swaps; n++) {
      if (u[in[n]] < 2.0f || u[in[n]] < 1.5f * u[out[n]] + 1.0f) break;
      all.push_back({u[in[n]] - u[out[n]], {l, in[n], out[n], nullptr}});
    }
  }
  std::stable_sort(all.begin(), all.end(), [](const auto &a, const auto &b) { return a.first > b.first; });
  if (all.size() > max_swaps) all.resize(max_swaps);
  for (auto &a : all) {
    HostLayer &hl = *host_layers_[a.second.l];
    a.second.piece = spare_piece(align256(2 * hl.gate_bytes + hl.down_bytes));
    swap_next_.push_back(a.second);
  }
}

// The last round's choice: its halves are in swap_buf_, its outgoing experts in their pieces, its victims in theirs.
// After joining the swap thread.
uint32_t Model::apply_round() {
  settle_moves();
  const auto t0 = std::chrono::steady_clock::now();
  uint64_t slot_max = 0;
  for (auto &L : cards_[0]->layers) slot_max = std::max(slot_max, L.slot_bytes);
  const uint32_t applied = (uint32_t)swap_next_.size();
  uint32_t swapped = 0;
  if (applied) {
    // A refill's slot past the cache's blocks: the blocks it needs, from the spare ones (plan_refills counted them).
    std::vector<int32_t> top(Q_LAYERS, -1);
    for (const SwapPlan &w : swap_next_)
      if (w.out == kNoExpert) top[w.l] = std::max(top[w.l], w.slot);
    for (uint32_t l = 0; l < Q_LAYERS; l++) {
      if (top[l] < 0) continue;
      size_t b = cards_[0]->cache[l].blocks.size();
      while (cache_slots(l, b) <= (uint32_t)top[l]) b++;
      for (auto &card : cards_) map_cache(*card, l, b);
    }
    if (!vm_ops_.empty()) run_vm(false);
    for (size_t i = 0; i < applied * ncards_; i++) {
      const SwapPlan &w = swap_next_[i / ncards_];
      Card &card = *cards_[i % ncards_];
      Layer &L = card.layers[w.l];
      const int32_t s = w.out == kNoExpert ? w.slot : host_layers_[w.l]->slot_of[w.out];
      card.up->copy(L.cache + (uint64_t)s * L.slot_bytes, swap_buf_ + i * slot_max, L.slot_bytes);
    }
    std::vector<bool> changed(Q_LAYERS, false);
    std::vector<const SwapPlan *> refills;
    for (const SwapPlan &w : swap_next_) {
      if (w.out == kNoExpert) {
        refills.push_back(&w);
      } else {
        swap_maps(w);
        swapped++;
      }
      changed[w.l] = true;
    }
    // Refills from the highest staging position down, so each one's place takes an expert that stays uncached.
    std::stable_sort(refills.begin(), refills.end(), [&](const SwapPlan *a, const SwapPlan *b) {
      return host_layers_[a->l]->stage_of[a->in] > host_layers_[b->l]->stage_of[b->in];
    });
    // A refill moves staging pieces, so the swaps' moves end first; a round as chosen holds swaps or refills, not
    // both.
    if (!refills.empty() && !moves_.empty()) {
      pool_->run(moves_.size(), [&](uint64_t i) {
        const HostLayer &hl = *host_layers_[moves_[i].l];
        std::memcpy(moves_[i].to, moves_[i].from, 2 * hl.gate_bytes + hl.down_bytes);
      });
      settle_moves();
    }
    std::vector<PieceMove> moves;
    for (const SwapPlan *w : refills) refill_maps(*w, moves);
    pool_->run(moves.size(), [&](uint64_t i) { std::memcpy(moves[i].to, moves[i].from, moves[i].bytes); });
    for (uint32_t l = 0; l < Q_LAYERS; l++) {
      if (!changed[l]) continue;
      HostLayer &hl = *host_layers_[l];
      for (auto &card : cards_) {
        card->up->copy(card->layers[l].slot_map, hl.slot_of.data(), Q_EXPERTS * 4);
        if (card->pc && !card->pc->stage_map.empty())
          card->up->copy(card->pc->stage_map[l], hl.stage_of.data(), Q_EXPERTS * 4);
      }
    }
    for (auto &card : cards_) card->up->synchronize();
    // Each outgoing expert into the piece its incoming one left, so every staging slot keeps its arena address and
    // the prompt path's copy lists stay as recorded: from the anonymous arena a list costs a bind per copy to record
    // and an unbind to drop (docs/measurements/MEASUREMENTS.md#2026-10-17-swaps-keep-the-copies).
    if (!moves_.empty())
      move_bg_ = std::thread([this] {
        for (const SwapMove &m : moves_) {
          const HostLayer &hl = *host_layers_[m.l];
          std::memcpy(m.to, m.from, 2 * hl.gate_bytes + hl.down_bytes);
        }
      });
    if (!refills.empty()) log_share("refilled the cache", t0);
  }
  swap_next_.clear();
  swaps_ += swapped;
  if (!evict_next_.empty()) apply_evict(std::chrono::steady_clock::now(), true);
  install_copies();
  return applied;
}

// Behind the next steps: with max_swaps, the choice of swaps from the counts use; the outgoing experts into their
// pieces, the incoming experts' halves, and the victims of a planned eviction into their pieces.
void Model::start_round(std::vector<float> use, uint32_t max_swaps) {
  if (swap_next_.empty() && evict_next_.empty() && max_swaps == 0) return;
  uint64_t slot_max = 0;
  for (auto &L : cards_[0]->layers) slot_max = std::max(slot_max, L.slot_bytes);
  const size_t most = std::max<size_t>(swap_next_.size(), max_swaps);
  if (swap_buf_bytes_ < most * ncards_ * slot_max) {
    swap_buf_bytes_ = (uint64_t)std::max<size_t>(64, most) * ncards_ * slot_max;
    swap_buf_ = static_cast<uint8_t *>(rt_->alloc_host(swap_buf_bytes_));
  }
  if (!swap_pool_) swap_pool_ = std::make_unique<Pool>(3);
  // A round that changes the staging records the prompt path's next copy list too: from the anonymous arena a
  // list costs a bind per copy to record and an unbind to free (docs/measurements/MEASUREMENTS.md#2026-10-17-swaps-keep-the-copies).
  // Swaps leave the staging's pieces where they are.
  bool restage = !evict_next_.empty() && cards_[0]->pc;
  for (const SwapPlan &w : swap_next_) restage |= w.out == kNoExpert && cards_[0]->pc;
  swap_bg_ = std::thread([this, slot_max, restage, use = std::move(use), max_swaps, old = std::move(retired_)]() mutable {
    old.clear();
    if (max_swaps) choose_swaps(use, max_swaps);
    std::vector<std::pair<uint32_t, uint32_t>> victims;  // (eviction, victim)
    for (uint32_t v = 0; v < evict_next_.size(); v++)
      for (uint32_t k = 0; k < evict_next_[v].victims.size(); k++) victims.push_back({v, k});
    const uint64_t nw = swap_next_.size() * (ncards_ + 1);
    swap_pool_->run(nw + victims.size(), [&](uint64_t i) {
      if (i >= nw) {
        const Evict &ev = evict_next_[victims[i - nw].first];
        const uint32_t k = victims[i - nw].second;
        fill_piece(ev.l, ev.victims[k], host_layers_[ev.l]->pieces[ev.first + k]);
        return;
      }
      const SwapPlan &w = swap_next_[i / (ncards_ + 1)];
      const uint32_t k = i % (ncards_ + 1);
      if (k == ncards_) {
        if (w.out != kNoExpert) fill_piece(w.l, w.out, w.piece);
        return;
      }
      // The incoming expert's half from its matrices in host memory, the arena's layout: gate and up as the file
      // has them, and down as one region per card (write_down), so card c's blocks are region c's whole rows.
      const GgufTensor &g = gguf_.tensor(blk(w.l, "ffn_gate_exps.weight")), &d = gguf_.tensor(blk(w.l, "ffn_down_exps.weight"));
      const uint64_t ed = d.bytes / Q_EXPERTS;
      Card &card = *cards_[k];
      Layer &L = card.layers[w.l];
      const ExpertWeights &x = host_layers_[w.l]->weights[w.in];
      const uint32_t f0 = card.c * card.ff, dqk = quant_block_values(d.type);
      const uint64_t gbpr = Q_EMBD / quant_block_values(g.type), part = ed / x.down_parts;
      uint8_t *buf = swap_buf_ + ((i / (ncards_ + 1)) * ncards_ + k) * slot_max;
      to_runs(g.type, {{x.gate, g.row_bytes(), f0, card.ff, 0, gbpr}}, buf);
      to_runs(g.type, {{x.up, g.row_bytes(), f0, card.ff, 0, gbpr}}, buf + L.up_off);
      to_runs(d.type, {{x.down + card.c * part, d.row_bytes() / x.down_parts, 0, Q_EMBD, 0, card.ff / dqk}},
              buf + L.down_off);
    });
    // The file's pages the round read, back to the host: a long session's swaps would otherwise refill the page
    // cache that load gave back, and a host that reclaims it moves the engine's buffers
    // (docs/measurements/MEASUREMENTS.md#2026-10-17-decode-stalls-eviction).
    for (const SwapPlan &w : swap_next_)
      if (w.out != kNoExpert) release_expert(w.l, w.out);
    for (const Evict &ev : evict_next_)
      for (uint32_t e : ev.victims) release_expert(ev.l, e);
    if (restage) {
      const Staging st = staging_after();
      for (auto &card : cards_) next_copies_.push_back(prompt_copies(*card, st, next_copy_count_));
    }
  });
  retired_.clear();
}

void Model::fill_piece(uint32_t l, uint32_t e, uint8_t *piece) {
  const GgufTensor &g = gguf_.tensor(blk(l, "ffn_gate_exps.weight")), &u = gguf_.tensor(blk(l, "ffn_up_exps.weight")),
                   &d = gguf_.tensor(blk(l, "ffn_down_exps.weight"));
  const uint64_t eg = g.bytes / Q_EXPERTS, ed = d.bytes / Q_EXPERTS;
  std::memcpy(piece, g.data + e * eg, eg);
  std::memcpy(piece + eg, u.data + e * eg, eg);
  write_down(piece + 2 * eg, d.data + e * ed, Q_EMBD, d.row_bytes(), host_layers_[l]->down_parts);
}

void Model::release_expert(uint32_t l, uint32_t e) {
  const GgufTensor &g = gguf_.tensor(blk(l, "ffn_gate_exps.weight")), &u = gguf_.tensor(blk(l, "ffn_up_exps.weight")),
                   &d = gguf_.tensor(blk(l, "ffn_down_exps.weight"));
  const uint64_t eg = g.bytes / Q_EXPERTS, ed = d.bytes / Q_EXPERTS;
  gguf_.release(g, e * eg, eg);
  gguf_.release(u, e * eg, eg);
  gguf_.release(d, e * ed, ed);
}

// The last round's moves done: each outgoing expert read from the arena piece its incoming one left, and the pieces
// the swap thread filled spare again. After joining the swap thread, which takes spare pieces.
void Model::settle_moves() {
  if (move_bg_.joinable()) move_bg_.join();
  for (const SwapMove &m : moves_) {
    HostLayer &hl = *host_layers_[m.l];
    hl.weights[m.e] = {m.to, m.to + hl.gate_bytes, m.to + 2 * hl.gate_bytes, hl.down_parts};
    spares_[align256(2 * hl.gate_bytes + hl.down_bytes)].push_back(m.from);
  }
  moves_.clear();
}

// A spare arena piece of `need` bytes: one an incoming expert left, or a new one.
uint8_t *Model::spare_piece(uint64_t need) {
  auto &v = spares_[need];
  if (!v.empty()) {
    uint8_t *p = v.back();
    v.pop_back();
    return p;
  }
  return arena_piece(need);
}

// The host's maps after a swap: in takes out's cache slot; out takes in's arena piece and staging slot, and its
// weights are read from the piece the swap thread filled until the move is settled (settle_moves).
void Model::swap_maps(const SwapPlan &w) {
  HostLayer &hl = *host_layers_[w.l];
  const int32_t s = hl.slot_of[w.out];
  uint8_t *p = hl.arena[w.in], *q = w.piece;
  moves_.push_back({w.l, w.out, p, q});
  hl.arena[w.in] = nullptr;
  hl.weights[w.in] = {};
  hl.arena[w.out] = p;
  hl.weights[w.out] = {q, q + hl.gate_bytes, q + 2 * hl.gate_bytes, hl.down_parts};
  const int32_t st = hl.stage_of[w.in];
  hl.stage_of[w.out] = st;
  hl.stage_of[w.in] = -1;
  if (st >= 0) hl.uncached[st] = w.out;
  hl.slot_of[w.in] = s;
  hl.slot_of[w.out] = -1;
}

}  // namespace engine::qwen4exp
