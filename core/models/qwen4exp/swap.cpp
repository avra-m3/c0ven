// Adaptive swaps (Strata's --adapt-swaps, docs/spec/STRATA_MEMORY.md): the cards' expert cache follows what the
// requests route to. Decode and prompt routing go into decayed per-layer counts; a swap round, between steps, lets
// the uncached experts used most replace the cached ones used least. A swap changes which experts the CPU computes,
// and so their rounding: runs that compare tokens turn swaps off (Options::adapt_swaps, ADR-0016's switches).
#include <algorithm>
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

// A round applies the previous round's choice and chooses the next. A thread builds the chosen experts' halves and
// copies the outgoing experts into spare arena pieces behind the steps, so the next round only copies and remaps.
uint32_t Model::swap_round(uint32_t max_swaps) {
  if (nslots_cached_ == 0 || nslots_cached_ >= Q_EXPERTS) return 0;
  if (swap_bg_.joinable()) swap_bg_.join();
  const uint64_t slot_max = [&] {
    uint64_t m = 0;
    for (auto &L : cards_[0]->layers) m = std::max(m, L.slot_bytes);
    return m;
  }();
  // The previous round's choice: its halves are in swap_buf_, its outgoing experts in their pieces.
  const uint32_t applied = (uint32_t)swap_next_.size();
  if (applied) {
    for (size_t i = 0; i < applied * ncards_; i++) {
      const SwapPlan &w = swap_next_[i / ncards_];
      Card &card = *cards_[i % ncards_];
      Layer &L = card.layers[w.l];
      const int32_t s = host_layers_[w.l]->slot_of[w.out];
      card.up->copy(L.cache + (uint64_t)s * L.slot_bytes, swap_buf_ + i * slot_max, L.slot_bytes);
    }
    // Each outgoing expert into the piece its incoming one leaves, so every staging slot keeps its arena address
    // and the prompt path's copy lists stay as recorded: from the anonymous arena a list costs a bind per copy to
    // record and an unbind to drop (docs/MEASUREMENTS.md#2026-10-17-swaps-keep-the-copies).
    pool_->run(applied, [&](uint64_t i) {
      const SwapPlan &w = swap_next_[i];
      const HostLayer &hl = *host_layers_[w.l];
      std::memcpy(hl.arena[w.in], w.piece, 2 * hl.gate_bytes + hl.down_bytes);
    });
    for (auto &card : cards_) card->up->synchronize();
    std::vector<bool> changed(Q_LAYERS, false);
    for (const SwapPlan &w : swap_next_) {
      swap_maps(w);
      changed[w.l] = true;
    }
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
  }
  swap_next_.clear();
  // This round's choice: in each layer, the uncached experts used most against the cached ones used least, while a
  // newcomer's count is clearly higher; then the round's best max_swaps by margin.
  std::vector<std::pair<float, SwapPlan>> all;
  for (uint32_t l = 0; l < Q_LAYERS; l++) {
    HostLayer &hl = *host_layers_[l];
    const float *u = &use_[(size_t)l * Q_EXPERTS];
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
  if (swap_buf_bytes_ < swap_next_.size() * ncards_ * slot_max) {
    swap_buf_bytes_ = (uint64_t)max_swaps * ncards_ * slot_max;
    swap_buf_ = static_cast<uint8_t *>(rt_->alloc_host(swap_buf_bytes_));
  }
  for (auto &x : use_) x *= 0.8f;  // the counts decay each round
  swaps_ += applied;
  // Behind the next steps: the outgoing experts into their pieces and the incoming experts' halves.
  if (!swap_next_.empty()) {
    if (!swap_pool_) swap_pool_ = std::make_unique<Pool>(3);
    swap_bg_ = std::thread([this, slot_max] {
      swap_pool_->run(swap_next_.size() * (ncards_ + 1), [&](uint64_t i) {
        const SwapPlan &w = swap_next_[i / (ncards_ + 1)];
        const GgufTensor &g = gguf_.tensor(blk(w.l, "ffn_gate_exps.weight")),
                         &u = gguf_.tensor(blk(w.l, "ffn_up_exps.weight")), &d = gguf_.tensor(blk(w.l, "ffn_down_exps.weight"));
        const uint64_t eg = g.bytes / Q_EXPERTS, ed = d.bytes / Q_EXPERTS;
        const uint32_t k = i % (ncards_ + 1);
        if (k == ncards_) {
          std::memcpy(w.piece, g.data + w.out * eg, eg);
          std::memcpy(w.piece + eg, u.data + w.out * eg, eg);
          write_down(w.piece + 2 * eg, d.data + w.out * ed, Q_EMBD, d.row_bytes(), host_layers_[w.l]->down_parts);
          return;
        }
        Card &card = *cards_[k];
        Layer &L = card.layers[w.l];
        const uint32_t f0 = card.c * card.ff, dqk = quant_block_values(d.type);
        const uint64_t gbpr = Q_EMBD / quant_block_values(g.type);
        uint8_t *buf = swap_buf_ + ((i / (ncards_ + 1)) * ncards_ + k) * slot_max;
        to_runs(g.type, {{g.data + w.in * eg, g.row_bytes(), f0, card.ff, 0, gbpr}}, buf);
        to_runs(g.type, {{u.data + w.in * eg, g.row_bytes(), f0, card.ff, 0, gbpr}}, buf + L.up_off);
        to_runs(d.type, {{d.data + w.in * ed, d.row_bytes(), 0, Q_EMBD, f0 / dqk, card.ff / dqk}}, buf + L.down_off);
      });
      // The file's pages the round read, back to the host: a long session's swaps would otherwise refill the page
      // cache that load gave back, and a host that reclaims it moves the engine's buffers
      // (docs/MEASUREMENTS.md#2026-10-17-decode-stalls-eviction).
      for (const SwapPlan &w : swap_next_) {
        const GgufTensor &g = gguf_.tensor(blk(w.l, "ffn_gate_exps.weight")),
                         &u = gguf_.tensor(blk(w.l, "ffn_up_exps.weight")), &d = gguf_.tensor(blk(w.l, "ffn_down_exps.weight"));
        const uint64_t eg = g.bytes / Q_EXPERTS, ed = d.bytes / Q_EXPERTS;
        for (uint32_t e : {w.in, w.out}) {
          gguf_.release(g, e * eg, eg);
          gguf_.release(u, e * eg, eg);
          gguf_.release(d, e * ed, ed);
        }
      }
    });
  }
  return applied;
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

// The host's maps after a swap: in takes out's cache slot; out takes in's arena piece, which now holds out's
// weights, and in's staging slot; the piece the swap thread filled is spare again.
void Model::swap_maps(const SwapPlan &w) {
  HostLayer &hl = *host_layers_[w.l];
  const int32_t s = hl.slot_of[w.out];
  uint8_t *p = hl.arena[w.in];
  spares_[align256(2 * hl.gate_bytes + hl.down_bytes)].push_back(w.piece);
  hl.arena[w.in] = nullptr;
  hl.weights[w.in] = {};
  hl.arena[w.out] = p;
  hl.weights[w.out] = {p, p + hl.gate_bytes, p + 2 * hl.gate_bytes, hl.down_parts};
  const int32_t st = hl.stage_of[w.in];
  hl.stage_of[w.out] = st;
  hl.stage_of[w.in] = -1;
  if (st >= 0) hl.uncached[st] = w.out;
  hl.slot_of[w.in] = s;
  hl.slot_of[w.out] = -1;
}

}  // namespace engine::qwen4exp
