// Level 3 (docs/TESTING.md): the prompt path's attention kernel, p_attn_xmx (kernels/models/qwen4exp/prompt_attn.cl),
// against attention in double precision over the same fp16 inputs and the same cells. Rows past 2,051 positions
// attend to the indexer's choice: here 512 random blocks, one of them partial, and the tail, laid out as p_idx_merge
// lays them out. One case's keys grow along the positions, so that the online softmax's maximum rises many times.
// Tolerance 2e-3 of the largest |reference| (TESTING.md, "Tolerances": fp16 products).
#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>

#include "ggml.h"
#include "layout.h"
#include "step.h"
#include "test_util.h"
#include "ze/library.h"

using namespace engine;

int main() {
  if (test::gpus() == 0) return test::no_gpu();
  ze::Runtime rt;
  ze::Library lib(rt);
  const uint32_t card = 0, n = 40;
  ze::List L(rt, card, ze::List::immediate);
  std::mt19937 rng(5);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  int failures = 0;
  // cards: the heads of one card when the model runs on that many (2: 12 query heads and 1 KV head; 1: 24 and 2).
  struct Case {
    uint32_t pos0;
    bool rising;
    uint32_t cards;
  };
  for (Case cs : {Case{0, false, 2}, Case{1500, false, 2}, Case{2100, false, 2}, Case{4000, false, 2},
                  Case{20000, false, 2}, Case{1000, true, 2}, Case{20000, true, 2}, Case{2100, false, 1},
                  Case{20000, true, 1}}) {
    const uint32_t pos0 = cs.pos0, ctx = pos0 + n;
    const uint32_t nq = Q_ATT_H / cs.cards, nkv = Q_ATT_KV / cs.cards, hp = nkv * Q_ATT_SLOTS;
    // K and V for every position, the queries (scaled by 1/16 as the kernel takes them) and the gates.
    std::vector<uint16_t> kh((size_t)ctx * nkv * Q_ATT_D), vh(kh.size()), q16((size_t)n * nq * Q_ATT_D),
        qt((size_t)n * Q_ATT_D * hp, 0);
    std::vector<float> Q((size_t)n * nq * 2 * Q_ATT_D);
    for (size_t i = 0; i < kh.size(); i++)
      kh[i] = ggml_fp32_to_fp16(normal(rng) * (cs.rising ? 1.0f + 3.0f * (float)(i / (nkv * Q_ATT_D)) / ctx : 1.0f));
    for (auto &x : vh) x = ggml_fp32_to_fp16(normal(rng));
    for (auto &x : Q) x = normal(rng) * 2.0f;
    for (uint32_t t = 0; t < n; t++)
      for (uint32_t h = 0; h < nq; h++)
        for (uint32_t d = 0; d < Q_ATT_D; d++) {
          uint16_t v = ggml_fp32_to_fp16(Q[((size_t)t * nq + h) * 2 * Q_ATT_D + d] / 16.0f);
          q16[((size_t)t * nq + h) * Q_ATT_D + d] = v;
          qt[((size_t)t * Q_ATT_D + d) * hp + h / (Q_ATT_H / Q_ATT_KV) * Q_ATT_SLOTS + h % (Q_ATT_H / Q_ATT_KV)] = v;
        }
    // Each row's cells, as p_idx_merge lays them out: the whole blocks in increasing order, then the partial one's
    // `extra` cells, then the tail.
    std::vector<uint32_t> cells((size_t)n * Q_IDX_CELLS), ncells(n);
    for (uint32_t t = 0; t < n; t++) {
      uint32_t nkv_t = pos0 + t + 1;
      uint32_t *c = &cells[(size_t)t * Q_IDX_CELLS];
      if (nkv_t <= Q_IDX_CELLS) {
        for (uint32_t i = 0; i < nkv_t; i++) c[i] = i;
        ncells[t] = nkv_t;
        continue;
      }
      uint32_t nb = nkv_t / 4, tail = nkv_t - 4 * nb, extra = Q_IDX_CELLS - Q_IDX_TOPK - tail;
      std::vector<uint32_t> all(nb);
      for (uint32_t b = 0; b < nb; b++) all[b] = b;
      std::shuffle(all.begin(), all.end(), rng);
      uint32_t want = Q_IDX_BLOCKS + (extra > 0);
      std::vector<uint32_t> chosen(all.begin(), all.begin() + want);
      std::sort(chosen.begin(), chosen.end());
      uint32_t worst = extra > 0 ? (uint32_t)(rng() % want) : want, at = 0;
      for (uint32_t i = 0; i < want; i++)
        if (i != worst)
          for (uint32_t k = 0; k < 4; k++) c[at++] = 4 * chosen[i] + k;
      for (uint32_t k = 0; k < extra; k++) c[at++] = 4 * chosen[worst] + k;
      for (uint32_t k = 0; k < tail; k++) c[at++] = 4 * nb + k;
      ncells[t] = at;
    }
    // The reference.
    std::vector<double> ref((size_t)n * nq * Q_ATT_D);
    for (uint32_t t = 0; t < n; t++)
      for (uint32_t h = 0; h < nq; h++) {
        const uint32_t *c = &cells[(size_t)t * Q_IDX_CELLS], kv = h / (Q_ATT_H / Q_ATT_KV);
        std::vector<double> s(ncells[t]);
        double mx = -INFINITY;
        for (uint32_t i = 0; i < ncells[t]; i++) {
          double dot = 0;
          for (uint32_t d = 0; d < Q_ATT_D; d++)
            dot += (double)ggml_fp16_to_fp32(q16[((size_t)t * nq + h) * Q_ATT_D + d]) *
                   ggml_fp16_to_fp32(kh[((size_t)c[i] * nkv + kv) * Q_ATT_D + d]);
          s[i] = dot;
          mx = std::max(mx, dot);
        }
        double sum = 0;
        for (auto &v : s) sum += v = std::exp(v - mx);
        for (uint32_t d = 0; d < Q_ATT_D; d++) {
          double acc = 0;
          for (uint32_t i = 0; i < ncells[t]; i++) acc += s[i] * ggml_fp16_to_fp32(vh[((size_t)c[i] * nkv + kv) * Q_ATT_D + d]);
          double gate = Q[((size_t)t * nq + h) * 2 * Q_ATT_D + Q_ATT_D + d];
          ref[((size_t)t * nq + h) * Q_ATT_D + d] = acc / sum / (1 + std::exp(-gate));
        }
      }
    auto dev = [&](const void *src, size_t bytes) {
      void *p = rt.alloc_device(card, std::max<size_t>(bytes, 256));
      L.copy(p, src, bytes);
      return p;
    };
    void *dkh = dev(kh.data(), kh.size() * 2), *dvh = dev(vh.data(), vh.size() * 2), *dqt = dev(qt.data(), qt.size() * 2),
         *dQ = dev(Q.data(), Q.size() * 4), *dcells = dev(cells.data(), cells.size() * 4), *dn = dev(ncells.data(), n * 4);
    auto *pp = static_cast<PromptPlan *>(rt.alloc_host(sizeof(PromptPlan)));
    *pp = {n, pos0, 0, 0};
    auto *o16 = static_cast<uint16_t *>(rt.alloc_host((size_t)n * nq * Q_ATT_D * 2));
    std::fill(o16, o16 + (size_t)n * nq * Q_ATT_D, 0x7e00);
    L.synchronize();
    const uint32_t rows = 16 / (hp / 8);  // 16 sub-groups a work-group, as prompt.cpp launches it
    L.launch(lib.get(card, "qwen4exp_attn", "p_attn_xmx"), {dqt, dQ, dkh, dvh, nq, hp, nkv, dcells, dn, pp, o16}, 1,
             (n + rows - 1) / rows, 1, 16 * 16);
    L.synchronize();
    double maxerr = 0, maxref = 0;
    for (size_t i = 0; i < ref.size(); i++) {
      maxref = std::max(maxref, std::fabs(ref[i]));
      double e = std::fabs(ref[i] - ggml_fp16_to_fp32(o16[i]));
      maxerr = std::isnan(e) ? INFINITY : std::max(maxerr, e);
    }
    double rel = maxerr / maxref;
    bool ok = rel <= 2e-3;
    failures += !ok;
    std::printf("p_attn_xmx, %u query heads, rows %u-%u%s: max error %.2e %s\n", nq, pos0, pos0 + n - 1,
                cs.rising ? ", rising keys" : "", rel, ok ? "ok" : "FAIL");
    for (void *p : {dkh, dvh, dqt, dQ, dcells, dn}) rt.free(p);
    rt.free(pp);
    rt.free(o16);
  }
  EXPECT(failures == 0, "%d case(s) over tolerance", failures);
  std::printf("PASS\n");
  return 0;
}
