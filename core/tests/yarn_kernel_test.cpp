// Level 3 (docs/TESTING.md): the rotation past the trained context in the decode and prompt kernels (idx_prep,
// p_idx_prep), against a scalar reference (docs/spec/MODEL_LAYER.md, section 3.3). A kernel's pow may differ from the
// host's by a few units in the last place, a large angle error at position 1,000,000 on the fastest pairs, so those
// (0-14, which YaRN leaves alone) are checked against the same kernel at s = 1. Tolerance 1e-2, on normed heads.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include "models/qwen4exp/layout.h"
#include "step.h"
#include "test_util.h"
#include "ze/library.h"

using namespace engine;

namespace {

constexpr uint32_t kRows = 4, kPos0 = 1000000, kRow = (Q_IDX_H + 1) * Q_IDX_D;

// The heads after the norm: what the kernels rotate.
std::vector<double> normed(const std::vector<float> &iq) {
  std::vector<double> out(iq.size());
  for (uint32_t t = 0; t < kRows; t++)
    for (uint32_t h = 0; h < Q_IDX_H; h++) {
      const float *x = &iq[(size_t)t * kRow + (size_t)h * Q_IDX_D];
      double ss = 0;
      for (uint32_t d = 0; d < Q_IDX_D; d++) ss += (double)x[d] * x[d];
      const double k = 1.0 / std::sqrt(ss / Q_IDX_D + Q_EPS);
      for (uint32_t d = 0; d < Q_IDX_D; d++) out[(size_t)t * kRow + (size_t)h * Q_IDX_D + d] = x[d] * k;
    }
  return out;
}

// Dimension d of a head at x, rotated at position pos with factor fs.
double reference(const double *x, uint32_t d, uint32_t pos, float fs) {
  if (d >= Q_ROT) return x[d];
  const uint32_t i = d % (Q_ROT / 2);
  const float theta = q_yarn_theta((float)pos * std::pow(Q_ROPE_BASE, -2.0f * (float)i / (float)Q_ROT), i, fs);
  const double c = std::cos((double)theta) * q_yarn_scale(fs), s = std::sin((double)theta) * q_yarn_scale(fs);
  return d < Q_ROT / 2 ? x[i] * c - x[i + Q_ROT / 2] * s : x[i] * s + x[i + Q_ROT / 2] * c;
}

}  // namespace

int main() {
  if (test::gpus() == 0) return test::no_gpu();
  ze::Runtime rt;
  ze::Library lib(rt);
  const uint32_t card = 0;
  ze::List L(rt, card, ze::List::immediate);
  std::mt19937 rng(7);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  std::vector<float> in((size_t)kRows * kRow);
  for (auto &x : in) x = normal(rng);
  const std::vector<double> x = normed(in);

  auto *iq = static_cast<float *>(rt.alloc_host(in.size() * 4));
  auto *qn = static_cast<float *>(rt.alloc_host(Q_IDX_D * 4));
  std::fill(qn, qn + Q_IDX_D, 1.0f);
  auto *raw = static_cast<uint8_t *>(rt.alloc_host(256));
  auto *pt = static_cast<uint32_t *>(rt.alloc_host(256));
  auto *sr = static_cast<SlotRope *>(rt.alloc_host(sizeof(SlotRope)));
  std::memset(sr, 0, sizeof(SlotRope));
  auto *plan = static_cast<StepPlan *>(rt.alloc_host(sizeof(StepPlan)));
  std::memset(plan, 0, sizeof(StepPlan));
  plan->nrows = kRows;
  for (uint32_t t = 0; t < kRows; t++) plan->rows[t].pos = kPos0 + t;
  auto *runs = static_cast<PromptPlan *>(rt.alloc_host(PROMPT_MAX_RUNS * sizeof(PromptPlan)));
  std::memset(runs, 0, PROMPT_MAX_RUNS * sizeof(PromptPlan));
  runs[0] = {kRows, kPos0, 0, 0};
  auto *ppos = static_cast<uint32_t *>(rt.alloc_host((size_t)kRows * 4 * 4));
  for (uint32_t e = 0; e < kRows; e++)
    for (uint32_t a = 0; a < 3; a++) ppos[4 * e + a] = kPos0 + e;

  const uint32_t npp = 1, zero = 0, one = 1, pstride = 4 * kRows;
  auto run = [&](bool prompt, float fs) {
    std::memcpy(iq, in.data(), in.size() * 4);
    if (prompt)
      L.launch(lib.get(card, "qwen4exp_prompt", "p_idx_prep"),
               {iq, qn, raw, pt, npp, runs, ppos, pstride, zero, one, fs}, Q_IDX_H, kRows, 1, Q_IDX_D);
    else
      L.launch(lib.get(card, "qwen4exp", "idx_prep"), {iq, qn, raw, pt, npp, plan, sr, zero, one, fs}, Q_IDX_H, kRows,
               1, Q_IDX_D);
    L.synchronize();
    return std::vector<float>(iq, iq + in.size());
  };

  const float fs = (float)Q_TRAINED_CTX / 1048576.0f, m = q_yarn_scale(fs);
  for (bool prompt : {false, true}) {
    const std::vector<float> trained = run(prompt, 1.0f), scaled = run(prompt, fs);
    double err_trained = 0, err_slow = 0, err_fast = 0, err_rest = 0;
    for (uint32_t t = 0; t < kRows; t++)
      for (uint32_t h = 0; h < Q_IDX_H; h++) {
        const size_t at = (size_t)t * kRow + (size_t)h * Q_IDX_D;
        for (uint32_t d = 0; d < Q_IDX_D; d++) {
          const uint32_t i = d % (Q_ROT / 2);
          if (d >= Q_ROT) {
            err_rest = std::max(err_rest, std::fabs((double)scaled[at + d] - x[at + d]));
          } else if (i <= Q_YARN_LOW) {
            err_fast = std::max(err_fast, std::fabs((double)scaled[at + d] - (double)trained[at + d] * m));
          } else {
            err_slow = std::max(err_slow, std::fabs((double)scaled[at + d] - reference(&x[at], d, kPos0 + t, fs)));
            err_trained =
                std::max(err_trained, std::fabs((double)trained[at + d] - reference(&x[at], d, kPos0 + t, 1.0f)));
          }
        }
      }
    std::printf("%s: unrotated %.2e, pairs 0-14 %.2e, pairs 15-31 %.2e, the same at s = 1 %.2e\n",
                prompt ? "p_idx_prep" : "idx_prep", err_rest, err_fast, err_slow, err_trained);
    EXPECT(err_rest <= 1e-2 && err_fast <= 1e-2 && err_slow <= 1e-2 && err_trained <= 1e-2, "over tolerance");
  }
  for (void *p : std::vector<void *>{iq, qn, raw, pt, sr, plan, runs, ppos}) rt.free(p);
  std::printf("PASS\n");
  return 0;
}
