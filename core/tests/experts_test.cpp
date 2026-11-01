// Level 1 (docs/TESTING.md): CpuExperts' SwiGLU limit (ExpertShape::swiglu_limit) against the formula written out
// here, applied to ggml's dot products of the same weights. compute and compute_many, the latter with h made in step
// with gate and up (Q8_0 down) and after them (Q4_K down), with an expert that several tokens share. With the limit 0
// the result is the plain product. Skips (77) on a CPU without the features the pinned ggml build uses.
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "cpu/experts.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "test_util.h"
#include "util/parallel.h"

using namespace engine;

namespace {

struct Expert {
  std::vector<uint8_t> gate, up, down;
};

std::vector<uint8_t> quantize(ggml_type type, const std::vector<float> &f, int64_t rows, int64_t K) {
  std::vector<uint8_t> q(rows * ggml_row_size(type, K));
  ggml_quantize_init(type);
  ggml_quantize_chunk(type, f.data(), q.data(), 0, rows, K, nullptr);
  return q;
}

// One expert's output for x, by the formula: g and u from ggml's dot products, then
// h = silu(min(g, L)) * clamp(u, -L, L) for L above 0, else silu(g) * u, then down h.
std::vector<float> reference(const ExpertShape &s, const Expert &ex, const float *x) {
  const uint32_t E = s.n_embd, F = s.n_ff;
  const auto *gt = ggml_get_type_traits_cpu((ggml_type)s.gate_type);
  const auto *dt = ggml_get_type_traits_cpu((ggml_type)s.down_type);
  std::vector<uint8_t> xq(ggml_row_size(gt->vec_dot_type, E)), hq(ggml_row_size(dt->vec_dot_type, F));
  ggml_get_type_traits_cpu(gt->vec_dot_type)->from_float(x, xq.data(), E);
  const size_t grow = ggml_row_size((ggml_type)s.gate_type, E), drow = ggml_row_size((ggml_type)s.down_type, F);
  const float L = s.swiglu_limit;
  std::vector<float> h(F), out(E);
  for (uint32_t r = 0; r < F; r++) {
    float g, u;
    gt->vec_dot((int)E, &g, 0, ex.gate.data() + r * grow, 0, xq.data(), 0, 1);
    gt->vec_dot((int)E, &u, 0, ex.up.data() + r * grow, 0, xq.data(), 0, 1);
    if (L > 0) {
      g = g < L ? g : L;
      u = u < -L ? -L : u > L ? L : u;
    }
    h[r] = g / (1.0f + std::exp(-g)) * u;
  }
  ggml_get_type_traits_cpu(dt->vec_dot_type)->from_float(h.data(), hq.data(), F);
  for (uint32_t i = 0; i < E; i++) dt->vec_dot((int)F, &out[i], 0, ex.down.data() + i * drow, 0, hq.data(), 0, 1);
  return out;
}

// The largest |a - b| over the largest |b|. A NaN in a or b makes it infinite.
double rel_error(const std::vector<float> &a, const std::vector<float> &b) {
  double err = 0, ref = 0;
  for (size_t i = 0; i < a.size(); i++) {
    double d = std::fabs((double)a[i] - (double)b[i]);
    err = std::isnan(d) ? INFINITY : std::max(err, d);
    ref = std::max(ref, (double)std::fabs(b[i]));
  }
  return err / ref;
}

}  // namespace

int main() {
#define NEEDS(f) \
  if (!__builtin_cpu_supports(f)) return test::skip("the CPU lacks " f ", which the ggml build uses")
  NEEDS("avx2");
  NEEDS("fma");
  NEEDS("f16c");
  NEEDS("bmi2");
  NEEDS("avx512f");
  NEEDS("avx512vbmi");
  NEEDS("avx512vnni");
  NEEDS("avx512bf16");
#undef NEEDS
  Pool pool(4);
  CpuExperts cpu(pool);
  std::mt19937 rng(11);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  // Gate and up values with a standard deviation of about 8, so a limit of 10 clamps a fair share of them.
  const uint32_t E = 256, F = 256, NEXP = 3, NTOK = 3;
  const float LIMIT = 10.0f;
  int failures = 0;
  for (ggml_type down_type : {GGML_TYPE_Q8_0, GGML_TYPE_Q4_K}) {
    std::vector<Expert> experts(NEXP);
    for (auto &ex : experts) {
      std::vector<float> f((size_t)F * E);
      for (auto &v : f) v = normal(rng) * 0.5f;
      ex.gate = quantize(GGML_TYPE_Q8_0, f, F, E);
      for (auto &v : f) v = normal(rng) * 0.5f;
      ex.up = quantize(GGML_TYPE_Q8_0, f, F, E);
      for (auto &v : f) v = normal(rng) * 0.05f;
      ex.down = quantize(down_type, f, E, F);
    }
    std::vector<float> x((size_t)NTOK * E);
    for (auto &v : x) v = normal(rng);
    // Token t takes experts t and t + 1 (mod NEXP), so each expert serves two tokens.
    const float w[2] = {0.7f, 0.3f};
    std::vector<std::vector<float>> plain(NTOK);
    for (float limit : {0.0f, LIMIT}) {
      ExpertShape s{GGML_TYPE_Q8_0, (uint32_t)down_type, E, F, limit};
      std::vector<CpuExperts::Job> jobs;
      std::vector<std::vector<float>> want(NTOK, std::vector<float>(E, 0.0f));
      for (uint32_t t = 0; t < NTOK; t++) {
        std::vector<ExpertWeights> ews;
        for (uint32_t k = 0; k < 2; k++) {
          const Expert &ex = experts[(t + k) % NEXP];
          ExpertWeights ew{ex.gate.data(), ex.up.data(), ex.down.data(), 1};
          ews.push_back(ew);
          jobs.push_back({ew, t, w[k]});
          std::vector<float> o = reference(s, ex, x.data() + t * E);
          for (uint32_t i = 0; i < E; i++) want[t][i] += w[k] * o[i];
        }
        std::vector<float> got(E);
        cpu.compute(s, ews, {w[0], w[1]}, x.data() + t * E, got.data());
        double e = rel_error(got, want[t]);
        bool ok = e <= 1e-5;
        failures += !ok;
        std::printf("down %-4s limit %4.1f compute      token %u: error %.2e %s\n", ggml_type_name(down_type), limit,
                    t, e, ok ? "ok" : "FAIL");
      }
      std::vector<float> y((size_t)NTOK * E);
      cpu.compute_many(s, jobs, NTOK, x.data(), E, y.data(), E);
      for (uint32_t t = 0; t < NTOK; t++) {
        std::vector<float> got(y.begin() + t * E, y.begin() + (t + 1) * E);
        double e = rel_error(got, want[t]);
        bool ok = e <= 1e-5;
        failures += !ok;
        std::printf("down %-4s limit %4.1f compute_many token %u: error %.2e %s\n", ggml_type_name(down_type), limit,
                    t, e, ok ? "ok" : "FAIL");
        if (limit == 0) plain[t] = want[t];
      }
      if (limit == 0) continue;
      // The limit must change the result, or the cases above show nothing.
      for (uint32_t t = 0; t < NTOK; t++) {
        double d = rel_error(want[t], plain[t]);
        EXPECT(d > 1e-2, "down %s token %u: the limit changes y by only %.2e", ggml_type_name(down_type), t, d);
      }
    }
  }
  EXPECT(failures == 0, "%d case(s) over tolerance", failures);
  std::printf("experts_test: ok\n");
  return 0;
}
