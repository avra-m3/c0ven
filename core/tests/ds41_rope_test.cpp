// Level 1 (docs/TESTING.md): deepseek41's rotary embedding (core/models/deepseek41/rope.h). Its constants against
// powf and ggml's YaRN dimensions; its frequencies against DeepSeek's precompute_freqs_cis in double; the rotation
// and its inverse against ggml's rope_ext and rope_ext_back as upstream llama.cpp calls them, at both bases and the
// positions where the window ring wraps; the inverse undoing the rotation; ref::rope_tail turning only the tail. No
// card.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numbers>
#include <random>
#include <vector>

#include "ggml-cpu.h"
#include "ggml.h"
#include "models/deepseek41/ref_ops.h"
#include "models/deepseek41/rope.h"
#include "test_util.h"

namespace {

constexpr uint32_t kPositions[] = {0, 1, 2, 127, 128, 135, 136, 4095, 16383, 16384, 65535, 262143};
constexpr uint32_t kNp = sizeof kPositions / sizeof kPositions[0];

// DeepSeek's frequency of pair i (model.py precompute_freqs_cis), in double.
double reference_freq(uint32_t i, bool yarn) {
  const double base = yarn ? D41_ROPE_BASE_C : D41_ROPE_BASE, dim = D41_ROT;
  double f = 1.0 / std::pow(base, 2.0 * i / dim);
  if (!yarn) return f;
  auto corrected = [&](double rotations) {
    return dim * std::log(D41_YARN_ORIG / (rotations * 2 * std::numbers::pi)) / (2 * std::log(base));
  };
  const double low = std::max(std::floor(corrected(D41_YARN_BETA_FAST)), 0.0);
  const double high = std::min(std::ceil(corrected(D41_YARN_BETA_SLOW)), dim - 1);
  const double ramp = std::clamp((i - low) / std::max(high - low, 1e-3), 0.0, 1.0);
  return f / D41_YARN_FACTOR * ramp + f * (1 - ramp);
}

// ggml's rope over kNp rows of D41_ROT values, row k at kPositions[k], with upstream llama.cpp's arguments for the
// layer kind (src/models/deepseek41.cpp, build_attention41).
std::vector<float> ggml_rope(const std::vector<float> &x, bool yarn, bool back) {
  ggml_init_params ip = {64u << 20, nullptr, false};
  ggml_context *ctx = ggml_init(ip);
  ggml_tensor *a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D41_ROT, 1, kNp);
  ggml_tensor *pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, kNp);
  std::memcpy(a->data, x.data(), x.size() * sizeof(float));
  for (uint32_t k = 0; k < kNp; k++) ((int32_t *)pos->data)[k] = (int32_t)kPositions[k];
  const float freq_scale = yarn ? 1.0f / D41_YARN_FACTOR : 1.0f, ext = yarn ? 1.0f : 0.0f;
  const float attn = yarn ? 1.0f / (1.0f + 0.1f * logf(1.0f / freq_scale)) : 1.0f;
  const float base = yarn ? D41_ROPE_BASE_C : D41_ROPE_BASE;
  const int orig = yarn ? D41_YARN_ORIG : 0;
  const float fast = yarn ? D41_YARN_BETA_FAST : 0.0f, slow = yarn ? D41_YARN_BETA_SLOW : 0.0f;
  ggml_tensor *r = back ? ggml_rope_ext_back(ctx, a, pos, nullptr, D41_ROT, GGML_ROPE_TYPE_NORMAL, orig, base,
                                             freq_scale, ext, attn, fast, slow)
                        : ggml_rope_ext(ctx, a, pos, nullptr, D41_ROT, GGML_ROPE_TYPE_NORMAL, orig, base, freq_scale,
                                        ext, attn, fast, slow);
  ggml_cgraph *gf = ggml_new_graph(ctx);
  ggml_build_forward_expand(gf, r);
  EXPECT(ggml_graph_compute_with_ctx(ctx, gf, 1) == GGML_STATUS_SUCCESS, "ggml's graph failed");
  std::vector<float> y((const float *)r->data, (const float *)r->data + x.size());
  ggml_free(ctx);
  return y;
}

}  // namespace

int main() {
  // ggml here is built for the reference CPU, so a CPU without these instructions cannot run it.
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
  // The steps are powf's, as ggml's rope takes them; the ramp's ends are ggml's.
  EXPECT(D41_ROPE_STEP == powf(D41_ROPE_BASE, -2.0f / D41_ROT), "D41_ROPE_STEP %a, powf %a", D41_ROPE_STEP,
         powf(D41_ROPE_BASE, -2.0f / D41_ROT));
  EXPECT(D41_ROPE_STEP_C == powf(D41_ROPE_BASE_C, -2.0f / D41_ROT), "D41_ROPE_STEP_C %a, powf %a", D41_ROPE_STEP_C,
         powf(D41_ROPE_BASE_C, -2.0f / D41_ROT));
  float dims[2];
  ggml_rope_yarn_corr_dims(D41_ROT, D41_YARN_ORIG, D41_ROPE_BASE_C, D41_YARN_BETA_FAST, D41_YARN_BETA_SLOW, dims);
  EXPECT(dims[0] == D41_YARN_LOW && dims[1] == D41_YARN_HIGH, "YaRN's dimensions %g .. %g", dims[0], dims[1]);

  // The angles over the position are DeepSeek's frequencies, to float's rounding of the running product.
  for (int yarn = 0; yarn < 2; yarn++)
    for (uint32_t i = 0; i < D41_ROT / 2; i++) {
      const double f = reference_freq(i, yarn), got = d41_rope_theta(1000, i, yarn) / 1000.0;
      EXPECT(std::fabs(got - f) <= 4e-6 * f, "yarn %d pair %u: frequency %.9g, DeepSeek's %.9g", yarn, i, got, f);
    }
  // Before the ramp a pair keeps base 160000's angle, and past it the angle is a sixteenth of that, exactly.
  for (uint32_t i = 0; i < D41_ROT / 2; i++) {
    float theta = 777.0f;
    for (uint32_t k = 0; k < i; k++) theta *= D41_ROPE_STEP_C;
    if (i <= D41_YARN_LOW) EXPECT(d41_rope_theta(777, i, 1) == theta, "pair %u before the ramp", i);
    if (i >= D41_YARN_HIGH) EXPECT(d41_rope_theta(777, i, 1) == theta / 16.0f, "pair %u past the ramp", i);
  }

  // The rotation and its inverse against ggml's, as upstream llama.cpp calls them.
  std::mt19937 rng(41);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<float> x((size_t)kNp * D41_ROT);
  for (float &v : x) v = nd(rng);
  for (int yarn = 0; yarn < 2; yarn++)
    for (int back = 0; back < 2; back++) {
      const std::vector<float> want = ggml_rope(x, yarn, back);
      double err = 0, peak = 0;
      for (uint32_t k = 0; k < kNp; k++) {
        std::vector<float> v(x.begin() + (size_t)k * D41_ROT, x.begin() + (size_t)(k + 1) * D41_ROT);
        d41_rope(v.data(), kPositions[k], yarn, back);
        for (uint32_t d = 0; d < D41_ROT; d++) {
          const double w = want[(size_t)k * D41_ROT + d];
          err = std::max(err, std::fabs(v[d] - w));
          peak = std::max(peak, std::fabs(w));
        }
      }
      std::printf("base %s, %s: largest difference from ggml %.3g of %.3g\n", yarn ? "160000 YaRN" : "10000",
                  back ? "inverse" : "forward", err, peak);
      EXPECT(err <= 1e-5 * peak, "the rotation differs from ggml's");
    }

  // The inverse undoes the rotation; each pair keeps its length; position 0 turns nothing.
  for (int yarn = 0; yarn < 2; yarn++)
    for (uint32_t k = 0; k < kNp; k++) {
      const float *v0 = x.data() + (size_t)k * D41_ROT;
      std::vector<float> v(v0, v0 + D41_ROT);
      d41_rope(v.data(), kPositions[k], yarn, false);
      for (uint32_t i = 0; i < D41_ROT / 2; i++) {
        const double a = std::hypot(v0[2 * i], v0[2 * i + 1]), b = std::hypot(v[2 * i], v[2 * i + 1]);
        EXPECT(std::fabs(a - b) <= 1e-6 * (1 + a), "pair %u's length at %u", i, kPositions[k]);
        if (kPositions[k] == 0) EXPECT(v[2 * i] == v0[2 * i] && v[2 * i + 1] == v0[2 * i + 1], "pair %u at 0", i);
      }
      d41_rope(v.data(), kPositions[k], yarn, true);
      for (uint32_t d = 0; d < D41_ROT; d++)
        EXPECT(std::fabs(v[d] - v0[d]) <= 1e-5f, "inverse at %u, dim %u: %g, was %g", kPositions[k], d, v[d], v0[d]);
    }

  // ref::rope_tail turns the last D41_ROT values of a head and of an index vector, and no other.
  for (uint32_t n : {(uint32_t)D41_HD, (uint32_t)D41_IDX_D}) {
    std::vector<float> v(n), w;
    for (float &e : v) e = nd(rng);
    w = v;
    engine::deepseek41::ref::rope_tail(w.data(), n, 16383, true, false);
    d41_rope(v.data() + n - D41_ROT, 16383, true, false);
    EXPECT(v == w, "rope_tail over %u values", n);
  }
  std::printf("ok\n");
  return 0;
}
