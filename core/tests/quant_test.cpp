// The weight type table of shared/quant.h against the pinned ggml, for every type the engine reads: block sizes,
// fields that cover each byte of a block once, the runs layout as a permutation of random blocks, and ggml's CPU dot
// product, which computes the experts the cards miss (core/cpu/experts.cpp), against its dequantisation in double
// precision: 1e-4 of the sum of |w x|. No card; the dot products need the CPU features the ggml build uses.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <strings.h>
#include <vector>

#include "cpu/features.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "load/gguf.h"
#include "load/runs.h"
#include "quant.h"
#include "test_util.h"

using namespace engine;

namespace {

// Values of a row of the dot product's activation type: ggml's dequantisation, or for q8_K, which has none, its
// float scale and 256 int8 values a block.
std::vector<double> act_values(ggml_type t, const uint8_t *row, int64_t K) {
  std::vector<double> v(K);
  if (t == GGML_TYPE_Q8_K) {
    for (int64_t b = 0; b < K / 256; b++) {
      const uint8_t *p = row + b * ggml_type_size(t);
      float d;
      std::memcpy(&d, p, 4);
      for (int i = 0; i < 256; i++) v[b * 256 + i] = (double)d * (int8_t)p[4 + i];
    }
    return v;
  }
  std::vector<float> f(K);
  ggml_get_type_traits(t)->to_float(row, f.data(), K);
  for (int64_t i = 0; i < K; i++) v[i] = f[i];
  return v;
}

}  // namespace

int main() {
  const std::string missing = missing_cpu_features(ENGINE_CPU_FEATURES);
  if (missing.empty()) ggml_cpu_init();
  std::mt19937 rng(5);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  for (uint32_t type : test::quant_types()) {
    const ggml_type t = (ggml_type)type;
    const char *name = ggml_type_label(type);
    const uint32_t bb = quant_block_bytes(type), qk = quant_block_values(type);
    EXPECT(strcasecmp(name, ggml_type_name(t)) == 0, "type %u: label %s, ggml's %s", type, name, ggml_type_name(t));
    EXPECT(qk == (uint32_t)ggml_blck_size(t) && bb == ggml_type_size(t), "%s: %u values in %u B, ggml's %lld in %zu",
           name, qk, bb, (long long)ggml_blck_size(t), ggml_type_size(t));

    // The fields cover each byte of a block once.
    std::vector<int> cover(bb, 0);
    for (uint32_t f = 0; f < quant_nfields(type); f++) {
      const uint32_t off = quant_field_offset(type, f), size = quant_field_size(type, f);
      EXPECT(off + size <= bb, "%s: field %u ends past the block", name, f);
      for (uint32_t i = off; i < off + size; i++) cover[i]++;
    }
    for (uint32_t i = 0; i < bb; i++) EXPECT(cover[i] == 1, "%s: byte %u in %d fields", name, i, cover[i]);

    // The runs layout of 37 random blocks, as 37 rows of one block, and back.
    const uint64_t n = 37;
    std::vector<uint8_t> blocks = test::random_blocks(type, n, rng), runs(runs_bytes(type, n)), back(runs.size());
    to_runs(type, {{blocks.data(), bb, 0, n, 0, 1}}, runs.data());
    from_runs(type, runs.data(), n, back.data());
    EXPECT(back == blocks, "%s: the runs layout is not a permutation", name);

    // Rows of random blocks: finite values, and the CPU dot product.
    const int64_t K = 2560, rows = 8;
    const size_t rb = ggml_row_size(t, K);
    std::vector<uint8_t> W = test::random_blocks(type, rows * K / qk, rng);
    std::vector<float> wf((size_t)(rows * K));
    ggml_get_type_traits(t)->to_float(W.data(), wf.data(), rows * K);
    for (float v : wf) EXPECT(std::isfinite(v), "%s: random blocks dequantise to a value not finite", name);
    if (!missing.empty()) {
      std::printf("%-8s table, fields and runs layout ok; dot product not checked: the CPU lacks %s\n", name,
                  missing.c_str());
      continue;
    }
    const auto *tr = ggml_get_type_traits_cpu(t);
    EXPECT(tr->vec_dot != nullptr, "%s: ggml has no CPU dot product", name);
    const auto *qt = ggml_get_type_traits_cpu(tr->vec_dot_type);
    EXPECT(qt->from_float != nullptr, "%s: ggml cannot quantise to %s", name, ggml_type_name(tr->vec_dot_type));
    std::vector<float> x(K);
    for (auto &v : x) v = normal(rng);
    std::vector<uint8_t> xq(ggml_row_size(tr->vec_dot_type, K));
    qt->from_float(x.data(), xq.data(), K);
    const std::vector<double> xv = act_values(tr->vec_dot_type, xq.data(), K);
    double worst = 0;
    for (int64_t r = 0; r < rows; r++) {
      float got;
      tr->vec_dot((int)K, &got, 0, W.data() + r * rb, 0, xq.data(), 0, 1);
      double ref = 0, mag = 0;
      for (int64_t i = 0; i < K; i++) {
        ref += (double)wf[r * K + i] * xv[i];
        mag += std::fabs((double)wf[r * K + i] * xv[i]);
      }
      worst = std::max(worst, std::fabs(got - ref) / mag);
    }
    EXPECT(worst <= 1e-4, "%s: the CPU dot product is %.2e of sum |w x| from ggml's dequantisation", name, worst);
    std::printf("%-8s table, fields and runs layout ok; CPU dot product with %s within %.2e of sum |w x|\n", name,
                ggml_type_name(tr->vec_dot_type), worst);
  }
  std::printf("PASS\n");
  return 0;
}
