// dot_rows_for's functions against ggml's vec_dot, bit for bit: for each type, 64 weight rows quantised by ggml from
// random values, against 1 to 9 activation rows. Skips (77) on a CPU without the features the pinned ggml build uses
// (cmake/ggml.cmake: AVX-512 among them), which its quantisers and vec_dot would need.
#include <cstring>
#include <random>
#include <vector>

#include "cpu/multirow.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "test_util.h"

using namespace engine;

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
  ggml_cpu_init();
  if (!dot_rows_for(GGML_TYPE_IQ2_S)) return test::skip("no AVX2, FMA or F16C");
  std::mt19937 rng(7);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  // Gate and up types are rows of 2,560 values; down's IQ4_NL is 320 or 640, a card's half of K or all of it.
  const struct {
    ggml_type type;
    int K;
  } cases[] = {{GGML_TYPE_IQ2_S, 2560}, {GGML_TYPE_IQ3_XXS, 2560}, {GGML_TYPE_IQ3_S, 2560},
               {GGML_TYPE_IQ4_NL, 320},  {GGML_TYPE_IQ4_NL, 640}};
  for (const auto &c : cases) {
    const auto *tr = ggml_get_type_traits_cpu(c.type);
    const auto *qt = ggml_get_type_traits_cpu(tr->vec_dot_type);
    const int rows = 64, RMAX = 9;
    const size_t rb = ggml_row_size(c.type, c.K), yb = ggml_row_size(tr->vec_dot_type, c.K);
    std::vector<float> f((size_t)rows * c.K);
    for (auto &v : f) v = normal(rng) * 0.02f;
    std::vector<uint8_t> W((size_t)rows * rb), Y((size_t)RMAX * yb);
    ggml_quantize_init(c.type);
    ggml_quantize_chunk(c.type, f.data(), W.data(), 0, rows, c.K, nullptr);
    std::vector<float> x(c.K);
    for (int r = 0; r < RMAX; r++) {
      for (auto &v : x) v = normal(rng);
      qt->from_float(x.data(), Y.data() + r * yb, c.K);
    }
    const void *ys[RMAX];
    for (int r = 0; r < RMAX; r++) ys[r] = Y.data() + r * yb;
    const DotRows dr = dot_rows_for(c.type);
    EXPECT(dr != nullptr, "%s: no function", ggml_type_name(c.type));
    uint32_t wrong = 0;
    for (int R = 1; R <= RMAX; R++)
      for (int i = 0; i < rows; i++) {
        float want[RMAX], got[RMAX];
        for (int r = 0; r < R; r++) tr->vec_dot(c.K, &want[r], 0, W.data() + i * rb, 0, ys[r], 0, 1);
        dr(c.K, W.data() + i * rb, ys, R, got);
        for (int r = 0; r < R; r++) wrong += std::memcmp(&want[r], &got[r], 4) != 0;
      }
    EXPECT(wrong == 0, "%s, K %d: %u sums differ from vec_dot", ggml_type_name(c.type), c.K, wrong);
    std::printf("%s, K %d: 1 to %d rows, same bits as vec_dot\n", ggml_type_name(c.type), c.K, RMAX);
  }
  EXPECT(dot_rows_for(GGML_TYPE_Q8_0) == nullptr, "a type without a multi-row path returned a function");
  std::printf("multirow_test: ok\n");
  return 0;
}
