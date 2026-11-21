// Level 2 (docs/TESTING.md): common.cl's gather_logits against the host. Raw logits for caller-chosen token ids on
// a card's slice [offset, offset + n) of a row, plus the slice's (max, sum_exp) for exact full-vocabulary log-sum-exp.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include "test_util.h"
#include "ze/library.h"

using namespace engine;

int main() {
  if (test::gpus() == 0) return test::no_gpu();
  ze::Runtime rt;
  ze::Library lib(rt);
  const uint32_t card = 0;
  ze::List up(rt, card, ze::List::immediate);
  auto k_gather = lib.get(card, "common", "gather_logits");
  std::mt19937 rng(42);
  std::normal_distribution<float> normal(0.0f, 4.0f);

  struct Case {
    uint32_t n, rows, row, offset;
    std::vector<int32_t> tokens;
  };
  const Case cases[] = {
      {124160, 1, 0, 0, {0, 1, 17, 65535, 124159, 124160, 200000}},
      {124160, 4, 2, 124160, {0, 124159, 124160, 124161, 200000, 248319}},
      {512, 3, 1, 0, {0, 511, 511, 256, 512}},
      {17, 2, 1, 17, {16, 17, 33, 34}},
  };
  int failures = 0;
  for (const Case &c : cases) {
    std::vector<float> v((size_t)c.rows * c.n);
    for (float &x : v) x = normal(rng);
    // Include one very low-probability token so gathering outside top-64 is checked explicitly.
    v[(size_t)c.row * c.n + 0] = -40.0f;
    v[(size_t)c.row * c.n + c.n - 1] = 25.0f;

    auto *dv = static_cast<float *>(rt.alloc_device(card, v.size() * 4));
    auto *dtok = static_cast<int32_t *>(rt.alloc_host(c.tokens.size() * 4));
    auto *dlog = static_cast<float *>(rt.alloc_host(c.tokens.size() * 4));
    auto *dstat = static_cast<float *>(rt.alloc_host(8));
    std::memcpy(dtok, c.tokens.data(), c.tokens.size() * 4);
    for (size_t i = 0; i < c.tokens.size(); i++) dlog[i] = -99999.0f;
    up.copy(dv, v.data(), v.size() * 4);
    const uint32_t ntok = (uint32_t)c.tokens.size();
    up.launch(k_gather, {dv, c.n, c.row, c.n, c.offset, dtok, ntok, dlog, dstat}, 1, 1, 1, 256);
    up.synchronize();

    const float *row = v.data() + (size_t)c.row * c.n;
    float mx = -INFINITY;
    for (uint32_t i = 0; i < c.n; i++) mx = std::max(mx, row[i]);
    double sum = 0.0;
    for (uint32_t i = 0; i < c.n; i++) sum += std::exp((double)row[i] - (double)mx);

    bool ok = std::memcmp(&dstat[0], &mx, 4) == 0;
    const double rel = std::abs((double)dstat[1] - sum) / std::max(1.0, sum);
    ok &= rel < 1e-5;
    for (size_t i = 0; i < c.tokens.size(); i++) {
      const int32_t tok = c.tokens[i];
      if (tok >= (int32_t)c.offset && (uint32_t)(tok - (int32_t)c.offset) < c.n) {
        ok &= std::memcmp(&dlog[i], &row[tok - (int32_t)c.offset], 4) == 0;
      } else {
        ok &= dlog[i] == -99999.0f;
      }
    }
    failures += !ok;
    std::printf("n %6u, offset %6u, row %u/%u: %s\n", c.n, c.offset, c.row, c.rows, ok ? "ok" : "FAIL");
    rt.free(dv);
    rt.free(dtok);
    rt.free(dlog);
    rt.free(dstat);
  }
  EXPECT(failures == 0, "%d case(s) wrong", failures);
  std::printf("PASS\n");
  return 0;
}
