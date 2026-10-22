// Level 2 (docs/TESTING.md): common.cl's topk_rows against the host. Each row's k largest values, ties going to the
// lower indices, written in increasing index order with the offset added, and the count; for rows long enough for
// topk_select_few's ranking and short ones, with and without ties, and a k it does not take, so topk_select runs:
// among them the indexer's, about half of up to 1,026 values.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>

#include "test_util.h"
#include "ze/library.h"

using namespace engine;

namespace {

// The kernels' order: the float's bits as an unsigned key (kernels/include/topk.h), so -0 ranks below +0.
uint32_t key(float x) {
  uint32_t u;
  std::memcpy(&u, &x, 4);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

}  // namespace

int main() {
  if (test::gpus() == 0) return test::no_gpu();
  ze::Runtime rt;
  ze::Library lib(rt);
  const uint32_t card = 0;
  ze::List up(rt, card, ze::List::immediate);
  auto k_topk = lib.get(card, "common", "topk_rows");
  std::mt19937 rng(77);
  std::normal_distribution<float> normal(0.0f, 3.0f);
  struct Case {
    uint32_t n, k, rows;
    bool ties;
  };
  const Case cases[] = {{124160, 64, 1, false}, {124160, 64, 2, true},  {124161, 64, 3, false}, {248320, 64, 1, true},
                        {124160, 8, 4, false},  {124160, 65, 1, false}, {9000, 64, 2, true},     {1000, 64, 1, true},
                        {50, 64, 1, false},     {1026, 513, 2, false},  {1026, 512, 1, true},    {584, 513, 2, true},
                        {2047, 1000, 1, true},  {600, 1, 1, true}};
  int failures = 0;
  for (const Case &c : cases) {
    std::vector<float> v((size_t)c.rows * c.n);
    for (auto &x : v) x = c.ties ? std::round(normal(rng) * 2.0f) / 2.0f : normal(rng);
    auto *dv = static_cast<float *>(rt.alloc_device(card, v.size() * 4));
    auto *nrows = static_cast<uint32_t *>(rt.alloc_host(4));
    auto *idx = static_cast<uint32_t *>(rt.alloc_host((size_t)c.rows * c.k * 4));
    auto *val = static_cast<float *>(rt.alloc_host((size_t)c.rows * c.k * 4));
    auto *count = static_cast<uint32_t *>(rt.alloc_host(c.rows * 4));
    up.copy(dv, v.data(), v.size() * 4);
    *nrows = c.rows;
    const uint32_t offset = 1000000;
    up.launch(k_topk, {dv, c.n, c.n, c.k, offset, nrows, idx, val, count}, c.rows, 1, 1, 1024);
    up.synchronize();
    bool ok = true;
    for (uint32_t r = 0; r < c.rows; r++) {
      // The host's answer: indices by key, then by index; the first k, in increasing order.
      const float *row = v.data() + (size_t)r * c.n;
      std::vector<uint32_t> order(c.n);
      std::iota(order.begin(), order.end(), 0u);
      std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return key(row[a]) > key(row[b]); });
      uint32_t want = std::min(c.k, c.n);
      std::vector<uint32_t> top(order.begin(), order.begin() + want);
      std::sort(top.begin(), top.end());
      ok &= count[r] == want;
      for (uint32_t i = 0; i < want; i++) {
        ok &= idx[(size_t)r * c.k + i] == top[i] + offset;
        ok &= std::memcmp(&val[(size_t)r * c.k + i], &row[top[i]], 4) == 0;
      }
    }
    failures += !ok;
    std::printf("n %6u, k %4u, %u rows%s: %s\n", c.n, c.k, c.rows, c.ties ? ", ties" : "", ok ? "ok" : "FAIL");
    rt.free(dv);
    rt.free(nrows);
    rt.free(idx);
    rt.free(val);
    rt.free(count);
  }
  EXPECT(failures == 0, "%d case(s) wrong", failures);
  std::printf("PASS\n");
  return 0;
}
