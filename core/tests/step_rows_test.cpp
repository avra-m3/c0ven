// Level 3 (docs/TESTING.md): the decode kernels that take more than 8 of a step's rows. layer.cl's moe_group puts every
// cached assignment of a step of up to STEP_MAX_ROWS rows in one group of at most MOE_GROUP_MAX, an expert more rows
// route to split in order of its assignments; common.cl's mv_xmx_bf16 gives rows 0 to 7 the same bits at every row
// count, and every row within 2e-3 of the largest |reference| (TESTING.md, "Tolerances").
#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include "models/qwen4exp/kernel_sizes.h"
#include "moe_tile.h"
#include "step.h"
#include "test_util.h"
#include "ze/library.h"

using namespace engine;

namespace {

uint16_t to_bf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return (uint16_t)((u + (0x7fffu + ((u >> 16) & 1))) >> 16);
}

float from_bf16(uint16_t h) {
  uint32_t u = (uint32_t)h << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

struct Group {
  int32_t slot;
  std::vector<uint32_t> asg;
};

// moe_group's groups on the host: in order of each group's first assignment.
std::vector<Group> host_groups(const std::vector<int32_t> &slots, uint32_t n) {
  std::vector<Group> out;
  for (uint32_t a = 0; a < n; a++) {
    const int32_t s = slots[a];
    if (s < 0) continue;
    int at = -1;
    for (int g = (int)out.size() - 1; g >= 0; g--)
      if (out[g].slot == s) {
        at = g;
        break;
      }
    if (at < 0 || out[at].asg.size() == MOE_GROUP_MAX) {
      out.push_back({s, {}});
      at = (int)out.size() - 1;
    }
    out[at].asg.push_back(a);
  }
  return out;
}

}  // namespace

int main() {
  if (test::gpus() == 0) return test::no_gpu();
  ze::Runtime rt;
  ze::Library lib(rt);
  const uint32_t card = 0;
  ze::List L(rt, card, ze::List::immediate);
  int failures = 0;

  // moe_group: expert slot 5 in every row, slot 7 in the first 12 rows, the rest distinct, every other row's last a
  // miss.
  {
    auto *plan = static_cast<StepPlan *>(rt.alloc_host(sizeof(StepPlan)));
    auto *slots = static_cast<int32_t *>(rt.alloc_host(STEP_MAX_ROWS * Q_USED * 4));
    auto *groups = static_cast<MoeGroup *>(rt.alloc_host(STEP_MAX_ROWS * Q_USED * sizeof(MoeGroup)));
    auto *ngroups = static_cast<uint32_t *>(rt.alloc_host(4));
    std::vector<int32_t> s(STEP_MAX_ROWS * Q_USED);
    for (uint32_t r = 0; r < STEP_MAX_ROWS; r++)
      for (uint32_t k = 0; k < Q_USED; k++) {
        int32_t v = 100 + (int32_t)(r * Q_USED + k);
        if (k == 0) v = 5;
        if (k == 1 && r < 12) v = 7;
        if (k == Q_USED - 1 && r % 2) v = -1;
        s[r * Q_USED + k] = v;
      }
    std::memcpy(slots, s.data(), s.size() * 4);
    for (uint32_t nrows : {1u, 8u, 9u, 12u, 16u}) {
      std::memset(plan, 0, sizeof(StepPlan));
      plan->nrows = nrows;
      std::memset(groups, 0xff, STEP_MAX_ROWS * Q_USED * sizeof(MoeGroup));
      L.launch(lib.get(card, "qwen4exp", "moe_group"), {slots, plan, groups, ngroups}, 1, 1, 1, MOE_GROUP_WG);
      L.synchronize();
      const std::vector<Group> want = host_groups(s, nrows * Q_USED);
      if (*ngroups != want.size()) {
        std::printf("FAIL moe_group, %u rows: %u groups, want %zu\n", nrows, *ngroups, want.size());
        failures++;
        continue;
      }
      for (size_t g = 0; g < want.size(); g++) {
        bool ok = groups[g].slot == want[g].slot && groups[g].n == want[g].asg.size();
        for (uint32_t i = 0; ok && i < groups[g].n; i++) ok = groups[g].asg[i] == want[g].asg[i];
        if (!ok) {
          std::printf("FAIL moe_group, %u rows: group %zu (slot %d, %u assignments)\n", nrows, g, groups[g].slot,
                      groups[g].n);
          failures++;
          break;
        }
      }
    }
  }

  // mv_xmx_bf16 at 8, 12 and 16 rows.
  {
    const uint32_t N = 96, K = 256, ks = 4;
    std::mt19937 rng(11);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<uint16_t> W((size_t)N * K), X((size_t)STEP_MAX_ROWS * K);
    for (auto &w : W) w = to_bf16(normal(rng) * 0.05f);
    for (auto &x : X) x = to_bf16(normal(rng));
    auto *dW = static_cast<uint16_t *>(rt.alloc_device(card, W.size() * 2));
    auto *dX = static_cast<uint16_t *>(rt.alloc_device(card, X.size() * 2));
    auto *ntok = static_cast<uint32_t *>(rt.alloc_host(4));
    auto *y = static_cast<float *>(rt.alloc_host((size_t)STEP_MAX_ROWS * N * 4));
    L.copy(dW, W.data(), W.size() * 2);
    L.copy(dX, X.data(), X.size() * 2);
    L.synchronize();
    const uint32_t zero = 0;
    const float one = 1.0f;
    uint16_t *no_yb = nullptr;
    std::vector<float> first8;
    for (uint32_t n : {8u, 12u, 16u}) {
      *ntok = n;
      std::fill(y, y + (size_t)STEP_MAX_ROWS * N, NAN);
      L.launch(lib.get(card, "common", "mv_xmx_bf16"), {dW, N, K, K, dX, ntok, y, N, zero, one, N, ks, no_yb},
               (N + 15) / 16, 1, 1, 16 * ks);
      L.synchronize();
      double big = 0, worst = 0;
      for (uint32_t t = 0; t < n; t++)
        for (uint32_t i = 0; i < N; i++) {
          double ref = 0;
          for (uint32_t k = 0; k < K; k++)
            ref += (double)from_bf16(W[(size_t)i * K + k]) * from_bf16(X[(size_t)t * K + k]);
          big = std::max(big, std::fabs(ref));
          worst = std::max(worst, std::fabs(ref - y[(size_t)t * N + i]));
        }
      if (!(worst <= 2e-3 * big)) {
        std::printf("FAIL mv_xmx_bf16, %u rows: off by %g of %g\n", n, worst, big);
        failures++;
      }
      if (n == 8) first8.assign(y, y + (size_t)8 * N);
      else if (std::memcmp(first8.data(), y, (size_t)8 * N * 4) != 0) {
        std::printf("FAIL mv_xmx_bf16, %u rows: rows 0-7 differ from the 8-row product\n", n);
        failures++;
      }
    }
  }

  if (failures) return 1;
  std::printf("step_rows_test: ok\n");
  return 0;
}
