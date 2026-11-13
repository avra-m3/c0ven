// Level 1 (docs/TESTING.md): whether a step's rows fit the KV pool (models/qwen4exp/model.h, step_fits_ring, which
// Model::step_fits asks at the ring's size now), on made-up rows and rings. No card.
#include <cstdint>
#include <cstdio>
#include <vector>

#include "models/qwen4exp/model.h"
#include "step.h"
#include "test_util.h"

using engine::StepIn;
using engine::qwen4exp::step_fits_ring;

namespace {

// n rows of slot from position pos on.
std::vector<StepIn> run(uint32_t slot, uint32_t pos, uint32_t n) {
  std::vector<StepIn> r;
  for (uint32_t i = 0; i < n; i++) r.push_back({slot, pos + i});
  return r;
}

std::vector<StepIn> cat(std::vector<StepIn> a, const std::vector<StepIn> &b) {
  a.insert(a.end(), b.begin(), b.end());
  return a;
}

bool fits(const std::vector<StepIn> &r, bool streamed, uint32_t row_claims, uint32_t ring) {
  return step_fits_ring(r.data(), (uint32_t)r.size(), streamed, row_claims, ring);
}

}  // namespace

int main() {
  static_assert(STEP_MAX_RUN == 8 && STEP_MAX_ROWS == 16, "the rows below are for 8 and 16");
  const uint32_t claims = 40;  // a row's claims at most, as kvs_row_claims gives for a shape

  // Slot 0 at positions 100 to 105 claims 59 pages (its last row's page, 26, and the 32 after it, short of 6 rows'
  // 240); slot 1 at 2000 to 2009, 400 (10 rows' 40, short of its last page's 535): 459 in all.
  const std::vector<StepIn> wide = cat(run(0, 100, 6), run(1, 2000, 10));
  EXPECT(fits(wide, true, claims, 460), "a ring of 460 holds 459");
  EXPECT(!fits(wide, true, claims, 459), "a ring of 459 does not");

  // A slot's rows counted once wherever they fall in the step: the same claims with the slots' rows interleaved.
  std::vector<StepIn> mixed;
  for (uint32_t i = 0; i < 6; i++) mixed.push_back({0, 100 + i}), mixed.push_back({1, 2000 + i});
  mixed = cat(mixed, run(1, 2006, 4));
  EXPECT(fits(mixed, true, claims, 460) && !fits(mixed, true, claims, 459), "interleaved");

  // Up to STEP_MAX_RUN rows always fit, the ring's fixed part holding their claims, and with every page resident up
  // to STEP_MAX_ROWS.
  EXPECT(fits(run(1, 2000, 8), true, claims, 0), "8 rows, no ring");
  EXPECT(fits(wide, false, claims, 0), "16 rows, every page resident");

  // No rows, or more than STEP_MAX_ROWS, never fit.
  EXPECT(!fits({}, true, claims, 1000) && !fits({}, false, claims, 1000), "no rows");
  const std::vector<StepIn> over = cat(wide, run(2, 0, 1));
  EXPECT(!fits(over, true, claims, 100000) && !fits(over, false, claims, 100000), "17 rows");

  std::printf("step_fits_test: ok\n");
  return 0;
}
