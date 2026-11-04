// Level 1 (docs/TESTING.md): YaRN's angles and magnitude (yarn.h) against docs/spec/MODEL_LAYER.md, section 3.3.
// No card.
#include <cmath>
#include <cstdio>

#include "models/qwen4exp/yarn.h"
#include "test_util.h"

int main() {
  const float fs = (float)Q_TRAINED_CTX / 1048576.0f;  // s = 4
  // At the trained context nothing changes, bit for bit.
  for (unsigned i = 0; i < 32; i++) EXPECT(q_yarn_theta(123.456f, i, 1.0f) == 123.456f, "pair %u at s = 1", i);
  EXPECT(q_yarn_scale(1.0f) == 1.0f, "the magnitude at s = 1");
  // Pairs up to 14 keep the trained angle, pairs from 22 take theta / s, and those between blend in eighths.
  for (unsigned i = 0; i <= 14; i++) EXPECT(q_yarn_theta(8.0f, i, fs) == 8.0f, "pair %u", i);
  for (unsigned i = 22; i < 32; i++) EXPECT(q_yarn_theta(8.0f, i, fs) == 2.0f, "pair %u", i);
  for (unsigned i = 15; i < 22; i++) {
    const float want = 8.0f - 6.0f * (float)(i - 14) / 8.0f;
    EXPECT(std::fabs(q_yarn_theta(8.0f, i, fs) - want) < 1e-5f, "pair %u", i);
  }
  // 1 + 0.1 ln s: the section's 1.0405 at s = 1.5 and 1.0693 at s = 2.
  EXPECT(std::fabs(q_yarn_scale(1.0f / 1.5f) - 1.0405f) < 1e-4f, "the magnitude at s = 1.5");
  EXPECT(std::fabs(q_yarn_scale(0.5f) - 1.0693f) < 1e-4f, "the magnitude at s = 2");
  std::printf("yarn_test: ok\n");
  return 0;
}
