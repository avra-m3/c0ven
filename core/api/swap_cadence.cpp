#include "api/swap_cadence.h"

#include <algorithm>

namespace engine {

namespace {
constexpr double kDecay = 0.98;
constexpr double kMinWeight = 8;     // samples a line needs
constexpr double kMinVariance = 0.25;  // of the layer counts, so that the line has a slope
}  // namespace

double MissCost::slope(const Sums &s) {
  if (s.n < kMinWeight) return 0;
  const double mx = s.x / s.n, var = s.xx / s.n - mx * mx;
  if (var < kMinVariance) return 0;
  return std::max(0.0, (s.xy / s.n - mx * s.y / s.n) / var);
}

void MissCost::add(uint32_t rows, uint32_t layers_missed, double ms) {
  if (rows == 0 || rows >= kWidths) return;
  Sums &s = sums_[rows];
  const double x = layers_missed;
  s.n = kDecay * s.n + 1;
  s.x = kDecay * s.x + x;
  s.y = kDecay * s.y + ms;
  s.xx = kDecay * s.xx + x * x;
  s.xy = kDecay * s.xy + x * ms;
}

double MissCost::ms(uint32_t rows, uint32_t layers_missed) const {
  rows = std::clamp(rows, 1u, kWidths - 1);
  for (uint32_t d = 0; d < kWidths; d++) {
    for (const uint32_t w : {rows - std::min(d, rows), rows + d}) {
      if (w == 0 || w >= kWidths) continue;
      if (const double b = slope(sums_[w]); b > 0) return b * layers_missed;
    }
  }
  return 0;
}

void SwapCadence::step(uint32_t rows, uint32_t layers_missed, double ms) {
  const double extra = cost_.ms(rows, layers_missed);
  if (extra > 0) fitted_ = true;
  missed_ms_ += extra;
  cost_.add(rows, layers_missed, ms);
}

void SwapCadence::round(double ms) {
  round_ms_ = ms;
  missed_ms_ = 0;
  plans_ = 0;
}

bool SwapCadence::due(uint32_t writers) const {
  if (round_ms_ <= 0 || !fitted_) return plans_ >= (writers > 1 ? kMaxPlans : kMinPlans);
  return plans_ >= kMaxPlans || (plans_ >= kMinPlans && missed_ms_ >= round_ms_);
}

}  // namespace engine
