// When a swap round is due: once the time the CPU's expert misses have cost since the last round reaches the time that
// round took. Apart from the C ABI's code so that it is tested without a card.
#pragma once

#include <cstdint>

namespace engine {

// How much time a missed layer adds to a step of a given width, from the steps the core runs: per width, a line
// through (layers missed, step ms), each sample's weight decayed by 0.98 a step.
class MissCost {
 public:
  void add(uint32_t rows, uint32_t layers_missed, double ms);
  // The time a step of rows with layers_missed layers missed adds; a width with no line yet takes the nearest width
  // with one, and 0 while there is none.
  double ms(uint32_t rows, uint32_t layers_missed) const;

 private:
  static constexpr uint32_t kWidths = 9;  // 0 to STEP_MAX_ROWS
  struct Sums {
    double n = 0, x = 0, y = 0, xx = 0, xy = 0;
  };
  // The line's slope in ms per layer missed; 0 until the samples are enough and their layer counts differ.
  static double slope(const Sums &s);
  Sums sums_[kWidths];
};

// The plans between swap rounds: the cost of the misses against the cost of a round, bounded by kMinPlans and
// kMaxPlans.
class SwapCadence {
 public:
  // A swap round gives the swap thread kMinPlans plans to build the next one, and the cache is not left longer than
  // kMaxPlans without one.
  static constexpr uint32_t kMinPlans = 4;
  static constexpr uint32_t kMaxPlans = 16;

  // A decode step or draft round of rows that missed layers_missed layers and took ms.
  void step(uint32_t rows, uint32_t layers_missed, double ms);
  // A plan has run.
  void plan() { plans_++; }
  // A swap round that took ms has run.
  void round(double ms);
  // Whether a round is due. With several writers, every kMaxPlans plans
  // (docs/measurements/MEASUREMENTS.md#2026-10-28-swaps-every-4th-plan). With one, every kMinPlans plans while no
  // round has been timed or no cost is fitted.
  bool due(uint32_t writers) const;

 private:
  MissCost cost_;
  double round_ms_ = 0;  // the last round's time
  double missed_ms_ = 0; // the misses' estimated cost since the last round
  uint32_t plans_ = 0;   // plans since the last round
  bool fitted_ = false;  // a cost was estimated for some step
};

}  // namespace engine
