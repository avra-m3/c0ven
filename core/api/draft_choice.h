// Which of the draft layer's drafts a round of several slots verifies
// (docs/adr/0077-a-decode-step-holds-16-rows.md): each draft's chance of being accepted, fitted to what the main model
// accepted, and the drafts of all slots ranked by it. Apart from the C ABI's code so that it is tested without a card.
#pragma once

#include <cstdint>
#include <vector>

#include "api/lookup.h"
#include "step.h"

namespace engine {

// The chance that the main model accepts a draft, given that it accepted the slot's drafts before it: a logistic
// model of the log-odds of the probability the sampler gave the draft, with one slope and an offset for each place
// in the block. It starts as that probability itself and is fitted again every kRefit drafts verified.
class DraftEstimate {
 public:
  static constexpr uint32_t kPlaces = STEP_MAX_RUN - 1;  // a slot's drafts in a round at most
  static constexpr uint32_t kRefit = 256;                // drafts verified between fits
  static constexpr double kDamping = 500.0;              // a fit on this many drafts moves half a Newton step
  static constexpr double kRidge = 1e-3;

  // Draft number place (0 the first after the token sampled last) that the sampler picked with probability p.
  double accept(uint32_t place, double p) const;
  // A verified draft: every draft the main model accepted, and the first it rejected.
  void observe(uint32_t place, double p, bool accepted);

  double slope() const { return slope_; }
  double offset(uint32_t place) const { return offset_[place < kPlaces ? place : kPlaces - 1]; }

 private:
  void refit();
  double slope_ = 1.0, offset_[kPlaces] = {};
  // Since the last fit: the gradient and the second derivatives of the log-likelihood, and each place's drafts.
  double g_slope_ = 0, h_slope_ = 0, g_offset_[kPlaces] = {}, h_offset_[kPlaces] = {}, h_cross_[kPlaces] = {};
  uint32_t n_[kPlaces] = {}, since_ = 0;
};

// The step's spare rows shared among several writing slots. The step holds max_rows rows and rows of them are taken
// (one a slot in the step). Each slot with a cap gets one row while rows are left, so that its acceptance stays
// measured; the rest go one at a time to the slot whose next draft is the likeliest to be accepted, by the acceptance
// a draft of its last rounds (DraftOdds), the lower slot first on a tie. caps[i] is the most slot i may make.
std::vector<uint32_t> share_rows(const std::vector<DraftOdds> &slots, const std::vector<uint32_t> &caps,
                                 uint32_t rows, uint32_t max_rows);

// The drafts each of several writing slots makes for the next round: its share_rows, cut to one more than
// choose_verified would take of them by the slot's acceptance a draft. A draft the round does not verify costs the
// draft layer a row for nothing, and the draft layer's pass widens with its rows. The one more leaves trim_drafts a
// choice by each draft's own chance. The whole share while fewer than two widths are timed.
std::vector<uint32_t> share_drafts(const std::vector<DraftOdds> &slots, const std::vector<uint32_t> &caps,
                                   uint32_t rows, uint32_t max_rows, const WindowTimes &times);

// Every this many choices that verify no draft at all, the core verifies each slot's first draft whatever
// choose_verified gave. A round with no drafts measures neither the estimate nor the wider windows' times, so an
// estimate or a time that is once too low would otherwise keep the choice at none for good.
constexpr uint32_t kVerifyCheck = 8;

// How many of each slot's drafts a round verifies. survival[i][j] is the chance that slot i's drafts 0 to j are all
// accepted, so it does not rise with j. The step has rows rows before any draft, each worth one token. The drafts of
// all slots are ranked by survival, and the round verifies the N likeliest, N the count that gives the most expected
// tokens a millisecond by the window times; at most max_rows rows in all. Every draft that fits while fewer than two
// widths are timed: one timed width prices a wider window in proportion to its rows, and no draft would pay.
std::vector<uint32_t> choose_verified(const std::vector<std::vector<double>> &survival, uint32_t rows,
                                      uint32_t max_rows, const WindowTimes &times);

}  // namespace engine
