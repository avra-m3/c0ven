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
  static constexpr uint32_t kRefit = 64;  // drafts verified between fits
  // A fit on this many drafts moves half a Newton step: half the drafts seen, from kDampingMin to kDampingMax, so
  // the first fits move most of the way and the later ones average over about kDampingMax drafts.
  static constexpr double kDampingMin = 32.0, kDampingMax = 512.0;
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
  uint64_t seen_ = 0;  // drafts verified in all
};

// The step's spare rows shared among several writing slots. The step holds max_rows rows and rows of them are taken
// (one a slot in the step). Each slot with a cap gets one row while rows are left, so that its acceptance stays
// measured; the rest go one at a time to the slot whose next draft is the likeliest to be accepted with those before
// it, by its last rounds' acceptance at each place (DraftOdds::layer_place), the lower slot first on a tie. caps[i] is
// the most slot i may make.
std::vector<uint32_t> share_rows(const std::vector<DraftOdds> &slots, const std::vector<uint32_t> &caps,
                                 uint32_t rows, uint32_t max_rows);

// The drafts each of several writing slots makes for the next round: the first N rows share_rows gives out, N the
// count that gives the most expected tokens a millisecond, and at least one for each slot that may draft. A round of
// rows + N rows costs window's time for its rows and drafter's for rows + N: the draft layer's pass after a round
// reads its rows and drafts one block row for each slot and draft. Until two widths of window are timed, the first
// rows up to a round of kStartRows, at least the floor. Then more of share_rows' order, to fill the block pass's
// power-of-2 width: those drafts cost the pass nothing, and choose_verified weighs them with the rest.
constexpr uint32_t kStartRows = STEP_MAX_RUN;
std::vector<uint32_t> share_drafts(const std::vector<DraftOdds> &slots, const std::vector<uint32_t> &caps,
                                   uint32_t rows, uint32_t max_rows, const WindowTimes &window,
                                   const WindowTimes &drafter);

// Every this many choices that verify no draft at all, the core verifies each slot's first draft whatever
// choose_verified gave. A round with no drafts measures neither the estimate nor the wider windows' times, so an
// estimate or a time that is once too low would otherwise keep the choice at none for good.
constexpr uint32_t kVerifyCheck = 8;

// How many of each slot's drafts a round verifies. survival[i][j] is the chance that slot i's drafts 0 to j are all
// accepted, so it does not rise with j. The step has rows rows before any draft, each worth one token. The drafts of
// all slots are ranked by survival, and the round verifies the N likeliest, N the count that gives the most expected
// tokens a millisecond by the window times and the draft layer's pass after the round, drafter's time for the
// round's rows or block, the block rows that made these drafts, if that is more; at most max_rows rows in all. Every
// draft that fits while fewer than two widths are timed: one timed width prices a wider window in proportion to its
// rows, and no draft would pay. extra more of the ranked drafts than that count, where there are, so that the widths
// next to the choice stay timed.
std::vector<uint32_t> choose_verified(const std::vector<std::vector<double>> &survival, uint32_t rows,
                                      uint32_t max_rows, const WindowTimes &times, const WindowTimes &drafter,
                                      uint32_t block, uint32_t extra = 0);

// Every this many rounds of several slots, the round verifies kExploreRows more drafts than choose_verified chose.
constexpr uint32_t kExplore = 8, kExploreRows = 2;

}  // namespace engine
