// Drafts from a slot's own tokens, and the choice between them and the draft layer's for a round of one slot
// (docs/adr/0041-drafts-from-the-slots-own-tokens.md), and the drafts of several slots, apart from the C ABI's code so
// that they are tested without a card.
#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <vector>

#include "step.h"

namespace engine {

// The shortest match of the slot's last tokens with an earlier run of them that proposes drafts.
constexpr uint32_t kLookupMin = 12;

// Every run of three tokens of a slot's history and where it ends, kept up to date as the history grows.
class Lookup {
 public:
  // The tokens that followed the longest earlier match (at least kLookupMin and at most 64 tokens) of tokens with
  // next appended, at most cap of them and none from the first token stop holds for; empty when there is none. The
  // match is looked for at the last 4 places the last three tokens ended, and the newest wins a tie. tokens only
  // grows between calls, or the index is built again.
  std::vector<int32_t> drafts(const std::vector<int32_t> &tokens, int32_t next, uint32_t cap,
                              const std::function<bool(int32_t)> &stop);
  // Brings the index up to tokens, as drafts does first: called as a prompt is read, so that the first round after it
  // does not index the whole prompt.
  void index(const std::vector<int32_t> &tokens);

 private:
  std::unordered_map<uint64_t, std::vector<uint32_t>> ends_;
  uint32_t indexed_ = 0;  // the tokens the index covers
};


// How a window's time grows with its rows, from the rounds the core runs: each width's running mean, and for a width
// not seen yet a line through the nearest widths seen (past the widest, through the two widest). A width's second
// time replaces its first, which may hold the first run of the width's lists.
class WindowTimes {
 public:
  void add(uint32_t rows, double ms);
  // The time of a window of rows; 0 while no width has been seen.
  double ms(uint32_t rows) const;

 private:
  static constexpr uint32_t kWidths = STEP_MAX_ROWS + 1;  // 0 to STEP_MAX_ROWS
  double ms_[kWidths] = {};
  uint32_t n_[kWidths] = {};
};

// How the window of a step or round of several slots grows with its rows, apart from the routed experts the CPU
// computes in it. A miss costs about the same whatever the window's width, while the misses a row change with the
// slots' text, so each width keeps its times and misses (decayed by 1/16 a round of the width), one cost a miss is
// fitted across the widths (each width's times against its misses), and a width's net time is its time less that
// cost of its misses. A width's second round replaces its first, which may hold the first run of the width's lists,
// and only a width timed twice is seen. For a width not seen, the line through the nearest seen ones (past the widest,
// through the two widest, no less than the widest; below the narrowest, through the two narrowest, no less than in
// proportion to the rows). A width is never dearer than a wider one seen.
class RoundTimes {
 public:
  void add(uint32_t rows, double ms, uint32_t misses);
  // A window of rows' time without misses; 0 while no width has been seen.
  double net(uint32_t rows) const { return net_[std::min(rows, kWidths - 1)]; }
  // The time a miss adds, in milliseconds; 0 until the widths' misses differ.
  double miss_ms() const { return miss_ms_; }
  // The misses a row of the rounds lately, of any width, decayed by 1/8 a round; 0 before the first.
  double misses_row() const { return misses_row_ < 0 ? 0 : misses_row_; }
  // A window of rows' time with the misses a row of the rounds lately; 0 while no width has been seen.
  double ms(uint32_t rows) const { return seen_ ? net(rows) + miss_ms_ * misses_row() * rows : 0; }
  // The widths seen so far, and whether rows is one.
  uint32_t seen() const { return seen_; }
  bool timed(uint32_t rows) const { return rows > 0 && rows < kWidths && w_[rows].rounds >= 2; }

 private:
  static constexpr uint32_t kWidths = STEP_MAX_ROWS + 1;  // 0 to STEP_MAX_ROWS
  struct Sums {
    double n = 0, m = 0, t = 0, mm = 0, mt = 0;  // weight, misses, ms, misses squared, misses times ms
    uint32_t rounds = 0;
  };
  void refresh();
  Sums w_[kWidths];
  double net_[kWidths] = {}, miss_ms_ = 0, misses_row_ = -1;
  uint32_t seen_ = 0;
};

// The places in a round of the draft layer's drafts whose acceptance DraftOdds keeps apart: the first kOddsPlaces.
constexpr uint32_t kOddsPlaces = 8;

// The rounds that must have tried or checked a slot's first draft before several slots' even split may give it none: a
// round without drafts measures no acceptance, so a low first guess would otherwise keep a request without drafts.
constexpr uint32_t kFirstChecks = 4;

// What one slot's rounds have accepted lately: the lookup's drafts accepted and its rounds that stopped at a draft,
// and the draft layer's drafts accepted and checked a round (up to the first one rejected), each decayed by 0.9 a
// round. Also, at each place of a round of the draft layer's drafts, the first draft's, the second's and so on:
// - made: the rounds that asked for a draft there after the draft layer made the one before (tried), and of them those
//   it made it in; the draft layer stops at a draft it is unsure of, so a round may hold fewer drafts than it asked;
// - accepted: the rounds with a draft there and every draft before it accepted (checked), and of them those that
//   accepted it;
// each decayed by 0.9 a round that tries or checks the place, so that a place few rounds reach keeps what they gave.
struct DraftOdds {
  double lookup_accepted = 9.0, lookup_failed = 1.0, layer_accepted = 2.0, layer_offered = 3.0;
  double place_accepted[kOddsPlaces] = {}, place_checked[kOddsPlaces] = {};
  double place_made[kOddsPlaces] = {}, place_tried[kOddsPlaces] = {};
  uint32_t first_checks = 0;  // the rounds that tried or checked the first place
  void lookup_round(uint32_t accepted, uint32_t drafts);
  // A round of the draft layer's drafts, of which it checked drafts and accepted accepted. asked is the drafts the
  // round asked of the draft layer, which may have made fewer; 0 where the round does not say.
  void layer_round(uint32_t accepted, uint32_t drafts, uint32_t asked = 0);
  // The draft layer's drafts accepted in a round of k: its acceptance a draft, accepted over checked, to the powers
  // 1 to k added (a draft counts only when those before it do).
  double layer_expected(uint32_t k) const;
  // The draft layer's drafts accepted in a round that asks for k, with each place's own figures: a draft counts when
  // it is made and accepted and those before it are. A place's acceptance is its rounds', which start from the place
  // before it's (the first place's from the acceptance a draft), as if that had been their first; its share made
  // likewise, the first place's from 1. A slot's second and third drafts are accepted less often than its first on
  // most text, which one acceptance for all of them hides.
  double place_expected(uint32_t k) const;
  // The drafts the draft layer is expected to make in a round that asks for k: a draft is made when those before it
  // are, at each place's share made.
  double place_drafts(uint32_t k) const;
  // The chance that it makes 0, 1, ... k of them.
  std::vector<double> drafts_made(uint32_t k) const;
};

// The drafts for a round of one slot: how many of the lookup's to take, 0 for the draft layer's. Each choice is worth
// its expected tokens (the row sampled last and the drafts accepted, the lookup's one after another with its
// acceptance so far, the draft layer's at its mean) over its window's time. The lookup is offered as many drafts as
// the draft layer makes and as many as a window holds.
uint32_t choose_drafts(const DraftOdds &odds, uint32_t layer_drafts, uint32_t draft_max, uint32_t lookup_drafts,
                       uint32_t max_rows, const WindowTimes &times);

// The drafts each of several writing slots makes while no window is timed, and every kSharedCheck-th time.
constexpr uint32_t kSharedDrafts = 2;

// Every this many choices for several slots the core makes kSharedDrafts each of a step of kEvenRows rows, whatever
// choose_shared_drafts gives. Rounds with fewer drafts, or none, do not measure the draft layer's acceptance or the
// wider windows' times, so a low acceptance or a slow window measured once would otherwise keep the choice low for
// good.
constexpr uint32_t kSharedCheck = 8;

// The rows several slots shared their drafts out of before a step held 16
// (docs/adr/0077-a-decode-step-holds-16-rows.md).
constexpr uint32_t kEvenRows = 8;

// The drafts of n slots with k at most each, of a step of rows rows: the rows past one a slot shared out, each slot
// spare / n and spare % n of them one more, from slot first on (past the last, from the first), each at most
// draft_max.
std::vector<uint32_t> even_drafts(uint32_t n, uint32_t k, uint32_t rows, uint32_t draft_max, uint32_t first = 0);

// The same counts, with the spare % n drafts of one more going by each slot's tokens with spare / n (the row sampled
// last and place_expected): first to the slots one more makes faster, its tokens over the round's time with it
// (round_ms, of each slot's drafts asked; with no time, any slot it adds to) above the slot's with spare / n; then
// to the slots that write the fewest. Two slots next to each other in that order, both made faster, are close where
// one more draft would take the first past the second. Where the line between the slots that take one more and
// those that do not falls inside a run of close slots, the run takes the drafts left in turns, from its slot
// first % its length on, in the slots' order. turns, where given, is the run's length; 1 where there is none.
std::vector<uint32_t> split_drafts(const std::vector<DraftOdds> &slots, uint32_t k, uint32_t rows, uint32_t draft_max,
                                   const std::function<double(const std::vector<uint32_t> &)> &round_ms,
                                   uint32_t first = 0, uint32_t *turns = nullptr);

// The drafts each of several writing slots may make for their next round
// (docs/adr/0078-several-slots-take-more-rows-only-when-the-slowest-writes-no-slower.md): one count k for all of
// them, shared out as split_drafts shares it, worth the slots' expected tokens (each the row sampled last and its
// drafts accepted, at each place's acceptance and share made: place_expected) over the round's time, the window's
// (window.ms) and the draft layer's pass after it (drafter, by the window's rows), at the rows the draft layer is
// expected to make (place_drafts). Where k leaves some slots one draft more than others, split_drafts gives them
// from first, and the core moves first on a choice at a time; so a slot's tokens and the round's time are their
// means over the turns of split_drafts' close slots.
// - even: the k with the most tokens a millisecond of a step of kEvenRows rows, and at least 1 while a slot's first
//   draft has been checked in fewer than kFirstChecks rounds; kSharedDrafts while fewer than two widths are timed.
// - wide: a k whose rows are more than kEvenRows, of a step of max_rows, that gives more tokens a millisecond in all
//   and its slowest slot at least the tokens a millisecond of the even split's slowest; the one with the most. Empty
//   when none does, or with max_rows at most kEvenRows. While fewer than two widths are timed: kSharedDrafts + 1 each
//   of max_rows rows, so that a wider width is timed.
// With a wide count, both carry their estimates for the trial: the slots' expected tokens and the round's time.
struct SharedChoice {
  std::vector<uint32_t> even, wide;
  double even_tokens = 0, even_ms = 0, wide_tokens = 0, wide_ms = 0;
};
SharedChoice choose_shared_drafts(const std::vector<DraftOdds> &slots, uint32_t draft_max, uint32_t max_rows,
                                  const RoundTimes &window, const WindowTimes &drafter, uint32_t first = 0);

// Whether rounds of several slots take choose_shared_drafts' wide count or its even split. Each arm keeps what its
// rounds gave against what choose_shared_drafts estimated for them: the tokens they wrote over the time of their
// windows and of the draft layer's passes after them, and the estimated tokens over the estimated time, each decayed
// by kTrialDecay a round of its own. An arm's correction is the first rate over the second, starting from
// kTrialPrior (as kTrialPriorRounds rounds at its first round's estimates). A round's estimates follow the slots'
// text and the misses of the moment, so the corrections compare the arms even when one last ran on other text. A
// round takes the arm whose estimate times its correction is the higher, and every kTrialCheck-th choice the other
// (every kTrialCheck / 2-th while the other has fewer than kTrialRounds rounds), so that its correction stays
// current. A choice without estimates is not counted and takes the wide count while the arms have as many rounds.
// The core keeps one for each number of slots.
constexpr uint32_t kTrialRounds = 8;
constexpr uint32_t kTrialCheck = 32;
constexpr double kTrialDecay = 1.0 / 16;
constexpr double kTrialPrior[2] = {1.0, 0.85};  // the even split's, the wide count's
constexpr uint32_t kTrialPriorRounds = 4;
class RoundTrial {
 public:
  bool wide(const SharedChoice &c);  // this choice's: true for the wide count
  // A round's tokens and window, with its arm's estimates.
  void add_round(bool wide, double tokens, double ms, double est_tokens, double est_ms);
  void add_time(bool wide, double ms);  // the draft layer's pass after a round
  double correction(bool wide) const;
  uint32_t rounds(bool wide) const { return rounds_[wide]; }

 private:
  double tokens_[2] = {}, ms_[2] = {}, est_tokens_[2] = {}, est_ms_[2] = {};
  uint32_t rounds_[2] = {}, picks_ = 0;
};

}  // namespace engine
