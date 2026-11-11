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

// What one slot's rounds have accepted lately: the lookup's drafts accepted and its rounds that stopped at a draft,
// and the draft layer's drafts accepted and checked a round (up to the first one rejected), each decayed by 0.9 a
// round.
struct DraftOdds {
  double lookup_accepted = 9.0, lookup_failed = 1.0, layer_accepted = 2.0, layer_offered = 3.0;
  void lookup_round(uint32_t accepted, uint32_t drafts);
  void layer_round(uint32_t accepted, uint32_t drafts);
  // The draft layer's drafts accepted in a round of k: its acceptance a draft, accepted over checked, to the powers
  // 1 to k added (a draft counts only when those before it do).
  double layer_expected(uint32_t k) const;
};

// The drafts for a round of one slot: how many of the lookup's to take, 0 for the draft layer's. Each choice is worth
// its expected tokens (the row sampled last and the drafts accepted, the lookup's one after another with its
// acceptance so far, the draft layer's at its mean) over its window's time. The lookup is offered as many drafts as
// the draft layer makes and as many as a window holds.
uint32_t choose_drafts(const DraftOdds &odds, uint32_t layer_drafts, uint32_t draft_max, uint32_t lookup_drafts,
                       uint32_t max_rows, const WindowTimes &times);

// The drafts each of several writing slots makes until two widths of their rounds are timed, and every
// kSharedCheck-th time.
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
// spare / n and the first spare % n one more, each at most draft_max.
std::vector<uint32_t> even_drafts(uint32_t n, uint32_t k, uint32_t rows, uint32_t draft_max);

// The drafts each of several writing slots makes for their next round
// (docs/adr/0078-several-slots-take-more-rows-only-when-none-writes-slower.md). One count k for all of them, shared
// out as even_drafts shares it, is worth the slots' expected tokens (each the row sampled last and its drafts
// accepted, at its acceptance) over the round's time: its window's (window.ms) and the draft layer's pass after it
// (drafter, by the window's rows). The even split is the k with the most tokens a millisecond of a step of kEvenRows
// rows, the choice before a step held 16. Of the k of a step of max_rows rows that give each slot at least its tokens
// a millisecond in the even split, the round takes the one with the most tokens a millisecond, or the even split
// when none has more. While fewer than two widths are timed: kSharedDrafts each of kEvenRows rows, and once that
// width is timed, kSharedDrafts + 1 each of max_rows rows, so that a wider width is timed too.
std::vector<uint32_t> choose_shared_drafts(const std::vector<DraftOdds> &slots, uint32_t draft_max, uint32_t max_rows,
                                           const RoundTimes &window, const WindowTimes &drafter);

}  // namespace engine
