// Drafts from a slot's own tokens, and the choice between them and the draft layer's for a round of one slot
// (docs/adr/0041-drafts-from-the-slots-own-tokens.md), apart from the C ABI's code so that they are tested without a
// card.
#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

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
  std::vector<int32_t> drafts(const std::vector<int32_t> &tokens, int32_t next, uint32_t cap, bool (*stop)(int32_t));
  // Brings the index up to tokens, as drafts does first: called as a prompt is read, so that the first round after it
  // does not index the whole prompt.
  void index(const std::vector<int32_t> &tokens);

 private:
  std::unordered_map<uint64_t, std::vector<uint32_t>> ends_;
  uint32_t indexed_ = 0;  // the tokens the index covers
};

// How a window's time grows with its rows, from the rounds the core runs: each width's running mean, and for a width
// not seen yet a line through the nearest widths seen.
class WindowTimes {
 public:
  void add(uint32_t rows, double ms);
  // The time of a window of rows; 0 while no width has been seen.
  double ms(uint32_t rows) const;

 private:
  static constexpr uint32_t kWidths = 9;  // 0 to STEP_MAX_ROWS
  double ms_[kWidths] = {};
};

// What one slot's rounds have accepted lately: the lookup's drafts accepted and its rounds that stopped at a draft,
// and the draft layer's drafts accepted and offered a round, each decayed by 0.9 a round.
struct DraftOdds {
  double lookup_accepted = 9.0, lookup_failed = 1.0, layer_accepted = 2.0, layer_offered = 3.0;
  void lookup_round(uint32_t accepted, uint32_t drafts);
  void layer_round(uint32_t accepted, uint32_t drafts);
  // The draft layer's drafts accepted in a round of k: its acceptance a draft, accepted over offered, to the powers
  // 1 to k added (a draft counts only when those before it do).
  double layer_expected(uint32_t k) const;
};

// The drafts for a round of one slot: how many of the lookup's to take, 0 for the draft layer's. Each choice is worth
// its expected tokens (the row sampled last and the drafts accepted, the lookup's one after another with its
// acceptance so far, the draft layer's at its mean) over its window's time. The lookup is offered as many drafts as
// the draft layer makes and as many as a window holds.
uint32_t choose_drafts(const DraftOdds &odds, uint32_t layer_drafts, uint32_t draft_max, uint32_t lookup_drafts,
                       uint32_t max_rows, const WindowTimes &times);

// The drafts each of several writing slots makes while the window times are not known.
constexpr uint32_t kSharedDrafts = 2;

// The drafts each of the slots makes in a round of several (one count for all), 0 to the most the window's rows
// hold: the count that gives the most tokens a millisecond, the slots' expected tokens (the row sampled last and the
// draft layer's drafts accepted) over the time of their windows together. kSharedDrafts until a window's time is known.
uint32_t choose_shared_drafts(const std::vector<DraftOdds> &slots, uint32_t draft_max, uint32_t max_rows,
                              const WindowTimes &times);

}  // namespace engine
