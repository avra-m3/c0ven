// Level 1 (docs/TESTING.md): drafts from a slot's own tokens and the choice between them and the draft layer's, and
// the drafts of several slots (api/lookup.h), on made-up histories and times. No card.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "api/lookup.h"
#include "test_util.h"

using engine::choose_drafts;
using engine::choose_shared_drafts;
using engine::DraftOdds;
using engine::even_drafts;
using engine::Lookup;
using engine::RoundTimes;
using engine::SharedChoice;
using engine::split_drafts;
using engine::WindowTimes;

namespace {

bool never(int32_t) { return false; }
bool is_99(int32_t t) { return t == 99; }

// n distinct tokens from first.
std::vector<int32_t> run(int32_t first, uint32_t n) {
  std::vector<int32_t> v;
  for (uint32_t i = 0; i < n; i++) v.push_back(first + (int32_t)i);
  return v;
}

std::vector<int32_t> cat(std::vector<int32_t> a, const std::vector<int32_t> &b) {
  a.insert(a.end(), b.begin(), b.end());
  return a;
}

// n slots whose draft layer's drafts are accepted, accepted of offered a round.
std::vector<DraftOdds> slots(uint32_t n, uint32_t accepted, uint32_t offered) {
  DraftOdds o;
  for (int i = 0; i < 200; i++) o.layer_round(accepted, offered);
  return std::vector<DraftOdds>(n, o);
}

std::vector<uint32_t> v(std::initializer_list<uint32_t> l) { return l; }

// n slots whose draft layer's drafts are accepted at each place as p gives, and made as m gives (all where m is
// empty), the places past them as their last, measured.
std::vector<DraftOdds> placed(uint32_t n, std::initializer_list<double> p, std::initializer_list<double> m = {}) {
  DraftOdds o;
  o.layer_accepted = *p.begin(), o.layer_offered = 1, o.first_checks = engine::kFirstChecks;
  uint32_t i = 0;
  for (double x : p) o.place_accepted[i] = x, o.place_checked[i] = 1, i++;
  i = 0;
  for (double x : m) o.place_made[i] = x, o.place_tried[i] = 1, i++;
  return std::vector<DraftOdds>(n, o);
}

}  // namespace

int main() {
  // A run of 30 tokens, other tokens, then the run's first 20 again: the lookup proposes what followed them.
  {
    const std::vector<int32_t> t = cat(cat(run(100, 30), run(500, 10)), run(100, 19));
    Lookup lk;
    const std::vector<int32_t> d = lk.drafts(t, 119, 7, never);
    EXPECT(d == run(120, 7), "repeat: %zu drafts, first %d", d.size(), d.empty() ? -1 : d[0]);
    EXPECT(lk.drafts(t, 119, 3, never) == run(120, 3), "cap 3");
    EXPECT(lk.drafts(t, 7, 7, never).empty(), "no match for another next");
    // Indexed ahead a piece at a time, as a prompt is read: the same drafts.
    Lookup ahead;
    ahead.index(std::vector<int32_t>(t.begin(), t.begin() + 25));
    ahead.index(t);
    EXPECT(ahead.drafts(t, 119, 7, never) == run(120, 7), "indexed ahead");
  }

  // A match shorter than kLookupMin proposes nothing; one of kLookupMin does.
  {
    const uint32_t m = engine::kLookupMin;
    std::vector<int32_t> t = cat(cat(run(100, 30), run(500, 10)), run(100, m - 2));
    Lookup lk;
    EXPECT(lk.drafts(t, 100 + (int32_t)m - 2, 7, never).empty(), "match of %u", m - 1);
    t.push_back(100 + (int32_t)m - 2);  // the history grows by one: the index follows it
    EXPECT(lk.drafts(t, 100 + (int32_t)m - 1, 7, never) == run(100 + (int32_t)m, 7), "match of %u", m);
  }

  // The drafts end before the first token stop holds for, and at next's position.
  {
    std::vector<int32_t> t = cat(run(100, 20), {99, 1, 2});
    t = cat(cat(t, run(500, 10)), run(100, 19));
    Lookup lk;
    EXPECT(lk.drafts(t, 119, 7, is_99).empty(), "stop at once");
    EXPECT(lk.drafts(t, 119, 7, never) == std::vector<int32_t>({99, 1, 2, 500, 501, 502, 503}), "no stop");
    // A stop that holds state, as the C ABI's code passes for the model's image tokens.
    const int32_t at = 2;
    EXPECT(lk.drafts(t, 119, 7, [&at](int32_t x) { return x == at; }) == std::vector<int32_t>({99, 1}),
           "a lambda's stop");
    // The run's start right after itself: the drafts run up to next and stop there.
    const std::vector<int32_t> u = cat(run(100, 20), run(100, 19));
    Lookup lu;
    const std::vector<int32_t> du = lu.drafts(u, 119, 7, never);
    EXPECT(du == std::vector<int32_t>({100, 101, 102, 103, 104, 105, 106}), "adjacent: %zu", du.size());
  }

  // Of two matches the longer wins, and of two as long the newer.
  {
    // A: 20 tokens then 1; B: the same 20 then 2; then the 20 again. Both match 20 (capped at 64 by neither): newer.
    std::vector<int32_t> t = cat(cat(run(100, 20), {1}), run(700, 5));
    t = cat(cat(cat(t, run(100, 20)), {2}), run(800, 5));
    t = cat(t, run(100, 19));
    Lookup lk;
    EXPECT(lk.drafts(t, 119, 1, never) == std::vector<int32_t>({2}), "newer of two");
    // A longer match further back wins: the history before the last copy agrees with the first copy only.
    std::vector<int32_t> v = cat(cat(cat(run(300, 10), run(100, 20)), {1}), run(700, 5));
    v = cat(cat(cat(v, run(100, 20)), {2}), run(800, 5));
    v = cat(cat(v, run(300, 10)), run(100, 19));
    Lookup lv;
    EXPECT(lv.drafts(v, 119, 1, never) == std::vector<int32_t>({1}), "longer of two");
  }

  // A window's time: seen widths, a line between them, and past the widest the line through the two widest.
  {
    WindowTimes w;
    EXPECT(w.ms(4) == 0, "nothing seen");
    w.add(1, 16.0);
    w.add(4, 26.0);
    EXPECT(w.ms(1) == 16.0 && w.ms(4) == 26.0, "seen");
    EXPECT(w.ms(2) > 19.3 && w.ms(2) < 19.4, "between: %g", w.ms(2));
    EXPECT(w.ms(8) > 39.3 && w.ms(8) < 39.4, "past, on the line through 1 and 4: %g", w.ms(8));
    w.add(8, 64.0);
    EXPECT(w.ms(8) == 64.0 && w.ms(6) == 45.0, "after 8: %g %g", w.ms(8), w.ms(6));
    // The second time replaces the first (a list's first run), and later ones join a running mean.
    w.add(8, 48.0);
    EXPECT(w.ms(8) == 48.0, "second: %g", w.ms(8));
    w.add(8, 64.0);
    EXPECT(w.ms(8) == 49.0, "mean: %g", w.ms(8));
  }

  // A round of several slots' time: each width's time less its misses at one cost a miss.
  {
    RoundTimes r;
    EXPECT(r.net(4) == 0 && r.seen() == 0, "nothing seen");
    // A width's first round is not trusted: one round leaves it unseen, and the second replaces it.
    r.add(4, 90.0, 0);
    EXPECT(r.seen() == 0, "one round: %u", r.seen());
    r.add(4, 24.0, 0);
    EXPECT(r.seen() == 1 && std::fabs(r.net(4) - 24.0) < 1e-9, "second: %g", r.net(4));
    // Below the narrowest seen, the line through 4 and 8 rows falls under the time in proportion to the rows.
    r.add(8, 80.0, 0), r.add(8, 80.0, 0);
    EXPECT(std::fabs(r.net(3) - 18) < 1e-9 && std::fabs(r.net(2) - 12) < 1e-9, "proportion: %g %g", r.net(3),
           r.net(2));
    // Widths 4, 6 and 8 cost 24, 28 and 32 ms, each miss 0.025 ms more, with misses that vary round to round.
    RoundTimes t;
    const double base[] = {0, 0, 0, 0, 24, 0, 28, 0, 32};
    for (int i = 0; i < 200; i++)
      for (uint32_t w : {4u, 6u, 8u}) {
        const uint32_t misses = (uint32_t)(w * (10 + (i * 37 + w * 11) % 90));
        t.add(w, base[w] + 0.025 * misses, misses);
      }
    EXPECT(std::fabs(t.miss_ms() - 0.025) < 1e-6, "cost a miss: %g", t.miss_ms());
    EXPECT(std::fabs(t.net(4) - 24) < 1e-3 && std::fabs(t.net(6) - 28) < 1e-3 && std::fabs(t.net(8) - 32) < 1e-3,
           "net: %g %g %g", t.net(4), t.net(6), t.net(8));
    // Between seen widths a line; past the widest, and below the narrowest, the line through the two widest or
    // narrowest.
    EXPECT(std::fabs(t.net(5) - 26) < 1e-3 && std::fabs(t.net(12) - 40) < 1e-3 && std::fabs(t.net(2) - 20) < 1e-3,
           "unseen: %g %g %g", t.net(5), t.net(12), t.net(2));
    EXPECT(std::fabs(t.net(1) - 18) < 1e-3, "one row: %g", t.net(1));
    // A width dearer than a wider one is priced as the wider.
    for (int i = 0; i < 2; i++) t.add(5, 40.0, 0);
    EXPECT(std::fabs(t.net(5) - 28) < 1e-3, "5 above 6: %g", t.net(5));
    // A window's time: its net time and its rows at the misses a row lately, which the first round sets.
    RoundTimes u;
    u.add(4, 30.0, 400);
    EXPECT(u.misses_row() == 100 && u.ms(4) == 0, "first round: %g %g", u.misses_row(), u.ms(4));
    for (int i = 0; i < 40; i++)
      for (uint32_t w : {4u, 8u}) {
        const uint32_t misses = w * (i % 2 ? 100 : 200);
        u.add(w, base[w] + 0.025 * misses, misses);
      }
    EXPECT(u.misses_row() > 100 && u.misses_row() < 200, "misses a row: %g", u.misses_row());
    EXPECT(std::fabs(u.ms(8) - (32 + 0.025 * u.misses_row() * 8)) < 1e-3, "with misses: %g", u.ms(8));
  }

  // The choice, with times like the cards' (1 row 16 ms, 4 rows 26, 8 rows 64).
  {
    WindowTimes w;
    w.add(1, 16.0);
    w.add(2, 19.3);
    w.add(3, 22.8);
    w.add(4, 26.0);
    w.add(8, 64.0);
    DraftOdds odds;  // the lookup's acceptance 0.9 to start with, the draft layer's 2 a round
    EXPECT(choose_drafts(odds, 3, 3, 1, 8, w) == 0, "one lookup draft: the draft layer");
    // 3 lookup drafts at 0.9 make 3.44 tokens in 26 ms, against the draft layer's 3 in 26.
    EXPECT(choose_drafts(odds, 3, 3, 7, 8, w) == 3, "high acceptance: %u", choose_drafts(odds, 3, 3, 7, 8, w));
    // With every lookup draft accepted lately, 7 at once (8 tokens in 64 ms) still loses to 3 (4 in 26).
    DraftOdds sure;
    for (int i = 0; i < 50; i++) sure.lookup_round(7, 7);
    EXPECT(choose_drafts(sure, 3, 3, 7, 8, w) == 3, "sure: 3");
    // When the 8-row window is cheap, the lookup takes it.
    WindowTimes fast = w;
    for (int i = 0; i < 100; i++) fast.add(8, 30.0);
    EXPECT(choose_drafts(sure, 3, 3, 7, 8, fast) == 7, "cheap 8 rows: %u", choose_drafts(sure, 3, 3, 7, 8, fast));
    // A lookup that keeps failing gives way to the draft layer.
    DraftOdds bad;
    for (int i = 0; i < 20; i++) bad.lookup_round(0, 3);
    EXPECT(choose_drafts(bad, 3, 3, 7, 8, w) == 0, "failing lookup");
    // Without the draft layer's drafts (all below its probability) the choice is against one row alone.
    EXPECT(choose_drafts(odds, 0, 3, 7, 8, w) == 3, "no layer drafts");
    // Nothing timed yet: no lookup.
    EXPECT(choose_drafts(sure, 3, 3, 7, 8, WindowTimes{}) == 0, "no times");
  }

  // The draft layer's acceptance a draft: accepted over checked, to the powers 1 to k. A round of 4 that accepts 1
  // checked 2.
  {
    DraftOdds o;
    for (int i = 0; i < 200; i++) o.layer_round(1, 4);  // 0.5 a draft
    EXPECT(o.layer_expected(0) == 0, "k 0");
    EXPECT(std::fabs(o.layer_expected(1) - 0.5) < 1e-6 && std::fabs(o.layer_expected(3) - 0.875) < 1e-6, "k 1, 3: %g %g",
           o.layer_expected(1), o.layer_expected(3));
    DraftOdds none;
    for (int i = 0; i < 200; i++) none.layer_round(0, 3);
    EXPECT(none.layer_expected(3) < 1e-6, "never accepted: %g", none.layer_expected(3));
    // Each place's own: the first always accepted, the second never; places no round reaches start from the place
    // before.
    EXPECT(std::fabs(o.place_expected(3) - 1) < 1e-6, "places: %g", o.place_expected(3));
    DraftOdds sure;
    for (int i = 0; i < 200; i++) sure.layer_round(3, 3);
    EXPECT(std::fabs(sure.place_expected(7) - 7) < 1e-6, "past the places reached: %g", sure.place_expected(7));
    // A round asked for 3 that made 1, accepted: the second place is never made, so a round of 3 makes 1 draft and
    // accepts it; the acceptance a draft does not see the drafts not made.
    DraftOdds short_made;
    for (int i = 0; i < 200; i++) short_made.layer_round(1, 1, 3);
    EXPECT(std::fabs(short_made.place_expected(3) - 1) < 1e-6 && std::fabs(short_made.place_drafts(3) - 1) < 1e-6 &&
               std::fabs(short_made.layer_expected(3) - 3) < 1e-6,
           "asked, not made: %g %g %g", short_made.place_expected(3), short_made.place_drafts(3),
           short_made.layer_expected(3));
    // Asked for 2, every draft made and accepted in half the rounds and only the first in the others: 1.5 drafts
    // made and accepted.
    DraftOdds half_made;
    for (int i = 0; i < 200; i++) half_made.layer_round(1 + i % 2, 1 + i % 2, 2);
    EXPECT(std::fabs(half_made.place_expected(2) - 1.5) < 0.05 && std::fabs(half_made.place_drafts(2) - 1.5) < 0.05,
           "half made: %g %g", half_made.place_expected(2), half_made.place_drafts(2));
    // Asked for 2 and made none: nothing made or accepted, the acceptance a draft as it was, the first place tried.
    DraftOdds none_made;
    for (int i = 0; i < 200; i++) none_made.layer_round(0, 0, 2);
    EXPECT(none_made.place_expected(2) < 1e-6 && none_made.place_drafts(2) < 1e-6 &&
               std::fabs(none_made.layer_expected(1) - 2.0 / 3) < 1e-6 && none_made.first_checks == 200,
           "none made: %g %g %g %u", none_made.place_expected(2), none_made.place_drafts(2),
           none_made.layer_expected(1), none_made.first_checks);
    // No round says what it asked for: every draft asked for is made.
    EXPECT(std::fabs(sure.place_drafts(5) - 5) < 1e-6, "made, not asked: %g", sure.place_drafts(5));
    // A place few rounds reach keeps what they gave: the second place rejected in 50 rounds, then 200 of one draft.
    DraftOdds kept;
    for (int i = 0; i < 50; i++) kept.layer_round(1, 2);
    for (int i = 0; i < 200; i++) kept.layer_round(1, 1);
    EXPECT(kept.place_expected(2) < 1.01, "kept: %g", kept.place_expected(2));
  }

  // The drafts of several slots shared out of a step's rows: the rows past one a slot, at most k and draft_max each.
  {
    EXPECT(even_drafts(3, 7, 16, 7) == v({5, 4, 4}), "3 of 16");
    EXPECT(even_drafts(3, 2, 8, 7) == v({2, 2, 1}), "3 of 8, 2 at most");
    EXPECT(even_drafts(2, 7, 8, 3) == v({3, 3}), "draft_max 3");
    EXPECT(even_drafts(8, 2, 8, 7) == std::vector<uint32_t>(8, 0), "8 of 8");
    EXPECT(even_drafts(9, 2, 8, 7) == std::vector<uint32_t>(9, 0), "more slots than rows");
    // The one draft more from slot first on, past the last from the first.
    EXPECT(even_drafts(3, 2, 8, 7, 1) == v({1, 2, 2}), "3 of 8 from the second");
    EXPECT(even_drafts(3, 2, 8, 7, 2) == v({2, 1, 2}), "3 of 8 from the third");
    EXPECT(even_drafts(3, 7, 16, 7, 2) == v({4, 4, 5}), "3 of 16 from the third");
    EXPECT(even_drafts(3, 7, 16, 7, 4) == v({4, 5, 4}), "first past the slots");
  }

  // The one draft more by the slots' tokens: first to the slots it makes faster, of them to the fewest, close slots
  // in turns. A round here takes 30 ms and 1 a row.
  {
    auto ms = [](const std::vector<uint32_t> &d) {
      double r = (double)d.size();
      for (uint32_t x : d) r += x;
      return 30 + r;
    };
    // Slots alike are all close: the drafts go as even_drafts gives them, from first on.
    for (uint32_t f = 0; f < 4; f++) {
      uint32_t turns = 0;
      EXPECT(split_drafts(slots(3, 3, 3), 2, 8, 7, ms, f, &turns) == even_drafts(3, 2, 8, 7, f) && turns == 3,
             "alike, from %u: %u turns", f, turns);
    }
    // 1 each of 8 rows: 1.9, 1.6 and 1.3 tokens in 36 ms. One more makes each faster: the slot at 1.3 takes one; the
    // second would take the slot at 1.6 to 1.96, past the one at 1.9, so those two take it in turns.
    const DraftOdds hi = placed(1, {0.9})[0], mid = placed(1, {0.6})[0], lo = placed(1, {0.3})[0];
    uint32_t turns = 0;
    EXPECT(split_drafts({hi, mid, lo}, 2, 8, 7, ms, 0, &turns) == v({2, 1, 2}) && turns == 2, "close: %u", turns);
    EXPECT(split_drafts({hi, mid, lo}, 2, 8, 7, ms, 1) == v({1, 2, 2}), "close, the next turn");
    EXPECT(split_drafts({hi, mid, lo}, 2, 8, 7, ms, 2) == v({2, 1, 2}), "close, past the run");
    // 4 each of 16 rows: 4.10, 2.31 and 1.43 tokens in 45 ms. A fifth draft adds 0.002 to the slot at 0.3, slower
    // in the round of 46 ms, so the slot at 2.31 takes it, short of the next.
    EXPECT(split_drafts({hi, mid, lo}, 7, 16, 7, ms, 0, &turns) == v({4, 5, 4}) && turns == 1, "fewest: %u", turns);
    EXPECT(split_drafts({lo, hi, mid}, 7, 16, 7, ms, 1) == v({4, 4, 5}), "fewest, another order");
    // With no time, every slot one more adds to is made faster: the slot at 0.3 takes it.
    auto none_timed = [](const std::vector<uint32_t> &) { return 0.0; };
    EXPECT(split_drafts({hi, mid, lo}, 7, 16, 7, none_timed) == v({4, 4, 5}), "no time");
    // Slots copying text at 0.97, 0.98 and 0.99 a draft are within a draft of each other: all take turns.
    const std::vector<DraftOdds> copy = {placed(1, {0.97})[0], placed(1, {0.98})[0], placed(1, {0.99})[0]};
    EXPECT(split_drafts(copy, 7, 16, 7, ms, 0, &turns) == v({5, 4, 4}) && turns == 3, "copy: %u turns", turns);
    EXPECT(split_drafts(copy, 7, 16, 7, ms, 2) == v({4, 4, 5}), "copy, third turn");
    // A slot that accepts none is not made faster: the two that accept every draft take the drafts.
    const DraftOdds sure = slots(1, 3, 3)[0], none = slots(1, 0, 3)[0];
    for (uint32_t f = 0; f < 3; f++) {
      EXPECT(split_drafts({sure, sure, none}, 2, 8, 7, ms, f) == v({2, 2, 1}), "none last, from %u", f);
      EXPECT(split_drafts({none, sure, sure}, 2, 8, 7, ms, f) == v({1, 2, 2}), "none first, from %u", f);
    }
    // No one more: k at most spare / n, or no spare rows.
    EXPECT(split_drafts({hi, mid, lo}, 1, 8, 7, ms, 0, &turns) == v({1, 1, 1}) && turns == 1, "k 1");
    EXPECT(split_drafts({hi, mid, lo}, 7, 3, 7, ms) == v({0, 0, 0}), "no spare rows");
    EXPECT(split_drafts({}, 7, 16, 7, ms).empty(), "no slots");
  }

  // The drafts of several slots, with times like the cards' (windows of 2 rows 20 ms, 4 23.4, 6 27.3, 8 32.2, 9 40,
  // 12 44, 16 48.1; the draft layer's pass 4.5 ms up to 8 rows and 6.5 past) and no misses.
  {
    RoundTimes w;
    WindowTimes dr;
    const double t[] = {0, 0, 20, 0, 23.4, 0, 27.3, 0, 32.2, 40, 0, 0, 44, 0, 0, 0, 48.1};
    for (int i = 0; i < 4; i++)
      for (uint32_t r = 1; r <= 16; r++)
        if (t[r] > 0) w.add(r, t[r], 0), dr.add(r, r > 8 ? 6.5 : 4.5);
    auto even = [&](const std::vector<DraftOdds> &s, uint32_t dmax, uint32_t rows, const RoundTimes &win) {
      return choose_shared_drafts(s, dmax, rows, win, dr).even;
    };
    auto wide = [&](const std::vector<DraftOdds> &s, uint32_t dmax, uint32_t rows, const RoundTimes &win) {
      return choose_shared_drafts(s, dmax, rows, win, dr).wide;
    };
    // No window timed: the even split 2 each of 8 rows; until two widths are timed, the wide count 3 each of 16 rows
    // where that is more than 8.
    RoundTimes first;
    EXPECT(even(slots(2, 3, 3), 7, 16, first) == v({2, 2}), "no times");
    EXPECT(wide(slots(2, 3, 3), 7, 16, first).empty(), "no times: 3 each of 2 slots is 8 rows");
    EXPECT(even(slots(3, 3, 3), 7, 16, first) == v({2, 2, 1}), "no times, 3 slots");
    EXPECT(wide(slots(3, 3, 3), 7, 16, first) == v({3, 3, 3}), "no times, 3 slots, wide");
    // Every draft accepted: the even split is 3 each (8 tokens in 32.2 ms); 7 each is 16 in 54.6 ms with the draft
    // layer's pass, against 8 in 36.7, faster for both.
    EXPECT(even(slots(2, 3, 3), 7, 16, w) == v({3, 3}), "sure");
    EXPECT(wide(slots(2, 3, 3), 7, 16, w) == v({7, 7}), "sure, wide");
    const SharedChoice sure = choose_shared_drafts(slots(2, 3, 3), 7, 16, w, dr);
    EXPECT(std::fabs(sure.wide_tokens - 16) < 1e-6 && std::fabs(sure.wide_ms - 54.6) < 1e-6 &&
               std::fabs(sure.even_tokens - 8) < 1e-6 && std::fabs(sure.even_ms - 36.7) < 1e-6,
           "sure, estimates: %g %g %g %g", sure.wide_tokens, sure.wide_ms, sure.even_tokens, sure.even_ms);
    // Four slots: 1 each of 8 rows (8 tokens in 36.7 ms), 3 each of 16 (16 in 54.6).
    EXPECT(even(slots(4, 3, 3), 7, 16, w) == v({1, 1, 1, 1}), "sure, 4 slots");
    EXPECT(wide(slots(4, 3, 3), 7, 16, w) == v({3, 3, 3, 3}), "sure, 4 slots, wide");
    // Two thirds accepted at every place: 2 each of 8 rows (4.2 tokens in 31.8 ms) beats 1 and 3, and every count of
    // more rows.
    EXPECT(even(placed(2, {2.0 / 3}), 7, 16, w) == v({2, 2}), "two thirds");
    EXPECT(wide(placed(2, {2.0 / 3}), 7, 16, w).empty(), "two thirds, wide");
    // Four slots accepting 0.68, 0.43 and 0.44 at the first three places: 1 each (6.7 tokens in 36.7 ms) beats 3 each
    // of 16 rows (8.4 in 54.6). At 0.82 for every place, 3 each would give 12.2 and be taken. On edits (0.98, 0.81,
    // 0.87) 3 each gives 13.9 against 7.9 and is taken.
    EXPECT(even(placed(4, {0.68, 0.43, 0.44}), 7, 16, w) == v({1, 1, 1, 1}), "prose, 4 slots");
    EXPECT(wide(placed(4, {0.68, 0.43, 0.44}), 7, 16, w).empty(), "prose, 4 slots, wide");
    EXPECT(wide(placed(4, {0.82}), 7, 16, w) == v({3, 3, 3, 3}), "one acceptance for every place, wide");
    EXPECT(wide(placed(4, {0.98, 0.81, 0.87}), 7, 16, w) == v({3, 3, 3, 3}), "edits, 4 slots, wide");
    // Three slots on prose as a log of every round on the cards had three writers: drafts made accepted 0.83, 0.74
    // and 0.65 at the first three places, made 0.81, 0.73 and 0.53 of the times asked for. 2, 2 and 1 make 3.6
    // drafts (5.74 tokens in 33.4 ms) and beat 1 each (2.4 drafts, 5.02 tokens in 30.7 ms); priced at the rows asked
    // (36.7 and 31.8 ms), 1 each was ahead. 2 each of 16 rows (6.11 tokens in 36.4 ms, 9 rows in 1 round of 5) and 3
    // each (6.48 in 39.9) are not taken.
    const std::vector<DraftOdds> prose = placed(3, {0.83, 0.74, 0.65}, {0.81, 0.73, 0.53});
    EXPECT(even(prose, 7, 16, w) == v({2, 2, 1}), "prose, 3 slots, drafts not made");
    EXPECT(wide(prose, 7, 16, w).empty(), "prose, 3 slots, drafts not made, wide");
    // Rows dear with misses (150 a row at 0.025 ms): at 0.3 a draft no drafts beat 1 each, but only once each slot's
    // first draft is measured; before, 1 each.
    RoundTimes missing;
    for (int i = 0; i < 4; i++)
      for (uint32_t r : {4u, 8u}) missing.add(r, t[r] + 0.025 * 150 * r, 150 * r);
    EXPECT(even(placed(4, {0.3}), 7, 16, missing) == v({0, 0, 0, 0}), "dear rows, measured");
    std::vector<DraftOdds> fresh = placed(4, {0.3});
    for (DraftOdds &o : fresh) o.first_checks = engine::kFirstChecks - 1;
    EXPECT(even(fresh, 7, 16, missing) == v({1, 1, 1, 1}), "dear rows, not measured yet");
    // Nothing accepted: no drafts.
    EXPECT(even(slots(2, 0, 3), 7, 16, w) == v({0, 0}), "never accepted");
    EXPECT(wide(slots(2, 0, 3), 7, 16, w).empty(), "never accepted, wide");
    // One slot that accepts all and one that accepts none: more rows give more tokens in all (9 in 54.6 ms against
    // 5 in 36.7), but the slot that accepts none, the slowest, writes slower in the longer round, so there is no wide
    // count.
    std::vector<DraftOdds> mixed = {slots(1, 3, 3)[0], slots(1, 0, 3)[0]};
    EXPECT(even(mixed, 7, 16, w) == v({3, 3}), "mixed");
    EXPECT(wide(mixed, 7, 16, w).empty(), "mixed, wide");
    // Three sure slots with draft_max 3: 3 each of 16 rows (12 tokens in 50.5 ms) gives more in all than 2, 2 and 1
    // of 8 (8 in 36.7). The first two slots write 4 tokens in 50.5 ms against 3 in 36.7, but the slowest, the third,
    // 4 against 2.
    EXPECT(even(slots(3, 3, 3), 3, 16, w) == v({2, 2, 1}), "3 slots, draft_max 3");
    EXPECT(wide(slots(3, 3, 3), 3, 16, w) == v({3, 3, 3}), "3 slots, draft_max 3, wide");
    // The same choice from the third slot on: 2, 1 and 2, with the same estimates (each slot's tokens and the round's
    // time are means over the three starts).
    const SharedChoice from0 = choose_shared_drafts(slots(3, 3, 3), 3, 16, w, dr, 0);
    const SharedChoice from2 = choose_shared_drafts(slots(3, 3, 3), 3, 16, w, dr, 2);
    EXPECT(from2.even == v({2, 1, 2}) && from2.wide == v({3, 3, 3}), "3 slots from the third");
    EXPECT(std::fabs(from0.even_tokens - from2.even_tokens) < 1e-9 && std::fabs(from0.even_ms - from2.even_ms) < 1e-9,
           "3 slots from the third, estimates: %g %g", from2.even_tokens, from2.even_ms);
    // The choice does not hang on the slots' order: two slots that accept every draft and one that accepts none, last
    // or first: the two take 2 each of 8 rows and the third 1 (7 tokens in 36.7 ms), and no wide count.
    std::vector<DraftOdds> none_last = {slots(1, 3, 3)[0], slots(1, 3, 3)[0], slots(1, 0, 3)[0]};
    std::vector<DraftOdds> none_first = {slots(1, 0, 3)[0], slots(1, 3, 3)[0], slots(1, 3, 3)[0]};
    const SharedChoice nl = choose_shared_drafts(none_last, 3, 16, w, dr, 0);
    const SharedChoice nf = choose_shared_drafts(none_first, 3, 16, w, dr, 0);
    EXPECT(nl.even == v({2, 2, 1}) && nf.even == v({1, 2, 2}) && nl.wide.empty() && nf.wide.empty(),
           "order of the slots");
    // The same with the third slot accepting none: it is the slowest in both, at 1 token in 36.7 ms against 1 in
    // 50.5, so there is no wide count.
    std::vector<DraftOdds> two_sure = {slots(1, 3, 3)[0], slots(1, 3, 3)[0], slots(1, 0, 3)[0]};
    EXPECT(wide(two_sure, 3, 16, w).empty(), "3 slots, one accepting none, wide");
    // Dear wide windows: no wide count.
    RoundTimes dear = w;
    for (int i = 0; i < 4; i++) dear.add(12, 70, 0), dear.add(16, 80, 0);
    EXPECT(wide(slots(2, 3, 3), 7, 16, dear).empty(), "dear");
    // A step of 8 rows at most: the even split only.
    EXPECT(even(slots(2, 3, 3), 7, 8, w) == v({3, 3}) && wide(slots(2, 3, 3), 7, 8, w).empty(), "8 rows");
    EXPECT(even({}, 7, 16, w).empty() && wide({}, 7, 16, w).empty(), "no slots");
  }

  // The trial of the wide count against the even split: the arm whose estimate times its correction (what its rounds
  // gave over what was estimated for them, from kTrialPrior) is the higher, and every kTrialCheck-th choice the other
  // (every kTrialCheck / 2-th while it has fewer than kTrialRounds rounds).
  {
    auto choice = [](double wide_tokens, double wide_ms, double even_tokens, double even_ms) {
      SharedChoice c;
      c.even = {1, 1, 1}, c.wide = {3, 3, 3};
      c.wide_tokens = wide_tokens, c.wide_ms = wide_ms, c.even_tokens = even_tokens, c.even_ms = even_ms;
      return c;
    };
    // Rounds that give wide_rate and even_rate tokens a millisecond, against estimates of 0.2 for both.
    auto run = [&](double wide_rate, double even_rate, uint32_t picks) {
      engine::RoundTrial tr;
      const SharedChoice c = choice(10, 50, 6, 30);
      uint32_t wides = 0;
      for (uint32_t i = 0; i < picks; i++) {
        const bool w = tr.wide(c);
        wides += w;
        tr.add_round(w, (w ? wide_rate : even_rate) * 30, 30, w ? 10 : 6, w ? 50 : 30);
      }
      return wides;
    };
    // Before any round: the even split where the estimates are level, the wide count where its estimate is 25%
    // higher (0.85 of it is 0.2125 against 0.2).
    engine::RoundTrial t1, t5;
    EXPECT(!t1.wide(choice(10, 50, 6, 30)) && t5.wide(choice(12.5, 50, 6, 30)), "prior");
    // A wide count that gives 0.25 against the even split's 0.2 takes every round but every 32nd once its correction
    // has passed the even split's, after 4 of its rounds (every 16th choice); one that gives 0.2 against 0.25 runs
    // every 16th round until it has kTrialRounds, then every 32nd.
    const uint32_t picks = 2000, faster = run(0.25, 0.2, picks), slower = run(0.2, 0.25, picks);
    EXPECT(faster >= picks - picks / engine::kTrialCheck - 5 * engine::kTrialCheck / 2, "faster wide: %u", faster);
    EXPECT(slower <= engine::kTrialRounds + picks / engine::kTrialCheck + 1, "slower wide: %u", slower);
    // Corrections carry to other text: on text A the wide count gave 0.8 of its estimate (0.81 or so with the prior)
    // and the even split all of it. On text B the wide count's estimate is 0.4 tokens a ms and the even split's
    // 0.25: about 0.32 against 0.25, the wide count. On text C 0.26 against 0.25: about 0.21, the even split.
    engine::RoundTrial t3;
    for (uint32_t i = 0; i < engine::kTrialRounds; i++)
      t3.add_round(true, 8, 40, 10, 40), t3.add_round(false, 9, 50, 9, 50);
    EXPECT(t3.correction(true) > 0.8 && t3.correction(true) < 0.85 && std::fabs(t3.correction(false) - 1) < 1e-9,
           "corrections: %g %g", t3.correction(true), t3.correction(false));
    EXPECT(t3.wide(choice(20, 50, 10, 40)), "text B");
    EXPECT(!t3.wide(choice(13, 50, 10, 40)), "text C");
    // The draft layer's pass after a round counts in its arm's time: 4 prior rounds of 0.85 x 10 tokens in 50 ms
    // decayed by 15/16, then 10 tokens in 40 ms and 10 more ms, against the estimates' 47.5 tokens in 237.5 ms.
    engine::RoundTrial t2;
    t2.add_round(true, 10, 40, 10, 50), t2.add_time(true, 10);
    EXPECT(std::fabs(t2.correction(true) - 41.875 / 47.5) < 1e-9 && t2.correction(false) == 1, "draft time: %g",
           t2.correction(true));
    // Without estimates (fewer than two widths timed) a choice is not counted and takes the wide count while the
    // arms have as many rounds; a round without estimates is not counted.
    engine::RoundTrial t4;
    bool all_wide = true;
    for (int i = 0; i < 5; i++) all_wide = all_wide && t4.wide(choice(10, 0, 6, 0));
    t4.add_round(true, 10, 40, 0, 0);
    EXPECT(all_wide && t4.rounds(true) == 0, "no estimates");
  }

  std::printf("lookup_test: ok\n");
  return 0;
}
