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
  }

  // The drafts of several slots shared out of a step's rows: the rows past one a slot, at most k and draft_max each.
  {
    EXPECT(even_drafts(3, 7, 16, 7) == v({5, 4, 4}), "3 of 16");
    EXPECT(even_drafts(3, 2, 8, 7) == v({2, 2, 1}), "3 of 8, 2 at most");
    EXPECT(even_drafts(2, 7, 8, 3) == v({3, 3}), "draft_max 3");
    EXPECT(even_drafts(8, 2, 8, 7) == std::vector<uint32_t>(8, 0), "8 of 8");
    EXPECT(even_drafts(9, 2, 8, 7) == std::vector<uint32_t>(9, 0), "more slots than rows");
  }

  // The drafts of several slots, with times like the cards' (windows of 2 rows 20 ms, 4 23.4, 6 27.3, 8 32.2, 12 44,
  // 16 48.1; the draft layer's pass 4.5 ms up to 8 rows and 6.5 past) and no misses.
  {
    RoundTimes w;
    WindowTimes dr;
    const double t[] = {0, 0, 20, 0, 23.4, 0, 27.3, 0, 32.2, 0, 0, 0, 44, 0, 0, 0, 48.1};
    for (int i = 0; i < 4; i++)
      for (uint32_t r = 1; r <= 16; r++)
        if (t[r] > 0) w.add(r, t[r], 0), dr.add(r, r > 8 ? 6.5 : 4.5);
    // Until two widths are timed: 2 each of 8 rows, then once that width is timed 3 each of 16 rows.
    RoundTimes first;
    EXPECT(choose_shared_drafts(slots(2, 3, 3), 7, 16, first, dr) == v({2, 2}), "no times");
    EXPECT(choose_shared_drafts(slots(3, 3, 3), 7, 16, first, dr) == v({2, 2, 1}), "no times, 3 slots");
    first.add(6, 27.3, 0), first.add(6, 27.3, 0);
    EXPECT(choose_shared_drafts(slots(2, 3, 3), 7, 16, first, dr) == v({3, 3}), "6 rows timed");
    // Every draft accepted: the even split is 3 each (8 tokens in 36.7 ms, 4 a slot); 7 each is 16 in 54.6 ms, 8 a
    // slot, faster for both.
    EXPECT(choose_shared_drafts(slots(2, 3, 3), 7, 16, w, dr) == v({7, 7}), "sure");
    // Four slots: 1 each of 8 rows (8 tokens in 36.7 ms), 3 each of 16 (16 in 54.6).
    EXPECT(choose_shared_drafts(slots(4, 3, 3), 7, 16, w, dr) == v({3, 3, 3, 3}), "sure, 4 slots");
    // Half accepted: 2 each of 8 rows (3.5 tokens in 31.8 ms) beats every count of more rows.
    EXPECT(choose_shared_drafts(slots(2, 1, 2), 7, 16, w, dr) == v({2, 2}), "half");
    // Nothing accepted: no drafts.
    EXPECT(choose_shared_drafts(slots(2, 0, 3), 7, 16, w, dr) == v({0, 0}), "never accepted");
    // One slot that accepts all and one that accepts none: more rows give more tokens in all (9 in 54.6 ms against
    // 5 in 36.7), but the slot that accepts none writes slower in the longer round, so the even split stays.
    std::vector<DraftOdds> mixed = {slots(1, 3, 3)[0], slots(1, 0, 3)[0]};
    EXPECT(choose_shared_drafts(mixed, 7, 16, w, dr) == v({3, 3}), "mixed");
    // Three sure slots with draft_max 3: 3 each of 16 rows (12 tokens in 50.5 ms) gives more in all than 2, 2 and 1
    // of 8 (8 in 36.7), but the first two slots would write 4 tokens in 50.5 ms against 3 in 36.7.
    EXPECT(choose_shared_drafts(slots(3, 3, 3), 3, 16, w, dr) == v({2, 2, 1}), "3 slots, draft_max 3");
    // Dear wide windows keep the even split.
    RoundTimes dear = w;
    for (int i = 0; i < 4; i++) dear.add(12, 70, 0), dear.add(16, 80, 0);
    EXPECT(choose_shared_drafts(slots(2, 3, 3), 7, 16, dear, dr) == v({3, 3}), "dear");
    // A step of 8 rows at most: the even split.
    EXPECT(choose_shared_drafts(slots(2, 3, 3), 7, 8, w, dr) == v({3, 3}), "8 rows");
    EXPECT(choose_shared_drafts({}, 7, 16, w, dr).empty(), "no slots");
  }

  std::printf("lookup_test: ok\n");
  return 0;
}
