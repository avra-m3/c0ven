// Level 1 (docs/TESTING.md): drafts from a slot's own tokens and the choice between them and the draft layer's
// (api/lookup.h), on made-up histories and times. No card.
#include <cstdint>
#include <vector>

#include "api/lookup.h"
#include "test_util.h"

using engine::choose_drafts;
using engine::DraftOdds;
using engine::Lookup;
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

  // A window's time: seen widths, a line between them, and past the widest in proportion.
  {
    WindowTimes w;
    EXPECT(w.ms(4) == 0, "nothing seen");
    w.add(1, 16.0);
    w.add(4, 26.0);
    EXPECT(w.ms(1) == 16.0 && w.ms(4) == 26.0, "seen");
    EXPECT(w.ms(2) > 19.3 && w.ms(2) < 19.4, "between: %g", w.ms(2));
    EXPECT(w.ms(8) == 52.0, "past: %g", w.ms(8));
    w.add(8, 64.0);
    EXPECT(w.ms(8) == 64.0 && w.ms(6) == 45.0, "after 8: %g %g", w.ms(8), w.ms(6));
    w.add(8, 80.0);
    EXPECT(w.ms(8) == 65.0, "mean: %g", w.ms(8));
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

  std::printf("lookup_test: ok\n");
  return 0;
}
