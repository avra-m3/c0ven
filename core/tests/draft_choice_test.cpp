// Level 1 (docs/TESTING.md): the draft estimate, the drafts shared among several slots and the drafts a round verifies
// (api/draft_choice.h), on made-up acceptances and times. No card.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "api/draft_choice.h"
#include "test_util.h"

using engine::choose_verified;
using engine::DraftEstimate;
using engine::DraftOdds;
using engine::share_drafts;
using engine::share_rows;
using engine::WindowTimes;

namespace {

// A slot whose rounds each offer offered drafts and accept the first accepted: every place before accepted always
// accepted, the place at it never.
DraftOdds odds(uint32_t accepted, uint32_t offered) {
  DraftOdds o;
  for (int i = 0; i < 200; i++) o.layer_round(accepted, offered);
  return o;
}

bool same(const std::vector<uint32_t> &a, const std::vector<uint32_t> &b) { return a == b; }

}  // namespace

int main() {
  // Before any draft is verified the estimate is the sampler's probability.
  {
    DraftEstimate e;
    EXPECT(std::fabs(e.accept(0, 0.8) - 0.8) < 1e-9 && std::fabs(e.accept(6, 0.3) - 0.3) < 1e-9, "unfitted: %g %g",
           e.accept(0, 0.8), e.accept(6, 0.3));
    EXPECT(e.accept(0, 1.0) < 1.0 && e.accept(0, 0.0) > 0.0, "the ends stay inside (0, 1)");
  }

  // Drafts accepted less often than the sampler's probability says, later places less still: the estimate follows.
  {
    DraftEstimate e;
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    auto truth = [](uint32_t place, double p) {
      return 1.0 / (1.0 + std::exp(-((place < 3 ? -0.5 : -1.5) + 0.5 * std::log(p / (1.0 - p)))));
    };
    for (int i = 0; i < 200000; i++) {
      const uint32_t place = (uint32_t)(i % 6);
      const double p = 0.5 + 0.49 * u(rng);
      e.observe(place, p, u(rng) < truth(place, p));
    }
    for (uint32_t place : {0u, 4u})
      for (double p : {0.6, 0.9}) {
        const double got = e.accept(place, p), want = truth(place, p);
        EXPECT(std::fabs(got - want) < 0.05, "place %u, p %g: %g, want %g (slope %g)", place, p, got, want, e.slope());
      }
    EXPECT(e.accept(0, 0.9) > e.accept(0, 0.6), "a likelier draft is accepted more often");
    // A place never verified keeps its start.
    EXPECT(e.offset(6) == 0.0, "place 6 never seen: %g", e.offset(6));
  }

  // The drafts each slot may make: one each first, then to the slot whose next draft is the likeliest to be accepted
  // with those before it.
  {
    const DraftOdds sure = odds(3, 3), once = odds(1, 2), never = odds(0, 3);
    // Three slots in a 16-row step: 13 spare rows. The sure slot takes its cap of 7; the other two share the rest.
    EXPECT(same(share_rows({sure, once, once}, {7, 7, 7}, 3, 16), {7, 3, 3}), "sure and two of one draft");
    EXPECT(same(share_rows({once, sure, never}, {7, 7, 7}, 3, 16), {4, 7, 2}), "a slot that never accepts");
    // Equal slots share alike, the lower slot first.
    EXPECT(same(share_rows({once, once, once}, {7, 7, 7}, 3, 16), {5, 4, 4}), "equal slots");
    EXPECT(same(share_rows({sure, sure}, {7, 7}, 2, 16), {7, 7}), "two slots, their caps");
    // A cap holds a slot short: 1 token left to write, or none.
    EXPECT(same(share_rows({sure, sure}, {1, 0}, 2, 16), {1, 0}), "caps");
    // No row to spare, and fewer spare rows than slots.
    EXPECT(same(share_rows({sure, sure}, {7, 7}, 16, 16), {0, 0}), "a full step");
    EXPECT(same(share_rows({once, sure, sure}, {7, 7, 7}, 14, 16), {1, 1, 0}), "two spare rows");
    // Rows other slots take in the step leave fewer to share.
    EXPECT(same(share_rows({sure}, {7}, 12, 16), {4}), "one writer beside 11 other rows");
  }

  // Each place's acceptance, and the shares by it.
  {
    DraftOdds fading;  // the first draft always accepted, the second half the time, the third never
    for (int i = 0; i < 100; i++) fading.layer_round(i % 2 ? 2 : 1, 3);
    EXPECT(fading.layer_place(0) > 0.95 && std::fabs(fading.layer_place(1) - 0.5) < 0.1 && fading.layer_place(2) < 0.1,
           "places: %g %g %g", fading.layer_place(0), fading.layer_place(1), fading.layer_place(2));
    // A place never checked has the rate of the one before it.
    EXPECT(std::fabs(fading.layer_place(3) - fading.layer_place(2)) < 1e-9, "place 3: %g", fading.layer_place(3));
    // Against a slot that accepts 3 drafts in 4 at every place, fading's second (0.48 with its first) goes before
    // the other's third (0.42), though fading accepts fewer of all its drafts (0.6).
    DraftOdds flat;
    for (uint32_t j = 0; j < DraftOdds::kPlaces; j++) flat.place_accepted[j] = 7.5, flat.place_checked[j] = 10;
    EXPECT(same(share_rows({fading, flat}, {7, 7}, 2, 6), {2, 2}), "fading and flat");
  }

  // The drafts each slot makes, by window times like the cards' (16.7 ms and 2.1 a row) and the draft layer's pass
  // after it (6 ms to 8 rows, 9 ms past them).
  {
    const DraftOdds sure = odds(3, 3), once = odds(1, 2), never = odds(0, 3);
    WindowTimes window, drafter;
    for (uint32_t r = 1; r <= 16; r++) window.add(r, 16.7 + 2.1 * r), drafter.add(r, r > 8 ? 9.0 : 6.0);
    // Fewer than two widths timed: rounds of kStartRows rows.
    EXPECT(same(share_drafts({once, once}, {7, 7}, 2, 16, WindowTimes{}, WindowTimes{}), {3, 3}), "no times");
    // Sure drafts all pay: each row adds a token for 2.1 ms, where a round gives one for 13 ms or more.
    EXPECT(same(share_drafts({sure, sure}, {7, 7}, 2, 16, window, drafter), {7, 7}), "two sure slots");
    // Slots whose second draft is never accepted make one each.
    EXPECT(same(share_drafts({once, once}, {7, 7}, 2, 16, window, drafter), {1, 1}), "two of one draft");
    // A slot that never accepts still drafts one, so that its acceptance stays measured. Beside a sure slot's 7 the
    // block pass of 10 rows runs at 16, and its drafts fill the rest.
    EXPECT(same(share_drafts({sure, never}, {7, 7}, 2, 16, window, drafter), {7, 7}), "one never accepts");
    EXPECT(same(share_drafts({never, never}, {7, 7}, 2, 16, window, drafter), {1, 1}), "none accepts");
    // Four writers whose first 4 drafts are accepted: 3 each, the step's 12 spare rows.
    const DraftOdds four = odds(4, 5);
    EXPECT(same(share_drafts({four, four, four, four}, {7, 7, 7, 7}, 4, 16, window, drafter), {3, 3, 3, 3}),
           "four writers");
    EXPECT(same(share_drafts({sure, sure}, {1, 0}, 2, 16, window, drafter), {1, 0}), "caps");
    // A dear pass past 8 rows keeps the block at 8: two sure slots make 3 each.
    WindowTimes dear;
    for (uint32_t r = 1; r <= 16; r++) dear.add(r, r > 8 ? 60.0 : 6.0);
    EXPECT(same(share_drafts({sure, sure}, {7, 7}, 2, 16, window, dear), {3, 3}), "dear wide pass");
  }

  // The drafts a round verifies, with times like the cards' (1 row 15.5 ms, 2 19.25, 3 21.9, 4 25.6, 6 34.5, 8 42.3).
  {
    WindowTimes w;
    const double t[] = {0, 15.5, 19.25, 21.9, 25.6, 0, 34.5, 0, 42.3};
    for (uint32_t r = 1; r <= 8; r++)
      if (t[r] > 0) w.add(r, t[r]);
    EXPECT(w.seen() == 6, "6 widths timed: %u", w.seen());
    // Fewer than two widths timed: every draft that fits.
    WindowTimes one;
    one.add(1, 15.5);
    EXPECT(same(choose_verified({{0.9, 0.1}, {0.2}}, 2, 16, WindowTimes{}, WindowTimes{}, 0), {2, 1}), "no times");
    EXPECT(same(choose_verified({{0.9, 0.1}, {0.2}}, 2, 16, one, WindowTimes{}, 0), {2, 1}), "one width timed");
    EXPECT(same(choose_verified({{0.9, 0.8, 0.7}, {0.9, 0.8, 0.7}}, 2, 4, one, WindowTimes{}, 0), {1, 1}),
           "no times, 2 spare rows");
    // Two slots, 19.25 ms without drafts (0.104 tokens a ms). A sure slot's drafts pay; an unsure slot's do not.
    // 2 + 0.95 + 0.9 + 0.85 in about 30 ms (5 rows) is 0.156; the unsure slot's first would make it 4.9 in 34.5.
    EXPECT(same(choose_verified({{0.95, 0.9, 0.85}, {0.2, 0.05}}, 2, 16, w, WindowTimes{}, 0), {3, 0}),
           "one sure slot");
    // Drafts are ranked across slots: slot 1's first goes before slot 0's second. Ranked 0.9, 0.8, 0.6, 0.3, 0.1:
    // 2 drafts give 3.7 tokens in 25.6 ms (0.145), 3 give 4.3 in about 30 (0.143).
    EXPECT(same(choose_verified({{0.9, 0.6, 0.3}, {0.8, 0.1}}, 2, 16, w, WindowTimes{}, 0), {1, 1}),
           "ranked across slots");
    // Nothing likely: no draft.
    EXPECT(same(choose_verified({{0.1}, {0.1}}, 2, 16, w, WindowTimes{}, 0), {0, 0}), "unlikely drafts");
    // A slot with no drafts, and no slots.
    EXPECT(same(choose_verified({{}, {0.95, 0.9}}, 2, 16, w, WindowTimes{}, 0), {0, 2}), "a slot without drafts");
    EXPECT(choose_verified({}, 0, 16, w, WindowTimes{}, 0).empty(), "no slots");
    // The step's rows bound the drafts: 3 slots and 4 rows leave one.
    EXPECT(same(choose_verified({{0.99}, {0.98}, {0.97}}, 3, 4, w, WindowTimes{}, 0), {1, 0, 0}), "one spare row");
    // Dear wide windows: fewer drafts.
    WindowTimes dear = w;
    for (int i = 0; i < 200; i++) dear.add(3, 60.0), dear.add(4, 90.0), dear.add(6, 150.0), dear.add(8, 200.0);
    EXPECT(same(choose_verified({{0.95, 0.9, 0.85}, {0.95, 0.9}}, 2, 16, dear, WindowTimes{}, 0), {0, 0}),
           "dear windows");
    // Past 8 rows the times are on the line through the two widest seen (3.9 ms a row): sure drafts fill the step.
    const std::vector<double> sure7(7, 0.99);
    EXPECT(same(choose_verified({sure7, sure7, sure7}, 3, 16, w, WindowTimes{}, 0), {5, 4, 4}),
           "sure drafts fill 16 rows");
    // The draft layer's pass after the round, 6 ms to 8 rows and 30 past them: the round stays at 8 rows, unless
    // the block that made the drafts ran past 8 already.
    WindowTimes pass;
    for (uint32_t r = 1; r <= 16; r++) pass.add(r, r > 8 ? 30.0 : 6.0);
    EXPECT(same(choose_verified({sure7, sure7, sure7}, 3, 16, w, pass, 8), {2, 2, 1}), "a dear pass past 8 rows");
    EXPECT(same(choose_verified({sure7, sure7, sure7}, 3, 16, w, pass, 16), {5, 4, 4}), "a block of 16 rows");
  }

  std::printf("draft_choice_test: ok\n");
  return 0;
}
