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
using engine::WindowTimes;

namespace {

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

  // The drafts each slot may make: one each first, then to the slot that accepts the most.
  {
    const DraftOdds sure = odds(3, 3), half = odds(1, 2), never = odds(0, 3);
    // Three slots in a 16-row step: 13 spare rows. The sure slot takes its cap of 7; the other two share the rest by
    // their acceptance.
    EXPECT(same(share_drafts({sure, half, half}, {7, 7, 7}, 3, 16), {7, 3, 3}), "sure and two halves");
    EXPECT(same(share_drafts({half, sure, never}, {7, 7, 7}, 3, 16), {5, 7, 1}), "a slot that never accepts keeps one");
    // Equal slots share alike, the lower slot first.
    EXPECT(same(share_drafts({half, half, half}, {7, 7, 7}, 3, 16), {5, 4, 4}), "equal slots");
    EXPECT(same(share_drafts({sure, sure}, {7, 7}, 2, 16), {7, 7}), "two slots, their caps");
    // A cap holds a slot short: 1 token left to write, or none.
    EXPECT(same(share_drafts({sure, sure}, {1, 0}, 2, 16), {1, 0}), "caps");
    // No row to spare, and fewer spare rows than slots.
    EXPECT(same(share_drafts({sure, sure}, {7, 7}, 16, 16), {0, 0}), "a full step");
    EXPECT(same(share_drafts({half, sure, sure}, {7, 7, 7}, 14, 16), {1, 1, 0}), "two spare rows");
    // Rows other slots take in the step leave fewer to share.
    EXPECT(same(share_drafts({sure}, {7}, 12, 16), {4}), "one writer beside 11 other rows");
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
    EXPECT(same(choose_verified({{0.9, 0.1}, {0.2}}, 2, 16, WindowTimes{}), {2, 1}), "no times");
    EXPECT(same(choose_verified({{0.9, 0.1}, {0.2}}, 2, 16, one), {2, 1}), "one width timed");
    EXPECT(same(choose_verified({{0.9, 0.8, 0.7}, {0.9, 0.8, 0.7}}, 2, 4, one), {1, 1}), "no times, 2 spare rows");
    // Two slots, 19.25 ms without drafts (0.104 tokens a ms). A sure slot's drafts pay; an unsure slot's do not.
    // 2 + 0.95 + 0.9 + 0.85 in about 30 ms (5 rows) is 0.156; the unsure slot's first would make it 4.9 in 34.5.
    EXPECT(same(choose_verified({{0.95, 0.9, 0.85}, {0.2, 0.05}}, 2, 16, w), {3, 0}), "one sure slot");
    // Drafts are ranked across slots: slot 1's first goes before slot 0's second. Ranked 0.9, 0.8, 0.6, 0.3, 0.1:
    // 2 drafts give 3.7 tokens in 25.6 ms (0.145), 3 give 4.3 in about 30 (0.143).
    EXPECT(same(choose_verified({{0.9, 0.6, 0.3}, {0.8, 0.1}}, 2, 16, w), {1, 1}), "ranked across slots");
    // Nothing likely: no draft.
    EXPECT(same(choose_verified({{0.1}, {0.1}}, 2, 16, w), {0, 0}), "unlikely drafts");
    // A slot with no drafts, and no slots.
    EXPECT(same(choose_verified({{}, {0.95, 0.9}}, 2, 16, w), {0, 2}), "a slot without drafts");
    EXPECT(choose_verified({}, 0, 16, w).empty(), "no slots");
    // The step's rows bound the drafts: 3 slots and 4 rows leave one.
    EXPECT(same(choose_verified({{0.99}, {0.98}, {0.97}}, 3, 4, w), {1, 0, 0}), "one spare row");
    // Dear wide windows: fewer drafts.
    WindowTimes dear = w;
    for (int i = 0; i < 200; i++) dear.add(3, 60.0), dear.add(4, 90.0), dear.add(6, 150.0), dear.add(8, 200.0);
    EXPECT(same(choose_verified({{0.95, 0.9, 0.85}, {0.95, 0.9}}, 2, 16, dear), {0, 0}), "dear windows");
    // Past 8 rows the times are on the line through the two widest seen (3.9 ms a row): sure drafts fill the step.
    const std::vector<double> sure7(7, 0.99);
    EXPECT(same(choose_verified({sure7, sure7, sure7}, 3, 16, w), {5, 4, 4}), "sure drafts fill 16 rows");
  }

  std::printf("draft_choice_test: ok\n");
  return 0;
}
