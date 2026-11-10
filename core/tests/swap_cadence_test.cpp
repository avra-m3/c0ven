// Level 1 (docs/TESTING.md): the swap cadence (api/swap_cadence.h) on made-up step times, and whether a round runs
// beside a prompt. No card.
#include <cmath>
#include <cstdint>
#include <cstdio>

#include "api/swap_cadence.h"
#include "test_util.h"

using engine::count_prompt_routing;
using engine::MissCost;
using engine::swap_round_runs;
using engine::SwapCadence;

namespace {

// Steps of one row: 10 ms and 0.1 ms for each layer missed, the layer counts cycling 0 to 39.
void steps(SwapCadence &c, uint32_t n) {
  for (uint32_t i = 0; i < n; i++) c.step(1, i % 40, 10.0 + 0.1 * (i % 40));
}

uint32_t plans_until_due(SwapCadence &c, uint32_t writers, uint32_t layers_missed) {
  for (uint32_t p = 1; p <= 64; p++) {
    c.plan();
    c.step(1, layers_missed, 10.0 + 0.1 * layers_missed);
    if (c.due(writers)) return p;
  }
  return 0;
}

}  // namespace

int main() {
  // The line through the samples gives the time a missed layer adds, at the width it was fitted on and, for a width
  // with no line, at the nearest width with one.
  {
    MissCost m;
    EXPECT(m.ms(1, 20) == 0, "no estimate before samples");
    for (uint32_t i = 0; i < 200; i++) m.add(2, i % 40, 12.0 + 0.25 * (i % 40));
    EXPECT(std::abs(m.ms(2, 20) - 5.0) < 1e-6, "width 2, 20 layers: %f", m.ms(2, 20));
    EXPECT(m.ms(2, 0) == 0, "no layers missed adds nothing");
    EXPECT(std::abs(m.ms(4, 10) - 2.5) < 1e-6, "width 4 takes width 2's line: %f", m.ms(4, 10));
    EXPECT(std::abs(m.ms(1, 10) - 2.5) < 1e-6, "width 1 takes width 2's line: %f", m.ms(1, 10));
  }

  // Steps that all miss the same layers have no slope to fit: no estimate. A step that gets faster with more layers
  // missed does not make a negative cost.
  {
    MissCost m;
    for (uint32_t i = 0; i < 100; i++) m.add(1, 10, 15.0);
    EXPECT(m.ms(1, 10) == 0, "one layer count");
    MissCost n;
    for (uint32_t i = 0; i < 100; i++) n.add(1, i % 40, 20.0 - 0.1 * (i % 40));
    EXPECT(n.ms(1, 30) == 0, "falling line");
  }

  // The line follows a change in what a layer costs.
  {
    MissCost m;
    for (uint32_t i = 0; i < 300; i++) m.add(1, i % 40, 10.0 + 0.1 * (i % 40));
    for (uint32_t i = 0; i < 600; i++) m.add(1, i % 40, 10.0 + 0.3 * (i % 40));
    EXPECT(std::abs(m.ms(1, 10) - 3.0) < 0.01, "after the change: %f", m.ms(1, 10));
  }

  // Until a round is timed and a cost fitted: every 4th plan, every 16th with several writers.
  {
    SwapCadence c;
    EXPECT(plans_until_due(c, 1, 10) == 4, "default, one writer");
    SwapCadence d;
    EXPECT(plans_until_due(d, 2, 10) == 16, "default, several writers");
    SwapCadence e;
    steps(e, 100);
    EXPECT(!e.due(1), "no plans yet");
    for (int i = 0; i < 4; i++) e.plan();
    EXPECT(e.due(1), "fitted but no round timed: every 4th plan");
  }

  // Costly misses: the rule is due at the shortest gap. A round that took 29 ms against 20 layers
  // missed (2 ms) a plan comes due at plan 15.
  {
    SwapCadence c;
    steps(c, 200);
    c.round(0.5);
    const uint32_t p = plans_until_due(c, 1, 30);
    EXPECT(p == 4, "cheap round, costly misses: %u plans", p);
    c.round(0.5);
    EXPECT(plans_until_due(c, 2, 30) == 16, "several writers: every 16th plan");
    SwapCadence s;
    steps(s, 200);
    s.round(29.0);
    EXPECT(plans_until_due(s, 1, 20) == 15, "29 ms round, 2 ms a plan");
  }

  // Cheap misses wait up to the longest gap; a round resets the count.
  {
    SwapCadence c;
    steps(c, 200);
    c.round(30.0);
    EXPECT(plans_until_due(c, 1, 1) == 16, "misses of 0.1 ms a plan");
    c.round(30.0);
    for (int i = 0; i < 3; i++) c.plan();
    EXPECT(!c.due(1), "3 plans after a round");
  }

  // A lone prompt: no round while it is read, due or not; one when it ends.
  EXPECT(!swap_round_runs(false, true, true, 0), "a lone prompt read, a round due");
  EXPECT(!swap_round_runs(false, false, true, 0), "a lone prompt read");
  EXPECT(swap_round_runs(true, false, false, 1), "a lone prompt ended");
  EXPECT(swap_round_runs(true, false, false, 0), "a prompt ended, its slot done at once");
  // A prompt read beside writers: rounds on the cadence.
  EXPECT(swap_round_runs(false, true, true, 1), "a prompt beside one writer, a round due");
  EXPECT(swap_round_runs(false, true, true, 3), "a prompt beside three writers, a round due");
  EXPECT(!swap_round_runs(false, false, true, 1), "a prompt beside one writer, no round due");
  EXPECT(!swap_round_runs(false, false, true, 3), "a prompt beside three writers, no round due");
  // A prompt that ends while another is read: its slot writes, so a round runs.
  EXPECT(swap_round_runs(true, false, true, 1), "one prompt ended, another read");
  // No prompt read: the cadence, as before.
  EXPECT(swap_round_runs(false, true, false, 2), "two writers, a round due");
  EXPECT(!swap_round_runs(false, false, false, 2), "two writers, no round due");
  EXPECT(!swap_round_runs(false, false, false, 0), "nothing to do");

  // A prompt's routing joins the counts only while no slot writes.
  EXPECT(count_prompt_routing(0), "a lone prompt's routing counts");
  EXPECT(!count_prompt_routing(1), "a prompt beside one writer");
  EXPECT(!count_prompt_routing(4), "a prompt beside four writers");

  // Beside a writer, a held plan count still grows, so the round comes at the next plan once a writer starts.
  {
    SwapCadence c;
    for (int i = 0; i < 4; i++) c.plan();
    EXPECT(!swap_round_runs(false, c.due(0), true, 0), "a lone prompt, four plans");
    c.plan();
    EXPECT(swap_round_runs(false, c.due(1), true, 1), "a writer starts beside the prompt");
  }

  std::printf("swap_cadence_test: ok\n");
  return 0;
}
