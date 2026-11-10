// Whether a decode step's rows run as a plain step or as a draft round, and how it is counted
// (docs/adr/0071-writers-draft-while-a-slot-reads-its-prompt.md). Apart from the C ABI's code so that it is tested
// without a card.
#pragma once

#include <cstdint>
#include <vector>

namespace engine {

// A row of a decode step, as the gate sees its slot before the step.
struct GateRow {
  bool writing = false;    // the row is one of the slot's own tokens: the slot has no prompt left
  uint32_t generated = 0;  // tokens the slot has sampled
  bool drafts = false;     // the slot holds drafts
};

enum class StepKind : uint8_t {
  kStep,    // a plain step: each row gives one token
  kRound,   // a lone slot's draft round
  kRounds,  // several slots' windows in one draft round
};

struct StepChoice {
  StepKind kind = StepKind::kStep;
  bool undrafted = false;      // a plain step, drafting on, every row writing: engine_stats.undrafted_steps
  bool beside_reader = false;  // a round, or an undrafted step, while another slot reads its prompt
};

// How a decode step of rows runs. A lone writing slot with drafts runs a round. Several run one round when every row
// is a writing slot's, none of them about to sample its second token, and some slot has drafts. A slot reading its
// prompt elsewhere does not change the choice; it only marks the step for the counters.
StepChoice choose_step(bool drafting, bool reading, const std::vector<GateRow> &rows);

}  // namespace engine
