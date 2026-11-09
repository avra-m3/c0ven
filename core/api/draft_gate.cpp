#include "api/draft_gate.h"

namespace engine {

StepChoice choose_step(bool drafting, bool reading, const std::vector<GateRow> &rows) {
  StepChoice c;
  if (rows.empty()) return c;
  bool all_writing = true, past_second = true, any = false;
  for (const GateRow &r : rows) {
    all_writing = all_writing && r.writing;
    past_second = past_second && r.generated > 1;
    any = any || r.drafts;
  }
  if (drafting && all_writing && any) {
    // A slot's second token already waits for the prompt steps after its first; a round would lengthen the step that
    // ends that wait.
    if (rows.size() == 1)
      c.kind = StepKind::kRound;
    else if (past_second)
      c.kind = StepKind::kRounds;
  }
  c.undrafted = c.kind == StepKind::kStep && drafting && all_writing;
  c.beside_reader = reading && (c.kind != StepKind::kStep || c.undrafted);
  return c;
}

}  // namespace engine
