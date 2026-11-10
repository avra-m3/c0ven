// Level 1 (docs/TESTING.md): the draft gate (api/draft_gate.h) on made-up rows. No card.
#include <vector>

#include "api/draft_gate.h"
#include "test_util.h"

using engine::choose_step;
using engine::GateRow;
using engine::StepKind;

namespace {

constexpr GateRow kWriter{true, 5, true};      // a writing slot with drafts
constexpr GateRow kBare{true, 5, false};       // a writing slot without drafts
constexpr GateRow kSecond{true, 1, true};      // a writing slot about to sample its second token
constexpr GateRow kPromptRow{false, 0, false};  // a slot's prompt token

}  // namespace

int main() {
  // Several writers, none reading: one round.
  {
    auto c = choose_step(true, false, {kWriter, kWriter, kBare});
    EXPECT(c.kind == StepKind::kRounds, "writers, none reading");
    EXPECT(!c.undrafted && !c.beside_reader, "writers, none reading: counters");
  }
  // The same writers while another slot reads its prompt: still one round, counted beside a reader.
  {
    auto c = choose_step(true, true, {kWriter, kWriter, kBare});
    EXPECT(c.kind == StepKind::kRounds, "writers beside a reader");
    EXPECT(!c.undrafted && c.beside_reader, "writers beside a reader: counters");
  }
  // A plan that mixes a prompt row with writers: a plain step, not undrafted.
  {
    for (bool reading : {false, true}) {
      auto c = choose_step(true, reading, {kWriter, kPromptRow});
      EXPECT(c.kind == StepKind::kStep, "a prompt row among writers, reading %d", reading);
      EXPECT(!c.undrafted && !c.beside_reader, "a prompt row among writers: counters, reading %d", reading);
    }
  }
  // A slot about to sample its second token keeps several writers to a plain step, read or not.
  {
    auto c = choose_step(true, false, {kWriter, kSecond});
    EXPECT(c.kind == StepKind::kStep && c.undrafted && !c.beside_reader, "second token, none reading");
    c = choose_step(true, true, {kWriter, kSecond});
    EXPECT(c.kind == StepKind::kStep && c.undrafted && c.beside_reader, "second token beside a reader");
  }
  // A lone writer: a round whenever it has drafts, its second token included, read or not.
  {
    for (bool reading : {false, true}) {
      auto c = choose_step(true, reading, {kWriter});
      EXPECT(c.kind == StepKind::kRound && c.beside_reader == reading, "lone writer, reading %d", reading);
      c = choose_step(true, reading, {kSecond});
      EXPECT(c.kind == StepKind::kRound, "lone writer's second token, reading %d", reading);
      c = choose_step(true, reading, {kBare});
      EXPECT(c.kind == StepKind::kStep && c.undrafted && c.beside_reader == reading, "lone bare writer, %d", reading);
    }
    auto c = choose_step(true, true, {kPromptRow});
    EXPECT(c.kind == StepKind::kStep && !c.undrafted && !c.beside_reader, "a lone prompt row");
  }
  // Writers without drafts: a plain step, counted undrafted.
  {
    auto c = choose_step(true, true, {kBare, kBare});
    EXPECT(c.kind == StepKind::kStep && c.undrafted && c.beside_reader, "writers without drafts");
  }
  // Drafting off: always a plain step, never counted.
  {
    auto c = choose_step(false, true, {kWriter, kWriter});
    EXPECT(c.kind == StepKind::kStep && !c.undrafted && !c.beside_reader, "drafting off");
  }
  // No rows: nothing.
  {
    auto c = choose_step(true, true, {});
    EXPECT(c.kind == StepKind::kStep && !c.undrafted && !c.beside_reader, "no rows");
  }
  std::printf("draft_gate_test: ok\n");
  return 0;
}
