// Level 1 (docs/TESTING.md): the tokens before a step's row (api/model.h, StepIn, row_at, row_after), as the C ABI's
// code fills them for decode rows, draft rows and prompt windows. No card.
#include <cstdint>
#include <cstdio>
#include <vector>

#include "api/model.h"
#include "test_util.h"

using engine::kNoToken;
using engine::row_after;
using engine::row_at;
using engine::StepIn;

namespace {

bool same(const StepIn &a, const StepIn &b) {
  return a.slot == b.slot && a.pos == b.pos && a.token == b.token && a.p1 == b.p1 && a.p2 == b.p2 && a.p3 == b.p3;
}

}  // namespace

int main() {
  // A row given five fields, as qwen4exp's code gives them, has no third token.
  const StepIn five{1, 7, 42, 41, 40};
  EXPECT(five.p3 == kNoToken, "p3 defaults to kNoToken");
  const StepIn none{};
  EXPECT(none.p1 == kNoToken && none.p2 == kNoToken && none.p3 == kNoToken, "an empty row has no tokens before it");
  const engine::Model::PromptRun run{0, 4, nullptr, 0, 3, 2};
  EXPECT(run.p3 == kNoToken, "a prompt run's p3 defaults to kNoToken");

  std::vector<int32_t> tokens;
  for (int32_t t = 100; t < 100 + 2 * (int32_t)STEP_MAX_ROWS + 8; t++) tokens.push_back(t);

  // At the start of a sequence the missing tokens are kNoToken.
  EXPECT(same(row_at(2, 0, 100, tokens), StepIn{2, 0, 100, kNoToken, kNoToken, kNoToken}), "row at position 0");
  EXPECT(same(row_at(2, 1, 101, tokens), StepIn{2, 1, 101, 100, kNoToken, kNoToken}), "row at position 1");
  EXPECT(same(row_at(2, 2, 102, tokens), StepIn{2, 2, 102, 101, 100, kNoToken}), "row at position 2");
  EXPECT(same(row_at(2, 3, 103, tokens), StepIn{2, 3, 103, 102, 101, 100}), "row at position 3");
  EXPECT(same(row_at(2, 9, 109, tokens), StepIn{2, 9, 109, 108, 107, 106}), "row at position 9");

  // Only tokens[0 .. pos) are read: a decode row's own token is not yet in the slot's tokens.
  const std::vector<int32_t> short_tokens(tokens.begin(), tokens.begin() + 3);
  EXPECT(same(row_at(0, 3, 7, short_tokens), StepIn{0, 3, 7, 102, 101, 100}), "row past the slot's tokens");

  // Draft rows and prompt windows chain from a first row; each row matches the one read from the whole sequence,
  // from any start and across windows of STEP_MAX_ROWS rows.
  for (uint32_t pos0 : {0u, 1u, 2u, 3u, 5u}) {
    StepIn row = row_at(1, pos0, tokens[pos0], tokens);
    for (uint32_t pos = pos0 + 1; pos < tokens.size(); pos++) {
      row = row_after(row, tokens[pos]);
      EXPECT(same(row, row_at(1, pos, tokens[pos], tokens)), "chain from %u, position %u", pos0, pos);
    }
  }

  std::printf("step_in_test: ok\n");
  return 0;
}
