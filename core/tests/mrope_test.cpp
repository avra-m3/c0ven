// Level 1 (docs/TESTING.md): the positions images give a prompt's cells (mrope.h) against the rule of transformers'
// get_rope_index, written out here group by group (docs/spec/MODEL_VISION.md, section 4.2): a run of text counts on
// from the running position; an image of h/2 x w/2 tokens gives (p, p + r, p + c) row-major, then the running
// position moves on by max(h, w) / 2. No card.
#include <array>
#include <cstdint>
#include <vector>

#include "mrope.h"
#include "test_util.h"

using engine::qwen4exp::ImageBlock;

namespace {

struct Group {
  bool image;
  uint32_t n, h, w;  // text: n tokens; image: h x w patches
};

// The reference: every cell's (t, h, w), and the images' blocks as mrope.h takes them.
void reference(const std::vector<Group> &groups, std::vector<std::array<uint32_t, 3>> &pos,
               std::vector<ImageBlock> &blocks) {
  uint32_t cur = 0;
  for (const Group &g : groups) {
    if (!g.image) {
      for (uint32_t i = 0; i < g.n; i++) pos.push_back({cur + i, cur + i, cur + i});
      cur += g.n;
      continue;
    }
    blocks.push_back({(uint32_t)pos.size(), g.h, g.w});
    for (uint32_t r = 0; r < g.h / 2; r++)
      for (uint32_t c = 0; c < g.w / 2; c++) pos.push_back({cur, cur + r, cur + c});
    cur += (g.h > g.w ? g.h : g.w) / 2;
  }
}

void check(const std::vector<Group> &groups) {
  std::vector<std::array<uint32_t, 3>> pos;
  std::vector<ImageBlock> blocks;
  reference(groups, pos, blocks);
  for (uint32_t cell = 0; cell < pos.size(); cell++) {
    auto p = engine::qwen4exp::mrope_position(blocks, cell);
    EXPECT(p == pos[cell], "cell %u: (%u, %u, %u), expected (%u, %u, %u)", cell, p[0], p[1], p[2], pos[cell][0],
           pos[cell][1], pos[cell][2]);
    const bool in_image = engine::qwen4exp::mrope_image_at(blocks, cell) != nullptr;
    if (!in_image) {
      const uint32_t d = engine::qwen4exp::mrope_delta(blocks, cell);
      EXPECT(cell - d == pos[cell][0], "cell %u: offset %u", cell, d);
    }
  }
  // Cells past the prompt continue as text.
  const uint32_t n = (uint32_t)pos.size();
  const uint32_t next = n ? pos[n - 1][0] + 1 : 0;
  auto p = engine::qwen4exp::mrope_position(blocks, n);
  if (blocks.empty() || blocks.back().first + blocks.back().tokens() < n)
    EXPECT(p[0] == next && p[1] == next && p[2] == next, "the cell after the prompt: %u, expected %u", p[0], next);
}

}  // namespace

int main() {
  check({{false, 40, 0, 0}});                                  // text only: (c, c, c)
  check({{false, 5, 0, 0}, {true, 0, 4, 6}, {false, 3, 0, 0}});  // wider than tall
  check({{false, 2, 0, 0}, {true, 0, 8, 2}, {false, 7, 0, 0}});  // taller than wide
  check({{false, 3, 0, 0}, {true, 0, 16, 16}, {false, 1, 0, 0}, {true, 0, 2, 10}, {false, 9, 0, 0}});
  check({{true, 0, 6, 4}, {false, 4, 0, 0}});                   // an image first
  check({{false, 6, 0, 0}, {true, 0, 2, 2}, {false, 1, 0, 0}});  // one token: no shift
  std::printf("mrope_test: ok\n");
  return 0;
}
