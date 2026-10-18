// The positions images give a prompt's cells (docs/spec/MODEL_VISION.md, section 4.2), as plain functions so a test
// can check them without a card.
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

namespace engine::qwen4exp {

// One image's block of placeholder tokens: it starts at cell first and holds grid_h / 2 rows of grid_w / 2 tokens.
struct ImageBlock {
  uint32_t first = 0, grid_h = 0, grid_w = 0;
  uint32_t tokens() const { return grid_h / 2 * (grid_w / 2); }
  // What the image adds to the offset between cells and positions: its tokens less max(h, w) / 2.
  uint32_t shift() const { return tokens() - std::max(grid_h, grid_w) / 2; }
};

// The offset between cell and position for a text cell, from the images that end at or before it.
inline uint32_t mrope_delta(const std::vector<ImageBlock> &images, uint32_t cell) {
  uint32_t delta = 0;
  for (const ImageBlock &b : images) {
    if (cell < b.first + b.tokens()) break;
    delta += b.shift();
  }
  return delta;
}

// A cell's (time, height, width). An image's token at row r and column c of its token grid, whose block starts when
// the running position is p, is (p, p + r, p + c); a text cell is its cell less the offset, three times.
inline std::array<uint32_t, 3> mrope_position(const std::vector<ImageBlock> &images, uint32_t cell) {
  uint32_t delta = 0;
  for (const ImageBlock &b : images) {
    if (cell < b.first) break;
    if (cell < b.first + b.tokens()) {
      const uint32_t k = cell - b.first, gw = b.grid_w / 2, p = b.first - delta;
      return {p, p + k / gw, p + k % gw};
    }
    delta += b.shift();
  }
  return {cell - delta, cell - delta, cell - delta};
}

// The image whose block holds cell, or null.
inline const ImageBlock *mrope_image_at(const std::vector<ImageBlock> &images, uint32_t cell) {
  for (const ImageBlock &b : images)
    if (cell >= b.first && cell < b.first + b.tokens()) return &b;
  return nullptr;
}

}  // namespace engine::qwen4exp
