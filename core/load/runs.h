// Weights into the runs layout the cards read (shared/quant.h; docs/spec/QUANT_FORMATS.md, "Wide loads").
#pragma once

#include <cstdint>
#include <vector>

namespace engine {

// A slice of a 2-D (or one matrix of a 3-D) tensor: rows [row0, row0 + rows) and, within each row, blocks
// [blk0, blk0 + nblk). row_bytes is the source row's size; base points at the matrix's first row.
struct SliceSpec {
  const uint8_t *base = nullptr;
  uint64_t row_bytes = 0;
  uint64_t row0 = 0, rows = 0;
  uint64_t blk0 = 0, nblk = 0;
};

// Bytes the slice takes in the runs layout (the same as in the file).
uint64_t runs_bytes(uint32_t type, uint64_t nblocks);

// Writes the slices' blocks, in order, as one runs-layout slice of all their blocks at dst. Several specs make one
// slice of gathered rows, as for a card's share of a matrix.
void to_runs(uint32_t type, const std::vector<SliceSpec> &specs, uint8_t *dst);

// The inverse for one slice of n blocks, into the file's layout: for tests.
void from_runs(uint32_t type, const uint8_t *src, uint64_t n, uint8_t *dst);

// The packed codebooks (QGRID_* in shared/quant.h), from ggml-common.h.
const std::vector<uint8_t> &codebooks();

}  // namespace engine
