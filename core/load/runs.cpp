#include "runs.h"

#include <cstring>
#include <stdexcept>

#include "quant.h"

namespace engine {

uint64_t runs_bytes(uint32_t type, uint64_t nblocks) { return nblocks * quant_block_bytes(type); }

void to_runs(uint32_t type, const std::vector<SliceSpec> &specs, uint8_t *dst) {
  uint32_t bb = quant_block_bytes(type);
  if (bb == 0) throw std::runtime_error("to_runs: unknown type");
  uint64_t n = 0;
  for (auto &s : specs) n += s.rows * s.nblk;
  uint32_t nf = quant_nfields(type);
  uint64_t k = 0;
  for (auto &s : specs) {
    for (uint64_t r = 0; r < s.rows; r++) {
      const uint8_t *row = s.base + (s.row0 + r) * s.row_bytes + s.blk0 * bb;
      for (uint64_t j = 0; j < s.nblk; j++, k++) {
        const uint8_t *blk = row + j * bb;
        for (uint32_t f = 0; f < nf; f++) {
          uint32_t sz = quant_field_size(type, f);
          std::memcpy(dst + quant_run_offset(type, f, n) + k * sz, blk + quant_field_offset(type, f), sz);
        }
      }
    }
  }
}

void from_runs(uint32_t type, const uint8_t *src, uint64_t n, uint8_t *dst) {
  uint32_t bb = quant_block_bytes(type);
  uint32_t nf = quant_nfields(type);
  for (uint64_t k = 0; k < n; k++)
    for (uint32_t f = 0; f < nf; f++) {
      uint32_t sz = quant_field_size(type, f);
      std::memcpy(dst + k * bb + quant_field_offset(type, f), src + quant_run_offset(type, f, n) + k * sz, sz);
    }
}

}  // namespace engine
