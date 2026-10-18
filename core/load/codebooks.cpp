// The i-quant codebooks are data in ggml-common.h at the pinned commit (QUANT_FORMATS.md, "The codebooks"). They
// are read from there, not copied into this repository, and packed in the order shared/quant.h gives.
#include <cstring>

#include "quant.h"
#include "runs.h"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

namespace engine {

const std::vector<uint8_t> &codebooks() {
  static const std::vector<uint8_t> packed = [] {
    std::vector<uint8_t> b(QGRID_BYTES);
    static_assert(sizeof(iq2s_grid) == QGRID_IQ3XXS - QGRID_IQ2S);
    static_assert(sizeof(iq3xxs_grid) == QGRID_IQ3S - QGRID_IQ3XXS);
    static_assert(sizeof(iq3s_grid) == QGRID_KVALUES - QGRID_IQ3S);
    static_assert(sizeof(kvalues_iq4nl) == QGRID_BYTES - QGRID_KVALUES);
    std::memcpy(b.data() + QGRID_IQ2S, iq2s_grid, sizeof iq2s_grid);
    std::memcpy(b.data() + QGRID_IQ3XXS, iq3xxs_grid, sizeof iq3xxs_grid);
    std::memcpy(b.data() + QGRID_IQ3S, iq3s_grid, sizeof iq3s_grid);
    std::memcpy(b.data() + QGRID_KVALUES, kvalues_iq4nl, sizeof kvalues_iq4nl);
    return b;
  }();
  return packed;
}

}  // namespace engine
