// The CPU's share of the routed experts: the misses, computed whole with ggml's CPU kernels on the weights in
// the file's layout (docs/DESIGN.md, section 9, "The CPU side"; ADR-0004). As the reference does, the activation is
// first converted to each weight type's dot-product type (docs/spec/MODEL_LAYER.md, section 6).
#pragma once

#include <cstdint>
#include <vector>

#include "util/parallel.h"

namespace engine {

// One expert's three matrices in the file's layout: gate and up are rows x K, down is K_out x rows. With down_parts
// of 2, down is stored as two regions one after the other, the first half of every row's blocks and then the second
// (each card stages only its own region for a prompt); the CPU dots the halves and adds them.
struct ExpertWeights {
  const uint8_t *gate = nullptr, *up = nullptr, *down = nullptr;
  uint32_t down_parts = 1;
};

struct ExpertShape {
  uint32_t gate_type = 0, down_type = 0;
  uint32_t n_embd = 0;  // K of gate and up, rows of down
  uint32_t n_ff = 0;    // rows of gate and up, K of down
};

class CpuExperts {
 public:
  explicit CpuExperts(Pool &pool);

  // y = sum over k of w[k] * down_k(silu(gate_k x) * (up_k x)), the experts taken in order.
  void compute(const ExpertShape &shape, const std::vector<ExpertWeights> &experts, const std::vector<float> &w,
               const float *x, float *y);

  // The same for several tokens at once: token t's experts are the jobs with tok == t, summed in the order given.
  // An expert several tokens need is one job per token, but its weights are read once for all of them.
  struct Job {
    ExpertWeights expert;
    uint32_t tok = 0;
    float w = 0;
  };
  void compute_many(const ExpertShape &shape, const std::vector<Job> &jobs, uint32_t ntok, const float *x,
                    uint64_t x_stride, float *y, uint64_t y_stride);

 private:
  Pool &pool_;
  std::vector<uint8_t> xq_, hq_;
  std::vector<float> g_, u_, out_;
  std::vector<uint8_t> xqm_;
};

}  // namespace engine
