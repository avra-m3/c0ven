// CpuExperts::compute_many keeps its one pool job (h_rows not 0) for every down type of the deepseek41 Unsloth file
// and of the Flash-Next files, and gives each token the bits compute gives it, on random blocks with an expert two
// tokens share. No card; the products need the CPU features the ggml build uses.
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "cpu/experts.h"
#include "cpu/features.h"
#include "load/gguf.h"
#include "quant.h"
#include "test_util.h"

using namespace engine;

int main() {
  // deepseek41's down types with its n_ff of 2,304 (q8_K blocks of 256), and Flash-Next's with 640 (q8_0 and q8_1
  // blocks of 32).
  const struct {
    uint32_t down_type, n_ff, rows;
  } cases[] = {{QT_Q2_K, 2304, 256}, {QT_IQ2_XS, 2304, 256}, {QT_IQ2_XXS, 2304, 256}, {QT_IQ2_S, 2304, 256},
               {QT_IQ3_XXS, 2304, 256}, {QT_IQ4_NL, 640, 32}, {QT_Q2_0, 640, 32}, {QT_Q8_0, 640, 32},
               {QT_Q5_1, 640, 32}};
  for (const auto &c : cases) {
    const uint32_t got = CpuExperts::h_rows({QT_IQ2_XXS, c.down_type, 256, c.n_ff});
    EXPECT(got == c.rows, "%s down, n_ff %u: h_rows %u, want %u", ggml_type_label(c.down_type), c.n_ff, got, c.rows);
  }
  const std::string missing = missing_cpu_features(ENGINE_CPU_FEATURES);
  if (!missing.empty()) {
    std::printf("h_rows ok; compute_many not checked: the CPU lacks %s\n", missing.c_str());
    return test::skip("the CPU lacks " + missing);
  }

  Pool pool(4);
  CpuExperts cpu(pool);
  std::mt19937 rng(11);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  const uint32_t E = 256, n_exp = 3, ntok = 2;
  for (const auto &c : cases) {
    const ExpertShape s{QT_IQ2_XXS, c.down_type, E, c.n_ff};
    const uint32_t F = c.n_ff;
    const uint64_t gblocks = (uint64_t)F * E / quant_block_values(s.gate_type);
    const uint64_t dblocks = (uint64_t)E * F / quant_block_values(s.down_type);
    std::vector<std::vector<uint8_t>> gate, up, down;
    std::vector<ExpertWeights> ew;
    for (uint32_t e = 0; e < n_exp; e++) {
      gate.push_back(test::random_blocks(s.gate_type, gblocks, rng));
      up.push_back(test::random_blocks(s.gate_type, gblocks, rng));
      down.push_back(test::random_blocks(s.down_type, dblocks, rng));
    }
    for (uint32_t e = 0; e < n_exp; e++) ew.push_back({gate[e].data(), up[e].data(), down[e].data(), 1});
    std::vector<float> x((size_t)ntok * E);
    for (auto &v : x) v = normal(rng);
    // Token 0 takes experts 0 and 1, token 1 experts 1 and 2: expert 1 is read once for both.
    const uint32_t pick[ntok][2] = {{0, 1}, {1, 2}};
    const float w[ntok][2] = {{0.75f, 0.25f}, {0.5f, 0.375f}};
    std::vector<CpuExperts::Job> jobs;
    for (uint32_t t = 0; t < ntok; t++)
      for (uint32_t k = 0; k < 2; k++) jobs.push_back({ew[pick[t][k]], t, w[t][k]});
    std::vector<float> got((size_t)ntok * E);
    cpu.compute_many(s, jobs, ntok, x.data(), E, got.data(), E);
    for (uint32_t t = 0; t < ntok; t++) {
      std::vector<float> want(E);
      cpu.compute(s, {ew[pick[t][0]], ew[pick[t][1]]}, {w[t][0], w[t][1]}, x.data() + (size_t)t * E, want.data());
      EXPECT(std::memcmp(want.data(), got.data() + (size_t)t * E, E * sizeof(float)) == 0,
             "%s down: compute_many's token %u differs from compute's", ggml_type_label(c.down_type), t);
    }
    std::printf("%-8s down, n_ff %u: %u rows of h at a time, compute_many equals compute bit for bit\n",
                ggml_type_label(c.down_type), F, CpuExperts::h_rows(s));
  }
  std::printf("PASS\n");
  return 0;
}
