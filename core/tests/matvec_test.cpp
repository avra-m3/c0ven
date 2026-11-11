// Level 3 (docs/TESTING.md): the matrix-vector kernels of every weight type the engine reads on random blocks, and
// of every type the file holds on real rows of the model, in the runs layout, against ggml's CPU dequantisation of the
// same rows multiplied in double precision by the same a8 activations. Tolerance 1e-5 of the largest |reference|
// (TESTING.md, "Tolerances"). Also checks that the runs layout is a permutation of the file's blocks.
#include <cmath>
#include <cstring>
#include <random>
#include <set>

#include "ggml.h"
#include "load/gguf.h"
#include "load/runs.h"
#include "quant.h"
#include "test_util.h"
#include "ze/library.h"

using namespace engine;

namespace {

// The a8 rule of kernels/common.cl, on the host.
std::vector<uint8_t> quantize_a8(const std::vector<float> &x) {
  size_t K = x.size();
  std::vector<uint8_t> a(K + K / 8);
  for (size_t u = 0; u < K / 32; u++) {
    float amax = 0;
    for (int i = 0; i < 32; i++) amax = std::fmax(amax, std::fabs(x[32 * u + i]));
    float d = amax / 127.0f, id = d != 0 ? 1.0f / d : 0.0f;
    for (int i = 0; i < 32; i++) a[32 * u + i] = (uint8_t)(int8_t)std::round(x[32 * u + i] * id);
    std::memcpy(a.data() + K + 4 * u, &d, 4);
  }
  return a;
}

double a8_value(const std::vector<uint8_t> &a, size_t K, size_t i) {
  float d;
  std::memcpy(&d, a.data() + K + 4 * (i / 32), 4);
  return (double)d * (int8_t)a[i];
}

}  // namespace

int main() {
  if (test::gpus() == 0) return test::no_gpu();
  ze::Runtime rt;
  ze::Library lib(rt);
  const uint32_t card = 0;
  auto *grids = static_cast<uint8_t *>(rt.alloc_device(card, QGRID_BYTES));
  ze::List up(rt, card, ze::List::immediate);
  up.copy(grids, codebooks().data(), QGRID_BYTES);
  up.synchronize();
  std::mt19937 rng(1234);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  int failures = 0;

  // The kernels on rows x K of type blocks (the file's layout), named what.
  auto check = [&](uint32_t type, const std::string &what, const uint8_t *blocks, uint64_t rows, uint64_t K) {
    const uint64_t bpr = K / quant_block_values(type), n = rows * bpr;
    std::vector<uint8_t> runs(runs_bytes(type, n)), back(runs.size());
    to_runs(type, {{blocks, bpr * quant_block_bytes(type), 0, rows, 0, bpr}}, runs.data());
    from_runs(type, runs.data(), n, back.data());
    EXPECT(std::memcmp(back.data(), blocks, back.size()) == 0, "%s: runs layout is not a permutation",
           ggml_type_label(type));

    // Reference rows from ggml.
    std::vector<float> wref(rows * K);
    ggml_get_type_traits((ggml_type)type)->to_float(blocks, wref.data(), (int64_t)(rows * K));

    const uint32_t ntok = 4;
    std::vector<uint8_t> act;
    std::vector<std::vector<uint8_t>> a8(ntok);
    for (uint32_t tk = 0; tk < ntok; tk++) {
      std::vector<float> x(K);
      for (auto &v : x) v = normal(rng) * (tk + 1);
      a8[tk] = quantize_a8(x);
      act.insert(act.end(), a8[tk].begin(), a8[tk].end());
    }
    auto *dW = static_cast<uint8_t *>(rt.alloc_device(card, runs.size()));
    auto *dA = static_cast<uint8_t *>(rt.alloc_device(card, act.size()));
    auto *dY = static_cast<float *>(rt.alloc_host(rows * ntok * sizeof(float)));
    auto *dN = static_cast<uint32_t *>(rt.alloc_host(sizeof(uint32_t)));  // the kernels read the token count here
    up.copy(dW, runs.data(), runs.size());
    up.copy(dA, act.data(), act.size());
    up.synchronize();

    // lanes 16, 8, 4 by matvec; then matvec_ks (ks > 0) with 8 lanes and K split 2, 4 and 8.
    const uint32_t cases[][2] = {{16, 0}, {8, 0}, {4, 0}, {8, 2}, {8, 4}, {8, 8}};
    for (const auto &[lanes, ks] : cases) {
      for (uint32_t nt : {1u, ntok}) {
        std::string mod = std::string("matvec_") + ggml_type_label(type) + "_l" + std::to_string(lanes);
        for (auto &c : mod) c = (char)std::tolower(c);
        std::fill(dY, dY + rows * ntok, NAN);
        *dN = nt;
        uint32_t rows32 = (uint32_t)rows, bpr32 = (uint32_t)bpr, K32 = (uint32_t)K, stride = (uint32_t)a8[0].size();
        uint32_t ystride = rows32;
        if (ks == 0) {
          uint32_t groups = (rows32 + 4 * (16 / lanes) - 1) / (4 * (16 / lanes));
          up.launch(lib.get(card, mod, "matvec"), {dW, rows32, bpr32, dA, K32, stride, dN, dY, ystride, grids}, groups,
                    1, 1, 64);
        } else {
          up.launch(lib.get(card, mod, "matvec_ks"), {dW, rows32, bpr32, dA, K32, stride, dN, dY, ystride, grids, ks},
                    (rows32 + 16 / lanes - 1) / (16 / lanes), 1, 1, 16 * ks);
        }
        up.synchronize();
        double maxerr = 0, maxref = 0;
        for (uint32_t tk = 0; tk < nt; tk++)
          for (uint64_t r = 0; r < rows; r++) {
            double ref = 0;
            for (uint64_t i = 0; i < K; i++) ref += (double)wref[r * K + i] * a8_value(a8[tk], K, i);
            maxref = std::max(maxref, std::fabs(ref));
            double e = std::fabs(ref - dY[tk * rows + r]);
            maxerr = std::isnan(e) ? INFINITY : std::max(maxerr, e);
          }
        double rel = maxerr / maxref;
        bool ok = rel <= 1e-5;
        failures += !ok;
        std::printf("%-8s %-28s rows %4llu x %5llu, lanes %2u, K split %u, tokens %u: max error %.2e %s\n",
                    ggml_type_label(type), what.c_str(), (unsigned long long)rows, (unsigned long long)K, lanes,
                    ks ? ks : 1, nt, rel, ok ? "ok" : "FAIL");
      }
    }
    rt.free(dW);
    rt.free(dA);
    rt.free(dY);
    rt.free(dN);
  };

  // Every type the engine reads, on random blocks: 300 rows of 2,560 values.
  for (uint32_t type : test::quant_types()) {
    const uint64_t rows = 300, K = 2560;
    std::vector<uint8_t> blocks = test::random_blocks(type, rows * K / quant_block_values(type), rng);
    check(type, "random blocks", blocks.data(), rows, K);
  }

  // Every quantised type of the file; the first tensor of each, by name order.
  std::string path = test::model_path();
  if (path.empty()) {
    std::printf("model not found under FLASH_MODEL_DIR: the file's rows not checked\n");
  } else {
    GgufModel model(path);
    std::set<uint32_t> types;
    for (auto &[name, ti] : model.tensors())
      if (quant_block_values(ti.type) > 1 && name.find("per_layer") == std::string::npos) types.insert(ti.type);
    for (uint32_t type : types) {
      const GgufTensor *t = nullptr;
      for (auto &[name, ti] : model.tensors())
        if (ti.type == type && name.find("per_layer") == std::string::npos) {
          t = &ti;
          break;
        }
      EXPECT(t, "no tensor of type %s", ggml_type_label(type));
      check(type, t->name, t->data, std::min<uint64_t>(t->shape[1], 1000), t->shape[0]);
    }
  }
  EXPECT(failures == 0, "%d case(s) over tolerance", failures);
  std::printf("PASS\n");
  return 0;
}
