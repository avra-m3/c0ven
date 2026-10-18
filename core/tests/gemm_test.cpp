// Level 2 and 3 (docs/TESTING.md): the XMX products of kernels/gemm.cl, fp16 and bf16, against the same 16-bit
// inputs multiplied in double precision. This also checks the register layout of the DPAS builtins that
// kernels/include/builtins.h states. Tolerance 2e-3 of the largest |reference| (TESTING.md, "Tolerances"). The bf16
// modules with other tiles (gemm_bf16_t4n<TN>) and the banded order must give the plain launch's bits, and the second
// output Y2 of an fp16 product the first's.
#include <cmath>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "ggml.h"
#include "test_util.h"
#include "ze/library.h"

using namespace engine;

namespace {

uint16_t to_bf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return (uint16_t)((u + (0x7fffu + ((u >> 16) & 1))) >> 16);
}
float from_bf16(uint16_t h) {
  uint32_t u = (uint32_t)h << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

}  // namespace

int main() {
  if (test::gpus() == 0) return test::no_gpu();
  ze::Runtime rt;
  ze::Library lib(rt);
  const uint32_t card = 0;
  ze::List L(rt, card, ze::List::immediate);
  std::mt19937 rng(7);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  int failures = 0;
  struct Shape {
    uint32_t T, N, K;
  };
  for (bool bf : {false, true})
    for (Shape s : {Shape{1, 33, 16}, Shape{37, 300, 320}, Shape{100, 513, 2560}, Shape{64, 4, 10240},
                    Shape{37, 2560, 320}}) {
      std::vector<uint16_t> W((size_t)s.N * s.K), X((size_t)s.T * s.K);
      std::vector<double> wf(W.size()), xf(X.size());
      auto fill = [&](std::vector<uint16_t> &v, std::vector<double> &f, float sc) {
        for (size_t i = 0; i < v.size(); i++) {
          float x = normal(rng) * sc;
          v[i] = bf ? to_bf16(x) : ggml_fp32_to_fp16(x);
          f[i] = bf ? from_bf16(v[i]) : ggml_fp16_to_fp32(v[i]);
        }
      };
      fill(W, wf, 0.05f);
      fill(X, xf, 1.0f);
      auto *dW = static_cast<uint16_t *>(rt.alloc_device(card, W.size() * 2));
      auto *dX = static_cast<uint16_t *>(rt.alloc_device(card, X.size() * 2));
      auto *dT = static_cast<uint32_t *>(rt.alloc_host(4));
      auto *dY = static_cast<float *>(rt.alloc_host((size_t)s.T * s.N * 4));
      *dT = s.T;
      L.copy(dW, W.data(), W.size() * 2);
      L.copy(dX, X.data(), X.size() * 2);
      L.synchronize();
      for (uint32_t mode : {0u, 1u}) {
        std::fill(dY, dY + (size_t)s.T * s.N, NAN);
        auto k = lib.get(card, bf ? "gemm_bf16" : "gemm_f16", "gemm");
        uint32_t zero = 0;
        float scale = 0.5f;
        uint16_t *null = nullptr;
        L.launch(k, {dW, s.N, s.K, s.K, dX, s.K, dT, dY, s.N, zero, mode, scale, null, zero}, (s.N + 127) / 128,
                 (s.T + 63) / 64, 1, 64);
        L.synchronize();
        std::vector<float> y0(dY, dY + (size_t)s.T * s.N);
        for (uint32_t band : {1u, 4u}) {
          std::fill(dY, dY + (size_t)s.T * s.N, NAN);
          L.launch(k, {dW, s.N, s.K, s.K, dX, s.K, dT, dY, s.N, zero, mode, scale, null, band}, (s.N + 127) / 128,
                   (s.T + 63) / 64, 1, 64);
          L.synchronize();
          bool same = std::memcmp(y0.data(), dY, y0.size() * 4) == 0;
          failures += !same;
          std::printf("%s T %4u N %4u K %5u mode %u: band %u %s\n", bf ? "bf16" : "fp16", s.T, s.N, s.K, mode, band,
                      same ? "same bits" : "FAIL, not the plain order's bits");
        }
        if (bf) {
          for (uint32_t tn : {2u, 1u}) {
            std::fill(dY, dY + (size_t)s.T * s.N, NAN);
            std::string mod = "gemm_bf16_t4n" + std::to_string(tn);
            L.launch(lib.get(card, mod, "gemm"),
                     {dW, s.N, s.K, s.K, dX, s.K, dT, dY, s.N, zero, mode, scale, null, zero},
                     (s.N + 64 * tn - 1) / (64 * tn), (s.T + 31) / 32, 1, 64);
            L.synchronize();
            bool same = std::memcmp(y0.data(), dY, y0.size() * 4) == 0;
            failures += !same;
            std::printf("%s T %4u N %4u K %5u mode %u: %s\n", mod.c_str(), s.T, s.N, s.K, mode,
                        same ? "same bits" : "FAIL, not gemm_bf16's bits");
          }
        }
        std::memcpy(dY, y0.data(), y0.size() * 4);
        if (s.N % 32 == 0) {
          // fp16 into Y (rows N apart) and Y2: the same bits.
          const size_t n16 = (size_t)s.T * s.N;
          auto *h1 = static_cast<uint16_t *>(rt.alloc_host(n16 * 2));
          auto *h2 = static_cast<uint16_t *>(rt.alloc_host(n16 * 2));
          std::fill(h1, h1 + n16, 0xffff);
          std::fill(h2, h2 + n16, 0);
          L.launch(k, {dW, s.N, s.K, s.K, dX, s.K, dT, (float *)h1, s.N, 1u, mode, scale, h2, zero}, (s.N + 127) / 128,
                   (s.T + 63) / 64, 1, 64);
          L.synchronize();
          bool same = std::memcmp(h1, h2, n16 * 2) == 0;
          failures += !same;
          std::printf("%s T %4u N %4u K %5u mode %u: fp16 Y2 %s\n", bf ? "bf16" : "fp16", s.T, s.N, s.K, mode,
                      same ? "same bits as Y" : "FAIL, not Y's bits");
          rt.free(h1);
          rt.free(h2);
        }
        double maxerr = 0, maxref = 0;
        for (uint32_t t = 0; t < s.T; t++)
          for (uint32_t n = 0; n < s.N; n++) {
            double r = 0;
            for (uint32_t i = 0; i < s.K; i++) r += wf[(size_t)n * s.K + i] * xf[(size_t)t * s.K + i];
            r *= scale;
            if (mode == 1) r = r / (1 + std::exp(-r));
            maxref = std::max(maxref, std::fabs(r));
            double e = std::fabs(r - dY[(size_t)t * s.N + n]);
            maxerr = std::isnan(e) ? INFINITY : std::max(maxerr, e);
          }
        double rel = maxerr / maxref;
        bool ok = rel <= 2e-3;
        failures += !ok;
        std::printf("%s T %4u N %4u K %5u mode %u: max error %.2e %s\n", bf ? "bf16" : "fp16", s.T, s.N, s.K, mode, rel,
                    ok ? "ok" : "FAIL");
      }
      rt.free(dW);
      rt.free(dX);
      rt.free(dT);
      rt.free(dY);
    }
  EXPECT(failures == 0, "%d case(s) over tolerance", failures);
  std::printf("PASS\n");
  return 0;
}
