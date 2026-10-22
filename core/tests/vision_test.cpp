// Level 3 (docs/TESTING.md): the image encoder's kernels (kernels/models/qwen4exp/vision.cl) against the rules of
// docs/spec/MODEL_VISION.md in double precision, on random inputs: v_embed, v_norm, v_qkv, v_attn, v_bias_gelu,
// v_residual and v_bias. Tolerances (TESTING.md, "Tolerances"): 1e-5 of the largest |reference| for f32 outputs,
// 2e-3 for fp16 ones, 1e-2 for bf16 ones. The GELU is checked at every fp16 value from -12 to 12 against ggml's
// table (section 3.4), exactly.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>

#include "ggml.h"
#include "test_util.h"
#include "ze/library.h"

using namespace engine;

namespace {

constexpr uint32_t D = 1152, H = 16, HD = 72, HDP = 80, ROT = 36, SEC = 18, PATCH = 768, GRID = 48;

float bf16_to_f32(uint16_t x) {
  uint32_t u = (uint32_t)x << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

int failures = 0;

void report(const char *what, const std::vector<double> &ref, const std::vector<double> &got, double tol) {
  double maxerr = 0, maxref = 0;
  for (size_t i = 0; i < ref.size(); i++) {
    maxref = std::max(maxref, std::fabs(ref[i]));
    const double e = std::fabs(ref[i] - got[i]);
    maxerr = std::isnan(e) ? INFINITY : std::max(maxerr, e);
  }
  const double rel = maxerr / maxref;
  const bool ok = rel <= tol;
  failures += !ok;
  std::printf("%-12s max error %.2e of %.3g %s\n", what, rel, maxref, ok ? "ok" : "FAIL");
}

// Section 2.2: patch n's row and column.
void patch_rc(uint32_t n, uint32_t gw, uint32_t &r, uint32_t &c) {
  const uint32_t win = n / 4;
  r = 2 * (win / (gw / 2)) + (n % 4) / 2;
  c = 2 * (win % (gw / 2)) + n % 2;
}

// ggml's GELU table, as ggml-cpu.c fills it.
std::vector<uint16_t> gelu_table() {
  std::vector<uint16_t> t(1 << 16);
  for (uint32_t i = 0; i < t.size(); i++) {
    const float x = ggml_fp16_to_fp32((ggml_fp16_t)i);
    t[i] = ggml_fp32_to_fp16(0.5f * x * (1.0f + tanhf(0.79788456080286535587989211986876f * x * (1.0f + 0.044715f * x * x))));
  }
  return t;
}

double gelu_tanh(double x) { return 0.5 * x * (1 + std::tanh(std::sqrt(2 / M_PI) * (x + 0.044715 * x * x * x))); }

}  // namespace

int main() {
  if (test::gpus() == 0) return test::no_gpu();
  ze::Runtime rt;
  ze::Library lib(rt);
  const uint32_t card = 0;
  ze::List L(rt, card, ze::List::immediate);
  auto k = [&](const char *name) { return lib.get(card, "qwen4exp_vision", name); };
  std::mt19937 rng(7);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  auto host = [&](size_t bytes) { return rt.alloc_host(std::max<size_t>(bytes, 256)); };
  std::vector<void *> held;
  auto copy_in = [&](const void *src, size_t bytes) {
    void *p = host(bytes);
    std::memcpy(p, src, bytes);
    held.push_back(p);
    return p;
  };
  const std::vector<uint16_t> gtab = gelu_table();
  void *gelu = copy_in(gtab.data(), gtab.size() * 2);

  // Images of gh x gw patches: one wider than tall and one at the table's own size.
  for (auto [gh, gw] : {std::pair<uint32_t, uint32_t>{6, 10}, {48, 48}}) {
    const uint32_t N = gh * gw;
    std::printf("image %u x %u patches\n", gh, gw);

    // v_embed, over the patches from 4 on (a partial first group would hide an offset error): a still image's one
    // frame, and a video's pair of frames with a matrix each.
    for (uint32_t F = 1; F <= 2; F++) {
      const uint32_t first = 4, n = N - first;
      std::vector<uint8_t> pix((size_t)n * F * PATCH);
      for (auto &v : pix) v = (uint8_t)(rng() & 255);
      std::vector<float> wt((size_t)F * PATCH * D), bias(D), tab((size_t)GRID * GRID * D);
      for (auto &v : wt) v = normal(rng) * 0.03f;
      for (auto &v : bias) v = normal(rng);
      for (auto &v : tab) v = normal(rng);
      auto *x = static_cast<float *>(host((size_t)n * D * 4));
      L.launch(k("v_embed"),
               {copy_in(pix.data(), pix.size()), n, first, gh, gw, F, copy_in(wt.data(), wt.size() * 4),
                copy_in(bias.data(), D * 4), copy_in(tab.data(), tab.size() * 4), x},
               (n + 7) / 8, 1, 1, 128);
      L.synchronize();
      std::vector<double> ref((size_t)n * D), got(x, x + (size_t)n * D);
      for (uint32_t t = 0; t < n; t++) {
        uint32_t r, c;
        patch_rc(first + t, gw, r, c);
        const double fy = gh > 1 ? (double)r * (GRID - 1) / (gh - 1) : 0, fx = gw > 1 ? (double)c * (GRID - 1) / (gw - 1) : 0;
        const uint32_t y0 = (uint32_t)fy, x0 = (uint32_t)fx, y1 = std::min(y0 + 1, GRID - 1), x1 = std::min(x0 + 1, GRID - 1);
        const double dy = fy - y0, dx = fx - x0;
        for (uint32_t o = 0; o < D; o++) {
          double e = bias[o];
          for (uint32_t f = 0; f < F; f++)
            for (uint32_t i = 0; i < PATCH; i++)
              e += (double)wt[((size_t)f * PATCH + i) * D + o] * (pix[((size_t)t * F + f) * PATCH + i] / 255.0 - 0.5) / 0.5;
          auto T = [&](uint32_t y, uint32_t xx) { return (double)tab[((size_t)y * GRID + xx) * D + o]; };
          e += (1 - dy) * (1 - dx) * T(y0, x0) + (1 - dy) * dx * T(y0, x1) + dy * (1 - dx) * T(y1, x0) + dy * dx * T(y1, x1);
          ref[(size_t)t * D + o] = e;
        }
      }
      report(F == 1 ? "v_embed" : "v_embed x2", ref, got, 1e-5);
      rt.free(x);
    }

    // v_norm.
    {
      std::vector<float> xs((size_t)N * D), g(D), b(D);
      for (auto &v : xs) v = normal(rng) * 3.0f + 1.0f;
      for (auto &v : g) v = normal(rng);
      for (auto &v : b) v = normal(rng);
      auto *out = static_cast<uint16_t *>(host((size_t)N * D * 2));
      L.launch(k("v_norm"), {copy_in(xs.data(), xs.size() * 4), N, copy_in(g.data(), D * 4), copy_in(b.data(), D * 4), out},
               N, 1, 1, 16);
      L.synchronize();
      std::vector<double> ref((size_t)N * D), got((size_t)N * D);
      for (uint32_t t = 0; t < N; t++) {
        double m = 0, v = 0;
        for (uint32_t i = 0; i < D; i++) m += xs[(size_t)t * D + i];
        m /= D;
        for (uint32_t i = 0; i < D; i++) v += (xs[(size_t)t * D + i] - m) * (xs[(size_t)t * D + i] - m);
        const double inv = 1 / std::sqrt(v / D + 1e-6);
        for (uint32_t i = 0; i < D; i++) {
          ref[(size_t)t * D + i] = (xs[(size_t)t * D + i] - m) * inv * g[i] + b[i];
          got[(size_t)t * D + i] = bf16_to_f32(out[(size_t)t * D + i]);
        }
      }
      report("v_norm", ref, got, 1e-2);
      rt.free(out);
    }

    // v_qkv, then v_attn on its output: the second half of the rows as queries, every row as keys, as one card of
    // two takes them.
    {
      std::vector<float> y((size_t)N * 3 * D), bias(3 * D);
      for (auto &v : y) v = normal(rng);
      for (auto &v : bias) v = normal(rng) * 0.5f;
      const size_t W = (size_t)H * HDP;
      auto *q = static_cast<uint16_t *>(host((size_t)N * W * 2));
      auto *kk = static_cast<uint16_t *>(host((size_t)N * W * 2));
      auto *v = static_cast<uint16_t *>(host((size_t)N * W * 2));
      for (auto *p : {q, kk, v}) std::memset(p, 0xff, (size_t)N * W * 2);
      // Rows from 2 on: the positions follow first.
      const uint32_t first = 2, n = N - first;
      L.launch(k("v_qkv"), {copy_in(y.data() + (size_t)first * 3 * D, (size_t)n * 3 * D * 4), copy_in(bias.data(), 3 * D * 4),
                            n, first, gw, q + first * W, kk + first * W, v + first * W},
               H * ROT / 64, n, 1, 64);
      L.launch(k("v_qkv"), {copy_in(y.data(), (size_t)first * 3 * D * 4), copy_in(bias.data(), 3 * D * 4), first, 0u, gw, q,
                            kk, v},
               H * ROT / 64, first, 1, 64);
      L.synchronize();
      std::vector<double> ref((size_t)N * 3 * W), got((size_t)N * 3 * W);
      for (uint32_t t = 0; t < N; t++) {
        uint32_t r, c;
        patch_rc(t, gw, r, c);
        for (uint32_t part = 0; part < 3; part++)
          for (uint32_t h = 0; h < H; h++)
            for (uint32_t d = 0; d < HDP; d++) {
              double val = 0;
              if (d < HD) {
                auto in = [&](uint32_t dd) {
                  const size_t a = (size_t)part * D + h * HD + dd;
                  return (double)y[(size_t)t * 3 * D + a] + bias[a];
                };
                val = in(d);
                if (part < 2) {
                  const uint32_t j = d % ROT;
                  const double th = (j < SEC ? r : c) * std::pow(10000.0, -2.0 * (j % SEC) / ROT);
                  val = d < ROT ? in(j) * std::cos(th) - in(j + ROT) * std::sin(th)
                                : in(j) * std::sin(th) + in(j + ROT) * std::cos(th);
                }
              }
              const uint16_t *src = part == 0 ? q : part == 1 ? kk : v;
              const size_t at = (size_t)t * W + h * HDP + d;
              ref[((size_t)t * 3 + part) * W + h * HDP + d] = val;
              got[((size_t)t * 3 + part) * W + h * HDP + d] = ggml_fp16_to_fp32(src[at]);
            }
      }
      report("v_qkv", ref, got, 2e-3);

      const uint32_t q0 = N / 2 / 4 * 4, nq = N - q0;
      const float scale = 1.0f / std::sqrt(72.0f);
      auto *o = static_cast<uint16_t *>(host((size_t)nq * D * 2));
      L.launch(lib.get(card, "qwen4exp_vision_attn", "v_attn"), {q + (size_t)q0 * W, nq, kk, v, N, scale, o}, H,
               (nq + 15) / 16, 1, 16);
      L.synchronize();
      std::vector<double> aref((size_t)nq * D), agot((size_t)nq * D);
      for (uint32_t t = 0; t < nq; t++)
        for (uint32_t h = 0; h < H; h++) {
          std::vector<double> s(N);
          double mx = -INFINITY, sum = 0;
          for (uint32_t i = 0; i < N; i++) {
            double dot = 0;
            for (uint32_t d = 0; d < HD; d++)
              dot += (double)ggml_fp16_to_fp32(q[(size_t)(q0 + t) * W + h * HDP + d]) *
                     ggml_fp16_to_fp32(kk[(size_t)i * W + h * HDP + d]);
            s[i] = dot * scale;
            mx = std::max(mx, s[i]);
          }
          for (auto &e : s) sum += e = std::exp(e - mx);
          for (uint32_t d = 0; d < HD; d++) {
            double acc = 0;
            for (uint32_t i = 0; i < N; i++) acc += s[i] * ggml_fp16_to_fp32(v[(size_t)i * W + h * HDP + d]);
            aref[(size_t)t * D + h * HD + d] = acc / sum;
            agot[(size_t)t * D + h * HD + d] = bf16_to_f32(o[(size_t)t * D + h * HD + d]);
          }
        }
      report("v_attn", aref, agot, 1e-2);
      for (void *p : {(void *)q, (void *)kk, (void *)v, (void *)o}) rt.free(p);
    }

    // v_bias_gelu at the up product's shape: 4,304 values into rows of 4,320, the rest zero.
    {
      const uint32_t cols = 4304, ldo = 4320, ldy = 4304;
      std::vector<float> y((size_t)N * ldy), b(cols);
      for (auto &e : y) e = normal(rng) * 3.0f;
      for (auto &e : b) e = normal(rng);
      auto *out = static_cast<uint16_t *>(host((size_t)N * ldo * 2));
      std::memset(out, 0xff, (size_t)N * ldo * 2);
      L.launch(k("v_bias_gelu"), {copy_in(y.data(), y.size() * 4), ldy, copy_in(b.data(), cols * 4), cols, N, out, ldo, gelu},
               (ldo + 63) / 64, N, 1, 64);
      L.synchronize();
      std::vector<double> ref((size_t)N * ldo), got((size_t)N * ldo);
      for (uint32_t t = 0; t < N; t++)
        for (uint32_t i = 0; i < ldo; i++) {
          ref[(size_t)t * ldo + i] = i < cols ? gelu_tanh((double)y[(size_t)t * ldy + i] + b[i]) : 0;
          got[(size_t)t * ldo + i] = bf16_to_f32(out[(size_t)t * ldo + i]);
        }
      report("v_bias_gelu", ref, got, 1e-2);
      rt.free(out);
    }

    // v_residual and v_bias.
    {
      std::vector<float> x0((size_t)N * D), y((size_t)N * D), b(D);
      for (auto &e : x0) e = normal(rng);
      for (auto &e : y) e = normal(rng);
      for (auto &e : b) e = normal(rng);
      auto *x = static_cast<float *>(copy_in(x0.data(), x0.size() * 4));
      auto *out = static_cast<float *>(host((size_t)N * D * 4));
      L.launch(k("v_residual"), {x, copy_in(y.data(), y.size() * 4), copy_in(b.data(), D * 4), N}, D / 64, N, 1, 64);
      L.launch(k("v_bias"), {copy_in(y.data(), y.size() * 4), copy_in(b.data(), D * 4), D, N, out}, D / 64, N, 1, 64);
      L.synchronize();
      std::vector<double> rr((size_t)N * D), rg(x, x + (size_t)N * D), br((size_t)N * D), bg(out, out + (size_t)N * D);
      for (size_t i = 0; i < rr.size(); i++) {
        rr[i] = (double)x0[i] + y[i] + b[i % D];
        br[i] = (double)y[i] + b[i % D];
      }
      report("v_residual", rr, rg, 1e-5);
      report("v_bias", br, bg, 1e-5);
      rt.free(out);
    }
  }

  // The GELU at every fp16 value in [-12, 12]: ggml's value, exactly.
  {
    std::vector<float> xs;
    for (uint32_t u = 0; u < 65536; u++) {
      const float f = ggml_fp16_to_fp32((uint16_t)u);
      if (std::isfinite(f) && std::fabs(f) <= 12.0f) xs.push_back(f);
    }
    const uint32_t n = (uint32_t)xs.size();
    std::vector<float> zero(n, 0.0f);
    auto *out = static_cast<uint16_t *>(host((size_t)n * 2));
    L.launch(k("v_bias_gelu"), {copy_in(xs.data(), n * 4), n, copy_in(zero.data(), n * 4), n, 1u, out, n, gelu}, (n + 63) / 64, 1, 1,
             64);
    L.synchronize();
    // The kernel writes bf16: compare after the same rounding.
    auto bf = [](float f) {
      uint32_t u;
      std::memcpy(&u, &f, 4);
      u = (u + 0x7fffu + ((u >> 16) & 1)) & 0xffff0000u;
      float r;
      std::memcpy(&r, &u, 4);
      return r;
    };
    uint32_t bad = 0;
    for (uint32_t i = 0; i < n; i++) {
      const float x = xs[i];
      const float want = x <= -10 ? 0.0f : x >= 10 ? x : ggml_fp16_to_fp32(gtab[ggml_fp32_to_fp16(x)]);
      bad += bf16_to_f32(out[i]) != bf(want);
    }
    std::printf("gelu         %u of %u fp16 inputs differ %s\n", bad, n, bad ? "FAIL" : "ok");
    failures += bad != 0;
    rt.free(out);
  }

  for (void *p : held) rt.free(p);
  EXPECT(failures == 0, "%d check(s) over tolerance", failures);
  std::printf("PASS\n");
  return 0;
}
