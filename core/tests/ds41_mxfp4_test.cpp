// Level 3 (docs/TESTING.md): the routed experts' kernels (kernels/quant/moe.cl with -DMOE_CLASSES) at MXFP4, the type
// of the drafter's experts, on three experts of the drafter's own file: part A and part B slots as the model builds
// them, at 1, 2, 3, 4, 6 and 8 rows. Limits: gate and up as a8 within half a code of ggml's weights in double, down
// within 1e-5 of the largest |reference|. Needs a card and DS41_DRAFTER, the drafter's file.
#include <atomic>
#include <cmath>
#include <thread>

#include "ds41_kernel_test.h"
#include "ggml.h"
#include "load/direct.h"
#include "load/gguf.h"
#include "load/runs.h"
#include "models/deepseek41/layout.h"
#include "moe_tile.h"
#include "quant.h"

using namespace engine;
using namespace ds41test;

namespace {

constexpr uint32_t E = D41_EMBD, FF = D41_FF, U = D41_DSP_USED, PA = D41_PART_A, PB = D41_PART_B;
constexpr uint32_t T = QT_MXFP4, QK = 32, X_STRIDE = E + E / 8, HA_STRIDE = PA + PA / 8;
constexpr uint32_t N_EXPERTS = 3;
constexpr uint32_t IDS[N_EXPERTS] = {0, 77, 127};
constexpr double DOWN_LIMIT = 1e-5, A8_LIMIT = 1e-6;

uint64_t align256(uint64_t x) { return (x + 255) & ~255ull; }

uint32_t local_size(uint32_t rows_a_sub_group, uint32_t lanes) {
  return 16 * MOE_WG_ROWS / std::max(rows_a_sub_group, 16 / lanes);
}

template <class F>
void in_threads(uint64_t n, const F &f) {
  const uint32_t nt = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
  std::vector<std::thread> ts;
  for (uint32_t t = 0; t < nt; t++) ts.emplace_back([&, t] { f(n * t / nt, n * (t + 1) / nt); });
  for (auto &t : ts) t.join();
}

D41Class make_class(uint32_t rows, uint64_t base) {
  const uint64_t g = align256(runs_bytes(T, (uint64_t)rows * (E / QK)));
  const uint64_t d = align256(runs_bytes(T, (uint64_t)E * (rows / QK)));
  return {base, 2 * g + d, g, 2 * g, rows, {}};
}

double a8_value(const uint8_t *row, uint32_t K, uint32_t i) {
  float d;
  std::memcpy(&d, row + K + 4 * (i / 32), 4);
  return (double)d * (int8_t)row[i];
}

double swiglu(double g, double u) {
  const double gc = std::min(g, (double)D41_SWIGLU_LIMIT);
  const double uc = std::clamp(u, -(double)D41_SWIGLU_LIMIT, (double)D41_SWIGLU_LIMIT);
  return gc / (1.0 + std::exp(-gc)) * uc;
}

// An expert's three matrices in the file's layout.
struct Expert {
  AlignedBuffer buf[3];
  const uint8_t *gate, *up, *down;
};

int run() {
  if (test::gpus() < 1) return test::skip("the MXFP4 kernels need a card");
  const char *path = std::getenv("DS41_DRAFTER");
  if (!path) return test::skip("DS41_DRAFTER is not set: the drafter's file of DeepSeek-V4.1-Flash-1017; MXFP4 is not tested");
  GgufModel gguf(path);
  DirectReader reader(gguf.files());
  const GgufTensor &tg = gguf.tensor("blk.0.ffn_gate_exps.weight"), &tu = gguf.tensor("blk.0.ffn_up_exps.weight");
  const GgufTensor &td = gguf.tensor("blk.0.ffn_down_exps.weight");
  EXPECT(tg.type == T && tu.type == T && td.type == T, "%s: its experts are not MXFP4", path);
  const uint64_t gbytes = tg.bytes / D41_DSP_EXPERTS, dbytes = td.bytes / D41_DSP_EXPERTS;
  const uint64_t grow = (uint64_t)(E / QK) * quant_block_bytes(T), drow = (uint64_t)(FF / QK) * quant_block_bytes(T);
  EXPECT(gbytes == FF * grow && dbytes == E * drow, "%s: an expert's size", path);
  std::vector<Expert> ex(N_EXPERTS);
  for (uint32_t k = 0; k < N_EXPERTS; k++) {
    const auto read = [&](const GgufTensor &t, uint64_t bytes, AlignedBuffer &buf) {
      const uint64_t off = t.offset + IDS[k] * bytes;
      buf = aligned_buffer(DirectReader::span(off, bytes));
      return reader.read(t.shard, off, bytes, buf.get());
    };
    ex[k].gate = read(tg, gbytes, ex[k].buf[0]);
    ex[k].up = read(tu, gbytes, ex[k].buf[1]);
    ex[k].down = read(td, dbytes, ex[k].buf[2]);
  }

  ze::Runtime rt(1);
  ze::Library lib(rt);
  Io io(rt, 0);
  Checker ck;
  std::mt19937 rng(20261110);
  // Part A slots, then part B slots, an expert in each.
  D41Class table[2] = {make_class(PA, 0), make_class(PB, 0)};
  table[1].base = N_EXPERTS * table[0].slot_bytes;
  const uint64_t cache_bytes = N_EXPERTS * (table[0].slot_bytes + table[1].slot_bytes);
  std::vector<uint8_t> host(cache_bytes);
  for (uint32_t cls = 0; cls < 2; cls++)
    for (uint32_t k = 0; k < N_EXPERTS; k++) {
      const D41Class &c = table[cls];
      const uint32_t r0 = cls ? PA : 0;
      uint8_t *s = host.data() + c.base + k * c.slot_bytes;
      to_runs(T, {{ex[k].gate, grow, r0, c.rows, 0, E / QK}}, s);
      to_runs(T, {{ex[k].up, grow, r0, c.rows, 0, E / QK}}, s + c.up_off);
      to_runs(T, {{ex[k].down, drow, 0, E, r0 / QK, c.rows / QK}}, s + c.down_off);
    }
  const uint32_t WMAX = STEP_MAX_ROWS;
  auto *cache = io.dev<uint8_t>(cache_bytes);
  io.put(cache, host);
  auto *classes = io.dev<D41Class>(2);
  io.put(classes, table, 2);
  auto *grids = io.dev<uint8_t>(QGRID_BYTES);
  io.put(grids, codebooks().data(), QGRID_BYTES);
  auto *x = io.dev<uint8_t>((size_t)WMAX * X_STRIDE);
  auto *groups = io.dev<D41Group>(WMAX * U);
  auto *ngroups = io.dev<uint32_t>(16);
  auto *ha = io.dev<uint8_t>((size_t)WMAX * U * HA_STRIDE);
  auto *y = io.dev<float>((size_t)WMAX * U * E);
  const auto traits = ggml_get_type_traits((ggml_type)T);

  for (uint32_t W : {1u, 2u, 3u, 4u, 6u, 8u}) {
    const std::string gmod = "deepseek41_moe_gate_up_mxfp4_g" + std::to_string(W);
    const std::string dmod = "deepseek41_moe_down_mxfp4_g" + std::to_string(W);
    const uint32_t lxg = local_size(D41_GU_ROWS(W), D41_GU_LANES), lxd = local_size(D41_DOWN_ROWS, D41_DOWN_LANES(T, W));
    const uint32_t nasg = W * U;
    // Row t's rank k goes to expert k: every group holds W assignments.
    const std::vector<float> xf = randn(rng, (size_t)W * E);
    std::vector<uint8_t> xa;
    std::vector<double> xd((size_t)W * E);
    for (uint32_t t = 0; t < W; t++) {
      const auto a = a8_row(xf.data() + (size_t)t * E, E);
      xa.insert(xa.end(), a.begin(), a.end());
      for (uint32_t i = 0; i < E; i++) xd[(size_t)t * E + i] = a8_value(a.data(), E, i);
    }
    io.put(x, xa);
    for (uint32_t cls = 0; cls < 2; cls++) {
      const D41Class &c = table[cls];
      const uint32_t j0 = cls ? PA : 0, K = c.rows;
      std::vector<D41Group> g(N_EXPERTS);
      for (uint32_t k = 0; k < N_EXPERTS; k++) {
        std::memset(&g[k], 0, sizeof g[k]);
        g[k].slot = (int32_t)k;
        g[k].n = W;
        g[k].cls = cls;
        for (uint32_t t = 0; t < W; t++) g[k].asg[t] = t * U + k;
      }
      const uint32_t ng = N_EXPERTS, per = U, xs = X_STRIDE, hs = HA_STRIDE, k_in = E, rows = E;
      io.put(groups, g);
      io.put(ngroups, &ng, 1);
      io.put(ha, std::vector<uint8_t>((size_t)nasg * HA_STRIDE, 0xAB));
      io.put(y, std::vector<float>((size_t)nasg * E, std::nanf("")));
      io.L.launch(lib.get(0, gmod, "moe_gate_up_c"), {cache, classes, groups, ngroups, x, k_in, xs, per, ha, hs, grids},
                  PA / MOE_WG_ROWS, nasg, 1, lxg);
      io.L.synchronize();
      io.L.launch(lib.get(0, dmod, "moe_down_c"), {cache, classes, groups, ngroups, rows, ha, hs, y, grids},
                  E / MOE_WG_ROWS, nasg, 1, lxd);
      io.L.synchronize();
      const auto ha_host = io.get(ha, (size_t)nasg * HA_STRIDE);
      const auto y_host = io.get(y, (size_t)nasg * E);

      for (uint32_t a = 0; a < nasg; a++) {
        const Expert &e = ex[a % U];
        const double *xr = xd.data() + (size_t)(a / U) * E;
        const uint8_t *row = ha_host.data() + (size_t)a * HA_STRIDE;
        // Gate and up in double, then the card's a8 row against them: half a code, and the unit's scale.
        std::vector<double> h(K);
        in_threads(K, [&](uint64_t i0, uint64_t i1) {
          std::vector<float> gw(E), uw(E);
          for (uint64_t i = i0; i < i1; i++) {
            traits->to_float(e.gate + (j0 + i) * grow, gw.data(), E);
            traits->to_float(e.up + (j0 + i) * grow, uw.data(), E);
            double gv = 0, uv = 0;
            for (uint32_t j = 0; j < E; j++) gv += (double)gw[j] * xr[j], uv += (double)uw[j] * xr[j];
            h[i] = swiglu(gv, uv);
          }
        });
        double worst = 0, largest = 0;
        for (uint32_t i = 0; i < K; i++) largest = std::max(largest, std::fabs(h[i]));
        for (uint32_t u = 0; u < K / 32; u++) {
          double amax = 0;
          for (uint32_t i = 0; i < 32; i++) amax = std::max(amax, std::fabs(h[32 * u + i]));
          float d;
          std::memcpy(&d, row + K + 4 * u, 4);
          worst = std::max(worst, std::fabs((double)d - amax / 127.0) / largest);
          for (uint32_t i = 0; i < 32; i++)
            worst = std::max(worst, (std::fabs(a8_value(row, K, 32 * u + i) - h[32 * u + i]) - 0.5 * d) / largest);
        }
        ck.note(cls ? "gate and up as a8, part B" : "gate and up as a8, part A", worst, A8_LIMIT);
        for (size_t b = K + K / 8; b < HA_STRIDE; b++)
          if (row[b] != 0xAB) ck.note("a8 h stores past its part", 1, 0);

        // down . h, with the card's a8 h.
        std::vector<double> hv(K), yr(E);
        for (uint32_t j = 0; j < K; j++) hv[j] = a8_value(row, K, j);
        in_threads(E, [&](uint64_t r0, uint64_t r1) {
          std::vector<float> dw(FF);
          for (uint64_t ri = r0; ri < r1; ri++) {
            traits->to_float(e.down + ri * drow, dw.data(), FF);
            double acc = 0;
            for (uint32_t j = 0; j < K; j++) acc += (double)dw[j0 + j] * hv[j];
            yr[ri] = acc;
          }
        });
        ck.rel(cls ? "down, part B" : "down, part A", y_host.data() + (size_t)a * E, yr.data(), E, DOWN_LIMIT);
      }
    }
    std::printf("%u rows: checked\n", W);
    std::fflush(stdout);
  }
  return ck.report();
}

}  // namespace

int main() {
  try {
    return run();
  } catch (const std::exception &e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
}
