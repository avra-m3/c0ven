// Level 3 (docs/TESTING.md): the experts' gate and up kernels built with and without -DSWIGLU_LIMIT=10.0f
// (kernels/include/swiglu.h), against silu(g) * u and silu(min(g, 10)) * clamp(u, -10, 10) in double precision, g and
// u from ggml's dequantisation of the same IQ4_XS weights and the same activations: matvec.cl's and moe.cl's
// moe_gate_up_g (1e-5, and equal to each other bit for bit), qgemm.cl's moe_gate_up_t in the file's layout (2e-3)
// and common.cl's swiglu (1e-5). Random weights, so no model is needed.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>

#include "ggml.h"
#include "load/runs.h"
#include "moe_tile.h"
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

uint64_t align256(uint64_t x) { return (x + 255) & ~255ull; }

// The SwiGLU with limit L (0: none), as kernels/include/swiglu.h defines it.
double swiglu(double g, double u, double L) {
  if (L > 0) {
    g = std::min(g, L);
    u = std::clamp(u, -L, L);
  }
  return g / (1 + std::exp(-g)) * u;
}

struct Checker {
  int failures = 0;
  // got as f32 values, or as fp16 when half.
  void check(const char *what, const std::vector<double> &ref, const void *got, bool half, double tol) {
    double maxerr = 0, maxref = 0;
    for (size_t i = 0; i < ref.size(); i++) {
      double v = half ? ggml_fp16_to_fp32(((const uint16_t *)got)[i]) : ((const float *)got)[i];
      maxref = std::max(maxref, std::fabs(ref[i]));
      double e = std::fabs(ref[i] - v);
      maxerr = std::isnan(e) ? INFINITY : std::max(maxerr, e);
    }
    double rel = maxerr / maxref;
    bool ok = rel <= tol;
    failures += !ok;
    std::printf("%-44s max error %.2e %s\n", what, rel, ok ? "ok" : "FAIL");
  }
};

}  // namespace

int main() {
  if (test::gpus() == 0) return test::no_gpu();
  ze::Runtime rt;
  ze::Library lib(rt);
  const uint32_t card = 0;
  auto *grids = static_cast<uint8_t *>(rt.alloc_device(card, QGRID_BYTES));
  ze::List L(rt, card, ze::List::immediate);
  L.copy(grids, codebooks().data(), QGRID_BYTES);
  L.synchronize();
  std::mt19937 rng(10);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  Checker ck;
  const double LIMIT = 10.0;

  // 4 experts of 64 rows of 512 values; token t's two assignments go to experts t and t + 1 (mod 4), so each expert
  // has two. Gate and up values have a standard deviation of about 8, so the limit clamps a fair share of them.
  const ggml_type type = GGML_TYPE_IQ4_XS;
  const uint32_t NEXP = 4, NTOK = 4, PER = 2, nasg = NTOK * PER, K = 512, rows = 64;
  const uint32_t bpr = K / quant_block_values(type);
  const size_t rb = ggml_row_size(type, K);
  std::vector<std::vector<uint8_t>> gq(NEXP), uq(NEXP);
  std::vector<std::vector<float>> gw(NEXP), uw(NEXP);
  ggml_quantize_init(type);
  for (uint32_t e = 0; e < NEXP; e++) {
    for (auto *pair : {&gq, &uq}) {
      std::vector<float> f((size_t)rows * K);
      for (auto &v : f) v = normal(rng) * 0.35f;
      (*pair)[e].resize(rows * rb);
      ggml_quantize_chunk(type, f.data(), (*pair)[e].data(), 0, rows, K, nullptr);
    }
    gw[e].resize((size_t)rows * K);
    uw[e].resize((size_t)rows * K);
    ggml_get_type_traits(type)->to_float(gq[e].data(), gw[e].data(), (int64_t)gw[e].size());
    ggml_get_type_traits(type)->to_float(uq[e].data(), uw[e].data(), (int64_t)uw[e].size());
  }
  auto expert_of = [&](uint32_t a) { return (a / PER + a % PER) % NEXP; };
  std::vector<std::vector<float>> x(NTOK, std::vector<float>(K));
  for (auto &t : x)
    for (auto &v : t) v = normal(rng);

  // g and u per assignment and row, for the a8 activations (decode) and the fp16 ones (prompt).
  std::vector<double> g8((size_t)nasg * rows), u8(g8.size()), g16(g8.size()), u16(g8.size());
  std::vector<uint8_t> act;
  for (uint32_t t = 0; t < NTOK; t++) {
    auto a = quantize_a8(x[t]);
    act.insert(act.end(), a.begin(), a.end());
    for (uint32_t j = 0; j < PER; j++) {
      const uint32_t m = t * PER + j, e = expert_of(m);
      for (uint32_t r = 0; r < rows; r++) {
        double ga = 0, ua = 0, gh = 0, uh = 0;
        for (uint32_t i = 0; i < K; i++) {
          double xa = a8_value(a, K, i), xh = ggml_fp16_to_fp32(ggml_fp32_to_fp16(x[t][i]));
          ga += gw[e][(size_t)r * K + i] * xa;
          ua += uw[e][(size_t)r * K + i] * xa;
          gh += gw[e][(size_t)r * K + i] * xh;
          uh += uw[e][(size_t)r * K + i] * xh;
        }
        g8[(size_t)m * rows + r] = ga;
        u8[(size_t)m * rows + r] = ua;
        g16[(size_t)m * rows + r] = gh;
        u16[(size_t)m * rows + r] = uh;
      }
    }
  }
  auto refs = [&](const std::vector<double> &g, const std::vector<double> &u, double lim) {
    std::vector<double> h(g.size());
    for (size_t i = 0; i < g.size(); i++) h[i] = swiglu(g[i], u[i], lim);
    return h;
  };
  // The limit must change h, or the checks below show nothing.
  {
    auto a = refs(g8, u8, 0), b = refs(g8, u8, LIMIT);
    double d = 0, m = 0;
    for (size_t i = 0; i < a.size(); i++) d = std::max(d, std::fabs(a[i] - b[i])), m = std::max(m, std::fabs(a[i]));
    EXPECT(d / m > 1e-2, "the limit changes h by only %.2e", d / m);
  }

  // The decode step's kernels: the experts in the runs layout, one slot each, gate then up; one group per expert.
  const uint64_t mat = align256(runs_bytes(type, (uint64_t)rows * bpr)), slot_bytes = 2 * mat, up_off = mat;
  std::vector<uint8_t> host(NEXP * slot_bytes);
  for (uint32_t e = 0; e < NEXP; e++) {
    to_runs(type, {{gq[e].data(), rb, 0, rows, 0, bpr}}, host.data() + e * slot_bytes);
    to_runs(type, {{uq[e].data(), rb, 0, rows, 0, bpr}}, host.data() + e * slot_bytes + up_off);
  }
  std::vector<MoeGroup> groups(NEXP);
  for (uint32_t e = 0; e < NEXP; e++) {
    std::memset(&groups[e], 0, sizeof(MoeGroup));
    groups[e].slot = (int32_t)e;
  }
  for (uint32_t m = 0; m < nasg; m++) {
    MoeGroup &G = groups[expert_of(m)];
    G.asg[G.n++] = m;
  }
  const uint32_t ng = NEXP, stride = K + K / 8;
  auto *cache = static_cast<uint8_t *>(rt.alloc_device(card, host.size()));
  auto *dG = static_cast<MoeGroup *>(rt.alloc_device(card, ng * sizeof(MoeGroup)));
  auto *dN = static_cast<uint32_t *>(rt.alloc_device(card, 4));
  auto *dA = static_cast<uint8_t *>(rt.alloc_device(card, act.size()));
  L.copy(cache, host.data(), host.size());
  L.copy(dG, groups.data(), ng * sizeof(MoeGroup));
  L.copy(dN, &ng, 4);
  L.copy(dA, act.data(), act.size());
  L.synchronize();
  const size_t nout = (size_t)nasg * rows;
  auto *hm = static_cast<float *>(rt.alloc_host(nout * 4));
  auto *hg = static_cast<float *>(rt.alloc_host(nout * 4));
  uint8_t *no_a8 = nullptr;
  for (bool limited : {false, true}) {
    const std::string sfx = limited ? "_limit10" : "";
    std::fill(hm, hm + nout, NAN);
    std::fill(hg, hg + nout, NAN);
    L.launch(lib.get(card, "matvec_iq4_xs_l16" + sfx, "moe_gate_up_g"),
             {cache, slot_bytes, dG, dN, rows, bpr, up_off, dA, K, stride, PER, hm, grids}, (rows + 3) / 4, ng, 1, 64);
    L.launch(lib.get(card, "moe_iq4_xs_g2" + sfx, "moe_gate_up_g"),
             {cache, slot_bytes, dG, dN, rows, bpr, up_off, dA, K, stride, PER, hg, grids, no_a8},
             (rows + MOE_WG_ROWS - 1) / MOE_WG_ROWS, ng, 1, 16 * MOE_WG_ROWS / MOE_GU_ROWS(2));
    L.synchronize();
    auto ref = refs(g8, u8, limited ? LIMIT : 0);
    ck.check(("matvec_iq4_xs_l16" + sfx + " moe_gate_up_g").c_str(), ref, hm, false, 1e-5);
    ck.check(("moe_iq4_xs_g2" + sfx + " moe_gate_up_g").c_str(), ref, hg, false, 1e-5);
    size_t differ = 0;
    for (size_t i = 0; i < nout; i++) differ += std::memcmp(&hm[i], &hg[i], 4) != 0;
    ck.failures += differ != 0;
    std::printf("%-44s %zu values differ %s\n", ("moe.cl against matvec.cl" + sfx).c_str(), differ,
                differ ? "FAIL" : "ok");
  }

  // The prompt path's kernel, the experts in the file's layout: the assignments grouped by expert, one tile each.
  const uint64_t pup_off = align256(rows * rb), pslot = 2 * pup_off;
  std::vector<uint8_t> pbuf(NEXP * pslot, 0);
  for (uint32_t e = 0; e < NEXP; e++) {
    std::memcpy(pbuf.data() + e * pslot, gq[e].data(), rows * rb);
    std::memcpy(pbuf.data() + e * pslot + pup_off, uq[e].data(), rows * rb);
  }
  std::vector<uint16_t> X((size_t)nasg * K);
  std::vector<uint32_t> order;  // row i of X is assignment order[i]
  auto *dTiles = static_cast<MoeTile *>(rt.alloc_host(NEXP * sizeof(MoeTile)));
  auto *dNT = static_cast<uint32_t *>(rt.alloc_host(4));
  for (uint32_t e = 0; e < NEXP; e++) {
    dTiles[e] = {e, e, (uint32_t)order.size(), groups[e].n};
    for (uint32_t j = 0; j < groups[e].n; j++) order.push_back(groups[e].asg[j]);
  }
  *dNT = NEXP;
  for (uint32_t i = 0; i < nasg; i++)
    for (uint32_t k = 0; k < K; k++) X[(size_t)i * K + k] = ggml_fp32_to_fp16(x[order[i] / PER][k]);
  auto *dW = static_cast<uint8_t *>(rt.alloc_device(card, pbuf.size() + QUANT_FILE_PAD));
  auto *dX = static_cast<uint16_t *>(rt.alloc_device(card, X.size() * 2));
  auto *dH = static_cast<uint16_t *>(rt.alloc_host(nout * 2));
  L.copy(dW, pbuf.data(), pbuf.size());
  L.copy(dX, X.data(), X.size() * 2);
  L.synchronize();
  for (bool limited : {false, true}) {
    const std::string mod = std::string("qgemm_iq4_xs_file") + (limited ? "_limit10" : "");
    std::fill(dH, dH + nout, 0x7e00);
    const uint32_t hsigma = 0;
    L.launch(lib.get(card, mod, "moe_gate_up_t"),
             {dW, pslot, dTiles, dNT, rows, bpr, pup_off, dX, K, K, dH, hsigma, grids}, rows / 64, NEXP, 1, 64);
    L.synchronize();
    auto h = refs(g16, u16, limited ? LIMIT : 0);
    std::vector<double> ref(nout);
    for (uint32_t i = 0; i < nasg; i++)
      for (uint32_t r = 0; r < rows; r++) ref[(size_t)i * rows + r] = h[(size_t)order[i] * rows + r];
    ck.check((mod + " moe_gate_up_t").c_str(), ref, dH, true, 2e-3);
  }

  // common.cl's swiglu on g8 and u8 as f32.
  const uint32_t n = (uint32_t)nout;
  auto *cg = static_cast<float *>(rt.alloc_host(nout * 4));
  auto *cu = static_cast<float *>(rt.alloc_host(nout * 4));
  std::vector<double> gf(nout), uf(nout);
  for (size_t i = 0; i < nout; i++) {
    cg[i] = (float)g8[i];
    cu[i] = (float)u8[i];
    gf[i] = cg[i];
    uf[i] = cu[i];
  }
  for (bool limited : {false, true}) {
    const std::string mod = std::string("common") + (limited ? "_limit10" : "");
    std::fill(hm, hm + nout, NAN);
    L.launch(lib.get(card, mod, "swiglu"), {cg, cu, hm, n}, (n + 63) / 64, 1, 1, 64);
    L.synchronize();
    ck.check((mod + " swiglu").c_str(), refs(gf, uf, limited ? LIMIT : 0), hm, false, 1e-5);
  }

  for (void *p : {(void *)cache, (void *)dG, (void *)dN, (void *)dA, (void *)hm, (void *)hg, (void *)dTiles,
                  (void *)dNT, (void *)dW, (void *)dX, (void *)dH, (void *)cg, (void *)cu, (void *)grids})
    rt.free(p);
  EXPECT(ck.failures == 0, "%d case(s) failed", ck.failures);
  std::printf("PASS\n");
  return 0;
}
