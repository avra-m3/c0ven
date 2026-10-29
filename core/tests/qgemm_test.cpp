// Level 3 (docs/TESTING.md): the prompt path's quantised products of kernels/quant/qgemm.cl, on real weights of the
// file, against ggml's CPU dequantisation of the same blocks multiplied in double precision by the same fp16
// activations. dequant_f16 for every type the dense weights use; moe_gate_up_t and moe_down_t for every expert type,
// and Q8_0 down on random weights when the file has none, in the runs layout (cached experts) and the file's (streamed
// experts), for card 1's half of an expert, so the slices start mid-row. Tolerance 2e-3 of the largest |reference|
// (TESTING.md, "Tolerances": fp16 products).
#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <set>

#include "ggml.h"
#include "load/gguf.h"
#include "load/runs.h"
#include "moe_tile.h"
#include "quant.h"
#include "test_util.h"
#include "ze/library.h"

using namespace engine;

namespace {

std::string lower(std::string s) {
  for (auto &c : s) c = (char)std::tolower(c);
  return s;
}

uint64_t align256(uint64_t x) { return (x + 255) & ~255ull; }

struct Checker {
  int failures = 0;
  void check(const char *what, const std::vector<double> &ref, const uint16_t *got) {
    double maxerr = 0, maxref = 0;
    for (size_t i = 0; i < ref.size(); i++) {
      maxref = std::max(maxref, std::fabs(ref[i]));
      double e = std::fabs(ref[i] - ggml_fp16_to_fp32(got[i]));
      maxerr = std::isnan(e) ? INFINITY : std::max(maxerr, e);
    }
    double rel = maxerr / maxref;
    bool ok = rel <= 2e-3;
    failures += !ok;
    std::printf("%-58s max error %.2e %s\n", what, rel, ok ? "ok" : "FAIL");
  }
};

}  // namespace

int main() {
  if (test::gpus() == 0) return test::no_gpu();
  std::string path = test::model_path();
  if (path.empty()) return test::skip("model not found under FLASH_MODEL_DIR");
  GgufModel model(path);
  ze::Runtime rt;
  ze::Library lib(rt);
  const uint32_t card = 0;
  auto *grids = static_cast<uint8_t *>(rt.alloc_device(card, QGRID_BYTES));
  ze::List L(rt, card, ze::List::immediate);
  L.copy(grids, codebooks().data(), QGRID_BYTES);
  L.synchronize();
  std::mt19937 rng(99);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  Checker ck;

  // dequant_f16 on the first 200 rows of a dense tensor of each quantised type the file has.
  std::set<uint32_t> dense_types;
  for (auto &[name, ti] : model.tensors())
    if (quant_block_values(ti.type) > 1 && name.find("exps") == std::string::npos &&
        name.find("per_layer") == std::string::npos)
      dense_types.insert(ti.type);
  for (uint32_t type : dense_types) {
    const GgufTensor *t = nullptr;
    for (auto &[name, ti] : model.tensors())
      if (ti.type == type && name.find("exps") == std::string::npos && name.find("per_layer") == std::string::npos) {
        t = &ti;
        break;
      }
    EXPECT(t, "no dense tensor of type %s", ggml_type_label(type));
    const uint32_t K = (uint32_t)t->shape[0], rows = 200, bpr = K / quant_block_values(type), units = K / 32;
    std::vector<uint8_t> runs(runs_bytes(type, (uint64_t)rows * bpr));
    to_runs(type, {{t->data, t->row_bytes(), 0, rows, 0, bpr}}, runs.data());
    std::vector<float> wf((size_t)rows * K);
    ggml_get_type_traits((ggml_type)type)->to_float(t->data, wf.data(), (int64_t)wf.size());
    auto *dW = static_cast<uint8_t *>(rt.alloc_device(card, runs.size()));
    auto *dO = static_cast<uint16_t *>(rt.alloc_host((size_t)rows * K * 2));
    L.copy(dW, runs.data(), runs.size());
    uint32_t zero = 0;
    L.launch(lib.get(card, "qgemm_" + lower(ggml_type_label(type)), "dequant_f16"),
             {dW, rows, bpr, zero, units, dO, K, grids}, (rows * units + 63) / 64, 1, 1, 64);
    L.synchronize();
    std::vector<double> ref(wf.begin(), wf.end());
    std::string what = std::string("dequant_f16 ") + ggml_type_label(type) + " " + t->name;
    ck.check(what.c_str(), ref, dO);
    rt.free(dW);
    rt.free(dO);
  }

  // The expert products: card 1's half of expert 7, 75 assignments in two tiles (64 + 11).
  const uint32_t E = 2560, F = 640, half = 320, f0 = 320, e = 7, nasg = 75;
  std::vector<uint16_t> X((size_t)nasg * E), Hh((size_t)nasg * half);
  std::vector<double> xf(X.size()), hf(Hh.size());
  for (size_t i = 0; i < X.size(); i++) {
    X[i] = ggml_fp32_to_fp16(normal(rng));
    xf[i] = ggml_fp16_to_fp32(X[i]);
  }
  for (size_t i = 0; i < Hh.size(); i++) {
    Hh[i] = ggml_fp32_to_fp16(normal(rng) * 0.1f);
    hf[i] = ggml_fp16_to_fp32(Hh[i]);
  }
  auto *dX = static_cast<uint16_t *>(rt.alloc_device(card, X.size() * 2));
  auto *dH = static_cast<uint16_t *>(rt.alloc_device(card, Hh.size() * 2));
  auto *dOut = static_cast<uint16_t *>(rt.alloc_host((size_t)nasg * E * 2));
  auto *dTiles = static_cast<MoeTile *>(rt.alloc_host(2 * sizeof(MoeTile)));
  auto *dN = static_cast<uint32_t *>(rt.alloc_host(4));
  dTiles[0] = {e, 0, 0, MOE_TILE};
  dTiles[1] = {e, 0, MOE_TILE, nasg - MOE_TILE};
  *dN = 2;
  // The expert kernels may take X and h with every 4 values permuted by sigma (shared/quant.h, quant_pairs_sigma),
  // and write h so; these are both orders.
  auto sigma = [](uint32_t i) { return (i & ~3u) | ((i & 1u) << 1) | ((i >> 1) & 1u); };
  std::vector<uint16_t> Xp(X.size()), Hp(Hh.size());
  for (size_t i = 0; i < X.size(); i++) Xp[i] = X[i - i % E + sigma((uint32_t)(i % E))];
  for (size_t i = 0; i < Hh.size(); i++) Hp[i] = Hh[i - i % half + sigma((uint32_t)(i % half))];
  auto *dXp = static_cast<uint16_t *>(rt.alloc_device(card, X.size() * 2));
  auto *dHp = static_cast<uint16_t *>(rt.alloc_device(card, Hh.size() * 2));
  L.copy(dX, X.data(), X.size() * 2);
  L.copy(dH, Hh.data(), Hh.size() * 2);
  L.copy(dXp, Xp.data(), X.size() * 2);
  L.copy(dHp, Hp.data(), Hh.size() * 2);
  L.synchronize();

  // One layer per gate type, and one per down type; Q8_0 down, which the IQ3_S file has none of, on a random expert
  // in the file's shape (2,560 rows of 640) when the file has none.
  std::vector<uint32_t> gate_layers, down_layers;
  std::vector<uint32_t> seen_g, seen_d;
  for (uint32_t l = 0; l < 48; l++) {
    uint32_t gt = model.tensor("blk." + std::to_string(l) + ".ffn_gate_exps.weight").type;
    uint32_t dt = model.tensor("blk." + std::to_string(l) + ".ffn_down_exps.weight").type;
    if (std::find(seen_g.begin(), seen_g.end(), gt) == seen_g.end()) seen_g.push_back(gt), gate_layers.push_back(l);
    if (std::find(seen_d.begin(), seen_d.end(), dt) == seen_d.end()) seen_d.push_back(dt), down_layers.push_back(l);
  }
  std::vector<uint8_t> random_down;
  if (std::find(seen_d.begin(), seen_d.end(), QT_Q8_0) == seen_d.end()) random_down = test::random_q8_0(E, F, 5);
  for (bool file : {false, true}) {
    for (uint32_t l : gate_layers) {
      const GgufTensor &g = model.tensor("blk." + std::to_string(l) + ".ffn_gate_exps.weight");
      const GgufTensor &u = model.tensor("blk." + std::to_string(l) + ".ffn_up_exps.weight");
      const uint32_t type = g.type, bpr = E / quant_block_values(type);
      const uint64_t eb = g.bytes / 512, rb = g.row_bytes();
      std::vector<uint8_t> buf;
      uint64_t up_off;
      if (file) {
        up_off = align256(half * rb);
        buf.assign(2 * up_off, 0);
        std::memcpy(buf.data(), g.data + e * eb + f0 * rb, half * rb);
        std::memcpy(buf.data() + up_off, u.data + e * eb + f0 * rb, half * rb);
      } else {
        uint64_t sb = runs_bytes(type, (uint64_t)half * bpr);
        up_off = align256(sb);
        buf.assign(2 * up_off, 0);
        to_runs(type, {{g.data + e * eb, rb, f0, half, 0, bpr}}, buf.data());
        to_runs(type, {{u.data + e * eb, rb, f0, half, 0, bpr}}, buf.data() + up_off);
      }
      std::vector<float> gw((size_t)F * E), uw((size_t)F * E);
      ggml_get_type_traits((ggml_type)type)->to_float(g.data + e * eb, gw.data(), (int64_t)gw.size());
      ggml_get_type_traits((ggml_type)type)->to_float(u.data + e * eb, uw.data(), (int64_t)uw.size());
      std::vector<double> ref((size_t)nasg * half);
      for (uint32_t a = 0; a < nasg; a++)
        for (uint32_t r = 0; r < half; r++) {
          double gs = 0, us = 0;
          for (uint32_t i = 0; i < E; i++) {
            gs += (double)gw[(size_t)(f0 + r) * E + i] * xf[(size_t)a * E + i];
            us += (double)uw[(size_t)(f0 + r) * E + i] * xf[(size_t)a * E + i];
          }
          ref[(size_t)a * half + r] = gs / (1 + std::exp(-gs)) * us;
        }
      auto *dW = static_cast<uint8_t *>(rt.alloc_device(card, buf.size()));
      L.copy(dW, buf.data(), buf.size());
      uint64_t slot_bytes = buf.size();
      std::string mod = "qgemm_" + lower(ggml_type_label(type)) + (file ? "_file" : "");
      const uint32_t sig = quant_pairs_sigma(type);
      for (uint32_t hsig : {0u, 1u}) {  // h in its own order, or in sigma order for a down type that takes it
        std::fill(dOut, dOut + (size_t)nasg * half, 0x7e00);
        L.launch(lib.get(card, mod, "moe_gate_up_t"),
                 {dW, slot_bytes, dTiles, dN, half, bpr, up_off, sig ? dXp : dX, E, E, dOut, hsig, grids}, half / 64, 2, 1,
                 64);
        L.synchronize();
        std::vector<uint16_t> back((size_t)nasg * half);
        for (size_t i = 0; i < back.size(); i++)
          back[i] = dOut[i - i % half + (hsig ? sigma((uint32_t)(i % half)) : (uint32_t)(i % half))];
        std::string what = std::string("moe_gate_up_t ") + ggml_type_label(type) + (file ? " file" : " runs") +
                           " blk." + std::to_string(l) + (hsig ? " h sigma" : "");
        ck.check(what.c_str(), ref, back.data());
      }
      rt.free(dW);
    }
    std::vector<int32_t> downs(down_layers.begin(), down_layers.end());
    if (!random_down.empty()) downs.push_back(-1);  // the random Q8_0 expert
    for (int32_t l : downs) {
      const GgufTensor *d = l >= 0 ? &model.tensor("blk." + std::to_string(l) + ".ffn_down_exps.weight") : nullptr;
      const uint32_t type = d ? d->type : QT_Q8_0, qk = quant_block_values(type);
      const uint64_t eb = d ? d->bytes / 512 : random_down.size(), rb = eb / E;
      const uint8_t *ed = d ? d->data + e * eb : random_down.data();
      std::vector<uint8_t> buf;
      uint32_t bpr, b0;
      if (file) {
        buf.assign(ed, ed + eb);
        bpr = F / qk;
        b0 = f0 / qk;
      } else {
        bpr = half / qk;
        b0 = 0;
        buf.assign(runs_bytes(type, (uint64_t)E * bpr), 0);
        to_runs(type, {{ed, rb, 0, E, f0 / qk, bpr}}, buf.data());
      }
      std::vector<float> dw((size_t)E * F);
      ggml_get_type_traits((ggml_type)type)->to_float(ed, dw.data(), (int64_t)dw.size());
      std::vector<double> ref((size_t)nasg * E);
      for (uint32_t a = 0; a < nasg; a++)
        for (uint32_t r = 0; r < E; r++) {
          double s = 0;
          for (uint32_t i = 0; i < half; i++) s += (double)dw[(size_t)r * F + f0 + i] * hf[(size_t)a * half + i];
          ref[(size_t)a * E + r] = s;
        }
      auto *dW = static_cast<uint8_t *>(rt.alloc_device(card, buf.size()));
      L.copy(dW, buf.data(), buf.size());
      std::fill(dOut, dOut + (size_t)nasg * E, 0x7e00);
      uint64_t slot_bytes = buf.size(), down_off = 0;
      std::string mod = "qgemm_" + lower(ggml_type_label(type)) + (file ? "_file" : "");
      L.launch(lib.get(card, mod, "moe_down_t"),
               {dW, slot_bytes, down_off, dTiles, dN, E, bpr, b0, quant_pairs_sigma(type) ? dHp : dH, half, dOut, grids},
               E / 64, 2, 1, 64);
      L.synchronize();
      std::string what = std::string("moe_down_t ") + ggml_type_label(type) + (file ? " file" : " runs") +
                         (l >= 0 ? " blk." + std::to_string(l) : std::string(" random"));
      ck.check(what.c_str(), ref, dOut);
      rt.free(dW);
    }
  }
  EXPECT(ck.failures == 0, "%d case(s) over tolerance", ck.failures);
  std::printf("PASS\n");
  return 0;
}
