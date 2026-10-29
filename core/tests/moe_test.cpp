// Level 3 (docs/TESTING.md): the decode step's grouped expert products of kernels/quant/moe.cl against matvec.cl's
// moe_gate_up_g and moe_down_g, which they must equal bit for bit, on real experts in the slot layout; gate and up's
// a8 output against quantize_a8 of the reference. Every expert type the file holds, for groups of up to 1, 2, 3, 4,
// 6 and 8 assignments of mixed sizes; down's rows end inside a work-group.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <random>

#include "ggml.h"
#include "load/gguf.h"
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

std::string lower(std::string s) {
  for (auto &c : s) c = (char)std::tolower(c);
  return s;
}

uint64_t align256(uint64_t x) { return (x + 255) & ~255ull; }

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
  ze::List up(rt, card, ze::List::immediate);
  up.copy(grids, codebooks().data(), QGRID_BYTES);
  up.synchronize();

  // 16 experts; each assignment of a token (10 a token) picks one, distinct within the token. Gate and up: a card's
  // 320 rows of 2,560 values; down: 2,500 rows of a card's 320.
  const uint32_t E = 16, PER = 10, EMBD = 2560, FF = 320, GU_ROWS = 320, DOWN_ROWS = 2500;
  std::mt19937 rng(4321);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  int failures = 0;
  for (bool down : {false, true}) {
    std::vector<uint32_t> types = down ? std::vector<uint32_t>{QT_IQ4_NL, QT_Q2_0, QT_Q8_0}
                                       : std::vector<uint32_t>{QT_IQ2_S, QT_IQ3_XXS, QT_IQ3_S, QT_IQ4_XS, QT_IQ2_XXS,
                                                                  QT_IQ2_XS, QT_IQ1_M};
    for (uint32_t type : types) {
      // The first layer whose experts are of the type.
      const GgufTensor *g = nullptr, *u = nullptr;
      for (uint32_t l = 0; !g && model.tensors().count("blk." + std::to_string(l) + ".ffn_gate_exps.weight"); l++) {
        std::string b = "blk." + std::to_string(l) + ".";
        const GgufTensor &t = model.tensor(b + (down ? "ffn_down_exps.weight" : "ffn_gate_exps.weight"));
        if (t.type == type) {
          g = &t;
          u = down ? nullptr : &model.tensor(b + "ffn_up_exps.weight");
        }
      }
      if (!g) {  // moe.cl's types the file does not have
        std::printf("%-5s %-8s no experts of the type in this file, skipped\n", down ? "down" : "gate",
                    ggml_type_label(type));
        continue;
      }
      const uint32_t rows = down ? DOWN_ROWS : GU_ROWS, K = down ? FF : EMBD;
      const uint32_t bpr = K / quant_block_values(type);
      const uint64_t n = (uint64_t)rows * bpr, mat = align256(runs_bytes(type, n));
      const uint64_t slot_bytes = down ? mat : 2 * mat, up_off = mat;
      std::vector<uint8_t> host(E * slot_bytes);
      const uint64_t per_expert = g->bytes / g->shape[2];
      for (uint32_t e = 0; e < E; e++) {
        uint8_t *s = host.data() + e * slot_bytes;
        to_runs(type, {{g->data + e * per_expert, g->row_bytes(), 0, rows, 0, bpr}}, s);
        if (u) to_runs(type, {{u->data + e * per_expert, u->row_bytes(), 0, rows, 0, bpr}}, s + up_off);
      }
      auto *cache = static_cast<uint8_t *>(rt.alloc_device(card, host.size()));
      up.copy(cache, host.data(), host.size());

      for (uint32_t A : {1u, 2u, 3u, 4u, 6u, 8u}) {
        // A tokens' assignments, grouped by expert in order of first use (as moe_group groups them).
        const uint32_t nasg = A * PER;
        std::vector<MoeGroup> groups;
        std::map<uint32_t, size_t> of;
        for (uint32_t t = 0; t < A; t++) {
          std::vector<uint32_t> pick(E);
          for (uint32_t e = 0; e < E; e++) pick[e] = e;
          std::shuffle(pick.begin(), pick.end(), rng);
          for (uint32_t j = 0; j < PER; j++) {
            uint32_t e = pick[j];
            if (!of.count(e)) {
              of[e] = groups.size();
              MoeGroup G;
              std::memset(&G, 0, sizeof G);
              G.slot = (int32_t)e;
              groups.push_back(G);
            }
            MoeGroup &G = groups[of[e]];
            G.asg[G.n++] = t * PER + j;
          }
        }
        uint32_t ng = (uint32_t)groups.size();
        // Activations: a token's x for gate and up, an assignment's h for down.
        std::vector<uint8_t> act;
        for (uint32_t i = 0; i < (down ? nasg : A); i++) {
          std::vector<float> x(K);
          for (auto &v : x) v = normal(rng) * (down ? 0.1f : 1.0f);
          auto a = quantize_a8(x);
          act.insert(act.end(), a.begin(), a.end());
        }
        const uint32_t stride = K + K / 8;
        auto *dG = static_cast<MoeGroup *>(rt.alloc_device(card, nasg * sizeof(MoeGroup)));
        auto *dN = static_cast<uint32_t *>(rt.alloc_device(card, 4));
        auto *dA = static_cast<uint8_t *>(rt.alloc_device(card, act.size()));
        const size_t nout = (size_t)nasg * rows;
        auto *ref = static_cast<float *>(rt.alloc_host(nout * 4));
        auto *got = static_cast<float *>(rt.alloc_host(nout * 4));
        const size_t a8 = (size_t)nasg * (rows + rows / 8);
        auto *ref_a8 = static_cast<uint8_t *>(rt.alloc_host(a8));
        auto *got_a8 = static_cast<uint8_t *>(rt.alloc_host(a8));
        std::memset(ref_a8, 0, a8);
        std::memset(got_a8, 0, a8);
        up.copy(dG, groups.data(), ng * sizeof(MoeGroup));
        up.copy(dN, &ng, 4);
        up.copy(dA, act.data(), act.size());
        up.synchronize();
        std::fill(ref, ref + nout, NAN);
        std::fill(got, got + nout, NAN);

        std::string mv = lower(std::string("matvec_") + ggml_type_label(type) + (down ? "_l4" : "_l16"));
        std::string mm = lower(std::string("moe_") + ggml_type_label(type) + "_g" + std::to_string(A));
        if (down) {
          uint32_t lx = 16 * MOE_WG_ROWS / MOE_DOWN_ROWS(type, A);
          uint64_t off = 0;
          up.launch(lib.get(card, mv, "moe_down_g"),
                    {cache, slot_bytes, dG, dN, rows, bpr, off, dA, K, stride, ref, grids}, (rows + 15) / 16, nasg,
                    1, 64);
          up.launch(lib.get(card, mm, "moe_down_g"),
                    {cache, slot_bytes, dG, dN, rows, bpr, off, dA, K, stride, got, grids},
                    (rows + MOE_WG_ROWS - 1) / MOE_WG_ROWS, nasg, 1, lx);
        } else {
          uint32_t lx = 16 * MOE_WG_ROWS / MOE_GU_ROWS(A);
          up.launch(lib.get(card, mv, "moe_gate_up_g"),
                    {cache, slot_bytes, dG, dN, rows, bpr, up_off, dA, K, stride, PER, ref, grids}, (rows + 3) / 4,
                    nasg, 1, 64);
          up.launch(lib.get(card, mm, "moe_gate_up_g"),
                    {cache, slot_bytes, dG, dN, rows, bpr, up_off, dA, K, stride, PER, got, grids, got_a8},
                    (rows + MOE_WG_ROWS - 1) / MOE_WG_ROWS, nasg, 1, lx);
          up.launch(lib.get(card, "common", "quantize_a8"), {ref, rows, rows, nasg, ref_a8}, (rows / 32 * nasg + 3) / 4,
                    1, 1, 64);
        }
        up.synchronize();
        size_t differ = 0, nonfinite = 0;
        for (size_t i = 0; i < nout; i++) {
          differ += std::memcmp(&ref[i], &got[i], 4) != 0;
          nonfinite += !std::isfinite(ref[i]);
        }
        if (!down) differ += std::memcmp(ref_a8, got_a8, a8) != 0;
        uint32_t largest = 0;
        for (auto &G : groups) largest = std::max(largest, G.n);
        bool ok = differ == 0 && nonfinite == 0;
        failures += !ok;
        std::printf("%-5s %-8s groups of up to %u: %2u groups, largest %u, %zu values: %zu differ, %zu not finite %s\n",
                    down ? "down" : "gate", ggml_type_label(type), A, ng, largest, nout, differ, nonfinite,
                    ok ? "ok" : "FAIL");
        rt.free(dG);
        rt.free(dN);
        rt.free(dA);
        rt.free(ref);
        rt.free(got);
        rt.free(ref_a8);
        rt.free(got_a8);
      }
      rt.free(cache);
    }
  }
  EXPECT(failures == 0, "%d case(s) differ", failures);
  std::printf("PASS\n");
  return 0;
}
