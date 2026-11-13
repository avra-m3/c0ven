// dot_rows_for's functions against ggml's vec_dot, bit for bit, on deepseek41's own weights: for each routed-expert
// type the file uses, rows of gate, up and down of the layers that hold it, against 1 to 9 activation rows. Gate and up
// meet the token embeddings of real prompts; down meets the hidden rows an expert makes from them. Needs DS41_MODEL, the
// file's first shard; skips (77) without it and on a CPU lacking what the tests' ggml build uses (cmake/ggml.cmake).
#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "cpu/features.h"
#include "cpu/multirow.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "load/gguf.h"
#include "models/deepseek41/layout.h"
#include "test_util.h"

using namespace engine;

namespace {

constexpr int kRowsMax = 9, kRowsTested = 96;
// Prompts' tokens: a start of text, a chat turn's markers, and words and code from the vocabulary.
constexpr int32_t kTokens[kRowsMax] = {0, 128803, 9226, 4117, 5466, 7379, 21015, 65106, 128804};

const ggml_type_traits_cpu *cpu_traits(uint32_t type) { return ggml_get_type_traits_cpu((ggml_type)type); }

// One layer's expert tensors of a kind ("gate", "up" or "down").
const GgufTensor &experts(const GgufModel &g, uint32_t layer, const char *kind) {
  return g.tensor("blk." + std::to_string(layer) + ".ffn_" + kind + "_exps.weight");
}

// Token embedding t as floats.
std::vector<float> embedding(const GgufModel &g, int32_t t) {
  const GgufTensor &te = g.tensor("token_embd.weight");
  std::vector<float> v(te.shape.at(0));
  ggml_get_type_traits((ggml_type)te.type)->to_float(te.data + (uint64_t)t * te.row_bytes(), v.data(), (int64_t)v.size());
  return v;
}

// n values of v quantised for the dots of type: the rows are the type's vec_dot_type.
std::vector<uint8_t> quantised(uint32_t type, const float *v, int n) {
  const auto *q = cpu_traits(cpu_traits(type)->vec_dot_type);
  std::vector<uint8_t> out(ggml_row_size(cpu_traits(type)->vec_dot_type, n));
  q->from_float(v, out.data(), n);
  return out;
}

// The rows kRowsTested of tensor t's expert e against the activation rows ys (K values each), each count of rows.
uint32_t wrong_sums(const GgufTensor &t, int e, int K, const std::vector<std::vector<uint8_t>> &ys, DotRows dr) {
  const auto *tr = cpu_traits(t.type);
  const uint64_t rows = t.shape.at(1), rb = t.row_bytes();
  EXPECT(t.shape.at(0) == (uint64_t)K, "%s: rows of %llu values, not %d", t.name.c_str(), (unsigned long long)t.shape.at(0), K);
  const void *yp[kRowsMax];
  for (int r = 0; r < kRowsMax; r++) yp[r] = ys[r].data();
  uint32_t wrong = 0;
  for (int i = 0; i < kRowsTested; i++) {
    // The first rows of the expert and rows spread over it.
    const uint64_t row = i < 32 ? (uint64_t)i : (uint64_t)(i - 32) * (rows - 32) / (kRowsTested - 32) + 32;
    const uint8_t *w = t.data + ((uint64_t)e * rows + row) * rb;
    float want[kRowsMax];
    for (int r = 0; r < kRowsMax; r++) tr->vec_dot(K, &want[r], 0, w, 0, yp[r], 0, 1);
    for (int R = 1; R <= kRowsMax; R++) {
      float got[kRowsMax];
      dr(K, w, yp, R, got);
      for (int r = 0; r < R; r++) wrong += std::memcmp(&want[r], &got[r], 4) != 0;
    }
  }
  return wrong;
}

}  // namespace

int main() {
  if (const std::string missing = missing_cpu_features(ENGINE_CPU_FEATURES); !missing.empty())
    return test::skip("the CPU lacks " + missing + ", which the tests' ggml build uses");
  const char *path = std::getenv("DS41_MODEL");
  if (!path) return test::skip("DS41_MODEL is not set: the first shard of DeepSeek-V4.1-Flash-1017's file");
  if (!dot_rows_for(GGML_TYPE_IQ2_S)) return test::skip("no AVX2, FMA or F16C");
  ggml_cpu_init();
  const GgufModel g(path);
  constexpr uint32_t E = D41_EMBD, FF = D41_FF;

  std::vector<std::vector<float>> x(kRowsMax);
  for (int r = 0; r < kRowsMax; r++) x[r] = embedding(g, kTokens[r]);

  // The first layer of each pair of gate-and-up and down types, and the layers' types as a whole.
  std::map<std::pair<uint32_t, uint32_t>, uint32_t> first;
  std::set<uint32_t> in_file;
  for (uint32_t l = 0; l < D41_LAYERS; l++) {
    const uint32_t gt = experts(g, l, "gate").type, dt = experts(g, l, "down").type;
    EXPECT(experts(g, l, "up").type == gt, "layer %u: up is not gate's type", l);
    first.emplace(std::make_pair(gt, dt), l);
    in_file.insert(gt);
    in_file.insert(dt);
  }
  // The types a pair tested, with the role.
  std::set<uint32_t> tested;
  for (const auto &[types, l] : first) {
    const auto [gt, dt] = types;
    const DotRows gr = dot_rows_for(gt), dr = dot_rows_for(dt);
    const GgufTensor &gate = experts(g, l, "gate"), &up = experts(g, l, "up"), &down = experts(g, l, "down");
    std::vector<std::vector<uint8_t>> xg(kRowsMax), hd(kRowsMax);
    uint32_t wrong = 0;
    for (int r = 0; r < kRowsMax; r++) xg[r] = quantised(gt, x[r].data(), (int)E);
    for (int e : {0, 131, 383}) {
      if (gr) wrong += wrong_sums(gate, e, (int)E, xg, gr) + wrong_sums(up, e, (int)E, xg, gr);
      if (!dr) continue;
      // Down's input: silu(gate) * up of this expert for each embedding, as the model makes it, quantised for down's dots.
      for (int r = 0; r < kRowsMax; r++) {
        std::vector<float> h(FF);
        for (uint32_t i = 0; i < FF; i++) {
          float gv, uv;
          cpu_traits(gt)->vec_dot((int)E, &gv, 0, gate.data + ((uint64_t)e * FF + i) * gate.row_bytes(), 0, xg[r].data(), 0, 1);
          cpu_traits(gt)->vec_dot((int)E, &uv, 0, up.data + ((uint64_t)e * FF + i) * up.row_bytes(), 0, xg[r].data(), 0, 1);
          gv = std::min(gv, D41_SWIGLU_LIMIT);
          uv = std::clamp(uv, -D41_SWIGLU_LIMIT, D41_SWIGLU_LIMIT);
          h[i] = gv / (1.0f + std::exp(-gv)) * uv;
        }
        hd[r] = quantised(dt, h.data(), (int)FF);
      }
      wrong += wrong_sums(down, e, (int)FF, hd, dr);
    }
    if (gr) tested.insert(gt);
    if (dr) tested.insert(dt);
    EXPECT(wrong == 0, "layer %u (gate and up %s, down %s): %u sums differ from vec_dot", l, ggml_type_name((ggml_type)gt),
           ggml_type_name((ggml_type)dt), wrong);
    std::printf("layer %u, gate and up %s%s, down %s%s: 1 to %d rows, same bits as vec_dot\n", l, ggml_type_name((ggml_type)gt),
                gr ? "" : " (no multi-row dot)", ggml_type_name((ggml_type)dt), dr ? "" : " (no multi-row dot)", kRowsMax);
  }
  // Every expert type of the file has a multi-row dot, and each ran on its real weights.
  for (uint32_t t : in_file) EXPECT(tested.count(t), "%s: the file uses it and it was not tested", ggml_type_name((ggml_type)t));
  std::printf("ds41_multirow_test: ok\n");
  return 0;
}
