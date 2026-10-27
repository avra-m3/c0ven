// The DFlash 2 drafter (core/models/qwen4exp/dflash.cpp) against a float CPU version of the same model, from its
// tensor files: the main model's features of 2,056 positions (a reference prompt, repeated) as context, then a block
// of 8 rows, each of which sees the last DF_WINDOW positions. Gate: each row's final hidden state within 2% (relative
// RMS) of the CPU's; the cards keep activations in bf16. Then two slots' blocks in one pass give each slot exactly the
// candidates it gets alone.
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <thread>

#include "ggml.h"
#include "load/gguf.h"
#include "models/qwen4exp/layout.h"
#include "models/qwen4exp/model.h"
#include "test_util.h"

using namespace engine;

namespace {

constexpr uint32_t E = Q_EMBD;

std::vector<int32_t> tokens_of(const std::string &line) {
  std::vector<int32_t> v;
  size_t p = line.find("\"tokens\":");
  if (p == std::string::npos) return v;
  p = line.find('[', p) + 1;
  while (line[p] != ']') {
    size_t q;
    v.push_back(std::stoi(line.substr(p), &q));
    p += q;
    if (line[p] == ',') p++;
    while (line[p] == ' ') p++;
  }
  return v;
}

float bf16f(uint16_t h) {
  uint32_t u = (uint32_t)h << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

std::vector<float> tensor(const std::string &dir, const std::string &name, size_t n) {
  std::ifstream f(dir + "/" + name, std::ios::binary);
  EXPECT(f, "cannot open %s", name.c_str());
  std::vector<uint16_t> h(n);
  f.read(reinterpret_cast<char *>(h.data()), (std::streamsize)n * 2);
  EXPECT((size_t)f.gcount() == n * 2, "%s is short", name.c_str());
  std::vector<float> v(n);
  for (size_t i = 0; i < n; i++) v[i] = bf16f(h[i]);
  return v;
}

// f(i) for i < n, over the host's threads.
template <class F>
void each(size_t n, F f) {
  const size_t T = std::max(1u, std::thread::hardware_concurrency());
  std::vector<std::thread> th;
  for (size_t t = 0; t < T; t++)
    th.emplace_back([&, t] {
      for (size_t i = t; i < n; i += T) f(i);
    });
  for (auto &x : th) x.join();
}

std::vector<float> matvec(const std::vector<float> &W, const float *x, size_t rows, size_t K) {
  std::vector<float> y(rows);
  for (size_t r = 0; r < rows; r++) {
    double a = 0;
    for (size_t k = 0; k < K; k++) a += (double)W[r * K + k] * x[k];
    y[r] = (float)a;
  }
  return y;
}

void rms_norm(float *x, const float *w, size_t n) {
  double ss = 0;
  for (size_t i = 0; i < n; i++) ss += (double)x[i] * x[i];
  const float rs = 1.0f / std::sqrt((float)(ss / n) + DF_EPS);
  for (size_t i = 0; i < n; i++) x[i] = x[i] * rs * w[i];
}

void rope(float *x, uint32_t pos) {
  for (uint32_t i = 0; i < DF_D / 2; i++) {
    const double th = pos * std::pow((double)DF_ROPE_BASE, -2.0 * i / DF_D);
    const float c = (float)std::cos(th), s = (float)std::sin(th), a = x[i], b = x[i + DF_D / 2];
    x[i] = a * c - b * s;
    x[i + DF_D / 2] = a * s + b * c;
  }
}

// The convolution's set (0 before a sublayer, 1 after it) over a block's rows x[t][E], dyn rows of DF_KP.
std::vector<float> conv(const std::vector<float> &x, const std::vector<float> &dyn, const std::vector<float> &base,
                        uint32_t set, uint32_t n) {
  std::vector<float> y((size_t)n * E);
  for (uint32_t t = 0; t < n; t++)
    for (uint32_t ch = 0; ch < E; ch++) {
      const float *d = &dyn[(size_t)t * DF_KP + set * (DF_KP / 2)];
      const uint32_t g = ch / DF_GROUP;
      float v = (base[(2 * set) * E + ch] + d[g]) * x[(size_t)t * E + ch];
      if (t > 0) v += (base[(2 * set + 1) * E + ch] + d[DF_KP / 4 + g]) * x[(size_t)(t - 1) * E + ch];
      y[(size_t)t * E + ch] = v;
    }
  return y;
}

// The CPU's final hidden rows of a block of n rows at positions pos0.., embedded as emb, over the context's features
// feat (positions 0 .. feat.size() - 1).
std::vector<float> reference(const std::string &dir, const std::vector<std::vector<float>> &feat,
                             std::vector<float> h, uint32_t pos0, uint32_t n) {
  const size_t nc = feat.size(), kvd = (size_t)DF_KV * DF_D, qd = (size_t)DF_H * DF_D;
  auto fc = tensor(dir, "fc.weight", (size_t)E * DF_FEAT);
  auto hnorm = tensor(dir, "hidden_norm.weight", E);
  std::vector<std::vector<float>> ctx(nc);
  each(nc, [&](size_t p) {
    ctx[p] = matvec(fc, feat[p].data(), E, DF_FEAT);
    rms_norm(ctx[p].data(), hnorm.data(), E);
  });
  fc.clear();
  for (uint32_t l = 0; l < DF_LAYERS; l++) {
    const std::string p = "layers." + std::to_string(l) + ".";
    auto in_norm = tensor(dir, p + "input_layernorm.weight", E), post_norm = tensor(dir, p + "post_attention_layernorm.weight", E);
    auto qn = tensor(dir, p + "self_attn.q_norm.weight", DF_D), kn = tensor(dir, p + "self_attn.k_norm.weight", DF_D);
    auto q = tensor(dir, p + "self_attn.q_proj.weight", qd * E), k = tensor(dir, p + "self_attn.k_proj.weight", kvd * E);
    auto v = tensor(dir, p + "self_attn.v_proj.weight", kvd * E), o = tensor(dir, p + "self_attn.o_proj.weight", E * qd);
    auto abase = tensor(dir, p + "attention_conv.base_kernel", 4 * E);
    auto akp = tensor(dir, p + "attention_conv.kernel_projection.weight", (size_t)DF_KP * E);
    auto mbase = tensor(dir, p + "mlp_conv.base_kernel", 4 * E);
    auto mkp = tensor(dir, p + "mlp_conv.kernel_projection.weight", (size_t)DF_KP * E);
    auto gate = tensor(dir, p + "mlp.gate_proj.weight", (size_t)DF_FF * E);
    auto up = tensor(dir, p + "mlp.up_proj.weight", (size_t)DF_FF * E);
    auto down = tensor(dir, p + "mlp.down_proj.weight", (size_t)E * DF_FF);
    // Keys and values: the context's (positions 0 ..), then the block's.
    std::vector<std::vector<float>> K(nc + n), V(nc + n);
    auto keys = [&](size_t j, const float *x, uint32_t pos) {
      auto kk = matvec(k, x, kvd, E);
      for (uint32_t hh = 0; hh < DF_KV; hh++) {
        rms_norm(&kk[hh * DF_D], kn.data(), DF_D);
        rope(&kk[hh * DF_D], pos);
      }
      K[j] = kk;
      V[j] = matvec(v, x, kvd, E);
    };
    each(nc, [&](size_t c) { keys(c, ctx[c].data(), (uint32_t)c); });
    // Attention.
    std::vector<float> x = h, dyn((size_t)n * DF_KP);
    for (uint32_t t = 0; t < n; t++) {
      rms_norm(&x[(size_t)t * E], in_norm.data(), E);
      auto d = matvec(akp, &x[(size_t)t * E], DF_KP, E);
      std::copy(d.begin(), d.end(), dyn.begin() + (size_t)t * DF_KP);
    }
    auto xc = conv(x, dyn, abase, 0, n);
    std::vector<std::vector<float>> Q;
    for (uint32_t t = 0; t < n; t++) {
      auto qq = matvec(q, &xc[(size_t)t * E], qd, E);
      for (uint32_t hh = 0; hh < DF_H; hh++) {
        rms_norm(&qq[hh * DF_D], qn.data(), DF_D);
        rope(&qq[hh * DF_D], pos0 + t);
      }
      Q.push_back(qq);
      keys(nc + t, &xc[(size_t)t * E], pos0 + t);
    }
    std::vector<float> a((size_t)n * E);
    for (uint32_t t = 0; t < n; t++) {
      std::vector<float> att(qd);
      for (uint32_t hh = 0; hh < DF_H; hh++) {
        const uint32_t kh = hh / (DF_H / DF_KV);
        // The row sees positions within DF_WINDOW of its own (the context's index is its position).
        const size_t lo = pos0 + t + 1 > DF_WINDOW ? pos0 + t + 1 - DF_WINDOW : 0;
        std::vector<double> s(K.size(), -1e300);
        double mx = -1e30, sum = 0;
        for (size_t j = lo; j < K.size(); j++) {
          double dt = 0;
          for (uint32_t d = 0; d < DF_D; d++) dt += (double)Q[t][hh * DF_D + d] * K[j][kh * DF_D + d];
          s[j] = dt / std::sqrt((double)DF_D);
          mx = std::max(mx, s[j]);
        }
        for (auto &e : s) sum += (e = std::exp(e - mx));
        for (uint32_t d = 0; d < DF_D; d++) {
          double acc = 0;
          for (size_t j = lo; j < K.size(); j++) acc += s[j] * V[j][kh * DF_D + d];
          att[hh * DF_D + d] = (float)(acc / sum);
        }
      }
      auto y = matvec(o, att.data(), E, qd);
      std::copy(y.begin(), y.end(), a.begin() + (size_t)t * E);
    }
    auto y = conv(a, dyn, abase, 1, n);
    for (size_t i = 0; i < h.size(); i++) h[i] += y[i];
    // The MLP.
    x = h;
    for (uint32_t t = 0; t < n; t++) {
      rms_norm(&x[(size_t)t * E], post_norm.data(), E);
      auto d = matvec(mkp, &x[(size_t)t * E], DF_KP, E);
      std::copy(d.begin(), d.end(), dyn.begin() + (size_t)t * DF_KP);
    }
    xc = conv(x, dyn, mbase, 0, n);
    std::vector<float> m((size_t)n * E);
    for (uint32_t t = 0; t < n; t++) {
      auto g = matvec(gate, &xc[(size_t)t * E], DF_FF, E), u = matvec(up, &xc[(size_t)t * E], DF_FF, E);
      for (uint32_t i = 0; i < DF_FF; i++) g[i] = g[i] / (1.0f + std::exp(-g[i])) * u[i];
      auto r = matvec(down, g.data(), E, DF_FF);
      std::copy(r.begin(), r.end(), m.begin() + (size_t)t * E);
    }
    y = conv(m, dyn, mbase, 1, n);
    for (size_t i = 0; i < h.size(); i++) h[i] += y[i];
    // The context's keys stay; the block's are made again from the next layer's input.
  }
  auto norm = tensor(dir, "norm.weight", E);
  for (uint32_t t = 0; t < n; t++) rms_norm(&h[(size_t)t * E], norm.data(), E);
  return h;
}

bool same(const std::vector<qwen4exp::Candidate> &a, const std::vector<qwen4exp::Candidate> &b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); i++)
    if (a[i].token != b[i].token || std::memcmp(&a[i].logit, &b[i].logit, 4) != 0) return false;
  return true;
}

}  // namespace

int main(int argc, char **argv) {
  if (test::gpus() == 0) return test::no_gpu();
  std::string path = test::model_path();
  if (path.empty()) return test::skip("model not found under FLASH_MODEL_DIR");
  const char *dd = std::getenv("FLASH_DFLASH_DIR");
  const std::string dir = dd ? dd : "/models/dflash";
  if (access((dir + "/dflash-manifest.json").c_str(), R_OK) != 0) return test::skip("no DFlash 2 files in " + dir);
  EXPECT(argc >= 2, "usage: dflash_test <reference dir>");
  const uint32_t N = DF_WINDOW + 8, B = 8;
  std::vector<int32_t> toks;
  {
    std::ifstream pf(std::string(argv[1]) + "/prompts.jsonl");
    std::string line;
    while (toks.size() < 64 && std::getline(pf, line)) toks = tokens_of(line);
  }
  EXPECT(toks.size() >= 64, "no reference prompt of 64 tokens");
  for (size_t i = toks.size(); i <= N; i++) toks.push_back(toks[i % 64]);
  qwen4exp::Options opt;
  opt.slots = 2;
  opt.max_ctx = 4096;
  opt.prompt_chunk = 0;
  opt.mtp_dir = dir;
  qwen4exp::Model model(path, opt);
  EXPECT(model.dflash(), "the drafter is not DFlash 2");

  // The prompt in steps of 8 rows, each step's features as the context, for both slots.
  std::vector<std::vector<float>> feat;
  for (uint32_t s = 0; s < 2; s++) {
    model.reset_slot(s);
    model.draft_start(s, 0);
    for (uint32_t i = 0; i < N; i += 8) {
      std::vector<qwen4exp::StepIn> rows;
      for (uint32_t k = i; k < i + 8; k++)
        rows.push_back({s, k, toks[k], k ? toks[k - 1] : qwen4exp::kNoToken, k > 1 ? toks[k - 2] : qwen4exp::kNoToken});
      model.step(rows.data(), 8);
      model.df_pass(rows.data(), 8, nullptr, 0);
      for (uint32_t r = 0; s == 0 && r < 8; r++) {
        auto f = model.df_features(r);
        feat.emplace_back(f.size());
        for (size_t j = 0; j < f.size(); j++) feat.back()[j] = bf16f(f[j]);
      }
    }
  }

  // One block of 8 on slot 0 against the CPU.
  qwen4exp::Model::DraftBlock blk{0, N, toks[N], B};
  auto alone8 = model.df_pass(nullptr, 0, &blk, 1);
  EXPECT(alone8.size() == 1 && alone8[0].size() == B - 1, "a block of 8 gives 7 drafts' candidates");
  auto hid = model.df_hidden();
  std::vector<float> emb((size_t)B * E);
  {
    GgufModel g(path);
    const GgufTensor &te = g.tensor("token_embd.weight");
    auto to_float = ggml_get_type_traits((ggml_type)te.type)->to_float;
    for (uint32_t t = 0; t < B; t++)
      to_float(te.data + (uint64_t)(t ? DF_MASK : toks[N]) * te.row_bytes(), &emb[(size_t)t * E], E);
  }
  auto ref = reference(dir, feat, emb, N, B);
  int failures = 0;
  for (uint32_t t = 0; t < B; t++) {
    double d2 = 0, r2 = 0;
    for (uint32_t i = 0; i < E; i++) {
      const double d = hid[(size_t)t * E + i] - ref[(size_t)t * E + i];
      d2 += d * d;
      r2 += (double)ref[(size_t)t * E + i] * ref[(size_t)t * E + i];
    }
    const double rel = std::sqrt(d2 / r2);
    std::printf("row %u: relative RMS difference %.4f\n", t, rel);
    failures += !(rel < 0.02);
  }

  // Two slots' blocks of 4 in one pass, against each alone.
  qwen4exp::Model::DraftBlock b0{0, N, toks[N], 4}, b1{1, N, toks[N - 1], 4};
  auto a0 = model.df_pass(nullptr, 0, &b0, 1), a1 = model.df_pass(nullptr, 0, &b1, 1);
  qwen4exp::Model::DraftBlock both[2] = {b0, b1};
  auto ab = model.df_pass(nullptr, 0, both, 2);
  uint32_t bad = 0;
  for (uint32_t k = 0; k < 3; k++) bad += !same(ab[0][k], a0[0][k]) + !same(ab[1][k], a1[0][k]);
  std::printf("two slots in one pass: %u of 6 drafts' candidates differ from alone\n", bad);
  failures += bad != 0;
  return failures ? 1 : 0;
}
