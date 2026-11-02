// Level 1 (docs/TESTING.md): deepseek41's scalar operations (core/models/deepseek41/ref_ops.h), which the kernel tests
// take as their reference, checked where an operation has a closed form or an invariant: Sinkhorn's columns and rows,
// its result against ggml's at any logits; the mixes' indexing; norms of constant vectors; the Engram gate at a dot of 0
// and at key = stream; the compressor's pooling; the index key against ggml's product; the indexer's relu and weights;
// top-k's ties; the candidate filter's pin; attention against a sink; the router's bias; the SwiGLU clamp; the output.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <random>
#include <vector>

#include "ggml-cpu.h"
#include "ggml.h"
#include "models/deepseek41/ref_ops.h"
#include "test_util.h"

namespace ref = engine::deepseek41::ref;

namespace {

std::mt19937 rng(41);

std::vector<float> randn(size_t n, float scale = 1.0f) {
  std::normal_distribution<float> nd(0.0f, scale);
  std::vector<float> v(n);
  for (float &x : v) x = nd(rng);
  return v;
}

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// ggml's ggml_dsv4_hc_comb over the n rows of D41_MIX logits at m: n rows of D41_HC * D41_HC, its dst first within a
// src, the order of HcMix.comb.
std::vector<float> ggml_comb(const float *m, uint32_t n, const float *scale, const float *base) {
  ggml_init_params ip = {64u << 20, nullptr, false};
  ggml_context *ctx = ggml_init(ip);
  ggml_tensor *tm = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D41_MIX, n);
  ggml_tensor *ts = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3), *tb = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D41_MIX);
  std::copy(m, m + (size_t)n * D41_MIX, (float *)tm->data);
  std::copy(scale, scale + 3, (float *)ts->data);
  std::copy(base, base + D41_MIX, (float *)tb->data);
  ggml_tensor *r = ggml_dsv4_hc_comb(ctx, tm, ts, tb, D41_HC_EPS, D41_SINKHORN);
  ggml_cgraph *gf = ggml_new_graph(ctx);
  ggml_build_forward_expand(gf, r);
  EXPECT(ggml_graph_compute_with_ctx(ctx, gf, 1) == GGML_STATUS_SUCCESS, "ggml's graph failed");
  std::vector<float> y((const float *)r->data, (const float *)r->data + (size_t)n * D41_HC * D41_HC);
  ggml_free(ctx);
  return y;
}

// attn_k . latent by ggml's mul_mat, as upstream's graph has it: BF16 weights and an F32 latent, which ggml converts.
std::vector<float> ggml_attn_k(const std::vector<uint16_t> &ak, const float *latent) {
  ggml_init_params ip = {16u << 20, nullptr, false};
  ggml_context *ctx = ggml_init(ip);
  ggml_tensor *w = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, D41_HD, D41_IDX_D);
  ggml_tensor *l = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D41_HD);
  std::copy(ak.begin(), ak.end(), (uint16_t *)w->data);
  std::copy(latent, latent + D41_HD, (float *)l->data);
  ggml_tensor *r = ggml_mul_mat(ctx, w, l);
  ggml_cgraph *gf = ggml_new_graph(ctx);
  ggml_build_forward_expand(gf, r);
  EXPECT(ggml_graph_compute_with_ctx(ctx, gf, 1) == GGML_STATUS_SUCCESS, "ggml's graph failed");
  std::vector<float> y((const float *)r->data, (const float *)r->data + D41_IDX_D);
  ggml_free(ctx);
  return y;
}

void norms() {
  std::vector<float> x(D41_HD, -3.0f), w(D41_HD, 2.0f), y(D41_HD);
  ref::rms_norm(x.data(), w.data(), D41_HD, y.data());
  for (float v : y) EXPECT(std::fabs(v + 2.0f) < 1e-6f, "RMSNorm of a constant: %g", v);
  // Scale-free, even for tiny values: the epsilon is 1e-20.
  std::vector<float> a = randn(D41_HD), b(D41_HD), c(D41_HD), small(D41_HD);
  for (uint32_t i = 0; i < D41_HD; i++) small[i] = a[i] * 1e-6f;
  ref::rms_norm(a.data(), nullptr, D41_HD, b.data());
  ref::rms_norm(small.data(), nullptr, D41_HD, c.data());
  for (uint32_t i = 0; i < D41_HD; i++) EXPECT(std::fabs(b[i] - c[i]) < 1e-5f, "RMSNorm scale %u", i);
}

void sinkhorn() {
  float scale[3] = {1, 1, 1}, base[D41_MIX] = {};
  // Close logits: rows and columns sum to 1. Logits 40 apart: the columns still do, since each round ends on them;
  // the rows have no closed form after 20 rounds, and the comparison with ggml below covers them.
  for (float spread : {0.5f, 20.0f})
    for (int t = 0; t < 500; t++) {
      std::uniform_real_distribution<float> u(-spread, spread);
      float m[D41_MIX];
      for (float &v : m) v = u(rng);
      const ref::HcMix mix = ref::hc_split(m, scale, base);
      for (uint32_t i = 0; i < D41_HC; i++) {
        double row = 0, col = 0;  // rows checked at close logits only
        for (uint32_t j = 0; j < D41_HC; j++) {
          row += mix.comb[i * D41_HC + j], col += mix.comb[j * D41_HC + i];
          EXPECT(mix.comb[i * D41_HC + j] > 0, "comb entry");
        }
        EXPECT(std::fabs(col - 1) < 3e-6, "spread %g: column %u sums to %.9f", spread, i, col);
        if (spread < 1) EXPECT(std::fabs(row - 1) < 1e-5, "spread %g: row %u sums to %.9f", spread, i, row);
        EXPECT(mix.pre[i] > D41_HC_EPS && mix.pre[i] < 1 + 2 * D41_HC_EPS, "pre %g", mix.pre[i]);
        EXPECT(mix.post[i] > 0 && mix.post[i] <= 2, "post %g", mix.post[i]);
      }
    }
  // The same mixes as ggml's, with a scale and bases that are not 1 and 0, over both ranges of logits.
  {
    const uint32_t n = 1000;
    const float cs[3] = {1, 1, 0.75f};
    std::vector<float> ms((size_t)n * D41_MIX), cb = randn(D41_MIX);
    for (uint32_t t = 0; t < n; t++) {
      std::uniform_real_distribution<float> u(t < n / 2 ? -0.5f : -20.0f, t < n / 2 ? 0.5f : 20.0f);
      for (uint32_t j = 0; j < D41_MIX; j++) ms[(size_t)t * D41_MIX + j] = u(rng);
    }
    const std::vector<float> want = ggml_comb(ms.data(), n, cs, cb.data());
    double err = 0;
    for (uint32_t t = 0; t < n; t++) {
      const ref::HcMix got = ref::hc_split(ms.data() + (size_t)t * D41_MIX, cs, cb.data());
      for (uint32_t i = 0; i < D41_HC * D41_HC; i++)
        err = std::max(err, std::fabs((double)got.comb[i] - want[(size_t)t * D41_HC * D41_HC + i]));
    }
    std::printf("comb: largest difference from ggml %.3g\n", err);
    EXPECT(err < 1e-5, "comb differs from ggml's");
  }
  // Zero logits: pre = sigmoid(0) + eps, post = 1, comb uniform. The scales and bases apply per part.
  float m[D41_MIX] = {};
  ref::HcMix mix = ref::hc_split(m, scale, base);
  for (uint32_t i = 0; i < D41_HC; i++) {
    EXPECT(std::fabs(mix.pre[i] - (0.5 + 1e-6)) < 1e-7, "pre at 0: %.9f", mix.pre[i]);
    EXPECT(std::fabs(mix.post[i] - 1.0f) < 1e-7f, "post at 0");
    for (uint32_t j = 0; j < D41_HC; j++) EXPECT(std::fabs(mix.comb[i * D41_HC + j] - 0.25f) < 1e-6f, "comb at 0");
  }
  m[1] = 2, m[D41_HC + 2] = 3, m[2 * D41_HC + 1 * D41_HC + 2] = 30;  // pre 1, post 2, comb from stream 1 to 2
  float s2[3] = {0.5f, 2.0f, 1.0f}, b2[D41_MIX] = {};
  b2[0] = 1;
  mix = ref::hc_split(m, s2, b2);
  EXPECT(std::fabs(mix.pre[1] - (sigmoid(1.0) + 1e-6)) < 1e-7, "pre[1] takes scale[0]");
  EXPECT(std::fabs(mix.pre[0] - (sigmoid(1.0) + 1e-6)) < 1e-7, "pre[0] takes base[0]");
  EXPECT(std::fabs(mix.post[2] - 2 * sigmoid(6.0)) < 1e-6, "post[2] takes scale[1]");
  EXPECT(mix.comb[1 * D41_HC + 2] > 0.95f, "comb[1][2]: %g", mix.comb[1 * D41_HC + 2]);  // 20 rounds stop short of 1

  // hc_project: one statistic over all four streams. X = c everywhere, fn's row j picks one value: m_j = 1.
  std::vector<float> X(D41_HC_DIM, 7.0f), fn((size_t)D41_MIX * D41_HC_DIM, 0.0f);
  for (uint32_t j = 0; j < D41_MIX; j++) fn[(size_t)j * D41_HC_DIM + j * 800] = 1.0f;
  float mm[D41_MIX];
  ref::hc_project(X.data(), fn.data(), mm);
  for (float v : mm) EXPECT(std::fabs(v - 1.0f) < 1e-6f, "hc_project: %g", v);
}

void collapse_and_post() {
  std::vector<float> X = randn(D41_HC_DIM), w(D41_EMBD, 1.0f), x(D41_EMBD), want(D41_EMBD);
  const float pre[D41_HC] = {0, 0, 1, 0};
  ref::hc_collapse_norm(X.data(), pre, w.data(), x.data());
  ref::rms_norm(X.data() + 2 * D41_EMBD, nullptr, D41_EMBD, want.data());
  for (uint32_t d = 0; d < D41_EMBD; d++) EXPECT(std::fabs(x[d] - want[d]) < 1e-6f, "collapse %u", d);
  // post = (1, 0, 0, 0), comb sends stream c to stream (c + 1) % 4.
  ref::HcMix mix = {};
  mix.post[0] = 1;
  for (uint32_t c = 0; c < D41_HC; c++) mix.comb[c * D41_HC + (c + 1) % D41_HC] = 1;
  std::vector<float> y = randn(D41_EMBD), out(D41_HC_DIM);
  ref::hc_post(X.data(), y.data(), mix, out.data());
  for (uint32_t c = 0; c < D41_HC; c++)
    for (uint32_t d = 0; d < D41_EMBD; d++) {
      const float e = X[(size_t)((c + 3) % D41_HC) * D41_EMBD + d] + (c == 0 ? y[d] : 0.0f);
      EXPECT(std::fabs(out[(size_t)c * D41_EMBD + d] - e) < 1e-6f, "hc_post stream %u", c);
    }
}

void engram() {
  // Keys orthogonal to the streams under the weights give a dot of 0: the gate is sigmoid(sqrt(1e-6)).
  std::vector<float> X = randn(D41_HC_DIM), kv = randn(D41_ENG_KV), w(D41_HC_DIM, 1.0f);
  for (uint32_t c = 0; c < D41_HC; c++)
    for (uint32_t d = 0; d < D41_EMBD; d++) kv[(size_t)c * D41_EMBD + d] = d % 2 ? 1.0f : 0.0f;
  for (uint32_t d = 1; d < D41_HC_DIM; d += 2) X[d] = 0;  // the streams on even dimensions only
  std::vector<float> X0 = X;
  float gate[D41_HC];
  ref::engram_gate(X.data(), kv.data(), w.data(), gate);
  const double g0 = sigmoid(std::sqrt(1e-6));
  for (uint32_t c = 0; c < D41_HC; c++) {
    EXPECT(std::fabs(gate[c] - g0) < 1e-7, "gate at a dot of 0: %.9f", gate[c]);
    for (uint32_t d = 0; d < D41_EMBD; d++) {
      const size_t i = (size_t)c * D41_EMBD + d;
      EXPECT(std::fabs(X[i] - (X0[i] + g0 * kv[(size_t)D41_HC * D41_EMBD + d])) < 1e-5, "Engram value");
    }
  }
  // Key = stream with unit weights: the sum of x^2 over mean(x^2) is 5120, so dot = sqrt(5120) per stream; key =
  // -stream: its negative.
  for (int sign : {1, -1}) {
    X = randn(D41_HC_DIM);
    for (size_t i = 0; i < (size_t)D41_HC_DIM; i++) kv[i] = sign * X[i];
    ref::engram_gate(X.data(), kv.data(), w.data(), gate);
    const double dt = sign * std::sqrt((double)D41_EMBD), g = sigmoid(std::copysign(std::sqrt(std::fabs(dt)), dt));
    for (float v : gate) EXPECT(std::fabs(v - g) < 1e-6, "gate at dot %g: %.9f, expected %.9f", dt, v, g);
  }
}

void compressor() {
  std::vector<float> kv = randn(2 * D41_HD), sc(2 * D41_HD, 0.0f), w(D41_HD, 1.0f), lat(D41_HD), want(D41_HD);
  // Equal scores: the mean of the two tokens. Token 1's scores far above: token 1.
  ref::compress(kv.data(), sc.data(), 2, w.data(), lat.data());
  for (uint32_t d = 0; d < D41_HD; d++) want[d] = (kv[d] + kv[D41_HD + d]) / 2;
  ref::rms_norm(want.data(), nullptr, D41_HD, want.data());
  for (uint32_t d = 0; d < D41_HD; d++) EXPECT(std::fabs(lat[d] - want[d]) < 1e-5f, "ratio 2, equal scores");
  for (uint32_t d = 0; d < D41_HD; d++) sc[D41_HD + d] = 100.0f;
  ref::compress(kv.data(), sc.data(), 2, w.data(), lat.data());
  ref::rms_norm(kv.data() + D41_HD, nullptr, D41_HD, want.data());
  for (uint32_t d = 0; d < D41_HD; d++) EXPECT(std::fabs(lat[d] - want[d]) < 1e-5f, "ratio 2, one token");
  ref::compress(kv.data(), nullptr, 1, w.data(), lat.data());
  ref::rms_norm(kv.data(), nullptr, D41_HD, want.data());
  for (uint32_t d = 0; d < D41_HD; d++) EXPECT(lat[d] == want[d], "ratio 1");

  // The compressed row: the index key from the latent before its rotation, then both rotated at the row's position.
  std::vector<float> ak((size_t)D41_IDX_D * D41_HD, 0.0f), kn(D41_IDX_D, 1.0f), row(D41_COMP_ROW);
  for (uint32_t i = 0; i < D41_IDX_D; i++) ak[(size_t)i * D41_HD + D41_HD - D41_IDX_D + i] = 1.0f;  // ik = latent tail
  ref::compressed_row(lat.data(), ak.data(), kn.data(), 20, row.data());
  std::vector<float> l2 = lat, ik(lat.end() - D41_IDX_D, lat.end());
  for (float &v : ik) v = ggml_bf16_to_fp32(ggml_fp32_to_bf16(v));
  ref::rms_norm(ik.data(), nullptr, D41_IDX_D, ik.data());
  ref::rope_tail(ik.data(), D41_IDX_D, 20, true, false);
  ref::rope_tail(l2.data(), D41_HD, 20, true, false);
  for (uint32_t d = 0; d < D41_HD; d++) EXPECT(row[d] == l2[d], "row's latent %u", d);
  for (uint32_t d = 0; d < D41_IDX_D; d++) EXPECT(std::fabs(row[D41_HD + d] - ik[d]) < 1e-6f, "row's key %u", d);

  // The key against ggml's product, and how far the same key lies when the latent is not rounded to bf16.
  std::vector<uint16_t> akb(ak.size());
  ak = randn(ak.size(), 0.05f);
  for (size_t i = 0; i < ak.size(); i++) {
    akb[i] = ggml_fp32_to_bf16(ak[i]).bits;
    ak[i] = ggml_bf16_to_fp32(ggml_bf16_t{akb[i]});
  }
  kn = randn(D41_IDX_D, 0.5f);
  std::vector<float> want_ik = ggml_attn_k(akb, lat.data()), plain(D41_IDX_D);
  for (uint32_t i = 0; i < D41_IDX_D; i++) {
    double s = 0;
    for (uint32_t d = 0; d < D41_HD; d++) s += (double)ak[(size_t)i * D41_HD + d] * lat[d];
    plain[i] = (float)s;
  }
  for (std::vector<float> *v : {&want_ik, &plain}) {
    ref::rms_norm(v->data(), kn.data(), D41_IDX_D, v->data());
    ref::rope_tail(v->data(), D41_IDX_D, 20, true, false);
  }
  ref::compressed_row(lat.data(), ak.data(), kn.data(), 20, row.data());
  double mx = 0, err = 0, err_plain = 0;
  for (uint32_t d = 0; d < D41_IDX_D; d++) {
    mx = std::max(mx, (double)std::fabs(want_ik[d]));
    err = std::max(err, (double)std::fabs(row[D41_HD + d] - want_ik[d]));
    err_plain = std::max(err_plain, (double)std::fabs(plain[d] - want_ik[d]));
  }
  std::printf("index key against ggml's product %.2e, unrounded latent %.2e\n", err / mx, err_plain / mx);
  EXPECT(err / mx < 1e-5, "index key against ggml's product: %g", err / mx);
  EXPECT(err_plain / mx > 1e-4, "a latent not in bf16 should move the key past the kernel test's limit: %g", err_plain / mx);
}

void indexer() {
  // Two heads: head 0 scores q.k, head 1 scores -q.k; relu keeps the positive one, weighted, over 64.
  const uint32_t n = 5;
  std::vector<float> iq((size_t)D41_IDX_H * D41_IDX_D, 0.0f), w(D41_IDX_H, 0.0f), ik((size_t)n * D41_COMP_ROW, 0.0f);
  std::vector<float> score(n);
  iq[0] = 1, iq[D41_IDX_D] = -1, w[0] = 3, w[1] = 5;
  for (uint32_t s = 0; s < n; s++) ik[(size_t)s * D41_COMP_ROW + D41_HD] = (float)s - 2;
  ref::index_score(iq.data(), w.data(), ik.data() + D41_HD, D41_COMP_ROW, n, score.data());
  for (uint32_t s = 0; s < n; s++) {
    const double v = (double)s - 2, want = (v > 0 ? 3 * v : -5 * v) / 64;
    EXPECT(std::fabs(score[s] - want) < 1e-6, "score %u: %g, expected %g", s, score[s], want);
  }
}

void top_k() {
  // Ties go to the lower index; the result is in increasing order.
  std::vector<float> s = {1, 3, 3, 0, 3, 2, 3};
  EXPECT((ref::top_k(s.data(), 7, 3) == std::vector<uint32_t>{1, 2, 4}), "ties at the threshold");
  EXPECT((ref::top_k(s.data(), 7, 5) == std::vector<uint32_t>{1, 2, 4, 5, 6}), "top 5");
  EXPECT(ref::top_k(s.data(), 7, 10).size() == 7, "k above n");
  std::vector<float> flat(1000, 0.5f);
  auto t = ref::top_k(flat.data(), 1000, 512);
  for (uint32_t i = 0; i < 512; i++) EXPECT(t[i] == i, "all equal: index %u is %u", i, t[i]);
  const float ninf = -std::numeric_limits<float>::infinity();
  std::vector<float> m = {ninf, 1, ninf, 2};
  EXPECT((ref::top_k(m.data(), 4, 3) == std::vector<uint32_t>{0, 1, 3}), "-inf taken last, lower first");
  // Random distinct values: the set of the k largest.
  std::vector<float> r = randn(16384);
  auto big = ref::top_k(r.data(), 16384, 512);
  std::vector<float> sorted = r;
  std::nth_element(sorted.begin(), sorted.begin() + 511, sorted.end(), std::greater<float>());
  for (uint32_t i : big) EXPECT(r[i] >= sorted[511], "a chosen value below the 512th");
}

void candidates() {
  // Up to 2,048 blocks every block is kept; the newest row's block whatever its score.
  std::vector<float> s = randn(16384);
  for (uint8_t k : ref::candidates(s.data(), 16384)) EXPECT(k == 1, "16,384 rows: every one kept");
  const uint32_t n = 2049 * D41_CAND_BLOCK + 3;  // 2,050 blocks, the last of 3 rows
  s.assign(n, 0.0f);
  for (uint32_t b = 0; b < 2050; b++)
    for (uint32_t i = 0; i < D41_CAND_BLOCK && b * D41_CAND_BLOCK + i < n; i++) s[b * D41_CAND_BLOCK + i] = (float)b;
  s[n - 1] = -100;  // the newest block scores lowest, and is kept by the pin
  s[n - 2] = s[n - 3] = -100;
  auto keep = ref::candidates(s.data(), n);
  // Kept: the pinned block 2049, then blocks 2048 down to 2: 2,048 blocks. Dropped: blocks 0 and 1.
  for (uint32_t i = 0; i < n; i++) EXPECT(keep[i] == (i >= 2 * D41_CAND_BLOCK), "row %u: %u", i, keep[i]);
}

void attention() {
  std::vector<float> q = randn((size_t)D41_H * D41_HD, 0.05f), kv = randn(4 * (size_t)D41_HD), sinks = randn(D41_H);
  std::vector<float> o((size_t)D41_H * D41_HD);
  // One key: o = v sigmoid(s - sink), the sink unscaled.
  const int32_t one[] = {-1, 2, -1};
  ref::sparse_attn(q.data(), kv.data(), D41_HD, one, 3, sinks.data(), o.data());
  for (uint32_t h = 0; h < D41_H; h++) {
    double s = 0;
    for (uint32_t d = 0; d < D41_HD; d++) s += (double)q[(size_t)h * D41_HD + d] * kv[2 * D41_HD + d];
    const double p = sigmoid(s / std::sqrt(512.0) - sinks[h]);
    for (uint32_t d = 0; d < D41_HD; d++)
      EXPECT(std::fabs(o[(size_t)h * D41_HD + d] - p * kv[2 * D41_HD + d]) < 1e-5, "one key, head %u", h);
  }
  // No key: 0. A sink far below: the softmax of the keys alone, here two equal keys.
  const int32_t none[] = {-1, -1};
  ref::sparse_attn(q.data(), kv.data(), D41_HD, none, 2, sinks.data(), o.data());
  for (float v : o) EXPECT(v == 0.0f, "no key");
  std::vector<float> low(D41_H, -1e4f);
  const int32_t twice[] = {1, 1};
  ref::sparse_attn(q.data(), kv.data(), D41_HD, twice, 2, low.data(), o.data());
  for (uint32_t h = 0; h < D41_H; h++)
    for (uint32_t d = 0; d < D41_HD; d++) EXPECT(std::fabs(o[(size_t)h * D41_HD + d] - kv[D41_HD + d]) < 1e-5, "two");
}

void output() {
  // o nonzero in group 3 only: a is zero outside rows 3,072 .. 4,095, and out = wo_b . a.
  const uint32_t G = D41_H / D41_OG * D41_HD, A = D41_OG * D41_OLR;
  std::vector<float> o((size_t)D41_H * D41_HD, 0.0f), wo_a = randn((size_t)A * G, 0.02f);
  std::vector<float> wo_b = randn((size_t)D41_EMBD * A, 0.02f), a(A), out(D41_EMBD);
  for (uint32_t i = 0; i < G; i++) o[3 * G + i] = (float)((int)(i % 7) - 3);
  ref::grouped_output(o.data(), wo_a.data(), wo_b.data(), a.data(), out.data());
  for (uint32_t r = 0; r < A; r++) {
    double want = 0;
    if (r / D41_OLR == 3)
      for (uint32_t i = 0; i < G; i++) want += (double)wo_a[(size_t)r * G + i] * o[3 * G + i];
    EXPECT(std::fabs(a[r] - want) < 1e-4, "a[%u]: %g, expected %g", r, a[r], want);
  }
  for (uint32_t i = 0; i < D41_EMBD; i += 97) {
    double want = 0;
    for (uint32_t r = 3 * D41_OLR; r < 4 * D41_OLR; r++) want += (double)wo_b[(size_t)i * A + r] * a[r];
    EXPECT(std::fabs(out[i] - want) < 1e-4, "out[%u]", i);
  }
}

void router() {
  // The bias chooses experts 100 .. 105, which the logits rank lowest; their weights come from the logits alone.
  std::vector<float> logits = randn(D41_EXPERTS), bias(D41_EXPERTS, 0.0f);
  for (uint32_t e = 100; e < 106; e++) logits[e] = -1.0f - (float)(e - 100) * 0.1f, bias[e] = 50.0f;
  const ref::Route r = ref::router(logits.data(), bias.data());
  double sum = 0, ssum = 0;
  for (uint32_t k = 0; k < D41_USED; k++) {
    EXPECT(r.id[k] == (int32_t)(100 + k), "expert %u: %d", k, r.id[k]);
    ssum += std::sqrt(std::log1p(std::exp((double)logits[100 + k])));
  }
  for (uint32_t k = 0; k < D41_USED; k++) {
    const double want = std::sqrt(std::log1p(std::exp((double)logits[100 + k]))) / ssum * 1.5;
    EXPECT(std::fabs(r.w[k] - want) < 1e-6, "weight %u: %g, expected %g", k, r.w[k], want);
    sum += r.w[k];
  }
  EXPECT(std::fabs(sum - 1.5) < 1e-6, "the weights sum to %g", sum);
  // Equal scores: the lower ids, in order.
  std::vector<float> flat(D41_EXPERTS, 0.3f), zero(D41_EXPERTS, 0.0f);
  const ref::Route t = ref::router(flat.data(), zero.data());
  for (uint32_t k = 0; k < D41_USED; k++) EXPECT(t.id[k] == (int32_t)k && std::fabs(t.w[k] - 0.25f) < 1e-6f, "ties");
}

void swiglu() {
  auto silu = [](double g) { return g / (1 + std::exp(-g)); };
  EXPECT(std::fabs(ref::swiglu(20, 2) - silu(10) * 2) < 1e-5, "gate above the limit");
  EXPECT(std::fabs(ref::swiglu(-20, 2) - silu(-20) * 2) < 1e-9, "gate below: not clamped");
  EXPECT(std::fabs(ref::swiglu(1, 30) - silu(1) * 10) < 1e-5, "up above");
  EXPECT(std::fabs(ref::swiglu(1, -30) + silu(1) * 10) < 1e-5, "up below");
  EXPECT(std::fabs(ref::swiglu(0.5f, 0.25f) - silu(0.5) * 0.25) < 1e-7, "inside");
}

}  // namespace

int main() {
  // ggml here is built for the reference CPU, so a CPU without these instructions cannot run it.
#define NEEDS(f) \
  if (!__builtin_cpu_supports(f)) return test::skip("the CPU lacks " f ", which the ggml build uses")
  NEEDS("avx2");
  NEEDS("fma");
  NEEDS("f16c");
  NEEDS("bmi2");
  NEEDS("avx512f");
  NEEDS("avx512vbmi");
  NEEDS("avx512vnni");
  NEEDS("avx512bf16");
#undef NEEDS
  norms();
  sinkhorn();
  collapse_and_post();
  engram();
  compressor();
  indexer();
  top_k();
  candidates();
  attention();
  output();
  router();
  swiglu();
  std::printf("ok\n");
  return 0;
}
