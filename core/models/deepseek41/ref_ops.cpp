#include "models/deepseek41/ref_ops.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <numeric>

#include "models/deepseek41/rope.h"

namespace engine::deepseek41::ref {

namespace {

double dot(const float *a, const float *b, uint32_t n) {
  double s = 0;
  for (uint32_t i = 0; i < n; i++) s += (double)a[i] * b[i];
  return s;
}

double rstd(const float *x, uint32_t n) { return 1.0 / std::sqrt(dot(x, x, n) / n + (double)D41_EPS); }

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// x at bf16's precision, to nearest with ties to even, as ggml converts it.
float round_bf16(float x) {
  if (std::isnan(x)) return x;
  const uint32_t u = std::bit_cast<uint32_t>(x);
  return std::bit_cast<float>(((u + (0x7fffu + ((u >> 16) & 1))) >> 16) << 16);
}

}  // namespace

void rms_norm(const float *x, const float *w, uint32_t n, float *y) {
  const double r = rstd(x, n);
  for (uint32_t i = 0; i < n; i++) y[i] = (float)(x[i] * r * (w ? w[i] : 1.0f));
}

void hc_project(const float *X, const float *fn, float *m) {
  const double r = rstd(X, D41_HC_DIM);
  for (uint32_t j = 0; j < D41_MIX; j++) m[j] = (float)(dot(fn + (size_t)j * D41_HC_DIM, X, D41_HC_DIM) * r);
}

HcMix hc_split(const float *m, const float *scale, const float *base) {
  constexpr uint32_t C = D41_HC;
  constexpr double eps = D41_HC_EPS;
  HcMix mix;
  for (uint32_t j = 0; j < C; j++) {
    mix.pre[j] = (float)(sigmoid((double)m[j] * scale[0] + base[j]) + eps);
    mix.post[j] = (float)(2 * sigmoid((double)m[C + j] * scale[1] + base[C + j]));
  }
  double c[C][C];
  for (uint32_t s = 0; s < C; s++) {
    double mx = -std::numeric_limits<double>::infinity(), sum = 0;
    for (uint32_t d = 0; d < C; d++) {
      c[s][d] = (double)m[2 * C + s * C + d] * scale[2] + base[2 * C + s * C + d];
      mx = std::max(mx, c[s][d]);
    }
    for (uint32_t d = 0; d < C; d++) sum += c[s][d] = std::exp(c[s][d] - mx);
    for (uint32_t d = 0; d < C; d++) c[s][d] = c[s][d] / sum + eps;
  }
  auto columns = [&] {
    for (uint32_t d = 0; d < C; d++) {
      double sum = 0;
      for (uint32_t s = 0; s < C; s++) sum += c[s][d];
      for (uint32_t s = 0; s < C; s++) c[s][d] /= sum + eps;
    }
  };
  columns();
  for (uint32_t it = 1; it < D41_SINKHORN; it++) {
    for (uint32_t s = 0; s < C; s++) {
      double sum = 0;
      for (uint32_t d = 0; d < C; d++) sum += c[s][d];
      for (uint32_t d = 0; d < C; d++) c[s][d] /= sum + eps;
    }
    columns();
  }
  for (uint32_t s = 0; s < C; s++)
    for (uint32_t d = 0; d < C; d++) mix.comb[s * C + d] = (float)c[s][d];
  return mix;
}

HcMix hc_mixes(const float *X, const float *fn, const float *scale, const float *base) {
  float m[D41_MIX];
  hc_project(X, fn, m);
  return hc_split(m, scale, base);
}

void hc_collapse(const float *X, const float *pre, float *x) {
  for (uint32_t d = 0; d < D41_EMBD; d++) {
    double s = 0;
    for (uint32_t c = 0; c < D41_HC; c++) s += (double)pre[c] * X[(size_t)c * D41_EMBD + d];
    x[d] = (float)s;
  }
}

void hc_collapse_norm(const float *X, const float *pre, const float *w, float *x) {
  hc_collapse(X, pre, x);
  rms_norm(x, w, D41_EMBD, x);
}

void hc_post(const float *X, const float *y, const HcMix &mix, float *out) {
  for (uint32_t dst = 0; dst < D41_HC; dst++)
    for (uint32_t d = 0; d < D41_EMBD; d++) {
      double s = (double)mix.post[dst] * y[d];
      for (uint32_t src = 0; src < D41_HC; src++) s += (double)mix.comb[src * D41_HC + dst] * X[(size_t)src * D41_EMBD + d];
      out[(size_t)dst * D41_EMBD + d] = (float)s;
    }
}

void engram_gate(float *X, const float *kv, const float *w, float *gate) {
  const float *value = kv + (size_t)D41_HC * D41_EMBD;
  for (uint32_t c = 0; c < D41_HC; c++) {
    float *x = X + (size_t)c * D41_EMBD;
    const float *key = kv + (size_t)c * D41_EMBD, *wc = w + (size_t)c * D41_EMBD;
    double s = 0;
    for (uint32_t d = 0; d < D41_EMBD; d++) s += (double)x[d] * wc[d] * key[d];
    const double dt = s * rstd(x, D41_EMBD) * rstd(key, D41_EMBD) / std::sqrt((double)D41_EMBD);
    const double g = sigmoid(std::copysign(std::sqrt(std::max(std::fabs(dt), (double)D41_ENG_CLAMP)), dt));
    for (uint32_t d = 0; d < D41_EMBD; d++) x[d] = (float)(x[d] + g * value[d]);
    if (gate) gate[c] = (float)g;
  }
}

void rope_tail(float *v, uint32_t n, uint32_t pos, bool yarn, bool inverse) {
  d41_rope(v + n - D41_ROT, pos, yarn, inverse);
}

void compress(const float *kv, const float *sc, uint32_t r, const float *w, float *latent) {
  for (uint32_t d = 0; d < D41_HD; d++) {
    if (r == 1) {
      latent[d] = kv[d];
      continue;
    }
    double mx = -std::numeric_limits<double>::infinity(), sum = 0, acc = 0;
    for (uint32_t t = 0; t < r; t++) mx = std::max(mx, (double)sc[(size_t)t * D41_HD + d]);
    for (uint32_t t = 0; t < r; t++) {
      const double e = std::exp(sc[(size_t)t * D41_HD + d] - mx);
      sum += e;
      acc += e * kv[(size_t)t * D41_HD + d];
    }
    latent[d] = (float)(acc / sum);
  }
  rms_norm(latent, w, D41_HD, latent);
}

void compressed_row(const float *latent, const float *attn_k, const float *k_norm, uint32_t pos, float *row) {
  // The index key is made from the latent before the latent's rotation (M orders the indexer first).
  float *ik = row + D41_HD;
  float rounded[D41_HD];
  for (uint32_t d = 0; d < D41_HD; d++) rounded[d] = round_bf16(latent[d]);
  for (uint32_t i = 0; i < D41_IDX_D; i++) ik[i] = (float)dot(attn_k + (size_t)i * D41_HD, rounded, D41_HD);
  rms_norm(ik, k_norm, D41_IDX_D, ik);
  rope_tail(ik, D41_IDX_D, pos, true, false);
  std::copy(latent, latent + D41_HD, row);
  rope_tail(row, D41_HD, pos, true, false);
}

void index_score(const float *iq, const float *w, const float *ik, uint32_t ik_stride, uint32_t n, float *score) {
  // 1 / sqrt(D41_IDX_D) / sqrt(D41_IDX_H) = 1 / 64 (M Indexer.softmax_scale and n_heads^-0.5).
  const double scale = 1.0 / std::sqrt((double)D41_IDX_D * D41_IDX_H);
  for (uint32_t s = 0; s < n; s++) {
    double acc = 0;
    for (uint32_t h = 0; h < D41_IDX_H; h++)
      acc += std::max(0.0, dot(iq + (size_t)h * D41_IDX_D, ik + (size_t)s * ik_stride, D41_IDX_D)) * (w[h] * scale);
    score[s] = (float)acc;
  }
}

std::vector<uint32_t> top_k(const float *s, uint32_t n, uint32_t k) {
  std::vector<uint32_t> idx(n);
  std::iota(idx.begin(), idx.end(), 0u);
  k = std::min(k, n);
  std::stable_sort(idx.begin(), idx.end(), [&](uint32_t a, uint32_t b) { return s[a] > s[b]; });
  idx.resize(k);
  std::sort(idx.begin(), idx.end());
  return idx;
}

std::vector<uint8_t> candidates(const float *score, uint32_t n) {
  std::vector<uint8_t> keep(n, 0);
  if (n == 0) return keep;
  const uint32_t nb = (n + D41_CAND_BLOCK - 1) / D41_CAND_BLOCK;
  std::vector<float> b(nb, -std::numeric_limits<float>::infinity());
  for (uint32_t s = 0; s < n; s++) b[s / D41_CAND_BLOCK] = std::max(b[s / D41_CAND_BLOCK], score[s]);
  b[(n - 1) / D41_CAND_BLOCK] = std::numeric_limits<float>::infinity();
  for (uint32_t blk : top_k(b.data(), nb, D41_CAND_BLOCKS)) {
    if (b[blk] == -std::numeric_limits<float>::infinity()) continue;
    for (uint32_t s = blk * D41_CAND_BLOCK; s < std::min(n, (blk + 1) * D41_CAND_BLOCK); s++) keep[s] = 1;
  }
  return keep;
}

void sparse_attn(const float *q, const float *kv, uint32_t kv_stride, const int32_t *idx, uint32_t nidx,
                 const float *sinks, float *o) {
  const double scale = 1.0 / std::sqrt((double)D41_HD);
  std::vector<double> s(nidx), acc(D41_HD);
  for (uint32_t h = 0; h < D41_H; h++) {
    const float *qh = q + (size_t)h * D41_HD;
    double m = -1e30;
    for (uint32_t i = 0; i < nidx; i++) {
      s[i] = idx[i] < 0 ? -std::numeric_limits<double>::infinity()
                        : dot(qh, kv + (size_t)idx[i] * kv_stride, D41_HD) * scale;
      m = std::max(m, s[i]);
    }
    double den = std::exp((double)sinks[h] - m);
    std::fill(acc.begin(), acc.end(), 0.0);
    for (uint32_t i = 0; i < nidx; i++) {
      if (idx[i] < 0) continue;
      const double e = std::exp(s[i] - m);
      den += e;
      const float *v = kv + (size_t)idx[i] * kv_stride;
      for (uint32_t d = 0; d < D41_HD; d++) acc[d] += e * v[d];
    }
    for (uint32_t d = 0; d < D41_HD; d++) o[(size_t)h * D41_HD + d] = (float)(acc[d] / den);
  }
}

void grouped_output(const float *o, const float *wo_a, const float *wo_b, float *a, float *out) {
  constexpr uint32_t G = D41_H / D41_OG * D41_HD;  // 4,096: a group's heads
  for (uint32_t g = 0; g < D41_OG; g++)
    for (uint32_t r = 0; r < D41_OLR; r++) {
      const size_t row = (size_t)g * D41_OLR + r;
      a[row] = (float)dot(wo_a + row * G, o + (size_t)g * G, G);
    }
  for (uint32_t i = 0; i < D41_EMBD; i++) out[i] = (float)dot(wo_b + (size_t)i * D41_OG * D41_OLR, a, D41_OG * D41_OLR);
}

void route(const float *logits, const float *bias, uint32_t n, uint32_t used, int32_t *id, float *w) {
  std::vector<float> s(n), b(n);
  for (uint32_t e = 0; e < n; e++) {
    // softplus as torch and ggml take it: x itself above 20.
    const float x = logits[e], sp = x > 20.0f ? x : std::log1p(std::exp(x));
    s[e] = std::sqrt(sp);
    b[e] = s[e] + bias[e];
  }
  std::vector<uint32_t> order(n);
  std::iota(order.begin(), order.end(), 0u);
  std::stable_sort(order.begin(), order.end(), [&](uint32_t x, uint32_t y) { return b[x] > b[y]; });
  float sum = 0;
  for (uint32_t k = 0; k < used; k++) id[k] = (int32_t)order[k], sum += s[order[k]];
  for (uint32_t k = 0; k < used; k++) w[k] = s[id[k]] / (sum + D41_EPS) * D41_ROUTE_SCALE;
}

Route router(const float *logits, const float *bias) {
  Route r;
  route(logits, bias, D41_EXPERTS, D41_USED, r.id, r.w);
  return r;
}

float swiglu(float g, float u) {
  g = std::min(g, D41_SWIGLU_LIMIT);
  u = std::clamp(u, -D41_SWIGLU_LIMIT, D41_SWIGLU_LIMIT);
  return g / (1.0f + std::exp(-g)) * u;
}

}  // namespace engine::deepseek41::ref
