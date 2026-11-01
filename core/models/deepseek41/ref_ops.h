// One scalar definition of each deepseek41 operation the kernels compute, for the kernel tests to compare against
// (ADR-0012 item 4). Each follows DeepSeek's inference/model.py (M) or kernel.py (K) and names its section of the
// deepseek41 forward-pass spec; the arithmetic is upstream llama.cpp's (ADR-0062 item 11): fp32 values, no bf16
// rounding, no FP8 or FP4 simulation. Sums and dot products accumulate in double.
#pragma once

#include <cstdint>
#include <vector>

#include "models/deepseek41/layout.h"

namespace engine::deepseek41::ref {

// y = w * x / sqrt(mean(x^2) + D41_EPS) over n values; w null: no weight. y may be x.
void rms_norm(const float *x, const float *w, uint32_t n, float *y);

// A sublayer's hyper-connection mixes (section 3b). comb[src * D41_HC + dst] weighs stream src into stream dst.
struct HcMix {
  float pre[D41_HC];
  float post[D41_HC];
  float comb[D41_HC * D41_HC];
};

// The 24 raw mixes (M Block.hc_mixes): m[j] = (fn_j . X) / sqrt(mean(X^2) + D41_EPS), X the 4 streams one after the
// other (D41_HC_DIM values), fn hc_*_fn's D41_MIX rows of D41_HC_DIM. One statistic over all 4 streams.
void hc_project(const float *X, const float *fn, float *m);

// The mixes from m, hc_*_scale (3) and hc_*_base (24) (K hc_split_sinkhorn_kernel): pre = sigmoid + D41_HC_EPS,
// post = 2 sigmoid, comb a row softmax plus D41_HC_EPS, then columns, then D41_SINKHORN - 1 times rows and columns,
// each divided by its sum plus D41_HC_EPS. comb's columns end summing to 1 less about D41_HC_EPS.
HcMix hc_split(const float *m, const float *scale, const float *base);

// Both of the above, from the streams.
HcMix hc_mixes(const float *X, const float *fn, const float *scale, const float *base);

// A sublayer's input (sections 3c, 3f, 4; M Block.hc_pre, then the norm): x = RMSNorm_w(sum_c pre[c] X_c). pre is the
// earlier sublayer's: the previous FFN's for attention, this layer's attention's for the FFN.
void hc_collapse_norm(const float *X, const float *pre, const float *w, float *x);

// The streams after a sublayer (sections 3e, 3h; M Block.hc_post): out_dst = post[dst] y + sum_src comb[src][dst]
// X_src. out must not alias X.
void hc_post(const float *X, const float *y, const HcMix &mix, float *out);

// Engram's injection (section 3a; M Engram.forward), in place on the 4 streams X. kv = engram_kv . g: a key for each
// stream, then the shared value (D41_ENG_KV values); w: engram_gate's rows, one a stream. Per stream c,
// dot = sum_d X_c w_c key_c / (rms(X_c) rms(key_c)) / sqrt(D41_EMBD), gate = sigmoid(sign(dot) sqrt(max(|dot|,
// D41_ENG_CLAMP))), X_c += gate value. gate, when not null, gets the 4 gates.
void engram_gate(float *X, const float *kv, const float *w, float *gate);

// The rotation of the last D41_ROT of n values (n = D41_HD or D41_IDX_D) to pos (section 3d; rope.h), or back from
// it. yarn: the layer has compressed rows.
void rope_tail(float *v, uint32_t n, uint32_t pos, bool yarn, bool inverse);

// A compressed group's latent before its rotation (section 3d, "the compressor"; M Compressor.forward). kv and sc are
// attn_compressor_kv . x and attn_compressor_gate . x for the group's r tokens, r rows of D41_HD each. r = 1:
// RMSNorm_w(kv), sc unused. r = 2: RMSNorm_w(sum_t kv_t softmax_t(sc_t)), the softmax per channel over the 2 tokens.
void compress(const float *kv, const float *sc, uint32_t r, const float *w, float *latent);

// A compressed row (section 3d; M Indexer.forward, then Attention._compress_kv) from the group's latent before its
// rotation: ik = RMSNorm_{k_norm}(attn_k . latent) rotated at pos, then the latent rotated at pos; row = [latent | ik]
// (D41_COMP_ROW values). attn_k: D41_IDX_D rows of D41_HD; pos = j r, the group's first position.
void compressed_row(const float *latent, const float *attn_k, const float *k_norm, uint32_t pos, float *row);

// The indexer's scores (section 3d, "Indexer"; M Indexer.forward): score[s] = sum_h relu(iq_h . ik_s) w_h / 64.
// iq: D41_IDX_H heads of D41_IDX_D, rotated at the query's position; w: indexer.proj . x, unscaled (D41_IDX_H); ik:
// n index keys, ik_stride floats apart.
void index_score(const float *iq, const float *w, const float *ik, uint32_t ik_stride, uint32_t n, float *score);

// The indices of the min(k, n) largest of s, in increasing order; of equal values the lower indices are taken
// (kernels/include/topk.h). -inf values are taken only when fewer than k others remain.
std::vector<uint32_t> top_k(const float *s, uint32_t n, uint32_t k);

// The candidate filter's first level (section 3d; M select_candidate_blocks) over n scores: each block of
// D41_CAND_BLOCK rows scores its largest, the block holding row n - 1 scores +inf, and the top D41_CAND_BLOCKS blocks
// not at -inf are kept. keep[s] is 1 for the rows of a kept block.
std::vector<uint8_t> candidates(const float *score, uint32_t n);

// Attention of the D41_H heads of q (D41_HD each) over the keys idx[0 .. nidx) of kv, kv_stride floats apart; K = V
// (section 3d, "Attention core"; K sparse_attn_kernel). An index of -1 is no key. With scale 1 / sqrt(D41_HD) and
// m = max(-1e30, max_i s_i): o_h = sum_i exp(s_i - m) v_i / (sum_i exp(s_i - m) + exp(sinks[h] - m)). A head with no
// key gives 0.
void sparse_attn(const float *q, const float *kv, uint32_t kv_stride, const int32_t *idx, uint32_t nidx,
                 const float *sinks, float *o);

// The attention's output (section 3d, "Output projection"; M Attention.forward): o_g, heads 8g .. 8g + 7 one after
// the other, a_g = wo_a's rows [1024 g, 1024 (g + 1)) . o_g, out = wo_b . a. o: D41_H x D41_HD, its rotation taken
// off; wo_a: 8,192 rows of 4,096; wo_b: D41_EMBD rows of 8,192. a gets the 8,192 values.
void grouped_output(const float *o, const float *wo_a, const float *wo_b, float *a, float *out);

// A token's routed experts (section 3g; M Gate.forward), from ffn_gate_inp . x and exp_probs_b: s = sqrt(softplus(
// logit)); the D41_USED largest s + bias, ids in decreasing order of it (ties to the lower id); w = s / (sum s +
// D41_EPS) D41_ROUTE_SCALE. The bias chooses; it does not weigh. In float, as the kernel computes it.
struct Route {
  int32_t id[D41_USED];
  float w[D41_USED];
};
Route router(const float *logits, const float *bias);

// The SwiGLU of an expert's gate and up values with the clamp (section 3g; M Expert.forward):
// silu(min(g, D41_SWIGLU_LIMIT)) clamp(u, -D41_SWIGLU_LIMIT, D41_SWIGLU_LIMIT).
float swiglu(float g, float u);

}  // namespace engine::deepseek41::ref
