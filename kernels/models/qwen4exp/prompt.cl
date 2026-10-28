// qwen4exp's prompt-path kernels: a chunk of runs of consecutive tokens, each of one sequence (PromptPlan) in one
// pass. The maths is the decode kernels' (layer.cl); here the products run on the XMX units (kernels/gemm.cl,
// kernels/quant/qgemm.cl) with 16-bit activations, and the experts' assignments are grouped by expert. Row t of a
// "[T][w]" buffer is at t * w; grids are recorded for the list's chunk size and a work-group past n returns. A kernel
// that follows a sequence takes one run's plan, its row t being the chunk's row0 + t. The recurrence (prompt_gdn.cl)
// and attention (prompt_attn.cl) are modules of their own.
#include "builtins.h"
#include "layout.h"
#include "moe_tile.h"
#include "step.h"
#include "topk.h"

#define RT_SG 32
#include "route.h"

static inline float silu(float x) { return x / (1.0f + exp(-x)); }
static inline float sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }
static inline float softplus(float x) { return x > 20.0f ? x : log(1.0f + exp(x)); }

static inline float wg_sum(float v, __local float *red) {
  v = sg_sum(v);
  uint nsg = get_num_sub_groups();
  if (sg_lane() == 0) red[sg_id()] = v;
  barrier(CLK_LOCAL_MEM_FENCE);
  float t = 0.0f;
  for (uint i = 0; i < nsg; i++) t += red[i];
  barrier(CLK_LOCAL_MEM_FENCE);
  return t;
}

static inline float wg_max(float v, __local float *red) {
  v = sg_max(v);
  uint nsg = get_num_sub_groups();
  if (sg_lane() == 0) red[sg_id()] = v;
  barrier(CLK_LOCAL_MEM_FENCE);
  float t = -INFINITY;
  for (uint i = 0; i < nsg; i++) t = fmax(t, red[i]);
  barrier(CLK_LOCAL_MEM_FENCE);
  return t;
}

static inline float rope(__local const float *x, uint d, uint pos) {
  if (d >= Q_ROT) return x[d];
  uint i = d % (Q_ROT / 2);
  float theta = (float)pos * pow(Q_ROPE_BASE, -2.0f * (float)i / (float)Q_ROT);
  float c = cos(theta), s = sin(theta);
  return d < Q_ROT / 2 ? x[i] * c - x[i + Q_ROT / 2] * s : x[i] * s + x[i + Q_ROT / 2] * c;
}

// Interleaved M-RoPE, as in layer.cl, by the positions of cell e of the chunk's table (step.h: (t, h, w, 0) per cell
// from the chunk's first block).
static inline float rope_at(__local const float *x, uint d, __global const uint *ppos, uint e) {
  uint i = d % (Q_ROT / 2), m = i % 3;
  return rope(x, d, ppos[4 * e + ((m == 1) ? 1 : (m == 2 && i < 30) ? 2 : 0)]);
}

// The table's entry for chunk row t.
static inline uint pos_entry(__global const PromptPlan *pp, uint t) { return pp->pos0 % Q_IDX_BLOCK + t; }

static inline void q8_0_store(__local const float *x, __global uchar *dst) {
  float amax = 0.0f;
  for (uint i = 0; i < 32; i++) amax = fmax(amax, fabs(x[i]));
  float d = amax / 127.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
  vstore_half(d, 0, (__global half *)dst);
  for (uint i = 0; i < 32; i++) ((__global char *)dst)[2 + i] = (char)round(x[i] * id);
}

static inline float q8_0_load(__global const uchar *row, uint d) {
  __global const uchar *b = row + (d / 32) * Q8_BLOCK;
  return vload_half(0, (__global const half *)b) * (float)((__global const char *)b)[2 + d % 32];
}

#define ROW_OR_RETURN(t) \
  if ((t) >= pp->n) return;

// The kernels that follow a sequence run once for all the chunk's runs (runs: PROMPT_MAX_RUNS plans in host memory,
// the unused ones empty). RUN_OF_ROW: t, a row of the chunk, becomes a row of the run that holds it, pp that run's
// plan and ri its index; a row past the last run returns. The runs fill the rows in order, so the search stops at the
// first run that ends after t: a chunk of one run reads one plan. RUN_OF_GROUP: the run is group z.
#define RUN_OF_ROW(t)                                                 \
  uint ri = 0;                                                        \
  while (ri + 1 < PROMPT_MAX_RUNS && (t) >= runs[ri].row0 + runs[ri].n) \
    ri++;                                                             \
  __global const PromptPlan *pp = runs + ri;                          \
  if ((t) < pp->row0 || (t) - pp->row0 >= pp->n) return;             \
  (t) -= pp->row0;
#define RUN_OF_GROUP()                     \
  const uint ri = get_group_id(2);         \
  __global const PromptPlan *pp = runs + ri;

// The hyper-connection kernels take rows r0 .. r1 - 1 of the chunk (all of them, or this card's own when the cards
// split the rows, prompt.cpp): group y is row r0 + y.
#define OWN_ROW_OR_RETURN(t) \
  if ((t) >= min(r1, pp->n)) return;

// ---------------------------------------------------------------------------------------------------------------
// Hyper-connections

// R[t] = four copies of the token's embedding e[t]. Groups (Q_HC_DIM / 64, T), 64.
__kernel void p_hc_init(__global const float *e, __global const PromptPlan *pp, __global float *R) {
  uint i = get_global_id(0), t = get_group_id(1);
  ROW_OR_RETURN(t)
  R[(ulong)t * Q_HC_DIM + i] = e[(ulong)t * Q_EMBD + i % Q_EMBD];
}

// The four streams' RMS norms of R[t], as bf16 for the bf16 products: outb[t][s] = R[t][s] * rr[t][s] * gamma[s],
// with rr[t][s] = 1 / rms kept for p_hc_mix. Groups (Q_HC, r1 - r0), 256.
__kernel void p_hc_norm(__global const float *in, __global const float *gamma, __global const PromptPlan *pp,
                        __global float *rr, __global ushort *outb, uint r0, uint r1) {
  __local float red[64];
  uint s = get_group_id(0), t = r0 + get_group_id(1);
  OWN_ROW_OR_RETURN(t)
  ulong base = (ulong)t * Q_HC_DIM + (ulong)s * Q_EMBD;
  float ss = 0.0f;
  for (uint i = get_local_id(0); i < Q_EMBD; i += get_local_size(0)) ss += in[base + i] * in[base + i];
  ss = wg_sum(ss, red);
  float r = rsqrt(ss / (float)Q_EMBD + Q_EPS);
  if (get_local_id(0) == 0) rr[t * Q_HC + s] = r;
  for (uint i = get_local_id(0); i < Q_EMBD; i += get_local_size(0))
    outb[base + i] = f32_to_bf16(in[base + i] * r * gamma[s * Q_EMBD + i]);
}

// The down and inject products' rows (Q_HC_LR + Q_HC values, lri_w apart): lrb[t] = bf16(silu(lri / 4)) for the
// up product, inj[t] = the inject values. Groups (ceil((Q_HC_LR + Q_HC) / 64), r1 - r0), 64.
__kernel void p_hc_lr(__global const float *lri, uint lri_w, __global const PromptPlan *pp, __global ushort *lrb,
                      __global float *inj, uint r0, uint r1) {
  uint i = get_global_id(0), t = r0 + get_group_id(1);
  OWN_ROW_OR_RETURN(t)
  if (i >= Q_HC_LR + Q_HC) return;
  float v = lri[(ulong)t * lri_w + i];
  if (i < Q_HC_LR) {
    v *= 0.25f;
    lrb[(ulong)t * Q_HC_LR + i] = f32_to_bf16(v / (1.0f + exp(-v)));
  } else {
    inj[t * Q_HC + i - Q_HC_LR] = v;
  }
}

// x[t][i] = 1/4 sum_s n[t][s][i] sigma(a[t][i][s]), with n = R * rr * gamma recomputed (p_hc_norm) and a in fp16,
// value-major (the core reorders the up projection's rows so); written as fp16 (for the quantised products) and bf16
// (for the BF16 ones), and into the other card's x16p and xbp when set. A sub-group takes 64 consecutive values, lane l
// values 16 k + l, so a row goes out in 128-byte block stores, which cross to the other card far faster than 2-byte
// ones (docs/measurements/MEASUREMENTS.md#2026-10-17-hc-split). Groups (Q_EMBD / 256, r1 - r0), 64.
REQD_SG __kernel void p_hc_mix(__global const float *R, __global const float *rr, __global const float *gamma,
                               __global const half *a, __global const PromptPlan *pp, __global half *x16,
                               __global ushort *xb, uint r0, uint r1, __global half *x16p, __global ushort *xbp) {
  uint t = r0 + get_group_id(1);
  OWN_ROW_OR_RETURN(t)
  const uint i0 = (get_group_id(0) * 4 + sg_id()) * 64, lane = sg_lane();
  ulong base = (ulong)t * Q_HC_DIM;
  ushort h[4], b[4];
  for (uint k = 0; k < 4; k++) {
    const uint i = i0 + 16 * k + lane;
    float acc = 0.0f;
    for (uint s = 0; s < Q_HC; s++) {
      float n = R[base + s * Q_EMBD + i] * rr[t * Q_HC + s] * gamma[s * Q_EMBD + i];
      acc += n * sigmoid(vload_half(base + i * Q_HC + s, a));
    }
    acc *= 0.25f;
    h[k] = as_ushort(convert_half(acc));
    b[k] = f32_to_bf16(acc);
  }
  const ulong at = (ulong)t * Q_EMBD + i0;
  intel_sub_group_block_write_us4((__global ushort *)x16 + at, (ushort4)(h[0], h[1], h[2], h[3]));
  intel_sub_group_block_write_us4(xb + at, (ushort4)(b[0], b[1], b[2], b[3]));
  if (x16p) {
    intel_sub_group_block_write_us4((__global ushort *)x16p + at, (ushort4)(h[0], h[1], h[2], h[3]));
    intel_sub_group_block_write_us4(xbp + at, (ushort4)(b[0], b[1], b[2], b[3]));
  }
}

// After an exchange: y = the two cards' fp16 partial sums added in card order (p0, then p1 when there are two
// cards), R[t][s] += 2 sigma(inj[t][s] / 4) y. Groups (Q_EMBD / 64, r1 - r0), 64.
__kernel void p_hc_combine(__global const half *p0, __global const half *p1, __global const float *inj,
                           __global const PromptPlan *pp, __global float *R, uint r0, uint r1) {
  uint i = get_global_id(0), t = r0 + get_group_id(1);
  OWN_ROW_OR_RETURN(t)
  ulong at = (ulong)t * Q_EMBD + i;
  float y = vload_half(at, p0);
  if (p1) y += vload_half(at, p1);
  for (uint s = 0; s < Q_HC; s++)
    R[(ulong)t * Q_HC_DIM + s * Q_EMBD + i] += 2.0f * sigmoid(inj[t * Q_HC + s] * 0.25f) * y;
}

// p_hc_combine, then p_hc_norm of the block that follows (gamma), in one pass over R: one work-group per row and
// stream keeps the stream's 2,560 values in registers between the update and the norm. Groups (Q_HC, r1 - r0), 256.
__kernel void p_hc_combine_norm(__global const half *p0, __global const half *p1, __global const float *inj,
                                __global const float *gamma, __global const PromptPlan *pp, __global float *R,
                                __global float *rr, __global ushort *outb, uint r0, uint r1) {
  __local float red[64];
  uint s = get_group_id(0), t = r0 + get_group_id(1), lid = get_local_id(0);
  OWN_ROW_OR_RETURN(t)
  const float g = 2.0f * sigmoid(inj[t * Q_HC + s] * 0.25f);
  const ulong base = (ulong)t * Q_HC_DIM + (ulong)s * Q_EMBD;
  float v[Q_EMBD / 256];
  float ss = 0.0f;
  for (uint k = 0; k < Q_EMBD / 256; k++) {
    uint i = lid + 256 * k;
    ulong at = (ulong)t * Q_EMBD + i;
    float y = vload_half(at, p0);
    if (p1) y += vload_half(at, p1);
    v[k] = R[base + i] + g * y;
    R[base + i] = v[k];
    ss += v[k] * v[k];
  }
  ss = wg_sum(ss, red);
  float r = rsqrt(ss / (float)Q_EMBD + Q_EPS);
  if (lid == 0) rr[t * Q_HC + s] = r;
  for (uint k = 0; k < Q_EMBD / 256; k++) {
    uint i = lid + 256 * k;
    outb[base + i] = f32_to_bf16(v[k] * r * gamma[s * Q_EMBD + i]);
  }
}

// Rows r0 .. r1 - 1 among the run's last STEP_MAX_ROWS of R into the other card's R (Rp), so that both cards hold
// the rows the head and the draft layer read. Groups (Q_HC_DIM / 64, STEP_MAX_ROWS), 64.
__kernel void p_hc_share(__global const float *R, __global const PromptPlan *runs, uint r0, uint r1,
                         __global float *Rp) {
  RUN_OF_GROUP()
  const uint n = pp->n, end = pp->row0 + n, i = get_global_id(0);
  const uint t = end - min(n, (uint)STEP_MAX_ROWS) + get_group_id(1);
  if (t < r0 || t >= min(r1, end)) return;
  Rp[(ulong)t * Q_HC_DIM + i] = R[(ulong)t * Q_HC_DIM + i];
}

// Each run's last row of R into row ri of r (the decode buffers), for the head. Groups (Q_HC_DIM / 64, 1, runs), 64.
__kernel void p_take_last(__global const float *R, __global const PromptPlan *runs, __global float *r) {
  RUN_OF_GROUP()
  uint i = get_global_id(0);
  if (pp->n == 0) return;
  r[(ulong)ri * Q_HC_DIM + i] = R[(ulong)(pp->row0 + pp->n - 1) * Q_HC_DIM + i];
}

// ---------------------------------------------------------------------------------------------------------------
// Gated DeltaNet: the convolution, norms and gates. Channels of m and u as in layer.cl.

// u[t][ch] = silu(conv over m at positions p-3 .. p): rows of the chunk, or the state for positions before it.
// Groups (ceil(nch / 64), T), 64.
__kernel void p_gdn_conv(__global const float *m, __global const float *w, __global const float *state,
                         ulong slot_stride, uint nch, __global const PromptPlan *runs, __global float *u) {
  uint ch = get_global_id(0), t = get_group_id(1);
  RUN_OF_ROW(t)
  m += (ulong)pp->row0 * nch;
  u += (ulong)pp->row0 * nch;
  if (ch >= nch) return;
  uint pos = pp->pos0 + t;
  __global const float *st = state + pp->slot * slot_stride;
  float mv[4];
  for (uint d = 1; d <= 3; d++)
    mv[3 - d] = t >= d ? m[(ulong)(t - d) * nch + ch] : st[((pos + 3 - d) % 3) * nch + ch];
  mv[3] = m[(ulong)t * nch + ch];
  __global const float *k = w + ch * Q_GDN_CONV;
  float acc = k[0] * mv[0];
  acc += k[1] * mv[1];
  acc += k[2] * mv[2];
  acc += k[3] * mv[3];
  u[(ulong)t * nch + ch] = silu(acc);
}

// The state keeps m at position p in slot p % 3: the chunk's last three rows go in. Groups ceil(nch / 64), 64.
__kernel void p_gdn_conv_save(__global const float *m, __global float *state, ulong slot_stride, uint nch,
                              __global const PromptPlan *runs) {
  RUN_OF_GROUP()
  uint ch = get_global_id(0);
  if (ch >= nch) return;
  uint n = pp->n;
  __global float *st = state + pp->slot * slot_stride;
  m += (ulong)pp->row0 * nch;
  for (uint t = n > 3 ? n - 3 : 0; t < n; t++) st[((pp->pos0 + t) % 3) * nch + ch] = m[(ulong)t * nch + ch];
}

// L2 norm of the q and k heads of each row, in place. Groups (2 nkh, T), 128.
__kernel void p_gdn_l2norm(__global float *u, uint nch, __global const PromptPlan *pp) {
  __local float red[64];
  uint t = get_group_id(1);
  ROW_OR_RETURN(t)
  __global float *x = u + (ulong)t * nch + get_group_id(0) * Q_GDN_D;
  uint d = get_local_id(0);
  float v = x[d];
  float ss = wg_sum(v * v, red);
  x[d] = v * rsqrt(ss + Q_EPS);
}

// gb[t] = g[nvh] then beta[nvh] from ab[t] = alpha[nvh] then beta_raw[nvh]. Groups (1, T), 64.
__kernel void p_gdn_gates(__global const float *ab, __global const float *A, __global const float *dt, uint nvh,
                          __global const PromptPlan *pp, __global float *gb) {
  uint h = get_local_id(0), t = get_group_id(1);
  ROW_OR_RETURN(t)
  if (h >= nvh) return;
  __global const float *r = ab + (ulong)t * 2 * nvh;
  gb[(ulong)t * 2 * nvh + h] = A[h] * softplus(r[h] + dt[h]);
  gb[(ulong)t * 2 * nvh + nvh + h] = sigmoid(r[nvh + h]);
}

// The gated output norm: o[t][lv] = norm(raw[t][lv]) * w * sigma(z[t][lv]), into fp16 for the output product.
// Groups (nvh, T), 128.
__kernel void p_gdn_out(__global const float *raw, __global const float *z, __global const float *norm, uint nvh,
                        __global const PromptPlan *pp, __global half *o16) {
  __local float red[64];
  uint lv = get_group_id(0), t = get_group_id(1), j = get_local_id(0);
  ROW_OR_RETURN(t)
  ulong at = ((ulong)t * nvh + lv) * Q_GDN_D + j;
  float out = raw[at];
  float ss = wg_sum(out * out, red);
  o16[at] = convert_half(out * rsqrt(ss / (float)Q_GDN_D + Q_EPS) * norm[j] * sigmoid(z[at]));
}

// ---------------------------------------------------------------------------------------------------------------
// Sparse attention

// Per row: groups [0, nq) norm and rotate the queries in place; [nq, nq + nkv) norm and rotate the keys and store keys
// and values in the 8-bit cache at the row's position, in the page the table pt gives it (layout.h).
// Groups (nq + nkv, T), 256.
// The queries also go out as fp16 scaled by 1/16: to qt, dimension-major with hp heads per dimension (p_attn_xmx),
// and to q16, [row][nq][256] (p_attn_tiled).
__kernel void p_attn_prep(__global float *Q, __global const float *K, __global const float *V,
                          __global const float *qn, __global const float *kn, uint nq, uint nkv, __global uchar *kc,
                          __global uchar *vc, __global const uint *pt, uint npp, __global const PromptPlan *runs,
                          __global const uint *ppos_all, uint pstride, __global half *qt, uint hp,
                          __global half *q16) {
  __local float x[Q_ATT_D], y[Q_ATT_D], red[64];
  uint g = get_group_id(0), t = get_group_id(1), d = get_local_id(0);
  RUN_OF_ROW(t)
  __global const uint *ppos = ppos_all + (ulong)ri * pstride;
  const ulong r0 = pp->row0;
  Q += r0 * nq * 2 * Q_ATT_D;
  K += r0 * nkv * Q_ATT_D;
  V += r0 * nkv * Q_ATT_D;
  qt += r0 * Q_ATT_D * hp;
  q16 += r0 * nq * Q_ATT_D;
  uint pos = pp->pos0 + t;
  bool is_q = g < nq;
  __global float *q = Q + (ulong)t * nq * 2 * Q_ATT_D;
  float v = is_q ? q[g * 2 * Q_ATT_D + d] : K[((ulong)t * nkv + g - nq) * Q_ATT_D + d];
  float ss = wg_sum(v * v, red);
  x[d] = v * rsqrt(ss / (float)Q_ATT_D + Q_EPS) * (is_q ? qn[d] : kn[d]);
  barrier(CLK_LOCAL_MEM_FENCE);
  float r = rope_at(x, d, ppos, pos_entry(pp, t));
  if (is_q) {
    q[g * 2 * Q_ATT_D + d] = r;
    qt[((ulong)t * Q_ATT_D + d) * hp + g] = convert_half(r * (1.0f / 16.0f));
    q16[((ulong)t * nq + g) * Q_ATT_D + d] = convert_half(r * (1.0f / 16.0f));
    return;
  }
  uint c = g - nq;
  ulong cell = (kv_cell(kv_page(pt, npp, pp->slot, pos), pos) * nkv + c) * KV_HEAD_BYTES;
  y[d] = r;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (d < Q_ATT_D / 32) q8_0_store(y + 32 * d, kc + cell + d * Q8_BLOCK);
  barrier(CLK_LOCAL_MEM_FENCE);
  y[d] = V[((ulong)t * nkv + c) * Q_ATT_D + d];
  barrier(CLK_LOCAL_MEM_FENCE);
  if (d < Q_ATT_D / 32) q8_0_store(y + 32 * d, vc + cell + d * Q8_BLOCK);
}

// The indexer's keys are split by position between the cards as in the decode step (layer.cl): block b belongs to
// card b % C, and its page p holds its pooled key and raw keys at local page p / C.

// The indexer's queries and raw keys: iq[t] holds 4 query heads then the key (640 values). Groups 0-3 norm and
// rotate a query head in place; group 4 stores the raw key when the row's block is this card's. Groups (5, T), 128.
__kernel void p_idx_prep(__global float *iq, __global const float *qn, __global uchar *raw, __global const uint *pt,
                         uint npp, __global const PromptPlan *runs, __global const uint *ppos_all, uint pstride,
                         uint c, uint C) {
  __local float x[Q_IDX_D], red[64];
  uint g = get_group_id(0), t = get_group_id(1), d = get_local_id(0);
  RUN_OF_ROW(t)
  __global const uint *ppos = ppos_all + (ulong)ri * pstride;
  uint pos = pp->pos0 + t, b = pos / Q_IDX_BLOCK;
  __global float *r = iq + ((ulong)pp->row0 + t) * (Q_IDX_H + 1) * Q_IDX_D;
  if (g < Q_IDX_H) {
    float v = r[g * Q_IDX_D + d];
    float ss = wg_sum(v * v, red);
    x[d] = v * rsqrt(ss / (float)Q_IDX_D + Q_EPS) * qn[d];
    barrier(CLK_LOCAL_MEM_FENCE);
    r[g * Q_IDX_D + d] = rope_at(x, d, ppos, pos_entry(pp, t));
    return;
  }
  if (b % C != c) return;
  x[d] = r[Q_IDX_H * Q_IDX_D + d];
  barrier(CLK_LOCAL_MEM_FENCE);
  ulong lp = kv_cell(kv_page(pt, npp, pp->slot, pos) / C, pos);
  if (d < Q_IDX_D / 32) q8_0_store(x + 32 * d, raw + lp * IDX_KEY_BYTES + d * Q8_BLOCK);
}

// Pools every block of this card the chunk completes: block b = pos0 / 4 + group, when 4b + 3 lies in the chunk;
// rotated by the positions of its first cell, which is the table's entry 4 group. Groups (T / 4 + 1), 128.
__kernel void p_idx_pool(__global const uchar *raw, __global const uint *pt, uint npp, __global const float *kn,
                         __global float *pooled, __global const PromptPlan *runs, __global const uint *ppos_all,
                         uint pstride, uint c, uint C) {
  __local float blk[Q_IDX_D], red[64];
  RUN_OF_GROUP()
  __global const uint *ppos = ppos_all + (ulong)ri * pstride;
  uint b = pp->pos0 / Q_IDX_BLOCK + get_group_id(0), d = get_local_id(0);
  uint last = Q_IDX_BLOCK * b + Q_IDX_BLOCK - 1;
  if (last < pp->pos0 || last >= pp->pos0 + pp->n || b % C != c) return;
  const uint lpage = kv_page(pt, npp, pp->slot, Q_IDX_BLOCK * b) / C;
  __global const uchar *rk = raw + kv_cell(lpage, 0) * IDX_KEY_BYTES;
  float mean = 0.0f;
  for (uint t = 0; t < Q_IDX_BLOCK; t++) mean += q8_0_load(rk + (ulong)t * IDX_KEY_BYTES, d);
  mean *= 1.0f / Q_IDX_BLOCK;
  float ss = wg_sum(mean * mean, red);
  blk[d] = mean * rsqrt(ss / (float)Q_IDX_D + Q_EPS) * kn[d];
  barrier(CLK_LOCAL_MEM_FENCE);
  pooled[(ulong)lpage * Q_IDX_D + d] = rope_at(blk, d, ppos, Q_IDX_BLOCK * get_group_id(0));
}

static inline uint idx_own(uint nb, uint c, uint C) { return nb > c ? (nb - c + C - 1) / C : 0; }

// scores[t][lb] = sum_h relu(iq_h . K_b) for this card's blocks complete at row t's position, by local index lb
// (block lb C + c). A work-item takes one block for IDX_SCORE_ROWS consecutive rows, reading its key once; each sum
// is the one-row expression in the same order, so the same bits. Groups (ceil(nb_max / 64), ceil(T / IDX_SCORE_ROWS)),
// 64; rows nb_max apart.
#define IDX_SCORE_ROWS 8
#if Q_IDX_H != 4
#error "p_idx_score holds the indexer's heads in a float4"
#endif
__kernel void p_idx_score(__global const float *iq, __global const float *pooled, __global const uint *pt, uint npp,
                          uint nb_max, __global const PromptPlan *runs, __global float *scores, uint c, uint C) {
  __local float4 qs[Q_IDX_D][IDX_SCORE_ROWS];  // qs[i][r]: the 4 heads' value i of row t0 + r
  RUN_OF_GROUP()
  const uint lb = get_global_id(0), t0 = IDX_SCORE_ROWS * get_group_id(1), lid = get_local_id(0);
  const uint n = pp->n;
  if (t0 >= n) return;
  iq += (ulong)pp->row0 * (Q_IDX_H + 1) * Q_IDX_D;
  scores += (ulong)pp->row0 * nb_max;
  const uint rows = min((uint)IDX_SCORE_ROWS, n - t0);
  // Below the width every cell is taken whatever the scores; the last row sees the most blocks.
  const uint nkv_last = pp->pos0 + t0 + rows;
  if (nkv_last <= Q_IDX_CELLS) return;
  for (uint j = lid; j < Q_IDX_D * IDX_SCORE_ROWS; j += 64) {
    const uint i = j / IDX_SCORE_ROWS, r = j % IDX_SCORE_ROWS;
    __global const float *q = iq + (ulong)(t0 + min(r, rows - 1)) * (Q_IDX_H + 1) * Q_IDX_D + i;
    qs[i][r] = (float4)(q[0], q[Q_IDX_D], q[2 * Q_IDX_D], q[3 * Q_IDX_D]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lb >= idx_own(nkv_last / Q_IDX_BLOCK, c, C)) return;
  const uint page = kv_page(pt, npp, pp->slot, (lb * C + c) * Q_IDX_BLOCK);
  __global const float *K = pooled + (ulong)(page / C) * Q_IDX_D;
  float4 dot[IDX_SCORE_ROWS];
  for (uint r = 0; r < IDX_SCORE_ROWS; r++) dot[r] = (float4)0.0f;
  for (uint i0 = 0; i0 < Q_IDX_D; i0 += 4) {
    const float4 k4 = vload4(0, K + i0);
    const float kv[4] = {k4.s0, k4.s1, k4.s2, k4.s3};
    __attribute__((opencl_unroll_hint)) for (uint u = 0; u < 4; u++)
      __attribute__((opencl_unroll_hint)) for (uint r = 0; r < IDX_SCORE_ROWS; r++) {
        const float4 q = qs[i0 + u][r];
        dot[r].s0 += q.s0 * kv[u];
        dot[r].s1 += q.s1 * kv[u];
        dot[r].s2 += q.s2 * kv[u];
        dot[r].s3 += q.s3 * kv[u];
      }
  }
  for (uint r = 0; r < rows; r++) {
    const uint t = t0 + r, nkv = pp->pos0 + t + 1;
    if (nkv <= Q_IDX_CELLS || lb >= idx_own(nkv / Q_IDX_BLOCK, c, C)) continue;
    float score = 0.0f;
    score += fmax(dot[r].s0, 0.0f);
    score += fmax(dot[r].s1, 0.0f);
    score += fmax(dot[r].s2, 0.0f);
    score += fmax(dot[r].s3, 0.0f);
    scores[(ulong)t * nb_max + lb] = score;
  }
}

// Row t's candidates on this card: its best Q_IDX_BLOCKS + 1 blocks in block order as (block, score) pairs, after their
// count; rows IDX_CAND words apart. Groups (T), TOPK_WG.
#define IDX_CAND (2 * (Q_IDX_BLOCKS + 1) + 2)
__kernel void p_idx_cand(__global const float *scores, uint nb_max, __global const PromptPlan *runs,
                         __global uint *blocks_all, __global float *bscores_all, __global uint *cand_all, uint c,
                         uint C) {
  __local uint hist[TOPK_HIST];
  __local uint scan[TOPK_WG];
  uint t = get_group_id(0), lid = get_local_id(0);
  RUN_OF_ROW(t)
  t += pp->row0;  // the chunk's row; the run's is t - row0
  uint nkv = pp->pos0 + t - pp->row0 + 1;
  __global uint *cand = cand_all + (ulong)t * IDX_CAND;
  if (nkv <= Q_IDX_CELLS) {
    if (lid == 0) cand[0] = 0;
    return;
  }
  uint own = idx_own(nkv / Q_IDX_BLOCK, c, C);
  __global uint *blocks = blocks_all + (ulong)t * (Q_IDX_BLOCKS + 1);
  __global float *bscores = bscores_all + (ulong)t * (Q_IDX_BLOCKS + 1);
  uint got = topk_select(scores + (ulong)t * nb_max, own, Q_IDX_BLOCKS + 1, blocks, bscores, hist, scan);
  barrier(CLK_GLOBAL_MEM_FENCE);
  for (uint i = lid; i < got; i += TOPK_WG) {
    cand[2 + 2 * i] = blocks[i] * C + c;
    cand[3 + 2 * i] = as_uint(bscores[i]);
  }
  if (lid == 0) cand[0] = got;
}

// The cells row t attends to, merged from the two cards' candidates (c0: card 0's, c1: card 1's or null) as
// layer.cl's idx_merge does, into cells[t] (Q_IDX_CELLS apart) and ncells[t]; also as a mask for p_attn_tiled: one
// bit per chosen block in sel[t] (nbw words), and wbx[t] = the partial block | its cells << 24 (0xffffff: none).
// merged, blocks and bscores are per-row scratch. Groups (T), TOPK_WG.
__kernel void p_idx_merge(__global const uint *c0, __global const uint *c1, __global const PromptPlan *runs,
                          __global float *merged_all, __global uint *blocks_all, __global float *bscores_all,
                          __global uint *cells_all, __global uint *ncells, __global uint *sel_all, uint nbw,
                          __global uint *wbx) {
  __local uint hist[TOPK_HIST];
  __local uint scan[TOPK_WG];
  __local uint mb[2 * (Q_IDX_BLOCKS + 1)];
  uint t = get_group_id(0), lid = get_local_id(0);
  RUN_OF_ROW(t)
  t += pp->row0;  // the chunk's row; the run's is t - row0
  uint nkv = pp->pos0 + t - pp->row0 + 1;
  __global uint *cells = cells_all + (ulong)t * Q_IDX_CELLS;
  if (nkv <= Q_IDX_CELLS) {
    for (uint i = lid; i < nkv; i += TOPK_WG) cells[i] = i;
    if (lid == 0) {
      ncells[t] = nkv;
      wbx[t] = 0xffffffu;
    }
    return;
  }
  __global const uint *lists[2] = {c0 + (ulong)t * IDX_CAND, c1 ? c1 + (ulong)t * IDX_CAND : 0};
  const uint C = c1 ? 2 : 1;
  __global float *merged = merged_all + (ulong)t * 2 * (Q_IDX_BLOCKS + 1);
  uint total = 0;
  for (uint k = 0; k < C; k++) total += lists[k][0];
  for (uint k = 0; k < C; k++) {
    __global const uint *ck = lists[k];
    uint nk = ck[0];
    for (uint i = lid; i < nk; i += TOPK_WG) {
      uint b = ck[2 + 2 * i], at = i;
      if (C > 1) {
        __global const uint *co = lists[1 - k];
        uint lo = 0, hi = co[0];
        while (lo < hi) {
          uint mid = (lo + hi) / 2;
          if (co[2 + 2 * mid] < b) lo = mid + 1; else hi = mid;
        }
        at += lo;
      }
      mb[at] = b;
      merged[at] = as_float(ck[3 + 2 * i]);
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  __global uint *sel = sel_all + (ulong)t * nbw;
  for (uint i = lid; i < nbw; i += TOPK_WG) sel[i] = 0;
  __global uint *blocks = blocks_all + (ulong)t * (Q_IDX_BLOCKS + 1);
  __global float *bscores = bscores_all + (ulong)t * (Q_IDX_BLOCKS + 1);
  uint nb = nkv / Q_IDX_BLOCK, tail = nkv - nb * Q_IDX_BLOCK;
  uint extra = Q_IDX_CELLS - Q_IDX_TOPK - tail;
  uint want = Q_IDX_BLOCKS + (extra > 0 ? 1 : 0);
  topk_select(merged, total, want, blocks, bscores, hist, scan);
  barrier(CLK_GLOBAL_MEM_FENCE);
  for (uint i = lid; i < want; i += TOPK_WG) blocks[i] = mb[blocks[i]];
  barrier(CLK_GLOBAL_MEM_FENCE);
  // The 513th best is the lowest score among the chosen (the last such in block order); want <= TOPK_WG.
  uint worst = want;
  if (extra > 0) {
    const float s = lid < want ? bscores[lid] : INFINITY;
    const float least = work_group_reduce_min(s);
    worst = work_group_reduce_max(lid < want && s == least ? lid : 0u);
  }
  if (lid == 0) wbx[t] = extra > 0 ? blocks[worst] | (extra << 24) : 0xffffffu;
  for (uint i = lid; i < want; i += TOPK_WG) {
    uint b = blocks[i];
    atomic_or(&sel[b / 32], 1u << (b % 32));
    if (i == worst) {
      for (uint k = 0; k < extra; k++) cells[4 * i + k] = 4 * b + k;
    } else {
      uint at = i < worst ? 4 * i : 4 * (i - 1) + extra;
      for (uint k = 0; k < Q_IDX_BLOCK; k++) cells[at + k] = 4 * b + k;
    }
  }
  if (lid < tail) cells[Q_IDX_TOPK + extra + lid] = nb * Q_IDX_BLOCK + lid;
  if (lid == 0) ncells[t] = Q_IDX_CELLS;
}

// The 8-bit K and V of every cached position (up to the run's end) as fp16, for p_attn_xmx and p_attn_tiled
// (prompt_attn.cl): kh and vh hold [position][nkv][256], each found through the page table pt. Not for a short run,
// which attends from the 8-bit cache (p_attn_rows). Groups (nkv * Q_ATT_D / 32, any), 64: one work-item per Q8_0 block
// of K or V, group y taking positions y, y + groups, and so on.
__kernel void p_kv16(__global const uchar *kc, __global const uchar *vc, __global const uint *pt, uint npp, uint nkv,
                     __global const PromptPlan *pp, __global half *kh, __global half *vh) {
  const uint g = get_global_id(0), end = pp->pos0 + pp->n;
  if (pp->n <= PROMPT_SHORT_RUN || g >= 2 * nkv * (Q_ATT_D / 32)) return;
  bool is_v = g >= nkv * (Q_ATT_D / 32);
  uint b = is_v ? g - nkv * (Q_ATT_D / 32) : g;  // block within the position's nkv heads
  for (uint pos = get_group_id(1); pos < end; pos += get_num_groups(1)) {
    __global const uchar *blk = (is_v ? vc : kc) + kv_cell(kv_page(pt, npp, pp->slot, pos), pos) * nkv * KV_HEAD_BYTES +
                                (b / (Q_ATT_D / 32)) * KV_HEAD_BYTES + (b % (Q_ATT_D / 32)) * Q8_BLOCK;
    float d = vload_half(0, (__global const half *)blk);
    __global half *out = (is_v ? vh : kh) + ((ulong)pos * nkv) * Q_ATT_D + 32 * b;
    for (uint i = 0; i < 32; i++) out[i] = convert_half(d * (float)((__global const char *)blk)[2 + i]);
  }
}

// ---------------------------------------------------------------------------------------------------------------
// The MoE block

// The router per row, as layer.cl's, by one sub-group of 32 per row: rl[t] holds the 512 router logits then the
// shared expert's gate. Writes ids[t] and w[t] (Q_USED each). The softmax's sum adds the sub-group's sums of each 32
// logits in order. Ties in the top 10 go to the lower expert, -0 and +0 being equal. Groups (ceil(T / 4)), 128.
__attribute__((intel_reqd_sub_group_size(32)))
__kernel void p_router(__global const float *rl, __global const PromptPlan *pp, __global int *ids_all,
                       __global float *w_all) {
  __local uint2 cand[4 * RT_SLM];
  const uint lane = sg_lane(), s = sg_id(), t = get_group_id(0) * 4 + s;
  ROW_OR_RETURN(t)
  __global const float *logits = rl + (ulong)t * (Q_EXPERTS + 1);
  float v[RT_PER];
  float mx = -INFINITY;
  for (uint j = 0; j < RT_PER; j++) {
    v[j] = logits[lane + RT_SG * j];
    mx = fmax(mx, v[j]);
  }
  mx = sub_group_reduce_max(mx);
  float sum = 0.0f;
  for (uint j = 0; j < RT_PER; j++) sum += sub_group_reduce_add(exp(v[j] - mx));
  uint key[RT_PER];
  for (uint j = 0; j < RT_PER; j++) key[j] = topk_key(v[j] + 0.0f);  // -0 as +0
  const uint2 win = rt_top(key, cand + s * RT_SLM);
  const float pk = lane < Q_USED ? exp(rt_value(win.x) - mx) / sum : 0.0f;
  float tot = 0.0f;
  for (uint k = 0; k < Q_USED; k++) tot += sub_group_broadcast(pk, k);
  if (lane < Q_USED) {
    ids_all[t * Q_USED + lane] = (int)win.y;
    w_all[t * Q_USED + lane] = pk / fmax(tot, 6.1035e-5f);
  }
}

// Counts each expert's assignments. cnt (Q_EXPERTS) is zeroed by p_moe_scan of the chunk before, and at load.
// Groups ceil(T * Q_USED / 64), 64.
__kernel void p_moe_count(__global const int *ids, __global const PromptPlan *pp, __global uint *cnt) {
  uint a = get_global_id(0);
  if (a >= pp->n * Q_USED) return;
  atomic_inc(&cnt[ids[a]]);
}

// One work-group of Q_EXPERTS: each expert's first row (base, an exclusive scan of the counts in expert order),
// and its tiles of MOE_TILE rows, into the list of cached experts (slot_map[e] >= 0, the cache slot) or of staged
// experts (stage_map[e] >= 0, the staging slot). Resets the counts and sets the fill cursors.
// With prof (the routing profile being gathered), adds the counts to it.
__kernel void p_moe_scan(__global uint *cnt, __global const int *slot_map, __global const int *stage_map,
                         __global uint *base, __global uint *cursor, __global MoeTile *tiles_c,
                         __global uint *ntiles_c, __global MoeTile *tiles_s, __global uint *ntiles_s,
                         __global uint *prof) {
  __local uint sc[Q_EXPERTS], tc[Q_EXPERTS], ts[Q_EXPERTS];
  uint e = get_local_id(0);
  uint n = cnt[e];
  uint nt = (n + MOE_TILE - 1) / MOE_TILE;
  bool cached = slot_map[e] >= 0;
  sc[e] = n;
  tc[e] = cached ? nt : 0;
  ts[e] = cached ? 0 : nt;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint off = 1; off < Q_EXPERTS; off <<= 1) {
    uint a = e >= off ? sc[e - off] : 0, b = e >= off ? tc[e - off] : 0, c = e >= off ? ts[e - off] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    sc[e] += a;
    tc[e] += b;
    ts[e] += c;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  uint first = sc[e] - n;
  if (prof) prof[e] += n;
  base[e] = first;
  cursor[e] = 0;
  cnt[e] = 0;
  uint tile0 = cached ? tc[e] - nt : ts[e] - nt;
  __global MoeTile *list = cached ? tiles_c : tiles_s;
  uint slot = (uint)(cached ? slot_map[e] : stage_map[e]);
  for (uint i = 0; i < nt; i++)
    list[tile0 + i] = (MoeTile){(uint)e, slot, first + i * MOE_TILE, min((uint)MOE_TILE, n - i * MOE_TILE)};
  if (e == Q_EXPERTS - 1) {
    *ntiles_c = tc[e];
    *ntiles_s = ts[e];
  }
}

// Each assignment a = t * Q_USED + k takes a row of its expert's run: inv[a] = row. The order within an expert's run
// depends on scheduling, which changes no result: every row is computed alone. Groups ceil(T * Q_USED / 64), 64.
__kernel void p_moe_fill(__global const int *ids, __global const PromptPlan *pp, __global const uint *base,
                         __global uint *cursor, __global uint *inv) {
  uint a = get_global_id(0);
  if (a >= pp->n * Q_USED) return;
  uint e = (uint)ids[a];
  inv[a] = base[e] + atomic_inc(&cursor[e]);
}

// xg[inv[t * Q_USED + k]] = x16[t]: the activations in expert order; with sigma, each 4 values permuted as the expert
// kernels' weight pairs are (shared/quant.h, quant_pairs_sigma). Scattering a token's row beats gathering rows of xg
// (docs/measurements/MEASUREMENTS.md#2026-10-15-prompt-moe-scatter), and 256-byte block moves beat 16 bytes a work-item
// (docs/measurements/MEASUREMENTS.md#2026-10-15-prompt-moe-scatter-blocks). Groups (1, T), Q_EMBD / 8: 128 values a sub-group.
REQD_SG __kernel void p_moe_scatter(__global const half *x16, __global const uint *inv, __global const PromptPlan *pp,
                                    uint sigma, __global half *xg) {
  uint t = get_group_id(1), lane = sg_lane(), at = 128 * sg_id();
  ROW_OR_RETURN(t)
  ushort8 v = sg_read_us8((__global const ushort *)(x16 + (ulong)t * Q_EMBD) + at);
  // Lane i holds values 16 j + i: sigma swaps the middle two of every 4 lanes.
  if (sigma) v = sg_shuffle_us8(v, (lane & ~3u) | ((lane & 1u) << 1) | ((lane >> 1) & 1u));
  uint rows[Q_USED];
  for (uint k = 0; k < Q_USED; k++) rows[k] = inv[t * Q_USED + k];
  for (uint k = 0; k < Q_USED; k++) sg_write_us8((__global ushort *)(xg + (ulong)rows[k] * Q_EMBD) + at, v);
}

// h16[t] = silu(g[t]) * u[t] for the shared expert, ff values per row. Groups (ceil(ff / 64), T), 64.
__kernel void p_swiglu(__global const float *g, __global const float *u, uint ff, __global const PromptPlan *pp,
                       __global half *h16) {
  uint i = get_global_id(0), t = get_group_id(1);
  ROW_OR_RETURN(t)
  if (i >= ff) return;
  h16[(ulong)t * ff + i] = convert_half(silu(g[(ulong)t * ff + i]) * u[(ulong)t * ff + i]);
}

// This card's partial sum of the block for row t: the routed experts' outputs y (fp16, in expert order, found through
// inv) weighted in rank order, then the shared expert's s gated by sigma(rl[t][512]); as fp16 into out, and into out2
// when set (the other card's buffer). Two values a work-item, so that a sub-group's store is 64 bytes, which the other
// card takes at full rate (docs/measurements/MEASUREMENTS.md#2026-10-15-prompt-exchange-stores). Groups (Q_EMBD / 128, T), 64.
__kernel void p_moe_combine(__global const half *y, __global const uint *inv, __global const float *w,
                            __global const float *s, __global const float *rl, __global const PromptPlan *pp,
                            __global half *out, __global half *out2) {
  uint i = 2 * get_global_id(0), t = get_group_id(1);
  ROW_OR_RETURN(t)
  float a0 = 0.0f, a1 = 0.0f;
  for (uint k = 0; k < Q_USED; k++) {
    const float wk = w[t * Q_USED + k];
    const ulong at = (ulong)inv[t * Q_USED + k] * Q_EMBD + i;
    a0 += wk * vload_half(at, y);
    a1 += wk * vload_half(at + 1, y);
  }
  const float g = sigmoid(rl[(ulong)t * (Q_EXPERTS + 1) + Q_EXPERTS]);
  a0 += g * s[(ulong)t * Q_EMBD + i];
  a1 += g * s[(ulong)t * Q_EMBD + i + 1];
  vstore_half2((float2)(a0, a1), 0, out + (ulong)t * Q_EMBD + i);
  if (out2) vstore_half2((float2)(a0, a1), 0, out2 + (ulong)t * Q_EMBD + i);
}

// ---------------------------------------------------------------------------------------------------------------
// The PLE module on layer 1, per row. pk[t] = W_k u[t] (Q_HC_DIM), pv[t] = W_v u[t] (Q_EMBD).

// Groups (Q_HC, T), 256.
__kernel void p_ple_gate(__global const float *k, __global const float *R, __global const float *gk,
                         __global const float *gq, __global const PromptPlan *pp, __global float *gates) {
  __local float red[64];
  uint s = get_group_id(0), t = get_group_id(1);
  ROW_OR_RETURN(t)
  __global const float *ks = k + (ulong)t * Q_HC_DIM + s * Q_EMBD;
  __global const float *rs = R + (ulong)t * Q_HC_DIM + s * Q_EMBD;
  float kk = 0.0f, rr = 0.0f;
  for (uint i = get_local_id(0); i < Q_EMBD; i += get_local_size(0)) {
    kk += ks[i] * ks[i];
    rr += rs[i] * rs[i];
  }
  kk = rsqrt(wg_sum(kk, red) / (float)Q_EMBD + Q_EPS);
  rr = rsqrt(wg_sum(rr, red) / (float)Q_EMBD + Q_EPS);
  float dot = 0.0f;
  for (uint i = get_local_id(0); i < Q_EMBD; i += get_local_size(0))
    dot += ks[i] * kk * gk[s * Q_EMBD + i] * rs[i] * rr * gq[s * Q_EMBD + i];
  float a = wg_sum(dot, red) * rsqrt((float)Q_EMBD);
  if (get_local_id(0) == 0) gates[t * Q_HC + s] = sigmoid(sign(a) * sqrt(fmax(fabs(a), 1e-6f)));
}

// Groups (Q_HC, T), 256.
__kernel void p_ple_norm(__global const float *v, __global const float *gates, __global const float *gc,
                         __global const PromptPlan *pp, __global float *N) {
  __local float red[64];
  uint s = get_group_id(0), t = get_group_id(1);
  ROW_OR_RETURN(t)
  float gate = gates[t * Q_HC + s];
  __global const float *vr = v + (ulong)t * Q_EMBD;
  float ss = 0.0f;
  for (uint i = get_local_id(0); i < Q_EMBD; i += get_local_size(0)) ss += gate * vr[i] * gate * vr[i];
  float r = rsqrt(wg_sum(ss, red) / (float)Q_EMBD + Q_EPS);
  for (uint i = get_local_id(0); i < Q_EMBD; i += get_local_size(0))
    N[(ulong)t * Q_HC_DIM + s * Q_EMBD + i] = gate * vr[i] * r * gc[s * Q_EMBD + i];
}

// The dilated convolution over N at positions p-9, p-6, p-3, p (rows of the chunk, or the history before it), then
// R[t] += G + silu(C). Groups (Q_HC_DIM / 64, T), 64.
__kernel void p_ple_conv(__global const float *N, __global const float *v, __global const float *gates,
                         __global const float *w, __global const float *hist, ulong slot_stride,
                         __global const PromptPlan *runs, __global float *R) {
  uint ch = get_global_id(0), t = get_group_id(1);
  RUN_OF_ROW(t)
  const ulong r0 = pp->row0;
  N += r0 * Q_HC_DIM;
  v += r0 * Q_EMBD;
  gates += r0 * Q_HC;
  R += r0 * Q_HC_DIM;
  uint pos = pp->pos0 + t;
  __global const float *h = hist + pp->slot * slot_stride;
  float nv[4];
  for (uint j = 0; j < 3; j++) {
    uint d = Q_PLE_DIL * (3 - j);
    nv[j] = t >= d ? N[(ulong)(t - d) * Q_HC_DIM + ch] : h[((pos + Q_PLE_HIST - d) % Q_PLE_HIST) * Q_HC_DIM + ch];
  }
  nv[3] = N[(ulong)t * Q_HC_DIM + ch];
  __global const float *k = w + ch * Q_GDN_CONV;
  float acc = k[0] * nv[0];
  acc += k[1] * nv[1];
  acc += k[2] * nv[2];
  acc += k[3] * nv[3];
  R[(ulong)t * Q_HC_DIM + ch] += gates[t * Q_HC + ch / Q_EMBD] * v[(ulong)t * Q_EMBD + ch % Q_EMBD] + silu(acc);
}

// The history keeps N at position p in slot p % 9: the chunk's last nine rows go in. Groups Q_HC_DIM / 64, 64.
__kernel void p_ple_save(__global const float *N, __global float *hist, ulong slot_stride,
                         __global const PromptPlan *runs) {
  RUN_OF_GROUP()
  uint ch = get_global_id(0);
  uint n = pp->n;
  __global float *h = hist + pp->slot * slot_stride;
  N += (ulong)pp->row0 * Q_HC_DIM;
  for (uint t = n > Q_PLE_HIST ? n - Q_PLE_HIST : 0; t < n; t++)
    h[((pp->pos0 + t) % Q_PLE_HIST) * Q_HC_DIM + ch] = N[(ulong)t * Q_HC_DIM + ch];
}
