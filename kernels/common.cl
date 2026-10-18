// Small kernels every model uses.
#include "a8.h"
#include "builtins.h"

// a8 activations (a8.h): x[t * x_stride + i] for i < K, t < ntok, into rows at out + t * (K + K / 8). One sub-group
// per 32 values, two per lane: groups ceil(K / 32 ntok / 4), 64.
REQD_SG __kernel void quantize_a8(__global const float *x, uint K, uint x_stride, uint ntok, __global uchar *out) {
  uint units = K / 32;
  uint g = get_group_id(0) * 4 + sg_id();
  if (g >= units * ntok) return;
  uint t = g / units, u = g % units;
  a8_store(vload2(sg_lane(), x + (ulong)t * x_stride + 32 * u), out + (ulong)t * (K + K / 8), K, u);
}

#include "topk.h"

static inline float silu(float x) { return x / (1.0f + exp(-x)); }
static inline float sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }

// y[t * y_stride + r] = post(scale * W[r] . x_t) for a BF16 matrix of rows x K (K a multiple of 8 lpr) and f32
// activations, t < ntok <= 8. post is the identity (mode 0), SiLU (1) or the sigmoid (2). lpr lanes share a row (16,
// 8, 4 or 2), so a sub-group takes 16 / lpr rows; 4 sub-groups per work-group: groups ceil(rows / (64 / lpr)), 64.
// Each lane reads 8 weights (16 bytes) at a time, and every token reuses them.
#define BF_MAXT 8
REQD_SG __kernel void matvec_bf16(__global const ushort *W, uint rows, uint K, __global const float *x, uint x_stride,
                                  uint ntok, __global float *y, uint y_stride, uint mode, float scale, uint lpr) {
  const uint lane = sg_lane(), sub = lane % lpr;
  uint r = (get_group_id(0) * 4 + sg_id()) * (SG_SIZE / lpr) + lane / lpr;
  const bool valid = r < rows;
  if (!valid) r = rows - 1;
  __global const ushort *w = W + (ulong)r * K;
  float acc[BF_MAXT];
  for (uint t = 0; t < BF_MAXT; t++) acc[t] = 0.0f;
  for (uint k = 8 * sub; k < K; k += 8 * lpr) {
    ushort8 wv = vload8(0, w + k);
    float8 wf = (float8)(bf16_to_f32(wv.s0), bf16_to_f32(wv.s1), bf16_to_f32(wv.s2), bf16_to_f32(wv.s3),
                         bf16_to_f32(wv.s4), bf16_to_f32(wv.s5), bf16_to_f32(wv.s6), bf16_to_f32(wv.s7));
    for (uint t = 0; t < BF_MAXT; t++) {
      if (t >= ntok) break;
      float8 xv = vload8(0, x + (ulong)t * x_stride + k);
      acc[t] += dot(wf.lo, xv.lo) + dot(wf.hi, xv.hi);
    }
  }
  for (uint t = 0; t < BF_MAXT; t++) {
    if (t >= ntok) break;
    float a = acc[t];
    for (uint m = 1; m < lpr; m <<= 1) a += sub_group_shuffle_xor(a, m);
    a *= scale;
    if (mode == 1) a = silu(a);
    if (mode == 2) a = sigmoid(a);
    if (valid && sub == 0) y[(ulong)t * y_stride + r] = a;
  }
}

// The same for few, long rows: one work-group of 256 per row splits K (a multiple of 8) and sums through local
// memory; ntok <= 8 tokens share each weight read (x and y rows x_stride and y_stride apart). post and scale apply to
// rows below post_rows only (the hyper-connections' down product carries their 4 inject rows after its 320).
// Groups rows, 256.
__kernel void matvec_bf16_ks(__global const ushort *W, uint rows, uint K, __global const float *x, uint x_stride,
                             __global const uint *ntok_p, __global float *y, uint y_stride, uint mode, float scale,
                             uint post_rows) {
  __local float red[BF_MAXT][16];
  const uint r = get_group_id(0), lid = get_local_id(0), ntok = *ntok_p;
  __global const ushort *w = W + (ulong)r * K;
  float acc[BF_MAXT];
  for (uint t = 0; t < BF_MAXT; t++) acc[t] = 0.0f;
  for (uint k = 8 * lid; k < K; k += 8 * 256) {
    ushort8 wv = vload8(0, w + k);
    float8 wf = (float8)(bf16_to_f32(wv.s0), bf16_to_f32(wv.s1), bf16_to_f32(wv.s2), bf16_to_f32(wv.s3),
                         bf16_to_f32(wv.s4), bf16_to_f32(wv.s5), bf16_to_f32(wv.s6), bf16_to_f32(wv.s7));
    for (uint t = 0; t < BF_MAXT; t++) {
      if (t >= ntok) break;
      float8 xv = vload8(0, x + (ulong)t * x_stride + k);
      acc[t] += dot(wf.lo, xv.lo) + dot(wf.hi, xv.hi);
    }
  }
  for (uint t = 0; t < BF_MAXT; t++) {
    if (t >= ntok) break;
    float a = sg_sum(acc[t]);
    if (sg_lane() == 0) red[t][sg_id()] = a;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid < ntok) {
    float a = 0.0f;
    for (uint i = 0; i < get_num_sub_groups(); i++) a += red[lid][i];
    if (r < post_rows) {
      a *= scale;
      if (mode == 1) a = silu(a);
      if (mode == 2) a = sigmoid(a);
    }
    y[(ulong)lid * y_stride + r] = a;
  }
}

// x's rows as bf16, for mv_xmx_bf16: xb[t][i] = bf16(x[t * x_stride + i]), i < K, t < *ntok (rows K apart).
// Groups (K / 64, width), 64.
__kernel void to_bf16_rows(__global const float *x, uint x_stride, uint K, __global const uint *ntok,
                           __global ushort *xb) {
  uint i = get_global_id(0), t = get_group_id(1);
  if (t >= *ntok || i >= K) return;
  xb[(ulong)t * K + i] = f32_to_bf16(x[(ulong)t * x_stride + i]);
}

// y[t * y_stride + n] = post(scale * W[n] . xb[t]) on the XMX units, t < *ntok <= 8; post (mode as matvec_bf16_ks)
// applies to n < post_rows, whose values also go to yb as bf16 (rows post_rows apart) when yb is set. xb: 8 bf16 rows
// of K, all read (to_bf16_rows); W: N rows ldw apart. W, xb 64-byte aligned; K, ldw multiples of 32. Groups
// ceil(N / 16), 16 ksplit (ksplit <= 64, K / ksplit a multiple of 16); sub-group 0 adds the ksplit tiles in order.
REQD_SG __kernel void mv_xmx_bf16(__global const ushort *W, uint N, uint K, uint ldw, __global const ushort *xb,
                                  __global const uint *ntok_p, __global float *y, uint y_stride, uint mode,
                                  float scale, uint post_rows, uint ksplit, __global ushort *yb) {
  __local float red[64][8][SG_SIZE];
  const uint lane = sg_lane(), s = sg_id(), n0 = get_group_id(0) * 16, ntok = *ntok_p;
  const uint kc = K / ksplit, k0 = s * kc;
  float8 acc = (float8)0.0f;
  for (uint k = k0; k < k0 + kc; k += 16) {
    ushort a[8];
    intel_sub_group_2d_block_read_16b_8r16x1c((__global void *)xb, K * 2, 8, K * 2, (int2)(k, 0), a);
    uint b[8];
    sg_load_cols16(W, K * 2, N, ldw * 2, k / 2, n0, b);
    acc = dpas_bf16(as_short8(vload8(0, a)), as_int8(vload8(0, b)), acc);
  }
  float r[8];
  vstore8(acc, 0, r);
  for (uint m = 0; m < 8; m++) red[s][m][lane] = r[m];
  barrier(CLK_LOCAL_MEM_FENCE);
  if (s != 0) return;
  const uint n = n0 + lane;
  for (uint m = 0; m < 8 && m < ntok; m++) {
    float v = 0.0f;
    for (uint q = 0; q < ksplit; q++) v += red[q][m][lane];
    if (n < post_rows) {
      v *= scale;
      if (mode == 1) v = silu(v);
      if (mode == 2) v = sigmoid(v);
    }
    if (n < N) y[(ulong)m * y_stride + n] = v;
    if (yb && n < post_rows) yb[(ulong)m * post_rows + n] = f32_to_bf16(v);
  }
}

// topk for rows: row r of v (v_stride apart) into idx and val at r * k, its count at count[r], for r < *nrows.
// Groups (rows), TOPK_WG.
REQD_SG __kernel void topk_rows(__global const float *v, uint v_stride, uint n, uint k, uint offset,
                                __global const uint *nrows, __global uint *idx, __global float *val,
                                __global uint *count) {
  __local uint hist[TOPK_HIST];
  __local uint scan[TOPK_WG];
  __local ulong cand[TOPK_CAND];
  uint r = get_group_id(0);
  if (r >= *nrows) return;
  uint got = topk_select_few(v + (ulong)r * v_stride, n, k, offset, idx + (ulong)r * k, val + (ulong)r * k, hist, scan,
                             cand);
  if (get_local_id(0) == 0) count[r] = got;
}

// out[i] = silu(g[i]) * u[i].
__kernel void swiglu(__global const float *g, __global const float *u, __global float *out, uint n) {
  uint i = get_global_id(0);
  if (i < n) out[i] = silu(g[i]) * u[i];
}

// The k largest of v[0..n) into idx and val (indices increasing), and their count into *count. One work-group of
// TOPK_WG work-items. offset is added to every index written.
REQD_SG __kernel void topk(__global const float *v, uint n, uint k, uint offset, __global uint *idx,
                           __global float *val, __global uint *count) {
  __local uint hist[TOPK_HIST];
  __local uint scan[TOPK_WG];
  __local ulong cand[TOPK_CAND];
  uint got = topk_select_few(v, n, k, offset, idx, val, hist, scan, cand);
  if (get_local_id(0) == 0) *count = got;
}
