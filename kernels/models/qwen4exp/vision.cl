// The image encoder's own kernels (docs/spec/MODEL_VISION.md, sections 2 and 3): the steps between its products,
// which are gemm_bf16's, in f32 except where a product or attention reads 16-bit values. v_embed embeds the patches;
// v_norm, v_qkv and v_attn make a layer's attention, v_bias_gelu its MLP's GELU, and v_residual adds to the residual;
// v_bias gives the merger's output.
#include "builtins.h"

#define V_D 1152     // the residual's width
#define V_H 16       // heads
#define V_HD 72      // a head's dimensions
#define V_ROT 36     // rotated pairs per head: (k, k + 36)
#define V_SEC 18     // pairs turned by the row; the next 18 by the column
#define V_PATCH 768  // 3 channels x 16 x 16 pixels: one frame of a patch
#define V_GRID 48    // the position table's side

#ifndef V_HDP
#define V_HDP 80     // a head's dimensions in the fp16 Q, K and V, padded with zeros to a multiple of 16
#endif
#define V_QKV_W (V_H * V_HDP)

// A patch's row and column in the image, from its place in the encoder's order: each 2 x 2 window's four patches in
// a row, windows row-major over the h/2 x w/2 grid (section 2.2). gw: the image's width in patches.
static inline uint2 patch_rc(uint n, uint gw) {
  const uint win = n / 4, R = win / (gw / 2), C = win % (gw / 2);
  return (uint2)(2 * R + (n % 4) / 2, 2 * C + n % 2);
}

#define V_EMB_PATCHES 8  // patches per work-group, sharing the weights' reads
#define V_EMB_WI 128     // work-items per work-group; each takes V_D / V_EMB_WI outputs

// e = sum over frames f of W_f p_f, + b + pos(row, col) (sections 2.3 and 2.4). pix: n patches of frames x 768 bytes,
// each frame channel, row, column, in the encoder's order, from patch first of an image of gh x gw patches. wt: frames
// matrices in f32, transposed, wt[f][k][o] (k < 768, o < 1,152); a still image's one is W0 + W1. pos_tab: the 48 x 48
// table, [y][x][1,152]. x: n rows of 1,152. Groups (ceil(n / 8)), 128 work-items.
__kernel void v_embed(__global const uchar *pix, uint n, uint first, uint gh, uint gw, uint frames,
                      __global const float *wt, __global const float *bias, __global const float *pos_tab,
                      __global float *x) {
  __local float p[V_EMB_PATCHES][V_PATCH];
  const uint p0 = V_EMB_PATCHES * get_group_id(0), li = get_local_id(0);
  const uint np = min((uint)V_EMB_PATCHES, n - p0);
  float acc[V_D / V_EMB_WI][V_EMB_PATCHES];
  for (uint o = 0; o < V_D / V_EMB_WI; o++)
    for (uint j = 0; j < V_EMB_PATCHES; j++) acc[o][j] = 0.0f;
  for (uint f = 0; f < frames; f++) {
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint i = li; i < V_EMB_PATCHES * V_PATCH; i += V_EMB_WI) {
      const uint j = i / V_PATCH, k = i % V_PATCH;
      // As llama.cpp: v / 255, then (v - mean) / std with mean and std 0.5 (tools/mtmd/clip-impl.h:726-735).
      p[j][k] = j < np ? ((float)pix[((ulong)(p0 + j) * frames + f) * V_PATCH + k] / 255.0f - 0.5f) / 0.5f : 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    __global const float *wf = wt + (ulong)f * V_PATCH * V_D;
    for (uint k = 0; k < V_PATCH; k++) {
      float pk[V_EMB_PATCHES];
      for (uint j = 0; j < V_EMB_PATCHES; j++) pk[j] = p[j][k];
      for (uint o = 0; o < V_D / V_EMB_WI; o++) {
        const float w = wf[(ulong)k * V_D + li + V_EMB_WI * o];
        for (uint j = 0; j < V_EMB_PATCHES; j++) acc[o][j] = fma(w, pk[j], acc[o][j]);
      }
    }
  }
  for (uint j = 0; j < np; j++) {
    const uint2 rc = patch_rc(first + p0 + j, gw);
    // Bilinear with aligned corners; at 48 x 48 the table as it is.
    const float fy = gh > 1 ? (float)rc.x * (V_GRID - 1) / (float)(gh - 1) : 0.0f;
    const float fx = gw > 1 ? (float)rc.y * (V_GRID - 1) / (float)(gw - 1) : 0.0f;
    const uint y0 = min((uint)fy, (uint)V_GRID - 1), x0 = min((uint)fx, (uint)V_GRID - 1);
    const uint y1 = min(y0 + 1, (uint)V_GRID - 1), x1 = min(x0 + 1, (uint)V_GRID - 1);
    const float dy = fy - (float)y0, dx = fx - (float)x0;
    __global const float *t00 = pos_tab + (ulong)(y0 * V_GRID + x0) * V_D, *t01 = pos_tab + (ulong)(y0 * V_GRID + x1) * V_D;
    __global const float *t10 = pos_tab + (ulong)(y1 * V_GRID + x0) * V_D, *t11 = pos_tab + (ulong)(y1 * V_GRID + x1) * V_D;
    for (uint o = 0; o < V_D / V_EMB_WI; o++) {
      const uint c = li + V_EMB_WI * o;
      const float pe = (1.0f - dy) * (1.0f - dx) * t00[c] + (1.0f - dy) * dx * t01[c] + dy * (1.0f - dx) * t10[c] +
                       dy * dx * t11[c];
      x[(ulong)(p0 + j) * V_D + c] = acc[o][j] + bias[c] + pe;
    }
  }
}

// out[t][i] = bf16(LayerNorm_1e-6(x[t]) * g + b), t < n. One sub-group per row: groups (n), 16 work-items.
REQD_SG __kernel void v_norm(__global const float *x, uint n, __global const float *g, __global const float *b,
                             __global ushort *out) {
  const uint t = get_group_id(0), lane = sg_lane();
  if (t >= n) return;
  __global const float *xr = x + (ulong)t * V_D;
  float s = 0.0f;
  for (uint i = lane; i < V_D; i += 16) s += xr[i];
  const float mean = sg_sum(s) / V_D;
  float v = 0.0f;
  for (uint i = lane; i < V_D; i += 16) {
    const float d = xr[i] - mean;
    v += d * d;
  }
  const float inv = 1.0f / sqrt(sg_sum(v) / V_D + 1e-6f);
  for (uint i = lane; i < V_D; i += 16) out[(ulong)t * V_D + i] = f32_to_bf16((xr[i] - mean) * inv * g[i] + b[i]);
}

// y: n rows of the QKV product, 3,456 values (Q, K, V, 16 heads of 72 each), before bias. Rows are patches first ..
// first + n - 1 of an image gw patches wide. q, k, v: n rows of 16 heads x V_HDP fp16 values, zero past 72; Q and K
// turned by section 3.2's rule. Groups (V_H * V_ROT / 64, n), 64 work-items: one per head and pair.
__kernel void v_qkv(__global const float *y, __global const float *bias, uint n, uint first, uint gw,
                    __global half *q, __global half *k, __global half *v) {
  const uint i = get_global_id(0), t = get_global_id(1);
  if (i >= V_H * V_ROT || t >= n) return;
  const uint h = i / V_ROT, j = i % V_ROT;
  const uint2 rc = patch_rc(first + t, gw);
  const float p = (float)(j < V_SEC ? rc.x : rc.y);
  const float theta = p * pow(10000.0f, -2.0f * (float)(j % V_SEC) / (float)V_ROT);
  const float cs = cos(theta), sn = sin(theta);
  __global const float *yr = y + (ulong)t * 3 * V_D;
  const ulong o = (ulong)t * V_QKV_W + h * V_HDP;
  for (uint part = 0; part < 3; part++) {
    const uint a = part * V_D + h * V_HD + j;
    const float x0 = yr[a] + bias[a], x1 = yr[a + V_ROT] + bias[a + V_ROT];
    __global half *dst = part == 0 ? q : part == 1 ? k : v;
    if (part < 2) {
      vstore_half(x0 * cs - x1 * sn, o + j, dst);
      vstore_half(x0 * sn + x1 * cs, o + j + V_ROT, dst);
    } else {
      vstore_half(x0, o + j, dst);
      vstore_half(x1, o + j + V_ROT, dst);
    }
    if (j < V_HDP - V_HD) vstore_half(0.0f, o + V_HD + j, dst);
  }
}

// o = softmax(scale q k^T) v per head: nq query rows, all npos keys, no mask. q, k, v: rows of 16 heads x V_HDP fp16
// (v_qkv's layout), 64-byte aligned; o: nq rows of 1,152 bf16, head-major. Sums in f32, scaled after; P to fp16 for
// P V (llama.cpp sums P V in fp16: the spec's section 7). A sub-group takes 16 query rows of one head, a lane a row,
// as S^T = K Q^T and O^T += V^T P^T: no shuffles in the softmax. Groups (V_H, ceil(nq / 16)), 16; 256 registers.
REQD_SG __kernel void v_attn(__global const half *q, uint nq, __global const half *k, __global const half *v,
                             uint npos, float scale, __global ushort *o) {
  const uint h = get_group_id(0), t0 = 16 * get_group_id(1), lane = sg_lane();
  if (t0 >= nq) return;
  const uint pitch = V_QKV_W * 2, x0 = h * V_HDP;
  // Q^T, the B operand: lane i is query t0 + i, qb[d][p] its dims 16 d + 2 p and 16 d + 2 p + 1.
  int8 qb[V_HDP / 16];
  for (uint d = 0; d < V_HDP / 16; d++) {
    uint t[8];
    intel_sub_group_2d_block_read_transpose_32b_16r8x1c((__global void *)q, pitch, nq, pitch,
                                                        (int2)(x0 / 2 + 8 * d, t0), t);
    qb[d] = as_int8(vload8(0, t));
  }
  // O^T by blocks of 16 dims: ae[j][m] is dim 16 j + 2 m of this lane's row, ao[j][m] dim 16 j + 2 m + 1.
  float8 ae[V_HDP / 16], ao[V_HDP / 16];
  for (uint j = 0; j < V_HDP / 16; j++) ae[j] = ao[j] = (float8)0.0f;
  float mrow = -INFINITY, lrow = 0.0f;
  for (uint key0 = 0; key0 < npos; key0 += 16) {
    // s0[m] = score of key key0 + m, s1[m] of key key0 + 8 + m, for this lane's row.
    float8 s0 = (float8)0.0f, s1 = (float8)0.0f;
    __attribute__((opencl_unroll_hint)) for (uint d = 0; d < V_HDP / 16; d++) {
      ushort a0[8], a1[8];
      intel_sub_group_2d_block_read_16b_8r16x1c((__global void *)k, pitch, npos, pitch, (int2)(x0 + 16 * d, key0), a0);
      intel_sub_group_2d_block_read_16b_8r16x1c((__global void *)k, pitch, npos, pitch, (int2)(x0 + 16 * d, key0 + 8),
                                                a1);
      s0 = dpas_f16(as_short8(vload8(0, a0)), qb[d], s0);
      s1 = dpas_f16(as_short8(vload8(0, a1)), qb[d], s1);
    }
    s0 *= scale;
    s1 *= scale;
    if (key0 + 16 > npos) {
      float sm[16];
      vstore8(s0, 0, sm);
      vstore8(s1, 1, sm);
      for (uint m = 0; m < 16; m++)
        if (key0 + m >= npos) sm[m] = -INFINITY;
      s0 = vload8(0, sm);
      s1 = vload8(1, sm);
    }
    const float8 mx8 = fmax(s0, s1);
    const float4 mx4 = fmax(mx8.lo, mx8.hi);
    const float mn = fmax(mrow, fmax(fmax(mx4.x, mx4.y), fmax(mx4.z, mx4.w)));
    const float alpha = exp(mrow - mn);
    const float8 p0 = exp(s0 - mn), p1 = exp(s1 - mn);
    const float8 ps = p0 + p1;
    const float4 ps4 = ps.lo + ps.hi;
    lrow = lrow * alpha + (ps4.x + ps4.y + ps4.z + ps4.w);
    mrow = mn;
    // P^T, the B operand: pairs of keys (2 p, 2 p + 1) for this lane's row.
    const half8 h0 = convert_half8(p0), h1 = convert_half8(p1);
    const int8 pb = as_int8((half16)(h0, h1));
    __attribute__((opencl_unroll_hint)) for (uint j = 0; j < V_HDP / 16; j++) {
      // V^T, the A operand: lane i is key key0 + i, vt[m] its dims 16 j + 2 m and 16 j + 2 m + 1.
      uint vt[8];
      intel_sub_group_2d_block_read_transpose_32b_16r8x1c((__global void *)v, pitch, npos, pitch,
                                                          (int2)(x0 / 2 + 8 * j, key0), vt);
      const uint8 w = vload8(0, vt);
      ae[j] = dpas_f16(as_short8(convert_ushort8(w & 0xffffu)), pb, ae[j] * alpha);
      ao[j] = dpas_f16(as_short8(convert_ushort8(w >> 16)), pb, ao[j] * alpha);
    }
  }
  const uint t = t0 + lane;
  if (t >= nq) return;
  const float inv = 1.0f / lrow;
  __global uint *row = (__global uint *)(o + (ulong)t * V_D + h * V_HD);
  for (uint j = 0; j < V_HDP / 16; j++) {
    float e[8], d[8];
    vstore8(ae[j] * inv, 0, e);
    vstore8(ao[j] * inv, 0, d);
    for (uint m = 0; m < 8; m++)
      if (16 * j + 2 * m < V_HD) row[8 * j + m] = (uint)f32_to_bf16(e[m]) | (uint)f32_to_bf16(d[m]) << 16;
  }
}

// ggml's GELU on the CPU (section 3.4): 0 at or below -10, x at or above 10, else the table's value at x rounded to
// fp16. The host fills the table as ggml does, with glibc's tanhf: the card's tanh differs by a few ulps, and for
// negative x the sum 1 + tanh cancels enough for that to change the fp16 result.
static inline float gelu_ref(float x, __global const half *table) {
  if (x <= -10.0f) return 0.0f;
  if (x >= 10.0f) return x;
  return vload_half(as_ushort(convert_half_rte(x)), table);
}

// out[t][i] = bf16(GELU(y[t][i] + b[i])), i < cols, t < n; rows ldy and ldo values apart, out zero from cols to ldo.
// table is the GELU's 65,536 fp16 values, by the bits of x as fp16. One work-item per value of out: global (ldo, n).
__kernel void v_bias_gelu(__global const float *y, uint ldy, __global const float *b, uint cols, uint n,
                          __global ushort *out, uint ldo, __global const half *table) {
  const uint i = get_global_id(0), t = get_global_id(1);
  if (i >= ldo || t >= n) return;
  out[(ulong)t * ldo + i] = i < cols ? f32_to_bf16(gelu_ref(y[(ulong)t * ldy + i] + b[i], table)) : 0;
}

// x[t][i] += y[t][i] + b[i], i < 1,152, t < n. Global (1,152, n).
__kernel void v_residual(__global float *x, __global const float *y, __global const float *b, uint n) {
  const uint i = get_global_id(0), t = get_global_id(1);
  if (i >= V_D || t >= n) return;
  x[(ulong)t * V_D + i] += y[(ulong)t * V_D + i] + b[i];
}

// out[t][i] = y[t][i] + b[i], i < cols, t < n; rows cols values apart. Global (cols, n).
__kernel void v_bias(__global const float *y, __global const float *b, uint cols, uint n, __global float *out) {
  const uint i = get_global_id(0), t = get_global_id(1);
  if (i >= cols || t >= n) return;
  out[(ulong)t * cols + i] = y[(ulong)t * cols + i] + b[i];
}
