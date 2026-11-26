// The image encoder's own kernels (docs/spec/MODEL_VISION.md, sections 2 and 3): the steps between its products,
// which are gemm_bf16's, in f32 except where a product or attention reads 16-bit values. v_embed embeds the patches;
// v_norm, v_qkv and v_attn make a layer's attention, v_bias_gelu its MLP's GELU, and v_residual adds to the residual;
// v_bias gives the merger's output.
#include "builtins.h"
#include "vision_sizes.h"

#if V_MERGE != 2
#error "patch_rc and the merge windows take 2 x 2 patches"
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
      // As llama.cpp: v / 255, then (v - mean) / std (tools/mtmd/clip-impl.h:726-735).
      p[j][k] = j < np ? ((float)pix[((ulong)(p0 + j) * frames + f) * V_PATCH + k] / 255.0f - V_PIX_MEAN) / V_PIX_STD
                       : 0.0f;
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
  const float inv = 1.0f / sqrt(sg_sum(v) / V_D + V_EPS);
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
