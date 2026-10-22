// The prompt path's products on the XMX units: many tokens against one 16-bit weight matrix. Built as fp16 (module
// gemm_f16) and, with -DBF16, as bf16 (module gemm_bf16) for the file's BF16 weights, whose reference products take
// bf16 activations (docs/spec/MODEL_LAYER.md, section 6).
// gemm:  Y[t][n] = post(scale * sum_k X[t][k] W[n][k]), t < *ntok, n < N; also into Y2 when set.
// to_16: f32 rows into fp16 or bf16 rows, for the activations.
#include "builtins.h"

#ifdef BF16
#define DPAS dpas_bf16
#else
#define DPAS dpas_f16
#endif

// Tile sizes from docs/measurements/MEASUREMENTS.md#2026-10-15-prompt-gemm-tiles; the TM 4 modules are for a short K or N.
#ifndef TM
#define TM 8      // tiles of 8 tokens per sub-group, a multiple of 4
#endif
#ifndef TN
#define TN 2      // tiles of 16 rows per sub-group
#endif
#ifndef GEMM_SG
#define GEMM_SG 4 // sub-groups per work-group, side by side along n
#endif

static inline float post(float x, uint mode) {
  if (mode == 1) return x / (1.0f + exp(-x));
  if (mode == 2) return 1.0f / (1.0f + exp(-x));
  return x;
}

// The sub-group's tile: 8 TM tokens from t0 by 16 TN rows from n0.
static inline void gemm_acc(__global const ushort *W, uint N, uint K, uint ldw, __global const ushort *X, uint ldx,
                            uint T, uint n0, uint t0, float8 acc[TM][TN]) {
  for (uint i = 0; i < TM; i++)
    for (uint j = 0; j < TN; j++) acc[i][j] = (float8)0.0f;
  for (uint k = 0; k < K; k += 16) {
    ushort a[8 * TM];
    for (uint i = 0; i < TM / 4; i++) sg_load_rows32(X, K * 2, T, ldx * 2, k, t0 + 32 * i, a + 32 * i);
    uint b[TN][8];
    for (uint j = 0; j < TN; j++) sg_load_cols16(W, K * 2, N, ldw * 2, k / 2, n0 + 16 * j, b[j]);
    for (uint i = 0; i < TM; i++) {
      short8 av = as_short8(vload8(i, a));
      for (uint j = 0; j < TN; j++) acc[i][j] = DPAS(av, as_int8(vload8(0, b[j])), acc[i][j]);
    }
  }
}

// A work-group takes 8 TM tokens by 64 TN rows: groups (ceil(N / (64 TN)), ceil(max tokens / (8 TM))), 64 work-items.
// K a multiple of 16; W, X 64-byte aligned, rows ldw, ldx elements apart, multiples of 32. Y: f32, or fp16 if yhalf.
// Y2 (yhalf only, N a multiple of 16 TN): the other card's copy of Y, rows ldy apart, 16-byte aligned. band > 0 runs
// bands of `band` row blocks, every token block of one before the next. Neither tiles nor bands change a sum's order.
REQD_SG __kernel void gemm(__global const ushort *W, uint N, uint K, uint ldw, __global const ushort *X, uint ldx,
                           __global const uint *ntok, __global float *Y, uint ldy, uint yhalf, uint mode, float scale,
                           __global ushort *Y2, uint band) {
  const uint lane = sg_lane();
  uint gn = get_group_id(0), gt = get_group_id(1);
  // A band's weights stay in L2 (docs/measurements/MEASUREMENTS.md#2026-10-15-prompt-gemm-bands).
  if (band) {
    const uint nb = get_num_groups(0), tb = get_num_groups(1), g = gn + gt * nb;
    const uint b = g / (band * tb), r = g % (band * tb), w = min(band, nb - b * band);
    gn = b * band + r % w;
    gt = r / w;
  }
  const uint n0 = (gn * GEMM_SG + sg_id()) * (16 * TN);
  const uint t0 = gt * (8 * TM);
  const uint T = *ntok;
  if (t0 >= T || n0 >= N) return;
  float8 acc[TM][TN];
  gemm_acc(W, N, K, ldw, X, ldx, T, n0, t0, acc);
  // A tile of 8 tokens x 16 rows goes out by one 2D block store when Y's rows are 64-byte aligned; single-value
  // stores bound a short K's product.
  const uint esize = yhalf ? 2 : 4;
  const bool block2d = ((ldy * esize) % 64) == 0 && (((ulong)Y) % 64) == 0;
  for (uint i = 0; i < TM; i++) {
    ushort h[TN][8];
    for (uint j = 0; j < TN; j++) {
      uint n = n0 + 16 * j + lane;
      float r[8];
      vstore8(acc[i][j], 0, r);
      for (uint m = 0; m < 8; m++) r[m] = post(r[m] * scale, mode);
      if (yhalf)
        for (uint m = 0; m < 8; m++) h[j][m] = as_ushort(convert_half(r[m]));
      if (block2d) {
        int2 at = (int2)(n0 + 16 * j, t0 + 8 * i);
        if (yhalf) {
          sg_store_rows8((__global ushort *)Y, N * 2, T, ldy * 2, at.x, at.y, h[j]);
        } else {
          uint v[8];
          for (uint m = 0; m < 8; m++) v[m] = as_uint(r[m]);
          intel_sub_group_2d_block_write_32b_8r16x1c((__global void *)Y, N * 4, T, ldy * 4, at, v);
        }
        continue;
      }
      for (uint m = 0; m < 8; m++) {
        uint t = t0 + 8 * i + m;
        if (t < T && n < N) {
          if (yhalf)
            vstore_half(r[m], (ulong)t * ldy + n, (__global half *)Y);
          else
            Y[(ulong)t * ldy + n] = r[m];
        }
      }
    }
    // One block write a token: small stores cross PCIe slowly (docs/measurements/MEASUREMENTS.md#2026-10-15-prompt-exchange-stores).
    if (Y2 && yhalf)
      for (uint m = 0; m < 8; m++) {
        const uint t = t0 + 8 * i + m;
        if (t >= T) break;
#if TN == 2
        intel_sub_group_block_write_us2(Y2 + (ulong)t * ldy + n0, (ushort2)(h[0][m], h[1][m]));
#else
        for (uint j = 0; j < TN; j++) intel_sub_group_block_write_us(Y2 + (ulong)t * ldy + n0 + 16 * j, h[j][m]);
#endif
      }
  }
}

// out[t * ld_out + i] = x[t * ld_in + i] as fp16 (or bf16), i < cols, t < *ntok. One work-item per value.
__kernel void to_16(__global const float *x, uint ld_in, uint cols, __global const uint *ntok, __global ushort *out,
                    uint ld_out) {
  uint i = get_global_id(0), t = get_global_id(1);
  if (i >= cols || t >= *ntok) return;
  float v = x[(ulong)t * ld_in + i];
#ifdef BF16
  out[(ulong)t * ld_out + i] = f32_to_bf16(v);
#else
  vstore_half(v, (ulong)t * ld_out + i, (__global half *)out);
#endif
}


