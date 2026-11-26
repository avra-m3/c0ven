// Xe-HPG prompt-path matrix products on the 8-lane XMX units: fp16 (gemm_f16) and bf16 (-DBF16, gemm_bf16).
// Stages 8*TM x 16 tiles of X in SLM and transposes 8-row tiles of W via per-sub-group SLM for coalesced loads.
#include "xmx.h"

#ifdef BF16
#define DPAS xmx_dpas_bf16
#else
#define DPAS xmx_dpas_f16
#endif

#ifndef TM
#define TM 8
#endif
#ifndef TN
#define TN 2
#endif
#ifndef GEMM_SG
#define GEMM_SG 4
#endif

#define HPG_SG (2 * GEMM_SG)

static inline float post(float x, uint mode) {
  if (mode == 1) return x / (1.0f + exp(-x));
  if (mode == 2) return 1.0f / (1.0f + exp(-x));
  return x;
}

// Same 14-argument contract and launch geometry as kernels/gemm.cl; 16 * GEMM_SG work-items form HPG_SG 8-lane sub-groups.
REQD_XMX_SG __kernel void gemm(__global const ushort *W, uint N, uint K, uint ldw, __global const ushort *X, uint ldx,
                               __global const uint *ntok, __global float *Y, uint ldy, uint yhalf, uint mode, float scale,
                               __global ushort *Y2, uint band) {
  __local uint la[2][8 * TM * 8];
  __local uint lb[HPG_SG][8][9];
  const uint lane = sg_lane(), sg = sg_id();
  uint gn = get_group_id(0), gt = get_group_id(1);
  if (band) {
    const uint nb = get_num_groups(0), tb = get_num_groups(1), g = gn + gt * nb;
    const uint b = g / (band * tb), r = g % (band * tb), w = min(band, nb - b * band);
    gn = b * band + r % w;
    gt = r / w;
  }
  const uint T = *ntok, t0 = gt * (8 * TM);
  // Only leave early on conditions uniform across the work-group so every sub-group enters all SLM barriers.
  if (t0 >= T || gn * (16 * TN * GEMM_SG) >= N) return;
  const uint n0 = (gn * HPG_SG + sg) * (8 * TN);
  float8 acc[TM][TN];
  for (uint i = 0; i < TM; i++)
    for (uint j = 0; j < TN; j++) acc[i][j] = (float8)0.0f;
  for (uint k = 0, step = 0; k < K; k += 16, step++) {
    __local uint *la_cur = la[step & 1u];
    for (uint idx = get_local_id(0); idx < 8 * TM * 8; idx += get_local_size(0)) {
      const uint r = idx >> 3, p = idx & 7u, row = t0 + r;
      uint val = 0u;
      if (row < T) {
        const ushort2 uv = vload2(0, X + (ulong)row * ldx + k + 2 * p);
        val = (uint)uv.x | ((uint)uv.y << 16);
      }
      la_cur[idx] = val;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    xmx_a_t a[TM];
    for (uint i = 0; i < TM; i++) {
      int8 av;
      __attribute__((opencl_unroll_hint)) for (uint m = 0; m < 8; m++) av[m] = (int)la_cur[(8 * i + m) * 8 + lane];
      a[i] = av;
    }
    for (uint j = 0; j < TN; j++) {
      const uint col0 = n0 + 8 * j;
      for (uint m = 0; m < 8; m++) {
        const uint col = col0 + m;
        lb[sg][lane][m] = col < N ? ((__global const uint *)(W + (ulong)col * ldw + k))[lane] : 0u;
      }
      sub_group_barrier(CLK_LOCAL_MEM_FENCE);
      int8 b;
      __attribute__((opencl_unroll_hint)) for (uint p = 0; p < 8; p++) b[p] = (int)lb[sg][p][lane];
      sub_group_barrier(CLK_LOCAL_MEM_FENCE);
      for (uint i = 0; i < TM; i++) acc[i][j] = DPAS(a[i], b, acc[i][j]);
    }
  }
  for (uint i = 0; i < TM; i++) {
    for (uint j = 0; j < TN; j++) {
      const uint col0 = n0 + 8 * j, n = col0 + lane;
      float r[8];
      vstore8(acc[i][j], 0, r);
      for (uint m = 0; m < 8; m++) r[m] = post(r[m] * scale, mode);
      ushort h[8];
      if (yhalf)
        for (uint m = 0; m < 8; m++) h[m] = as_ushort(convert_half(r[m]));
      const bool block16 = yhalf && (ldy % 8 == 0) && (((ulong)Y) % 16 == 0) && col0 + 8 <= N;
      for (uint m = 0; m < 8; m++) {
        const uint t = t0 + 8 * i + m;
        if (t >= T) break;
        if (block16) {
          intel_sub_group_block_write_us((__global ushort *)Y + (ulong)t * ldy + col0, h[m]);
        } else if (n < N) {
          if (yhalf)
            ((__global ushort *)Y)[(ulong)t * ldy + n] = h[m];
          else
            Y[(ulong)t * ldy + n] = r[m];
        }
      }
      if (Y2 && yhalf) {
        const bool y2_blk = (ldy % 8 == 0) && (((ulong)Y2) % 16 == 0) && col0 + 8 <= N;
        for (uint m = 0; m < 8; m++) {
          const uint t = t0 + 8 * i + m;
          if (t >= T) break;
          if (y2_blk)
            intel_sub_group_block_write_us(Y2 + (ulong)t * ldy + col0, h[m]);
          else if (n < N)
            Y2[(ulong)t * ldy + n] = h[m];
        }
      }
    }
  }
}
