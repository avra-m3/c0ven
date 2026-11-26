// Xe-HPG (8-lane) XMX primitives for shared kernels: 8x8x16 DPAS, scalar/SLM operand loads, and SLM IQ4 lookup.
#ifndef KERNELS_XMX_HPG_H
#define KERNELS_XMX_HPG_H

#include "builtins.h"

#define XMX_LANES 8
#define REQD_XMX_SG __attribute__((intel_reqd_sub_group_size(XMX_LANES)))

typedef int8 xmx_a_t;

static inline float8 xmx_dpas_f16(xmx_a_t a, int8 b, float8 c) {
  return intel_sub_group_f16_f16_matrix_mad_k16(a, b, c);
}
static inline float8 xmx_dpas_bf16(xmx_a_t a, int8 b, float8 c) {
  return intel_sub_group_bf16_bf16_matrix_mad_k16(a, b, c);
}

// Reads an 8-row by 16-column tile of 16-bit values from base[row0..row0+7][k0..k0+15] with 0 padding past height.
static inline xmx_a_t xmx_load_a8(__global const ushort *base, uint K, uint height, uint k0, uint row0) {
  const uint lane = sg_lane();
  int8 a;
  __attribute__((opencl_unroll_hint)) for (uint m = 0; m < 8; m++) {
    if (row0 + m < height) {
      const ushort2 v = vload2(0, base + (ulong)(row0 + m) * K + k0 + 2 * lane);
      a[m] = (int)((uint)v.x | ((uint)v.y << 16));
    } else {
      a[m] = 0;
    }
  }
  return a;
}

// Reads 8 columns of B (rows n0..n0+7 of W at columns k0..k0+15, stride ldw) into DPAS B layout, zero past N.
static inline int8 xmx_load_b(__global const ushort *W, uint K, uint N, uint ldw, uint k0, uint n0) {
  (void)K;
  const uint n = n0 + sg_lane();
  if (n >= N) return (int8)0;
  return as_int8(vload8(0, (__global const int *)(W + (ulong)n * ldw + k0)));
}

// Stores one 32-element Q8_1 unit from local float row32 by one 8-lane sub-group (4 values per lane).
static inline void xmx_a8_store(__local const float *row32, __global uchar *a, uint K, uint u) {
  const uint lane = sg_lane();
  const float4 v = vload4(lane, row32);
  const float amax = sub_group_reduce_max(fmax(fmax(fabs(v.x), fabs(v.y)), fmax(fabs(v.z), fabs(v.w))));
  const float d = amax / 127.0f;
  const float id = d != 0.0f ? 1.0f / d : 0.0f;
  vstore4((char4)((char)round(v.x * id), (char)round(v.y * id), (char)round(v.z * id), (char)round(v.w * id)), lane,
          (__global char *)(a + 32 * u));
  if (lane == 0) ((__global float *)(a + K))[u] = d;
}

// Under SIMD8 a 16-entry codebook spans two sub-groups, so read directly from SLM instead of shuffling across 16 lanes.
#define XMX_IQ4_GET(tbl, grid, idx) ((void)(tbl), convert_half((float)((__local const char *)(grid))[(idx)]))

// Loads 2 indexer rows (4 heads per row, 16 dimensions) from q[row][d][h]; H == 4 puts d = 2*lane and 2*lane+1
// consecutively in one 16-byte load.
static inline xmx_a_t xmx_idx_qa(__global const ushort *q, uint row0, uint row1, uint s, uint D, uint H) {
  const uint lane = sg_lane();
  const ulong r0 = ((ulong)row0 * D + 16 * s + 2 * lane) * H;
  const ulong r1 = ((ulong)row1 * D + 16 * s + 2 * lane) * H;
  const ushort8 u0 = vload8(0, q + r0), u1 = vload8(0, q + r1);
  const uint4 p0 = convert_uint4(u0.lo) | (convert_uint4(u0.hi) << 16);
  const uint4 p1 = convert_uint4(u1.lo) | (convert_uint4(u1.hi) << 16);
  return as_int8((uint8)(p0, p1));
}

// Double-buffered local staging of 64 x 32 half values (2 * 1024 uints = 8 KiB) for cooperative qgemm A loads.
#define XMX_QGEMM_SLM __local uint xmx_qg_a[2 * 2 * 64 * 8];
#define XMX_QGEMM_SLM_PARAM , __local uint *xmx_qg_a
#define XMX_QGEMM_SLM_ARG , xmx_qg_a
#define XMX_QGEMM_LOAD_A(u, X, K, end_row, ldx, k0, row0, nt, a) \
  xmx_qgemm_load_a((u), xmx_qg_a, (X), (K), (end_row), (ldx), (k0), (row0), (nt), (a))

static inline void xmx_qgemm_load_a(uint step_u, __local uint *slm, __global const ushort *X, uint K, uint end_row,
                                    uint ldx, uint k0, uint row0, uint nt, xmx_a_t a[2][8]) {
  (void)K;
  __local uint *buf = slm + (step_u & 1u) * (2 * 64 * 8);
  const uint lid = get_local_id(0), lane = sg_lane();
  const uint total = nt * 8 * 16;
  for (uint idx = lid; idx < total; idx += 64) {
    const uint r = idx >> 4, cp = idx & 15u, h = cp >> 3, p = cp & 7u;
    const uint row = row0 + r;
    const uint val = row < end_row ? ((__global const uint *)(X + (ulong)row * ldx + k0))[cp] : 0u;
    buf[(h * 64 + r) * 8 + p] = val;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint h = 0; h < 2; h++) {
    __local const uint *src = buf + h * 64 * 8;
    for (uint i = 0; i < nt; i++) {
      int8 v;
      __attribute__((opencl_unroll_hint)) for (uint m = 0; m < 8; m++) v[m] = (int)src[(8 * i + m) * 8 + lane];
      a[h][i] = v;
    }
  }
}

static inline void xmx_qgemm_write_tile(float8 acc, __global half *Y, uint rows, uint end_row, uint row0, uint row_s,
                                        uint t_start, uint i, bool block2d) {
  (void)block2d;
  if (row0 + sg_lane() >= rows) return;
  float y[8];
  vstore8(acc, 0, y);
  for (uint m = 0; m < 8; m++) {
    const uint t = t_start + 8 * i + m;
    if (t < end_row) Y[(ulong)t * rows + row_s] = convert_half(y[m]);
  }
}

#endif
