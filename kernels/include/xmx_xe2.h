// Xe2 (16-lane) XMX primitives for shared kernels: 8x16x16 DPAS, 2D block loads/stores, and 16-lane IQ4 lookup.
#ifndef KERNELS_XMX_XE2_H
#define KERNELS_XMX_XE2_H

#include "a8.h"
#include "builtins.h"

#define XMX_LANES 16
#define REQD_XMX_SG REQD_SG

typedef short8 xmx_a_t;

static inline float8 xmx_dpas_f16(xmx_a_t a, int8 b, float8 c) { return dpas_f16(a, b, c); }
static inline float8 xmx_dpas_bf16(xmx_a_t a, int8 b, float8 c) { return dpas_bf16(a, b, c); }

// Reads an 8-row by 16-column tile of 16-bit values from base[row0..row0+7][k0..k0+15] with 0 padding past height.
static inline xmx_a_t xmx_load_a8(__global const ushort *base, uint K, uint height, uint k0, uint row0) {
  ushort a[8];
  intel_sub_group_2d_block_read_16b_8r16x1c((__global void *)base, K * 2, height, K * 2, (int2)(k0, row0), a);
  return as_short8(vload8(0, a));
}

// Reads 16 columns of B (rows n0..n0+15 of W at columns k0..k0+15, stride ldw), transposed into DPAS B layout.
static inline int8 xmx_load_b(__global const ushort *W, uint K, uint N, uint ldw, uint k0, uint n0) {
  uint b[8];
  sg_load_cols16(W, K * 2, N, ldw * 2, k0 / 2, n0, b);
  return as_int8(vload8(0, b));
}

// Stores one 32-element Q8_1 unit from local float row32 into global quantized activation row a.
static inline void xmx_a8_store(__local const float *row32, __global uchar *a, uint K, uint u) {
  a8_store(vload2(sg_lane(), row32), a, K, u);
}

#define XMX_IQ4_GET(tbl, grid, idx) ((void)(grid), intel_sub_group_shuffle((tbl), (idx)))

// Loads 2 indexer rows (4 heads per row, 16 dimensions) from q[row][d][h] for one DPAS call.
static inline xmx_a_t xmx_idx_qa(__global const ushort *q, uint row0, uint row1, uint s, uint D, uint H) {
  const uint lane = sg_lane();
  const ulong r0 = ((ulong)row0 * D + 16 * s + lane) * H;
  const ulong r1 = ((ulong)row1 * D + 16 * s + lane) * H;
  return as_short8((ushort8)(vload4(0, q + r0), vload4(0, q + r1)));
}

// Loads 32-row blocks of A directly from global memory into xmx_a_t tiles on Xe2 (no local staging needed).
#define XMX_QGEMM_SLM
#define XMX_QGEMM_SLM_PARAM
#define XMX_QGEMM_SLM_ARG
#define XMX_QGEMM_LOAD_A(u, X, K, end_row, ldx, k0, row0, nt, a) \
  xmx_qgemm_load_a((X), (K), (end_row), (ldx), (k0), (row0), (nt), (a))

static inline void xmx_qgemm_load_a(__global const ushort *X, uint K, uint end_row, uint ldx, uint k0, uint row0,
                                    uint nt, xmx_a_t a[2][8]) {
  for (uint h = 0; h < 2; h++) {
    for (uint c = 0; c < nt / 4; c++) {
      ushort buf[32];
      sg_load_rows32(X, K * 2, end_row, ldx * 2, k0 + 16 * h, row0 + 32 * c, buf);
      for (uint k = 0; k < 4; k++) a[h][4 * c + k] = as_short8(vload8(k, buf));
    }
  }
}

// Writes one 8x16 DPAS output tile of fp16 values to Y[t.start + 8*i ..][r0 .. r0+15], clamped to end_row and rows.
static inline void xmx_qgemm_write_tile(float8 acc, __global half *Y, uint rows, uint end_row, uint row0, uint row_s,
                                        uint t_start, uint i, bool block2d) {
  const uint r = row0 + sg_lane();
  float y[8];
  vstore8(acc, 0, y);
  if (block2d) {
    ushort h[8];
    for (uint m = 0; m < 8; m++) h[m] = as_ushort(convert_half(y[m]));
    sg_store_rows8((__global ushort *)Y, rows * 2, end_row, rows * 2, row0, t_start + 8 * i, h);
  } else if (r < rows) {
    for (uint m = 0; m < 8; m++) {
      const uint t = t_start + 8 * i + m;
      if (t < end_row) Y[(ulong)t * rows + row_s] = convert_half(y[m]);
    }
  }
}

#endif
