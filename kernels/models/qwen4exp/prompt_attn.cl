// Sparse attention over a prompt chunk on the XMX units (docs/spec/MODEL_LAYER.md, 3.5 and 3.6), built with 256
// registers. One sub-group takes one row and 8 of its query heads (the A operand's 8 rows), and walks the row's
// cells 32 at a time with an online softmax: S = Q K^T by dpas calls over the 256 dimensions, then O += P V, one
// call per 16 dimensions and 16 cells. The output, 8 heads x 256, stays in registers.
#include "builtins.h"
#include "layout.h"
#include "step.h"

static inline float sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }

// Chunks that end at most here take p_attn_tiled, the others p_attn_xmx (both are recorded; each returns at once
// when it is not the one).
#define ATT_XMX_ROWS 4

#ifndef ATT_TILED_MAX
#define ATT_TILED_MAX 12288
#endif

// qt: each row's queries as fp16 scaled by 1/16, dimension-major, qt[t][d][h], hp heads a dimension (nq rounded up to
// 8, the extra heads zero). kh, vh: p_kv16's fp16 K and V, [position][nkv][256]. A work-group holds ATT_XMX_ROWS
// consecutive rows of hp / 8 sub-groups each, which share many cells through the core's cache.
// Groups (1, ceil(T / ATT_XMX_ROWS)), 16 hp / 8 ATT_XMX_ROWS work-items.
REQD_SG __kernel void p_attn_xmx(__global const half *qt, __global const float *Q, __global const half *kh,
                                 __global const half *vh, uint nq, uint hp, uint nkv, __global const uint *cells_all,
                                 __global const uint *ncells, __global const PromptPlan *pp, __global half *o16) {
  const uint hg = hp / 8, sg = get_sub_group_id();
  const uint t = ATT_XMX_ROWS * get_group_id(1) + sg / hg, h0 = 8 * (sg % hg), lane = sg_lane();
  if (t >= pp->n || pp->pos0 + pp->n <= ATT_TILED_MAX) return;
  const ulong r0 = pp->row0;
  qt += r0 * Q_ATT_D * hp;
  Q += r0 * nq * 2 * Q_ATT_D;
  cells_all += r0 * Q_IDX_CELLS;
  ncells += r0;
  o16 += r0 * nq * Q_ATT_D;
  const uint c = h0 / (Q_ATT_H / Q_ATT_KV);
  const uint n = ncells[t];
  __global const uint *cells = cells_all + (ulong)t * Q_IDX_CELLS;
  __global const ushort *q = (__global const ushort *)qt + (ulong)t * Q_ATT_D * hp + h0;
  float8 o[16];
  for (uint j = 0; j < 16; j++) o[j] = (float8)0.0f;
  float8 mrow = (float8)(-INFINITY), lrow = (float8)0.0f;
  for (uint i0 = 0; i0 < n; i0 += 32) {
    // Two tiles of 16 cells: lane = cell, component = head.
    uint cell[2];
    bool valid[2];
    float8 s[2];
    for (uint u = 0; u < 2; u++) {
      valid[u] = i0 + 16 * u + lane < n;
      cell[u] = cells[valid[u] ? i0 + 16 * u + lane : i0];
      __global const uint *krow = (__global const uint *)(kh + ((ulong)cell[u] * nkv + c) * Q_ATT_D);
      float8 s0 = (float8)0.0f, s1 = (float8)0.0f;
      __attribute__((opencl_unroll_hint)) for (uint k = 0; k < Q_ATT_D / 16; k += 2) {
        short8 a0 = as_short8(vload8(0, q + (ulong)(16 * k + lane) * hp));
        short8 a1 = as_short8(vload8(0, q + (ulong)(16 * k + 16 + lane) * hp));
        s0 = dpas_f16(a0, as_int8(vload8(0, krow + 8 * k)), s0);
        s1 = dpas_f16(a1, as_int8(vload8(0, krow + 8 * k + 8)), s1);
      }
      s[u] = valid[u] ? s0 + s1 : (float8)(-INFINITY);
    }
    // The online softmax, per head (component).
    float sa[8], sb[8], mr[8], lr[8], alpha[8], pa[8], pb[8];
    vstore8(s[0], 0, sa);
    vstore8(s[1], 0, sb);
    vstore8(mrow, 0, mr);
    vstore8(lrow, 0, lr);
    __attribute__((opencl_unroll_hint)) for (uint m = 0; m < 8; m++) {
      float bm = sub_group_reduce_max(fmax(sa[m], sb[m]));
      float mn = fmax(mr[m], bm);
      alpha[m] = exp(mr[m] - mn);
      pa[m] = exp(sa[m] - mn);
      pb[m] = exp(sb[m] - mn);
      lr[m] = lr[m] * alpha[m] + sub_group_reduce_add(pa[m] + pb[m]);
      mr[m] = mn;
    }
    mrow = vload8(0, mr);
    lrow = vload8(0, lr);
    float8 al = vload8(0, alpha);
    for (uint j = 0; j < 16; j++) o[j] *= al;
    short8 P[2] = {as_short8(convert_half8(vload8(0, pa))), as_short8(convert_half8(vload8(0, pb)))};
    // O += P V, a quarter of the dimensions at a time: B[k][n] = V[cell k][d + n], from block reads of the cells'
    // rows (lane n gets dimensions n, n + 16, n + 32, n + 48 of the quarter).
    __attribute__((opencl_unroll_hint)) for (uint qd = 0; qd < 4; qd++) {
      __attribute__((opencl_unroll_hint)) for (uint u = 0; u < 2; u++) {
        ushort r[16][4];
        __attribute__((opencl_unroll_hint)) for (uint k = 0; k < 16; k++) {
          uint ck = sub_group_broadcast(cell[u], k);
          ushort4 v = intel_sub_group_block_read_us4((__global const ushort *)(vh + ((ulong)ck * nkv + c) * Q_ATT_D) +
                                                     64 * qd);
          r[k][0] = v.s0;
          r[k][1] = v.s1;
          r[k][2] = v.s2;
          r[k][3] = v.s3;
        }
        __attribute__((opencl_unroll_hint)) for (uint j = 0; j < 4; j++) {
          uint pr[8];
          __attribute__((opencl_unroll_hint)) for (uint q2 = 0; q2 < 8; q2++)
            pr[q2] = (uint)r[2 * q2][j] | ((uint)r[2 * q2 + 1][j] << 16);
          o[4 * qd + j] = dpas_f16(P[u], as_int8(vload8(0, pr)), o[4 * qd + j]);
        }
      }
    }
  }
  // Out: head h0 + m, dimension 64 (j / 4) + 16 (j % 4) + lane.
  float lr[8];
  vstore8(lrow, 0, lr);
  for (uint j = 0; j < 16; j++) {
    float ov[8];
    vstore8(o[j], 0, ov);
    uint d = 64 * (j / 4) + 16 * (j % 4) + lane;
    for (uint m = 0; m < 8; m++) {
      uint h = h0 + m;
      if (h >= nq) break;
      float gate = Q[((ulong)t * nq + h) * 2 * Q_ATT_D + Q_ATT_D + d];
      o16[((ulong)t * nq + h) * Q_ATT_D + d] = convert_half(ov[m] / lr[m] * sigmoid(gate));
    }
  }
}

// ---------------------------------------------------------------------------------------------------------------
// The same attention for contexts up to ATT_TILED_MAX positions, tiled: a sub-group takes 8 consecutive rows of one
// query head and walks the keys in tiles of 16 positions, each K and V tile loaded once for the 8 rows by 2D block
// reads. Up to ATT_TILED_MAX the tiles cost at most twice the chosen cells and share their loads, so this is faster.
static inline float8 sel8(float8 v, float8 w, uint m) {  // v with component m of w; m a compile-time constant
  float a[8], b[8];
  vstore8(v, 0, a);
  vstore8(w, 0, b);
  a[m] = b[m];
  return vload8(0, a);
}

#if Q_IDX_BLOCK != 4
#error "tile_keys takes a tile's 16 positions as 4 blocks"
#endif
// The positions key0 .. key0 + 15 a row takes, as 16 bits (bit j: key0 + j); 0 when the row is not live. pm, dense,
// nbm and xm are the row's position, density, whole blocks and wbx; selrow its sel words. The tile's 4 blocks lie in
// one word, as key0 is a multiple of 16. Each lane computes its own row's bits and the sub-group shares them by
// broadcasts (docs/measurements/MEASUREMENTS.md#2026-10-15-prompt-attention-mask).
static inline uint tile_keys(uint key0, uint pm, bool dense, uint nbm, uint xm, __global const uint *selrow,
                             bool live) {
  if (!live || pm < key0) return 0;
  const uint causal = pm - key0 >= 15 ? 0xffffu : (2u << (pm - key0)) - 1;
  const uint b0 = key0 / Q_IDX_BLOCK;
  if (dense || b0 >= nbm) return causal;
  // Blocks before nbm count when chosen; the row's tail (blocks from nbm on) always.
  const uint need = nbm - b0 >= 4 ? 0xfu : (1u << (nbm - b0)) - 1;
  const uint chosen = (selrow[b0 / 32] >> (b0 % 32)) & need, blocks = chosen | (~need & 0xfu);
  uint keys = 0;
  for (uint i = 0; i < 4; i++)
    if ((blocks >> i) & 1) keys |= 0xfu << (4 * i);
  // The partial block: only its first `extra` positions.
  const uint xb = xm & 0xffffffu;
  if (xb >= b0 && xb < min(b0 + 4, nbm) && ((chosen >> (xb - b0)) & 1))
    keys &= ~(((0xfu << (xm >> 24)) & 0xfu) << (4 * (xb - b0)));
  return causal & keys;
}

// q16: the queries, fp16, scaled by 1/16, [row][nq][256]. The indexer's choice is a mask: a position up to row t's
// counts if the row is dense (llama.cpp's 2,051 cells hold all of it), in its tail, or in a chosen block (sel: a bit
// a block, rows nbw words apart), but not past the first `extra` cells of the partial block (wbx[t] = block |
// extra << 24, block 0xffffff when none). Groups (nq, ceil(T / 8)), 16 work-items.
REQD_SG __kernel void p_attn_tiled(__global const half *q16, __global const float *Q, __global const half *kh,
                                   __global const half *vh, uint nq, uint nkv, __global const uint *sel, uint nbw,
                                   __global const uint *wbx, __global const PromptPlan *pp, __global half *o16) {
  const uint h = get_group_id(0), t0 = 8 * get_group_id(1), lane = sg_lane();
  const uint n = pp->n, pos0 = pp->pos0;
  if (t0 >= n || pos0 + n > ATT_TILED_MAX) return;
  const ulong r0 = pp->row0;
  q16 += r0 * nq * Q_ATT_D;
  Q += r0 * nq * 2 * Q_ATT_D;
  sel += r0 * nbw;
  wbx += r0;
  o16 += r0 * nq * Q_ATT_D;
  const uint c = h / (Q_ATT_H / Q_ATT_KV);
  const uint rows = min(8u, n - t0);
  const uint last = pos0 + t0 + rows - 1;  // the largest position among the rows
  // The queries: 16 A operands of 8 rows x 16 dimensions.
  short8 qa[16];
  for (uint k = 0; k < 16; k++) {
    ushort a[8];
    intel_sub_group_2d_block_read_16b_8r16x1c((__global void *)(q16 + h * Q_ATT_D), Q_ATT_D * 2, n, nq * Q_ATT_D * 2,
                                              (int2)(16 * k, t0), a);
    qa[k] = as_short8(vload8(0, a));
  }
  // Each row's mask parameters, one row per lane 0-7.
  const uint my_t = t0 + min(lane, rows - 1);
  const uint my_pos = pos0 + my_t;
  const bool my_dense = my_pos + 1 <= Q_IDX_CELLS;
  const uint my_nb = (my_pos + 1) / Q_IDX_BLOCK;
  const uint my_wbx = wbx[my_t];
  float8 o[16];
  for (uint j = 0; j < 16; j++) o[j] = (float8)0.0f;
  float8 mrow = (float8)(-INFINITY), lrow = (float8)0.0f;
  __global const ushort *kb = (__global const ushort *)(kh + c * Q_ATT_D);
  __global const ushort *vb = (__global const ushort *)(vh + c * Q_ATT_D);
  const uint pitch = nkv * Q_ATT_D * 2, height = last + 1;
  for (uint key0 = 0; key0 <= last; key0 += 16) {
    // The mask: lane = key, component m = row m.
    const uint keys = tile_keys(key0, my_pos, my_dense, my_nb, my_wbx, sel + (ulong)my_t * nbw, lane < rows);
    if (!sub_group_any(keys != 0)) continue;
    float8 mask = (float8)(-INFINITY);
    __attribute__((opencl_unroll_hint)) for (uint m = 0; m < 8; m++)
      if ((sub_group_broadcast(keys, m) >> lane) & 1) mask = sel8(mask, (float8)0.0f, m);
    float8 s0 = (float8)0.0f, s1 = (float8)0.0f;
    __attribute__((opencl_unroll_hint)) for (uint k = 0; k < 16; k += 2) {
      uint kt0[8], kt1[8];
      intel_sub_group_2d_block_read_transpose_32b_16r8x1c((__global void *)kb, Q_ATT_D * 2, height, pitch,
                                                          (int2)(8 * k, key0), kt0);
      intel_sub_group_2d_block_read_transpose_32b_16r8x1c((__global void *)kb, Q_ATT_D * 2, height, pitch,
                                                          (int2)(8 * k + 8, key0), kt1);
      s0 = dpas_f16(qa[k], as_int8(vload8(0, kt0)), s0);
      s1 = dpas_f16(qa[k + 1], as_int8(vload8(0, kt1)), s1);
    }
    float8 s = s0 + s1 + mask;
    float sm[8], mr[8], lr[8], alpha[8], pr[8];
    vstore8(s, 0, sm);
    vstore8(mrow, 0, mr);
    vstore8(lrow, 0, lr);
    __attribute__((opencl_unroll_hint)) for (uint m = 0; m < 8; m++) {
      float bm = sub_group_reduce_max(sm[m]);
      float mn = fmax(mr[m], bm);
      bool none = mn == -INFINITY;
      alpha[m] = none ? 1.0f : exp(mr[m] - mn);
      pr[m] = none ? 0.0f : exp(sm[m] - mn);
      lr[m] = lr[m] * alpha[m] + sub_group_reduce_add(pr[m]);
      mr[m] = mn;
    }
    mrow = vload8(0, mr);
    lrow = vload8(0, lr);
    float8 al = vload8(0, alpha);
    short8 pa = as_short8(convert_half8(vload8(0, pr)));
    __attribute__((opencl_unroll_hint)) for (uint j = 0; j < 16; j++) {
      uint vt[8];
      intel_sub_group_2d_block_read_transform_16b_16r16x1c((__global void *)vb, Q_ATT_D * 2, height, pitch,
                                                           (int2)(16 * j, key0), vt);
      o[j] = dpas_f16(pa, as_int8(vload8(0, vt)), o[j] * al);
    }
  }
  float lr[8];
  vstore8(lrow, 0, lr);
  for (uint j = 0; j < 16; j++) {
    float ov[8];
    vstore8(o[j], 0, ov);
    uint d = 16 * j + lane;
    for (uint m = 0; m < rows; m++) {
      ulong at = ((ulong)(t0 + m) * nq + h);
      float gate = Q[at * 2 * Q_ATT_D + Q_ATT_D + d];
      o16[at * Q_ATT_D + d] = convert_half(ov[m] / lr[m] * sigmoid(gate));
    }
  }
}
