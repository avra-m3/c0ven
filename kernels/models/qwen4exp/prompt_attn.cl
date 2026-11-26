// Sparse attention over a prompt chunk on the XMX units (docs/spec/MODEL_LAYER.md, 3.5 and 3.6), built with 256
// registers, for every run longer than a short run, at any context. One sub-group takes one row and 8 of its query
// heads (the A operand's 8 rows), and walks the row's cells 32 at a time with an online softmax: S = Q K^T by dpas
// calls over the 256 dimensions, then O += P V, one call per 16 dimensions and 16 cells. The queries (16 A operands)
// and the output, 8 heads x 256, stay in registers.
#include "prompt_attn_common.h"

// The online softmax keeps the maximum it scales by until a score passes it by more than this, so P stays under
// e^8 in fp16 and the sub-group skips the maximum over its lanes and the rescaling of O on most steps.
#define ATT_MAX_SLACK 8.0f

#if Q_IDX_BLOCK != 4
#error "p_attn_xmx reads V in runs of 4 cells"
#endif
#if Q_ATT_SLOTS % 8 || Q_ATT_SLOTS < Q_ATT_H / Q_ATT_KV
#error "Q_ATT_SLOTS holds a KV head's query heads in whole sub-groups of 8"
#endif

// qt: each row's queries as fp16 scaled by 1/16, dimension-major, qt[t][d][slot], hp = nkv Q_ATT_SLOTS slots a
// dimension: KV head c's heads in slots c Q_ATT_SLOTS on, the slots past its 12 heads zero. So a sub-group's 8 slots
// never span two KV heads. kh, vh: p_kv16's fp16 K and V, [position][nkv][256]. cells: p_idx_merge's, whose first
// 2,048 cells are 512 whole blocks of 4 (the dense rows' cells are 0, 1, 2, ...), so a tile of 16 cells is mostly 4
// runs of 4 positions. A work-group holds `rows` consecutive rows of hp / 8 sub-groups each, which share many cells
// through the core's cache. Groups (1, ceil(T / rows)), 16 hp / 8 rows work-items (prompt.cpp: 16 sub-groups).
REQD_SG __kernel void p_attn_xmx(__global const half *qt, __global const float *Q, __global const half *kh,
                                 __global const half *vh, uint nq, uint hp, uint nkv, __global const uint *cells_all,
                                 __global const uint *ncells, __global const PromptPlan *pp, __global half *o16) {
  const uint hg = hp / 8, sg = get_sub_group_id(), rows = get_num_sub_groups() / hg;
  const uint t = rows * get_group_id(1) + sg / hg, h0 = 8 * (sg % hg), lane = sg_lane();
  if (t >= pp->n || pp->n <= PROMPT_SHORT_RUN) return;
  const ulong r0 = pp->row0;
  qt += r0 * Q_ATT_D * hp;
  Q += r0 * nq * 2 * Q_ATT_D;
  cells_all += r0 * Q_IDX_CELLS;
  ncells += r0;
  o16 += r0 * nq * Q_ATT_D;
  // KV head c; the sub-group's slots are its heads hk0 .. hk0 + 7, those under 12 real.
  const uint c = h0 / Q_ATT_SLOTS, hk0 = h0 % Q_ATT_SLOTS;
  const uint n = ncells[t];
  __global const uint *cells = cells_all + (ulong)t * Q_IDX_CELLS;
  __global const ushort *q = (__global const ushort *)qt + (ulong)t * Q_ATT_D * hp + h0;
  // V as a 2D surface for the block reads: one row a position, this KV head's 256 values.
  const uint height = pp->pos0 + pp->n, pitch = nkv * Q_ATT_D * 2;
  __global const ushort *vb = (__global const ushort *)(vh + c * Q_ATT_D);
  short8 qa[16];
  for (uint k = 0; k < 16; k++) qa[k] = as_short8(vload8(0, q + (ulong)(16 * k + lane) * hp));
  float8 o[16];
  for (uint j = 0; j < 16; j++) o[j] = (float8)0.0f;
  float8 mrow = (float8)(-INFINITY), lrow = (float8)0.0f;  // lrow: each lane's own part of the sums
  for (uint i0 = 0; i0 < n; i0 += 32) {
    // Two tiles of 16 cells: lane = cell, component = head. A tile is "runs" when its cells are 4 runs of 4
    // consecutive positions and every lane has one; g0 holds each run's first position.
    uint cell[2], g0[2][4];
    bool valid[2], runs[2];
    float8 s[2];
    for (uint u = 0; u < 2; u++) {
      valid[u] = i0 + 16 * u + lane < n;
      cell[u] = cells[valid[u] ? i0 + 16 * u + lane : i0];
      runs[u] = sub_group_all(valid[u]);
      for (uint g = 0; g < 4; g++) {
        g0[u][g] = sub_group_broadcast(cell[u], 4 * g);
        runs[u] = runs[u] && sub_group_broadcast(cell[u], 4 * g + 3) == g0[u][g] + 3;
      }
      __global const uint *krow = (__global const uint *)(kh + ((ulong)cell[u] * nkv + c) * Q_ATT_D);
      float8 s0 = (float8)0.0f, s1 = (float8)0.0f;
      __attribute__((opencl_unroll_hint)) for (uint k = 0; k < Q_ATT_D / 16; k += 2) {
        s0 = dpas_f16(qa[k], as_int8(vload8(0, krow + 8 * k)), s0);
        s1 = dpas_f16(qa[k + 1], as_int8(vload8(0, krow + 8 * k + 8)), s1);
      }
      s[u] = valid[u] ? s0 + s1 : (float8)(-INFINITY);
    }
    // The online softmax, per head (component). Each lane checks its own scores against the maximum in use; only
    // when one passes it by more than ATT_MAX_SLACK is the maximum over the lanes taken and O rescaled.
    float8 bm = fmax(s[0], s[1]);
    if (sub_group_any(any(bm > mrow + ATT_MAX_SLACK))) {
      float b8[8];
      vstore8(bm, 0, b8);
      for (uint m = 0; m < 8; m++) b8[m] = sub_group_reduce_max(b8[m]);
      const float8 mn = fmax(mrow, vload8(0, b8)), al = exp(mrow - mn);
      for (uint j = 0; j < 16; j++) o[j] *= al;
      lrow *= al;
      mrow = mn;
    }
    const float8 ea = exp(s[0] - mrow), eb = exp(s[1] - mrow);
    lrow += ea + eb;
    short8 P[2] = {as_short8(convert_half8(ea)), as_short8(convert_half8(eb))};
    // O += P V, a quarter of the dimensions at a time: B[k][n] = V[cell k][d + n], cells paired in each 32-bit word.
    __attribute__((opencl_unroll_hint)) for (uint qd = 0; qd < 4; qd++) {
      __attribute__((opencl_unroll_hint)) for (uint u = 0; u < 2; u++) {
        if (runs[u]) {
          // A run's 4 positions by 32 dimensions in one 2D block read: lane n gets dimension n, then n + 16, of the
          // run's 4 positions in turn.
          __attribute__((opencl_unroll_hint)) for (uint j2 = 0; j2 < 2; j2++) {
            uint ba[8], bb[8];
            __attribute__((opencl_unroll_hint)) for (uint g = 0; g < 4; g++) {
              ushort r8[8];
              intel_sub_group_2d_block_read_16b_4r16x2c((__global void *)vb, Q_ATT_D * 2, height, pitch,
                                                        (int2)(64 * qd + 32 * j2, g0[u][g]), r8);
              ba[2 * g] = (uint)r8[0] | ((uint)r8[1] << 16);
              ba[2 * g + 1] = (uint)r8[2] | ((uint)r8[3] << 16);
              bb[2 * g] = (uint)r8[4] | ((uint)r8[5] << 16);
              bb[2 * g + 1] = (uint)r8[6] | ((uint)r8[7] << 16);
            }
            o[4 * qd + 2 * j2] = dpas_f16(P[u], as_int8(vload8(0, ba)), o[4 * qd + 2 * j2]);
            o[4 * qd + 2 * j2 + 1] = dpas_f16(P[u], as_int8(vload8(0, bb)), o[4 * qd + 2 * j2 + 1]);
          }
          continue;
        }
        // Any other tile (the partial block and the tail, a row's last cells): each cell's row by block reads (lane
        // n gets dimensions n, n + 16, n + 32, n + 48 of the quarter), paired afterwards.
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
  // Out: head c 12 + hk0 + m, dimension 64 (j / 4) + 16 (j % 4) + lane.
  for (uint x = 1; x < 16; x *= 2) lrow += intel_sub_group_shuffle_xor(lrow, x);
  float lr[8];
  vstore8(lrow, 0, lr);
  for (uint j = 0; j < 16; j++) {
    float ov[8];
    vstore8(o[j], 0, ov);
    uint d = 64 * (j / 4) + 16 * (j % 4) + lane;
    for (uint m = 0; m < 8; m++) {
      if (hk0 + m >= Q_ATT_H / Q_ATT_KV) break;
      uint h = c * (Q_ATT_H / Q_ATT_KV) + hk0 + m;
      float gate = Q[((ulong)t * nq + h) * 2 * Q_ATT_D + Q_ATT_D + d];
      o16[((ulong)t * nq + h) * Q_ATT_D + d] = convert_half(ov[m] / lr[m] * sigmoid(gate));
    }
  }
}
