// Xe-HPG sparse prompt attention on the 8-lane XMX units, sharing p_attn_rows, p_attn_rows_merge and
// p_idx_score_xmx via prompt_attn_common.h. Work-group 256 work-items form 16 pairs of 8-lane sub-groups; each half of
// a pair processes alternating 16-cell tiles and merges once per row at the end via SLM.
#include "prompt_attn_common.h"

#define ATT_MAX_SLACK 8.0f

#if Q_ATT_SLOTS % 8 || Q_ATT_SLOTS < Q_ATT_H / Q_ATT_KV
#error "Q_ATT_SLOTS holds a KV head's query heads in whole groups of 8"
#endif

REQD_XMX_SG __kernel void p_attn_xmx(__global const half *qt, __global const float *Q, __global const half *kh,
                                     __global const half *vh, uint nq, uint hp, uint nkv,
                                     __global const uint *cells_all, __global const uint *ncells,
                                     __global const PromptPlan *pp, __global half *o16) {
  __local float pair_ml[16][2][2][8];
  __local float pair_o[16][4][8][8];
  if (pp->n <= PROMPT_SHORT_RUN) return;

  const uint hg = hp / 8, sg = get_sub_group_id(), pair = sg >> 1, half_id = sg & 1u;
  const uint rows = (get_num_sub_groups() / 2) / hg;
  const uint t = rows * get_group_id(1) + pair / hg, h0 = 8 * (pair % hg), lane = sg_lane();
  const bool active = t < pp->n;

  const ulong r0 = pp->row0;
  qt += r0 * Q_ATT_D * hp;
  Q += r0 * nq * 2 * Q_ATT_D;
  cells_all += r0 * Q_IDX_CELLS;
  ncells += r0;
  o16 += r0 * nq * Q_ATT_D;

  const uint c = h0 / Q_ATT_SLOTS, hk0 = h0 % Q_ATT_SLOTS;
  float8 o[32];
  for (uint j = 0; j < 32; j++) o[j] = (float8)0.0f;
  float8 mrow = (float8)(-INFINITY), lrow = (float8)0.0f;

  if (active) {
    const uint n = ncells[t];
    __global const uint *cells = cells_all + (ulong)t * Q_IDX_CELLS;
    __global const ushort *q = (__global const ushort *)qt + (ulong)t * Q_ATT_D * hp + h0;
    int8 qa[16];
    for (uint k = 0; k < 16; k++) {
      const ushort8 q0 = vload8(0, q + (ulong)(16 * k + 2 * lane) * hp);
      const ushort8 q1 = vload8(0, q + (ulong)(16 * k + 2 * lane + 1) * hp);
      qa[k] = as_int8(convert_uint8(q0) | (convert_uint8(q1) << 16));
    }

    for (uint i0 = 16 * half_id; i0 < n; i0 += 32) {
      uint cell[2];
      bool valid[2];
      float8 sc[2];
      for (uint u = 0; u < 2; u++) {
        valid[u] = i0 + 8 * u + lane < n;
        cell[u] = cells[valid[u] ? i0 + 8 * u + lane : i0];
        __global const uint *krow = (__global const uint *)(kh + ((ulong)cell[u] * nkv + c) * Q_ATT_D);
        float8 s0 = (float8)0.0f, s1 = (float8)0.0f;
        __attribute__((opencl_unroll_hint)) for (uint k = 0; k < Q_ATT_D / 16; k += 2) {
          s0 = xmx_dpas_f16(qa[k], as_int8(vload8(0, krow + 8 * k)), s0);
          s1 = xmx_dpas_f16(qa[k + 1], as_int8(vload8(0, krow + 8 * k + 8)), s1);
        }
        sc[u] = valid[u] ? s0 + s1 : (float8)(-INFINITY);
      }

      const float8 bm = fmax(sc[0], sc[1]);
      if (sub_group_any(any(bm > mrow + ATT_MAX_SLACK))) {
        float b8[8];
        vstore8(bm, 0, b8);
        for (uint m = 0; m < 8; m++) b8[m] = sub_group_reduce_max(b8[m]);
        const float8 mn = fmax(mrow, vload8(0, b8)), al = exp(mrow - mn);
        for (uint j = 0; j < 32; j++) o[j] *= al;
        lrow *= al;
        mrow = mn;
      }

      const float8 ea = exp(sc[0] - mrow), eb = exp(sc[1] - mrow);
      lrow += ea + eb;
      const ushort8 ha = as_ushort8(convert_half8(ea)), hb = as_ushort8(convert_half8(eb));
      const uint l0 = (lane & 3u) * 2u, l1 = l0 + 1u;
      const ushort8 a0 = intel_sub_group_shuffle(ha, l0), a1 = intel_sub_group_shuffle(ha, l1);
      const ushort8 b0 = intel_sub_group_shuffle(hb, l0), b1 = intel_sub_group_shuffle(hb, l1);
      const ushort8 p0 = lane < 4 ? a0 : b0, p1 = lane < 4 ? a1 : b1;
      const int8 P = as_int8(convert_uint8(p0) | (convert_uint8(p1) << 16));

      __attribute__((opencl_unroll_hint)) for (uint qd = 0; qd < 4; qd++) {
        ushort r[16][8];
        __attribute__((opencl_unroll_hint)) for (uint k = 0; k < 16; k++) {
          const uint ck = sub_group_broadcast(cell[k >> 3], k & 7u);
          const ushort8 v = intel_sub_group_block_read_us8((__global const ushort *)(vh + ((ulong)ck * nkv + c) * Q_ATT_D) +
                                                           64 * qd);
          r[k][0] = v.s0;
          r[k][1] = v.s1;
          r[k][2] = v.s2;
          r[k][3] = v.s3;
          r[k][4] = v.s4;
          r[k][5] = v.s5;
          r[k][6] = v.s6;
          r[k][7] = v.s7;
        }
        __attribute__((opencl_unroll_hint)) for (uint j = 0; j < 8; j++) {
          uint pr[8];
          __attribute__((opencl_unroll_hint)) for (uint q2 = 0; q2 < 8; q2++)
            pr[q2] = (uint)r[2 * q2][j] | ((uint)r[2 * q2 + 1][j] << 16);
          o[8 * qd + j] = xmx_dpas_f16(P, as_int8(vload8(0, pr)), o[8 * qd + j]);
        }
      }
    }
    for (uint x = 1; x < 8; x *= 2) lrow += intel_sub_group_shuffle_xor(lrow, x);
  }

  if (lane == 0) {
    vstore8(mrow, 0, pair_ml[pair][half_id][0]);
    vstore8(lrow, 0, pair_ml[pair][half_id][1]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (active) {
    const float8 m0 = vload8(0, pair_ml[pair][0][0]), m1 = vload8(0, pair_ml[pair][1][0]);
    const float8 l0 = vload8(0, pair_ml[pair][0][1]), l1 = vload8(0, pair_ml[pair][1][1]);
    const float8 mn = fmax(fmax(m0, m1), (float8)(-1e30f)), a0 = exp(m0 - mn), a1 = exp(m1 - mn);
    const float8 ltot = fmax(l0 * a0 + l1 * a1, (float8)1e-30f);
    const float8 scale = (half_id == 0 ? a0 : a1) / ltot;
    for (uint j = 0; j < 32; j++) o[j] *= scale;
  }

  for (uint p = 0; p < 8; p++) {
    if (half_id == 1)
      for (uint k = 0; k < 4; k++) vstore8(o[4 * p + k], 0, pair_o[pair][k][lane]);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (half_id == 0 && active) {
      for (uint k = 0; k < 4; k++) {
        const uint j = 4 * p + k;
        float ov[8];
        vstore8(o[j] + vload8(0, pair_o[pair][k][lane]), 0, ov);
        const uint d = 8 * j + lane;
        for (uint m = 0; m < 8; m++) {
          if (hk0 + m >= Q_ATT_H / Q_ATT_KV) break;
          const uint h = c * (Q_ATT_H / Q_ATT_KV) + hk0 + m;
          const float gate = Q[((ulong)t * nq + h) * 2 * Q_ATT_D + Q_ATT_D + d];
          o16[((ulong)t * nq + h) * Q_ATT_D + d] = convert_half(ov[m] * sigmoid(gate));
        }
      }
    }
    if (p + 1 < 8) barrier(CLK_LOCAL_MEM_FENCE);
  }
}
