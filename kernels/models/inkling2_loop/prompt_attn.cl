// FP16 causal SWA and global attention kernels with relative-position bias and per-query sinks for prompt
// chunks: XMX dpas over contiguous runs longer than PROMPT_SHORT_RUN, and split-KV over short runs.
#include "common_i2.h"
#include "step.h"

#define ATT_MAX_SLACK 8.0f
#define ATT_SPLIT_TILE 64
#define ATT_MAX_SPLITS 32
#define ATT_PART (I2_HEAD_D + 2)

// Long-run causal attention on XMX units, compiled with 256 GRF.
// Groups (1, ceil(pp->n / rows_per_wg), 1), local (16 * 16, 1, 1) where hg = nkv, rows_per_wg = 16 / hg.
REQD_SG __kernel void p_i2_attn_xmx(__global const half *qt, __global const float *gate, ulong gate_stride,
                                    __global const float *sink, ulong sink_stride, __global const float *rb,
                                    ulong rb_stride, __global const half *kh, __global const half *vh, uint nq,
                                    uint hp, uint nkv, uint gqa, uint is_swa, uint swa, __global const PromptPlan *pp,
                                    __global half *o16) {
  if (pp->n <= PROMPT_SHORT_RUN) return;
  const uint hg = nkv, sg = get_sub_group_id(), rows = get_num_sub_groups() / hg;
  if (sg >= rows * hg) return;
  const uint t = rows * get_group_id(1) + sg / hg, c = sg % hg, lane = sg_lane();
  if (t >= pp->n) return;

  const uint r0 = pp->row0, row_idx = r0 + t;
  const uint qpos = pp->pos0 + t;
  const uint k_start = (is_swa && pp->pos0 >= swa) ? (pp->pos0 - swa + 1u) : 0u;
  const uint k_end = pp->pos0 + pp->n;
  const uint height = k_end - k_start;
  const uint pitch = nkv * I2_HEAD_D * 2u;

  const uint j_first = (is_swa && qpos >= swa) ? ((qpos - swa + 1u) - k_start) : 0u;
  const uint j_end = (qpos - k_start) + 1u;

  const __global ushort *q = (__global const ushort *)qt + ((ulong)row_idx * I2_HEAD_D) * hp + (ulong)c * 8u;
  const __global ushort *vb = (__global const ushort *)(vh + c * I2_HEAD_D);

  short8 qa[I2_HEAD_D / 16];
  for (uint k = 0; k < I2_HEAD_D / 16; k++) qa[k] = as_short8(vload8(0, q + (ulong)(16u * k + lane) * hp));

  float8 o[I2_HEAD_D / 16];
  for (uint j = 0; j < I2_HEAD_D / 16; j++) o[j] = (float8)0.0f;

  float m_init[8], l_init[8];
  for (uint m = 0; m < 8; m++) {
    m_init[m] = m < gqa ? sink[(ulong)row_idx * sink_stride + c * gqa + m] : 0.0f;
    l_init[m] = (lane == 0 && m < gqa) ? 1.0f : 0.0f;
  }
  float8 mrow = vload8(0, m_init), lrow = vload8(0, l_init);

  const uint tile_start = (j_first / 32u) * 32u;
  for (uint i0 = tile_start; i0 < j_end; i0 += 32u) {
    uint cell[2], j0[2];
    bool valid[2], full[2];
    float8 s[2];
    for (uint u = 0; u < 2; u++) {
      const uint idx = i0 + 16u * u + lane;
      valid[u] = idx >= j_first && idx < j_end;
      cell[u] = valid[u] ? idx : 0u;
      j0[u] = i0 + 16u * u;
      full[u] = sub_group_all(valid[u]);
      __global const uint *krow = (__global const uint *)(kh + ((ulong)cell[u] * nkv + c) * I2_HEAD_D);
      float8 s0 = (float8)0.0f, s1 = (float8)0.0f;
      __attribute__((opencl_unroll_hint)) for (uint k = 0; k < I2_HEAD_D / 16; k += 2) {
        s0 = dpas_f16(qa[k], as_int8(vload8(0, krow + 8u * k)), s0);
        s1 = dpas_f16(qa[k + 1u], as_int8(vload8(0, krow + 8u * k + 8u)), s1);
      }
      float sv[8];
      vstore8(s0 + s1, 0, sv);
      if (valid[u]) {
        const uint dist = qpos - (k_start + idx);
        for (uint m = 0; m < gqa; m++) {
          if (dist < I2_REL_EXT) sv[m] += rb[((ulong)row_idx * rb_stride + c * gqa + m) * I2_REL_EXT + dist];
        }
        for (uint m = gqa; m < 8; m++) sv[m] = -INFINITY;
      } else {
        for (uint m = 0; m < 8; m++) sv[m] = -INFINITY;
      }
      s[u] = vload8(0, sv);
    }

    const float8 bm = fmax(s[0], s[1]);
    if (sub_group_any(any(bm > mrow + ATT_MAX_SLACK))) {
      float b8[8];
      vstore8(bm, 0, b8);
      for (uint m = 0; m < gqa; m++) b8[m] = sub_group_reduce_max(b8[m]);
      for (uint m = gqa; m < 8; m++) b8[m] = 0.0f;
      const float8 mn = fmax(mrow, vload8(0, b8)), al = exp(mrow - mn);
      for (uint j = 0; j < I2_HEAD_D / 16; j++) o[j] *= al;
      lrow *= al;
      mrow = mn;
    }

    const float8 ea = exp(s[0] - mrow), eb = exp(s[1] - mrow);
    lrow += ea + eb;
    const short8 P[2] = {as_short8(convert_half8(ea)), as_short8(convert_half8(eb))};

    __attribute__((opencl_unroll_hint)) for (uint qd = 0; qd < 2; qd++) {
      __attribute__((opencl_unroll_hint)) for (uint u = 0; u < 2; u++) {
        if (full[u]) {
          __attribute__((opencl_unroll_hint)) for (uint j2 = 0; j2 < 2; j2++) {
            uint ba[8], bb[8];
            __attribute__((opencl_unroll_hint)) for (uint g = 0; g < 4; g++) {
              ushort r8[8];
              intel_sub_group_2d_block_read_16b_4r16x2c((__global void *)vb, I2_HEAD_D * 2u, height, pitch,
                                                        (int2)(64u * qd + 32u * j2, j0[u] + 4u * g), r8);
              ba[2 * g] = (uint)r8[0] | ((uint)r8[1] << 16);
              ba[2 * g + 1] = (uint)r8[2] | ((uint)r8[3] << 16);
              bb[2 * g] = (uint)r8[4] | ((uint)r8[5] << 16);
              bb[2 * g + 1] = (uint)r8[6] | ((uint)r8[7] << 16);
            }
            o[4u * qd + 2u * j2] = dpas_f16(P[u], as_int8(vload8(0, ba)), o[4u * qd + 2u * j2]);
            o[4u * qd + 2u * j2 + 1u] = dpas_f16(P[u], as_int8(vload8(0, bb)), o[4u * qd + 2u * j2 + 1u]);
          }
          continue;
        }
        ushort rv[16][4];
        __attribute__((opencl_unroll_hint)) for (uint k = 0; k < 16; k++) {
          const uint ck = sub_group_broadcast(cell[u], k);
          const ushort4 v = intel_sub_group_block_read_us4(
              (__global const ushort *)(vh + ((ulong)ck * nkv + c) * I2_HEAD_D) + 64u * qd);
          rv[k][0] = v.s0;
          rv[k][1] = v.s1;
          rv[k][2] = v.s2;
          rv[k][3] = v.s3;
        }
        __attribute__((opencl_unroll_hint)) for (uint j = 0; j < 4; j++) {
          uint pr[8];
          __attribute__((opencl_unroll_hint)) for (uint q2 = 0; q2 < 8; q2++)
            pr[q2] = (uint)rv[2u * q2][j] | ((uint)rv[2u * q2 + 1u][j] << 16);
          o[4u * qd + j] = dpas_f16(P[u], as_int8(vload8(0, pr)), o[4u * qd + j]);
        }
      }
    }
  }

  for (uint x = 1; x < 16; x *= 2) lrow += intel_sub_group_shuffle_xor(lrow, x);
  float lr[8];
  vstore8(lrow, 0, lr);
  for (uint j = 0; j < I2_HEAD_D / 16; j++) {
    float ov[8];
    vstore8(o[j], 0, ov);
    const uint d = 64u * (j / 4u) + 16u * (j % 4u) + lane;
    for (uint m = 0; m < gqa; m++) {
      const uint h = c * gqa + m;
      const float gv = sigmoid(gate[(ulong)row_idx * gate_stride + (ulong)h * I2_HEAD_D + d]);
      o16[((ulong)row_idx * nq + h) * I2_HEAD_D + d] = convert_half((ov[m] / lr[m]) * gv);
    }
  }
}

static inline __global const PromptPlan *short_row(__global const PromptPlan *runs, uint y) {
  __global const PromptPlan *pp = runs + y / PROMPT_SHORT_RUN;
  return (pp->n == 0 || pp->n > PROMPT_SHORT_RUN || (y % PROMPT_SHORT_RUN) >= pp->n) ? 0 : pp;
}

// Short-run split-KV FP16 attention for prompt runs of up to PROMPT_SHORT_RUN tokens.
// Groups (nkv, PROMPT_MAX_RUNS * PROMPT_SHORT_RUN, ATT_MAX_SPLITS), local (I2_HEAD_D, 1, 1).
REQD_SG __kernel void p_i2_attn_rows(__global const float *Q, ulong q_stride, __global const float *rb, ulong rb_stride,
                                     __global const half *kc, __global const half *vc, __global const uint *ptab,
                                     uint npp, uint nq, uint nkv, uint gqa, uint is_swa, uint swa, uint swa_cap,
                                     __global const PromptPlan *runs, __global float *part) {
  __local float ql[I2_GQA_MAX][I2_HEAD_D];
  __local float p[I2_GQA_MAX][ATT_SPLIT_TILE];
  __local uint cl[ATT_SPLIT_TILE];
  const uint c = get_group_id(0), y = get_group_id(1), s = get_group_id(2), d = get_local_id(0);
  __global const PromptPlan *pp = short_row(runs, y);
  if (!pp) return;

  const uint t = y % PROMPT_SHORT_RUN;
  const uint r = pp->row0 + t;
  const uint pos = pp->pos0 + t;
  const uint first_kp = (is_swa && pos >= swa) ? (pos - swa + 1u) : 0u;
  const uint nkeys = pos - first_kp + 1u;
  const uint per = i2_attn_per_split(nkeys);
  const uint ns = i2_attn_num_splits(nkeys);
  if (s >= ns) return;

  const uint start_kp = first_kp + s * per;
  const uint end_kp = min(pos + 1u, start_kp + per);
  for (uint h = 0; h < gqa; h++) ql[h][d] = Q[(ulong)r * q_stride + (ulong)(c * gqa + h) * I2_HEAD_D + d];

  const uint sg = sg_id(), lane = sg_lane(), nsg = get_num_sub_groups();
  const ulong cell_stride = (ulong)nkv * I2_HEAD_D;
  const float q_scale = rsqrt((float)I2_HEAD_D);
  __global const half *K = kc + c * I2_HEAD_D;
  __global const half *V = vc + c * I2_HEAD_D;

  float m_wg[I2_GQA_MAX], l_wg[I2_GQA_MAX], acc[I2_GQA_MAX];
  for (uint h = 0; h < gqa; h++) {
    m_wg[h] = -INFINITY;
    l_wg[h] = 0.0f;
    acc[h] = 0.0f;
  }

  for (uint t0 = start_kp; t0 < end_kp; t0 += ATT_SPLIT_TILE) {
    const uint m = min((uint)ATT_SPLIT_TILE, end_kp - t0);
    if (d < m) {
      const uint kp = t0 + d;
      cl[d] = is_swa ? ((uint)pp->slot * swa_cap + (kp % swa_cap))
                     : i2_kv_cell(i2_kv_page(ptab, npp, pp->slot, kp), kp);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint i = sg; i < m; i += nsg) {
      const float8 kv = vload_half8(0, K + (ulong)cl[i] * cell_stride + 8u * lane);
      const uint dist = pos - (t0 + i);
      for (uint h = 0; h < gqa; h++) {
        const __local float *qh = &ql[h][8u * lane];
        float dot8 = kv.s0 * qh[0] + kv.s1 * qh[1] + kv.s2 * qh[2] + kv.s3 * qh[3] + kv.s4 * qh[4] +
                     kv.s5 * qh[5] + kv.s6 * qh[6] + kv.s7 * qh[7];
        dot8 = sg_sum(dot8);
        if (lane == 0) {
          const float b = dist < I2_REL_EXT ? rb[((ulong)r * rb_stride + (c * gqa + h)) * I2_REL_EXT + dist] : 0.0f;
          p[h][i] = dot8 * q_scale + b;
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint h = 0; h < gqa; h++) {
      float mx = -INFINITY;
      for (uint i = 0; i < m; i++) mx = fmax(mx, p[h][i]);
      const float m_new = fmax(m_wg[h], mx);
      const float alpha = m_wg[h] == -INFINITY ? 0.0f : exp(m_wg[h] - m_new);
      float l_tile = 0.0f;
      acc[h] *= alpha;
      for (uint i = 0; i < m; i++) {
        const float w = exp(p[h][i] - m_new);
        l_tile += w;
        acc[h] += w * (float)V[(ulong)cl[i] * cell_stride + d];
      }
      m_wg[h] = m_new;
      l_wg[h] = l_wg[h] * alpha + l_tile;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  __global float *po = part + (((ulong)y * nq + c * gqa) * ATT_MAX_SPLITS + s) * ATT_PART;
  for (uint h = 0; h < gqa; h++) {
    __global float *dst = po + (ulong)h * ATT_MAX_SPLITS * ATT_PART;
    dst[d] = acc[h];
    if (d == 0) {
      dst[I2_HEAD_D] = m_wg[h];
      dst[I2_HEAD_D + 1] = l_wg[h];
    }
  }
}

// Merges short-run attention splits with per-query sinks and writes FP16 `o16`.
// Groups (nq, PROMPT_MAX_RUNS * PROMPT_SHORT_RUN, 1), local (I2_HEAD_D, 1, 1).
REQD_SG __kernel void p_i2_attn_rows_merge(__global const float *part, __global const float *sink, ulong sink_stride,
                                           __global const float *gate, ulong gate_stride, uint nq, uint is_swa,
                                           uint swa, __global const PromptPlan *runs, __global half *o16) {
  const uint lh = get_group_id(0), y = get_group_id(1), d = get_local_id(0);
  __global const PromptPlan *pp = short_row(runs, y);
  if (!pp || lh >= nq) return;

  const uint t = y % PROMPT_SHORT_RUN;
  const uint r = pp->row0 + t;
  const uint pos = pp->pos0 + t;
  const uint first_kp = (is_swa && pos >= swa) ? (pos - swa + 1u) : 0u;
  const uint ns = i2_attn_num_splits(pos - first_kp + 1u);
  __global const float *pt = part + ((ulong)y * nq + lh) * ATT_MAX_SPLITS * ATT_PART;

  const float sk = sink[(ulong)r * sink_stride + lh];
  float mx = sk;
  for (uint s = 0; s < ns; s++) mx = fmax(mx, pt[(ulong)s * ATT_PART + I2_HEAD_D]);
  float sum = exp(sk - mx), acc = 0.0f;
  for (uint s = 0; s < ns; s++) {
    const float f = exp(pt[(ulong)s * ATT_PART + I2_HEAD_D] - mx);
    acc += f * pt[(ulong)s * ATT_PART + d];
    sum += f * pt[(ulong)s * ATT_PART + I2_HEAD_D + 1];
  }
  const float gv = sigmoid(gate[(ulong)r * gate_stride + (ulong)lh * I2_HEAD_D + d]);
  o16[((ulong)r * nq + lh) * I2_HEAD_D + d] = convert_half((acc / sum) * gv);
}
