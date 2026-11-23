// Inline helpers shared across inkling2_loop OpenCL kernel modules.
#ifndef KERNELS_INKLING2_COMMON_I2_H
#define KERNELS_INKLING2_COMMON_I2_H

#include "a8.h"
#include "builtins.h"
#include "kernel_sizes.h"
#include "moe_tile.h"
#include "step.h"

#ifndef A8_BYTES
#define A8_BYTES(K) ((K) + (K) / 8u)
#endif

#define I2_HIST (I2_CONV - 1)
#define I2_GQA_MAX 8
#define I2_NORM_WG 256
#define I2_ATT_SPLIT_TILE 64
#define I2_ATT_MAX_SPLITS 32
#define I2_ATT_PART (I2_HEAD_D + 2)

static inline float sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }

static inline float wg_sum(float v, __local float *red) {
  v = sg_sum(v);
  const uint nsg = get_num_sub_groups();
  if (sg_lane() == 0) red[sg_id()] = v;
  barrier(CLK_LOCAL_MEM_FENCE);
  float t = 0.0f;
  for (uint i = 0; i < nsg; i++) t += red[i];
  barrier(CLK_LOCAL_MEM_FENCE);
  return t;
}

static inline uint i2_kv_page(__global const uint *pt, uint npp, uint slot, uint pos) {
  return pt[(ulong)slot * npp + pos / KV_PAGE_CELLS];
}

static inline uint i2_kv_cell(uint page, uint pos) {
  return page * KV_PAGE_CELLS + (pos % KV_PAGE_CELLS);
}

static inline uint i2_attn_per_split(uint nkeys) {
  const uint chunks = (nkeys + 63u) / 64u;
  return chunks <= 32u ? 64u : ((chunks + 31u) / 32u) * 64u;
}

static inline uint i2_attn_num_splits(uint nkeys) {
  const uint per = i2_attn_per_split(nkeys);
  return (nkeys + per - 1u) / per;
}

static inline void i2_store_a8_slm(__local const float *buf, __global uchar *out, uint K) {
  const uint sg = sg_id(), nsg = get_num_sub_groups(), lane = sg_lane();
  for (uint u = sg; u < K / 32u; u += nsg) {
    a8_store(vload2(lane, buf + 32u * u), out, K, u);
  }
}

static inline void i2_push_norm_row(__global const float *x, float eps, __global float *dst, __local float *red) {
  const uint lid = get_local_id(0);
  float ss = 0.0f;
  for (uint i = lid; i < I2_EMBD; i += I2_NORM_WG) {
    const float v = x[i];
    ss += v * v;
  }
  ss = wg_sum(ss, red);
  const float rs = rsqrt(ss / (float)I2_EMBD + eps);
  for (uint i = lid; i < I2_EMBD; i += I2_NORM_WG) dst[i] = x[i] * rs;
}

static inline void i2_res_mix_row(__global const float *s_norm, uint nb, uint s_max_rows, __global const float *x,
                                  uint row, __global const float *queries, uint nq, float eps, __global float *out,
                                  ulong out_stride, __global float *push_norm, uint push_idx, uint p_max_rows,
                                  __local float *red, __local float weights[4][I2_RES_MAX + 1]) {
  const uint lid = get_local_id(0);
  float xc[I2_EMBD / I2_NORM_WG];
  float ss = 0.0f;
  for (uint k = 0; k < I2_EMBD / I2_NORM_WG; k++) {
    const uint i = lid + k * I2_NORM_WG;
    const float v = x[i];
    xc[k] = v;
    ss += v * v;
  }
  ss = wg_sum(ss, red);
  const float rs = rsqrt(ss / (float)I2_EMBD + eps);
  for (uint k = 0; k < I2_EMBD / I2_NORM_WG; k++) {
    xc[k] *= rs;
    if (push_norm) {
      const uint i = lid + k * I2_NORM_WG;
      push_norm[((ulong)push_idx * p_max_rows + row) * I2_EMBD + i] = xc[k];
    }
  }

  float dots[4][I2_RES_MAX + 1];
  for (uint q = 0; q < nq; q++)
    for (uint b = 0; b <= nb; b++) dots[q][b] = 0.0f;

  for (uint k = 0; k < I2_EMBD / I2_NORM_WG; k++) {
    const uint i = lid + k * I2_NORM_WG;
    float qv[4];
    for (uint q = 0; q < nq; q++) qv[q] = queries[(ulong)q * I2_EMBD + i];
    for (uint b = 0; b < nb; b++) {
      const float sv = s_norm[((ulong)b * s_max_rows + row) * I2_EMBD + i];
      for (uint q = 0; q < nq; q++) dots[q][b] += sv * qv[q];
    }
    for (uint q = 0; q < nq; q++) dots[q][nb] += xc[k] * qv[q];
  }

  const float scale = rsqrt((float)I2_EMBD);
  for (uint q = 0; q < nq; q++) {
    float sc[I2_RES_MAX + 1];
    float mx = -INFINITY;
    for (uint b = 0; b <= nb; b++) {
      sc[b] = wg_sum(dots[q][b], red) * scale;
      mx = fmax(mx, sc[b]);
    }
    if (lid == 0) {
      float sum = 0.0f;
      for (uint b = 0; b <= nb; b++) {
        const float e = exp(sc[b] - mx);
        weights[q][b] = e;
        sum += e;
      }
      const float inv = 1.0f / sum;
      for (uint b = 0; b <= nb; b++) weights[q][b] *= inv;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  for (uint k = 0; k < I2_EMBD / I2_NORM_WG; k++) {
    const uint i = lid + k * I2_NORM_WG;
    float acc[4];
    for (uint q = 0; q < nq; q++) acc[q] = weights[q][nb] * xc[k];
    for (uint b = 0; b < nb; b++) {
      const float sv = s_norm[((ulong)b * s_max_rows + row) * I2_EMBD + i];
      for (uint q = 0; q < nq; q++) acc[q] += weights[q][b] * sv;
    }
    for (uint q = 0; q < nq; q++) out[(ulong)q * out_stride + (ulong)row * I2_EMBD + i] = acc[q];
  }
}

static inline void i2_qkvr_prep_blk(uint blk, uint row, uint slot, uint pos, __global float *Q, ulong q_stride,
                                    __global const float *q_clip, __global const float *K, ulong k_stride,
                                    __global const float *V, ulong v_stride, __global float *R, ulong r_stride,
                                    __global const float *r_clip, uint nq, uint nkv, uint gqa, uint is_swa, float eps,
                                    float qk_clip, float rel_clip, uint log_floor, float log_alpha, uint swa_cap,
                                    __global const uint *ptab, uint npp, __global half *kc, __global half *vc,
                                    __global half *qt, uint hp) {
  const uint lane = sg_lane();
  const float tau = is_swa ? 1.0f : (1.0f + log_alpha * log(fmax(1.0f, ((float)pos + 1.0f) / (float)log_floor)));

  if (blk < nq) {
    const uint h = blk;
    __global float *qh = Q + (ulong)row * q_stride + (ulong)h * I2_HEAD_D;
    __global const float *qw = q_clip + (ulong)h * I2_HEAD_D;
    float v[I2_HEAD_D / SG_SIZE];
    float ss = 0.0f;
    for (uint j = 0; j < I2_HEAD_D / SG_SIZE; j++) {
      const float x = qh[lane + j * SG_SIZE] * qw[lane + j * SG_SIZE];
      v[j] = x;
      ss += x * x;
    }
    const float norm = sqrt(sg_sum(ss)) + 1e-6f;
    const float factor = fmin(1.0f, qk_clip / norm) * tau;
    const float q_scale = rsqrt((float)I2_HEAD_D);
    const uint c = h / gqa, hk = h % gqa;
    for (uint j = 0; j < I2_HEAD_D / SG_SIZE; j++) {
      const uint d = lane + j * SG_SIZE;
      const float qv = v[j] * factor;
      qh[d] = qv;
      if (qt) {
        const ulong qt_base = ((ulong)row * I2_HEAD_D + d) * hp + c * 8u;
        qt[qt_base + hk] = convert_half(qv * q_scale);
        if (hk == 0) {
          for (uint p = gqa; p < 8u; p++) qt[qt_base + p] = (half)0.0f;
        }
      }
    }
    return;
  }

  const ulong cell = is_swa ? ((ulong)slot * swa_cap + (pos % swa_cap))
                            : i2_kv_cell(i2_kv_page(ptab, npp, slot, pos), pos);

  if (blk < nq + nkv) {
    const uint h = blk - nq;
    __global const float *kh = K + (ulong)row * k_stride + (ulong)h * I2_HEAD_D;
    __global half *kd = kc + (cell * nkv + h) * I2_HEAD_D;
    float v[I2_HEAD_D / SG_SIZE];
    float ss = 0.0f;
    for (uint j = 0; j < I2_HEAD_D / SG_SIZE; j++) {
      const float x = kh[lane + j * SG_SIZE];
      v[j] = x;
      ss += x * x;
    }
    const float norm = sqrt(sg_sum(ss)) + 1e-6f;
    const float factor = fmin(1.0f, qk_clip / norm);
    for (uint j = 0; j < I2_HEAD_D / SG_SIZE; j++) kd[lane + j * SG_SIZE] = convert_half(v[j] * factor);
    return;
  }

  if (blk < nq + 2 * nkv) {
    const uint h = blk - nq - nkv;
    __global const float *vh = V + (ulong)row * v_stride + (ulong)h * I2_HEAD_D;
    __global half *vd = vc + (cell * nkv + h) * I2_HEAD_D;
    float v[I2_HEAD_D / SG_SIZE];
    float ss = 0.0f;
    for (uint j = 0; j < I2_HEAD_D / SG_SIZE; j++) {
      const float x = vh[lane + j * SG_SIZE];
      v[j] = x;
      ss += x * x;
    }
    const float rs = rsqrt(sg_sum(ss) / (float)I2_HEAD_D + eps);
    for (uint j = 0; j < I2_HEAD_D / SG_SIZE; j++) vd[lane + j * SG_SIZE] = convert_half(v[j] * rs);
    return;
  }

  const uint h = blk - nq - 2 * nkv;
  __global float *rh = R + (ulong)row * r_stride + (ulong)h * I2_REL_D;
  __global const float *rw = r_clip + (ulong)h * I2_REL_D;
  const float x = rh[lane] * rw[lane];
  const float norm = sqrt(sg_sum(x * x)) + 1e-6f;
  const float factor = fmin(1.0f, rel_clip / norm) * tau;
  rh[lane] = x * factor;
}

static inline float i2_attn_merge_dim(__global const float *pt, float sk, float gate_val, uint ns, uint d) {
  float mx = sk;
  for (uint s = 0; s < ns; s++) mx = fmax(mx, pt[(ulong)s * I2_ATT_PART + I2_HEAD_D]);
  float sum = exp(sk - mx), acc = 0.0f;
  for (uint s = 0; s < ns; s++) {
    const float f = exp(pt[(ulong)s * I2_ATT_PART + I2_HEAD_D] - mx);
    acc += f * pt[(ulong)s * I2_ATT_PART + d];
    sum += f * pt[(ulong)s * I2_ATT_PART + I2_HEAD_D + 1];
  }
  return (acc / sum) * sigmoid(gate_val);
}

#endif
