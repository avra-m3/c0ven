// Streaming hidden-state probe kernels (core/models/qwen4exp/probe.cpp): tapping the four hyper-connection residual
// streams with optional unweighted RMSNorm, accumulating prompt QKV projections across tapped layers, fused sliding-
// window MQA + residual + RMSNorm + classifier head, KV ring update, and MLP head.
#include "builtins.h"
#include "layout.h"
#include "step.h"

static inline float probe_wg_sum(float v, __local float *red) {
  v = sg_sum(v);
  const uint nsg = get_num_sub_groups();
  if (sg_lane() == 0) red[sg_id()] = v;
  barrier(CLK_LOCAL_MEM_FENCE);
  float t = 0.0f;
  for (uint i = 0; i < nsg; i++) t += red[i];
  barrier(CLK_LOCAL_MEM_FENCE);
  return t;
}

static inline float probe_silu(float x) { return x / (1.0f + exp(-x)); }

// Decode step tap: mean of Q_HC residual streams of row r (from plan), optional unweighted RMSNorm, written as BF16
// into out[(r * n_taps + tap_k) * Q_EMBD ..]. Grid: (W, 1, 1), local 256.
REQD_SG __kernel void probe_tap(__global const float *R, uint tap_k, uint n_taps, __global const StepPlan *plan,
                                float eps, uint do_norm, __global ushort *out) {
  __local float red[16];
  const uint r = get_group_id(0);
  if (r >= plan->nrows) return;
  const uint lid = get_local_id(0), lsz = get_local_size(0);
  __global const float *base = R + (ulong)r * Q_HC_DIM;
  float ss = 0.0f;
  if (do_norm) {
    for (uint i = lid; i < Q_EMBD; i += lsz) {
      float m = base[i];
      for (uint s = 1; s < Q_HC; s++) m += base[s * Q_EMBD + i];
      m *= Q_HC_MEAN;
      ss += m * m;
    }
    ss = probe_wg_sum(ss, red);
  }
  const float inv = do_norm ? rsqrt(ss / (float)Q_EMBD + eps) : 1.0f;
  __global ushort *dst = out + ((ulong)r * n_taps + tap_k) * Q_EMBD;
  for (uint i = lid; i < Q_EMBD; i += lsz) {
    float m = base[i];
    for (uint s = 1; s < Q_HC; s++) m += base[s * Q_EMBD + i];
    dst[i] = f32_to_bf16(m * Q_HC_MEAN * inv);
  }
}

// Prompt chunk tap: rows r0 .. r0 + *nrows - 1 of P.R into contiguous [rows, Q_EMBD] BF16 buffer (P.xb).
// Grid: (max_rows, 1, 1), local 256.
REQD_SG __kernel void p_probe_tap(__global const float *R, uint r0, __global const uint *nrows, float eps, uint do_norm,
                                  __global ushort *out) {
  __local float red[16];
  const uint r = get_group_id(0);
  if (r >= *nrows) return;
  const uint lid = get_local_id(0), lsz = get_local_size(0);
  __global const float *base = R + (ulong)(r0 + r) * Q_HC_DIM;
  float ss = 0.0f;
  if (do_norm) {
    for (uint i = lid; i < Q_EMBD; i += lsz) {
      float m = base[i];
      for (uint s = 1; s < Q_HC; s++) m += base[s * Q_EMBD + i];
      m *= Q_HC_MEAN;
      ss += m * m;
    }
    ss = probe_wg_sum(ss, red);
  }
  const float inv = do_norm ? rsqrt(ss / (float)Q_EMBD + eps) : 1.0f;
  __global ushort *dst = out + (ulong)r * Q_EMBD;
  for (uint i = lid; i < Q_EMBD; i += lsz) {
    float m = base[i];
    for (uint s = 1; s < Q_HC; s++) m += base[s * Q_EMBD + i];
    dst[i] = f32_to_bf16(m * Q_HC_MEAN * inv);
  }
}

// Elementwise copy (add == 0) or accumulate (add != 0) of (*nrows * cols) FP32 values. Grid: ((max*cols+255)/256), 256.
__kernel void probe_accum(__global float *dst, __global const float *src, __global const uint *nrows, uint cols,
                          uint add) {
  const uint idx = get_global_id(0);
  const ulong total = (ulong)(*nrows) * cols;
  if ((ulong)idx >= total) return;
  dst[idx] = add ? (dst[idx] + src[idx]) : src[idx];
}

// Fused SingProbeAttn head for n consecutive rows [row0 .. row0 + n - 1] at positions [pos0 .. pos0 + n - 1].
// Reads K/V from slot's BF16 ring for t < pos0 and from qkv (BF16-rounded) for pos0 <= t <= pos, so all n rows run
// in parallel with no ring collision. Also copies n_lin linear/drift scalars (adding bias for non-drift) to out.
// Supports qd = H * D <= 1024, D <= 64 (multiple of 16), H <= 16. Grid: (n, 1, 1), local 128 (8 sub-groups of 16).
REQD_SG __kernel void probe_mqa_head(__global const float *qkv, uint qkv_stride, uint row0, uint pos0, uint slot_start,
                                     uint n, uint H, uint D, uint window, __global const ushort *k_ring,
                                     __global const ushort *v_ring, __global const ushort *o_proj,
                                     __global const float *norm_w, float eps, __global const float *cls_w,
                                     __global const float *cls_b, uint base_labels, __global const float *lin_b,
                                     __global const uint *lin_is_drift, uint n_lin, __global float *out,
                                     uint out_stride) {
  __local float attn_buf[1024];
  __local float hid_buf[1024];
  __local float red[16];

  const uint j = get_group_id(0);
  if (j >= n) return;
  const uint lid = get_local_id(0), lsz = get_local_size(0);
  const uint sg = sg_id(), lane = sg_lane(), nsg = get_num_sub_groups();
  const uint qd = H * D;
  const uint pos = pos0 + j;
  const uint lo = max(slot_start, pos + 1 > window ? pos + 1 - window : 0u);
  const float scale = rsqrt((float)D);
  __global const float *row_qkv = qkv + (ulong)(row0 + j) * qkv_stride;

  // Causal sliding-window MQA with online softmax; one sub-group handles heads sg, sg + nsg, ...
  for (uint h = sg; h < H; h += nsg) {
    float qv[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const uint chunks = D / SG_SIZE;
    for (uint c = 0; c < chunks; c++) qv[c] = row_qkv[h * D + c * SG_SIZE + lane];
    float mx = -INFINITY, lsum = 0.0f;
    for (uint t = lo; t <= pos; t++) {
      float kv[4], vv[4];
      if (t < pos0) {
        const ulong cell = (ulong)(t % window) * D;
        for (uint c = 0; c < chunks; c++) {
          const uint d = c * SG_SIZE + lane;
          kv[c] = bf16_to_f32(k_ring[cell + d]);
          vv[c] = bf16_to_f32(v_ring[cell + d]);
        }
      } else {
        __global const float *tk = qkv + (ulong)(row0 + (t - pos0)) * qkv_stride + qd;
        __global const float *tv = tk + D;
        for (uint c = 0; c < chunks; c++) {
          const uint d = c * SG_SIZE + lane;
          kv[c] = bf16_to_f32(f32_to_bf16(tk[d]));
          vv[c] = bf16_to_f32(f32_to_bf16(tv[d]));
        }
      }
      float dot = 0.0f;
      for (uint c = 0; c < chunks; c++) dot += qv[c] * kv[c];
      const float s = sg_sum(dot) * scale;
      const float nmx = fmax(mx, s);
      const float alpha = mx == -INFINITY ? 0.0f : exp(mx - nmx);
      const float beta = exp(s - nmx);
      lsum = lsum * alpha + beta;
      for (uint c = 0; c < chunks; c++) acc[c] = acc[c] * alpha + beta * vv[c];
      mx = nmx;
    }
    const float inv = lsum > 0.0f ? 1.0f / lsum : 0.0f;
    for (uint c = 0; c < chunks; c++) attn_buf[h * D + c * SG_SIZE + lane] = acc[c] * inv;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // Output projection o_proj @ attn + q, then RMSNorm(eps) * norm_w.
  float ss = 0.0f;
  for (uint r = lid; r < qd; r += lsz) {
    __global const ushort *orow = o_proj + (ulong)r * qd;
    float s = row_qkv[r];
    for (uint c = 0; c < qd; c++) s += bf16_to_f32(orow[c]) * attn_buf[c];
    hid_buf[r] = s;
    ss += s * s;
  }
  ss = probe_wg_sum(ss, red);
  const float rinv = rsqrt(ss / (float)qd + eps);
  for (uint r = lid; r < qd; r += lsz) hid_buf[r] = hid_buf[r] * rinv * norm_w[r];
  barrier(CLK_LOCAL_MEM_FENCE);

  // Linear classifier head plus optional linear/drift passthrough.
  __global float *dst = out + (ulong)j * out_stride;
  for (uint l = lid; l < base_labels; l += lsz) {
    __global const float *cw = cls_w + (ulong)l * qd;
    float s = cls_b[l];
    for (uint r = 0; r < qd; r++) s += cw[r] * hid_buf[r];
    dst[l] = s;
  }
  for (uint li = lid; li < n_lin; li += lsz) {
    const float d = row_qkv[qd + 2 * D + li];
    dst[base_labels + li] = lin_is_drift[li] ? d : (d + lin_b[li]);
  }
}

// Writes the last m (<= window) rows of a committed run into the slot's BF16 K and V rings.
// Grid: ((m * D + 63) / 64, 1, 1), local 64.
__kernel void probe_kv_write(__global const float *qkv, uint qkv_stride, uint qd, uint D, uint row0, uint pos0, uint m,
                             uint window, __global ushort *k_ring, __global ushort *v_ring) {
  const uint idx = get_global_id(0);
  if (idx >= m * D) return;
  const uint j = idx / D, d = idx % D;
  const uint cell = (pos0 + j) % window;
  __global const float *rk = qkv + (ulong)(row0 + j) * qkv_stride + qd;
  k_ring[(ulong)cell * D + d] = f32_to_bf16(rk[d]);
  v_ring[(ulong)cell * D + d] = f32_to_bf16(rk[D + d]);
}

// SingProbeMlp + optional linear/drift head for n rows in proj [rows, mid + n_lin]. Grid: (n, 1, 1), local 64.
__kernel void probe_mlp_head(__global const float *proj, uint proj_stride, uint row0, uint n, uint mid,
                             __global const float *fc1_b, __global const float *fc2_w, __global const float *fc2_b,
                             uint base_labels, __global const float *lin_b, __global const uint *lin_is_drift,
                             uint n_lin, __global float *out, uint out_stride) {
  const uint j = get_group_id(0);
  if (j >= n) return;
  const uint lid = get_local_id(0), lsz = get_local_size(0);
  __global const float *row = proj + (ulong)(row0 + j) * proj_stride;
  __global float *dst = out + (ulong)j * out_stride;
  for (uint l = lid; l < base_labels; l += lsz) {
    __global const float *w2 = fc2_w + (ulong)l * mid;
    float s = fc2_b[l];
    for (uint i = 0; i < mid; i++) s += w2[i] * probe_silu(row[i] + fc1_b[i]);
    dst[l] = s;
  }
  for (uint li = lid; li < n_lin; li += lsz) {
    const float d = row[mid + li];
    dst[base_labels + li] = lin_is_drift[li] ? d : (d + lin_b[li]);
  }
}

// Pure linear/drift probe head: copies n rows from proj [rows, n_lin], adding bias for non-drift labels.
// Grid: ((n * n_lin + 63) / 64, 1, 1), local 64.
__kernel void probe_linear_head(__global const float *proj, uint proj_stride, uint row0, uint n,
                                __global const float *lin_b, __global const uint *lin_is_drift, uint n_lin,
                                __global float *out, uint out_stride) {
  const uint idx = get_global_id(0);
  if (idx >= n * n_lin) return;
  const uint j = idx / n_lin, li = idx % n_lin;
  const float d = proj[(ulong)(row0 + j) * proj_stride + li];
  out[(ulong)j * out_stride + li] = lin_is_drift[li] ? d : (d + lin_b[li]);
}
