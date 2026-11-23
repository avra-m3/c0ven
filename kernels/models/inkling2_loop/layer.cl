// Decode-step kernels for inkling2_loop: attention-residuals mixing, loop control, causal shortconv,
// relative-position FP16 SWA/global attention with sinks, sigmoid top-8 routing, and general n-card allreduce.
#include "common_i2.h"
#include "step.h"
#include "topk.h"

#define RT_SG SG_SIZE
#include "route.h"

#define ROW_OR_RETURN(r) \
  if ((r) >= plan->nrows) return;

REQD_SG __kernel void i2_push_norm(__global const float *src, ulong src_stride, float eps,
                                   __global const StepPlan *plan, __global float *dst_base, uint push_idx,
                                   uint max_rows) {
  __local float red[64];
  const uint r = get_group_id(1);
  ROW_OR_RETURN(r)
  i2_push_norm_row(src + (ulong)r * src_stride, eps, dst_base + ((ulong)push_idx * max_rows + r) * I2_EMBD, red);
}

REQD_SG __kernel void i2_res_mix(__global const float *s_norm, uint nb, uint s_max_rows, __global const float *cur,
                                 ulong cur_stride, __global const float *queries, uint nq, float eps,
                                 __global const StepPlan *plan, __global float *out, ulong out_stride,
                                 __global float *push_norm, uint push_idx, uint p_max_rows) {
  __local float red[64];
  __local float weights[4][I2_RES_MAX + 1];
  const uint r = get_group_id(1);
  ROW_OR_RETURN(r)
  i2_res_mix_row(s_norm, nb, s_max_rows, cur + (ulong)r * cur_stride, r, queries, nq, eps, out, out_stride,
                 push_norm, push_idx, p_max_rows, red, weights);
}

REQD_SG __kernel void i2_ctrl_prep(__global const float *cur, ulong cur_stride, __global const float *msg,
                                   ulong msg_stride, float eps, __global const StepPlan *plan, __global float *out,
                                   ulong out_stride) {
  __local float red[64];
  const uint side = get_group_id(0), r = get_group_id(1);
  ROW_OR_RETURN(r)
  __global const float *src = side == 0 ? cur + (ulong)r * cur_stride : msg + (ulong)r * msg_stride;
  i2_push_norm_row(src, eps, out + (ulong)r * out_stride + (ulong)side * I2_EMBD, red);
}

// F32 matrix-vector multiply Y[r][row] = sum_k W[row][k] * X[r][k] across all rows in `plan`.
// Groups (ceil(rows / 8), 1, 1), local (128, 1, 1): 8 sub-groups per work-group, 1 row per sub-group.
REQD_SG __kernel void i2_f32_matvec(__global const float *W, __global const float *X, ulong x_stride,
                                    __global const StepPlan *plan, __global float *Y, ulong y_stride, uint rows,
                                    uint K) {
  const uint row = get_group_id(0) * 8 + sg_id(), lane = sg_lane();
  if (row >= rows) return;
  const uint nr = plan->nrows;
  __global const float *wr = W + (ulong)row * K;
  float acc[STEP_MAX_ROWS];
  for (uint r = 0; r < nr; r++) acc[r] = 0.0f;
  for (uint k = lane; k < K; k += SG_SIZE) {
    const float wv = wr[k];
    for (uint r = 0; r < nr; r++) acc[r] += wv * X[(ulong)r * x_stride + k];
  }
  for (uint r = 0; r < nr; r++) {
    const float v = sg_sum(acc[r]);
    if (lane == 0) Y[(ulong)r * y_stride + row] = v;
  }
}

// Fuses gain/forget projections from rank-64 `gate` and applies (1 - tanh(forget)) * cur + tanh(gain) * msg.
// Groups (ceil(I2_EMBD / 8), 1, 1), local (128, 1, 1).
REQD_SG __kernel void i2_ctrl_apply(__global const float *gate, ulong gate_stride, __global const float *gain_w,
                                    __global const float *forget_w, __global const float *cur, ulong cur_stride,
                                    __global const float *msg, ulong msg_stride, __global const StepPlan *plan,
                                    __global float *out, ulong out_stride) {
  const uint i = get_group_id(0) * 8 + sg_id(), lane = sg_lane();
  if (i >= I2_EMBD) return;
  const uint nr = plan->nrows;
  const float4 gw = vload4(lane, gain_w + (ulong)i * I2_GATE_RANK);
  const float4 fw = vload4(lane, forget_w + (ulong)i * I2_GATE_RANK);
  for (uint r = 0; r < nr; r++) {
    const float4 g = vload4(lane, gate + (ulong)r * gate_stride);
    const float gv = sg_sum(dot(gw, g));
    const float fv = sg_sum(dot(fw, g));
    if (lane == 0) {
      const float cv = cur[(ulong)r * cur_stride + i];
      const float mv = msg[(ulong)r * msg_stride + i];
      out[(ulong)r * out_stride + i] = cv - tanh(fv) * cv + tanh(gv) * mv;
    }
  }
}

// Weighted RMS norm for `nstreams` streams (`s = group x`): writes float `out_f` and/or `a8` (`out_a8`).
// Groups (nstreams, width, 1), local (256, 1, 1).
REQD_SG __kernel void i2_rms_norm(__global const float *x, ulong x_stride, __global const float *gamma, float eps,
                                  __global const StepPlan *plan, __global float *out_f, ulong out_f_stride,
                                  __global uchar *out_a8, ulong a8_stride, uint nstreams, uint share_gamma) {
  __local float red[64];
  __local float buf[I2_EMBD];
  const uint s = get_group_id(0), r = get_group_id(1), lid = get_local_id(0);
  if (s >= nstreams) return;
  ROW_OR_RETURN(r)
  __global const float *src = x + (ulong)s * x_stride + (ulong)r * I2_EMBD;
  __global const float *gw = gamma + (share_gamma ? 0 : (ulong)s * I2_EMBD);
  float ss = 0.0f;
  for (uint i = lid; i < I2_EMBD; i += I2_NORM_WG) {
    const float v = src[i];
    ss += v * v;
  }
  ss = wg_sum(ss, red);
  const float rs = rsqrt(ss / (float)I2_EMBD + eps);
  for (uint i = lid; i < I2_EMBD; i += I2_NORM_WG) {
    const float y = src[i] * rs * gw[i];
    if (out_f) out_f[(ulong)s * out_f_stride + (ulong)r * I2_EMBD + i] = y;
    buf[i] = y;
  }
  if (!out_a8) return;
  barrier(CLK_LOCAL_MEM_FENCE);
  i2_store_a8_slm(buf, out_a8 + (ulong)s * a8_stride + (ulong)r * A8_BYTES(I2_EMBD), I2_EMBD);
}

// Causal short convolution over channels `0 .. nch - 1`. Groups (ceil(nch / 64), 1, 1), local (64, 1, 1).
REQD_SG __kernel void i2_shortconv_step(__global const float *x_in, ulong in_stride, __global const float *w,
                                        __global const float *dyn, uint nkv_dyn, __global float *state,
                                        ulong slot_stride, ulong layer_st_off, uint total_ch, uint ch_off, uint nch,
                                        __global const StepPlan *plan, __global float *x_out, ulong out_stride,
                                        __global float *w_conv, ulong w_stride) {
  const uint ch = get_global_id(0);
  if (ch >= nch) return;
  __global const float *kw = w + (ulong)ch * I2_CONV;
  const bool window = plan->window != 0;
  const uint nr = plan->nrows;
  const float inv_scale = dyn ? (1.0f / 3.0f) : rsqrt(5.0f);
  const uint h = dyn ? ch / I2_HEAD_D : 0;

  for (uint r = 0; r < nr; r++) {
    const uint pos = plan->rows[r].pos;
    const uint slot = plan->rows[r].slot;
    __global float *st = state + (ulong)slot * slot_stride + layer_st_off + ch_off;

    float mv[I2_HIST];
    for (uint d = 1; d <= I2_HIST; d++) {
      mv[I2_HIST - d] = (window && r >= d && plan->rows[r - d].slot == slot)
                            ? x_in[(ulong)(r - d) * in_stride + ch]
                            : st[((pos + I2_HIST - d) % I2_HIST) * total_ch + ch];
    }
    const float xt = x_in[(ulong)r * in_stride + ch];
    float acc = xt + kw[0] * mv[0] + kw[1] * mv[1] + kw[2] * mv[2] + kw[3] * xt;
    if (dyn) {
      __global const float *dp = dyn + ((ulong)r * nkv_dyn + h) * I2_CONV;
      acc += dp[0] * xt + dp[1] * mv[2] + dp[2] * mv[1] + dp[3] * mv[0];
    }
    x_out[(ulong)r * out_stride + ch] = acc * inv_scale;
    if (window)
      w_conv[(ulong)r * w_stride + ch_off + ch] = xt;
    else
      st[(pos % I2_HIST) * total_ch + ch] = xt;
  }
}

// Commits accepted window shortconv inputs across all logical layers. Groups (ceil(total_ch / 64), logical, 1), (64).
REQD_SG __kernel void i2_shortconv_commit(__global const float *w_conv, __global float *state, ulong slot_stride,
                                          uint total_ch, ulong layer_w_stride, ulong layer_st_stride,
                                          __global const CommitPlan *cp) {
  const uint ch = get_global_id(0), il = get_group_id(1);
  if (ch >= total_ch) return;
  __global const float *wc = w_conv + (ulong)il * layer_w_stride;
  __global float *st_layer = state + (ulong)il * layer_st_stride;
  for (uint k = 0; k < cp->nruns; k++) {
    __global const CommitRun *run = cp->runs + k;
    __global float *st = st_layer + (ulong)run->slot * slot_stride;
    for (uint r = 0; r < run->n; r++)
      st[((run->pos0 + r) % I2_HIST) * total_ch + ch] = wc[(ulong)(run->row0 + r) * total_ch + ch];
  }
}

REQD_SG __kernel void i2_qkvr_prep(__global float *Q, ulong q_stride, __global const float *q_clip,
                                   __global const float *K, ulong k_stride, __global const float *V, ulong v_stride,
                                   __global float *R, ulong r_stride, __global const float *r_clip, uint nq, uint nkv,
                                   uint gqa, uint is_swa, float eps, float qk_clip, float rel_clip, uint log_floor,
                                   float log_alpha, uint swa_cap, __global const uint *ptab, uint npp,
                                   __global half *kc, __global half *vc, __global const StepPlan *plan) {
  const uint blk = get_group_id(0), r = get_group_id(1);
  ROW_OR_RETURN(r)
  i2_qkvr_prep_blk(blk, r, plan->rows[r].slot, plan->rows[r].pos, Q, q_stride, q_clip, K, k_stride, V, v_stride, R,
                   r_stride, r_clip, nq, nkv, gqa, is_swa, eps, qk_clip, rel_clip, log_floor, log_alpha, swa_cap,
                   ptab, npp, kc, vc, 0, 0);
}

// Projects R onto relative-position bias table. Groups (nq, width, I2_REL_EXT / 64), local (64, 1, 1).
REQD_SG __kernel void i2_rel_table(__global const float *R, ulong r_stride, __global const float *rel_bias, uint nq,
                                   __global const StepPlan *plan, __global float *rb, ulong rb_stride) {
  __local float rl[I2_REL_D];
  const uint h = get_group_id(0), r = get_group_id(1), lid = get_local_id(0);
  const uint dist = get_group_id(2) * 64u + lid;
  if (h >= nq) return;
  ROW_OR_RETURN(r)
  if (lid < I2_REL_D) rl[lid] = R[(ulong)r * r_stride + (ulong)h * I2_REL_D + lid];
  barrier(CLK_LOCAL_MEM_FENCE);
  if (dist >= I2_REL_EXT) return;
  __global const float *b = rel_bias + (ulong)dist * I2_REL_D;
  float acc = 0.0f;
  for (uint i = 0; i < I2_REL_D; i++) acc += rl[i] * b[i];
  rb[((ulong)r * rb_stride + h) * I2_REL_EXT + dist] = acc;
}

// Split-KV FP16 decode attention. Groups (nkv, width, I2_ATT_MAX_SPLITS), local (I2_HEAD_D, 1, 1).
REQD_SG __kernel void i2_attn_split(__global const float *Q, ulong q_stride, __global const float *rb, ulong rb_stride,
                                    __global const half *kc, __global const half *vc, __global const uint *ptab,
                                    uint npp, uint nq, uint nkv, uint gqa, uint is_swa, uint swa, uint swa_cap,
                                    __global const StepPlan *plan, __global float *part) {
  __local float ql[I2_GQA_MAX][I2_HEAD_D];
  __local float p[I2_GQA_MAX][I2_ATT_SPLIT_TILE];
  __local uint cl[I2_ATT_SPLIT_TILE];
  const uint c = get_group_id(0), r = get_group_id(1), s = get_group_id(2), d = get_local_id(0);
  ROW_OR_RETURN(r)
  const uint pos = plan->rows[r].pos, slot = plan->rows[r].slot;
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

  for (uint t0 = start_kp; t0 < end_kp; t0 += I2_ATT_SPLIT_TILE) {
    const uint m = min((uint)I2_ATT_SPLIT_TILE, end_kp - t0);
    if (d < m) {
      const uint kp = t0 + d;
      cl[d] = is_swa ? ((uint)slot * swa_cap + (kp % swa_cap))
                     : i2_kv_cell(i2_kv_page(ptab, npp, slot, kp), kp);
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

  __global float *po = part + (((ulong)r * nq + c * gqa) * I2_ATT_MAX_SPLITS + s) * I2_ATT_PART;
  for (uint h = 0; h < gqa; h++) {
    __global float *dst = po + (ulong)h * I2_ATT_MAX_SPLITS * I2_ATT_PART;
    dst[d] = acc[h];
    if (d == 0) {
      dst[I2_HEAD_D] = m_wg[h];
      dst[I2_HEAD_D + 1] = l_wg[h];
    }
  }
}

// Merges splits with attention sink logit, applies sigmoid(gate), and writes float and a8.
// Groups (nq, width, 1), local (I2_HEAD_D, 1, 1).
REQD_SG __kernel void i2_attn_merge(__global const float *part, __global const float *sink, ulong sink_stride,
                                    __global const float *gate, ulong gate_stride, uint nq, uint is_swa, uint swa,
                                    __global const StepPlan *plan, __global float *out, ulong out_stride,
                                    __global uchar *oa) {
  __local float os[I2_HEAD_D];
  const uint lh = get_group_id(0), r = get_group_id(1), d = get_local_id(0);
  if (lh >= nq) return;
  ROW_OR_RETURN(r)
  const uint pos = plan->rows[r].pos;
  const uint first_kp = (is_swa && pos >= swa) ? (pos - swa + 1u) : 0u;
  const uint ns = i2_attn_num_splits(pos - first_kp + 1u);
  __global const float *pt = part + ((ulong)r * nq + lh) * I2_ATT_MAX_SPLITS * I2_ATT_PART;
  const float y = i2_attn_merge_dim(pt, sink[(ulong)r * sink_stride + lh],
                                    gate[(ulong)r * gate_stride + (ulong)lh * I2_HEAD_D + d], ns, d);
  if (out) out[(ulong)r * out_stride + (ulong)lh * I2_HEAD_D + d] = y;
  if (!oa) return;
  os[d] = y;
  barrier(CLK_LOCAL_MEM_FENCE);
  const uint sg = sg_id(), lane = sg_lane();
  if (sg < I2_HEAD_D / 32u) {
    a8_store(vload2(lane, os + 32u * sg), oa + (ulong)r * A8_BYTES(nq * I2_HEAD_D), nq * I2_HEAD_D,
             lh * (I2_HEAD_D / 32u) + sg);
  }
}

#define GROUP_WG 256
REQD_SG __kernel void i2_router_group(__global const float *rl, ulong rl_stride, __global const float *bias,
                                      float scale, __global const StepPlan *plan, __global int *ids_all,
                                      __global float *w_all, __global MoeGroup *groups, __global uint *ngroups) {
  __local uint2 cand[STEP_MAX_ROWS * RT_SLM];
  __local float probs[STEP_MAX_ROWS * I2_EXPERTS];
  __local int sl[GROUP_WG];
  __local uint head_of[GROUP_WG], scan[GROUP_WG];

  const uint sg = sg_id(), lid = get_local_id(0);
  const uint nr = plan->nrows;
  if (sg < nr) {
    i2_route_row(rl + (ulong)sg * rl_stride, bias, scale, cand + sg * RT_SLM, probs + sg * I2_EXPERTS,
                 ids_all + sg * I2_USED, w_all + sg * I2_USED);
  }
  barrier(CLK_GLOBAL_MEM_FENCE);

  const uint n = nr * I2_USED;
  const int e = lid < n ? ids_all[lid] : -1;
  sl[lid] = e;
  barrier(CLK_LOCAL_MEM_FENCE);

  bool head = false;
  if (e >= 0) {
    uint prev_cnt = 0;
    for (uint b = 0; b < lid; b++) prev_cnt += (sl[b] == e);
    head = (prev_cnt % MOE_GROUP_MAX) == 0;
  }
  scan[lid] = head ? 1u : 0u;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint off = 1; off < GROUP_WG; off <<= 1) {
    const uint v = lid >= off ? scan[lid - off] : 0u;
    barrier(CLK_LOCAL_MEM_FENCE);
    scan[lid] += v;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (head) head_of[lid] = scan[lid] - 1u;
  barrier(CLK_LOCAL_MEM_FENCE);

  if (e >= 0 && head) {
    const uint g = head_of[lid];
    uint cnt = 0;
    for (uint b = lid; b < n && cnt < MOE_GROUP_MAX; b++) {
      if (sl[b] == e) groups[g].asg[cnt++] = b;
    }
    groups[g].slot = (uint)e;
    groups[g].n = cnt;
  }
  if (lid == GROUP_WG - 1u) *ngroups = scan[GROUP_WG - 1u];
}

__kernel void i2_moe_combine(__global const float *y, __global const float *w, __global const StepPlan *plan,
                             __global float *out, ulong out_stride) {
  const uint i = get_global_id(0), r = get_group_id(1);
  ROW_OR_RETURN(r)
  if (i >= I2_EMBD) return;
  float acc = 0.0f;
  for (uint k = 0; k < I2_USED; k++) acc += w[r * I2_USED + k] * y[((ulong)r * I2_USED + k) * I2_EMBD + i];
  out[(ulong)r * out_stride + i] = acc;
}

__kernel void i2_allreduce_add(__global const float *local_p, ulong local_stride, __global const float *recv,
                               ulong recv_stride, uint ncards, uint width, __global const StepPlan *plan,
                               __global float *out, ulong out_stride) {
  const uint i = get_global_id(0), r = get_group_id(1);
  ROW_OR_RETURN(r)
  if (i >= I2_EMBD) return;
  float sum = local_p[(ulong)r * local_stride + i];
  for (uint p = 0; p + 1 < ncards; p++) sum += recv[((ulong)p * width + r) * recv_stride + i];
  out[(ulong)r * out_stride + i] = sum;
}

__kernel void i2_residual_add(__global const float *a, ulong a_stride, __global const float *b, ulong b_stride,
                              __global const StepPlan *plan, __global float *dst, ulong dst_stride, uint cols) {
  const uint i = get_global_id(0), r = get_group_id(1);
  ROW_OR_RETURN(r)
  if (i >= cols) return;
  dst[(ulong)r * dst_stride + i] = a[(ulong)r * a_stride + i] + b[(ulong)r * b_stride + i];
}

__kernel void i2_softcap(__global float *logits, ulong stride, __global const StepPlan *plan, uint cols, float cap) {
  const uint i = get_global_id(0), r = get_group_id(1);
  ROW_OR_RETURN(r)
  if (i >= cols || cap <= 0.0f) return;
  const ulong idx = (ulong)r * stride + i;
  logits[idx] = cap * tanh(logits[idx] / cap);
}
