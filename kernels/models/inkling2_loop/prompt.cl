// Prompt-chunk kernels for inkling2_loop: depth-wise res_mix, F32 tiled control & router, shortconv across
// runs, Q/K/V/R clipping & qt packing, relative-bias projection, KV gathering, and MoE routing/tiling/combine.
#include "common_i2.h"
#include "moe_tile.h"
#include "step.h"
#include "topk.h"

#define RT_SG SG_SIZE
#include "route.h"

#define ROW_OR_RETURN(t) \
  if ((t) >= pp->n) return;

REQD_SG __kernel void p_i2_push_norm(__global const float *src, ulong src_stride, float eps,
                                     __global const PromptPlan *pp, __global float *dst_base, uint push_idx,
                                     uint s_max_rows) {
  __local float red[64];
  const uint t = get_group_id(1);
  ROW_OR_RETURN(t)
  i2_push_norm_row(src + (ulong)t * src_stride, eps, dst_base + ((ulong)push_idx * s_max_rows + t) * I2_EMBD, red);
}

REQD_SG __kernel void p_i2_res_mix(__global const float *s_norm, uint nb, uint s_max_rows, __global const float *cur,
                                   ulong cur_stride, __global const float *queries, uint nq, float eps,
                                   __global const PromptPlan *pp, __global float *out, ulong out_stride,
                                   __global float *push_norm, uint push_idx, uint p_max_rows) {
  __local float red[64];
  __local float weights[4][I2_RES_MAX + 1];
  const uint t = get_group_id(1);
  ROW_OR_RETURN(t)
  i2_res_mix_row(s_norm, nb, s_max_rows, cur + (ulong)t * cur_stride, t, queries, nq, eps, out, out_stride,
                 push_norm, push_idx, p_max_rows, red, weights);
}

REQD_SG __kernel void p_i2_ctrl_prep(__global const float *cur, ulong cur_stride, __global const float *msg,
                                     ulong msg_stride, float eps, __global const PromptPlan *pp, __global float *out,
                                     ulong out_stride) {
  __local float red[64];
  const uint side = get_group_id(0), t = get_group_id(1);
  ROW_OR_RETURN(t)
  __global const float *src = side == 0 ? cur + (ulong)t * cur_stride : msg + (ulong)t * msg_stride;
  i2_push_norm_row(src, eps, out + (ulong)t * out_stride + (ulong)side * I2_EMBD, red);
}

REQD_SG __kernel void p_i2_f32_gemm(__global const float *W, __global const float *X, ulong x_stride,
                                    __global const PromptPlan *pp, __global float *Y, ulong y_stride, uint rows,
                                    uint K) {
  const uint row = get_group_id(0) * 8 + sg_id(), lane = sg_lane();
  const uint t0 = get_group_id(1) * 8;
  if (row >= rows || t0 >= pp->n) return;
  const uint m = min(8u, pp->n - t0);
  __global const float *wr = W + (ulong)row * K;
  float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  for (uint k = lane; k < K; k += SG_SIZE) {
    const float wv = wr[k];
    for (uint j = 0; j < m; j++) acc[j] += wv * X[(ulong)(t0 + j) * x_stride + k];
  }
  for (uint j = 0; j < m; j++) {
    const float v = sg_sum(acc[j]);
    if (lane == 0) Y[(ulong)(t0 + j) * y_stride + row] = v;
  }
}

REQD_SG __kernel void p_i2_f16_proj(__global const half *W, __global const half *X, ulong x_stride,
                                    __global const PromptPlan *pp, __global float *Y, ulong y_stride, uint rows,
                                    uint K) {
  const uint row = get_group_id(0) * 8 + sg_id(), lane = sg_lane();
  const uint t0 = get_group_id(1) * 8;
  if (row >= rows || t0 >= pp->n) return;
  const uint m = min(8u, pp->n - t0);
  __global const half *wr = W + (ulong)row * K;
  float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  for (uint k = lane; k < K; k += SG_SIZE) {
    const float wv = (float)wr[k];
    for (uint j = 0; j < m; j++) acc[j] += wv * (float)X[(ulong)(t0 + j) * x_stride + k];
  }
  for (uint j = 0; j < m; j++) {
    const float v = sg_sum(acc[j]);
    if (lane == 0) Y[(ulong)(t0 + j) * y_stride + row] = v;
  }
}

REQD_SG __kernel void p_i2_ctrl_apply(__global const float *gate, ulong gate_stride, __global const float *gain_w,
                                      __global const float *forget_w, __global const float *cur, ulong cur_stride,
                                      __global const float *msg, ulong msg_stride, __global const PromptPlan *pp,
                                      __global float *out, ulong out_stride) {
  const uint i = get_group_id(0) * 8 + sg_id(), lane = sg_lane();
  const uint t0 = get_group_id(1) * 8;
  if (i >= I2_EMBD || t0 >= pp->n) return;
  const uint m = min(8u, pp->n - t0);
  const float4 gw = vload4(lane, gain_w + (ulong)i * I2_GATE_RANK);
  const float4 fw = vload4(lane, forget_w + (ulong)i * I2_GATE_RANK);
  for (uint j = 0; j < m; j++) {
    const uint t = t0 + j;
    const float4 g = vload4(lane, gate + (ulong)t * gate_stride);
    const float gv = sg_sum(dot(gw, g));
    const float fv = sg_sum(dot(fw, g));
    if (lane == 0) {
      const float cv = cur[(ulong)t * cur_stride + i];
      const float mv = msg[(ulong)t * msg_stride + i];
      out[(ulong)t * out_stride + i] = cv - tanh(fv) * cv + tanh(gv) * mv;
    }
  }
}

REQD_SG __kernel void p_i2_rms_norm_f16(__global const float *x, ulong x_stride, __global const float *gamma, float eps,
                                        __global const PromptPlan *pp, __global half *out16, ulong out16_stride,
                                        uint nstreams, uint share_gamma) {
  __local float red[64];
  const uint s = get_group_id(0), t = get_group_id(1), lid = get_local_id(0);
  if (s >= nstreams) return;
  ROW_OR_RETURN(t)
  __global const float *src = x + (ulong)s * x_stride + (ulong)t * I2_EMBD;
  __global const float *gw = gamma + (share_gamma ? 0 : (ulong)s * I2_EMBD);
  float ss = 0.0f;
  for (uint i = lid; i < I2_EMBD; i += I2_NORM_WG) {
    const float v = src[i];
    ss += v * v;
  }
  ss = wg_sum(ss, red);
  const float rs = rsqrt(ss / (float)I2_EMBD + eps);
  __global half *dst = out16 + (ulong)s * out16_stride + (ulong)t * I2_EMBD;
  for (uint i = lid; i < I2_EMBD; i += I2_NORM_WG) dst[i] = convert_half(src[i] * rs * gw[i]);
}

REQD_SG __kernel void p_i2_shortconv(__global const float *x_in, ulong in_stride, __global const float *w,
                                     __global const float *dyn, ulong dyn_stride, uint nkv_dyn, __global float *state,
                                     ulong slot_stride, ulong layer_st_off, uint total_ch, uint ch_off, uint nch,
                                     __global const PromptPlan *runs, __global float *x_out, ulong out_stride) {
  const uint ch = get_global_id(0);
  if (ch >= nch) return;
  __global const PromptPlan *rp = runs + get_group_id(2);
  if (rp->n == 0) return;

  __global const float *kw = w + (ulong)ch * I2_CONV;
  __global float *st = state + (ulong)rp->slot * slot_stride + layer_st_off + ch_off;
  const float inv_scale = dyn ? (1.0f / 3.0f) : rsqrt(5.0f);
  const uint h = dyn ? ch / I2_HEAD_D : 0;

  for (uint t = 0; t < rp->n; t++) {
    const uint row = rp->row0 + t;
    const uint pos = rp->pos0 + t;
    float mv[I2_HIST];
    for (uint d = 1; d <= I2_HIST; d++) {
      mv[I2_HIST - d] = t >= d ? x_in[(ulong)(row - d) * in_stride + ch]
                               : st[((pos + I2_HIST - d) % I2_HIST) * total_ch + ch];
    }
    const float xt = x_in[(ulong)row * in_stride + ch];
    float acc = xt + kw[0] * mv[0] + kw[1] * mv[1] + kw[2] * mv[2] + kw[3] * xt;
    if (dyn) {
      __global const float *dp = dyn + (ulong)row * dyn_stride + (ulong)h * I2_CONV;
      acc += dp[0] * xt + dp[1] * mv[2] + dp[2] * mv[1] + dp[3] * mv[0];
    }
    x_out[(ulong)row * out_stride + ch] = acc * inv_scale;
    st[(pos % I2_HIST) * total_ch + ch] = xt;
  }
}

REQD_SG __kernel void p_i2_qkvr_prep(__global float *Q, ulong q_stride, __global const float *q_clip,
                                     __global const float *K, ulong k_stride, __global const float *V, ulong v_stride,
                                     __global float *R, ulong r_stride, __global const float *r_clip, uint nq, uint nkv,
                                     uint gqa, uint is_swa, float eps, float qk_clip, float rel_clip, uint log_floor,
                                     float log_alpha, uint swa_cap, __global const uint *ptab, uint npp,
                                     __global half *kc, __global half *vc, __global half *qt, uint hp,
                                     __global const PromptPlan *runs, uint nruns, __global const PromptPlan *chunk) {
  const uint blk = get_group_id(0), t = get_group_id(1);
  if (t >= chunk->n) return;
  uint slot = 0, pos = t;
  for (uint r = 0; r < nruns; r++) {
    __global const PromptPlan *rp = runs + r;
    if (rp->n > 0 && t >= rp->row0 && t < rp->row0 + rp->n) {
      slot = rp->slot;
      pos = rp->pos0 + (t - rp->row0);
      break;
    }
  }
  i2_qkvr_prep_blk(blk, t, slot, pos, Q, q_stride, q_clip, K, k_stride, V, v_stride, R, r_stride, r_clip, nq, nkv,
                   gqa, is_swa, eps, qk_clip, rel_clip, log_floor, log_alpha, swa_cap, ptab, npp, kc, vc, qt, hp);
}

REQD_SG __kernel void p_i2_rel_table(__global const float *R, ulong r_stride, __global const float *rel_bias, uint nq,
                                     __global const PromptPlan *pp, __global float *rb, ulong rb_stride) {
  __local float rl[I2_REL_D];
  const uint h = get_group_id(0), t = get_group_id(1), lid = get_local_id(0);
  const uint dist = get_group_id(2) * 64u + lid;
  if (h >= nq) return;
  ROW_OR_RETURN(t)
  if (lid < I2_REL_D) rl[lid] = R[(ulong)t * r_stride + (ulong)h * I2_REL_D + lid];
  barrier(CLK_LOCAL_MEM_FENCE);
  if (dist >= I2_REL_EXT) return;
  __global const float *b = rel_bias + (ulong)dist * I2_REL_D;
  float acc = 0.0f;
  for (uint i = 0; i < I2_REL_D; i++) acc += rl[i] * b[i];
  rb[((ulong)t * rb_stride + h) * I2_REL_EXT + dist] = acc;
}

__kernel void p_i2_kv_gather(__global const half *kc, __global const half *vc, __global const uint *ptab, uint npp,
                             uint nkv, uint is_swa, uint swa, uint swa_cap, __global const PromptPlan *rp,
                             __global half *kh, __global half *vh) {
  if (rp->n <= PROMPT_SHORT_RUN) return;
  const uint k_start = (is_swa && rp->pos0 >= swa) ? (rp->pos0 - swa + 1u) : 0u;
  const uint ncells = (rp->pos0 + rp->n) - k_start;
  const uint col = get_global_id(0), cols_total = nkv * (I2_HEAD_D / 8u);
  if (col >= cols_total) return;
  const uint h = col / (I2_HEAD_D / 8u), blk8 = col % (I2_HEAD_D / 8u);
  const ulong row_halfs = (ulong)nkv * I2_HEAD_D;

  for (uint j = get_group_id(1); j < ncells; j += get_num_groups(1)) {
    const uint pos = k_start + j;
    const ulong cell = is_swa ? ((ulong)rp->slot * swa_cap + (pos % swa_cap))
                              : i2_kv_cell(i2_kv_page(ptab, npp, rp->slot, pos), pos);
    const ulong src_off = (cell * nkv + h) * I2_HEAD_D + 8u * blk8;
    const ulong dst_off = (ulong)j * row_halfs + (ulong)h * I2_HEAD_D + 8u * blk8;
    *((__global uint4 *)(kh + dst_off)) = *((__global const uint4 *)(kc + src_off));
    *((__global uint4 *)(vh + dst_off)) = *((__global const uint4 *)(vc + src_off));
  }
}

REQD_SG __kernel void p_i2_router(__global const float *rl, ulong rl_stride, __global const float *bias, float scale,
                                  __global const PromptPlan *pp, __global int *ids_all, __global float *w_all) {
  __local uint2 cand[4 * RT_SLM];
  __local float probs[4 * I2_EXPERTS];
  const uint sg = sg_id(), t = get_group_id(0) * 4u + sg;
  ROW_OR_RETURN(t)
  i2_route_row(rl + (ulong)t * rl_stride, bias, scale, cand + sg * RT_SLM, probs + sg * I2_EXPERTS,
               ids_all + (ulong)t * I2_USED, w_all + (ulong)t * I2_USED);
}

__kernel void p_i2_moe_count(__global const int *ids, __global const PromptPlan *pp, __global uint *cnt) {
  const uint a = get_global_id(0);
  if (a >= pp->n * I2_USED) return;
  atomic_inc(&cnt[ids[a]]);
}

__kernel void p_i2_moe_scan(__global uint *cnt, __global const int *slot_map, __global const int *stage_map,
                            __global uint *base, __global uint *cursor, __global MoeTile *tiles_c,
                            __global uint *ntiles_c, __global MoeTile *tiles_s, __global uint *ntiles_s) {
  __local uint sc[I2_EXPERTS], tc[I2_EXPERTS], ts[I2_EXPERTS];
  const uint e = get_local_id(0), n = cnt[e];
  const bool cached = slot_map[e] >= 0;
  const uint nt = (n + MOE_TILE - 1u) / MOE_TILE;
  sc[e] = n;
  tc[e] = cached ? nt : 0u;
  ts[e] = cached ? 0u : nt;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint off = 1u; off < I2_EXPERTS; off <<= 1u) {
    const uint a = e >= off ? sc[e - off] : 0u;
    const uint b = e >= off ? tc[e - off] : 0u;
    const uint c = e >= off ? ts[e - off] : 0u;
    barrier(CLK_LOCAL_MEM_FENCE);
    sc[e] += a;
    tc[e] += b;
    ts[e] += c;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const uint first = sc[e] - n;
  base[e] = first;
  cursor[e] = 0u;
  cnt[e] = 0u;
  const uint tile0 = cached ? tc[e] - nt : ts[e] - nt;
  __global MoeTile *list = cached ? tiles_c : tiles_s;
  const uint slot = (uint)(cached ? slot_map[e] : stage_map[e]);
  for (uint i = 0; i < nt; i++)
    list[tile0 + i] = (MoeTile){(uint)e, slot, first + i * MOE_TILE, min((uint)MOE_TILE, n - i * MOE_TILE)};
  if (e == I2_EXPERTS - 1u) {
    *ntiles_c = tc[e];
    *ntiles_s = ts[e];
  }
}

__kernel void p_i2_moe_fill(__global const int *ids, __global const PromptPlan *pp, __global const uint *base,
                            __global uint *cursor, __global uint *inv) {
  const uint a = get_global_id(0);
  if (a >= pp->n * I2_USED) return;
  const uint e = (uint)ids[a];
  inv[a] = base[e] + atomic_inc(&cursor[e]);
}

// Groups (1, T, 1), local (288, 1, 1) -> I2_EMBD / 128 = 18 sub-groups * 16 = 288 items.
REQD_SG __kernel void p_i2_moe_scatter(__global const half *x16, __global const uint *inv, __global const PromptPlan *pp,
                                       uint sigma, __global half *xg) {
  const uint t = get_group_id(1), lane = sg_lane(), at = 128u * sg_id();
  ROW_OR_RETURN(t)
  ushort8 v = sg_read_us8((__global const ushort *)(x16 + (ulong)t * I2_EMBD) + at);
  if (sigma) v = sg_shuffle_us8(v, (lane & ~3u) | ((lane & 1u) << 1) | ((lane >> 1) & 1u));
  uint rows[I2_USED];
  for (uint k = 0; k < I2_USED; k++) rows[k] = inv[t * I2_USED + k];
  for (uint k = 0; k < I2_USED; k++) sg_write_us8((__global ushort *)(xg + (ulong)rows[k] * I2_EMBD) + at, v);
}

__kernel void p_i2_moe_combine(__global const half *y, __global const uint *inv, __global const float *w,
                               __global const PromptPlan *pp, __global half *out) {
  const uint i = 2u * get_global_id(0), t = get_group_id(1);
  ROW_OR_RETURN(t)
  if (i >= I2_EMBD) return;
  float a0 = 0.0f, a1 = 0.0f;
  for (uint k = 0; k < I2_USED; k++) {
    const float wk = w[t * I2_USED + k];
    const ulong at = (ulong)inv[t * I2_USED + k] * I2_EMBD + i;
    a0 += wk * vload_half(at, y);
    a1 += wk * vload_half(at + 1u, y);
  }
  out[(ulong)t * I2_EMBD + i] = convert_half(a0);
  out[(ulong)t * I2_EMBD + i + 1u] = convert_half(a1);
}

__kernel void p_i2_allreduce_f16(__global const half *local_p, ulong local_stride, __global const half *recv,
                                 ulong recv_stride, uint ncards, uint max_rows, __global const PromptPlan *pp,
                                 __global float *out, ulong out_stride) {
  const uint i = get_global_id(0), t = get_group_id(1);
  ROW_OR_RETURN(t)
  if (i >= I2_EMBD) return;
  float sum = (float)local_p[(ulong)t * local_stride + i];
  for (uint p = 0; p + 1 < ncards; p++) sum += (float)recv[((ulong)p * max_rows + t) * recv_stride + i];
  out[(ulong)t * out_stride + i] = sum;
}

__kernel void p_i2_residual_add(__global const float *a, ulong a_stride, __global const float *b, ulong b_stride,
                                __global const PromptPlan *pp, __global float *dst, ulong dst_stride) {
  const uint i = get_global_id(0), t = get_group_id(1);
  ROW_OR_RETURN(t)
  if (i >= I2_EMBD) return;
  dst[(ulong)t * dst_stride + i] = a[(ulong)t * a_stride + i] + b[(ulong)t * b_stride + i];
}

__kernel void p_i2_take_last(__global const float *p_cur, ulong p_cur_stride, __global const float *p_s_norm,
                             uint nb, uint p_max_rows, __global const PromptPlan *runs, __global float *d_cur,
                             ulong d_cur_stride, __global float *d_s_norm, uint d_max_rows) {
  const uint i = get_global_id(0), r = get_group_id(2);
  if (i >= I2_EMBD) return;
  __global const PromptPlan *rp = runs + r;
  if (rp->n == 0) return;
  const uint last = rp->row0 + rp->n - 1u;
  d_cur[(ulong)r * d_cur_stride + i] = p_cur[(ulong)last * p_cur_stride + i];
  for (uint b = 0; b < nb; b++) {
    d_s_norm[((ulong)b * d_max_rows + r) * I2_EMBD + i] =
        p_s_norm[((ulong)b * p_max_rows + last) * I2_EMBD + i];
  }
}
