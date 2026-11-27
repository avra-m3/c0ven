// Shared prompt attention and indexer kernels for both Xe2 and Xe-HPG: p_attn_rows, p_attn_rows_merge, and
// p_idx_score_xmx (using xmx.h). Included by prompt_attn.cl and prompt_attn_hpg.cl.
#ifndef KERNELS_QWEN4EXP_PROMPT_ATTN_COMMON_H
#define KERNELS_QWEN4EXP_PROMPT_ATTN_COMMON_H

#include "layout.h"
#include "step.h"
#include "xmx.h"

static inline float sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }

// A short run's rows (at most PROMPT_SHORT_RUN: a decode row and its drafts) attend as the decode step's attn_split
// and attn_merge (layer.cl) do: over the cells p_idx_merge chose, read from the 8-bit cache through the page table,
// with no fp16 copy of the sequence. p_attn_rows: work-group (KV head c, row y, split s) takes cells
// [s ATT_SPLIT, +ATT_SPLIT) of row y % PROMPT_SHORT_RUN of run y / PROMPT_SHORT_RUN, for the ATT_GQA query heads of
// KV head c, and writes each head's unnormalised output, max and sum into part (ATT_PART floats a head and split, rows
// PROMPT_MAX_RUNS PROMPT_SHORT_RUN of them). Groups (nkv, PROMPT_MAX_RUNS PROMPT_SHORT_RUN, ATT_SPLITS), 256.
#define ATT_SPLIT 64
#define ATT_SPLITS ((Q_IDX_CELLS + ATT_SPLIT - 1) / ATT_SPLIT)
#define ATT_PART (Q_ATT_D + 2)
#define ATT_GQA (Q_ATT_H / Q_ATT_KV)

// Row y of the short runs' grid: its run's plan, or null when the run is long or holds no row y.
static inline __global const PromptPlan *short_row(__global const PromptPlan *runs, uint y) {
  __global const PromptPlan *pp = runs + y / PROMPT_SHORT_RUN;
  return pp->n > PROMPT_SHORT_RUN || y % PROMPT_SHORT_RUN >= pp->n ? 0 : pp;
}

REQD_SG __kernel void p_attn_rows(__global const float *Q, __global const uchar *kc, __global const uchar *vc,
                                  __global const uint *pt, uint npp, uint nq, uint nkv, __global const uint *cells_all,
                                  __global const uint *ncells, __global const PromptPlan *runs, __global float *part,
                                  __global const uchar *hk, __global const uchar *hv) {
  // Streamed (hk, hv given): the cells are read in place from the host copy, which holds every page; the 8-bit
  // cache then holds only the resident pages, and a page index into it would read past its end.
  __local float q[ATT_GQA][Q_ATT_D], p[ATT_GQA][ATT_SPLIT];
  __local uint cl[ATT_SPLIT];
  uint c = get_group_id(0), y = get_group_id(1), s = get_group_id(2), d = get_local_id(0);
  __global const PromptPlan *pp = short_row(runs, y);
  if (!pp) return;
  const uint r = pp->row0 + y % PROMPT_SHORT_RUN;  // the chunk's row
  uint n = ncells[r], i0 = s * ATT_SPLIT;
  if (i0 >= n) return;
  uint m = min((uint)ATT_SPLIT, n - i0);
  for (uint h = 0; h < ATT_GQA; h++) q[h][d] = Q[((ulong)r * nq + c * ATT_GQA + h) * 2 * Q_ATT_D + d];
  __global const uint *cells = cells_all + (ulong)r * Q_IDX_CELLS + i0;
  if (d < m) {
    const uint pos = cells[d];
    cl[d] = (uint)kv_cell(kv_page(pt, npp, pp->slot, pos), pos);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  __global const uchar *K = (hk ? hk : kc) + c * KV_HEAD_BYTES;
  __global const uchar *V = (hv ? hv : vc) + c * KV_HEAD_BYTES;
  ulong cell_bytes = (ulong)nkv * KV_HEAD_BYTES;
  uint lane = get_sub_group_local_id(), sg = get_sub_group_id(), nsg = get_num_sub_groups();
  for (uint i = sg; i < m; i += nsg) {
    // Lane l holds values 16l .. 16l+15: half of block l/2.
    __global const uchar *blk = K + cl[i] * cell_bytes + (lane / 2) * Q8_BLOCK;
    float db = vload_half(0, (__global const half *)blk);
    char16 v = vload16(0, (__global const char *)(blk + 2 + 16 * (lane % 2)));
    float kv[16] = {v.s0, v.s1, v.s2, v.s3, v.s4, v.s5, v.s6, v.s7, v.s8, v.s9, v.sa, v.sb, v.sc, v.sd, v.se, v.sf};
    for (uint h = 0; h < ATT_GQA; h++) {
      __local const float *qh = &q[h][16 * lane];
      float part_ = 0.0f;
      for (uint t = 0; t < 16; t++) part_ += qh[t] * kv[t];
      float dot = sub_group_reduce_add(db * part_);
      if (lane == 0) p[h][i] = dot * Q_ATT_SCALE;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  __global float *po = part + (((ulong)y * nq + c * ATT_GQA) * ATT_SPLITS + s) * ATT_PART;
  if (sg < ATT_GQA) {
    float mx = -INFINITY;
    for (uint i = lane; i < m; i += 16) mx = fmax(mx, p[sg][i]);
    mx = sub_group_reduce_max(mx);
    float sum = 0.0f;
    for (uint i = lane; i < m; i += 16) {
      float e = exp(p[sg][i] - mx);
      p[sg][i] = e;
      sum += e;
    }
    sum = sub_group_reduce_add(sum);
    if (lane == 0) {
      po[(ulong)sg * ATT_SPLITS * ATT_PART + Q_ATT_D] = mx;
      po[(ulong)sg * ATT_SPLITS * ATT_PART + Q_ATT_D + 1] = sum;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  __global const uchar *vb = V + (d / 32) * Q8_BLOCK;
  float acc[ATT_GQA];
  for (uint h = 0; h < ATT_GQA; h++) acc[h] = 0.0f;
  // Eight cells' loads issued together, then summed in order.
  uint i = 0;
  for (; i + 8 <= m; i += 8) {
    float v[8];
    for (uint j = 0; j < 8; j++) {
      __global const uchar *b = vb + cl[i + j] * cell_bytes;
      v[j] = vload_half(0, (__global const half *)b) * (float)((__global const char *)b)[2 + d % 32];
    }
    for (uint j = 0; j < 8; j++)
      for (uint h = 0; h < ATT_GQA; h++) acc[h] += p[h][i + j] * v[j];
  }
  for (; i < m; i++) {
    __global const uchar *b = vb + cl[i] * cell_bytes;
    float v = vload_half(0, (__global const half *)b) * (float)((__global const char *)b)[2 + d % 32];
    for (uint h = 0; h < ATT_GQA; h++) acc[h] += p[h][i] * v;
  }
  for (uint h = 0; h < ATT_GQA; h++) po[(ulong)h * ATT_SPLITS * ATT_PART + d] = acc[h];
}

// p_attn_rows' splits combined in order, then the output gate, into o16 as the long runs' kernels write it. Groups
// (nq, PROMPT_MAX_RUNS PROMPT_SHORT_RUN), 256.
REQD_SG __kernel void p_attn_rows_merge(__global const float *Q, uint nq, __global const uint *ncells,
                                        __global const PromptPlan *runs, __global const float *part, __global half *o16) {
  uint lh = get_group_id(0), y = get_group_id(1), d = get_local_id(0);
  __global const PromptPlan *pp = short_row(runs, y);
  if (!pp) return;
  const uint r = pp->row0 + y % PROMPT_SHORT_RUN;
  uint ns = (ncells[r] + ATT_SPLIT - 1) / ATT_SPLIT;
  __global const float *pt = part + ((ulong)y * nq + lh) * ATT_SPLITS * ATT_PART;
  float mx = -INFINITY;
  for (uint s = 0; s < ns; s++) mx = fmax(mx, pt[s * ATT_PART + Q_ATT_D]);
  float acc = 0.0f, sum = 0.0f;
  for (uint s = 0; s < ns; s++) {
    float f = exp(pt[s * ATT_PART + Q_ATT_D] - mx);
    acc += f * pt[s * ATT_PART + d];
    sum += f * pt[s * ATT_PART + Q_ATT_D + 1];
  }
  float gate = Q[((ulong)r * nq + lh) * 2 * Q_ATT_D + Q_ATT_D + d];
  o16[((ulong)r * nq + lh) * Q_ATT_D + d] = convert_half(acc / sum * sigmoid(gate));
}

// The prompt indexer's scores on the XMX units: scores[t][lb] = sum_h relu(iq_h . K_b) for this card's blocks complete
// at row t's position, by local index lb (block lb C + c), rows nb_max apart.
#define IDX_XMX_ROWS 16
#define IDX_XMX_TB 2
#if Q_IDX_H != 4 || Q_IDX_D % 16
#error "p_idx_score_xmx tiles 2 rows by the indexer's 4 heads, and 16 dimensions a step"
#endif
static inline uint idx_own_x(uint nb, uint c, uint C) { return nb > c ? (nb - c + C - 1) / C : 0; }
REQD_XMX_SG __kernel void p_idx_score_xmx(__global const ushort *qh, __global const ushort *ql, __global const half *kh,
                                          __global const half *kl, __global const uint *pt, uint npp, uint nb_max,
                                          __global const PromptPlan *runs, __global float *scores, uint c, uint C) {
  __global const PromptPlan *pp = runs + get_group_id(2);
  const uint lane = sg_lane(), n = pp->n, t0 = IDX_XMX_ROWS * get_group_id(1);
  if (t0 >= n) return;
  scores += (ulong)pp->row0 * nb_max;
  const uint rows = min((uint)IDX_XMX_ROWS, n - t0);
  const uint nkv_last = pp->pos0 + t0 + rows;
  if (nkv_last <= Q_IDX_CELLS) return;
  const uint nown = idx_own_x(nkv_last / Q_IDX_BLOCK, c, C);
  const uint b0 = (get_group_id(0) * get_num_sub_groups() + get_sub_group_id()) * XMX_LANES * IDX_XMX_TB;
  if (b0 >= nown) return;
  ulong K[IDX_XMX_TB];
  for (uint j = 0; j < IDX_XMX_TB; j++) {
    const uint lb = min(b0 + XMX_LANES * j + lane, nown - 1);
    K[j] = (ulong)(kv_page(pt, npp, pp->slot, (lb * C + c) * Q_IDX_BLOCK) / C) * Q_IDX_D;
  }
  float8 acc[IDX_XMX_ROWS / 2][IDX_XMX_TB];
  for (uint i = 0; i < IDX_XMX_ROWS / 2; i++)
    for (uint j = 0; j < IDX_XMX_TB; j++) acc[i][j] = (float8)0.0f;
  for (uint s = 0; s < Q_IDX_D / 16; s++) {
    int8 bh[IDX_XMX_TB], bl[IDX_XMX_TB];
    for (uint j = 0; j < IDX_XMX_TB; j++) {
      bh[j] = as_int8(vload8(0, (__global const uint *)(kh + K[j]) + 8 * s));
      bl[j] = as_int8(vload8(0, (__global const uint *)(kl + K[j]) + 8 * s));
    }
    for (uint i = 0; i < IDX_XMX_ROWS / 2; i++) {
      const uint row_r0 = pp->row0 + t0 + min(2 * i, rows - 1);
      const uint row_r1 = pp->row0 + t0 + min(2 * i + 1, rows - 1);
      const xmx_a_t a = xmx_idx_qa(qh, row_r0, row_r1, s, Q_IDX_D, Q_IDX_H);
      const xmx_a_t b = xmx_idx_qa(ql, row_r0, row_r1, s, Q_IDX_D, Q_IDX_H);
      for (uint j = 0; j < IDX_XMX_TB; j++) {
        acc[i][j] = xmx_dpas_f16(a, bh[j], acc[i][j]);
        acc[i][j] = xmx_dpas_f16(a, bl[j], acc[i][j]);
        acc[i][j] = xmx_dpas_f16(b, bh[j], acc[i][j]);
      }
    }
  }
  uint lim[IDX_XMX_ROWS];  // row r's blocks on this card, those complete at its position (none below the width)
  for (uint r = 0; r < IDX_XMX_ROWS; r++) {
    const uint nkv = pp->pos0 + t0 + r + 1;
    lim[r] = r < rows && nkv > Q_IDX_CELLS ? idx_own_x(nkv / Q_IDX_BLOCK, c, C) : 0;
  }
  for (uint i = 0; i < IDX_XMX_ROWS / 2; i++)
    for (uint j = 0; j < IDX_XMX_TB; j++) {
      const uint lb = b0 + XMX_LANES * j + lane;
      const float8 a = acc[i][j];
      for (uint u = 0; u < 2; u++) {
        const uint r = 2 * i + u, t = t0 + r;
        if (lb >= lim[r]) continue;
        float score = 0.0f;
        score += fmax(u ? a.s4 : a.s0, 0.0f);
        score += fmax(u ? a.s5 : a.s1, 0.0f);
        score += fmax(u ? a.s6 : a.s2, 0.0f);
        score += fmax(u ? a.s7 : a.s3, 0.0f);
        scores[(ulong)t * nb_max + lb] = score;
      }
    }
}

#endif
