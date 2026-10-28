// The DFlash 2 drafter's kernels (core/models/qwen4exp/dflash.cpp): the main model's features, the norms, the
// block's convolutions, attention over a slot's ring and the MLP's gate. Rows follow a DraftPlan; grids are recorded
// for a width and return past plan->nrows. Each card holds nq query heads and nkv KV heads (docs/DESIGN.md, section 5).
#include "builtins.h"
#include "layout.h"
#include "step.h"

static inline float silu(float x) { return x / (1.0f + exp(-x)); }

// Sum over the work-group (at most 1024 work-items, sub-groups of 16). red has 64 floats.
static inline float wg_sum(float v, __local float *red) {
  v = sg_sum(v);
  uint nsg = get_num_sub_groups();
  if (sg_lane() == 0) red[sg_id()] = v;
  barrier(CLK_LOCAL_MEM_FENCE);
  float t = 0.0f;
  for (uint i = 0; i < nsg; i++) t += red[i];
  barrier(CLK_LOCAL_MEM_FENCE);
  return t;
}

// Row r's features from the residual's four streams after one of the five layers (k): their mean, as bf16, into
// out[r][k * Q_EMBD ..]. The decode step's rows. Groups (Q_EMBD / 64, width), 64.
__kernel void df_feat(__global const float *R, uint k, __global const StepPlan *plan, __global ushort *out) {
  uint i = get_global_id(0), r = get_group_id(1);
  if (r >= plan->nrows) return;
  __global const float *x = R + (ulong)r * Q_HC_DIM + i;
  float m = (x[0] + x[Q_EMBD] + x[2 * Q_EMBD] + x[3 * Q_EMBD]) * 0.25f;
  out[(ulong)r * DF_FEAT + k * Q_EMBD + i] = f32_to_bf16(m);
}

// The same for a prompt run's last rows (at most STEP_MAX_ROWS), row j of them into out[j]: only the rows r0 .. r1 - 1
// of the chunk, which this card holds. Groups (Q_EMBD / 64, STEP_MAX_ROWS), 64.
__kernel void p_df_feat(__global const float *R, __global const PromptPlan *runs, uint k, uint r0, uint r1,
                        __global ushort *out_all) {
  // One launch for all the chunk's runs: run z's rows into out_all + z STEP_MAX_ROWS DF_FEAT.
  __global const PromptPlan *pp = runs + get_group_id(2);
  __global ushort *out = out_all + (ulong)get_group_id(2) * STEP_MAX_ROWS * DF_FEAT;
  const uint n = pp->n, m = min(n, (uint)STEP_MAX_ROWS), j = get_group_id(1), i = get_global_id(0);
  if (j >= m) return;
  const uint t = pp->row0 + n - m + j;
  if (t < r0 || t >= r1) return;
  __global const float *x = R + (ulong)t * Q_HC_DIM + i;
  float v = (x[0] + x[Q_EMBD] + x[2 * Q_EMBD] + x[3 * Q_EMBD]) * 0.25f;
  out[(ulong)j * DF_FEAT + k * Q_EMBD + i] = f32_to_bf16(v);
}

// RMS norm of row r with weight w (no +1): into out and as bf16 into outb, each when not null. Groups (width), 256.
__kernel void df_norm(__global const float *x, __global const float *w, __global const DraftPlan *plan,
                      __global float *out, __global ushort *outb) {
  __local float red[64];
  uint r = get_group_id(0);
  if (r >= plan->nrows) return;
  __global const float *xr = x + (ulong)r * Q_EMBD;
  float ss = 0.0f;
  for (uint i = get_local_id(0); i < Q_EMBD; i += get_local_size(0)) ss += xr[i] * xr[i];
  ss = wg_sum(ss, red);
  float rs = rsqrt(ss / (float)Q_EMBD + DF_EPS);
  for (uint i = get_local_id(0); i < Q_EMBD; i += get_local_size(0)) {
    float v = xr[i] * rs * w[i];
    if (out) out[(ulong)r * Q_EMBD + i] = v;
    if (outb) outb[(ulong)r * Q_EMBD + i] = f32_to_bf16(v);
  }
}

// One set of a convolution's taps over the block's rows: y_t = sum_o (base[set][o] + dyn_t[set][o][group]) x_{t-o},
// where x_{t-1} is zero at a block's first row. dyn rows hold DF_KP weights, set-major, then tap, then group; base is
// [set][tap][Q_EMBD]. A tap's rows of x are read whole, so out may not be x. Groups (Q_EMBD / 64, width), 64.
static inline float df_tap(__global const float *x, __global const float *dyn, uint set, __global const float *base,
                           __global const DraftPlan *plan, uint t, uint ch) {
  __global const float *d = dyn + (ulong)t * DF_KP + set * (DF_KP / 2);
  const uint g = ch / DF_GROUP;
  float y = (base[(2 * set) * Q_EMBD + ch] + d[g]) * x[(ulong)t * Q_EMBD + ch];
  if (t > plan->rows[t].row0)
    y += (base[(2 * set + 1) * Q_EMBD + ch] + d[DF_KP / 4 + g]) * x[(ulong)(t - 1) * Q_EMBD + ch];
  return y;
}

// The convolution before a sublayer (set 0): into out and as bf16 into outb. Groups (Q_EMBD / 64, width), 64.
__kernel void df_conv(__global const float *x, __global const float *dyn, __global const float *base,
                      __global const DraftPlan *plan, __global float *out, __global ushort *outb) {
  uint ch = get_global_id(0), t = get_group_id(1);
  if (t >= plan->nrows) return;
  float y = df_tap(x, dyn, 0, base, plan, t, ch);
  out[(ulong)t * Q_EMBD + ch] = y;
  outb[(ulong)t * Q_EMBD + ch] = f32_to_bf16(y);
}

// After an exchange: the sublayer's output is the cards' partial sums added in card order (P: ncards blocks of
// STEP_MAX_ROWS rows); the convolution after it (set 1), then the residual h += it. Groups (Q_EMBD / 64, width), 64.
__kernel void df_finish(__global const float *P, uint ncards, __global const float *dyn, __global const float *base,
                        __global const DraftPlan *plan, __global float *h) {
  uint ch = get_global_id(0), t = get_group_id(1);
  if (t >= plan->nrows) return;
  __global const float *d = dyn + (ulong)t * DF_KP + DF_KP / 2;
  const uint g = ch / DF_GROUP;
  float a = 0.0f;
  for (uint c = 0; c < ncards; c++) a += P[((ulong)c * STEP_MAX_ROWS + t) * Q_EMBD + ch];
  float y = (base[2 * Q_EMBD + ch] + d[g]) * a;
  if (t > plan->rows[t].row0) {
    float b = 0.0f;
    for (uint c = 0; c < ncards; c++) b += P[((ulong)c * STEP_MAX_ROWS + t - 1) * Q_EMBD + ch];
    y += (base[3 * Q_EMBD + ch] + d[DF_KP / 4 + g]) * b;
  }
  h[(ulong)t * Q_EMBD + ch] += y;
}

// Rotary embedding over a whole head, pairs (i, i + 64), angle pos * base^(-2i / 128), as Qwen3's rotate_half.
static inline float df_rope(__local const float *x, uint d, uint pos) {
  uint i = d % (DF_D / 2);
  float theta = (float)pos * pow(DF_ROPE_BASE, -2.0f * (float)i / (float)DF_D);
  float c = cos(theta), s = sin(theta);
  return d < DF_D / 2 ? x[i] * c - x[i + DF_D / 2] * s : x[i] * s + x[i + DF_D / 2] * c;
}

// Q rows hold nq heads, K and V rows nkv heads, of DF_D. Work-groups [0, nq) norm and rotate the queries in place;
// [nq, nq + nkv) norm and rotate the keys and store keys and values in the slot's ring at the row's position.
// Groups (nq + nkv, width), DF_D.
__kernel void df_prep(__global float *Q, __global const float *K, __global const float *V, __global const float *qn,
                      __global const float *kn, uint nq, uint nkv, __global float *rk, __global float *rv,
                      __global const DraftPlan *plan) {
  __local float x[DF_D], red[64];
  uint g = get_group_id(0), r = get_group_id(1), d = get_local_id(0);
  if (r >= plan->nrows) return;
  const uint pos = plan->rows[r].pos;
  const bool is_q = g < nq;
  float v = is_q ? Q[((ulong)r * nq + g) * DF_D + d] : K[((ulong)r * nkv + g - nq) * DF_D + d];
  float ss = wg_sum(v * v, red);
  x[d] = v * rsqrt(ss / (float)DF_D + DF_EPS) * (is_q ? qn[d] : kn[d]);
  barrier(CLK_LOCAL_MEM_FENCE);
  float y = df_rope(x, d, pos);
  if (is_q) {
    Q[((ulong)r * nq + g) * DF_D + d] = y;
    return;
  }
  const uint c = g - nq;
  const ulong cell = ((ulong)plan->rows[r].slot * DF_RING + pos % DF_RING) * nkv + c;
  rk[cell * DF_D + d] = y;
  rv[cell * DF_D + d] = V[((ulong)r * nkv + c) * DF_D + d];
}

// Attention, non-causal in the block: row r attends to its slot's positions lo_r .. its block's last. A block's rows
// share their keys and values, so the work-group of its first row reads each cell once for all of them: it covers
// the cells from the first row's lo (the block's smallest) in splits of DF_SPLIT, for one KV head's DF_GQA query
// heads, and masks the cells before each row's own lo. Per row, query head and split it writes the weighted sum of V,
// then the scores' maximum and sum (DF_PART floats). Groups (nkv, width, DF_SPLITS), DF_D.
#define DF_SPLIT 64
#define DF_SPLITS ((DF_WINDOW + STEP_MAX_ROWS + DF_SPLIT - 1) / DF_SPLIT)
#define DF_PART (DF_D + 2)
#define DF_GQA (DF_H / DF_KV)
#define DF_PAIRS (STEP_MAX_ROWS * DF_GQA)
#if DF_D != 2 * DF_SPLIT
#error "df_attn's work-items score a split's cells in two halves of the pairs"
#endif
REQD_SG __kernel void df_attn(__global const float *Q, __global const float *rk, __global const float *rv, uint nq,
                              uint nkv, __global const DraftPlan *plan, __global float *part) {
  __local float q[DF_PAIRS][DF_D], p[DF_PAIRS][DF_SPLIT];
  const uint c = get_group_id(0), b0 = get_group_id(1), s = get_group_id(2), d = get_local_id(0);
  if (b0 >= plan->nrows || plan->rows[b0].row0 != b0) return;
  const uint nb = plan->rows[b0].n, lo = plan->rows[b0].lo, np = nb * DF_GQA;
  const uint n = plan->rows[b0].pos + nb - lo, i0 = s * DF_SPLIT;
  if (i0 >= n) return;
  const uint m = min((uint)DF_SPLIT, n - i0);
  // Pairs past the block's are zero, so every loop below runs a fixed count and its arrays stay in registers.
  for (uint j = 0; j < DF_PAIRS; j++)
    q[j][d] = j < np ? Q[((ulong)(b0 + j / DF_GQA) * nq + c * DF_GQA + j % DF_GQA) * DF_D + d] : 0.0f;
  barrier(CLK_LOCAL_MEM_FENCE);
  const ulong base = (ulong)plan->rows[b0].slot * DF_RING;
  const uint lane = sg_lane(), sg = sg_id(), nsg = get_num_sub_groups();
  const float scale = 0.08838834764831845f;  // 1 / sqrt(DF_D)
  {
    // Work-item d scores cell d % DF_SPLIT against half the pairs on its own: no reduction across work-items.
    const uint i = d % DF_SPLIT, j0 = d / DF_SPLIT * (DF_PAIRS / 2), at = lo + i0 + i;
    float sc[DF_PAIRS / 2];
    __attribute__((opencl_unroll_hint)) for (uint k = 0; k < DF_PAIRS / 2; k++) sc[k] = 0.0f;
    if (i < m) {
      __global const float *kr = rk + ((base + at % DF_RING) * nkv + c) * DF_D;
      for (uint e = 0; e < DF_D; e += 16) {
        const float16 kv = vload16(0, kr + e);
        __attribute__((opencl_unroll_hint)) for (uint k = 0; k < DF_PAIRS / 2; k++) {
          const float16 qv = vload16(0, &q[j0 + k][e]);
          sc[k] += dot(kv.s0123, qv.s0123) + dot(kv.s4567, qv.s4567) + dot(kv.s89ab, qv.s89ab) +
                   dot(kv.scdef, qv.scdef);
        }
      }
      __attribute__((opencl_unroll_hint)) for (uint k = 0; k < DF_PAIRS / 2; k++)
        if (j0 + k < np) p[j0 + k][i] = at >= plan->rows[b0 + (j0 + k) / DF_GQA].lo ? sc[k] * scale : -INFINITY;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint j = sg; j < np; j += nsg) {
    float mx = -INFINITY;
    for (uint i = lane; i < m; i += SG_SIZE) mx = fmax(mx, p[j][i]);
    mx = sub_group_reduce_max(mx);
    float sum = 0.0f;
    for (uint i = lane; i < m; i += SG_SIZE) {
      float e = mx == -INFINITY ? 0.0f : exp(p[j][i] - mx);  // a split wholly before the row's window weighs 0
      p[j][i] = e;
      sum += e;
    }
    sum = sub_group_reduce_add(sum);
    if (lane == 0) {
      __global float *po = part + (((ulong)(b0 + j / DF_GQA) * nq + c * DF_GQA + j % DF_GQA) * DF_SPLITS + s) * DF_PART;
      po[DF_D] = mx;
      po[DF_D + 1] = sum;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  // p of pairs past the block's is never written: zero it here, as their sums are never read.
  for (uint j = np + sg; j < DF_PAIRS; j += nsg)
    for (uint i = lane; i < DF_SPLIT; i += SG_SIZE) p[j][i] = 0.0f;
  barrier(CLK_LOCAL_MEM_FENCE);
  float acc[DF_PAIRS];
  __attribute__((opencl_unroll_hint)) for (uint j = 0; j < DF_PAIRS; j++) acc[j] = 0.0f;
  // Four cells at a time: one local read of each pair's weights for four values of V.
  uint i = 0;
  for (; i + 4 <= m; i += 4) {
    float4 v;
    v.x = rv[((base + (lo + i0 + i) % DF_RING) * nkv + c) * DF_D + d];
    v.y = rv[((base + (lo + i0 + i + 1) % DF_RING) * nkv + c) * DF_D + d];
    v.z = rv[((base + (lo + i0 + i + 2) % DF_RING) * nkv + c) * DF_D + d];
    v.w = rv[((base + (lo + i0 + i + 3) % DF_RING) * nkv + c) * DF_D + d];
    __attribute__((opencl_unroll_hint)) for (uint j = 0; j < DF_PAIRS; j++) acc[j] += dot(vload4(0, &p[j][i]), v);
  }
  for (; i < m; i++) {
    const float v = rv[((base + (lo + i0 + i) % DF_RING) * nkv + c) * DF_D + d];
    __attribute__((opencl_unroll_hint)) for (uint j = 0; j < DF_PAIRS; j++) acc[j] += p[j][i] * v;
  }
  __attribute__((opencl_unroll_hint)) for (uint j = 0; j < DF_PAIRS; j++)
    if (j < np) part[(((ulong)(b0 + j / DF_GQA) * nq + c * DF_GQA + j % DF_GQA) * DF_SPLITS + s) * DF_PART + d] = acc[j];
}

// df_attn's splits combined in order: into out and as bf16 into outb (rows nq DF_D apart). The splits are the
// block's, from its first row's lo. Groups (nq, width), DF_D.
__kernel void df_merge(uint nq, __global const DraftPlan *plan, __global const float *part, __global float *out,
                       __global ushort *outb) {
  const uint lh = get_group_id(0), r = get_group_id(1), d = get_local_id(0);
  if (r >= plan->nrows) return;
  const uint b0 = plan->rows[r].row0;
  const uint n = plan->rows[b0].pos + plan->rows[b0].n - plan->rows[b0].lo, ns = (n + DF_SPLIT - 1) / DF_SPLIT;
  __global const float *pt = part + ((ulong)r * nq + lh) * DF_SPLITS * DF_PART;
  float mx = -INFINITY;
  for (uint s = 0; s < ns; s++) mx = fmax(mx, pt[s * DF_PART + DF_D]);
  float acc = 0.0f, sum = 0.0f;
  for (uint s = 0; s < ns; s++) {
    const float ms = pt[s * DF_PART + DF_D];
    const float f = ms == -INFINITY ? 0.0f : exp(ms - mx);
    acc += f * pt[s * DF_PART + d];
    sum += f * pt[s * DF_PART + DF_D + 1];
  }
  const float y = acc / sum;
  out[((ulong)r * nq + lh) * DF_D + d] = y;
  outb[((ulong)r * nq + lh) * DF_D + d] = f32_to_bf16(y);
}

// silu(g) u as bf16, rows of n. Groups (ceil(n / 64), width), 64.
__kernel void df_swiglu(__global const float *g, __global const float *u, uint n, __global const DraftPlan *plan,
                        __global ushort *outb) {
  uint i = get_global_id(0), t = get_group_id(1);
  if (t >= plan->nrows || i >= n) return;
  outb[(ulong)t * n + i] = f32_to_bf16(silu(g[(ulong)t * n + i]) * u[(ulong)t * n + i]);
}
