// qwen4exp's decode-step kernels: hyper-connections, Gated DeltaNet, sparse attention and its key indexer, the router,
// the experts' combine, layer 1's PLE module. Maths: docs/spec/MODEL_LAYER.md (sections per kernel) and MODEL_IO.md;
// card split: docs/DESIGN.md, section 5 (each card passes its own head counts). A step's plan->nrows rows (StepPlan)
// are tokens of several sequences or a draft window's positions; row r is at r * width, grids are recorded for the
// width and return past nrows. Recurrent state updates row by row; all rows' K and V are stored before any attends.
#include "a8.h"
#include "builtins.h"
#include "layout.h"
#include "moe_tile.h"
#include "step.h"
#include "topk.h"

#define RT_SG SG_SIZE
#include "route.h"

static inline float silu(float x) { return x / (1.0f + exp(-x)); }
static inline float sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }
static inline float softplus(float x) { return x > 20.0f ? x : log(1.0f + exp(x)); }

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

static inline float wg_max(float v, __local float *red) {
  v = sg_max(v);
  uint nsg = get_num_sub_groups();
  if (sg_lane() == 0) red[sg_id()] = v;
  barrier(CLK_LOCAL_MEM_FENCE);
  float t = -INFINITY;
  for (uint i = 0; i < nsg; i++) t = fmax(t, red[i]);
  barrier(CLK_LOCAL_MEM_FENCE);
  return t;
}

// NeoX RoPE on the first Q_ROT dimensions of a head held in local memory: pairs (i, i + 32), angle
// pos * base^(-2i / 64) (section 3.3). Every work-item of the group calls it; d is its dimension.
static inline float rope(__local const float *x, uint d, uint pos) {
  if (d >= Q_ROT) return x[d];
  uint i = d % (Q_ROT / 2);
  float theta = (float)pos * pow(Q_ROPE_BASE, -2.0f * (float)i / (float)Q_ROT);
  float c = cos(theta), s = sin(theta);
  return d < Q_ROT / 2 ? x[i] * c - x[i + Q_ROT / 2] * s : x[i] * s + x[i + Q_ROT / 2] * c;
}

// Interleaved M-RoPE (docs/spec/MODEL_VISION.md, section 4.2): pair i turns by the height position when i % 3 is 1,
// by the width position when i % 3 is 2 and i < 30, by the time position otherwise. With t = h = w it is rope().
static inline float rope3(__local const float *x, uint d, uint t, uint h, uint w) {
  uint i = d % (Q_ROT / 2), m = i % 3;
  return rope(x, d, m == 1 ? h : (m == 2 && i < 30) ? w : t);
}

// One Q8_0 block of 32 values from local memory into dst (d = max|x| / 127, q = round(x / d)).
static inline void q8_0_store(__local const float *x, __global uchar *dst) {
  float amax = 0.0f;
  for (uint i = 0; i < 32; i++) amax = fmax(amax, fabs(x[i]));
  float d = amax / 127.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
  vstore_half(d, 0, (__global half *)dst);
  for (uint i = 0; i < 32; i++) ((__global char *)dst)[2 + i] = (char)round(x[i] * id);
}

static inline float q8_0_load(__global const uchar *row, uint d) {
  __global const uchar *b = row + (d / 32) * Q8_BLOCK;
  return vload_half(0, (__global const half *)b) * (float)((__global const char *)b)[2 + d % 32];
}

// ---------------------------------------------------------------------------------------------------------------
// Hyper-connections (section 1)

#define ROW_OR_RETURN(r) \
  if ((r) >= plan->nrows) return;

// The residual starts as four copies of the token's embedding. Groups (Q_HC_DIM / 64, width), 64.
__kernel void hc_init(__global const float *x, __global const StepPlan *plan, __global float *R) {
  uint i = get_global_id(0), r = get_group_id(1);
  ROW_OR_RETURN(r)
  R[(ulong)r * Q_HC_DIM + i] = x[(ulong)r * Q_EMBD + i % Q_EMBD];
}

// The four streams' RMS norms: out[r][s] = in[r][s] / rms * gamma[s], and as bf16 into outb when it is not null
// (for mv_xmx_bf16). Groups (Q_HC, width), 256.
__kernel void rms_norm(__global const float *in, __global const float *gamma, __global const StepPlan *plan,
                       __global float *out, __global ushort *outb) {
  __local float red[64];
  uint g = get_group_id(0), r = get_group_id(1);
  ROW_OR_RETURN(r)
  ulong base = (ulong)r * Q_HC_DIM + (ulong)g * Q_EMBD;
  float ss = 0.0f;
  for (uint i = get_local_id(0); i < Q_EMBD; i += get_local_size(0)) ss += in[base + i] * in[base + i];
  ss = wg_sum(ss, red);
  float rs = rsqrt(ss / (float)Q_EMBD + Q_EPS);
  for (uint i = get_local_id(0); i < Q_EMBD; i += get_local_size(0)) {
    float v = in[base + i] * rs * gamma[g * Q_EMBD + i];
    out[base + i] = v;
    if (outb) outb[base + i] = f32_to_bf16(v);
  }
}

// The mix's output (1.1, step 3): x[i] = 1/4 sum_s n_s[i] * sigma(a_s[i]), with a value-major: a_s[i] = a[4 i + s]
// (the core reorders the up projection's rows so). Groups (Q_EMBD / 64, width), 64.
__kernel void hc_mix(__global const float *n, __global const float *a, __global const StepPlan *plan,
                     __global float *x) {
  uint i = get_global_id(0), r = get_group_id(1);
  ROW_OR_RETURN(r)
  float acc = 0.0f;
  for (uint s = 0; s < Q_HC; s++)
    acc += n[(ulong)r * Q_HC_DIM + s * Q_EMBD + i] * sigmoid(a[(ulong)r * Q_HC_DIM + i * Q_HC + s]);
  x[(ulong)r * Q_EMBD + i] = acc * 0.25f;
}

// The up projection (BF16, Q_HC_DIM rows of K = Q_HC_LR, value-major) and hc_mix in one: a sub-group takes rows
// 4i .. 4i+3, four lanes per row as matvec_bf16 with lpr 4, so it holds a_s[i] for every stream and writes x[i]
// with hc_mix's sum. lr rows lr_stride apart. Groups (Q_EMBD / 4, 1), 64; rows up to plan->nrows.
#define UPM_LPR 4
REQD_SG __kernel void hc_up_mix(__global const ushort *W, __global const float *lr, uint lr_stride,
                                __global const float *n, __global const StepPlan *plan, __global float *x) {
  const uint lane = sg_lane(), sub = lane % UPM_LPR;
  const uint i = get_group_id(0) * 4 + sg_id();
  const uint row = Q_HC * i + lane / UPM_LPR;
  __global const ushort *w = W + (ulong)row * Q_HC_LR;
  const uint ntok = plan->nrows;
  float acc[STEP_MAX_ROWS];
  for (uint t = 0; t < STEP_MAX_ROWS; t++) acc[t] = 0.0f;
  for (uint k = 8 * sub; k < Q_HC_LR; k += 8 * UPM_LPR) {
    ushort8 wv = vload8(0, w + k);
    float8 wf = (float8)(bf16_to_f32(wv.s0), bf16_to_f32(wv.s1), bf16_to_f32(wv.s2), bf16_to_f32(wv.s3),
                         bf16_to_f32(wv.s4), bf16_to_f32(wv.s5), bf16_to_f32(wv.s6), bf16_to_f32(wv.s7));
    for (uint t = 0; t < STEP_MAX_ROWS; t++) {
      if (t >= ntok) break;
      float8 xv = vload8(0, lr + (ulong)t * lr_stride + k);
      acc[t] += dot(wf.lo, xv.lo) + dot(wf.hi, xv.hi);
    }
  }
  for (uint t = 0; t < STEP_MAX_ROWS; t++) {
    if (t >= ntok) break;
    float a = acc[t];
    for (uint m = 1; m < UPM_LPR; m <<= 1) a += sub_group_shuffle_xor(a, m);
    float mix = 0.0f;
    for (uint s = 0; s < Q_HC; s++)
      mix += n[(ulong)t * Q_HC_DIM + s * Q_EMBD + i] * sigmoid(sub_group_shuffle(a, UPM_LPR * s));
    if (lane == 0) x[(ulong)t * Q_EMBD + i] = mix * 0.25f;
  }
}

// hc_up_mix on the XMX units: dpas up projection of lrb (8 bf16 rows of Q_HC_LR, all read; to_bf16_rows), 16 of W's
// value-major rows a sub-group (4 values' a for the 4 streams), then hc_mix's sum in stream order. A work-group's 8
// sub-groups write 32 values of x, also as bf16 into xb (rows Q_EMBD apart) and as a8 into xa (a8.h; rows
// Q_EMBD + Q_EMBD / 8 bytes apart, sub-group m storing row m). Groups (Q_HC_DIM / 128), 128.
#define UPM_SGS 8
REQD_SG __kernel void hc_up_mix_xmx(__global const ushort *W, __global const ushort *lrb, __global const float *n,
                                    __global const StepPlan *plan, __global float *x, __global uchar *xa,
                                    __global ushort *xb) {
  __local float xs[STEP_MAX_ROWS][32];
  const uint lane = sg_lane(), n0 = (get_group_id(0) * UPM_SGS + sg_id()) * 16, ntok = plan->nrows;
  float8 acc = (float8)0.0f;
  for (uint k = 0; k < Q_HC_LR; k += 16) {
    ushort a[8];
    intel_sub_group_2d_block_read_16b_8r16x1c((__global void *)lrb, Q_HC_LR * 2, 8, Q_HC_LR * 2, (int2)(k, 0), a);
    uint b[8];
    sg_load_cols16(W, Q_HC_LR * 2, Q_HC_DIM, Q_HC_LR * 2, k / 2, n0, b);
    acc = dpas_bf16(as_short8(vload8(0, a)), as_int8(vload8(0, b)), acc);
  }
  // Lane l holds a_s[i] for s = l % 4, i = n0 / 4 + l / 4.
  const uint s = lane % Q_HC, i = n0 / Q_HC + lane / Q_HC;
  float r[8];
  vstore8(acc, 0, r);
  for (uint m = 0; m < 8; m++) {
    if (m >= ntok) break;
    float term = n[(ulong)m * Q_HC_DIM + s * Q_EMBD + i] * sigmoid(r[m]);
    float mix = 0.0f;
    for (uint q = 0; q < Q_HC; q++) mix += sub_group_shuffle(term, (lane & ~3u) + q);
    if (s == 0) {
      x[(ulong)m * Q_EMBD + i] = mix * 0.25f;
      xb[(ulong)m * Q_EMBD + i] = f32_to_bf16(mix * 0.25f);
      xs[m][i % 32] = mix * 0.25f;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const uint m = sg_id();
  if (m < ntok) a8_store(vload2(lane, xs[m]), xa + (ulong)m * (Q_EMBD + Q_EMBD / 8), Q_EMBD, get_group_id(0));
}

// After an exchange (1.2): y = the cards' partial sums added in card order, R_s += 2 sigma(inject_s / 4) y. P holds
// ncards blocks of [width][Q_EMBD]; inject rows are inj_stride apart. Groups (Q_EMBD / 64, width), 64.
__kernel void hc_combine(__global const float *P, uint ncards, uint width, __global const float *inject,
                         uint inj_stride, __global const StepPlan *plan, __global float *R) {
  uint i = get_global_id(0), r = get_group_id(1);
  ROW_OR_RETURN(r)
  float y = 0.0f;
  for (uint c = 0; c < ncards; c++) y += P[((ulong)c * width + r) * Q_EMBD + i];
  for (uint s = 0; s < Q_HC; s++)
    R[(ulong)r * Q_HC_DIM + s * Q_EMBD + i] += 2.0f * sigmoid(inject[r * inj_stride + s] * 0.25f) * y;
}

// hc_combine, then rms_norm of the block that follows (gamma) into out and outb, in one pass: the same sums in the
// same order as the two kernels. Up to two cards. Groups (Q_HC, width), 256.
__kernel void hc_combine_norm(__global const float *P, uint ncards, uint width, __global const float *inject,
                              uint inj_stride, __global const float *gamma, __global const StepPlan *plan,
                              __global float *R, __global float *out, __global ushort *outb) {
  __local float red[64];
  uint s = get_group_id(0), r = get_group_id(1);
  ROW_OR_RETURN(r)
  const float g = 2.0f * sigmoid(inject[r * inj_stride + s] * 0.25f);
  ulong base = (ulong)r * Q_HC_DIM + (ulong)s * Q_EMBD;
  // Every load first, so they are in flight together (P may be in host memory), then the updates.
  float v[Q_EMBD / 256], y[Q_EMBD / 256], y1[Q_EMBD / 256];
  for (uint k = 0; k < Q_EMBD / 256; k++) {
    uint i = get_local_id(0) + 256 * k;
    v[k] = R[base + i];
    y[k] = P[(ulong)r * Q_EMBD + i];
    y1[k] = ncards > 1 ? P[((ulong)width + r) * Q_EMBD + i] : 0.0f;
  }
  float ss = 0.0f;
  for (uint k = 0; k < Q_EMBD / 256; k++) {
    uint i = get_local_id(0) + 256 * k;
    float yy = y[k];
    if (ncards > 1) yy += y1[k];
    v[k] = v[k] + g * yy;
    R[base + i] = v[k];
    ss += v[k] * v[k];
  }
  ss = wg_sum(ss, red);
  float rs = rsqrt(ss / (float)Q_EMBD + Q_EPS);
  for (uint k = 0; k < Q_EMBD / 256; k++) {
    uint i = get_local_id(0) + 256 * k;
    float o = v[k] * rs * gamma[s * Q_EMBD + i];
    out[base + i] = o;
    outb[base + i] = f32_to_bf16(o);
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Gated DeltaNet (section 2). This card's share: nkh key heads and nvh = 3 nkh value heads, local value head lv
// reading local key head lv % nkh. Channels of m and u: [q: nkh x 128 | k: nkh x 128 | v: nvh x 128], rows nch apart.

// The causal convolution and SiLU (2.3), row after row, then the L2 norm of the q and k heads (2.4). The ring keeps
// the last three m by position. In a window (plan->window) it is not written: rows take earlier rows' m of their own
// run from m, wm keeps the window's m (rows nch apart), and gdn_conv_commit writes the accepted rows'. Groups
// (nch / 128), 128, one a head of channels; the first nqk (q, k) are normed in sub-groups of 32 as p_gdn_l2norm is,
// for the same sums.
__attribute__((intel_reqd_sub_group_size(32)))
__kernel void gdn_conv(__global const float *m, __global const float *w, __global float *state, ulong slot_stride,
                       uint nch, uint nqk, __global const StepPlan *plan, __global float *u, __global float *wm) {
  __local float red[64];
  const uint ch = get_global_id(0);
  const bool norm = get_group_id(0) < nqk;
  __global const float *k = w + ch * Q_GDN_CONV;
  const bool window = plan->window != 0;
  for (uint r = 0; r < plan->nrows; r++) {
    uint pos = plan->rows[r].pos;
    __global float *st = state + plan->rows[r].slot * slot_stride;
    float mv[3];
    for (uint d = 1; d <= 3; d++)
      mv[3 - d] = window && r >= d && plan->rows[r - d].slot == plan->rows[r].slot ? m[(ulong)(r - d) * nch + ch]
                                                                                 : st[((pos + 3 - d) % 3) * nch + ch];
    float mt = m[(ulong)r * nch + ch];
    float acc = k[0] * mv[0];
    acc += k[1] * mv[1];
    acc += k[2] * mv[2];
    acc += k[3] * mt;
    float v = silu(acc);
    if (norm) v = v * rsqrt(wg_sum(v * v, red) + Q_EPS);
    u[(ulong)r * nch + ch] = v;
    if (window)
      wm[(ulong)r * nch + ch] = mt;
    else
      st[(pos % 3) * nch + ch] = mt;
  }
}

// The commit of a window's convolution rings: each run's accepted rows' m into its slot's ring. Groups
// ceil(nch / 64), 64.
__kernel void gdn_conv_commit(__global const float *wm, __global float *state, ulong slot_stride, uint nch,
                              __global const CommitPlan *cp) {
  uint ch = get_global_id(0);
  if (ch >= nch) return;
  for (uint k = 0; k < cp->nruns; k++) {
    __global const CommitRun *run = cp->runs + k;
    __global float *st = state + run->slot * slot_stride;
    for (uint r = 0; r < run->n; r++) st[((run->pos0 + r) % 3) * nch + ch] = wm[(ulong)(run->row0 + r) * nch + ch];
  }
}

// The recurrence's state by column: value column j of a head is 4 neighbouring lanes of one sub-group, lane part p
// holding the column's state entries S[j][32p .. 32p+31] in registers. Sums over the column's 128 entries are each
// lane's 32 in order, then the 4 parts by xor shuffles, which give all 4 lanes the same bits.
#define GDN_PARTS 4
#define GDN_PER (Q_GDN_D / GDN_PARTS)
static inline float gdn_sum4(float x) {
  x += sub_group_shuffle_xor(x, 1);
  return x + sub_group_shuffle_xor(x, 2);
}

static inline void gdn_load(__global const float *col, uint p, float s[GDN_PER]) {
  for (uint m = 0; m < GDN_PER / 4; m++) {
    float4 v = vload4(m, col + GDN_PER * p);
    s[4 * m] = v.x, s[4 * m + 1] = v.y, s[4 * m + 2] = v.z, s[4 * m + 3] = v.w;
  }
}

static inline void gdn_store(__global float *col, uint p, const float s[GDN_PER]) {
  for (uint m = 0; m < GDN_PER / 4; m++)
    vstore4((float4)(s[4 * m], s[4 * m + 1], s[4 * m + 2], s[4 * m + 3]), m, col + GDN_PER * p);
}

// One row's step of the recurrence (2.5) on a lane's part of column j: s <- s decay + k (beta (v - decay s . k)),
// returning s . q. The commit replays it, so both use this one function.
static inline float gdn_step(float s[GDN_PER], uint p, __local const float *k, __local const float *q, float v,
                             float decay, float beta) {
  float sk = 0.0f;
  for (uint i = 0; i < GDN_PER; i++) sk += s[i] * decay * k[GDN_PER * p + i];
  sk = gdn_sum4(sk);
  float dj = beta * (v - sk);
  float out = 0.0f;
  for (uint i = 0; i < GDN_PER; i++) {
    s[i] = s[i] * decay + k[GDN_PER * p + i] * dj;
    out += s[i] * q[GDN_PER * p + i];
  }
  return gdn_sum4(out);
}

// The recurrence (2.5) and the gated output norm (2.6), row after row: one work-group of 512 per local value head,
// 4 lanes per value column, the state in registers across rows. z and o rows are nvh x 128. In a window
// (plan->window) the slots' state is not written: the first run's last state goes to the scratch Sw (one slot's),
// and gdn_commit replays the accepted rows on each slot from u and gb, which the window leaves in place. Groups
// (nvh, width), 512.
REQD_SG __kernel void gdn_recur(__global const float *u, uint nch, __global const float *ab, __global const float *A,
                                __global const float *dt, __global float *gb, __global const float *z,
                                __global float *S, ulong slot_stride, __global const float *norm, uint nkh, uint nvh,
                                __global const StepPlan *plan, __global float *o, __global float *Sw,
                                __global uchar *oa) {
  __local float q[Q_GDN_D], k[Q_GDN_D], red[64], os[Q_GDN_D];
  const uint lv = get_group_id(0), lid = get_local_id(0), j = lid / GDN_PARTS, p = lid % GDN_PARTS;
  const uint kh = lv % nkh;
  const bool window = plan->window != 0;
  const ulong col = ((ulong)lv * Q_GDN_D + j) * Q_GDN_D;
  // Rows of distinct slots are independent. Outside a window, work-group (lv, r) takes row r if every row is another
  // slot's, else (lv, 0) takes all. A window has one run of rows a slot: (lv, g) takes the g-th run, and with one run,
  // the common case, (lv, 0) takes it without looking for runs.
  const uint nr = plan->nrows, g = get_group_id(1);
  uint r0 = 0, r1 = nr;
  if (!window) {
    bool apart = true;
    for (uint a = 0; a < nr && apart; a++)
      for (uint b = a + 1; b < nr; b++) apart = apart && plan->rows[a].slot != plan->rows[b].slot;
    if (apart) r0 = g, r1 = min(g + 1, nr);
    else if (g > 0) return;
  } else if (plan->rows[nr - 1].slot != plan->rows[0].slot) {
    uint k = 0;
    for (r0 = 0; r0 < nr && k < g; r0++) k += r0 + 1 < nr && plan->rows[r0 + 1].slot != plan->rows[r0].slot;
    if (r0 >= nr) return;
    for (r1 = r0 + 1; r1 < nr && plan->rows[r1].slot == plan->rows[r0].slot;) r1++;
  } else if (g > 0) {
    return;
  }
  float st[GDN_PER];
  for (uint r = r0; r < r1; r++) {
    __global float *sr = S + plan->rows[r].slot * slot_stride + col;
    if (r == r0 || !window || plan->rows[r].slot != plan->rows[r - 1].slot) gdn_load(sr, p, st);
    __global const float *ur = u + (ulong)r * nch;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid < Q_GDN_D) {
      q[lid] = ur[kh * Q_GDN_D + lid];
      k[lid] = ur[(nkh + kh) * Q_GDN_D + lid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float v = ur[(2 * nkh + lv) * Q_GDN_D + j];
    // The gates (2.2): ab rows hold alpha[nvh] then beta_raw[nvh]; gb rows keep g[nvh] then beta[nvh] for the commit.
    __global const float *a = ab + (ulong)r * 2 * nvh;
    const float g = A[lv] * softplus(a[lv] + dt[lv]), beta = sigmoid(a[nvh + lv]);
    if (lid == 0) {
      gb[(ulong)r * 2 * nvh + lv] = g;
      gb[(ulong)r * 2 * nvh + nvh + lv] = beta;
    }
    float out = gdn_step(st, p, k, q, v, exp(g), beta);
    if (!window) gdn_store(sr, p, st);
    out *= rsqrt((float)Q_GDN_D);
    float ss = wg_sum(p == 0 ? out * out : 0.0f, red);
    out = out * rsqrt(ss / (float)Q_GDN_D + Q_EPS) * norm[j] * sigmoid(z[((ulong)r * nvh + lv) * Q_GDN_D + j]);
    if (p == 0) {
      o[((ulong)r * nvh + lv) * Q_GDN_D + j] = out;
      os[j] = out;
    }
    // The head's 128 values into oa as a8 (a8.h): sub-group i < 4 takes values 32 i .. 32 i + 31.
    barrier(CLK_LOCAL_MEM_FENCE);
    const uint K = nvh * Q_GDN_D, i = sg_id();
    if (i < Q_GDN_D / 32) a8_store(vload2(sg_lane(), os + 32 * i), oa + (ulong)r * (K + K / 8), K, lv * 4 + i);
  }
  if (window && g == 0) gdn_store(Sw + col, p, st);
}

// The commit of a window's DeltaNet state: each run's accepted rows replayed on its slot's state from the window's u
// (rows nch apart) and gb. Groups nvh, 512.
REQD_SG __kernel void gdn_commit(__global const float *u, uint nch, __global const float *gb, __global float *S,
                                 ulong slot_stride, uint nkh, uint nvh, __global const CommitPlan *cp) {
  __local float q[Q_GDN_D], k[Q_GDN_D];
  const uint lv = get_group_id(0), lid = get_local_id(0), j = lid / GDN_PARTS, p = lid % GDN_PARTS;
  const uint kh = lv % nkh;
  for (uint run = 0; run < cp->nruns; run++) {
    __global const CommitRun *cr = cp->runs + run;
    __global float *s = S + cr->slot * slot_stride + ((ulong)lv * Q_GDN_D + j) * Q_GDN_D;
    float st[GDN_PER];
    gdn_load(s, p, st);
    for (uint r = cr->row0; r < cr->row0 + cr->n; r++) {
      __global const float *ur = u + (ulong)r * nch;
      barrier(CLK_LOCAL_MEM_FENCE);
      if (lid < Q_GDN_D) {
        q[lid] = ur[kh * Q_GDN_D + lid];
        k[lid] = ur[(nkh + kh) * Q_GDN_D + lid];
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      float v = ur[(2 * nkh + lv) * Q_GDN_D + j];
      float decay = exp(gb[(ulong)r * 2 * nvh + lv]), beta = gb[(ulong)r * 2 * nvh + nvh + lv];
      (void)gdn_step(st, p, k, q, v, decay, beta);
    }
    gdn_store(s, p, st);
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Sparse attention (section 3). This card's share: nq query heads (12 per card on two cards) and nkv KV heads; local
// query head lh reads local KV head lh / 12.

// Q rows hold nq heads of 512: query (256) then gate (256). K and V rows hold nkv heads of 256. Work-groups
// [0, nq) norm and rotate the queries in place; [nq, nq + nkv) norm and rotate the keys and store keys and values in
// the 8-bit cache at the row's position (3.2, 3.3, 3.7), in the page the table pt gives it (layout.h).
// Groups (nq + nkv, width), 256.
__kernel void attn_prep(__global float *Q, __global const float *K, __global const float *V, __global const float *qn,
                        __global const float *kn, uint nq, uint nkv, __global uchar *kc, __global uchar *vc,
                        __global const uint *pt, uint npp, __global const StepPlan *plan,
                        __global const SlotRope *sr) {
  __local float x[Q_ATT_D], y[Q_ATT_D], red[64];
  uint g = get_group_id(0), r = get_group_id(1), d = get_local_id(0);
  ROW_OR_RETURN(r)
  uint pos = plan->rows[r].pos;
  bool is_q = g < nq;
  __global float *q = Q + (ulong)r * nq * 2 * Q_ATT_D;
  float v = is_q ? q[g * 2 * Q_ATT_D + d] : K[((ulong)r * nkv + g - nq) * Q_ATT_D + d];
  float ss = wg_sum(v * v, red);
  x[d] = v * rsqrt(ss / (float)Q_ATT_D + Q_EPS) * (is_q ? qn[d] : kn[d]);
  barrier(CLK_LOCAL_MEM_FENCE);
  float rv = rope(x, d, pos - sr[plan->rows[r].slot].delta);
  if (is_q) {
    q[g * 2 * Q_ATT_D + d] = rv;
    return;
  }
  uint c = g - nq;
  ulong cell = (kv_cell(kv_page(pt, npp, plan->rows[r].slot, pos), pos) * nkv + c) * KV_HEAD_BYTES;
  y[d] = rv;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (d < Q_ATT_D / 32) q8_0_store(y + 32 * d, kc + cell + d * Q8_BLOCK);
  barrier(CLK_LOCAL_MEM_FENCE);
  y[d] = V[((ulong)r * nkv + c) * Q_ATT_D + d];
  barrier(CLK_LOCAL_MEM_FENCE);
  if (d < Q_ATT_D / 32) q8_0_store(y + 32 * d, vc + cell + d * Q8_BLOCK);
}

// The indexer's keys are split by position between the cards (docs/DESIGN.md, section 4): block b belongs to card
// b % C, and its page p (p % C == b % C) holds its pooled key and the raw keys of its 4 positions at local page p / C
// (layout.h). Each card scores its own blocks and sends its best (idx_cand); both cards merge the two lists to the
// same selection (idx_merge).

// The indexer's query and raw key (3.4): iq rows hold 4 query heads then the key (640). Work-groups 0-3 norm and
// rotate a query head in place; work-group 4 stores the raw key (8-bit) when the row's block is this card's.
// Groups (5, width), 128.
__kernel void idx_prep(__global float *iq, __global const float *qn, __global uchar *raw, __global const uint *pt,
                       uint npp, __global const StepPlan *plan, __global const SlotRope *sr, uint c, uint C) {
  __local float x[Q_IDX_D], red[64];
  uint g = get_group_id(0), r = get_group_id(1), d = get_local_id(0);
  ROW_OR_RETURN(r)
  uint pos = plan->rows[r].pos, b = pos / Q_IDX_BLOCK;
  __global float *row = iq + (ulong)r * (Q_IDX_H + 1) * Q_IDX_D;
  if (g < Q_IDX_H) {
    float v = row[g * Q_IDX_D + d];
    float ss = wg_sum(v * v, red);
    x[d] = v * rsqrt(ss / (float)Q_IDX_D + Q_EPS) * qn[d];
    barrier(CLK_LOCAL_MEM_FENCE);
    row[g * Q_IDX_D + d] = rope(x, d, pos - sr[plan->rows[r].slot].delta);
    return;
  }
  if (b % C != c) return;
  x[d] = row[Q_IDX_H * Q_IDX_D + d];
  barrier(CLK_LOCAL_MEM_FENCE);
  ulong lp = kv_cell(kv_page(pt, npp, plan->rows[r].slot, pos) / C, pos);
  if (d < Q_IDX_D / 32) q8_0_store(x + 32 * d, raw + lp * IDX_KEY_BYTES + d * Q8_BLOCK);
}

// Pools the block a row completes (its position is a block's last), on the block's card: the mean of its 4 raw keys,
// normed, rotated by the positions of the block's first cell (sr). After idx_prep, so a window's raw keys are all
// stored. Groups (width), 128.
__kernel void idx_pool(__global const uchar *raw, __global const uint *pt, uint npp, __global const float *kn,
                       __global float *pooled, __global const StepPlan *plan, __global const SlotRope *sr, uint c,
                       uint C) {
  __local float blk[Q_IDX_D], red[64];
  uint r = get_group_id(0), d = get_local_id(0);
  ROW_OR_RETURN(r)
  uint pos = plan->rows[r].pos, slot = plan->rows[r].slot;
  uint b = pos / Q_IDX_BLOCK;
  if (pos % Q_IDX_BLOCK != Q_IDX_BLOCK - 1 || b % C != c) return;
  const uint lpage = kv_page(pt, npp, slot, pos) / C;
  __global const uchar *rk = raw + kv_cell(lpage, 0) * IDX_KEY_BYTES;
  float mean = 0.0f;
  for (uint t = 0; t < Q_IDX_BLOCK; t++) mean += q8_0_load(rk + (ulong)t * IDX_KEY_BYTES, d);
  mean *= 1.0f / Q_IDX_BLOCK;
  float ss = wg_sum(mean * mean, red);
  blk[d] = mean * rsqrt(ss / (float)Q_IDX_D + Q_EPS) * kn[d];
  barrier(CLK_LOCAL_MEM_FENCE);
  __global const SlotRope *s = sr + slot;
  uint b0 = Q_IDX_BLOCK * b, t = b0 - s->delta, h = t, w = t;
  if (b0 == s->blk) t = s->t, h = s->h, w = s->w;
  pooled[(ulong)lpage * Q_IDX_D + d] = rope3(blk, d, t, h, w);
}

// This card's blocks complete at a row's position: b < nb with b % C == c, local indices 0 .. own - 1.
static inline uint idx_own(uint nb, uint c, uint C) { return nb > c ? (nb - c + C - 1) / C : 0; }

// score[lb] = sum_h relu(iq_h . K_b) for this card's blocks complete at the row's position (3.4, step 4), by local
// index lb (block lb C + c); rows nb_max apart. Groups (ceil(nb_max / 64), width), 64.
__kernel void idx_score(__global const float *iq, __global const float *pooled, __global const uint *pt, uint npp,
                        uint nb_max, __global const StepPlan *plan, __global float *scores, uint c, uint C) {
  uint lb = get_global_id(0), r = get_group_id(1);
  ROW_OR_RETURN(r)
  uint nkv = plan->rows[r].pos + 1;
  if (nkv <= Q_IDX_CELLS || lb >= idx_own(nkv / Q_IDX_BLOCK, c, C)) return;
  const uint page = kv_page(pt, npp, plan->rows[r].slot, (lb * C + c) * Q_IDX_BLOCK);
  __global const float *K = pooled + (ulong)(page / C) * Q_IDX_D;
  __global const float *q = iq + (ulong)r * (Q_IDX_H + 1) * Q_IDX_D;
  float score = 0.0f;
  for (uint h = 0; h < Q_IDX_H; h++) {
    float dot = 0.0f;
    for (uint i = 0; i < Q_IDX_D; i++) dot += q[h * Q_IDX_D + i] * K[i];
    score += fmax(dot, 0.0f);
  }
  scores[(ulong)r * nb_max + lb] = score;
}

// A row's candidates on this card: its best Q_IDX_BLOCKS + 1 blocks, in block order, as (block, score) pairs, after
// their count: cand rows IDX_CAND words apart. Rows within the width take every cell and send none. Groups (width),
// TOPK_WG.
#define IDX_CAND (2 * (Q_IDX_BLOCKS + 1) + 2)
__kernel void idx_cand(__global const float *scores, uint nb_max, __global const StepPlan *plan,
                       __global uint *blocks_all, __global float *bscores_all, __global uint *cand_all, uint c,
                       uint C) {
  __local uint hist[TOPK_HIST];
  __local uint scan[TOPK_WG];
  uint r = get_group_id(0), lid = get_local_id(0);
  ROW_OR_RETURN(r)
  uint nkv = plan->rows[r].pos + 1;
  __global uint *cand = cand_all + (ulong)r * IDX_CAND;
  if (nkv <= Q_IDX_CELLS) {
    if (lid == 0) cand[0] = 0;
    return;
  }
  uint own = idx_own(nkv / Q_IDX_BLOCK, c, C);
  __global uint *blocks = blocks_all + (ulong)r * (Q_IDX_BLOCKS + 1);
  __global float *bscores = bscores_all + (ulong)r * (Q_IDX_BLOCKS + 1);
  uint got = topk_select(scores + (ulong)r * nb_max, own, Q_IDX_BLOCKS + 1, blocks, bscores, hist, scan);
  barrier(CLK_GLOBAL_MEM_FENCE);
  for (uint i = lid; i < got; i += TOPK_WG) vstore2((uint2)(blocks[i] * C + c, as_uint(bscores[i])), 1 + i, cand);
  if (lid == 0) cand[0] = got;
}

// How many of the n increasing blocks in l are below b.
static inline uint idx_below(__local const uint *l, uint n, uint b) {
  uint lo = 0, hi = n;
  while (lo < hi) {
    uint mid = (lo + hi) / 2;
    if (l[mid] < b) lo = mid + 1; else hi = mid;
  }
  return lo;
}

// The cells row r attends to (3.4, steps 5-6), increasing, into cells[r] (Q_IDX_CELLS apart) and ncells[r], alike on
// all cards: every cached position up to llama.cpp's 2,051; beyond, the tail after the last whole block, the 512 best
// blocks and the first cells of the 513th, from the cards' candidates (cand: C blocks of width rows, IDX_CAND words a
// row) merged in block order, so ties go to the lower block as on one card. Groups (width), TOPK_WG.
__kernel void idx_merge(__global const uint *cand_all, uint C, uint width, __global const StepPlan *plan,
                        __global float *merged_all, __global uint *blocks_all, __global float *bscores_all,
                        __global uint *cells_all, __global uint *ncells) {
  __local uint hist[TOPK_HIST];
  __local uint scan[TOPK_WG];
  __local uint mb[2 * (Q_IDX_BLOCKS + 1)], lb[2][Q_IDX_BLOCKS + 1];
  uint r = get_group_id(0), lid = get_local_id(0);
  ROW_OR_RETURN(r)
  uint nkv = plan->rows[r].pos + 1;
  __global uint *cells = cells_all + (ulong)r * Q_IDX_CELLS;
  if (nkv <= Q_IDX_CELLS) {
    for (uint i = lid; i < nkv; i += TOPK_WG) cells[i] = i;
    if (lid == 0) ncells[r] = nkv;
    return;
  }
  // The cards' lists (C is 1 or 2) are host memory: read in one round trip, a count and a pair per work-item (a list
  // has at most Q_IDX_BLOCKS + 1 < TOPK_WG pairs), with the blocks copied to local memory for the searches.
  __global const uint *c0 = cand_all + (ulong)r * IDX_CAND, *c1 = cand_all + ((ulong)width + r) * IDX_CAND;
  const bool has = lid <= Q_IDX_BLOCKS;
  uint n0 = c0[0], n1 = C > 1 ? c1[0] : 0;
  uint2 p0 = has ? vload2(1 + lid, c0) : (uint2)(0), p1 = has && C > 1 ? vload2(1 + lid, c1) : (uint2)(0);
  if (lid < n0) lb[0][lid] = p0.x;
  if (lid < n1) lb[1][lid] = p1.x;
  barrier(CLK_LOCAL_MEM_FENCE);
  // The merged list, by block: each candidate's place is its rank in its own list plus the other list's candidates
  // of lower blocks.
  __global float *merged = merged_all + (ulong)r * 2 * (Q_IDX_BLOCKS + 1);
  const uint total = n0 + n1;
  if (lid < n0) {
    uint at = lid + idx_below(lb[1], n1, p0.x);
    mb[at] = p0.x;
    merged[at] = as_float(p0.y);
  }
  if (lid < n1) {
    uint at = lid + idx_below(lb[0], n0, p1.x);
    mb[at] = p1.x;
    merged[at] = as_float(p1.y);
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  __global uint *blocks = blocks_all + (ulong)r * (Q_IDX_BLOCKS + 1);
  __global float *bscores = bscores_all + (ulong)r * (Q_IDX_BLOCKS + 1);
  uint nb = nkv / Q_IDX_BLOCK, tail = nkv - nb * Q_IDX_BLOCK;
  uint extra = Q_IDX_CELLS - Q_IDX_TOPK - tail;  // cells of the 513th block: 3 - tail
  uint want = Q_IDX_BLOCKS + (extra > 0 ? 1 : 0);
  topk_select(merged, total, want, blocks, bscores, hist, scan);
  barrier(CLK_GLOBAL_MEM_FENCE);
  for (uint i = lid; i < want; i += TOPK_WG) blocks[i] = mb[blocks[i]];
  barrier(CLK_GLOBAL_MEM_FENCE);
  // The 513th best is the lowest score among the chosen (the last such in block order); want <= TOPK_WG.
  uint worst = want;
  if (extra > 0) {
    const float s = lid < want ? bscores[lid] : INFINITY;
    const float least = work_group_reduce_min(s);
    worst = work_group_reduce_max(lid < want && s == least ? lid : 0u);
  }
  for (uint i = lid; i < want; i += TOPK_WG) {
    uint b = blocks[i];
    if (i == worst) {
      for (uint t = 0; t < extra; t++) cells[4 * i + t] = 4 * b + t;
    } else {
      uint at = i < worst ? 4 * i : 4 * (i - 1) + extra;
      for (uint t = 0; t < Q_IDX_BLOCK; t++) cells[at + t] = 4 * b + t;
    }
  }
  if (lid < tail) cells[Q_IDX_TOPK + extra + lid] = nb * Q_IDX_BLOCK + lid;
  if (lid == 0) ncells[r] = Q_IDX_CELLS;
}

// Attention over the chosen cells (3.5) and the output gate (3.6), split so every core has work: work-group (KV
// head, row, s) takes cells [s * ATT_SPLIT, +ATT_SPLIT), found through the page table pt, for the ATT_GQA query heads
// of that KV head, loading K and V once for all of them, and writes each head's unnormalised output, max and sum into
// part (ATT_PART floats a head and split); attn_merge combines the splits in order. Groups (nkv, width, ATT_SPLITS),
// 256.
#define ATT_SPLIT 64
#define ATT_SPLITS ((Q_IDX_CELLS + ATT_SPLIT - 1) / ATT_SPLIT)
#define ATT_PART (Q_ATT_D + 2)
#define ATT_GQA (Q_ATT_H / Q_ATT_KV)
REQD_SG __kernel void attn_split(__global const float *Q, __global const uchar *kc, __global const uchar *vc,
                                 __global const uint *pt, uint npp, uint nq, uint nkv, __global const uint *cells_all,
                                 __global const uint *ncells, __global const StepPlan *plan, __global float *part) {
  __local float q[ATT_GQA][Q_ATT_D], p[ATT_GQA][ATT_SPLIT];
  __local uint cl[ATT_SPLIT];
  uint c = get_group_id(0), r = get_group_id(1), s = get_group_id(2), d = get_local_id(0);
  ROW_OR_RETURN(r)
  uint n = ncells[r], i0 = s * ATT_SPLIT;
  if (i0 >= n) return;
  uint m = min((uint)ATT_SPLIT, n - i0);
  for (uint h = 0; h < ATT_GQA; h++) q[h][d] = Q[((ulong)r * nq + c * ATT_GQA + h) * 2 * Q_ATT_D + d];
  __global const uint *cells = cells_all + (ulong)r * Q_IDX_CELLS + i0;
  if (d < m) {
    const uint pos = cells[d];
    cl[d] = (uint)kv_cell(kv_page(pt, npp, plan->rows[r].slot, pos), pos);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  __global const uchar *K = kc + c * KV_HEAD_BYTES;
  __global const uchar *V = vc + c * KV_HEAD_BYTES;
  ulong cell_bytes = (ulong)nkv * KV_HEAD_BYTES;
  uint lane = sg_lane(), sg = sg_id(), nsg = get_num_sub_groups();
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
      if (lane == 0) p[h][i] = dot * (1.0f / 16.0f);
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  __global float *po = part + (((ulong)r * nq + c * ATT_GQA) * ATT_SPLITS + s) * ATT_PART;
  if (sg < ATT_GQA) {
    float mx = -INFINITY;
    for (uint i = lane; i < m; i += SG_SIZE) mx = fmax(mx, p[sg][i]);
    mx = sub_group_reduce_max(mx);
    float sum = 0.0f;
    for (uint i = lane; i < m; i += SG_SIZE) {
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

// The splits of attn_split combined in order, then the output gate; unless oa is null, the rows also go to oa as a8
// (a8.h; rows nq 256 + nq 32 bytes apart). Groups (nq, width), 256.
REQD_SG __kernel void attn_merge(__global const float *Q, uint nq, __global const uint *ncells,
                                 __global const StepPlan *plan, __global const float *part, __global float *out,
                                 __global uchar *oa) {
  __local float os[Q_ATT_D];
  uint lh = get_group_id(0), r = get_group_id(1), d = get_local_id(0);
  ROW_OR_RETURN(r)
  uint ns = (ncells[r] + ATT_SPLIT - 1) / ATT_SPLIT;
  __global const float *pt = part + ((ulong)r * nq + lh) * ATT_SPLITS * ATT_PART;
  float mx = -INFINITY;
  for (uint s = 0; s < ns; s++) mx = fmax(mx, pt[s * ATT_PART + Q_ATT_D]);
  float acc = 0.0f, sum = 0.0f;
  for (uint s = 0; s < ns; s++) {
    float f = exp(pt[s * ATT_PART + Q_ATT_D] - mx);
    acc += f * pt[s * ATT_PART + d];
    sum += f * pt[s * ATT_PART + Q_ATT_D + 1];
  }
  float gate = Q[((ulong)r * nq + lh) * 2 * Q_ATT_D + Q_ATT_D + d];
  const float y = acc / sum * sigmoid(gate);
  out[((ulong)r * nq + lh) * Q_ATT_D + d] = y;
  if (!oa) return;
  os[d] = y;
  barrier(CLK_LOCAL_MEM_FENCE);
  const uint K = nq * Q_ATT_D, i = sg_id();
  if (i < Q_ATT_D / 32) a8_store(vload2(sg_lane(), os + 32 * i), oa + (ulong)r * (K + K / 8), K, lh * 8 + i);
}

// ---------------------------------------------------------------------------------------------------------------
// The MoE block (section 4)

// The softmax's maximum and sum, lane holding experts lane + 16 j.
static inline float2 rt_softmax(__global const float *logits) {
  const uint lane = sg_lane();
  float v[RT_PER];
  float mx = -INFINITY;
  for (uint j = 0; j < RT_PER; j++) {
    v[j] = logits[lane + SG_SIZE * j];
    mx = fmax(mx, v[j]);
  }
  mx = sub_group_reduce_max(mx);
  float sum = 0.0f;
  for (uint j = 0; j < RT_PER; j++) sum += exp(v[j] - mx);
  return (float2)(mx, sub_group_reduce_add(sum));
}

// Row r's results from its winners (lane k < 10: expert id, logit v, slot) and the softmax's (max, sum): the experts,
// weights and slots, nmiss[r], and with a job, the misses and the routed experts. Returns the row's miss count.
static inline uint rt_finish(uint r, int id, float v, int slot, float2 ms, __global int *ids_all, __global float *w_all,
                             __global int *slots_all, __global uint *nmiss, __global MissJob *job) {
  const uint lane = sg_lane();
  float pk = lane < Q_USED ? exp(v - ms.x) / ms.y : 0.0f;
  float tot = 0.0f;
  for (uint k = 0; k < Q_USED; k++) tot += sub_group_broadcast(pk, k);
  float wk = pk / fmax(tot, 6.1035e-5f);
  if (lane < Q_USED) {
    ids_all[r * Q_USED + lane] = id;
    w_all[r * Q_USED + lane] = wk;
    slots_all[r * Q_USED + lane] = slot;
    if (job) job->all[lane] = id;
  }
  // The misses, in rank order: lane k's place is the misses ranked before it.
  uint miss = lane < Q_USED && slot < 0;
  uint at = sub_group_scan_exclusive_add(miss);
  uint n = sub_group_reduce_add(miss);
  if (lane == 0) nmiss[r] = n;
  if (job && miss) {
    job->ids[at] = id;
    job->w[at] = wk;
  }
  return n;
}

// The job's miss count, then the step's number behind a system-scope release fence; by the row's sub-group.
static inline void rt_publish(__global MissJob *job, uint n, __global const StepPlan *plan) {
  sub_group_barrier(CLK_GLOBAL_MEM_FENCE);
  if (sg_lane() == 0) {
    job->nmiss = n;
    atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_release, memory_scope_all_svm_devices);
    atomic_store_explicit((volatile __global atomic_uint *)&job->seq, plan->step, memory_order_release,
                          memory_scope_all_svm_devices);
  }
}

// Row r, all of it, by its sub-group (cand: RT_SLM of local memory). Returns the lane's slot.
static inline int route(uint r, __global const float *rl, __global const int *slot_map, __global const float *x,
                        __global int *ids_all, __global float *w_all, __global int *slots_all, __global uint *nmiss,
                        __global MissJob *job, __global const StepPlan *plan, __local uint2 *cand) {
  const uint lane = sg_lane();
  __global const float *logits = rl + (ulong)r * (Q_EXPERTS + 1);
  const float2 ms = rt_softmax(logits);
  uint key[RT_PER];
  for (uint j = 0; j < RT_PER; j++) key[j] = topk_key(logits[lane + SG_SIZE * j]);
  const uint2 win = rt_top(key, cand);
  const int id = (int)win.y, slot = lane < Q_USED ? slot_map[id] : 0;
  const uint n = rt_finish(r, id, rt_value(win.x), slot, ms, ids_all, w_all, slots_all, nmiss, job);
  if (!job) return slot;
  if (n > 0)
    for (uint i = lane; i < Q_EMBD / 4; i += SG_SIZE) vstore4(vload4(i, x + (ulong)r * Q_EMBD), i, job->x);
  rt_publish(job, n, plan);
  return slot;
}

// The router (4.2), a sub-group per row so nothing waits on a work-group barrier: softmax over the 512 logits (rl rows
// hold them, then the shared expert's gate), the 10 largest, weights renormalised, into ids, w and slots at r * Q_USED
// (slot_map: an int per expert, negative when not cached), the uncached count into nmiss[r]. With jobs (card 0), row
// r's MissJob gets the misses and x when there are any, then the step's number. Groups (width), 16.
REQD_SG __kernel void router(__global const float *rl, __global const int *slot_map, __global const float *x,
                             __global int *ids_all, __global float *w_all, __global int *slots_all,
                             __global uint *nmiss, __global MissJob *jobs, __global const StepPlan *plan) {
  __local uint2 cand[RT_SLM];
  const uint r = get_group_id(0);
  ROW_OR_RETURN(r)
  route(r, rl, slot_map, x, ids_all, w_all, slots_all, nmiss, jobs ? jobs + r : 0, plan, cand);
}

// Waits for the CPU's answer to a layer's MissJobs (card 0): returns at once when no row had a miss (the skip); else
// spins on the answer's flag with a system-scope acquire fence before each volatile load, the only form that always
// sees the CPU's store (docs/measurements/MEASUREMENTS.md#2026-10-11-doorbell). Bounded: gives up after 2^30 polls and sets *stuck.
// One work-item.
__kernel void cpu_wait(__global const uint *nmiss, __global const uint *ans_seq, __global const StepPlan *plan,
                       __global uint *stuck) {
  uint any = 0;
  for (uint r = 0; r < plan->nrows; r++) any += nmiss[r];
  if (any == 0) return;
  const uint want = plan->step;
  for (uint i = 0; i < (1u << 30); i++) {
    atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_acquire, memory_scope_all_svm_devices);
    if (*(volatile __global const uint *)ans_seq == want) return;
  }
  *stuck = 1;
}

// This card's partial sum of the block (4.3, 4.4) for row r: the cached experts' outputs y[r][k] weighted in rank
// order, the shared expert's s gated by sigma(the router's last logit), and on card 0 the CPU's sum for the misses
// (cpu, else null) when the row had any. Groups (Q_EMBD / 64, width), 64.
__kernel void moe_combine(__global const float *y, __global const float *w, __global const int *slots,
                          __global const float *s, __global const float *rl, __global const float *cpu,
                          __global const uint *nmiss, __global const StepPlan *plan, __global float *out) {
  uint i = get_global_id(0), r = get_group_id(1);
  ROW_OR_RETURN(r)
  float acc = 0.0f;
  for (uint k = 0; k < Q_USED; k++)
    if (slots[r * Q_USED + k] >= 0) acc += w[r * Q_USED + k] * y[((ulong)r * Q_USED + k) * Q_EMBD + i];
  acc += sigmoid(rl[(ulong)r * (Q_EXPERTS + 1) + Q_EXPERTS]) * s[(ulong)r * Q_EMBD + i];
  if (cpu && nmiss[r] > 0) {
    atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_acquire, memory_scope_all_svm_devices);
    acc += ((volatile __global const float *)cpu)[(ulong)r * Q_EMBD + i];
  }
  out[(ulong)r * Q_EMBD + i] = acc;
}

// The step's cached experts grouped (MoeGroup): every assignment a = r * Q_USED + k of the plan's rows whose slot is
// on the card joins its expert's group. A row routes to 10 distinct experts, so a group has at most one assignment
// per row: STEP_MAX_ROWS <= MOE_GROUP_MAX. Groups are numbered by their first assignment, and members ordered, so
// the result does not depend on scheduling. One work-group of 128 (>= STEP_MAX_ROWS * Q_USED).
static inline void group_slots(__global const int *slots, __global const StepPlan *plan, __global MoeGroup *groups,
                               __global uint *ngroups, __local int *sl, __local uint *head_of, __local uint *scan) {
  const uint a = get_local_id(0), n = plan->nrows * Q_USED;
  sl[a] = a < n ? slots[a] : -1;
  barrier(CLK_LOCAL_MEM_FENCE);
  int s = sl[a];
  uint leader = a, rank = 0;
  if (s >= 0)
    for (uint b = 0; b < a; b++)
      if (sl[b] == s) {
        if (leader == a) leader = b;
        rank++;
      }
  bool head = s >= 0 && leader == a;
  scan[a] = head ? 1 : 0;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint off = 1; off < 128; off <<= 1) {
    uint v = a >= off ? scan[a - off] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    scan[a] += v;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (head) head_of[a] = scan[a] - 1;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (s >= 0) {
    uint g = head_of[leader];
    groups[g].asg[rank] = a;
    if (head) {
      uint cnt = 0;
      for (uint b = a; b < n; b++) cnt += sl[b] == s;
      groups[g].slot = s;
      groups[g].n = cnt;
    }
  }
  if (a == 127) *ngroups = scan[127];
}

__kernel void moe_group(__global const int *slots, __global const StepPlan *plan, __global MoeGroup *groups,
                        __global uint *ngroups) {
  __local int sl[128];
  __local uint head_of[128], scan[128];
  group_slots(slots, plan, groups, ngroups, sl, head_of, scan);
}

// group_slots for a step of one row, by its sub-group: lane a < Q_USED holds assignment a's slot.
static inline void group_row(int slot, __global MoeGroup *groups, __global uint *ngroups) {
  const uint a = sg_lane();
  const int s = a < Q_USED ? slot : -1;
  uint leader = a, rank = 0, cnt = 0;
  for (uint b = 0; b < Q_USED; b++) {
    const int sb = sub_group_broadcast(s, b);
    if (s >= 0 && sb == s) {
      if (b < a) {
        if (leader == a) leader = b;
        rank++;
      } else {
        cnt++;
      }
    }
  }
  const uint head = s >= 0 && leader == a;
  const uint g = (uint)sg_shuffle_ul((ulong)sub_group_scan_exclusive_add(head), leader);
  if (s >= 0) {
    groups[g].asg[rank] = a;
    if (head) {
      groups[g].slot = s;
      groups[g].n = cnt;
    }
  }
  const uint ng = sub_group_reduce_add(head);
  if (a == 0) *ngroups = ng;
}

// router and moe_group in one launch: sub-group r routes row r, then the work-group groups the slots. A step of one
// row splits the routing: sub-group 0 takes the top 10, 1 the softmax's sum, 2 copies slot_map to local memory, and
// the others copy x to the MissJob, misses or not; after a barrier, sub-group 0 finishes, publishes the job and
// groups the row's slots. Groups 1, 128.
REQD_SG __kernel void router_group(__global const float *rl, __global const int *slot_map, __global const float *x,
                                   __global int *ids_all, __global float *w_all, __global int *slots_all,
                                   __global uint *nmiss, __global MissJob *jobs, __global const StepPlan *plan,
                                   __global MoeGroup *groups, __global uint *ngroups) {
  __local int sl[128];
  __local uint head_of[128], scan[128];
  __local uint2 cand[STEP_MAX_ROWS * RT_SLM];
  __local int sm[Q_EXPERTS];
  __local float2 ms;
  const uint sg = sg_id(), lane = sg_lane();
  if (plan->nrows == 1) {
    uint2 win = (uint2)(0, 0);
    if (sg == 0) {
      uint key[RT_PER];
      for (uint j = 0; j < RT_PER; j++) key[j] = topk_key(rl[lane + SG_SIZE * j]);
      win = rt_top(key, cand);
    } else if (sg == 1) {
      const float2 v = rt_softmax(rl);
      if (lane == 0) ms = v;
    } else if (sg == 2) {
      for (uint j = 0; j < RT_PER; j++) sm[lane + SG_SIZE * j] = slot_map[lane + SG_SIZE * j];
    } else if (jobs) {
      for (uint i = get_local_id(0) - 3 * SG_SIZE; i < Q_EMBD / 4; i += 128 - 3 * SG_SIZE)
        vstore4(vload4(i, x), i, jobs->x);
      atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_release, memory_scope_all_svm_devices);
    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    if (sg != 0) return;
    const int id = (int)win.y, slot = lane < Q_USED ? sm[id] : 0;
    const uint n = rt_finish(0, id, rt_value(win.x), slot, ms, ids_all, w_all, slots_all, nmiss, jobs);
    if (jobs) rt_publish(jobs, n, plan);
    group_row(slot, groups, ngroups);
    return;
  }
  if (sg < plan->nrows)
    route(sg, rl, slot_map, x, ids_all, w_all, slots_all, nmiss, jobs ? jobs + sg : 0, plan, cand + sg * RT_SLM);
  barrier(CLK_GLOBAL_MEM_FENCE);
  group_slots(slots_all, plan, groups, ngroups, sl, head_of, scan);
}

// swiglu_rows, stored only as a8 (a8.h) into ha (rows ff + ff / 8 bytes apart): silu(g) * u for 32 of a row's ff
// values (ff a multiple of 32) per sub-group, two per lane. Groups (ceil(ff / 128), width), 64.
REQD_SG __kernel void swiglu_a8(__global const float *g, __global const float *u, uint ff, __global const StepPlan *plan,
                                __global uchar *ha) {
  const uint r = get_group_id(1), unit = get_group_id(0) * 4 + sg_id();
  ROW_OR_RETURN(r)
  if (unit >= ff / 32) return;
  const ulong i = (ulong)r * ff + 32 * unit + 2 * sg_lane();
  a8_store((float2)(silu(g[i]) * u[i], silu(g[i + 1]) * u[i + 1]), ha + (ulong)r * (ff + ff / 8), ff, unit);
}

// silu(g) * u for the shared expert's ff values of each row. Groups (ceil(ff / 64), width), 64.
__kernel void swiglu_rows(__global const float *g, __global const float *u, uint ff, __global const StepPlan *plan,
                          __global float *h) {
  uint i = get_global_id(0), r = get_group_id(1);
  ROW_OR_RETURN(r)
  if (i < ff) h[(ulong)r * ff + i] = silu(g[(ulong)r * ff + i]) * u[(ulong)r * ff + i];
}

// ---------------------------------------------------------------------------------------------------------------
// The PLE module on layer 1 (MODEL_IO.md, 1.3). k rows = W_k u (4 x 2560), v rows = W_v u (2560).

// Per stream s and row: the gate sigma(sign(a) sqrt(max(|a|, 1e-6))), a = norm(k_s) . norm(R_s) / sqrt(2560)
// (steps 1, 3, 4). Groups (Q_HC, width), 256.
__kernel void ple_gate(__global const float *k, __global const float *R, __global const float *gk,
                       __global const float *gq, __global const StepPlan *plan, __global float *gates) {
  __local float red[64];
  uint s = get_group_id(0), r = get_group_id(1);
  ROW_OR_RETURN(r)
  __global const float *ks = k + (ulong)r * Q_HC_DIM + s * Q_EMBD;
  __global const float *rs = R + (ulong)r * Q_HC_DIM + s * Q_EMBD;
  float kk = 0.0f, rr = 0.0f;
  for (uint i = get_local_id(0); i < Q_EMBD; i += get_local_size(0)) {
    kk += ks[i] * ks[i];
    rr += rs[i] * rs[i];
  }
  kk = rsqrt(wg_sum(kk, red) / (float)Q_EMBD + Q_EPS);
  rr = rsqrt(wg_sum(rr, red) / (float)Q_EMBD + Q_EPS);
  float dot = 0.0f;
  for (uint i = get_local_id(0); i < Q_EMBD; i += get_local_size(0))
    dot += ks[i] * kk * gk[s * Q_EMBD + i] * rs[i] * rr * gq[s * Q_EMBD + i];
  float a = wg_sum(dot, red) * rsqrt((float)Q_EMBD);
  if (get_local_id(0) == 0) gates[r * Q_HC + s] = sigmoid(sign(a) * sqrt(fmax(fabs(a), 1e-6f)));
}

// Per stream s and row: G_s = gate_s v, N_s = norm(G_s) * g_c (steps 5, 6). Groups (Q_HC, width), 256.
__kernel void ple_norm(__global const float *v, __global const float *gates, __global const float *gc,
                       __global const StepPlan *plan, __global float *N) {
  __local float red[64];
  uint s = get_group_id(0), r = get_group_id(1);
  ROW_OR_RETURN(r)
  float gate = gates[r * Q_HC + s];
  __global const float *vr = v + (ulong)r * Q_EMBD;
  float ss = 0.0f;
  for (uint i = get_local_id(0); i < Q_EMBD; i += get_local_size(0)) ss += gate * vr[i] * gate * vr[i];
  float rs = rsqrt(wg_sum(ss, red) / (float)Q_EMBD + Q_EPS);
  for (uint i = get_local_id(0); i < Q_EMBD; i += get_local_size(0))
    N[(ulong)r * Q_HC_DIM + s * Q_EMBD + i] = gate * vr[i] * rs * gc[s * Q_EMBD + i];
}

// Per channel, row after row: the dilated causal convolution and SiLU (step 7), then R += G + C (step 8). The
// history keeps the last 9 N in a ring by position. In a window (plan->window) it is not written: a row takes the
// earlier rows' N of its own run from N itself, and the commit (ple_commit) writes the accepted rows' (N stays in
// place for it). Groups Q_HC_DIM / 64, 64.
__kernel void ple_conv(__global const float *N, __global const float *v, __global const float *gates,
                       __global const float *w, __global float *hist, ulong slot_stride,
                       __global const StepPlan *plan, __global float *R) {
  uint ch = get_global_id(0);
  __global const float *k = w + ch * Q_GDN_CONV;
  const bool window = plan->window != 0;
  for (uint r = 0; r < plan->nrows; r++) {
    uint pos = plan->rows[r].pos;
    __global float *h = hist + plan->rows[r].slot * slot_stride;
    float nv[3];
    for (uint jj = 0; jj < 3; jj++) {
      uint d = Q_PLE_DIL * (3 - jj);
      nv[jj] = window && r >= d && plan->rows[r - d].slot == plan->rows[r].slot
                   ? N[(ulong)(r - d) * Q_HC_DIM + ch]
                   : h[((pos + Q_PLE_HIST - d) % Q_PLE_HIST) * Q_HC_DIM + ch];
    }
    float nt = N[(ulong)r * Q_HC_DIM + ch];
    float acc = k[0] * nv[0];
    acc += k[1] * nv[1];
    acc += k[2] * nv[2];
    acc += k[3] * nt;
    if (!window) h[(pos % 9) * Q_HC_DIM + ch] = nt;
    R[(ulong)r * Q_HC_DIM + ch] += gates[r * Q_HC + ch / Q_EMBD] * v[(ulong)r * Q_EMBD + ch % Q_EMBD] + silu(acc);
  }
}

// The commit of a window's PLE history, run by run. Groups Q_HC_DIM / 64, 64.
__kernel void ple_commit(__global const float *N, __global float *hist, ulong slot_stride,
                         __global const CommitPlan *cp) {
  uint ch = get_global_id(0);
  for (uint k = 0; k < cp->nruns; k++) {
    __global const CommitRun *run = cp->runs + k;
    __global float *h = hist + run->slot * slot_stride;
    for (uint r = 0; r < run->n; r++)
      h[((run->pos0 + r) % 9) * Q_HC_DIM + ch] = N[(ulong)(run->row0 + r) * Q_HC_DIM + ch];
  }
}

// ---------------------------------------------------------------------------------------------------------------
// The draft (MTP) layer's own steps (MODEL_IO.md, section 3); its decoder layer runs on the kernels above.

// mtp_norm_e: out[r] = RMSNorm(e[r]) * g over 2,560 values (the token's embedding). Groups (width), 256.
__kernel void mtp_norm_e(__global const float *e, __global const float *g, __global const StepPlan *plan,
                         __global float *out) {
  __local float red[64];
  uint r = get_group_id(0);
  ROW_OR_RETURN(r)
  __global const float *x = e + (ulong)r * Q_EMBD;
  float ss = 0.0f;
  for (uint i = get_local_id(0); i < Q_EMBD; i += get_local_size(0)) ss += x[i] * x[i];
  float rs = rsqrt(wg_sum(ss, red) / (float)Q_EMBD + Q_EPS);
  for (uint i = get_local_id(0); i < Q_EMBD; i += get_local_size(0)) out[(ulong)r * Q_EMBD + i] = x[i] * rs * g[i];
}

// mtp_norm_h: out[r] = RMSNorm(H[src_row(r)]) * g over all 10,240 values at once (not per stream). src_row: rows of
// H in order, or, with chain, the row in 4-bit field r of plan->pad (the draft before's own residual). Groups
// (width), 256.
__kernel void mtp_norm_h(__global const float *H, __global const float *g, uint chain, __global const StepPlan *plan,
                         __global float *out) {
  __local float red[64];
  uint r = get_group_id(0);
  ROW_OR_RETURN(r)
  __global const float *x = H + (ulong)(chain ? (plan->pad >> (4 * r)) & 15 : r) * Q_HC_DIM;
  float ss = 0.0f;
  for (uint i = get_local_id(0); i < Q_HC_DIM; i += get_local_size(0)) ss += x[i] * x[i];
  float rs = rsqrt(wg_sum(ss, red) / (float)Q_HC_DIM + Q_EPS);
  for (uint i = get_local_id(0); i < Q_HC_DIM; i += get_local_size(0)) out[(ulong)r * Q_HC_DIM + i] = x[i] * rs * g[i];
}

// R[r][s] = hs[r][s] + e[r] for each stream s. Groups (Q_HC_DIM / 64, width), 64.
__kernel void mtp_sum(__global const float *hs, __global const float *e, __global const StepPlan *plan,
                      __global float *R) {
  uint i = get_global_id(0), r = get_group_id(1);
  ROW_OR_RETURN(r)
  R[(ulong)r * Q_HC_DIM + i] = hs[(ulong)r * Q_HC_DIM + i] + e[(ulong)r * Q_EMBD + i % Q_EMBD];
}

// The cells a draft row attends to: the last min(Q_IDX_CELLS, positions since first[slot]) up to its own, densely.
// Groups (width), 256.
__kernel void mtp_cells(__global const uint *first, __global const StepPlan *plan, __global uint *cells_all,
                        __global uint *ncells) {
  uint r = get_group_id(0);
  ROW_OR_RETURN(r)
  uint pos = plan->rows[r].pos, f = min(first[plan->rows[r].slot], pos);
  uint n = min((uint)Q_IDX_CELLS, pos + 1 - f);
  __global uint *cells = cells_all + (ulong)r * Q_IDX_CELLS;
  for (uint i = get_local_id(0); i < n; i += get_local_size(0)) cells[i] = pos + 1 - n + i;
  if (get_local_id(0) == 0) ncells[r] = n;
}
