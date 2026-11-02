// deepseek41's decode-step kernels between its products: hyper-connections, Engram and the MoE block's. The maths is
// the deepseek41 forward pass; core/models/deepseek41/ref_ops.h holds one scalar definition of each. A row is one step's
// token: X is [rows][D41_HC][D41_EMBD] f32, ss one sum of squares of a row's D41_HC_DIM values. Row counts are the
// recorded width, passed by value, and a padded row costs kernel time only. ds_router alone reads the plan, for the
// real row count and the step, so those words must reach the card before it runs.
#include "a8.h"
#include "builtins.h"
#include "layout.h"
#include "ops.h"
#include "topk.h"

#define RT_SG SG_SIZE
#include "route.h"

static inline float sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }

// Comes before reading what another agent wrote to host memory, as the card caches host-memory lines (ADR-0011). It
// invalidates the card's cache, so a launch that reads only card memory leaves it out.
static inline void host_read_fence(void) {
  atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_acquire, memory_scope_all_svm_devices);
}

// One chunk of a sublayer's input, values i .. i + 63, into each form its consumers read; a null pointer has none.
// job's x lies in host memory: 256-byte stores.
static inline void store_x(float4 x, uint i, __global uchar *xa, __global ushort *xb, __global float *xf,
                           __global D41MissJob *job) {
  if (xf) wr4(xf + i, x);
  if (job) wr4((__global float *)job->x + i, x);
  if (xb)
    intel_sub_group_block_write_us4(xb + i, (ushort4)(f32_to_bf16(x.s0), f32_to_bf16(x.s1), f32_to_bf16(x.s2),
                                                      f32_to_bf16(x.s3)));
  if (xa) {
    a8_store_block(x.lo, xa, D41_EMBD, i / 32);
    a8_store_block(x.hi, xa, D41_EMBD, i / 32 + 1);
  }
}

#define HC_VALUES (D41_EMBD / D41_HC_SGS)  // values a sub-group
#define HC_CHUNKS (HC_VALUES / 64)         // its chunks of 64
#define HC_LOCALS                                                      \
  __local float red[2][D41_HC_SGS];                                    \
  const uint r = get_group_id(0), s = sg_id(), lane = sg_lane();       \
  const uint i0 = s * HC_VALUES;                                       \
  __global uchar *xar = xa ? xa + (ulong)r * D41_A8_ROW : 0;           \
  __global ushort *xbr = xb ? xb + (ulong)r * D41_EMBD : 0;            \
  __global float *xfr = xf ? xf + (ulong)r * D41_EMBD : 0;

// ds_hc_init, fused with layer 0's attention collapse (pre = [1, 0, 0, 0]) and norm: X_c = e, x = RMSNorm_norm(e). e:
// the embedding rows, f32. ss[r] gets the row's sum of squares. Any of xa (a8), xb (bf16) and xf (f32) may be null.
// Groups W, D41_HC_SGS * 16.
REQD_SG __kernel void ds_hc_init(__global const float *e, __global const float *norm, __global float *X,
                                 __global float *ss, __global uchar *xa, __global ushort *xb, __global float *xf) {
  HC_LOCALS
  __global const float *er = e + (ulong)r * D41_EMBD;
  __global float *Xr = X + (ulong)r * D41_HC_DIM;
  float4 xs[HC_CHUNKS];
  float4 sq = (float4)0.0f;
  for (uint b = 0; b < HC_CHUNKS; b++) {
    const uint i = i0 + 64 * b;
    const float4 v = rd4(er + i);
    wr4(Xr + i, v);
    wr4(Xr + D41_EMBD + i, v);
    wr4(Xr + 2 * D41_EMBD + i, v);
    wr4(Xr + 3 * D41_EMBD + i, v);
    xs[b] = v;
    sq = mad(v, v, sq);
  }
  const float post = sg_sum(sum4(sq));
  if (lane == 0) red[0][s] = post;
  barrier(CLK_LOCAL_MEM_FENCE);
  const float total = posted_sum(red[0], D41_HC_SGS);
  const float rs = rsqrt(total / (float)D41_EMBD + D41_EPS);
  for (uint b = 0; b < HC_CHUNKS; b++) {
    const uint i = i0 + 64 * b;
    store_x(xs[b] * rs * rd4(norm + i), i, xar, xbr, xfr, 0);
  }
  if (s == 0 && lane == 0) ss[r] = total * (float)D41_HC;
}

// The 24 raw mixes of W rows: m[r][j] = (fn_j . X_r) / sqrt(ss[r] / D41_HC_DIM + D41_EPS), fn D41_MIX rows of
// D41_HC_DIM (hc_attn_fn or hc_ffn_fn). A work-group is a row of fn; its sub-groups split K, and all W rows share each
// weight read, sums through local memory. Groups D41_MIX, 16 ksplit with ksplit dividing 320 and at most 32.
REQD_SG __kernel void ds_hc_mix(__global const float *fn, __global const float *X, __global const float *ss, uint W,
                                __global float *m) {
  __local float red[STEP_MAX_ROWS][32];
  const uint j = get_group_id(0), s = sg_id(), lane = sg_lane();
  const uint nsg = get_local_size(0) / SG_SIZE, kc = D41_HC_DIM / nsg;
  __global const float *f = fn + (ulong)j * D41_HC_DIM + s * kc;
  __global const float *x = X + s * kc;
  float4 a0 = (float4)0.0f, a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
#define MIX_ROW(t) a##t = mad(w, rd4(xk + (t) * D41_HC_DIM), a##t);
  for (uint k = 0; k < kc; k += 64) {
    const float4 w = rd4(f + k);
    __global const float *xk = x + k;
    MIX_ROW(0)
    if (W > 1) {
      MIX_ROW(1)
      if (W > 2) {
        MIX_ROW(2)
        if (W > 3) {
          MIX_ROW(3)
          if (W > 4) {
            MIX_ROW(4)
            if (W > 5) {
              MIX_ROW(5)
              if (W > 6) {
                MIX_ROW(6)
                if (W > 7) MIX_ROW(7)
              }
            }
          }
        }
      }
    }
  }
#undef MIX_ROW
  float p[STEP_MAX_ROWS] = {sum4(a0), sum4(a1), sum4(a2), sum4(a3), sum4(a4), sum4(a5), sum4(a6), sum4(a7)};
  for (uint t = 0; t < W; t++) {
    const float v = sg_sum(p[t]);
    if (lane == 0) red[t][s] = v;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (s != 0) return;
  for (uint t = 0; t < W; t++) {
    const float v = posted_sum(red[t], nsg);
    if (lane == 0) m[(ulong)t * D41_MIX + j] = v * rsqrt(ss[t] / (float)D41_HC_DIM + D41_EPS);
  }
}

// A sublayer's mixes from its raw mixes: pre, post and the Sinkhorn-normalised comb (ref_ops.h, hc_split). m: [W][24];
// scale: 3 values; base: D41_MIX values. One sub-group a row, lane src * 4 + dst holding that entry of comb, so a
// row's sum is two swaps of lanes and a column's two others. Groups W, 16.
REQD_SG __kernel void ds_hc_sinkhorn(__global const float *m, __global const float *scale, __global const float *base,
                                     __global D41HcMix *mix) {
  const uint r = get_group_id(0), lane = sg_lane();
  __global const float *mr = m + (ulong)r * D41_MIX;
#define SUM_ROWS(v) ({ const float t_ = (v) + sg_xor((v), 1); t_ + sg_xor(t_, 2); })
#define SUM_COLS(v) ({ const float t_ = (v) + sg_xor((v), 4); t_ + sg_xor(t_, 8); })
  const float l = rd1(mr + 2 * D41_HC) * scale[2] + rd1(base + 2 * D41_HC);
  float mx = fmax(l, sg_xor(l, 1));
  mx = fmax(mx, sg_xor(mx, 2));
  const float e = exp(l - mx);
  float c = e / SUM_ROWS(e) + D41_HC_EPS;
  c /= SUM_COLS(c) + D41_HC_EPS;
  for (uint it = 1; it < D41_SINKHORN; it++) {
    c /= SUM_ROWS(c) + D41_HC_EPS;
    c /= SUM_COLS(c) + D41_HC_EPS;
  }
#undef SUM_ROWS
#undef SUM_COLS
  wr1((__global float *)mix[r].comb, c);
  const float p = sigmoid(rd1(mr) * scale[lane < D41_HC ? 0 : 1] + rd1(base));
  if (lane < D41_HC)
    mix[r].pre[lane] = p + D41_HC_EPS;
  else if (lane < 2 * D41_HC)
    mix[r].post[lane - D41_HC] = 2.0f * p;
}

// A sublayer's end: X'_dst = post[dst] y + sum_src comb[src][dst] X_src, y = y0 + y1, the cards' partial sums (y1 null
// with one card; a swap keeps the bits); y_host is nonzero when either lies in host memory. ss[r] gets X''s sum of
// squares. With norm set, the next input x = RMSNorm_norm(sum pre[c] X'_c) goes into any of xa (a8), xb (bf16), xf
// (f32) and job's x; norm null writes X' only. X and Xo must not overlap. Groups W, D41_HC_SGS * 16.
REQD_SG __kernel void ds_hc_post_norm(__global const float *X, __global const float *y0, __global const float *y1,
                                      uint y_host, __global const D41HcMix *mix, __global const float *norm,
                                      __global float *Xo, __global float *ss, __global uchar *xa, __global ushort *xb,
                                      __global float *xf, __global D41MissJob *job) {
  HC_LOCALS
  __global const float *Xr = X + (ulong)r * D41_HC_DIM;
  __global const float *y0r = y0 + (ulong)r * D41_EMBD;
  __global const float *y1r = y1 ? y1 + (ulong)r * D41_EMBD : 0;
  __global float *Or = Xo + (ulong)r * D41_HC_DIM;
  if (y_host) host_read_fence();
  const float4 post = vload4(0, (__global const float *)mix[r].post);
  const float4 pre = vload4(0, (__global const float *)mix[r].pre);
  const float16 c = vload16(0, (__global const float *)mix[r].comb);
  float4 xs[HC_CHUNKS];
  float4 sq = (float4)0.0f, xx = (float4)0.0f;
  for (uint b = 0; b < HC_CHUNKS; b++) {
    const uint i = i0 + 64 * b;
    // Loads come first, so that they are in flight together when y lies in host memory.
    const float4 x0 = rd4(Xr + i), x1 = rd4(Xr + D41_EMBD + i), x2 = rd4(Xr + 2 * D41_EMBD + i),
                 x3 = rd4(Xr + 3 * D41_EMBD + i);
    float4 y = rd4(y0r + i);
    if (y1) y += rd4(y1r + i);
    const float4 o0 = post.s0 * y + c.s0 * x0 + c.s4 * x1 + c.s8 * x2 + c.sc * x3;
    const float4 o1 = post.s1 * y + c.s1 * x0 + c.s5 * x1 + c.s9 * x2 + c.sd * x3;
    const float4 o2 = post.s2 * y + c.s2 * x0 + c.s6 * x1 + c.sa * x2 + c.se * x3;
    const float4 o3 = post.s3 * y + c.s3 * x0 + c.s7 * x1 + c.sb * x2 + c.sf * x3;
    wr4(Or + i, o0);
    wr4(Or + D41_EMBD + i, o1);
    wr4(Or + 2 * D41_EMBD + i, o2);
    wr4(Or + 3 * D41_EMBD + i, o3);
    sq = mad(o0, o0, sq);
    sq = mad(o1, o1, sq);
    sq = mad(o2, o2, sq);
    sq = mad(o3, o3, sq);
    if (norm) {
      xs[b] = pre.s0 * o0 + pre.s1 * o1 + pre.s2 * o2 + pre.s3 * o3;
      xx = mad(xs[b], xs[b], xx);
    }
  }
  const float psq = sg_sum(sum4(sq)), pxx = norm ? sg_sum(sum4(xx)) : 0.0f;
  if (lane == 0) {
    red[0][s] = psq;
    red[1][s] = pxx;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (norm) {
    const float rs = rsqrt(posted_sum(red[1], D41_HC_SGS) / (float)D41_EMBD + D41_EPS);
    for (uint b = 0; b < HC_CHUNKS; b++) {
      const uint i = i0 + 64 * b;
      store_x(xs[b] * rs * rd4(norm + i), i, xar, xbr, xfr, job ? job + r : 0);
    }
  }
  if (s == 0) {
    const float total = posted_sum(red[0], D41_HC_SGS);
    if (lane == 0) ss[r] = total;
  }
}

// Engram's gates for key streams [c0, c1), a card's share (ref_ops.h, engram_gate), and the send of its value rows
// [v0, v1). kv: rows kv_stride apart, each the keys of those streams, then those values. w: engram_gate's rows for the
// same streams. Each gate goes into own and, when peer is set, peer (the other card's memory or pinned host memory),
// which also gets the value rows; with no peer the card owns every stream and value. Groups (c1 - c0, W), 256.
REQD_SG __kernel void ds_engram_gate(__global const float *X, __global const float *kv, uint kv_stride,
                                     __global const float *w, uint c0, uint c1, uint v0, uint v1,
                                     __global D41EngramSend *own, __global D41EngramSend *peer) {
  __local float red[3][16];
  const uint g = get_group_id(0), r = get_group_id(1), s = sg_id(), lane = sg_lane(), c = c0 + g, n = c1 - c0;
  const uint i0 = s * (D41_EMBD / 16);
  __global const float *Xc = X + (ulong)r * D41_HC_DIM + c * D41_EMBD;
  __global const float *kr = kv + (ulong)r * kv_stride;
  __global const float *key = kr + g * D41_EMBD;
  __global const float *wc = w + g * D41_EMBD;
  float4 dt = (float4)0.0f, xx = dt, kk = dt;
  for (uint i = i0; i < i0 + D41_EMBD / 16; i += 64) {
    const float4 x = rd4(Xc + i), k = rd4(key + i), wv = rd4(wc + i);
    dt = mad(x * wv, k, dt);
    xx = mad(x, x, xx);
    kk = mad(k, k, kk);
  }
  const float pd = sg_sum(sum4(dt)), px = sg_sum(sum4(xx)), pk = sg_sum(sum4(kk));
  if (lane == 0) {
    red[0][s] = pd;
    red[1][s] = px;
    red[2][s] = pk;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (s == 0) {
    const float d = posted_sum(red[0], 16), sx = posted_sum(red[1], 16), sk = posted_sum(red[2], 16);
    const float rstd = rsqrt(sx / (float)D41_EMBD + D41_EPS) * rsqrt(sk / (float)D41_EMBD + D41_EPS);
    const float dot_c = d * rstd * rsqrt((float)D41_EMBD);
    const float gate = sigmoid(copysign(sqrt(fmax(fabs(dot_c), D41_ENG_CLAMP)), dot_c));
    wr1((__global float *)own[r].gate[c], gate);
    if (peer) wr1((__global float *)peer[r].gate[c], gate);
  }
  if (peer) {
    // A work-group is one stream of one row and sends its share of the value rows, in the 64-byte stores PCIe needs.
    // v1 - v0 is a multiple of 256 (c1 - c0), v0 of 16.
    const uint share = (v1 - v0) / n, seg = share / 16, at = g * share + s * seg;
    __global const float *src = kr + n * D41_EMBD + at;
    __global float *dst = (__global float *)peer[r].value + v0 + at;
    for (uint q = 0; q < seg; q += 16) wr1(dst + q, rd1(src + q));
  }
}

// Engram's injection on the 4 streams in place, X_c += gate_c value, then the attention input from the changed X:
// x = RMSNorm_norm(sum pre[c] X_c) with the previous layer's pre, into any of xa, xb and xf; ss[r] gets X's sum of
// squares. Gates of the streams [c0, c1) and values [v0, v1) are this card's (own, and kv as ds_engram_gate reads it),
// the others in peer (peer_host nonzero when that is host memory). v0, v1: multiples of 64. Groups W, D41_HC_SGS * 16.
REQD_SG __kernel void ds_engram_apply(__global float *X, __global const float *kv, uint kv_stride, uint c0, uint c1,
                                      uint v0, uint v1, __global const D41EngramSend *own,
                                      __global const D41EngramSend *peer, uint peer_host,
                                      __global const D41HcMix *mix, __global const float *norm, __global float *ss,
                                      __global uchar *xa, __global ushort *xb, __global float *xf) {
  HC_LOCALS
  __global float *Xr = X + (ulong)r * D41_HC_DIM;
  __global const float *vr = kv + (ulong)r * kv_stride + (c1 - c0) * D41_EMBD;
  __global const float *pv = peer ? (__global const float *)peer[r].value : 0;
  if (peer_host) host_read_fence();
#define GATE(c) ((c) >= c0 && (c) < c1 ? own[r].gate[c][0] : peer[r].gate[c][0])
  const float g0 = GATE(0), g1 = GATE(1), g2 = GATE(2), g3 = GATE(3);
#undef GATE
  const float4 pre = vload4(0, (__global const float *)mix[r].pre);
  float4 xs[HC_CHUNKS];
  float4 sq = (float4)0.0f, xx = (float4)0.0f;
  for (uint b = 0; b < HC_CHUNKS; b++) {
    const uint i = i0 + 64 * b;
    const float4 v = rd4(i >= v0 && i < v1 ? vr + (i - v0) : pv + i);
    const float4 x0 = rd4(Xr + i) + g0 * v, x1 = rd4(Xr + D41_EMBD + i) + g1 * v,
                 x2 = rd4(Xr + 2 * D41_EMBD + i) + g2 * v, x3 = rd4(Xr + 3 * D41_EMBD + i) + g3 * v;
    wr4(Xr + i, x0);
    wr4(Xr + D41_EMBD + i, x1);
    wr4(Xr + 2 * D41_EMBD + i, x2);
    wr4(Xr + 3 * D41_EMBD + i, x3);
    sq = mad(x0, x0, sq);
    sq = mad(x1, x1, sq);
    sq = mad(x2, x2, sq);
    sq = mad(x3, x3, sq);
    xs[b] = pre.s0 * x0 + pre.s1 * x1 + pre.s2 * x2 + pre.s3 * x3;
    xx = mad(xs[b], xs[b], xx);
  }
  const float psq = sg_sum(sum4(sq)), pxx = sg_sum(sum4(xx));
  if (lane == 0) {
    red[0][s] = psq;
    red[1][s] = pxx;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const float rs = rsqrt(posted_sum(red[1], D41_HC_SGS) / (float)D41_EMBD + D41_EPS);
  for (uint b = 0; b < HC_CHUNKS; b++) {
    const uint i = i0 + 64 * b;
    store_x(xs[b] * rs * rd4(norm + i), i, xar, xbr, xfr, 0);
  }
  if (s == 0) {
    const float total = posted_sum(red[0], D41_HC_SGS);
    if (lane == 0) ss[r] = total;
  }
}

// s = sqrt(softplus(logit)); softplus is the logit itself above 20, as torch and ggml take it.
static inline float rt_score(float x) { return sqrt(x > 20.0f ? x : log1p(exp(x))); }

// A MissJob header's three stores of 64 bytes a sub-group: first and second, then third over first's bytes, behind a
// release fence when fence is set. ds_job_store_rate times these same stores.
static inline void rt_job_stores(__global uint *h, uint first, uint second, uint third, bool fence) {
  intel_sub_group_block_write2(h, (uint2)(first, second));
  if (fence) atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_release, memory_scope_all_svm_devices);
  intel_sub_group_block_write(h, third);
}

// A row's MissJob header, by its sub-group (lane k < D41_USED holds rank k's id and weight, with miss set when it is
// not cached), in two 64-byte stores across PCIe, then the step behind a release fence in a third store of the same
// first 64 bytes, whose other words do not change. Unused entries hold -1 and 0. A padded row has id -1 and no miss.
static inline void rt_job(__global D41MissJob *job, uint n, int id, float wk, uint miss, uint step) {
  const uint lane = sg_lane();
  uint first = lane == 0 ? D41_NO_STEP : lane == 1 ? n : lane < 8 ? 0xffffffffu : 0u;
  uint second = 0;
  uint at = 0;
  for (uint k = 0; k < D41_USED; k++) {
    const uint idk = (uint)sub_group_broadcast(id, k);
    const uint wbits = as_uint(sub_group_broadcast(wk, k));
    if (sub_group_broadcast(miss, k)) {
      if (lane == 2 + at) first = idk;
      if (lane == 8 + at) first = wbits;
      at++;
    }
    // job->all[k]: words 14 and 15 in the first store, the others in the second.
    if (lane == 14 + k) first = idk;
    if (lane + 2 == k) second = idk;
  }
  rt_job_stores((__global uint *)job, first, second, lane == 0 ? step : first, true);
}

// The step's cached experts as D41Group, by sub-groups 0, 1, 2, lane l of sub-group j holding assignment 16 j + l. sl:
// the n assignments' slot pairs, -1 for none. Groups are numbered by their first assignment and members ordered, so the
// result does not depend on scheduling. Pair k is class flip with slot k below split, class 1 - flip with k - split
// above. Every work-item of the work-group calls this; heads is local memory.
static inline void rt_group(__local const int *sl, uint n, uint split, uint flip, __global D41Group *groups,
                            __global uint *ngroups, __local ulong *heads) {
  const uint j = sg_id(), lane = sg_lane(), banks = (n + 15) / 16;
  ulong mask = 0;
  int s = -1;
  uint a = 0, leader = 0, rank = 0, count = 0;
  if (j < banks) {
    const int v0 = (int)intel_sub_group_block_read((__local const uint *)sl);
    const int v1 = n > 16 ? (int)intel_sub_group_block_read((__local const uint *)sl + 16) : -1;
    const int v2 = n > 32 ? (int)intel_sub_group_block_read((__local const uint *)sl + 32) : -1;
    a = 16 * j + lane;
    s = a < n ? (j == 0 ? v0 : j == 1 ? v1 : v2) : -1;
    leader = a;
    for (uint b = 0; b < n; b++) {
      const int sb = sub_group_broadcast(b < 16 ? v0 : b < 32 ? v1 : v2, b & 15);
      const uint match = sb == s;
      leader = match ? min(leader, b) : leader;
      rank += match && b < a;
      count += match;
    }
    // Each lane holds its own bit, so the sum is the sub-group's bits.
    mask = sub_group_reduce_add(s >= 0 && leader == a ? 1ul << a : 0ul);
    if (lane == 0 && banks > 1) heads[j] = mask;
  }
  if (banks > 1) {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (j >= banks) return;
    mask = heads[0] | heads[1] | (banks > 2 ? heads[2] : 0ul);
  } else if (j >= banks) {
    return;
  }
  if (s >= 0) {
    __global D41Group *g = groups + popcount(mask & ((1ul << leader) - 1));
    g->asg[rank] = a;
    if (leader == a) {
      const uint upper = (uint)s >= split;
      g->slot = s - (upper ? (int)split : 0);
      g->n = count;
      g->cls = upper ^ flip;
    }
  }
  if (j == 0 && lane == 0) *ngroups = (uint)popcount(mask);
}

// The router and its grouping (rt_group): a row's sub-groups score its experts' columns in turn, its first takes the
// top 6 of s + bias, ties to the lower expert, w = s / (sum s + D41_EPS) D41_ROUTE_SCALE. Past
// the plan's *real a row has ids and slots -1, w and nmiss 0. slot_map: each expert's slot pair, negative when not
// cached; jobs: [W], 64 B aligned, null on a card that does not post. Groups 1, 16 W D41_RT_SGS(W).
REQD_SG __kernel void ds_router(__global const float *logits, __global const float *bias,
                                __global const int *slot_map, uint W, uint split, uint flip,
                                __global const uint *real, __global const uint *step, __global int *ids,
                                __global float *w, __global int *slots, __global uint *nmiss,
                                __global D41MissJob *jobs, __global D41Group *groups, __global uint *ngroups) {
  __local uint keys[STEP_MAX_ROWS][D41_EXPERTS];
  __local uint2 cand[STEP_MAX_ROWS * RT_SLM];
  __local int sl[STEP_MAX_ROWS * D41_USED];
  __local ulong heads[3];
  const uint sgs = get_num_sub_groups() / W, sg = sg_id(), r = sg / sgs, part = sg % sgs, lane = sg_lane();
  const uint nreal = *real;
  __global const float *lr = logits + (ulong)r * D41_EXPERTS;
  if (r < nreal)
    for (uint j = part; j < RT_PER; j += sgs) {
      const uint k = topk_key(rt_score(rd1(lr + 16 * j)) + rd1(bias + 16 * j));
      intel_sub_group_block_write((__local uint *)&keys[r][16 * j], k);
    }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (part == 0) {
    int id = -1, slot = -1;
    float wk = 0.0f;
    if (r < nreal) {
      uint key[RT_PER];
      for (uint j = 0; j < RT_PER; j += 4) {
        const uint4 k = intel_sub_group_block_read4((__local const uint *)&keys[r][16 * j]);
        key[j] = k.s0, key[j + 1] = k.s1, key[j + 2] = k.s2, key[j + 3] = k.s3;
      }
      id = (int)rt_top(key, cand + r * RT_SLM).y;
      const float sk = rt_score(lr[id]);
      float tot = 0.0f;
      for (uint k = 0; k < D41_USED; k++) tot += sub_group_broadcast(sk, k);
      wk = sk / (tot + D41_EPS) * D41_ROUTE_SCALE;
      slot = slot_map[id];
    }
    const uint miss = lane < D41_USED && r < nreal && slot < 0;
    const uint n = sub_group_reduce_add(miss);
    if (lane < D41_USED) {
      const uint a = r * D41_USED + lane;
      ids[a] = id;
      w[a] = wk;
      slots[a] = slot < 0 ? -1 : slot;
      sl[a] = slot < 0 ? -1 : slot;
    }
    if (lane == 0) nmiss[r] = n;
    if (jobs) rt_job(jobs + r, n, id, wk, miss, *step);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  rt_group(sl, W * D41_USED, split, flip, groups, ngroups, heads);
}

// The shared expert's SwiGLU with the clamp, stored only as a8 (a8.h) into ha, rows ff + ff / 8 bytes apart:
// silu(min(g, limit)) clamp(u, -limit, limit). g, u: ff values a row, ff the card's rows, a multiple of 64, rows stride
// values apart: ff in two buffers, 2 ff with u = g + ff when one matvec launch wrote gate and up as one matrix. Groups
// W, D41_SWIGLU_SGS * 16; sub-groups take the chunks of 64 in turn.
REQD_SG __kernel void ds_swiglu_clamp_a8(__global const float *g, __global const float *u, uint stride, uint ff,
                                         __global uchar *ha) {
  const uint r = get_group_id(0), sgs = get_num_sub_groups();
  __global const float *gr = g + (ulong)r * stride, *ur = u + (ulong)r * stride;
  __global uchar *a = ha + (ulong)r * (ff + ff / 8);
  for (uint i = sg_id() * 64; i < ff; i += sgs * 64) {
    const float4 gc = fmin(rd4(gr + i), (float4)D41_SWIGLU_LIMIT);
    const float4 uc = clamp(rd4(ur + i), (float4)-D41_SWIGLU_LIMIT, (float4)D41_SWIGLU_LIMIT);
    const float4 h = gc / (1.0f + exp(-gc)) * uc;
    a8_store_block(h.lo, a, ff, i / 32);
    a8_store_block(h.hi, a, ff, i / 32 + 1);
  }
}

// A card's partial sum: its cached experts in rank order (y: [W][D41_USED][D41_EMBD], w and slots the router's), then
// shared, then cpu, the CPU's answer in host memory or null, for a row with misses: an acquire fence, then the 16-byte
// loads that read host memory well, together in flight. Into out and, when not null, peer (the other card's memory or
// pinned host memory), 256 bytes a sub-group. Groups W, sub-groups * 16; sub-groups divide D41_EMBD / 64 / CMB_BATCH.
#define CMB_BATCH 5
static inline void moe_combine(__global const float *y, __global const float *w, __global const int *slots,
                               __global const float *shared, __global const float *cpu, __global const uint *nmiss,
                               __global float *out, __global float *peer, bool fence) {
  const uint r = get_group_id(0), sgs = get_num_sub_groups(), lane = sg_lane();
  const uint a = r * D41_USED;
  const float3 w0 = vload3(0, w + a), w1 = vload3(1, w + a);
  const int3 s0 = vload3(0, slots + a), s1 = vload3(1, slots + a);
  const float wk[D41_USED] = {sub_group_broadcast(w0.s0, 0), sub_group_broadcast(w0.s1, 0),
                              sub_group_broadcast(w0.s2, 0), sub_group_broadcast(w1.s0, 0),
                              sub_group_broadcast(w1.s1, 0), sub_group_broadcast(w1.s2, 0)};
  const int on[D41_USED] = {sub_group_broadcast(s0.s0, 0), sub_group_broadcast(s0.s1, 0),
                            sub_group_broadcast(s0.s2, 0), sub_group_broadcast(s1.s0, 0),
                            sub_group_broadcast(s1.s1, 0), sub_group_broadcast(s1.s2, 0)};
  const uint host = cpu != 0 && sub_group_broadcast(nmiss[r], 0) != 0;
  if (host) {
    // One fence a row: it waits on the card's whole cache, so one a sub-group outlasts the loads
    // (docs/measurements/MEASUREMENTS.md#2026-11-02-deepseek41-router).
    if (fence && sg_id() == 0) host_read_fence();
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  __global const float *yr = y + (ulong)a * D41_EMBD;
  __global const float *sr = shared + (ulong)r * D41_EMBD;
  __global const float *cr = cpu ? cpu + (ulong)r * D41_EMBD : 0;
  __global float *ot = out + (ulong)r * D41_EMBD;
  __global float *pr = peer ? peer + (ulong)r * D41_EMBD : 0;
  for (uint c = sg_id(); c < D41_EMBD / 64; c += CMB_BATCH * sgs) {
    // A host load takes a PCIe round trip, so a sub-group's loads go out before any sum needs one.
    float4 from_cpu[CMB_BATCH];
    if (host)
      for (uint b = 0; b < CMB_BATCH; b++) from_cpu[b] = vload4((c + b * sgs) * 16 + lane, cr);
    for (uint b = 0; b < CMB_BATCH; b++) {
      const uint q = (c + b * sgs) * 16 + lane;  // the chunk's 16-byte unit for this lane
      float4 acc = (float4)0.0f;
      for (uint k = 0; k < D41_USED; k++)
        if (on[k] >= 0) acc += wk[k] * vload4(q, yr + k * D41_EMBD);
      acc += vload4(q, sr);
      if (host) acc += from_cpu[b];
      vstore4(acc, q, ot);
      if (peer) vstore4(acc, q, pr);
    }
  }
}
REQD_SG __kernel void ds_moe_combine(__global const float *y, __global const float *w, __global const int *slots,
                                     __global const float *shared, __global const float *cpu,
                                     __global const uint *nmiss, __global float *out, __global float *peer) {
  moe_combine(y, w, slots, shared, cpu, nmiss, out, peer, true);
}

// A card's send of its partial sums: y [W][D41_EMBD] in its memory into the same rows of peer, the other card's memory
// or pinned host memory, in 256-byte stores, as every store across PCIe is 64 bytes or wider. Even and odd exchanges use
// different peer buffers: the other card may still be reading the last one
// (docs/measurements/MEASUREMENTS.md#2026-11-02-deepseek41-exchange). Groups W, sub-groups * 16; sub-groups divide 80.
REQD_SG __kernel void ds_exch_send(__global const float *y, __global float *peer) {
  const uint n = D41_EMBD / get_num_sub_groups(), at = get_group_id(0) * D41_EMBD + sg_id() * n;
  for (uint i = at; i < at + n; i += 64) wr4(peer + i, rd4(y + i));
}

// A list's wait (ADR-0011, ADR-0062 item 5) on the miss answer, the other card's exchange flag or the Engram flag: one
// work-item polls flag until it holds *want, and sets *stuck past D41_WAIT_TICKS in this wait or D41_LIST_TICKS since the
// list's first polling wait stamped *start, which the plan's copy zeroes. A wait that finds *stuck set does not poll.
// Pointers other than want, stuck and start may be null. stamp gets the clock at entry and exit. Group 1.
REQD_SG __kernel void ds_wait(__global uint *raise, __global const uint *flush, __global const uint *flag,
                              __global const uint *want, __global const uint *nmiss, uint W, __global uint *stuck,
                              __global ulong *start, __global ulong *stamp) {
  if (stamp) stamp[0] = clock_read_device();
  const uint step = *want;
  // The card's flag to the other card. flush is a word of the other card's memory: its read arrives after the stores
  // before it on that path.
  if (raise) {
    if (flush) {
      host_read_fence();
      (void)*(volatile __global const uint *)flush;
    }
    atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_release, memory_scope_all_svm_devices);
    atomic_store_explicit((volatile __global atomic_uint *)raise, step, memory_order_release,
                          memory_scope_all_svm_devices);
  }
  bool poll = flag != 0 && *stuck == 0;
  // No row of W missed: the CPU has no answer to give.
  if (poll && nmiss) {
    uint any = 0;
    for (uint r = 0; r < W; r++) any |= nmiss[r];
    poll = any != 0;
  }
  if (poll) {
    const ulong t0 = clock_read_device();
    if (*start == 0) *start = t0;
    const ulong end = min(t0 + D41_WAIT_TICKS, *start + D41_LIST_TICKS);
    bool seen = false;
    for (uint i = 0; i < D41_WAIT_POLLS && !seen; i++) {
      host_read_fence();
      seen = *(volatile __global const uint *)flag == step;
      if (!seen && clock_read_device() >= end) break;
    }
    if (!seen) *stuck = 1;
  }
  if (stamp) stamp[1] = clock_read_device();
}

// The kernels below are ds41_router_test's controls, which no model list runs.

// n headers 128 bytes apart into h, sub-groups taking them in turn, by rt_job's stores (kind 0), one word a lane (1) or
// two bytes a lane (2). No fence, so that the time is the stores'. Groups 1, sub-groups * 16.
REQD_SG __kernel void ds_job_store_rate(__global uint *h, uint n, uint kind) {
  const uint lane = sg_lane(), sgs = get_num_sub_groups();
  for (uint i = sg_id(); i < n; i += sgs) {
    __global uint *p = h + (ulong)i * 32;
    if (kind == 2) {
      __global ushort *b = (__global ushort *)p;
      b[lane] = (ushort)lane;
      b[32 + lane] = (ushort)lane;
      b[16 + lane] = (ushort)lane;
      b[48 + lane] = (ushort)lane;
      b[lane] = (ushort)(lane + 1);
    } else if (kind == 1) {
      p[lane] = lane;
      p[16 + lane] = lane;
      p[lane] = lane + 1;
    } else {
      rt_job_stores(p, lane, lane, lane + 1, false);
    }
  }
}

// ds_moe_combine without its fence, to show what the fence does.
REQD_SG __kernel void ds_moe_combine_no_fence(__global const float *y, __global const float *w,
                                              __global const int *slots, __global const float *shared,
                                              __global const float *cpu, __global const uint *nmiss,
                                              __global float *out, __global float *peer) {
  moe_combine(y, w, slots, shared, cpu, nmiss, out, peer, false);
}

// ds_moe_combine's reads of the CPU's answer, then each row holds for ticks of the device clock with no fence, then the
// combine, in one launch, with no kernel's end between the reads. stamp[0] gets the clock when row 0 starts to hold.
REQD_SG __kernel void ds_moe_combine_read_twice(__global const float *y, __global const float *w,
                                                __global const int *slots, __global const float *shared,
                                                __global const float *cpu, __global const uint *nmiss,
                                                __global float *out, uint ticks, __global ulong *stamp, uint fence) {
  const uint r = get_group_id(0), sgs = get_num_sub_groups(), lane = sg_lane();
  __global const float *cr = cpu + (ulong)r * D41_EMBD;
  __global float *ot = out + (ulong)r * D41_EMBD;
  // The store keeps the reads.
  for (uint c = sg_id(); c < D41_EMBD / 64; c += sgs) vstore4(vload4(c * 16 + lane, cr), c * 16 + lane, ot);
  barrier(CLK_LOCAL_MEM_FENCE);
  if (sg_id() == 0 && lane == 0) {
    const ulong t0 = clock_read_device();
    if (r == 0) stamp[0] = t0;
    while (clock_read_device() - t0 < ticks) {
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  moe_combine(y, w, slots, shared, cpu, nmiss, out, 0, fence != 0);
}

// Holds a list for ticks of the device clock. Unlike ds_wait it runs no fence. stamp[0] gets the clock at entry. Group
// 1.
REQD_SG __kernel void ds_hold(uint ticks, __global ulong *stamp) {
  const ulong t0 = clock_read_device();
  stamp[0] = t0;
  while (clock_read_device() - t0 < ticks) {
  }
}
