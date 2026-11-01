// deepseek41's decode-step kernels between its products: hyper-connections and Engram. The maths is the deepseek41
// forward pass; core/models/deepseek41/ref_ops.h holds one scalar definition of each. A row is one step's token: X is
// [rows][D41_HC][D41_EMBD] f32, ss one sum of squares of a row's D41_HC_DIM values. Row counts are the recorded width,
// passed by value, so these kernels read no plan and a padded row costs kernel time only.
#include "a8.h"
#include "builtins.h"
#include "layout.h"

static inline float sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }

// Sub-group block moves of f32 values. Lane i holds p[i], or p[16 j + i] as element j. Reads need 4-byte alignment,
// writes 16-byte.
static inline float rd1(__global const float *p) {
  return as_float(intel_sub_group_block_read((__global const uint *)p));
}
static inline float4 rd4(__global const float *p) {
  return as_float4(intel_sub_group_block_read4((__global const uint *)p));
}
static inline void wr1(__global float *p, float v) { intel_sub_group_block_write((__global uint *)p, as_uint(v)); }
static inline void wr4(__global float *p, float4 v) { intel_sub_group_block_write4((__global uint *)p, as_uint4(v)); }

static inline float sum4(float4 v) { return (v.s0 + v.s1) + (v.s2 + v.s3); }

// Comes before reading what another agent wrote to host memory, as the card caches host-memory lines (ADR-0011). It
// invalidates the card's cache, so a launch that reads only card memory leaves it out.
static inline void host_read_fence(void) {
  atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_acquire, memory_scope_all_svm_devices);
}

// The total of the sub-groups' posted sums, after the barrier: lanes read the posts together, in a fixed order.
static inline float posted_sum(__local const float *red, uint nsg) {
  float t = 0.0f;
  for (uint i = sg_lane(); i < nsg; i += SG_SIZE) t += red[i];
  return sg_sum(t);
}

// a8 unit u of the row at a (a8.h's rule), from 32 values held as element j of lane i = value 16 j + i.
static inline void a8_store_block(float2 v, __global uchar *a, uint u) {
  const float amax = sub_group_reduce_max(fmax(fabs(v.x), fabs(v.y)));
  const float d = amax / 127.0f;
  const float id = d != 0.0f ? 1.0f / d : 0.0f;
  intel_sub_group_block_write_uc2(a + 32 * u, as_uchar2(convert_char2(round(v * id))));
  if (sg_lane() == 0) ((__global float *)(a + D41_EMBD))[u] = d;
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
    a8_store_block(x.lo, xa, i / 32);
    a8_store_block(x.hi, xa, i / 32 + 1);
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
