// deepseek41's decode-step kernels around attention: the q and KV norms, the window ring, the compressor, the indexer
// and the attention. The maths is the deepseek41 forward pass; core/models/deepseek41/ref_ops.h holds one scalar definition of
// each. Caches are f16, block-written, indexed by position, so a step can run again and a rejected draft overwrites
// only its own cells.

// An angle's bits must equal ggml's: at position 16,383 one unit in the last place of float is 1e-3 radians. Above the
// includes, since a pragma covers only what follows it and rope.h's d41_yarn adds two products.
#pragma OPENCL FP_CONTRACT OFF

#include "a8.h"
#include "builtins.h"
#include "layout.h"
#include "ops.h"
#include "rope.h"
#include "topk.h"

// The raw angle (d41_rope_theta before YaRN) of pair pair0 + lane / 2: the same chain of float products as ggml's.
static inline float rope_chain(uint pos, uint pair0, uint yarn) {
  const float step = yarn ? D41_ROPE_STEP_C : D41_ROPE_STEP;
  float t = (float)pos;
  for (uint k = 0; k < pair0; k++) t *= step;
  const uint n = sg_lane() / 2;
  for (uint k = 0; k < n; k++) t *= step;
  return t;
}

// The next element's raw angles: 16 values on, so 8 pairs on.
static inline float rope_next8(float t, uint yarn) {
  const float step = yarn ? D41_ROPE_STEP_C : D41_ROPE_STEP;
  for (uint k = 0; k < 8; k++) t *= step;
  return t;
}

// 16 rotated values, lane l holding the value of pair pair0 + l / 2 (even lane the pair's first), t its raw angle.
static inline float rope_16(float v, float t, uint pair0, uint yarn) {
  const uint lane = sg_lane();
  const float th = yarn ? d41_yarn(t, pair0 + lane / 2) : t;
  float c;
  const float sn = sincos(th, &c);
  const float o = sg_xor(v, 1);
  return mad(v, c, (lane & 1) ? o * sn : -o * sn);
}

static inline float2 rope_2(float2 v, uint pos, uint pair0, uint yarn) {
  const float t0 = rope_chain(pos, pair0, yarn);
  return (float2)(rope_16(v.s0, t0, pair0, yarn), rope_16(v.s1, rope_next8(t0, yarn), pair0 + 8, yarn));
}

static inline float4 rope_4(float4 v, uint pos, uint yarn) {
  return (float4)(rope_2(v.lo, pos, 0, yarn), rope_2(v.hi, pos, 16, yarn));
}

// One row's qr or kv: qr (group 0) goes to a8 in qa and, when set, bf16 in qb; kv (group 1) is rotated at the row's
// position (yarn: the layer has compressed rows) and goes into ring [slot][D41_RING][D41_HD] at the row's cell. qf and
// kvf, when set, get the f32 values, for the tests. Inputs and outputs are rows D41_QLR or D41_HD apart. Groups
// (2, W), D41_QKV_SGS * 16.
REQD_SG __kernel void ds_qkv_norm(__global const float *qr, __global const float *q_norm, __global const float *kv,
                                  __global const float *kv_norm, __global const D41Row *rows, uint yarn,
                                  __global uchar *qa, __global ushort *qb, __global float *qf, __global ushort *ring,
                                  __global float *kvf) {
  __local float red[D41_QKV_SGS];
  const uint r = get_group_id(1), s = sg_id(), lane = sg_lane();
  if (get_group_id(0) == 0) {
    const uint i0 = s * (D41_QLR / D41_QKV_SGS);
    __global const float *in = qr + (ulong)r * D41_QLR;
    float4 xs[D41_QLR / D41_QKV_SGS / 64];
    float4 sq = (float4)0.0f;
    for (uint b = 0; b < D41_QLR / D41_QKV_SGS / 64; b++) {
      xs[b] = rd4(in + i0 + 64 * b);
      sq = mad(xs[b], xs[b], sq);
    }
    const float post = sg_sum(sum4(sq));
    if (lane == 0) red[s] = post;
    barrier(CLK_LOCAL_MEM_FENCE);
    const float rs = rsqrt(posted_sum(red, D41_QKV_SGS) / (float)D41_QLR + D41_EPS);
    __global uchar *qar = qa + (ulong)r * D41_QR_A8_ROW;
    for (uint b = 0; b < D41_QLR / D41_QKV_SGS / 64; b++) {
      const uint i = i0 + 64 * b;
      const float4 y = xs[b] * rs * rd4(q_norm + i);
      a8_store_block(y.lo, qar, D41_QLR, i / 32);
      a8_store_block(y.hi, qar, D41_QLR, i / 32 + 1);
      if (qb)
        intel_sub_group_block_write_us4(qb + (ulong)r * D41_QLR + i,
                                        (ushort4)(f32_to_bf16(y.s0), f32_to_bf16(y.s1), f32_to_bf16(y.s2),
                                                  f32_to_bf16(y.s3)));
      if (qf) wr4(qf + (ulong)r * D41_QLR + i, y);
    }
  } else {
    const uint i0 = s * (D41_HD / D41_QKV_SGS);
    __global const float *in = kv + (ulong)r * D41_HD;
    const float4 x0 = rd4(in + i0), x1 = rd4(in + i0 + 64);
    const float4 sq = mad(x1, x1, x0 * x0);
    const float post = sg_sum(sum4(sq));
    if (lane == 0) red[s] = post;
    barrier(CLK_LOCAL_MEM_FENCE);
    const float rs = rsqrt(posted_sum(red, D41_QKV_SGS) / (float)D41_HD + D41_EPS);
    const uint slot = rows[r].slot, pos = rows[r].pos;
    const float4 y0 = x0 * rs * rd4(kv_norm + i0);
    float4 y1 = x1 * rs * rd4(kv_norm + i0 + 64);
    if (i0 + 64 == D41_NOPE) y1 = rope_4(y1, pos, yarn);
    if (slot != D41_NO_ROW) {
      __global ushort *cell = ring + ((ulong)slot * D41_RING + rows[r].ring) * D41_HD + i0;
      intel_sub_group_block_write_us4(cell, as_ushort4(convert_half4(y0)));
      intel_sub_group_block_write_us4(cell + 64, as_ushort4(convert_half4(y1)));
    }
    if (kvf) {
      wr4(kvf + (ulong)r * D41_HD + i0, y0);
      wr4(kvf + (ulong)r * D41_HD + i0 + 64, y1);
    }
  }
}

#define COMP_SGS 16
#define COMP_VALUES (D41_HD / COMP_SGS)     // latent values a sub-group: 32
#define COMP_KROWS (D41_IDX_D / COMP_SGS)   // rows of attn_k a sub-group: 8

// A KV source's compressor. Ratio 2: an even position keeps kv' and sc in cr's cell at position % D41_CRING; an odd
// one completes the group from row rows[r].prev of this step or else from cr. Ratio 1: every row is a group; sc and cr
// are not read. A group's row, [rotated latent | index key] f16, goes to pool at d41_comp_index. rowf, when set, gets
// rows of D41_COMP_ROWF f32. kv, sc: rows of D41_HD f32. attn_k: D41_IDX_D rows of D41_HD bf16. Groups W, 256.
REQD_SG __kernel void ds_compress(__global const float *kv, __global const float *sc, __global float *cr,
                                  __global const D41Row *rows, uint ratio, __global const float *norm,
                                  __global const ushort *attn_k, __global const float *k_norm,
                                  __global const uint *pt, uint npp, __global ushort *pool, __global float *rowf) {
  __local float red[COMP_SGS];
  __local float __attribute__((aligned(64))) lat[D41_HD];
  __local float __attribute__((aligned(64))) ikl[D41_IDX_D];
  const uint r = get_group_id(0), s = sg_id(), lane = sg_lane();
  const uint slot = rows[r].slot, pos = rows[r].pos;
  if (slot == D41_NO_ROW) return;
  const uint i0 = s * COMP_VALUES;
  float2 x = rd2(kv + (ulong)r * D41_HD + i0);
  if (ratio == 2) {
    const float2 sc1 = rd2(sc + (ulong)r * D41_HD + i0);
    __global float *cells = cr + (ulong)slot * D41_CRING * D41_CRING_CELL;
    if (!(pos & 1)) {
      __global float *cell = cells + (pos % D41_CRING) * D41_CRING_CELL + i0;
      wr2(cell, x);
      wr2(cell + D41_HD, sc1);
      return;
    }
    const uint prev = rows[r].prev;
    __global const float *cell0 = cells + ((pos - 1) % D41_CRING) * D41_CRING_CELL + i0;
    const float2 kv0 = rd2(prev != D41_NO_ROW ? kv + (ulong)prev * D41_HD + i0 : cell0);
    const float2 sc0 = rd2(prev != D41_NO_ROW ? sc + (ulong)prev * D41_HD + i0 : cell0 + D41_HD);
    // The per-channel softmax over two tokens: the larger score's weight is 1, the other's e.
    const float2 e = exp(-fabs(sc1 - sc0));
    const int2 one_wins = sc1 >= sc0;
    const float2 big = select(kv0, x, one_wins), small = select(x, kv0, one_wins);
    x = (big + e * small) / (1.0f + e);
  }
  const float post = sg_sum(x.s0 * x.s0 + x.s1 * x.s1);
  if (lane == 0) red[s] = post;
  barrier(CLK_LOCAL_MEM_FENCE);

  const uint at = ratio == 2 ? pos & ~1u : pos;  // the group's first position, where the row is rotated
  __global ushort *row = pool + d41_comp_index(d41_page(pt, npp, slot, at), at / ratio, ratio) * D41_COMP_ROW;
  __global float *rf = rowf ? rowf + (ulong)r * D41_COMP_ROWF : 0;
  {
    const float rs = rsqrt(posted_sum(red, COMP_SGS) / (float)D41_HD + D41_EPS);
    const float2 y = x * rs * rd2(norm + i0);
    // attn_k is BF16, so ggml's product converts its input to bf16; the row keeps the f32 values.
    intel_sub_group_block_write2((__local uint *)(lat + i0),
                                 (uint2)((uint)f32_to_bf16(y.s0), (uint)f32_to_bf16(y.s1)) << 16);
    const float2 o = i0 >= D41_NOPE ? rope_2(y, at, (i0 - D41_NOPE) / 2, 1) : y;
    intel_sub_group_block_write_us2(row + i0, as_ushort2(convert_half2(o)));
    if (rf) {
      wr2(rf + i0, o);
      wr2(rf + D41_COMP_ROW + i0, y);
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // attn_k . latent, rows split among the sub-groups, the weights in file order and the bf16 latent from local memory.
  __global const ushort *ak = attn_k + (ulong)(s * COMP_KROWS) * D41_HD;
  float4 acc[COMP_KROWS];
  for (uint i = 0; i < COMP_KROWS; i++) acc[i] = (float4)0.0f;
  for (uint c = 0; c < D41_HD; c += 64) {
    const float4 l = as_float4(intel_sub_group_block_read4((__local const uint *)(lat + c)));
    for (uint i = 0; i < COMP_KROWS; i++) {
      const ushort4 w = intel_sub_group_block_read_us4(ak + (ulong)i * D41_HD + c);
      acc[i] = mad(as_float4(convert_uint4(w) << 16), l, acc[i]);
    }
  }
  for (uint i = 0; i < COMP_KROWS; i++) {
    const float v = sg_sum(sum4(acc[i]));
    if (lane == 0) ikl[s * COMP_KROWS + i] = v;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (s != 0) return;
  float8 ik = as_float8(intel_sub_group_block_read8((__local const uint *)ikl));
  const float ss = sg_sum(dot(ik.lo, ik.lo) + dot(ik.hi, ik.hi));
  ik *= rsqrt(ss / (float)D41_IDX_D + D41_EPS) * rd8(k_norm);
  ik.hi = rope_4(ik.hi, at, 1);
  intel_sub_group_block_write_us8(row + D41_HD, as_ushort8(convert_half8(ik)));
  if (rf) wr8(rf + D41_HD, ik);
}

#define IDX_CHUNK SG_SIZE                                  // dimensions of a head that one pass over the heads takes
#define IDX_CHUNKS (D41_IDX_D / IDX_CHUNK)                 // 8
#define IDX_SGS (D41_IDX_WG_ROWS / SG_SIZE)                // 4
#define IDX_FIRST_ROT ((D41_IDX_D - D41_ROT) / IDX_CHUNK)  // the first chunk that rotates

// A chunk's value in lane l, rotated by its pair's cosine and sine, the sine negated in even lanes.
static inline float idx_rot(float v, float cs, float ss) { return mad(v, cs, sg_xor(v, 1) * ss); }

// Two f16 values held in a word, low half first.
static inline float2 f16_pair(uint v) { return convert_float2(as_half2(v)); }

// a + the dot of one chunk with a row: q's lane d is the head's dimension d, k.sd is dimension d of this lane's row.
static inline float idx_dot(float a, float16 k, float q) {
#define D(d, sd) a = mad(k.sd, sub_group_broadcast(q, d), a);
  D(0, s0) D(1, s1) D(2, s2) D(3, s3) D(4, s4) D(5, s5) D(6, s6) D(7, s7)
  D(8, s8) D(9, s9) D(10, sa) D(11, sb) D(12, sc) D(13, sd) D(14, se) D(15, sf)
#undef D
  return a;
}

// A row's indexer scores: score[j] = sum_h relu(iq_h . ik_j) w_h / 64 for j < (pos + 1) / ratio of its KV source's
// pool, rows of D41_CTX_CAP values apart; a later group or a padded row returns, and at most 15 values past the last
// change. iq: rows of D41_IDX_H x D41_IDX_D f32, not yet rotated to the row's position. w: rows of D41_IDX_H f32. pool:
// compressed rows, by d41_comp_index through pt (or none). Groups (D41_IDX_WGS, W), D41_IDX_WG_ROWS.
REQD_SG __kernel void ds_idx_score(__global const float *iq, __global const float *w, __global const D41Row *rows,
                                   uint ratio, __global const uint *pt, uint npp, __global const ushort *pool,
                                   __global float *score) {
  // The rotated q by chunk: it does not fit in registers beside the keys. Lanes hold rows, so heads need no reduction.
  __local float __attribute__((aligned(64))) lq[D41_IDX_D * D41_IDX_H];
  const uint r = get_group_id(1), jg = get_group_id(0) * D41_IDX_WG_ROWS;
  const uint slot = rows[r].slot, pos = rows[r].pos;
  if (slot == D41_NO_ROW) return;
  const uint n = (pos + 1) / ratio;
  if (jg >= n) return;
  const uint s = sg_id(), lane = sg_lane();

  // Two chunks a sub-group, so the rotating ones take each pair's angle once.
  const uint c0 = s * (IDX_CHUNKS / IDX_SGS);
  __global const float *q0 = iq + (ulong)r * D41_IDX_H * D41_IDX_D;
#define RD(e) rd1(p + (e) * D41_IDX_D)
#define RO(e) idx_rot(RD(e), cs, ss)
  if (c0 < IDX_FIRST_ROT) {
    for (uint i = 0; i < IDX_CHUNKS / IDX_SGS; i++) {
      const uint c = c0 + i;
      for (uint hg = 0; hg < D41_IDX_H / 8; hg++) {
        __global const float *p = q0 + (ulong)(8 * hg) * D41_IDX_D + c * IDX_CHUNK;
        const float8 v = (float8)(RD(0), RD(1), RD(2), RD(3), RD(4), RD(5), RD(6), RD(7));
        intel_sub_group_block_write8((__local uint *)(lq + (c * D41_IDX_H + 8 * hg) * IDX_CHUNK), as_uint8(v));
      }
    }
  } else {
    const float sign = (lane & 1) ? 1.0f : -1.0f;
    float th = rope_chain(pos, (c0 - IDX_FIRST_ROT) * 8, 1);
    for (uint i = 0; i < IDX_CHUNKS / IDX_SGS; i++) {
      const uint c = c0 + i;
      float cs;
      const float ss = sign * sincos(d41_yarn(th, (c - IDX_FIRST_ROT) * 8 + lane / 2), &cs);
      th = rope_next8(th, 1);
      for (uint hg = 0; hg < D41_IDX_H / 8; hg++) {
        __global const float *p = q0 + (ulong)(8 * hg) * D41_IDX_D + c * IDX_CHUNK;
        const float8 v = (float8)(RO(0), RO(1), RO(2), RO(3), RO(4), RO(5), RO(6), RO(7));
        intel_sub_group_block_write8((__local uint *)(lq + (c * D41_IDX_H + 8 * hg) * IDX_CHUNK), as_uint8(v));
      }
    }
  }
#undef RO
#undef RD
  barrier(CLK_LOCAL_MEM_FENCE);

  const uint j0 = jg + s * SG_SIZE;
  if (j0 >= n) return;
  // A lane past the last row reads the sub-group's first, since its page may not exist.
  const uint j = j0 + lane < n ? j0 + lane : j0;
  const ulong row = d41_comp_index(d41_page(pt, npp, slot, j * ratio), j, ratio);
  __global const uint8 *key = (__global const uint8 *)(pool + row * D41_COMP_ROW + D41_HD);

  float a[D41_IDX_H];
#define Z(h) a[h] = a[(h) + 1] = a[(h) + 2] = a[(h) + 3] = a[(h) + 4] = a[(h) + 5] = a[(h) + 6] = a[(h) + 7] = 0.0f;
  Z(0) Z(8) Z(16) Z(24)
#undef Z
  for (uint c = 0; c < IDX_CHUNKS; c++) {
    const uint8 raw = key[c];
    const float16 k = (float16)(f16_pair(raw.s0), f16_pair(raw.s1), f16_pair(raw.s2), f16_pair(raw.s3),
                                f16_pair(raw.s4), f16_pair(raw.s5), f16_pair(raw.s6), f16_pair(raw.s7));
    __local const uint *lc = (__local const uint *)(lq + c * D41_IDX_H * IDX_CHUNK);
#define G(h)                                                                         \
  {                                                                                  \
    const float8 q = as_float8(intel_sub_group_block_read8(lc + (h) * IDX_CHUNK));   \
    a[h] = idx_dot(a[h], k, q.s0);                                                   \
    a[(h) + 1] = idx_dot(a[(h) + 1], k, q.s1);                                       \
    a[(h) + 2] = idx_dot(a[(h) + 2], k, q.s2);                                       \
    a[(h) + 3] = idx_dot(a[(h) + 3], k, q.s3);                                       \
    a[(h) + 4] = idx_dot(a[(h) + 4], k, q.s4);                                       \
    a[(h) + 5] = idx_dot(a[(h) + 5], k, q.s5);                                       \
    a[(h) + 6] = idx_dot(a[(h) + 6], k, q.s6);                                       \
    a[(h) + 7] = idx_dot(a[(h) + 7], k, q.s7);                                       \
  }
    G(0) G(8) G(16) G(24)
#undef G
  }

  // 1 / sqrt(D41_IDX_D * D41_IDX_H) = 1 / 64. The sum starts at +0, so no score is -0, which topk_key ranks below +0.
  const float2 wh = rd2(w + (ulong)r * D41_IDX_H) * (1.0f / 64.0f);
  float sum = 0.0f;
#define T(h, x, l) sum = mad(fmax(a[h], 0.0f), sub_group_broadcast(x, l), sum);
#define T4(h, x, l) T(h, x, l) T((h) + 1, x, (l) + 1) T((h) + 2, x, (l) + 2) T((h) + 3, x, (l) + 3)
#define T16(h, x) T4(h, x, 0) T4((h) + 4, x, 4) T4((h) + 8, x, 8) T4((h) + 12, x, 12)
  T16(0, wh.s0) T16(16, wh.s1)
#undef T16
#undef T4
#undef T
  wr1(score + (ulong)r * D41_CTX_CAP + j0, sum);
}

// topk.h's topk_bin, its first sums in loads of 8 bins: a pass would otherwise wait on a load for each of 128 bins
// (docs/measurements/MEASUREMENTS.md#2026-11-02-deepseek41-idx-select).
static inline void idx_bin(__local const uint *hist, uint nb, uint remaining, __local uint *scan) {
  const uint sgs = get_sub_group_size(), lane = sg_lane(), per = nb / sgs;
  uint s = 0;
  __local const uint8 *lb = (__local const uint8 *)(hist + nb - (lane + 1) * per);
  for (uint j = 0; j < per / 8; j++) {
    const uint8 c = lb[j];
    s += (c.s0 + c.s1) + (c.s2 + c.s3) + (c.s4 + c.s5) + (c.s6 + c.s7);
  }
  const uint above = sub_group_scan_inclusive_add(s);
  const uint f = sub_group_reduce_min(above >= remaining ? lane : sgs);
  uint acc = sub_group_broadcast(above - s, f);
  for (uint j0 = 0; j0 < per; j0 += sgs) {
    const uint b = nb - 1 - f * per - j0 - lane;
    const uint h = j0 + lane < per ? hist[b] : 0;
    const uint at = acc + sub_group_scan_inclusive_add(h);
    const uint j = sub_group_reduce_min(at >= remaining && j0 + lane < per ? lane : sgs);
    if (j < sgs) {
      if (lane == j) {
        scan[0] = b;
        scan[1] = remaining - (at - h);
      }
      return;
    }
    acc = sub_group_broadcast(at, sgs - 1);
  }
}

// Clears bins [0, nb): 32 a sub-group, in one block store.
static inline void idx_clear(__local uint *h, uint nb) {
  const uint b = sg_id() * 32;
  if (b < nb) intel_sub_group_block_write2((__local uint *)(h + b), (uint2)(0u));
}

// One key of the final scan, in order of index.
static inline void idx_take(uint key, uint i, uint prefix, uint *eq_take, uint *out, __global uint *idx) {
  if (key > prefix || (key == prefix && *eq_take > 0)) {
    *eq_take -= key == prefix;
    idx[(*out)++] = i;
  }
}

// topk.h's topk_select for k = D41_TOPK, without values, changed for a whole row of scores, where the histograms'
// atomics and one-value loads bound the time (docs/measurements/MEASUREMENTS.md#2026-11-02-deepseek41-idx-select).
// Keys below t do not count: 512 pairs of work-items each hold a key at or above the smallest of the pairs' maxima, so
// the top k are all at or above it. Every scan loads 4 values.
static inline void idx_select(__global const float *v, uint n, __global uint *idx, __local uint *hist,
                              __local uint *scan) {
  const uint lid = get_local_id(0);
  if (D41_TOPK >= n) {
    for (uint i = lid; i < n; i += TOPK_WG) idx[i] = i;
    return;
  }
  const uint lane = sg_lane(), sgs = get_sub_group_size(), n4 = n / 4;
  uint t = 0;
  if (n >= 4 * TOPK_WG) {
    uint mx = 0;
    for (uint j = lid; j < n4; j += TOPK_WG) {
      const float4 x = vload4(j, v);
      mx = max(mx, max(max(topk_key(x.s0), topk_key(x.s1)), max(topk_key(x.s2), topk_key(x.s3))));
    }
    t = work_group_reduce_min(max(mx, sub_group_shuffle_xor(mx, 1u)));
  }
  const bool wide = n > 8192;
  const uint passes = wide ? 3 : 4, hbins = wide ? TOPK_BINS : 256;
  idx_clear(hist, hbins);
  barrier(CLK_LOCAL_MEM_FENCE);
  uint prefix = 0, mask = 0, remaining = D41_TOPK;
  for (uint pass = 0; pass < passes; pass++) {
    const uint width = wide ? (pass < 2 ? 11 : 10) : 8;
    const uint shift = wide ? (pass == 0 ? 21 : pass == 1 ? 10 : 0) : 24 - 8 * pass;
    const uint nb = 1u << width, dmask = nb - 1;
    __local uint *h = hist + TOPK_BINS * (pass & 1);
    for (uint j0 = 0; j0 < n4; j0 += TOPK_WG) {
      const uint j = j0 + lid;
      uint bins[4] = {TOPK_NONE, TOPK_NONE, TOPK_NONE, TOPK_NONE};
      if (j < n4) {
        const float4 x = vload4(j, v);
        const uint keys[4] = {topk_key(x.s0), topk_key(x.s1), topk_key(x.s2), topk_key(x.s3)};
        for (uint q = 0; q < 4; q++)
          if (keys[q] >= t && (keys[q] & mask) == prefix) bins[q] = (keys[q] >> shift) & dmask;
      }
      for (uint q = 0; q < 4; q++) topk_count(h, bins[q], lane, sgs);
    }
    if (lid < n - 4 * n4) {
      const uint key = topk_key(v[4 * n4 + lid]);
      if (key >= t && (key & mask) == prefix) atomic_inc(&h[(key >> shift) & dmask]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sg_id() == 0) idx_bin(h, nb, remaining, scan);
    idx_clear(hist + TOPK_BINS * (~pass & 1), hbins);
    barrier(CLK_LOCAL_MEM_FENCE);
    prefix |= scan[0] << shift;
    mask |= dmask << shift;
    remaining = scan[1];
  }
  // As topk_select's end: a run of indices a work-item, each placed after the keys chosen in the runs before it.
  const uint chunk = (n + TOPK_WG - 1) / TOPK_WG;
  const uint lo = min(lid * chunk, n), hi = min(lo + chunk, n);
  uint gt = 0, eq = 0;
  uint i = lo;
  for (; i + 4 <= hi; i += 4) {
    const float4 x = vload4(0, v + i);
    const uint4 key = (uint4)(topk_key(x.s0), topk_key(x.s1), topk_key(x.s2), topk_key(x.s3));
    gt += (key.s0 > prefix) + (key.s1 > prefix) + (key.s2 > prefix) + (key.s3 > prefix);
    eq += (key.s0 == prefix) + (key.s1 == prefix) + (key.s2 == prefix) + (key.s3 == prefix);
  }
  for (; i < hi; i++) {
    const uint key = topk_key(v[i]);
    gt += key > prefix;
    eq += key == prefix;
  }
  const uint sg = sg_id();
  uint gt_before = sub_group_scan_exclusive_add(gt), eq_before = sub_group_scan_exclusive_add(eq);
  if (lane == sgs - 1) {
    scan[2 + 2 * sg] = gt_before + gt;
    scan[3 + 2 * sg] = eq_before + eq;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  uint g = 0, e = 0;
  for (uint b = lane; b < sg; b += sgs) {
    g += scan[2 + 2 * b];
    e += scan[3 + 2 * b];
  }
  gt_before += sub_group_reduce_add(g);
  eq_before += sub_group_reduce_add(e);
  uint eq_take = eq_before >= remaining ? 0 : min(eq, remaining - eq_before);
  uint out = gt_before + min(eq_before, remaining);
  for (i = lo; i + 4 <= hi; i += 4) {
    const float4 x = vload4(0, v + i);
    idx_take(topk_key(x.s0), i, prefix, &eq_take, &out, idx);
    idx_take(topk_key(x.s1), i + 1, prefix, &eq_take, &out, idx);
    idx_take(topk_key(x.s2), i + 2, prefix, &eq_take, &out, idx);
    idx_take(topk_key(x.s3), i + 3, prefix, &eq_take, &out, idx);
  }
  for (; i < hi; i++) idx_take(topk_key(v[i]), i, prefix, &eq_take, &out, idx);
}

// A row's top min(D41_TOPK, n) of score, n = (pos + 1) / ratio, as indices in increasing order into idx at row r of
// D41_TOPK entries; of equal scores the lower indices. Entries past them and a padded row's are not written. score:
// rows of D41_CTX_CAP values, as ds_idx_score leaves them. Groups W, TOPK_WG.
REQD_SG __kernel void ds_idx_topk(__global const float *score, __global const D41Row *rows, uint ratio,
                                  __global uint *idx) {
  __local uint __attribute__((aligned(64))) hist[TOPK_HIST];
  __local uint scan[TOPK_WG];
  const uint r = get_group_id(0);
  const uint slot = rows[r].slot, pos = rows[r].pos;
  if (slot == D41_NO_ROW) return;
  const uint n = (pos + 1) / ratio;
  if (n == 0) return;
  idx_select(score + (ulong)r * D41_CTX_CAP, n, idx + (ulong)r * D41_TOPK, hist, scan);
}

#define ATT_BATCHES (D41_ATT_KEYS / SG_SIZE)      // 8: lots of 16 keys, the lanes of a dot-product sub-group
#define ATT_CHUNKS (D41_HD / SG_SIZE)             // 32: a head's chunks of 16 values
#define ATT_CHUNK_VALUES (D41_ATT_HEADS * SG_SIZE)  // a chunk of the group's 8 heads
#define ATT_SG_CHUNKS (ATT_CHUNKS / D41_ATT_SGS)  // 2: chunks a sub-group holds of q, and writes of the output
#define ATT_SG_DIMS (ATT_SG_CHUNKS * SG_SIZE)     // 32
#define ATT_DOT_CHUNKS (ATT_CHUNKS / 2)           // 16: each lot's two sub-groups take half of the chunks
#define ATT_SCALE 0.044194173824159216f           // 512^-0.5

// How many splits of keys a row has: the window, and its compressed rows (ratio 0: none).
static inline uint att_compressed(uint pos, uint ratio) { return ratio ? min((pos + 1) / ratio, (uint)D41_TOPK) : 0; }
static inline uint att_splits(uint pos, uint ratio) {
  return 1 + (att_compressed(pos, ratio) + D41_ATT_KEYS - 1) / D41_ATT_KEYS;
}

// Each key's place as a multiple of 64 bytes, lane l holding key i0 + l, i0 a sub-group's first. Split 0 is the window,
// the last vis positions up to the row's, oldest first; a split after it reads idx from entry (split - 1) D41_ATT_KEYS.
// A lane past the m keys takes the sub-group's first, since idx past them is not written.
static inline uint att_key_off(__global const D41Row *row, uint split, uint vis, uint i0, uint m, uint ratio,
                               __global const uint *idx, __global const uint *pt, uint npp) {
  const uint lane = sg_lane(), slot = row->slot, i = i0 + lane;
  if (split == 0) {
    const int c = (int)row->ring - (int)(vis - 1) + (int)(i < m ? i : i0);
    return (slot * D41_RING + (uint)(c < 0 ? c + D41_RING : c)) * (D41_HD * 2 / 64);
  }
  uint j = intel_sub_group_block_read(idx + (split - 1) * D41_ATT_KEYS + i0);
  j = i < m ? j : sub_group_broadcast(j, 0);
  return (uint)d41_comp_index(d41_page(pt, npp, slot, j * ratio), j, ratio) * (D41_COMP_ROW_BYTES / 64);
}

// One split of one row for 8 heads: for its keys k_i, s_i = (q_h . k_i) 512^-0.5 and, with m the head's largest s_i,
// o = sum_i exp(s_i - m) k_i, l = sum_i exp(s_i - m). ratio 0: the window only, with base 10000, in a grid of one
// split. A split past the row's keys or a padded row returns. Groups (heads / 8, W, splits), D41_ATT_WG.
REQD_SG __kernel void ds_attn_split(__global const float *q,  // the row's rows of this card's heads, not yet rotated
                                    __global const D41Row *rows, uint ratio,
                                    __global const ushort *ring,  // split 0's keys, f16: [slot][D41_RING][D41_HD]
                                    __global const uint *pt, uint npp,
                                    __global const ushort *pool,  // later splits' keys, f16, by d41_comp_index, pt
                                    __global const uint *idx,     // their numbers, rows of D41_TOPK
                                    __global float *part) {       // D41_ATT_PART (layout.h)
  __local float __attribute__((aligned(64))) lq[D41_HD * D41_ATT_HEADS];       // [chunk][head][16]
  __local float __attribute__((aligned(64))) lp[2 * D41_ATT_KEYS * D41_ATT_HEADS];  // [half of the chunks][lot][head][16]
  __local uint __attribute__((aligned(64))) lo[D41_ATT_KEYS];
  const uint hg = get_group_id(0), r = get_group_id(1), split = get_group_id(2);
  const uint nh = get_num_groups(0) * D41_ATT_HEADS;
  __global const D41Row *row = rows + r;
  const uint slot = row->slot, pos = row->pos;
  if (slot == D41_NO_ROW || split >= att_splits(pos, ratio)) return;
  const uint s = sg_id(), lane = sg_lane();
  const uint vis = min(pos + 1, (uint)D41_WIN);
  const uint rest = att_compressed(pos, ratio) - (split ? (split - 1) * D41_ATT_KEYS : 0);
  const uint m = split ? min(rest, (uint)D41_ATT_KEYS) : vis;
  const uint nb = (m + SG_SIZE - 1) / SG_SIZE;
  const uint lot = s / 2, side = s & 1;
  __global const uchar *keys = (__global const uchar *)(split ? pool : ring);
  __global const float *q0 = q + ((ulong)r * nh + hg * D41_ATT_HEADS) * D41_HD;

  // A sub-group's 32 values of each head; a rotated chunk takes each pair's angle once for the 8 heads.
  float2 x[D41_ATT_HEADS];
  for (uint h = 0; h < D41_ATT_HEADS; h++) x[h] = rd2(q0 + h * D41_HD + s * ATT_SG_DIMS);
  if (s * ATT_SG_DIMS >= D41_NOPE) {
    const uint yarn = ratio != 0, pair0 = (s * ATT_SG_DIMS - D41_NOPE) / 2;
    const float sign = (lane & 1) ? 1.0f : -1.0f;
    float th = rope_chain(pos, pair0, yarn), cs0, cs1;
    const float sn0 = sign * sincos(yarn ? d41_yarn(th, pair0 + lane / 2) : th, &cs0);
    th = rope_next8(th, yarn);
    const float sn1 = sign * sincos(yarn ? d41_yarn(th, pair0 + 8 + lane / 2) : th, &cs1);
    for (uint h = 0; h < D41_ATT_HEADS; h++)
      x[h] = (float2)(idx_rot(x[h].s0, cs0, sn0), idx_rot(x[h].s1, cs1, sn1));
  }
  __local uint *lq0 = (__local uint *)(lq + s * ATT_SG_CHUNKS * ATT_CHUNK_VALUES);
  intel_sub_group_block_write8(lq0, as_uint8((float8)(x[0].s0, x[1].s0, x[2].s0, x[3].s0, x[4].s0, x[5].s0, x[6].s0,
                                                      x[7].s0)));
  intel_sub_group_block_write8(lq0 + ATT_CHUNK_VALUES,
                               as_uint8((float8)(x[0].s1, x[1].s1, x[2].s1, x[3].s1, x[4].s1, x[5].s1, x[6].s1,
                                                 x[7].s1)));
  if (!side && lot < nb) {
    const uint off = att_key_off(row, split, vis, lot * SG_SIZE, m, ratio, idx + (ulong)r * D41_TOPK, pt, npp);
    intel_sub_group_block_write(lo + lot * SG_SIZE, off);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // The dot products: lanes hold keys, 16 of a lot, so a head needs no reduction, and two sub-groups share a lot's chunks.
  __local float *lp1 = lp + side * D41_ATT_KEYS * D41_ATT_HEADS;
  if (lot < nb) {
    const uint off = intel_sub_group_block_read(lo + lot * SG_SIZE);
    __global const uint8 *key = (__global const uint8 *)(keys + ((ulong)off << 6));
    float8 a = (float8)0.0f;
    for (uint c = side * ATT_DOT_CHUNKS; c < (side + 1) * ATT_DOT_CHUNKS; c++) {
      const float16 k = convert_float16(as_half16(key[c]));
      const float8 qc = as_float8(intel_sub_group_block_read8((__local const uint *)(lq + c * ATT_CHUNK_VALUES)));
      a.s0 = idx_dot(a.s0, k, qc.s0);
      a.s1 = idx_dot(a.s1, k, qc.s1);
      a.s2 = idx_dot(a.s2, k, qc.s2);
      a.s3 = idx_dot(a.s3, k, qc.s3);
      a.s4 = idx_dot(a.s4, k, qc.s4);
      a.s5 = idx_dot(a.s5, k, qc.s5);
      a.s6 = idx_dot(a.s6, k, qc.s6);
      a.s7 = idx_dot(a.s7, k, qc.s7);
    }
    a *= ATT_SCALE;
    if (lot * SG_SIZE + lane >= m) a = (float8)(-INFINITY);
    intel_sub_group_block_write8((__local uint *)(lp1 + lot * ATT_CHUNK_VALUES), as_uint8(a));
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // Each head's largest score and its probabilities, by the sub-group of that number.
  __global float *ph = part + ((ulong)r * nh + hg * D41_ATT_HEADS + s) * D41_ATT_PART;
  if (s < D41_ATT_HEADS) {
    float v[ATT_BATCHES];
    float mx = -INFINITY;
    for (uint b = 0; b < nb; b++) {
      const uint at = b * ATT_CHUNK_VALUES + s * SG_SIZE;
      v[b] = as_float(intel_sub_group_block_read((__local const uint *)(lp + at))) +
             as_float(intel_sub_group_block_read((__local const uint *)(lp + D41_ATT_KEYS * D41_ATT_HEADS + at)));
      mx = fmax(mx, v[b]);
    }
    mx = sg_max(mx);
    float sum = 0.0f;
    for (uint b = 0; b < nb; b++) {
      const float e = exp(v[b] - mx);
      sum += e;
      intel_sub_group_block_write((__local uint *)(lp + b * ATT_CHUNK_VALUES + s * SG_SIZE), as_uint(e));
    }
    sum = sg_sum(sum);
    if (lane == 0) vstore2((float2)(mx, sum), 0, ph + D41_ATT_ML + 2 * split);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // The weighted keys: lanes hold values, 32 of each head a sub-group, each key's probabilities one value in all lanes.
  float2 acc[D41_ATT_HEADS];
  for (uint h = 0; h < D41_ATT_HEADS; h++) acc[h] = (float2)0.0f;
  for (uint b = 0; b < nb; b++) {
    const float8 p = as_float8(intel_sub_group_block_read8((__local const uint *)(lp + b * ATT_CHUNK_VALUES)));
    const uint offs = intel_sub_group_block_read(lo + b * SG_SIZE);
#define A(h, e, l) acc[h] = mad(k, (float2)sub_group_broadcast(p.e, l), acc[h]);
#define KEY(l)                                                                                                         \
  {                                                                                                                    \
    __global const ushort *kp = (__global const ushort *)(keys + ((ulong)sub_group_broadcast(offs, l) << 6));          \
    const float2 k = convert_float2(as_half2(intel_sub_group_block_read_us2(kp + s * ATT_SG_DIMS)));                   \
    A(0, s0, l) A(1, s1, l) A(2, s2, l) A(3, s3, l) A(4, s4, l) A(5, s5, l) A(6, s6, l) A(7, s7, l)                    \
  }
    KEY(0) KEY(1) KEY(2) KEY(3) KEY(4) KEY(5) KEY(6) KEY(7)
    KEY(8) KEY(9) KEY(10) KEY(11) KEY(12) KEY(13) KEY(14) KEY(15)
#undef KEY
#undef A
  }
  __global float *out = part + ((ulong)r * nh + hg * D41_ATT_HEADS) * D41_ATT_PART + split * D41_HD + s * ATT_SG_DIMS;
  for (uint h = 0; h < D41_ATT_HEADS; h++) wr2(out + (ulong)h * D41_ATT_PART, acc[h]);
}

// A row's attention output from its splits: o_h = sum_split exp(m_split - M) o_split / den, den the same sum of the l
// and exp(sink - M), M the largest of the m, the sink and -1e30 (ggml's form, so exp does not overflow at a large sink);
// then its rotation taken off. A padded row returns. Groups (heads, W), 256, heads a multiple of 8.
REQD_SG __kernel void ds_attn_merge(__global const float *part,   // as ds_attn_split leaves it, with the same ratio
                                    __global const float *sinks,  // D41_H f32
                                    uint first_head,              // the card's first; groups of heads count from it
                                    __global const D41Row *rows, uint ratio,
                                    __global uchar *oa,    // a8, [groups][W] rows of 8 heads, D41_OG_A8_ROW bytes
                                    __global float *of) {  // when set, the f32 values, rows of heads of D41_HD
  const uint lh = get_group_id(0), r = get_group_id(1), nh = get_num_groups(0);
  const uint slot = rows[r].slot, pos = rows[r].pos;
  if (slot == D41_NO_ROW) return;
  const uint lane = sg_lane(), ns = att_splits(pos, ratio), i0 = sg_id() * 32;
  __global const float *ph = part + ((ulong)r * nh + lh) * D41_ATT_PART;
  // Lane 2 s holds split s's m, the next its l.
  const float ml = rd1(ph + D41_ATT_ML);
  const bool is_m = !(lane & 1) && lane / 2 < ns;
  const float sink = sinks[first_head + lh];
  const float big = fmax(sg_max(is_m ? ml : -INFINITY), fmax(sink, -1e30f));
  const float e = exp(ml - big);
  const float den = sg_sum(is_m ? e * sg_xor(ml, 1) : 0.0f) + exp(sink - big);
  const float f = e / den;
  float2 y = (float2)0.0f;
  for (uint sp = 0; sp < ns; sp++) y = mad(rd2(ph + sp * D41_HD + i0), (float2)sub_group_broadcast(f, 2 * sp), y);
  if (i0 >= D41_NOPE) {
    const uint yarn = ratio != 0, pair0 = (i0 - D41_NOPE) / 2;
    const float sign = (lane & 1) ? -1.0f : 1.0f;
    float th = rope_chain(pos, pair0, yarn), cs0, cs1;
    const float sn0 = sign * sincos(yarn ? d41_yarn(th, pair0 + lane / 2) : th, &cs0);
    th = rope_next8(th, yarn);
    const float sn1 = sign * sincos(yarn ? d41_yarn(th, pair0 + 8 + lane / 2) : th, &cs1);
    y = (float2)(idx_rot(y.s0, cs0, sn0), idx_rot(y.s1, cs1, sn1));
  }
  if (of) wr2(of + ((ulong)r * nh + lh) * D41_HD + i0, y);
  const uint g = lh / D41_OG_HEADS;
  a8_store_block(y, oa + ((ulong)g * get_num_groups(1) + r) * D41_OG_A8_ROW, D41_OG_K,
                 (lh % D41_OG_HEADS) * (D41_HD / 32) + i0 / 32);
}
