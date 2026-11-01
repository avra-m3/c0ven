// deepseek41's decode-step kernels around attention: the q and KV norms, the window ring and the compressor. The maths
// is the deepseek41 forward pass; core/models/deepseek41/ref_ops.h holds one scalar definition of each. Caches are
// f16, block-written, indexed by position, so a step can run again and a rejected draft overwrites only its own cells.

// An angle's bits must equal ggml's: at position 16,383 one unit in the last place of float is 1e-3 radians. Above the
// includes, since a pragma covers only what follows it and rope.h's d41_yarn adds two products.
#pragma OPENCL FP_CONTRACT OFF

#include "a8.h"
#include "builtins.h"
#include "layout.h"
#include "ops.h"
#include "rope.h"

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
