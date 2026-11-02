// The decode step's expert products by groups (shared/moe_tile.h), bit for bit matvec.cl's moe_gate_up_g and
// moe_down_g: the lanes take the same units in the same order (16 a row for gate and up, 4 for down). The work around
// the dot products, which bounds those kernels, is cheaper: 32-bit field offsets stepping by a constant, vector loads,
// several rows a sub-group per activation load, i-quant signs as dot4(g ^ m, q) - dot4(m, q), m 0xff in negative bytes.
// Modules moe_<type>_g<GROUP> (-DQTYPE, -DGROUP: 1, 2, 3, 4, 6 or 8); a work-group takes MOE_WG_ROWS rows of one
// expert.
//
// With -DMOE_CLASSES a module holds deepseek41's kernels instead, at the end of this file: moe_gate_up_c, unless
// -DCL_DOWN_ONLY, and moe_down_c, unless -DCL_GATE_UP_ONLY.
#include "a8.h"
#include "decode.h"
#include "moe_tile.h"
#ifdef MOE_CLASSES
#include "layout.h"
#define SWIGLU_LIMIT D41_SWIGLU_LIMIT
#endif
#include "swiglu.h"

#define A GROUP

#define RUN(f) ((uint)quant_run_offset(QTYPE, f, n))  // field f's run in a slice of n blocks

static inline float silu(float x) { return x / (1.0f + exp(-x)); }

static inline uint ld_u32x(__global const uchar *p) { return *(__global const uint *)p; }

// An activation unit: 8 words of codes and the scale.
static inline float act_unit(__global const uchar *a, uint K, uint u, uint q[8]) {
  uint4 qa = vload4(0, (__global const uint *)(a + 32 * u)), qb = vload4(1, (__global const uint *)(a + 32 * u));
  q[0] = qa.s0; q[1] = qa.s1; q[2] = qa.s2; q[3] = qa.s3;
  q[4] = qb.s0; q[5] = qb.s1; q[6] = qb.s2; q[7] = qb.s3;
  return ((__global const float *)(a + K))[u];
}

// ---------------------------------------------------------------------------------------------------------------
// Gate and up: 256 values a block, 8 units. Lane l takes unit l & 7 of blocks (l >> 3), (l >> 3) + 2, ...

#if QTYPE == QT_IQ2_S || QTYPE == QT_IQ3_XXS || QTYPE == QT_IQ3_S || QTYPE == QT_IQ4_XS || QTYPE == QT_IQ2_XXS || \
    QTYPE == QT_IQ2_XS || QTYPE == QT_IQ1_S || QTYPE == QT_IQ1_M || QTYPE == QT_Q2_K

// Unit s of block k as 8 magnitude words g and 8 nibbles of sign bits (bit j: byte j negative), or for IQ4_XS, IQ1_S,
// IQ1_M and Q2_K as 8 signed words and no signs; lo, hi, mn and mh as decode_unit gives them.
static inline void gu_decode(__global const uchar *W, uint n, uint k, uint s, uint g[8], uint sg[8], float *lo,
                             float *hi, float *mn, float *mh GRID_PARAM) {
  *mn = *mh = 0.0f;
#if QTYPE == QT_IQ3_XXS
  uint2 qs = vload2(0, (__global const uint *)(W + RUN(0) + k * 64 + 8 * s));
  uint aux = ld_u32x(W + RUN(1) + k * 32 + 4 * s);
  __local const uint *gr = (__local const uint *)grid;
  for (uint l = 0; l < 4; l++) {
    uint word = l < 2 ? qs.s0 : qs.s1, sh = 16 * (l & 1);
    uint s7 = (aux >> (7 * l)) & 127;
    uint bits = s7 | ((popcount(s7) & 1) << 7);
    g[2 * l] = gr[(word >> sh) & 255u];
    g[2 * l + 1] = gr[(word >> (sh + 8)) & 255u];
    sg[2 * l] = bits & 15;
    sg[2 * l + 1] = bits >> 4;
  }
  *lo = *hi = vload_half(0, (__global const half *)(W + RUN(2) + k * 2)) * (0.5f + (float)(aux >> 28)) * 0.5f;
#elif QTYPE == QT_IQ3_S
  uint2 qs = vload2(0, (__global const uint *)(W + RUN(0) + k * 64 + 8 * s));
  uint sg4 = ld_u32x(W + RUN(1) + k * 32 + 4 * s);
  uint qh = W[RUN(2) + k * 8 + s];
  uint sc = (W[RUN(3) + k * 4 + s / 2] >> (4 * (s & 1))) & 15;
  __local const uint *gr = (__local const uint *)grid;
  for (uint l = 0; l < 4; l++) {
    uint word = l < 2 ? qs.s0 : qs.s1, sh = 16 * (l & 1);
    uint bits = (sg4 >> (8 * l)) & 255u;
    g[2 * l] = gr[((word >> sh) & 255u) | (((qh >> (2 * l)) & 1) << 8)];
    g[2 * l + 1] = gr[((word >> (sh + 8)) & 255u) | (((qh >> (2 * l + 1)) & 1) << 8)];
    sg[2 * l] = bits & 15;
    sg[2 * l + 1] = bits >> 4;
  }
  *lo = *hi = vload_half(0, (__global const half *)(W + RUN(4) + k * 2)) * (float)(1 + 2 * sc);
#elif QTYPE == QT_IQ2_S
  uint qs4 = ld_u32x(W + RUN(0) + k * 32 + 4 * s);
  uint sg4 = ld_u32x(W + RUN(1) + k * 32 + 4 * s);
  uint qh = W[RUN(2) + k * 8 + s];
  uint sc = W[RUN(3) + k * 8 + s];
  for (uint l = 0; l < 4; l++) {
    uint idx = ((qs4 >> (8 * l)) & 255u) | (((qh >> (2 * l)) & 3u) << 8);
    uint2 e = *(GRID_AS const uint2 *)(grid + 8 * idx);
    uint bits = (sg4 >> (8 * l)) & 255u;
    g[2 * l] = e.s0;
    g[2 * l + 1] = e.s1;
    sg[2 * l] = bits & 15;
    sg[2 * l + 1] = bits >> 4;
  }
  float d = vload_half(0, (__global const half *)(W + RUN(4) + k * 2)) * 0.25f;
  *lo = d * (0.5f + (float)(sc & 15));
  *hi = d * (0.5f + (float)(sc >> 4));
#elif QTYPE == QT_IQ2_XXS || QTYPE == QT_IQ2_XS
  uint2 qs = vload2(0, (__global const uint *)(W + RUN(0) + k * 64 + 8 * s));
  for (uint l = 0; l < 4; l++) {
#if QTYPE == QT_IQ2_XXS
    uint idx = (qs.s0 >> (8 * l)) & 255u, s7 = (qs.s1 >> (7 * l)) & 127u;
#else
    uint v = ((l < 2 ? qs.s0 : qs.s1) >> (16 * (l & 1))) & 0xffffu;
    uint idx = v & 511u, s7 = v >> 9;
#endif
    uint2 e = *(GRID_AS const uint2 *)(grid + 8 * idx);
    uint bits = s7 | ((popcount(s7) & 1) << 7);
    g[2 * l] = e.s0;
    g[2 * l + 1] = e.s1;
    sg[2 * l] = bits & 15;
    sg[2 * l + 1] = bits >> 4;
  }
#if QTYPE == QT_IQ2_XXS
  *lo = *hi = vload_half(0, (__global const half *)(W + RUN(1) + k * 2)) * (0.5f + (float)(qs.s1 >> 28)) * 0.25f;
#else
  uint sc = W[RUN(1) + k * 8 + s];
  float d = vload_half(0, (__global const half *)(W + RUN(2) + k * 2)) * 0.25f;
  *lo = d * (0.5f + (float)(sc & 15));
  *hi = d * (0.5f + (float)(sc >> 4));
#endif
#elif QTYPE == QT_IQ1_S
  uint qs4 = ld_u32x(W + RUN(0) + k * 32 + 4 * s);
  uint qh = *(__global const ushort *)(W + RUN(1) + k * 16 + 2 * s);
  for (uint l = 0; l < 4; l++) {
    uint2 e = *(GRID_AS const uint2 *)(grid + 8 * (((qs4 >> (8 * l)) & 255u) | (((qh >> (3 * l)) & 7u) << 8)));
    g[2 * l] = iq1_word(e.s0, qh & 0x8000u);
    g[2 * l + 1] = iq1_word(e.s1, qh & 0x8000u);
    sg[2 * l] = sg[2 * l + 1] = 0;
  }
  *lo = *hi = vload_half(0, (__global const half *)(W + RUN(2) + k * 2)) * (float)(2 * ((qh >> 12) & 7u) + 1) * 0.125f;
#elif QTYPE == QT_Q2_K
  uint8 qv = vload8(0, (__global const uint *)(W + RUN(0) + k * 64 + 32 * (s / 4)));
  uint qw[8] = {qv.s0, qv.s1, qv.s2, qv.s3, qv.s4, qv.s5, qv.s6, qv.s7};
  uint shift = 2 * (s % 4);
  for (uint j = 0; j < 8; j++) {
    g[j] = (qw[j] >> shift) & 0x03030303u;
    sg[j] = 0;
  }
  uint sc = *(__global const ushort *)(W + RUN(1) + k * 16 + 2 * s);
  float d = vload_half(0, (__global const half *)(W + RUN(2) + k * 4));
  float dmin = vload_half(1, (__global const half *)(W + RUN(2) + k * 4));
  *lo = d * (float)(sc & 15u);
  *hi = d * (float)((sc >> 8) & 15u);
  *mn = dmin * (float)((sc >> 4) & 15u);
  *mh = dmin * (float)(sc >> 12);
#elif QTYPE == QT_IQ1_M
  uint qs4 = ld_u32x(W + RUN(0) + k * 32 + 4 * s);
  uint qh2 = *(__global const ushort *)(W + RUN(1) + k * 16 + 2 * s);
  uint2 sc = vload2(0, (__global const uint *)(W + RUN(2) + k * 8));
  float d = (float)as_half((ushort)(((sc.s0 >> 12) & 0x000fu) | ((sc.s0 >> 24) & 0x00f0u) | ((sc.s1 >> 4) & 0x0f00u) |
                                    ((sc.s1 >> 16) & 0xf000u)));
  uint ss = (((s < 4 ? sc.s0 : sc.s1) >> (16 * ((s / 2) & 1))) & 0xffffu) >> (6 * (s & 1));
  for (uint l = 0; l < 4; l++) {
    uint h = (qh2 >> (4 * l)) & 15u;
    uint2 e = *(GRID_AS const uint2 *)(grid + 8 * (((qs4 >> (8 * l)) & 255u) | ((h & 7u) << 8)));
    g[2 * l] = iq1_word(e.s0, h & 8u);
    g[2 * l + 1] = iq1_word(e.s1, h & 8u);
    sg[2 * l] = sg[2 * l + 1] = 0;
  }
  *lo = d * (float)(2 * (ss & 7) + 1) * 0.125f;
  *hi = d * (float)(2 * ((ss >> 3) & 7) + 1) * 0.125f;
#elif QTYPE == QT_IQ4_XS
  uint4 qv = vload4(0, (__global const uint *)(W + RUN(0) + k * 128 + 16 * s));
  uint xw[4] = {qv.s0, qv.s1, qv.s2, qv.s3};
  for (uint j = 0; j < 4; j++) {
    uint lo4 = 0, hi4 = 0;
    for (uint b = 0; b < 4; b++) {
      uint byte = (xw[j] >> (8 * b)) & 0xff;
      lo4 |= (uint)grid[byte & 15] << (8 * b);
      hi4 |= (uint)grid[byte >> 4] << (8 * b);
    }
    g[j] = lo4;
    g[4 + j] = hi4;
    sg[j] = sg[4 + j] = 0;
  }
  uint sl = (W[RUN(1) + k * 4 + s / 2] >> (4 * (s & 1))) & 15;
  uint sh = (*(__global const ushort *)(W + RUN(2) + k * 2) >> (2 * s)) & 3;
  *lo = *hi = vload_half(0, (__global const half *)(W + RUN(3) + k * 2)) * (float)((int)(sl | (sh << 4)) - 32);
#endif
}

#if QTYPE == QT_IQ4_XS || QTYPE == QT_IQ1_S || QTYPE == QT_IQ1_M || QTYPE == QT_Q2_K
#define SIGNED_GRID 0
#else
#define SIGNED_GRID 1
#endif

// Q2_K's mins' term, as unit_dot forms it.
static inline float gu_mins(const uint q[8], float mn, float mh) {
  int sa0 = 0, sa1 = 0;
  for (uint j = 0; j < 4; j++) sa0 += dot4(q[j], 0x01010101u);
  for (uint j = 4; j < 8; j++) sa1 += dot4(q[j], 0x01010101u);
  return mins_term(mn, mh, sa0, sa1);
}

// sum over the unit of w * x, as unit_dot gives it, with the words still split from their signs.
static inline float gu_dot1(const uint g[8], const uint sg[8], const uint q[8], float lo, float hi, float mn,
                            float mh) {
  int s0 = 0, s1 = 0;
  for (uint j = 0; j < 8; j++) {
#if SIGNED_GRID
    uint m = sign_mask4(sg[j]);
    int v = dot4(g[j] ^ m, q[j]) - dot4(m, q[j]);
#else
    int v = dot4(g[j], q[j]);
#endif
    if (j < 4) s0 += v; else s1 += v;
  }
  float r = lo * (float)s0 + hi * (float)s1;
#if QTYPE == QT_Q2_K
  r -= gu_mins(q, mn, mh);
#endif
  return r;
}

// The words with their signs applied, for several activations. No grid byte is 0, so (g ^ m) + (m & 1) has no carry
// between bytes.
static inline void gu_signed(uint g[8], const uint sg[8]) {
#if SIGNED_GRID
  for (uint j = 0; j < 8; j++) {
    uint m = sign_mask4(sg[j]);
    g[j] = (g[j] ^ m) + (m & 0x01010101u);
  }
#endif
}

static inline float gu_dotw(const uint w[8], const uint q[8], float lo, float hi, float mn, float mh) {
  int s0 = 0, s1 = 0;
  for (uint j = 0; j < 4; j++) s0 += dot4(w[j], q[j]);
  for (uint j = 4; j < 8; j++) s1 += dot4(w[j], q[j]);
  float r = lo * (float)s0 + hi * (float)s1;
#if QTYPE == QT_Q2_K
  r -= gu_mins(q, mn, mh);
#endif
  return r;
}

// h[a * rows + r] = silu(gate_r . x_a) * (up_r . x_a) for the group's assignments a, x_a at act + (a / per_tok) *
// act_stride. The arguments are moe_gate_up_g's, then ha: unless null, h's rows also as a8 (a8.h; rows rows + rows / 8
// bytes apart, rows a multiple of MOE_WG_ROWS), each work-group's 32 values of an assignment by one sub-group.
#ifndef MOE_CLASSES
#define R MOE_GU_ROWS(A)
REQD_SG __kernel void moe_gate_up_g(__global const uchar *cache, ulong slot_bytes, __global const MoeGroup *groups,
                                    __global const uint *ngroups, uint rows, uint bpr, ulong up_off,
                                    __global const uchar *act, uint K, uint act_stride, uint per_tok,
                                    __global float *h, __global const uchar *grids, __global uchar *ha) {
  __local float hs[A][MOE_WG_ROWS];
  LOAD_GRID(grids)
  if (get_group_id(1) >= *ngroups) return;
  __global const MoeGroup *gp = groups + get_group_id(1);
  const uint na = A == 1 ? 1 : gp->n;
  uint asg[A];
  for (uint t = 0; t < A; t++) asg[t] = t < na ? gp->asg[t] : gp->asg[0];
  const uint lane = sg_lane(), s = lane & 7;
  const uint r0 = (get_group_id(0) * (MOE_WG_ROWS / R) + sg_id()) * R;
  __global const uchar *G = cache + (ulong)gp->slot * slot_bytes;
  __global const uchar *U = G + up_off;
  const uint n = rows * bpr;
  float ag[R * A], au[R * A];
  uint kb[R];
  for (uint i = 0; i < R; i++) {
    for (uint t = 0; t < A; t++) ag[i * A + t] = au[i * A + t] = 0.0f;
    kb[i] = min(r0 + i, rows - 1) * bpr + (lane >> 3);
  }
  for (uint b = lane >> 3; b < bpr; b += 2) {
    const uint u = 8 * b + s;
#if GROUP == 1
    uint q[8];
    float da = act_unit(act + (asg[0] / per_tok) * act_stride, K, u, q);
    for (uint i = 0; i < R; i++) {
      uint g[8], sg[8];
      float lo, hi, mn, mh;
      gu_decode(G, n, kb[i], s, g, sg, &lo, &hi, &mn, &mh GRID_ARG);
      ag[i] += da * gu_dot1(g, sg, q, lo, hi, mn, mh);
      gu_decode(U, n, kb[i], s, g, sg, &lo, &hi, &mn, &mh GRID_ARG);
      au[i] += da * gu_dot1(g, sg, q, lo, hi, mn, mh);
      kb[i] += 2;
    }
#else
    // The weights decoded once, for every assignment.
    for (uint i = 0; i < R; i++) {
      uint wg[8], wu[8], sg[8];
      float glo, ghi, gmn, gmh, ulo, uhi, umn, umh;
      gu_decode(G, n, kb[i], s, wg, sg, &glo, &ghi, &gmn, &gmh GRID_ARG);
      gu_signed(wg, sg);
      gu_decode(U, n, kb[i], s, wu, sg, &ulo, &uhi, &umn, &umh GRID_ARG);
      gu_signed(wu, sg);
      for (uint t = 0; t < A; t++) {
        if (t < na) {
          uint q[8];
          float da = act_unit(act + (asg[t] / per_tok) * act_stride, K, u, q);
          ag[i * A + t] += da * gu_dotw(wg, q, glo, ghi, gmn, gmh);
          au[i * A + t] += da * gu_dotw(wu, q, ulo, uhi, umn, umh);
        }
      }
      kb[i] += 2;
    }
#endif
  }
  for (uint i = 0; i < R; i++) {
    const uint r = r0 + i;
    for (uint t = 0; t < A; t++) {
      float gv = sg_sum(ag[i * A + t]), uv = sg_sum(au[i * A + t]);
      if (t < na && r < rows && lane == 0) {
        const float hv = SWIGLU(gv, uv);
        h[asg[t] * rows + r] = hv;
        hs[t][r % MOE_WG_ROWS] = hv;
      }
    }
  }
  if (!ha) return;
  barrier(CLK_LOCAL_MEM_FENCE);
  const uint t = sg_id();
  if (t < na) a8_store(vload2(lane, hs[t]), ha + (ulong)gp->asg[t] * (rows + rows / 8), rows, get_group_id(0));
}

#undef R
#endif

#endif

// ---------------------------------------------------------------------------------------------------------------
// Down: short rows (the card's ff values). 4 lanes a row, lane l taking units l, l + 4, ...; a sub-group pass covers
// 4 rows, and a sub-group makes MOE_DOWN_ROWS / 4 passes, rows base, base + 4, ...

#if QTYPE == QT_IQ4_NL || QTYPE == QT_Q2_0 || QTYPE == QT_Q8_0 || QTYPE == QT_Q2_K || QTYPE == QT_Q5_1

#define DL 4

static inline float lanes4_sum(float x) {
  x += sg_xor(x, 1);
  return x + sg_xor(x, 2);
}

// Unit u of a row whose blocks start at block kr, as 8 signed words and decode_unit's lo, hi, mn and mh.
static inline float4 down_decode(__global const uchar *W, uint n, uint kr, uint u, uint w[8] GRID_PARAM) {
#if QTYPE == QT_IQ4_NL
  const uint k = kr + u;
  uint4 qv = vload4(0, (__global const uint *)(W + RUN(0) + k * 16));
  uint xw[4] = {qv.s0, qv.s1, qv.s2, qv.s3};
  for (uint j = 0; j < 4; j++) {
    uint lo4 = 0, hi4 = 0;
    for (uint b = 0; b < 4; b++) {
      uint byte = (xw[j] >> (8 * b)) & 0xff;
      lo4 |= (uint)grid[byte & 15] << (8 * b);
      hi4 |= (uint)grid[byte >> 4] << (8 * b);
    }
    w[j] = lo4;
    w[4 + j] = hi4;
  }
  float d = vload_half(0, (__global const half *)(W + RUN(1) + k * 2));
  return (float4)(d, d, 0.0f, 0.0f);
#elif QTYPE == QT_Q8_0
  // Q8_0: the 32 signed bytes as they are, no codebook.
  const uint k = kr + u;
  uint8 qv = vload8(0, (__global const uint *)(W + RUN(0) + k * 32));
  w[0] = qv.s0; w[1] = qv.s1; w[2] = qv.s2; w[3] = qv.s3;
  w[4] = qv.s4; w[5] = qv.s5; w[6] = qv.s6; w[7] = qv.s7;
  float d = vload_half(0, (__global const half *)(W + RUN(1) + k * 2));
  return (float4)(d, d, 0.0f, 0.0f);
#elif QTYPE == QT_Q2_K
  // Q2_K: the codes 0-3, with two scales and two mins.
  const uint k = kr + u / 8, s = u % 8;
  uint8 qv = vload8(0, (__global const uint *)(W + RUN(0) + k * 64 + 32 * (s / 4)));
  uint qw[8] = {qv.s0, qv.s1, qv.s2, qv.s3, qv.s4, qv.s5, qv.s6, qv.s7};
  for (uint j = 0; j < 8; j++) w[j] = (qw[j] >> (2 * (s % 4))) & 0x03030303u;
  uint sc = *(__global const ushort *)(W + RUN(1) + k * 16 + 2 * s);
  float d = vload_half(0, (__global const half *)(W + RUN(2) + k * 4));
  float dmin = vload_half(1, (__global const half *)(W + RUN(2) + k * 4));
  return (float4)(d * (float)(sc & 15u), d * (float)((sc >> 8) & 15u), dmin * (float)((sc >> 4) & 15u),
                  dmin * (float)(sc >> 12));
#elif QTYPE == QT_Q5_1
  // Q5_1: the codes 0-31, 4 bits in qs and the fifth in qh, with a minimum.
  const uint k = kr + u;
  uint4 qv = vload4(0, (__global const uint *)(W + RUN(0) + k * 16));
  uint xw[4] = {qv.s0, qv.s1, qv.s2, qv.s3};
  uint qh = ld_u32x(W + RUN(1) + k * 4);
  for (uint j = 0; j < 4; j++) {
    w[j] = (xw[j] & 0x0f0f0f0fu) | (spread4(qh >> (4 * j)) << 4);
    w[4 + j] = ((xw[j] >> 4) & 0x0f0f0f0fu) | (spread4(qh >> (16 + 4 * j)) << 4);
  }
  float d = vload_half(0, (__global const half *)(W + RUN(2) + k * 4));
  float m = vload_half(1, (__global const half *)(W + RUN(2) + k * 4));
  return (float4)(d, d, -m, -m);
#else
  // Q2_0: the codes 0-3 unsigned here; the caller subtracts the activation's sum for the -1.
  const uint k = kr + u / 2;
  uint2 qv = vload2(0, (__global const uint *)(W + RUN(0) + k * 16 + 8 * (u & 1)));
  for (uint j = 0; j < 8; j++) {
    uint x = ((j < 4 ? qv.s0 : qv.s1) >> (8 * (j & 3))) & 255u;
    w[j] = (x & 0x03u) | ((x & 0x0cu) << 6) | ((x & 0x30u) << 12) | ((x & 0xc0u) << 18);
  }
  float d = vload_half(0, (__global const half *)(W + RUN(1) + k * 2));
  return (float4)(d, d, 0.0f, 0.0f);
#endif
}

// The unit's sum, as unit_dot gives it, for down_decode's scales sc. qs0 and qs1: for Q2_0, Q2_K and Q5_1, the sums of
// q[0..3] and q[4..7]. The dot products are summed in two loops apart: one loop that picks s0 or s1 by index gave
// wrong sums for Q2_0 (IGC 2.20).
static inline float down_dot(const uint w[8], const uint q[8], float4 sc, int qs0, int qs1) {
  int s0 = 0, s1 = 0;
  for (uint j = 0; j < 4; j++) s0 += dot4(w[j], q[j]);
  for (uint j = 4; j < 8; j++) s1 += dot4(w[j], q[j]);
#if QTYPE == QT_Q2_K || QTYPE == QT_Q5_1
  float r = sc.s0 * (float)s0 + sc.s1 * (float)s1;
  r -= mins_term(sc.s2, sc.s3, qs0, qs1);
  return r;
#else
#if QTYPE == QT_Q2_0
  s0 -= qs0;
  s1 -= qs1;
#endif
  const float d = sc.s0;
  return d * (float)s0 + d * (float)s1;
#endif
}

static inline void q_sums(const uint q[8], int *qs0, int *qs1) {
  *qs0 = *qs1 = 0;
#if QTYPE == QT_Q2_0 || QTYPE == QT_Q2_K || QTYPE == QT_Q5_1
  for (uint j = 0; j < 4; j++) *qs0 += dot4(q[j], 0x01010101u);
  for (uint j = 4; j < 8; j++) *qs1 += dot4(q[j], 0x01010101u);
#endif
}

// y[a * rows + r] = down_r . h_a for the group's assignments a, h_a at act + a * act_stride. The arguments are
// moe_down_g's.
#define P (MOE_DOWN_ROWS(QTYPE, A) / (SG_SIZE / DL))  // passes
REQD_SG __kernel void moe_down_g(__global const uchar *cache, ulong slot_bytes, __global const MoeGroup *groups,
                                 __global const uint *ngroups, uint rows, uint bpr, ulong down_off,
                                 __global const uchar *act, uint K, uint act_stride, __global float *y,
                                 __global const uchar *grids) {
  LOAD_GRID(grids)
  if (get_group_id(1) >= *ngroups) return;
  __global const MoeGroup *gp = groups + get_group_id(1);
  const uint na = A == 1 ? 1 : gp->n;
  uint asg[A];
  for (uint t = 0; t < A; t++) asg[t] = t < na ? gp->asg[t] : gp->asg[0];
  const uint lane = sg_lane() % DL, units = K / 32;
  const uint base = (get_group_id(0) * (MOE_WG_ROWS / MOE_DOWN_ROWS(QTYPE, A)) + sg_id()) * MOE_DOWN_ROWS(QTYPE, A) +
                    sg_lane() / DL;
  __global const uchar *W = cache + (ulong)gp->slot * slot_bytes + down_off;
  const uint n = rows * bpr;
  float acc[P * A];
  uint kr[P];
  for (uint i = 0; i < P; i++) {
    for (uint t = 0; t < A; t++) acc[i * A + t] = 0.0f;
    kr[i] = min(base + (SG_SIZE / DL) * i, rows - 1) * bpr;
  }
  for (uint u = lane; u < units; u += DL) {
#if GROUP == 1
    uint q[8];
    int qs0, qs1;
    float da = act_unit(act + asg[0] * act_stride, K, u, q);
    q_sums(q, &qs0, &qs1);
    for (uint i = 0; i < P; i++) {
      uint w[8];
      float4 sc = down_decode(W, n, kr[i], u, w GRID_ARG);
      acc[i] += da * down_dot(w, q, sc, qs0, qs1);
    }
#else
    for (uint i = 0; i < P; i++) {
      uint w[8];
      float4 sc = down_decode(W, n, kr[i], u, w GRID_ARG);
      for (uint t = 0; t < A; t++) {
        if (t < na) {
          uint q[8];
          int qs0, qs1;
          float da = act_unit(act + asg[t] * act_stride, K, u, q);
          q_sums(q, &qs0, &qs1);
          acc[i * A + t] += da * down_dot(w, q, sc, qs0, qs1);
        }
      }
    }
#endif
  }
  for (uint i = 0; i < P; i++) {
    const uint r = base + (SG_SIZE / DL) * i;
    for (uint t = 0; t < A; t++) {
      float v = lanes4_sum(acc[i * A + t]);
      if (t < na && r < rows && sg_lane() % DL == 0) y[asg[t] * rows + r] = v;
    }
  }
}

#undef P

#endif

// ---------------------------------------------------------------------------------------------------------------
// deepseek41's routed experts (ADR-0062 item 2): a launch mixes slot classes. A group's record names its class in the
// layer's table (D41Class), which gives where its slot lies, its rows and its down K, so part A, part B and a whole
// expert share one launch. GU_LANES and DOWN_LANES are the lanes a row; a sub-group keeps moe_gate_up_g's and
// moe_down_g's rows. Rows and K are multiples of MOE_WG_ROWS and of the 256 values of a block.

#ifdef MOE_CLASSES

#ifndef GU_LANES
#define GU_LANES D41_GU_LANES
#endif
#ifndef DOWN_LANES
#define DOWN_LANES D41_DOWN_LANES(QTYPE, A)
#endif
// ds41_moe_test's sweep sets the rows a sub-group.
#ifdef CL_ROWS
#define CL_GU_ROWS CL_ROWS
#define CL_DOWN_ROWS CL_ROWS
#else
#define CL_GU_ROWS D41_GU_ROWS(A)
#define CL_DOWN_ROWS D41_DOWN_ROWS
#endif
#define CL_SETS(L) (SG_SIZE / (L))                         // rows a sub-group works on at once
#define CL_RS(L, rows) ((rows) < CL_SETS(L) ? CL_SETS(L) : (rows))  // rows a sub-group
#define CL_PASSES(L, rows) (CL_RS(L, rows) / CL_SETS(L))   // rows a lane takes per activation load

#ifdef GRID_CONSTANT
#define CL_GRIDS __constant uchar *
#else
#define CL_GRIDS __global const uchar *
#endif

static inline float cl_sum(float x, uint lanes) {
  if (lanes == SG_SIZE) return sg_sum(x);
  for (uint m = 1; m < lanes; m <<= 1) x += sg_xor(x, m);
  return x;
}

#if QTYPE == QT_IQ1_S || QTYPE == QT_IQ1_M
// A weight is lo or hi times 8 g + delta, g a grid byte of -1, 0 or 1 and delta 1 or -1, so a dot is 8 dot4(g, q) plus
// or minus a sum of q's bytes: the same whole numbers as dots with 8 g + delta, without forming those bytes. IQ1_S has
// one delta a unit and one scale, IQ1_M a delta for each of a unit's 4 runs of 8 values. m: 0 or -1, the delta's sign.
static inline void cl_decode(__global const uchar *W, uint n, uint k, uint s, uint g[8], uint m[8], float *lo,
                             float *hi GRID_PARAM) {
#if QTYPE == QT_IQ1_S
  const uint qs4 = ld_u32x(W + RUN(0) + k * 32 + 4 * s);
  const uint qh = *(__global const ushort *)(W + RUN(1) + k * 16 + 2 * s);
  for (uint l = 0; l < 4; l++) {
    const uint2 e = *(GRID_AS const uint2 *)(grid + 8 * (((qs4 >> (8 * l)) & 255u) | (((qh >> (3 * l)) & 7u) << 8)));
    g[2 * l] = e.s0;
    g[2 * l + 1] = e.s1;
  }
  m[0] = (uint)(-(int)(qh >> 15));
  *lo = *hi = vload_half(0, (__global const half *)(W + RUN(2) + k * 2)) * (float)(2 * ((qh >> 12) & 7u) + 1) * 0.125f;
#else
  const uint qs4 = ld_u32x(W + RUN(0) + k * 32 + 4 * s);
  const uint qh2 = *(__global const ushort *)(W + RUN(1) + k * 16 + 2 * s);
  const uint2 sc = vload2(0, (__global const uint *)(W + RUN(2) + k * 8));
  const float d = (float)as_half((ushort)(((sc.s0 >> 12) & 0x000fu) | ((sc.s0 >> 24) & 0x00f0u) |
                                          ((sc.s1 >> 4) & 0x0f00u) | ((sc.s1 >> 16) & 0xf000u)));
  const uint ss = (((s < 4 ? sc.s0 : sc.s1) >> (16 * ((s / 2) & 1))) & 0xffffu) >> (6 * (s & 1));
  for (uint l = 0; l < 4; l++) {
    const uint h = (qh2 >> (4 * l)) & 15u;
    const uint2 e = *(GRID_AS const uint2 *)(grid + 8 * (((qs4 >> (8 * l)) & 255u) | ((h & 7u) << 8)));
    g[2 * l] = e.s0;
    g[2 * l + 1] = e.s1;
    m[l] = (uint)(-(int)(h >> 3));
  }
  *lo = d * (float)(2 * (ss & 7) + 1) * 0.125f;
  *hi = d * (float)(2 * ((ss >> 3) & 7) + 1) * 0.125f;
#endif
}

// The sums of q's bytes a unit's dots need.
static inline void cl_qsums(const uint q[8], int qp[4]) {
#if QTYPE == QT_IQ1_S
  int a = 0;
  for (uint j = 0; j < 8; j++) a += dot4(q[j], 0x01010101u);
  qp[0] = a;
#else
  for (uint l = 0; l < 4; l++) qp[l] = dot4(q[2 * l], 0x01010101u) + dot4(q[2 * l + 1], 0x01010101u);
#endif
}

static inline float cl_dot(const uint g[8], const uint m[8], const uint q[8], const int qp[4], float lo, float hi) {
#if QTYPE == QT_IQ1_S
  int s = 0;
  for (uint j = 0; j < 8; j++) s += dot4(g[j], q[j]);
  return lo * (float)(8 * s + ((qp[0] ^ (int)m[0]) - (int)m[0]));
#else
  int s[4];
  for (uint l = 0; l < 4; l++)
    s[l] = 8 * (dot4(g[2 * l], q[2 * l]) + dot4(g[2 * l + 1], q[2 * l + 1])) + ((qp[l] ^ (int)m[l]) - (int)m[l]);
  return lo * (float)(s[0] + s[1]) + hi * (float)(s[2] + s[3]);
#endif
}

// For several activations: nothing to do to the words first.
static inline void cl_prepare(uint g[8], const uint m[8]) {}
#define cl_dot_many cl_dot

#else

static inline void cl_decode(__global const uchar *W, uint n, uint k, uint s, uint g[8], uint sg[8], float *lo,
                             float *hi GRID_PARAM) {
  float mn, mh;
  gu_decode(W, n, k, s, g, sg, lo, hi, &mn, &mh GRID_ARG);
}
static inline void cl_qsums(const uint q[8], int qp[4]) {}
static inline float cl_dot(const uint g[8], const uint sg[8], const uint q[8], const int qp[4], float lo, float hi) {
  return gu_dot1(g, sg, q, lo, hi, 0.0f, 0.0f);
}
static inline void cl_prepare(uint g[8], const uint sg[8]) { gu_signed(g, sg); }
static inline float cl_dot_many(const uint g[8], const uint sg[8], const uint q[8], const int qp[4], float lo,
                                float hi) {
  return gu_dotw(g, q, lo, hi, 0.0f, 0.0f);
}

#endif

#ifndef CL_DOWN_ONLY
// Stored only as a8, into ha, ha_stride bytes an assignment: a row of a class of r rows is r codes and r / 8 bytes of
// scales. cache: the card's experts, D41Class.base from its start. K: 5,120 here. x: a8 at act + (a / per_tok) *
// act_stride for assignment a. Groups past *ngroups return, and a group's work-groups past its rows; so groups (the
// largest class's rows / MOE_WG_ROWS, the most groups), local size 16 MOE_WG_ROWS / CL_RS.
#define L GU_LANES
#define P CL_PASSES(L, CL_GU_ROWS)
#define RS CL_RS(L, CL_GU_ROWS)
REQD_SG __kernel void moe_gate_up_c(__global const uchar *cache, __global const D41Class *classes,
                                    __global const D41Group *groups, __global const uint *ngroups,
                                    __global const uchar *act, uint K, uint act_stride, uint per_tok,
                                    __global uchar *ha, uint ha_stride, CL_GRIDS grids) {
  __local float hs[A][MOE_WG_ROWS];
  if (get_group_id(1) >= *ngroups) return;
  __global const D41Group *gp = groups + get_group_id(1);
  __global const D41Class *cl = classes + gp->cls;
  const uint rows = cl->rows, bpr = K / 256;
  if (get_group_id(0) * MOE_WG_ROWS >= rows) return;
  LOAD_GRID(grids)
  const uint na = A == 1 ? 1 : gp->n;
  uint asg[A];
  for (uint t = 0; t < A; t++) asg[t] = t < na ? gp->asg[t] : gp->asg[0];
  const uint lane = sg_lane();
  const uint r0 = (get_group_id(0) * (MOE_WG_ROWS / RS) + sg_id()) * RS + lane / L;
  __global const uchar *G = cache + cl->base + (ulong)gp->slot * cl->slot_bytes;
  __global const uchar *U = G + cl->up_off;
  const uint n = rows * bpr, units = K / 32;
  float ag[P * A], au[P * A];
  uint kb[P];
  for (uint i = 0; i < P; i++) {
    for (uint t = 0; t < A; t++) ag[i * A + t] = au[i * A + t] = 0.0f;
    kb[i] = (r0 + CL_SETS(L) * i) * bpr;
  }
  for (uint u = lane % L; u < units; u += L) {
    const uint b = u >> 3, s = u & 7;
#if GROUP == 1
    uint q[8];
    int qp[4];
    const float da = act_unit(act + (asg[0] / per_tok) * act_stride, K, u, q);
    cl_qsums(q, qp);
    for (uint i = 0; i < P; i++) {
      uint g[8], sg[8];
      float lo, hi;
      cl_decode(G, n, kb[i] + b, s, g, sg, &lo, &hi GRID_ARG);
      ag[i] += da * cl_dot(g, sg, q, qp, lo, hi);
      cl_decode(U, n, kb[i] + b, s, g, sg, &lo, &hi GRID_ARG);
      au[i] += da * cl_dot(g, sg, q, qp, lo, hi);
    }
#else
    for (uint i = 0; i < P; i++) {
      uint wg[8], wu[8], sg[8], su[8];
      float glo, ghi, ulo, uhi;
      cl_decode(G, n, kb[i] + b, s, wg, sg, &glo, &ghi GRID_ARG);
      cl_prepare(wg, sg);
      cl_decode(U, n, kb[i] + b, s, wu, su, &ulo, &uhi GRID_ARG);
      cl_prepare(wu, su);
      for (uint t = 0; t < A; t++) {
        if (t < na) {
          uint q[8];
          int qp[4];
          const float da = act_unit(act + (asg[t] / per_tok) * act_stride, K, u, q);
          cl_qsums(q, qp);
          ag[i * A + t] += da * cl_dot_many(wg, sg, q, qp, glo, ghi);
          au[i * A + t] += da * cl_dot_many(wu, su, q, qp, ulo, uhi);
        }
      }
    }
#endif
  }
  for (uint i = 0; i < P; i++) {
    const uint r = r0 + CL_SETS(L) * i;
    for (uint t = 0; t < A; t++) {
      const float gv = cl_sum(ag[i * A + t], L), uv = cl_sum(au[i * A + t], L);
      if (t < na && lane % L == 0) hs[t][r % MOE_WG_ROWS] = SWIGLU(gv, uv);
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const uint t = sg_id();
  if (t < na) a8_store(vload2(lane, hs[t]), ha + (ulong)asg[t] * ha_stride, rows, get_group_id(0));
}
#undef RS
#undef P
#undef L
#endif

#ifndef CL_GATE_UP_ONLY
// y[a * rows + r] = down_r . h_a for r < rows, the class's K values of h_a, a8 at act + a * act_stride; y's rows a
// multiple of MOE_WG_ROWS. Groups (rows / MOE_WG_ROWS, the most groups), local size 16 MOE_WG_ROWS / CL_RS.
#define L DOWN_LANES
#define P CL_PASSES(L, CL_DOWN_ROWS)
#define RS CL_RS(L, CL_DOWN_ROWS)
REQD_SG __kernel void moe_down_c(__global const uchar *cache, __global const D41Class *classes,
                                 __global const D41Group *groups, __global const uint *ngroups, uint rows,
                                 __global const uchar *act, uint act_stride, __global float *y, CL_GRIDS grids) {
  if (get_group_id(1) >= *ngroups) return;
  LOAD_GRID(grids)
  __global const D41Group *gp = groups + get_group_id(1);
  __global const D41Class *cl = classes + gp->cls;
  const uint K = cl->rows, bpr = K / 256, units = K / 32;
  const uint na = A == 1 ? 1 : gp->n;
  uint asg[A];
  for (uint t = 0; t < A; t++) asg[t] = t < na ? gp->asg[t] : gp->asg[0];
  const uint lane = sg_lane();
  const uint base = (get_group_id(0) * (MOE_WG_ROWS / RS) + sg_id()) * RS + lane / L;
  __global const uchar *W = cache + cl->base + (ulong)gp->slot * cl->slot_bytes + cl->down_off;
  const uint n = rows * bpr;
  float acc[P * A];
  uint kr[P];
  for (uint i = 0; i < P; i++) {
    for (uint t = 0; t < A; t++) acc[i * A + t] = 0.0f;
    kr[i] = (base + CL_SETS(L) * i) * bpr;
  }
  for (uint u = lane % L; u < units; u += L) {
    const uint b = u >> 3, s = u & 7;
#if GROUP == 1
    uint q[8];
    int qp[4];
    const float da = act_unit(act + asg[0] * act_stride, K, u, q);
    cl_qsums(q, qp);
    for (uint i = 0; i < P; i++) {
      uint g[8], sg[8];
      float lo, hi;
      cl_decode(W, n, kr[i] + b, s, g, sg, &lo, &hi GRID_ARG);
      acc[i] += da * cl_dot(g, sg, q, qp, lo, hi);
    }
#else
    for (uint i = 0; i < P; i++) {
      uint g[8], sg[8];
      float lo, hi;
      cl_decode(W, n, kr[i] + b, s, g, sg, &lo, &hi GRID_ARG);
      cl_prepare(g, sg);
      for (uint t = 0; t < A; t++) {
        if (t < na) {
          uint q[8];
          int qp[4];
          const float da = act_unit(act + asg[t] * act_stride, K, u, q);
          cl_qsums(q, qp);
          acc[i * A + t] += da * cl_dot_many(g, sg, q, qp, lo, hi);
        }
      }
    }
#endif
  }
  for (uint i = 0; i < P; i++) {
    const uint r = base + CL_SETS(L) * i;
    for (uint t = 0; t < A; t++) {
      const float v = cl_sum(acc[i * A + t], L);
      if (t < na && lane % L == 0) y[(ulong)asg[t] * rows + r] = v;
    }
  }
}
#undef RS
#undef P
#undef L
#endif

#endif
