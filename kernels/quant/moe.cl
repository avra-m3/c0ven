// The decode step's expert products by groups (shared/moe_tile.h), bit for bit matvec.cl's moe_gate_up_g and
// moe_down_g: the lanes take the same units in the same order (16 a row for gate and up, 4 for down). The work around
// the dot products, which bounds those kernels, is cheaper: 32-bit field offsets stepping by a constant, vector loads,
// several rows a sub-group per activation load, i-quant signs as dot4(g ^ m, q) - dot4(m, q), m 0xff in negative bytes.
// Modules moe_<type>_g<GROUP> (-DQTYPE, -DGROUP: 1, 2, 3, 4, 6 or 8); a work-group takes MOE_WG_ROWS rows of one
// expert.
#include "a8.h"
#include "decode.h"
#include "moe_tile.h"

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
    QTYPE == QT_IQ2_XS || QTYPE == QT_IQ1_M

// Unit s of block k as 8 magnitude words g and 8 nibbles of sign bits (bit j: byte j negative), or for IQ4_XS and
// IQ1_M as 8 signed words and no signs; lo and hi as decode_unit gives them.
static inline void gu_decode(__global const uchar *W, uint n, uint k, uint s, uint g[8], uint sg[8], float *lo,
                             float *hi GRID_PARAM) {
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
    uint2 e = *(__local const uint2 *)(grid + 8 * idx);
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
    uint2 e = *(__local const uint2 *)(grid + 8 * idx);
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
#elif QTYPE == QT_IQ1_M
  uint qs4 = ld_u32x(W + RUN(0) + k * 32 + 4 * s);
  uint qh2 = *(__global const ushort *)(W + RUN(1) + k * 16 + 2 * s);
  uint2 sc = vload2(0, (__global const uint *)(W + RUN(2) + k * 8));
  float d = (float)as_half((ushort)(((sc.s0 >> 12) & 0x000fu) | ((sc.s0 >> 24) & 0x00f0u) | ((sc.s1 >> 4) & 0x0f00u) |
                                    ((sc.s1 >> 16) & 0xf000u)));
  uint ss = (((s < 4 ? sc.s0 : sc.s1) >> (16 * ((s / 2) & 1))) & 0xffffu) >> (6 * (s & 1));
  for (uint l = 0; l < 4; l++) {
    uint h = (qh2 >> (4 * l)) & 15u;
    uint2 e = *(__local const uint2 *)(grid + 8 * (((qs4 >> (8 * l)) & 255u) | ((h & 7u) << 8)));
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

#if QTYPE == QT_IQ4_XS || QTYPE == QT_IQ1_M
#define SIGNED_GRID 0
#else
#define SIGNED_GRID 1
#endif

// sum over the unit of w * x, as unit_dot gives it, with the words still split from their signs.
static inline float gu_dot1(const uint g[8], const uint sg[8], const uint q[8], float lo, float hi) {
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
  return lo * (float)s0 + hi * (float)s1;
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

static inline float gu_dotw(const uint w[8], const uint q[8], float lo, float hi) {
  int s0 = 0, s1 = 0;
  for (uint j = 0; j < 4; j++) s0 += dot4(w[j], q[j]);
  for (uint j = 4; j < 8; j++) s1 += dot4(w[j], q[j]);
  return lo * (float)s0 + hi * (float)s1;
}

// h[a * rows + r] = silu(gate_r . x_a) * (up_r . x_a) for the group's assignments a, x_a at act + (a / per_tok) *
// act_stride. The arguments are moe_gate_up_g's, then ha: unless null, h's rows also as a8 (a8.h; rows rows + rows / 8
// bytes apart, rows a multiple of MOE_WG_ROWS), each work-group's 32 values of an assignment by one sub-group.
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
      float lo, hi;
      gu_decode(G, n, kb[i], s, g, sg, &lo, &hi GRID_ARG);
      ag[i] += da * gu_dot1(g, sg, q, lo, hi);
      gu_decode(U, n, kb[i], s, g, sg, &lo, &hi GRID_ARG);
      au[i] += da * gu_dot1(g, sg, q, lo, hi);
      kb[i] += 2;
    }
#else
    // The weights decoded once, for every assignment.
    for (uint i = 0; i < R; i++) {
      uint wg[8], wu[8], sg[8];
      float glo, ghi, ulo, uhi;
      gu_decode(G, n, kb[i], s, wg, sg, &glo, &ghi GRID_ARG);
      gu_signed(wg, sg);
      gu_decode(U, n, kb[i], s, wu, sg, &ulo, &uhi GRID_ARG);
      gu_signed(wu, sg);
      for (uint t = 0; t < A; t++) {
        if (t < na) {
          uint q[8];
          float da = act_unit(act + (asg[t] / per_tok) * act_stride, K, u, q);
          ag[i * A + t] += da * gu_dotw(wg, q, glo, ghi);
          au[i * A + t] += da * gu_dotw(wu, q, ulo, uhi);
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
        const float hv = silu(gv) * uv;
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

// ---------------------------------------------------------------------------------------------------------------
// Down: short rows (the card's ff values). 4 lanes a row, lane l taking units l, l + 4, ...; a sub-group pass covers
// 4 rows, and a sub-group makes MOE_DOWN_ROWS / 4 passes, rows base, base + 4, ...

#if QTYPE == QT_IQ4_NL || QTYPE == QT_Q2_0

#define DL 4

static inline float lanes4_sum(float x) {
  x += sg_xor(x, 1);
  return x + sg_xor(x, 2);
}

// Unit u of a row whose blocks start at block kr, as 8 signed words and its scale.
static inline float down_decode(__global const uchar *W, uint n, uint kr, uint u, uint w[8] GRID_PARAM) {
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
  return vload_half(0, (__global const half *)(W + RUN(1) + k * 2));
#else
  // Q2_0: the codes 0-3 unsigned here; the caller subtracts the activation's sum for the -1.
  const uint k = kr + u / 2;
  uint2 qv = vload2(0, (__global const uint *)(W + RUN(0) + k * 16 + 8 * (u & 1)));
  for (uint j = 0; j < 8; j++) {
    uint x = ((j < 4 ? qv.s0 : qv.s1) >> (8 * (j & 3))) & 255u;
    w[j] = (x & 0x03u) | ((x & 0x0cu) << 6) | ((x & 0x30u) << 12) | ((x & 0xc0u) << 18);
  }
  return vload_half(0, (__global const half *)(W + RUN(1) + k * 2));
#endif
}

// The unit's sum, as unit_dot gives it. qs0 and qs1: for Q2_0, the sums of q[0..3] and q[4..7]. The dot products
// are summed in two loops apart: one loop that picks s0 or s1 by index gave wrong sums for Q2_0 (IGC 2.20).
static inline float down_dot(const uint w[8], const uint q[8], float d, int qs0, int qs1) {
  int s0 = 0, s1 = 0;
  for (uint j = 0; j < 4; j++) s0 += dot4(w[j], q[j]);
  for (uint j = 4; j < 8; j++) s1 += dot4(w[j], q[j]);
#if QTYPE == QT_Q2_0
  s0 -= qs0;
  s1 -= qs1;
#endif
  return d * (float)s0 + d * (float)s1;
}

static inline void q_sums(const uint q[8], int *qs0, int *qs1) {
  *qs0 = *qs1 = 0;
#if QTYPE == QT_Q2_0
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
      float d = down_decode(W, n, kr[i], u, w GRID_ARG);
      acc[i] += da * down_dot(w, q, d, qs0, qs1);
    }
#else
    for (uint i = 0; i < P; i++) {
      uint w[8];
      float d = down_decode(W, n, kr[i], u, w GRID_ARG);
      for (uint t = 0; t < A; t++) {
        if (t < na) {
          uint q[8];
          int qs0, qs1;
          float da = act_unit(act + asg[t] * act_stride, K, u, q);
          q_sums(q, &qs0, &qs1);
          acc[i * A + t] += da * down_dot(w, q, d, qs0, qs1);
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
