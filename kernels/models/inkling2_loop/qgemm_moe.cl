// Prompt-path XMX expert gate/up projection with SiTU activation for inkling2_loop, one weight type per module.
// Down projection uses the shared kernels/quant/qgemm.cl moe_down_t module directly.
#include "decode.h"
#include "moe_tile.h"

#define QG_SG 4
#define MT (MOE_TILE / 8)

static inline float i2_sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }

static inline float i2_situ(float g, float u, float beta, float linear_beta) {
  const float s = (beta * tanh(g / beta)) * i2_sigmoid(g);
  return linear_beta > 0.0f ? s * (linear_beta * tanh(u / linear_beta)) : s * u;
}

#ifndef HAS_UNIT_RAW
static inline void unit_to_f16(const uint w[8], float lo, float hi, float mn, float mh, int8 *b0, int8 *b1) {
  uint p[16];
  for (uint i = 0; i < 16; i++) {
    const float sc = i < 8 ? lo : hi, m = i < 8 ? mn : mh;
    const uint word = w[i / 2];
    const uint sh = 16 * (i & 1);
    const float v0 = sc * (float)(char)((word >> sh) & 0xff) - m;
    const float v1 = sc * (float)(char)((word >> (sh + 8)) & 0xff) - m;
    p[i] = (uint)as_ushort(convert_half(v0)) | ((uint)as_ushort(convert_half(v1)) << 16);
  }
  *b0 = as_int8((uint8)(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]));
  *b1 = as_int8((uint8)(p[8], p[9], p[10], p[11], p[12], p[13], p[14], p[15]));
}

static inline void decode_pairs_own(__global const uchar *W, ulong n, ulong k, uint s, int8 b[2] GRID_PARAM) {
  uint w[8];
  float lo, hi, mn, mh;
  decode_unit(W, n, k, s, w, &lo, &hi, &mn, &mh GRID_ARG);
  unit_to_f16(w, lo, hi, mn, mh, &b[0], &b[1]);
}
#endif

// Work-groups: (ceil(rows / 64), max_tiles, 1), local (64, 1, 1).
// base[exp_idx] holds [gate | up] starting at 0 and up_off in runs layout (slot = t.slot).
REQD_SG __kernel void moe_gate_up_situ_t(__global const uchar *base, ulong slot_bytes, __global const MoeTile *tiles,
                                         __global const uint *ntiles, uint rows, uint bpr, ulong up_off,
                                         __global const ushort *X, uint K, uint ldx, __global half *H, uint hsigma,
                                         float beta, float lin_beta, __global const uchar *grids) {
  LOAD_GRID(grids)
  if (get_group_id(1) >= *ntiles) return;
  const MoeTile t = tiles[get_group_id(1)];
  const uint lane = sg_lane();
  const uint r = (get_group_id(0) * QG_SG + sg_id()) * 16 + lane;
  const uint rr = min(r, rows - 1);
  __global const uchar *G = base + (ulong)t.slot * slot_bytes;
  __global const uchar *U = G + up_off;
  const ulong n = (ulong)rows * bpr;
  float8 ag[MT], au[MT];
  for (uint i = 0; i < MT; i++) ag[i] = au[i] = (float8)0.0f;
  const uint units = K / 32;
#ifdef HAS_BLOCK_RAW
  const half tbl = raw_table(GRID_ONLY_ARG);
  uint8 bg = block_raw(G, n, (ulong)rr * bpr), bu = block_raw(U, n, (ulong)rr * bpr);
  uint8 rg = unit_raw_in(G, n, (ulong)rr * bpr, 0, bg), ru = unit_raw_in(U, n, (ulong)rr * bpr, 0, bu);
#elif defined(HAS_UNIT_RAW)
  const half tbl = raw_table(GRID_ONLY_ARG);
  uint8 rg = unit_raw(G, n, (ulong)rr * bpr, 0), ru = unit_raw(U, n, (ulong)rr * bpr, 0);
#endif
  for (uint u = 0; u < units; u++) {
    ushort a[2][8 * MT];
    for (uint h = 0; h < 2; h++)
      for (uint c = 0; c < MT / 4; c++)
        sg_load_rows32(X, K * 2, t.start + t.count, ldx * 2, 32 * u + 16 * h, t.start + 32 * c, a[h] + 32 * c);
    uint pr[16];
    int8 b[2];
    const ulong k = (ulong)rr * bpr + u / UNITS_PER_BLOCK;
#ifdef HAS_UNIT_RAW
    const uint un = min(u + 1, units - 1), sn = un % UNITS_PER_BLOCK;
    const ulong kn = (ulong)rr * bpr + un / UNITS_PER_BLOCK;
#ifdef HAS_BLOCK_RAW
    if (sn == 0 && un != u) {
      bg = block_raw(G, n, kn);
      bu = block_raw(U, n, kn);
    }
    uint8 rg_next = unit_raw_in(G, n, kn, sn, bg), ru_next = unit_raw_in(U, n, kn, sn, bu);
#else
    uint8 rg_next = unit_raw(G, n, kn, sn), ru_next = unit_raw(U, n, kn, sn);
#endif
    raw_pairs(rg, pr, tbl GRID_ARG);
    b[0] = as_int8(vload8(0, pr));
    b[1] = as_int8(vload8(1, pr));
#else
    decode_pairs_own(G, n, k, u % UNITS_PER_BLOCK, b GRID_ARG);
#endif
    for (uint h = 0; h < 2; h++)
      for (uint i = 0; i < MT; i++) ag[i] = dpas_f16(as_short8(vload8(i, a[h])), b[h], ag[i]);
#ifdef HAS_UNIT_RAW
    raw_pairs(ru, pr, tbl GRID_ARG);
    rg = rg_next;
    ru = ru_next;
    b[0] = as_int8(vload8(0, pr));
    b[1] = as_int8(vload8(1, pr));
#else
    decode_pairs_own(U, n, k, u % UNITS_PER_BLOCK, b GRID_ARG);
#endif
    for (uint h = 0; h < 2; h++)
      for (uint i = 0; i < MT; i++) au[i] = dpas_f16(as_short8(vload8(i, a[h])), b[h], au[i]);
  }
  if (r >= rows) return;
  for (uint i = 0; i < MT; i++) {
    float g[8], v[8];
    vstore8(ag[i], 0, g);
    vstore8(au[i], 0, v);
    for (uint m = 0; m < 8; m++) {
      if (8 * i + m < t.count) {
        H[(ulong)(t.start + 8 * i + m) * rows + (hsigma ? sigma4(r) : r)] =
            convert_half(i2_situ(g[m], v[m], beta, lin_beta));
      }
    }
  }
}
