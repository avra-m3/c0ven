// The prompt path's products with quantised weights: one weight type per module (-DQTYPE), in the runs layout or,
// with -DLAYOUT_FILE, the file's (decode.h). Row r of a slice starts at block r * bpr + b0 and has `units` units of 32
// values, so a slice can take part of each row (the file's down matrices hold both cards' halves); run offsets need
// the slice's block count, rows * bpr. dequant_f16 feeds kernels/gemm.cl. The expert kernels decode each lane's row
// into dpas B operands on the fly (decode_unit_pairs), with the activations grouped by expert (prompt.cl).
#include "decode.h"

#include "moe_tile.h"
#include "swiglu.h"

#define QG_SG 4  // sub-groups per work-group: 64 rows
#define MT (MOE_TILE / 8)  // dpas tiles of 8 assignments
#if MOE_TILE % 64
#error "moe_down_t takes half a tile as MT / 2 dpas tiles, a multiple of 4"
#endif

static inline float silu(float x) { return x / (1.0f + exp(-x)); }

#ifndef HAS_UNIT_RAW
// The 32 values of a unit as 16 pairs of fp16 in their own order: b0 for values 0-15, b1 for 16-31.
static inline void unit_to_f16(const uint w[8], float lo, float hi, float mn, float mh, int8 *b0, int8 *b1) {
  uint p[16];
  for (uint i = 0; i < 16; i++) {
    float sc = i < 8 ? lo : hi, m = i < 8 ? mn : mh;
    uint word = w[i / 2];
    uint sh = 16 * (i & 1);
    float v0 = sc * (float)(char)((word >> sh) & 0xff) - m;
    float v1 = sc * (float)(char)((word >> (sh + 8)) & 0xff) - m;
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

// out[r * ldo + i] = value i of row r, fp16. One work-item per unit: groups ceil(rows * units / 64), 64. When a
// sub-group's 16 units lie one after another in out (units a multiple of 16, rows ldo = 32 units apart), its 1 KB goes
// through local memory so that each of four stores writes 256 consecutive bytes, which is much faster than 64-byte
// stores (docs/measurements/MEASUREMENTS.md#2026-10-15-prompt-dequant-stores). Otherwise each work-item stores its unit.
REQD_SG __kernel void dequant_f16(__global const uchar *W, uint rows, uint bpr, uint b0, uint units,
                                  __global half *out, uint ldo, __global const uchar *grids) {
  __local uint buf[64 / SG_SIZE][16 * SG_SIZE];
  LOAD_GRID(grids)
  const uint g = get_global_id(0), lane = sg_lane(), total = rows * units;
  const bool whole = units % SG_SIZE == 0 && ldo == 32 * units && g - lane + SG_SIZE <= total;
  const uint gc = min(g, total - 1), r = gc / units, u = gc % units;
  ulong n = (ulong)rows * bpr;
  uint w[8];
  float lo, hi, mn, mh;
  decode_unit(W, n, (ulong)r * bpr + b0 + u / UNITS_PER_BLOCK, u % UNITS_PER_BLOCK, w, &lo, &hi, &mn, &mh GRID_ARG);
  // The unit's 32 values as 16 words of two fp16.
  uint p[16];
  for (uint j = 0; j < 16; j++) {
    float sc = j < 8 ? lo : hi, m = j < 8 ? mn : mh;
    float v0 = sc * (float)(char)((w[j / 2] >> (16 * (j & 1))) & 0xff) - m;
    float v1 = sc * (float)(char)((w[j / 2] >> (16 * (j & 1) + 8)) & 0xff) - m;
    p[j] = (uint)as_ushort(convert_half(v0)) | ((uint)as_ushort(convert_half(v1)) << 16);
  }
  if (whole) {
    __local uint *b = buf[sg_id()];
    for (uint j = 0; j < 16; j++) b[16 * lane + j] = p[j];
    sub_group_barrier(CLK_LOCAL_MEM_FENCE);
    __global uint4 *o = (__global uint4 *)(out + (ulong)(g - lane) * 32);
    for (uint k = 0; k < 64 / SG_SIZE; k++) o[SG_SIZE * k + lane] = vload4(SG_SIZE * k + lane, b);
  } else if (g < total) {
    __global uint *o = (__global uint *)(out + (ulong)r * ldo + 32 * u);
    vstore8(vload8(0, p), 0, o);
    vstore8(vload8(1, p), 1, o);
  }
}

// The tile's weights: an expert in a cache slot (runs layout) or in a staging slot (file layout).
static inline __global const uchar *tile_base(__global const uchar *base, ulong slot_bytes, MoeTile t) {
  return base + (ulong)t.slot * slot_bytes;
}

// Work-groups (rows / 64, max tiles), 64 work-items. Tile y's assignments are rows t.start .. t.start + t.count of X
// (fp16, ldx apart, 2,560 wide), in the gate type's pair order (quant_pairs_sigma); h goes to the same rows of H (fp16,
// rows wide), in the down type's order: sigma if hsigma. gate and up are rows x K slices of bpr blocks per row, up_off
// bytes apart.
REQD_SG __kernel void moe_gate_up_t(__global const uchar *base, ulong slot_bytes, __global const MoeTile *tiles,
                                    __global const uint *ntiles, uint rows, uint bpr, ulong up_off,
                                    __global const ushort *X, uint K, uint ldx, __global half *H, uint hsigma,
                                    __global const uchar *grids) {
  LOAD_GRID(grids)
  if (get_group_id(1) >= *ntiles) return;
  MoeTile t = tiles[get_group_id(1)];
  const uint lane = sg_lane();
  const uint r = (get_group_id(0) * QG_SG + sg_id()) * 16 + lane;
  const uint rr = min(r, rows - 1);
  __global const uchar *G = tile_base(base, slot_bytes, t);
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
    ulong k = (ulong)rr * bpr + u / UNITS_PER_BLOCK;
#ifdef HAS_UNIT_RAW
    uint un = min(u + 1, units - 1), sn = un % UNITS_PER_BLOCK;
    ulong kn = (ulong)rr * bpr + un / UNITS_PER_BLOCK;
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
    for (uint m = 0; m < 8; m++)
      if (8 * i + m < t.count) H[(ulong)(t.start + 8 * i + m) * rows + (hsigma ? sigma4(r) : r)] =
            convert_half(SWIGLU(g[m], v[m]));
  }
}

// down_tile: moe_down_t for one tile, in nt dpas tiles of 8 assignments. Types with raw_words load the next unit's
// fields while they decode this one, as raw_pairs' types do. y goes out by 2D block writes when Y allows them.
static inline void down_tile(MoeTile t, __global const uchar *D, uint rows, uint bpr, uint b0,
                             __global const ushort *Hin, uint K, __global half *Y, const uint nt GRID_PARAM) {
  const uint lane = sg_lane();
  const uint r0 = (get_group_id(0) * QG_SG + sg_id()) * 16, r = r0 + lane;
  const uint rr = min(r, rows - 1);
  const ulong n = (ulong)rows * bpr;
  float8 acc[MT];
  for (uint i = 0; i < nt; i++) acc[i] = (float8)0.0f;
  const uint units = K / 32;
#ifdef HAS_UNIT_RAW
  const half tbl = raw_table(GRID_ONLY_ARG);
#endif
#if defined(HAS_UNIT_RAW) || defined(HAS_RAW_WORDS)
  uint8 rd = unit_raw(D, n, (ulong)rr * bpr + b0, 0);
#endif
  for (uint u = 0; u < units; u++) {
    ushort a[2][8 * MT];
    for (uint h = 0; h < 2; h++)
      for (uint c = 0; c < nt / 4; c++)
        sg_load_rows32(Hin, K * 2, t.start + t.count, K * 2, 32 * u + 16 * h, t.start + 32 * c, a[h] + 32 * c);
    int8 b[2];
    uint ub = b0 * UNITS_PER_BLOCK + u;  // the unit's index along the stored row
#if defined(HAS_UNIT_RAW) || defined(HAS_RAW_WORDS)
    uint ubn = b0 * UNITS_PER_BLOCK + min(u + 1, units - 1);
    uint8 rd_next = unit_raw(D, n, (ulong)rr * bpr + ubn / UNITS_PER_BLOCK, ubn % UNITS_PER_BLOCK);
#ifdef HAS_UNIT_RAW
    uint pr[16];
    raw_pairs(rd, pr, tbl GRID_ARG);
    b[0] = as_int8(vload8(0, pr));
    b[1] = as_int8(vload8(1, pr));
#else
    uint w[8];
    float lo, hi, mn, mh;
    raw_words(rd, w, &lo, &hi, &mn, &mh GRID_ARG);
    unit_to_f16(w, lo, hi, mn, mh, &b[0], &b[1]);
#endif
    rd = rd_next;
#else
    decode_pairs_own(D, n, (ulong)rr * bpr + ub / UNITS_PER_BLOCK, ub % UNITS_PER_BLOCK, b GRID_ARG);
#endif
    for (uint h = 0; h < 2; h++)
      for (uint i = 0; i < nt; i++) acc[i] = dpas_f16(as_short8(vload8(i, a[h])), b[h], acc[i]);
  }
  // A block write leaves out the rows past the tile and the columns past rows.
  const bool block2d = (rows * 2) % 64 == 0 && ((ulong)Y) % 64 == 0;
  for (uint i = 0; i < nt; i++) {
    float y[8];
    vstore8(acc[i], 0, y);
    if (block2d) {
      ushort h[8];
      for (uint m = 0; m < 8; m++) h[m] = as_ushort(convert_half(y[m]));
      sg_store_rows8((__global ushort *)Y, rows * 2, t.start + t.count, rows * 2, r0, t.start + 8 * i, h);
    } else if (r < rows)
      for (uint m = 0; m < 8; m++)
        if (8 * i + m < t.count) Y[(ulong)(t.start + 8 * i + m) * rows + r] = convert_half(y[m]);
  }
}

// Work-groups (rows / 64, max tiles), 64 work-items. y = down h: down is rows x K (K = units x 32 of this card's
// inputs, from block b0 of rows of bpr blocks); h rows are K wide; y goes to the tile's rows of Y (fp16, rows wide). A
// tile of up to MOE_TILE / 2 assignments takes half the dpas calls and loads of h
// (docs/measurements/MEASUREMENTS.md#2026-10-15-prompt-expert-down).
REQD_SG __kernel void moe_down_t(__global const uchar *base, ulong slot_bytes, ulong down_off,
                                 __global const MoeTile *tiles, __global const uint *ntiles, uint rows, uint bpr,
                                 uint b0, __global const ushort *Hin, uint K, __global half *Y,
                                 __global const uchar *grids) {
  LOAD_GRID(grids)
  if (get_group_id(1) >= *ntiles) return;
  MoeTile t = tiles[get_group_id(1)];
  __global const uchar *D = tile_base(base, slot_bytes, t) + down_off;
  if (t.count <= MOE_TILE / 2)
    down_tile(t, D, rows, bpr, b0, Hin, K, Y, MT / 2 GRID_ARG);
  else
    down_tile(t, D, rows, bpr, b0, Hin, K, Y, MT GRID_ARG);
}

// The 32 values of unit s of block k as dequant_f16 writes them: 16 pairs of fp16, b[0] the first 8.
static inline void dense_pairs(__global const uchar *W, ulong n, ulong k, uint s, int8 b[2] GRID_PARAM) {
  uint w[8];
  float lo, hi, mn, mh;
  decode_unit(W, n, k, s, w, &lo, &hi, &mn, &mh GRID_ARG);
  uint p[16];
  for (uint j = 0; j < 16; j++) {
    float sc = j < 8 ? lo : hi, m = j < 8 ? mn : mh;
    float v0 = sc * (float)(char)((w[j / 2] >> (16 * (j & 1))) & 0xff) - m;
    float v1 = sc * (float)(char)((w[j / 2] >> (16 * (j & 1) + 8)) & 0xff) - m;
    p[j] = (uint)as_ushort(convert_half(v0)) | ((uint)as_ushort(convert_half(v1)) << 16);
  }
  b[0] = as_int8(vload8(0, p));
  b[1] = as_int8(vload8(1, p));
}

// dense_t for one sub-group's 16 rows and nt dpas tiles of 8 tokens from t0.
static inline void dense_tile(__global const uchar *W, uint rows, uint bpr, __global const ushort *X, uint K, uint ldx,
                              uint T, uint t0, __global float *Y, uint ldy, uint yhalf, const uint nt GRID_PARAM) {
  const uint lane = sg_lane();
  const uint r0 = (get_group_id(0) * QG_SG + sg_id()) * 16, r = r0 + lane;
  const uint rr = min(r, rows - 1);
  const ulong n = (ulong)rows * bpr;
  float8 acc[MT];
  for (uint i = 0; i < nt; i++) acc[i] = (float8)0.0f;
  const uint units = K / 32;
  for (uint u = 0; u < units; u++) {
    ushort a[2][8 * MT];
    for (uint h = 0; h < 2; h++)
      for (uint c = 0; c < nt / 4; c++) sg_load_rows32(X, K * 2, T, ldx * 2, 32 * u + 16 * h, t0 + 32 * c, a[h] + 32 * c);
    int8 b[2];
    dense_pairs(W, n, (ulong)rr * bpr + u / UNITS_PER_BLOCK, u % UNITS_PER_BLOCK, b GRID_ARG);
    for (uint h = 0; h < 2; h++)
      for (uint i = 0; i < nt; i++) acc[i] = dpas_f16(as_short8(vload8(i, a[h])), b[h], acc[i]);
  }
  const uint esize = yhalf ? 2 : 4;
  const bool block2d = ((ldy * esize) % 64) == 0 && (((ulong)Y) % 64) == 0;
  for (uint i = 0; i < nt; i++) {
    float v[8];
    vstore8(acc[i], 0, v);
    ushort h[8];
    if (yhalf)
      for (uint m = 0; m < 8; m++) h[m] = as_ushort(convert_half(v[m]));
    if (block2d) {
      int2 at = (int2)(r0, t0 + 8 * i);
      if (yhalf) {
        sg_store_rows8((__global ushort *)Y, rows * 2, T, ldy * 2, at.x, at.y, h);
      } else {
        uint w[8];
        for (uint m = 0; m < 8; m++) w[m] = as_uint(v[m]);
        intel_sub_group_2d_block_write_32b_8r16x1c((__global void *)Y, rows * 4, T, ldy * 4, at, w);
      }
    } else {
      for (uint m = 0; m < 8; m++) {
        const uint t = t0 + 8 * i + m;
        if (t < T && r < rows) {
          if (yhalf)
            vstore_half(v[m], (ulong)t * ldy + r, (__global half *)Y);
          else
            Y[(ulong)t * ldy + r] = v[m];
        }
      }
    }
  }
}

// Work-groups (ceil(rows / 64), ceil(max tokens / MOE_TILE)), 64 work-items. Y[t][r] = sum_k X[t][k] W[r][k] for
// t < *ntok, the weights decoded on the fly as dequant_f16 decodes them and the products summed in gemm_f16's order,
// so Y has dequant_f16 and gemm_f16's bits. W rows x K, bpr blocks a row; X fp16, rows ldx apart, 64-byte aligned; Y
// f32, or fp16 if yhalf, rows ldy apart. Each token tile decodes the weights again, so this is for few tokens.
REQD_SG __kernel void dense_t(__global const uchar *W, uint rows, uint bpr, __global const ushort *X, uint K, uint ldx,
                              __global const uint *ntok, __global float *Y, uint ldy, uint yhalf,
                              __global const uchar *grids) {
  const uint T = *ntok, t0 = get_group_id(1) * MOE_TILE;
  if (t0 >= T) return;  // before the grid's barrier: the whole work-group leaves
  LOAD_GRID(grids)
  if (T - t0 <= MOE_TILE / 2)
    dense_tile(W, rows, bpr, X, K, ldx, T, t0, Y, ldy, yhalf, MT / 2 GRID_ARG);
  else
    dense_tile(W, rows, bpr, X, K, ldx, T, t0, Y, ldy, yhalf, MT GRID_ARG);
}
