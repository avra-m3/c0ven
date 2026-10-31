// Decoding 32 weights of a block in the runs layout (shared/quant.h) into 8 words of 4 signed bytes, with the float
// scales that make their integer dot product with an activation unit real (docs/spec/QUANT_FORMATS.md, by section).
// A unit is 32 consecutive values of a row (unit s of block k); for x = da * q (int8 q) every type gives sum w x =
// da * (lo * dot(w[0..3], q[0..3]) + hi * dot(w[4..7], q[4..7]) - mn * sum(q)).
// QTYPE is the weight type, set per module (-DQTYPE=...). The slice starts at W.
#ifndef KERNELS_QUANT_DECODE_H
#define KERNELS_QUANT_DECODE_H

#include "builtins.h"
#include "quant.h"

#define QK quant_block_values(QTYPE)
#define UNITS_PER_BLOCK (QK / 32)
// FIELD(f, k): field f of block k. In the runs layout the slice holds n blocks, one run per field; with -DLAYOUT_FILE
// the blocks are as the file stores them, one after another (the experts streamed to the cards for a prompt).
#ifdef LAYOUT_FILE
#define FIELD(f, k) (W + (ulong)(k) * quant_block_bytes(QTYPE) + quant_field_offset(QTYPE, f))
#else
#define FIELD(f, k) (W + quant_run_offset(QTYPE, f, n) + (ulong)(k) * quant_field_size(QTYPE, f))
#endif

// A 32-bit load from a field. File blocks keep 16-bit alignment only (18, 34 or 98 bytes long).
#ifdef LAYOUT_FILE
static inline uint dec_u32(__global const uchar *p) { return (uint)ld_u16(p) | ((uint)ld_u16(p + 2) << 16); }
#else
static inline uint dec_u32(__global const uchar *p) { return ld_u32(p); }
#endif

// Bytes of the codebook a type needs in shared local memory, and where it starts in the packed buffer.
#if QTYPE == QT_IQ2_S
#define GRID_OFF QGRID_IQ2S
#define GRID_BYTES 8192
#elif QTYPE == QT_IQ3_XXS
#define GRID_OFF QGRID_IQ3XXS
#define GRID_BYTES 1024
#elif QTYPE == QT_IQ3_S
#define GRID_OFF QGRID_IQ3S
#define GRID_BYTES 2048
#elif QTYPE == QT_IQ4_NL || QTYPE == QT_IQ4_XS
#define GRID_OFF QGRID_KVALUES
#define GRID_BYTES 16
#elif QTYPE == QT_IQ2_XXS
#define GRID_OFF QGRID_IQ2XXS
#define GRID_BYTES 2048
#elif QTYPE == QT_IQ2_XS
#define GRID_OFF QGRID_IQ2XS
#define GRID_BYTES 4096
#elif QTYPE == QT_IQ1_S || QTYPE == QT_IQ1_M
#define GRID_OFF QGRID_IQ1S
#define GRID_BYTES 16384
#else
#define GRID_BYTES 0
#endif

#if GRID_BYTES > 0
#define GRID_PARAM , __local const uchar *grid
#define GRID_ONLY_PARAM __local const uchar *grid
#define GRID_ONLY_ARG grid
#define GRID_ARG , grid
// Copies the type's codebook into shared local memory; every work-item of the group takes part.
#define LOAD_GRID(src)                                                                      \
  __local uint grid_words[GRID_BYTES / 4];                                                  \
  for (uint i_ = get_local_id(0); i_ < GRID_BYTES / 4; i_ += get_local_size(0))             \
    grid_words[i_] = ((__global const uint *)((src) + GRID_OFF))[i_];                       \
  barrier(CLK_LOCAL_MEM_FENCE);                                                             \
  __local const uchar *grid = (__local const uchar *)grid_words;
#else
#define GRID_PARAM
#define GRID_ONLY_PARAM void
#define GRID_ONLY_ARG
#define GRID_ARG
#define LOAD_GRID(src)
#endif

// IQ4_NL / IQ4_XS: 16 bytes, byte j holds value j (low nibble) and value j + 16 (high nibble), through the table.
// iq4_word: one word of them, 4 bytes, into the values' words lo and hi.
static inline void iq4_word(uint x, uint *lo, uint *hi, __local const uchar *kv) {
  uint l = 0, h = 0;
  for (uint b = 0; b < 4; b++) {
    uint byte = (x >> (8 * b)) & 0xff;
    l |= (uint)kv[byte & 15] << (8 * b);
    h |= (uint)kv[byte >> 4] << (8 * b);
  }
  *lo = l;
  *hi = h;
}
static inline void iq4_unpack(__global const uchar *qs, uint w[8], __local const uchar *kv) {
  for (uint i = 0; i < 4; i++) iq4_word(dec_u32(qs + 4 * i), &w[i], &w[4 + i], kv);
}

// Q2_0: one byte, 4 values of 2 bits, into 4 signed bytes q - 1.
static inline uint q2_byte(uint x) {
  uint t = (x & 0x03u) | ((x & 0x0cu) << 6) | ((x & 0x30u) << 12) | ((x & 0xc0u) << 18);
  return sub_bytes(t, 0x01010101u);
}

// Q4_0 / Q5_0 / Q5_1: 4 bits of 4 values (one per byte) into the low bits of their bytes.
static inline uint spread4(uint x) {
  return (x & 1u) | ((x & 2u) << 7) | ((x & 4u) << 14) | ((x & 8u) << 21);
}

// Q3_K: the 6-bit scale i of 16 (12 bytes: low nibbles of 0-7, high nibbles of 0-7, then 2-bit tops in 8-11).
static inline int q3k_scale(__global const uchar *sc, uint i) {
  uint lo4 = i < 8 ? sc[i] & 15 : sc[i - 8] >> 4;
  uint hi2 = (sc[8 + (i & 3)] >> (2 * (i / 4))) & 3;
  return (int)(lo4 | (hi2 << 4)) - 32;
}

// IQ1_S / IQ1_M: a grid word (4 values of -1, 0 or 1) times 8, plus or minus 1: 8 (g + delta) with delta = +-1/8.
static inline uint iq1_word(uint g, uint neg) {
  uint x = (g << 3) & 0xf8f8f8f8u;
  return neg ? sub_bytes(x, 0x01010101u) : x | 0x01010101u;
}

static inline void decode_unit(__global const uchar *W, ulong n, ulong k, uint s, uint w[8], float *lo, float *hi,
                               float *mn GRID_PARAM) {
  *mn = 0.0f;
#if QTYPE == QT_Q8_0
  // Q8_0: y = d * q.
  __global const uchar *qs = FIELD(0, k);
  for (uint i = 0; i < 8; i++) w[i] = dec_u32(qs + 4 * i);
  *lo = *hi = ld_half(FIELD(1, k));
#elif QTYPE == QT_Q4_0 || QTYPE == QT_Q5_0
  // Q4_0: y = d * (q - 8), byte j holding value j (low nibble) and value j + 16 (high nibble). Q5_0: bit j of qh is
  // bit 4 of value j; y = d * (q - 16).
  __global const uchar *qs = FIELD(0, k);
#if QTYPE == QT_Q5_0
  uint qh = dec_u32(FIELD(1, k));
  for (uint i = 0; i < 4; i++) {
    uint x = dec_u32(qs + 4 * i);
    w[i] = sub_bytes((x & 0x0f0f0f0fu) | (spread4(qh >> (4 * i)) << 4), 0x10101010u);
    w[4 + i] = sub_bytes(((x >> 4) & 0x0f0f0f0fu) | (spread4(qh >> (16 + 4 * i)) << 4), 0x10101010u);
  }
  *lo = *hi = ld_half(FIELD(2, k));
#else
  for (uint i = 0; i < 4; i++) {
    uint x = dec_u32(qs + 4 * i);
    w[i] = sub_bytes(x & 0x0f0f0f0fu, 0x08080808u);
    w[4 + i] = sub_bytes((x >> 4) & 0x0f0f0f0fu, 0x08080808u);
  }
  *lo = *hi = ld_half(FIELD(1, k));
#endif
#elif QTYPE == QT_Q5_1
  // Q5_1: Q5_0's bits without the offset of 16, and a min; y = d * q + m.
  __global const uchar *qs = FIELD(0, k);
  uint qh = dec_u32(FIELD(1, k));
  for (uint i = 0; i < 4; i++) {
    uint x = dec_u32(qs + 4 * i);
    w[i] = (x & 0x0f0f0f0fu) | (spread4(qh >> (4 * i)) << 4);
    w[4 + i] = ((x >> 4) & 0x0f0f0f0fu) | (spread4(qh >> (16 + 4 * i)) << 4);
  }
  __global const uchar *dm = FIELD(2, k);
  *lo = *hi = ld_half(dm);
  *mn = -ld_half(dm + 2);
#elif QTYPE == QT_Q3_K
  // Q3_K: y = d * (scale - 32) * (q - 4 * !h); unit s takes bits 2(s mod 4) of qs bytes 32(s / 4) .. + 31, and bit
  // s of the hmask bytes; its two halves of 16 have scales 2s and 2s + 1.
  __global const uchar *qs = FIELD(0, k) + 32 * (s / 4);
  __global const uchar *hm = FIELD(1, k);
  uint shift = 2 * (s % 4);
  for (uint i = 0; i < 8; i++) {
    uint low = (dec_u32(qs + 4 * i) >> shift) & 0x03030303u;
    uint h = (dec_u32(hm + 4 * i) >> s) & 0x01010101u;
    w[i] = sub_bytes(low | (h << 2), 0x04040404u);
  }
  float d = ld_half(FIELD(3, k));
  *lo = d * (float)q3k_scale(FIELD(2, k), 2 * s);
  *hi = d * (float)q3k_scale(FIELD(2, k), 2 * s + 1);
#elif QTYPE == QT_IQ2_XXS
  // IQ2_XXS: y = d * (0.5 + s) / 4 * grid byte * sign; 4 8-bit indices, then 4 x 7 sign bits (8th for even parity)
  // and the 4-bit scale.
  __global const uchar *qs = FIELD(0, k) + 8 * s;
  uint idx = dec_u32(qs), aux = dec_u32(qs + 4);
  for (uint l = 0; l < 4; l++) {
    uint2 e = *(__local const uint2 *)(grid + 8 * ((idx >> (8 * l)) & 255));
    uint s7 = (aux >> (7 * l)) & 127;
    uint signs = s7 | ((popcount(s7) & 1) << 7);
    w[2 * l] = apply_signs4(e.x, signs & 15);
    w[2 * l + 1] = apply_signs4(e.y, signs >> 4);
  }
  *lo = *hi = ld_half(FIELD(1, k)) * (0.5f + (float)(aux >> 28)) * 0.25f;
#elif QTYPE == QT_IQ2_XS
  // IQ2_XS: as IQ2_XXS, but each 16-bit word holds a 9-bit index and 7 sign bits, and the scale is a nibble per half.
  __global const uchar *qs = FIELD(0, k) + 8 * s;
  for (uint l = 0; l < 4; l++) {
    uint v = ld_u16(qs + 2 * l);
    uint2 e = *(__local const uint2 *)(grid + 8 * (v & 511));
    uint s7 = v >> 9;
    uint signs = s7 | ((popcount(s7) & 1) << 7);
    w[2 * l] = apply_signs4(e.x, signs & 15);
    w[2 * l + 1] = apply_signs4(e.y, signs >> 4);
  }
  uint sc = FIELD(1, k)[s];
  float d = ld_half(FIELD(2, k)) * 0.25f;
  *lo = d * (0.5f + (float)(sc & 15));
  *hi = d * (0.5f + (float)(sc >> 4));
#elif QTYPE == QT_IQ1_S
  // IQ1_S: as IQ1_M, with one 3-bit scale and one delta sign a unit, in bits 12-14 and 15 of its 16-bit qh word, whose
  // bits 3l .. 3l + 2 are the high bits of index l; d is a field of its own.
  __global const uchar *qs = FIELD(0, k) + 4 * s;
  uint qh = ld_u16(FIELD(1, k) + 2 * s);
  for (uint l = 0; l < 4; l++) {
    uint2 e = *(__local const uint2 *)(grid + 8 * (qs[l] | (((qh >> (3 * l)) & 7) << 8)));
    w[2 * l] = iq1_word(e.x, qh & 0x8000);
    w[2 * l + 1] = iq1_word(e.y, qh & 0x8000);
  }
  *lo = *hi = ld_half(FIELD(2, k)) * (float)(2 * ((qh >> 12) & 7) + 1) * 0.125f;
#elif QTYPE == QT_IQ1_M
  // IQ1_M: y = dl * (grid value + delta), delta +-1/8 per 8 values, grid values -1, 0 or 1; so y = dl / 8 * (8 g +- 1),
  // whole numbers. Indices are 8 bits of qs and 3 of qh; d is the top nibbles of the 4 scale words, and the unit's
  // two 3-bit scales are bits 6(s mod 2) of word s / 2.
  __global const uchar *qs = FIELD(0, k) + 4 * s;
  __global const uchar *qh = FIELD(1, k) + 2 * s;
  __global const uchar *scw = FIELD(2, k);
  uint s0 = ld_u16(scw), s1 = ld_u16(scw + 2), s2 = ld_u16(scw + 4), s3 = ld_u16(scw + 6);
  float d = (float)as_half((ushort)((s0 >> 12) | ((s1 >> 8) & 0x00f0u) | ((s2 >> 4) & 0x0f00u) | (s3 & 0xf000u)));
  uint ss = ld_u16(scw + 2 * (s / 2)) >> (6 * (s & 1));
  for (uint l = 0; l < 4; l++) {
    uint h = qh[l / 2] >> (4 * (l & 1));
    uint2 e = *(__local const uint2 *)(grid + 8 * (qs[l] | ((h & 7) << 8)));
    w[2 * l] = iq1_word(e.x, h & 8);
    w[2 * l + 1] = iq1_word(e.y, h & 8);
  }
  *lo = d * (float)(2 * (ss & 7) + 1) * 0.125f;
  *hi = d * (float)(2 * ((ss >> 3) & 7) + 1) * 0.125f;
#elif QTYPE == QT_Q2_0
  // Q2_0: value j is bits 2(j mod 4) of byte j / 4; y = (q - 1) * d.
  __global const uchar *qs = FIELD(0, k) + 8 * s;
  for (uint i = 0; i < 8; i++) w[i] = q2_byte(qs[i]);
  *lo = *hi = ld_half(FIELD(1, k));
#elif QTYPE == QT_IQ4_NL
  // IQ4_NL: y = d * kvalues[nibble].
  iq4_unpack(FIELD(0, k), w, grid);
  *lo = *hi = ld_half(FIELD(1, k));
#elif QTYPE == QT_IQ4_XS
  // IQ4_XS: y = d * (ls - 32) * kvalues[nibble], ls the 6-bit scale of sub-block s.
  iq4_unpack(FIELD(0, k) + 16 * s, w, grid);
  uint sl = (FIELD(1, k)[s / 2] >> (4 * (s & 1))) & 15;
  uint sh = (ld_u16(FIELD(2, k)) >> (2 * s)) & 3;
  *lo = *hi = ld_half(FIELD(3, k)) * (float)((int)(sl | (sh << 4)) - 32);
#elif QTYPE == QT_Q4_K || QTYPE == QT_Q5_K
  // Q4_K: y = d * sc * q - dmin * m; Q5_K adds bit 4 of q from qh.
  __global const uchar *qs = FIELD(0, k) + 32 * (s / 2);
  uint shift = 4 * (s & 1);
  for (uint i = 0; i < 8; i++) w[i] = (dec_u32(qs + 4 * i) >> shift) & 0x0f0f0f0fu;
#if QTYPE == QT_Q5_K
  __global const uchar *qh = FIELD(1, k);
  for (uint i = 0; i < 8; i++) w[i] |= ((dec_u32(qh + 4 * i) >> s) & 0x01010101u) << 4;
  __global const uchar *sc = FIELD(2, k);
  __global const uchar *dm = FIELD(3, k);
#else
  __global const uchar *sc = FIELD(1, k);
  __global const uchar *dm = FIELD(2, k);
#endif
  uint scale, minv;
  if (s < 4) {
    scale = sc[s] & 63;
    minv = sc[s + 4] & 63;
  } else {
    scale = (sc[s + 4] & 15) | ((sc[s - 4] >> 6) << 4);
    minv = (sc[s + 4] >> 4) | ((sc[s] >> 6) << 4);
  }
  *lo = *hi = ld_half(dm) * (float)scale;
  *mn = ld_half(dm + 2) * (float)minv;
#elif QTYPE == QT_Q6_K
  // Q6_K: y = d * scales[v / 16] * (q - 32), q from 4 low bits in ql and 2 high bits in qh.
  uint h = s / 4, q4 = s % 4;
  __global const uchar *ql = FIELD(0, k) + 64 * h + 32 * (q4 & 1);
  __global const uchar *qh = FIELD(1, k) + 32 * h;
  uint lshift = 4 * (q4 >> 1), hshift = 2 * q4;
  for (uint i = 0; i < 8; i++) {
    uint low = (dec_u32(ql + 4 * i) >> lshift) & 0x0f0f0f0fu;
    uint high = ((dec_u32(qh + 4 * i) >> hshift) & 0x03030303u) << 4;
    w[i] = sub_bytes(low | high, 0x20202020u);
  }
  __global const char *sc = (__global const char *)(FIELD(2, k));
  float d = ld_half(FIELD(3, k));
  *lo = d * (float)sc[2 * s];
  *hi = d * (float)sc[2 * s + 1];
#elif QTYPE == QT_IQ2_S
  // IQ2_S: y = d * (0.5 + s) / 4 * grid[index][j] * sign; groups 0-1 take the low scale nibble, 2-3 the high.
  __global const uchar *qs = FIELD(0, k) + 4 * s;
  __global const uchar *sg = FIELD(1, k) + 4 * s;
  uint qh = FIELD(2, k)[s];
  uint sc = FIELD(3, k)[s];
  for (uint g = 0; g < 4; g++) {
    uint idx = qs[g] | (((qh >> (2 * g)) & 3) << 8);
    __local const uint *e = (__local const uint *)(grid + 8 * idx);
    uint signs = sg[g];
    w[2 * g] = apply_signs4(e[0], signs & 15);
    w[2 * g + 1] = apply_signs4(e[1], signs >> 4);
  }
  float d = ld_half(FIELD(4, k)) * 0.25f;
  *lo = d * (0.5f + (float)(sc & 15));
  *hi = d * (0.5f + (float)(sc >> 4));
#elif QTYPE == QT_IQ3_XXS
  // IQ3_XXS: y = d * (0.5 + s) / 2 * grid byte * sign; signs from 7-bit indices with the 8th bit for even parity.
  __global const uchar *qs = FIELD(0, k) + 8 * s;
  uint aux = dec_u32(FIELD(1, k) + 4 * s);
  __local const uint *g = (__local const uint *)grid;
  for (uint l = 0; l < 4; l++) {
    uint s7 = (aux >> (7 * l)) & 127;
    uint signs = s7 | ((popcount(s7) & 1) << 7);
    w[2 * l] = apply_signs4(g[qs[2 * l]], signs & 15);
    w[2 * l + 1] = apply_signs4(g[qs[2 * l + 1]], signs >> 4);
  }
  *lo = *hi = ld_half(FIELD(2, k)) * (0.5f + (float)(aux >> 28)) * 0.5f;
#elif QTYPE == QT_IQ3_S
  // IQ3_S: y = d * (1 + 2s) * grid byte * sign; 9-bit indices, plain sign bits.
  __global const uchar *qs = FIELD(0, k) + 8 * s;
  __global const uchar *sg = FIELD(1, k) + 4 * s;
  uint qh = FIELD(2, k)[s];
  uint sc = (FIELD(3, k)[s / 2] >> (4 * (s & 1))) & 15;
  __local const uint *g = (__local const uint *)grid;
  for (uint l = 0; l < 4; l++) {
    uint signs = sg[l];
    uint i0 = qs[2 * l] | (((qh >> (2 * l)) & 1) << 8);
    uint i1 = qs[2 * l + 1] | (((qh >> (2 * l + 1)) & 1) << 8);
    w[2 * l] = apply_signs4(g[i0], signs & 15);
    w[2 * l + 1] = apply_signs4(g[i1], signs >> 4);
  }
  *lo = *hi = ld_half(FIELD(4, k)) * (float)(1 + 2 * sc);
#else
#error "decode_unit: QTYPE not handled"
#endif
}

// The activation format "a8" for a vector of K values: K int8 codes, then K / 32 float scales.
#define A8_BYTES(K) ((K) + (K) / 8)

// One unit of activations: 8 words of codes and the scale.
static inline float load_act(__global const uchar *a, uint K, uint u, uint q[8]) {
  __global const uint *p = (__global const uint *)(a + 32 * u);
  for (uint i = 0; i < 8; i++) q[i] = p[i];
  return ((__global const float *)(a + K))[u];
}

static inline float unit_dot(const uint w[8], const uint q[8], float lo, float hi, float mn) {
  int s0 = 0, s1 = 0;
  for (uint i = 0; i < 4; i++) s0 += dot4(w[i], q[i]);
  for (uint i = 4; i < 8; i++) s1 += dot4(w[i], q[i]);
  float r = lo * (float)s0 + hi * (float)s1;
  if (mn != 0.0f) {
    int sa = 0;
    for (uint i = 0; i < 8; i++) sa += dot4(q[i], 0x01010101u);
    r -= mn * (float)sa;
  }
  return r;
}


// ---------------------------------------------------------------------------------------------------------------
// The same unit as 16 fp16 pairs, the B operand of two dpas calls (pairs 0-7: values 0-15), for the prompt path. A
// word's bytes b0 b1 b2 b3 become pairs (b0, b2) and (b1, b3): OR-ing bytes into 0x6400 gives halves 1024 + b, two at
// once. So values come in sigma order per 4, and so must the activations (p_moe_scatter's x, qgemm.cl's h).

// sigma: swaps the middle two of every four. Its own inverse.
static inline uint sigma4(uint i) { return (i & ~3u) | ((i & 1u) << 1) | ((i >> 1) & 1u); }

// Unsigned bytes (magnitudes) to two half2 words, (b0, b2) and (b1, b3), each times sc, with the sign bits of
// `signs` (bit j for byte j) applied.
static inline void mag4_pairs(uint x, uint signs, half sc, uint *p0, uint *p1) {
  half2 a = as_half2((x & 0x00ff00ffu) | 0x64006400u) - (half2)(1024.0h);
  half2 b = as_half2(((x >> 8) & 0x00ff00ffu) | 0x64006400u) - (half2)(1024.0h);
  uint m0 = ((signs & 1u) << 15) | ((signs & 4u) << 29);
  uint m1 = ((signs & 2u) << 14) | ((signs & 8u) << 28);
  *p0 = as_uint(a * (half2)sc) ^ m0;
  *p1 = as_uint(b * (half2)sc) ^ m1;
}

// Signed bytes, times sc, minus mn.
static inline void int4_pairs(uint x, half sc, half mn, uint *p0, uint *p1) {
  x ^= 0x80808080u;
  half2 a = as_half2((x & 0x00ff00ffu) | 0x64006400u) - (half2)(1152.0h);
  half2 b = as_half2(((x >> 8) & 0x00ff00ffu) | 0x64006400u) - (half2)(1152.0h);
  *p0 = as_uint(a * (half2)sc - (half2)mn);
  *p1 = as_uint(b * (half2)sc - (half2)mn);
}

static inline void decode_unit_pairs(__global const uchar *W, ulong n, ulong k, uint s, uint p[16] GRID_PARAM) {
#if QTYPE == QT_IQ2_S
  __global const uchar *qs = FIELD(0, k) + 4 * s;
  uint signs = dec_u32(FIELD(1, k) + 4 * s);
  uint qh = FIELD(2, k)[s];
  uint sc = FIELD(3, k)[s];
  float d = ld_half(FIELD(4, k)) * 0.25f;
  half dl = convert_half(d * (0.5f + (float)(sc & 15))), dh = convert_half(d * (0.5f + (float)(sc >> 4)));
  __attribute__((opencl_unroll_hint)) for (uint g = 0; g < 4; g++) {
    uint idx = qs[g] | (((qh >> (2 * g)) & 3) << 8);
    uint2 e = *(__local const uint2 *)(grid + 8 * idx);
    uint sg = signs >> (8 * g);
    half dg = g < 2 ? dl : dh;
    mag4_pairs(e.x, sg & 15, dg, &p[4 * g], &p[4 * g + 1]);
    mag4_pairs(e.y, (sg >> 4) & 15, dg, &p[4 * g + 2], &p[4 * g + 3]);
  }
#elif QTYPE == QT_IQ3_XXS
  __global const uchar *qs = FIELD(0, k) + 8 * s;
  uint aux = dec_u32(FIELD(1, k) + 4 * s);
  half dg = convert_half(ld_half(FIELD(2, k)) * (0.5f + (float)(aux >> 28)) * 0.5f);
  __local const uint *g = (__local const uint *)grid;
  __attribute__((opencl_unroll_hint)) for (uint l = 0; l < 4; l++) {
    uint s7 = (aux >> (7 * l)) & 127;
    uint sg = s7 | ((popcount(s7) & 1) << 7);
    mag4_pairs(g[qs[2 * l]], sg & 15, dg, &p[4 * l], &p[4 * l + 1]);
    mag4_pairs(g[qs[2 * l + 1]], sg >> 4, dg, &p[4 * l + 2], &p[4 * l + 3]);
  }
#elif QTYPE == QT_IQ3_S
  __global const uchar *qs = FIELD(0, k) + 8 * s;
  uint signs = dec_u32(FIELD(1, k) + 4 * s);
  uint qh = FIELD(2, k)[s];
  uint sc = (FIELD(3, k)[s / 2] >> (4 * (s & 1))) & 15;
  half dg = convert_half(ld_half(FIELD(4, k)) * (float)(1 + 2 * sc));
  __local const uint *g = (__local const uint *)grid;
  __attribute__((opencl_unroll_hint)) for (uint l = 0; l < 4; l++) {
    uint sg = signs >> (8 * l);
    uint i0 = qs[2 * l] | (((qh >> (2 * l)) & 1) << 8);
    uint i1 = qs[2 * l + 1] | (((qh >> (2 * l + 1)) & 1) << 8);
    mag4_pairs(g[i0], sg & 15, dg, &p[4 * l], &p[4 * l + 1]);
    mag4_pairs(g[i1], (sg >> 4) & 15, dg, &p[4 * l + 2], &p[4 * l + 3]);
  }
#else
  uint w[8];
  float lo, hi, mn;
  decode_unit(W, n, k, s, w, &lo, &hi, &mn GRID_ARG);
  half hl = convert_half(lo), hh = convert_half(hi), hm = convert_half(mn);
  __attribute__((opencl_unroll_hint)) for (uint i = 0; i < 8; i++)
    int4_pairs(w[i], i < 4 ? hl : hh, hm, &p[2 * i], &p[2 * i + 1]);
#endif
}


// ---------------------------------------------------------------------------------------------------------------
// The experts' types split into unit_raw (a unit's raw fields) and raw_pairs (decoding them), so a kernel loads the
// next unit while it decodes this one. Only quant_pairs_sigma's types (shared/quant.h) decode to pairs: IQ4_NL, IQ4_XS
// and Q2_0 are faster decoded to bytes (an unrecorded probe), so their raw_pairs branches below are not built.
#if QTYPE == QT_IQ2_S || QTYPE == QT_IQ3_XXS || QTYPE == QT_IQ3_S || QTYPE == QT_IQ4_NL || QTYPE == QT_IQ4_XS || \
    QTYPE == QT_Q2_0
static inline uint8 unit_raw(__global const uchar *W, ulong n, ulong k, uint s) {
  uint8 r = (uint8)(0);
#if QTYPE == QT_IQ2_S
  r.s0 = dec_u32(FIELD(0, k) + 4 * s);
  r.s1 = dec_u32(FIELD(1, k) + 4 * s);
  r.s2 = (uint)FIELD(2, k)[s] | ((uint)FIELD(3, k)[s] << 8);
  r.s3 = ld_u16(FIELD(4, k));
#elif QTYPE == QT_IQ3_XXS
  r.s0 = dec_u32(FIELD(0, k) + 8 * s);
  r.s1 = dec_u32(FIELD(0, k) + 8 * s + 4);
  r.s2 = dec_u32(FIELD(1, k) + 4 * s);
  r.s3 = ld_u16(FIELD(2, k));
#elif QTYPE == QT_IQ3_S
  r.s0 = dec_u32(FIELD(0, k) + 8 * s);
  r.s1 = dec_u32(FIELD(0, k) + 8 * s + 4);
  r.s2 = dec_u32(FIELD(1, k) + 4 * s);
  r.s3 = (uint)FIELD(2, k)[s] | ((uint)((FIELD(3, k)[s / 2] >> (4 * (s & 1))) & 15) << 8);
  r.s4 = ld_u16(FIELD(4, k));
#elif QTYPE == QT_IQ4_NL
  for (uint i = 0; i < 4; i++) r[i] = dec_u32(FIELD(0, k) + 4 * i);
  r.s4 = ld_u16(FIELD(1, k));
#elif QTYPE == QT_IQ4_XS
  for (uint i = 0; i < 4; i++) r[i] = dec_u32(FIELD(0, k) + 16 * s + 4 * i);
  uint sl = (FIELD(1, k)[s / 2] >> (4 * (s & 1))) & 15;
  uint sh = (ld_u16(FIELD(2, k)) >> (2 * s)) & 3;
  r.s4 = sl | (sh << 4);
  r.s5 = ld_u16(FIELD(3, k));
#elif QTYPE == QT_Q2_0
  r.s0 = dec_u32(FIELD(0, k) + 8 * s);
  r.s1 = dec_u32(FIELD(0, k) + 8 * s + 4);
  r.s2 = ld_u16(FIELD(1, k));
#endif
  return r;
}
#endif

#if QTYPE == QT_IQ2_S || QTYPE == QT_IQ3_XXS || QTYPE == QT_IQ3_S
#define HAS_UNIT_RAW 1

#if QTYPE == QT_IQ2_S || QTYPE == QT_IQ3_S
// The fields every unit of a block shares (qh, scales and d), loaded once a block by block_raw; unit_raw_in loads only
// the unit's own fields and takes the rest from them. In the runs layout each field is a run of its own, so unit_raw
// keeps five lines of each row in use for both matrices of 16 rows, more than L1 holds for the sub-groups an Xe core
// runs (docs/measurements/MEASUREMENTS.md#2026-10-15-prompt-expert-block-fields). IQ3_XXS shares only d and gains nothing from it.
#define HAS_BLOCK_RAW 1

static inline uint8 block_raw(__global const uchar *W, ulong n, ulong k) {
  uint8 x = (uint8)(0);
  x.s0 = dec_u32(FIELD(2, k));
  x.s1 = dec_u32(FIELD(2, k) + 4);
#if QTYPE == QT_IQ2_S
  x.s2 = dec_u32(FIELD(3, k));
  x.s3 = dec_u32(FIELD(3, k) + 4);
  x.s4 = ld_u16(FIELD(4, k));
#else
  x.s2 = dec_u32(FIELD(3, k));
  x.s4 = ld_u16(FIELD(4, k));
#endif
  return x;
}

// unit_raw's result for unit s of block k, whose shared fields block_raw gave as x.
static inline uint8 unit_raw_in(__global const uchar *W, ulong n, ulong k, uint s, uint8 x) {
  uint8 r = (uint8)(0);
  uint qh = ((s < 4 ? x.s0 : x.s1) >> (8 * (s & 3))) & 255;
#if QTYPE == QT_IQ2_S
  r.s0 = dec_u32(FIELD(0, k) + 4 * s);
  r.s1 = dec_u32(FIELD(1, k) + 4 * s);
  r.s2 = qh | ((((s < 4 ? x.s2 : x.s3) >> (8 * (s & 3))) & 255) << 8);
  r.s3 = x.s4;
#else
  r.s0 = dec_u32(FIELD(0, k) + 8 * s);
  r.s1 = dec_u32(FIELD(0, k) + 8 * s + 4);
  r.s2 = dec_u32(FIELD(1, k) + 4 * s);
  r.s3 = qh | (((x.s2 >> (4 * s)) & 15) << 8);
  r.s4 = x.s4;
#endif
  return r;
}
#endif

static inline half h_of(uint bits) { return as_half((ushort)bits); }

// tbl: lane i holds the codebook's value i as a half (IQ4_NL and IQ4_XS), for shuffles by nibble.
static inline void raw_pairs(uint8 r, uint p[16], half tbl GRID_PARAM) {
#if QTYPE == QT_IQ2_S
  float d = (float)h_of(r.s3) * 0.25f;
  uint sc = r.s2 >> 8, qh = r.s2 & 255;
  half dl = convert_half(d * (0.5f + (float)(sc & 15))), dh = convert_half(d * (0.5f + (float)(sc >> 4)));
  __attribute__((opencl_unroll_hint)) for (uint g = 0; g < 4; g++) {
    uint idx = ((r.s0 >> (8 * g)) & 255) | (((qh >> (2 * g)) & 3) << 8);
    uint2 e = *(__local const uint2 *)(grid + 8 * idx);
    uint sg = r.s1 >> (8 * g);
    half dg = g < 2 ? dl : dh;
    mag4_pairs(e.x, sg & 15, dg, &p[4 * g], &p[4 * g + 1]);
    mag4_pairs(e.y, (sg >> 4) & 15, dg, &p[4 * g + 2], &p[4 * g + 3]);
  }
#elif QTYPE == QT_IQ3_XXS
  uint aux = r.s2;
  half dg = convert_half((float)h_of(r.s3) * (0.5f + (float)(aux >> 28)) * 0.5f);
  __local const uint *g = (__local const uint *)grid;
  __attribute__((opencl_unroll_hint)) for (uint l = 0; l < 4; l++) {
    uint q2 = l < 2 ? r.s0 : r.s1;
    uint s7 = (aux >> (7 * l)) & 127;
    uint sg = s7 | ((popcount(s7) & 1) << 7);
    mag4_pairs(g[(q2 >> (16 * (l & 1))) & 255], sg & 15, dg, &p[4 * l], &p[4 * l + 1]);
    mag4_pairs(g[(q2 >> (16 * (l & 1) + 8)) & 255], sg >> 4, dg, &p[4 * l + 2], &p[4 * l + 3]);
  }
#elif QTYPE == QT_IQ3_S
  uint qh = r.s3 & 255, sc = r.s3 >> 8;
  half dg = convert_half((float)h_of(r.s4) * (float)(1 + 2 * sc));
  __local const uint *g = (__local const uint *)grid;
  __attribute__((opencl_unroll_hint)) for (uint l = 0; l < 4; l++) {
    uint q2 = l < 2 ? r.s0 : r.s1;
    uint sg = r.s2 >> (8 * l);
    uint i0 = ((q2 >> (16 * (l & 1))) & 255) | (((qh >> (2 * l)) & 1) << 8);
    uint i1 = ((q2 >> (16 * (l & 1) + 8)) & 255) | (((qh >> (2 * l + 1)) & 1) << 8);
    mag4_pairs(g[i0], sg & 15, dg, &p[4 * l], &p[4 * l + 1]);
    mag4_pairs(g[i1], (sg >> 4) & 15, dg, &p[4 * l + 2], &p[4 * l + 3]);
  }
#elif QTYPE == QT_IQ4_NL || QTYPE == QT_IQ4_XS
#if QTYPE == QT_IQ4_NL
  half2 d2 = (half2)h_of(r.s4);
#else
  half2 d2 = (half2)convert_half((float)h_of(r.s5) * (float)((int)r.s4 - 32));
#endif
  // Byte j holds value j (low nibble) and value j + 16 (high nibble); pairs in sigma order: (v0, v2), (v1, v3), ...
  __attribute__((opencl_unroll_hint)) for (uint i = 0; i < 4; i++) {
    uint x = r[i];
    __attribute__((opencl_unroll_hint)) for (uint hn = 0; hn < 2; hn++) {
      uint sh = 4 * hn;
      half a0 = intel_sub_group_shuffle(tbl, (x >> sh) & 15), a2 = intel_sub_group_shuffle(tbl, (x >> (16 + sh)) & 15);
      half a1 = intel_sub_group_shuffle(tbl, (x >> (8 + sh)) & 15), a3 = intel_sub_group_shuffle(tbl, (x >> (24 + sh)) & 15);
      p[8 * hn + 2 * i] = as_uint((half2)(a0, a2) * d2);
      p[8 * hn + 2 * i + 1] = as_uint((half2)(a1, a3) * d2);
    }
  }
#elif QTYPE == QT_Q2_0
  // Value j is bits 2 (j mod 4) of byte j / 4, y = (q - 1) d: from byte b, the pairs (v0, v2) and (v1, v3).
  half2 d2 = (half2)h_of(r.s2);
  __attribute__((opencl_unroll_hint)) for (uint b = 0; b < 8; b++) {
    uint x = ((b < 4 ? r.s0 : r.s1) >> (8 * (b & 3))) & 255;
    uint e0 = (x & 0x03u) | ((x & 0x30u) << 12), e1 = ((x >> 2) & 0x03u) | (((x >> 2) & 0x30u) << 12);
    p[2 * b] = as_uint((as_half2(e0 | 0x64006400u) - (half2)(1025.0h)) * d2);
    p[2 * b + 1] = as_uint((as_half2(e1 | 0x64006400u) - (half2)(1025.0h)) * d2);
  }
#endif
}

// The codebook's values as halves, one per lane (IQ4 types): for raw_pairs' shuffles.
static inline half raw_table(GRID_ONLY_PARAM) {
#if QTYPE == QT_IQ4_NL || QTYPE == QT_IQ4_XS
  return convert_half((float)((__local const char *)grid)[sg_lane()]);
#else
  return 0.0h;
#endif
}
#endif

#if QTYPE == QT_IQ4_NL || QTYPE == QT_Q2_0
#define HAS_RAW_WORDS 1
// decode_unit's words and scales from unit_raw's fields.
static inline void raw_words(uint8 r, uint w[8], float *lo, float *hi, float *mn GRID_PARAM) {
  *mn = 0.0f;
#if QTYPE == QT_IQ4_NL
  for (uint i = 0; i < 4; i++) iq4_word(r[i], &w[i], &w[4 + i], grid);
  *lo = *hi = (float)as_half((ushort)r.s4);
#else
  for (uint i = 0; i < 8; i++) w[i] = q2_byte(((i < 4 ? r.s0 : r.s1) >> (8 * (i & 3))) & 255);
  *lo = *hi = (float)as_half((ushort)r.s2);
#endif
}
#endif

#endif
