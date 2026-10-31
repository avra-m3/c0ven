// The weight formats the engine reads, as ggml numbers them, and the "runs" layout the cards hold them in.
// Formats: docs/spec/QUANT_FORMATS.md. The runs layout is its "Wide loads" section: a slice of n blocks is stored as
// one run per field, field f of block k at run_offset(f, n) + k * field_size(f), the largest fields first.
#ifndef SHARED_QUANT_H
#define SHARED_QUANT_H

#include "types.h"

// ggml type ids (ggml/include/ggml.h at the pinned commit).
#define QT_F32 0
#define QT_F16 1
#define QT_Q4_0 2
#define QT_Q5_0 6
#define QT_Q5_1 7
#define QT_Q8_0 8
#define QT_Q8_1 9
#define QT_Q3_K 11
#define QT_Q4_K 12
#define QT_Q5_K 13
#define QT_Q6_K 14
#define QT_Q8_K 15
#define QT_IQ2_XXS 16
#define QT_IQ2_XS 17
#define QT_IQ3_XXS 18
#define QT_IQ1_S 19
#define QT_IQ4_NL 20
#define QT_IQ3_S 21
#define QT_IQ2_S 22
#define QT_IQ4_XS 23
#define QT_IQ1_M 29
#define QT_BF16 30
#define QT_Q2_0 42

// Values per block.
SHARED_FN u32 quant_block_values(u32 t) {
  switch (t) {
    case QT_Q4_0: case QT_Q5_0: case QT_Q5_1: case QT_Q8_0: case QT_Q8_1: case QT_IQ4_NL: return 32;
    case QT_Q2_0: return 64;
    case QT_Q3_K: case QT_Q4_K: case QT_Q5_K: case QT_Q6_K: case QT_Q8_K:
    case QT_IQ1_S: case QT_IQ1_M: case QT_IQ2_XXS: case QT_IQ2_XS:
    case QT_IQ3_XXS: case QT_IQ3_S: case QT_IQ2_S: case QT_IQ4_XS: return 256;
    default: return 1;
  }
}

// Bytes per block; 0 for a type the engine does not read.
SHARED_FN u32 quant_block_bytes(u32 t) {
  switch (t) {
    case QT_F32: return 4;
    case QT_F16: case QT_BF16: return 2;
    case QT_Q4_0: return 18;
    case QT_Q5_0: return 22;
    case QT_Q5_1: return 24;
    case QT_Q8_0: return 34;
    case QT_Q8_1: return 36;
    case QT_Q3_K: return 110;
    case QT_Q4_K: return 144;
    case QT_Q5_K: return 176;
    case QT_Q6_K: return 210;
    case QT_Q8_K: return 292;
    case QT_IQ1_S: return 50;
    case QT_IQ1_M: return 56;
    case QT_IQ2_XXS: return 66;
    case QT_IQ2_XS: return 74;
    case QT_IQ3_XXS: return 98;
    case QT_IQ4_NL: return 18;
    case QT_IQ3_S: return 110;
    case QT_IQ2_S: return 82;
    case QT_IQ4_XS: return 136;
    case QT_Q2_0: return 18;
    default: return 0;
  }
}

// Fields of a block in run order: (offset in the file's block, size). Quants first, f16 scales last.
SHARED_FN u32 quant_nfields(u32 t) {
  switch (t) {
    case QT_IQ2_S: case QT_IQ3_S: return 5;
    case QT_IQ4_XS: case QT_Q3_K: case QT_Q5_K: case QT_Q6_K: return 4;
    case QT_IQ1_S: case QT_IQ1_M: case QT_IQ2_XS: case QT_IQ3_XXS:
    case QT_Q4_K: case QT_Q5_0: case QT_Q5_1: return 3;
    case QT_IQ2_XXS: case QT_IQ4_NL: case QT_Q2_0: case QT_Q4_0: case QT_Q8_0: return 2;
    default: return 1;
  }
}

// Encoded as (offset << 8) | size.
SHARED_FN u32 quant_field_(u32 t, u32 f) {
  switch (t) {
    case QT_IQ2_S: {  // qs (grid index low bits), signs, qh, scales, d
      const u32 v[5] = {(2u << 8) | 32, (34u << 8) | 32, (66u << 8) | 8, (74u << 8) | 8, (0u << 8) | 2};
      return v[f];
    }
    case QT_IQ1_S: {  // qs (grid index low bits), qh (index high bits, scale and delta sign), d
      const u32 v[3] = {(2u << 8) | 32, (34u << 8) | 16, (0u << 8) | 2};
      return v[f];
    }
    case QT_IQ1_M: {  // qs (grid index low bits), qh, scales (with d in their top nibbles)
      const u32 v[3] = {(0u << 8) | 32, (32u << 8) | 16, (48u << 8) | 8};
      return v[f];
    }
    case QT_IQ2_XXS: {  // grid indices and scale-and-sign words, d
      const u32 v[2] = {(2u << 8) | 64, (0u << 8) | 2};
      return v[f];
    }
    case QT_IQ2_XS: {  // 9-bit grid indices with 7 sign bits, scales, d
      const u32 v[3] = {(2u << 8) | 64, (66u << 8) | 8, (0u << 8) | 2};
      return v[f];
    }
    case QT_IQ3_XXS: {  // grid indices, scale-and-sign words, d
      const u32 v[3] = {(2u << 8) | 64, (66u << 8) | 32, (0u << 8) | 2};
      return v[f];
    }
    case QT_IQ3_S: {  // qs, signs, qh, scales, d
      const u32 v[5] = {(2u << 8) | 64, (74u << 8) | 32, (66u << 8) | 8, (106u << 8) | 4, (0u << 8) | 2};
      return v[f];
    }
    case QT_IQ4_XS: {  // qs, scales_l, scales_h, d
      const u32 v[4] = {(8u << 8) | 128, (4u << 8) | 4, (2u << 8) | 2, (0u << 8) | 2};
      return v[f];
    }
    case QT_IQ4_NL: case QT_Q2_0: {  // qs, d
      const u32 v[2] = {(2u << 8) | 16, (0u << 8) | 2};
      return v[f];
    }
    case QT_Q4_0: {  // qs, d
      const u32 v[2] = {(2u << 8) | 16, (0u << 8) | 2};
      return v[f];
    }
    case QT_Q5_0: {  // qs, qh, d
      const u32 v[3] = {(6u << 8) | 16, (2u << 8) | 4, (0u << 8) | 2};
      return v[f];
    }
    case QT_Q5_1: {  // qs, qh, d and m
      const u32 v[3] = {(8u << 8) | 16, (4u << 8) | 4, (0u << 8) | 4};
      return v[f];
    }
    case QT_Q8_0: {  // qs, d
      const u32 v[2] = {(2u << 8) | 32, (0u << 8) | 2};
      return v[f];
    }
    case QT_Q3_K: {  // qs, hmask, scales, d
      const u32 v[4] = {(32u << 8) | 64, (0u << 8) | 32, (96u << 8) | 12, (108u << 8) | 2};
      return v[f];
    }
    case QT_Q4_K: {  // qs, scales, d and dmin
      const u32 v[3] = {(16u << 8) | 128, (4u << 8) | 12, (0u << 8) | 4};
      return v[f];
    }
    case QT_Q5_K: {  // qs, qh, scales, d and dmin
      const u32 v[4] = {(48u << 8) | 128, (16u << 8) | 32, (4u << 8) | 12, (0u << 8) | 4};
      return v[f];
    }
    case QT_Q6_K: {  // ql, qh, scales, d
      const u32 v[4] = {(0u << 8) | 128, (128u << 8) | 64, (192u << 8) | 16, (208u << 8) | 2};
      return v[f];
    }
    default: return quant_block_bytes(t);
  }
}

SHARED_FN u32 quant_field_offset(u32 t, u32 f) { return quant_field_(t, f) >> 8; }
SHARED_FN u32 quant_field_size(u32 t, u32 f) { return quant_field_(t, f) & 0xff; }

// Where field f's run starts in a slice of n blocks.
SHARED_FN u64 quant_run_offset(u32 t, u32 f, u64 n) {
  u64 off = 0;
  for (u32 g = 0; g < f; g++) off += quant_field_size(t, g);
  return off * n;
}

// Whether the prompt path's expert kernels (kernels/quant/qgemm.cl) take a type's values in the order sigma, the
// middle two of every four swapped (kernels/quant/decode.h, decode_unit_pairs), or in their own order. Their
// activations are stored in the same order.
SHARED_FN u32 quant_pairs_sigma(u32 t) { return t == QT_IQ2_S || t == QT_IQ3_XXS || t == QT_IQ3_S; }

// The codebooks, packed into one buffer the core uploads to each card (from ggml-common.h at the pinned commit;
// QUANT_FORMATS.md, "The codebooks").
#define QGRID_IQ2S 0            // u64 x 1024
#define QGRID_IQ3XXS 8192       // u32 x 256
#define QGRID_IQ3S 9216         // u32 x 512
#define QGRID_KVALUES 11264     // i8 x 16 (kvalues_iq4nl)
#define QGRID_IQ2XXS 11280      // u64 x 256
#define QGRID_IQ2XS 13328       // u64 x 512
#define QGRID_IQ1S 17424        // u64 x 2048 (iq1s_grid, IQ1_S's and IQ1_M's)
#define QGRID_BYTES 33808

#endif
