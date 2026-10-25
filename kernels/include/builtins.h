// The one header that names Intel's OpenCL builtins (ADR-0003). Each was checked on the B60
// (docs/measurements/MEASUREMENTS.md#2026-10-11-kernel-spike, "Builtins that compiled and gave correct results").
#ifndef KERNELS_BUILTINS_H
#define KERNELS_BUILTINS_H

#include "types.h"

#define SG_SIZE 16
#define REQD_SG __attribute__((intel_reqd_sub_group_size(SG_SIZE)))

// Signed 4 x int8 dot product (DP4A).
static inline int dot4(uint a, uint b) { return dot_4x8packed_ss_int(a, b); }

// The card's clock in cycles of its core clock (cl_khr_kernel_clock). The headers do not declare it; IGC's library
// has it (docs/measurements/MEASUREMENTS.md#2026-10-25-kernel-clock).
ulong __attribute__((overloadable)) clock_read_device(void);
static inline ulong card_clock(void) { return clock_read_device(); }

static inline float sg_sum(float x) { return sub_group_reduce_add(x); }
static inline float sg_max(float x) { return sub_group_reduce_max(x); }
static inline float sg_xor(float x, uint m) { return sub_group_shuffle_xor(x, m); }
static inline ulong sg_shuffle_ul(ulong x, uint lane) { return intel_sub_group_shuffle(x, lane); }
static inline uint sg_lane(void) { return get_sub_group_local_id(); }
static inline uint sg_id(void) { return get_sub_group_id(); }

// f16 at a 2-byte aligned address.
static inline float ld_half(__global const uchar *p) { return vload_half(0, (__global const half *)p); }
static inline uint ld_u32(__global const uchar *p) { return *(__global const uint *)p; }
static inline ushort ld_u16(__global const uchar *p) { return *(__global const ushort *)p; }

// Bytewise a - b on packed int8 (no borrow between bytes), in whole words: bits 0-6 of each byte from one subtraction
// that sets bit 7 of a and clears it in b, so no borrow leaves a byte, then bit 7 as a7 ^ b7 ^ the borrow into it.
// IGC compiles the char4 form to many more instructions (docs/measurements/MEASUREMENTS.md#2026-10-15-sub-bytes-words).
static inline uint sub_bytes(uint a, uint b) {
  return ((a | 0x80808080u) - (b & 0x7f7f7f7fu)) ^ ((a ^ ~b) & 0x80808080u);
}
// Four sign bits to four 0x00 / 0xFF bytes.
static inline uint sign_mask4(uint bits) { return ((bits * 0x00204081u) & 0x01010101u) * 0xffu; }
// Negates the bytes of v whose sign bit is set.
static inline uint apply_signs4(uint v, uint bits) {
  uint m = sign_mask4(bits);
  return sub_bytes(v ^ m, m);
}

static inline float bf16_to_f32(ushort x) { return as_float((uint)x << 16); }

// The XMX units (DPAS), sub-group of 16: C (8 x 16) += A (8 x 16) B (16 x 16). Lane i holds column i of A (a[m] =
// A[m][i]), column i of B as 8 pairs of 16-bit values (B[2p][i], B[2p + 1][i] in b[p], low half first), and column
// i of C (c[m] = C[m][i]). Checked by core/tests/gemm_test.cpp.
static inline float8 dpas_f16(short8 a, int8 b, float8 c) { return intel_sub_group_f16_f16_matrix_mad_k16(a, b, c); }
static inline float8 dpas_bf16(short8 a, int8 b, float8 c) { return intel_sub_group_bf16_bf16_matrix_mad_k16(a, b, c); }

// Sub-group block read: lane i gets p[i]. p must be 4-byte aligned.
static inline ushort sg_read_us(__global const ushort *p) { return intel_sub_group_block_read_us(p); }
// 8 values a lane, 256 bytes: lane i gets (or writes) p[16 j + i] as element j. p must be 16-byte aligned.
// Checked by core/tests/greedy_test.cpp given a prompt chunk, through the prompt path's p_moe_scatter.
static inline ushort8 sg_read_us8(__global const ushort *p) { return intel_sub_group_block_read_us8(p); }
static inline void sg_write_us8(__global ushort *p, ushort8 v) { intel_sub_group_block_write_us8(p, v); }
// Element j of the result: element j of x in lane `from`.
static inline ushort8 sg_shuffle_us8(ushort8 x, uint from) { return intel_sub_group_shuffle(x, from); }

// f32 to bf16, rounding to nearest even, NaN kept quiet (ggml's GGML_FP32_TO_BF16).
static inline ushort f32_to_bf16(float f) {
  uint u = as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return (ushort)((u >> 16) | 64);
  return (ushort)((u + (0x7fffu + ((u >> 16) & 1))) >> 16);
}

// 2D block reads of a 16-bit matrix: rows `pitch` bytes apart (a multiple of 64; base 64-byte aligned), `width` bytes
// and `height` rows in bounds, 0 outside. sg_load_rows32 (the A operand of four dpas calls): rows y0..y0+31, lane i
// column x0 + i, dst[r] = M[y0 + r][x0 + i].
static inline void sg_load_rows32(__global const ushort *base, uint width, uint height, uint pitch, uint x0, uint y0,
                                  ushort dst[32]) {
  intel_sub_group_2d_block_read_16b_32r16x1c((__global void *)base, width, height, pitch, (int2)(x0, y0), dst);
}
// sg_load_cols16 (the B operand of one dpas call): 32-bit pairs x0..x0+7 of rows y0..y0+15, transposed: lane i gets
// row y0 + i, dst[p] = the pair at column x0 + p.
static inline void sg_load_cols16(__global const ushort *base, uint width, uint height, uint pitch, uint x0, uint y0,
                                  uint dst[8]) {
  intel_sub_group_2d_block_read_transpose_32b_16r8x1c((__global void *)base, width, height, pitch, (int2)(x0, y0),
                                                      dst);
}

// 2D block write of a 16-bit matrix, the bounds and alignment as the reads': rows y0..y0+7, columns x0..x0+15, lane
// i writing column x0 + i, M[y0 + r][x0 + i] = src[r] (a dpas tile's layout); writes outside the bounds are dropped.
// Checked by core/tests/gemm_test.cpp.
static inline void sg_store_rows8(__global ushort *base, uint width, uint height, uint pitch, uint x0, uint y0,
                                  ushort src[8]) {
  intel_sub_group_2d_block_write_16b_8r16x1c((__global void *)base, width, height, pitch, (int2)(x0, y0), src);
}

#endif
