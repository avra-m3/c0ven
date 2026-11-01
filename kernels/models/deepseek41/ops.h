// What deepseek41's kernel files share: sub-group block moves of f32 values, the sum of posted sums and a8's store.
#ifndef DEEPSEEK41_OPS_H
#define DEEPSEEK41_OPS_H

#include "builtins.h"

// Lane i holds p[i], or p[16 j + i] as element j. Reads need 4-byte alignment, writes 16-byte.
static inline float rd1(__global const float *p) {
  return as_float(intel_sub_group_block_read((__global const uint *)p));
}
static inline float2 rd2(__global const float *p) {
  return as_float2(intel_sub_group_block_read2((__global const uint *)p));
}
static inline float4 rd4(__global const float *p) {
  return as_float4(intel_sub_group_block_read4((__global const uint *)p));
}
static inline float8 rd8(__global const float *p) {
  return as_float8(intel_sub_group_block_read8((__global const uint *)p));
}
static inline void wr1(__global float *p, float v) { intel_sub_group_block_write((__global uint *)p, as_uint(v)); }
static inline void wr2(__global float *p, float2 v) { intel_sub_group_block_write2((__global uint *)p, as_uint2(v)); }
static inline void wr4(__global float *p, float4 v) { intel_sub_group_block_write4((__global uint *)p, as_uint4(v)); }
static inline void wr8(__global float *p, float8 v) { intel_sub_group_block_write8((__global uint *)p, as_uint8(v)); }

static inline float sum4(float4 v) { return (v.s0 + v.s1) + (v.s2 + v.s3); }

// The total of the sub-groups' posted sums, after the barrier: lanes read the posts together, in a fixed order.
static inline float posted_sum(__local const float *red, uint nsg) {
  float t = 0.0f;
  for (uint i = sg_lane(); i < nsg; i += SG_SIZE) t += red[i];
  return sg_sum(t);
}

// a8 unit u of a row of K values at a (a8.h's rule), from 32 values held as element j of lane i = value 16 j + i.
static inline void a8_store_block(float2 v, __global uchar *a, uint K, uint u) {
  const float amax = sub_group_reduce_max(fmax(fabs(v.x), fabs(v.y)));
  const float d = amax / 127.0f;
  const float id = d != 0.0f ? 1.0f / d : 0.0f;
  intel_sub_group_block_write_uc2(a + 32 * u, as_uchar2(convert_char2(round(v * id))));
  if (sg_lane() == 0) ((__global float *)(a + K))[u] = d;
}

#endif
