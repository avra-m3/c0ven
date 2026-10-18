// a8 activations (kernels/quant/decode.h): a row of K values as K int8 codes, then K / 32 float scales. Scale
// max|x| / 127, codes round(x / scale): the q8_1 rule (docs/spec/QUANT_FORMATS.md), with the scale kept in f32.
#ifndef KERNELS_A8_H
#define KERNELS_A8_H

#include "builtins.h"

// Unit u (values 32 u .. 32 u + 31) of the row at a, by one sub-group of 16: lane l holds values 2 l and 2 l + 1.
static inline void a8_store(float2 v, __global uchar *a, uint K, uint u) {
  const uint lane = sg_lane();
  const float amax = sub_group_reduce_max(fmax(fabs(v.x), fabs(v.y)));
  const float d = amax / 127.0f;
  const float id = d != 0.0f ? 1.0f / d : 0.0f;
  vstore2((char2)((char)round(v.x * id), (char)round(v.y * id)), lane, (__global char *)(a + 32 * u));
  if (lane == 0) ((__global float *)(a + K))[u] = d;
}

#endif
