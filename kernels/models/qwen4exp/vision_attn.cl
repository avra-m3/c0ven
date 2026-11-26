// Vision attention kernel (docs/spec/MODEL_VISION.md, section 3.2) for Xe2 16-lane XMX units.
#include "builtins.h"
#include "vision_sizes.h"

#define V_QKV_W (V_H * V_HDP)

// o = softmax(scale q k^T) v per head: nq query rows, all npos keys, no mask. q, k, v: rows of 16 heads x V_HDP fp16
// (v_qkv's layout), 64-byte aligned; o: nq rows of 1,152 bf16, head-major. Sums in f32, scaled after; P to fp16 for
// P V (llama.cpp sums P V in fp16: the spec's section 7). A sub-group takes 16 query rows of one head, a lane a row,
// as S^T = K Q^T and O^T += V^T P^T: no shuffles in the softmax. Groups (V_H, ceil(nq / 16)), 16; 256 registers.
REQD_SG __kernel void v_attn(__global const half *q, uint nq, __global const half *k, __global const half *v,
                             uint npos, float scale, __global ushort *o) {
  const uint h = get_group_id(0), t0 = 16 * get_group_id(1), lane = sg_lane();
  if (t0 >= nq) return;
  const uint pitch = V_QKV_W * 2, x0 = h * V_HDP;
  // Q^T, the B operand: lane i is query t0 + i, qb[d][p] its dims 16 d + 2 p and 16 d + 2 p + 1.
  int8 qb[V_HDP / 16];
  for (uint d = 0; d < V_HDP / 16; d++) {
    uint t[8];
    intel_sub_group_2d_block_read_transpose_32b_16r8x1c((__global void *)q, pitch, nq, pitch,
                                                        (int2)(x0 / 2 + 8 * d, t0), t);
    qb[d] = as_int8(vload8(0, t));
  }
  // O^T by blocks of 16 dims: ae[j][m] is dim 16 j + 2 m of this lane's row, ao[j][m] dim 16 j + 2 m + 1.
  float8 ae[V_HDP / 16], ao[V_HDP / 16];
  for (uint j = 0; j < V_HDP / 16; j++) ae[j] = ao[j] = (float8)0.0f;
  float mrow = -INFINITY, lrow = 0.0f;
  for (uint key0 = 0; key0 < npos; key0 += 16) {
    // s0[m] = score of key key0 + m, s1[m] of key key0 + 8 + m, for this lane's row.
    float8 s0 = (float8)0.0f, s1 = (float8)0.0f;
    __attribute__((opencl_unroll_hint)) for (uint d = 0; d < V_HDP / 16; d++) {
      ushort a0[8], a1[8];
      intel_sub_group_2d_block_read_16b_8r16x1c((__global void *)k, pitch, npos, pitch, (int2)(x0 + 16 * d, key0), a0);
      intel_sub_group_2d_block_read_16b_8r16x1c((__global void *)k, pitch, npos, pitch, (int2)(x0 + 16 * d, key0 + 8),
                                                a1);
      s0 = dpas_f16(as_short8(vload8(0, a0)), qb[d], s0);
      s1 = dpas_f16(as_short8(vload8(0, a1)), qb[d], s1);
    }
    s0 *= scale;
    s1 *= scale;
    if (key0 + 16 > npos) {
      float sm[16];
      vstore8(s0, 0, sm);
      vstore8(s1, 1, sm);
      for (uint m = 0; m < 16; m++)
        if (key0 + m >= npos) sm[m] = -INFINITY;
      s0 = vload8(0, sm);
      s1 = vload8(1, sm);
    }
    const float8 mx8 = fmax(s0, s1);
    const float4 mx4 = fmax(mx8.lo, mx8.hi);
    const float mn = fmax(mrow, fmax(fmax(mx4.x, mx4.y), fmax(mx4.z, mx4.w)));
    const float alpha = exp(mrow - mn);
    const float8 p0 = exp(s0 - mn), p1 = exp(s1 - mn);
    const float8 ps = p0 + p1;
    const float4 ps4 = ps.lo + ps.hi;
    lrow = lrow * alpha + (ps4.x + ps4.y + ps4.z + ps4.w);
    mrow = mn;
    // P^T, the B operand: pairs of keys (2 p, 2 p + 1) for this lane's row.
    const half8 h0 = convert_half8(p0), h1 = convert_half8(p1);
    const int8 pb = as_int8((half16)(h0, h1));
    __attribute__((opencl_unroll_hint)) for (uint j = 0; j < V_HDP / 16; j++) {
      // V^T, the A operand: lane i is key key0 + i, vt[m] its dims 16 j + 2 m and 16 j + 2 m + 1.
      uint vt[8];
      intel_sub_group_2d_block_read_transpose_32b_16r8x1c((__global void *)v, pitch, npos, pitch,
                                                          (int2)(x0 / 2 + 8 * j, key0), vt);
      const uint8 w = vload8(0, vt);
      ae[j] = dpas_f16(as_short8(convert_ushort8(w & 0xffffu)), pb, ae[j] * alpha);
      ao[j] = dpas_f16(as_short8(convert_ushort8(w >> 16)), pb, ao[j] * alpha);
    }
  }
  const uint t = t0 + lane;
  if (t >= nq) return;
  const float inv = 1.0f / lrow;
  __global uint *row = (__global uint *)(o + (ulong)t * V_D + h * V_HD);
  for (uint j = 0; j < V_HDP / 16; j++) {
    float e[8], d[8];
    vstore8(ae[j] * inv, 0, e);
    vstore8(ao[j] * inv, 0, d);
    for (uint m = 0; m < 8; m++)
      if (16 * j + 2 * m < V_HD) row[8 * j + m] = (uint)f32_to_bf16(e[m]) | (uint)f32_to_bf16(d[m]) << 16;
  }
}
