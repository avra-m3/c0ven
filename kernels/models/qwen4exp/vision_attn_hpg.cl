// Xe-HPG vision attention kernel (docs/spec/MODEL_VISION.md, section 3.2) on the 8-lane XMX units.
// Two 8-lane sub-groups per work-group cover 16 query rows of one head: groups (V_H, ceil(nq / 16)), 16.
#include "vision_sizes.h"
#include "xmx.h"

#define V_QKV_W (V_H * V_HDP)

REQD_XMX_SG __kernel void v_attn(__global const half *q, uint nq, __global const half *k, __global const half *v,
                                 uint npos, float scale, __global ushort *o) {
  const uint h = get_group_id(0), t0 = 16 * get_group_id(1) + 8 * sg_id(), lane = sg_lane();
  if (t0 >= nq) return;
  const uint x0 = h * V_HDP, t = t0 + lane;
  const __global ushort *qu = (const __global ushort *)q;
  const __global ushort *ku = (const __global ushort *)k;
  const __global ushort *vu = (const __global ushort *)v;

  int8 qb[V_HDP / 16];
  for (uint d = 0; d < V_HDP / 16; d++) qb[d] = xmx_load_b(qu, V_QKV_W, nq, V_QKV_W, 16 * d + x0, t0);

  float8 ae[V_HDP / 16], ao[V_HDP / 16];
  for (uint j = 0; j < V_HDP / 16; j++) ae[j] = ao[j] = (float8)0.0f;
  float mrow = -INFINITY, lrow = 0.0f;

  for (uint key0 = 0; key0 < npos; key0 += 16) {
    float8 s0 = (float8)0.0f, s1 = (float8)0.0f;
    __attribute__((opencl_unroll_hint)) for (uint d = 0; d < V_HDP / 16; d++) {
      const int8 a0 = xmx_load_a8(ku, V_QKV_W, npos, x0 + 16 * d, key0);
      const int8 a1 = xmx_load_a8(ku, V_QKV_W, npos, x0 + 16 * d, key0 + 8);
      s0 = xmx_dpas_f16(a0, qb[d], s0);
      s1 = xmx_dpas_f16(a1, qb[d], s1);
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

    const half8 h0 = convert_half8(p0), h1 = convert_half8(p1);
    const int8 pb = as_int8((half16)(h0, h1));
    const uint k0 = key0 + 2 * lane, k1 = k0 + 1;
    __attribute__((opencl_unroll_hint)) for (uint j = 0; j < V_HDP / 16; j++) {
      const ushort16 v0 = k0 < npos ? vload16(0, vu + (ulong)k0 * V_QKV_W + x0 + 16 * j) : (ushort16)0;
      const ushort16 v1 = k1 < npos ? vload16(0, vu + (ulong)k1 * V_QKV_W + x0 + 16 * j) : (ushort16)0;
      const int8 va_lo = as_int8(convert_uint8(v0.lo) | (convert_uint8(v1.lo) << 16));
      const int8 va_hi = as_int8(convert_uint8(v0.hi) | (convert_uint8(v1.hi) << 16));
      ae[j] = xmx_dpas_f16(va_lo, pb, ae[j] * alpha);
      ao[j] = xmx_dpas_f16(va_hi, pb, ao[j] * alpha);
    }
  }

  if (t >= nq) return;
  const float inv = 1.0f / lrow;
  __global ushort *row = o + (ulong)t * V_D + h * V_HD;
  for (uint j = 0; j < V_HDP / 16; j++) {
    float e[8], d[8];
    vstore8(ae[j] * inv, 0, e);
    vstore8(ao[j] * inv, 0, d);
    for (uint m = 0; m < 8; m++) {
      if (16 * j + m < V_HD) row[16 * j + m] = f32_to_bf16(e[m]);
      if (16 * j + 8 + m < V_HD) row[16 * j + 8 + m] = f32_to_bf16(d[m]);
    }
  }
}
