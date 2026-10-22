// The Gated DeltaNet recurrence over a prompt chunk (docs/spec/MODEL_LAYER.md, 2.5), token after token. A value
// head's 128 x 128 state S (S[i][c]: column c, row i) is split four ways by row: sub-group r holds rows 32 r .. 32 r
// + 31 of 16 columns, a column a lane, in registers; the four add their partial dot products in local memory. Four
// times the sub-groups give each EU several chains to overlap. The loop that updates S and takes S q also takes the
// next token's S k: one barrier a token, each sum in order (docs/measurements/MEASUREMENTS.md#2026-10-15-prompt-gdn-fused-loop).
#include "builtins.h"
#include "layout.h"
#include "step.h"

#define GR 4                    // sub-groups per work-group, splitting the rows
#define GROWS (Q_GDN_D / GR)    // rows per sub-group

// Groups (nvh * 8): work-group g takes columns 16 (g % 8) .. + 15 of local value head g / 8; GR x 16 work-items.
// u rows hold [q | k | v] (nch values), gb rows g[nvh] then beta[nvh]. raw[t][lv][c] receives the head's output
// before its norm, scaled by 1/sqrt(128).
REQD_SG __kernel void p_gdn_recur(__global const float *u, uint nch, __global const float *gb,
                                  __global float *S, ulong slot_stride, uint nkh, uint nvh,
                                  __global const PromptPlan *pp, __global float *raw) {
  // [token % 2][S k or S q][sub-group][lane]
  __local float part[2][2][GR][16];
  const uint lv = get_group_id(0) / 8, cb = get_group_id(0) % 8, lane = sg_lane(), r = sg_id();
  const uint c = 16 * cb + lane, kh = lv % nkh, i0 = GROWS * r;
  __global float *s_col = S + pp->slot * slot_stride + ((ulong)lv * Q_GDN_D + c) * Q_GDN_D + i0;
  const uint n = pp->n;
  if (n == 0) return;
  u += (ulong)pp->row0 * nch;
  gb += (ulong)pp->row0 * 2 * nvh;
  raw += (ulong)pp->row0 * nvh * Q_GDN_D;
  float s[GROWS];
  for (uint i = 0; i < GROWS; i++) s[i] = s_col[i];
  const float scale = rsqrt((float)Q_GDN_D);
  // Each token's inputs are loaded two tokens ahead.
  float qv[GROWS / 16], kv[GROWS / 16], v, ga, be;
#define GDN_LOAD(tt)                                                         \
  {                                                                          \
    __global const float *row_ = u + (ulong)(tt) * nch;                      \
    for (uint b = 0; b < GROWS / 16; b++) {                                  \
      qv[b] = row_[kh * Q_GDN_D + i0 + 16 * b + lane];                       \
      kv[b] = row_[(nkh + kh) * Q_GDN_D + i0 + 16 * b + lane];               \
    }                                                                        \
    v = row_[(2 * nkh + lv) * Q_GDN_D + c];                                  \
    ga = gb[(ulong)(tt) * 2 * nvh + lv];                                     \
    be = gb[(ulong)(tt) * 2 * nvh + nvh + lv];                               \
  }
  GDN_LOAD(0)
  float qn[GROWS / 16], kn[GROWS / 16], vn = v, dn = exp(ga), bn = be;
  for (uint b = 0; b < GROWS / 16; b++) qn[b] = qv[b], kn[b] = kv[b];
  GDN_LOAD(min(1u, n - 1))
  float sk = 0.0f;
  __attribute__((opencl_unroll_hint)) for (uint i = 0; i < GROWS; i++)
    sk += s[i] * dn * sub_group_broadcast(kn[i / 16], i % 16);
  part[0][0][r][lane] = sk;
  for (uint t = 0; t < n; t++) {
    float qc[GROWS / 16], kc[GROWS / 16], vc = vn, decay = dn, beta = bn;
    for (uint b = 0; b < GROWS / 16; b++) qc[b] = qn[b], kc[b] = kn[b];
    for (uint b = 0; b < GROWS / 16; b++) qn[b] = qv[b], kn[b] = kv[b];
    vn = v, dn = exp(ga), bn = be;
    GDN_LOAD(min(t + 2, n - 1))
    barrier(CLK_LOCAL_MEM_FENCE);
    // cur: this token's S k, then its S q. alt: the last token's S q, then the next token's S k.
    __local float (*cur)[GR][16] = part[t % 2], (*alt)[GR][16] = part[(t + 1) % 2];
    if (r == 0 && t > 0)
      raw[((ulong)(t - 1) * nvh + lv) * Q_GDN_D + c] =
          ((alt[1][0][lane] + alt[1][1][lane]) + (alt[1][2][lane] + alt[1][3][lane])) * scale;
    float dj = beta * (vc - ((cur[0][0][lane] + cur[0][1][lane]) + (cur[0][2][lane] + cur[0][3][lane])));
    // S q for this token and S k for the next, two chains of adds in one loop.
    float out = 0.0f, skn = 0.0f;
    __attribute__((opencl_unroll_hint)) for (uint i = 0; i < GROWS; i++) {
      float si = s[i] * decay + sub_group_broadcast(kc[i / 16], i % 16) * dj;
      s[i] = si;
      out += si * sub_group_broadcast(qc[i / 16], i % 16);
      skn += si * dn * sub_group_broadcast(kn[i / 16], i % 16);
    }
    cur[1][r][lane] = out;
    alt[0][r][lane] = skn;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  __local float (*last)[GR][16] = part[(n - 1) % 2];
  if (r == 0)
    raw[((ulong)(n - 1) * nvh + lv) * Q_GDN_D + c] =
        ((last[1][0][lane] + last[1][1][lane]) + (last[1][2][lane] + last[1][3][lane])) * scale;
  for (uint i = 0; i < GROWS; i++) s_col[i] = s[i];
}
