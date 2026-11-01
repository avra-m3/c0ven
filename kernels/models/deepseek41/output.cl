// deepseek41's attention output product, attn_output_a in groups: one launch for a card's groups, with a8 as its output
// since attn_output_b reads that. Each weight type and lanes a row is a module, deepseek41_og_<type>_l<lanes> (-DQTYPE,
// -DLANES: 16, 8 or 4), and the dot products are those of kernels/quant/matvec.cl.
#include "decode.h"
#include "layout.h"
#include "ops.h"

#define RPS (SG_SIZE / LANES)  // rows a sub-group
#define MAXT STEP_MAX_ROWS

static inline float lanes_sum(float x) {
#if LANES == 16
  return sg_sum(x);
#else
  for (uint m = 1; m < LANES; m <<= 1) x += sg_xor(x, m);
  return x;
#endif
}

// For each of ntok activation rows at act, a[t][g D41_OLR + i] = W[g D41_OLR + i] . act's row t of group g, for the
// card's groups g and i < D41_OLR; a padded row's result is not used. Groups (D41_OLR / D41_OG_WG_ROWS, groups),
// D41_OG_WG_ROWS / RPS * 16.
REQD_SG __kernel void ds_output_groups(__global const uchar *W,  // D41_OLR rows of D41_OG_K a group, runs layout
                                       uint bpr,                 // blocks a row
                                       __global const uchar *act,  // a8 from ds_attn_merge: [groups][ntok] rows
                                       uint ntok,
                                       __global uchar *oa,    // a as a8, rows of groups D41_OLR values
                                       __global float *af) {  // when set, the f32 values
  __local float __attribute__((aligned(64))) ys[MAXT][D41_OG_WG_ROWS];
  const uint u0 = get_group_id(0), g = get_group_id(1), lane = sg_lane();
  const uint K = get_num_groups(1) * D41_OLR;  // the output's values
  const ulong n = (ulong)K * bpr;
  const uint rl = sg_id() * RPS + lane / LANES;
  const uint r = g * D41_OLR + u0 * D41_OG_WG_ROWS + rl;
  __global const uchar *ag = act + (ulong)g * ntok * D41_OG_A8_ROW;
  float acc[MAXT];
  for (uint t = 0; t < MAXT; t++) acc[t] = 0.0f;
  for (uint u = lane % LANES; u < D41_OG_K / 32; u += LANES) {
    uint w[8];
    float lo, hi, mn;
    decode_unit(W, n, (ulong)r * bpr + u / UNITS_PER_BLOCK, u % UNITS_PER_BLOCK, w, &lo, &hi, &mn);
    for (uint t = 0; t < MAXT; t++) {
      if (t < ntok) {
        uint q[8];
        float da = load_act(ag + t * D41_OG_A8_ROW, D41_OG_K, u, q);
        acc[t] += da * unit_dot(w, q, lo, hi, mn);
      }
    }
  }
  for (uint t = 0; t < MAXT; t++) acc[t] = lanes_sum(acc[t]);
  if (lane % LANES == 0)
    for (uint t = 0; t < ntok; t++) ys[t][rl] = acc[t];
  barrier(CLK_LOCAL_MEM_FENCE);
  // Each activation row's a8 unit, by the sub-group of its number.
  const uint t = sg_id();
  if (t >= ntok) return;
  const uint unit = g * (D41_OLR / D41_OG_WG_ROWS) + u0;
  const float2 v = as_float2(intel_sub_group_block_read2((__local const uint *)ys[t]));
  a8_store_block(v, oa + (ulong)t * (K + K / 8), K, unit);
  if (af) wr2(af + (ulong)t * K + unit * D41_OG_WG_ROWS, v);
}
