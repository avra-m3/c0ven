// Sigmoid-biased top-8 expert routing for inkling2_loop: one sub-group of RT_SG lanes per row.
#ifndef KERNELS_INKLING2_ROUTE_H
#define KERNELS_INKLING2_ROUTE_H

#include "kernel_sizes.h"

#ifndef RT_SG
#define RT_SG SG_SIZE
#endif

#if I2_EXPERTS % RT_SG
#error "I2_EXPERTS must be a multiple of RT_SG"
#endif

#define RT_PER (I2_EXPERTS / RT_SG)
#define RT_CAND (2 * RT_SG)
#define RT_SLM (RT_CAND + RT_SG)

static inline float i2_sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }
static inline float i2_rt_value(uint key) { return as_float(key & 0x80000000u ? key & 0x7fffffffu : ~key); }

static inline void i2_rt_fill(const uint key[RT_PER], bool on, bool all, uint lk, uint lj, uint q[3], uint qj[3]) {
  for (uint j = 0; j < RT_PER; j++) {
    const uint kj = key[j];
    if (!on || !(all || kj < lk || (kj == lk && j > lj)) || kj <= q[2]) continue;
    if (kj > q[1]) {
      q[2] = q[1], qj[2] = qj[1];
      if (kj > q[0])
        q[1] = q[0], qj[1] = qj[0], q[0] = kj, qj[0] = j;
      else
        q[1] = kj, qj[1] = j;
    } else {
      q[2] = kj, qj[2] = j;
    }
  }
}

// Ranks top I2_USED experts by key = topk_key(sigmoid(logit) + exp_probs_b). Returns winner (key, expert_id)
// in lanes 0 .. I2_USED - 1, breaking ties toward lower expert ID.
static inline uint2 i2_rt_top(const uint key[RT_PER], __local uint2 *cand) {
  const uint lane = sg_lane();
  uint m = 0;
  for (uint j = 0; j < RT_PER; j++) m = max(m, key[j]);
  uint above = 0;
  for (uint l = 0; l < RT_SG; l++) above += sub_group_broadcast(m, l) > m;
  const uint t = sub_group_reduce_min(above < I2_USED ? m : 0xffffffffu);
  uint mask = 0;
  for (uint j = 0; j < RT_PER; j++) mask |= (uint)(key[j] >= t) << j;
  const uint nc = popcount(mask), n = sub_group_reduce_add(nc);
  if (n <= RT_CAND) {
    const uint base = sub_group_scan_exclusive_add(nc);
    for (uint j = 0; j < RT_PER; j++)
      cand[(mask >> j) & 1 ? base + popcount(mask & ((1u << j) - 1)) : RT_CAND + lane] =
          (uint2)(key[j], lane + RT_SG * j);
    sub_group_barrier(CLK_LOCAL_MEM_FENCE);
    const uint2 c0 = lane < n ? cand[lane] : (uint2)(0, I2_EXPERTS);
    const uint2 c1 = lane + RT_SG < n ? cand[lane + RT_SG] : (uint2)(0, I2_EXPERTS);
    uint r0 = 0, r1 = 0;
    for (uint i = 0; i < RT_SG; i++) {
      const uint ok = sub_group_broadcast(c0.x, i), oe = sub_group_broadcast(c0.y, i);
      r0 += ok > c0.x || (ok == c0.x && oe < c0.y);
      r1 += ok > c1.x || (ok == c1.x && oe < c1.y);
    }
    if (n > RT_SG) {
      for (uint i = 0; i < RT_SG; i++) {
        const uint ok = sub_group_broadcast(c1.x, i), oe = sub_group_broadcast(c1.y, i);
        r0 += ok > c0.x || (ok == c0.x && oe < c0.y);
        r1 += ok > c1.x || (ok == c1.x && oe < c1.y);
      }
    }
    sub_group_barrier(CLK_LOCAL_MEM_FENCE);
    if (lane < n && r0 < I2_USED) cand[r0] = c0;
    if (lane + RT_SG < n && r1 < I2_USED) cand[r1] = c1;
    sub_group_barrier(CLK_LOCAL_MEM_FENCE);
    return cand[lane < I2_USED ? lane : 0];
  }
  uint q[3] = {0, 0, 0}, qj[3] = {0, 0, 0}, lk = 0, lj = 0;
  i2_rt_fill(key, true, true, lk, lj, q, qj);
  uint2 win = (uint2)(0, 0);
  for (uint k = 0; k < I2_USED; k++) {
    const uint best = sub_group_reduce_max(q[0]);
    const uint e = q[0] == best ? lane + RT_SG * qj[0] : I2_EXPERTS;
    const uint we = sub_group_reduce_min(e);
    if (lane == k || (k == 0 && lane >= I2_USED)) win = (uint2)(best, we);
    if (we == e) {
      lk = q[0], lj = qj[0];
      q[0] = q[1], qj[0] = qj[1], q[1] = q[2], qj[1] = qj[2], q[2] = 0;
    }
    if (k + 1 < I2_USED && sub_group_reduce_max((uint)(q[0] == 0))) i2_rt_fill(key, q[0] == 0, false, lk, lj, q, qj);
  }
  return win;
}

// Routes one row of logits (`rl`) with optional selection bias (`bias`, nullable).
// Writes winner IDs and normalized weights (`expert_weights_scale`) into ids_out and w_out.
// `probs` must hold RT_SG floats * RT_PER (I2_EXPERTS floats) per routing sub-group.
static inline void i2_route_row(__global const float *rl, __global const float *bias, float scale,
                                __local uint2 *cand, __local float *probs, __global int *ids_out,
                                __global float *w_out) {
  const uint lane = sg_lane();
  uint key[RT_PER];
  for (uint j = 0; j < RT_PER; j++) {
    const uint e = lane + RT_SG * j;
    const float p = i2_sigmoid(rl[e]);
    probs[e] = p;
    const float sel = bias ? p + bias[e] : p;
    key[j] = topk_key(sel);
  }
  sub_group_barrier(CLK_LOCAL_MEM_FENCE);
  const uint2 win = i2_rt_top(key, cand);
  const float p_win = lane < I2_USED ? probs[win.y] : 0.0f;
  const float p_sum = sub_group_reduce_add(p_win) + 1e-20f;
  if (lane < I2_USED) {
    ids_out[lane] = (int)win.y;
    w_out[lane] = (p_win / p_sum) * scale;
  }
}

#endif
