// Decode MoE expert kernels for inkling2_loop: per-expert base pointers (VRAM cache or host USM) and SiTU.
// Built per QTYPE and GROUP in {1, 2, 4, 8}.
#include "a8.h"
#include "decode.h"
#include "kernel_sizes.h"
#include "moe_tile.h"

#define A GROUP
#define WG_SG 4
#define WG_SIZE (WG_SG * SG_SIZE)
#define GU_ROW_CHUNK 32

static inline float i2_sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }

static inline float i2_situ(float g, float u, float beta, float linear_beta) {
  const float s = (beta * tanh(g / beta)) * i2_sigmoid(g);
  return linear_beta > 0.0f ? s * (linear_beta * tanh(u / linear_beta)) : s * u;
}

// Compute SiTU(gate . x, up . x) over `rows` (= ff_n, a multiple of 32) and quantize h to a8 (`ha`).
// Grid: (rows / 32, max_groups, 1), local (64, 1, 1). exp_ptrs[gp->slot] points to [gate | up | down] in runs layout.
REQD_SG __kernel void i2_moe_gate_up_g(__global const ulong *exp_ptrs, __global const MoeGroup *groups,
                                       __global const uint *ngroups, uint rows, uint bpr, ulong up_off,
                                       __global const uchar *act, uint K, uint act_stride, float beta, float lin_beta,
                                       __global uchar *ha, __global const uchar *grids) {
  __local float hs[A][GU_ROW_CHUNK];
  LOAD_GRID(grids)
  if (get_group_id(1) >= *ngroups) return;
  __global const MoeGroup *gp = groups + get_group_id(1);
  const uint na = A == 1 ? 1 : gp->n;
  uint asg[A];
  for (uint t = 0; t < A; t++) asg[t] = t < na ? gp->asg[t] : gp->asg[0];

  __global const uchar *G = (__global const uchar *)(uintptr_t)exp_ptrs[gp->slot];
  __global const uchar *U = G + up_off;
  const uint lane = sg_lane(), sg = sg_id();
  const ulong n = (ulong)rows * bpr;
  const uint units = K / 32;
  const uint row_base = get_group_id(0) * GU_ROW_CHUNK;

  // 4 sub-groups of 16 cover 32 output rows in 8 passes (1 row per sub-group per pass).
  for (uint pass = 0; pass < GU_ROW_CHUNK / WG_SG; pass++) {
    const uint r = row_base + pass * WG_SG + sg;
    const uint rr = min(r, rows - 1);
    float ag[A], au[A];
    for (uint t = 0; t < A; t++) ag[t] = au[t] = 0.0f;
    for (uint u = lane; u < units; u += SG_SIZE) {
      const ulong k = (ulong)rr * bpr + u / UNITS_PER_BLOCK;
      const uint s = u % UNITS_PER_BLOCK;
      uint wg[8], wu[8];
      float glo, ghi, gmn, gmh, ulo, uhi, umn, umh;
      decode_unit(G, n, k, s, wg, &glo, &ghi, &gmn, &gmh GRID_ARG);
      decode_unit(U, n, k, s, wu, &ulo, &uhi, &umn, &umh GRID_ARG);
      for (uint t = 0; t < A; t++) {
        if (t < na) {
          uint q[8];
          const float da = load_act(act + (ulong)(asg[t] / I2_USED) * act_stride, K, u, q);
          ag[t] += da * unit_dot(wg, q, glo, ghi, gmn, gmh);
          au[t] += da * unit_dot(wu, q, ulo, uhi, umn, umh);
        }
      }
    }
    for (uint t = 0; t < A; t++) {
      const float gv = sg_sum(ag[t]), uv = sg_sum(au[t]);
      if (t < na && r < rows && lane == 0) hs[t][pass * WG_SG + sg] = i2_situ(gv, uv, beta, lin_beta);
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint t = sg; t < na; t += WG_SG) {
    a8_store(vload2(lane, hs[t]), ha + (ulong)gp->asg[t] * A8_BYTES(rows), rows, get_group_id(0));
  }
}

// Compute down . h over `rows` (= I2_EMBD) for grouped assignments, reading from exp_ptrs[gp->slot] + down_off.
// Grid: (ceil(rows / 8), max_groups, 1), local (64, 1, 1); 2 passes of 4 SGs = 8 output rows per work-group.
REQD_SG __kernel void i2_moe_down_g(__global const ulong *exp_ptrs, ulong down_off, __global const MoeGroup *groups,
                                    __global const uint *ngroups, uint rows, uint bpr, __global const uchar *act,
                                    uint K, uint act_stride, __global float *y, __global const uchar *grids) {
  LOAD_GRID(grids)
  if (get_group_id(1) >= *ngroups) return;
  __global const MoeGroup *gp = groups + get_group_id(1);
  const uint na = A == 1 ? 1 : gp->n;
  uint asg[A];
  for (uint t = 0; t < A; t++) asg[t] = t < na ? gp->asg[t] : gp->asg[0];

  __global const uchar *W = (__global const uchar *)(uintptr_t)exp_ptrs[gp->slot] + down_off;
  const uint lane = sg_lane(), sg = sg_id();
  const ulong n = (ulong)rows * bpr;
  const uint units = K / 32;
  const uint row_base = get_group_id(0) * 8;

  for (uint pass = 0; pass < 2; pass++) {
    const uint r = row_base + pass * WG_SG + sg;
    if (r >= rows) continue;
    float acc[A];
    for (uint t = 0; t < A; t++) acc[t] = 0.0f;
    for (uint u = lane; u < units; u += SG_SIZE) {
      const ulong k = (ulong)r * bpr + u / UNITS_PER_BLOCK;
      const uint s = u % UNITS_PER_BLOCK;
      uint w[8];
      float lo, hi, mn, mh;
      decode_unit(W, n, k, s, w, &lo, &hi, &mn, &mh GRID_ARG);
      for (uint t = 0; t < A; t++) {
        if (t < na) {
          uint q[8];
          const float da = load_act(act + (ulong)asg[t] * act_stride, K, u, q);
          acc[t] += da * unit_dot(w, q, lo, hi, mn, mh);
        }
      }
    }
    for (uint t = 0; t < A; t++) {
      const float v = sg_sum(acc[t]);
      if (t < na && lane == 0) y[(ulong)asg[t] * rows + r] = v;
    }
  }
}
