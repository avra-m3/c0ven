// Matrix-vector products with quantised weights in the runs layout and a8 activations (decode.h): one weight type per
// module (-DQTYPE), LANES lanes of a 16-wide sub-group per row (-DLANES: 16, 8 or 4), WG_SG sub-groups a work-group,
// up to MAXT tokens sharing each weight read (-DMAXT: 8, or 16 for the decode steps of more than 8 rows). matvec: a
// dense slice, rows x K. matvec_ks: K split over a work-group's sub-groups, for slices of few rows. moe_gate_up_g:
// h[a] = silu(gate x) * up x for each assignment a (row, expert) of a group. moe_down_g: y[a] = down h[a] for each,
// with its own activation.
#include "decode.h"
#include "moe_tile.h"
#include "swiglu.h"

#ifndef LANES
#define LANES 16
#endif
#define WG_SG 4
#define RPS (SG_SIZE / LANES)          // rows per sub-group
#ifndef MAXT
#define MAXT 8
#endif

static inline float lanes_sum(float x) {
#if LANES == 16
  return sg_sum(x);
#else
  for (uint m = 1; m < LANES; m <<= 1) x += sg_xor(x, m);
  return x;
#endif
}

// Accumulates row r's dot products with ntok activations (a8, K values, act_stride bytes apart) into acc: the
// lane's units u0, u0 + du, ..., then the sum over the row's lanes.
static inline void row_dot(__global const uchar *W, ulong n, uint r, uint bpr, __global const uchar *act, uint K,
                           uint act_stride, uint ntok, uint u0, uint du, float acc[MAXT] GRID_PARAM) {
  uint units = K / 32;
  for (uint t = 0; t < MAXT; t++) acc[t] = 0.0f;
  for (uint u = u0; u < units; u += du) {
    uint w[8];
    float lo, hi, mn, mh;
    ulong k = (ulong)r * bpr + u / UNITS_PER_BLOCK;
    decode_unit(W, n, k, u % UNITS_PER_BLOCK, w, &lo, &hi, &mn, &mh GRID_ARG);
    for (uint t = 0; t < MAXT; t++) {
      if (t < ntok) {
        uint q[8];
        float da = load_act(act + t * act_stride, K, u, q);
        acc[t] += da * unit_dot(w, q, lo, hi, mn, mh);
      }
    }
  }
  for (uint t = 0; t < MAXT; t++) acc[t] = lanes_sum(acc[t]);
}

// y[t * y_stride + r] = W[r] . x_t for r < rows, t < *ntok_p <= MAXT: the step plan's rows, read at run time, so a
// list recorded for a width computes only the rows of the step it runs. The slice holds rows * bpr blocks.
REQD_SG __kernel void matvec(__global const uchar *W, uint rows, uint bpr, __global const uchar *act, uint K,
                             uint act_stride, __global const uint *ntok_p, __global float *y, uint y_stride,
                             __global const uchar *grids) {
  LOAD_GRID(grids)
  const uint ntok = *ntok_p;
  uint r = (get_group_id(0) * WG_SG + sg_id()) * RPS + sg_lane() / LANES;
  bool valid = r < rows;
  if (!valid) r = rows - 1;  // lanes stay in the sub-group's reductions
  float acc[MAXT];
  row_dot(W, (ulong)rows * bpr, r, bpr, act, K, act_stride, ntok, sg_lane() % LANES, LANES, acc GRID_ARG);
  if (valid && sg_lane() % LANES == 0)
    for (uint t = 0; t < ntok; t++) y[t * y_stride + r] = acc[t];
}

// matvec with K split over the work-group's ks sub-groups (ks <= 8): a slice of few rows has too few sub-groups
// otherwise to keep the card busy. A work-group takes RPS rows; its sub-group s takes units lane + LANES (s + ks i) of
// each. The sub-groups' sums meet in local memory and sub-group 0 adds them in order of s. Groups ceil(rows / RPS),
// 16 ks.
REQD_SG __kernel void matvec_ks(__global const uchar *W, uint rows, uint bpr, __global const uchar *act, uint K,
                                uint act_stride, __global const uint *ntok_p, __global float *y, uint y_stride,
                                __global const uchar *grids, uint ks) {
  __local float part[MAXT][8][SG_SIZE];
  LOAD_GRID(grids)
  const uint s = sg_id(), ntok = *ntok_p;
  uint r = get_group_id(0) * RPS + sg_lane() / LANES;
  bool valid = r < rows;
  if (!valid) r = rows - 1;
  float acc[MAXT];
  row_dot(W, (ulong)rows * bpr, r, bpr, act, K, act_stride, ntok, sg_lane() % LANES + LANES * s, LANES * ks,
          acc GRID_ARG);
  for (uint t = 0; t < ntok; t++) part[t][s][sg_lane()] = acc[t];
  barrier(CLK_LOCAL_MEM_FENCE);
  if (s != 0) return;
  for (uint t = 0; t < ntok; t++) {
    float v = 0.0f;
    for (uint j = 0; j < ks; j++) v += part[t][j][sg_lane()];
    if (valid && sg_lane() % LANES == 0) y[t * y_stride + r] = v;
  }
}

static inline float silu(float x) { return x / (1.0f + exp(-x)); }

// An expert's products by groups (MoeGroup): group g (group id 1) is one cached expert and up to MOE_GROUP_MAX
// assignments, so each weight row is decoded once for all of them. The expert's gate slice starts at its slot, its
// up slice up_off bytes and its down slice down_off bytes into it, each rows x K. Assignment a's activation is a8 at
// act + (a / per_tok) * act_stride (gate and up) or act + a * act_stride (down); outputs go to h[a] and y[a].
// Groups past *ngroups return.
static inline void row_dot_g(__global const uchar *W, ulong n, uint r, uint bpr, __global const uchar *act, uint K,
                             uint act_stride, uint div, const uint asg[MOE_GROUP_MAX], uint na,
                             float acc[MOE_GROUP_MAX] GRID_PARAM) {
  uint lane = sg_lane() % LANES;
  uint units = K / 32;
  for (uint t = 0; t < MOE_GROUP_MAX; t++) acc[t] = 0.0f;
  for (uint u = lane; u < units; u += LANES) {
    uint w[8];
    float lo, hi, mn, mh;
    ulong k = (ulong)r * bpr + u / UNITS_PER_BLOCK;
    decode_unit(W, n, k, u % UNITS_PER_BLOCK, w, &lo, &hi, &mn, &mh GRID_ARG);
    for (uint t = 0; t < MOE_GROUP_MAX; t++) {
      if (t < na) {
        uint q[8];
        float da = load_act(act + (asg[t] / div) * act_stride, K, u, q);
        acc[t] += da * unit_dot(w, q, lo, hi, mn, mh);
      }
    }
  }
  for (uint t = 0; t < MOE_GROUP_MAX; t++) acc[t] = lanes_sum(acc[t]);
}

REQD_SG __kernel void moe_gate_up_g(__global const uchar *cache, ulong slot_bytes, __global const MoeGroup *groups,
                                    __global const uint *ngroups, uint rows, uint bpr, ulong up_off,
                                    __global const uchar *act, uint K, uint act_stride, uint per_tok,
                                    __global float *h, __global const uchar *grids) {
  LOAD_GRID(grids)
  if (get_group_id(1) >= *ngroups) return;
  MoeGroup g = groups[get_group_id(1)];
  uint r = (get_group_id(0) * WG_SG + sg_id()) * RPS + sg_lane() / LANES;
  bool valid = r < rows;
  if (!valid) r = rows - 1;
  __global const uchar *G = cache + (ulong)g.slot * slot_bytes;
  ulong n = (ulong)rows * bpr;
  float gv[MOE_GROUP_MAX], uv[MOE_GROUP_MAX];
  row_dot_g(G, n, r, bpr, act, K, act_stride, per_tok, g.asg, g.n, gv GRID_ARG);
  row_dot_g(G + up_off, n, r, bpr, act, K, act_stride, per_tok, g.asg, g.n, uv GRID_ARG);
  if (valid && sg_lane() % LANES == 0)
    for (uint t = 0; t < MOE_GROUP_MAX; t++)
      if (t < g.n) h[g.asg[t] * rows + r] = SWIGLU(gv[t], uv[t]);
}

REQD_SG __kernel void moe_down_g(__global const uchar *cache, ulong slot_bytes, __global const MoeGroup *groups,
                                 __global const uint *ngroups, uint rows, uint bpr, ulong down_off,
                                 __global const uchar *act, uint K, uint act_stride, __global float *y,
                                 __global const uchar *grids) {
  LOAD_GRID(grids)
  if (get_group_id(1) >= *ngroups) return;
  MoeGroup g = groups[get_group_id(1)];
  uint r = (get_group_id(0) * WG_SG + sg_id()) * RPS + sg_lane() / LANES;
  bool valid = r < rows;
  if (!valid) r = rows - 1;
  __global const uchar *D = cache + (ulong)g.slot * slot_bytes + down_off;
  float acc[MOE_GROUP_MAX];
  row_dot_g(D, (ulong)rows * bpr, r, bpr, act, K, act_stride, 1, g.asg, g.n, acc GRID_ARG);
  if (valid && sg_lane() % LANES == 0)
    for (uint t = 0; t < MOE_GROUP_MAX; t++)
      if (t < g.n) y[g.asg[t] * rows + r] = acc[t];
}
