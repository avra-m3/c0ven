// deepseek41's prompt chunk path: the small kernels between its products (core/models/deepseek41/prompt.cpp). A chunk's
// rows are f32 streams as the decode step's, its products take f16 activations, and its routed experts' outputs come
// back as f16 rows by assignment.
#include "builtins.h"
#include "layout.h"
#include "combine.h"

static inline float silu(float x) { return x / (1.0f + exp(-x)); }

// out[i] = f16(in[i]) for i < n, n a multiple of 4. Groups ceil(n / 4 / 256), 256.
__kernel void dp_to_f16(__global const float *in, __global half *out, ulong n) {
  const ulong i = (ulong)get_global_id(0) * 4;
  if (i < n) vstore_half4_rte(vload4(0, in + i), 0, out + i);
}

// Row a of dst is row rowidx[a] of src, both D41_EMBD f16 wide; with sigma, the middle two values of every four swap,
// the order of the experts' gate and up types that decode to pairs (shared/quant.h). Groups (rows), 256.
__kernel void dp_gather(__global const ushort *src, __global const uint *rowidx, __global ushort *dst, uint sigma) {
  const uint a = get_group_id(0);
  __global const ushort4 *s = (__global const ushort4 *)(src + (ulong)rowidx[a] * D41_EMBD);
  __global ushort4 *d = (__global ushort4 *)(dst + (ulong)a * D41_EMBD);
  for (uint i = get_local_id(0); i < D41_EMBD / 4; i += get_local_size(0)) {
    const ushort4 v = s[i];
    d[i] = sigma ? v.s0213 : v;
  }
}

// The shared expert's SwiGLU with the clamp, as ds_swiglu_clamp_a8 but into f16 rows of ff: gu holds a row's ff gate
// values and then its ff up values, ld values apart. Groups (rows), 256.
__kernel void dp_swiglu_f16(__global const float *gu, uint ld, uint ff, __global half *out) {
  const uint t = get_group_id(0);
  for (uint i = get_local_id(0) * 4; i < ff; i += get_local_size(0) * 4) {
    const float4 g = fmin(vload4(0, gu + (ulong)t * ld + i), (float4)D41_SWIGLU_LIMIT);
    const float4 u = clamp(vload4(0, gu + (ulong)t * ld + ff + i), (float4)-D41_SWIGLU_LIMIT, (float4)D41_SWIGLU_LIMIT);
    vstore_half4_rte(g / (1.0f + exp(-g)) * u, 0, out + (ulong)t * ff + i);
  }
}

// A row's partial sum, as moe_combine's: its cached experts in rank order, then shared, then the CPU's answer when the
// row has misses. y: f16 rows of D41_EMBD, row yidx[row * D41_USED + rank] of it, or -1 for an expert the card does not
// hold. Into out and, when set, peer. Groups (rows), 256.
__kernel void dp_combine(__global const half *y, __global const int *yidx, __global const float *w,
                         __global const float *shared, __global const float *cpu, __global const uint *nmiss,
                         __global float *out, __global float *peer) {
  const uint r = get_group_id(0), a = r * D41_USED;
  const bool host = cpu != 0 && nmiss[r] != 0;
  if (host) {
    if (get_local_id(0) == 0) host_read_fence();
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  float wk[D41_USED];
  int at[D41_USED];
  for (uint k = 0; k < D41_USED; k++) {
    wk[k] = w[a + k];
    at[k] = yidx[a + k];
  }
  for (uint c = get_local_id(0) * 4; c < D41_EMBD; c += get_local_size(0) * 4) {
    float4 acc = (float4)0.0f;
    for (uint k = 0; k < D41_USED; k++)
      if (at[k] >= 0) acc += wk[k] * vload_half4(0, y + (ulong)at[k] * D41_EMBD + c);
    acc += vload4(0, shared + (ulong)r * D41_EMBD + c);
    if (host) acc += vload4(0, cpu + (ulong)r * D41_EMBD + c);
    vstore4(acc, 0, out + (ulong)r * D41_EMBD + c);
    if (peer) vstore4(acc, 0, peer + (ulong)r * D41_EMBD + c);
  }
}

// Window cells of one slot from one ring to another: for q = q0 .. q0 + n - 1, cell q % src_cells of src into cell
// q % dst_cells of dst. A ring is [slot][cells][D41_HD] f16. Groups (n), 256.
__kernel void dp_ring_move(__global const uint *src, uint src_cells, __global uint *dst, uint dst_cells, uint slot,
                           uint q0) {
  const uint q = q0 + get_group_id(0);
  __global const uint *s = src + ((ulong)slot * src_cells + q % src_cells) * (D41_HD / 2);
  __global uint *d = dst + ((ulong)slot * dst_cells + q % dst_cells) * (D41_HD / 2);
  for (uint i = get_local_id(0); i < D41_HD / 2; i += get_local_size(0)) d[i] = s[i];
}
