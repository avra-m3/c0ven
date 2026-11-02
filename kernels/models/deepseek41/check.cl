// Kernels for deepseek41's tests, which no model list runs: a comparison on the card, so a soak spends its time in the
// exchanges and not in reading results back, and ds41_router_test's controls.
#include "builtins.h"
#include "layout.h"
#include "combine.h"
#include "ops.h"

// Adds to *bad the count of got's words that differ from want's, bit for bit, in [W][D41_HC_DIM] of each. It stores
// only when one differs. Groups W, sub-groups * 16; sub-groups divide 320.
REQD_SG __kernel void ds_check_same(__global const uint *got, __global const uint *want, __global uint *bad) {
  const uint n = D41_HC_DIM / get_num_sub_groups(), at = get_group_id(0) * D41_HC_DIM + sg_id() * n;
  uint4 diff = (uint4)(0);
  for (uint i = at; i < at + n; i += 64)
    diff -= as_uint4(intel_sub_group_block_read4(got + i) != intel_sub_group_block_read4(want + i));
  const uint count = sub_group_reduce_add(diff.s0 + diff.s1 + diff.s2 + diff.s3);
  if (count && sg_lane() == 0)
    atomic_fetch_add_explicit((volatile __global atomic_uint *)bad, count, memory_order_relaxed, memory_scope_device);
}

// n headers 128 bytes apart into h, sub-groups taking them in turn, by rt_job's stores (kind 0), one word a lane (1) or
// two bytes a lane (2). No fence, so that the time is the stores'. Groups 1, sub-groups * 16.
REQD_SG __kernel void ds_job_store_rate(__global uint *h, uint n, uint kind) {
  const uint lane = sg_lane(), sgs = get_num_sub_groups();
  for (uint i = sg_id(); i < n; i += sgs) {
    __global uint *p = h + (ulong)i * 32;
    if (kind == 2) {
      __global ushort *b = (__global ushort *)p;
      b[lane] = (ushort)lane;
      b[32 + lane] = (ushort)lane;
      b[16 + lane] = (ushort)lane;
      b[48 + lane] = (ushort)lane;
      b[lane] = (ushort)(lane + 1);
    } else if (kind == 1) {
      p[lane] = lane;
      p[16 + lane] = lane;
      p[lane] = lane + 1;
    } else {
      rt_job_stores(p, lane, lane, lane + 1, false);
    }
  }
}

// ds_moe_combine without its fence, to show what the fence does.
REQD_SG __kernel void ds_moe_combine_no_fence(__global const float *y, __global const float *w,
                                              __global const int *slots, __global const float *shared,
                                              __global const float *cpu, __global const uint *nmiss,
                                              __global float *out, __global float *peer) {
  moe_combine(y, w, slots, shared, cpu, nmiss, out, peer, false);
}

// ds_moe_combine's reads of the CPU's answer, then each row holds for ticks of the device clock with no fence, then the
// combine, in one launch, with no kernel's end between the reads. stamp[0] gets the clock when row 0 starts to hold.
REQD_SG __kernel void ds_moe_combine_read_twice(__global const float *y, __global const float *w,
                                                __global const int *slots, __global const float *shared,
                                                __global const float *cpu, __global const uint *nmiss,
                                                __global float *out, uint ticks, __global ulong *stamp, uint fence) {
  const uint r = get_group_id(0), sgs = get_num_sub_groups(), lane = sg_lane();
  __global const float *cr = cpu + (ulong)r * D41_EMBD;
  __global float *ot = out + (ulong)r * D41_EMBD;
  // The store keeps the reads.
  for (uint c = sg_id(); c < D41_EMBD / 64; c += sgs) vstore4(vload4(c * 16 + lane, cr), c * 16 + lane, ot);
  barrier(CLK_LOCAL_MEM_FENCE);
  if (sg_id() == 0 && lane == 0) {
    const ulong t0 = clock_read_device();
    if (r == 0) stamp[0] = t0;
    while (clock_read_device() - t0 < ticks) {
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  moe_combine(y, w, slots, shared, cpu, nmiss, out, 0, fence != 0);
}

// Holds a list for ticks of the device clock. Unlike ds_wait it runs no fence. stamp[0] gets the clock at entry. Group
// 1.
REQD_SG __kernel void ds_hold(uint ticks, __global ulong *stamp) {
  const ulong t0 = clock_read_device();
  stamp[0] = t0;
  while (clock_read_device() - t0 < ticks) {
  }
}
