// A kernel for deepseek41's tests, which no model list runs: a comparison on the card, so a soak spends its time in the
// exchanges and not in reading results back.
#include "builtins.h"
#include "layout.h"

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
