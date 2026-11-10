// What layer.cl's kernels share with check.cl's controls of them: the fence before reading host memory, the
// MissJob header's stores and a card's partial sum. The includer has builtins.h and layout.h.
#ifndef KERNELS_DEEPSEEK41_COMBINE_H
#define KERNELS_DEEPSEEK41_COMBINE_H

// Comes before reading what another agent wrote to host memory, as the card caches host-memory lines (ADR-0011). It
// invalidates the card's cache, so a launch that reads only card memory leaves it out.
static inline void host_read_fence(void) {
  atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_acquire, memory_scope_all_svm_devices);
}

// A MissJob header's three stores of 64 bytes a sub-group: first and second, then third over first's bytes, behind a
// release fence when fence is set. check.cl's ds_job_store_rate times these same stores.
static inline void rt_job_stores(__global uint *h, uint first, uint second, uint third, bool fence) {
  intel_sub_group_block_write2(h, (uint2)(first, second));
  if (fence) atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_release, memory_scope_all_svm_devices);
  intel_sub_group_block_write(h, third);
}

// A card's partial sum: its cached experts in rank order (y: [W][D41_USED][D41_EMBD], w and slots the router's), then
// shared, then cpu, the CPU's answer in host memory or null, for a row with misses: an acquire fence, then the 16-byte
// loads that read host memory well, together in flight. Into out and, when not null, peer (the other card's memory or
// pinned host memory), 256 bytes a sub-group. Groups W, sub-groups * 16; sub-groups divide D41_EMBD / 64 / CMB_BATCH.
#define CMB_BATCH 5
static inline void moe_combine(__global const float *y, __global const float *w, __global const int *slots,
                               __global const float *shared, __global const float *cpu, __global const uint *nmiss,
                               __global float *out, __global float *peer, bool fence) {
  const uint r = get_group_id(0), sgs = get_num_sub_groups(), lane = sg_lane();
  const uint a = r * D41_USED;
  // D41_USED is a multiple of 3: the weights and slots load three at a time.
  float wk[D41_USED];
  int on[D41_USED];
  for (uint t = 0; t < D41_USED / 3; t++) {
    const float3 wt = vload3(t, w + a);
    const int3 st = vload3(t, slots + a);
    wk[3 * t] = sub_group_broadcast(wt.s0, 0);
    wk[3 * t + 1] = sub_group_broadcast(wt.s1, 0);
    wk[3 * t + 2] = sub_group_broadcast(wt.s2, 0);
    on[3 * t] = sub_group_broadcast(st.s0, 0);
    on[3 * t + 1] = sub_group_broadcast(st.s1, 0);
    on[3 * t + 2] = sub_group_broadcast(st.s2, 0);
  }
  const uint host = cpu != 0 && sub_group_broadcast(nmiss[r], 0) != 0;
  if (host) {
    // One fence a row: it waits on the card's whole cache, so one a sub-group outlasts the loads
    // (docs/measurements/MEASUREMENTS.md#2026-11-02-deepseek41-router).
    if (fence && sg_id() == 0) host_read_fence();
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  __global const float *yr = y + (ulong)a * D41_EMBD;
  __global const float *sr = shared + (ulong)r * D41_EMBD;
  __global const float *cr = cpu ? cpu + (ulong)r * D41_EMBD : 0;
  __global float *ot = out + (ulong)r * D41_EMBD;
  __global float *pr = peer ? peer + (ulong)r * D41_EMBD : 0;
  for (uint c = sg_id(); c < D41_EMBD / 64; c += CMB_BATCH * sgs) {
    // A host load takes a PCIe round trip, so a sub-group's loads go out before any sum needs one.
    float4 from_cpu[CMB_BATCH];
    if (host)
      for (uint b = 0; b < CMB_BATCH; b++) from_cpu[b] = vload4((c + b * sgs) * 16 + lane, cr);
    for (uint b = 0; b < CMB_BATCH; b++) {
      const uint q = (c + b * sgs) * 16 + lane;  // the chunk's 16-byte unit for this lane
      float4 acc = (float4)0.0f;
      for (uint k = 0; k < D41_USED; k++)
        if (on[k] >= 0) acc += wk[k] * vload4(q, yr + k * D41_EMBD);
      acc += vload4(q, sr);
      if (host) acc += from_cpu[b];
      vstore4(acc, q, ot);
      if (peer) vstore4(acc, q, pr);
    }
  }
}

#endif
