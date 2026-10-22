// The k largest of n floats, by one work-group: a radix select on order-preserving keys, then a scan that writes
// the chosen indices in increasing order, so the result does not depend on scheduling. Ties at the threshold go to
// the lower indices.
#ifndef KERNELS_TOPK_H
#define KERNELS_TOPK_H

#include "builtins.h"

#define TOPK_WG 1024

// Float bits as an unsigned key with the same order.
static inline uint topk_key(float x) {
  uint u = as_uint(x);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// topk_select's local memory: hist, two histograms of TOPK_BINS; scan, 2 + 2 words per sub-group.
#define TOPK_BINS 2048
#define TOPK_HIST (2 * TOPK_BINS)

// The bin of the remaining-th largest of the keys counted in hist (nb bins, a multiple of the sub-group's size), and
// how many of them remain to be taken in it, into scan[0] and scan[1]; by one sub-group. Lane l sums `per` bins
// downwards from nb - 1 - l per, so its inclusive scan counts the keys at or above them; the first lane to reach
// remaining holds the bin, and the sub-group then scans that lane's bins, a sub-group's worth at a time.
static inline void topk_bin(__local const uint *hist, uint nb, uint remaining, __local uint *scan) {
  const uint sgs = get_sub_group_size(), lane = sg_lane(), per = nb / sgs;
  uint s = 0;
  for (uint j = 0; j < per; j++) s += hist[nb - 1 - lane * per - j];
  const uint above = sub_group_scan_inclusive_add(s);
  const uint f = sub_group_reduce_min(above >= remaining ? lane : sgs);
  uint acc = sub_group_broadcast(above - s, f);
  for (uint j0 = 0; j0 < per; j0 += sgs) {
    const uint b = nb - 1 - f * per - j0 - lane;
    const uint h = j0 + lane < per ? hist[b] : 0;
    const uint at = acc + sub_group_scan_inclusive_add(h);
    const uint j = sub_group_reduce_min(at >= remaining && j0 + lane < per ? lane : sgs);
    if (j < sgs) {
      if (lane == j) {
        scan[0] = b;
        scan[1] = remaining - (at - h);
      }
      return;
    }
    acc = sub_group_broadcast(at, sgs - 1);
  }
}

// Counts key into h[bin], bin < 4096 (TOPK_NONE: not counted). The lanes of a sub-group in its lowest bin add theirs
// in one atomic: most keys share their first bits (the indexer's scores are non-negative floats of a few exponents),
// and one bin would otherwise take nearly every increment of a pass.
#define TOPK_NONE 4096u
static inline void topk_count(__local uint *h, uint bin, uint lane, uint sgs) {
  const uint b0 = sub_group_reduce_min(bin);
  if (b0 == TOPK_NONE) return;
  const uint same = sub_group_reduce_add(bin == b0 ? 1u : 0u);
  const uint first = sub_group_reduce_min(bin == b0 ? lane : sgs);
  if (lane == first)
    atomic_add(&h[b0], same);
  else if (bin != b0 && bin != TOPK_NONE)
    atomic_inc(&h[bin]);
}

// Writes the indices of the k largest of v[0..n) into idx (increasing), their values into val if not null, and
// returns min(k, n). Called by all TOPK_WG work-items; hist and scan hold TOPK_HIST and TOPK_WG words. The k-th
// largest key is found a few bits at a time from the top, each part by sub-group 0 from a histogram of the keys that
// match the parts above; then a scan writes the chosen indices in order.
static inline uint topk_select(__global const float *v, uint n, uint k, __global uint *idx, __global float *val,
                               __local uint *hist, __local uint *scan) {
  const uint lid = get_local_id(0);
  if (k >= n) {
    for (uint i = lid; i < n; i += TOPK_WG) {
      idx[i] = i;
      if (val) val[i] = v[i];
    }
    return n;
  }
  // Few values: four passes of 8 bits, whose histograms are quick to clear and scan; many: three of 11, 11 and 10.
  const bool wide = n > 8192;
  const uint passes = wide ? 3 : 4, hbins = wide ? TOPK_BINS : 256;
  for (uint b = lid; b < hbins; b += TOPK_WG) hist[b] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);
  uint prefix = 0, mask = 0, remaining = k;
  const uint lane = sg_lane(), sgs = get_sub_group_size(), n4 = n / 4;
  for (uint pass = 0; pass < passes; pass++) {
    const uint width = wide ? (pass < 2 ? 11 : 10) : 8;
    const uint shift = wide ? (pass == 0 ? 21 : pass == 1 ? 10 : 0) : 24 - 8 * pass;
    const uint nb = 1u << width, dmask = nb - 1;
    __local uint *h = hist + TOPK_BINS * (pass & 1);
    for (uint j0 = 0; j0 < n4; j0 += TOPK_WG) {
      const uint j = j0 + lid;
      uint bins[4] = {TOPK_NONE, TOPK_NONE, TOPK_NONE, TOPK_NONE};
      if (j < n4) {
        const float4 x = vload4(j, v);
        const uint keys[4] = {topk_key(x.s0), topk_key(x.s1), topk_key(x.s2), topk_key(x.s3)};
        for (uint q = 0; q < 4; q++)
          if ((keys[q] & mask) == prefix) bins[q] = (keys[q] >> shift) & dmask;
      }
      for (uint q = 0; q < 4; q++) topk_count(h, bins[q], lane, sgs);
    }
    if (lid < n - 4 * n4) {
      const uint key = topk_key(v[4 * n4 + lid]);
      if ((key & mask) == prefix) atomic_inc(&h[(key >> shift) & dmask]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sg_id() == 0) topk_bin(h, nb, remaining, scan);
    __local uint *h_next = hist + TOPK_BINS * (~pass & 1);
    for (uint b = lid; b < hbins; b += TOPK_WG) h_next[b] = 0;  // the next pass's histogram
    barrier(CLK_LOCAL_MEM_FENCE);
    prefix |= scan[0] << shift;
    mask |= dmask << shift;
    remaining = scan[1];
  }
  // prefix is the k-th largest key; take every key above it and the first `remaining` equal to it. Work-item lid
  // holds a run of indices; its first place is the count of keys above prefix in the runs before it, plus the
  // equal ones before it that are taken: min(remaining, the equal ones before it).
  const uint chunk = (n + TOPK_WG - 1) / TOPK_WG;
  const uint lo = min(lid * chunk, n), hi = min(lo + chunk, n);
  uint gt = 0, eq = 0;
  for (uint i = lo; i < hi; i++) {
    uint key = topk_key(v[i]);
    gt += key > prefix;
    eq += key == prefix;
  }
  const uint sg = sg_id();
  uint gt_before = sub_group_scan_exclusive_add(gt), eq_before = sub_group_scan_exclusive_add(eq);
  if (lane == sgs - 1) {
    scan[2 + 2 * sg] = gt_before + gt;
    scan[3 + 2 * sg] = eq_before + eq;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  uint g = 0, e = 0;
  for (uint s = lane; s < sg; s += sgs) {
    g += scan[2 + 2 * s];
    e += scan[3 + 2 * s];
  }
  gt_before += sub_group_reduce_add(g);
  eq_before += sub_group_reduce_add(e);
  uint eq_take = eq_before >= remaining ? 0 : min(eq, remaining - eq_before);
  uint out = gt_before + min(eq_before, remaining);
  for (uint i = lo; i < hi; i++) {
    uint key = topk_key(v[i]);
    bool take = key > prefix;
    if (key == prefix && eq_take > 0) {
      take = true;
      eq_take--;
    }
    if (take) {
      idx[out] = i;
      if (val) val[out] = v[i];
      out++;
    }
  }
  return k;
}

// topk_select's result with offset added to every index, faster for long rows and k up to one per sub-group. Only the
// keys at or above the smallest sub-group maximum are ranked, in local memory, as 64-bit words (the key above the
// index's complement): a larger word is a larger key or an equal key at a lower index, topk_select's order. With more
// than TOPK_CAND of them, a short row or a larger k, it is topk_select.
#define TOPK_CAND 2048
static inline uint topk_select_few(__global const float *v, uint n, uint k, uint offset, __global uint *idx,
                                   __global float *val, __local uint *hist, __local uint *scan, __local ulong *cand) {
  const uint lid = get_local_id(0);
  if (k <= get_num_sub_groups() && n >= 8 * TOPK_WG) {
    // Each work-item's largest key, four values a load.
    const uint n4 = n / 4;
    uint mx = 0;
#pragma unroll 4
    for (uint i = lid; i < n4; i += TOPK_WG) {
      float4 x = vload4(i, v);
      mx = max(mx, max(max(topk_key(x.s0), topk_key(x.s1)), max(topk_key(x.s2), topk_key(x.s3))));
    }
    for (uint i = 4 * n4 + lid; i < n; i += TOPK_WG) mx = max(mx, topk_key(v[i]));
    const uint t = work_group_reduce_min(sub_group_reduce_max(mx));
    if (lid == 0) scan[0] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    // The keys at or above t, in any order.
#pragma unroll 4
    for (uint i = lid; i < n4; i += TOPK_WG) {
      float4 x = vload4(i, v);
      uint key[4] = {topk_key(x.s0), topk_key(x.s1), topk_key(x.s2), topk_key(x.s3)};
      for (uint j = 0; j < 4; j++) {
        if (key[j] >= t) {
          uint at = atomic_inc(&scan[0]);
          if (at < TOPK_CAND) cand[at] = ((ulong)key[j] << 32) | ~(4 * i + j);
        }
      }
    }
    for (uint i = 4 * n4 + lid; i < n; i += TOPK_WG) {
      uint key = topk_key(v[i]);
      if (key >= t) {
        uint at = atomic_inc(&scan[0]);
        if (at < TOPK_CAND) cand[at] = ((ulong)key << 32) | ~i;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const uint m = scan[0];
    if (m <= TOPK_CAND) {
      // Each one's rank, the count of larger words: a sub-group loads SG_SIZE of them at a time and passes them
      // round by shuffles. The first k go to scan[1 + rank].
      for (uint a0 = 0; a0 < m; a0 += TOPK_WG) {
        const uint a = a0 + lid;
        const ulong ca = a < m ? cand[a] : 0;
        uint rank = 0;
        for (uint b0 = 0; b0 < m; b0 += SG_SIZE) {
          ulong cb = b0 + sg_lane() < m ? cand[b0 + sg_lane()] : 0;
          for (uint j = 0; j < SG_SIZE; j++) rank += sg_shuffle_ul(cb, j) > ca;
        }
        if (a < m && rank < k) scan[1 + rank] = ~(uint)ca;
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      // Each one's place among them by index.
      if (lid < k) {
        uint ia = scan[1 + lid], pos = 0;
        for (uint b = 0; b < k; b++) pos += scan[1 + b] < ia;
        idx[pos] = ia + offset;
        if (val) val[pos] = v[ia];
      }
      return k;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  uint got = topk_select(v, n, k, idx, val, hist, scan);
  barrier(CLK_GLOBAL_MEM_FENCE);
  for (uint i = lid; i < got; i += TOPK_WG) idx[i] += offset;
  return got;
}

#endif
