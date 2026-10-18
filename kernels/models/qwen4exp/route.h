// The router's top 10 (docs/spec/MODEL_LAYER.md, 4.2) by one sub-group of RT_SG lanes for a row of 512 logits, lane
// holding experts lane + RT_SG j: layer.cl's with sub-groups of 16, prompt.cl's with 32. The includer defines RT_SG.
#ifndef KERNELS_QWEN4EXP_ROUTE_H
#define KERNELS_QWEN4EXP_ROUTE_H

#define RT_PER (Q_EXPERTS / RT_SG)
#define RT_CAND (2 * RT_SG)       // the candidates rt_top ranks in registers
#define RT_SLM (RT_CAND + RT_SG)  // rt_top's local memory: the candidates, then a spare entry per lane

// The float whose topk_key is key.
static inline float rt_value(uint key) { return as_float(key & 0x80000000u ? key & 0x7fffffffu : ~key); }

// Into a lane's queue of its 3 largest keys (q, largest first, their j in qj) when on: the keys taken after (lk, lj),
// in the order the top 10 takes them (a larger key, then a lower j), or all of them. The keys come in order of j and
// move the queue only when larger, so of equal keys the lower j goes first.
static inline void rt_fill(const uint key[RT_PER], bool on, bool all, uint lk, uint lj, uint q[3], uint qj[3]) {
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

// The 10 largest of a row's keys (topk_key's order, lane holding experts lane + RT_SG j) by its sub-group: lane k < 10
// gets winner k's (key, expert), largest first, ties to the lower expert. The candidates are the keys at or above the
// smallest of the 10 largest lane maxima, at most the 10th largest key. Up to RT_CAND go through cand (RT_SLM) to the
// lanes, 2 each, placed by the count before them; with more (equal keys), arg-max rounds over each lane's top 3.
static inline uint2 rt_top(const uint key[RT_PER], __local uint2 *cand) {
  const uint lane = sg_lane();
  uint m0 = 0, m1 = 0, m2 = 0, m3 = 0;
  for (uint j = 0; j < RT_PER; j += 4) {
    m0 = max(m0, key[j]);
    m1 = max(m1, key[j + 1]);
    m2 = max(m2, key[j + 2]);
    m3 = max(m3, key[j + 3]);
  }
  const uint m = max(max(m0, m1), max(m2, m3));
  uint above = 0;
  for (uint l = 0; l < RT_SG; l++) above += sub_group_broadcast(m, l) > m;
  const uint t = sub_group_reduce_min(above < Q_USED ? m : 0xffffffffu);
  uint mask = 0;
  for (uint j = 0; j < RT_PER; j++) mask |= (uint)(key[j] >= t) << j;
  const uint nc = popcount(mask), n = sub_group_reduce_add(nc);
  if (n <= RT_CAND) {
    // Every key goes to its candidate's place, in order of lane and j, or to the lane's spare entry: no branches.
    const uint base = sub_group_scan_exclusive_add(nc);
    for (uint j = 0; j < RT_PER; j++)
      cand[(mask >> j) & 1 ? base + popcount(mask & ((1u << j) - 1)) : RT_CAND + lane] =
          (uint2)(key[j], lane + RT_SG * j);
    sub_group_barrier(CLK_LOCAL_MEM_FENCE);
    // Past n, (0, Q_EXPERTS): it goes before no candidate.
    const uint2 c0 = lane < n ? cand[lane] : (uint2)(0, Q_EXPERTS);
    const uint2 c1 = lane + RT_SG < n ? cand[lane + RT_SG] : (uint2)(0, Q_EXPERTS);
    uint r0 = 0, r1 = 0;
    for (uint i = 0; i < RT_SG; i++) {
      const uint ok = sub_group_broadcast(c0.x, i), oe = sub_group_broadcast(c0.y, i);
      r0 += ok > c0.x || (ok == c0.x && oe < c0.y);
      r1 += ok > c1.x || (ok == c1.x && oe < c1.y);
    }
    if (n > RT_SG)
      for (uint i = 0; i < RT_SG; i++) {
        const uint ok = sub_group_broadcast(c1.x, i), oe = sub_group_broadcast(c1.y, i);
        r0 += ok > c0.x || (ok == c0.x && oe < c0.y);
        r1 += ok > c1.x || (ok == c1.x && oe < c1.y);
      }
    sub_group_barrier(CLK_LOCAL_MEM_FENCE);
    if (lane < n && r0 < Q_USED) cand[r0] = c0;
    if (lane + RT_SG < n && r1 < Q_USED) cand[r1] = c1;
    sub_group_barrier(CLK_LOCAL_MEM_FENCE);
    return cand[lane < Q_USED ? lane : 0];
  }
  uint q[3] = {0, 0, 0}, qj[3] = {0, 0, 0}, lk = 0, lj = 0;
  rt_fill(key, true, true, lk, lj, q, qj);
  uint2 win = (uint2)(0, 0);
  for (uint k = 0; k < Q_USED; k++) {
    const uint best = sub_group_reduce_max(q[0]);
    const uint e = q[0] == best ? lane + RT_SG * qj[0] : Q_EXPERTS;
    const uint we = sub_group_reduce_min(e);
    if (lane == k || (k == 0 && lane >= Q_USED)) win = (uint2)(best, we);
    if (we == e) {
      lk = q[0], lj = qj[0];
      q[0] = q[1], qj[0] = qj[1], q[1] = q[2], qj[1] = qj[2], q[2] = 0;
    }
    if (k + 1 < Q_USED && sub_group_reduce_max((uint)(q[0] == 0))) rt_fill(key, q[0] == 0, false, lk, lj, q, qj);
  }
  return win;
}

#endif
