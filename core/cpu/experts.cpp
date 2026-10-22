#include "experts.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <immintrin.h>
#include <stdexcept>

#include "ggml-cpu.h"
#include "ggml.h"
#include "multirow.h"

namespace engine {

namespace {
constexpr uint32_t kRowsPerTask = 64;
// compute_many's tasks: kRowsGU rows of gate or of up, and kRowsDown rows of down, small enough that one expert
// spreads evenly over 16 workers. h is made kRowsH rows at a time.
constexpr uint32_t kRowsGU = 16, kRowsH = 32, kRowsDown = 32;
static_assert(kRowsH % kRowsGU == 0);

// Row r of an expert's down against h (quantised to the down's dot-product type), F values: in one dot, or with the
// row stored in parts (ExpertWeights::down_parts) as one dot per part, added in order.
void down_row(const ggml_type_traits_cpu *dt, const ExpertWeights &ew, uint32_t r, uint32_t E, uint32_t F,
              uint32_t down_type, const uint8_t *hq, float *out) {
  const uint32_t parts = ew.down_parts, Fp = F / parts;
  const size_t prow = ggml_row_size((ggml_type)down_type, Fp), hp = ggml_row_size(dt->vec_dot_type, Fp);
  if (parts == 1) {
    dt->vec_dot((int)F, out, 0, ew.down + r * prow, 0, hq, 0, 1);
    return;
  }
  float sum = 0.0f;
  for (uint32_t p = 0; p < parts; p++) {
    float v;
    dt->vec_dot((int)Fp, &v, 0, ew.down + ((size_t)p * E + r) * prow, 0, hq + p * hp, 0, 1);
    sum = p == 0 ? v : sum + v;
  }
  *out = sum;
}

// The same row for the jobs js of one expert, the weights decoded once for up to kRowsTogether of them at a time
// (multirow.h); each job's sum has the bits down_row gives it.
constexpr uint32_t kRowsTogether = 16;
void down_rows(DotRows dr, const ggml_type_traits_cpu *dt, const ExpertWeights &ew, uint32_t r, uint32_t E, uint32_t F,
               uint32_t down_type, const uint8_t *hq, size_t hq_bytes, const std::vector<uint32_t> &js, float *out) {
  const uint32_t parts = ew.down_parts, Fp = F / parts;
  const size_t prow = ggml_row_size((ggml_type)down_type, Fp), hp = ggml_row_size(dt->vec_dot_type, Fp);
  for (size_t j0 = 0; j0 < js.size(); j0 += kRowsTogether) {
    const uint32_t R = (uint32_t)std::min<size_t>(kRowsTogether, js.size() - j0);
    const void *ys[kRowsTogether];
    float v[kRowsTogether], sum[kRowsTogether];
    for (uint32_t p = 0; p < parts; p++) {
      for (uint32_t i = 0; i < R; i++) ys[i] = hq + js[j0 + i] * hq_bytes + p * hp;
      dr((int)Fp, ew.down + ((size_t)p * E + r) * prow, ys, (int)R, v);
      for (uint32_t i = 0; i < R; i++) sum[i] = p == 0 ? v[i] : sum[i] + v[i];
    }
    for (uint32_t i = 0; i < R; i++) out[(size_t)js[j0 + i] * E + r] = sum[i];
  }
}
}

CpuExperts::CpuExperts(Pool &pool) : pool_(pool) { ggml_cpu_init(); }

void CpuExperts::compute(const ExpertShape &s, const std::vector<ExpertWeights> &experts, const std::vector<float> &w,
                         const float *x, float *y) {
  const uint32_t E = s.n_embd, F = s.n_ff, n = (uint32_t)experts.size();
  for (uint32_t i = 0; i < E; i++) y[i] = 0.0f;
  if (n == 0) return;
  const auto *gt = ggml_get_type_traits_cpu((ggml_type)s.gate_type);
  const auto *dt = ggml_get_type_traits_cpu((ggml_type)s.down_type);
  const auto *gq = ggml_get_type_traits_cpu(gt->vec_dot_type);
  const auto *dq = ggml_get_type_traits_cpu(dt->vec_dot_type);
  if (!gt->vec_dot || !dt->vec_dot) throw std::runtime_error("ggml has no CPU dot product for an expert type");
  const size_t grow = ggml_row_size((ggml_type)s.gate_type, E);
  const size_t xq_bytes = ggml_row_size(gt->vec_dot_type, E), hq_bytes = ggml_row_size(dt->vec_dot_type, F);

  xq_.resize(xq_bytes);
  gq->from_float(x, xq_.data(), E);
  g_.resize((size_t)n * F);
  u_.resize((size_t)n * F);
  // Gate and up: tasks of kRowsPerTask rows of one matrix of one expert.
  const uint32_t ft = (F + kRowsPerTask - 1) / kRowsPerTask;
  pool_.run((uint64_t)n * 2 * ft, [&](uint64_t task) {
    uint32_t e = (uint32_t)(task / (2 * ft)), which = (uint32_t)(task / ft % 2), t = (uint32_t)(task % ft);
    const uint8_t *m = which ? experts[e].up : experts[e].gate;
    float *out = (which ? u_.data() : g_.data()) + (size_t)e * F;
    for (uint32_t r = t * kRowsPerTask; r < std::min(F, (t + 1) * kRowsPerTask); r++)
      gt->vec_dot((int)E, out + r, 0, m + r * grow, 0, xq_.data(), 0, 1);
  });
  hq_.resize((size_t)n * hq_bytes);
  std::vector<float> h(F);
  for (uint32_t e = 0; e < n; e++) {
    for (uint32_t r = 0; r < F; r++) {
      float gv = g_[(size_t)e * F + r];
      h[r] = gv / (1.0f + std::exp(-gv)) * u_[(size_t)e * F + r];
    }
    dq->from_float(h.data(), hq_.data() + e * hq_bytes, F);
  }
  out_.resize((size_t)n * E);
  const uint32_t et = (E + kRowsPerTask - 1) / kRowsPerTask;
  pool_.run((uint64_t)n * et, [&](uint64_t task) {
    uint32_t e = (uint32_t)(task / et), t = (uint32_t)(task % et);
    for (uint32_t r = t * kRowsPerTask; r < std::min(E, (t + 1) * kRowsPerTask); r++)
      down_row(dt, experts[e], r, E, F, s.down_type, hq_.data() + e * hq_bytes, out_.data() + (size_t)e * E + r);
  });
  for (uint32_t e = 0; e < n; e++)
    for (uint32_t i = 0; i < E; i++) y[i] += w[e] * out_[(size_t)e * E + i];
}

void CpuExperts::compute_many(const ExpertShape &s, const std::vector<Job> &jobs, uint32_t ntok, const float *x,
                              uint64_t x_stride, float *y, uint64_t y_stride) {
  const uint32_t E = s.n_embd, F = s.n_ff, n = (uint32_t)jobs.size();
  if (n == 0) {
    for (uint32_t t = 0; t < ntok; t++)
      for (uint32_t i = 0; i < E; i++) y[t * y_stride + i] = 0.0f;
    return;
  }
  const auto *gt = ggml_get_type_traits_cpu((ggml_type)s.gate_type);
  const auto *dt = ggml_get_type_traits_cpu((ggml_type)s.down_type);
  const auto *gq = ggml_get_type_traits_cpu(gt->vec_dot_type);
  const auto *dq = ggml_get_type_traits_cpu(dt->vec_dot_type);
  if (!gt->vec_dot || !dt->vec_dot) throw std::runtime_error("ggml has no CPU dot product for an expert type");
  const size_t grow = ggml_row_size((ggml_type)s.gate_type, E);
  const size_t xq_bytes = ggml_row_size(gt->vec_dot_type, E), hq_bytes = ggml_row_size(dt->vec_dot_type, F);
  xqm_.resize((size_t)ntok * xq_bytes);
  for (uint32_t t = 0; t < ntok; t++) gq->from_float(x + t * x_stride, xqm_.data() + t * xq_bytes, E);
  // The distinct experts, each with its jobs.
  std::vector<const uint8_t *> distinct;
  std::vector<std::vector<uint32_t>> of;
  for (uint32_t j = 0; j < n; j++) {
    size_t d = 0;
    while (d < distinct.size() && distinct[d] != jobs[j].expert.gate) d++;
    if (d == distinct.size()) distinct.push_back(jobs[j].expert.gate), of.emplace_back();
    of[d].push_back(j);
  }
  const uint32_t nd = (uint32_t)distinct.size();
  g_.resize((size_t)n * F);
  u_.resize((size_t)n * F);
  hq_.resize((size_t)n * hq_bytes);
  // h = silu(gate) * up for rows [r0, r1) of every job of expert d, quantised to the down's dot-product type in
  // place: blocks of that type never straddle kRowsH rows (a multiple of its block size), so h's blocks come out as
  // quantising the whole row would make them.
  const int64_t hblk = ggml_blck_size(dt->vec_dot_type);
  const size_t hblk_bytes = ggml_type_size(dt->vec_dot_type);
  const bool fused = kRowsH % hblk == 0 && F % kRowsH == 0;
  auto swiglu = [&](uint32_t d, uint32_t r0, uint32_t r1) {
    float h[kRowsH];
    for (uint32_t j : of[d]) {
      for (uint32_t r = r0; r < r1; r++) {
        float gv = g_[(size_t)j * F + r];
        h[r - r0] = gv / (1.0f + std::exp(-gv)) * u_[(size_t)j * F + r];
      }
      dq->from_float(h, hq_.data() + j * hq_bytes + (size_t)(r0 / hblk) * hblk_bytes, r1 - r0);
    }
  };
  // Gate and up: tasks of kRowsGU rows of one matrix of one expert, for all its jobs, gate and up alternating. The
  // last of the tasks over kRowsH rows to finish computes their h.
  const uint32_t ft = (F + kRowsGU - 1) / kRowsGU, ht = F / kRowsH, per_h = 2 * kRowsH / kRowsGU;
  const uint64_t ngu = (uint64_t)nd * 2 * ft;
  std::vector<std::atomic<uint32_t>> pending(fused ? (size_t)nd * ht : 0);
  // An expert that several jobs share decodes each weight row once for all of them, where the type allows.
  const DotRows gr = dot_rows_for(s.gate_type), dr = dot_rows_for(s.down_type);
  auto gate_up = [&](uint64_t task) {
    uint32_t d = (uint32_t)(task / (2 * ft)), which = (uint32_t)(task % 2), t = (uint32_t)(task / 2 % ft);
    const ExpertWeights &ew = jobs[of[d][0]].expert;
    const uint8_t *m = which ? ew.up : ew.gate;
    const uint32_t r0 = t * kRowsGU, r1 = std::min(F, (t + 1) * kRowsGU);
    float *outs = which ? u_.data() : g_.data();
    if (gr && of[d].size() > 1) {
      for (size_t j0 = 0; j0 < of[d].size(); j0 += kRowsTogether) {
        const uint32_t R = (uint32_t)std::min<size_t>(kRowsTogether, of[d].size() - j0);
        const void *ys[kRowsTogether];
        float v[kRowsTogether];
        for (uint32_t i = 0; i < R; i++) ys[i] = xqm_.data() + jobs[of[d][j0 + i]].tok * xq_bytes;
        for (uint32_t r = r0; r < r1; r++) {
          gr((int)E, m + r * grow, ys, (int)R, v);
          for (uint32_t i = 0; i < R; i++) outs[(size_t)of[d][j0 + i] * F + r] = v[i];
        }
      }
    } else {
      for (uint32_t r = r0; r < r1; r++)
        for (uint32_t j : of[d]) gt->vec_dot((int)E, outs + (size_t)j * F + r, 0, m + r * grow, 0, xqm_.data() + jobs[j].tok * xq_bytes, 0, 1);
    }
    if (!fused) return;
    const uint32_t b = r0 / kRowsH;
    if (pending[(size_t)d * ht + b].fetch_add(1, std::memory_order_acq_rel) == per_h - 1)
      swiglu(d, b * kRowsH, (b + 1) * kRowsH);
  };
  // Down: tasks of kRowsDown rows of every expert, each summing its rows of y over the jobs in order.
  out_.resize((size_t)n * E);
  const uint32_t et = (E + kRowsDown - 1) / kRowsDown;
  auto down = [&](uint64_t t) {
    const uint32_t r0 = (uint32_t)t * kRowsDown, r1 = std::min(E, r0 + kRowsDown);
    for (uint32_t d = 0; d < nd; d++) {
      const ExpertWeights &ew = jobs[of[d][0]].expert;
      if (dr && of[d].size() > 1) {
        for (uint32_t r = r0; r < r1; r++) down_rows(dr, dt, ew, r, E, F, s.down_type, hq_.data(), hq_bytes, of[d], out_.data());
        continue;
      }
      for (uint32_t r = r0; r < r1; r++)
        for (uint32_t j : of[d])
          down_row(dt, ew, r, E, F, s.down_type, hq_.data() + j * hq_bytes, out_.data() + (size_t)j * E + r);
    }
    for (uint32_t tk = 0; tk < ntok; tk++)
      for (uint32_t r = r0; r < r1; r++) y[tk * y_stride + r] = 0.0f;
    for (uint32_t j = 0; j < n; j++) {
      float *yt = y + jobs[j].tok * y_stride;
      for (uint32_t r = r0; r < r1; r++) yt[r] += jobs[j].w * out_[(size_t)j * E + r];
    }
  };
  if (fused) {
    // One pool job, so the workers wake once: the down tasks come after every gate and up task, and wait for them
    // to finish. The pool hands out tasks in order, so by then each of those is running on some thread.
    std::atomic<uint64_t> gu_done{0};
    pool_.run(ngu + et, [&](uint64_t task) {
      if (task < ngu) {
        gate_up(task);
        gu_done.fetch_add(1, std::memory_order_release);
        return;
      }
      while (gu_done.load(std::memory_order_acquire) < ngu) _mm_pause();
      down(task - ngu);
    });
    return;
  }
  pool_.run(ngu, gate_up);
  pool_.run(nd, [&](uint64_t d) {
    for (uint32_t j : of[d]) {
      std::vector<float> h(F);
      for (uint32_t r = 0; r < F; r++) {
        float gv = g_[(size_t)j * F + r];
        h[r] = gv / (1.0f + std::exp(-gv)) * u_[(size_t)j * F + r];
      }
      dq->from_float(h.data(), hq_.data() + j * hq_bytes, F);
    }
  });
  pool_.run(et, down);
}

}  // namespace engine
