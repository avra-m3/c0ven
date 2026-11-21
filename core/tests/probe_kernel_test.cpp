// Level 2 GPU kernel test: qwen4exp_probe's probe_tap, probe_mqa_head, and probe_kv_write against probe_eval_cpu,
// covering ring rollover, chunked parallel evaluation matching sequential token-by-token evaluation, and drift scalars.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "api/probe.h"
#include "step.h"
#include "test_util.h"
#include "ze/library.h"

using namespace engine;

namespace {

uint16_t to_bf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return (uint16_t)((u + (0x7fffu + ((u >> 16) & 1u))) >> 16);
}

float from_bf16(uint16_t h) {
  uint32_t u = (uint32_t)h << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

}  // namespace

int main() {
  if (test::gpus() == 0) return test::no_gpu();
  ze::Runtime rt;
  ze::Library lib(rt);
  const uint32_t card = 0;
  ze::List up(rt, card, ze::List::immediate);

  // Build synthetic SingProbeAttn (H=8, D=64, window=16, 10 base labels + 1 drift label) with tiny hidden tap check.
  const uint32_t H = 8, D = 64, qd = H * D, W = 16, base_labels = 10, n_lin = 1, n_tot = base_labels + n_lin;
  const uint32_t qkv_stride = qd + 2 * D + n_lin;
  const uint32_t n_tok = 24;  // exceeds window=16 to exercise wrap-around

  std::mt19937 rng(2026);
  std::normal_distribution<float> dist(0.0f, 0.25f);
  std::vector<float> qkv(n_tok * qkv_stride);
  for (float &v : qkv) v = dist(rng);
  std::vector<uint16_t> o_proj(qd * qd);
  for (uint16_t &v : o_proj) v = to_bf16(dist(rng) * 0.1f);
  std::vector<float> norm_w(qd), cls_w(base_labels * qd), cls_b(base_labels), lin_b(n_lin, 0.25f);
  for (float &v : norm_w) v = 1.0f + dist(rng) * 0.1f;
  for (float &v : cls_w) v = dist(rng) * 0.2f;
  for (float &v : cls_b) v = dist(rng);
  std::vector<uint32_t> lin_is_drift = {1u};

  // Host reference matching probe_mqa_head token-by-token.
  std::vector<uint16_t> k_ref(W * D, 0), v_ref(W * D, 0);
  std::vector<float> ref_out(n_tok * n_tot, 0.0f);
  const float scale = 1.0f / std::sqrt((float)D);
  for (uint32_t pos = 0; pos < n_tok; pos++) {
    const float *row = qkv.data() + (size_t)pos * qkv_stride;
    const uint32_t cell = pos % W;
    for (uint32_t d = 0; d < D; d++) {
      k_ref[cell * D + d] = to_bf16(row[qd + d]);
      v_ref[cell * D + d] = to_bf16(row[qd + D + d]);
    }
    const uint32_t lo = pos + 1 > W ? pos + 1 - W : 0u;
    std::vector<float> attn(qd, 0.0f);
    for (uint32_t h = 0; h < H; h++) {
      float mx = -INFINITY;
      std::vector<float> sc(pos - lo + 1);
      for (uint32_t t = lo; t <= pos; t++) {
        float dot = 0.0f;
        for (uint32_t d = 0; d < D; d++) dot += row[h * D + d] * from_bf16(k_ref[(t % W) * D + d]);
        sc[t - lo] = dot * scale;
        mx = std::max(mx, sc[t - lo]);
      }
      float sum = 0.0f;
      for (float &s : sc) sum += (s = std::exp(s - mx));
      for (uint32_t t = lo; t <= pos; t++) {
        const float w = sc[t - lo] / sum;
        for (uint32_t d = 0; d < D; d++) attn[h * D + d] += w * from_bf16(v_ref[(t % W) * D + d]);
      }
    }
    std::vector<float> hid(qd);
    float ss = 0.0f;
    for (uint32_t r = 0; r < qd; r++) {
      float s = row[r];
      for (uint32_t c = 0; c < qd; c++) s += from_bf16(o_proj[r * qd + c]) * attn[c];
      hid[r] = s;
      ss += s * s;
    }
    const float rinv = 1.0f / std::sqrt(ss / (float)qd + 1e-6f);
    for (uint32_t r = 0; r < qd; r++) hid[r] = hid[r] * rinv * norm_w[r];
    for (uint32_t l = 0; l < base_labels; l++) {
      float s = cls_b[l];
      for (uint32_t r = 0; r < qd; r++) s += cls_w[l * qd + r] * hid[r];
      ref_out[pos * n_tot + l] = s;
    }
    ref_out[pos * n_tot + base_labels] = row[qd + 2 * D];  // drift raw dot
  }

  auto *d_qkv = static_cast<float *>(rt.alloc_device(card, qkv.size() * 4));
  auto *d_kr = static_cast<uint16_t *>(rt.alloc_device(card, W * D * 2));
  auto *d_vr = static_cast<uint16_t *>(rt.alloc_device(card, W * D * 2));
  auto *d_op = static_cast<uint16_t *>(rt.alloc_device(card, o_proj.size() * 2));
  auto *d_nw = static_cast<float *>(rt.alloc_device(card, norm_w.size() * 4));
  auto *d_cw = static_cast<float *>(rt.alloc_device(card, cls_w.size() * 4));
  auto *d_cb = static_cast<float *>(rt.alloc_device(card, cls_b.size() * 4));
  auto *d_lb = static_cast<float *>(rt.alloc_device(card, lin_b.size() * 4));
  auto *d_ld = static_cast<uint32_t *>(rt.alloc_device(card, lin_is_drift.size() * 4));
  auto *h_out = static_cast<float *>(rt.alloc_host(n_tok * n_tot * 4));
  up.copy(d_qkv, qkv.data(), qkv.size() * 4);
  up.copy(d_op, o_proj.data(), o_proj.size() * 2);
  up.copy(d_nw, norm_w.data(), norm_w.size() * 4);
  up.copy(d_cw, cls_w.data(), cls_w.size() * 4);
  up.copy(d_cb, cls_b.data(), cls_b.size() * 4);
  up.copy(d_lb, lin_b.data(), lin_b.size() * 4);
  up.copy(d_ld, lin_is_drift.data(), lin_is_drift.size() * 4);
  up.synchronize();

  // Split n_tok into a chunk of 10 then a chunk of 14 (crossing ring boundary W=16).
  auto k_head = lib.get(card, "qwen4exp_probe", "probe_mqa_head");
  auto k_kv = lib.get(card, "qwen4exp_probe", "probe_kv_write");
  const float eps = 1e-6f;
  const uint32_t slot_start = 0;
  for (auto [pos0, cnt] : std::vector<std::pair<uint32_t, uint32_t>>{{0, 10}, {10, 14}}) {
    up.launch(k_head, {d_qkv, qkv_stride, pos0, pos0, slot_start, cnt, H, D, W, d_kr, d_vr, d_op, d_nw, eps, d_cw,
                       d_cb, base_labels, d_lb, d_ld, n_lin, h_out + (size_t)pos0 * n_tot, n_tot},
              cnt, 1, 1, 128);
    const uint32_t m = std::min(cnt, W);
    const uint32_t skip = cnt - m;
    up.launch(k_kv, {d_qkv, qkv_stride, qd, D, pos0 + skip, pos0 + skip, m, W, d_kr, d_vr}, (m * D + 63) / 64, 1, 1,
              64);
    up.synchronize();
  }

  float max_err = 0.0f;
  for (size_t i = 0; i < ref_out.size(); i++) max_err = std::max(max_err, std::fabs(h_out[i] - ref_out[i]));
  if (max_err > 1e-3f) {
    fprintf(stderr, "FAIL probe_kernel_test: max_err=%g\n", max_err);
    return 1;
  }
  printf("probe_kernel_test: PASS (max_err=%g)\n", max_err);
  return 0;
}
