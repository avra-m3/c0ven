// Unit tests for streaming hidden-state probes: safetensors loading, SingProbeAttn/MLP/linear forward math,
// calibration and sidecar alarms, 10-token rolling-mean streaming over 64K tokens, drift spans, and checkpoints.
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "api/probe.h"

namespace fs = std::filesystem;

#define CHECK(cond, ...)                                    \
  do {                                                      \
    if (!(cond)) {                                          \
      fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
      fprintf(stderr, __VA_ARGS__);                         \
      fprintf(stderr, "\n");                                \
      exit(1);                                              \
    }                                                       \
  } while (0)

namespace {

uint16_t to_bf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return (uint16_t)((u + (0x7fffu + ((u >> 16) & 1u))) >> 16);
}

struct TensorOut {
  std::string name;
  std::string dtype;
  std::vector<uint64_t> shape;
  std::vector<uint8_t> bytes;
};

void write_safetensors(const fs::path &path, const std::vector<TensorOut> &tensors) {
  std::string hdr = "{";
  uint64_t off = 0;
  for (size_t i = 0; i < tensors.size(); i++) {
    const auto &t = tensors[i];
    if (i) hdr += ",";
    hdr += "\"" + t.name + "\":{\"dtype\":\"" + t.dtype + "\",\"shape\":[";
    for (size_t j = 0; j < t.shape.size(); j++) {
      if (j) hdr += ",";
      hdr += std::to_string(t.shape[j]);
    }
    hdr += "],\"data_offsets\":[" + std::to_string(off) + "," + std::to_string(off + t.bytes.size()) + "]}";
    off += t.bytes.size();
  }
  hdr += "}";
  std::ofstream out(path, std::ios::binary);
  uint64_t n = hdr.size();
  out.write(reinterpret_cast<const char *>(&n), 8);
  out.write(hdr.data(), (std::streamsize)hdr.size());
  for (const auto &t : tensors) out.write(reinterpret_cast<const char *>(t.bytes.data()), (std::streamsize)t.bytes.size());
}

TensorOut make_bf16(const std::string &name, const std::vector<uint64_t> &shape, const std::vector<float> &vals) {
  TensorOut t{name, "BF16", shape, {}};
  t.bytes.resize(vals.size() * 2);
  for (size_t i = 0; i < vals.size(); i++) {
    uint16_t b = to_bf16(vals[i]);
    std::memcpy(t.bytes.data() + i * 2, &b, 2);
  }
  return t;
}

TensorOut make_f32(const std::string &name, const std::vector<uint64_t> &shape, const std::vector<float> &vals) {
  TensorOut t{name, "F32", shape, {}};
  t.bytes.resize(vals.size() * 4);
  std::memcpy(t.bytes.data(), vals.data(), t.bytes.size());
  return t;
}

void write_text(const fs::path &path, const std::string &text) {
  std::ofstream out(path);
  out << text;
}

void test_singprobe_attn_sliding_window_and_calibration(const fs::path &root) {
  // Tiny SingProbeAttn: hidden=4, 2 taps ([1, 3]) -> feat=8, 2 heads x dim=2 -> q_dim=4, window=3, 10 labels.
  const fs::path dir = root / "sing_attn";
  fs::create_directories(dir);
  write_text(dir / "config.json", R"({
    "model_type": "sing_probe_attn",
    "hidden_size": 4,
    "base_model_layer_ids": [1, 3],
    "num_attention_heads": 2,
    "head_dim": 2,
    "sliding_window": 3,
    "num_labels": 10
  })");

  // Identity-like projections so analytical output is easy to check:
  // q = feat[0..3], k = feat[4..5], v = feat[6..7], o_proj = I, norm = 1, cls_w[4] = [1,0,0,0], cls_w[7] = [0,1,0,0].
  std::vector<float> pq(4 * 8, 0.0f), pk(2 * 8, 0.0f), pv(2 * 8, 0.0f), op(4 * 4, 0.0f), nw(4, 1.0f);
  for (uint32_t i = 0; i < 4; i++) pq[i * 8 + i] = 1.0f;
  for (uint32_t i = 0; i < 2; i++) pk[i * 8 + 4 + i] = 1.0f;
  for (uint32_t i = 0; i < 2; i++) pv[i * 8 + 6 + i] = 1.0f;
  for (uint32_t i = 0; i < 4; i++) op[i * 4 + i] = 1.0f;
  std::vector<float> cw(10 * 4, 0.0f), cb(10, 0.0f);
  cw[4 * 4 + 0] = 2.0f;   // agent_safety
  cw[7 * 4 + 1] = 4.0f;   // safe (FLAG_SAFE)
  cw[8 * 4 + 2] = 3.0f;   // unsafe_response
  cb[4] = -0.5f;
  write_safetensors(dir / "model.safetensors", {
      make_bf16("proj_q.weight", {4, 8}, pq),
      make_bf16("proj_k.weight", {2, 8}, pk),
      make_bf16("proj_v.weight", {2, 8}, pv),
      make_bf16("o_proj.weight", {4, 4}, op),
      make_bf16("norm.weight", {4}, nw),
      make_bf16("classifier.weight", {10, 4}, cw),
      make_bf16("classifier.bias", {10}, cb),
  });

  // Sidecar drift alarm sharing layers [1, 3].
  const fs::path side = dir / "drift";
  fs::create_directories(side);
  write_text(side / "config.json", R"({
    "model_type": "c0ven_probe_v1",
    "hidden_size": 4,
    "base_model_layer_ids": [1, 3],
    "labels": [
      {"name": "prompt_injection_drift", "scope": "drift", "threshold": 0.6}
    ]
  })");
  write_safetensors(side / "model.safetensors", {
      make_f32("linear.weight", {1, 8}, {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}),
      make_f32("linear.bias", {1}, {-1.0f}),
  });
  write_text(dir / "calibration.json", R"({
    "model_name": "singprobe-cal",
    "roll_window": 2,
    "thresholds": {"agent_safety": 0.7},
    "scale": {"agent_safety": 1.5},
    "bias": {"agent_safety": 0.25},
    "extra_alarms": "drift"
  })");

  const engine::ProbeWeights pw = engine::load_probe(dir.string(), 48, 4);
  CHECK(pw.spec.kind == engine::ProbeKind::SingProbeAttn, "kind");
  CHECK(pw.spec.model_name == "singprobe-cal", "cal model_name");
  CHECK(pw.spec.n_labels() == 11, "10 singprobe + 1 drift");
  CHECK(pw.spec.slot_kv_bytes() == 2ull * 3 * 2 * 2, "slot_kv_bytes");
  CHECK((pw.spec.labels[7].flags & ENGINE_PROBE_FLAG_SAFE) != 0, "safe flag");

  // Verify sliding window eviction: pos 0 writes large v=[10,0]; by pos 3 (window=3), pos 0 is evicted.
  engine::ProbeCpuSlot slot;
  slot.init(pw.spec);
  std::vector<float> out(11);
  auto feat_of = [](std::vector<float> f) {
    std::vector<uint16_t> b(f.size());
    for (size_t i = 0; i < f.size(); i++) b[i] = to_bf16(f[i]);
    return b;
  };
  auto f_spike = feat_of({0, 1, 0, 0, 1, 0, 10, 0});
  auto f_quiet = feat_of({0, 1, 0, 0, 1, 0, 0, 0});
  engine::probe_eval_cpu(pw, slot, 0, f_spike.data(), out.data());
  const float l0 = out[4];  // sees spike in v[0] -> high agent_safety logit
  engine::probe_eval_cpu(pw, slot, 1, f_quiet.data(), out.data());
  engine::probe_eval_cpu(pw, slot, 2, f_quiet.data(), out.data());
  engine::probe_eval_cpu(pw, slot, 3, f_quiet.data(), out.data());
  const float l3 = out[4];  // pos 0 evicted (window [1, 3] has v=[0,0])
  CHECK(l0 > l3 + 1.0f, "sliding window did not evict pos 0: l0=%f l3=%f", l0, l3);

  // Safe label alone (index 7 high, others low) must NOT trip input_flagged.
  engine::ProbeTracker tr;
  const engine_token_span span{2, 5};
  tr.init(&pw.spec, 6, ENGINE_PROBE_ENABLED | ENGINE_PROBE_BLOCK_INPUT | ENGINE_PROBE_BLOCK_OUTPUT, &span, 1);
  std::vector<float> benign(11, -6.0f);
  benign[7] = 8.0f;  // safe ~ 1.0
  benign[10] = 0.0f;  // drift dot at pre-span pos 1
  for (uint32_t p = 0; p < 4; p++) tr.feed_token(p, benign.data(), 11);
  CHECK(!tr.input_flagged(), "high safe score must not flag input");
  CHECK(tr.to_c().input_scores[7] > 0.99f, "safe score recorded");

  // Drift spike at end of untrusted span (pos 4 == span.end - 1): dot jumps 0 -> +5, diff - 1 = +4 -> sigmoid > 0.98.
  std::vector<float> drifted = benign;
  drifted[10] = 5.0f;
  tr.feed_token(4, drifted.data(), 11);
  CHECK(tr.input_flagged(), "drift alarm must flag at untrusted span end");
  CHECK(tr.block_input_tripped(), "block_input must trip");
  CHECK(tr.to_c().input_scores[10] > 0.95f, "drift score %f", tr.to_c().input_scores[10]);

  // Output token at pos 6 flags unsafe_response.
  std::vector<float> bad_out(11, -5.0f);
  bad_out[8] = 4.0f;
  tr.feed_token(6, bad_out.data(), 11);
  CHECK(tr.output_flagged() && tr.block_output_tripped(), "output probe must flag on pos >= prompt_len");
}

void test_singprobe_mlp_and_64k_rolling_mean_and_checkpoint(const fs::path &root) {
  const fs::path dir = root / "sing_mlp";
  fs::create_directories(dir);
  write_text(dir / "config.json", R"({
    "model_type": "sing_probe_mlp",
    "hidden_size": 2,
    "base_model_layer_ids": [0],
    "intermediate_size": 2,
    "num_labels": 10
  })");
  std::vector<float> w1 = {1.0f, 0.0f, 0.0f, 1.0f};
  std::vector<float> b1 = {0.0f, 0.0f};
  std::vector<float> w2(10 * 2, 0.0f), b2(10, 0.0f);
  w2[0] = 2.0f;  // sexual = 2 * silu(x0)
  write_safetensors(dir / "model.safetensors", {
      make_bf16("fc1.weight", {2, 2}, w1),
      make_f32("fc1.bias", {2}, b1),
      make_f32("fc2.weight", {10, 2}, w2),
      make_f32("fc2.bias", {10}, b2),
  });
  const engine::ProbeWeights pw = engine::load_probe(dir.string(), 4, 2);
  CHECK(pw.spec.kind == engine::ProbeKind::SingProbeMlp, "mlp kind");
  engine::ProbeCpuSlot slot;
  slot.init(pw.spec);
  uint16_t feat[2] = {to_bf16(2.0f), to_bf16(0.0f)};
  float out[10] = {};
  engine::probe_eval_cpu(pw, slot, 0, feat, out);
  const float want = 2.0f * (2.0f / (1.0f + std::exp(-2.0f)));
  CHECK(std::fabs(out[0] - want) < 1e-3f, "mlp logit %f != %f", out[0], want);

  // 65,536-token rolling mean: single-token spike should be diluted by 10-token window; sustained 10-token burst peaks.
  engine::ProbeTracker tr;
  tr.init(&pw.spec, 65536, ENGINE_PROBE_ENABLED, nullptr, 0);
  float low[10], high[10];
  for (int i = 0; i < 10; i++) low[i] = -10.0f, high[i] = -10.0f;
  high[0] = 10.0f;  // sigmoid ~ 0.99995
  // 1-token spike at pos 1000: rolling mean stays ~ 0.10.
  for (uint32_t p = 0; p < 50000; p++) tr.feed_token(p, p == 1000 ? high : low, 10);
  CHECK(tr.to_c().input_scores[0] > 0.09f && tr.to_c().input_scores[0] < 0.12f, "1-tok spike mean %f",
        tr.to_c().input_scores[0]);
  CHECK(!tr.input_flagged(), "1-tok spike must not flag");

  // Snapshot at pos 50000, restore into a second tracker, then feed 10 consecutive high tokens -> flags!
  const auto snap = tr.snapshot();
  engine::ProbeTracker tr2;
  tr2.init(&pw.spec, 65536, ENGINE_PROBE_ENABLED, nullptr, 0);
  tr2.restore(snap, 50000);
  for (uint32_t p = 50000; p < 65536; p++) tr2.feed_token(p, p < 50010 ? high : low, 10);
  CHECK(tr2.to_c().input_scores[0] > 0.99f, "sustained burst mean %f", tr2.to_c().input_scores[0]);
  CHECK(tr2.input_flagged(), "sustained 10-token burst must flag");
}

}  // namespace

int main() {
  const fs::path root = fs::temp_directory_path() / ("c0ven_probe_test_" + std::to_string(::getpid()));
  fs::remove_all(root);
  fs::create_directories(root);
  test_singprobe_attn_sliding_window_and_calibration(root);
  test_singprobe_mlp_and_64k_rolling_mean_and_checkpoint(root);
  fs::remove_all(root);
  printf("probe_test: PASS\n");
  return 0;
}
