// Streaming hidden-state probes: configuration, safetensors/calibration loading, CPU reference forward pass, and
// per-slot rolling-mean and untrusted-span drift tracking. Supports SingProbe attention/MLP checkpoints and custom
// linear or drift alarms (c0ven_probe_v1) without external runtime dependencies.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "engine_api.h"

namespace engine {

enum class ProbeKind : uint32_t {
  None = 0,
  SingProbeAttn = 1,
  SingProbeMlp = 2,
  Linear = 3,
};

struct ProbeLabelSpec {
  std::string name;
  float threshold = 0.5f;
  uint32_t scope = ENGINE_PROBE_SCOPE_INPUT;
  uint32_t flags = 0;
  float cal_scale = 1.0f;
  float cal_bias = 0.0f;
  bool is_linear = false;       // true for c0ven_probe_v1 standalone or extra_alarms sidecar labels
  uint32_t linear_index = 0;    // row in ProbeWeights::linear_w / linear_b
  float linear_bias = 0.0f;     // bias term applied after before/after subtraction for drift labels
};

struct ProbeSpec {
  ProbeKind kind = ProbeKind::None;
  std::string model_name;
  std::string kind_name;
  uint32_t hidden_size = 0;
  std::vector<uint32_t> layer_ids;
  uint32_t num_heads = 0;
  uint32_t head_dim = 0;
  uint32_t window = 0;
  uint32_t intermediate_size = 0;
  uint32_t base_labels = 0;
  uint32_t linear_labels = 0;
  uint32_t roll_window = 10;
  float norm_eps = 1e-6f;
  bool rms_norm_taps = true;
  std::vector<ProbeLabelSpec> labels;

  uint32_t n_labels() const { return (uint32_t)labels.size(); }
  uint32_t n_taps() const { return (uint32_t)layer_ids.size(); }
  uint32_t feat_dim() const { return n_taps() * hidden_size; }
  uint32_t q_dim() const { return num_heads * head_dim; }
  uint32_t kv_dim() const { return head_dim; }  // MQA: 1 KV head
  // Per-slot KV ring in bytes (BF16 K and V), 0 for stateless probes.
  uint64_t slot_kv_bytes() const {
    return kind == ProbeKind::SingProbeAttn ? (uint64_t)2 * window * kv_dim() * 2 : 0;
  }
  engine_probe_info to_c() const;
};

struct ProbeWeights {
  ProbeSpec spec;
  // SingProbeAttn weights (BF16 projections, F32 norm and classifier).
  std::vector<uint16_t> proj_q;  // [q_dim, feat_dim]
  std::vector<uint16_t> proj_k;  // [kv_dim, feat_dim]
  std::vector<uint16_t> proj_v;  // [kv_dim, feat_dim]
  std::vector<uint16_t> o_proj;  // [q_dim, q_dim]
  std::vector<float> norm;       // [q_dim]
  std::vector<float> cls_w;      // [base_labels, q_dim]
  std::vector<float> cls_b;      // [base_labels]
  // SingProbeMlp weights.
  std::vector<uint16_t> fc1_w;   // [intermediate_size, feat_dim]
  std::vector<float> fc1_b;      // [intermediate_size]
  std::vector<float> fc2_w;      // [base_labels, intermediate_size]
  std::vector<float> fc2_b;      // [base_labels]
  // Linear and drift alarm weights (standalone c0ven_probe_v1 or extra_alarms sidecar).
  std::vector<uint16_t> linear_w;  // [linear_labels, feat_dim]
  std::vector<float> linear_b;     // [linear_labels]
};

// Loads config.json, model.safetensors, and optional calibration.json (plus optional extra_alarms) from dir.
// model_layers and model_hidden validate bounds when non-zero.
ProbeWeights load_probe(const std::string &dir, uint32_t model_layers = 0, uint32_t model_hidden = 0);

// Unweighted RMSNorm on one vector of len floats into BF16 (mean-of-streams preprocessing for tapped layers).
void probe_rms_norm_bf16(const float *in, uint32_t len, float eps, uint16_t *out);

// CPU reference state for SingProbeAttn's per-slot BF16 KV ring (used by tests and mock validation).
struct ProbeCpuSlot {
  uint32_t start = 0;
  std::vector<uint16_t> k_ring;  // [window, kv_dim]
  std::vector<uint16_t> v_ring;  // [window, kv_dim]

  void init(const ProbeSpec &spec);
  void reset(uint32_t pos = 0);
};

// Evaluates one token at sequence position pos from concatenated tapped BF16 features [feat_dim].
// Writes spec.n_labels() raw outputs into out (pre-sigmoid logits for normal labels, w.h dot for drift labels).
void probe_eval_cpu(const ProbeWeights &pw, ProbeCpuSlot &slot, uint32_t pos, const uint16_t *feat_bf16, float *out);

// Per-slot streaming score accumulator: rolling mean of calibrated probabilities over roll_window tokens, running
// maximum across prompt (input) and generated (output) tokens, and before/after untrusted-span drift evaluation.
class ProbeTracker {
 public:
  struct SpanState {
    uint32_t start = 0;
    uint32_t end = 0;
    bool pre_ready = false;
    bool post_ready = false;
    std::vector<float> pre_dots;
  };

  struct Snapshot {
    uint32_t n_labels = 0;
    uint32_t roll_window = 10;
    uint32_t in_count = 0;
    std::vector<float> in_ring;    // [roll_window * n_labels]
    std::vector<double> in_sum;    // [n_labels]
    std::vector<float> in_max;     // [n_labels]
    bool in_flagged = false;
    std::vector<float> last_dots;  // [n_labels] at checkpoint position - 1
    bool has_last_dots = false;
    std::vector<SpanState> spans;  // active span pre_dots captured before checkpoint
  };

  ProbeTracker() = default;
  void init(const ProbeSpec *spec, uint32_t prompt_len, uint32_t flags, const engine_token_span *spans,
            uint32_t n_spans);
  void reset();
  bool active() const { return spec_ != nullptr && (flags_ & ENGINE_PROBE_ENABLED) != 0; }
  uint32_t flags() const { return flags_; }
  bool input_flagged() const { return in_flagged_; }
  bool output_flagged() const { return out_flagged_; }
  bool block_input_tripped() const { return active() && (flags_ & ENGINE_PROBE_BLOCK_INPUT) && in_flagged_; }
  bool block_output_tripped() const { return active() && (flags_ & ENGINE_PROBE_BLOCK_OUTPUT) && out_flagged_; }

  // Feeds one committed token's raw outputs [n_labels] at sequence position pos.
  void feed_token(uint32_t pos, const float *raw, uint32_t n);

  Snapshot snapshot() const;
  void restore(const Snapshot &snap, uint32_t checkpoint_pos);
  engine_slot_probe to_c() const;

 private:
  const ProbeSpec *spec_ = nullptr;
  uint32_t prompt_len_ = 0;
  uint32_t flags_ = 0;
  uint32_t roll_window_ = 10;
  uint32_t in_count_ = 0;
  uint32_t out_count_ = 0;
  std::vector<float> in_ring_;
  std::vector<float> out_ring_;
  std::vector<double> in_sum_;
  std::vector<double> out_sum_;
  std::vector<float> in_max_;
  std::vector<float> out_max_;
  bool in_flagged_ = false;
  bool out_flagged_ = false;
  std::vector<SpanState> spans_;
  std::vector<float> last_dots_;
  bool has_last_dots_ = false;

  void update_drift(uint32_t pos, const float *raw);
};

}  // namespace engine
