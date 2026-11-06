#include "models/qwen4exp/shape.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <variant>

#include "models/qwen4exp/kernel_sizes.h"
#include "models/qwen4exp/layout.h"
#include "quant.h"

namespace engine::qwen4exp {

namespace {

const GgufValue &value(const std::map<std::string, GgufValue> &meta, const std::string &key) {
  auto it = meta.find("qwen4exp." + key);
  if (it == meta.end()) throw std::runtime_error("GGUF key missing: qwen4exp." + key);
  return it->second;
}

uint32_t size_of(const std::map<std::string, GgufValue> &meta, const std::string &key) {
  const int64_t *v = std::get_if<int64_t>(&value(meta, key));
  if (!v) throw std::runtime_error("GGUF key is not an integer: qwen4exp." + key);
  if (*v < 0 || *v > UINT32_MAX) throw std::runtime_error("GGUF key is out of range: qwen4exp." + key);
  return (uint32_t)*v;
}

float real_of(const std::map<std::string, GgufValue> &meta, const std::string &key) {
  const GgufValue &v = value(meta, key);
  if (const double *d = std::get_if<double>(&v)) return (float)*d;
  if (const int64_t *i = std::get_if<int64_t>(&v)) return (float)*i;
  throw std::runtime_error("GGUF key is not a number: qwen4exp." + key);
}

const std::vector<int64_t> &ints_of(const std::map<std::string, GgufValue> &meta, const std::string &key) {
  const auto *v = std::get_if<std::vector<int64_t>>(&value(meta, key));
  if (!v) throw std::runtime_error("GGUF key is not an integer array: qwen4exp." + key);
  return *v;
}

static_assert(KV_PAGE_CELLS == Q_IDX_BLOCK, "a page is one indexer block");
static_assert(Q_ATT_SCALE * Q_ATT_SCALE * Q_ATT_D == 1.0f, "Q_ATT_SCALE is 1 / sqrt(Q_ATT_D)");
static_assert(3 * Q_ROPE_SEC_H >= Q_ROT / 2 - 1 && Q_ROPE_SEC_T + Q_ROPE_SEC_H + Q_ROPE_SEC_W == Q_ROT / 2,
              "rope3 turns every pair by one of the three positions");

// The counts kernels/models/qwen4exp/CMakeLists.txt builds modules for.
constexpr uint32_t kBuiltExperts = 512, kBuiltExpertsSmall = 256;

std::string shape_text(const std::vector<uint64_t> &shape) {
  std::string s;
  for (uint64_t d : shape) s += (s.empty() ? "" : "x") + std::to_string(d);
  return s;
}

}  // namespace

Shape read_shape(const std::map<std::string, GgufValue> &meta, const std::map<std::string, GgufTensor> &tensors) {
  Shape s{};
  s.embd = size_of(meta, "embedding_length");
  s.layers = size_of(meta, "block_count");
  s.hc = size_of(meta, "hyper_connection.count");
  s.hc_lr = size_of(meta, "hyper_connection.low_rank");
  s.gdn_kh = size_of(meta, "ssm.group_count");
  s.gdn_vh = size_of(meta, "ssm.time_step_rank");
  s.gdn_d = size_of(meta, "ssm.state_size");
  s.gdn_inner = size_of(meta, "ssm.inner_size");
  s.gdn_conv = size_of(meta, "ssm.conv_kernel");
  s.att_h = size_of(meta, "attention.head_count");
  s.att_kv = size_of(meta, "attention.head_count_kv");
  s.att_key = size_of(meta, "attention.key_length");
  s.att_value = size_of(meta, "attention.value_length");
  s.rot = size_of(meta, "rope.dimension_count");
  s.rope_base = real_of(meta, "rope.freq_base");
  s.rope_sections = ints_of(meta, "rope.dimension_sections");
  s.eps = real_of(meta, "attention.layer_norm_rms_epsilon");
  s.idx_h = size_of(meta, "attention.indexer.head_count");
  s.idx_d = size_of(meta, "attention.indexer.key_length");
  s.idx_topk = size_of(meta, "attention.indexer.top_k");
  s.compress_ratios = ints_of(meta, "attention.compress_ratios");
  if (!s.compress_ratios.empty())
    s.idx_block = (uint32_t)*std::max_element(s.compress_ratios.begin(), s.compress_ratios.end());
  s.attn_interval = size_of(meta, "full_attention_interval");
  s.experts = size_of(meta, "expert_count");
  s.used = size_of(meta, "expert_used_count");
  s.ff = size_of(meta, "expert_feed_forward_length");
  s.shared_ff = size_of(meta, "expert_shared_feed_forward_length");
  s.ple_heads = (uint32_t)ints_of(meta, "ple.head_vocab_sizes").size();
  s.ple_row = size_of(meta, "embedding_length_per_layer_input");
  s.ple_dil = size_of(meta, "ple.ngram_size");
  auto out = tensors.find("output.weight");
  if (out == tensors.end()) throw std::runtime_error("tensor missing from the model: output.weight");
  if (out->second.shape.size() != 2) throw std::runtime_error("output.weight is not a matrix");
  if (out->second.shape[1] > UINT32_MAX) throw std::runtime_error("output.weight has too many rows");
  s.vocab = (uint32_t)out->second.shape[1];
  return s;
}

std::vector<std::string> shape_mismatches(const Shape &s, const std::map<std::string, GgufTensor> &tensors) {
  std::vector<std::string> bad;
  const struct {
    const char *name;
    uint32_t file, built;
  } sizes[] = {
      {"embedding_length", s.embd, Q_EMBD},
      {"hyper_connection.count", s.hc, Q_HC},
      {"hyper_connection.low_rank", s.hc_lr, Q_HC_LR},
      {"ssm.state_size", s.gdn_d, Q_GDN_D},
      {"ssm.conv_kernel", s.gdn_conv, Q_GDN_CONV},
      {"attention.head_count", s.att_h, Q_ATT_H},
      {"attention.head_count_kv", s.att_kv, Q_ATT_KV},
      {"attention.key_length", s.att_key, Q_ATT_D},
      {"attention.value_length", s.att_value, Q_ATT_D},
      {"rope.dimension_count", s.rot, Q_ROT},
      {"attention.indexer.head_count", s.idx_h, Q_IDX_H},
      {"attention.indexer.key_length", s.idx_d, Q_IDX_D},
      {"attention.indexer.top_k", s.idx_topk, Q_IDX_TOPK},
      {"the largest of attention.compress_ratios", s.idx_block, Q_IDX_BLOCK},
      {"expert_used_count", s.used, Q_USED},
      {"ple.ngram_size", s.ple_dil, Q_PLE_DIL},
  };
  for (const auto &f : sizes)
    if (f.file != f.built)
      bad.push_back(std::string(f.name) + " is " + std::to_string(f.file) + ", this build has " +
                    std::to_string(f.built));
  if (s.experts != kBuiltExperts && s.experts != kBuiltExpertsSmall)
    bad.push_back("expert_count is " + std::to_string(s.experts) + ", this build has kernels for 512 and 256");
  if (s.rope_base != Q_ROPE_BASE) bad.push_back("rope.freq_base differs from this build's");
  if (s.rope_sections != std::vector<int64_t>{Q_ROPE_SEC_T, Q_ROPE_SEC_H, Q_ROPE_SEC_W, 0})
    bad.push_back("rope.dimension_sections differs from this build's " + std::to_string(Q_ROPE_SEC_T) + ", " +
                  std::to_string(Q_ROPE_SEC_H) + ", " + std::to_string(Q_ROPE_SEC_W) + ", 0");
  if (s.eps != Q_EPS) bad.push_back("attention.layer_norm_rms_epsilon differs from this build's");

  // Sizes the code divides by or walks over.
  const struct {
    const char *name;
    uint32_t value;
  } counts[] = {
      {"block_count", s.layers},
      {"ssm.group_count", s.gdn_kh},
      {"full_attention_interval", s.attn_interval},
      {"expert_feed_forward_length", s.ff},
      {"output.weight's rows", s.vocab},
  };
  for (const auto &c : counts)
    if (c.value == 0) bad.push_back(std::string(c.name) + " is 0");
  if (!bad.empty()) return bad;

  // How the sizes fit together: each key head has the same number of value heads, the shared expert is as wide as the
  // others, and the n-gram heads' rows make one embedding (docs/spec/MODEL_LAYER.md, MODEL_IO.md).
  if (s.gdn_vh % s.gdn_kh)
    bad.push_back("ssm.time_step_rank " + std::to_string(s.gdn_vh) + " is not a multiple of ssm.group_count " +
                  std::to_string(s.gdn_kh));
  if (s.gdn_inner != s.gdn_vh * s.gdn_d) bad.push_back("ssm.inner_size is not time_step_rank x state_size");
  if (s.shared_ff != s.ff) bad.push_back("expert_shared_feed_forward_length differs from expert_feed_forward_length");
  if (s.ple_heads * s.ple_row != s.embd) bad.push_back("the n-gram heads' rows do not make one embedding");

  // Only attention layers have an indexer.
  if (s.compress_ratios.size() != s.layers) {
    bad.push_back("attention.compress_ratios has " + std::to_string(s.compress_ratios.size()) + " values for " +
                  std::to_string(s.layers) + " layers");
  } else {
    for (uint32_t l = 0; l < s.layers; l++)
      if (s.compress_ratios[l] != (s.attn(l) ? (int64_t)s.idx_block : 0))
        bad.push_back("attention.compress_ratios[" + std::to_string(l) + "] is " +
                      std::to_string(s.compress_ratios[l]));
  }

  // The experts' slices are found by these shapes, and each layer's router has a row per expert.
  const struct {
    const char *name;
    std::vector<uint64_t> shape;
  } weights[] = {
      {"ffn_gate_exps.weight", {s.embd, s.ff, s.experts}},
      {"ffn_up_exps.weight", {s.embd, s.ff, s.experts}},
      {"ffn_down_exps.weight", {s.ff, s.embd, s.experts}},
      {"ffn_gate_inp.weight", {s.embd, s.experts}},
  };
  for (uint32_t l = 0; l < s.layers; l++)
    for (const auto &w : weights) {
      const std::string name = "blk." + std::to_string(l) + "." + w.name;
      auto it = tensors.find(name);
      if (it == tensors.end())
        bad.push_back(name + " is missing");
      else if (it->second.shape != w.shape)
        bad.push_back(name + " is " + shape_text(it->second.shape) + ", expected " + shape_text(w.shape));
    }
  return bad;
}

std::string expert_module(const std::string &base, uint32_t experts) {
  return experts == kBuiltExperts ? base : base + "_e" + std::to_string(experts);
}

std::vector<std::string> split_mismatches(const Shape &s, const std::map<std::string, GgufTensor> &tensors, uint32_t n) {
  std::vector<std::string> bad;
  const struct {
    const char *name;
    uint32_t value;
  } shared[] = {
      {"ssm.group_count", s.gdn_kh},
      {"ssm.time_step_rank", s.gdn_vh},
      {"expert_feed_forward_length", s.ff},
      {"output.weight's rows", s.vocab},
  };
  for (const auto &f : shared)
    if (f.value % n) bad.push_back(std::string(f.name) + " " + std::to_string(f.value) + " does not divide equally");
  for (uint32_t l = 0; l < s.layers; l++) {
    const std::string name = "blk." + std::to_string(l) + ".ffn_down_exps.weight";
    auto it = tensors.find(name);
    if (it == tensors.end()) continue;
    const uint32_t block = quant_block_values(it->second.type);
    if (block && (s.ff / n) % block)
      bad.push_back(name + ": a card's " + std::to_string(s.ff / n) + " columns are not whole blocks of " +
                    std::to_string(block));
  }
  return bad;
}

}  // namespace engine::qwen4exp
