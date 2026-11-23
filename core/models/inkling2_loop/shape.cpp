// Metadata extraction and tensor shape / multi-card split validation for inkling2_loop.
#include "models/inkling2_loop/shape.h"

#include <cmath>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <variant>

#include "engine_api.h"
#include "models/inkling2_loop/kernel_sizes.h"
#include "models/inkling2_loop/layout.h"
#include "quant.h"

namespace engine::inkling2_loop {

namespace {

int64_t meta_i64(const std::map<std::string, GgufValue> &meta, const std::string &key) {
  auto it = meta.find(key);
  if (it == meta.end()) throw std::runtime_error("GGUF key missing: " + key);
  if (auto p = std::get_if<int64_t>(&it->second)) return *p;
  if (auto p = std::get_if<bool>(&it->second)) return *p ? 1 : 0;
  throw std::runtime_error("GGUF key is not an integer: " + key);
}

int64_t meta_i64_opt(const std::map<std::string, GgufValue> &meta, const std::string &key, int64_t fallback) {
  auto it = meta.find(key);
  if (it == meta.end()) return fallback;
  if (auto p = std::get_if<int64_t>(&it->second)) return *p;
  if (auto p = std::get_if<bool>(&it->second)) return *p ? 1 : 0;
  throw std::runtime_error("GGUF key is not an integer: " + key);
}

double meta_f64(const std::map<std::string, GgufValue> &meta, const std::string &key) {
  auto it = meta.find(key);
  if (it == meta.end()) throw std::runtime_error("GGUF key missing: " + key);
  if (auto p = std::get_if<double>(&it->second)) return *p;
  if (auto p = std::get_if<int64_t>(&it->second)) return (double)*p;
  throw std::runtime_error("GGUF key is not a number: " + key);
}

double meta_f64_opt(const std::map<std::string, GgufValue> &meta, const std::string &key, double fallback) {
  auto it = meta.find(key);
  if (it == meta.end()) return fallback;
  if (auto p = std::get_if<double>(&it->second)) return *p;
  if (auto p = std::get_if<int64_t>(&it->second)) return (double)*p;
  throw std::runtime_error("GGUF key is not a number: " + key);
}

const std::vector<int64_t> &meta_ints(const std::map<std::string, GgufValue> &meta, const std::string &key) {
  auto it = meta.find(key);
  if (it == meta.end()) throw std::runtime_error("GGUF key missing: " + key);
  if (auto p = std::get_if<std::vector<int64_t>>(&it->second)) return *p;
  throw std::runtime_error("GGUF key is not an array: " + key);
}

uint32_t meta_uniform_or_i64(const std::map<std::string, GgufValue> &meta, const std::string &key) {
  auto it = meta.find(key);
  if (it == meta.end()) throw std::runtime_error("GGUF key missing: " + key);
  if (auto p = std::get_if<int64_t>(&it->second)) {
    if (*p <= 0) throw std::runtime_error("GGUF key must be positive: " + key);
    return (uint32_t)*p;
  }
  if (auto p = std::get_if<std::vector<int64_t>>(&it->second)) {
    if (p->empty()) throw std::runtime_error("GGUF array is empty: " + key);
    const int64_t first = (*p)[0];
    if (first <= 0) throw std::runtime_error("GGUF key must be positive: " + key);
    for (int64_t v : *p) {
      if (v != first) throw std::runtime_error("GGUF array is not uniform: " + key);
    }
    return (uint32_t)first;
  }
  throw std::runtime_error("GGUF key is not an integer or integer array: " + key);
}

std::string shape_str(const std::vector<uint64_t> &v) {
  std::ostringstream o;
  o << "{";
  for (size_t i = 0; i < v.size(); i++) o << (i ? ", " : "") << v[i];
  o << "}";
  return o.str();
}

void check_tensor(std::vector<std::string> &bad, const std::map<std::string, GgufTensor> &tensors,
                  const std::string &name, const std::vector<uint64_t> &want, bool required = true) {
  auto it = tensors.find(name);
  if (it == tensors.end()) {
    if (required) bad.push_back("missing tensor " + name);
    return;
  }
  if (it->second.shape != want) {
    bad.push_back(name + ": shape " + shape_str(it->second.shape) + ", expected " + shape_str(want));
  }
}

bool supported_quant_type(uint32_t type) {
  switch (type) {
    case QT_F32:
    case QT_F16:
    case QT_BF16:
    case QT_Q8_0:
    case QT_IQ4_NL:
    case QT_Q4_0:
    case QT_Q5_0:
    case QT_Q5_1:
    case QT_Q4_K:
    case QT_Q5_K:
    case QT_Q6_K:
    case QT_IQ4_XS: return true;
    default: return false;
  }
}

}  // namespace

uint32_t Shape::global_layers() const {
  uint32_t n = 0;
  const uint32_t L = logical_layers();
  for (uint32_t il = 0; il < L; il++)
    if (!is_swa_logical(il)) n++;
  return n;
}

uint32_t Shape::swa_layers() const { return logical_layers() - global_layers(); }

int32_t Shape::global_idx(uint32_t il) const {
  if (is_swa_logical(il)) return -1;
  int32_t idx = 0;
  for (uint32_t j = 0; j < il; j++)
    if (!is_swa_logical(j)) idx++;
  return idx;
}

int32_t Shape::swa_idx(uint32_t il) const {
  if (!is_swa_logical(il)) return -1;
  int32_t idx = 0;
  for (uint32_t j = 0; j < il; j++)
    if (is_swa_logical(j)) idx++;
  return idx;
}

uint32_t Shape::max_res_blocks() const {
  uint32_t n = 0;
  for (uint32_t ip = 0; ip < phys_layers; ip++) {
    if (ip % res_block_size == 0) n++;
  }
  return n;
}

Shape read_shape(const std::map<std::string, GgufValue> &meta, const std::map<std::string, GgufTensor> &) {
  const auto key = [](const char *k) { return std::string("inkling2_loop.") + k; };
  Shape s;
  s.embd = (uint32_t)meta_i64(meta, key("embedding_length"));
  s.phys_layers = (uint32_t)meta_i64(meta, key("block_count"));
  s.core_start = (uint32_t)meta_i64(meta, key("loop.core_layer_start"));
  s.core_end = (uint32_t)meta_i64(meta, key("loop.core_layer_end"));
  s.loop_count = (uint32_t)meta_i64(meta, key("loop.count"));
  s.gate_rank = (uint32_t)meta_i64(meta, key("loop.gate_rank"));
  s.loop_varying = meta_i64_opt(meta, key("loop.varying"), 0) != 0;

  s.context = (uint32_t)meta_i64(meta, key("context_length"));
  s.vocab = (uint32_t)meta_i64(meta, key("vocab_size"));
  s.vocab_unpadded = (uint32_t)meta_i64(meta, key("vocab_size_unpadded"));

  s.head_q = (uint32_t)meta_i64(meta, key("attention.head_count"));
  s.head_kv = (uint32_t)meta_i64(meta, key("attention.head_count_kv"));
  s.head_d = (uint32_t)meta_i64(meta, key("attention.key_length"));
  const uint32_t head_vd = (uint32_t)meta_i64(meta, key("attention.value_length"));
  if (head_vd != s.head_d) throw std::runtime_error("inkling2_loop: attention.value_length != attention.key_length");

  s.rel_d = (uint32_t)meta_i64(meta, key("attention.relative_length"));
  s.rel_ext = (uint32_t)meta_i64(meta, key("attention.relative_extent"));
  s.sliding_window = (uint32_t)meta_i64(meta, key("attention.sliding_window"));
  s.res_block_size = (uint32_t)meta_i64(meta, key("attn_res.block_size"));
  s.conv = (uint32_t)meta_i64(meta, key("shortconv.l_cache"));

  s.experts = (uint32_t)meta_i64(meta, key("expert_count"));
  s.used = (uint32_t)meta_i64(meta, key("expert_used_count"));
  s.ff = meta_uniform_or_i64(meta, key("expert_feed_forward_length"));

  s.eps = (float)meta_f64(meta, key("attention.layer_norm_rms_epsilon"));
  s.qk_clip = (float)meta_f64(meta, key("attention.qk_clip"));
  s.rel_clip = (float)meta_f64(meta, key("attention.relative_clip"));
  s.situ_beta = (float)meta_f64(meta, key("activation.situ_beta"));
  s.situ_linear_beta = (float)meta_f64(meta, key("activation.situ_linear_beta"));
  s.expert_weights_scale = (float)meta_f64_opt(meta, key("expert_weights_scale"), 1.0);
  s.log_scale_floor = (uint32_t)meta_i64(meta, key("attention.log_scale_floor"));
  s.log_scale_alpha = (float)meta_f64(meta, key("attention.log_scale_alpha"));
  s.softcap = (float)meta_f64_opt(meta, key("final_logit_softcapping"), 0.0);

  const auto pat = meta_ints(meta, key("attention.sliding_window_pattern"));
  s.swa_phys.resize(pat.size());
  for (size_t i = 0; i < pat.size(); i++) s.swa_phys[i] = pat[i] != 0 ? 1 : 0;
  return s;
}

std::vector<std::string> shape_mismatches(const Shape &s, const std::map<std::string, GgufTensor> &tensors) {
  std::vector<std::string> bad;
  if (s.embd != I2_EMBD) bad.push_back("embedding_length=" + std::to_string(s.embd) + ", kernel requires " + std::to_string(I2_EMBD));
  if (s.head_d != I2_HEAD_D) bad.push_back("attention.key_length=" + std::to_string(s.head_d) + ", kernel requires " + std::to_string(I2_HEAD_D));
  if (s.rel_d != I2_REL_D) bad.push_back("attention.relative_length=" + std::to_string(s.rel_d) + ", kernel requires " + std::to_string(I2_REL_D));
  if (s.rel_ext != I2_REL_EXT) bad.push_back("attention.relative_extent=" + std::to_string(s.rel_ext) + ", kernel requires " + std::to_string(I2_REL_EXT));
  if (s.conv != I2_CONV) bad.push_back("shortconv.l_cache=" + std::to_string(s.conv) + ", kernel requires " + std::to_string(I2_CONV));
  if (s.experts != I2_EXPERTS) bad.push_back("expert_count=" + std::to_string(s.experts) + ", kernel requires " + std::to_string(I2_EXPERTS));
  if (s.used != I2_USED) bad.push_back("expert_used_count=" + std::to_string(s.used) + ", kernel requires " + std::to_string(I2_USED));
  if (s.gate_rank != I2_GATE_RANK) bad.push_back("loop.gate_rank=" + std::to_string(s.gate_rank) + ", kernel requires " + std::to_string(I2_GATE_RANK));

  if (!s.phys_layers || s.swa_phys.size() != s.phys_layers)
    bad.push_back("sliding_window_pattern length must match block_count");
  if (!s.res_block_size || !s.core_start || s.core_start >= s.core_end || s.core_end > s.phys_layers ||
      s.core_start % s.res_block_size != 0 || s.core_end % s.res_block_size != 0 ||
      (s.phys_layers - s.core_end) % s.res_block_size != 0)
    bad.push_back("invalid loop core bounds vs attn_res.block_size");
  if (s.loop_count < (s.loop_varying ? 1u : 2u))
    bad.push_back("loop.count must be at least " + std::to_string(s.loop_varying ? 1 : 2));
  if (s.max_res_blocks() + 1 > I2_RES_MAX)
    bad.push_back("attn_res block count exceeds I2_RES_MAX");

  if (!s.head_kv || s.head_q % s.head_kv != 0)
    bad.push_back("attention.head_count must be divisible by head_count_kv");
  if (!s.ff || s.ff % 32 != 0) bad.push_back("expert_feed_forward_length must be a positive multiple of 32");
  if (!s.vocab_unpadded || s.vocab_unpadded > s.vocab)
    bad.push_back("vocab_size_unpadded must be in [1, vocab_size]");
  if (!s.sliding_window || !s.log_scale_floor)
    bad.push_back("sliding_window and log_scale_floor must be positive");
  for (float v : {s.eps, s.qk_clip, s.rel_clip, s.situ_beta, s.situ_linear_beta}) {
    if (!std::isfinite(v) || v <= 0.0f) {
      bad.push_back("norm epsilon, clip thresholds, and SiTU betas must be finite and positive");
      break;
    }
  }
  if (!std::isfinite(s.log_scale_alpha) || s.log_scale_alpha < 0.0f)
    bad.push_back("attention.log_scale_alpha must be finite and non-negative");

  if (!bad.empty()) return bad;

  check_tensor(bad, tensors, "token_embd.weight", {s.embd, s.vocab});
  check_tensor(bad, tensors, "token_embd_norm.weight", {s.embd});
  check_tensor(bad, tensors, "output_norm.weight", {s.embd});
  check_tensor(bad, tensors, "output_res_query.weight", {s.embd});
  check_tensor(bad, tensors, "output.weight", {s.embd, s.vocab});

  const uint64_t qd = s.q_dim(), kvd = s.kv_dim();
  for (uint32_t ip = 0; ip < s.phys_layers; ip++) {
    const std::string b = "blk." + std::to_string(ip) + ".";
    const bool swa = s.is_swa_phys(ip);
    check_tensor(bad, tensors, b + "attn_q.weight", {s.embd, qd});
    check_tensor(bad, tensors, b + "attn_k.weight", {s.embd, kvd});
    check_tensor(bad, tensors, b + "attn_v.weight", {s.embd, kvd});
    check_tensor(bad, tensors, b + "attn_output.weight", {qd, s.embd});
    check_tensor(bad, tensors, b + "attn_gate.weight", {s.embd, qd});
    check_tensor(bad, tensors, b + "attn_q_clip.weight", {s.head_d, s.head_q});
    check_tensor(bad, tensors, b + "attn_sink_proj.weight", {s.embd, s.head_q});
    check_tensor(bad, tensors, b + "attn_rel.weight", {s.embd, (uint64_t)s.rel_d * s.head_q});
    check_tensor(bad, tensors, b + "attn_rel_bias.weight", {s.rel_d, s.rel_ext});
    check_tensor(bad, tensors, b + "attn_rel_clip.weight", {s.rel_d, s.head_q});
    check_tensor(bad, tensors, b + "attn_k_conv.weight", {s.conv, kvd});
    check_tensor(bad, tensors, b + "attn_v_conv.weight", {s.conv, kvd});
    check_tensor(bad, tensors, b + "attn_post_conv.weight", {s.conv, s.embd});
    check_tensor(bad, tensors, b + "ffn_post_conv.weight", {s.conv, s.embd});

    if (swa) {
      check_tensor(bad, tensors, b + "attn_norm.weight", {s.embd});
    } else {
      check_tensor(bad, tensors, b + "attn_q_inp_norm.weight", {s.embd});
      check_tensor(bad, tensors, b + "attn_k_inp_norm.weight", {s.embd});
      check_tensor(bad, tensors, b + "attn_v_inp_norm.weight", {s.embd});
      check_tensor(bad, tensors, b + "attn_k_conv_proj.weight", {s.embd, (uint64_t)s.conv * s.head_kv});
      check_tensor(bad, tensors, b + "attn_v_conv_proj.weight", {s.embd, (uint64_t)s.conv * s.head_kv});
    }
    if (ip > 0) {
      if (swa) check_tensor(bad, tensors, b + "attn_res_query.weight", {s.embd});
      else check_tensor(bad, tensors, b + "attn_res_query.weight", {s.embd, 3});
    }
    check_tensor(bad, tensors, b + "ffn_res_query.weight", {s.embd});
    check_tensor(bad, tensors, b + "ffn_norm.weight", {s.embd});
    check_tensor(bad, tensors, b + "ffn_gate_inp.weight", {s.embd, s.experts});
    check_tensor(bad, tensors, b + "exp_probs_b.bias", {s.experts}, false);
    check_tensor(bad, tensors, b + "ffn_gate_up_exps.weight", {s.embd, (uint64_t)2 * s.ff, s.experts});
    check_tensor(bad, tensors, b + "ffn_down_exps.weight", {s.ff, s.embd, s.experts});

    if (s.is_loop_core(ip)) {
      check_tensor(bad, tensors, b + "loop_msg_query.weight", {s.embd, I2_LOOP_MSGS});
      for (const char *pfx : {"loop_q_ctrl_", "loop_k_ctrl_", "loop_v_ctrl_", "loop_ffn_ctrl_"}) {
        check_tensor(bad, tensors, b + pfx + "down.weight", {(uint64_t)2 * s.embd, s.gate_rank});
        check_tensor(bad, tensors, b + pfx + "gain.weight", {s.gate_rank, s.embd});
        check_tensor(bad, tensors, b + pfx + "forget.weight", {s.gate_rank, s.embd});
      }
    }
  }

  for (const auto &[name, t] : tensors) {
    if (!supported_quant_type(t.type))
      bad.push_back(name + ": unsupported tensor type " + std::string(ggml_type_label(t.type)));
  }
  return bad;
}

std::vector<std::string> split_mismatches(const Shape &s, uint32_t ncards,
                                          const std::map<std::string, GgufTensor> &tensors) {
  std::vector<std::string> bad;
  if (ncards == 0 || ncards > ENGINE_MAX_CARDS) {
    bad.push_back("card count " + std::to_string(ncards) + " out of [1, " + std::to_string(ENGINE_MAX_CARDS) + "]");
    return bad;
  }
  if (ncards > s.head_kv) {
    bad.push_back("card count " + std::to_string(ncards) + " exceeds head_count_kv " + std::to_string(s.head_kv));
    return bad;
  }
  uint32_t ff_align = 32;
  for (uint32_t ip = 0; ip < s.phys_layers; ip++) {
    const std::string key = "blk." + std::to_string(ip) + ".ffn_down_exps.weight";
    if (auto it = tensors.find(key); it != tensors.end()) {
      ff_align = std::lcm(ff_align, quant_block_values(it->second.type));
    }
  }
  if (s.ff % ff_align != 0 || s.ff / ff_align < ncards) {
    bad.push_back("expert_feed_forward_length " + std::to_string(s.ff) + " cannot split cleanly across " +
                  std::to_string(ncards) + " cards at alignment " + std::to_string(ff_align));
    return bad;
  }
  const auto slices = card_slices(s, ncards, ff_align);
  for (uint32_t ip = 0; ip < s.phys_layers; ip++) {
    const std::string b = "blk." + std::to_string(ip) + ".";
    if (auto it = tensors.find(b + "attn_output.weight"); it != tensors.end()) {
      const uint32_t qk = quant_block_values(it->second.type);
      for (const auto &cs : slices) {
        if ((cs.q0 * s.head_d) % qk != 0 || (cs.nq * s.head_d) % qk != 0) {
          bad.push_back(b + "attn_output.weight: card " + std::to_string(cs.c) + " Q column slice is not aligned to " +
                        ggml_type_label(it->second.type) + " block size " + std::to_string(qk));
          break;
        }
      }
    }
    if (auto it = tensors.find(b + "ffn_down_exps.weight"); it != tensors.end()) {
      const uint32_t qk = quant_block_values(it->second.type);
      for (const auto &cs : slices) {
        if (cs.ff0 % qk != 0 || cs.ff_n % qk != 0) {
          bad.push_back(b + "ffn_down_exps.weight: card " + std::to_string(cs.c) + " ff column slice is not aligned to " +
                        ggml_type_label(it->second.type) + " block size " + std::to_string(qk));
          break;
        }
      }
    }
  }
  return bad;
}

}  // namespace engine::inkling2_loop
