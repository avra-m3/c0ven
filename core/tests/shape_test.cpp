// Level 1 (docs/TESTING.md): read_shape and shape_mismatches on metadata and a tensor table built here. No card and no
// model file.
#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "models/qwen4exp/layout.h"
#include "models/qwen4exp/shape.h"
#include "test_util.h"

using engine::GgufTensor;
using engine::GgufValue;
using engine::qwen4exp::read_shape;
using engine::qwen4exp::shape_mismatches;

namespace {

using Meta = std::map<std::string, GgufValue>;
using Tensors = std::map<std::string, GgufTensor>;

// The metadata of a file this build runs, as IQ3_S's header has it.
Meta good_meta() {
  std::vector<int64_t> ratios(Q_LAYERS, 0);
  for (uint32_t l = 0; l < Q_LAYERS; l++)
    if (Q_IS_ATTN(l)) ratios[l] = Q_IDX_BLOCK;
  Meta m;
  const auto set = [&](const std::string &key, GgufValue v) { m["qwen4exp." + key] = std::move(v); };
  set("embedding_length", (int64_t)Q_EMBD);
  set("block_count", (int64_t)Q_LAYERS);
  set("hyper_connection.count", (int64_t)Q_HC);
  set("hyper_connection.low_rank", (int64_t)Q_HC_LR);
  set("ssm.group_count", (int64_t)Q_GDN_KH);
  set("ssm.time_step_rank", (int64_t)Q_GDN_VH);
  set("ssm.state_size", (int64_t)Q_GDN_D);
  set("ssm.inner_size", (int64_t)(Q_GDN_VH * Q_GDN_D));
  set("ssm.conv_kernel", (int64_t)Q_GDN_CONV);
  set("attention.head_count", (int64_t)Q_ATT_H);
  set("attention.head_count_kv", (int64_t)Q_ATT_KV);
  set("attention.key_length", (int64_t)Q_ATT_D);
  set("attention.value_length", (int64_t)Q_ATT_D);
  set("rope.dimension_count", (int64_t)Q_ROT);
  set("rope.freq_base", (double)Q_ROPE_BASE);
  set("attention.layer_norm_rms_epsilon", (double)Q_EPS);
  set("attention.indexer.head_count", (int64_t)Q_IDX_H);
  set("attention.indexer.key_length", (int64_t)Q_IDX_D);
  set("attention.indexer.top_k", (int64_t)Q_IDX_TOPK);
  set("attention.compress_ratios", ratios);
  set("full_attention_interval", (int64_t)4);
  set("expert_count", (int64_t)Q_EXPERTS);
  set("expert_used_count", (int64_t)Q_USED);
  set("expert_feed_forward_length", (int64_t)Q_FF);
  set("expert_shared_feed_forward_length", (int64_t)Q_FF);
  set("ple.head_vocab_sizes", std::vector<int64_t>(Q_PLE_HEADS, 1));
  set("embedding_length_per_layer_input", (int64_t)Q_PLE_ROW);
  set("ple.ngram_size", (int64_t)Q_PLE_DIL);
  return m;
}

void add(Tensors &t, const std::string &name, std::vector<uint64_t> shape) {
  GgufTensor g;
  g.name = name;
  g.shape = std::move(shape);
  t[name] = g;
}

Tensors tensors_for(uint64_t experts) {
  Tensors t;
  add(t, "output.weight", {Q_EMBD, Q_VOCAB});
  for (uint32_t l = 0; l < Q_LAYERS; l++) {
    const std::string b = "blk." + std::to_string(l) + ".";
    add(t, b + "ffn_gate_exps.weight", {Q_EMBD, Q_FF, experts});
    add(t, b + "ffn_up_exps.weight", {Q_EMBD, Q_FF, experts});
    add(t, b + "ffn_down_exps.weight", {Q_FF, Q_EMBD, experts});
    add(t, b + "ffn_gate_inp.weight", {Q_EMBD, experts});
  }
  return t;
}

std::vector<std::string> mismatches(const Meta &m, const Tensors &t) { return shape_mismatches(read_shape(m, t), t); }

bool names(const std::vector<std::string> &bad, const std::string &text) {
  for (const std::string &b : bad)
    if (b.find(text) != std::string::npos) return true;
  return false;
}

std::string read_error(const Meta &m, const Tensors &t) {
  try {
    read_shape(m, t);
  } catch (const std::runtime_error &e) {
    return e.what();
  }
  return "";
}

}  // namespace

int main() {
  const Meta good = good_meta();
  const Tensors tensors = tensors_for(Q_EXPERTS);
  EXPECT(mismatches(good, tensors).empty(), "a file of this build's shape: '%s'",
         mismatches(good, tensors).at(0).c_str());

  // ISTA-DASLab's Coder: half the experts, in the metadata and the tensors alike.
  {
    Meta m = good;
    m["qwen4exp.expert_count"] = (int64_t)256;
    const auto bad = mismatches(m, tensors_for(256));
    EXPECT(bad.size() == 1 && names(bad, "expert_count is 256, this build has 512"), "a file of 256 experts");
  }

  // Metadata that does not describe the tensors.
  {
    const auto bad = mismatches(good, tensors_for(256));
    EXPECT(names(bad, "blk.0.ffn_gate_exps.weight is 2560x640x256, expected 2560x640x512"), "gate shape");
    EXPECT(names(bad, "blk.47.ffn_down_exps.weight"), "down shape on the last layer");
    EXPECT(names(bad, "blk.0.ffn_gate_inp.weight"), "router shape");
  }
  {
    Tensors t = tensors;
    t.erase("blk.5.ffn_up_exps.weight");
    EXPECT(names(mismatches(good, t), "blk.5.ffn_up_exps.weight is missing"), "a missing expert tensor");
  }

  // Every size is compared, and a difference does not hide another.
  {
    Meta m = good;
    m["qwen4exp.embedding_length"] = (int64_t)1024;
    m["qwen4exp.attention.head_count"] = (int64_t)8;
    const auto bad = mismatches(m, tensors);
    EXPECT(names(bad, "embedding_length is 1024"), "embedding_length");
    EXPECT(names(bad, "attention.head_count is 8"), "head_count");
  }
  {
    Tensors t = tensors;
    t["output.weight"].shape[1] = 1000;
    EXPECT(names(mismatches(good, t), "output.weight's rows is 1000"), "vocabulary");
  }
  {
    Meta m = good;
    m["qwen4exp.rope.freq_base"] = 10000.0;
    EXPECT(names(mismatches(m, tensors), "rope.freq_base"), "rope base");
  }

  // Which layers have an indexer.
  {
    Meta m = good;
    std::get<std::vector<int64_t>>(m["qwen4exp.attention.compress_ratios"])[0] = Q_IDX_BLOCK;
    EXPECT(names(mismatches(m, tensors), "attention.compress_ratios[0] is 4"), "a ratio on a recurrent layer");
    m = good;
    std::get<std::vector<int64_t>>(m["qwen4exp.attention.compress_ratios"]).pop_back();
    EXPECT(names(mismatches(m, tensors), "has 47 values for 48 layers"), "a ratio short");
  }

  // What read_shape refuses.
  {
    Meta m = good;
    m.erase("qwen4exp.expert_count");
    EXPECT(read_error(m, tensors).find("GGUF key missing: qwen4exp.expert_count") != std::string::npos, "missing key");
    m = good;
    m["qwen4exp.expert_count"] = std::string("512");
    EXPECT(read_error(m, tensors).find("not an integer") != std::string::npos, "wrong kind");
    m = good;
    m["qwen4exp.expert_count"] = (int64_t)-1;
    EXPECT(read_error(m, tensors).find("out of range") != std::string::npos, "negative");
    Tensors t = tensors;
    t.erase("output.weight");
    EXPECT(read_error(good, t).find("output.weight") != std::string::npos, "no output.weight");
  }

  std::printf("shape_test: ok\n");
  return 0;
}
