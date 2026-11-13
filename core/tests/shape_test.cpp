// Level 1 (docs/TESTING.md): read_shape and shape_mismatches on metadata and a tensor table built here. No card and no
// model file.
#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "models/qwen4exp/kernel_sizes.h"
#include "models/qwen4exp/layout.h"
#include "models/qwen4exp/shape.h"
#include "test_util.h"

using engine::GgufTensor;
using engine::GgufValue;
using engine::qwen4exp::expert_module;
using engine::qwen4exp::read_shape;
using engine::qwen4exp::shape_mismatches;
using engine::qwen4exp::split_mismatches;

namespace {

using Meta = std::map<std::string, GgufValue>;
using Tensors = std::map<std::string, GgufTensor>;

// IQ3_S's sizes that layout.h no longer holds.
constexpr uint32_t kLayers = 48, kKeyHeads = 16, kValueHeads = 48, kFF = 640, kVocab = 248320, kExperts = 512;

// The compress ratios of a file of this many layers: the indexer's block on every fourth layer.
std::vector<int64_t> ratios_for(uint32_t layers) {
  std::vector<int64_t> ratios(layers, 0);
  for (uint32_t l = 3; l < layers; l += 4) ratios[l] = Q_IDX_BLOCK;
  return ratios;
}

// The metadata of a file this build runs, as IQ3_S's header has it.
Meta good_meta() {
  Meta m;
  const auto set = [&](const std::string &key, GgufValue v) { m["qwen4exp." + key] = std::move(v); };
  set("embedding_length", (int64_t)Q_EMBD);
  set("block_count", (int64_t)kLayers);
  set("hyper_connection.count", (int64_t)Q_HC);
  set("hyper_connection.low_rank", (int64_t)Q_HC_LR);
  set("ssm.group_count", (int64_t)kKeyHeads);
  set("ssm.time_step_rank", (int64_t)kValueHeads);
  set("ssm.state_size", (int64_t)Q_GDN_D);
  set("ssm.inner_size", (int64_t)(kValueHeads * Q_GDN_D));
  set("ssm.conv_kernel", (int64_t)Q_GDN_CONV);
  set("attention.head_count", (int64_t)Q_ATT_H);
  set("attention.head_count_kv", (int64_t)Q_ATT_KV);
  set("attention.key_length", (int64_t)Q_ATT_D);
  set("attention.value_length", (int64_t)Q_ATT_D);
  set("context_length", (int64_t)262144);
  set("rope.dimension_count", (int64_t)Q_ROT);
  set("rope.freq_base", (double)Q_ROPE_BASE);
  set("rope.dimension_sections", std::vector<int64_t>{Q_ROPE_SEC_T, Q_ROPE_SEC_H, Q_ROPE_SEC_W, 0});
  set("attention.layer_norm_rms_epsilon", (double)Q_EPS);
  set("attention.indexer.head_count", (int64_t)Q_IDX_H);
  set("attention.indexer.key_length", (int64_t)Q_IDX_D);
  set("attention.indexer.top_k", (int64_t)Q_IDX_TOPK);
  set("attention.compress_ratios", ratios_for(kLayers));
  set("full_attention_interval", (int64_t)4);
  set("expert_count", (int64_t)kExperts);
  set("expert_used_count", (int64_t)Q_USED);
  set("expert_feed_forward_length", (int64_t)kFF);
  set("expert_shared_feed_forward_length", (int64_t)kFF);
  set("ple.head_vocab_sizes", std::vector<int64_t>(16, 1));
  set("embedding_length_per_layer_input", (int64_t)160);
  set("ple.ngram_size", (int64_t)Q_PLE_DIL);
  return m;
}

void add(Tensors &t, const std::string &name, std::vector<uint64_t> shape) {
  GgufTensor g;
  g.name = name;
  g.shape = std::move(shape);
  t[name] = g;
}

Tensors tensors_for(uint64_t experts, uint32_t layers = kLayers) {
  Tensors t;
  add(t, "output.weight", {Q_EMBD, kVocab});
  for (uint32_t l = 0; l < layers; l++) {
    const std::string b = "blk." + std::to_string(l) + ".";
    add(t, b + "ffn_gate_exps.weight", {Q_EMBD, kFF, experts});
    add(t, b + "ffn_up_exps.weight", {Q_EMBD, kFF, experts});
    add(t, b + "ffn_down_exps.weight", {kFF, Q_EMBD, experts});
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
  const Tensors tensors = tensors_for(kExperts);
  EXPECT(mismatches(good, tensors).empty(), "a file of this build's shape: '%s'",
         mismatches(good, tensors).at(0).c_str());

  // ISTA-DASLab's Coder: half the experts, in the metadata and the tensors alike. Other counts have no kernels.
  {
    Meta m = good;
    m["qwen4exp.expert_count"] = (int64_t)256;
    EXPECT(mismatches(m, tensors_for(256)).empty(), "a file of 256 experts");
    m["qwen4exp.expert_count"] = (int64_t)128;
    const auto bad = mismatches(m, tensors_for(128));
    EXPECT(bad.size() == 1 && names(bad, "expert_count is 128, this build has kernels for 512 and 256"),
           "a file of 128 experts");
    EXPECT(expert_module("qwen4exp", 512) == "qwen4exp", "the 512 module's name");
    EXPECT(expert_module("qwen4exp_prompt", 256) == "qwen4exp_prompt_e256", "the 256 module's name");
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
    Meta m = good;
    m["qwen4exp.rope.freq_base"] = 10000.0;
    EXPECT(names(mismatches(m, tensors), "rope.freq_base"), "rope base");
  }
  {
    Meta m = good;
    m["qwen4exp.rope.dimension_sections"] = std::vector<int64_t>{16, 16, 0, 0};
    EXPECT(names(mismatches(m, tensors), "rope.dimension_sections"), "rope sections");
  }

  // The sizes that come from the file: another layer count or vocabulary is not a difference.
  {
    Meta m = good;
    m["qwen4exp.block_count"] = (int64_t)24;
    m["qwen4exp.attention.compress_ratios"] = ratios_for(24);
    Tensors t = tensors_for(kExperts, 24);
    t["output.weight"].shape[1] = 1000;
    const auto s = read_shape(m, t);
    EXPECT(shape_mismatches(s, t).empty(), "24 layers: '%s'", shape_mismatches(s, t).at(0).c_str());
    EXPECT(s.layers == 24 && s.vocab == 1000, "the file's sizes");
    EXPECT(s.attn(3) && !s.attn(4) && s.attn(23) && s.attn_layers() == 6, "which layers are attention");
    EXPECT(s.ple_hist() == Q_PLE_HIST, "positions of PLE history");
    EXPECT(s.kv_head_bytes() == KV_HEAD_BYTES, "a KV head's bytes in a cell");
    EXPECT(s.idx_key_bytes() == IDX_KEY_BYTES, "an indexer key's bytes in a cell");
    EXPECT(s.idx_cells() == Q_IDX_CELLS && s.idx_blocks() == Q_IDX_BLOCKS, "the indexer's cells and blocks");
    EXPECT(s.att_splits() == (Q_IDX_CELLS + 63) / 64, "attn_split's parts");
    EXPECT(s.idx_cand() == 2 * (Q_IDX_BLOCKS + 1) + 2, "words per row of the indexer's candidates");
    auto no_conv = s;
    no_conv.gdn_conv = 0;
    EXPECT(no_conv.ple_hist() == 0, "a convolution size of 0 does not wrap");
  }

  // Rules the sizes have to keep.
  {
    Meta m = good;
    m["qwen4exp.ssm.time_step_rank"] = (int64_t)40;
    m["qwen4exp.ssm.inner_size"] = (int64_t)(40 * Q_GDN_D);
    EXPECT(names(mismatches(m, tensors), "ssm.time_step_rank 40 is not a multiple"), "value heads");
    m = good;
    m["qwen4exp.ssm.inner_size"] = (int64_t)1;
    EXPECT(names(mismatches(m, tensors), "ssm.inner_size"), "inner size");
    m = good;
    m["qwen4exp.expert_shared_feed_forward_length"] = (int64_t)320;
    EXPECT(names(mismatches(m, tensors), "expert_shared_feed_forward_length"), "shared expert width");
    m = good;
    m["qwen4exp.embedding_length_per_layer_input"] = (int64_t)100;
    EXPECT(names(mismatches(m, tensors), "n-gram heads"), "n-gram rows");
    for (const char *key : {"block_count", "ssm.group_count", "full_attention_interval", "expert_feed_forward_length"}) {
      m = good;
      m[std::string("qwen4exp.") + key] = (int64_t)0;
      EXPECT(names(mismatches(m, tensors), std::string(key) + " is 0"), "%s of 0", key);
    }
  }

  // Which layers have an indexer.
  {
    Meta m = good;
    std::get<std::vector<int64_t>>(m["qwen4exp.attention.compress_ratios"])[0] = Q_IDX_BLOCK;
    EXPECT(names(mismatches(m, tensors), "attention.compress_ratios[0] is 4"), "a ratio on a recurrent layer");
    m = good;
    std::get<std::vector<int64_t>>(m["qwen4exp.attention.compress_ratios"]).pop_back();
    EXPECT(names(mismatches(m, tensors), "has 47 values for 48 layers"), "a ratio short");
    m = good;
    m["qwen4exp.attention.compress_ratios"] = std::vector<int64_t>(kLayers, 0);
    EXPECT(names(mismatches(m, tensors), "compress_ratios is 0"), "no layer with a ratio");
  }

  // Sharing between cards.
  {
    const auto s = read_shape(good, tensors);
    EXPECT(split_mismatches(s, tensors, 1).empty() && split_mismatches(s, tensors, 2).empty(), "one and two cards");
    Tensors t = tensors;
    t["output.weight"].shape[1] = 1001;
    const auto odd = read_shape(good, t);
    EXPECT(split_mismatches(odd, t, 1).empty(), "any vocabulary on one card");
    EXPECT(names(split_mismatches(odd, t, 2), "output.weight's rows 1001 does not divide equally"), "odd vocabulary");
    Meta m = good;
    m["qwen4exp.ssm.group_count"] = (int64_t)15;
    EXPECT(names(split_mismatches(read_shape(m, tensors), tensors, 2), "ssm.group_count 15"), "odd key heads");
    t = tensors;
    t["blk.2.ffn_down_exps.weight"].type = QT_Q2_0;
    EXPECT(split_mismatches(read_shape(good, t), t, 2).empty(), "Q2_0 down: 320 columns are 5 blocks");
    m = good;
    m["qwen4exp.expert_feed_forward_length"] = (int64_t)200;
    EXPECT(names(split_mismatches(read_shape(m, t), t, 2), "blk.2.ffn_down_exps.weight: a card's 100 columns"),
           "down columns off the blocks");
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
