// qwen4exp's sizes as its file gives them, and the check of a file against what this build can run: the sizes layout.h
// and kernel_sizes.h still fix, and the rules the others have to keep. The keys are in docs/spec/MODEL_LAYER.md, "The
// numbers" (ADR-0062).
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "load/gguf.h"
#include "models/qwen4exp/layout.h"

namespace engine::qwen4exp {

struct Shape {
  uint32_t embd, layers;
  uint32_t hc, hc_lr;
  uint32_t gdn_kh, gdn_vh, gdn_d, gdn_inner, gdn_conv;
  uint32_t att_h, att_kv, att_key, att_value;
  uint32_t rot;
  float rope_base, eps;
  uint32_t idx_h, idx_d, idx_topk;
  uint32_t idx_block;      // the largest of attention.compress_ratios
  uint32_t attn_interval;
  uint32_t experts, used, ff, shared_ff;
  uint32_t vocab;  // output.weight's rows
  uint32_t ple_heads, ple_row, ple_dil;
  std::vector<int64_t> compress_ratios;  // one a layer
  std::vector<int64_t> rope_sections;    // rope.dimension_sections: time, height, width, and a fourth of 0

  bool attn(uint32_t l) const { return (l + 1) % attn_interval == 0; }  // else a Gated DeltaNet layer
  uint32_t attn_layers() const { return layers / attn_interval; }
  uint32_t hc_dim() const { return hc * embd; }
  uint32_t kv_head_bytes() const { return att_key / 32 * Q8_BLOCK; }  // one head of one cell in the 8-bit KV cache
  uint32_t idx_key_bytes() const { return idx_d / 32 * Q8_BLOCK; }  // one indexer key of one cell, the same way
  uint32_t idx_cells() const { return idx_topk + idx_block - 1; }  // the most cells a row attends to
  uint32_t idx_blocks() const { return idx_topk / idx_block; }  // the whole blocks those cells hold
  uint32_t att_splits() const { return (idx_cells() + 63) / 64; }  // attn_split's ATT_SPLITS
  uint32_t idx_cand() const { return 2 * (idx_blocks() + 1) + 2; }  // words per row of the indexer's candidates
  uint32_t ple_hist() const { return gdn_conv ? ple_dil * (gdn_conv - 1) : 0; }  // positions of history
};

// The file's sizes, from GgufModel::metadata() and tensors(). Throws when a key or output.weight is missing, a value
// has the wrong kind, or a size does not fit in 32 bits.
Shape read_shape(const std::map<std::string, GgufValue> &meta, const std::map<std::string, GgufTensor> &tensors);

// One line for each way the file differs from the sizes fixed at build time, has an expert count with no kernels,
// breaks a rule, or has expert tensors that differ from its own sizes; empty when the build can run it. attn(),
// attn_layers(), idx_blocks() and what is made from it need this to be empty: they divide by a size it checks.
std::vector<std::string> shape_mismatches(const Shape &s, const std::map<std::string, GgufTensor> &tensors);

// The module of kernels built for this expert count, from its base name: expert_module("qwen4exp", 256) is
// "qwen4exp_e256". The count must be one shape_mismatches accepts.
std::string expert_module(const std::string &base, uint32_t experts);

// The same for sharing the model between n cards: each card takes an equal share of the heads, the expert width and the
// vocabulary, and a share of a layer's down experts is a whole number of blocks.
std::vector<std::string> split_mismatches(const Shape &s, const std::map<std::string, GgufTensor> &tensors, uint32_t n);

}  // namespace engine::qwen4exp
