// qwen4exp's sizes as its file gives them, and the check of a file against the sizes this build has in layout.h. The
// keys are in docs/spec/MODEL_LAYER.md, "The numbers" (ADR-0062).
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "load/gguf.h"

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
  uint32_t attn_interval;  // layer l is attention when (l + 1) % attn_interval == 0
  uint32_t experts, used, ff, shared_ff;
  uint32_t vocab;  // output.weight's rows
  uint32_t ple_heads, ple_row, ple_dil;
  std::vector<int64_t> compress_ratios;  // one a layer
};

// The file's sizes, from GgufModel::metadata() and tensors(). Throws when a key or output.weight is missing, a value
// has the wrong kind, or a size does not fit in 32 bits.
Shape read_shape(const std::map<std::string, GgufValue> &meta, const std::map<std::string, GgufTensor> &tensors);

// One line for each way the file differs from this build's sizes, or its expert tensors from its own sizes; empty when
// the build can run it.
std::vector<std::string> shape_mismatches(const Shape &s, const std::map<std::string, GgufTensor> &tensors);

}  // namespace engine::qwen4exp
