// Metadata-driven dimensions and validation for an inkling2_loop GGUF model.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "load/gguf.h"

namespace engine::inkling2_loop {

struct Shape {
  uint32_t embd = 0;
  uint32_t phys_layers = 0;
  uint32_t core_start = 0;
  uint32_t core_end = 0;
  uint32_t loop_count = 0;
  uint32_t gate_rank = 0;
  bool loop_varying = false;

  uint32_t context = 0;
  uint32_t vocab = 0;
  uint32_t vocab_unpadded = 0;

  uint32_t head_q = 0;
  uint32_t head_kv = 0;
  uint32_t head_d = 0;
  uint32_t rel_d = 0;
  uint32_t rel_ext = 0;
  uint32_t sliding_window = 0;
  uint32_t res_block_size = 0;
  uint32_t conv = 0;

  uint32_t experts = 0;
  uint32_t used = 0;
  uint32_t ff = 0;

  float eps = 1e-6f;
  float qk_clip = 0.0f;
  float rel_clip = 0.0f;
  float situ_beta = 0.0f;
  float situ_linear_beta = 0.0f;
  float expert_weights_scale = 1.0f;
  uint32_t log_scale_floor = 1;
  float log_scale_alpha = 0.0f;
  float softcap = 0.0f;

  // Per physical layer [0 .. phys_layers): 1 if SWA, 0 if global attention.
  std::vector<uint8_t> swa_phys;

  uint32_t core_len() const { return core_end - core_start; }
  uint32_t logical_layers() const { return phys_layers + (loop_count - 1) * core_len(); }
  bool is_loop_core(uint32_t ip) const { return ip >= core_start && ip < core_end; }
  uint32_t phys_of(uint32_t il) const { return il < phys_layers ? il : core_start + (il - phys_layers) % core_len(); }
  uint32_t pass_of(uint32_t il) const { return il < phys_layers ? 0 : 1 + (il - phys_layers) / core_len(); }
  bool is_swa_phys(uint32_t ip) const { return swa_phys.at(ip) != 0; }
  bool is_swa_logical(uint32_t il) const { return is_swa_phys(phys_of(il)); }
  uint32_t gqa_ratio() const { return head_q / head_kv; }
  uint32_t q_dim() const { return head_q * head_d; }
  uint32_t kv_dim() const { return head_kv * head_d; }

  uint32_t global_layers() const;
  uint32_t swa_layers() const;
  int32_t global_idx(uint32_t il) const;
  int32_t swa_idx(uint32_t il) const;
  uint32_t max_res_blocks() const;
};

Shape read_shape(const std::map<std::string, GgufValue> &meta, const std::map<std::string, GgufTensor> &tensors);

// Structural and shape problems on the model's GGUF metadata and tensors; empty when the model can run.
std::vector<std::string> shape_mismatches(const Shape &s, const std::map<std::string, GgufTensor> &tensors);

// Multi-card partitioning problems for `ncards` cards (`1 <= ncards <= ENGINE_MAX_CARDS`); empty when valid.
std::vector<std::string> split_mismatches(const Shape &s, uint32_t ncards,
                                          const std::map<std::string, GgufTensor> &tensors);

}  // namespace engine::inkling2_loop
