// deepseek41's drafter on the CPU, from ref_ops.h and ggml's dot products as reference.h is: what a card's drafter
// is compared with, and what is itself compared with upstream llama.cpp's graph for the drafter's file. That file is
// a second GGUF: D41_DSP_LAYERS layers shaped as the model's window-only ones, two tables that bias a draft by the
// token before it, and a row that gives a draft's chance of being kept. The embeddings and output head are the model's.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "models/deepseek41/reference.h"

namespace engine::deepseek41 {

struct Draft {
  int32_t token = 0;
  float keep = 0;  // the drafter's own chance, 0 to 1, that the model keeps this token
};

// Picks draft slot's token from its vocab() logits. Not given, a draft is the largest logit, the lowest id of equals.
using DraftChoice = std::function<int32_t(uint32_t slot, const float *logits)>;

class RefDrafter {
 public:
  // Opens the drafter at path and reads its dense tensors. target must outlive it. Throws when a tensor is missing or
  // has another shape, or when the file's filesystem refuses O_DIRECT.
  RefDrafter(const std::string &path, RefModel &target, unsigned threads = 16);
  ~RefDrafter();
  RefDrafter(const RefDrafter &) = delete;
  RefDrafter &operator=(const RefDrafter &) = delete;

  // Forgets every fed position.
  void reset();

  // Stores position pos's keys from features: D41_DSP_FEATURES values, the model's inputs to layers D41_DSP_TARGET0
  // and the D41_DSP_TARGETS - 1 after it at that position, one after the other (RefModel::layer_input). The probe
  // gets main_x (layer -1) and kv_injected (each layer).
  void feed(uint32_t pos, const float *features, const RefProbe &probe = nullptr);
  // The same for the token target just ran.
  void feed(const RefProbe &probe = nullptr);

  // n drafts, 1 to D41_DSP_BLOCK, of the tokens after anchor, which is the token at position pos and not yet fed.
  // Draft i is chosen from the block's row i plus the bias of the token before it: anchor, then the drafts chosen.
  // The probe gets dsp_keys (the keys a row sees, one value), the block's values row after row under upstream's
  // names, then dsp_logits (n rows of vocab()) and dsp_keep (n values), all at layer -1 but for a layer's own.
  std::vector<Draft> draft(int32_t anchor, uint32_t pos, uint32_t n, const DraftChoice &choose = nullptr,
                           const RefProbe &probe = nullptr);

  int32_t noise_token() const;
  uint64_t dense_bytes() const;
  uint64_t expert_bytes() const;
  // Of the last draft: row r's routed experts at a layer, D41_DSP_USED ids.
  const int32_t *route(uint32_t layer, uint32_t row) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace engine::deepseek41
