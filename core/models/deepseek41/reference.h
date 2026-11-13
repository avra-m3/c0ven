// The whole deepseek41 forward pass on the CPU, one token at a time, from ref_ops.h and ggml's dot products: what the
// card's layers are compared with, and what is itself compared with upstream llama.cpp (ADR-0080 items 6 and 11). It
// reads the model through GgufModel's table with O_DIRECT, so nothing of the file enters the page cache: the dense
// tensors once into anonymous memory (5.31 GiB for GSQ-RCO's IQ1_M file), a token's experts and Engram rows as needed.
// Text tokens only: no image spans.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "models/deepseek41/ref_ops.h"

namespace engine::deepseek41 {

// Called during decode with each step's values, under the name upstream's graph gives them (its cb(), layer -1 for the
// ones outside a layer). The values are f32 in upstream's order and last only for the call: n values at v.
//   layer -1: hc_init, result_norm, result_output
//   layer il: engram_rows, engram_kv, engram_gate, engram_out, attn_norm, qr_norm, q, kv, comp_latent, idx_score,
//             attn_derope, attn_wo_a, attn_out, hc_attn_post, ffn_norm, ffn_moe_logits, ffn_moe_out, ffn_shexp,
//             ffn_out, l_out
// comp_latent comes only when a row is made and idx_score only at a layer with its own top-k and rows to score.
// hc_attn_pre, hc_ffn_pre and hc_out are not given: ref::hc_collapse_norm joins the collapse and the norm.
using RefProbe = std::function<void(const char *name, int layer, const float *v, size_t n)>;

struct RefOptions {
  unsigned threads = 16;                  // for the products and the O_DIRECT reads
  uint32_t max_positions = D41_CTX_CAP;   // positions before decode throws; the compressed rows grow with them
};

struct RefStats {
  uint64_t dense_bytes = 0;   // anonymous memory holding the dense tensors
  uint64_t expert_bytes = 0;  // read for routed experts since the model opened, without O_DIRECT's widening
  uint64_t engram_rows = 0;   // Engram rows read since the model opened
};

class RefModel {
 public:
  // Opens the model at path, its first shard when split, and reads its dense tensors. The block count may be below
  // D41_LAYERS (a generated model); every other size must be the layout's, but for the vocabulary, the Engram table's
  // rows and token_embd's and output's rows. Throws when a tensor is missing or has another shape, or when the
  // file's filesystem refuses O_DIRECT.
  explicit RefModel(const std::string &path, const RefOptions &options = {});
  ~RefModel();
  RefModel(const RefModel &) = delete;
  RefModel &operator=(const RefModel &) = delete;

  // Forgets every position: the next decode is at position 0.
  void reset();

  // Runs token at the next position and writes vocab() logits to logits, which may be null. Throws past max_positions
  // or for a token outside the vocabulary.
  void decode(int32_t token, float *logits, const RefProbe &probe = nullptr);

  uint32_t position() const;  // the next decode's position
  uint32_t layers() const;
  uint32_t vocab() const;
  const RefStats &stats() const;

  // Of the last decode, for a layer below layers(): its routed experts, and its index source's top-k, in increasing
  // order, empty when the layer has no compressed rows or none is visible yet.
  const ref::Route &route(uint32_t layer) const;
  const std::vector<uint32_t> &top_k(uint32_t layer) const;

  // What the drafter takes from the model (drafter.h). Of the last decode, D41_EMBD values: the mean of the 4 streams
  // entering the layer, after its Engram rows.
  const float *layer_input(uint32_t layer) const;
  // A token's embedding (D41_EMBD values), and the output head on an x already normed (vocab() logits).
  void embedding(int32_t token, float *out);
  void head(const float *x, float *logits);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace engine::deepseek41
