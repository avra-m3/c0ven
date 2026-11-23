// Level 1: inkling2_loop Shape extraction, tensor validation, and 1..8-card split checks.
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "models/inkling2_loop/kernel_sizes.h"
#include "models/inkling2_loop/layout.h"
#include "models/inkling2_loop/shape.h"
#include "quant.h"
#include "test_util.h"

using engine::GgufTensor;
using engine::GgufValue;
using engine::inkling2_loop::card_slices;
using engine::inkling2_loop::conv_channels;
using engine::inkling2_loop::read_shape;
using engine::inkling2_loop::shape_mismatches;
using engine::inkling2_loop::split_mismatches;
using engine::inkling2_loop::swa_ring_cap;

namespace {

using Meta = std::map<std::string, GgufValue>;
using Tensors = std::map<std::string, GgufTensor>;

Meta good_meta() {
  Meta m;
  const auto set = [&](const char *k, GgufValue v) { m[std::string("inkling2_loop.") + k] = std::move(v); };
  set("embedding_length", (int64_t)I2_EMBD);
  set("block_count", (int64_t)24);
  set("loop.core_layer_start", (int64_t)6);
  set("loop.core_layer_end", (int64_t)18);
  set("loop.count", (int64_t)3);
  set("loop.gate_rank", (int64_t)I2_GATE_RANK);
  set("loop.varying", false);
  set("context_length", (int64_t)262144);
  set("vocab_size", (int64_t)201024);
  set("vocab_size_unpadded", (int64_t)200058);
  set("attention.head_count", (int64_t)24);
  set("attention.head_count_kv", (int64_t)8);
  set("attention.key_length", (int64_t)I2_HEAD_D);
  set("attention.value_length", (int64_t)I2_HEAD_D);
  set("attention.layer_norm_rms_epsilon", 1e-6);
  set("attention.qk_clip", 22.627417);
  set("attention.relative_length", (int64_t)I2_REL_D);
  set("attention.relative_extent", (int64_t)I2_REL_EXT);
  set("attention.relative_clip", 8.0);
  set("attention.log_scale_floor", (int64_t)8192);
  set("attention.log_scale_alpha", 0.1);
  set("attention.sliding_window", (int64_t)1024);
  std::vector<int64_t> pat;
  for (int i = 0; i < 24; i++) pat.push_back((i % 6) != 5 ? 1 : 0);
  set("attention.sliding_window_pattern", pat);
  set("attn_res.block_size", (int64_t)6);
  set("shortconv.l_cache", (int64_t)I2_CONV);
  set("expert_count", (int64_t)I2_EXPERTS);
  set("expert_used_count", (int64_t)I2_USED);
  set("expert_feed_forward_length", (int64_t)1152);
  set("expert_weights_scale", 8.0);
  set("activation.situ_beta", 4.0);
  set("activation.situ_linear_beta", 25.0);
  set("final_logit_softcapping", 30.0);
  return m;
}

void add(Tensors &t, const std::string &name, std::vector<uint64_t> shape, uint32_t type = QT_Q8_0) {
  GgufTensor g;
  g.name = name;
  g.type = type;
  g.shape = std::move(shape);
  t[name] = g;
}

Tensors good_tensors(uint32_t moe_type = QT_Q8_0, uint32_t dense_type = QT_Q8_0) {
  Tensors t;
  add(t, "token_embd.weight", {I2_EMBD, 201024}, dense_type);
  add(t, "token_embd_norm.weight", {I2_EMBD}, QT_F32);
  add(t, "output_norm.weight", {I2_EMBD}, QT_F32);
  add(t, "output_res_query.weight", {I2_EMBD}, QT_F32);
  add(t, "output.weight", {I2_EMBD, 201024}, dense_type);

  for (uint32_t ip = 0; ip < 24; ip++) {
    const std::string b = "blk." + std::to_string(ip) + ".";
    const bool swa = (ip % 6) != 5;
    add(t, b + "attn_q.weight", {I2_EMBD, 3072}, dense_type);
    add(t, b + "attn_k.weight", {I2_EMBD, 1024}, dense_type);
    add(t, b + "attn_v.weight", {I2_EMBD, 1024}, dense_type);
    add(t, b + "attn_output.weight", {3072, I2_EMBD}, dense_type);
    add(t, b + "attn_gate.weight", {I2_EMBD, 3072}, dense_type);
    add(t, b + "attn_q_clip.weight", {I2_HEAD_D, 24}, QT_F32);
    add(t, b + "attn_sink_proj.weight", {I2_EMBD, 24}, dense_type);
    add(t, b + "attn_rel.weight", {I2_EMBD, 384}, dense_type);
    add(t, b + "attn_rel_bias.weight", {I2_REL_D, I2_REL_EXT}, QT_F32);
    add(t, b + "attn_rel_clip.weight", {I2_REL_D, 24}, QT_F32);
    add(t, b + "attn_k_conv.weight", {I2_CONV, 1024}, QT_F32);
    add(t, b + "attn_v_conv.weight", {I2_CONV, 1024}, QT_F32);
    add(t, b + "attn_post_conv.weight", {I2_CONV, I2_EMBD}, QT_F32);
    add(t, b + "ffn_post_conv.weight", {I2_CONV, I2_EMBD}, QT_F32);
    if (swa) {
      add(t, b + "attn_norm.weight", {I2_EMBD}, QT_F32);
    } else {
      add(t, b + "attn_q_inp_norm.weight", {I2_EMBD}, QT_F32);
      add(t, b + "attn_k_inp_norm.weight", {I2_EMBD}, QT_F32);
      add(t, b + "attn_v_inp_norm.weight", {I2_EMBD}, QT_F32);
      add(t, b + "attn_k_conv_proj.weight", {I2_EMBD, 32}, dense_type);
      add(t, b + "attn_v_conv_proj.weight", {I2_EMBD, 32}, dense_type);
    }
    if (ip > 0) {
      if (swa) add(t, b + "attn_res_query.weight", {I2_EMBD}, QT_F32);
      else add(t, b + "attn_res_query.weight", {I2_EMBD, 3}, QT_F32);
    }
    add(t, b + "ffn_res_query.weight", {I2_EMBD}, QT_F32);
    add(t, b + "ffn_norm.weight", {I2_EMBD}, QT_F32);
    add(t, b + "ffn_gate_inp.weight", {I2_EMBD, I2_EXPERTS}, QT_F32);
    add(t, b + "exp_probs_b.bias", {I2_EXPERTS}, QT_F32);
    add(t, b + "ffn_gate_up_exps.weight", {I2_EMBD, 2304, I2_EXPERTS}, moe_type);
    add(t, b + "ffn_down_exps.weight", {1152, I2_EMBD, I2_EXPERTS}, moe_type);
    if (ip >= 6 && ip < 18) {
      add(t, b + "loop_msg_query.weight", {I2_EMBD, I2_LOOP_MSGS}, QT_F32);
      for (const char *p : {"loop_q_ctrl_", "loop_k_ctrl_", "loop_v_ctrl_", "loop_ffn_ctrl_"}) {
        add(t, b + p + "down.weight", {2 * I2_EMBD, I2_GATE_RANK}, QT_F32);
        add(t, b + p + "gain.weight", {I2_GATE_RANK, I2_EMBD}, QT_F32);
        add(t, b + p + "forget.weight", {I2_GATE_RANK, I2_EMBD}, QT_F32);
      }
    }
  }
  return t;
}

}  // namespace

int main() {
  const Meta m = good_meta();
  const Tensors t = good_tensors();
  const auto s = read_shape(m, t);

  EXPECT(s.logical_layers() == 48, "logical layers = %u", s.logical_layers());
  EXPECT(s.global_layers() == 8 && s.swa_layers() == 40, "global=%u swa=%u", s.global_layers(), s.swa_layers());
  EXPECT(s.phys_of(24) == 6 && s.pass_of(24) == 1, "il 24 -> ip 6 pass 1");
  EXPECT(s.phys_of(47) == 17 && s.pass_of(47) == 2, "il 47 -> ip 17 pass 2");
  EXPECT(!s.is_swa_logical(29) && !s.is_swa_logical(47), "looped global slots");
  EXPECT(s.global_idx(47) == 7 && s.swa_idx(47) == -1, "global_idx(47)");
  EXPECT(s.max_res_blocks() == 4, "max_res_blocks=%u", s.max_res_blocks());
  EXPECT(shape_mismatches(s, t).empty(), "good model has no shape mismatches");

  for (uint32_t n = 1; n <= 8; n++) {
    EXPECT(split_mismatches(s, n, t).empty(), "Q8_0 split on %u cards", n);
    const auto sl = card_slices(s, n);
    uint32_t kv = 0, q = 0, ff = 0, voc = 0;
    for (const auto &c : sl) {
      EXPECT(c.nkv >= 1 && c.nq == c.nkv * 3 && c.ff_n % 32 == 0, "slice %u/%u invalid", c.c, n);
      kv += c.nkv;
      q += c.nq;
      ff += c.ff_n;
      voc += c.vocab_n;
    }
    EXPECT(kv == s.head_kv && q == s.head_q && ff == s.ff && voc == s.vocab_unpadded, "split sum on %u", n);
  }
  EXPECT(conv_channels(s, card_slices(s, 2)[0].nkv) == 5632, "2-card conv channels = 5632");
  EXPECT(swa_ring_cap(s) == 1088, "swa ring cap = 1088");
  EXPECT(!split_mismatches(s, 9, t).empty(), "9 cards rejected");

  // Q4_K / IQ4_XS alignment: 1, 2, 4 cards align 768 * 3 / n to 256, whereas 3 cards gives 768? Wait: 3,3,2 -> nq = 9,9,6 -> 1152,1152,768. 1152 % 256 != 0.
  const Tensors t_q4k = good_tensors(QT_IQ4_NL, QT_Q4_K);
  EXPECT(split_mismatches(s, 1, t_q4k).empty(), "Q4_K 1-card ok");
  EXPECT(split_mismatches(s, 2, t_q4k).empty(), "Q4_K 2-card ok");
  EXPECT(split_mismatches(s, 4, t_q4k).empty(), "Q4_K 4-card ok");
  EXPECT(!split_mismatches(s, 3, t_q4k).empty(), "Q4_K 3-card caught unaligned attn_output");
  EXPECT(!split_mismatches(s, 8, t_q4k).empty(), "Q4_K 8-card caught unaligned attn_output");

  // Missing loop tensor or wrong shape caught.
  Tensors t_bad = t;
  t_bad.erase("blk.6.loop_msg_query.weight");
  EXPECT(!shape_mismatches(s, t_bad).empty(), "missing loop tensor caught");

  std::printf("inkling2_shape_test: ok\n");
  return 0;
}
