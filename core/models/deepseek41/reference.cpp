#include "models/deepseek41/reference.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>

#include "api/model.h"
#include "cpu/experts.h"
#include "models/deepseek41/engram.h"
#include "models/deepseek41/ref_file.h"
#include "util/parallel.h"

namespace engine::deepseek41 {

namespace {

using Mat = RefMat;

std::string blk(uint32_t il, const char *name) { return "blk." + std::to_string(il) + "." + name; }

}  // namespace

struct RefModel::Impl {
  struct Layer {
    Mat q_a, q_b, kv, out_a, out_b, gate_inp, sh_gate, sh_up, sh_down;
    Mat comp_kv, comp_gate, idx_q_b, idx_proj, engram_kv;  // only at the layers that have them
    std::vector<float> attn_norm, q_a_norm, kv_norm, sinks, ffn_norm, bias;
    std::vector<float> attn_fn, attn_scale, attn_base, ffn_fn, ffn_scale, ffn_base;
    std::vector<float> comp_norm, attn_k, k_norm, engram_gate;
    const GgufTensor *gate_exps = nullptr, *up_exps = nullptr, *down_exps = nullptr;
  };

  RefOptions opt;
  Pool pool;
  RefFile file;
  CpuExperts experts;
  EngramHash hash;
  RefStats stats;
  uint32_t n_layer, n_vocab;
  const GgufTensor *embd, *table;
  Mat output;
  std::vector<float> output_norm;
  std::vector<Layer> layer;
  uint32_t pos = 0;
  std::array<int32_t, 3> history{kNoToken, kNoToken, kNoToken};  // the last token first
  std::vector<float> window;                                     // [layer][D41_WIN][D41_HD], at f16's values
  std::vector<std::vector<float>> rows;                          // [source layer][row][D41_COMP_ROW], at f16's values
  std::vector<float> state_kv, state_sc;                         // [source layer][position % 2][D41_HD], in f32
  std::vector<std::vector<uint32_t>> topk;
  std::vector<uint8_t> keep;                                     // the candidate filter's rows, this step
  std::vector<ref::Route> routes;
  std::vector<float> inputs;                                     // [layer][D41_EMBD]: the streams' mean entering it

  Impl(const std::string &path, const RefOptions &o);

  const GgufTensor &tensor(const std::string &name, std::initializer_list<uint64_t> shape) const {
    return file.tensor(name, shape);
  }
  Mat mat(const std::string &name, uint32_t k, uint32_t rows_) { return file.mat(name, k, rows_); }
  std::vector<float> floats(const std::string &name, std::initializer_list<uint64_t> shape) {
    return file.floats(name, shape);
  }
  void load_layer(uint32_t il);

  void mul(const Mat &m, uint32_t first, uint32_t n, const float *x, float *y) { file.mul(m, first, n, x, y); }
  void mul(const Mat &m, const float *x, float *y) { file.mul(m, x, y); }
  void read_row(const GgufTensor &t, uint64_t row, float *out) { file.read_row(t, row, out); }
  void attention(uint32_t il, const float *x, float *attn, const RefProbe &probe);
  void run_experts(uint32_t il, const ref::Route &r, const float *x, float *y);
  void decode(int32_t token, float *logits, const RefProbe &probe);
  void reset();
};

RefModel::Impl::Impl(const std::string &path, const RefOptions &o)
    : opt(o),
      pool(o.threads > 0 ? o.threads - 1 : 0),
      file(path, o.threads, pool),
      experts(pool),
      hash(EngramParams::from_gguf(file.model)),
      n_layer((uint32_t)file.model.get_int("deepseek41.block_count")) {
  if (n_layer == 0 || n_layer > D41_LAYERS) throw std::runtime_error("deepseek41: a block count outside 1 to 40");
  embd = &file.model.tensor("token_embd.weight");
  if (embd->shape.size() != 2 || embd->shape[0] != D41_EMBD) throw std::runtime_error("token_embd.weight: its shape");
  n_vocab = (uint32_t)embd->shape[1];
  table = &file.model.tensor("per_layer_token_embd.weight");
  if (table->shape.size() != 2 || table->shape[0] != D41_ENG_ROW) {
    throw std::runtime_error("per_layer_token_embd.weight: its shape");
  }
  output = mat("output.weight", D41_EMBD, n_vocab);
  output_norm = floats("output_norm.weight", {D41_EMBD});
  layer.resize(n_layer);
  for (uint32_t il = 0; il < n_layer; il++) load_layer(il);
  window.resize((size_t)n_layer * D41_WIN * D41_HD);
  rows.resize(n_layer);
  state_kv.resize((size_t)n_layer * 2 * D41_HD);
  state_sc.resize((size_t)n_layer * 2 * D41_HD);
  topk.resize(n_layer);
  routes.resize(n_layer);
  inputs.resize((size_t)n_layer * D41_EMBD);
}

void RefModel::Impl::load_layer(uint32_t il) {
  Layer &l = layer[il];
  const auto m = [&](const char *name, uint32_t k, uint32_t r) { return mat(blk(il, name), k, r); };
  const auto f = [&](const char *name, std::initializer_list<uint64_t> s) { return floats(blk(il, name), s); };
  l.q_a = m("attn_q_a.weight", D41_EMBD, D41_QLR);
  l.q_b = m("attn_q_b.weight", D41_QLR, D41_H * D41_HD);
  l.kv = m("attn_kv.weight", D41_EMBD, D41_HD);
  l.out_a = m("attn_output_a.weight", D41_OG_K, D41_OG * D41_OLR);
  l.out_b = m("attn_output_b.weight", D41_OG * D41_OLR, D41_EMBD);
  l.gate_inp = m("ffn_gate_inp.weight", D41_EMBD, D41_EXPERTS);
  l.sh_gate = m("ffn_gate_shexp.weight", D41_EMBD, D41_FF);
  l.sh_up = m("ffn_up_shexp.weight", D41_EMBD, D41_FF);
  l.sh_down = m("ffn_down_shexp.weight", D41_FF, D41_EMBD);
  if (l.sh_gate.type != l.sh_up.type) throw std::runtime_error(blk(il, "ffn_up_shexp.weight") + ": not gate's type");
  l.attn_norm = f("attn_norm.weight", {D41_EMBD});
  l.q_a_norm = f("attn_q_a_norm.weight", {D41_QLR});
  l.kv_norm = f("attn_kv_a_norm.weight", {D41_HD});
  l.sinks = f("attn_sinks.weight", {D41_H});
  l.ffn_norm = f("ffn_norm.weight", {D41_EMBD});
  l.bias = f("exp_probs_b.bias", {D41_EXPERTS});
  l.attn_fn = f("hc_attn_fn.weight", {D41_HC_DIM, D41_MIX});
  l.attn_scale = f("hc_attn_scale.weight", {3});
  l.attn_base = f("hc_attn_base.weight", {D41_MIX});
  l.ffn_fn = f("hc_ffn_fn.weight", {D41_HC_DIM, D41_MIX});
  l.ffn_scale = f("hc_ffn_scale.weight", {3});
  l.ffn_base = f("hc_ffn_base.weight", {D41_MIX});
  const int li = (int)il;
  if (D41_KV_SRC(li) == li) {
    l.comp_kv = m("attn_compressor_kv.weight", D41_EMBD, D41_HD);
    if (D41_RATIO(li) == 2) l.comp_gate = m("attn_compressor_gate.weight", D41_EMBD, D41_HD);
    l.comp_norm = f("attn_compressor_norm.weight", {D41_HD});
    l.attn_k = f("indexer.attn_k.weight", {D41_HD, D41_IDX_D});
    l.k_norm = f("indexer.k_norm.weight", {D41_IDX_D});
  }
  if (D41_IDX_SRC(li) == li && D41_RATIO(li) != 0) {
    l.idx_q_b = m("indexer.attn_q_b.weight", D41_QLR, D41_IDX_H * D41_IDX_D);
    l.idx_proj = m("indexer.proj.weight", D41_EMBD, D41_IDX_H);
  }
  if (D41_ENGRAM(li) >= 0) {
    l.engram_kv = m("engram_kv.weight", D41_ENG_G, D41_ENG_KV);
    l.engram_gate = f("engram_gate.weight", {D41_EMBD, D41_HC});
  }
  l.gate_exps = &tensor(blk(il, "ffn_gate_exps.weight"), {D41_EMBD, D41_FF, D41_EXPERTS});
  l.up_exps = &tensor(blk(il, "ffn_up_exps.weight"), {D41_EMBD, D41_FF, D41_EXPERTS});
  l.down_exps = &tensor(blk(il, "ffn_down_exps.weight"), {D41_FF, D41_EMBD, D41_EXPERTS});
  if (l.gate_exps->type != l.up_exps->type) throw std::runtime_error(blk(il, "ffn_up_exps.weight") + ": not gate's type");
}

void RefModel::Impl::attention(uint32_t il, const float *x, float *attn, const RefProbe &probe) {
  const Layer &l = layer[il];
  const int li = (int)il;
  const uint32_t p = pos, ratio = (uint32_t)D41_RATIO(li);
  const bool yarn = ratio != 0;
  const auto show = [&](const char *name, const float *v, size_t n) {
    if (probe) probe(name, (int)il, v, n);
  };

  std::vector<float> qr(D41_QLR), q((size_t)D41_H * D41_HD), kv(D41_HD);
  mul(l.q_a, x, qr.data());
  ref::rms_norm(qr.data(), l.q_a_norm.data(), D41_QLR, qr.data());
  show("qr_norm", qr.data(), D41_QLR);
  mul(l.q_b, qr.data(), q.data());
  for (uint32_t h = 0; h < D41_H; h++) ref::rope_tail(q.data() + (size_t)h * D41_HD, D41_HD, p, yarn, false);
  show("q", q.data(), q.size());

  mul(l.kv, x, kv.data());
  ref::rms_norm(kv.data(), l.kv_norm.data(), D41_HD, kv.data());
  ref::rope_tail(kv.data(), D41_HD, p, yarn, false);
  show("kv", kv.data(), D41_HD);
  float *cell = window.data() + ((size_t)il * D41_WIN + p % D41_WIN) * D41_HD;
  for (uint32_t d = 0; d < D41_HD; d++) cell[d] = round_f16(kv[d]);

  // The compressed rows this layer reads, and its top-k.
  const std::vector<float> *comp = nullptr;
  topk[il].clear();
  if (ratio != 0) {
    const int src = D41_KV_SRC(li);
    if (src == li) {
      std::vector<float> latent(D41_HD);
      bool made = true;
      if (ratio == 1) {
        mul(l.comp_kv, x, state_kv.data() + (size_t)il * 2 * D41_HD);
        ref::compress(state_kv.data() + (size_t)il * 2 * D41_HD, nullptr, 1, l.comp_norm.data(), latent.data());
      } else {
        const size_t at = ((size_t)il * 2 + p % 2) * D41_HD;
        mul(l.comp_kv, x, state_kv.data() + at);
        mul(l.comp_gate, x, state_sc.data() + at);
        made = p % 2 == 1;
        if (made) {
          ref::compress(state_kv.data() + (size_t)il * 2 * D41_HD, state_sc.data() + (size_t)il * 2 * D41_HD, 2,
                        l.comp_norm.data(), latent.data());
        }
      }
      if (made) {
        show("comp_latent", latent.data(), D41_HD);
        const uint32_t j = p / ratio;
        std::vector<float> row(D41_COMP_ROW);
        ref::compressed_row(latent.data(), l.attn_k.data(), l.k_norm.data(), j * ratio, row.data());
        std::transform(row.begin(), row.end(), row.begin(), round_f16);
        rows[il].insert(rows[il].end(), row.begin(), row.end());
      }
    }
    comp = &rows[(size_t)src];
    const uint32_t n = (p + 1) / ratio;
    const uint32_t isrc = (uint32_t)D41_IDX_SRC(li);
    if (isrc == il && n > 0) {
      std::vector<float> iq((size_t)D41_IDX_H * D41_IDX_D), w(D41_IDX_H), score(n);
      mul(l.idx_q_b, qr.data(), iq.data());
      for (uint32_t h = 0; h < D41_IDX_H; h++) ref::rope_tail(iq.data() + (size_t)h * D41_IDX_D, D41_IDX_D, p, yarn, false);
      mul(l.idx_proj, x, w.data());
      ref::index_score(iq.data(), w.data(), comp->data() + D41_HD, D41_COMP_ROW, n, score.data());
      show("idx_score", score.data(), n);
      // Layer 20 chooses the candidate blocks and takes its own top-k from all its scores; 24, 28, 32 and 36 keep to them.
      if (il == 20) {
        keep = ref::candidates(score.data(), n);
      } else if (il > 20) {
        for (uint32_t s = 0; s < n; s++)
          if (!keep[s]) score[s] = -std::numeric_limits<float>::infinity();
      }
      topk[il] = ref::top_k(score.data(), n, D41_TOPK);
    } else if (isrc != il) {
      topk[il] = topk[isrc];
    }
  }

  // The keys: the window's visible positions, then the top-k's rows.
  const uint32_t nwin = std::min(p + 1, (uint32_t)D41_WIN), nkeys = nwin + (uint32_t)topk[il].size();
  std::vector<float> keys((size_t)nkeys * D41_HD), o((size_t)D41_H * D41_HD);
  std::vector<int32_t> idx(nkeys);
  for (uint32_t i = 0; i < nkeys; i++) {
    idx[i] = (int32_t)i;
    const uint32_t kp = p + 1 - nwin + i;
    const float *from = i < nwin ? window.data() + ((size_t)il * D41_WIN + kp % D41_WIN) * D41_HD
                                 : comp->data() + (size_t)topk[il][i - nwin] * D41_COMP_ROW;
    std::copy(from, from + D41_HD, keys.data() + (size_t)i * D41_HD);
  }
  // ggml's flash attention converts q to the keys' type before its dot products.
  std::vector<float> q16(q.size());
  std::transform(q.begin(), q.end(), q16.begin(), round_f16);
  ref::sparse_attn(q16.data(), keys.data(), D41_HD, idx.data(), nkeys, l.sinks.data(), o.data());
  for (uint32_t h = 0; h < D41_H; h++) ref::rope_tail(o.data() + (size_t)h * D41_HD, D41_HD, p, yarn, true);
  show("attn_derope", o.data(), o.size());

  std::vector<float> a((size_t)D41_OG * D41_OLR);
  for (uint32_t g = 0; g < D41_OG; g++)
    mul(l.out_a, g * D41_OLR, D41_OLR, o.data() + (size_t)g * D41_OG_K, a.data() + (size_t)g * D41_OLR);
  show("attn_wo_a", a.data(), a.size());
  mul(l.out_b, a.data(), attn);
  show("attn_out", attn, D41_EMBD);
}

void RefModel::Impl::run_experts(uint32_t il, const ref::Route &r, const float *x, float *y) {
  const Layer &l = layer[il];
  file.routed(experts, *l.gate_exps, *l.up_exps, *l.down_exps, D41_EXPERTS, r.id, r.w, D41_USED, x, y);
}

void RefModel::Impl::decode(int32_t token, float *logits, const RefProbe &probe) {
  if (pos >= opt.max_positions) throw std::runtime_error("deepseek41 reference: past its positions");
  if (token < 0 || (uint32_t)token >= n_vocab) throw std::runtime_error("deepseek41 reference: a token outside the vocabulary");
  const auto show = [&](const char *name, int il, const float *v, size_t n) {
    if (probe) probe(name, il, v, n);
  };

  std::vector<float> X(D41_HC_DIM), X1(D41_HC_DIM), x(D41_EMBD), attn(D41_EMBD), gate_l(D41_EXPERTS);
  std::vector<float> moe(D41_EMBD), shared(D41_EMBD), ffn(D41_EMBD);
  read_row(*embd, (uint64_t)token, X.data());
  for (uint32_t c = 1; c < D41_HC; c++) std::copy_n(X.data(), D41_EMBD, X.data() + (size_t)c * D41_EMBD);
  show("hc_init", -1, X.data(), X.size());

  float pre[D41_HC] = {1.0f, 0.0f, 0.0f, 0.0f};
  for (uint32_t il = 0; il < n_layer; il++) {
    const Layer &l = layer[il];
    const int e = D41_ENGRAM((int)il);
    if (e >= 0) {
      const auto ids = hash.rows(token, history[0], history[1], history[2], (uint32_t)e);
      std::vector<float> g(D41_ENG_G), kv(D41_ENG_KV);
      for (uint32_t c = 0; c < D41_ENG_COLS; c++) read_row(*table, ids[c], g.data() + (size_t)c * D41_ENG_ROW);
      stats.engram_rows += D41_ENG_COLS;
      show("engram_rows", (int)il, g.data(), g.size());
      mul(l.engram_kv, g.data(), kv.data());
      show("engram_kv", (int)il, kv.data(), kv.size());
      float gates[D41_HC];
      ref::engram_gate(X.data(), kv.data(), l.engram_gate.data(), gates);
      show("engram_gate", (int)il, gates, D41_HC);
      show("engram_out", (int)il, X.data(), X.size());
    }
    float *in = inputs.data() + (size_t)il * D41_EMBD;
    for (uint32_t d = 0; d < D41_EMBD; d++) {
      double sum = 0;
      for (uint32_t c = 0; c < D41_HC; c++) sum += X[(size_t)c * D41_EMBD + d];
      in[d] = (float)(sum / D41_HC);
    }

    const ref::HcMix am = ref::hc_mixes(X.data(), l.attn_fn.data(), l.attn_scale.data(), l.attn_base.data());
    ref::hc_collapse_norm(X.data(), pre, l.attn_norm.data(), x.data());
    show("attn_norm", (int)il, x.data(), x.size());
    attention(il, x.data(), attn.data(), probe);
    ref::hc_post(X.data(), attn.data(), am, X1.data());
    show("hc_attn_post", (int)il, X1.data(), X1.size());

    // Each collapse uses the mixes of the sublayer before it (spec sections 3c and 3f).
    const ref::HcMix fm = ref::hc_mixes(X1.data(), l.ffn_fn.data(), l.ffn_scale.data(), l.ffn_base.data());
    ref::hc_collapse_norm(X1.data(), am.pre, l.ffn_norm.data(), x.data());
    show("ffn_norm", (int)il, x.data(), x.size());
    std::copy_n(fm.pre, D41_HC, pre);

    mul(l.gate_inp, x.data(), gate_l.data());
    show("ffn_moe_logits", (int)il, gate_l.data(), gate_l.size());
    routes[il] = ref::router(gate_l.data(), l.bias.data());
    run_experts(il, routes[il], x.data(), moe.data());
    show("ffn_moe_out", (int)il, moe.data(), moe.size());

    ExpertShape s;
    s.gate_type = l.sh_gate.type;
    s.down_type = l.sh_down.type;
    s.n_embd = D41_EMBD;
    s.n_ff = D41_FF;
    s.swiglu_limit = D41_SWIGLU_LIMIT;
    experts.compute(s, {ExpertWeights{l.sh_gate.data, l.sh_up.data, l.sh_down.data, 1}}, {1.0f}, x.data(), shared.data());
    show("ffn_shexp", (int)il, shared.data(), shared.size());

    for (uint32_t d = 0; d < D41_EMBD; d++) ffn[d] = moe[d] + shared[d];
    show("ffn_out", (int)il, ffn.data(), ffn.size());
    ref::hc_post(X1.data(), ffn.data(), fm, X.data());
    show("l_out", (int)il, X.data(), X.size());
  }

  ref::hc_collapse_norm(X.data(), pre, output_norm.data(), x.data());
  show("result_norm", -1, x.data(), x.size());
  if (logits || probe) {
    std::vector<float> own;
    float *out = logits;
    if (!out) {
      own.resize(n_vocab);
      out = own.data();
    }
    mul(output, x.data(), out);
    show("result_output", -1, out, n_vocab);
  }

  history = {token, history[0], history[1]};
  pos++;
}

void RefModel::Impl::reset() {
  pos = 0;
  history = {kNoToken, kNoToken, kNoToken};
  for (auto &r : rows) r.clear();
  for (auto &t : topk) t.clear();
  keep.clear();
}

RefModel::RefModel(const std::string &path, const RefOptions &options) : impl_(std::make_unique<Impl>(path, options)) {}
RefModel::~RefModel() = default;
void RefModel::reset() { impl_->reset(); }
void RefModel::decode(int32_t token, float *logits, const RefProbe &probe) { impl_->decode(token, logits, probe); }
uint32_t RefModel::position() const { return impl_->pos; }
uint32_t RefModel::layers() const { return impl_->n_layer; }
uint32_t RefModel::vocab() const { return impl_->n_vocab; }
const RefStats &RefModel::stats() const {
  impl_->stats.dense_bytes = impl_->file.dense_bytes;
  impl_->stats.expert_bytes = impl_->file.expert_bytes;
  return impl_->stats;
}
const float *RefModel::layer_input(uint32_t layer) const {
  if (layer >= impl_->n_layer) throw std::runtime_error("deepseek41 reference: no such layer");
  return impl_->inputs.data() + (size_t)layer * D41_EMBD;
}
void RefModel::embedding(int32_t token, float *out) {
  if (token < 0 || (uint32_t)token >= impl_->n_vocab) throw std::runtime_error("deepseek41 reference: a token outside the vocabulary");
  impl_->read_row(*impl_->embd, (uint64_t)token, out);
}
void RefModel::head(const float *x, float *logits) { impl_->mul(impl_->output, x, logits); }
const ref::Route &RefModel::route(uint32_t layer) const { return impl_->routes.at(layer); }
const std::vector<uint32_t> &RefModel::top_k(uint32_t layer) const { return impl_->topk.at(layer); }

}  // namespace engine::deepseek41
