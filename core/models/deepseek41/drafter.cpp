#include "models/deepseek41/drafter.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "cpu/experts.h"
#include "models/deepseek41/ref_file.h"
#include "util/parallel.h"

namespace engine::deepseek41 {

namespace {

constexpr uint32_t kNoPos = UINT32_MAX;

std::string blk(uint32_t il, const char *name) { return "blk." + std::to_string(il) + "." + name; }

}  // namespace

struct RefDrafter::Impl {
  struct Layer {
    RefMat q_a, q_b, kv, out_a, out_b, gate_inp, sh_gate, sh_up, sh_down;
    std::vector<float> attn_norm, q_a_norm, kv_norm, sinks, ffn_norm, bias;
    std::vector<float> attn_fn, attn_scale, attn_base, ffn_fn, ffn_scale, ffn_base;
    const GgufTensor *gate_exps = nullptr, *up_exps = nullptr, *down_exps = nullptr;
  };

  RefModel &target;
  Pool pool;
  RefFile file;
  CpuExperts experts;
  uint32_t n_vocab;
  int32_t noise;
  RefMat fc, w2;
  const GgufTensor *w1;
  std::vector<float> enc_norm, output_norm, conf;
  Layer layer[D41_DSP_LAYERS];
  std::vector<float> ring;                // [layer][D41_WIN][D41_HD], at f16's values
  uint32_t ring_pos[D41_WIN];             // the position each cell holds, or kNoPos
  int32_t routes[D41_DSP_LAYERS][D41_DSP_BLOCK][D41_DSP_USED] = {};

  Impl(const std::string &path, RefModel &t, unsigned threads);
  void feed(uint32_t pos, const float *features, const RefProbe &probe);
  std::vector<Draft> draft(int32_t anchor, uint32_t pos, uint32_t n, const DraftChoice &choose, const RefProbe &probe);
  void reset() { std::fill_n(ring_pos, D41_WIN, kNoPos); }
};

RefDrafter::Impl::Impl(const std::string &path, RefModel &t, unsigned threads)
    : target(t), pool(threads > 0 ? threads - 1 : 0), file(path, threads, pool), experts(pool), n_vocab(t.vocab()) {
  if (file.model.get_int("dflash.block_count") != D41_DSP_LAYERS) {
    throw std::runtime_error("deepseek41 drafter: not the layout's layer count");
  }
  noise = (int32_t)file.model.get_int("tokenizer.ggml.mask_token_id");
  if (noise < 0 || (uint32_t)noise >= n_vocab) throw std::runtime_error("deepseek41 drafter: its noise token");
  fc = file.mat("fc.weight", D41_DSP_FEATURES, D41_EMBD);
  enc_norm = file.floats("enc.output_norm.weight", {D41_EMBD});
  output_norm = file.floats("output_norm.weight", {D41_EMBD});
  conf = file.floats("conf_proj.weight", {D41_EMBD + D41_DSP_RANK, 1});
  w1 = &file.tensor("markov_w1.weight", {D41_DSP_RANK, n_vocab});
  w2 = file.mat("markov_w2.weight", D41_DSP_RANK, n_vocab);
  for (uint32_t il = 0; il < D41_DSP_LAYERS; il++) {
    Layer &l = layer[il];
    const auto m = [&](const char *name, uint32_t k, uint32_t r) { return file.mat(blk(il, name), k, r); };
    const auto f = [&](const char *name, std::initializer_list<uint64_t> s) { return file.floats(blk(il, name), s); };
    l.q_a = m("attn_q_a.weight", D41_EMBD, D41_QLR);
    l.q_b = m("attn_q_b.weight", D41_QLR, D41_H * D41_HD);
    l.kv = m("attn_kv.weight", D41_EMBD, D41_HD);
    l.out_a = m("attn_output_a.weight", D41_OG_K, D41_OG * D41_OLR);
    l.out_b = m("attn_output_b.weight", D41_OG * D41_OLR, D41_EMBD);
    l.gate_inp = m("ffn_gate_inp.weight", D41_EMBD, D41_DSP_EXPERTS);
    l.sh_gate = m("ffn_gate_shexp.weight", D41_EMBD, D41_FF);
    l.sh_up = m("ffn_up_shexp.weight", D41_EMBD, D41_FF);
    l.sh_down = m("ffn_down_shexp.weight", D41_FF, D41_EMBD);
    if (l.sh_gate.type != l.sh_up.type) throw std::runtime_error(blk(il, "ffn_up_shexp.weight") + ": not gate's type");
    l.attn_norm = f("attn_norm.weight", {D41_EMBD});
    l.q_a_norm = f("attn_q_a_norm.weight", {D41_QLR});
    l.kv_norm = f("attn_kv_a_norm.weight", {D41_HD});
    l.sinks = f("attn_sinks.weight", {D41_H});
    l.ffn_norm = f("ffn_norm.weight", {D41_EMBD});
    l.bias = f("exp_probs_b.bias", {D41_DSP_EXPERTS});
    l.attn_fn = f("hc_attn_fn.weight", {D41_HC_DIM, D41_MIX});
    l.attn_scale = f("hc_attn_scale.weight", {3});
    l.attn_base = f("hc_attn_base.weight", {D41_MIX});
    l.ffn_fn = f("hc_ffn_fn.weight", {D41_HC_DIM, D41_MIX});
    l.ffn_scale = f("hc_ffn_scale.weight", {3});
    l.ffn_base = f("hc_ffn_base.weight", {D41_MIX});
    l.gate_exps = &file.tensor(blk(il, "ffn_gate_exps.weight"), {D41_EMBD, D41_FF, D41_DSP_EXPERTS});
    l.up_exps = &file.tensor(blk(il, "ffn_up_exps.weight"), {D41_EMBD, D41_FF, D41_DSP_EXPERTS});
    l.down_exps = &file.tensor(blk(il, "ffn_down_exps.weight"), {D41_FF, D41_EMBD, D41_DSP_EXPERTS});
    if (l.gate_exps->type != l.up_exps->type) throw std::runtime_error(blk(il, "ffn_up_exps.weight") + ": not gate's type");
  }
  ring.resize((size_t)D41_DSP_LAYERS * D41_WIN * D41_HD);
  reset();
}

void RefDrafter::Impl::feed(uint32_t pos, const float *features, const RefProbe &probe) {
  std::vector<float> x(D41_EMBD), kv(D41_HD);
  file.mul(fc, features, x.data());
  ref::rms_norm(x.data(), enc_norm.data(), D41_EMBD, x.data());
  if (probe) probe("main_x", -1, x.data(), x.size());
  for (uint32_t il = 0; il < D41_DSP_LAYERS; il++) {
    const Layer &l = layer[il];
    file.mul(l.kv, x.data(), kv.data());
    ref::rms_norm(kv.data(), l.kv_norm.data(), D41_HD, kv.data());
    ref::rope_tail(kv.data(), D41_HD, pos, false, false);
    if (probe) probe("kv_injected", (int)il, kv.data(), kv.size());
    float *cell = ring.data() + ((size_t)il * D41_WIN + pos % D41_WIN) * D41_HD;
    for (uint32_t d = 0; d < D41_HD; d++) cell[d] = round_f16(kv[d]);
  }
  ring_pos[pos % D41_WIN] = pos;
}

std::vector<Draft> RefDrafter::Impl::draft(int32_t anchor, uint32_t pos, uint32_t n, const DraftChoice &choose,
                                           const RefProbe &probe) {
  if (n == 0 || n > D41_DSP_BLOCK) throw std::runtime_error("deepseek41 drafter: a block outside 1 to 5 tokens");
  if (anchor < 0 || (uint32_t)anchor >= n_vocab) throw std::runtime_error("deepseek41 drafter: a token outside the vocabulary");
  const auto show = [&](const char *name, int il, const float *v, size_t count) {
    if (probe) probe(name, il, v, count);
  };

  // The fed keys the block sees: the D41_WIN positions before it, those that are stored.
  std::vector<uint32_t> cells;
  for (uint32_t back = std::min(pos, (uint32_t)D41_WIN); back > 0; back--) {
    const uint32_t p = pos - back;
    if (ring_pos[p % D41_WIN] == p) cells.push_back(p % D41_WIN);
  }
  const uint32_t nkeys = (uint32_t)cells.size() + n;
  std::vector<int32_t> idx(nkeys);
  for (uint32_t i = 0; i < nkeys; i++) idx[i] = (int32_t)i;
  const float seen = (float)nkeys;
  show("dsp_keys", -1, &seen, 1);

  // Row after row of each value, so that a probe gets a block as upstream's graph holds it.
  std::vector<float> X((size_t)n * D41_HC_DIM), X1((size_t)n * D41_HC_DIM), x((size_t)n * D41_EMBD);
  std::vector<float> pre((size_t)n * D41_HC, 0.0f);
  for (uint32_t r = 0; r < n; r++) {
    float *Xr = X.data() + (size_t)r * D41_HC_DIM;
    target.embedding(r == 0 ? anchor : noise, Xr);
    for (uint32_t c = 1; c < D41_HC; c++) std::copy_n(Xr, D41_EMBD, Xr + (size_t)c * D41_EMBD);
    pre[(size_t)r * D41_HC] = 1.0f;
  }
  show("hc_init", -1, X.data(), X.size());

  const size_t qn = (size_t)D41_H * D41_HD, an = (size_t)D41_OG * D41_OLR;
  std::vector<float> qr((size_t)n * D41_QLR), q(n * qn), kv((size_t)n * D41_HD), keys((size_t)nkeys * D41_HD);
  std::vector<float> o(n * qn), a(n * an), attn((size_t)n * D41_EMBD), q16(qn);
  std::vector<float> gate_l((size_t)n * D41_DSP_EXPERTS), moe((size_t)n * D41_EMBD), shared((size_t)n * D41_EMBD);
  std::vector<float> ffn((size_t)n * D41_EMBD);
  std::vector<ref::HcMix> am(n), fm(n);

  for (uint32_t il = 0; il < D41_DSP_LAYERS; il++) {
    const Layer &l = layer[il];
    for (uint32_t r = 0; r < n; r++) {
      const float *Xr = X.data() + (size_t)r * D41_HC_DIM;
      float *xr = x.data() + (size_t)r * D41_EMBD;
      am[r] = ref::hc_mixes(Xr, l.attn_fn.data(), l.attn_scale.data(), l.attn_base.data());
      ref::hc_collapse_norm(Xr, pre.data() + (size_t)r * D41_HC, l.attn_norm.data(), xr);

      float *qrr = qr.data() + (size_t)r * D41_QLR, *qq = q.data() + r * qn, *kr = kv.data() + (size_t)r * D41_HD;
      file.mul(l.q_a, xr, qrr);
      ref::rms_norm(qrr, l.q_a_norm.data(), D41_QLR, qrr);
      file.mul(l.q_b, qrr, qq);
      for (uint32_t h = 0; h < D41_H; h++) ref::rope_tail(qq + (size_t)h * D41_HD, D41_HD, pos + r, false, false);
      file.mul(l.kv, xr, kr);
      ref::rms_norm(kr, l.kv_norm.data(), D41_HD, kr);
      ref::rope_tail(kr, D41_HD, pos + r, false, false);
    }
    show("attn_norm", (int)il, x.data(), x.size());
    show("qr_norm", (int)il, qr.data(), qr.size());
    show("q", (int)il, q.data(), q.size());
    show("kv", (int)il, kv.data(), kv.size());

    for (uint32_t i = 0; i < cells.size(); i++) {
      const float *from = ring.data() + ((size_t)il * D41_WIN + cells[i]) * D41_HD;
      std::copy_n(from, D41_HD, keys.data() + (size_t)i * D41_HD);
    }
    std::transform(kv.begin(), kv.end(), keys.begin() + (ptrdiff_t)(cells.size() * D41_HD), round_f16);

    for (uint32_t r = 0; r < n; r++) {
      float *orow = o.data() + r * qn, *ar = a.data() + r * an;
      // ggml's flash attention converts q to the keys' type before its dot products.
      std::transform(q.begin() + (ptrdiff_t)(r * qn), q.begin() + (ptrdiff_t)((r + 1) * qn), q16.begin(), round_f16);
      ref::sparse_attn(q16.data(), keys.data(), D41_HD, idx.data(), nkeys, l.sinks.data(), orow);
      for (uint32_t h = 0; h < D41_H; h++) ref::rope_tail(orow + (size_t)h * D41_HD, D41_HD, pos + r, false, true);
      for (uint32_t g = 0; g < D41_OG; g++)
        file.mul(l.out_a, g * D41_OLR, D41_OLR, orow + (size_t)g * D41_OG_K, ar + (size_t)g * D41_OLR);
      file.mul(l.out_b, ar, attn.data() + (size_t)r * D41_EMBD);
    }
    show("attn_derope", (int)il, o.data(), o.size());
    show("attn_out", (int)il, attn.data(), attn.size());

    for (uint32_t r = 0; r < n; r++) {
      const float *Xr = X.data() + (size_t)r * D41_HC_DIM;
      float *X1r = X1.data() + (size_t)r * D41_HC_DIM, *xr = x.data() + (size_t)r * D41_EMBD;
      ref::hc_post(Xr, attn.data() + (size_t)r * D41_EMBD, am[r], X1r);
      // Each collapse uses the mixes of the sublayer before it, as the model's layers do.
      fm[r] = ref::hc_mixes(X1r, l.ffn_fn.data(), l.ffn_scale.data(), l.ffn_base.data());
      ref::hc_collapse_norm(X1r, am[r].pre, l.ffn_norm.data(), xr);
      std::copy_n(fm[r].pre, D41_HC, pre.data() + (size_t)r * D41_HC);
    }
    show("hc_attn_post", (int)il, X1.data(), X1.size());
    show("ffn_norm", (int)il, x.data(), x.size());

    for (uint32_t r = 0; r < n; r++) {
      const float *xr = x.data() + (size_t)r * D41_EMBD;
      float *gl = gate_l.data() + (size_t)r * D41_DSP_EXPERTS, w[D41_DSP_USED];
      file.mul(l.gate_inp, xr, gl);
      ref::route(gl, l.bias.data(), D41_DSP_EXPERTS, D41_DSP_USED, routes[il][r], w);
      file.routed(experts, *l.gate_exps, *l.up_exps, *l.down_exps, D41_DSP_EXPERTS, routes[il][r], w, D41_DSP_USED, xr,
                  moe.data() + (size_t)r * D41_EMBD);

      ExpertShape s;
      s.gate_type = l.sh_gate.type;
      s.down_type = l.sh_down.type;
      s.n_embd = D41_EMBD;
      s.n_ff = D41_FF;
      s.swiglu_limit = D41_SWIGLU_LIMIT;
      experts.compute(s, {ExpertWeights{l.sh_gate.data, l.sh_up.data, l.sh_down.data, 1}}, {1.0f}, xr,
                      shared.data() + (size_t)r * D41_EMBD);
      for (uint32_t d = 0; d < D41_EMBD; d++) {
        const size_t at = (size_t)r * D41_EMBD + d;
        ffn[at] = moe[at] + shared[at];
      }
      ref::hc_post(X1.data() + (size_t)r * D41_HC_DIM, ffn.data() + (size_t)r * D41_EMBD, fm[r],
                   X.data() + (size_t)r * D41_HC_DIM);
    }
    show("ffn_moe_logits", (int)il, gate_l.data(), gate_l.size());
    show("ffn_moe_out", (int)il, moe.data(), moe.size());
    show("ffn_shexp", (int)il, shared.data(), shared.size());
    show("ffn_out", (int)il, ffn.data(), ffn.size());
    show("l_out", (int)il, X.data(), X.size());
  }

  // The keep chance reads the collapse before its norm; the head reads it after.
  std::vector<float> out((size_t)n * D41_EMBD), normed((size_t)n * D41_EMBD), logits((size_t)n * n_vocab);
  for (uint32_t r = 0; r < n; r++) {
    float *orow = out.data() + (size_t)r * D41_EMBD, *nr = normed.data() + (size_t)r * D41_EMBD;
    ref::hc_collapse(X.data() + (size_t)r * D41_HC_DIM, pre.data() + (size_t)r * D41_HC, orow);
    ref::rms_norm(orow, output_norm.data(), D41_EMBD, nr);
    target.head(nr, logits.data() + (size_t)r * n_vocab);
  }
  show("hc_out", -1, out.data(), out.size());
  show("result_norm", -1, normed.data(), normed.size());
  show("result_output", -1, logits.data(), logits.size());

  std::vector<Draft> drafts(n);
  std::vector<float> row(D41_DSP_RANK), bias(n_vocab), keep(n);
  int32_t prev = anchor;
  for (uint32_t r = 0; r < n; r++) {
    float *lr = logits.data() + (size_t)r * n_vocab;
    const float *orow = out.data() + (size_t)r * D41_EMBD;
    file.read_row(*w1, (uint64_t)prev, row.data());
    file.mul(w2, row.data(), bias.data());
    for (uint32_t v = 0; v < n_vocab; v++) lr[v] += bias[v];
    double sum = 0;
    for (uint32_t d = 0; d < D41_EMBD; d++) sum += (double)conf[d] * orow[d];
    for (uint32_t d = 0; d < D41_DSP_RANK; d++) sum += (double)conf[D41_EMBD + d] * row[d];
    keep[r] = (float)(1.0 / (1.0 + std::exp(-sum)));
    prev = choose ? choose(r, lr) : (int32_t)(std::max_element(lr, lr + n_vocab) - lr);
    if (prev < 0 || (uint32_t)prev >= n_vocab) throw std::runtime_error("deepseek41 drafter: a chosen token outside the vocabulary");
    drafts[r] = Draft{prev, keep[r]};
  }
  show("dsp_logits", -1, logits.data(), logits.size());
  show("dsp_keep", -1, keep.data(), keep.size());
  return drafts;
}

RefDrafter::RefDrafter(const std::string &path, RefModel &target, unsigned threads)
    : impl_(std::make_unique<Impl>(path, target, threads)) {}
RefDrafter::~RefDrafter() = default;
void RefDrafter::reset() { impl_->reset(); }
void RefDrafter::feed(uint32_t pos, const float *features, const RefProbe &probe) { impl_->feed(pos, features, probe); }
void RefDrafter::feed(const RefProbe &probe) {
  RefModel &t = impl_->target;
  if (t.position() == 0) throw std::runtime_error("deepseek41 drafter: the model has run no token");
  if (t.layers() < D41_DSP_TARGET0 + D41_DSP_TARGETS) throw std::runtime_error("deepseek41 drafter: the model lacks the layers it reads");
  std::vector<float> features(D41_DSP_FEATURES);
  for (uint32_t k = 0; k < D41_DSP_TARGETS; k++)
    std::copy_n(t.layer_input(D41_DSP_TARGET0 + k), D41_EMBD, features.data() + (size_t)k * D41_EMBD);
  impl_->feed(t.position() - 1, features.data(), probe);
}
std::vector<Draft> RefDrafter::draft(int32_t anchor, uint32_t pos, uint32_t n, const DraftChoice &choose,
                                     const RefProbe &probe) {
  return impl_->draft(anchor, pos, n, choose, probe);
}
int32_t RefDrafter::noise_token() const { return impl_->noise; }
uint64_t RefDrafter::dense_bytes() const { return impl_->file.dense_bytes; }
uint64_t RefDrafter::expert_bytes() const { return impl_->file.expert_bytes; }
const int32_t *RefDrafter::route(uint32_t layer, uint32_t row) const {
  if (layer >= D41_DSP_LAYERS || row >= D41_DSP_BLOCK) throw std::runtime_error("deepseek41 drafter: no such route");
  return impl_->routes[layer][row];
}

}  // namespace engine::deepseek41
