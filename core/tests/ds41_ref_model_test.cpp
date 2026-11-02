// Level 1 (docs/TESTING.md): RefModel (core/models/deepseek41/reference.h) on a generated model, against a walk of the
// forward pass written here from direct calls to ref_ops. ref_ops' sizes are the layout's, so the model has the real
// widths with a small vocabulary and Engram table, 26 layers (each kind of layer) and random weights in the file's
// types. Layers share their large tensors' bytes, and experts past the 8th are holes the router's bias never picks.
// It checks the order of the steps, the caches and what layers share; only the real file can check the numbers.
//
//   ds41_ref_model_test --dump <file> --model <first shard>  compares RefModel with upstream's values in a dump, as
//   reference/deepseek41/refgen.cpp writes them, and prints each step's worst relative RMS. Fails above 1e-3.
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "ggml-cpu.h"
#include "ggml.h"
#include "models/deepseek41/engram.h"
#include "models/deepseek41/reference.h"
#include "models/deepseek41/ref_ops.h"
#include "models/deepseek41/rope.h"
#include "quant.h"
#include "test_util.h"

using namespace engine::deepseek41;

namespace {

constexpr uint32_t kVocab = 64, kLive = 8;  // kLive: experts with bytes in the file

struct Weight {
  uint32_t type = 0;
  std::vector<uint64_t> shape;
  std::vector<uint8_t> bytes;   // for experts, the kLive first
  uint64_t file_bytes = 0;      // the tensor's whole size in the file
  uint64_t offset = 0;          // in the file's data section
};

struct Writer {
  std::vector<uint8_t> b;
  template <class T>
  void put(T v) {
    const auto *p = reinterpret_cast<const uint8_t *>(&v);
    b.insert(b.end(), p, p + sizeof v);
  }
  void str(const std::string &s) {
    put<uint64_t>(s.size());
    b.insert(b.end(), s.begin(), s.end());
  }
};

std::mt19937 rng(4101);

std::vector<float> randn(size_t n, float scale) {
  std::normal_distribution<float> nd(0.0f, scale);
  std::vector<float> v(n);
  for (float &x : v) x = nd(rng);
  return v;
}

std::vector<uint8_t> as_bytes(const std::vector<float> &v) {
  std::vector<uint8_t> b(v.size() * 4);
  std::memcpy(b.data(), v.data(), b.size());
  return b;
}

Weight weight(uint32_t type, std::vector<uint64_t> shape, uint64_t live_rows = 0) {
  Weight w;
  w.type = type;
  w.shape = std::move(shape);
  uint64_t elements = 1;
  for (auto d : w.shape) elements *= d;
  w.file_bytes = engine::ggml_type_bytes(type, elements);
  const uint64_t n = live_rows ? live_rows * w.shape[0] : elements;
  if (type == QT_F32) {
    w.bytes = as_bytes(randn(n, 0.01f));
  } else if (type == QT_BF16) {
    w.bytes.resize(n * 2);
    auto *p = reinterpret_cast<ggml_bf16_t *>(w.bytes.data());
    for (float x : randn(n, 0.05f)) *p++ = ggml_fp32_to_bf16(x);
  } else {
    w.bytes = test::random_blocks(type, n / quant_block_values(type), rng);
  }
  return w;
}

Weight floats(std::vector<uint64_t> shape, std::vector<float> v) {
  Weight w;
  w.type = QT_F32;
  w.shape = std::move(shape);
  w.bytes = as_bytes(v);
  w.file_bytes = w.bytes.size();
  return w;
}

// The model: the tensors by name, several names naming one blob. The hash's numbers are small.
struct Model {
  uint32_t layers = 0;
  std::vector<Weight> blobs;
  std::map<std::string, uint32_t> names;
  std::vector<uint64_t> mult, off, mod;
  std::vector<int32_t> token_map;
  std::string path;

  uint32_t add(Weight w) {
    blobs.push_back(std::move(w));
    return (uint32_t)blobs.size() - 1;
  }
  void name(const std::string &n, uint32_t blob) { names[n] = blob; }
  const Weight &at(const std::string &n) const { return blobs[names.at(n)]; }
  const Weight &at(uint32_t il, const std::string &n) const { return at("blk." + std::to_string(il) + "." + n); }
  const float *f(uint32_t il, const std::string &n) const {
    return reinterpret_cast<const float *>(at(il, n).bytes.data());
  }
};

Model make_model(uint32_t layers) {
  Model m;
  m.layers = layers;
  for (uint32_t e = 0; e < kEngramTables; e++) {
    for (uint32_t i = 0; i < D41_ENG_NGRAM; i++) m.mult.push_back((rng() | 1ull) + ((uint64_t)(rng() & 0xffff) << 32));
  }
  uint64_t rows = 0;
  static const uint64_t primes[] = {89, 97, 101, 103, 107, 109, 113, 127};
  for (uint32_t c = 0; c < kEngramTables * D41_ENG_COLS; c++) {
    m.mod.push_back(primes[c % 8]);
    m.off.push_back(rows);
    rows += primes[c % 8];
  }
  for (uint32_t t = 0; t < kVocab; t++) m.token_map.push_back((int32_t)t);

  m.name("token_embd.weight", m.add(weight(QT_Q5_K, {D41_EMBD, kVocab})));
  m.name("output.weight", m.add(weight(QT_Q5_K, {D41_EMBD, kVocab})));
  m.name("output_norm.weight", m.add(floats({D41_EMBD}, randn(D41_EMBD, 1.0f))));
  m.name("per_layer_token_embd.weight", m.add(weight(QT_IQ4_NL, {D41_ENG_ROW, rows})));

  // The large tensors, one of each. Each layer has its own norms, sinks and mixes' scales and bases.
  const uint32_t q_a = m.add(weight(QT_Q8_0, {D41_EMBD, D41_QLR})), q_b = m.add(weight(QT_Q5_K, {D41_QLR, D41_H * D41_HD}));
  const uint32_t kv = m.add(weight(QT_Q8_0, {D41_EMBD, D41_HD}));
  const uint32_t o_a = m.add(weight(QT_Q5_K, {D41_OG_K, D41_OG * D41_OLR}));
  const uint32_t o_b = m.add(weight(QT_Q5_K, {D41_OG * D41_OLR, D41_EMBD}));
  const uint32_t fn_a = m.add(weight(QT_F32, {D41_HC_DIM, D41_MIX})), fn_f = m.add(weight(QT_F32, {D41_HC_DIM, D41_MIX}));
  const uint32_t gi = m.add(weight(QT_BF16, {D41_EMBD, D41_EXPERTS}));
  const uint32_t sg = m.add(weight(QT_Q5_K, {D41_EMBD, D41_FF})), su = m.add(weight(QT_Q5_K, {D41_EMBD, D41_FF}));
  const uint32_t sd = m.add(weight(QT_Q5_K, {D41_FF, D41_EMBD}));
  const uint32_t c_kv = m.add(weight(QT_BF16, {D41_EMBD, D41_HD})), c_g = m.add(weight(QT_BF16, {D41_EMBD, D41_HD}));
  const uint32_t ik = m.add(weight(QT_BF16, {D41_HD, D41_IDX_D}));
  const uint32_t iq = m.add(weight(QT_BF16, {D41_QLR, D41_IDX_H * D41_IDX_D}));
  const uint32_t ip = m.add(weight(QT_BF16, {D41_EMBD, D41_IDX_H}));
  const uint32_t ek = m.add(weight(QT_Q5_K, {D41_ENG_G, D41_ENG_KV}));
  const uint32_t eg = m.add(weight(QT_IQ1_S, {D41_EMBD, D41_FF, D41_EXPERTS}, kLive * D41_FF));
  const uint32_t eu = m.add(weight(QT_IQ1_S, {D41_EMBD, D41_FF, D41_EXPERTS}, kLive * D41_FF));
  const uint32_t ed = m.add(weight(QT_IQ1_M, {D41_FF, D41_EMBD, D41_EXPERTS}, kLive * D41_EMBD));

  std::vector<float> bias(D41_EXPERTS, -1000.0f);
  std::uniform_real_distribution<float> small(-0.1f, 0.1f);
  for (uint32_t e = 0; e < kLive; e++) bias[e] = 1000.0f + small(rng);

  for (uint32_t il = 0; il < layers; il++) {
    const std::string b = "blk." + std::to_string(il) + ".";
    const int li = (int)il;
    const auto own = [&](const char *n, std::vector<uint64_t> shape, float mean, float sd_) {
      std::vector<float> v = randn(shape[0], sd_);
      for (float &x : v) x += mean;
      m.name(b + n, m.add(floats(std::move(shape), std::move(v))));
    };
    m.name(b + "attn_q_a.weight", q_a);
    m.name(b + "attn_q_b.weight", q_b);
    m.name(b + "attn_kv.weight", kv);
    m.name(b + "attn_output_a.weight", o_a);
    m.name(b + "attn_output_b.weight", o_b);
    m.name(b + "hc_attn_fn.weight", fn_a);
    m.name(b + "hc_ffn_fn.weight", fn_f);
    m.name(b + "ffn_gate_inp.weight", gi);
    m.name(b + "ffn_gate_shexp.weight", sg);
    m.name(b + "ffn_up_shexp.weight", su);
    m.name(b + "ffn_down_shexp.weight", sd);
    m.name(b + "ffn_gate_exps.weight", eg);
    m.name(b + "ffn_up_exps.weight", eu);
    m.name(b + "ffn_down_exps.weight", ed);
    m.name(b + "exp_probs_b.bias", m.add(floats({D41_EXPERTS}, bias)));
    own("attn_norm.weight", {D41_EMBD}, 1.0f, 0.2f);
    own("attn_q_a_norm.weight", {D41_QLR}, 1.0f, 0.2f);
    own("attn_kv_a_norm.weight", {D41_HD}, 1.0f, 0.2f);
    own("attn_sinks.weight", {D41_H}, 0.0f, 1.0f);
    own("ffn_norm.weight", {D41_EMBD}, 1.0f, 0.2f);
    own("hc_attn_scale.weight", {3}, 1.0f, 0.2f);
    own("hc_attn_base.weight", {D41_MIX}, 0.0f, 1.0f);
    own("hc_ffn_scale.weight", {3}, 1.0f, 0.2f);
    own("hc_ffn_base.weight", {D41_MIX}, 0.0f, 1.0f);
    if (D41_KV_SRC(li) == li) {
      m.name(b + "attn_compressor_kv.weight", c_kv);
      if (D41_RATIO(li) == 2) m.name(b + "attn_compressor_gate.weight", c_g);
      own("attn_compressor_norm.weight", {D41_HD}, 1.0f, 0.2f);
      m.name(b + "indexer.attn_k.weight", ik);
      own("indexer.k_norm.weight", {D41_IDX_D}, 1.0f, 0.2f);
    }
    if (D41_RATIO(li) != 0 && D41_IDX_SRC(li) == li) {
      m.name(b + "indexer.attn_q_b.weight", iq);
      m.name(b + "indexer.proj.weight", ip);
    }
    if (D41_ENGRAM(li) >= 0) {
      m.name(b + "engram_kv.weight", ek);
      m.name(b + "engram_gate.weight", m.add(floats({D41_EMBD, D41_HC}, randn((size_t)D41_EMBD * D41_HC, 1.0f))));
    }
  }
  return m;
}

// Writes m to path: blobs go in order, each expert tensor's holes left unwritten.
void write_model(Model &m, const std::string &path) {
  m.path = path;
  uint64_t at = 0;
  for (Weight &w : m.blobs) {
    w.offset = at;
    at += (w.file_bytes + 4095) / 4096 * 4096;
  }
  Writer h;
  h.put<uint32_t>(0x46554747u);
  h.put<uint32_t>(3);
  h.put<uint64_t>(m.names.size());
  h.put<uint64_t>(9);
  const auto u32_key = [&](const char *k, uint32_t v) {
    h.str(k);
    h.put<uint32_t>(4);
    h.put<uint32_t>(v);
  };
  const auto u64s_key = [&](const char *k, const std::vector<uint64_t> &v) {
    h.str(k);
    h.put<uint32_t>(9);
    h.put<uint32_t>(10);
    h.put<uint64_t>(v.size());
    for (uint64_t x : v) h.put<uint64_t>(x);
  };
  h.str("general.architecture");
  h.put<uint32_t>(8);
  h.str("deepseek41");
  u32_key("general.alignment", 4096);
  u32_key("deepseek41.block_count", m.layers);
  u64s_key("deepseek41.ple.layer_multipliers", m.mult);
  u64s_key("deepseek41.ple.head_offsets", m.off);
  u64s_key("deepseek41.ple.head_vocab_sizes", m.mod);
  h.str("deepseek41.engram.token_map");
  h.put<uint32_t>(9);
  h.put<uint32_t>(5);
  h.put<uint64_t>(m.token_map.size());
  for (int32_t x : m.token_map) h.put<int32_t>(x);
  u32_key("deepseek41.engram.pad_token_id", 2);
  u32_key("deepseek41.embedding_length", D41_EMBD);
  for (const auto &[name, blob] : m.names) {
    const Weight &w = m.blobs[blob];
    h.str(name);
    h.put<uint32_t>((uint32_t)w.shape.size());
    for (uint64_t d : w.shape) h.put<uint64_t>(d);
    h.put<uint32_t>(w.type);
    h.put<uint64_t>(w.offset);
  }
  const uint64_t start = (h.b.size() + 4095) / 4096 * 4096;
  const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
  EXPECT(fd >= 0, "cannot create %s", path.c_str());
  EXPECT(::pwrite(fd, h.b.data(), h.b.size(), 0) == (ssize_t)h.b.size(), "header");
  for (const Weight &w : m.blobs)
    EXPECT(::pwrite(fd, w.bytes.data(), w.bytes.size(), (off_t)(start + w.offset)) == (ssize_t)w.bytes.size(), "weights");
  EXPECT(::ftruncate(fd, (off_t)(start + at)) == 0, "size");
  ::close(fd);
}

// y = W x as ggml's mul_mat computes it, for rows [first, first + n) of a matrix of k values a row: x converted to the
// weights' dot type, one dot product a row. Not threaded, and separate from RefModel's product.
void product(const Weight &w, uint32_t k, uint32_t first, uint32_t n, const float *x, float *y, uint64_t byte0 = 0) {
  const auto *t = ggml_get_type_traits_cpu((ggml_type)w.type);
  const auto *q = ggml_get_type_traits_cpu(t->vec_dot_type);
  std::vector<uint8_t> xq(ggml_row_size(t->vec_dot_type, k));
  q->from_float(x, xq.data(), k);
  const size_t rb = ggml_row_size((ggml_type)w.type, k);
  for (uint32_t r = 0; r < n; r++) t->vec_dot((int)k, y + r, 0, w.bytes.data() + byte0 + (first + r) * rb, 0, xq.data(), 0, 1);
}

void product(const Weight &w, const float *x, float *y) { product(w, (uint32_t)w.shape[0], 0, (uint32_t)w.shape[1], x, y); }

// product above against ggml's own graph, on the types the model's dense tensors use.
void check_product_is_ggml(const Weight &w) {
  constexpr uint32_t rows = 8;
  const uint32_t k = (uint32_t)w.shape[0];
  const std::vector<float> x = randn(k, 1.0f);
  std::vector<float> y(rows);
  product(w, k, 0, rows, x.data(), y.data());
  ggml_init_params ip = {256u << 20, nullptr, false};
  ggml_context *ctx = ggml_init(ip);
  ggml_tensor *a = ggml_new_tensor_2d(ctx, (ggml_type)w.type, k, rows);
  std::memcpy(a->data, w.bytes.data(), ggml_nbytes(a));
  ggml_tensor *b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, k);
  std::copy(x.begin(), x.end(), (float *)b->data);
  ggml_tensor *r = ggml_mul_mat(ctx, a, b);
  ggml_cgraph *gf = ggml_new_graph(ctx);
  ggml_build_forward_expand(gf, r);
  EXPECT(ggml_graph_compute_with_ctx(ctx, gf, 1) == GGML_STATUS_SUCCESS, "ggml's graph failed");
  for (uint32_t i = 0; i < rows; i++)
    EXPECT(y[i] == ((const float *)r->data)[i], "type %u row %u: %g, ggml %g", w.type, i, y[i], ((const float *)r->data)[i]);
  ggml_free(ctx);
}

double rel_rms(const float *a, const float *b, size_t n) {
  double d = 0, s = 0;
  for (size_t i = 0; i < n; i++) {
    if (std::isinf(a[i]) && a[i] == b[i]) continue;
    d += ((double)a[i] - b[i]) * ((double)a[i] - b[i]);
    s += (double)b[i] * b[i];
  }
  return s > 0 ? std::sqrt(d / s) : std::sqrt(d);
}

using Values = std::map<std::pair<std::string, int>, std::vector<float>>;

float to_f16(float x) { return ggml_fp16_to_fp32(ggml_fp32_to_fp16(x)); }

// The forward pass from direct calls, its state its own, comparing each step with what RefModel gave.
class Walk {
 public:
  Walk(const Model &m) : m_(m), hash_(params(m)), rows_(m.layers), topk_(m.layers), routes_(m.layers) {
    window_.resize((size_t)m.layers * D41_WIN * D41_HD);
    sk_.resize((size_t)m.layers * 2 * D41_HD);
    ss_.resize((size_t)m.layers * 2 * D41_HD);
  }

  void decode(int32_t t, const Values &got, std::vector<float> *logits) {
    got_ = &got;
    std::vector<float> X(D41_HC_DIM), X1(D41_HC_DIM), x(D41_EMBD), attn(D41_EMBD);
    row_to_f32(m_.at("token_embd.weight"), (uint64_t)t, X.data());
    for (uint32_t c = 1; c < D41_HC; c++) std::copy_n(X.data(), D41_EMBD, X.data() + (size_t)c * D41_EMBD);
    same("hc_init", -1, X);
    float pre[D41_HC] = {1, 0, 0, 0};
    for (uint32_t il = 0; il < m_.layers; il++) {
      const int e = D41_ENGRAM((int)il);
      if (e >= 0) engram(il, (uint32_t)e, t, X);
      const ref::HcMix am = ref::hc_mixes(X.data(), m_.f(il, "hc_attn_fn.weight"), m_.f(il, "hc_attn_scale.weight"),
                                          m_.f(il, "hc_attn_base.weight"));
      ref::hc_collapse_norm(X.data(), pre, m_.f(il, "attn_norm.weight"), x.data());
      same("attn_norm", (int)il, x);
      attention(il, x.data(), attn);
      ref::hc_post(X.data(), attn.data(), am, X1.data());
      same("hc_attn_post", (int)il, X1);
      const ref::HcMix fm = ref::hc_mixes(X1.data(), m_.f(il, "hc_ffn_fn.weight"), m_.f(il, "hc_ffn_scale.weight"),
                                          m_.f(il, "hc_ffn_base.weight"));
      ref::hc_collapse_norm(X1.data(), am.pre, m_.f(il, "ffn_norm.weight"), x.data());
      same("ffn_norm", (int)il, x);
      std::copy_n(fm.pre, D41_HC, pre);
      std::vector<float> ffn = moe(il, x.data());
      ref::hc_post(X1.data(), ffn.data(), fm, X.data());
      same("l_out", (int)il, X);
    }
    ref::hc_collapse_norm(X.data(), pre, m_.at("output_norm.weight").bytes.empty() ? nullptr
                          : reinterpret_cast<const float *>(m_.at("output_norm.weight").bytes.data()), x.data());
    same("result_norm", -1, x);
    logits->resize(kVocab);
    product(m_.at("output.weight"), x.data(), logits->data());
    same("result_output", -1, *logits);
    p3_ = p2_, p2_ = p1_, p1_ = t;
    pos_++;
  }

  const ref::Route &route(uint32_t il) const { return routes_[il]; }
  const std::vector<uint32_t> &top_k(uint32_t il) const { return topk_[il]; }
  double worst() const { return worst_; }
  const std::map<std::string, int> &seen() const { return seen_; }

 private:
  static EngramParams params(const Model &m) {
    EngramParams p;
    p.multipliers = m.mult;
    p.offsets = m.off;
    p.moduli = m.mod;
    p.token_map = m.token_map;
    p.pad = 2;
    return p;
  }

  void same(const char *name, int il, const std::vector<float> &v) { same(name, il, v.data(), v.size()); }
  void same(const char *name, int il, const float *v, size_t n) {
    const auto it = got_->find({name, il});
    EXPECT(it != got_->end(), "pos %u: RefModel gave no %s at layer %d", pos_, name, il);
    EXPECT(it->second.size() == n, "pos %u: %s at layer %d: %zu values, not %zu", pos_, name, il, it->second.size(), n);
    const double e = rel_rms(it->second.data(), v, n);
    EXPECT(e < 2e-5, "pos %u: %s at layer %d differs by %g", pos_, name, il, e);
    worst_ = std::max(worst_, e);
    seen_[name]++;
  }

  void row_to_f32(const Weight &w, uint64_t row, float *out) const {
    const size_t rb = ggml_row_size((ggml_type)w.type, (int64_t)w.shape[0]);
    ggml_get_type_traits((ggml_type)w.type)->to_float(w.bytes.data() + row * rb, out, (int64_t)w.shape[0]);
  }

  void engram(uint32_t il, uint32_t e, int32_t t, std::vector<float> &X) {
    const auto ids = hash_.rows(t, p1_, p2_, p3_, e);
    std::vector<float> g(D41_ENG_G), kv(D41_ENG_KV);
    for (uint32_t c = 0; c < D41_ENG_COLS; c++) row_to_f32(m_.at("per_layer_token_embd.weight"), ids[c], g.data() + (size_t)c * D41_ENG_ROW);
    same("engram_rows", (int)il, g);
    product(m_.at(il, "engram_kv.weight"), g.data(), kv.data());
    same("engram_kv", (int)il, kv);
    float gates[D41_HC];
    ref::engram_gate(X.data(), kv.data(), m_.f(il, "engram_gate.weight"), gates);
    same("engram_gate", (int)il, gates, D41_HC);
    same("engram_out", (int)il, X);
  }

  void attention(uint32_t il, const float *x, std::vector<float> &attn) {
    const int li = (int)il;
    const uint32_t ratio = (uint32_t)D41_RATIO(li), p = pos_;
    const bool yarn = ratio != 0;
    std::vector<float> qr(D41_QLR), q((size_t)D41_H * D41_HD), kv(D41_HD);
    product(m_.at(il, "attn_q_a.weight"), x, qr.data());
    ref::rms_norm(qr.data(), m_.f(il, "attn_q_a_norm.weight"), D41_QLR, qr.data());
    same("qr_norm", li, qr);
    product(m_.at(il, "attn_q_b.weight"), qr.data(), q.data());
    for (uint32_t h = 0; h < D41_H; h++) ref::rope_tail(&q[(size_t)h * D41_HD], D41_HD, p, yarn, false);
    same("q", li, q);
    product(m_.at(il, "attn_kv.weight"), x, kv.data());
    ref::rms_norm(kv.data(), m_.f(il, "attn_kv_a_norm.weight"), D41_HD, kv.data());
    ref::rope_tail(kv.data(), D41_HD, p, yarn, false);
    same("kv", li, kv);
    for (uint32_t d = 0; d < D41_HD; d++) window_[((size_t)il * D41_WIN + p % D41_WIN) * D41_HD + d] = to_f16(kv[d]);

    std::vector<uint32_t> &tk = topk_[il];
    tk.clear();
    const std::vector<float> *comp = nullptr;
    if (ratio) {
      const int src = D41_KV_SRC(li);
      comp = &rows_[(size_t)src];
      if (src == li) compressor(il, ratio, x);
      const uint32_t n = (p + 1) / ratio, isrc = (uint32_t)D41_IDX_SRC(li);
      if (isrc == il && n > 0) {
        std::vector<float> iq((size_t)D41_IDX_H * D41_IDX_D), w(D41_IDX_H), score(n);
        product(m_.at(il, "indexer.attn_q_b.weight"), qr.data(), iq.data());
        for (uint32_t h = 0; h < D41_IDX_H; h++) ref::rope_tail(&iq[(size_t)h * D41_IDX_D], D41_IDX_D, p, yarn, false);
        product(m_.at(il, "indexer.proj.weight"), x, w.data());
        ref::index_score(iq.data(), w.data(), comp->data() + D41_HD, D41_COMP_ROW, n, score.data());
        same("idx_score", li, score);
        if (il == 20) keep_ = ref::candidates(score.data(), n);
        if (il > 20)
          for (uint32_t s = 0; s < n; s++)
            if (!keep_[s]) score[s] = -std::numeric_limits<float>::infinity();
        tk = ref::top_k(score.data(), n, D41_TOPK);
      } else if (isrc != il) {
        tk = topk_[isrc];
      }
    }

    // Keys in another order from RefModel's: the sum over them does not depend on it past a double's rounding.
    const uint32_t nwin = std::min(p + 1, (uint32_t)D41_WIN), nkeys = nwin + (uint32_t)tk.size();
    std::vector<float> keys((size_t)nkeys * D41_HD), o((size_t)D41_H * D41_HD), q16(q.size());
    std::vector<int32_t> idx(nkeys);
    for (uint32_t i = 0; i < nkeys; i++) {
      idx[i] = (int32_t)(nkeys - 1 - i);
      const float *from = i < nwin ? &window_[((size_t)il * D41_WIN + (p - i) % D41_WIN) * D41_HD]
                                   : comp->data() + (size_t)tk[i - nwin] * D41_COMP_ROW;
      std::copy_n(from, D41_HD, keys.data() + (size_t)i * D41_HD);
    }
    std::transform(q.begin(), q.end(), q16.begin(), to_f16);
    ref::sparse_attn(q16.data(), keys.data(), D41_HD, idx.data(), nkeys, m_.f(il, "attn_sinks.weight"), o.data());
    for (uint32_t h = 0; h < D41_H; h++) ref::rope_tail(&o[(size_t)h * D41_HD], D41_HD, p, yarn, true);
    same("attn_derope", li, o);
    std::vector<float> a((size_t)D41_OG * D41_OLR);
    for (uint32_t g = 0; g < D41_OG; g++)
      product(m_.at(il, "attn_output_a.weight"), D41_OG_K, g * D41_OLR, D41_OLR, &o[(size_t)g * D41_OG_K], &a[(size_t)g * D41_OLR]);
    same("attn_wo_a", li, a);
    product(m_.at(il, "attn_output_b.weight"), a.data(), attn.data());
    same("attn_out", li, attn);
  }

  void compressor(uint32_t il, uint32_t ratio, const float *x) {
    const uint32_t p = pos_;
    std::vector<float> latent(D41_HD);
    float *kv0 = &sk_[(size_t)il * 2 * D41_HD], *sc0 = &ss_[(size_t)il * 2 * D41_HD];
    if (ratio == 1) {
      product(m_.at(il, "attn_compressor_kv.weight"), x, kv0);
      ref::compress(kv0, nullptr, 1, m_.f(il, "attn_compressor_norm.weight"), latent.data());
    } else {
      product(m_.at(il, "attn_compressor_kv.weight"), x, kv0 + (p % 2) * D41_HD);
      product(m_.at(il, "attn_compressor_gate.weight"), x, sc0 + (p % 2) * D41_HD);
      if (p % 2 == 0) return;
      ref::compress(kv0, sc0, 2, m_.f(il, "attn_compressor_norm.weight"), latent.data());
    }
    same("comp_latent", (int)il, latent);
    // BF16 weights as f32: compressed_row rounds the latent itself, as ggml's product does.
    std::vector<float> attn_k((size_t)D41_IDX_D * D41_HD);
    ggml_bf16_to_fp32_row(reinterpret_cast<const ggml_bf16_t *>(m_.at(il, "indexer.attn_k.weight").bytes.data()), attn_k.data(), (int64_t)attn_k.size());
    std::vector<float> row(D41_COMP_ROW);
    ref::compressed_row(latent.data(), attn_k.data(), m_.f(il, "indexer.k_norm.weight"), p / ratio * ratio, row.data());
    std::transform(row.begin(), row.end(), row.begin(), to_f16);
    rows_[il].insert(rows_[il].end(), row.begin(), row.end());
  }

  std::vector<float> moe(uint32_t il, const float *x) {
    std::vector<float> logits(D41_EXPERTS), y(D41_EMBD, 0.0f), sh(D41_EMBD, 0.0f), out(D41_EMBD);
    product(m_.at(il, "ffn_gate_inp.weight"), x, logits.data());
    same("ffn_moe_logits", (int)il, logits);
    routes_[il] = ref::router(logits.data(), m_.f(il, "exp_probs_b.bias"));
    const Weight &eg = m_.at(il, "ffn_gate_exps.weight"), &eu = m_.at(il, "ffn_up_exps.weight"), &ed = m_.at(il, "ffn_down_exps.weight");
    const size_t gb = ggml_row_size((ggml_type)eg.type, D41_EMBD) * D41_FF, db = ggml_row_size((ggml_type)ed.type, D41_FF) * D41_EMBD;
    for (uint32_t k = 0; k < D41_USED; k++) {
      const int32_t e = routes_[il].id[k];
      EXPECT(e >= 0 && (uint32_t)e < kLive, "layer %u routes to expert %d, a hole in the file", il, e);
      swiglu_down(eg, eu, ed, (size_t)e * gb, (size_t)e * db, routes_[il].w[k], x, y.data());
    }
    same("ffn_moe_out", (int)il, y);
    swiglu_down(m_.at(il, "ffn_gate_shexp.weight"), m_.at(il, "ffn_up_shexp.weight"), m_.at(il, "ffn_down_shexp.weight"), 0, 0, 1.0f, x, sh.data());
    same("ffn_shexp", (int)il, sh);
    for (uint32_t d = 0; d < D41_EMBD; d++) out[d] = y[d] + sh[d];
    same("ffn_out", (int)il, out);
    return out;
  }

  // y += w down(swiglu(gate x, up x)), the gate and up at byte g of their tensors, the down at byte d.
  void swiglu_down(const Weight &g, const Weight &u, const Weight &dn, size_t gb, size_t db, float w, const float *x, float *y) {
    std::vector<float> ga(D41_FF), up(D41_FF), h(D41_FF), o(D41_EMBD);
    product(g, D41_EMBD, 0, D41_FF, x, ga.data(), gb);
    product(u, D41_EMBD, 0, D41_FF, x, up.data(), gb);
    for (uint32_t i = 0; i < D41_FF; i++) h[i] = ref::swiglu(ga[i], up[i]);
    product(dn, D41_FF, 0, D41_EMBD, h.data(), o.data(), db);
    for (uint32_t d = 0; d < D41_EMBD; d++) y[d] += w * o[d];
  }

  const Model &m_;
  EngramHash hash_;
  const Values *got_ = nullptr;
  uint32_t pos_ = 0;
  int32_t p1_ = -1, p2_ = -1, p3_ = -1;
  std::vector<float> window_, sk_, ss_;
  std::vector<std::vector<float>> rows_;
  std::vector<std::vector<uint32_t>> topk_;
  std::vector<uint8_t> keep_;
  std::vector<ref::Route> routes_;
  double worst_ = 0;
  std::map<std::string, int> seen_;
};

RefProbe capture(Values &into) {
  return [&into](const char *name, int layer, const float *v, size_t n) {
    into[{name, layer}].assign(v, v + n);
  };
}

// Every step of T tokens equal to the walk's, the same routes and top-k, reads only of the routed experts, and the
// same logits again after a reset.
void compare_with_walk(const Model &m, uint32_t T, const std::vector<int32_t> &tokens) {
  RefModel model(m.path, RefOptions{8, 1024});
  EXPECT(model.layers() == m.layers && model.vocab() == kVocab, "the model's sizes");
  Walk walk(m);
  std::vector<std::vector<float>> first;
  for (uint32_t t = 0; t < T; t++) {
    Values got;
    std::vector<float> logits(kVocab), direct;
    model.decode(tokens[t], logits.data(), capture(got));
    walk.decode(tokens[t], got, &direct);
    for (uint32_t il = 0; il < m.layers; il++) {
      const ref::Route &a = model.route(il), &b = walk.route(il);
      for (uint32_t k = 0; k < D41_USED; k++) {
        EXPECT(a.id[k] == b.id[k] && a.w[k] == b.w[k], "pos %u layer %u route %u", t, il, k);
      }
      EXPECT(model.top_k(il) == walk.top_k(il), "pos %u layer %u top-k", t, il);
    }
    first.push_back(std::move(logits));
  }
  EXPECT(model.position() == T, "position");
  std::printf("%u layers, %u tokens: worst relative RMS %.2e over the steps:", m.layers, T, walk.worst());
  for (const auto &[name, n] : walk.seen()) std::printf(" %s %d", name.c_str(), n);
  std::printf("\n");

  const auto &eg = m.at(0, "ffn_gate_exps.weight"), &ed = m.at(0, "ffn_down_exps.weight");
  const uint64_t expert = 2 * eg.file_bytes / D41_EXPERTS + ed.file_bytes / D41_EXPERTS;
  EXPECT(model.stats().expert_bytes == (uint64_t)T * m.layers * D41_USED * expert, "expert bytes read: %llu",
         (unsigned long long)model.stats().expert_bytes);
  const uint64_t engram_layers = (m.layers > 1) + (m.layers > 14);
  EXPECT(model.stats().engram_rows == T * engram_layers * D41_ENG_COLS, "Engram rows read");
  EXPECT(model.stats().dense_bytes < (1ull << 30), "the shared blobs are held once");

  // Layer kinds: compressed rows appear at ratio 2 from position 1 and at ratio 1 from position 0.
  if (m.layers > 20 && T > 2) {
    EXPECT(model.top_k(1).empty(), "a window-only layer has no top-k");
    EXPECT(model.top_k(3) == model.top_k(2) && model.top_k(21) == model.top_k(20), "consumers reuse the source's top-k");
    EXPECT(model.top_k(2).size() == T / 2 && model.top_k(20).size() == T, "the top-k holds every visible row");
  }

  model.reset();
  EXPECT(model.position() == 0, "reset");
  for (uint32_t t = 0; t < std::min(T, 3u); t++) {
    std::vector<float> logits(kVocab);
    model.decode(tokens[t], logits.data());
    EXPECT(logits == first[t], "pos %u: another run's logits differ after a reset", t);
  }
  bool threw = false;
  try {
    model.decode((int32_t)kVocab, nullptr);
  } catch (const std::runtime_error &) {
    threw = true;
  }
  EXPECT(threw, "a token outside the vocabulary");
}

// Each record: u32 magic, u32 position, i32 token, i32 layer, u32 name bytes, u32 values; the name, then f32 values. A
// tensor's name is upstream's, layer -1 outside a layer. Returns the worst relative RMS over the steps, printing each.
double compare_dump(const std::string &dump, const std::string &model_path) {
  FILE *f = std::fopen(dump.c_str(), "rb");
  EXPECT(f, "cannot read %s", dump.c_str());
  RefModel model(model_path);
  std::map<std::string, double> worst;
  uint32_t pos = 0xffffffffu;
  Values got;
  uint32_t head[6];
  while (std::fread(head, sizeof head, 1, f) == 1) {
    EXPECT(head[0] == 0x44343144u, "%s: not a record", dump.c_str());
    if (head[1] != pos) {
      if (head[1] == 0) model.reset();
      EXPECT(head[1] == model.position(), "dump at position %u, the model at %u", head[1], model.position());
      got.clear();
      model.decode((int32_t)head[2], nullptr, capture(got));
      pos = head[1];
    }
    std::string name(head[4], '\0');
    std::vector<float> v(head[5]);
    EXPECT(std::fread(name.data(), 1, head[4], f) == head[4], "%s: short", dump.c_str());
    EXPECT(std::fread(v.data(), 4, head[5], f) == head[5], "%s: short", dump.c_str());
    const auto it = got.find({name, (int32_t)head[3]});
    if (it == got.end()) continue;
    // Upstream's index score covers its whole cache, the rows past the visible ones at -inf.
    const size_t n = std::min(v.size(), it->second.size());
    double &w = worst[name];
    w = std::max(w, rel_rms(it->second.data(), v.data(), n));
  }
  std::fclose(f);
  double all = 0;
  for (const auto &[name, w] : worst) {
    std::printf("%-20s worst relative RMS %.2e\n", name.c_str(), w);
    all = std::max(all, w);
  }
  return all;
}

}  // namespace

int main(int argc, char **argv) {
#define NEEDS(f) \
  if (!__builtin_cpu_supports(f)) return test::skip("the CPU lacks " f ", which the ggml build uses")
  NEEDS("avx2");
  NEEDS("fma");
  NEEDS("f16c");
  NEEDS("bmi2");
  NEEDS("avx512f");
  NEEDS("avx512vbmi");
  NEEDS("avx512vnni");
  NEEDS("avx512bf16");
#undef NEEDS
  ggml_cpu_init();  // product below reads its f16 table
  if (argc == 5 && std::string(argv[1]) == "--dump" && std::string(argv[3]) == "--model") {
    const double w = compare_dump(argv[2], argv[4]);
    return w <= 1e-3 ? 0 : 1;
  }

  namespace fs = std::filesystem;
  const fs::path dir = fs::current_path() / "ds41_ref_model_test.tmp";
  fs::remove_all(dir);
  fs::create_directories(dir);

  Model m = make_model(26);
  for (const char *n : {"attn_q_b.weight", "attn_kv.weight", "ffn_gate_inp.weight"}) check_product_is_ggml(m.at(0, n));

  std::uniform_int_distribution<int32_t> tok(0, kVocab - 1);
  std::vector<int32_t> tokens(160);
  for (int32_t &t : tokens) t = tok(rng);
  tokens[3] = 2;  // the pad token itself

  write_model(m, (dir / "tiny26.gguf").string());
  compare_with_walk(m, 10, tokens);

  // Past the window, on the first layers alone, sharing the bytes above.
  m.layers = 3;
  for (auto it = m.names.begin(); it != m.names.end();) {
    int il = -1;
    if (std::sscanf(it->first.c_str(), "blk.%d.", &il) == 1 && il >= 3) {
      it = m.names.erase(it);
    } else {
      ++it;
    }
  }
  write_model(m, (dir / "tiny3.gguf").string());
  compare_with_walk(m, D41_WIN + 7, tokens);

  // The dump comparison, on a dump of RefModel itself.
  const std::string dump = (dir / "dump.bin").string();
  {
    FILE *f = std::fopen(dump.c_str(), "wb");
    EXPECT(f, "cannot write %s", dump.c_str());
    RefModel model(m.path, RefOptions{8, 1024});
    for (uint32_t t = 0; t < 3; t++) {
      model.decode(tokens[t], nullptr, [&](const char *name, int layer, const float *v, size_t n) {
        const uint32_t head[6] = {0x44343144u, t, (uint32_t)tokens[t], (uint32_t)layer, (uint32_t)std::strlen(name), (uint32_t)n};
        std::fwrite(head, sizeof head, 1, f);
        std::fwrite(name, 1, head[4], f);
        std::fwrite(v, 4, n, f);
      });
    }
    std::fclose(f);
  }
  EXPECT(compare_dump(dump, m.path) == 0.0, "RefModel against its own dump");

  fs::remove_all(dir);
  std::printf("ds41_ref_model_test: ok\n");
  return 0;
}
