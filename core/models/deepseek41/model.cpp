#include "models/deepseek41/model.h"

#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "ggml.h"
#include "moe_tile.h"
#include "quant.h"

namespace engine::deepseek41 {

namespace {

constexpr uint32_t E = D41_EMBD, D = D41_HC_DIM;
constexpr uint32_t kHeads = D41_H / 2;                           // a card's query heads
constexpr uint32_t kGroups = D41_OG / 2;                         // and its output groups
constexpr uint32_t kEngA8Row = D41_ENG_G + D41_ENG_G / 8;        // a token's gathered Engram rows as a8
constexpr uint32_t kEngKvRows = 2 * E + E / 2;                   // a card's rows of engram_kv: two keys, half the value
constexpr uint32_t kTopkLocal = 1024;                            // ds_idx_topk's and topk_rows' work-group
constexpr std::chrono::seconds kPublishLimit{10};
const void *const none = nullptr;

std::string blk(uint32_t l, const char *name) { return "blk." + std::to_string(l) + "." + name; }

std::string mv_module(uint32_t type, uint32_t lanes) {
  std::string s = std::string("matvec_") + ggml_type_label(type) + "_l" + std::to_string(lanes);
  for (char &ch : s) ch = (char)std::tolower((unsigned char)ch);
  return s;
}

std::string moe_module(const char *part, uint32_t type, uint32_t group) {
  std::string s = std::string("deepseek41_moe_") + part + "_" + ggml_type_label(type) + "_g" + std::to_string(group);
  for (char &ch : s) ch = (char)std::tolower((unsigned char)ch);
  return s;
}

uint64_t align256(uint64_t v) { return (v + 255) / 256 * 256; }

// The slot class of rows rows of gate and up, and the same columns of down, with its slots from base on.
D41Class slot_class(uint32_t gate_type, uint32_t down_type, uint32_t rows, uint64_t base) {
  const uint64_t g = align256(runs_bytes(gate_type, (uint64_t)rows * (E / 256)));
  const uint64_t d = align256(runs_bytes(down_type, (uint64_t)E * (rows / 256)));
  return {base, 2 * g + d, g, 2 * g, rows, {}};
}

// The recorded width that holds n rows: the routed experts' kernels are built for these.
uint32_t width(uint32_t n) { return n <= 4 ? n : n <= 6 ? 6 : 8; }

// A row of K values as a8 (kernels/include/a8.h).
void a8_row(const float *x, uint32_t K, uint8_t *a) {
  for (uint32_t u = 0; u < K / 32; u++) {
    float amax = 0;
    for (uint32_t i = 0; i < 32; i++) amax = std::fmax(amax, std::fabs(x[32 * u + i]));
    const float d = amax / 127.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
    for (uint32_t i = 0; i < 32; i++) a[32 * u + i] = (uint8_t)(int8_t)std::round(x[32 * u + i] * id);
    std::memcpy(a + K + 4 * u, &d, 4);
  }
}

// A routing profile (tools/route_profile.cpp): lines "layer expert count", '#' starting a comment line.
std::vector<uint64_t> read_profile(const std::string &path) {
  std::vector<uint64_t> v((size_t)D41_LAYERS * D41_EXPERTS, 0);
  FILE *f = std::fopen(path.c_str(), "r");
  if (!f) throw std::runtime_error("cannot read the routing profile " + path);
  char line[256];
  while (std::fgets(line, sizeof line, f)) {
    unsigned l, e;
    unsigned long long c;
    if (line[0] != '#' && std::sscanf(line, "%u %u %llu", &l, &e, &c) == 3 && l < D41_LAYERS && e < D41_EXPERTS)
      v[(size_t)l * D41_EXPERTS + e] = c;
  }
  std::fclose(f);
  return v;
}

}  // namespace

template <class T>
T *Model::dev(Card &card, size_t n) {
  return static_cast<T *>(rt_->alloc_device(card.c, n * sizeof(T), 4096));
}

template <class T>
T *Model::upload(Card &card, const T *src, size_t n) {
  T *p = dev<T>(card, n);
  card.up->copy(p, src, n * sizeof(T));
  card.up->synchronize();
  return p;
}

const GgufTensor &Model::tensor(const std::string &name, std::initializer_list<uint64_t> shape) const {
  const GgufTensor &t = gguf_.tensor(name);
  if (!std::equal(t.shape.begin(), t.shape.end(), shape.begin(), shape.end()))
    throw std::runtime_error(name + ": not the shape deepseek41's layout gives");
  return t;
}

// Read past the page cache, as every dense tensor is (ADR-0074 item 3), and kept until its layer is on both cards.
const uint8_t *Model::held(const GgufTensor &t) {
  auto [it, fresh] = held_.try_emplace({t.shard, t.offset});
  if (fresh) {
    it->second = aligned_buffer(DirectReader::span(t.offset, t.bytes));
    ssd_->direct().read(t.shard, t.offset, t.bytes, it->second.get());
  }
  return it->second.get() + t.offset % DirectReader::kAlign;
}

float *Model::upload_floats(Card &card, const std::string &name, std::initializer_list<uint64_t> shape, size_t first,
                            size_t n) {
  const GgufTensor &t = tensor(name, shape);
  std::vector<float> v(t.elements());
  if (t.type == QT_F32) {
    std::memcpy(v.data(), held(t), t.bytes);
  } else {
    ggml_get_type_traits((ggml_type)t.type)->to_float(held(t), v.data(), (int64_t)v.size());
  }
  return upload<float>(card, v.data() + first, n ? n : v.size());
}

uint16_t *Model::upload_bf16(Card &card, const std::string &name, std::initializer_list<uint64_t> shape) {
  const GgufTensor &t = tensor(name, shape);
  if (t.type != QT_BF16) throw std::runtime_error(name + ": the engine reads it as BF16 only");
  return upload<uint16_t>(card, reinterpret_cast<const uint16_t *>(held(t)), t.elements());
}

Model::Mat Model::mat(Card &card, uint32_t type, uint32_t K, const std::vector<SliceSpec> &specs) {
  Mat m;
  m.type = type;
  m.K = K;
  m.bpr = K / quant_block_values(type);
  for (const SliceSpec &s : specs) m.rows += (uint32_t)s.rows;
  std::vector<uint8_t> buf(runs_bytes(type, (uint64_t)m.rows * m.bpr));
  to_runs(type, specs, buf.data());
  m.p = upload<uint8_t>(card, buf.data(), buf.size());
  return m;
}

Model::Model(const std::string &path, const Options &opt, const Debug &debug) : opt_(opt), debug_(debug), gguf_(path) {
  const auto t0 = std::chrono::steady_clock::now();
  if (gguf_.get_string("general.architecture") != "deepseek41") throw std::runtime_error("not a deepseek41 model");
  if (gguf_.get_int("deepseek41.block_count") != D41_LAYERS)
    throw std::runtime_error(path + ": deepseek41 with other than 40 layers");
  if (opt_.slots == 0) throw std::runtime_error("deepseek41 needs a slot");
  if (opt_.max_ctx == 0 || opt_.max_ctx > D41_CTX_CAP || opt_.max_ctx % KV_PAGE_CELLS)
    throw std::runtime_error("deepseek41 reads at most 16,384 positions a sequence, a multiple of 4");
  if (!opt_.mtp_dir.empty()) throw std::runtime_error("deepseek41 has no drafter");
  if (!opt_.vision_path.empty()) throw std::runtime_error("deepseek41 reads no images");
  if (opt_.max_cards < 2) throw std::runtime_error("deepseek41 runs on two cards");
  rt_ = std::make_unique<ze::Runtime>(2);
  lib_ = std::make_unique<ze::Library>(*rt_);
  pool_ = std::make_unique<Pool>(opt_.cpu_threads > 1 ? opt_.cpu_threads - 1 : 0);
  npp_ = opt_.max_ctx / KV_PAGE_CELLS;
  kv_pages_ = opt_.kv_pages ? opt_.kv_pages : opt_.slots * npp_;

  const GgufTensor &embd = gguf_.tensor("token_embd.weight");
  if (embd.shape.size() != 2 || embd.shape[0] != E) throw std::runtime_error("token_embd.weight: its shape");
  vocab_ = (uint32_t)embd.shape[1];

  uint64_t largest = 0;
  for (uint32_t l = 0; l < D41_LAYERS; l++) {
    const GgufTensor &g = tensor(blk(l, "ffn_gate_exps.weight"), {E, D41_FF, D41_EXPERTS});
    const GgufTensor &u = tensor(blk(l, "ffn_up_exps.weight"), {E, D41_FF, D41_EXPERTS});
    const GgufTensor &d = tensor(blk(l, "ffn_down_exps.weight"), {D41_FF, E, D41_EXPERTS});
    if (g.type != u.type) throw std::runtime_error(blk(l, "ffn_up_exps.weight") + ": not gate's type");
    LayerExperts le;
    le.shape.gate_type = g.type;
    le.shape.down_type = d.type;
    le.shape.n_embd = E;
    le.shape.n_ff = D41_FF;
    le.shape.swiglu_limit = D41_SWIGLU_LIMIT;
    le.gate = {g.shard, g.offset};
    le.up = {u.shard, u.offset};
    le.down = {d.shard, d.offset};
    le.gate_bytes = g.bytes / D41_EXPERTS;
    le.down_bytes = d.bytes / D41_EXPERTS;
    largest = std::max(largest, le.expert_bytes());
    experts_.push_back(le);
  }
  ssd_ = std::make_unique<SsdReader>(gguf_.files(), staging_ring_bytes(largest, STEP_MAX_ROWS));
  arena_ = std::make_unique<ExpertArena>(opt_.arena_bytes, kArenaPieceBytes, opt_.hugetlb_arena);
  miss_ = std::make_unique<MissJobs>(experts_, *arena_, *ssd_, *pool_);
  hash_ = std::make_unique<EngramHash>(EngramParams::from_gguf(gguf_));
  engram_ = std::make_unique<EngramReader>(gguf_);
  eng_f_.resize((size_t)STEP_MAX_ROWS * D41_ENG_G);

  const auto host = [&](size_t bytes) {
    void *p = rt_->alloc_host(bytes);
    std::memset(p, 0, bytes);
    return p;
  };
  plan_ = static_cast<StepWords *>(host(sizeof(StepWords)));
  embed_ = static_cast<float *>(host((size_t)STEP_MAX_ROWS * E * 4));
  for (auto &a : eng_a8_) a = static_cast<uint8_t *>(host((size_t)STEP_MAX_ROWS * kEngA8Row));
  jobs_ = static_cast<D41MissJob *>(host((size_t)D41_LAYERS * STEP_MAX_ROWS * sizeof(D41MissJob)));
  cpu_y_ = static_cast<float *>(host((size_t)D41_LAYERS * STEP_MAX_ROWS * E * 4));
  ans_flag_ = static_cast<uint32_t *>(host((size_t)D41_LAYERS * 64));
  flags_ = static_cast<uint32_t *>(host((size_t)2 * kFlags * 64));
  stuck_ = static_cast<uint32_t *>(host(2 * 64));
  top_idx_ = static_cast<uint32_t *>(host((size_t)2 * STEP_MAX_ROWS * opt_.candidates * 4));
  top_val_ = static_cast<float *>(host((size_t)2 * STEP_MAX_ROWS * opt_.candidates * 4));
  top_n_ = static_cast<uint32_t *>(host(2 * STEP_MAX_ROWS * 4));
  if (debug_.probes) probe_ = static_cast<float *>(host((size_t)2 * D41_LAYERS * STEP_MAX_ROWS * D * 4));
  pt_.assign((size_t)opt_.slots * npp_, D41_NO_PAGE);

  for (uint32_t c = 0; c < 2; c++) {
    cards_[c] = std::make_unique<Card>();
    Card &card = *cards_[c];
    card.c = c;
    card.up = std::make_unique<ze::List>(*rt_, c, ze::List::immediate);
    card.grids = upload<uint8_t>(card, codebooks().data(), QGRID_BYTES);
    card.layers.resize(D41_LAYERS);
    alloc_buffers(card);
  }
  for (uint32_t l = 0; l < D41_LAYERS; l++) load_layer(l);
  {
    const GgufTensor &out = tensor("output.weight", {E, vocab_});
    for (auto &cp : cards_) {
      Card &card = *cp;
      card.vocab0 = card.c * (vocab_ / 2);
      card.vocab_n = card.c ? vocab_ - vocab_ / 2 : vocab_ / 2;
      card.head = mat(card, out.type, E, {{held(out), out.row_bytes(), card.vocab0, card.vocab_n, 0, E / quant_block_values(out.type)}});
      card.out_norm = upload_floats(card, "output_norm.weight", {E});
      card.logits = dev<float>(card, (size_t)STEP_MAX_ROWS * card.vocab_n);
    }
    held_.clear();
  }
  load_experts();

  info_.cards = 2;
  info_.vocab = vocab_;
  for (uint32_t c = 0; c < 2; c++) info_.device_bytes[c] = rt_->device_bytes(c);
  info_.arena_anon = true;
  info_.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

Model::~Model() = default;

void Model::alloc_buffers(Card &card) {
  const size_t R = STEP_MAX_ROWS;
  card.plan = dev<StepWords>(card, 1);
  card.pt = upload<uint32_t>(card, pt_.data(), pt_.size());
  card.e = dev<float>(card, R * E);
  for (auto &a : card.eng_a8) a = dev<uint8_t>(card, R * kEngA8Row);
  card.X = dev<float>(card, R * D);
  card.X1 = dev<float>(card, R * D);
  card.ss = dev<float>(card, 64);
  card.m = dev<float>(card, R * D41_MIX);
  card.amix = dev<D41HcMix>(card, R);
  card.fmix = dev<D41HcMix>(card, R);
  card.xa = dev<uint8_t>(card, R * D41_A8_ROW);
  card.xb = dev<uint16_t>(card, R * E);
  card.qr = dev<float>(card, R * D41_QLR);
  card.kv = dev<float>(card, R * D41_HD);
  card.qa = dev<uint8_t>(card, R * D41_QR_A8_ROW);
  card.qb = dev<uint16_t>(card, R * D41_QLR);
  card.q = dev<float>(card, R * kHeads * D41_HD);
  card.ckv = dev<float>(card, R * D41_HD);
  card.csc = dev<float>(card, R * D41_HD);
  card.iq = dev<float>(card, R * D41_IDX_H * D41_IDX_D);
  card.iw = dev<float>(card, R * D41_IDX_H);
  card.score = dev<float>(card, R * D41_CTX_CAP);
  card.idx = dev<uint32_t>(card, R * D41_TOPK + 16);
  card.part = dev<float>(card, R * kHeads * D41_ATT_PART);
  card.oa = dev<uint8_t>(card, kGroups * R * D41_OG_A8_ROW);
  card.aa = dev<uint8_t>(card, R * (kGroups * D41_OLR + kGroups * D41_OLR / 8));
  card.y = dev<float>(card, R * E);
  for (auto &b : card.in) b = dev<float>(card, R * E);
  card.ekv = dev<float>(card, R * kEngKvRows);
  for (auto &s : card.eown) s = dev<D41EngramSend>(card, R);
  for (auto &s : card.ein) s = dev<D41EngramSend>(card, R);
  card.rlogits = dev<float>(card, R * D41_EXPERTS);
  card.ids = dev<int32_t>(card, R * D41_USED);
  card.w = dev<float>(card, R * D41_USED + 16);
  card.slots = dev<int32_t>(card, R * D41_USED + 16);
  card.nmiss = dev<uint32_t>(card, 64);
  card.groups = dev<D41Group>(card, R * D41_USED);
  card.ngroups = dev<uint32_t>(card, 64);
  card.ha = dev<uint8_t>(card, R * D41_USED * (D41_PART_A + D41_PART_A / 8));
  card.sgu = dev<float>(card, R * 2 * D41_PART_A);
  card.sha = dev<uint8_t>(card, R * (D41_PART_A + D41_PART_A / 8));
  card.shared = dev<float>(card, R * E);
  card.y6 = dev<float>(card, R * D41_USED * E);
  card.out = dev<float>(card, R * E);
}

void Model::load_layer(uint32_t l) {
  const int li = (int)l;
  const auto name = [&](const char *n) { return blk(l, n); };
  for (auto &cp : cards_) {
    Card &card = *cp;
    Layer &L = card.layers[l];
    const auto slice = [&](const char *n, uint64_t K, uint64_t rows, uint64_t row0, uint64_t nrows, uint64_t col0 = 0,
                           uint64_t ncols = 0) {
      const GgufTensor &t = tensor(name(n), {K, rows});
      const uint64_t qk = quant_block_values(t.type);
      if (ncols == 0) ncols = K;
      if (col0 % qk || ncols % qk) throw std::runtime_error(t.name + ": the cards' cut is not on its blocks");
      return mat(card, t.type, (uint32_t)ncols, {{held(t), t.row_bytes(), row0, nrows, col0 / qk, ncols / qk}});
    };
    L.attn_fn = upload_floats(card, name("hc_attn_fn.weight"), {D, D41_MIX});
    L.attn_scale = upload_floats(card, name("hc_attn_scale.weight"), {3});
    L.attn_base = upload_floats(card, name("hc_attn_base.weight"), {D41_MIX});
    L.ffn_fn = upload_floats(card, name("hc_ffn_fn.weight"), {D, D41_MIX});
    L.ffn_scale = upload_floats(card, name("hc_ffn_scale.weight"), {3});
    L.ffn_base = upload_floats(card, name("hc_ffn_base.weight"), {D41_MIX});
    L.attn_norm = upload_floats(card, name("attn_norm.weight"), {E});
    L.ffn_norm = upload_floats(card, name("ffn_norm.weight"), {E});
    L.q_norm = upload_floats(card, name("attn_q_a_norm.weight"), {D41_QLR});
    L.kv_norm = upload_floats(card, name("attn_kv_a_norm.weight"), {D41_HD});
    L.sinks = upload_floats(card, name("attn_sinks.weight"), {D41_H});
    L.bias = upload_floats(card, name("exp_probs_b.bias"), {D41_EXPERTS});
    L.q_a = slice("attn_q_a.weight", E, D41_QLR, 0, D41_QLR);
    L.kv = slice("attn_kv.weight", E, D41_HD, 0, D41_HD);
    L.q_b = slice("attn_q_b.weight", D41_QLR, D41_H * D41_HD, (uint64_t)card.c * kHeads * D41_HD, kHeads * D41_HD);
    L.out_a = slice("attn_output_a.weight", D41_OG_K, D41_OG * D41_OLR, (uint64_t)card.c * kGroups * D41_OLR,
                    kGroups * D41_OLR);
    if (L.out_a.type != QT_Q5_K) throw std::runtime_error(name("attn_output_a.weight") + ": the engine reads it as Q5_K only");
    L.out_b = slice("attn_output_b.weight", D41_OG * D41_OLR, E, 0, E, (uint64_t)card.c * kGroups * D41_OLR,
                    kGroups * D41_OLR);
    L.gate_inp = upload_bf16(card, name("ffn_gate_inp.weight"), {E, D41_EXPERTS});
    {
      // The larger part of the shared expert on the card that does not post the layer's misses (ADR-0074 item 2).
      const bool posts = l % 2 == card.c;
      const uint64_t r0 = posts ? D41_PART_A : 0;
      L.sh_ff = posts ? D41_PART_B : D41_PART_A;
      const GgufTensor &g = tensor(name("ffn_gate_shexp.weight"), {E, D41_FF});
      const GgufTensor &u = tensor(name("ffn_up_shexp.weight"), {E, D41_FF});
      if (g.type != u.type) throw std::runtime_error(u.name + ": not gate's type");
      const uint64_t nb = E / quant_block_values(g.type);
      L.sh_gu = mat(card, g.type, E, {{held(g), g.row_bytes(), r0, L.sh_ff, 0, nb}, {held(u), u.row_bytes(), r0, L.sh_ff, 0, nb}});
      L.sh_down = slice("ffn_down_shexp.weight", D41_FF, E, 0, E, r0, L.sh_ff);
    }
    L.ring = dev<uint16_t>(card, (size_t)opt_.slots * D41_RING * D41_HD);
    if (D41_KV_SRC(li) == li) {
      const uint32_t ratio = (uint32_t)D41_RATIO(li);
      L.comp_kv = upload_bf16(card, name("attn_compressor_kv.weight"), {E, D41_HD});
      if (ratio == 2) {
        L.comp_gate = upload_bf16(card, name("attn_compressor_gate.weight"), {E, D41_HD});
        L.cr = dev<float>(card, (size_t)opt_.slots * D41_CRING * D41_CRING_CELL);
      }
      L.comp_norm = upload_floats(card, name("attn_compressor_norm.weight"), {D41_HD});
      L.attn_k = upload_bf16(card, name("indexer.attn_k.weight"), {D41_HD, D41_IDX_D});
      L.k_norm = upload_floats(card, name("indexer.k_norm.weight"), {D41_IDX_D});
      L.pool = dev<uint16_t>(card, (size_t)kv_pages_ * (KV_PAGE_CELLS / ratio) * D41_COMP_ROW);
    }
    if (D41_RATIO(li) != 0 && D41_IDX_SRC(li) == li) {
      L.idx_q_b = upload_bf16(card, name("indexer.attn_q_b.weight"), {D41_QLR, D41_IDX_H * D41_IDX_D});
      L.idx_proj = upload_bf16(card, name("indexer.proj.weight"), {E, D41_IDX_H});
    }
    if (D41_ENGRAM(li) >= 0) {
      // The card's two streams' keys, then its half of the value rows.
      const GgufTensor &t = tensor(name("engram_kv.weight"), {D41_ENG_G, D41_ENG_KV});
      const uint64_t nb = D41_ENG_G / quant_block_values(t.type);
      L.engram_kv = mat(card, t.type, D41_ENG_G,
                        {{held(t), t.row_bytes(), (uint64_t)card.c * 2 * E, 2 * E, 0, nb},
                         {held(t), t.row_bytes(), (uint64_t)D + card.c * (E / 2), E / 2, 0, nb}});
      L.engram_gate = upload_floats(card, name("engram_gate.weight"), {E, D41_HC}, (size_t)card.c * 2 * E, 2 * E);
    }
  }
  held_.clear();
}

// The tiers, hottest first by the profile's routings a byte: the cards' slots, then the arena. The rest stay in the
// file. Without a profile each layer's first experts go in turn.
void Model::load_experts() {
  std::vector<uint64_t> profile;
  if (opt_.routing_profile && !opt_.routing_profile->empty() && *opt_.routing_profile != "none")
    profile = read_profile(*opt_.routing_profile);
  struct Hot {
    uint32_t l, e;
  };
  std::vector<Hot> hot;
  for (uint32_t e = 0; e < D41_EXPERTS; e++)
    for (uint32_t l = 0; l < D41_LAYERS; l++) hot.push_back({l, e});
  if (!profile.empty())
    std::stable_sort(hot.begin(), hot.end(), [&](const Hot &a, const Hot &b) {
      return profile[(size_t)a.l * D41_EXPERTS + a.e] * experts_[b.l].expert_bytes() >
             profile[(size_t)b.l * D41_EXPERTS + b.e] * experts_[a.l].expert_bytes();
    });

  // A card's room, less what the layers' odd pairs may put on one card more than the other.
  uint64_t room = UINT64_MAX;
  for (uint32_t c = 0; c < 2; c++) {
    const uint64_t total = rt_->cards()[c].memory, used = rt_->device_bytes(c) + opt_.reserve_bytes + (64ull << 20);
    info_.expert_budget[c] = total > used ? total - used : 0;
    room = std::min(room, info_.expert_budget[c]);
  }
  const uint64_t per_layer = opt_.experts_per_layer < 0 ? D41_EXPERTS : (uint64_t)opt_.experts_per_layer;
  std::vector<std::vector<uint32_t>> on_cards(D41_LAYERS), in_arena(D41_LAYERS);
  uint64_t card_bytes = 0, arena_bytes = 0, cached = 0;
  for (const Hot &h : hot) {
    const LayerExperts &le = experts_[h.l];
    const uint64_t slots = slot_class(le.shape.gate_type, le.shape.down_type, D41_PART_A, 0).slot_bytes +
                           slot_class(le.shape.gate_type, le.shape.down_type, D41_PART_B, 0).slot_bytes;
    const uint64_t in_ram = align256(le.expert_bytes());
    if (on_cards[h.l].size() < per_layer && card_bytes + slots <= 2 * room) {
      on_cards[h.l].push_back(h.e);
      card_bytes += slots;
      cached++;
    } else if (arena_bytes + in_ram <= opt_.arena_bytes) {
      in_arena[h.l].push_back(h.e);
      arena_bytes += in_ram;
    }
  }
  for (uint32_t l = 0; l < D41_LAYERS; l++) load_cache(l, on_cards[l]);
  info_.experts_per_layer = (uint32_t)(cached / D41_LAYERS);
  if (opt_.arena_bytes == 0) return;
  for (uint32_t l = 0; l < D41_LAYERS; l++) arena_->fill(ssd_->direct(), l, experts_[l], in_arena[l]);
  const ArenaReport r = arena_->settle();
  info_.arena_bytes = r.bytes;
}

// Layer l's experts ids on the cards: pair k's part A on one card and part B on the other, the cards taking turns so
// that both hold the same bytes.
void Model::load_cache(uint32_t l, const std::vector<uint32_t> &ids) {
  const LayerExperts &le = experts_[l];
  const uint32_t gt = le.shape.gate_type, dt = le.shape.down_type;
  const uint32_t n = (uint32_t)ids.size(), split = (n + 1) / 2;
  std::vector<int32_t> map(D41_EXPERTS, -1);
  for (uint32_t k = 0; k < n; k++) map[ids[k]] = (int32_t)k;
  D41Class tables[2][2];
  for (auto &cp : cards_) {
    Card &card = *cp;
    Layer &ly = card.layers[l];
    ly.slot_map = upload<int32_t>(card, map.data(), map.size());
    if (n == 0) continue;
    ly.split = split;
    ly.flip = card.c ^ (l & 1);
    const uint64_t part_a = ly.flip == 0 ? split : n - split;
    D41Class *t = tables[card.c];
    t[0] = slot_class(gt, dt, D41_PART_A, 0);
    t[1] = slot_class(gt, dt, D41_PART_B, part_a * t[0].slot_bytes);
    ly.cache = dev<uint8_t>(card, part_a * t[0].slot_bytes + (n - part_a) * t[1].slot_bytes);
    ly.classes = upload<D41Class>(card, t, 2);
  }
  if (n == 0) return;

  const uint64_t gu_blocks = E / 256, rb = gu_blocks * quant_block_bytes(gt), db = (uint64_t)(D41_FF / 256) * quant_block_bytes(dt);
  const uint64_t buf_bytes = std::max(le.gate_bytes, le.down_bytes) + 2 * DirectReader::kAlign;
  const uint64_t slot_bytes = slot_class(gt, dt, D41_PART_A, 0).slot_bytes;
  std::mutex mu;
  std::exception_ptr failed;
  const DirectReader &reader = ssd_->direct();
  ssd_->direct().pool().run(n, [&](uint64_t k) {
    try {
      AlignedBuffer buf = aligned_buffer(buf_bytes);
      std::vector<uint8_t> slot[2] = {std::vector<uint8_t>(slot_bytes), std::vector<uint8_t>(slot_bytes)};
      const D41Class *cls[2];
      uint64_t r0[2];
      for (uint32_t c = 0; c < 2; c++) {
        const uint32_t part = (uint32_t)(k >= split) ^ cards_[c]->layers[l].flip;
        cls[c] = &tables[c][part];
        r0[c] = part ? D41_PART_A : 0;
      }
      const uint32_t e = ids[k];
      const uint8_t *g = reader.read_one(le.gate.file, le.gate.offset + e * le.gate_bytes, le.gate_bytes, buf.get());
      for (uint32_t c = 0; c < 2; c++) to_runs(gt, {{g, rb, r0[c], cls[c]->rows, 0, gu_blocks}}, slot[c].data());
      const uint8_t *u = reader.read_one(le.up.file, le.up.offset + e * le.gate_bytes, le.gate_bytes, buf.get());
      for (uint32_t c = 0; c < 2; c++)
        to_runs(gt, {{u, rb, r0[c], cls[c]->rows, 0, gu_blocks}}, slot[c].data() + cls[c]->up_off);
      const uint8_t *d = reader.read_one(le.down.file, le.down.offset + e * le.down_bytes, le.down_bytes, buf.get());
      for (uint32_t c = 0; c < 2; c++)
        to_runs(dt, {{d, db, 0, E, r0[c] / 256, cls[c]->rows / 256}}, slot[c].data() + cls[c]->down_off);
      for (uint32_t c = 0; c < 2; c++) {
        Card &card = *cards_[c];
        const uint64_t at = k < split ? k : k - split;
        std::lock_guard<std::mutex> lk(card.up_mu);
        card.up->copy(card.layers[l].cache + cls[c]->base + at * cls[c]->slot_bytes, slot[c].data(), cls[c]->slot_bytes);
        card.up->synchronize();
      }
    } catch (...) {
      std::lock_guard<std::mutex> lk(mu);
      if (!failed) failed = std::current_exception();
    }
  });
  if (failed) std::rethrow_exception(failed);
}

ze_kernel_handle_t Model::kernel(Card &card, const char *module, const char *name) {
  return lib_->get(card.c, module, name);
}

void Model::matvec(ze::List &L, Card &card, const char *name, const Mat &m, const uint8_t *act, float *y, uint32_t W,
                   uint32_t lanes, uint32_t ks) {
  L.label(name);
  const uint32_t stride = m.K + m.K / 8, rps = 16 / lanes;
  const std::string mod = mv_module(m.type, lanes);
  if (ks) {
    L.launch(lib_->get(card.c, mod, "matvec_ks"), {m.p, m.rows, m.bpr, act, m.K, stride, W, y, m.rows, card.grids, ks},
             (m.rows + rps - 1) / rps, 1, 1, 16 * ks);
  } else {
    L.launch(lib_->get(card.c, mod, "matvec"), {m.p, m.rows, m.bpr, act, m.K, stride, W, y, m.rows, card.grids},
             (m.rows + 4 * rps - 1) / (4 * rps), 1, 1, 64);
  }
}

// y[t][n] = w[n] . xb[t] for the step's real rows; xb holds STEP_MAX_ROWS rows, all read.
void Model::product_bf16(ze::List &L, Card &card, const char *name, const uint16_t *w, uint32_t N, uint32_t K,
                         const uint16_t *xb, float *y) {
  L.label(name);
  const uint32_t zero = 0, ks = D41_RT_KS;
  const float one = 1.0f;
  L.launch(kernel(card, "common", "mv_xmx_bf16"), {w, N, K, K, xb, &card.plan->real, y, N, zero, one, zero, ks, none},
           (N + 15) / 16, 1, 1, 16 * ks);
}

void Model::wait(ze::List &L, Card &card, const char *name, uint32_t *raise, const uint32_t *flag, const uint32_t *nmiss,
                 uint32_t W) {
  L.label(name);
  L.launch(kernel(card, "deepseek41", "ds_wait"),
           {raise, none, flag, &card.plan->step, nmiss, W, &card.plan->stuck, &card.plan->start, none}, 1, 1, 1, 1);
}

// Engram's injection before layer l's attention: the card's keys and half the value, the gates and the value's halves
// exchanged, then X changed in place and the attention's input made from it with the last sublayer's mixes.
void Model::record_engram(ze::List &L, Card &card, uint32_t l, uint32_t W) {
  Layer &ly = card.layers[l];
  Card &other = *cards_[1 - card.c];
  const uint32_t e = (uint32_t)D41_ENGRAM((int)l), k = 2 * D41_LAYERS + e;
  const uint32_t c0 = 2 * card.c, c1 = c0 + 2, v0 = card.c * (E / 2), v1 = v0 + E / 2, stride = kEngKvRows;
  const uint32_t in_card = 0;
  matvec(L, card, "engram_kv", ly.engram_kv, card.eng_a8[e], card.ekv, W, 8);
  L.launch(kernel(card, "deepseek41", "ds_engram_gate"),
           {card.X, card.ekv, stride, ly.engram_gate, c0, c1, v0, v1, card.eown[e], other.ein[e]}, c1 - c0, W, 1, 256);
  wait(L, card, "wait: other card, Engram", flag(card.c, k), flag(other.c, k));
  L.launch(kernel(card, "deepseek41", "ds_engram_apply"),
           {card.X, card.ekv, stride, c0, c1, v0, v1, card.eown[e], card.ein[e], in_card, card.fmix, ly.attn_norm,
            card.ss, card.xa, card.xb, none},
           W, 1, 1, D41_HC_SGS * 16);
}

// Layer l's attention from x (xa, xb) to the card's partial sum, the exchange, and the MoE block's input.
void Model::record_attention(ze::List &L, Card &card, uint32_t l, uint32_t W) {
  Layer &ly = card.layers[l];
  Card &other = *cards_[1 - card.c];
  const int li = (int)l;
  const uint32_t ratio = (uint32_t)D41_RATIO(li), yarn = ratio != 0, first_head = card.c * kHeads;
  const bool kv_src = D41_KV_SRC(li) == li, idx_src = ratio != 0 && D41_IDX_SRC(li) == li;
  // A window-only layer reads no compressed row; it is given layer 2's pool so that the pointer is one it may read.
  uint16_t *pool = card.layers[ratio ? (uint32_t)D41_KV_SRC(li) : 2].pool;
  D41Row *rows = card.plan->rows;
  const auto attn = [&](const char *name) { return kernel(card, "deepseek41_attn", name); };

  L.launch(kernel(card, "deepseek41", "ds_hc_mix"), {ly.attn_fn, card.X, card.ss, W, card.m}, D41_MIX, 1, 1,
           16 * D41_HC_KSPLIT);
  L.launch(kernel(card, "deepseek41", "ds_hc_sinkhorn"), {card.m, ly.attn_scale, ly.attn_base, card.amix}, W, 1, 1, 16);

  matvec(L, card, "attn_q_a", ly.q_a, card.xa, card.qr, W, 4, 4);
  matvec(L, card, "attn_kv", ly.kv, card.xa, card.kv, W, 8, 4);
  if (kv_src) {
    product_bf16(L, card, "compressor_kv", ly.comp_kv, D41_HD, E, card.xb, card.ckv);
    if (ratio == 2) product_bf16(L, card, "compressor_gate", ly.comp_gate, D41_HD, E, card.xb, card.csc);
  }
  const void *qb = idx_src ? card.qb : nullptr;
  L.launch(attn("ds_qkv_norm"), {card.qr, ly.q_norm, card.kv, ly.kv_norm, rows, yarn, card.qa, qb, none, ly.ring, none},
           2, W, 1, D41_QKV_SGS * 16);
  matvec(L, card, "attn_q_b", ly.q_b, card.qa, card.q, W, 4);
  if (kv_src) {
    const void *sc = ratio == 2 ? card.csc : nullptr, *cr = ly.cr;
    L.launch(attn("ds_compress"),
             {card.ckv, sc, cr, rows, ratio, ly.comp_norm, ly.attn_k, ly.k_norm, card.pt, npp_, ly.pool, none}, W, 1, 1,
             256);
  }
  if (idx_src) {
    product_bf16(L, card, "indexer q_b", ly.idx_q_b, D41_IDX_H * D41_IDX_D, D41_QLR, card.qb, card.iq);
    product_bf16(L, card, "indexer proj", ly.idx_proj, D41_IDX_H, E, card.xb, card.iw);
    L.launch(attn("ds_idx_score"), {card.iq, card.iw, rows, ratio, card.pt, npp_, pool, card.score}, D41_IDX_WGS, W, 1,
             D41_IDX_WG_ROWS);
    L.launch(attn("ds_idx_topk"), {card.score, rows, ratio, card.idx}, W, 1, 1, kTopkLocal);
  }
  L.launch(attn("ds_attn_split"), {card.q, rows, ratio, ly.ring, card.pt, npp_, pool, card.idx, card.part},
           kHeads / D41_ATT_HEADS, W, ratio ? D41_ATT_SPLITS : 1, D41_ATT_WG);
  L.launch(attn("ds_attn_merge"), {card.part, ly.sinks, first_head, rows, ratio, card.oa, none}, kHeads, W, 1, 256);
  L.launch(kernel(card, "deepseek41_og", "ds_output_groups"), {ly.out_a.p, ly.out_a.bpr, card.oa, W, card.aa, none},
           D41_OLR / D41_OG_WG_ROWS, kGroups, 1, D41_OG_WG_ROWS / (16 / D41_OG_LANES) * 16);
  matvec(L, card, "attn_output_b", ly.out_b, card.aa, card.y, W, 8);

  const uint32_t k = 2 * l, in_card = 0;
  L.launch(kernel(card, "deepseek41", "ds_exch_send"), {card.y, other.in[0]}, W, 1, 1, D41_EX_SGS * 16);
  wait(L, card, "wait: other card, attention", flag(card.c, k), flag(other.c, k));
  const void *job = l % 2 == card.c ? jobs_ + (size_t)l * STEP_MAX_ROWS : nullptr;
  L.launch(kernel(card, "deepseek41", "ds_hc_post_norm"),
           {card.X, card.y, card.in[0], in_card, card.amix, ly.ffn_norm, card.X1, card.ss, card.xa, card.xb, none, job},
           W, 1, 1, D41_HC_SGS * 16);
  if (probe_ && card.c == 0) L.copy(probe_ + (size_t)l * STEP_MAX_ROWS * D, card.X1, (size_t)W * D * 4);
}

// Layer l's MoE block: the router, the shared expert's part, the CPU's answer on the posting card, the exchange, and the
// next sublayer's input.
void Model::record_ffn(ze::List &L, Card &card, uint32_t l, uint32_t W) {
  Layer &ly = card.layers[l];
  Card &other = *cards_[1 - card.c];
  const bool posts = l % 2 == card.c;
  const uint32_t ff = ly.sh_ff, two_ff = 2 * ff, in_card = 0;

  L.launch(kernel(card, "deepseek41", "ds_hc_mix"), {ly.ffn_fn, card.X1, card.ss, W, card.m}, D41_MIX, 1, 1,
           16 * D41_HC_KSPLIT);
  L.launch(kernel(card, "deepseek41", "ds_hc_sinkhorn"), {card.m, ly.ffn_scale, ly.ffn_base, card.fmix}, W, 1, 1, 16);

  product_bf16(L, card, "router product", ly.gate_inp, D41_EXPERTS, E, card.xb, card.rlogits);
  const void *jobs = posts ? jobs_ + (size_t)l * STEP_MAX_ROWS : nullptr;
  L.launch(kernel(card, "deepseek41", "ds_router"),
           {card.rlogits, ly.bias, ly.slot_map, W, ly.split, ly.flip, &card.plan->real, &card.plan->step, card.ids, card.w,
            card.slots, card.nmiss, jobs, card.groups, card.ngroups},
           1, 1, 1, 16 * W * D41_RT_SGS(W));

  if (ly.cache) {
    const LayerExperts &le = experts_[l];
    const uint32_t per = D41_USED, xs = D41_A8_ROW, hs = D41_PART_A + D41_PART_A / 8, nasg = W * D41_USED;
    const uint32_t lxg = 16 * MOE_WG_ROWS / std::max<uint32_t>(D41_GU_ROWS(W), 16 / D41_GU_LANES);
    const uint32_t lxd = 16 * MOE_WG_ROWS / std::max<uint32_t>(D41_DOWN_ROWS, 16 / D41_DOWN_LANES(le.shape.down_type, W));
    L.launch(lib_->get(card.c, moe_module("gate_up", le.shape.gate_type, W), "moe_gate_up_c"),
             {ly.cache, ly.classes, card.groups, card.ngroups, card.xa, E, xs, per, card.ha, hs, card.grids},
             D41_PART_A / MOE_WG_ROWS, nasg, 1, lxg);
    L.launch(lib_->get(card.c, moe_module("down", le.shape.down_type, W), "moe_down_c"),
             {ly.cache, ly.classes, card.groups, card.ngroups, E, card.ha, hs, card.y6, card.grids}, E / MOE_WG_ROWS, nasg,
             1, lxd);
  }
  matvec(L, card, "shared gate and up", ly.sh_gu, card.xa, card.sgu, W, 8);
  L.launch(kernel(card, "deepseek41", "ds_swiglu_clamp_a8"), {card.sgu, card.sgu + ff, two_ff, ff, card.sha}, W, 1, 1,
           D41_SWIGLU_SGS * 16);
  matvec(L, card, "shared down", ly.sh_down, card.sha, card.shared, W, 4);

  const void *cpu = posts ? cpu_y_ + (size_t)l * STEP_MAX_ROWS * E : nullptr;
  if (posts) wait(L, card, "wait: CPU experts", nullptr, ans_flag_ + 16 * l, card.nmiss, W);
  L.launch(kernel(card, "deepseek41", "ds_moe_combine"),
           {card.y6, card.w, card.slots, card.shared, cpu, card.nmiss, card.out, other.in[1]}, W, 1, 1,
           D41_CMB_SGS(W) * 16);
  const uint32_t k = 2 * l + 1;
  wait(L, card, "wait: other card, MoE", flag(card.c, k), flag(other.c, k));

  // Before an Engram layer only the streams are made: its injection comes before the collapse.
  const bool last = l + 1 == D41_LAYERS, engram = !last && D41_ENGRAM((int)l + 1) >= 0;
  const void *norm = engram ? nullptr : last ? card.out_norm : card.layers[l + 1].attn_norm;
  const void *xa = engram ? nullptr : card.xa, *xb = engram ? nullptr : card.xb;
  L.launch(kernel(card, "deepseek41", "ds_hc_post_norm"),
           {card.X1, card.out, card.in[1], in_card, card.fmix, norm, card.X, card.ss, xa, xb, none, none}, W, 1, 1,
           D41_HC_SGS * 16);
  if (probe_ && card.c == 0)
    L.copy(probe_ + ((size_t)D41_LAYERS + l) * STEP_MAX_ROWS * D, card.X, (size_t)W * D * 4);
}

void Model::record(Card &card, uint32_t W) {
  auto list = std::make_unique<ze::List>(*rt_, card.c, ze::List::regular, false, false);
  ze::List &L = *list;
  if (opt_.trace) {
    constexpr size_t cap = 4096;
    card.trace_ts[W] = static_cast<uint64_t *>(rt_->alloc_host(cap * 8));
    L.trace(card.trace_ts[W], cap);
  }
  L.copy(card.plan, plan_, sizeof(StepWords));
  L.copy(card.e, embed_, (size_t)W * E * 4);
  for (uint32_t e = 0; e < kEngramTables; e++) L.copy(card.eng_a8[e], eng_a8_[e], (size_t)W * kEngA8Row);
  L.launch(kernel(card, "deepseek41", "ds_hc_init"),
           {card.e, card.layers[0].attn_norm, card.X, card.ss, card.xa, card.xb, none}, W, 1, 1, D41_HC_SGS * 16);
  for (uint32_t l = 0; l < D41_LAYERS; l++) {
    if (D41_ENGRAM((int)l) >= 0) record_engram(L, card, l, W);
    record_attention(L, card, l, W);
    record_ffn(L, card, l, W);
  }
  matvec(L, card, "output", card.head, card.xa, card.logits, W, 4);
  L.launch(kernel(card, "common", "topk_rows"),
           {card.logits, card.vocab_n, card.vocab_n, opt_.candidates, card.vocab0, &card.plan->real,
            top_idx_ + (size_t)card.c * STEP_MAX_ROWS * opt_.candidates,
            top_val_ + (size_t)card.c * STEP_MAX_ROWS * opt_.candidates, top_n_ + (size_t)card.c * STEP_MAX_ROWS},
           W, 1, 1, kTopkLocal);
  L.copy(stuck_ + 16 * card.c, &card.plan->stuck, 4);
  L.close();
  card.lists[W] = std::move(list);
}

std::vector<std::vector<Candidate>> Model::step(const StepIn *rows, uint32_t n, bool, bool) {
  if (n == 0 || n > STEP_MAX_ROWS) throw std::runtime_error("a step takes 1 to 8 rows");
  for (uint32_t r = 0; r < n; r++) {
    if (rows[r].slot >= opt_.slots) throw std::runtime_error("slot out of range");
    if (rows[r].pos >= opt_.max_ctx) throw std::runtime_error("position past the context");
    if (rows[r].token < 0 || (uint32_t)rows[r].token >= vocab_) throw std::runtime_error("a token outside the vocabulary");
    // The kernels index the pool by the page: one not set would be a write outside it.
    if (pt_[(size_t)rows[r].slot * npp_ + rows[r].pos / KV_PAGE_CELLS] == D41_NO_PAGE)
      throw std::runtime_error("a row's page is not set");
  }
  const uint32_t W = width(n);
  for (auto &card : cards_)
    if (!card->lists.count(W)) record(*card, W);
  using Clock = std::chrono::steady_clock;
  const auto ms = [](Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
  const auto t_start = Clock::now();
  times_ = {};

  last_width_ = W;
  plan_->real = n;
  plan_->step = ++step_;
  plan_->stuck = 0;
  plan_->start = 0;
  const GgufTensor &te = gguf_.tensor("token_embd.weight");
  const auto to_float = ggml_get_type_traits((ggml_type)te.type)->to_float;
  for (uint32_t r = 0; r < W; r++) {
    if (r >= n) {
      plan_->rows[r] = {D41_NO_ROW, 0, 0, D41_NO_ROW};
      std::memset(embed_ + (size_t)r * E, 0, E * 4);
      for (auto &a : eng_a8_) std::memset(a + (size_t)r * kEngA8Row, 0, kEngA8Row);
      continue;
    }
    uint32_t prev = D41_NO_ROW;
    for (uint32_t q = 0; q < r; q++)
      if (rows[q].slot == rows[r].slot && rows[q].pos + 1 == rows[r].pos) prev = q;
    plan_->rows[r] = {rows[r].slot, rows[r].pos, rows[r].pos % D41_RING, prev};
    to_float(te.data + (uint64_t)rows[r].token * te.row_bytes(), embed_ + (size_t)r * E, E);
  }
  // Both layers' Engram rows before the lists start, which copy them to the cards.
  for (uint32_t e = 0; e < kEngramTables; e++) {
    std::vector<uint64_t> ids((size_t)n * D41_ENG_COLS);
    for (uint32_t r = 0; r < n; r++) {
      const auto h = hash_->rows(rows[r].token, rows[r].p1, rows[r].p2, rows[r].p3, e);
      std::copy(h.begin(), h.end(), ids.begin() + (size_t)r * D41_ENG_COLS);
    }
    engram_->gather(ids.data(), (uint32_t)ids.size(), eng_f_.data());
    for (uint32_t r = 0; r < n; r++)
      a8_row(eng_f_.data() + (size_t)r * D41_ENG_G, D41_ENG_G, eng_a8_[e] + (size_t)r * kEngA8Row);
  }

  auto t_mark = Clock::now();
  times_.before = ms(t_start, t_mark);
  for (auto &card : cards_) card->lists.at(W)->execute();
  stats_ = {};
  counts_ = {};
  std::exception_ptr failed;
  uint32_t answered = 0;
  // Where a step that failed stopped, from the flags each card raised.
  const auto progress = [&] {
    std::string s = "; the CPU answered " + std::to_string(answered) + " layers";
    for (uint32_t c = 0; c < 2; c++) {
      uint32_t raised = 0;
      for (uint32_t k = 0; k < kFlags; k++) raised += *flag(c, k) == step_;
      s += ", card " + std::to_string(c) + " reached " + std::to_string(raised) + " of " + std::to_string(kFlags) + " exchanges";
    }
    return s;
  };
  try {
    for (uint32_t l = 0; l < D41_LAYERS; l++, answered++) {
      // The posting card's records (ADR-0011): spin until each real row's holds this step.
      D41MissJob *jobs = jobs_ + (size_t)l * STEP_MAX_ROWS;
      for (uint32_t r = 0; r < n; r++) {
        std::atomic_ref<uint32_t> seq(jobs[r].seq);
        const auto t0 = std::chrono::steady_clock::now();
        for (uint32_t k = 0; seq.load(std::memory_order_acquire) != step_; k++) {
          _mm_pause();
          if ((k & 0xffff) == 0 && std::chrono::steady_clock::now() - t0 > kPublishLimit)
            throw std::runtime_error("card " + std::to_string(l % 2) + " did not publish layer " + std::to_string(l) +
                                     "'s experts");
        }
      }
      const auto t_seen = Clock::now();
      times_.records += ms(t_mark, t_seen);
      MissAsk ask;
      ask.layer = l;
      ask.rows = W;
      ask.real_rows = n;
      ask.step = step_;
      ask.jobs = jobs;
      ask.answer = cpu_y_ + (size_t)l * STEP_MAX_ROWS * E;
      ask.answer_flag = ans_flag_ + 16 * l;
      const MissCounts c = miss_->run_layer(ask);
      t_mark = Clock::now();
      times_.answers += ms(t_seen, t_mark);
      counts_.routings += c.routings;
      counts_.arena_jobs += c.arena_jobs;
      counts_.ssd_jobs += c.ssd_jobs;
      counts_.ssd_experts += c.ssd_experts;
      counts_.read_rounds += c.read_rounds;
      counts_.cpu_pool_runs += c.cpu_pool_runs;
      stats_.misses += c.routings;
      stats_.layers_missed += c.routings != 0;
    }
  } catch (...) {
    failed = std::current_exception();
  }
  // The lists end by themselves: a wait the CPU never answers gives up (ADR-0074 item 5).
  try {
    for (auto &card : cards_) card->lists.at(W)->synchronize();
    times_.after = ms(t_mark, Clock::now());
    if (failed) std::rethrow_exception(failed);
  } catch (const std::exception &e) {
    throw std::runtime_error(e.what() + progress());
  }
  for (uint32_t c = 0; c < 2; c++)
    if (stuck_[16 * c]) throw std::runtime_error("card " + std::to_string(c) + " gave up a wait of the step" + progress());

  std::vector<std::vector<Candidate>> out(n);
  for (uint32_t r = 0; r < n; r++) {
    auto &o = out[r];
    for (uint32_t c = 0; c < 2; c++) {
      const size_t base = ((size_t)c * STEP_MAX_ROWS + r) * opt_.candidates;
      for (uint32_t i = 0; i < top_n_[c * STEP_MAX_ROWS + r]; i++)
        o.push_back({(int32_t)top_idx_[base + i], top_val_[base + i]});
    }
    std::sort(o.begin(), o.end(), [](const Candidate &a, const Candidate &b) {
      return a.logit != b.logit ? a.logit > b.logit : a.token < b.token;
    });
    if (o.size() > opt_.candidates) o.resize(opt_.candidates);
  }
  return out;
}

std::vector<float> Model::logits(uint32_t row) {
  if (row >= STEP_MAX_ROWS) throw std::runtime_error("no such row");
  std::vector<float> v(vocab_);
  for (auto &card : cards_) {
    card->up->copy(v.data() + card->vocab0, card->logits + (size_t)row * card->vocab_n, (size_t)card->vocab_n * 4);
    card->up->synchronize();
  }
  return v;
}

std::vector<TraceRow> Model::trace(uint32_t c) const {
  const Card &card = *cards_[c];
  const auto ts = card.trace_ts.find(last_width_);
  if (ts == card.trace_ts.end()) throw std::runtime_error("the last step was not traced");
  const std::vector<std::string> &names = card.lists.at(last_width_)->trace_names();
  const ze::Card &dev = rt_->cards()[c];
  std::vector<TraceRow> rows;
  for (size_t i = 0; i < names.size(); i++) {
    auto row = std::find_if(rows.begin(), rows.end(), [&](const TraceRow &r) { return r.kernel == names[i]; });
    if (row == rows.end()) row = rows.insert(rows.end(), TraceRow{names[i], 0, 0});
    row->launches++;
    row->ms += (double)((ts->second[i + 1] - ts->second[i]) & dev.timestamp_mask) * 1e3 / (double)dev.timer_hz;
  }
  return rows;
}

const float *Model::probe(uint32_t layer, bool ffn) const {
  if (!probe_ || layer >= D41_LAYERS) throw std::runtime_error("no such probe");
  return probe_ + ((size_t)(ffn ? D41_LAYERS : 0) + layer) * STEP_MAX_ROWS * D;
}

void Model::set_images(uint32_t, const Image *, uint32_t n) {
  if (n) throw std::runtime_error("deepseek41 reads no images");
}

std::vector<std::vector<Candidate>> Model::mtp_catchup(const StepIn *, uint32_t, const std::vector<uint32_t> &,
                                                       const uint32_t *) {
  throw std::runtime_error("deepseek41 has no drafter");
}
std::vector<Candidate> Model::mtp_draft(uint32_t, uint32_t, int32_t, uint32_t) {
  throw std::runtime_error("deepseek41 has no drafter");
}
std::vector<std::vector<Candidate>> Model::mtp_drafts(const StepIn *, uint32_t, const uint32_t *) {
  throw std::runtime_error("deepseek41 has no drafter");
}
std::vector<std::vector<Candidate>> Model::mtp_catchup_prompt(const StepIn *, uint32_t, uint32_t,
                                                              const std::vector<uint32_t> &) {
  throw std::runtime_error("deepseek41 has no drafter");
}
std::vector<std::vector<std::vector<Candidate>>> Model::df_pass(const StepIn *, uint32_t, const DraftBlock *, uint32_t) {
  throw std::runtime_error("deepseek41 has no drafter");
}
void Model::df_context_prompt(const StepIn *, uint32_t, uint32_t) {
  throw std::runtime_error("deepseek41 has no drafter");
}
std::vector<std::vector<Candidate>> Model::prompt_runs(const PromptRun *, uint32_t, const int32_t *, uint32_t) {
  throw std::runtime_error("deepseek41 has no prompt path: its prompts are read as decode steps");
}

// A slot's state outside the pages: every layer's window, then each ratio-2 source's compressor ring. Both cards hold
// the same bits, so card 0's are saved.
uint64_t Model::slot_state_bytes() {
  uint64_t n = (uint64_t)D41_LAYERS * D41_RING * D41_HD * 2;
  for (const Layer &ly : cards_[0]->layers)
    if (ly.cr) n += (uint64_t)D41_CRING * D41_CRING_CELL * 4;
  return n;
}

void Model::save_slot(uint32_t slot, uint8_t *dst) {
  if (slot >= opt_.slots) throw std::runtime_error("slot out of range");
  Card &card = *cards_[0];
  const size_t ring = (size_t)D41_RING * D41_HD * 2, cr = (size_t)D41_CRING * D41_CRING_CELL * 4;
  for (const Layer &ly : card.layers) {
    card.up->copy(dst, reinterpret_cast<const uint8_t *>(ly.ring) + slot * ring, ring);
    dst += ring;
    if (!ly.cr) continue;
    card.up->copy(dst, reinterpret_cast<const uint8_t *>(ly.cr) + slot * cr, cr);
    dst += cr;
  }
  card.up->synchronize();
}

void Model::load_slot(uint32_t slot, const uint8_t *src) {
  if (slot >= opt_.slots) throw std::runtime_error("slot out of range");
  const size_t ring = (size_t)D41_RING * D41_HD * 2, cr = (size_t)D41_CRING * D41_CRING_CELL * 4;
  for (auto &cp : cards_) {
    const uint8_t *at = src;
    for (const Layer &ly : cp->layers) {
      cp->up->copy(reinterpret_cast<uint8_t *>(ly.ring) + slot * ring, at, ring);
      at += ring;
      if (!ly.cr) continue;
      cp->up->copy(reinterpret_cast<uint8_t *>(ly.cr) + slot * cr, at, cr);
      at += cr;
    }
    cp->up->synchronize();
  }
}

void Model::sleep() { throw std::runtime_error("deepseek41 does not sleep yet"); }
void Model::wake() { throw std::runtime_error("deepseek41 does not sleep yet"); }

void Model::set_pages(uint32_t slot, uint32_t first, const uint32_t *pages, uint32_t n) {
  if (slot >= opt_.slots || first + n > npp_) throw std::runtime_error("pages outside the slot's table");
  for (uint32_t i = 0; i < n; i++)
    if (pages[i] != D41_NO_PAGE && pages[i] >= kv_pages_) throw std::runtime_error("a page outside the pool");
  const size_t at = (size_t)slot * npp_ + first;
  std::copy(pages, pages + n, pt_.begin() + (std::ptrdiff_t)at);
  for (auto &cp : cards_) {
    cp->up->copy(cp->pt + at, pt_.data() + at, (size_t)n * 4);
    cp->up->synchronize();
  }
}

// The compressed rows whose positions all lie in the page's first cells; a ratio-2 row's odd half is in the slot's state.
void Model::copy_cells(uint32_t from, uint32_t to, uint32_t cells) {
  if (from >= kv_pages_ || to >= kv_pages_ || cells >= KV_PAGE_CELLS) throw std::runtime_error("copy_cells: a page or count out of range");
  for (auto &cp : cards_) {
    for (uint32_t l = 0; l < D41_LAYERS; l++) {
      const Layer &ly = cp->layers[l];
      if (!ly.pool) continue;
      const uint32_t ratio = (uint32_t)D41_RATIO((int)l), per = KV_PAGE_CELLS / ratio, rows = cells / ratio;
      if (rows == 0) continue;
      const size_t row = D41_COMP_ROW_BYTES;
      cp->up->copy(reinterpret_cast<uint8_t *>(ly.pool) + (size_t)to * per * row,
                   reinterpret_cast<const uint8_t *>(ly.pool) + (size_t)from * per * row, rows * row);
    }
    cp->up->synchronize();
  }
}

}  // namespace engine::deepseek41
