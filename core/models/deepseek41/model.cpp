#include "models/deepseek41/model.h"
#include "models/deepseek41/prompt.h"

#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <utility>

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
// The swaps' rule: a routing counts kHeatDecay times less each step; an expert comes in when it leads the one going out
// by kSwapLead times and kSwapFloor routings; a profile counts as kProfileSteps steps; at most kSwapBatch pairs change
// at once, kSwapsALayer of a layer.
constexpr float kHeatDecay = 0.998f, kSwapLead = 1.0f, kSwapFloor = 0.5f, kProfileSteps = 200.0f;
constexpr size_t kSwapBatch = 16, kSwapsALayer = 2, kFileSeen = 64;
// Threads writing experts into the arena, each with its own reads from the file. Few, since reads queued beside the
// Engram rows' and the CPU's slow both.
constexpr uint32_t kRefillThreads = 2;

double now_ms() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
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
  const uint64_t g = align256(runs_bytes(gate_type, (uint64_t)rows * (E / quant_block_values(gate_type))));
  const uint64_t d = align256(runs_bytes(down_type, (uint64_t)E * (rows / quant_block_values(down_type))));
  return {base, 2 * g + d, g, 2 * g, rows, {}};
}

// A layer's kind. The drafter's layers, D41_LAYERS and on, are window-only and have no Engram.
bool drafter_layer(uint32_t l) { return l >= D41_LAYERS; }
uint32_t ratio_of(uint32_t l) { return drafter_layer(l) ? 0 : (uint32_t)D41_RATIO((int)l); }
bool kv_source(uint32_t l) { return !drafter_layer(l) && D41_KV_SRC((int)l) == (int)l; }
bool idx_source(uint32_t l) { return ratio_of(l) != 0 && D41_IDX_SRC((int)l) == (int)l; }
int engram_of(uint32_t l) { return drafter_layer(l) ? -1 : D41_ENGRAM((int)l); }
uint32_t experts_of(uint32_t l) { return drafter_layer(l) ? D41_DSP_EXPERTS : D41_EXPERTS; }
uint32_t used_of(uint32_t l) { return drafter_layer(l) ? D41_DSP_USED : D41_USED; }
// The kernels built for a layer's expert counts.
const char *layer_module(uint32_t l) { return drafter_layer(l) ? "deepseek41_dsp" : "deepseek41"; }

// attn_output_a's kernels, by its type.
const char *og_module(uint32_t type) {
  if (type == QT_Q5_K) return "deepseek41_og";
  if (type == QT_Q8_0) return "deepseek41_og_q8_0";
  throw std::runtime_error("attn_output_a.weight: the engine reads it as Q5_K or Q8_0 only");
}

// x as bf16, to the nearest, ties to even.
uint16_t to_bf16(float x) {
  uint32_t b;
  std::memcpy(&b, &x, 4);
  return (uint16_t)((b + 0x7fffu + ((b >> 16) & 1u)) >> 16);
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
  const GgufTensor &t = source().tensor(name);
  if (!std::equal(t.shape.begin(), t.shape.end(), shape.begin(), shape.end()))
    throw std::runtime_error(name + ": not the shape deepseek41's layout gives");
  return t;
}

// Read past the page cache, as every dense tensor is (ADR-0080 item 3), and kept until its layer is on both cards.
const uint8_t *Model::held(const GgufTensor &t) {
  auto [it, fresh] = held_.try_emplace({t.shard, t.offset});
  if (fresh) {
    it->second = aligned_buffer(DirectReader::span(t.offset, t.bytes));
    reader().read(t.shard, t.offset, t.bytes, it->second.get());
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
  if (t.type == QT_BF16) return upload<uint16_t>(card, reinterpret_cast<const uint16_t *>(held(t)), t.elements());
  // The drafter's router is F32 in its file; the cards' product reads bf16.
  if (t.type != QT_F32) throw std::runtime_error(name + ": the engine reads it as BF16 or F32 only");
  const float *f = reinterpret_cast<const float *>(held(t));
  std::vector<uint16_t> v(t.elements());
  for (size_t i = 0; i < v.size(); i++) v[i] = to_bf16(f[i]);
  return upload<uint16_t>(card, v.data(), v.size());
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
  if (!opt_.vision_path.empty()) throw std::runtime_error("deepseek41 reads no images");
  if (opt_.max_cards < 2) throw std::runtime_error("deepseek41 runs on two cards");
  rt_ = std::make_unique<ze::Runtime>(2);
  lib_ = std::make_unique<ze::Library>(*rt_);
  pool_ = std::make_unique<Pool>(opt_.cpu_threads, true);
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
  engram_thread_ = std::thread([this] { engram_work(); });

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
  eng_flag_ = static_cast<uint32_t *>(host((size_t)kEngramTables * 64));
  flags_ = static_cast<uint32_t *>(host((size_t)2 * kFlags * 64));
  stuck_ = static_cast<uint32_t *>(host(2 * 64));
  top_idx_ = static_cast<uint32_t *>(host((size_t)2 * STEP_MAX_ROWS * opt_.candidates * 4));
  top_val_ = static_cast<float *>(host((size_t)2 * STEP_MAX_ROWS * opt_.candidates * 4));
  top_n_ = static_cast<uint32_t *>(host(2 * STEP_MAX_ROWS * 4));
  if (debug_.probes) probe_ = static_cast<float *>(host((size_t)2 * kAllLayers * STEP_MAX_ROWS * D * 4));
  if (!opt_.mtp_dir.empty()) {
    dsp_ = std::make_unique<Drafter>(opt_.mtp_dir);
    dsp_->x = static_cast<float *>(host((size_t)STEP_MAX_ROWS * D * 4));
    dsp_->mix = static_cast<D41HcMix *>(host(STEP_MAX_ROWS * sizeof(D41HcMix)));
    dsp_->r = static_cast<uint16_t *>(host((size_t)STEP_MAX_ROWS * D41_DSP_RANK * 2));
    dsp_->words = static_cast<uint32_t *>(host(64));
    if (debug_.probes) {
      dsp_->probe_x = static_cast<float *>(host((size_t)STEP_MAX_ROWS * E * 4));
      dsp_->probe_kv = static_cast<float *>(host((size_t)D41_DSP_LAYERS * STEP_MAX_ROWS * D41_HD * 4));
      dsp_->probe_ids = static_cast<int32_t *>(host((size_t)D41_DSP_LAYERS * STEP_MAX_ROWS * D41_DSP_USED * 4));
    }
  }
  pairs_.resize(kAllLayers);
  split_.resize(kAllLayers);
  tables_.resize(kAllLayers);
  pt_.assign((size_t)opt_.slots * npp_, D41_NO_PAGE);

  for (uint32_t c = 0; c < 2; c++) {
    cards_[c] = std::make_unique<Card>();
    Card &card = *cards_[c];
    card.c = c;
    card.up = std::make_unique<ze::List>(*rt_, c, ze::List::immediate);
    card.grids = upload<uint8_t>(card, codebooks().data(), QGRID_BYTES);
    card.layers.resize(dsp_ ? kAllLayers : D41_LAYERS);
    alloc_buffers(card);
    if (opt_.prompt_chunk) alloc_chunk(card);
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
  // Before the model's experts, which take the room that is left.
  if (dsp_) load_drafter();
  load_experts();

  info_.cards = 2;
  info_.vocab = vocab_;
  for (uint32_t c = 0; c < 2; c++) info_.device_bytes[c] = rt_->device_bytes(c);
  info_.arena_anon = true;
  info_.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

Model::~Model() {
  if (engram_thread_.joinable()) {
    {
      std::lock_guard<std::mutex> lk(engram_mu_);
      engram_stop_ = true;
    }
    engram_cv_.notify_all();
    engram_thread_.join();
  }
  {
    std::lock_guard<std::mutex> lk(swap_mu_);
    swap_stop_ = true;
  }
  swap_cv_.notify_all();
  refill_cv_.notify_all();
  for (std::thread &t : loaders_) t.join();
  for (std::thread &t : refillers_) t.join();
}

// A step's Engram rows, a table at a time in the order its layers come: gathered, made a8 in host memory, then the
// table's flag, which the cards' lists wait for before they copy the rows.
void Model::engram_work() {
  uint64_t seen = 0;
  for (;;) {
    {
      std::unique_lock<std::mutex> lk(engram_mu_);
      engram_cv_.wait(lk, [&] { return engram_stop_ || engram_asked_ != seen; });
      if (engram_stop_) return;
      seen = engram_asked_;
    }
    const auto t0 = std::chrono::steady_clock::now();
    try {
      const uint32_t n = engram_n_;
      std::vector<uint64_t> ids((size_t)n * D41_ENG_COLS);
      for (uint32_t e = 0; e < kEngramTables; e++) {
        for (uint32_t r = 0; r < n; r++) {
          const StepIn &row = engram_rows_[r];
          const auto h = hash_->rows(row.token, row.p1, row.p2, row.p3, e);
          std::copy(h.begin(), h.end(), ids.begin() + (size_t)r * D41_ENG_COLS);
        }
        engram_->gather(ids.data(), (uint32_t)ids.size(), eng_f_.data());
        for (uint32_t r = 0; r < n; r++)
          a8_row(eng_f_.data() + (size_t)r * D41_ENG_G, D41_ENG_G, eng_a8_[e] + (size_t)r * kEngA8Row);
        std::atomic_ref<uint32_t>(eng_flag_[16 * e]).store(step_, std::memory_order_release);
      }
    } catch (...) {
      engram_failed_ = std::current_exception();
    }
    times_.engram = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    engram_done_.store(seen, std::memory_order_release);
  }
}

// Until the thread has finished the step's rows, or failed: the step's rows are the caller's.
void Model::engram_wait() {
  uint64_t asked;
  {
    std::lock_guard<std::mutex> lk(engram_mu_);
    asked = engram_asked_;
  }
  for (uint32_t k = 0; engram_done_.load(std::memory_order_acquire) != asked; k++) {
    if (k < 4096) {
      _mm_pause();
    } else {
      std::this_thread::yield();
    }
  }
}

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
  if (!dsp_) return;
  card.fa = dev<uint8_t>(card, R * D41_DSP_A8_ROW);
  card.dmain = dev<float>(card, R * E);
  card.dxa = dev<uint8_t>(card, R * D41_A8_ROW);
  card.dkv = dev<float>(card, R * D41_HD);
  if (debug_.probes) {
    card.dxf = dev<float>(card, R * E);
    card.dkvf = dev<float>(card, D41_DSP_LAYERS * R * D41_HD);
  }
}

// A drafter layer's tensors are numbered from 0 in its own file, which dsp_loading_ selects.
void Model::load_layer(uint32_t l) {
  const auto name = [&](const char *n) { return blk(drafter_layer(l) ? l - D41_LAYERS : l, n); };
  const uint32_t n_experts = experts_of(l);
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
    L.bias = upload_floats(card, name("exp_probs_b.bias"), {n_experts});
    L.q_a = slice("attn_q_a.weight", E, D41_QLR, 0, D41_QLR);
    L.kv = slice("attn_kv.weight", E, D41_HD, 0, D41_HD);
    L.q_b = slice("attn_q_b.weight", D41_QLR, D41_H * D41_HD, (uint64_t)card.c * kHeads * D41_HD, kHeads * D41_HD);
    L.out_a = slice("attn_output_a.weight", D41_OG_K, D41_OG * D41_OLR, (uint64_t)card.c * kGroups * D41_OLR,
                    kGroups * D41_OLR);
    og_module(L.out_a.type);
    L.out_b = slice("attn_output_b.weight", D41_OG * D41_OLR, E, 0, E, (uint64_t)card.c * kGroups * D41_OLR,
                    kGroups * D41_OLR);
    L.gate_inp = upload_bf16(card, name("ffn_gate_inp.weight"), {E, n_experts});
    {
      // The larger part of the shared expert on the card that does not post the layer's misses (ADR-0080 item 2).
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
    if (kv_source(l)) {
      const uint32_t ratio = ratio_of(l);
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
    if (idx_source(l)) {
      L.idx_q_b = upload_bf16(card, name("indexer.attn_q_b.weight"), {D41_QLR, D41_IDX_H * D41_IDX_D});
      L.idx_proj = upload_bf16(card, name("indexer.proj.weight"), {E, D41_IDX_H});
    }
    if (engram_of(l) >= 0) {
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

// The drafter's dense tensors and all of its experts onto the cards, and its token tables into host memory.
void Model::load_drafter() {
  Drafter &d = *dsp_;
  dsp_loading_ = true;
  if (d.gguf.get_string("general.architecture") != "dflash" || d.gguf.get_int("dflash.block_count") != D41_DSP_LAYERS)
    throw std::runtime_error(opt_.mtp_dir + ": not deepseek41's drafter");
  d.noise = (int32_t)d.gguf.get_int("tokenizer.ggml.mask_token_id");
  if (d.noise < 0 || (uint32_t)d.noise >= vocab_) throw std::runtime_error(opt_.mtp_dir + ": its noise token");
  {
    const GgufTensor &t = tensor("conf_proj.weight", {E + D41_DSP_RANK, 1});
    if (t.type != QT_F32) throw std::runtime_error(t.name + ": the engine reads it as F32 only");
    const float *f = reinterpret_cast<const float *>(held(t));
    d.conf.assign(f, f + E + D41_DSP_RANK);
  }
  const auto table = [&](const char *name, AlignedBuffer &buf) {
    const GgufTensor &t = tensor(name, {D41_DSP_RANK, vocab_});
    if (t.type != QT_BF16) throw std::runtime_error(t.name + ": the engine reads it as BF16 only");
    buf = aligned_buffer(DirectReader::span(t.offset, t.bytes));
    return reinterpret_cast<const uint16_t *>(d.reader.read(t.shard, t.offset, t.bytes, buf.get()));
  };
  d.w1 = table("markov_w1.weight", d.w1_buf);
  AlignedBuffer w2_buf;
  const uint16_t *w2 = table("markov_w2.weight", w2_buf);
  const GgufTensor &fc = tensor("fc.weight", {D41_DSP_FEATURES, E});
  for (auto &cp : cards_) {
    Card &card = *cp;
    card.w2 = upload<uint16_t>(card, w2 + (size_t)card.vocab0 * D41_DSP_RANK, (size_t)card.vocab_n * D41_DSP_RANK);
    card.rb = dev<uint16_t>(card, (size_t)STEP_MAX_ROWS * D41_DSP_RANK);
    card.bias = dev<float>(card, card.vocab_n);
    card.biased = dev<float>(card, card.vocab_n);
    card.pick = dev<uint32_t>(card, 16);
    card.fc = mat(card, fc.type, D41_DSP_FEATURES,
                  {{held(fc), fc.row_bytes(), 0, E, 0, D41_DSP_FEATURES / quant_block_values(fc.type)}});
    card.enc_norm = upload_floats(card, "enc.output_norm.weight", {E});
    card.draft_norm = upload_floats(card, "output_norm.weight", {E});
  }
  held_.clear();
  for (uint32_t il = 0; il < D41_DSP_LAYERS; il++) {
    const GgufTensor &g = tensor(blk(il, "ffn_gate_exps.weight"), {E, D41_FF, D41_DSP_EXPERTS});
    const GgufTensor &u = tensor(blk(il, "ffn_up_exps.weight"), {E, D41_FF, D41_DSP_EXPERTS});
    const GgufTensor &dn = tensor(blk(il, "ffn_down_exps.weight"), {D41_FF, E, D41_DSP_EXPERTS});
    if (g.type != u.type) throw std::runtime_error(u.name + ": not gate's type");
    LayerExperts le;
    le.shape.gate_type = g.type;
    le.shape.down_type = dn.type;
    le.shape.n_embd = E;
    le.shape.n_ff = D41_FF;
    le.shape.swiglu_limit = D41_SWIGLU_LIMIT;
    le.gate = {g.shard, g.offset};
    le.up = {u.shard, u.offset};
    le.down = {dn.shard, dn.offset};
    le.gate_bytes = g.bytes / D41_DSP_EXPERTS;
    le.down_bytes = dn.bytes / D41_DSP_EXPERTS;
    le.experts = D41_DSP_EXPERTS;
    d.experts.push_back(le);
  }
  std::vector<uint32_t> all(D41_DSP_EXPERTS);
  for (uint32_t e = 0; e < D41_DSP_EXPERTS; e++) all[e] = e;
  for (uint32_t l = D41_LAYERS; l < kAllLayers; l++) {
    load_layer(l);
    load_cache(l, all);
  }
  dsp_loading_ = false;
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
  if (opt_.adapt_swaps) {
    heat_.assign((size_t)D41_LAYERS * D41_EXPERTS, 0.0f);
    missed_.assign(D41_LAYERS, {});
    for (uint32_t l = 0; l < D41_LAYERS && !profile.empty(); l++) {
      uint64_t sum = 0;
      for (uint32_t e = 0; e < D41_EXPERTS; e++) sum += profile[(size_t)l * D41_EXPERTS + e];
      for (uint32_t e = 0; e < D41_EXPERTS && sum; e++)
        heat_[(size_t)l * D41_EXPERTS + e] =
            (float)profile[(size_t)l * D41_EXPERTS + e] / (float)sum * D41_USED * kProfileSteps;
    }
    uint64_t slot_bytes = 0;
    for (const auto &t : tables_) slot_bytes = std::max(slot_bytes, t[0][0].slot_bytes);
    for (auto &cp : cards_) {
      cp->swap_up = std::make_unique<ze::List>(*rt_, cp->c, ze::List::immediate, true);
      cp->swap_slot = static_cast<uint8_t *>(rt_->alloc_host(slot_bytes));
    }
    for (uint32_t c = 0; c < 2; c++) loaders_.emplace_back([this, c] { swap_load(c); });
    for (uint32_t i = 0; i < kRefillThreads; i++) refillers_.emplace_back([this] { swap_refill(); });
  }
  if (opt_.arena_bytes == 0) return;
  for (uint32_t l = 0; l < D41_LAYERS; l++) arena_->fill(ssd_->direct(), l, experts_[l], in_arena[l]);
  const ArenaReport r = arena_->settle();
  info_.arena_bytes = r.bytes;
}

// Layer l's experts ids on the cards: pair k's part A on one card and part B on the other, the cards taking turns so
// that both hold the same bytes.
void Model::load_cache(uint32_t l, const std::vector<uint32_t> &ids) {
  const LayerExperts &le = layer_experts(l);
  const uint32_t gt = le.shape.gate_type, dt = le.shape.down_type;
  const uint32_t n = (uint32_t)ids.size(), split = (n + 1) / 2;
  std::vector<int32_t> map(D41_EXPERTS, -1);
  for (uint32_t k = 0; k < n; k++) map[ids[k]] = (int32_t)k;
  pairs_[l] = ids;
  split_[l] = split;
  for (auto &cp : cards_) {
    Card &card = *cp;
    Layer &ly = card.layers[l];
    ly.slot_map = upload<int32_t>(card, map.data(), map.size());
    ly.flip = card.c ^ (l & 1);
    const uint64_t part_a = ly.flip == 0 ? split : n - split;
    auto &t = tables_[l][card.c];
    t[0] = slot_class(gt, dt, D41_PART_A, 0);
    t[1] = slot_class(gt, dt, D41_PART_B, part_a * t[0].slot_bytes);
    if (n == 0) continue;
    ly.split = split;
    ly.cache = dev<uint8_t>(card, part_a * t[0].slot_bytes + (n - part_a) * t[1].slot_bytes);
    ly.classes = upload<D41Class>(card, t.data(), 2);
  }
  if (n == 0) return;

  const uint64_t buf_bytes = std::max(le.gate_bytes, le.down_bytes) + 2 * DirectReader::kAlign;
  const uint64_t slot_bytes = tables_[l][0][0].slot_bytes;
  std::mutex mu;
  std::exception_ptr failed;
  DirectReader &from = drafter_layer(l) ? dsp_->reader : ssd_->direct();
  const DirectReader &reader = from;
  from.pool().run(n, [&](uint64_t k) {
    try {
      AlignedBuffer bufs[3] = {aligned_buffer(buf_bytes), aligned_buffer(buf_bytes), aligned_buffer(buf_bytes)};
      std::vector<uint8_t> slot(slot_bytes);
      const uint32_t e = ids[k];
      const uint8_t *g = reader.read_one(le.gate.file, le.gate.offset + e * le.gate_bytes, le.gate_bytes, bufs[0].get());
      const uint8_t *u = reader.read_one(le.up.file, le.up.offset + e * le.gate_bytes, le.gate_bytes, bufs[1].get());
      const uint8_t *d = reader.read_one(le.down.file, le.down.offset + e * le.down_bytes, le.down_bytes, bufs[2].get());
      for (uint32_t c = 0; c < 2; c++) {
        Card &card = *cards_[c];
        make_slot(l, c, (uint32_t)k, g, u, d, nullptr, slot.data());
        std::lock_guard<std::mutex> lk(card.up_mu);
        card.up->copy(slot_at(l, c, (uint32_t)k), slot.data(), pair_class(l, c, (uint32_t)k).slot_bytes);
        card.up->synchronize();
      }
    } catch (...) {
      std::lock_guard<std::mutex> lk(mu);
      if (!failed) failed = std::current_exception();
    }
  });
  if (failed) std::rethrow_exception(failed);
}

// Card c's slot of layer l's pair, from an expert's matrices: gate and up in the file's layout, and down in the file's
// layout, or, with down_b, as the arena holds it: part A's blocks of every row at down and part B's at down_b.
void Model::make_slot(uint32_t l, uint32_t c, uint32_t pair, const uint8_t *gate, const uint8_t *up, const uint8_t *down,
                      const uint8_t *down_b, uint8_t *slot) const {
  const LayerExperts &le = layer_experts(l);
  const uint32_t gt = le.shape.gate_type, dt = le.shape.down_type;
  const D41Class &t = pair_class(l, c, pair);
  const bool part_b = t.rows == D41_PART_B;
  const uint64_t dq = quant_block_values(dt);
  const uint64_t gu_blocks = E / quant_block_values(gt), rb = gu_blocks * quant_block_bytes(gt), bb = quant_block_bytes(dt);
  const uint64_t r0 = part_b ? D41_PART_A : 0, nb = t.rows / dq;
  to_runs(gt, {{gate, rb, r0, t.rows, 0, gu_blocks}}, slot);
  to_runs(gt, {{up, rb, r0, t.rows, 0, gu_blocks}}, slot + t.up_off);
  if (down_b) {
    to_runs(dt, {{part_b ? down_b : down, nb * bb, 0, E, 0, nb}}, slot + t.down_off);
  } else {
    to_runs(dt, {{down, (uint64_t)(D41_FF / dq) * bb, 0, E, r0 / dq, nb}}, slot + t.down_off);
  }
}

// The class of a pair's slot on card c: part A for a pair below the split on the card whose flip is 0, and so on.
const D41Class &Model::pair_class(uint32_t l, uint32_t c, uint32_t pair) const {
  return tables_[l][c][(uint32_t)(pair >= split_[l]) ^ c ^ (l & 1)];
}

uint8_t *Model::slot_at(uint32_t l, uint32_t c, uint32_t pair) const {
  const D41Class &t = pair_class(l, c, pair);
  return cards_[c]->layers[l].cache + t.base + (uint64_t)(pair < split_[l] ? pair : pair - split_[l]) * t.slot_bytes;
}

// The routers' word for an expert of layer l on both cards: its pair, or -1.
void Model::map_expert(uint32_t l, uint32_t expert, int32_t pair) {
  for (auto &cp : cards_) cp->up->copy(cp->layers[l].slot_map + expert, &pair, 4);
}

// Between steps: the arena places the refill threads wrote are mapped, and the experts the loaders wrote are mapped
// on the cards, which frees the arena places of those that came from it for the experts that left.
void Model::advance_swaps() {
  if (swap_state_.load(std::memory_order_acquire) == SwapState::failed) std::rethrow_exception(swap_failed_);
  std::vector<Refill> done;
  {
    std::lock_guard<std::mutex> lk(swap_mu_);
    done.swap(refill_done_);
  }
  for (const Refill &r : done) {
    arena_->put(r.l, experts_[r.l], r.out, r.place);
    refilling_.erase({r.l, r.out});
  }
  if (swap_state_.load(std::memory_order_acquire) != SwapState::loaded) return;
  const double mapped_at = now_ms();
  std::vector<Refill> queued;
  for (Swap &w : swaps_) {
    map_expert(w.l, w.in, (int32_t)w.pair);
    pairs_[w.l][w.pair] = w.in;
    w.place = arena_->take(w.l, w.in);
    // in came from the file: out takes the place of the arena's least used expert of the layer, if used less.
    const uint32_t cold = w.place ? kNoExpert : coldest_in_arena(w.l);
    const float *heat = heat_.data() + (size_t)w.l * D41_EXPERTS;
    if (cold != kNoExpert && heat[cold] < heat[w.out]) w.place = arena_->take(w.l, cold);
    if (w.place) queued.push_back({w.l, w.out, w.place, mapped_at});
  }
  for (auto &cp : cards_) cp->up->synchronize();
  swaps_done_ += swaps_.size();
  swap_stats_.batches++, swap_stats_.pairs += swaps_.size(), swap_stats_.steps_to_map += step_no_ - chosen_step_;
  swap_stats_.to_loaded += loaded_at_ - chosen_at_, swap_stats_.to_mapped += mapped_at - chosen_at_;
  swaps_.clear();
  for (const Refill &r : queued) refilling_.insert({r.l, r.out});
  {
    std::lock_guard<std::mutex> lk(swap_mu_);
    for (const Refill &r : queued) refill_queue_.push_back(r);
    swap_state_.store(SwapState::idle, std::memory_order_release);
  }
  refill_cv_.notify_all();
}

// An expert the file gave, or that left the cards, goes to an arena place in the background.
void Model::queue_refill(uint32_t l, uint32_t out, uint8_t *place) {
  refilling_.insert({l, out});
  {
    std::lock_guard<std::mutex> lk(swap_mu_);
    refill_queue_.push_back({l, out, place, now_ms()});
  }
  refill_cv_.notify_all();
}

SwapStats Model::swap_stats() const {
  std::lock_guard<std::mutex> lk(swap_mu_);
  return swap_stats_;
}

// The arena's least used expert of layer l, or kNoExpert.
uint32_t Model::coldest_in_arena(uint32_t l) const {
  const float *heat = heat_.data() + (size_t)l * D41_EXPERTS;
  uint32_t out = kNoExpert;
  for (uint32_t e = 0; e < D41_EXPERTS; e++)
    if (arena_->weights(l, e).gate && (out == kNoExpert || heat[e] < heat[out])) out = e;
  return out;
}

// After a step of n rows: the routings' counts, and with the loaders idle the next changes: the experts missed
// since the last choice that lead their layer's least used pairs, and the experts the file gave.
void Model::count_routing(uint32_t n) {
  step_no_++;
  heat_unit_ /= kHeatDecay;
  if (heat_unit_ > 1e6f) {
    for (float &h : heat_) h /= heat_unit_;
    heat_unit_ = 1.0f;
  }
  for (uint32_t l = 0; l < D41_LAYERS; l++)
    for (uint32_t r = 0; r < n; r++)
      for (int32_t e : jobs_[(size_t)l * STEP_MAX_ROWS + r].all)
        if (e >= 0) heat_[(size_t)l * D41_EXPERTS + (uint32_t)e] += heat_unit_;
  for (uint32_t l = 0; l < D41_LAYERS; l++)
    for (uint32_t r = 0; r < n; r++) {
      const D41MissJob &job = jobs_[(size_t)l * STEP_MAX_ROWS + r];
      for (uint32_t i = 0; i < job.nmiss && i < D41_USED; i++) {
        const auto e = (uint32_t)job.ids[i];
        if (std::find(missed_[l].begin(), missed_[l].end(), e) == missed_[l].end()) missed_[l].push_back(e);
        const std::pair<uint32_t, uint32_t> seen{l, e};
        if (!arena_->weights(l, e).gate && file_seen_.size() < kFileSeen &&
            std::find(file_seen_.begin(), file_seen_.end(), seen) == file_seen_.end())
          file_seen_.push_back(seen);
      }
    }
  // A queue of refills longer than two batches holds the next choice back.
  if (swap_state_.load(std::memory_order_acquire) != SwapState::idle || refilling_.size() > 2 * kSwapBatch) return;

  struct Lead {
    float by;
    Swap swap;
  };
  std::vector<Lead> leads;
  for (uint32_t l = 0; l < D41_LAYERS; l++) {
    const float *heat = heat_.data() + (size_t)l * D41_EXPERTS;
    // The experts missed since the last choice, most used first, against the layer's least used pairs.
    std::vector<uint32_t> in, out;
    for (uint32_t e : missed_[l])
      if (std::find(pairs_[l].begin(), pairs_[l].end(), e) == pairs_[l].end() && !refilling_.count({l, e})) in.push_back(e);
    missed_[l].clear();
    for (uint32_t k = 0; k < pairs_[l].size(); k++)
      if (pairs_[l][k] != kNoExpert) out.push_back(k);
    const size_t count = std::min({kSwapsALayer, in.size(), out.size()});
    std::partial_sort(in.begin(), in.begin() + (std::ptrdiff_t)count, in.end(), [&](uint32_t a, uint32_t b) { return heat[a] > heat[b]; });
    std::partial_sort(out.begin(), out.begin() + (std::ptrdiff_t)count, out.end(),
                      [&](uint32_t a, uint32_t b) { return heat[pairs_[l][a]] < heat[pairs_[l][b]]; });
    for (size_t i = 0; i < count; i++) {
      const uint32_t leaving = pairs_[l][out[i]];
      const float by = heat[in[i]] - kSwapLead * heat[leaving] - kSwapFloor * heat_unit_;
      if (by > 0.0f) leads.push_back({by, {l, out[i], in[i], leaving, nullptr}});
    }
  }
  std::sort(leads.begin(), leads.end(), [](const Lead &a, const Lead &b) { return a.by > b.by; });
  if (leads.size() > kSwapBatch) leads.resize(kSwapBatch);
  // An expert the file gave takes the arena place of its layer's least used expert there, once it has been used
  // more: a refill thread reads it in, so a later step that routes to it finds it in memory.
  uint32_t admitted = 0;
  for (const auto &[l, e] : file_seen_) {
    if (admitted == kSwapBatch) break;
    const float *heat = heat_.data() + (size_t)l * D41_EXPERTS;
    const bool coming = std::any_of(leads.begin(), leads.end(), [&, l = l, e = e](const Lead &x) { return x.swap.l == l && x.swap.in == e; });
    if (coming || arena_->weights(l, e).gate || refilling_.count({l, e}) ||
        std::find(pairs_[l].begin(), pairs_[l].end(), e) != pairs_[l].end())
      continue;
    const uint32_t out = coldest_in_arena(l);
    if (out == kNoExpert || heat[e] <= heat[out]) continue;
    queue_refill(l, e, arena_->take(l, out));
    swaps_done_++;
    admitted++;
  }
  file_seen_.clear();
  if (leads.empty()) return;
  // The pairs' experts leave the cards now: their slots are written while the next steps run.
  for (const Lead &lead : leads) {
    const Swap &w = lead.swap;
    map_expert(w.l, w.out, -1);
    pairs_[w.l][w.pair] = kNoExpert;
    swaps_.push_back(w);
  }
  for (auto &cp : cards_) cp->up->synchronize();
  chosen_step_ = step_no_, chosen_at_ = now_ms();
  {
    std::lock_guard<std::mutex> lk(swap_mu_);
    load_gen_++;
    loading_left_ = (uint32_t)loaders_.size();
    swap_state_.store(SwapState::loading, std::memory_order_release);
  }
  swap_cv_.notify_all();
}

// A loader: writes its card's half of each chosen pair's slot, from the arena or the file.
void Model::swap_load(uint32_t c) {
  uint64_t gate_bytes = 0, down_bytes = 0;
  for (const LayerExperts &le : experts_) gate_bytes = std::max(gate_bytes, le.gate_bytes), down_bytes = std::max(down_bytes, le.down_bytes);
  AlignedBuffer bufs[3] = {aligned_buffer(gate_bytes + 2 * DirectReader::kAlign), aligned_buffer(gate_bytes + 2 * DirectReader::kAlign),
                           aligned_buffer(down_bytes + 2 * DirectReader::kAlign)};
  const DirectReader &reader = ssd_->direct();
  Card &card = *cards_[c];
  uint64_t seen = 0;
  for (;;) {
    {
      std::unique_lock<std::mutex> lk(swap_mu_);
      swap_cv_.wait(lk, [&] { return swap_stop_ || load_gen_ != seen; });
      if (swap_stop_) return;
      seen = load_gen_;
    }
    double make = 0, upload = 0;
    std::exception_ptr error;
    try {
      for (const Swap &w : swaps_) {
        const LayerExperts &le = experts_[w.l];
        const uint8_t *m[3];
        const ExpertWeights held = arena_->weights(w.l, w.in);
        const uint8_t *down_b = nullptr;
        if (held.gate) {
          m[0] = held.gate, m[1] = held.up, m[2] = held.down;
          down_b = held.down + (uint64_t)E * kPartABlocks * quant_block_bytes(le.shape.down_type);
        } else {
          m[0] = reader.read_one(le.gate.file, le.gate.offset + w.in * le.gate_bytes, le.gate_bytes, bufs[0].get());
          m[1] = reader.read_one(le.up.file, le.up.offset + w.in * le.gate_bytes, le.gate_bytes, bufs[1].get());
          m[2] = reader.read_one(le.down.file, le.down.offset + w.in * le.down_bytes, le.down_bytes, bufs[2].get());
        }
        const double t0 = now_ms();
        make_slot(w.l, c, w.pair, m[0], m[1], m[2], down_b, card.swap_slot);
        const double t1 = now_ms();
        card.swap_up->copy(slot_at(w.l, c, w.pair), card.swap_slot, pair_class(w.l, c, w.pair).slot_bytes);
        card.swap_up->synchronize();
        make += t1 - t0, upload += now_ms() - t1;
      }
    } catch (...) {
      error = std::current_exception();
    }
    std::lock_guard<std::mutex> lk(swap_mu_);
    swap_stats_.make += make, swap_stats_.upload += upload;
    if (error) {
      swap_failed_ = error;
      swap_state_.store(SwapState::failed, std::memory_order_release);
    }
    if (--loading_left_ == 0 && swap_state_.load(std::memory_order_acquire) == SwapState::loading) {
      loaded_at_ = now_ms();
      swap_state_.store(SwapState::loaded, std::memory_order_release);
    }
  }
}

// A refill thread: reads an expert from the file and writes it into its arena place.
void Model::swap_refill() {
  uint64_t gate_bytes = 0, down_bytes = 0;
  for (const LayerExperts &le : experts_) gate_bytes = std::max(gate_bytes, le.gate_bytes), down_bytes = std::max(down_bytes, le.down_bytes);
  AlignedBuffer bufs[3] = {aligned_buffer(gate_bytes + 2 * DirectReader::kAlign), aligned_buffer(gate_bytes + 2 * DirectReader::kAlign),
                           aligned_buffer(down_bytes + 2 * DirectReader::kAlign)};
  const DirectReader &reader = ssd_->direct();
  for (;;) {
    Refill r;
    {
      std::unique_lock<std::mutex> lk(swap_mu_);
      refill_cv_.wait(lk, [&] { return swap_stop_ || !refill_queue_.empty(); });
      if (swap_stop_) return;
      r = refill_queue_.front();
      refill_queue_.pop_front();
    }
    double read = 0, write = 0;
    try {
      const LayerExperts &le = experts_[r.l];
      const double t0 = now_ms();
      const uint8_t *m[3];
      m[0] = reader.read_one(le.gate.file, le.gate.offset + r.out * le.gate_bytes, le.gate_bytes, bufs[0].get());
      m[1] = reader.read_one(le.up.file, le.up.offset + r.out * le.gate_bytes, le.gate_bytes, bufs[1].get());
      m[2] = reader.read_one(le.down.file, le.down.offset + r.out * le.down_bytes, le.down_bytes, bufs[2].get());
      const double t1 = now_ms();
      ExpertArena::write(le, m[0], m[1], m[2], r.place);
      read = t1 - t0, write = now_ms() - t1;
    } catch (...) {
      std::lock_guard<std::mutex> lk(swap_mu_);
      swap_failed_ = std::current_exception();
      swap_state_.store(SwapState::failed, std::memory_order_release);
      continue;
    }
    std::lock_guard<std::mutex> lk(swap_mu_);
    refill_done_.push_back(r);
    swap_stats_.refills++, swap_stats_.refill_read += read, swap_stats_.refill_write += write;
    swap_stats_.refill_wait += now_ms() - r.queued;
  }
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
  const uint32_t e = (uint32_t)engram_of(l), k = 2 * kAllLayers + e;
  const uint32_t c0 = 2 * card.c, c1 = c0 + 2, v0 = card.c * (E / 2), v1 = v0 + E / 2, stride = kEngKvRows;
  const uint32_t in_card = 0;
  // The host gathers the table's rows while the layers before run.
  wait(L, card, "wait: Engram rows", nullptr, eng_flag_ + 16 * e);
  L.copy(card.eng_a8[e], eng_a8_[e], (size_t)W * kEngA8Row);
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
  const uint32_t ratio = ratio_of(l), yarn = ratio != 0, first_head = card.c * kHeads;
  const bool kv_src = kv_source(l), idx_src = idx_source(l);
  // A window-only layer reads no compressed row; it is given layer 2's pool so that the pointer is one it may read.
  uint16_t *pool = card.layers[ratio ? (uint32_t)D41_KV_SRC((int)l) : 2].pool;
  // A drafter layer's rows are one block: each sees the window before it and the block's rows.
  const void *block = drafter_layer(l) ? &card.plan->real : nullptr;
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
  L.launch(attn("ds_attn_split"), {card.q, rows, ratio, ly.ring, card.pt, npp_, pool, card.idx, card.part, block},
           kHeads / D41_ATT_HEADS, W, drafter_layer(l) ? 2 : ratio ? D41_ATT_SPLITS : 1, D41_ATT_WG);
  L.launch(attn("ds_attn_merge"), {card.part, ly.sinks, first_head, rows, ratio, card.oa, none, block}, kHeads, W, 1,
           256);
  L.launch(kernel(card, og_module(ly.out_a.type), "ds_output_groups"), {ly.out_a.p, ly.out_a.bpr, card.oa, W, card.aa, none},
           D41_OLR / D41_OG_WG_ROWS, kGroups, 1, D41_OG_WG_ROWS / (16 / D41_OG_LANES) * 16);
  matvec(L, card, "attn_output_b", ly.out_b, card.aa, card.y, W, 8);

  const uint32_t k = 2 * l, in_card = 0;
  L.launch(kernel(card, "deepseek41", "ds_exch_send"), {card.y, other.in[0]}, W, 1, 1, D41_EX_SGS * 16);
  wait(L, card, "wait: other card, attention", flag(card.c, k), flag(other.c, k));
  // The drafter's experts are all on the cards: its layers post nothing to the CPU.
  const void *job = !drafter_layer(l) && l % 2 == card.c ? jobs_ + (size_t)l * STEP_MAX_ROWS : nullptr;
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
  const bool posts = !drafter_layer(l) && l % 2 == card.c;
  const uint32_t ff = ly.sh_ff, two_ff = 2 * ff, in_card = 0;

  L.launch(kernel(card, "deepseek41", "ds_hc_mix"), {ly.ffn_fn, card.X1, card.ss, W, card.m}, D41_MIX, 1, 1,
           16 * D41_HC_KSPLIT);
  L.launch(kernel(card, "deepseek41", "ds_hc_sinkhorn"), {card.m, ly.ffn_scale, ly.ffn_base, card.fmix}, W, 1, 1, 16);

  product_bf16(L, card, "router product", ly.gate_inp, experts_of(l), E, card.xb, card.rlogits);
  const void *jobs = posts ? jobs_ + (size_t)l * STEP_MAX_ROWS : nullptr;
  L.launch(kernel(card, layer_module(l), "ds_router"),
           {card.rlogits, ly.bias, ly.slot_map, W, ly.split, ly.flip, &card.plan->real, &card.plan->step, card.ids, card.w,
            card.slots, card.nmiss, jobs, card.groups, card.ngroups},
           1, 1, 1, 16 * W * D41_RT_SGS(W));
  if (drafter_layer(l) && dsp_->probe_ids && card.c == 0)
    L.copy(dsp_->probe_ids + (size_t)(l - D41_LAYERS) * STEP_MAX_ROWS * D41_DSP_USED, card.ids,
           (size_t)W * D41_DSP_USED * 4);

  if (ly.cache) {
    const LayerExperts &le = layer_experts(l);
    const uint32_t per = used_of(l), xs = D41_A8_ROW, hs = D41_PART_A + D41_PART_A / 8, nasg = W * per;
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
  L.launch(kernel(card, layer_module(l), "ds_moe_combine"),
           {card.y6, card.w, card.slots, card.shared, cpu, card.nmiss, card.out, other.in[1]}, W, 1, 1,
           D41_CMB_SGS(W) * 16);
  const uint32_t k = 2 * l + 1;
  wait(L, card, "wait: other card, MoE", flag(card.c, k), flag(other.c, k));

  // Before an Engram layer only the streams are made: its injection comes before the collapse.
  const bool last = l + 1 == D41_LAYERS || l + 1 == kAllLayers, engram = !last && engram_of(l + 1) >= 0;
  const void *norm = engram ? nullptr : !last ? card.layers[l + 1].attn_norm : drafter_layer(l) ? card.draft_norm : card.out_norm;
  const void *xa = engram ? nullptr : card.xa, *xb = engram ? nullptr : card.xb;
  L.launch(kernel(card, "deepseek41", "ds_hc_post_norm"),
           {card.X1, card.out, card.in[1], in_card, card.fmix, norm, card.X, card.ss, xa, xb, none, none}, W, 1, 1,
           D41_HC_SGS * 16);
  if (probe_ && card.c == 0)
    L.copy(probe_ + ((size_t)kAllLayers + l) * STEP_MAX_ROWS * D, card.X, (size_t)W * D * 4);
}

// The step's rows into the drafter's keys, from their features in fa: the drafter's input, then each of its layers' key
// at the row's position into that layer's window. A row the model later drops is stored again when its position runs.
void Model::record_feed(ze::List &L, Card &card, uint32_t W) {
  const uint32_t yarn = 0;
  D41Row *rows = card.plan->rows;
  matvec(L, card, "drafter input", card.fc, card.fa, card.dmain, W, 8);
  L.launch(kernel(card, "deepseek41", "ds_dsp_input"), {card.dmain, card.enc_norm, card.dxa, card.dxf}, W, 1, 1,
           D41_HC_SGS * 16);
  for (uint32_t il = 0; il < D41_DSP_LAYERS; il++) {
    Layer &ly = card.layers[D41_LAYERS + il];
    const void *kvf = card.dkvf ? card.dkvf + (size_t)il * STEP_MAX_ROWS * D41_HD : nullptr;
    matvec(L, card, "drafter key", ly.kv, card.dxa, card.dkv, W, 8, 4);
    L.launch(kernel(card, "deepseek41_attn", "ds_qkv_norm"),
             {none, none, card.dkv, ly.kv_norm, rows, yarn, none, none, none, ly.ring, kvf}, 2, W, 1,
             D41_QKV_SGS * 16);
  }
  if (dsp_->probe_x && card.c == 0) {
    L.copy(dsp_->probe_x, card.dxf, (size_t)W * E * 4);
    L.copy(dsp_->probe_kv, card.dkvf, (size_t)D41_DSP_LAYERS * STEP_MAX_ROWS * D41_HD * 4);
  }
}

// A draft block's list: the rows are the block, through the drafter's layers. The keep chance reads what it leaves.
void Model::record_draft(Card &card, uint32_t W) {
  auto list = std::make_unique<ze::List>(*rt_, card.c, ze::List::regular, false, false);
  ze::List &L = *list;
  if (opt_.trace) {
    constexpr size_t cap = 4096;
    card.draft_trace_ts[W] = static_cast<uint64_t *>(rt_->alloc_host(cap * 8));
    L.trace(card.draft_trace_ts[W], cap);
  }
  L.copy(card.plan, plan_, sizeof(StepWords));
  L.copy(card.e, embed_, (size_t)W * E * 4);
  L.launch(kernel(card, "deepseek41", "ds_hc_init"),
           {card.e, card.layers[D41_LAYERS].attn_norm, card.X, card.ss, card.xa, card.xb, none}, W, 1, 1,
           D41_HC_SGS * 16);
  for (uint32_t l = D41_LAYERS; l < kAllLayers; l++) {
    record_attention(L, card, l, W);
    record_ffn(L, card, l, W);
  }
  // The keep chance reads a row's streams before the output norm.
  if (card.c == 0) {
    L.copy(dsp_->x, card.X, (size_t)W * D * 4);
    L.copy(dsp_->mix, card.fmix, (size_t)W * sizeof(D41HcMix));
  }
  L.copy(stuck_ + 16 * card.c, &card.plan->stuck, 4);
  L.close();
  card.draft_lists[W] = std::move(list);
}

// The model's output head over the last block's rows, and the first draft's candidates: run only when the first
// draft's keep chance says a draft is worth choosing.
void Model::record_head(Card &card, uint32_t W) {
  auto list = std::make_unique<ze::List>(*rt_, card.c, ze::List::regular, false, false);
  ze::List &L = *list;
  if (opt_.trace) {
    constexpr size_t cap = 256;
    card.head_trace_ts[W] = static_cast<uint64_t *>(rt_->alloc_host(cap * 8));
    L.trace(card.head_trace_ts[W], cap);
  }
  matvec(L, card, "output", card.head, card.xa, card.logits, W, 4);
  record_pick(L, card);
  L.close();
  card.head_lists[W] = std::move(list);
}

// One draft's candidates, into the host's first row: the logits of the row the host named plus the bias of the token
// before it, whose row of markov_w1 the host gave, over the card's part of the vocabulary.
void Model::record_pick(ze::List &L, Card &card) {
  const uint32_t zero = 0, ks = 4, rank = D41_DSP_RANK;
  const float one = 1.0f;
  L.copy(card.pick, dsp_->words, 64);
  L.copy(card.rb, dsp_->r, (size_t)D41_DSP_RANK * 2);
  L.label("draft bias");
  L.launch(kernel(card, "common", "mv_xmx_bf16"),
           {card.w2, card.vocab_n, rank, rank, card.rb, card.pick + 1, card.bias, card.vocab_n, zero, one, zero, ks, none},
           (card.vocab_n + 15) / 16, 1, 1, 16 * ks);
  L.launch(kernel(card, "deepseek41", "ds_dsp_bias_add"), {card.logits, card.vocab_n, card.bias, card.pick, card.biased},
           (card.vocab_n + 255) / 256, 1, 1, 256);
  L.launch(kernel(card, "common", "topk_rows"),
           {card.biased, card.vocab_n, card.vocab_n, opt_.candidates, card.vocab0, card.pick + 1,
            top_idx_ + (size_t)card.c * STEP_MAX_ROWS * opt_.candidates,
            top_val_ + (size_t)card.c * STEP_MAX_ROWS * opt_.candidates, top_n_ + (size_t)card.c * STEP_MAX_ROWS},
           1, 1, 1, kTopkLocal);
}

// What the cards read to choose draft row: its number and the row of the token before it.
void Model::set_pick(uint32_t row, int32_t prev) {
  dsp_->words[0] = row;
  dsp_->words[1] = 1;
  std::memcpy(dsp_->r, dsp_->w1 + (size_t)prev * D41_DSP_RANK, (size_t)D41_DSP_RANK * 2);
}

// The candidates the cards' last choice left, largest first.
std::vector<Candidate> Model::picked() const {
  std::vector<Candidate> o;
  for (uint32_t c = 0; c < 2; c++) {
    const size_t base = (size_t)c * STEP_MAX_ROWS * opt_.candidates;
    for (uint32_t i = 0; i < top_n_[c * STEP_MAX_ROWS]; i++) o.push_back({(int32_t)top_idx_[base + i], top_val_[base + i]});
  }
  std::sort(o.begin(), o.end(), [](const Candidate &x, const Candidate &y) {
    return x.logit != y.logit ? x.logit > y.logit : x.token < y.token;
  });
  if (o.size() > opt_.candidates) o.resize(opt_.candidates);
  return o;
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
  L.launch(kernel(card, "deepseek41", "ds_hc_init"),
           {card.e, card.layers[0].attn_norm, card.X, card.ss, card.xa, card.xb, none}, W, 1, 1, D41_HC_SGS * 16);
  for (uint32_t l = 0; l < D41_LAYERS; l++) {
    if (engram_of(l) >= 0) record_engram(L, card, l, W);
    // The drafter reads the streams as these layers do; with the last of them every row's keys are stored.
    if (dsp_ && l >= D41_DSP_TARGET0 && l < D41_DSP_TARGET0 + D41_DSP_TARGETS) {
      L.launch(kernel(card, "deepseek41", "ds_dsp_feature"), {card.X, l - D41_DSP_TARGET0, card.fa}, W, 1, 1,
               D41_HC_SGS * 16);
      if (l + 1 == D41_DSP_TARGET0 + D41_DSP_TARGETS) record_feed(L, card, W);
    }
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
  if (opt_.adapt_swaps) advance_swaps();
  using Clock = std::chrono::steady_clock;
  const auto ms = [](Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
  const auto t_start = Clock::now();
  times_ = {};
  // The Engram rows' thread starts now; whatever ends the step waits for it, since it reads the caller's rows.
  struct EngramDone {
    Model &m;
    ~EngramDone() { m.engram_wait(); }
  } engram_done{*this};
  plan_->step = ++step_;
  {
    std::lock_guard<std::mutex> lk(engram_mu_);
    engram_rows_ = rows;
    engram_n_ = n;
    engram_asked_++;
  }
  engram_cv_.notify_one();

  last_width_ = W;
  last_draft_ = false;
  plan_->real = n;
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
      times_.layer_records[l] = (float)ms(t_mark, t_seen);
      times_.records += times_.layer_records[l];
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
      times_.layer_answers[l] = (float)ms(t_seen, t_mark);
      times_.answers += times_.layer_answers[l];
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
  // The lists end by themselves: a wait the CPU never answers gives up (ADR-0080 item 5).
  try {
    for (auto &card : cards_) card->lists.at(W)->synchronize();
    times_.after = ms(t_mark, Clock::now());
    engram_wait();
    if (engram_failed_) std::rethrow_exception(std::exchange(engram_failed_, nullptr));
    if (failed) std::rethrow_exception(failed);
  } catch (const std::exception &e) {
    throw std::runtime_error(e.what() + progress());
  }
  for (uint32_t c = 0; c < 2; c++)
    if (stuck_[16 * c]) throw std::runtime_error("card " + std::to_string(c) + " gave up a wait of the step" + progress());

  if (opt_.adapt_swaps) count_routing(n);

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
  const auto &stamps = last_draft_ ? card.draft_trace_ts : card.trace_ts;
  const auto ts = stamps.find(last_width_);
  if (ts == stamps.end()) throw std::runtime_error("the last step was not traced");
  const ze::Card &dev = rt_->cards()[c];
  std::vector<TraceRow> rows;
  const auto add = [&](const ze::List &list, const uint64_t *stamp) {
    const std::vector<std::string> &names = list.trace_names();
    for (size_t i = 0; i < names.size(); i++) {
      auto row = std::find_if(rows.begin(), rows.end(), [&](const TraceRow &r) { return r.kernel == names[i]; });
      if (row == rows.end()) row = rows.insert(rows.end(), TraceRow{names[i], 0, 0});
      row->launches++;
      row->ms += (double)((stamp[i + 1] - stamp[i]) & dev.timestamp_mask) * 1e3 / (double)dev.timer_hz;
    }
  };
  add(*(last_draft_ ? card.draft_lists : card.lists).at(last_width_), ts->second);
  // A block whose drafts were chosen ran the head's list too.
  if (last_draft_ && dsp_->head_ran) add(*card.head_lists.at(last_width_), card.head_trace_ts.at(last_width_));
  return rows;
}

const float *Model::probe(uint32_t layer, bool ffn) const {
  if (!probe_ || layer >= kAllLayers) throw std::runtime_error("no such probe");
  return probe_ + ((size_t)(ffn ? kAllLayers : 0) + layer) * STEP_MAX_ROWS * D;
}

void Model::set_images(uint32_t, const Image *, uint32_t n) {
  if (n) throw std::runtime_error("deepseek41 reads no images");
}

std::vector<std::vector<Candidate>> Model::mtp_catchup(const StepIn *, uint32_t, const std::vector<uint32_t> &,
                                                       const uint32_t *) {
  throw std::runtime_error("deepseek41 drafts by blocks");
}
std::vector<Candidate> Model::mtp_draft(uint32_t, uint32_t, int32_t, uint32_t) {
  throw std::runtime_error("deepseek41 drafts by blocks");
}
std::vector<std::vector<Candidate>> Model::mtp_drafts(const StepIn *, uint32_t, const uint32_t *) {
  throw std::runtime_error("deepseek41 drafts by blocks");
}
std::vector<std::vector<Candidate>> Model::mtp_catchup_prompt(const StepIn *, uint32_t, uint32_t,
                                                              const std::vector<uint32_t> &) {
  throw std::runtime_error("deepseek41 drafts by blocks");
}
std::vector<std::vector<std::vector<Candidate>>> Model::df_pass(const StepIn *, uint32_t, const DraftBlock *blocks,
                                                                uint32_t nblocks) {
  if (!dsp_) throw std::runtime_error("deepseek41 was loaded without its drafter");
  if (nblocks == 0) return {};
  if (nblocks > 1) throw std::runtime_error("deepseek41 drafts one block a pass");
  const DraftBlock &b = blocks[0];
  if (b.n == 0 || b.n > D41_DSP_BLOCK) throw std::runtime_error("a draft block holds 1 to 5 rows");
  if (b.slot >= opt_.slots) throw std::runtime_error("slot out of range");
  if (b.token < 0 || (uint32_t)b.token >= vocab_) throw std::runtime_error("a token outside the vocabulary");
  const uint32_t n = b.n, W = width(n);
  for (auto &card : cards_)
    if (!card->draft_lists.count(W)) record_draft(*card, W);
  last_width_ = W;
  last_draft_ = true;
  plan_->step = ++step_;
  plan_->real = n;
  plan_->stuck = 0;
  plan_->start = 0;
  // The block's token, then the drafter's noise token at the positions after it.
  const GgufTensor &te = gguf_.tensor("token_embd.weight");
  const auto to_float = ggml_get_type_traits((ggml_type)te.type)->to_float;
  for (uint32_t r = 0; r < W; r++) {
    if (r >= n) {
      plan_->rows[r] = {D41_NO_ROW, 0, 0, D41_NO_ROW};
      std::memset(embed_ + (size_t)r * E, 0, E * 4);
      continue;
    }
    plan_->rows[r] = {b.slot, b.pos + r, (b.pos + r) % D41_RING, D41_NO_ROW};
    to_float(te.data + (uint64_t)(r == 0 ? b.token : dsp_->noise) * te.row_bytes(), embed_ + (size_t)r * E, E);
  }
  for (auto &card : cards_) card->draft_lists.at(W)->execute();
  for (auto &card : cards_) card->draft_lists.at(W)->synchronize();
  for (uint32_t c = 0; c < 2; c++)
    if (stuck_[16 * c]) throw std::runtime_error("card " + std::to_string(c) + " gave up a wait of the draft block");

  Drafter &d = *dsp_;
  d.n = n;
  d.width = W;
  d.token = b.token;
  d.head_ran = false;
  d.out.resize((size_t)n * E);
  for (uint32_t r = 0; r < n; r++) {
    const float *X = d.x + (size_t)r * D, *pre = d.mix[r].pre;
    float *out = d.out.data() + (size_t)r * E;
    for (uint32_t i = 0; i < E; i++)
      out[i] = pre[0] * X[i] + pre[1] * X[E + i] + pre[2] * X[2 * E + i] + pre[3] * X[3 * E + i];
  }
  return {{}};
}

std::vector<CardDraft> Model::drafts(const DraftPick &pick, float min_keep, uint32_t most) {
  if (!dsp_ || dsp_->n == 0) throw std::runtime_error("deepseek41 has drafted no block");
  Drafter &d = *dsp_;
  const auto bf16 = [](uint16_t v) {
    const uint32_t b = (uint32_t)v << 16;
    float f;
    std::memcpy(&f, &b, 4);
    return f;
  };
  std::vector<CardDraft> out;
  int32_t prev = d.token;
  for (uint32_t r = 0; r < std::min(d.n, most); r++) {
    CardDraft o;
    const float *h = d.out.data() + (size_t)r * E;
    const uint16_t *w1 = d.w1 + (size_t)prev * D41_DSP_RANK;
    double sum = 0;
    for (uint32_t i = 0; i < E; i++) sum += (double)d.conf[i] * h[i];
    for (uint32_t k = 0; k < D41_DSP_RANK; k++) sum += (double)d.conf[E + k] * bf16(w1[k]);
    o.keep = (float)(1.0 / (1.0 + std::exp(-sum)));
    if (o.keep < min_keep) break;
    set_pick(r, prev);
    // The first draft chosen runs the head over every row; a later one adds its bias to its row's logits.
    const bool head = r == 0;
    for (auto &card : cards_) {
      if (head && !card->head_lists.count(d.width)) record_head(*card, d.width);
      if (!head && !card->pick_list) {
        card->pick_list = std::make_unique<ze::List>(*rt_, card->c, ze::List::regular, false, false);
        record_pick(*card->pick_list, *card);
        card->pick_list->close();
      }
      (head ? card->head_lists.at(d.width) : card->pick_list)->execute();
    }
    for (auto &card : cards_) (head ? card->head_lists.at(d.width) : card->pick_list)->synchronize();
    d.head_ran = true;
    o.candidates = picked();
    if (o.candidates.empty()) throw std::runtime_error("a draft row has no candidate");
    o.token = pick ? pick(r, o.candidates) : o.candidates.front().token;
    if (o.token < 0 || (uint32_t)o.token >= vocab_) throw std::runtime_error("a chosen draft outside the vocabulary");
    prev = o.token;
    out.push_back(std::move(o));
  }
  return out;
}

int32_t Model::noise_token() const {
  if (!dsp_) throw std::runtime_error("deepseek41 was loaded without its drafter");
  return dsp_->noise;
}
const float *Model::draft_input() const {
  if (!dsp_ || !dsp_->probe_x) throw std::runtime_error("no such probe");
  return dsp_->probe_x;
}
const float *Model::draft_key(uint32_t layer) const {
  if (!dsp_ || !dsp_->probe_kv || layer >= D41_DSP_LAYERS) throw std::runtime_error("no such probe");
  return dsp_->probe_kv + (size_t)layer * STEP_MAX_ROWS * D41_HD;
}
const int32_t *Model::draft_route(uint32_t layer, uint32_t row) const {
  if (!dsp_ || !dsp_->probe_ids || layer >= D41_DSP_LAYERS || row >= dsp_->n) throw std::runtime_error("no such probe");
  return dsp_->probe_ids + ((size_t)layer * STEP_MAX_ROWS + row) * D41_DSP_USED;
}
const float *Model::draft_out(uint32_t row) const {
  if (!dsp_ || row >= dsp_->n) throw std::runtime_error("no such draft row");
  return dsp_->out.data() + (size_t)row * E;
}

// The drafter is fed by the decode steps (they store each row's keys): a chunk's last rows run as steps (prompt.cpp), so the
// context rows have nothing left to do.
void Model::df_context_prompt(const StepIn *, uint32_t, uint32_t) {}

// A slot's state outside the pages: every layer's window, then each ratio-2 source's compressor ring. Both cards hold
// the same bits, so card 0's are saved.
uint64_t Model::slot_state_bytes() {
  uint64_t n = (uint64_t)cards_[0]->layers.size() * D41_RING * D41_HD * 2;
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
