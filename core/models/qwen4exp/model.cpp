// qwen4exp's loading, its arena of uncached experts, and the decode step (model.h).
#include "model_impl.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <immintrin.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <sys/mman.h>
#include <thread>

#include "ggml.h"
#include "layout.h"
#include "load/runs.h"
#include "quant.h"

namespace engine::qwen4exp {

using namespace detail;

namespace {

// A routing profile (tools/route_profile.cpp): lines "layer expert count", '#' starting a comment line.
std::vector<uint64_t> read_routing_profile(const std::string &path) {
  std::vector<uint64_t> v((size_t)Q_LAYERS * Q_EXPERTS, 0);
  FILE *f = std::fopen(path.c_str(), "r");
  if (!f) throw std::runtime_error("cannot read the routing profile " + path);
  char line[256];
  while (std::fgets(line, sizeof line, f)) {
    if (line[0] == '#') continue;
    unsigned l, e;
    unsigned long long c;
    if (std::sscanf(line, "%u %u %llu", &l, &e, &c) == 3 && l < Q_LAYERS && e < Q_EXPERTS) v[(size_t)l * Q_EXPERTS + e] = c;
  }
  std::fclose(f);
  return v;
}

}  // namespace

ze_kernel_handle_t Model::kernel(uint32_t card, const std::string &module, const std::string &name) {
  return lib_->get(card, module, name);
}

uint32_t Model::ncards_of(const Card &card) { return card.C; }

uint32_t Model::ev(uint32_t layer, uint32_t which) const { return layer * (2 * ncards_ + 2) + which; }
uint32_t Model::ngram_ev() const { return Q_LAYERS * (2 * ncards_ + 2); }
uint32_t Model::idx_ev(uint32_t attn, uint32_t c) const { return Q_LAYERS * (2 * ncards_ + 2) + 1 + attn * ncards_ + c; }

template <class T>
T *Model::upload(Card &card, const void *src, size_t bytes) {
  auto *p = static_cast<T *>(rt_->alloc_device(card.c, bytes));
  if (src) {
    card.up->copy(p, src, bytes);
    card.up->synchronize();
  }
  return p;
}
// Also used by prompt.cpp and mtp.cpp.
template int32_t *Model::upload<int32_t>(Card &, const void *, size_t);
template uint32_t *Model::upload<uint32_t>(Card &, const void *, size_t);
template uint16_t *Model::upload<uint16_t>(Card &, const void *, size_t);
template uint8_t *Model::upload<uint8_t>(Card &, const void *, size_t);
template float *Model::upload<float>(Card &, const void *, size_t);
template StepPlan *Model::upload<StepPlan>(Card &, const void *, size_t);

// A card's share of a weight: the given row ranges, and within each row the given ranges of blocks (all of the row
// when empty), in that order, in the runs layout (raw rows for BF16, F16 and F32).
Model::Mat Model::mat(Card &card, const GgufTensor &t, const Ranges &rows, const Ranges &cols) {
  Mat m;
  m.type = t.type;
  uint64_t qk = quant_block_values(t.type);
  uint64_t bpr_full = t.shape[0] / qk;
  Ranges bc = cols.empty() ? Ranges{{0, bpr_full}} : cols;
  uint64_t nb = 0, nr = 0;
  for (auto &r : bc) nb += r.second;
  for (auto &r : rows) nr += r.second;
  m.rows = (uint32_t)nr;
  m.bpr = (uint32_t)nb;
  m.K = (uint32_t)(nb * qk);
  std::vector<uint8_t> buf(nr * nb * quant_block_bytes(t.type));
  if (is_raw(t.type)) {
    uint8_t *dst = buf.data();
    uint32_t bb = quant_block_bytes(t.type);
    for (auto &rr : rows)
      for (uint64_t r = rr.first; r < rr.first + rr.second; r++)
        for (auto &c : bc) {
          std::memcpy(dst, t.data + r * t.row_bytes() + c.first * bb, c.second * bb);
          dst += c.second * bb;
        }
  } else {
    std::vector<SliceSpec> specs;
    if (cols.empty()) {
      for (auto &rr : rows) specs.push_back({t.data, t.row_bytes(), rr.first, rr.second, 0, bpr_full});
    } else {
      for (auto &rr : rows)
        for (uint64_t r = rr.first; r < rr.first + rr.second; r++)
          for (auto &c : bc) specs.push_back({t.data, t.row_bytes(), r, 1, c.first, c.second});
    }
    to_runs(t.type, specs, buf.data());
  }
  m.p = upload<uint8_t>(card, buf.data(), buf.size());
  return m;
}

Model::Model(const std::string &path, const Options &opt) : opt_(opt), gguf_(path) {
  auto t0 = std::chrono::steady_clock::now();
  if (gguf_.get_string("general.architecture") != "qwen4exp") throw std::runtime_error("not a qwen4exp model");
  if (opt_.max_ctx % Q_IDX_BLOCK) throw std::runtime_error("max_ctx must be a multiple of 4");
  rt_ = std::make_unique<ze::Runtime>(1);
  lib_ = std::make_unique<ze::Library>(*rt_);
  ncards_ = std::min(opt_.max_cards, rt_->ncards());
  if (ncards_ != 1 && ncards_ != 2) throw std::runtime_error("the engine runs on one or two cards");
  // The copy engines read an anonymous arena through a userptr binding of each copy's source (ADR-0035, item 2), which
  // needs no shared system USM. Out of fault mode (ADR-0037) the cards report none: 0x0 here.
  for (uint32_t c = 0; c < ncards_ && opt_.anon_arena; c++)
    std::fprintf(stderr, "engine: %s: shared system USM capabilities 0x%x\n", rt_->cards()[c].name.c_str(),
                 rt_->cards()[c].shared_system);
  npp_ = (opt_.max_ctx / KV_PAGE_CELLS + ncards_ - 1) / ncards_ * ncards_;
  kv_pages_ = opt_.kv_pages ? (opt_.kv_pages + ncards_ - 1) / ncards_ * ncards_ : opt_.slots * npp_;
  // Sharing moves experts in and out of the cache, so it needs the swaps, and a cache the engine sizes.
  share_ = opt_.share_kv && opt_.adapt_swaps && opt_.experts_per_layer < 0;
  if (share_) blk_ = rt_->vm_block(0);
  pool_ = std::make_unique<Pool>(opt_.cpu_threads > 1 ? opt_.cpu_threads - 1 : 0);
  cpu_ = std::make_unique<CpuExperts>(*pool_);
  ngram_ = std::make_unique<Ngram>(gguf_);
  events_ = std::make_unique<ze::EventPool>(*rt_, Q_LAYERS * (2 * ncards_ + 2) + 1 + kAttn * ncards_);

  plan_ = static_cast<StepPlan *>(rt_->alloc_host(sizeof(StepPlan)));
  std::memset(plan_, 0, sizeof(StepPlan));
  embed_ = static_cast<float *>(rt_->alloc_host(STEP_MAX_ROWS * Q_EMBD * 4));
  ngram_u_ = static_cast<float *>(rt_->alloc_host(STEP_MAX_ROWS * Q_EMBD * 4));
  ex1_ = static_cast<float *>(rt_->alloc_host((size_t)Q_LAYERS * ncards_ * STEP_MAX_ROWS * Q_EMBD * 4));
  ex2_ = static_cast<float *>(rt_->alloc_host((size_t)Q_LAYERS * ncards_ * STEP_MAX_ROWS * Q_EMBD * 4));
  jobs_ = static_cast<MissJob *>(rt_->alloc_host((size_t)Q_LAYERS * STEP_MAX_ROWS * sizeof(MissJob)));
  std::memset(jobs_, 0, (size_t)Q_LAYERS * STEP_MAX_ROWS * sizeof(MissJob));
  ans_seq_ = static_cast<uint32_t *>(rt_->alloc_host((size_t)Q_LAYERS * 64));
  std::memset(ans_seq_, 0, (size_t)Q_LAYERS * 64);
  stuck_ = static_cast<uint32_t *>(rt_->alloc_host(64));
  *stuck_ = 0;
  step_ts_ = static_cast<uint64_t *>(rt_->alloc_host((size_t)ncards_ * 2 * 8));
  waited_ = static_cast<uint64_t *>(rt_->alloc_host((size_t)Q_LAYERS * 8));
  std::memset(waited_, 0, (size_t)Q_LAYERS * 8);
  cpu_y_ = static_cast<float *>(rt_->alloc_host((size_t)Q_LAYERS * STEP_MAX_ROWS * Q_EMBD * 4));
  icand_ = static_cast<uint32_t *>(rt_->alloc_host((size_t)kAttn * ncards_ * STEP_MAX_ROWS * kIdxCand * 4));
  top_idx_ = static_cast<uint32_t *>(rt_->alloc_host((size_t)ncards_ * STEP_MAX_ROWS * opt_.candidates * 4));
  top_val_ = static_cast<float *>(rt_->alloc_host((size_t)ncards_ * STEP_MAX_ROWS * opt_.candidates * 4));
  top_n_ = static_cast<uint32_t *>(rt_->alloc_host(ncards_ * STEP_MAX_ROWS * 4));
  head_plan_ = static_cast<StepPlan *>(rt_->alloc_host(sizeof(StepPlan)));
  std::memset(head_plan_, 0, sizeof(StepPlan));
  head_plan_->nrows = 1;
  cplan_ = static_cast<CommitPlan *>(rt_->alloc_host(sizeof(CommitPlan)));
  std::memset(cplan_, 0, sizeof(CommitPlan));
  rope_ = static_cast<SlotRope *>(rt_->alloc_host(opt_.slots * sizeof(SlotRope)));
  std::memset(rope_, 0, opt_.slots * sizeof(SlotRope));
  images_.resize(opt_.slots);
  blocks_.resize(opt_.slots);
  slot_gen_.assign(opt_.slots, 0);

  tmax_ = std::min(opt_.prompt_chunk, opt_.max_ctx / 64 * 64);
  kv_margin_ = std::max<uint32_t>(2048, 2 * tmax_ / KV_PAGE_CELLS);  // two prompt chunks, or 8,192 positions
  if (tmax_) {
    if (opt_.prompt_chunk % 64) throw std::runtime_error("prompt_chunk must be a multiple of 64");
    pevents_ = std::make_unique<ze::EventPool>(*rt_, 7 * Q_LAYERS * ncards_ + 1 + kAttn * ncards_ + ncards_);
    pplan_ = static_cast<PromptPlan *>(rt_->alloc_host((1 + PROMPT_MAX_RUNS) * sizeof(PromptPlan)));
    std::memset(pplan_, 0, (1 + PROMPT_MAX_RUNS) * sizeof(PromptPlan));
    prows_ = static_cast<uint32_t *>(rt_->alloc_host(2 * 4));
    std::memset(prows_, 0, 2 * 4);
    ppos_ = static_cast<uint32_t *>(rt_->alloc_host((size_t)PROMPT_MAX_RUNS * ppos_run() * 4));
    std::memset(ppos_, 0, (size_t)PROMPT_MAX_RUNS * ppos_run() * 4);
    pembed_ = static_cast<float *>(rt_->alloc_host((size_t)tmax_ * Q_EMBD * 4));
    pngram_ = static_cast<float *>(rt_->alloc_host((size_t)tmax_ * Q_EMBD * 4));
    pngram16_ = static_cast<uint16_t *>(rt_->alloc_host((size_t)tmax_ * Q_EMBD * 2));
    std::memset(pngram16_, 0, (size_t)tmax_ * Q_EMBD * 2);
  }
  if (!opt_.mtp_dir.empty()) {
    mtp_on_ = true;
    mplan_ = static_cast<StepPlan *>(rt_->alloc_host(sizeof(StepPlan)));
    std::memset(mplan_, 0, sizeof(StepPlan));
    membed_ = static_cast<float *>(rt_->alloc_host(STEP_MAX_ROWS * Q_EMBD * 4));
    mex1_ = static_cast<float *>(rt_->alloc_host((size_t)ncards_ * STEP_MAX_ROWS * Q_EMBD * 4));
    mex2_ = static_cast<float *>(rt_->alloc_host((size_t)ncards_ * STEP_MAX_ROWS * Q_EMBD * 4));
    mtop_idx_ = static_cast<uint32_t *>(rt_->alloc_host((size_t)ncards_ * STEP_MAX_ROWS * opt_.candidates * 4));
    mtop_val_ = static_cast<float *>(rt_->alloc_host((size_t)ncards_ * STEP_MAX_ROWS * opt_.candidates * 4));
    mtop_n_ = static_cast<uint32_t *>(rt_->alloc_host(ncards_ * STEP_MAX_ROWS * 4));
    mevents_ = std::make_unique<ze::EventPool>(*rt_, 2 * ncards_);
  }
  use_.assign((size_t)Q_LAYERS * Q_EXPERTS, 0.0f);
  load_cards(true);
  info_.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// Everything on the cards, from the file: the weights, each card's share of the cached experts, the state and the
// buffers, the recorded lists, and a warm-up chunk. first: the constructor's load, which also chooses the cached
// experts and fills the arena with the others; after a sleep, the cards cache the experts they cached when they slept.
void Model::load_cards(bool first) {
  const auto t0 = std::chrono::steady_clock::now();
  reader_ = std::make_unique<DirectReader>(gguf_.files());
  struct Done {
    Model *m;
    ~Done() {
      m->unstage();
      m->reader_.reset();
    }
  } done{this};
  for (uint32_t c = 0; c < ncards_; c++) {
    auto card = std::make_unique<Card>();
    card->c = c;
    card->C = ncards_;
    card->up = std::make_unique<ze::List>(*rt_, c, ze::List::immediate);
    cards_.push_back(std::move(card));
  }
  for (auto &card : cards_) load_card(*card);
  if (tmax_)
    for (auto &card : cards_) load_prompt(*card);
  load_vision(first);
  if (mtp_on_)
    for (auto &card : cards_) load_mtp(*card);
  load_experts(first);
  std::fprintf(stderr, "engine: read %.2f GB of weights with O_DIRECT in %.2f s%s\n", reader_->bytes_read() / 1e9,
               std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
               first ? ", the arena with them" : "");
  reader_.reset();
  if (first && opt_.anon_arena) settle_arena();
  for (auto &card : cards_) {
    for (uint32_t W = 1; W <= STEP_MAX_ROWS; W *= 2) record(*card, W);
    record_commit(*card);
    if (mtp_on_) {
      for (uint32_t W = 1; W <= STEP_MAX_ROWS; W *= 2) record_mtp(*card, W, false);
      for (uint32_t W = 1; W <= STEP_MAX_ROWS; W *= 2) record_mtp(*card, W, true);
    }
  }
  if (tmax_) {
    for (auto &card : cards_) record_prompt_copy(*card);
    vision_slots();
    // Every chunk size's lists now, so no request waits for one to be recorded. The engine reads a short prompt piece
    // as decode windows, so a chunk of several runs is seldom under 256 rows; a smaller one's lists are recorded when
    // first used.
    for (uint32_t T = 64;; T = std::min(2 * T, tmax_)) {
      for (auto &card : cards_) prompt_list(*card, T, 1);
      if (T >= 256)
        for (auto &card : cards_) prompt_list(*card, T, PROMPT_MAX_RUNS);
      if (T == tmax_) break;
    }
  }
  for (uint32_t s = 0; s < opt_.slots; s++) reset_slot(s);
  if (tmax_) {
    // One chunk of the largest size, once, because a list's first run is much slower than the next ones. Varied
    // tokens, so the experts' tiles and the attention's sparse rows run as for a real prompt.
    std::vector<int32_t> warm(tmax_);
    for (uint32_t i = 0; i < tmax_; i++) warm[i] = (int32_t)((7919ull * i + 13) % 150000);
    prompt(0, 0, warm.data(), tmax_, kNoToken, kNoToken);
    reset_slot(0);
  }
  info_.cards = ncards_;
  info_.experts_per_layer = nslots_cached_;
  if (share_) {
    uint32_t sum = 0;
    for (auto &hl : host_layers_) sum += Q_EXPERTS - (uint32_t)hl->uncached.size();
    info_.experts_per_layer = sum / Q_LAYERS;
    log_share("loaded");
  }
  info_.arena_anon = opt_.anon_arena;
  info_.vocab = Q_VOCAB;
  for (uint32_t c = 0; c < ncards_; c++) info_.device_bytes[c] = rt_->device_bytes(c);
  // The file's pages back to the host, but the token embedding's, which each step reads: everything else is on the
  // cards or in the arena now, and a swap reads its experts again. Otherwise the mapping holds most of the file in
  // the page cache (docs/measurements/MEASUREMENTS.md#2026-10-14-decode-stalls) on a host short of memory (MEASUREMENTS.md#the-test-machine).
  for (const auto &[name, t] : gguf_.tensors())
    if (name != "token_embd.weight") gguf_.release(t);
  // Every page of the arena mapped for the CPU before the first step, after a sleep too. The load wrote them all,
  // but the CPU's mappings may be gone, and an expert's first miss would fault
  // (docs/measurements/MEASUREMENTS.md#2026-10-15-arena-touch).
  std::vector<std::pair<const uint8_t *, uint64_t>> pieces;
  for (auto &hl : host_layers_)
    for (const uint8_t *p : hl->arena)
      if (p) pieces.push_back({p, 2 * hl->gate_bytes + hl->down_bytes});
  pool_->run(pieces.size(), [&](uint64_t i) {
    for (uint64_t o = 0; o < pieces[i].second; o += 4096) (void)*(volatile const uint8_t *)(pieces[i].first + o);
  });
}

void Model::sleep() {
  if (asleep_) return;
  wait_commit();
  if (swap_bg_.joinable()) swap_bg_.join();
  settle_moves();
  // A swap round's choice not yet applied: its pieces go back to the spares; its halves were for these cards.
  for (const SwapPlan &w : swap_next_) {
    const HostLayer &hl = *host_layers_[w.l];
    if (w.piece) spares_[align256(2 * hl.gate_bytes + hl.down_bytes)].push_back(w.piece);
  }
  swap_next_.clear();
  evict_next_.clear();  // its victims stay cached; their pieces stay free positions
  next_copies_.clear();
  retired_.clear();
  venc_slot_ = -1;  // the last image encoded is no slot's any more
  for (auto &card : cards_) {
    for (auto &[w, ts] : card->trace) rt_->free(ts);
    if (card->pc)
      for (auto &[t, ts] : card->pc->trace) rt_->free(ts);
  }
  cards_.clear();  // the lists and their queues
  for (uint32_t c = 0; c < ncards_; c++) {
    rt_->free_card(c);
    info_.device_bytes[c] = 0;
  }
  asleep_ = true;
}

void Model::wake() {
  if (!asleep_) return;
  auto t0 = std::chrono::steady_clock::now();
  try {
    load_cards(false);
  } catch (...) {
    asleep_ = false;
    sleep();  // what the failed load put on the cards
    throw;
  }
  asleep_ = false;
  info_.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// need bytes of the arena, in pieces of 1 GiB, as small pieces are slower to allocate and to read
// (docs/measurements/MEASUREMENTS.md#2026-10-13-design-inputs). With Options::anon_arena (the default, ADR-0035) a piece is the
// process's own memory, 2 MiB aligned for huge pages, which the driver cannot take from the CPU; without it, the
// driver's host memory (DESIGN.md, section 6).
uint8_t *Model::arena_piece(uint64_t need) {
  if (arena_left_ < need) {
    if (opt_.anon_arena) {
      const uint64_t huge = 2ull << 20, bytes = kArenaPiece + huge;
      void *m = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      if (m == MAP_FAILED) throw std::runtime_error("anon_arena: mmap of a piece failed");
      // Locked as the load fills it: a page the kernel swaps out drops the cards' userptr bindings of it, and a copy
      // that faults it back in under reclaim can fail and reset the copy engine. A memory lock too small for it
      // (Docker's default) stops the load, unless ENGINE_UNLOCKED_ARENA is set (docs/DEVELOPING.md).
      if (mlock2(m, bytes, MLOCK_ONFAULT) != 0) {
        const int e = errno;
        if (!std::getenv("ENGINE_UNLOCKED_ARENA")) {
          munmap(m, bytes);
          throw std::runtime_error(std::string("anon_arena: the arena cannot be locked in memory (mlock2: ") +
                                   std::strerror(e) +
                                   "); run the container with --ulimit memlock=-1:-1, or set ENGINE_UNLOCKED_ARENA=1 "
                                   "to load it unlocked");
        }
        if (!arena_lock_errno_) arena_lock_errno_ = e;
      }
      arena_at_ = reinterpret_cast<uint8_t *>((reinterpret_cast<uintptr_t>(m) + huge - 1) & ~(huge - 1));
      anon_pieces_.push_back({m, bytes, arena_at_, 0});
      madvise(arena_at_, kArenaPiece, MADV_HUGEPAGE);
    } else {
      arena_at_ = static_cast<uint8_t *>(rt_->alloc_host(kArenaPiece));
    }
    arena_left_ = kArenaPiece;
  }
  uint8_t *p = arena_at_;
  arena_at_ += need;
  arena_left_ -= need;
  if (opt_.anon_arena && !anon_pieces_.empty()) anon_pieces_.back().used = kArenaPiece - arena_left_;
  return p;
}

// Each piece's filled part in huge pages before any list binds it (record_prompt_copy). Compaction moves small pages,
// locked ones too while vm.compact_unevictable_allowed is 1, but not huge ones, and khugepaged would collapse the
// small ones later, under the bindings. Best effort: a range the kernel cannot collapse now stays in small pages.
void Model::settle_arena() {
  const uint64_t huge = 2ull << 20;
  const auto t0 = std::chrono::steady_clock::now();
  uint64_t ranges = 0, collapsed = 0;
  for (const AnonPiece &a : anon_pieces_)
    for (uint64_t o = 0; o + huge <= a.used; o += huge, ranges++)
      if (madvise(a.at + o, huge, MADV_COLLAPSE) == 0) collapsed++;
  const std::string lock = arena_lock_errno_ ? std::string("not locked (mlock2: ") + std::strerror(arena_lock_errno_) +
                                                   "), so the kernel may swap it out"
                                             : "locked in memory";
  std::fprintf(stderr, "engine: the arena: %llu of %llu 2 MiB ranges in huge pages, in %.2f s; %s\n",
               (unsigned long long)collapsed, (unsigned long long)ranges,
               std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), lock.c_str());
}

Model::~Model() {
  wait_commit();
  if (swap_bg_.joinable()) swap_bg_.join();
  if (move_bg_.joinable()) move_bg_.join();
  for (const AnonPiece &a : anon_pieces_) munmap(a.map, a.map_bytes);
}

const GgufTensor &Model::src(const std::string &name) {
  if (auto it = staged_.find(name); it != staged_.end()) return it->second.t;
  const GgufTensor &t = gguf_.tensor(name);
  StagedTensor x{t, aligned_buffer(DirectReader::span(t.offset, t.bytes))};
  x.t.data = reader_->read(t.shard, t.offset, t.bytes, x.buf.get());
  return staged_.emplace(name, std::move(x)).first->second.t;
}

void Model::load_card(Card &card) {
  const uint32_t C = ncards_, c = card.c;
  card.nkh = Q_GDN_KH / C;
  card.nvh = Q_GDN_VH / C;
  card.nch = 2 * card.nkh * Q_GDN_D + card.nvh * Q_GDN_D;
  card.nq = Q_ATT_H / C;
  card.nkv = Q_ATT_KV / C;
  card.ff = Q_FF / C;
  card.vocab_n = Q_VOCAB / C;
  card.vocab0 = c * card.vocab_n;
  const uint32_t kh0 = c * card.nkh, q0 = c * card.nq, kv0 = c * card.nkv, f0 = c * card.ff;
  const uint32_t S = opt_.slots;

  card.grids = upload<uint8_t>(card, codebooks().data(), QGRID_BYTES);
  auto raw = [&](const std::string &name) {
    const GgufTensor &t = src(name);
    if (t.type == QT_F16) {  // only blk.1.ple_conv1d: widened to f32 once
      std::vector<float> f(t.elements());
      ggml_get_type_traits(GGML_TYPE_F16)->to_float(t.data, f.data(), (int64_t)f.size());
      return upload<uint8_t>(card, f.data(), f.size() * 4);
    }
    return upload<uint8_t>(card, t.data, t.bytes);
  };
  auto f32 = [&](const std::string &name) { return reinterpret_cast<float *>(raw(name)); };
  // A file may store a tensor the kernels take as BF16 in a quantised type (blk.1.ple_key is Q2_0 in the Q2_0 file):
  // widened to BF16 once.
  auto bf16 = [&](const std::string &name) {
    const GgufTensor &t = src(name);
    if (t.type == QT_BF16) return reinterpret_cast<uint16_t *>(raw(name));
    std::vector<float> f(t.elements());
    ggml_get_type_traits((ggml_type)t.type)->to_float(t.data, f.data(), (int64_t)f.size());
    std::vector<uint16_t> b(f.size());
    for (size_t i = 0; i < f.size(); i++) b[i] = bf16_bits(f[i]);
    return upload<uint16_t>(card, b.data(), b.size() * 2);
  };
  // The hyper-connections' up projections, rows reordered from stream-major (s * 2560 + i) to value-major
  // (i * 4 + s), so the 4 streams of a value sit in 4 neighbouring rows: the prompt path sums them inside its product
  // (gemm_hcmix), and the decode step's hc_mix reads them so.
  auto hc_up_rows = [&](const std::string &name) {
    const GgufTensor &t = src(name);
    const uint64_t rb = t.row_bytes();
    std::vector<uint8_t> v(t.bytes);
    for (uint32_t st = 0; st < Q_HC; st++)
      for (uint32_t i = 0; i < Q_EMBD; i++)
        std::memcpy(v.data() + ((uint64_t)i * Q_HC + st) * rb, t.data + ((uint64_t)st * Q_EMBD + i) * rb, rb);
    return upload<uint16_t>(card, v.data(), v.size());
  };
  auto gather_f32 = [&](const std::string &name, const Ranges &idx, uint32_t width) {
    const GgufTensor &t = src(name);
    const float *src = reinterpret_cast<const float *>(t.data);
    std::vector<float> v;
    for (auto &r : idx)
      for (uint64_t i = r.first; i < r.first + r.second; i++)
        v.insert(v.end(), src + i * width, src + (i + 1) * width);
    return upload<float>(card, v.data(), v.size() * 4);
  };

  // The GDN heads of this card: key heads [kh0, kh0 + nkh), value heads kh + 16 j for j = 0..2, j-major.
  Ranges vh, qkv_rows, z_rows, out_cols;
  for (uint32_t j = 0; j < 3; j++) vh.push_back({16 * j + kh0, card.nkh});
  qkv_rows.push_back({Q_GDN_D * kh0, Q_GDN_D * card.nkh});
  qkv_rows.push_back({Q_GDN_KH * Q_GDN_D + Q_GDN_D * kh0, Q_GDN_D * card.nkh});
  for (auto &r : vh) qkv_rows.push_back({2 * Q_GDN_KH * Q_GDN_D + Q_GDN_D * r.first, Q_GDN_D * r.second});
  for (auto &r : vh) z_rows.push_back({Q_GDN_D * r.first, Q_GDN_D * r.second});
  for (auto &r : vh) out_cols.push_back({r.first / 2, r.second / 2});  // two heads per 256-value block

  uint32_t gi = 0, ai = 0;
  card.layers.resize(Q_LAYERS);
  for (uint32_t l = 0; l < Q_LAYERS; l++) {
    Layer &L = card.layers[l];
    L.attn = Q_IS_ATTN(l);
    L.idx = L.attn ? ai++ : gi++;
    const char *hn[2][4] = {{"hc_attn_norm.weight", "hc_attn_down.weight", "hc_attn_up.weight", "hc_attn_inject.weight"},
                            {"hc_ffn_norm.weight", "hc_ffn_down.weight", "hc_ffn_up.weight", "hc_ffn_inject.weight"}};
    for (int i = 0; i < 2; i++) {
      L.hc_norm[i] = f32(blk(l, hn[i][0]));
      {  // down's 320 rows, then inject's 4: one product for both in the prompt path
        const GgufTensor &td = src(blk(l, hn[i][1])), &tj = src(blk(l, hn[i][3]));
        std::vector<uint8_t> v(td.data, td.data + td.bytes);
        v.insert(v.end(), tj.data, tj.data + tj.bytes);
        L.hc_down[i] = upload<uint16_t>(card, v.data(), v.size());
        L.hc_inj[i] = L.hc_down[i] + (size_t)Q_HC_LR * Q_HC_DIM;
      }
      L.hc_up[i] = hc_up_rows(blk(l, hn[i][2]));
    }
    if (!L.attn) {
      L.qkv = mat(card, src(blk(l, "attn_qkv.weight")), qkv_rows);
      L.z = mat(card, src(blk(l, "attn_gate.weight")), z_rows);
      L.out = mat(card, src(blk(l, "ssm_out.weight")), {{0, Q_EMBD}}, out_cols);
      {
        const GgufTensor &ta = src(blk(l, "ssm_alpha.weight")), &tb = src(blk(l, "ssm_beta.weight"));
        std::vector<uint8_t> v;
        for (const GgufTensor *t : {&ta, &tb})
          for (auto &r : vh) v.insert(v.end(), t->data + r.first * t->row_bytes(),
                                      t->data + (r.first + r.second) * t->row_bytes());
        L.ab = upload<uint16_t>(card, v.data(), v.size());
      }
      std::vector<std::pair<uint64_t, uint64_t>> ch;  // conv channels follow qkv's rows
      for (auto &r : qkv_rows) ch.push_back(r);
      L.conv = gather_f32(blk(l, "ssm_conv1d.weight"), ch, Q_GDN_CONV);
      L.A = gather_f32(blk(l, "ssm_a"), vh, 1);
      L.dt = gather_f32(blk(l, "ssm_dt.bias"), vh, 1);
      L.ssm_norm = f32(blk(l, "ssm_norm.weight"));
      L.conv_stride = 3ull * card.nch;
      L.S_stride = (uint64_t)card.nvh * Q_GDN_D * Q_GDN_D;
      L.conv_state = upload<float>(card, nullptr, S * L.conv_stride * 4);
      L.wu = upload<float>(card, nullptr, (size_t)STEP_MAX_ROWS * card.nch * 4);
      L.wgb = upload<float>(card, nullptr, (size_t)STEP_MAX_ROWS * 2 * card.nvh * 4);
      L.wm = upload<float>(card, nullptr, (size_t)STEP_MAX_ROWS * card.nch * 4);
      L.S = upload<float>(card, nullptr, S * L.S_stride * 4);
    } else {
      L.q = mat(card, src(blk(l, "attn_q.weight")), {{2 * Q_ATT_D * q0, 2 * Q_ATT_D * card.nq}});
      L.k = mat(card, src(blk(l, "attn_k.weight")), {{Q_ATT_D * kv0, Q_ATT_D * card.nkv}});
      L.v = mat(card, src(blk(l, "attn_v.weight")), {{Q_ATT_D * kv0, Q_ATT_D * card.nkv}});
      L.o = mat(card, src(blk(l, "attn_output.weight")), {{0, Q_EMBD}}, {{q0, card.nq}});
      L.qn = f32(blk(l, "attn_q_norm.weight"));
      L.kn = f32(blk(l, "attn_k_norm.weight"));
      L.iqn = f32(blk(l, "indexer.q_norm.weight"));
      L.ikn = f32(blk(l, "indexer.k_norm.weight"));
      {
        const GgufTensor &tq = src(blk(l, "indexer.q_proj.weight")),
                         &tk = src(blk(l, "indexer.k_proj.weight"));
        std::vector<uint8_t> v(tq.data, tq.data + tq.bytes);
        v.insert(v.end(), tk.data, tk.data + tk.bytes);
        L.idx_w = upload<uint16_t>(card, v.data(), v.size());
      }
      // The pool of pages (layout.h); the indexer's keys of this card's pages only (split by position: DESIGN.md,
      // section 4).
      const uint64_t cells = (uint64_t)kv_pages_ * KV_PAGE_CELLS, own = (uint64_t)kv_pages_ / C;
      if (share_) {  // backed as pages come into use (kv_share.cpp)
        const uint32_t x = 4 * L.idx;
        L.kc = kv_region(card, x, KV_PAGE_CELLS * card.nkv * KV_HEAD_BYTES, 1);
        L.vc = kv_region(card, x + 1, KV_PAGE_CELLS * card.nkv * KV_HEAD_BYTES, 1);
        L.raw = kv_region(card, x + 2, KV_PAGE_CELLS * IDX_KEY_BYTES, C);
        L.pooled = reinterpret_cast<float *>(kv_region(card, x + 3, Q_IDX_D * 4, C));
      } else {
        L.kc = upload<uint8_t>(card, nullptr, cells * card.nkv * KV_HEAD_BYTES);
        L.vc = upload<uint8_t>(card, nullptr, cells * card.nkv * KV_HEAD_BYTES);
        L.raw = upload<uint8_t>(card, nullptr, own * KV_PAGE_CELLS * IDX_KEY_BYTES);
        L.pooled = upload<float>(card, nullptr, own * Q_IDX_D * 4);
      }
    }
    {
      const GgufTensor &tr = src(blk(l, "ffn_gate_inp.weight")),
                       &ts = src(blk(l, "ffn_gate_inp_shexp.weight"));
      std::vector<uint8_t> v(tr.data, tr.data + tr.bytes);
      v.insert(v.end(), ts.data, ts.data + ts.bytes);
      L.router = upload<uint16_t>(card, v.data(), v.size());
    }
    L.sh_gate = mat(card, src(blk(l, "ffn_gate_shexp.weight")), {{f0, card.ff}});
    L.sh_up = mat(card, src(blk(l, "ffn_up_shexp.weight")), {{f0, card.ff}});
    const GgufTensor &sd = src(blk(l, "ffn_down_shexp.weight"));
    uint32_t sqk = quant_block_values(sd.type);
    L.sh_down = mat(card, sd, {{0, Q_EMBD}}, {{f0 / sqk, card.ff / sqk}});
    unstage();
  }

  card.out_norm = f32("output_hc_norm.weight");
  card.out_down = bf16("output_hc_down.weight");
  card.out_up = hc_up_rows("output_hc_up.weight");
  card.head = mat(card, src("output.weight"), {{card.vocab0, card.vocab_n}});
  card.ple_k = bf16("blk.1.ple_key.weight");
  card.ple_v = bf16("blk.1.ple_value.weight");
  card.ple_gk = f32("blk.1.ple_norm_key.weight");
  card.ple_gq = f32("blk.1.ple_norm_query.weight");
  card.ple_gc = f32("blk.1.ple_norm_conv.weight");
  card.ple_conv = f32("blk.1.ple_conv1d.weight");
  {
    // Each slot's own run of pages while they fit in the pool; page 0 past it, until set_pages.
    std::vector<uint32_t> t((size_t)S * npp_);
    for (size_t i = 0; i < t.size(); i++) t[i] = i < kv_pages_ ? (uint32_t)i : 0;
    card.ptab = upload<uint32_t>(card, t.data(), t.size() * 4);
  }
  card.ple_stride = (uint64_t)Q_PLE_HIST * Q_HC_DIM;
  card.ple_hist = upload<float>(card, nullptr, S * card.ple_stride * 4);
  unstage();

  // The decode step's buffers, for STEP_MAX_ROWS rows.
  constexpr size_t W = STEP_MAX_ROWS;
  auto fbuf = [&](size_t n) { return upload<float>(card, nullptr, W * n * 4); };
  const uint32_t nb_max = opt_.max_ctx / Q_IDX_BLOCK;
  card.R = fbuf(Q_HC_DIM);
  card.n = fbuf(Q_HC_DIM);
  card.plan = upload<StepPlan>(card, nullptr, sizeof(StepPlan));
  card.xb16 = upload<uint16_t>(card, nullptr, (size_t)STEP_MAX_ROWS * Q_HC_DIM * 2);
  card.lrb16 = upload<uint16_t>(card, nullptr, (size_t)STEP_MAX_ROWS * Q_HC_LR * 2);
  card.xbf16 = upload<uint16_t>(card, nullptr, (size_t)STEP_MAX_ROWS * Q_EMBD * 2);
  card.ngb16 = upload<uint16_t>(card, nullptr, (size_t)STEP_MAX_ROWS * Q_EMBD * 2);
  card.lr = fbuf(Q_HC_LR + Q_HC);  // the down product's 320 values, then its 4 inject rows
  card.a = fbuf(Q_HC_DIM);
  card.inj[0] = fbuf(Q_HC_LR + Q_HC);  // the inject values are this + Q_HC_LR (the down product writes both)
  card.inj[1] = fbuf(Q_HC_LR + Q_HC);
  card.x = fbuf(Q_EMBD);
  card.m = fbuf(card.nch);
  card.z = fbuf(card.nvh * Q_GDN_D);
  card.ab = fbuf(2 * card.nvh);
  card.gb = fbuf(2 * card.nvh);
  card.u = fbuf(card.nch);
  card.Sw = upload<float>(card, nullptr, (size_t)card.nvh * Q_GDN_D * Q_GDN_D * 4);
  card.o = fbuf(std::max(card.nvh * Q_GDN_D, card.nq * Q_ATT_D));
  card.Q = fbuf(card.nq * 2 * Q_ATT_D);
  card.Kt = fbuf(card.nkv * Q_ATT_D);
  card.Vt = fbuf(card.nkv * Q_ATT_D);
  card.idxq = fbuf(Q_IDX_H * Q_IDX_D + Q_IDX_D);
  card.scores = fbuf(nb_max);
  card.bscores = fbuf(Q_IDX_BLOCKS + 1);
  card.att = fbuf((size_t)STEP_MAX_ROWS * card.nq * kAttSplits * (Q_ATT_D + 2));
  card.blocks = upload<uint32_t>(card, nullptr, W * (Q_IDX_BLOCKS + 1) * 4);
  card.cells = upload<uint32_t>(card, nullptr, W * Q_IDX_CELLS * 4);
  card.ncells = upload<uint32_t>(card, nullptr, W * 4);
  card.rl = fbuf(Q_EXPERTS + 1);
  card.w = fbuf(Q_USED);
  card.ids = upload<int32_t>(card, nullptr, W * Q_USED * 4);
  card.merged = fbuf(2 * (Q_IDX_BLOCKS + 1));
  card.icand = upload<uint32_t>(card, nullptr, W * kIdxCand * 4);
  card.slots = upload<int32_t>(card, nullptr, W * Q_USED * 4);
  card.nmiss = upload<uint32_t>(card, nullptr, Q_LAYERS * W * 4);
  card.groups = reinterpret_cast<MoeGroup *>(upload<uint8_t>(card, nullptr, W * Q_USED * sizeof(MoeGroup)));
  card.ngroups = upload<uint32_t>(card, nullptr, 4);
  card.h = fbuf((size_t)Q_USED * card.ff);
  card.y = fbuf((size_t)Q_USED * Q_EMBD);
  card.shg = fbuf(card.ff);
  card.shu = fbuf(card.ff);
  card.shh = fbuf(card.ff);
  card.s = fbuf(Q_EMBD);
  card.pk = fbuf(Q_HC_DIM);
  card.pv = fbuf(Q_EMBD);
  card.gates = fbuf(Q_HC);
  card.N = fbuf(Q_HC_DIM);
  card.logits = fbuf(card.vocab_n);
  card.xa8 = upload<uint8_t>(card, nullptr, W * a8_bytes(Q_EMBD));
  card.oa8 = upload<uint8_t>(card, nullptr, W * a8_bytes(std::max(card.nvh * Q_GDN_D, card.nq * Q_ATT_D)));
  card.ha8 = upload<uint8_t>(card, nullptr, W * Q_USED * a8_bytes(card.ff));
  card.sha8 = upload<uint8_t>(card, nullptr, W * a8_bytes(card.ff));
  card.yo = fbuf(Q_EMBD);
  if (ncards_ > 1) card.xp = upload<float>(card, nullptr, (size_t)2 * ncards_ * STEP_MAX_ROWS * Q_EMBD * 4);
}

// Which experts the cards cache, and the arena of the others for the CPU. Each card holds half of every cached
// expert: rows [f0, f0 + ff) of gate and up, the matching columns of down. Both cards cache the same experts. first:
// choose them and fill the arena; otherwise the cards cache the experts the host's maps name, swaps included.
void Model::load_experts(bool first) {
  // Bytes per slot, per layer.
  std::vector<uint64_t> slot_bytes(Q_LAYERS);
  uint64_t per_set = 0;
  for (uint32_t l = 0; l < Q_LAYERS; l++) {
    const GgufTensor &g = gguf_.tensor(blk(l, "ffn_gate_exps.weight")), &d = gguf_.tensor(blk(l, "ffn_down_exps.weight"));
    Card &c0 = *cards_[0];
    uint64_t gb = runs_bytes(g.type, (uint64_t)c0.ff * (Q_EMBD / quant_block_values(g.type)));
    uint64_t db = runs_bytes(d.type, (uint64_t)Q_EMBD * (c0.ff / quant_block_values(d.type)));
    for (auto &card : cards_) {
      Layer &L = card->layers[l];
      L.gu_type = g.type;
      L.down_type = d.type;
      L.up_off = align256(gb);
      L.down_off = L.up_off + align256(gb);
      L.slot_bytes = L.down_off + align256(db);
    }
    slot_bytes[l] = cards_[0]->layers[l].slot_bytes;
    per_set += slot_bytes[l];
  }
  // The prompt path's staging holds two layers of the experts not cached, each card's share of them: its rows of gate
  // and up, and its region of down. The arena keeps down in one region per card, so a card stages only the blocks it
  // reads (docs/measurements/MEASUREMENTS.md#2026-10-17-staging-half-downs).
  std::vector<uint64_t> stg_bytes(Q_LAYERS);
  for (uint32_t l = 0; l < Q_LAYERS; l++) {
    const GgufTensor &g = gguf_.tensor(blk(l, "ffn_gate_exps.weight")), &d = gguf_.tensor(blk(l, "ffn_down_exps.weight"));
    uint64_t gh = align256((uint64_t)cards_[0]->ff * g.row_bytes());
    stg_bytes[l] = 2 * gh + align256(d.bytes / Q_EXPERTS / ncards_);
  }
  auto staging = [&](uint64_t n) {
    if (tmax_ == 0) return (uint64_t)0;
    uint64_t mx = 0;
    for (uint32_t l = 0; l < Q_LAYERS; l++) mx = std::max(mx, (Q_EXPERTS - n) * stg_bytes[l]);
    return 2 * mx;
  };
  uint64_t budget = UINT64_MAX;
  for (auto &card : cards_) {
    uint64_t total = rt_->cards()[card->c].memory,
             used = rt_->device_bytes(card->c) + 2 * Q_LAYERS * Q_EXPERTS * 4;
    uint64_t free = total > used + opt_.reserve_bytes ? total - used - opt_.reserve_bytes : 0;
    budget = std::min(budget, free);
  }
  uint64_t n = nslots_cached_;
  std::vector<uint64_t> profile;
  std::vector<uint32_t> count(Q_LAYERS);  // with sharing: each layer's cached experts
  if (share_) {
    // After a sleep, each layer's cached experts in slots 0 .. count - 1, in their order before it.
    for (uint32_t l = 0; !first && l < Q_LAYERS; l++) {
      HostLayer &hl = *host_layers_[l];
      std::vector<std::pair<int32_t, uint32_t>> by;
      for (uint32_t e = 0; e < Q_EXPERTS; e++)
        if (hl.slot_of[e] >= 0) by.push_back({hl.slot_of[e], e});
      std::sort(by.begin(), by.end());
      for (uint32_t i = 0; i < by.size(); i++) hl.slot_of[by[i].second] = (int32_t)i;
      count[l] = (uint32_t)by.size();
    }
    share_cards(first, budget, stg_bytes, count);
    n = nslots_cached_;
    if (first && !opt_.routing_profile.empty()) profile = read_routing_profile(opt_.routing_profile);
  } else if (first) {
    n = std::min<uint64_t>(Q_EXPERTS, budget / per_set);
    while (n > 0 && n * per_set + staging(n) > budget) n--;
    if (opt_.experts_per_layer >= 0) n = std::min<uint64_t>(n, (uint64_t)opt_.experts_per_layer);
    nslots_cached_ = (uint32_t)n;
    if (!opt_.routing_profile.empty()) profile = read_routing_profile(opt_.routing_profile);
  } else if (n * per_set + staging(n) > budget) {
    throw std::runtime_error("the cards no longer have room for the experts they cached before the sleep");
  }

  // Each layer's maps and caches, then every expert this load reads, through one pipeline: batches of 64 read from
  // the file with O_DIRECT while the CPU rearranges the batch before for both cards (to_runs) and copies the arena's
  // into their pieces (ADR-0032). A layer's cached experts come in slot order, then (first) its
  // arena's.
  struct Batch {
    uint32_t l, first, count;
  };
  std::vector<std::vector<uint32_t>> order(Q_LAYERS);
  std::vector<Batch> batches;
  constexpr uint32_t kBatch = 64;
  uint64_t per_expert = 0;  // staging bytes of one expert's three matrices, each widened for O_DIRECT
  constexpr uint64_t A = DirectReader::kAlign;
  auto cap = [](uint64_t bytes) { return (bytes + A - 1) / A * A + A; };
  for (uint32_t l = 0; l < Q_LAYERS; l++) {
    if (!share_) count[l] = (uint32_t)n;
    if (first)
      host_layers_.push_back(host_layer(l, count[l], profile, stg_bytes[l], share_ ? Q_EXPERTS - cache_slots(l, 0) : 0));
    const HostLayer &hl = *host_layers_[l];
    std::vector<uint32_t> &o = order[l];
    o.resize(count[l]);
    for (uint32_t e = 0; e < Q_EXPERTS; e++)
      if (hl.slot_of[e] >= 0) o[hl.slot_of[e]] = e;
    if (first)
      for (uint32_t e = 0; e < Q_EXPERTS; e++)
        if (hl.slot_of[e] < 0) o.push_back(e);
    for (uint32_t i = 0; i < o.size(); i += kBatch)
      batches.push_back({l, i, std::min<uint32_t>(kBatch, (uint32_t)o.size() - i)});
    per_expert = std::max(per_expert, 2 * cap(hl.gate_bytes) + cap(hl.down_bytes));
    for (auto &card : cards_) {
      Layer &L = card->layers[l];
      L.slot_map = upload<int32_t>(*card, hl.slot_of.data(), Q_EXPERTS * 4);
      if (n && !share_) L.cache = upload<uint8_t>(*card, nullptr, n * L.slot_bytes);
    }
  }
  struct Staged {
    AlignedBuffer buf;
    std::vector<ExpertWeights> at;  // each expert's three matrices in buf
  };
  Staged st[2];
  for (Staged &x : st) x.buf = aligned_buffer(kBatch * per_expert);
  uint64_t slot_max = 0;
  for (auto &L : cards_[0]->layers) slot_max = std::max(slot_max, L.slot_bytes);
  std::vector<std::vector<uint8_t>> runs(ncards_, std::vector<uint8_t>(kBatch * slot_max));
  auto read = [&](Staged &x, const Batch &B) {
    const GgufTensor *t[3] = {&gguf_.tensor(blk(B.l, "ffn_gate_exps.weight")),
                              &gguf_.tensor(blk(B.l, "ffn_up_exps.weight")),
                              &gguf_.tensor(blk(B.l, "ffn_down_exps.weight"))};
    const HostLayer &hl = *host_layers_[B.l];
    const uint64_t size[3] = {hl.gate_bytes, hl.gate_bytes, hl.down_bytes};
    const uint64_t off[3] = {0, cap(hl.gate_bytes), 2 * cap(hl.gate_bytes)};
    x.at.assign(B.count, {});
    reader_->pool().run(3 * B.count, [&](uint64_t j) {
      const uint32_t i = (uint32_t)(j / 3), w = (uint32_t)(j % 3), e = order[B.l][B.first + i];
      const uint8_t *p = reader_->read_one(t[w]->shard, t[w]->offset + e * size[w], size[w],
                                           x.buf.get() + i * per_expert + off[w]);
      (w == 0 ? x.at[i].gate : w == 1 ? x.at[i].up : x.at[i].down) = p;
    });
  };
  auto place = [&](const Staged &x, const Batch &B) {
    HostLayer &hl = *host_layers_[B.l];
    const uint32_t cached = B.first < count[B.l] ? std::min<uint32_t>(B.count, count[B.l] - B.first) : 0;
    const uint32_t gt = hl.shape.gate_type, dt = hl.shape.down_type, dqk = quant_block_values(dt);
    const uint64_t gbpr = Q_EMBD / quant_block_values(gt), grb = hl.gate_row_bytes, drb = hl.down_bytes / Q_EMBD;
    pool_->run((uint64_t)cached * ncards_ + (B.count - cached), [&](uint64_t j) {
      if (j >= (uint64_t)cached * ncards_) {  // an arena expert: its three matrices into its piece
        const uint32_t i = cached + (uint32_t)(j - (uint64_t)cached * ncards_), e = order[B.l][B.first + i];
        uint8_t *p = hl.arena[e];
        std::memcpy(p, x.at[i].gate, hl.gate_bytes);
        std::memcpy(p + hl.gate_bytes, x.at[i].up, hl.gate_bytes);
        write_down(p + 2 * hl.gate_bytes, x.at[i].down, Q_EMBD, drb, hl.down_parts);
        return;
      }
      const uint32_t c = (uint32_t)(j / cached), i = (uint32_t)(j % cached);
      const Card &card = *cards_[c];
      const Layer &L = card.layers[B.l];
      const uint32_t f0 = card.c * card.ff;
      uint8_t *slot = runs[c].data() + i * L.slot_bytes;
      to_runs(gt, {{x.at[i].gate, grb, f0, card.ff, 0, gbpr}}, slot);
      to_runs(gt, {{x.at[i].up, grb, f0, card.ff, 0, gbpr}}, slot + L.up_off);
      to_runs(dt, {{x.at[i].down, drb, 0, Q_EMBD, f0 / dqk, card.ff / dqk}}, slot + L.down_off);
    });
    if (!cached) return;
    for (auto &card : cards_) {
      const Layer &L = card->layers[B.l];
      card->up->copy(L.cache + (uint64_t)B.first * L.slot_bytes, runs[card->c].data(), (uint64_t)cached * L.slot_bytes);
    }
    for (auto &card : cards_) card->up->synchronize();
  };
  std::thread reading;
  std::exception_ptr failed;
  try {
    if (!batches.empty()) read(st[0], batches[0]);
    for (size_t b = 0; b < batches.size(); b++) {
      if (b + 1 < batches.size())
        reading = std::thread([&, b] {
          try {
            read(st[(b + 1) % 2], batches[b + 1]);
          } catch (...) {
            failed = std::current_exception();
          }
        });
      place(st[b % 2], batches[b]);
      if (reading.joinable()) reading.join();
      if (failed) std::rethrow_exception(failed);
    }
  } catch (...) {
    if (reading.joinable()) reading.join();
    throw;
  }
}

// Layer l's host side: which n experts the cards cache, and the others' arena pieces.
std::unique_ptr<Model::HostLayer> Model::host_layer(uint32_t l, uint64_t n, const std::vector<uint64_t> &profile,
                                                    uint64_t stg_bytes, uint32_t positions) {
  auto hl = std::make_unique<HostLayer>();
  const GgufTensor &g = gguf_.tensor(blk(l, "ffn_gate_exps.weight")), &d = gguf_.tensor(blk(l, "ffn_down_exps.weight"));
  hl->shape = {g.type, d.type, Q_EMBD, Q_FF};
  hl->slot_of.assign(Q_EXPERTS, -1);
  hl->stage_of.assign(Q_EXPERTS, -1);
  // The n experts the profile routes to most (ties to the lower id), or the first n without one.
  {
    std::vector<uint32_t> order(Q_EXPERTS);
    for (uint32_t e = 0; e < Q_EXPERTS; e++) order[e] = e;
    if (!profile.empty())
      std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        return profile[l * Q_EXPERTS + a] > profile[l * Q_EXPERTS + b];
      });
    std::sort(order.begin(), order.begin() + n);  // slots in id order
    for (uint32_t i = 0; i < n; i++) hl->slot_of[order[i]] = (int32_t)i;
  }
  const uint64_t eg = g.bytes / Q_EXPERTS, ed = d.bytes / Q_EXPERTS;  // one expert's matrix
  hl->gate_bytes = eg;
  hl->down_bytes = ed;
  hl->gate_row_bytes = g.row_bytes();
  hl->down_parts = ncards_;
  hl->stg_up_off = align256((uint64_t)cards_[0]->ff * g.row_bytes());
  hl->stg_down_off = 2 * hl->stg_up_off;
  hl->stg_bytes = stg_bytes;
  hl->weights.resize(Q_EXPERTS);
  hl->arena.assign(Q_EXPERTS, nullptr);
  for (uint32_t e = 0; e < Q_EXPERTS; e++) {
    if (hl->slot_of[e] >= 0) continue;
    auto *p = arena_piece(align256(2 * eg + ed));  // filled by load_experts' pipeline
    hl->arena[e] = p;
    hl->weights[e] = {p, p + eg, p + 2 * eg, hl->down_parts};
    hl->stage_of[e] = (int32_t)hl->uncached.size();
    hl->uncached.push_back(e);
    hl->pieces.push_back(p);
    info_.arena_bytes += 2 * eg + ed;
  }
  // With sharing, a piece for every expert the cache may leave out, after the others (kv_share.cpp).
  while (hl->pieces.size() < positions) {
    hl->pieces.push_back(arena_piece(align256(2 * eg + ed)));
    info_.arena_bytes += 2 * eg + ed;
  }
  return hl;
}

// y[t] = m x[t] for ntok tokens of a8 activations, rows of m.rows apart.
void Model::matvec(ze::List &list, Card &card, const Mat &m, const uint8_t *act, float *y, uint32_t ntok,
                   ze_event_handle_t signal) {
  uint32_t lanes = lanes_for(m.type, m.K);
  uint32_t stride = (uint32_t)a8_bytes(m.K);
  if (opt_.trace)
    list.label(std::string("matvec ") + ggml_type_label(m.type) + " " + std::to_string(m.rows) + "x" + std::to_string(m.K));
  if (m.rows <= kMvKsRows && lanes == 8) {
    // Few rows: K split over 4 sub-groups of 8 lanes per row (matvec_ks), so that the card has enough of them.
    const uint32_t ks = 4;
    list.launch(kernel(card.c, mv_module(m.type, 8), "matvec_ks"),
                {m.p, m.rows, m.bpr, act, m.K, stride, ntok, y, m.rows, card.grids, ks}, (m.rows + 1) / 2, 1, 1,
                16 * ks, 1, 1, signal);
    return;
  }
  auto k = kernel(card.c, mv_module(m.type, lanes), "matvec");
  uint32_t rows_per_wg = 4 * (16 / lanes);
  list.launch(k, {m.p, m.rows, m.bpr, act, m.K, stride, ntok, y, m.rows, card.grids},
              (m.rows + rows_per_wg - 1) / rows_per_wg, 1, 1, 64, 1, 1, signal);
}

// The decode step for up to W rows, recorded once per card and width (1, 2, 4, 8): the kernels read the step plan
// in host memory, so a list serves any step of up to W rows. Products run for all W rows (their weights are read
// once whatever the count); row-wise kernels skip the rows past the plan's.
void Model::record(Card &card, uint32_t W) {
  // Not in order: the launches between L.together(true) and L.together(false) read the same inputs and write their
  // own outputs, so they may run side by side; each other command waits for those before it.
  auto list = std::make_unique<ze::List>(*rt_, card.c, ze::List::regular, false, false);
  ze::List &L = *list;
  const uint32_t c = card.c, C = ncards_;
  auto common = [&](const char *name) { return kernel(c, "common", name); };
  auto q4 = [&](const char *name) { return kernel(c, "qwen4exp", name); };
  const float one = 1.0f, quarter = 0.25f;
  const uint32_t zero = 0, mode_silu = 1;
  // The kernels read the plan from the card's copy: a word in host memory makes each launch slower
  // (docs/measurements/MEASUREMENTS.md#2026-10-16-bf16-products-occupancy).
  const StepPlan *plan = card.plan;
  const uint32_t *nrows = &plan->nrows;

  if (opt_.trace) {
    card.trace[W] = static_cast<uint64_t *>(rt_->alloc_host(8192 * 8));
    L.trace(card.trace[W], 8192);
  }
  L.copy(card.plan, plan_, sizeof(StepPlan));
  L.timestamp(step_ts_ + 2 * c);
  // The BF16 products, on the XMX units with bf16 activations (xb: 8 rows of K), as the reference rounds a BF16
  // product's activation (docs/spec/MODEL_LAYER.md, section 6) and the prompt path's gemm_bf16 takes it. K split in 4,
  // or in 16 for a few rows.
  uint16_t *no_yb = nullptr;
  auto bf16mv = [&](const uint16_t *W_, uint32_t rows, uint32_t K, const uint16_t *xb, float *y) {
    const uint32_t ks = rows < 128 ? 16 : 4;
    if (opt_.trace) L.label("mv_xmx_bf16 " + std::to_string(rows) + "x" + std::to_string(K));
    L.launch(common("mv_xmx_bf16"), {W_, rows, K, K, xb, nrows, y, rows, zero, one, rows, ks, no_yb}, (rows + 15) / 16, 1,
             1, 16 * ks);
  };
  auto quant = [&](const float *x, uint32_t K, uint32_t ntok, uint8_t *out) {
    uint32_t units = K / 32 * ntok;
    L.launch(common("quantize_a8"), {x, K, K, ntok, out}, (units + 3) / 4, 1, 1, 64);
  };
  // The hyper-connection mix of block i (0: mixer, 1: MoE) into card.x and its a8 card.xa8. down's 320 rows and
  // inject's 4 follow each other in memory: one product (matvec_bf16_ks, a work-group per row) writes lr then the
  // inject values into lri.
  // normed: the exchange before it did the norm (hc_combine_norm).
  auto hc_mix = [&](const float *norm, const uint16_t *down, const uint16_t *up, float *lri, bool normed) {
    const uint32_t post = Q_HC_LR, lw = Q_HC_LR + Q_HC;
    if (!normed) L.launch(q4("rms_norm"), {card.R, norm, plan, card.n, card.xb16}, Q_HC, W, 1, 256);
    // On the XMX units, with bf16 activations as the reference's BF16 products take them: the norm wrote xb16, and
    // the down product writes lr as bf16 for the up product. K split in 8, faster than 16 for 1-8 rows
    // (docs/measurements/MEASUREMENTS.md#2026-10-15-hc-down-split).
    if (opt_.trace) L.label("mv_xmx_bf16 down");
    L.launch(common("mv_xmx_bf16"), {down, lw, (uint32_t)Q_HC_DIM, (uint32_t)Q_HC_DIM, card.xb16, nrows, lri, lw, mode_silu,
                                     quarter, post, 8u, card.lrb16}, (lw + 15) / 16, 1, 1, 16 * 8);
    L.launch(q4("hc_up_mix_xmx"), {up, card.lrb16, card.n, plan, card.x, card.xa8, card.xbf16}, Q_HC_DIM / 128, 1, 1,
             128);
  };
  // next_norm: the norm of the block that follows, done with the combine, or null.
  auto exchange = [&](float *partials, uint32_t layer, uint32_t base, const float *inj, const float *next_norm) {
    if (C > 1) L.wait({(*events_)[ev(layer, base + (1 - c))]});
    const uint32_t lw = Q_HC_LR + Q_HC;
    const uint32_t rows_apart = STEP_MAX_ROWS;  // each card's block in ex1_ and ex2_ holds STEP_MAX_ROWS rows
    if (next_norm)
      L.launch(q4("hc_combine_norm"), {partials, C, rows_apart, inj, lw, next_norm, plan, card.R, card.n, card.xb16}, Q_HC, W, 1,
               256);
    else
      L.launch(q4("hc_combine"), {partials, C, rows_apart, inj, lw, plan, card.R}, Q_EMBD / 64, W, 1, 64);
  };
  const size_t ex_layer = (size_t)C * STEP_MAX_ROWS * Q_EMBD;  // floats per layer in ex1_ and ex2_
  // The mixer's out product, this card's partial sums into ex1_; from 2 rows via card memory and one copy to host
  // (docs/measurements/MEASUREMENTS.md#2026-10-16-out-products-host-writes). With two cards, 2 rows or more exchange card to card
  // (ADR-0030, docs/measurements/MEASUREMENTS.md#2026-10-16-peer-exchange): a card's block of xp is copied to the same block of the
  // other's, and the combine reads both in card order, so the sums match. k: 0 the mixer's exchange, 1 the MoE's.
  const bool peer = C > 1 && W > 1;
  auto xp_block = [&](const Card &to, uint32_t k) { return to.xp + ((size_t)k * C + c) * STEP_MAX_ROWS * Q_EMBD; };
  auto send_peer = [&](uint32_t k, uint32_t layer, uint32_t base) {
    L.copy(xp_block(*cards_[1 - c], k), xp_block(card, k), (size_t)W * Q_EMBD * 4, (*events_)[ev(layer, base + c)]);
  };
  auto out_product = [&](const Mat &m, float *p1, uint32_t layer) {
    if (W == 1) {
      matvec(L, card, m, card.oa8, p1, W, (*events_)[ev(layer, c)]);
      return;
    }
    if (peer) {
      matvec(L, card, m, card.oa8, xp_block(card, 0), W);
      send_peer(0, layer, 0);
      return;
    }
    matvec(L, card, m, card.oa8, card.yo, W);
    L.copy(p1, card.yo, (size_t)W * Q_EMBD * 4, (*events_)[ev(layer, c)]);
  };

  L.launch(q4("hc_init"), {embed_, plan, card.R}, Q_HC_DIM / 64, W, 1, 64);
  bool normed = false;  // the layer's first norm done by the exchange before it
  for (uint32_t l = 0; l < Q_LAYERS; l++) {
    Layer &Ly = card.layers[l];
    if (l == 1) {  // the PLE module (MODEL_IO.md, 1.3), after the host has read the rows (ADR-0015)
      L.wait({(*events_)[ngram_ev()]});
      L.launch(common("to_bf16_rows"), {ngram_u_, Q_EMBD, Q_EMBD, nrows, card.ngb16}, Q_EMBD / 64, W, 1, 64);
      bf16mv(card.ple_k, Q_HC_DIM, Q_EMBD, card.ngb16, card.pk);
      bf16mv(card.ple_v, Q_EMBD, Q_EMBD, card.ngb16, card.pv);
      L.launch(q4("ple_gate"), {card.pk, card.R, card.ple_gk, card.ple_gq, plan, card.gates}, Q_HC, W, 1, 256);
      L.launch(q4("ple_norm"), {card.pv, card.gates, card.ple_gc, plan, card.N}, Q_HC, W, 1, 256);
      L.launch(q4("ple_conv"), {card.N, card.pv, card.gates, card.ple_conv, card.ple_hist, card.ple_stride, plan, card.R},
               Q_HC_DIM / 64, 1, 1, 64);
    }
    hc_mix(Ly.hc_norm[0], Ly.hc_down[0], Ly.hc_up[0], card.inj[0], normed);
    float *p1 = ex1_ + l * ex_layer + (size_t)c * STEP_MAX_ROWS * Q_EMBD;
    if (!Ly.attn) {
      L.together(true);
      matvec(L, card, Ly.qkv, card.xa8, card.m, W);
      matvec(L, card, Ly.z, card.xa8, card.z, W);
      bf16mv(Ly.ab, 2 * card.nvh, Q_EMBD, card.xbf16, card.ab);
      L.together(false);
      // u, the gates and m stay in the layer's own buffers, for a window's commit.
      L.launch(q4("gdn_conv"), {card.m, Ly.conv, Ly.conv_state, Ly.conv_stride, card.nch, 2 * card.nkh, plan, Ly.wu,
                                Ly.wm}, card.nch / Q_GDN_D, 1, 1, Q_GDN_D);
      L.launch(q4("gdn_recur"), {Ly.wu, card.nch, card.ab, Ly.A, Ly.dt, Ly.wgb, card.z, Ly.S, Ly.S_stride, Ly.ssm_norm,
                                 card.nkh, card.nvh, plan, card.o, card.Sw, card.oa8}, card.nvh, W, 1, 4 * Q_GDN_D);
      out_product(Ly.out, p1, l);
    } else {
      L.together(true);
      matvec(L, card, Ly.q, card.xa8, card.Q, W);
      matvec(L, card, Ly.k, card.xa8, card.Kt, W);
      matvec(L, card, Ly.v, card.xa8, card.Vt, W);
      bf16mv(Ly.idx_w, Q_IDX_H * Q_IDX_D + Q_IDX_D, Q_EMBD, card.xbf16, card.idxq);
      L.together(false);
      L.together(true);
      L.launch(q4("attn_prep"), {card.Q, card.Kt, card.Vt, Ly.qn, Ly.kn, card.nq, card.nkv, Ly.kc, Ly.vc, card.ptab,
                                 npp_, plan, rope_}, card.nq + card.nkv, W, 1, Q_ATT_D);
      // The indexer, split by position: each card scores its own blocks and sends its best; both merge the same.
      L.launch(q4("idx_prep"), {card.idxq, Ly.iqn, Ly.raw, card.ptab, npp_, plan, rope_, c, C}, Q_IDX_H + 1, W, 1,
               Q_IDX_D);
      L.together(false);
      L.launch(q4("idx_pool"), {Ly.raw, card.ptab, npp_, Ly.ikn, Ly.pooled, plan, rope_, c, C}, W, 1, 1, Q_IDX_D);
      uint32_t nbl = (opt_.max_ctx / Q_IDX_BLOCK + C - 1) / C;
      L.launch(q4("idx_score"), {card.idxq, Ly.pooled, card.ptab, npp_, nbl, plan, card.scores, c, C},
               (nbl + 63) / 64, W, 1, 64);
      // This card's list goes to its own memory, then in one copy to host memory: faster than the kernel writing
      // host memory a pair at a time.
      uint32_t *cand = icand_ + (size_t)Ly.idx * C * STEP_MAX_ROWS * kIdxCand;
      L.launch(q4("idx_cand"), {card.scores, nbl, plan, card.blocks, card.bscores, card.icand, c, C}, W, 1, 1, 1024);
      L.copy(cand + (size_t)c * STEP_MAX_ROWS * kIdxCand, card.icand, (size_t)W * kIdxCand * 4,
             (*events_)[idx_ev(Ly.idx, c)]);
      if (C > 1) L.wait({(*events_)[idx_ev(Ly.idx, 1 - c)]});
      const uint32_t rows_apart_c = STEP_MAX_ROWS;
      L.launch(q4("idx_merge"), {cand, C, rows_apart_c, plan, card.merged, card.blocks, card.bscores, card.cells,
                                 card.ncells}, W, 1, 1, 1024);
      L.launch(q4("attn_split"), {card.Q, Ly.kc, Ly.vc, card.ptab, npp_, card.nq, card.nkv, card.cells, card.ncells,
                                  plan, card.att}, card.nkv, W, kAttSplits, Q_ATT_D);
      L.launch(q4("attn_merge"), {card.Q, card.nq, card.ncells, plan, card.att, card.o, card.oa8}, card.nq, W, 1,
               Q_ATT_D);
      out_product(Ly.o, p1, l);
    }
    exchange(peer ? card.xp : ex1_ + l * ex_layer, l, 0, card.inj[0] + Q_HC_LR, Ly.hc_norm[1]);

    // The MoE block. The router's chain (the router, then the routed experts) and the shared expert's run side by
    // side, a step of each at a time.
    hc_mix(Ly.hc_norm[1], Ly.hc_down[1], Ly.hc_up[1], card.inj[1], true);
    MissJob *jobs = c == 0 ? jobs_ + (size_t)l * STEP_MAX_ROWS : nullptr;
    uint32_t *nmiss = card.nmiss + l * STEP_MAX_ROWS;
    L.together(true);
    bf16mv(Ly.router, Q_EXPERTS + 1, Q_EMBD, card.xbf16, card.rl);
    matvec(L, card, Ly.sh_gate, card.xa8, card.shg, W);
    matvec(L, card, Ly.sh_up, card.xa8, card.shu, W);
    L.together(false);
    // With cached experts, the router also groups them (router_group), so an expert several rows need is decoded
    // once for all of them.
    L.together(true);
    if (nslots_cached_ == 0)
      L.launch(q4("router"), {card.rl, Ly.slot_map, card.x, card.ids, card.w, card.slots, nmiss, jobs, plan}, W, 1, 1,
               kSgSize);
    else
      L.launch(q4("router_group"), {card.rl, Ly.slot_map, card.x, card.ids, card.w, card.slots, nmiss, jobs, plan,
                                    card.groups, card.ngroups}, 1, 1, 1, 128);
    L.launch(q4("swiglu_a8"), {card.shg, card.shu, card.ff, plan, card.sha8}, (card.ff / 32 + 3) / 4, W, 1, 64);
    L.together(false);
    if (nslots_cached_ > 0) {
      // moe.cl's kernels where they take the type: the same results, faster
      // (docs/measurements/MEASUREMENTS.md#2026-10-14-decode-expert-kernels).
      std::string gm = moe_module(Ly.gu_type, W, false), dm = moe_module(Ly.down_type, W, true);
      uint32_t rpw = gm.empty() ? 4 : MOE_WG_ROWS, lx = gm.empty() ? 64 : kSgSize * MOE_WG_ROWS / MOE_GU_ROWS(W);
      uint32_t gbpr = Q_EMBD / quant_block_values(Ly.gu_type);
      uint32_t K = Q_EMBD, xs = (uint32_t)a8_bytes(Q_EMBD), per = Q_USED;
      L.together(true);
      // moe.cl's also writes h as a8.
      if (gm.empty())
        L.launch(kernel(c, mv_module(Ly.gu_type, 16), "moe_gate_up_g"),
                 {Ly.cache, Ly.slot_bytes, card.groups, card.ngroups, card.ff, gbpr, Ly.up_off, card.xa8, K, xs, per,
                  card.h, card.grids},
                 (card.ff + rpw - 1) / rpw, W * Q_USED, 1, lx);
      else
        L.launch(kernel(c, gm, "moe_gate_up_g"),
                 {Ly.cache, Ly.slot_bytes, card.groups, card.ngroups, card.ff, gbpr, Ly.up_off, card.xa8, K, xs, per,
                  card.h, card.grids, card.ha8},
                 (card.ff + rpw - 1) / rpw, W * Q_USED, 1, lx);
      matvec(L, card, Ly.sh_down, card.sha8, card.s, W);
      L.together(false);
      if (gm.empty()) quant(card.h, card.ff, W * Q_USED, card.ha8);
      uint32_t drpw = dm.empty() ? 16 : MOE_WG_ROWS;
      uint32_t dlx = dm.empty() ? 64 : kSgSize * MOE_WG_ROWS / MOE_DOWN_ROWS(Ly.down_type, W);
      uint32_t dbpr = card.ff / quant_block_values(Ly.down_type);
      uint32_t stride = (uint32_t)a8_bytes(card.ff);
      L.launch(kernel(c, dm.empty() ? mv_module(Ly.down_type, 4) : dm, "moe_down_g"),
               {Ly.cache, Ly.slot_bytes, card.groups, card.ngroups, (uint32_t)Q_EMBD, dbpr, Ly.down_off, card.ha8, card.ff,
                stride, card.y, card.grids},
               (Q_EMBD + drpw - 1) / drpw, W * Q_USED, 1, dlx);
    } else {
      matvec(L, card, Ly.sh_down, card.sha8, card.s, W);
    }
    if (c == 0) L.launch(q4("cpu_wait"), {nmiss, ans_seq_ + 16 * l, plan, stuck_, waited_ + l}, 1, 1, 1, 1);
    float *p2 = ex2_ + l * ex_layer + (size_t)c * STEP_MAX_ROWS * Q_EMBD;
    const float *cpu = c == 0 ? cpu_y_ + (size_t)l * STEP_MAX_ROWS * Q_EMBD : nullptr;
    if (peer) {
      L.launch(q4("moe_combine"), {card.y, card.w, card.slots, card.s, card.rl, cpu, nmiss, plan, xp_block(card, 1)},
               Q_EMBD / 64, W, 1, 64);
      send_peer(1, l, C);
    } else {
      L.launch(q4("moe_combine"), {card.y, card.w, card.slots, card.s, card.rl, cpu, nmiss, plan, p2}, Q_EMBD / 64, W,
               1, 64, 1, 1, (*events_)[ev(l, C + c)]);
    }
    // The next layer's first norm with this combine, but before layer 1, whose PLE module changes R first.
    normed = l + 1 < Q_LAYERS && l + 1 != 1;
    exchange(peer ? card.xp + (size_t)C * STEP_MAX_ROWS * Q_EMBD : ex2_ + l * ex_layer, l, C, card.inj[1] + Q_HC_LR,
             normed ? card.layers[l + 1].hc_norm[0] : nullptr);
  }
  record_head(L, card, plan, W);
  L.timestamp(step_ts_ + 2 * c + 1);
  L.close();
  card.lists[W] = std::move(list);
}

// A window's commit (CommitPlan): the accepted rows' recurrent state, replayed from what the window kept.
void Model::record_commit(Card &card) {
  card.commit = std::make_unique<ze::List>(*rt_, card.c, ze::List::regular);
  ze::List &L = *card.commit;
  auto q4 = [&](const char *name) { return kernel(card.c, "qwen4exp", name); };
  for (auto &Ly : card.layers) {
    if (Ly.attn) continue;
    L.launch(q4("gdn_conv_commit"), {Ly.wm, Ly.conv_state, Ly.conv_stride, card.nch, cplan_}, (card.nch + 63) / 64, 1, 1,
             64);
    L.launch(q4("gdn_commit"), {Ly.wu, card.nch, Ly.wgb, Ly.S, Ly.S_stride, card.nkh, card.nvh, cplan_}, card.nvh, 1, 1,
             4 * Q_GDN_D);
  }
  L.launch(q4("ple_commit"), {card.N, card.ple_hist, card.ple_stride, cplan_}, Q_HC_DIM / 64, 1, 1, 64);
  L.close();
}

void Model::commit(uint32_t slot, uint32_t pos0, uint32_t n) {
  CommitRun run{n, pos0, slot, 0};
  commit_runs(&run, 1);
}

void Model::commit_runs(const CommitRun *runs, uint32_t n) {
  if (n == 0 || n > STEP_MAX_ROWS) throw std::runtime_error("commit out of range");
  for (uint32_t k = 0; k < n; k++)
    if (runs[k].slot >= opt_.slots || runs[k].n == 0 || runs[k].row0 + runs[k].n > STEP_MAX_ROWS)
      throw std::runtime_error("commit out of range");
  wait_commit();
  cplan_->nruns = n;
  std::copy(runs, runs + n, cplan_->runs);
  for (auto &card : cards_) card->commit->execute();
  commit_pending_ = true;
}

void Model::wait_commit() {
  if (!commit_pending_) return;
  for (auto &card : cards_) card->commit->synchronize();
  commit_pending_ = false;
}

// The final mixer and this card's share of the head (MODEL_IO.md, section 2), from the residuals in card.R: each
// row's 64 largest logits into the host's candidate buffers. plan gives the rows.
void Model::record_head(ze::List &L, Card &card, const StepPlan *plan, uint32_t W) {
  const uint32_t c = card.c;
  auto common = [&](const char *name) { return kernel(c, "common", name); };
  auto q4 = [&](const char *name) { return kernel(c, "qwen4exp", name); };
  const float one = 1.0f, quarter = 0.25f;
  const uint32_t zero = 0, mode_silu = 1, gs = Q_EMBD;
  uint16_t *no_bf16 = nullptr;
  L.launch(q4("rms_norm"), {card.R, card.out_norm, plan, card.n, no_bf16}, Q_HC, W, 1, 256);
  {
    uint32_t post = Q_HC_LR, lw = Q_HC_LR + Q_HC;
    L.launch(common("matvec_bf16_ks"), {card.out_down, (uint32_t)Q_HC_LR, (uint32_t)Q_HC_DIM, card.n, (uint32_t)Q_HC_DIM,
                                        &plan->nrows, card.lr, lw, mode_silu, quarter, post}, Q_HC_LR, 1, 1, 256);
    uint32_t lpr = bf16_lanes(Q_HC_LR);
    L.launch(common("matvec_bf16"), {card.out_up, (uint32_t)Q_HC_DIM, (uint32_t)Q_HC_LR, card.lr, lw, W, card.a,
                                     (uint32_t)Q_HC_DIM, zero, one, lpr}, (Q_HC_DIM + 64 / lpr - 1) / (64 / lpr), 1, 1, 64);
  }
  L.launch(q4("hc_mix"), {card.n, card.a, plan, card.x}, Q_EMBD / 64, W, 1, 64);
  uint32_t units = Q_EMBD / 32 * W;
  L.launch(common("quantize_a8"), {card.x, gs, gs, W, card.xa8}, (units + 3) / 4, 1, 1, 64);
  matvec(L, card, card.head, card.xa8, card.logits, W);
  uint32_t *ti = top_idx_ + (size_t)c * STEP_MAX_ROWS * opt_.candidates;
  float *tv = top_val_ + (size_t)c * STEP_MAX_ROWS * opt_.candidates;
  L.launch(common("topk_rows"), {card.logits, card.vocab_n, card.vocab_n, opt_.candidates, card.vocab0, &plan->nrows, ti, tv,
                                 top_n_ + (size_t)c * STEP_MAX_ROWS}, W, 1, 1, 1024);
}

void Model::reset_slot(uint32_t slot) {
  if (slot >= opt_.slots) throw std::runtime_error("slot out of range");
  wait_commit();
  images_[slot].clear();
  blocks_[slot].clear();
  const uint32_t zero = 0;
  for (auto &card : cards_) {
    for (auto &L : card->layers) {
      if (L.attn) continue;
      card->up->fill(L.conv_state + slot * L.conv_stride, &zero, 4, L.conv_stride * 4);
      card->up->fill(L.S + slot * L.S_stride, &zero, 4, L.S_stride * 4);
    }
    card->up->fill(card->ple_hist + slot * card->ple_stride, &zero, 4, card->ple_stride * 4);
    card->up->synchronize();
  }
}

std::vector<Candidate> Model::step(uint32_t slot, uint32_t pos, int32_t token, int32_t p1, int32_t p2) {
  StepIn r{slot, pos, token, p1, p2};
  return std::move(step(&r, 1)[0]);
}

std::vector<std::vector<Candidate>> Model::step(const StepIn *rows, uint32_t n, bool window, bool prompt) {
  if (n == 0 || n > STEP_MAX_ROWS) throw std::runtime_error("a step takes 1 to 8 rows");
  if (window)
    for (uint32_t r = 1; r < n; r++) {
      bool same = rows[r].slot == rows[r - 1].slot;
      for (uint32_t q = 0; q + 1 < r && !same; q++)
        if (rows[q].slot == rows[r].slot) throw std::runtime_error("a window has one run of rows a sequence");
      if (same && rows[r].pos != rows[r - 1].pos + 1)
        throw std::runtime_error("a window's run is consecutive positions of one sequence");
    }
  for (uint32_t r = 0; r < n; r++) {
    if (rows[r].slot >= opt_.slots) throw std::runtime_error("slot out of range");
    if (rows[r].pos >= opt_.max_ctx) throw std::runtime_error("position past the context");
  }
  wait_commit();
  uint32_t W = 1;
  while (W < n) W *= 2;
  last_width_ = W;
  plan_->nrows = n;
  plan_->window = window ? 1 : 0;
  plan_->step++;
  const GgufTensor &te = gguf_.tensor("token_embd.weight");
  auto to_float = ggml_get_type_traits((ggml_type)te.type)->to_float;
  for (uint32_t r = 0; r < n; r++) {
    plan_->rows[r] = {rows[r].slot, rows[r].pos, (uint32_t)rows[r].token, STEP_ROW_LOGITS};
    to_float(te.data + (uint64_t)rows[r].token * te.row_bytes(), embed_ + (size_t)r * Q_EMBD, Q_EMBD);
    bool first = true;  // the slot's first row of the step sets its rotation
    for (uint32_t q = 0; q < r; q++) first = first && rows[q].slot != rows[r].slot;
    if (first) set_rope(rows[r].slot, rows[r].pos, true);
  }
  for (uint32_t i = 0; i < events_->size(); i++) ZE_CHECK(zeEventHostReset((*events_)[i]));
  for (uint32_t c = ncards_; c-- > 0;) cards_[c]->lists.at(W)->execute();
  // The n-gram rows, all read at once while the cards run layer 0 (ADR-0015); they wait for the event before layer 1.
  {
    int32_t t[STEP_MAX_ROWS], p1[STEP_MAX_ROWS], p2[STEP_MAX_ROWS];
    for (uint32_t r = 0; r < n; r++) t[r] = rows[r].token, p1[r] = rows[r].p1, p2[r] = rows[r].p2;
    ngram_->gather_rows(t, p1, p2, n, ngram_u_);
  }
  ZE_CHECK(zeEventHostSignal((*events_)[ngram_ev()]));

  stats_ = {};
  std::vector<CpuExperts::Job> jobs;
  std::vector<float> xs((size_t)n * Q_EMBD);
  const uint32_t step = plan_->step;
  for (uint32_t l = 0; l < Q_LAYERS; l++) {
    // The layer's records (ADR-0011): spin until card 0 publishes each row's for this step; then all the rows'
    // misses at once, each expert's weights read once.
    jobs.clear();
    HostLayer &hl = *host_layers_[l];
    for (uint32_t r = 0; r < n; r++) {
      MissJob &J = jobs_[(size_t)l * STEP_MAX_ROWS + r];
      std::atomic_ref<uint32_t> seq(J.seq);
      auto t0 = std::chrono::steady_clock::now();
      for (uint32_t k = 0; seq.load(std::memory_order_acquire) != step; k++) {
        _mm_pause();
        if ((k & 0xffff) == 0 && std::chrono::steady_clock::now() - t0 > std::chrono::seconds(10))
          throw std::runtime_error("card 0 did not publish layer " + std::to_string(l) + "'s experts");
      }
      const uint32_t nm = J.nmiss;
      if (opt_.adapt_swaps) note_routing(l, J.all, Q_USED, prompt ? kPromptRouting : 1.0f);
      if (nm == 0) continue;  // the skip
      std::memcpy(xs.data() + (size_t)r * Q_EMBD, J.x, Q_EMBD * 4);
      for (uint32_t k = 0; k < nm; k++) jobs.push_back({hl.weights[J.ids[k]], r, J.w[k]});
    }
    if (jobs.empty()) continue;
    stats_.misses += (uint32_t)jobs.size();
    stats_.layers_missed++;
    cpu_->compute_many(hl.shape, jobs, n, xs.data(), Q_EMBD, cpu_y_ + (size_t)l * STEP_MAX_ROWS * Q_EMBD, Q_EMBD);
    std::atomic_ref<uint32_t>(ans_seq_[16 * l]).store(step, std::memory_order_release);
  }
  for (auto &card : cards_) card->lists.at(W)->synchronize();
  if (*stuck_) throw std::runtime_error("card 0 gave up waiting for the CPU's experts");
  stats_.rows = n;
  for (uint32_t c = 0; c < ncards_ && c < 2; c++) {
    const ze::Card &zc = rt_->cards()[c];
    uint64_t ticks = (step_ts_[2 * c + 1] - step_ts_[2 * c]) & zc.timestamp_mask;
    stats_.card_ns[c] = (uint64_t)((double)ticks * 1e9 / (double)zc.timer_hz);
  }
  uint64_t cycles = 0;
  for (uint32_t l = 0; l < Q_LAYERS; l++) cycles += waited_[l];
  stats_.cpu_wait_ns = cycles * 1000 / std::max(1u, rt_->cards()[0].clock_mhz);

  std::vector<std::vector<Candidate>> out(n);
  for (uint32_t r = 0; r < n; r++) {
    auto &o = out[r];
    for (uint32_t c = 0; c < ncards_; c++) {
      size_t base = ((size_t)c * STEP_MAX_ROWS + r) * opt_.candidates;
      for (uint32_t i = 0; i < top_n_[c * STEP_MAX_ROWS + r]; i++) o.push_back({(int32_t)top_idx_[base + i], top_val_[base + i]});
    }
    std::sort(o.begin(), o.end(), [](const Candidate &a, const Candidate &b) {
      return a.logit != b.logit ? a.logit > b.logit : a.token < b.token;
    });
    if (o.size() > opt_.candidates) o.resize(opt_.candidates);
  }
  return out;
}

}  // namespace engine::qwen4exp
