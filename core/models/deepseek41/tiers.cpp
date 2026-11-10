#include "models/deepseek41/tiers.h"

#include <sys/mman.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <utility>

#include "ggml.h"

namespace engine::deepseek41 {

namespace {

constexpr uint64_t kAlign = DirectReader::kAlign;

uint64_t align_down(uint64_t v, uint64_t a) { return v / a * a; }
uint64_t align_up(uint64_t v, uint64_t a) { return align_down(v + a - 1, a); }
uint64_t matrix_offset(const MatrixPlace &m, uint64_t bytes, uint32_t id) { return m.offset + id * bytes; }

// MADV_COLLAPSE on every whole 2 MiB range of [at, at + bytes). A range the kernel cannot collapse now stays in small
// pages. Returns those it collapsed.
uint64_t collapse(uint8_t *at, uint64_t bytes, uint64_t *ranges) {
  uint64_t done = 0;
  for (uint64_t o = 0; o + kHugeBytes <= bytes; o += kHugeBytes, (*ranges)++)
    if (madvise(at + o, kHugeBytes, MADV_COLLAPSE) == 0) done++;
  return done;
}

}  // namespace

AnonMemory::AnonMemory(uint64_t bytes, bool reserved) : map_bytes_(bytes + kHugeBytes), bytes_(bytes) {
  // A private mapping reserves its pages here, so a short pool fails now and not at a later fault.
  void *huge = reserved ? mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0)
                        : MAP_FAILED;
  if (huge != MAP_FAILED) {
    map_ = huge;
    map_bytes_ = bytes;
    at_ = static_cast<uint8_t *>(huge);
    reserved_ = true;
    return;
  }
  map_ = mmap(nullptr, map_bytes_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (map_ == MAP_FAILED) throw std::runtime_error("the arena: mmap of " + std::to_string(map_bytes_) + " B failed");
  // Locked as it fills: a page the kernel swaps out drops the cards' bindings of it, and comes back slowly.
  if (mlock2(map_, map_bytes_, MLOCK_ONFAULT) != 0) {
    lock_errno_ = errno;
    if (!std::getenv("ENGINE_UNLOCKED_ARENA")) {
      munmap(map_, map_bytes_);
      throw std::runtime_error(std::string("the arena cannot be locked in memory (mlock2: ") +
                               std::strerror(lock_errno_) +
                               "); run the container with --ulimit memlock=-1:-1, or set ENGINE_UNLOCKED_ARENA=1 "
                               "to load it unlocked");
    }
  }
  at_ = reinterpret_cast<uint8_t *>(align_up(reinterpret_cast<uintptr_t>(map_), kHugeBytes));
  madvise(at_, bytes_, MADV_HUGEPAGE);
}

AnonMemory::~AnonMemory() {
  if (map_) munmap(map_, map_bytes_);
}

AnonMemory::AnonMemory(AnonMemory &&o) noexcept
    : map_(std::exchange(o.map_, nullptr)), map_bytes_(o.map_bytes_), bytes_(o.bytes_), at_(o.at_),
      lock_errno_(o.lock_errno_), reserved_(o.reserved_) {}

ExpertArena::ExpertArena(uint64_t budget, uint64_t piece_bytes, bool reserved)
    : budget_(budget), piece_bytes_(piece_bytes), reserved_(reserved) {
  if (piece_bytes % kHugeBytes != 0) throw std::runtime_error("the arena: a piece must be whole huge pages");
}

uint8_t *ExpertArena::place(uint64_t need) {
  if (need > piece_bytes_) throw std::runtime_error("the arena: an expert is larger than a piece");
  if (left_ < need) {
    pieces_.emplace_back(piece_bytes_, reserved_);
    if (reserved_ && !pieces_.back().reserved()) {
      std::fprintf(stderr, "engine: the deepseek41 arena: no reserved huge pages for piece %zu; it and the pieces after it "
                           "are in the process's own memory\n", pieces_.size() - 1);
      reserved_ = false;
    }
    piece_used_.push_back(0);
    left_ = piece_bytes_;
  }
  uint8_t *p = pieces_.back().data() + piece_used_.back();
  piece_used_.back() += need;
  left_ -= need;
  used_ += need;
  return p;
}

uint32_t ExpertArena::fill(DirectReader &reader, uint32_t layer, const LayerExperts &l,
                           const std::vector<uint32_t> &ids) {
  const uint64_t need = align_up(l.expert_bytes(), 256);
  const size_t block = ggml_type_size((ggml_type)l.shape.down_type);
  const uint64_t rows = l.shape.n_embd, part_a = kPartABlocks * block, part_b = kPartBBlocks * block;
  if (l.down_bytes != rows * (part_a + part_b))
    throw std::runtime_error("the arena: a down is not " + std::to_string(kPartABlocks + kPartBBlocks) +
                             " blocks a row");
  if (weights_.size() <= layer) weights_.resize(layer + 1);
  if (weights_[layer].empty()) weights_[layer].resize(l.experts);

  // Places first, in the order given, so an expert's address does not depend on the threads.
  std::vector<std::pair<uint32_t, uint8_t *>> places;
  for (uint32_t id : ids) {
    if (id >= l.experts) throw std::runtime_error("the arena: no expert " + std::to_string(id));
    if (used_ + need > budget_) break;
    places.emplace_back(id, place(need));
    experts_++;
  }

  // The worst span of a matrix: its bytes and a block at each end.
  const uint64_t buf_bytes = std::max(l.gate_bytes, l.down_bytes) + 2 * kAlign;
  std::mutex mu;
  std::vector<AlignedBuffer> spare;
  std::exception_ptr failed;
  auto task = [&](uint64_t i) {
    AlignedBuffer buf;
    {
      std::lock_guard<std::mutex> lk(mu);
      if (failed) return;
      if (!spare.empty()) buf = std::move(spare.back()), spare.pop_back();
    }
    try {
      if (!buf) buf = aligned_buffer(buf_bytes);
      const auto [id, to] = places[i];
      const uint8_t *g = reader.read_one(l.gate.file, matrix_offset(l.gate, l.gate_bytes, id), l.gate_bytes, buf.get());
      std::memcpy(to, g, l.gate_bytes);
      const uint8_t *u = reader.read_one(l.up.file, matrix_offset(l.up, l.gate_bytes, id), l.gate_bytes, buf.get());
      std::memcpy(to + l.gate_bytes, u, l.gate_bytes);
      const uint8_t *d = reader.read_one(l.down.file, matrix_offset(l.down, l.down_bytes, id), l.down_bytes, buf.get());
      write_down(l, d, to + 2 * l.gate_bytes);
    } catch (...) {
      std::lock_guard<std::mutex> lk(mu);
      if (!failed) failed = std::current_exception();
    }
    std::lock_guard<std::mutex> lk(mu);
    if (buf) spare.push_back(std::move(buf));
  };
  if (!places.empty()) reader.pool().run(places.size(), task);
  if (failed) std::rethrow_exception(failed);

  for (const auto &[id, to] : places) put(layer, l, id, to);
  return (uint32_t)places.size();
}

// A down in the file's layout as the arena holds it: part A's blocks of every row, then part B's.
void ExpertArena::write_down(const LayerExperts &l, const uint8_t *down, uint8_t *to) {
  const size_t block = ggml_type_size((ggml_type)l.shape.down_type);
  const uint64_t rows = l.shape.n_embd, part_a = kPartABlocks * block, part_b = kPartBBlocks * block;
  uint8_t *b = to + rows * part_a;
  for (uint64_t r = 0; r < rows; r++) {
    std::memcpy(to + r * part_a, down + r * (part_a + part_b), part_a);
    std::memcpy(b + r * part_b, down + r * (part_a + part_b) + part_a, part_b);
  }
}

void ExpertArena::write(const LayerExperts &l, const uint8_t *gate, const uint8_t *up, const uint8_t *down, uint8_t *to) {
  std::memcpy(to, gate, l.gate_bytes);
  std::memcpy(to + l.gate_bytes, up, l.gate_bytes);
  write_down(l, down, to + 2 * l.gate_bytes);
}

uint8_t *ExpertArena::take(uint32_t layer, uint32_t id) {
  if (layer >= weights_.size() || id >= weights_[layer].size()) return nullptr;
  ExpertWeights &w = weights_[layer][id];
  uint8_t *place = const_cast<uint8_t *>(w.gate);
  w = {};
  return place;
}

void ExpertArena::put(uint32_t layer, const LayerExperts &l, uint32_t id, uint8_t *place) {
  if (weights_.size() <= layer) weights_.resize(layer + 1);
  if (weights_[layer].empty()) weights_[layer].resize(l.experts);
  ExpertWeights &w = weights_[layer][id];
  w.gate = place;
  w.up = place + l.gate_bytes;
  w.down = place + 2 * l.gate_bytes;
  w.down_parts = 2;
  w.down_blocks[0] = kPartABlocks;
  w.down_blocks[1] = kPartBBlocks;
}

ArenaReport ExpertArena::settle() {
  const auto t0 = std::chrono::steady_clock::now();
  ArenaReport rep;
  rep.pieces = pieces_.size();
  rep.experts = experts_;
  rep.bytes = used_;
  uint64_t sum = 0;
  for (size_t p = 0; p < pieces_.size(); p++) {
    if (!rep.lock_errno) rep.lock_errno = pieces_[p].lock_errno();
    if (pieces_[p].reserved()) {
      rep.reserved_pieces++;
      rep.ranges += piece_used_[p] / kHugeBytes;
      rep.huge_ranges += piece_used_[p] / kHugeBytes;
    } else {
      rep.huge_ranges += collapse(pieces_[p].data(), piece_used_[p], &rep.ranges);
    }
    // A read, since the fill has written these pages: the first miss on a page pays its fault otherwise.
    for (uint64_t o = 0; o < piece_used_[p]; o += kAlign, rep.touched_pages++)
      sum += *static_cast<volatile const uint8_t *>(pieces_[p].data() + o);
  }
  (void)sum;
  rep.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  const std::string lock = rep.lock_errno ? std::string("not locked (mlock2: ") + std::strerror(rep.lock_errno) +
                                                "), so the kernel may swap it out"
                                          : "locked in memory";
  std::fprintf(stderr, "engine: the deepseek41 arena: %llu experts, %llu of %llu pieces in reserved huge pages, %llu of %llu 2 MiB "
                       "ranges in huge pages, in %.2f s; %s\n",
               (unsigned long long)rep.experts, (unsigned long long)rep.reserved_pieces, (unsigned long long)rep.pieces,
               (unsigned long long)rep.huge_ranges, (unsigned long long)rep.ranges, rep.seconds, lock.c_str());
  return rep;
}

ExpertWeights ExpertArena::weights(uint32_t layer, uint32_t id) const {
  if (layer >= weights_.size() || id >= weights_[layer].size()) return {};
  return weights_[layer][id];
}

uint64_t staging_ring_bytes(uint64_t largest_expert, uint32_t rows) {
  return 2 * std::min<uint64_t>((uint64_t)D41_USED * rows, D41_EXPERTS) * (largest_expert + 3 * kAlign);
}

RingLayout::RingLayout(uint64_t capacity) : capacity_(capacity), high_(align_down(capacity, kAlign)) {}

RingExpert RingLayout::place(uint32_t side, const LayerExperts &l, uint32_t id) {
  const uint64_t spans[3] = {DirectReader::span(matrix_offset(l.gate, l.gate_bytes, id), l.gate_bytes),
                             DirectReader::span(matrix_offset(l.up, l.gate_bytes, id), l.gate_bytes),
                             DirectReader::span(matrix_offset(l.down, l.down_bytes, id), l.down_bytes)};
  if (spans[0] + spans[1] + spans[2] > high_ - low_)
    throw std::runtime_error("the staging ring of " + std::to_string(capacity_) + " B cannot hold this read round");
  RingRead reads[3];
  for (int m = 0; m < 3; m++) {
    if (side == 0) {
      reads[m] = {low_, spans[m]};
      low_ += spans[m];
    } else {
      high_ -= spans[m];
      reads[m] = {high_, spans[m]};
    }
  }
  return {reads[0], reads[1], reads[2]};
}

void RingLayout::release(uint32_t side) {
  if (side == 0)
    low_ = 0;
  else
    high_ = align_down(capacity_, kAlign);
}

SsdReader::SsdReader(const std::vector<std::string> &paths, uint64_t ring_bytes, unsigned threads,
                     std::chrono::nanoseconds wait_limit)
    : reader_(paths, threads), ring_(align_up(ring_bytes, kHugeBytes)), layout_(ring_bytes), wait_limit_(wait_limit) {
  uint64_t ranges = 0;
  collapse(ring_.data(), ring_.bytes(), &ranges);
  // A write, since reading an untouched page would map the shared zero page and the first read into it would fault.
  for (uint64_t o = 0; o < ring_.bytes(); o += kAlign) *static_cast<volatile uint8_t *>(ring_.data() + o) = 0;
  thread_ = std::thread([this] { round_thread(); });
}

SsdReader::~SsdReader() {
  {
    std::lock_guard<std::mutex> lk(mu_);
    stop_ = true;
  }
  cv_.notify_all();
  thread_.join();
}

void SsdReader::add_pieces(const MatrixPlace &first, uint64_t bytes, uint32_t id, const RingRead &read,
                           const uint8_t **weights) {
  const uint64_t offset = matrix_offset(first, bytes, id), a = align_down(offset, kAlign), end = offset + bytes;
  uint8_t *to = ring_.data() + read.at;
  for (uint64_t at = 0; a + at < end; at += DirectReader::kPiece)
    pieces_.push_back({first.file, a + at, std::min(DirectReader::kPiece, end - (a + at)), to + at});
  *weights = to + (offset - a);
}

void SsdReader::start(uint32_t side, const LayerExperts &l, const std::vector<uint32_t> &ids,
                      std::vector<ExpertWeights> &out) {
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (pending_) throw std::runtime_error("the last SSD read round is still running");
    error_.clear();  // the failure of a round wait gave up on, already reported
    pieces_.clear();
    out.resize(ids.size());
    try {
      for (size_t i = 0; i < ids.size(); i++) {
        const RingExpert r = layout_.place(side, l, ids[i]);
        out[i] = ExpertWeights{};
        add_pieces(l.gate, l.gate_bytes, ids[i], r.gate, &out[i].gate);
        add_pieces(l.up, l.gate_bytes, ids[i], r.up, &out[i].up);
        add_pieces(l.down, l.down_bytes, ids[i], r.down, &out[i].down);
      }
    } catch (...) {
      layout_.release(side);
      throw;
    }
    if (pieces_.empty()) return;
    pending_ = true;
    pieces_left_ = pieces_.size();
    rounds_++;
  }
  cv_.notify_all();
}

void SsdReader::wait() {
  std::unique_lock<std::mutex> lk(mu_);
  if (!cv_.wait_for(lk, wait_limit_, [&] { return !pending_; })) {
    const Piece &first = pieces_.front();
    throw std::runtime_error("the SSD reads did not land in " +
                             std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(wait_limit_).count()) +
                             " ms: " + std::to_string(pieces_left_.load()) + " of " + std::to_string(pieces_.size()) +
                             " pieces left, from shard " + std::to_string(first.file) + " offset " +
                             std::to_string(first.offset));
  }
  if (!error_.empty()) throw std::runtime_error(std::exchange(error_, {}));
}

void SsdReader::round_thread() {
  std::unique_lock<std::mutex> lk(mu_);
  for (;;) {
    cv_.wait(lk, [&] { return stop_ || pending_; });
    if (stop_) return;
    lk.unlock();
    std::mutex err_mu;
    std::string err;
    const uint64_t runs0 = reader_.pool().runs_started();
    reader_.pool().run(pieces_.size(), [&](uint64_t i) {
      const Piece &p = pieces_[i];
      const auto t0 = std::chrono::steady_clock::now();
      try {
        if (read_hook_) read_hook_();
        reader_.read_one(p.file, p.offset, p.bytes, p.to);
      } catch (const std::exception &e) {
        std::lock_guard<std::mutex> elk(err_mu);
        if (err.empty()) err = e.what();
      }
      if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(1)) {
        slow_reads_.fetch_add(1, std::memory_order_relaxed);
        std::fprintf(stderr, "engine: an SSD read of an expert took over 1 s\n");
      }
      pieces_left_.fetch_sub(1, std::memory_order_relaxed);
    });
    pool_runs_.fetch_add(reader_.pool().runs_started() - runs0, std::memory_order_relaxed);
    lk.lock();
    error_ = std::move(err);
    pending_ = false;
    cv_.notify_all();
  }
}

MissJobs::MissJobs(std::vector<LayerExperts> layers, const ExpertArena &arena, SsdReader &reader, Pool &cpu,
                   std::chrono::nanoseconds wait_limit)
    : layers_(std::move(layers)), arena_(arena), reader_(reader), pool_(cpu), cpu_(cpu), wait_limit_(wait_limit) {
  thread_ = std::thread([this] { job_thread(); });
}

MissJobs::~MissJobs() {
  {
    std::lock_guard<std::mutex> lk(mu_);
    stop_ = true;
  }
  cv_.notify_all();
  thread_.join();
}

MissCounts MissJobs::run_layer(const MissAsk &ask) {
  if (ask.layer >= layers_.size()) throw std::runtime_error("no such layer's experts: " + std::to_string(ask.layer));
  const LayerExperts &l = layers_[ask.layer];
  const uint32_t real = std::min(ask.real_rows, ask.rows);
  MissCounts c;
  arena_jobs_.clear();
  ssd_jobs_.clear();
  ssd_refs_.clear();
  ssd_ids_.clear();
  for (uint32_t r = 0; r < real; r++) {
    const D41MissJob &J = ask.jobs[r];
    if (J.nmiss > D41_USED) throw std::runtime_error("a miss record holds " + std::to_string(J.nmiss) + " experts");
    for (uint32_t k = 0; k < J.nmiss; k++) {
      if (J.ids[k] < 0 || (uint32_t)J.ids[k] >= l.experts)
        throw std::runtime_error("a miss record names expert " + std::to_string(J.ids[k]));
      const uint32_t id = (uint32_t)J.ids[k];
      c.routings++;
      if (const ExpertWeights w = arena_.weights(ask.layer, id); w.gate) {
        arena_jobs_.push_back({w, r, J.w[k]});
        continue;
      }
      const size_t slot = (size_t)(std::find(ssd_ids_.begin(), ssd_ids_.end(), id) - ssd_ids_.begin());
      if (slot == ssd_ids_.size()) ssd_ids_.push_back(id);
      ssd_refs_.push_back({(uint32_t)slot, r, J.w[k]});
    }
  }
  if (c.routings == 0) return c;
  c.arena_jobs = (uint32_t)arena_jobs_.size();
  c.ssd_jobs = (uint32_t)ssd_refs_.size();
  c.ssd_experts = (uint32_t)ssd_ids_.size();

  const uint64_t runs0 = pool_.runs_started();
  const uint64_t x_stride = sizeof(D41MissJob) / sizeof(float);
  const size_t E = l.shape.n_embd, values = real * E;
  const uint32_t side = ask.layer % 2;
  const bool ssd = !ssd_ids_.empty(), arena = !arena_jobs_.empty();
  if (ssd) {
    reader_.start(side, l, ssd_ids_, ssd_weights_);
    c.read_rounds = 1;
  }
  // The ring stays out of use until its reads have landed, whatever fails first.
  std::exception_ptr failed;
  try {
    if (arena) {
      if (arena_start_hook_) arena_start_hook_();
      arena_y_.resize(values);
      cpu_.compute_many(l.shape, arena_jobs_, real, ask.jobs[0].x, x_stride, arena_y_.data(), E);
    }
    if (arena_hook_) arena_hook_();
  } catch (...) {
    failed = std::current_exception();
  }
  try {
    if (ssd) reader_.wait();
  } catch (...) {
    if (!failed) failed = std::current_exception();
  }
  if (failed) {
    if (ssd) reader_.release(side);
    std::rethrow_exception(failed);
  }
  if (ssd) {
    for (const SsdJob &j : ssd_refs_) ssd_jobs_.push_back({ssd_weights_[j.expert], j.tok, j.w});
    ssd_y_.resize(values);
    try {
      cpu_.compute_many(l.shape, ssd_jobs_, real, ask.jobs[0].x, x_stride, ssd_y_.data(), E);
    } catch (...) {
      reader_.release(side);
      throw;
    }
    reader_.release(side);
  }
  c.cpu_pool_runs = (uint32_t)(pool_.runs_started() - runs0);

  for (uint32_t r = 0; r < ask.rows; r++) {
    float *y = ask.answer + r * ask.answer_stride;
    if (r >= real) {
      std::memset(y, 0, E * sizeof(float));
    } else if (arena && ssd) {
      const float *a = arena_y_.data() + r * E, *s = ssd_y_.data() + r * E;
      for (size_t i = 0; i < E; i++) y[i] = a[i] + s[i];
    } else {
      std::memcpy(y, (arena ? arena_y_ : ssd_y_).data() + r * E, E * sizeof(float));
    }
  }
  if (ask.answer_flag) std::atomic_ref<uint32_t>(*ask.answer_flag).store(ask.step, std::memory_order_release);
  return c;
}

void MissJobs::submit(const MissAsk &ask) {
  {
    std::lock_guard<std::mutex> lk(mu_);
    queue_.push_back(ask);
  }
  cv_.notify_all();
}

void MissJobs::wait() {
  std::unique_lock<std::mutex> lk(mu_);
  if (!idle_.wait_for(lk, wait_limit_, [&] { return queue_.empty() && !busy_; })) {
    const uint32_t layer = busy_ ? running_layer_ : queue_.front().layer;
    std::string what = "layer " + std::to_string(layer) + "'s miss job did not finish in " +
                       std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(wait_limit_).count()) +
                       " ms, with " + std::to_string(queue_.size()) + " more jobs waiting";
    if (!error_.empty()) what += "; before it, " + std::exchange(error_, {});
    queue_.clear();
    abandoned_ = busy_;
    throw std::runtime_error(what);
  }
  if (!error_.empty()) throw std::runtime_error(std::exchange(error_, {}));
}

void MissJobs::job_thread() {
  std::unique_lock<std::mutex> lk(mu_);
  for (;;) {
    cv_.wait(lk, [&] { return stop_ || !queue_.empty(); });
    if (stop_) return;
    const MissAsk ask = queue_.front();
    queue_.pop_front();
    busy_ = true;
    running_layer_ = ask.layer;
    lk.unlock();
    std::string err;
    try {
      run_layer(ask);
    } catch (const std::exception &e) {
      err = "layer " + std::to_string(ask.layer) + ": " + e.what();
    }
    lk.lock();
    if (!std::exchange(abandoned_, false) && error_.empty()) error_ = std::move(err);
    busy_ = false;
    idle_.notify_all();
  }
}

HostDeadlines::HostDeadlines(std::chrono::nanoseconds limit) : limit_(limit) {
  thread_ = std::thread([this] { run(); });
}

HostDeadlines::~HostDeadlines() {
  {
    std::lock_guard<std::mutex> lk(mu_);
    stop_ = true;
  }
  cv_.notify_all();
  thread_.join();
}

uint64_t HostDeadlines::hold(DeadlineEvent &ev, uint32_t *stuck) {
  std::lock_guard<std::mutex> lk(mu_);
  const uint64_t id = next_++;
  holds_.push_back({id, &ev, stuck, false, false, {}});
  cv_.notify_all();
  return id;
}

bool HostDeadlines::release(uint64_t id) {
  std::lock_guard<std::mutex> lk(mu_);
  for (size_t i = 0; i < holds_.size(); i++) {
    if (holds_[i].id != id) continue;
    const bool fired = holds_[i].fired;
    holds_.erase(holds_.begin() + (std::ptrdiff_t)i);
    return fired;
  }
  throw std::runtime_error("releasing a deadline that is not held");
}

void HostDeadlines::run() {
  std::unique_lock<std::mutex> lk(mu_);
  for (;;) {
    if (stop_) return;
    bool polling = false;
    for (Hold &h : holds_) {
      if (h.fired) continue;
      polling = true;
      const auto now = std::chrono::steady_clock::now();
      if (!h.seen && h.ev->reached()) h.seen = true, h.at = now;
      if (!h.seen || now - h.at < limit_) continue;
      // The flag first: the list reads it once the signal lets it go on.
      std::atomic_ref<uint32_t>(*h.stuck).store(1, std::memory_order_release);
      h.ev->signal();
      h.fired = true;
    }
    if (polling)
      cv_.wait_for(lk, std::chrono::milliseconds(1));
    else
      cv_.wait(lk, [&] {
        return stop_ || std::any_of(holds_.begin(), holds_.end(), [](const Hold &h) { return !h.fired; });
      });
  }
}

}  // namespace engine::deepseek41
