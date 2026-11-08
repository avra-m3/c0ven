#include "models/qwen4exp/ngram.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "ggml.h"
#include "load/direct.h"
#include "quant.h"

namespace engine::qwen4exp {

namespace {
constexpr uint64_t kPage = 4096;
}  // namespace

Ngram::Ngram(const GgufModel &model, uint64_t cache_rows, unsigned readers, bool in_ram, unsigned loader_threads,
             uint64_t huge_bytes) {
  auto m = model.get_ints("qwen4exp.ple.layer_multipliers");
  auto v = model.get_ints("qwen4exp.ple.head_vocab_sizes");
  auto o = model.get_ints("qwen4exp.ple.head_offsets");
  if (m.size() != 3 || v.size() != kNgramHeads || o.size() != kNgramHeads)
    throw std::runtime_error("unexpected ple.* arrays");
  per_ngram_ = (uint32_t)model.get_int("qwen4exp.ple.heads_per_ngram");
  if (2 * per_ngram_ != kNgramHeads) throw std::runtime_error("ple.heads_per_ngram is not half the n-gram heads");
  for (int i = 0; i < 3; i++) mult_[i] = (uint64_t)m[i];
  for (uint32_t i = 0; i < kNgramHeads; i++) vocab_[i] = (uint64_t)v[i], offset_[i] = (uint64_t)o[i];
  reset_ = model.get_int("qwen4exp.ple.eos_token_id");
  const GgufTensor &t = model.tensor("per_layer_token_embd.weight");
  type_ = t.type;
  row_bytes_ = t.row_bytes();
  row_values_ = t.shape[0];
  // kNgramHeads is the hash's (rows); a row's width is the file's, and gather's caller sized u for one embedding.
  if (row_values_ * kNgramHeads != (uint64_t)model.get_int("qwen4exp.embedding_length"))
    throw std::runtime_error("the n-gram table's rows do not make one embedding");
  data_offset_ = t.offset;
  if (row_bytes_ > kPage) throw std::runtime_error("n-gram rows larger than a page");
  // ADR-0015: the table is read past the page cache; a filesystem that refuses O_DIRECT stops the engine.
  fd_ = ::open(model.files().at(t.shard).c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
  if (fd_ < 0) throw std::runtime_error("cannot open the n-gram table's shard with O_DIRECT (ADR-0015)");
  readers_ = std::make_unique<Pool>(readers > 1 ? readers - 1 : 0);
  rows_ = t.bytes / row_bytes_;
  if (in_ram) {
    ram_bytes_ = (t.bytes + DirectReader::kAlign - 1) / DirectReader::kAlign * DirectReader::kAlign;
    map_ram(huge_bytes);
  }
  const bool huge_cache =
      !in_ram && huge_bytes && RowCache::slots_in(huge_bytes, row_bytes_) > cache_rows && map_cache(huge_bytes);
  if (!huge_cache) {
    cache_mem_.resize((RowCache::bytes_for(cache_rows, row_bytes_) + 7) / 8);
    cache_.init(reinterpret_cast<uint8_t *>(cache_mem_.data()), cache_rows, row_bytes_);
  }
  if (ram_) {
    // The table read in 1 GiB ranges, each widened to the file's alignment into a buffer, then copied in place.
    DirectReader reader(model.files(), loader_threads);
    constexpr uint64_t kRange = 1ull << 30;
    auto buf = aligned_buffer(kRange + 2 * DirectReader::kAlign);
    const auto t0 = std::chrono::steady_clock::now();
    for (uint64_t at = 0; at < t.bytes; at += kRange) {
      const uint64_t n = std::min(kRange, t.bytes - at);
      std::memcpy(ram_ + at, reader.read(t.shard, data_offset_ + at, n, buf.get()), n);
    }
    std::fprintf(stderr, "engine: the n-gram table in RAM: %.2f GB read in %.2f s\n", t.bytes / 1e9,
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  }
}

Ngram::~Ngram() {
  if (fd_ >= 0) close(fd_);
  for (const Mapping &m : maps_) munmap(m.at, m.bytes);
}

// The table's place in RAM: its first huge_bytes in reserved huge pages and the rest in the process's own memory, at
// one range of addresses, so row i is at ram_ + i x row_bytes. With no huge pages to be had, ram_ stays null.
void Ngram::map_ram(uint64_t huge_bytes) {
  const int anon = MAP_PRIVATE | MAP_ANONYMOUS;
  if (huge_bytes == 0) {
    void *m = mmap(nullptr, ram_bytes_, PROT_READ | PROT_WRITE, anon, -1, 0);
    if (m == MAP_FAILED) throw std::runtime_error("cannot map " + std::to_string(ram_bytes_) + " B for the n-gram table");
    madvise(m, ram_bytes_, MADV_HUGEPAGE);
    ram_ = static_cast<uint8_t *>(m);
    maps_.push_back({m, ram_bytes_});
    return;
  }
  // A range aligned to 1 GiB, the largest huge page, then the two parts mapped over it.
  constexpr uint64_t kAlignHuge = 1ull << 30;
  const uint64_t huge = huge_bytes, rest = huge >= ram_bytes_ ? 0 : ram_bytes_ - huge, span = huge + rest;
  void *r = mmap(nullptr, span + kAlignHuge, PROT_NONE, anon | MAP_NORESERVE, -1, 0);
  if (r == MAP_FAILED) throw std::runtime_error("cannot reserve addresses for the n-gram table");
  uint8_t *base = reinterpret_cast<uint8_t *>((reinterpret_cast<uintptr_t>(r) + kAlignHuge - 1) & ~(kAlignHuge - 1));
  const size_t lead = (size_t)(base - static_cast<uint8_t *>(r)), tail = kAlignHuge - lead;
  if (lead) munmap(r, lead);
  if (tail) munmap(base + span, tail);
  void *h = mmap(base, huge, PROT_READ | PROT_WRITE, anon | MAP_FIXED | MAP_HUGETLB, -1, 0);
  if (h == MAP_FAILED) {
    const int e = errno;
    munmap(base, span);
    std::fprintf(stderr,
                 "engine: the n-gram table: no reserved huge pages for %.2f GiB (mmap: %s); it stays in the file\n",
                 huge / double(1ull << 30), std::strerror(e));
    return;
  }
  maps_.push_back({base, huge});
  if (rest) {
    void *o = mmap(base + huge, rest, PROT_READ | PROT_WRITE, anon | MAP_FIXED, -1, 0);
    if (o == MAP_FAILED) throw std::runtime_error("cannot map " + std::to_string(rest) + " B for the n-gram table");
    madvise(o, rest, MADV_HUGEPAGE);
    maps_.push_back({o, rest});
  }
  ram_ = base;
  huge_bytes_ = huge;
}

// The cache in huge_bytes of reserved huge pages, faulted in now so that no step waits for a page to be cleared.
// false: the pages could not be mapped, and the log says so.
bool Ngram::map_cache(uint64_t huge_bytes) {
  void *m = mmap(nullptr, huge_bytes, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_POPULATE, -1, 0);
  if (m == MAP_FAILED) {
    const int e = errno;
    std::fprintf(stderr,
                 "engine: the n-gram cache: no reserved huge pages for %.2f GiB (mmap: %s); it keeps its default "
                 "size in the process's own memory\n",
                 huge_bytes / double(1ull << 30), std::strerror(e));
    return false;
  }
  maps_.push_back({m, huge_bytes});
  cache_.init(static_cast<uint8_t *>(m), RowCache::slots_in(huge_bytes, row_bytes_), row_bytes_);
  huge_bytes_ = huge_bytes;
  std::fprintf(stderr, "engine: the n-gram cache: %llu rows in %.2f GiB of reserved huge pages\n",
               (unsigned long long)cache_.slots(), huge_bytes / double(1ull << 30));
  return true;
}

double Ngram::read_latency_us() {
  alignas(kPage) static uint8_t buf[kPage];
  std::mt19937_64 rng(7);
  constexpr int kReads = 256;
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < kReads; i++) {
    const uint64_t off = data_offset_ + (rng() % rows_) * row_bytes_;
    if (pread(fd_, buf, kPage, (off_t)(off & ~(kPage - 1))) < 0) return 0;
  }
  return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / kReads;
}

std::array<uint64_t, kNgramHeads> Ngram::rows(int32_t t, int32_t p1, int32_t p2) const {
  // A missing or reset predecessor makes it, and every older one, the reset token.
  uint64_t c0 = (uint64_t)t;
  bool reset1 = p1 < 0 || p1 == reset_;
  uint64_t c1 = reset1 ? (uint64_t)reset_ : (uint64_t)p1;
  bool reset2 = reset1 || p2 < 0 || p2 == reset_;
  uint64_t c2 = reset2 ? (uint64_t)reset_ : (uint64_t)p2;
  uint64_t mixed2 = (c0 * mult_[0]) ^ (c1 * mult_[1]);
  uint64_t mixed3 = mixed2 ^ (c2 * mult_[2]);
  std::array<uint64_t, kNgramHeads> r;
  for (uint32_t h = 0; h < kNgramHeads; h++) r[h] = ((h < per_ngram_ ? mixed2 : mixed3) % vocab_[h]) + offset_[h];
  return r;
}

// Reads the rows ids[i] into out + i * row_bytes, in parallel: one aligned read of the page(s) holding each row.
void Ngram::read_rows(const std::vector<uint64_t> &ids, uint8_t *out) {
  if (ram_) {
    for (size_t i = 0; i < ids.size(); i++) std::memcpy(out + i * row_bytes_, ram_ + ids[i] * row_bytes_, row_bytes_);
    return;
  }
  readers_->run(ids.size(), [&](uint64_t i) {
    alignas(kPage) static thread_local uint8_t buf[2 * kPage];
    uint64_t off = data_offset_ + ids[i] * row_bytes_, page = off & ~(kPage - 1);
    uint64_t len = (off + row_bytes_ > page + kPage) ? 2 * kPage : kPage;
    ssize_t got = pread(fd_, buf, len, (off_t)page);
    if (got < (ssize_t)(off - page + row_bytes_)) throw std::runtime_error("short read from the n-gram table");
    std::memcpy(out + i * row_bytes_, buf + (off - page), row_bytes_);
  });
}

void Ngram::gather(int32_t t, int32_t p1, int32_t p2, float *u) { gather_rows(&t, &p1, &p2, 1, u); }

void Ngram::gather_rows(const int32_t *t, const int32_t *p1, const int32_t *p2, uint32_t n, float *u) {
  std::vector<uint64_t> ids((size_t)n * kNgramHeads);
  for (uint32_t i = 0; i < n; i++) {
    auto r = rows(t[i], p1[i], p2[i]);
    std::memcpy(&ids[(size_t)i * kNgramHeads], r.data(), sizeof r);
  }
  fill(ids, n, u);
}

void Ngram::gather_many(const int32_t *tokens, uint32_t n, int32_t p1, int32_t p2, float *u) {
  // Every row the tokens name; the distinct ones not in the cache are read.
  std::vector<uint64_t> ids((size_t)n * kNgramHeads);
  for (uint32_t i = 0; i < n; i++) {
    int32_t q1 = i >= 1 ? tokens[i - 1] : p1, q2 = i >= 2 ? tokens[i - 2] : i == 1 ? p1 : p2;
    auto r = rows(tokens[i], q1, q2);
    std::memcpy(&ids[(size_t)i * kNgramHeads], r.data(), sizeof r);
  }
  fill(ids, n, u);
}

void Ngram::fill(const std::vector<uint64_t> &ids, uint32_t n, float *u) {
  stats_.rows += ids.size();
  // where[k]: the row's bytes for ids[k] (a cache slot, or the read buffer).
  std::vector<const uint8_t *> where(ids.size(), nullptr);
  std::unordered_map<uint64_t, uint32_t> want;  // row id -> index in miss
  std::vector<uint64_t> miss;
  std::vector<uint32_t> miss_of(ids.size(), UINT32_MAX);
  for (size_t k = 0; k < ids.size(); k++) {
    uint64_t id = ids[k];
    if (const uint8_t *row = cache_.find(id)) {
      where[k] = row;
      stats_.hits++;
      continue;
    }
    auto [it, fresh] = want.try_emplace(id, (uint32_t)miss.size());
    if (fresh) miss.push_back(id);
    miss_of[k] = it->second;
  }
  std::vector<uint8_t> got(miss.size() * row_bytes_);
  if (!miss.empty()) read_rows(miss, got.data());
  stats_.reads += miss.size();
  for (size_t k = 0; k < ids.size(); k++)
    if (miss_of[k] != UINT32_MAX) where[k] = &got[(size_t)miss_of[k] * row_bytes_];
  // Dequantise, then keep the new rows (a later row in the same slot replaces an earlier one). A decode step's few
  // rows are dequantised here: waking the readers would take longer (docs/measurements/MEASUREMENTS.md#2026-10-16-host-glue).
  auto to_float = ggml_get_type_traits((ggml_type)type_)->to_float;
  const uint64_t w = row_values_;
  auto dequantise = [&](uint64_t i) {
    for (uint64_t h = 0; h < kNgramHeads; h++) to_float(where[i * kNgramHeads + h], u + (i * kNgramHeads + h) * w, w);
  };
  if (n <= 8) {
    for (uint32_t i = 0; i < n; i++) dequantise(i);
  } else {
    readers_->run(n, dequantise);
  }
  for (size_t j = 0; j < miss.size(); j++) cache_.put(miss[j], &got[j * row_bytes_]);
}

}  // namespace engine::qwen4exp
