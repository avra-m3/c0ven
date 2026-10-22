#include "ngram.h"

#include <fcntl.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "ggml.h"
#include "quant.h"

namespace engine::qwen4exp {

namespace {
constexpr uint64_t kPage = 4096;
constexpr uint64_t kEmpty = ~0ull;
}  // namespace

Ngram::Ngram(const GgufModel &model, uint32_t cache_rows, unsigned readers) {
  auto m = model.get_ints("qwen4exp.ple.layer_multipliers");
  auto v = model.get_ints("qwen4exp.ple.head_vocab_sizes");
  auto o = model.get_ints("qwen4exp.ple.head_offsets");
  if (m.size() != 3 || v.size() != 16 || o.size() != 16) throw std::runtime_error("unexpected ple.* arrays");
  for (int i = 0; i < 3; i++) mult_[i] = (uint64_t)m[i];
  for (int i = 0; i < 16; i++) vocab_[i] = (uint64_t)v[i], offset_[i] = (uint64_t)o[i];
  reset_ = model.get_int("qwen4exp.ple.eos_token_id");
  const GgufTensor &t = model.tensor("per_layer_token_embd.weight");
  type_ = t.type;
  row_bytes_ = t.row_bytes();
  data_offset_ = t.offset;
  if (row_bytes_ > kPage) throw std::runtime_error("n-gram rows larger than a page");
  // ADR-0015: the table is read past the page cache; a filesystem that refuses O_DIRECT stops the engine.
  fd_ = ::open(model.files().at(t.shard).c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
  if (fd_ < 0) throw std::runtime_error("cannot open the n-gram table's shard with O_DIRECT (ADR-0015)");
  slots_ = cache_rows;
  tag_.assign(slots_, kEmpty);
  data_.resize((size_t)slots_ * row_bytes_);
  readers_ = std::make_unique<Pool>(readers > 1 ? readers - 1 : 0);
}

Ngram::~Ngram() {
  if (fd_ >= 0) close(fd_);
}

std::array<uint64_t, 16> Ngram::rows(int32_t t, int32_t p1, int32_t p2) const {
  // A missing or reset predecessor makes it, and every older one, the reset token.
  uint64_t c0 = (uint64_t)t;
  bool reset1 = p1 < 0 || p1 == reset_;
  uint64_t c1 = reset1 ? (uint64_t)reset_ : (uint64_t)p1;
  bool reset2 = reset1 || p2 < 0 || p2 == reset_;
  uint64_t c2 = reset2 ? (uint64_t)reset_ : (uint64_t)p2;
  uint64_t mixed2 = (c0 * mult_[0]) ^ (c1 * mult_[1]);
  uint64_t mixed3 = mixed2 ^ (c2 * mult_[2]);
  std::array<uint64_t, 16> r;
  for (int h = 0; h < 16; h++) r[h] = ((h < 8 ? mixed2 : mixed3) % vocab_[h]) + offset_[h];
  return r;
}

// Reads the rows ids[i] into out + i * row_bytes, in parallel: one aligned read of the page(s) holding each row.
void Ngram::read_rows(const std::vector<uint64_t> &ids, uint8_t *out) {
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
  std::vector<uint64_t> ids((size_t)n * 16);
  for (uint32_t i = 0; i < n; i++) {
    auto r = rows(t[i], p1[i], p2[i]);
    std::memcpy(&ids[(size_t)i * 16], r.data(), sizeof r);
  }
  fill(ids, n, u);
}

void Ngram::gather_many(const int32_t *tokens, uint32_t n, int32_t p1, int32_t p2, float *u) {
  // Every row the tokens name; the distinct ones not in the cache are read.
  std::vector<uint64_t> ids((size_t)n * 16);
  for (uint32_t i = 0; i < n; i++) {
    int32_t q1 = i >= 1 ? tokens[i - 1] : p1, q2 = i >= 2 ? tokens[i - 2] : i == 1 ? p1 : p2;
    auto r = rows(tokens[i], q1, q2);
    std::memcpy(&ids[(size_t)i * 16], r.data(), sizeof r);
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
    uint32_t slot = (uint32_t)(id % slots_);
    if (tag_[slot] == id) {
      where[k] = &data_[(size_t)slot * row_bytes_];
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
  auto dequantise = [&](uint64_t i) {
    for (int h = 0; h < 16; h++) to_float(where[i * 16 + h], u + i * 2560 + 160 * h, 160);
  };
  if (n <= 8) {
    for (uint32_t i = 0; i < n; i++) dequantise(i);
  } else {
    readers_->run(n, dequantise);
  }
  for (size_t j = 0; j < miss.size(); j++) {
    uint32_t slot = (uint32_t)(miss[j] % slots_);
    tag_[slot] = miss[j];
    std::memcpy(&data_[(size_t)slot * row_bytes_], &got[j * row_bytes_], row_bytes_);
  }
}

}  // namespace engine::qwen4exp
