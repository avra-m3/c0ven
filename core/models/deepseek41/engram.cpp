#include "models/deepseek41/engram.h"

#include <fcntl.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#include "ggml.h"
#include "quant.h"

namespace engine::deepseek41 {

namespace {

constexpr uint64_t kPage = 4096;
constexpr uint64_t kEmpty = ~0ull;

std::vector<uint64_t> u64s(const GgufModel &model, const std::string &key, size_t n) {
  const auto v = model.get_ints(key);
  if (v.size() != n) throw std::runtime_error(key + ": " + std::to_string(v.size()) + " values, not " + std::to_string(n));
  return std::vector<uint64_t>(v.begin(), v.end());
}

}  // namespace

EngramParams EngramParams::from_gguf(const GgufModel &model) {
  EngramParams p;
  p.multipliers = u64s(model, "deepseek41.ple.layer_multipliers", kEngramTables * D41_ENG_NGRAM);
  p.offsets = u64s(model, "deepseek41.ple.head_offsets", kEngramTables * D41_ENG_COLS);
  p.moduli = u64s(model, "deepseek41.ple.head_vocab_sizes", kEngramTables * D41_ENG_COLS);
  const auto map = model.get_ints("deepseek41.engram.token_map");
  p.token_map.assign(map.begin(), map.end());
  p.pad = (int32_t)model.get_int("deepseek41.engram.pad_token_id");
  return p;
}

EngramHash::EngramHash(EngramParams p) : p_(std::move(p)) {
  if (p_.multipliers.size() != kEngramTables * D41_ENG_NGRAM || p_.offsets.size() != kEngramTables * D41_ENG_COLS ||
      p_.moduli.size() != kEngramTables * D41_ENG_COLS)
    throw std::runtime_error("Engram hash: the ple arrays do not match two tables of 4-grams in 8 heads");
  for (uint64_t m : p_.moduli)
    if (m == 0) throw std::runtime_error("Engram hash: a column of no rows");
  pad_ = compressed(p_.pad);
  if (pad_ == kNoToken) throw std::runtime_error("Engram hash: the pad token is not in the token map");
}

int32_t EngramHash::compressed(int32_t t) const {
  if (t < 0 || (size_t)t >= p_.token_map.size()) return kNoToken;
  const int32_t c = p_.token_map[(size_t)t];
  return c < 0 ? kNoToken : c;
}

std::array<uint64_t, D41_ENG_COLS> EngramHash::rows(int32_t t, int32_t p1, int32_t p2, int32_t p3,
                                                    uint32_t e) const {
  // Shift s is the token s places back; once one is absent or dead, it and every longer shift are the pad. The
  // products stay below 2^63, so uint64 gives the reference's int64 values.
  const int32_t src[D41_ENG_NGRAM] = {t, p1, p2, p3};
  uint64_t tok[D41_ENG_NGRAM];
  bool blocked = false;
  for (uint32_t s = 0; s < D41_ENG_NGRAM; s++) {
    const int32_t c = compressed(src[s]);
    blocked = blocked || c == kNoToken;
    tok[s] = (uint64_t)(blocked ? pad_ : c);
  }
  const uint64_t *mult = &p_.multipliers[(size_t)e * D41_ENG_NGRAM];
  const uint64_t *mod = &p_.moduli[(size_t)e * D41_ENG_COLS], *off = &p_.offsets[(size_t)e * D41_ENG_COLS];
  std::array<uint64_t, D41_ENG_COLS> r;
  uint64_t rolling = tok[0] * mult[0];
  for (uint32_t o = 1; o < D41_ENG_NGRAM; o++) {
    rolling ^= tok[o] * mult[o];
    for (uint32_t h = 0; h < D41_ENG_HEADS; h++) {
      const uint32_t col = (o - 1) * D41_ENG_HEADS + h;
      r[col] = rolling % mod[col] + off[col];
    }
  }
  return r;
}

void engram_dequant(const uint8_t *row, float *out) {
  ggml_get_type_traits(GGML_TYPE_IQ4_NL)->to_float(row, out, D41_ENG_ROW);
}

EngramReader::EngramReader(const GgufModel &model, uint32_t cache_rows, unsigned readers) {
  const GgufTensor &t = model.tensor("per_layer_token_embd.weight");
  if (t.type != QT_IQ4_NL || t.shape.at(0) != D41_ENG_ROW)
    throw std::runtime_error("the Engram table is not IQ4_NL rows of 256 values");
  data_offset_ = t.offset;
  rows_ = t.rows();
  fd_ = ::open(model.files().at(t.shard).c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
  if (fd_ < 0) throw std::runtime_error("cannot open the Engram table's shard with O_DIRECT (ADR-0015)");
  slots_ = cache_rows;
  tag_.assign(slots_, kEmpty);
  data_.resize((size_t)slots_ * D41_ENG_ROW_BYTES);
  readers_ = std::make_unique<Pool>(readers > 1 ? readers - 1 : 0);
}

EngramReader::~EngramReader() {
  if (fd_ >= 0) ::close(fd_);
}

double EngramReader::read_latency_us() {
  alignas(kPage) static uint8_t buf[kPage];
  std::mt19937_64 rng(7);
  constexpr int kReads = 256;
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < kReads; i++) {
    const uint64_t off = data_offset_ + (rng() % rows_) * D41_ENG_ROW_BYTES;
    if (pread(fd_, buf, kPage, (off_t)(off & ~(kPage - 1))) < 0) return 0;
  }
  return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / kReads;
}

void EngramReader::read_rows(const std::vector<uint64_t> &ids, uint8_t *out) {
  std::atomic<bool> short_read{false};
  readers_->run(ids.size(), [&](uint64_t i) {
    alignas(kPage) static thread_local uint8_t buf[2 * kPage];
    const uint64_t off = data_offset_ + ids[i] * D41_ENG_ROW_BYTES, page = off & ~(kPage - 1);
    const uint64_t in_page = off - page, len = in_page + D41_ENG_ROW_BYTES > kPage ? 2 * kPage : kPage;
    // At the file's end the read comes back short of len; the row is what it must hold.
    const ssize_t got = pread(fd_, buf, len, (off_t)page);
    if (got < (ssize_t)(in_page + D41_ENG_ROW_BYTES)) {
      short_read = true;
      return;
    }
    std::memcpy(out + i * D41_ENG_ROW_BYTES, buf + in_page, D41_ENG_ROW_BYTES);
  });
  if (short_read) throw std::runtime_error("short read from the Engram table");
}

void EngramReader::gather(const uint64_t *ids, uint32_t n, float *out) {
  for (uint32_t k = 0; k < n; k++)
    if (ids[k] >= rows_) throw std::runtime_error("Engram row " + std::to_string(ids[k]) + " past the table's end");
  stats_.rows += n;
  std::vector<const uint8_t *> where(n, nullptr);
  std::unordered_map<uint64_t, uint32_t> want;  // row id -> index in miss
  std::vector<uint64_t> miss;
  std::vector<uint32_t> miss_of(n, UINT32_MAX);
  for (uint32_t k = 0; k < n; k++) {
    const uint32_t slot = (uint32_t)(ids[k] % slots_);
    if (tag_[slot] == ids[k]) {
      where[k] = &data_[(size_t)slot * D41_ENG_ROW_BYTES];
      stats_.hits++;
      continue;
    }
    auto [it, fresh] = want.try_emplace(ids[k], (uint32_t)miss.size());
    if (fresh) miss.push_back(ids[k]);
    miss_of[k] = it->second;
  }
  std::vector<uint8_t> got(miss.size() * D41_ENG_ROW_BYTES);
  if (!miss.empty()) read_rows(miss, got.data());
  stats_.reads += miss.size();
  for (uint32_t k = 0; k < n; k++)
    if (miss_of[k] != UINT32_MAX) where[k] = &got[(size_t)miss_of[k] * D41_ENG_ROW_BYTES];
  // A decode step's rows are dequantised here, since waking the readers would take longer, as for Flash-Next's rows
  // (docs/measurements/MEASUREMENTS.md#2026-10-16-host-glue).
  const auto dequant = [&](uint64_t k) { engram_dequant(where[k], out + k * D41_ENG_ROW); };
  if (n <= kEngramTables * D41_ENG_COLS * STEP_MAX_ROWS) {
    for (uint32_t k = 0; k < n; k++) dequant(k);
  } else {
    readers_->run(n, dequant);
  }
  // A later row in the same slot replaces an earlier one. Only now: where[] may point into a slot.
  for (size_t j = 0; j < miss.size(); j++) {
    const uint32_t slot = (uint32_t)(miss[j] % slots_);
    tag_[slot] = miss[j];
    std::memcpy(&data_[(size_t)slot * D41_ENG_ROW_BYTES], &got[j * D41_ENG_ROW_BYTES], D41_ENG_ROW_BYTES);
  }
}

}  // namespace engine::deepseek41
