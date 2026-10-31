// The n-gram embedding's rows (docs/spec/MODEL_IO.md, 1.2 and 1.4): the hash of a token and the two before it, and
// the 16 rows of the table it names. The table stays on the SSD (ADR-0015), read with O_DIRECT 4 KiB reads past the
// page cache by a pool of reader threads, many because the SSD needs many reads in flight
// (docs/measurements/MEASUREMENTS.md#2026-10-15-ngram-gather), and rows already read are cached in RAM. Where the host
// has room, the control plane asks for the whole table in RAM instead (ADR-0057).
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "load/gguf.h"
#include "util/parallel.h"

namespace engine::qwen4exp {

constexpr int32_t kNoToken = -1;

struct NgramStats {
  uint64_t rows = 0;    // rows asked for
  uint64_t hits = 0;    // served by the cache
  uint64_t reads = 0;   // read from the SSD
};

class Ngram {
 public:
  // cache_rows: rows the cache holds (raw, ~100 B each); readers: threads reading the SSD. in_ram: read the whole
  // table into the process's memory now, with loader_threads threads, and serve every row from it.
  Ngram(const GgufModel &model, uint32_t cache_rows = 1u << 20, unsigned readers = 128, bool in_ram = false,
        unsigned loader_threads = 8);
  ~Ngram();

  // The 16 row ids for token t at a position whose two predecessors are p1 (just before) and p2 (kNoToken when
  // there is none).
  std::array<uint64_t, 16> rows(int32_t t, int32_t p1, int32_t p2) const;

  // u: 16 x 160 = 2,560 values, row h at u[160 h], dequantised from IQ4_NL.
  void gather(int32_t t, int32_t p1, int32_t p2, float *u);

  // The same for n consecutive tokens (p1 and p2 precede tokens[0]): u[i * 2,560 ...] for token i. Each distinct row
  // is read once.
  void gather_many(const int32_t *tokens, uint32_t n, int32_t p1, int32_t p2, float *u);

  // The same for n tokens with their own predecessors, as a decode step's rows have: all their reads at once.
  void gather_rows(const int32_t *t, const int32_t *p1, const int32_t *p2, uint32_t n, float *u);

  const NgramStats &stats() const { return stats_; }
  bool in_ram() const { return ram_ != nullptr; }
  // The mean time of one 4 KiB row read from the SSD, from 256 reads at random rows one at a time, microseconds.
  double read_latency_us();

 private:
  void read_rows(const std::vector<uint64_t> &ids, uint8_t *out);
  void fill(const std::vector<uint64_t> &ids, uint32_t n, float *u);  // ids: 16 a token

  uint8_t *ram_ = nullptr;  // the whole table's rows, when in RAM
  size_t ram_bytes_ = 0;
  uint64_t rows_ = 0;
  uint64_t mult_[3];
  uint64_t vocab_[16], offset_[16];
  int64_t reset_;
  int fd_ = -1;
  uint64_t data_offset_ = 0, row_bytes_ = 0;
  uint32_t type_ = 0;
  // The cache: direct-mapped, row id % slots; touched only by the calling thread.
  uint32_t slots_ = 0;
  std::vector<uint64_t> tag_;
  std::vector<uint8_t> data_;
  std::unique_ptr<Pool> readers_;
  NgramStats stats_;
};

}  // namespace engine::qwen4exp
