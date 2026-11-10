// The n-gram embedding's rows (docs/spec/MODEL_IO.md, 1.2 and 1.4): the hash of a token and the two before it, and
// the kNgramHeads rows of the table it names. The table stays on the SSD (ADR-0015), read with O_DIRECT 4 KiB reads past the
// page cache by a pool of reader threads, many because the SSD needs many reads in flight
// (docs/measurements/MEASUREMENTS.md#2026-10-15-ngram-gather), and rows already read are cached in RAM. Where the host
// has room, the control plane asks for the whole table in RAM instead (ADR-0057), and with --hugepages it may give the
// table or its cache the reserved huge pages the arena leaves (ADR-0070).
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "api/model.h"
#include "load/gguf.h"
#include "models/qwen4exp/row_cache.h"
#include "util/parallel.h"

namespace engine::qwen4exp {

using engine::kNoToken;
using engine::NgramStats;

// The table's rows a token takes: the hash's heads. The file's ple.head_vocab_sizes has to have this many.
constexpr uint32_t kNgramHeads = 16;

class Ngram {
 public:
  // cache_rows: rows the cache holds (raw, ~100 B each); readers: threads reading the SSD. in_ram: read the whole
  // table into the process's memory now, with loader_threads threads, and serve every row from it. huge_bytes: bytes
  // of the host's reserved huge pages, a multiple of their size; with in_ram the table's first huge_bytes are in
  // them and the rest in the process's own memory, without it the cache fills them, when they hold more than
  // cache_rows rows. If the pages cannot be mapped, the log says so, and the table stays in the file with a cache of
  // cache_rows rows.
  Ngram(const GgufModel &model, uint64_t cache_rows = 1u << 20, unsigned readers = 128, bool in_ram = false,
        unsigned loader_threads = 8, uint64_t huge_bytes = 0);
  ~Ngram();

  // The 16 row ids for token t at a position whose two predecessors are p1 (just before) and p2 (kNoToken when
  // there is none).
  std::array<uint64_t, kNgramHeads> rows(int32_t t, int32_t p1, int32_t p2) const;

  // u: 16 rows of the table's width w, the file's embedding, row h at u[w h], dequantised from IQ4_NL.
  void gather(int32_t t, int32_t p1, int32_t p2, float *u);

  // The same for n consecutive tokens (p1 and p2 precede tokens[0]): u[i * 16 w ...] for token i. Each distinct row
  // is read once.
  void gather_many(const int32_t *tokens, uint32_t n, int32_t p1, int32_t p2, float *u);

  // The same for n tokens with their own predecessors, as a decode step's rows have: all their reads at once.
  void gather_rows(const int32_t *t, const int32_t *p1, const int32_t *p2, uint32_t n, float *u);

  const NgramStats &stats() const { return stats_; }
  bool in_ram() const { return ram_ != nullptr; }
  uint64_t cache_rows() const { return cache_.slots(); }
  // The table's or the cache's bytes in reserved huge pages.
  uint64_t huge_bytes() const { return huge_bytes_; }
  // The mean time of one 4 KiB row read from the SSD, from 256 reads at random rows one at a time, microseconds.
  double read_latency_us();

 private:
  void read_rows(const std::vector<uint64_t> &ids, uint8_t *out);
  void fill(const std::vector<uint64_t> &ids, uint32_t n, float *u);  // ids: 16 a token

  void map_ram(uint64_t huge_bytes);
  bool map_cache(uint64_t huge_bytes);

  uint8_t *ram_ = nullptr;  // the whole table's rows, when in RAM
  size_t ram_bytes_ = 0;
  struct Mapping {
    void *at;
    size_t bytes;
  };
  std::vector<Mapping> maps_;  // what the table in RAM or the cache in huge pages mapped, for the destructor
  uint64_t huge_bytes_ = 0;
  uint64_t rows_ = 0;
  uint64_t mult_[3];
  uint64_t vocab_[kNgramHeads], offset_[kNgramHeads];
  uint32_t per_ngram_ = 0;  // ple.heads_per_ngram: the first this many heads hash two tokens, the rest three
  int64_t reset_;
  int fd_ = -1;
  uint64_t data_offset_ = 0, row_bytes_ = 0, row_values_ = 0;
  uint32_t type_ = 0;
  // The cache, touched only by the calling thread; in cache_mem_ unless it is in huge pages.
  RowCache cache_;
  std::vector<uint64_t> cache_mem_;
  std::unique_ptr<Pool> readers_;
  NgramStats stats_;
};

}  // namespace engine::qwen4exp
