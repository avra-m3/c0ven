// deepseek41's Engram rows (spec section 2; DeepSeek's inference/engram.py, NgramHashState): the hash of a token and
// the three before it, which names 24 rows of per_layer_token_embd for each of the two Engram layers, and the rows
// dequantised from IQ4_NL. The table stays in the file, read with O_DIRECT past the page cache (ADR-0015, ADR-0074
// item 7).
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include "api/model.h"
#include "load/gguf.h"
#include "models/deepseek41/layout.h"
#include "util/parallel.h"

namespace engine::deepseek41 {

constexpr uint32_t kEngramTables = 2;  // the Engram layers, 1 and 14 (D41_ENGRAM)

// The hash's numbers, as the file's metadata holds them.
struct EngramParams {
  std::vector<uint64_t> multipliers;  // ple.layer_multipliers: D41_ENG_NGRAM a table, odd
  std::vector<uint64_t> offsets;      // ple.head_offsets: D41_ENG_COLS a table, the column's first row in the file
  std::vector<uint64_t> moduli;       // ple.head_vocab_sizes: D41_ENG_COLS a table, the column's rows (a prime)
  std::vector<int32_t> token_map;     // engram.token_map: each token's compressed id
  int32_t pad = 2;                    // engram.pad_token_id: the token, not its compressed id

  // Reads them from the model's metadata; throws when an array has the wrong length.
  static EngramParams from_gguf(const GgufModel &model);
};

class EngramHash {
 public:
  // Throws when the arrays' lengths do not match the layout, or pad is not a token.
  explicit EngramHash(EngramParams p);

  // The D41_ENG_COLS rows of table e for token t, whose three predecessors are p1 (just before it), p2 and p3; the
  // columns are the 2-gram's 8 heads, then the 3-gram's, then the 4-gram's. A token that is absent (before position 0)
  // or dead (in an image) is kNoToken: it and every token before it hash as the pad token. Rows are the file's.
  std::array<uint64_t, D41_ENG_COLS> rows(int32_t t, int32_t p1, int32_t p2, int32_t p3, uint32_t e) const;

  int32_t compressed(int32_t t) const;  // t's compressed id, or kNoToken when t is not a token
  int32_t pad() const { return pad_; }  // the pad token's compressed id

 private:
  EngramParams p_;
  int32_t pad_;
};

// One row of the table, D41_ENG_ROW values from its D41_ENG_ROW_BYTES of IQ4_NL blocks, by ggml's dequantisation.
void engram_dequant(const uint8_t *row, float *out);

// The table's rows read with O_DIRECT by a pool of reader threads, and a direct-mapped cache of the rows already read
// (ADR-0015). A read is the 4 KiB page holding the row, or the two pages when it crosses one: 144 does not divide
// 4,096. Refuses a file on a filesystem without O_DIRECT, and a table that is not IQ4_NL rows of D41_ENG_ROW values.
class EngramReader {
 public:
  EngramReader(const GgufModel &model, uint32_t cache_rows = 1u << 20, unsigned readers = 128);
  ~EngramReader();
  EngramReader(const EngramReader &) = delete;
  EngramReader &operator=(const EngramReader &) = delete;

  // The n rows ids, dequantised: row k's D41_ENG_ROW values at out + D41_ENG_ROW k. Each distinct row is read once.
  // Throws on a row past the table's end or a short read. One thread at a time.
  void gather(const uint64_t *ids, uint32_t n, float *out);

  const NgramStats &stats() const { return stats_; }
  // The mean time of one 4 KiB read at a random row, from 256 reads one at a time, microseconds.
  double read_latency_us();

 private:
  void read_rows(const std::vector<uint64_t> &ids, uint8_t *out);

  int fd_ = -1;
  uint64_t data_offset_ = 0, rows_ = 0;
  uint32_t slots_ = 0;
  std::vector<uint64_t> tag_;
  std::vector<uint8_t> data_;
  std::unique_ptr<Pool> readers_;
  NgramStats stats_;
};

}  // namespace engine::deepseek41
