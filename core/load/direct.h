// Reads of the model's files past the page cache (ADR-0032): O_DIRECT, so a load never
// fills the host's RAM with the file's pages. A range is read whole into an aligned buffer, in pieces of 1 MiB by a
// pool of threads, as many as probe_threads finds fastest on the file's disk (8 on the test machine's T500:
// docs/measurements/MEASUREMENTS.md#2026-10-16-load-read-rates); the caller gets a pointer to its first byte inside the
// buffer. A filesystem that refuses O_DIRECT stops the engine, as for the n-gram table (ADR-0015).
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "util/parallel.h"

namespace engine {

class DirectReader {
 public:
  static constexpr uint64_t kAlign = 4096;
  static constexpr uint64_t kPiece = 1 << 20;

  explicit DirectReader(const std::vector<std::string> &paths, unsigned threads = 8);
  ~DirectReader();
  DirectReader(const DirectReader &) = delete;
  DirectReader &operator=(const DirectReader &) = delete;

  // The thread count of 4, 8 and 16 that reads the file fastest: 256 MiB each, from different places so the disk's
  // own cache does not serve them. rate: that count's bytes a second.
  static unsigned probe_threads(const std::string &path, double *rate);

  // Bytes a buffer needs to hold the read of [offset, offset + bytes): the range widened to kAlign at both ends.
  static uint64_t span(uint64_t offset, uint64_t bytes);
  // Reads [offset, offset + bytes) of file into buf, which is kAlign-aligned and holds span(offset, bytes); returns
  // where the range's first byte landed. Thread-safe, and reads on the caller's thread: for several ranges at once,
  // call it from the pool's threads (pool()).
  const uint8_t *read_one(uint32_t file, uint64_t offset, uint64_t bytes, uint8_t *buf) const;
  // The same, split into pieces of kPiece read by the pool.
  const uint8_t *read(uint32_t file, uint64_t offset, uint64_t bytes, uint8_t *buf);
  Pool &pool() { return pool_; }
  uint64_t bytes_read() const { return bytes_read_; }  // since the reader opened

 private:
  std::vector<int> fds_;
  std::vector<std::string> paths_;
  Pool pool_;
  mutable std::atomic<uint64_t> bytes_read_{0};
};

// A kAlign-aligned buffer, freed with free().
struct AlignedFree {
  void operator()(uint8_t *p) const;
};
using AlignedBuffer = std::unique_ptr<uint8_t, AlignedFree>;
AlignedBuffer aligned_buffer(uint64_t bytes);

}  // namespace engine
