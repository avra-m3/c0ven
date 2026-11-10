// The n-gram table's row cache (ngram.h): direct-mapped, slot = row id % slots, in memory its owner gives it, so the
// owner can put it in reserved huge pages. The memory holds the slots' tags, then their rows. One thread uses it.
#pragma once

#include <cstdint>
#include <cstring>

namespace engine::qwen4exp {

class RowCache {
 public:
  static constexpr uint64_t kEmpty = ~0ull;

  // The bytes a cache of `slots` rows of `row_bytes` takes, and the slots that `bytes` hold.
  static uint64_t bytes_for(uint64_t slots, uint64_t row_bytes) { return slots * (sizeof(uint64_t) + row_bytes); }
  static uint64_t slots_in(uint64_t bytes, uint64_t row_bytes) { return bytes / (sizeof(uint64_t) + row_bytes); }

  // mem: at least bytes_for(slots, row_bytes), 8-byte aligned; owned by the caller and kept while the cache is used.
  void init(uint8_t *mem, uint64_t slots, uint64_t row_bytes) {
    tag_ = reinterpret_cast<uint64_t *>(mem);
    data_ = mem + slots * sizeof(uint64_t);
    slots_ = slots;
    row_bytes_ = row_bytes;
    std::memset(tag_, 0xff, slots * sizeof(uint64_t));
  }

  // The cached row's bytes, or null.
  const uint8_t *find(uint64_t id) const {
    const uint64_t s = id % slots_;
    return tag_[s] == id ? data_ + s * row_bytes_ : nullptr;
  }

  // Keeps the row in its slot, in place of the one there.
  void put(uint64_t id, const uint8_t *row) {
    const uint64_t s = id % slots_;
    tag_[s] = id;
    std::memcpy(data_ + s * row_bytes_, row, row_bytes_);
  }

  uint64_t slots() const { return slots_; }

 private:
  uint64_t *tag_ = nullptr;
  uint8_t *data_ = nullptr;
  uint64_t slots_ = 0, row_bytes_ = 0;
};

}  // namespace engine::qwen4exp
