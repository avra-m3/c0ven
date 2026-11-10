// Level 1 (docs/TESTING.md): the n-gram table's row cache (models/qwen4exp/row_cache.h) in memory the test gives it,
// and its sizes for a cache that fills 17 GiB of reserved huge pages. No card, no model file.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <vector>

#include "models/qwen4exp/row_cache.h"

using engine::qwen4exp::RowCache;

#define CHECK(cond, what)                                                  \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, what); \
      return 1;                                                            \
    }                                                                      \
  } while (0)

int main() {
  // The IQ3_S file's table: 320,001,536 rows of 90 B (160 IQ4_NL values).
  constexpr uint64_t kRow = 90, kRows = 320001536ull, kGiB = 1ull << 30;
  const uint64_t slots = RowCache::slots_in(17 * kGiB, kRow);
  CHECK(slots == 17 * kGiB / 98, "a slot is its row and an 8-byte tag");
  CHECK(slots > (1ull << 27) && slots < kRows, "17 GiB holds more than 134M rows and less than the table");
  CHECK(RowCache::bytes_for(slots, kRow) <= 17 * kGiB, "the slots fit the bytes they came from");

  // A small cache: misses, hits, a row replaced in its slot, and ids past 2^32, as a table of more rows would have.
  constexpr uint64_t kSlots = 1000;
  std::vector<uint64_t> mem((RowCache::bytes_for(kSlots, kRow) + 7) / 8);
  RowCache c;
  c.init(reinterpret_cast<uint8_t *>(mem.data()), kSlots, kRow);
  CHECK(c.slots() == kSlots, "slots");
  uint8_t row[kRow];
  for (uint64_t id : std::initializer_list<uint64_t>{0, 7, kRows - 1, (1ull << 40) + 3}) {
    CHECK(c.find(id) == nullptr, "an empty cache holds no row");
    std::memset(row, (int)(id & 0xff) ^ 0x5a, kRow);
    c.put(id, row);
    const uint8_t *got = c.find(id);
    CHECK(got != nullptr && std::memcmp(got, row, kRow) == 0, "a row put is found, byte for byte");
  }
  std::memset(row, 0x11, kRow);
  c.put(7 + kSlots, row);
  CHECK(c.find(7) == nullptr, "a row in the same slot replaces the one there");
  CHECK(c.find(7 + kSlots) != nullptr && c.find(7 + kSlots)[0] == 0x11, "and is found");
  CHECK(c.find(0) != nullptr, "the other slots keep their rows");
  std::printf("row cache: %llu slots in 17 GiB of 90-byte rows; lookups and replacement as expected\n",
              (unsigned long long)slots);
  return 0;
}
