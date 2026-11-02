// Level 1 (docs/TESTING.md): deepseek41's Engram hash (core/models/deepseek41/engram.h) against the rows of
// reference/deepseek41/engram_rows.json, which tests/host/deepseek41/test_engram_hash.py writes with its port of
// DeepSeek's engram.py: 1,000 windows, among them positions 0-2, pad tokens and image tokens. Every row must be
// equal. Then an IQ4_NL row of known values through engram_dequant, and EngramReader on a small file made here, in the
// working directory (ctest's is the build tree, which takes O_DIRECT), with rows across 4 KiB boundaries. No card, no
// model file.
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ggml.h"
#include "models/deepseek41/engram.h"
#include "quant.h"
#include "test_util.h"

using namespace engine;
using namespace engine::deepseek41;

namespace {

// The JSON the reference is written in: objects, arrays, integers, strings and booleans.
struct Json {
  std::string text;  // a number's digits, a string's characters, "true" or "false"
  std::vector<Json> items;
  std::map<std::string, Json> fields;

  const Json &operator[](const std::string &key) const {
    auto it = fields.find(key);
    EXPECT(it != fields.end(), "the reference has no \"%s\"", key.c_str());
    return it->second;
  }
  int64_t i64() const { return std::stoll(text); }
  uint64_t u64() const { return std::stoull(text); }
  bool truth() const { return text == "true"; }
};

class Parser {
 public:
  explicit Parser(const std::string &s) : s_(s) {}

  Json value() {
    space();
    Json v;
    EXPECT(p_ < s_.size(), "the reference ends early");
    const char c = s_[p_];
    if (c == '{') {
      p_++;
      while (space(), s_[p_] != '}') {
        Json key = value();
        space();
        EXPECT(s_[p_] == ':', "':' expected at %zu", p_);
        p_++;
        v.fields[key.text] = value();
        space();
        if (s_[p_] == ',') p_++;
      }
      p_++;
    } else if (c == '[') {
      p_++;
      while (space(), s_[p_] != ']') {
        v.items.push_back(value());
        space();
        if (s_[p_] == ',') p_++;
      }
      p_++;
    } else if (c == '"') {
      const size_t end = s_.find('"', p_ + 1);
      v.text = s_.substr(p_ + 1, end - p_ - 1);
      p_ = end + 1;
    } else {
      const size_t start = p_;
      while (p_ < s_.size() && (std::isalnum((unsigned char)s_[p_]) || s_[p_] == '-')) p_++;
      EXPECT(p_ > start, "unexpected '%c' at %zu", c, start);
      v.text = s_.substr(start, p_ - start);
    }
    return v;
  }

 private:
  void space() {
    while (p_ < s_.size() && std::isspace((unsigned char)s_[p_])) p_++;
  }
  const std::string &s_;
  size_t p_ = 0;
};

std::vector<uint64_t> u64s(const Json &a) {
  std::vector<uint64_t> v;
  for (const Json &e : a.items) v.push_back(e.u64());
  return v;
}

// kvalues_iq4nl (docs/spec/QUANT_FORMATS.md, its SHA-256 there begins 61aa4754).
constexpr int kIq4Nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

// An IQ4_NL block (docs/spec/QUANT_FORMATS.md): d (f16), then 16 bytes, byte j's low nibble value j and high nibble
// value j + 16; y = d * kvalues[nibble]. Here block b has d = 0.25 + b / 8 and value k of the row the nibble k % 16,
// or 15 - k % 16 in odd blocks. Exact in f32: an f16 scale times a small integer.
void iq4_nl_check() {
  static_assert(D41_ENG_ROW_BYTES == 8 * 18, "a row is 8 blocks of 18 B");
  uint8_t row[D41_ENG_ROW_BYTES] = {};
  const uint16_t ds[8] = {0x3400, 0x3600, 0x3800, 0x3900, 0x3a00, 0x3b00, 0x3c00, 0x3c80};
  const float dv[8] = {0.25f, 0.375f, 0.5f, 0.625f, 0.75f, 0.875f, 1.0f, 1.125f};
  const auto nibble = [](int k) { return (k / 32) % 2 ? 15 - k % 16 : k % 16; };
  for (int b = 0; b < 8; b++) {
    uint8_t *block = row + 18 * b;
    std::memcpy(block, &ds[b], 2);
    for (int j = 0; j < 16; j++) block[2 + j] = (uint8_t)(nibble(32 * b + j) | nibble(32 * b + j + 16) << 4);
  }
  float out[D41_ENG_ROW];
  engram_dequant(row, out);
  for (int k = 0; k < D41_ENG_ROW; k++) {
    const float want = dv[k / 32] * (float)kIq4Nl[nibble(k)];
    EXPECT(out[k] == want, "IQ4_NL value %d: %g, expected %g", k, out[k], want);
  }
}

template <class T>
void put(std::ofstream &f, T v) {
  f.write(reinterpret_cast<const char *>(&v), sizeof v);
}

// A GGUF of version 3 with the Engram table only. Returns the table's first byte.
uint64_t write_table(const std::string &path, uint64_t rows, const std::vector<uint8_t> &data) {
  std::ofstream f(path, std::ios::binary);
  const std::string name = "per_layer_token_embd.weight";
  put<uint32_t>(f, 0x46554747u);
  put<uint32_t>(f, 3);
  put<uint64_t>(f, 1);
  put<uint64_t>(f, 0);
  put<uint64_t>(f, name.size());
  f.write(name.data(), (std::streamsize)name.size());
  put<uint32_t>(f, 2);
  put<uint64_t>(f, D41_ENG_ROW);
  put<uint64_t>(f, rows);
  put<uint32_t>(f, QT_IQ4_NL);
  put<uint64_t>(f, 0);
  const uint64_t start = ((uint64_t)f.tellp() + 31) / 32 * 32;
  f.write("\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0",
          (std::streamsize)(start - (uint64_t)f.tellp()));
  f.write(reinterpret_cast<const char *>(data.data()), (std::streamsize)data.size());
  return start;
}

// EngramReader against the file's own bytes: every row alone, among them each row across a 4 KiB boundary and the
// file's last; a call with repeats; rows sharing a cache slot; a row past the end.
void reader_check() {
  constexpr uint64_t ROWS = 512;
  std::mt19937_64 rng(41);
  std::vector<uint8_t> data(ROWS * D41_ENG_ROW_BYTES);
  for (auto &b : data) b = (uint8_t)rng();
  for (uint64_t r = 0; r < ROWS; r++)
    for (uint32_t b = 0; b < 8; b++) {
      const uint16_t d = (uint16_t)(0x3400 + r + 7 * b);  // positive and finite, differing by row and block
      std::memcpy(&data[r * D41_ENG_ROW_BYTES + 18 * b], &d, 2);
    }
  const auto want = [&](uint64_t r, uint32_t k) {
    const uint8_t *block = &data[r * D41_ENG_ROW_BYTES + 18 * (k / 32)];
    uint16_t d;
    std::memcpy(&d, block, 2);
    const uint32_t j = k % 32, q = j < 16 ? block[2 + j] & 15 : block[2 + j - 16] >> 4;
    return ggml_fp16_to_fp32(d) * (float)kIq4Nl[q];
  };
  const auto same = [&](const uint64_t *ids, uint32_t n, const std::vector<float> &out) {
    for (uint32_t i = 0; i < n; i++)
      for (uint32_t k = 0; k < D41_ENG_ROW; k++)
        EXPECT(out[i * D41_ENG_ROW + k] == want(ids[i], k), "row %llu value %u: %g, expected %g",
               (unsigned long long)ids[i], k, out[i * D41_ENG_ROW + k], want(ids[i], k));
  };
  const std::string path = "ds41_engram_table.gguf";
  const uint64_t start = write_table(path, ROWS, data);
  {
    GgufModel model(path);
    EXPECT(model.tensor("per_layer_token_embd.weight").offset == start, "the table's offset");
    EngramReader reader(model, 16, 8);
    std::vector<float> out(ROWS * D41_ENG_ROW);
    uint32_t crossing = 0;
    for (uint64_t r = 0; r < ROWS; r++) {
      const uint64_t off = start + r * D41_ENG_ROW_BYTES;
      crossing += off / 4096 != (off + D41_ENG_ROW_BYTES - 1) / 4096;
      reader.gather(&r, 1, out.data());
      same(&r, 1, out);
    }
    EXPECT(crossing >= 10, "%u rows cross a 4 KiB boundary", crossing);
    EXPECT(reader.stats().reads == ROWS && reader.stats().rows == ROWS, "%llu reads of %llu rows",
           (unsigned long long)reader.stats().reads, (unsigned long long)reader.stats().rows);

    // 100, 101 and 102 are out of the 16 slots by now, 511 is in its slot; 100 repeats and is read once.
    uint64_t ids[] = {100, 101, 100, 102, 511, 100};
    const NgramStats s0 = reader.stats();
    reader.gather(ids, 6, out.data());
    same(ids, 6, out);
    EXPECT(reader.stats().reads - s0.reads == 3 && reader.stats().hits - s0.hits == 1,
           "repeats: %llu reads, %llu hits, expected 3 and 1",
           (unsigned long long)(reader.stats().reads - s0.reads), (unsigned long long)(reader.stats().hits - s0.hits));
    // 100 % 16 == 116 % 16: both asked for at once, then each in turn, the slot replaced.
    uint64_t pair[] = {116, 100};
    reader.gather(pair, 2, out.data());
    same(pair, 2, out);
    reader.gather(&pair[0], 1, out.data());
    same(&pair[0], 1, out);
    // More rows than a step's run on the reader threads: the same values.
    std::vector<uint64_t> many(ROWS);
    for (uint64_t r = 0; r < ROWS; r++) many[r] = ROWS - 1 - r;
    reader.gather(many.data(), (uint32_t)ROWS, out.data());
    same(many.data(), (uint32_t)ROWS, out);

    const uint64_t past = ROWS;
    bool refused = false;
    try {
      reader.gather(&past, 1, out.data());
    } catch (const std::runtime_error &) {
      refused = true;
    }
    EXPECT(refused, "a row past the table's end was read");
    EXPECT(reader.read_latency_us() > 0, "read_latency_us");
    std::printf("EngramReader: %llu rows equal the file's bytes, %u of them across a 4 KiB boundary\n",
                (unsigned long long)ROWS, crossing);
  }
  ::unlink(path.c_str());
}

}  // namespace

int main(int argc, char **argv) {
  EXPECT(argc >= 2, "usage: ds41_engram_test <reference/deepseek41/engram_rows.json>");
  std::ifstream f(argv[1]);
  if (!f) return test::skip(std::string("no reference rows at ") + argv[1]);
  std::stringstream ss;
  ss << f.rdbuf();
  const std::string text = ss.str();
  const Json ref = Parser(text).value();

  EngramParams p;
  p.multipliers = u64s(ref["layer_multipliers"]);
  p.offsets = u64s(ref["head_offsets"]);
  p.moduli = u64s(ref["head_vocab_sizes"]);
  p.pad = (int32_t)ref["pad_token_id"].i64();
  // The reference holds the compressed ids of its windows' tokens only; the others are never read.
  p.token_map.assign(D41_VOCAB, 0);
  std::vector<uint8_t> known(D41_VOCAB, 0);
  for (const Json &pair : ref["token_map"].items) {
    const int64_t t = pair.items.at(0).i64();
    p.token_map.at((size_t)t) = (int32_t)pair.items.at(1).i64();
    known[(size_t)t] = 1;
  }
  const EngramHash hash(p);
  EXPECT(hash.pad() == p.token_map[(size_t)p.pad], "the pad's compressed id");

  uint32_t windows = 0, rows = 0;
  for (const Json &w : ref["windows"].items) {
    const int64_t pos = w["pos"].i64();
    int32_t in[D41_ENG_NGRAM];
    for (uint32_t s = 0; s < D41_ENG_NGRAM; s++) {
      // Shift s is absent before position 0, and dead in an image: both are kNoToken.
      const int64_t t = w["tokens"].items.at(s).i64();
      const bool absent = pos - (int64_t)s < 0, dead = w["dead"].items.at(s).truth();
      if (!absent && !dead) EXPECT(known.at((size_t)t), "token %lld is not in the reference's map", (long long)t);
      in[s] = absent || dead ? kNoToken : (int32_t)t;
    }
    const std::vector<uint64_t> want = u64s(w["rows"]);
    EXPECT(want.size() == kEngramTables * D41_ENG_COLS, "window %u: %zu rows", windows, want.size());
    for (uint32_t e = 0; e < kEngramTables; e++) {
      const auto got = hash.rows(in[0], in[1], in[2], in[3], e);
      for (uint32_t c = 0; c < D41_ENG_COLS; c++, rows++)
        EXPECT(got[c] == want[e * D41_ENG_COLS + c], "window %u (pos %lld), table %u, column %u: %llu, expected %llu",
               windows, (long long)pos, e, c, (unsigned long long)got[c],
               (unsigned long long)want[e * D41_ENG_COLS + c]);
    }
    windows++;
  }
  EXPECT(windows == 1000, "%u windows in the reference", windows);
  std::printf("%u windows, %u rows: equal to the port of engram.py\n", windows, rows);

  iq4_nl_check();
  std::printf("an IQ4_NL row of known values: equal\n");
  reader_check();
  return 0;
}
