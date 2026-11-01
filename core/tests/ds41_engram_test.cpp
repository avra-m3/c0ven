// Level 1 (docs/TESTING.md): deepseek41's Engram hash (core/models/deepseek41/engram.h) against the rows of
// reference/deepseek41/engram_rows.json, which tests/host/deepseek41/test_engram_hash.py writes with its port of
// DeepSeek's engram.py: 1,000 windows, among them positions 0-2, pad tokens and image tokens. Every row must be
// equal. Then a Q5_K row of known values through engram_dequant. No card, no model file.
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "models/deepseek41/engram.h"
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

// A Q5_K block (docs/spec/QUANT_FORMATS.md): d and dmin (f16), 12 bytes of 6-bit scales and mins, the values' fifth
// bits (32 bytes), their low nibbles (128 bytes). Here d = 0.5, dmin = 0.25, every scale 3 and min 2, value k's
// low nibble k % 16 and its fifth bit set for k % 3 == 0: value k is 0.5 * 3 * q - 0.25 * 2.
void q5_k_check() {
  uint8_t b[176] = {};
  const uint16_t d = 0x3800, dmin = 0x3400;  // 0.5, 0.25
  std::memcpy(b, &d, 2);
  std::memcpy(b + 2, &dmin, 2);
  for (int j = 0; j < 4; j++) b[4 + j] = 3, b[8 + j] = 2, b[12 + j] = 0x23;  // sub-blocks 4-7 from the high bytes
  uint8_t *qh = b + 16, *qs = b + 48;
  std::vector<int> q(D41_ENG_ROW);
  for (int k = 0; k < D41_ENG_ROW; k++) {
    // Value k is in 64-value group g = k / 64, half h = (k % 64) / 32, lane l = k % 32.
    const int g = k / 64, h = (k % 64) / 32, l = k % 32, lo = k % 16, hi = k % 3 == 0;
    qs[32 * g + l] |= (uint8_t)(lo << (4 * h));
    qh[l] |= (uint8_t)(hi << (2 * g + h));
    q[k] = lo + 16 * hi;
  }
  float out[D41_ENG_ROW];
  engram_dequant(b, out);
  for (int k = 0; k < D41_ENG_ROW; k++) {
    const float want = 0.5f * 3 * q[k] - 0.25f * 2;
    EXPECT(out[k] == want, "Q5_K value %d: %g, expected %g", k, out[k], want);
  }
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

  q5_k_check();
  std::printf("a Q5_K row of known values: equal\n");
  return 0;
}
