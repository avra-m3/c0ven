#include "models/deepseek41/engram.h"

#include <stdexcept>
#include <string>
#include <utility>

#include "ggml.h"

namespace engine::deepseek41 {

namespace {

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
  ggml_get_type_traits(GGML_TYPE_Q5_K)->to_float(row, out, D41_ENG_ROW);
}

}  // namespace engine::deepseek41
