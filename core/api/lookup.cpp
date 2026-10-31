#include "api/lookup.h"

#include <algorithm>

namespace engine {

namespace {

uint64_t key(int32_t a, int32_t b, int32_t c) {
  return ((uint64_t)(uint32_t)a << 42) ^ ((uint64_t)(uint32_t)b << 21) ^ (uint64_t)(uint32_t)c;
}

}  // namespace

std::vector<int32_t> Lookup::drafts(const std::vector<int32_t> &t, int32_t next, uint32_t cap,
                                     bool (*stop)(int32_t)) {
  index(t);
  std::vector<int32_t> d;
  const uint32_t p = (uint32_t)t.size();  // next's position
  if (p < 2 || cap == 0) return d;
  auto tok = [&](uint32_t i) { return i < p ? t[i] : next; };
  auto it = ends_.find(key(t[p - 2], t[p - 1], next));
  if (it == ends_.end()) return d;
  const std::vector<uint32_t> &ends = it->second;
  uint32_t best = 0, len = 0;
  for (size_t i = ends.size(), seen = 0; i > 0 && seen < 4; i--, seen++) {
    const uint32_t e = ends[i - 1];
    uint32_t n = 3;
    while (n < 64 && e >= n && tok(e - n) == tok(p - n)) n++;
    if (n > len) best = e, len = n;
  }
  if (len < kLookupMin) return d;
  for (uint32_t j = best + 1; j <= p && d.size() < cap && !stop(tok(j)); j++) d.push_back(tok(j));
  return d;
}

void Lookup::index(const std::vector<int32_t> &t) {
  if (indexed_ > t.size()) ends_.clear(), indexed_ = 0;
  for (; indexed_ < t.size(); indexed_++)
    if (indexed_ >= 2) ends_[key(t[indexed_ - 2], t[indexed_ - 1], t[indexed_])].push_back(indexed_);
}

void WindowTimes::add(uint32_t rows, double ms) {
  if (rows == 0 || rows >= kWidths) return;
  double &m = ms_[rows];
  m = m == 0 ? ms : m + (ms - m) / 16;
}

double WindowTimes::ms(uint32_t rows) const {
  rows = std::min(rows, kWidths - 1);
  if (ms_[rows] > 0) return ms_[rows];
  uint32_t lo = rows, hi = rows;
  while (lo > 0 && ms_[lo] == 0) lo--;
  while (hi < kWidths && ms_[hi] == 0) hi++;
  if (lo > 0 && hi < kWidths) return ms_[lo] + (ms_[hi] - ms_[lo]) * (rows - lo) / (hi - lo);
  if (lo > 0) return ms_[lo] * rows / lo;  // past the widest seen: in proportion to the rows
  if (hi < kWidths) return ms_[hi];        // below the narrowest seen: no faster than it
  return 0;
}

void DraftOdds::lookup_round(uint32_t accepted, uint32_t drafts) {
  lookup_accepted = 0.9 * lookup_accepted + accepted;
  lookup_failed = 0.9 * lookup_failed + (accepted < drafts ? 1 : 0);
}

void DraftOdds::layer_round(uint32_t accepted, uint32_t drafts) {
  layer_accepted = 0.9 * layer_accepted + 0.1 * accepted;
  layer_offered = 0.9 * layer_offered + 0.1 * drafts;
}

double DraftOdds::layer_expected(uint32_t k) const {
  const double p = layer_offered > 0 ? std::min(1.0, layer_accepted / layer_offered) : 0.0;
  double e = 0, pi = 1;
  for (uint32_t i = 0; i < k; i++) e += (pi *= p);
  return e;
}

uint32_t choose_drafts(const DraftOdds &odds, uint32_t layer_drafts, uint32_t draft_max, uint32_t lookup_drafts,
                       uint32_t max_rows, const WindowTimes &times) {
  if (lookup_drafts < 2 || max_rows < 2 || times.ms(1) == 0) return 0;
  const double q = odds.lookup_accepted / (odds.lookup_accepted + odds.lookup_failed);
  const uint32_t k = std::min(layer_drafts, max_rows - 1);
  double best = (1.0 + std::min<double>(odds.layer_accepted, k)) / times.ms(k + 1);
  uint32_t take = 0;
  for (uint32_t cap : {draft_max, max_rows - 1}) {
    const uint32_t l = std::min({cap, lookup_drafts, max_rows - 1});
    double e = 0, ql = 1;
    for (uint32_t i = 0; i < l; i++) e += (ql *= q);
    const double r = (1.0 + e) / times.ms(l + 1);
    if (l >= 2 && r > best) best = r, take = l;
  }
  return take;
}

uint32_t choose_shared_drafts(const std::vector<DraftOdds> &slots, uint32_t draft_max, uint32_t max_rows,
                              const WindowTimes &times) {
  const uint32_t n = (uint32_t)slots.size();
  if (n == 0 || n >= max_rows) return 0;
  const uint32_t most = std::min(draft_max, (max_rows - n) / n);
  if (times.ms(n) == 0) return std::min(kSharedDrafts, most);
  double best = 0;
  uint32_t take = 0;
  for (uint32_t k = 0; k <= most; k++) {
    double tokens = 0;
    for (const DraftOdds &o : slots) tokens += 1.0 + o.layer_expected(k);
    const double r = tokens / times.ms(n * (k + 1));
    if (r > best) best = r, take = k;
  }
  return take;
}

}  // namespace engine
