#include "api/lookup.h"

#include <algorithm>

namespace engine {

namespace {

uint64_t key(int32_t a, int32_t b, int32_t c) {
  return ((uint64_t)(uint32_t)a << 42) ^ ((uint64_t)(uint32_t)b << 21) ^ (uint64_t)(uint32_t)c;
}

}  // namespace

std::vector<int32_t> Lookup::drafts(const std::vector<int32_t> &t, int32_t next, uint32_t cap,
                                     const std::function<bool(int32_t)> &stop) {
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
  m = n_[rows]++ < 2 ? ms : m + (ms - m) / 16;
}

double WindowTimes::ms(uint32_t rows) const {
  rows = std::min(rows, kWidths - 1);
  if (ms_[rows] > 0) return ms_[rows];
  uint32_t lo = rows, hi = rows;
  while (lo > 0 && ms_[lo] == 0) lo--;
  while (hi < kWidths && ms_[hi] == 0) hi++;
  if (lo > 0 && hi < kWidths) return ms_[lo] + (ms_[hi] - ms_[lo]) * (rows - lo) / (hi - lo);
  if (lo > 0) {
    // Past the widest seen: on the line through the two widest seen, so the fixed part of a window is not scaled with
    // the rows. Times in proportion to the rows overprice a wide window, and a width never chosen is never measured.
    uint32_t below = lo - 1;
    while (below > 0 && ms_[below] == 0) below--;
    if (below == 0) return ms_[lo] * rows / lo;
    return ms_[lo] + std::max(0.0, (ms_[lo] - ms_[below]) / (lo - below)) * (rows - lo);
  }
  if (hi < kWidths) return ms_[hi];        // below the narrowest seen: no faster than it
  return 0;
}

void RoundTimes::add(uint32_t rows, double ms, uint32_t misses) {
  if (rows == 0 || rows >= kWidths) return;
  Sums &s = w_[rows];
  if (s.rounds++ == 1) s = Sums{0, 0, 0, 0, 0, 2};
  const double keep = 1.0 - 1.0 / 16, m = misses;
  s.n = s.n * keep + 1, s.m = s.m * keep + m, s.t = s.t * keep + ms;
  s.mm = s.mm * keep + m * m, s.mt = s.mt * keep + m * ms;
  const double r = m / rows;
  misses_row_ = misses_row_ < 0 ? r : misses_row_ + (r - misses_row_) / 8;
  refresh();
}

void RoundTimes::refresh() {
  double num = 0, den = 0;
  seen_ = 0;
  for (uint32_t w = 1; w < kWidths; w++) {
    const Sums &s = w_[w];
    if (s.rounds < 2) continue;
    seen_++;
    num += s.mt - s.m * s.t / s.n;
    den += s.mm - s.m * s.m / s.n;
  }
  miss_ms_ = den > 0 ? std::max(0.0, num / den) : 0.0;
  double at[kWidths] = {};
  for (uint32_t w = 1; w < kWidths; w++)
    if (w_[w].rounds >= 2) at[w] = std::max(1e-3, (w_[w].t - miss_ms_ * w_[w].m) / w_[w].n);
  // A width never dearer than a wider one seen.
  for (uint32_t w = kWidths - 1, wider = 0; w > 0; w--) {
    if (at[w] == 0) continue;
    if (wider) at[w] = std::min(at[w], at[wider]);
    wider = w;
  }
  for (uint32_t r = 0; r < kWidths; r++) {
    net_[r] = 0;
    if (seen_ == 0 || r == 0) continue;
    if (at[r] > 0) {
      net_[r] = at[r];
      continue;
    }
    uint32_t lo = r, hi = r;
    while (lo > 0 && at[lo] == 0) lo--;
    while (hi < kWidths && at[hi] == 0) hi++;
    if (lo > 0 && hi < kWidths) {
      net_[r] = at[lo] + (at[hi] - at[lo]) * (r - lo) / (hi - lo);
    } else if (hi < kWidths) {
      // Below the narrowest seen: on the line through the two narrowest seen, and no less than in proportion to the
      // rows. A width priced too high is never run, so never timed.
      uint32_t above = hi + 1;
      while (above < kWidths && at[above] == 0) above++;
      net_[r] = above == kWidths ? at[hi]
                                 : std::max(at[hi] * r / hi, at[hi] - (at[above] - at[hi]) / (above - hi) * (hi - r));
    } else {
      // Past the widest seen: on the line through the two widest seen, as WindowTimes.
      uint32_t below = lo - 1;
      while (below > 0 && at[below] == 0) below--;
      net_[r] = below == 0 ? at[lo] : at[lo] + std::max(0.0, (at[lo] - at[below]) / (lo - below)) * (r - lo);
    }
  }
}

void DraftOdds::lookup_round(uint32_t accepted, uint32_t drafts) {
  lookup_accepted = 0.9 * lookup_accepted + accepted;
  lookup_failed = 0.9 * lookup_failed + (accepted < drafts ? 1 : 0);
}

void DraftOdds::layer_round(uint32_t accepted, uint32_t drafts) {
  layer_accepted = 0.9 * layer_accepted + 0.1 * accepted;
  // The drafts checked: those accepted and the first one rejected; the ones after it say nothing of the acceptance.
  layer_offered = 0.9 * layer_offered + 0.1 * std::min(drafts, accepted + 1);
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

std::vector<uint32_t> even_drafts(uint32_t n, uint32_t k, uint32_t rows, uint32_t draft_max) {
  std::vector<uint32_t> d(n, 0);
  const uint32_t spare = rows > n ? rows - n : 0;
  for (uint32_t i = 0; i < n; i++) d[i] = std::min({k, draft_max, spare / n + (i < spare % n ? 1u : 0u)});
  return d;
}

std::vector<uint32_t> choose_shared_drafts(const std::vector<DraftOdds> &slots, uint32_t draft_max, uint32_t max_rows,
                                           const RoundTimes &window, const WindowTimes &drafter) {
  const uint32_t n = (uint32_t)slots.size(), even_rows = std::min(kEvenRows, max_rows);
  if (n == 0) return {};
  auto width = [n](const std::vector<uint32_t> &d) {
    uint32_t w = n;
    for (uint32_t x : d) w += x;
    return w;
  };
  if (window.seen() < 2) {
    std::vector<uint32_t> d = even_drafts(n, kSharedDrafts, even_rows, draft_max);
    if (!window.timed(width(d))) return d;
    std::vector<uint32_t> wide = even_drafts(n, kSharedDrafts + 1, max_rows, draft_max);
    return window.timed(width(wide)) ? d : wide;
  }
  auto ms = [&](const std::vector<uint32_t> &d) { return window.ms(width(d)) + drafter.ms(width(d)); };
  auto tokens = [&](uint32_t i, uint32_t d) { return 1.0 + slots[i].layer_expected(d); };
  // The even split, as before a step held 16 rows: the count with the most tokens a millisecond of its window.
  std::vector<uint32_t> even;
  double even_rate = -1;
  for (uint32_t k = 0; k <= draft_max; k++) {
    std::vector<uint32_t> d = even_drafts(n, k, even_rows, draft_max);
    double t = 0;
    for (uint32_t i = 0; i < n; i++) t += tokens(i, d[i]);
    const double m = window.ms(width(d));
    if (t / m > even_rate) even_rate = t / m, even = std::move(d);
  }
  // A count of the whole step's rows, where no slot writes slower than in the even split, both with the draft
  // layer's pass.
  const double even_ms = ms(even);
  std::vector<uint32_t> best = even;
  double best_rate = 0;
  for (uint32_t i = 0; i < n; i++) best_rate += tokens(i, even[i]) / even_ms;
  for (uint32_t k = 1; k <= draft_max; k++) {
    std::vector<uint32_t> d = even_drafts(n, k, max_rows, draft_max);
    const double m = ms(d);
    double t = 0;
    bool slower = false;
    for (uint32_t i = 0; i < n; i++) {
      const double e = tokens(i, d[i]);
      slower |= e / m < tokens(i, even[i]) / even_ms;
      t += e;
    }
    if (!slower && t / m > best_rate) best_rate = t / m, best = std::move(d);
  }
  return best;
}

}  // namespace engine
