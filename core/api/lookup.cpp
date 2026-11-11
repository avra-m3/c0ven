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

void DraftOdds::layer_round(uint32_t accepted, uint32_t drafts, uint32_t asked) {
  // The drafts checked: those accepted and the first one rejected; the ones after it say nothing of the acceptance.
  // A round asked for drafts of which none were made leaves them be.
  if (drafts > 0 || asked == 0) {
    layer_accepted = 0.9 * layer_accepted + 0.1 * accepted;
    layer_offered = 0.9 * layer_offered + 0.1 * std::min(drafts, accepted + 1);
  }
  // The places checked: the same. The places tried: those asked for up to the first the draft layer did not make.
  const uint32_t checked = std::min(drafts, accepted + 1), tried = std::min(asked, drafts + 1);
  first_checks += checked > 0 || tried > 0;
  for (uint32_t i = 0; i < std::min(checked, kOddsPlaces); i++) {
    place_checked[i] = 0.9 * place_checked[i] + 0.1;
    place_accepted[i] = 0.9 * place_accepted[i] + (i < accepted ? 0.1 : 0.0);
  }
  for (uint32_t i = 0; i < std::min(tried, kOddsPlaces); i++) {
    place_tried[i] = 0.9 * place_tried[i] + 0.1;
    place_made[i] = 0.9 * place_made[i] + (i < drafts ? 0.1 : 0.0);
  }
}

double DraftOdds::layer_expected(uint32_t k) const {
  const double p = layer_offered > 0 ? std::min(1.0, layer_accepted / layer_offered) : 0.0;
  double e = 0, pi = 1;
  for (uint32_t i = 0; i < k; i++) e += (pi *= p);
  return e;
}

double DraftOdds::place_expected(uint32_t k) const {
  // A place's acceptance and share made: its rounds' (a weight of 1 less 0.9 to the power of their number), and the
  // place before it's for the rest.
  double p = layer_offered > 0 ? std::min(1.0, layer_accepted / layer_offered) : 0.0, m = 1;
  double e = 0, pi = 1;
  for (uint32_t i = 0; i < k; i++) {
    if (i < kOddsPlaces) {
      p = place_accepted[i] + (1 - place_checked[i]) * p;
      m = place_made[i] + (1 - place_tried[i]) * m;
    }
    e += (pi *= p * m);
  }
  return e;
}

double DraftOdds::place_drafts(uint32_t k) const {
  const std::vector<double> c = drafts_made(k);
  double e = 0;
  for (uint32_t j = 1; j <= k; j++) e += j * c[j];
  return e;
}

std::vector<double> DraftOdds::drafts_made(uint32_t k) const {
  std::vector<double> c(k + 1, 0.0);
  double m = 1, pi = 1;  // the place's share made; the chance of making every draft up to it
  for (uint32_t i = 0; i < k; i++) {
    if (i < kOddsPlaces) m = place_made[i] + (1 - place_tried[i]) * m;
    c[i] = pi * (1 - m);
    pi *= m;
  }
  c[k] = pi;
  return c;
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

SharedChoice choose_shared_drafts(const std::vector<DraftOdds> &slots, uint32_t draft_max, uint32_t max_rows,
                                  const RoundTimes &window, const WindowTimes &drafter) {
  const uint32_t n = (uint32_t)slots.size();
  SharedChoice c;
  if (n == 0) return c;
  auto width = [n](const std::vector<uint32_t> &d) {
    uint32_t w = n;
    for (uint32_t x : d) w += x;
    return w;
  };
  // A round's time: the window's, at the misses a row of the rounds lately, and the draft layer's pass after it, for
  // each width the round may have, by the chance of each slot's drafts made.
  auto ms = [&](const std::vector<uint32_t> &d) {
    std::vector<double> chance(1, 1.0);  // of each number of drafts made in all
    for (uint32_t i = 0; i < n; i++) {
      const std::vector<double> c = slots[i].drafts_made(d[i]);
      std::vector<double> next(chance.size() + d[i], 0.0);
      for (size_t a = 0; a < chance.size(); a++)
        for (size_t b = 0; b < c.size(); b++) next[a + b] += chance[a] * c[b];
      chance = std::move(next);
    }
    double t = 0;
    for (uint32_t x = 0; x < chance.size(); x++)
      if (chance[x] > 0) t += chance[x] * (window.ms(n + x) + drafter.ms(n + x));
    return t;
  };
  // The slots' tokens for drafts d, in all and the fewest.
  auto sum_low = [&](const std::vector<uint32_t> &d, double &low) {
    double t = 0;
    for (uint32_t i = 0; i < n; i++) {
      const double e = 1.0 + slots[i].place_expected(d[i]);
      t += e;
      low = i == 0 ? e : std::min(low, e);
    }
    return t;
  };
  // The even split: of the counts the spare rows of kEvenRows allow, the one with the most tokens a millisecond, none
  // only once every slot's first draft is measured; kSharedDrafts while fewer than two widths are timed.
  const uint32_t even_rows = std::min(kEvenRows, max_rows), spare = even_rows > n ? even_rows - n : 0;
  const uint32_t most = std::min(draft_max, (spare + n - 1) / n);
  uint32_t take = std::min(kSharedDrafts, most);
  if (window.seen() >= 2) {
    bool measured = true;
    for (const DraftOdds &o : slots) measured = measured && o.first_checks >= kFirstChecks;
    double best = 0, low = 0;
    take = 0;
    for (uint32_t k = measured ? 0 : std::min(1u, most); k <= most; k++) {
      const std::vector<uint32_t> d = even_drafts(n, k, even_rows, draft_max);
      const double r = sum_low(d, low) / ms(d);
      if (r > best) best = r, take = k;
    }
  }
  c.even = even_drafts(n, take, even_rows, draft_max);
  if (max_rows <= kEvenRows) return c;
  // Until two widths are timed, kSharedDrafts + 1 each of max_rows rows, where that is more than kEvenRows, so that
  // a wider width is timed.
  if (window.seen() < 2) {
    std::vector<uint32_t> wide = even_drafts(n, kSharedDrafts + 1, max_rows, draft_max);
    if (width(wide) > kEvenRows) c.wide = std::move(wide);
    return c;
  }
  // A count of the whole step's rows that gives more tokens a millisecond in total, and its slowest slot no fewer
  // than the even split's slowest.
  const double even_ms = ms(c.even);
  double even_low = 0;
  const double even_tokens = sum_low(c.even, even_low);
  double best_rate = even_tokens / even_ms;
  for (uint32_t k = 1; k <= draft_max; k++) {
    std::vector<uint32_t> d = even_drafts(n, k, max_rows, draft_max);
    if (width(d) <= kEvenRows) continue;
    const double m = ms(d);
    double low = 0;
    const double t = sum_low(d, low);
    if (low / m >= even_low / even_ms && t / m > best_rate)
      best_rate = t / m, c.wide = std::move(d), c.wide_tokens = t, c.wide_ms = m;
  }
  if (!c.wide.empty()) c.even_tokens = even_tokens, c.even_ms = even_ms;
  return c;
}

bool RoundTrial::wide(const SharedChoice &c) {
  if (c.wide_ms <= 0 || c.even_ms <= 0) return rounds_[1] <= rounds_[0];
  picks_++;
  const bool faster =
      c.wide_tokens / c.wide_ms * correction(true) > c.even_tokens / c.even_ms * correction(false);
  const uint32_t check = rounds_[!faster] < kTrialRounds ? kTrialCheck / 2 : kTrialCheck;
  return picks_ % check == 0 ? !faster : faster;
}

void RoundTrial::add_round(bool wide, double tokens, double ms, double est_tokens, double est_ms) {
  if (est_tokens <= 0 || est_ms <= 0) return;
  if (rounds_[wide] == 0) {
    tokens_[wide] = kTrialPriorRounds * kTrialPrior[wide] * est_tokens, ms_[wide] = kTrialPriorRounds * est_ms;
    est_tokens_[wide] = kTrialPriorRounds * est_tokens, est_ms_[wide] = kTrialPriorRounds * est_ms;
  }
  const double keep = 1 - kTrialDecay;
  tokens_[wide] = tokens_[wide] * keep + tokens, ms_[wide] = ms_[wide] * keep + ms;
  est_tokens_[wide] = est_tokens_[wide] * keep + est_tokens, est_ms_[wide] = est_ms_[wide] * keep + est_ms;
  rounds_[wide]++;
}

void RoundTrial::add_time(bool wide, double ms) {
  if (rounds_[wide] > 0) ms_[wide] += ms;
}

double RoundTrial::correction(bool wide) const {
  if (ms_[wide] <= 0 || est_ms_[wide] <= 0 || est_tokens_[wide] <= 0) return kTrialPrior[wide];
  return (tokens_[wide] / ms_[wide]) / (est_tokens_[wide] / est_ms_[wide]);
}

}  // namespace engine
