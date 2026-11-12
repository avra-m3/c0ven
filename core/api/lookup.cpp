#include "api/lookup.h"

#include <algorithm>
#include <cmath>

namespace engine {

namespace {

uint64_t key(int32_t a, int32_t b, int32_t c) {
  return ((uint64_t)(uint32_t)a << 42) ^ ((uint64_t)(uint32_t)b << 21) ^ (uint64_t)(uint32_t)c;
}

// A window of r rows' time from the widths seen (at[w] > 0, w from 1 to widths - 1), for WindowTimes and RoundTimes
// alike: as lookup.h describes it. 0 while none is seen.
double width_ms(const double *at, uint32_t widths, uint32_t r) {
  r = std::min(r, widths - 1);
  if (r == 0) return 0;
  if (at[r] > 0) return at[r];
  uint32_t lo = r, hi = r;
  while (lo > 0 && at[lo] == 0) lo--;
  while (hi < widths && at[hi] == 0) hi++;
  if (lo > 0 && hi < widths) return at[lo] + (at[hi] - at[lo]) * (r - lo) / (hi - lo);
  if (lo > 0) {
    // Past the widest seen: on a line from the widest, so the fixed part of a window is not scaled with the rows. Times
    // in proportion to the rows overprice a wide window, and a width never chosen is never measured. The gentler of
    // the two slopes, so one slow round at the widest does not price every wider width out.
    uint32_t below = lo - 1, narrowest = 1;
    while (below > 0 && at[below] == 0) below--;
    if (below == 0) return at[lo] * r / lo;
    while (at[narrowest] == 0) narrowest++;
    const double slope = std::min((at[lo] - at[below]) / (lo - below), (at[lo] - at[narrowest]) / (lo - narrowest));
    return at[lo] + std::max(0.0, slope) * (r - lo);
  }
  if (hi == widths) return 0;
  // Below the narrowest seen: on the line through the two narrowest seen, no dearer than the narrowest and no less
  // than in proportion to the rows. A width priced too high is never run, so never timed.
  uint32_t above = hi + 1;
  while (above < widths && at[above] == 0) above++;
  if (above == widths) return at[hi];
  const double line = at[hi] - (at[above] - at[hi]) / (above - hi) * (hi - r);
  return std::max(at[hi] * r / hi, std::min(at[hi], line));
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

double WindowTimes::ms(uint32_t rows) const { return width_ms(ms_, kWidths, rows); }

void RoundTimes::add(uint32_t rows, double ms, uint32_t misses) {
  if (rows == 0 || rows >= kWidths) return;
  Sums &s = w_[rows];
  if (s.rounds++ == 1) s = Sums{0, 0, 0, 0, 0, 0, 2};
  const double keep = 1.0 - 1.0 / 16, m = misses;
  s.n = s.n * keep + 1, s.m = s.m * keep + m, s.t = s.t * keep + ms;
  s.mm = s.mm * keep + m * m, s.mt = s.mt * keep + m * ms, s.tt = s.tt * keep + ms * ms;
  const double r = m / rows;
  misses_row_ = misses_row_ < 0 ? r : misses_row_ + (r - misses_row_) / 8;
  refresh();
}

void RoundTimes::refresh() {
  double num = 0, den = 0, tt = 0, dof = -1;
  seen_ = 0;
  for (uint32_t w = 1; w < kWidths; w++) {
    const Sums &s = w_[w];
    if (s.rounds < 2) continue;
    seen_++;
    num += s.mt - s.m * s.t / s.n;
    den += s.mm - s.m * s.m / s.n;
    tt += s.tt - s.t * s.t / s.n;
    dof += s.n - 1;
  }
  miss_ms_ = 0;
  if (den > 0 && dof > 0) {
    const double c = num / den, se = std::sqrt(std::max(0.0, tt - c * num) / dof / den);
    miss_ms_ = std::max(0.0, c - se);
  }
  double at[kWidths] = {};
  for (uint32_t w = 1; w < kWidths; w++)
    if (w_[w].rounds >= 2) at[w] = std::max(1e-3, (w_[w].t - miss_ms_ * w_[w].m) / w_[w].n);
  // A width never dearer than a wider one seen.
  for (uint32_t w = kWidths - 1, wider = 0; w > 0; w--) {
    if (at[w] == 0) continue;
    if (wider) at[w] = std::min(at[w], at[wider]);
    wider = w;
  }
  for (uint32_t r = 0; r < kWidths; r++) net_[r] = width_ms(at, kWidths, r);
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

std::vector<uint32_t> even_drafts(uint32_t n, uint32_t k, uint32_t rows, uint32_t draft_max, uint32_t first) {
  std::vector<uint32_t> d(n, 0);
  const uint32_t spare = rows > n ? rows - n : 0;
  for (uint32_t i = 0; i < n; i++)
    d[i] = std::min({k, draft_max, spare / n + ((i + n - first % n) % n < spare % n ? 1u : 0u)});
  return d;
}

std::vector<uint32_t> split_drafts(const std::vector<DraftOdds> &slots, uint32_t k, uint32_t rows, uint32_t draft_max,
                                   const std::function<double(const std::vector<uint32_t> &)> &round_ms, uint32_t first,
                                   uint32_t *turns) {
  const uint32_t n = (uint32_t)slots.size();
  if (turns) *turns = 1;
  if (n == 0) return {};
  const uint32_t spare = rows > n ? rows - n : 0, base = std::min({k, draft_max, spare / n});
  const uint32_t more = std::min(k, draft_max) > base ? spare % n : 0;
  std::vector<uint32_t> d(n, base);
  if (more == 0) return d;
  // Each slot's tokens with base drafts, what one more adds, and whether that makes it faster.
  std::vector<double> tokens(n), gain(n);
  std::vector<bool> faster(n);
  const double t0 = round_ms(d);
  for (uint32_t i = 0; i < n; i++) {
    tokens[i] = 1.0 + slots[i].place_expected(base);
    gain[i] = 1.0 + slots[i].place_expected(base + 1) - tokens[i];
    d[i]++;
    faster[i] = gain[i] > 0 && (t0 <= 0 || (tokens[i] + gain[i]) / round_ms(d) > tokens[i] / t0);
    d[i]--;
  }
  // The slots it makes faster first, then by their tokens, fewest first.
  std::vector<uint32_t> order(n);
  for (uint32_t i = 0; i < n; i++) order[i] = i;
  std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
    if (faster[a] != faster[b]) return (bool)faster[a];
    return tokens[a] < tokens[b];
  });
  // Next to each other in that order, a and b are close where one more draft would take a past b.
  auto close = [&](uint32_t a, uint32_t b) { return faster[a] && faster[b] && tokens[a] + gain[a] > tokens[b]; };
  uint32_t lo = more - 1, hi = more;
  if (!close(order[lo], order[hi])) {
    for (uint32_t j = 0; j < more; j++) d[order[j]]++;
    return d;
  }
  while (lo > 0 && close(order[lo - 1], order[lo])) lo--;
  while (hi + 1 < n && close(order[hi], order[hi + 1])) hi++;
  for (uint32_t j = 0; j < lo; j++) d[order[j]]++;
  std::vector<uint32_t> run(order.begin() + lo, order.begin() + hi + 1);
  std::sort(run.begin(), run.end());
  const uint32_t g = (uint32_t)run.size();
  for (uint32_t t = 0; t < more - lo; t++) d[run[(first + t) % g]]++;
  if (turns) *turns = g;
  return d;
}

SharedChoice choose_shared_drafts(const std::vector<DraftOdds> &slots, uint32_t draft_max, uint32_t max_rows,
                                  const RoundTimes &window, const WindowTimes &drafter, uint32_t first,
                                  const std::function<bool(const std::vector<uint32_t> &)> &fits) {
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
  // A count k each of rows rows: the slots' tokens in all and the fewest a slot has, and the round's time; and its
  // split from slot first on (split). Where some slots get one draft more, each slot's tokens and the time are means
  // over the n slots that draft may start from, as the core moves that start each choice.
  auto price = [&](uint32_t k, uint32_t rows, double &total, double &low, std::vector<uint32_t> &split) {
    std::vector<double> tokens(n, 0.0);
    double t = 0;
    uint32_t starts = 1;
    for (uint32_t f = 0; f < starts; f++) {
      std::vector<uint32_t> d = split_drafts(slots, k, rows, draft_max, ms, f, f == 0 ? &starts : nullptr);
      for (uint32_t i = 0; i < n; i++) tokens[i] += 1.0 + slots[i].place_expected(d[i]);
      t += ms(d);
      if (f == first % starts) split = std::move(d);  // split_drafts gives the same from first as from first % starts
    }
    total = 0;
    for (uint32_t i = 0; i < n; i++) {
      total += tokens[i] / starts;
      low = i == 0 ? tokens[i] / starts : std::min(low, tokens[i] / starts);
    }
    return t / starts;
  };
  // The even split: of the counts the spare rows of kEvenRows allow, the one with the most tokens a millisecond, none
  // only once every slot's first draft is measured; kSharedDrafts while fewer than two widths are timed.
  const uint32_t even_rows = std::min(kEvenRows, max_rows), spare = even_rows > n ? even_rows - n : 0;
  const uint32_t most = std::min(draft_max, (spare + n - 1) / n);
  double even_tokens = 0, even_low = 0, even_ms = 0;
  if (window.seen() >= 2) {
    bool measured = true;
    for (const DraftOdds &o : slots) measured = measured && o.first_checks >= kFirstChecks;
    double best = -1;
    for (uint32_t k = measured ? 0 : std::min(1u, most); k <= most; k++) {
      double total = 0, low = 0;
      std::vector<uint32_t> d;
      const double m = price(k, even_rows, total, low, d);
      if (total / m > best) best = total / m, even_tokens = total, even_low = low, even_ms = m, c.even = std::move(d);
    }
  } else {
    c.even = split_drafts(slots, std::min(kSharedDrafts, most), even_rows, draft_max, ms, first);
  }
  if (max_rows <= kEvenRows) return c;
  // Until two widths are timed, kSharedDrafts + 1 each of max_rows rows, where that is more than kEvenRows, so that
  // a wider width is timed.
  if (window.seen() < 2) {
    std::vector<uint32_t> wide = split_drafts(slots, kSharedDrafts + 1, max_rows, draft_max, ms, first);
    if (width(wide) > kEvenRows && (!fits || fits(wide))) c.wide = std::move(wide);
    return c;
  }
  // A count of the whole step's rows that gives more tokens a millisecond in total, and its slowest slot no fewer
  // than the even split's slowest (ahead); else, for the trial's checks, the one with the most tokens a millisecond.
  double best_rate = 0;
  for (uint32_t k = 1; k <= draft_max; k++) {
    if (width(even_drafts(n, k, max_rows, draft_max)) <= kEvenRows) continue;
    double t = 0, low = 0;
    std::vector<uint32_t> d;
    const double m = price(k, max_rows, t, low, d);
    const bool ahead = low / m >= even_low / even_ms && t / m > even_tokens / even_ms;
    const bool better = c.wide.empty() || (ahead != c.wide_ahead ? ahead : t / m > best_rate);
    if (better && (!fits || fits(d))) {
      best_rate = t / m, c.wide_ahead = ahead, c.wide_tokens = t, c.wide_ms = m;
      c.wide = std::move(d);
    }
  }
  if (!c.wide.empty()) c.even_tokens = even_tokens, c.even_ms = even_ms;
  return c;
}

std::vector<uint32_t> window_drafts(const std::vector<uint32_t> &made, uint32_t rows, uint32_t first) {
  const uint32_t n = (uint32_t)made.size();
  std::vector<uint32_t> d(n, 0);
  uint32_t left = rows > n ? rows - n : 0;
  for (bool more = true; more && left > 0;) {
    more = false;
    for (uint32_t t = 0; t < n && left > 0; t++) {
      const uint32_t i = (first + t) % n;
      if (d[i] < made[i]) d[i]++, left--, more = true;
    }
  }
  return d;
}

bool RoundTrial::wide(const SharedChoice &c) {
  if (c.wide_ms <= 0 || c.even_ms <= 0) return rounds_[1] <= rounds_[0];
  picks_++;
  const bool faster =
      c.wide_ahead && c.wide_tokens / c.wide_ms * correction(true) > c.even_tokens / c.even_ms * correction(false);
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
