#include "api/draft_choice.h"

#include <algorithm>
#include <cmath>

namespace engine {

namespace {

// The log-odds of p, kept finite.
double log_odds(double p) {
  p = std::clamp(p, 1e-3, 1.0 - 1e-3);
  return std::log(p / (1.0 - p));
}

double sigmoid(double z) { return 1.0 / (1.0 + std::exp(-z)); }

// The order in which share_rows gives out the spare rows: a slot index for each row.
std::vector<uint32_t> share_order(const std::vector<DraftOdds> &slots, const std::vector<uint32_t> &caps,
                                  uint32_t rows, uint32_t max_rows) {
  const size_t n = slots.size();
  std::vector<uint32_t> order, d(n, 0);
  uint32_t spare = max_rows > rows ? max_rows - rows : 0;
  for (size_t i = 0; i < n && spare > 0; i++)
    if (caps[i] > 0) d[i] = 1, spare--, order.push_back((uint32_t)i);
  std::vector<double> next(n);  // the chance that slot i's next draft and those before it are accepted
  for (size_t i = 0; i < n; i++) next[i] = std::pow(slots[i].layer_rate(), d[i] + 1);
  for (; spare > 0; spare--) {
    size_t best = n;
    for (size_t i = 0; i < n; i++)
      if (d[i] < caps[i] && (best == n || next[i] > next[best])) best = i;
    if (best == n) break;
    d[best]++;
    next[best] *= slots[best].layer_rate();
    order.push_back((uint32_t)best);
  }
  return order;
}

}  // namespace

double DraftEstimate::accept(uint32_t place, double p) const {
  return sigmoid(offset(place) + slope_ * log_odds(p));
}

void DraftEstimate::observe(uint32_t place, double p, bool accepted) {
  if (place >= kPlaces) return;
  const double x = log_odds(p), mu = sigmoid(offset_[place] + slope_ * x), r = (accepted ? 1.0 : 0.0) - mu;
  const double w = mu * (1.0 - mu);
  g_slope_ += r * x;
  h_slope_ += w * x * x;
  g_offset_[place] += r;
  h_offset_[place] += w;
  h_cross_[place] += w * x;
  n_[place]++;
  seen_++;
  if (++since_ >= kRefit) refit();
}

// One damped Newton step on the drafts since the last fit. The offsets meet only through the slope, so the slope is
// solved first and each offset follows from it.
void DraftEstimate::refit() {
  double s = h_slope_ + kRidge, g = g_slope_;
  for (uint32_t j = 0; j < kPlaces; j++) {
    const double d = h_offset_[j] + kRidge;
    s -= h_cross_[j] * h_cross_[j] / d;
    g -= h_cross_[j] * g_offset_[j] / d;
  }
  const double ds = s > 0 ? g / s : 0.0, damping = std::clamp(seen_ / 2.0, kDampingMin, kDampingMax);
  slope_ = std::max(0.0, slope_ + ds * since_ / (since_ + damping));
  for (uint32_t j = 0; j < kPlaces; j++) {
    const double dj = (g_offset_[j] - h_cross_[j] * ds) / (h_offset_[j] + kRidge);
    offset_[j] += dj * n_[j] / (n_[j] + damping);
    g_offset_[j] = h_offset_[j] = h_cross_[j] = 0;
    n_[j] = 0;
  }
  g_slope_ = h_slope_ = 0;
  since_ = 0;
}

std::vector<uint32_t> share_rows(const std::vector<DraftOdds> &slots, const std::vector<uint32_t> &caps,
                                 uint32_t rows, uint32_t max_rows) {
  std::vector<uint32_t> d(slots.size(), 0);
  for (uint32_t i : share_order(slots, caps, rows, max_rows)) d[i]++;
  return d;
}

std::vector<uint32_t> share_drafts(const std::vector<DraftOdds> &slots, const std::vector<uint32_t> &caps,
                                   uint32_t rows, uint32_t max_rows, const WindowTimes &times) {
  const std::vector<uint32_t> order = share_order(slots, caps, rows, max_rows);
  std::vector<uint32_t> d(slots.size(), 0);
  // The first rows of the order are one for each slot that may draft.
  uint32_t floor = 0;
  for (size_t i = 0; i < slots.size(); i++) floor += caps[i] > 0;
  floor = std::min<uint32_t>(floor, (uint32_t)order.size());
  uint32_t take = (uint32_t)order.size();
  if (times.seen() >= 2) {
    double tokens = rows, best = 0;
    std::vector<uint32_t> c(slots.size(), 0);
    take = floor;
    for (uint32_t k = 0; k <= order.size(); k++) {
      if (k > 0) {
        const uint32_t i = order[k - 1];
        tokens += std::pow(slots[i].layer_rate(), ++c[i]);
      }
      if (k < floor) continue;
      const double r = tokens / times.ms(rows + k);
      if (r > best) best = r, take = k;
    }
  }
  for (uint32_t k = 0; k < take; k++) d[order[k]]++;
  return d;
}

std::vector<uint32_t> choose_verified(const std::vector<std::vector<double>> &survival, uint32_t rows,
                                      uint32_t max_rows, const WindowTimes &times, uint32_t extra) {
  struct Draft {
    double survival;
    uint32_t slot, place;
  };
  std::vector<Draft> all;
  for (uint32_t i = 0; i < survival.size(); i++)
    for (uint32_t j = 0; j < survival[i].size(); j++) all.push_back({survival[i][j], i, j});
  // A slot's earlier draft goes before its later one on a tie, so the drafts taken are each slot's first.
  std::sort(all.begin(), all.end(), [](const Draft &a, const Draft &b) {
    if (a.survival != b.survival) return a.survival > b.survival;
    return a.place != b.place ? a.place < b.place : a.slot < b.slot;
  });
  const uint32_t most = std::min<uint32_t>((uint32_t)all.size(), max_rows > rows ? max_rows - rows : 0);
  uint32_t take = most;
  if (times.seen() >= 2) {
    double best = rows / times.ms(rows), tokens = rows;
    take = 0;
    for (uint32_t k = 1; k <= most; k++) {
      tokens += all[k - 1].survival;
      const double r = tokens / times.ms(rows + k);
      if (r > best) best = r, take = k;
    }
    take = std::min(most, take + extra);
  }
  std::vector<uint32_t> d(survival.size(), 0);
  for (uint32_t k = 0; k < take; k++) d[all[k].slot]++;
  return d;
}

}  // namespace engine
