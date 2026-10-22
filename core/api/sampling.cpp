#include "api/sampling.h"

#include <algorithm>
#include <cmath>

namespace engine {

using qwen4exp::Candidate;
using qwen4exp::kNoToken;

bool penalised(const engine_settings &s) {
  return s.penalty_last_n > 0 && (s.penalty_repeat != 1.0f || s.penalty_freq != 0.0f || s.penalty_present != 0.0f);
}

double uniform(uint64_t seed, uint32_t pos) {
  uint64_t z = seed + 0x9e3779b97f4a7c15ull * (pos + 1ull);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  z ^= z >> 31;
  return (double)(z >> 11) * (1.0 / 9007199254740992.0);
}

std::pair<int32_t, double> sample_p(const std::vector<Candidate> &cand, const engine_settings &s, uint32_t pos,
                                    const std::vector<int32_t> &hist) {
  if (cand.empty()) return {kNoToken, 0.0};
  std::vector<Candidate> c = cand;
  if (penalised(s)) {
    // A penalty can move a candidate below tokens the cards did not return; only the candidates are reordered.
    const auto w0 = hist.end() - (std::ptrdiff_t)std::min<size_t>(hist.size(), s.penalty_last_n);
    for (Candidate &x : c) {
      const auto n = std::count(w0, hist.end(), x.token);
      if (n == 0) continue;
      x.logit = x.logit <= 0.0f ? x.logit * s.penalty_repeat : x.logit / s.penalty_repeat;
      x.logit -= (float)n * s.penalty_freq + s.penalty_present;
    }
    std::stable_sort(c.begin(), c.end(), [](const Candidate &a, const Candidate &b) { return a.logit > b.logit; });
  }
  if (s.temperature <= 0.0f) {
    double sum = 0;
    for (const Candidate &x : c) sum += std::exp((double)x.logit - c[0].logit);
    return {c[0].token, 1.0 / sum};
  }
  size_t k = s.top_k == 0 ? c.size() : std::min<size_t>(s.top_k, c.size());
  if (s.top_p > 0.0f && s.top_p < 1.0f) {  // on the logits before the temperature, as llama.cpp
    double sum = 0, acc = 0;
    for (size_t i = 0; i < k; i++) sum += std::exp((double)c[i].logit - c[0].logit);
    for (size_t i = 0; i < k; i++) {
      acc += std::exp((double)c[i].logit - c[0].logit) / sum;
      if (acc >= s.top_p) {
        k = i + 1;
        break;
      }
    }
  }
  if (s.min_p > 0.0f) {
    const double floor = (double)c[0].logit + std::log((double)s.min_p);
    size_t keep = 1;
    while (keep < k && c[keep].logit >= floor) keep++;
    k = keep;
  }
  std::vector<double> p(k);
  double sum = 0;
  for (size_t i = 0; i < k; i++) sum += p[i] = std::exp((c[i].logit - c[0].logit) / s.temperature);
  double r = uniform(s.seed, pos) * sum;
  for (size_t i = 0; i < k; i++) {
    r -= p[i];
    if (r <= 0) return {c[i].token, p[i] / sum};
  }
  return {c[k - 1].token, p[k - 1] / sum};
}

}  // namespace engine
