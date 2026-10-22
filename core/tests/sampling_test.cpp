// Level 1 (docs/TESTING.md): the sampler (api/sampling.h) against llama.cpp's rules, worked by hand on a few
// candidates (docs/spec/MODEL_IO.md, section 2.4): the penalties first, then top-k, top-p on the logits before the
// temperature, min-p, and the temperature last. No card.
#include <cmath>
#include <cstdint>
#include <set>
#include <vector>

#include "api/sampling.h"
#include "test_util.h"

using engine::sample_p;
using engine::qwen4exp::Candidate;

namespace {

engine_settings greedy() {
  engine_settings s{};
  s.temperature = 0.0f;
  return s;
}

// The tokens a sampler gives over many positions.
std::set<int32_t> drawn(const std::vector<Candidate> &c, const engine_settings &s, const std::vector<int32_t> &hist) {
  std::set<int32_t> out;
  for (uint32_t pos = 0; pos < 4000; pos++) out.insert(sample_p(c, s, pos, hist).first);
  return out;
}

}  // namespace

int main() {
  const std::vector<Candidate> c{{10, 5.0f}, {11, 4.9f}, {12, -1.0f}, {13, -1.2f}};

  // Without penalties greedy takes the top candidate, and its probability is the plain softmax's.
  {
    auto [tok, p] = sample_p(c, greedy(), 0, {});
    double sum = 0;
    for (const Candidate &x : c) sum += std::exp((double)x.logit - 5.0);
    EXPECT(tok == 10 && std::fabs(p - 1.0 / sum) < 1e-12, "greedy: %d %g", tok, p);
  }

  // A repeat penalty divides a positive logit: 5.0 / 1.1 = 4.55 falls below 4.9.
  {
    engine_settings s = greedy();
    s.penalty_last_n = 64;
    s.penalty_repeat = 1.1f;
    EXPECT(sample_p(c, s, 0, {10}).first == 11, "repeat penalty on a positive logit");
    // The window holds only the last penalty_last_n tokens: 10 is outside it.
    s.penalty_last_n = 2;
    EXPECT(sample_p(c, s, 0, {10, 3, 4}).first == 10, "a token outside the window is not counted");
  }

  // A repeat penalty multiplies a negative logit: -1.0 * 1.5 = -1.5 falls below -1.2.
  {
    std::vector<Candidate> neg{{12, -1.0f}, {13, -1.2f}};
    engine_settings s = greedy();
    s.penalty_last_n = 64;
    s.penalty_repeat = 1.5f;
    EXPECT(sample_p(neg, s, 0, {12}).first == 13, "repeat penalty on a negative logit");
  }

  // Frequency counts each occurrence, presence once: 5.0 - 2 x 0.04 = 4.92 stays above 4.9, - 3 x 0.04 does not.
  {
    engine_settings s = greedy();
    s.penalty_last_n = 64;
    s.penalty_repeat = 1.0f;
    s.penalty_freq = 0.04f;
    EXPECT(sample_p(c, s, 0, {10, 10}).first == 10, "frequency, twice");
    EXPECT(sample_p(c, s, 0, {10, 10, 10}).first == 11, "frequency, three times");
    s.penalty_freq = 0.0f;
    s.penalty_present = 0.11f;
    EXPECT(sample_p(c, s, 0, {10}).first == 11, "presence");
    s.penalty_present = 0.09f;
    EXPECT(sample_p(c, s, 0, {10, 10, 10}).first == 10, "presence counts once");
  }

  // Top-p before the temperature. At temperature 1, logits 2, 1, 0 have probabilities 0.665, 0.245, 0.090, so top-p
  // 0.7 keeps two. At temperature 100 all three are near 1/3, and top-p applied after it would keep all three.
  {
    const std::vector<Candidate> t{{20, 2.0f}, {21, 1.0f}, {22, 0.0f}};
    engine_settings s{};
    s.temperature = 100.0f;
    s.top_p = 0.7f;
    EXPECT(drawn(t, s, {}) == (std::set<int32_t>{20, 21}), "top-p on the logits before the temperature");
  }

  // Min-p keeps the tokens at least min_p times as likely as the top one, before the temperature: logits 0, -1, -3
  // against log(0.1) = -2.30 keep two.
  {
    const std::vector<Candidate> m{{30, 0.0f}, {31, -1.0f}, {32, -3.0f}};
    engine_settings s{};
    s.temperature = 100.0f;
    s.min_p = 0.1f;
    EXPECT(drawn(m, s, {}) == (std::set<int32_t>{30, 31}), "min-p");
    // Min-p reads the penalised logits: 30 penalised to -1 puts the floor at -3.30, and 32 is kept.
    s.penalty_last_n = 64;
    s.penalty_repeat = 1.0f;
    s.penalty_present = 1.0f;
    EXPECT(drawn(m, s, {30}) == (std::set<int32_t>{30, 31, 32}), "min-p after the penalties");
  }

  // Top-k cuts first: top-k 1 is greedy whatever the temperature.
  {
    engine_settings s{};
    s.temperature = 100.0f;
    s.top_k = 1;
    EXPECT(drawn(c, s, {}) == (std::set<int32_t>{10}), "top-k 1");
  }

  std::printf("sampling_test: ok\n");
  return 0;
}
