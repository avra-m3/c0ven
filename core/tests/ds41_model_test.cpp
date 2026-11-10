// deepseek41's decode step on the cards against RefModel on the CPU, on the real file: the streams after each
// sublayer, the logits and the first choice, one row a step. Needs two cards and DS41_MODEL, the file's first shard.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include "models/deepseek41/model.h"
#include "models/deepseek41/reference.h"
#include "test_util.h"

using namespace engine;
using namespace engine::deepseek41;

namespace {

constexpr double kKlBound = 0.04;  // ADR-0074's bound, mean nats a position

double rel_rms(const float *got, const float *want, size_t n) {
  double d = 0, w = 0;
  for (size_t i = 0; i < n; i++) {
    d += ((double)got[i] - want[i]) * ((double)got[i] - want[i]);
    w += (double)want[i] * want[i];
  }
  return std::sqrt(d / std::max(w, 1e-30));
}

// KL(want || got) of two rows of logits, in nats.
double kl_logits(const std::vector<float> &want, const std::vector<float> &got) {
  const auto log_sum = [](const std::vector<float> &x) {
    const double top = *std::max_element(x.begin(), x.end());
    double z = 0;
    for (float t : x) z += std::exp((double)t - top);
    return top + std::log(z);
  };
  const double zw = log_sum(want), zg = log_sum(got);
  double kl = 0;
  for (size_t i = 0; i < want.size(); i++) kl += std::exp(want[i] - zw) * ((want[i] - zw) - (got[i] - zg));
  return kl;
}

uint32_t env_u32(const char *name, uint32_t otherwise) {
  const char *e = std::getenv(name);
  return e ? (uint32_t)std::strtoul(e, nullptr, 10) : otherwise;
}

}  // namespace

namespace {

int run() {
  if (test::gpus() < 2) return test::skip("deepseek41's step needs two cards");
  const char *path = std::getenv("DS41_MODEL");
  if (!path) return test::skip("DS41_MODEL is not set: the first shard of DeepSeek-V4.1-Flash-1017's file");
  const uint32_t steps = env_u32("DS41_STEPS", 12);

  Options opt;
  opt.slots = 1;
  opt.max_ctx = 1024;
  opt.arena_bytes = (uint64_t)env_u32("DS41_ARENA_GIB", 0) << 30;
  opt.hugetlb_arena = std::getenv("ENGINE_HUGETLB_ARENA") != nullptr;
  const auto t_load = std::chrono::steady_clock::now();
  deepseek41::Model model(path, opt, Debug{true});
  std::printf("loaded in %.1f s: %u cards, arena %.1f GiB\n",
              std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load).count(), model.info().cards,
              (double)model.info().arena_bytes / (1ull << 30));
  std::vector<uint32_t> pages(model.pages_per_slot());
  for (uint32_t i = 0; i < pages.size(); i++) pages[i] = i;
  bool refused = false;
  try {
    const StepIn row{0, 0, 0, -1, -1, -1};
    model.step(&row, 1);
  } catch (const std::runtime_error &) {
    refused = true;
  }
  EXPECT(refused, "a row whose page is not set was run");
  model.set_pages(0, 0, pages.data(), (uint32_t)pages.size());
  RefModel ref(path, RefOptions{16, steps});
  const uint32_t vocab = ref.vocab();
  EXPECT(model.info().vocab == vocab, "the vocabulary: %u, RefModel %u", model.info().vocab, vocab);

  // Any tokens do: each step feeds RefModel's first choice back, after a fixed start.
  std::vector<int32_t> tokens = {0, 9211, 1418, 344, 31032, 5106};
  std::vector<float> want(vocab);
  std::map<std::pair<bool, int>, std::vector<float>> streams;
  double kl_sum = 0, worst_stream = 0, ms_sum = 0;
  uint32_t same = 0;
  for (uint32_t t = 0; t < steps; t++) {
    streams.clear();
    ref.decode(tokens[t], want.data(), [&](const char *name, int layer, const float *v, size_t n) {
      const std::string s = name;
      if (s == "hc_attn_post" || s == "l_out") streams[{s == "l_out", layer}].assign(v, v + n);
    });
    const StepIn row = row_at(0, t, tokens[t], tokens);
    const auto t0 = std::chrono::steady_clock::now();
    const auto cands = model.step(&row, 1);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (t) ms_sum += ms;

    double worst = 0;
    int worst_layer = -1;
    bool worst_ffn = false;
    for (const auto &[key, v] : streams) {
      EXPECT(v.size() == D41_HC_DIM, "RefModel's streams hold %zu values", v.size());
      const double e = rel_rms(model.probe((uint32_t)key.second, key.first), v.data(), v.size());
      EXPECT(!std::isnan(e), "position %u, layer %d: a stream is not a number", t, key.second);
      if (e > worst) worst = e, worst_layer = key.second, worst_ffn = key.first;
      if (std::getenv("DS41_STREAMS")) std::printf("  position %u layer %d %s: %.2e\n", t, key.second, key.first ? "ffn" : "attention", e);
    }
    EXPECT(streams.size() == 2 * ref.layers(), "RefModel gave %zu streams", streams.size());
    worst_stream = std::max(worst_stream, worst);

    const std::vector<float> got = model.logits(0);
    EXPECT(got.size() == vocab, "the logits hold %zu values", got.size());
    const double kl = kl_logits(want, got);
    const auto top_want = (int32_t)(std::max_element(want.begin(), want.end()) - want.begin());
    const auto top_got = (int32_t)(std::max_element(got.begin(), got.end()) - got.begin());
    EXPECT(!cands.empty() && !cands[0].empty() && cands[0][0].token == top_got, "the first candidate is the largest logit");
    kl_sum += kl;
    same += top_want == top_got;
    const MissCounts &c = model.miss_counts();
    std::printf("position %u token %d: %.1f ms, streams worst %.2e (layer %d %s), logits %.2e, KL %.5f, choice %d %s; CPU %llu arena, %llu SSD\n",
                t, tokens[t], ms, worst, worst_layer, worst_ffn ? "ffn" : "attention", rel_rms(got.data(), want.data(), vocab), kl,
                top_got, top_want == top_got ? "same" : "differs", (unsigned long long)c.arena_jobs, (unsigned long long)c.ssd_jobs);
    std::fflush(stdout);
    if (tokens.size() <= t + 1) tokens.push_back(top_want);
  }
  const double kl = kl_sum / steps;
  std::printf("%u positions: mean KL %.5f, %u first choices the same, streams worst %.2e, %.1f ms a step after the first\n", steps, kl,
              same, worst_stream, steps > 1 ? ms_sum / (steps - 1) : 0.0);
  EXPECT(kl <= kKlBound, "the logits' mean KL against RefModel is %.5f, above %.2f", kl, kKlBound);
  std::printf("ds41_model_test: ok\n");
  return 0;
}

}  // namespace

// A failure ends through the destructors, so no list is left running on a card.
int main() {
  try {
    return run();
  } catch (const std::exception &e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
}
