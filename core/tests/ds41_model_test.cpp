// deepseek41's decode step on the cards against RefModel on the CPU, on the real file: the streams after each
// sublayer, the logits and the first choice, one row a step. Needs two cards and DS41_MODEL, the file's first shard.
// DS41_STEPS: positions run. DS41_TOKENS: a dump of upstream's whose tokens start the sequence. DS41_ARENA_GIB,
// DS41_CARD_EXPERTS, DS41_PROFILE: the arena's size, the experts a layer on the cards, a routing profile. DS41_TRACE:
// the last step's kernels and their times. DS41_NO_STREAMS: the lists copy no streams out, as a served step.
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

// The logits against RefModel's, in nats: half the positions within kMedianKl and all but one in kLooseOf within kLooseKl.
// A position where the model is unsure moves far on a rounding, as RefModel does against upstream there, so the mean of
// a few positions is not bounded here.
constexpr double kMedianKl = 0.01, kLooseKl = 0.1;
constexpr uint32_t kLooseOf = 16;

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

// The tokens of a dump of upstream's values (reference/deepseek41/refgen.cpp), one a position.
std::vector<int32_t> dump_tokens(const char *path) {
  FILE *f = std::fopen(path, "rb");
  EXPECT(f, "cannot read %s", path);
  std::vector<int32_t> tokens;
  uint32_t head[6];
  while (std::fread(head, 1, sizeof head, f) == sizeof head) {
    EXPECT(head[0] == 0x44343144u, "%s: not a record", path);
    if (head[1] == tokens.size()) tokens.push_back((int32_t)head[2]);
    std::fseek(f, (long)head[4] + 4 * (long)head[5], SEEK_CUR);
  }
  std::fclose(f);
  return tokens;
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
  opt.experts_per_layer = std::getenv("DS41_CARD_EXPERTS") ? std::atoi(std::getenv("DS41_CARD_EXPERTS")) : -1;
  if (const char *profile = std::getenv("DS41_PROFILE")) opt.routing_profile = profile;
  opt.trace = std::getenv("DS41_TRACE") != nullptr;
  const bool probes = std::getenv("DS41_NO_STREAMS") == nullptr;
  const auto t_load = std::chrono::steady_clock::now();
  deepseek41::Model model(path, opt, Debug{probes});
  std::printf("loaded in %.1f s: %u cards with %u experts a layer and %.1f, %.1f GiB in use, arena %.1f GiB\n",
              std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load).count(), model.info().cards,
              model.info().experts_per_layer, (double)model.info().device_bytes[0] / (1ull << 30),
              (double)model.info().device_bytes[1] / (1ull << 30), (double)model.info().arena_bytes / (1ull << 30));
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

  // Any tokens do: each step feeds RefModel's first choice back, after a fixed start or DS41_TOKENS's, a dump.
  std::vector<int32_t> tokens = {0, 9211, 1418, 344, 31032, 5106};
  if (const char *dump = std::getenv("DS41_TOKENS")) tokens = dump_tokens(dump);
  std::vector<float> want(vocab);
  std::map<std::pair<bool, int>, std::vector<float>> streams;
  double kl_sum = 0, worst_stream = 0, ms_sum = 0;
  StepTimes parts;
  std::vector<double> kls;
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
    if (t) {
      const StepTimes &p = model.step_times();
      ms_sum += ms;
      parts.before += p.before, parts.records += p.records, parts.answers += p.answers, parts.after += p.after;
    }

    double worst = 0;
    int worst_layer = -1;
    bool worst_ffn = false;
    for (const auto &[key, v] : streams) {
      if (!probes) break;
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
    kls.push_back(kl);
    same += top_want == top_got;
    const MissCounts &c = model.miss_counts();
    std::printf("position %u token %d: %.1f ms, streams worst %.2e (layer %d %s), logits %.2e, KL %.5f, choice %d %s; CPU %llu of %llu routings, %llu from the file\n",
                t, tokens[t], ms, worst, worst_layer, worst_ffn ? "ffn" : "attention", rel_rms(got.data(), want.data(), vocab), kl,
                top_got, top_want == top_got ? "same" : "differs", (unsigned long long)c.routings, (unsigned long long)D41_LAYERS * D41_USED,
                (unsigned long long)c.ssd_jobs);
    std::fflush(stdout);
    if (tokens.size() <= t + 1) tokens.push_back(top_want);
  }
  std::sort(kls.begin(), kls.end());
  const double median = kls[kls.size() / 2], after = steps > 1 ? steps - 1 : 1;
  const auto loose = (uint32_t)(kls.end() - std::upper_bound(kls.begin(), kls.end(), kLooseKl));
  std::printf("%u positions: KL median %.5f, mean %.5f, %u above %.1f; %u first choices the same; streams worst %.2e\n", steps, median,
              kl_sum / steps, loose, kLooseKl, same, worst_stream);
  std::printf("a step after the first: %.1f ms; before the lists %.1f, waiting for records %.1f, the CPU's experts %.1f, to the lists' end %.1f\n",
              ms_sum / after, parts.before / after, parts.records / after, parts.answers / after, parts.after / after);
  for (uint32_t c = 0; c < 2 && opt.trace; c++) {
    std::vector<TraceRow> rows = model.trace(c);
    double sum = 0;
    for (const TraceRow &r : rows) sum += r.ms;
    std::sort(rows.begin(), rows.end(), [](const TraceRow &a, const TraceRow &b) { return a.ms > b.ms; });
    std::printf("card %u, the last step's list: %.2f ms\n", c, sum);
    for (const TraceRow &r : rows) std::printf("  %-40s %4u launches %7.3f ms\n", r.kernel.c_str(), r.launches, r.ms);
  }
  EXPECT(median <= kMedianKl, "the logits' median KL against RefModel is %.5f, above %.2f", median, kMedianKl);
  EXPECT(loose <= steps / kLooseOf + 1, "%u of %u positions have a KL above %.1f", loose, steps, kLooseKl);
  EXPECT(same * 10 >= steps * 9, "%u of %u first choices are RefModel's", same, steps);
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
