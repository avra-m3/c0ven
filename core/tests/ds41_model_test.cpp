// deepseek41's decode step on the cards against RefModel on the CPU, on the real file: the streams after each
// sublayer, the logits and the first choice. Needs two cards and DS41_MODEL, the file's first shard. DS41_STEPS:
// positions run; DS41_ROWS: positions a step; DS41_TOKENS: a dump of upstream's whose tokens start the sequence;
// DS41_ARENA_GIB, DS41_CARD_EXPERTS, DS41_PROFILE: the tiers' sizes and a routing profile; DS41_SWAPS: experts change
// tiers by use; DS41_TRACE: the last step's kernels; DS41_NO_STREAMS: no streams, RefModel first, then the steps back
// to back; DS41_LAYERS: the host's waits by layer; DS41_CPU_THREADS: the CPU's expert workers; DS41_ROUTES: a file
// that gets each position's routed experts a layer, then how many of them the cards lacked and their ids; DS41_NO_REF: no RefModel, the step's own first choice goes on, nothing is
// compared and the test skips; DS41_DRAFTER: the drafter's file, loaded so that each step stores its keys;
// DS41_PROMPT_CHUNK: Options::prompt_chunk, whose buffers take card memory from the expert cache (0: none).
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
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

  const bool compare = std::getenv("DS41_NO_REF") == nullptr;
  const bool probes = compare && std::getenv("DS41_NO_STREAMS") == nullptr;
  // Any tokens do: each position feeds RefModel's first choice back, after a fixed start or DS41_TOKENS's, a dump.
  std::vector<int32_t> tokens = {0, 9211, 1418, 344, 31032, 5106};
  if (const char *dump = std::getenv("DS41_TOKENS")) tokens = dump_tokens(dump);
  using Streams = std::map<std::pair<bool, int>, std::vector<float>>;
  auto ref = compare ? std::make_unique<RefModel>(path, RefOptions{16, steps}) : nullptr;
  const uint32_t vocab = D41_VOCAB, layers = D41_LAYERS;
  EXPECT(!ref || (ref->vocab() == vocab && ref->layers() == layers), "RefModel's vocabulary or layers");
  const auto ref_decode = [&](uint32_t pos, std::vector<float> &want, Streams *streams) {
    ref->decode(tokens[pos], want.data(), [&](const char *name, int layer, const float *v, size_t count) {
      const std::string s = name;
      if (streams && (s == "hc_attn_post" || s == "l_out")) (*streams)[{s == "l_out", layer}].assign(v, v + count);
    });
    if (tokens.size() <= pos + 1) tokens.push_back((int32_t)(std::max_element(want.begin(), want.end()) - want.begin()));
  };
  // Without the streams RefModel runs every position first and leaves: the steps then run back to back, as a
  // request's do, and the arena may take the memory RefModel held.
  std::vector<std::vector<float>> ahead;
  for (uint32_t pos = 0; pos < steps && compare && !probes; pos++) {
    ahead.emplace_back(vocab);
    ref_decode(pos, ahead.back(), nullptr);
  }
  if (!probes) ref.reset();

  Options opt;
  opt.slots = 1;
  opt.max_ctx = 1024;
  opt.arena_bytes = (uint64_t)env_u32("DS41_ARENA_GIB", 0) << 30;
  opt.hugetlb_arena = std::getenv("ENGINE_HUGETLB_ARENA") != nullptr;
  opt.experts_per_layer = std::getenv("DS41_CARD_EXPERTS") ? std::atoi(std::getenv("DS41_CARD_EXPERTS")) : -1;
  if (const char *profile = std::getenv("DS41_PROFILE")) opt.routing_profile = profile;
  opt.trace = std::getenv("DS41_TRACE") != nullptr;
  opt.adapt_swaps = std::getenv("DS41_SWAPS") != nullptr;
  if (const char *drafter = std::getenv("DS41_DRAFTER")) opt.mtp_dir = drafter;
  opt.cpu_threads = env_u32("DS41_CPU_THREADS", opt.cpu_threads);
  opt.prompt_chunk = env_u32("DS41_PROMPT_CHUNK", opt.prompt_chunk);
  FILE *routes = std::getenv("DS41_ROUTES") ? std::fopen(std::getenv("DS41_ROUTES"), "w") : nullptr;
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
  EXPECT(model.info().vocab == vocab, "the vocabulary: %u, RefModel %u", model.info().vocab, vocab);

  const uint32_t rows_a_step = std::min<uint32_t>(env_u32("DS41_ROWS", 1), STEP_MAX_ROWS);
  EXPECT(rows_a_step > 0, "DS41_ROWS is 1 to %d", STEP_MAX_ROWS);
  std::vector<std::vector<float>> want(rows_a_step, std::vector<float>(vocab));
  std::vector<Streams> streams(rows_a_step);
  double kl_sum = 0, worst_stream = 0, ms_sum = 0;
  StepTimes parts;
  std::vector<double> kls, step_ms;
  std::vector<std::array<float, D41_LAYERS>> layer_records, layer_answers;
  uint32_t same = 0;
  uint32_t step_count = 0;
  for (uint32_t t = 0; t < steps;) {
    // Without RefModel a step takes the tokens there are: its last row's first choice is the next.
    const uint32_t n = std::min({rows_a_step, steps - t, compare ? rows_a_step : (uint32_t)tokens.size() - t});
    std::vector<StepIn> rows;
    for (uint32_t i = 0; i < n; i++) {
      if (probes) ref_decode(t + i, want[i], &streams[i]);
      rows.push_back(row_at(0, t + i, tokens[t + i], tokens));
    }
    const auto t0 = std::chrono::steady_clock::now();
    const auto cands = model.step(rows.data(), n);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    const StepTimes p = model.step_times();
    if (t) {
      ms_sum += ms;
      step_ms.push_back(ms);
      layer_records.push_back(p.layer_records);
      layer_answers.push_back(p.layer_answers);
      parts.before += p.before, parts.engram += p.engram, parts.records += p.records, parts.answers += p.answers;
      parts.after += p.after;
      step_count++;
    }
    const MissCounts &c = model.miss_counts();
    EXPECT(cands.size() == n, "the step gave %zu rows of candidates", cands.size());
    if (!compare && tokens.size() == t + n) tokens.push_back(cands[n - 1].at(0).token);
    for (uint32_t i = 0; i < n && routes; i++)
      for (uint32_t l = 0; l < D41_LAYERS; l++) {
        const int32_t *e = model.routed(l, i), *m;
        const uint32_t nm = model.missed(l, i, &m);
        std::fprintf(routes, "%u %u %d %d %d %d %d %d %u", t + i, l, e[0], e[1], e[2], e[3], e[4], e[5], nm);
        for (uint32_t k = 0; k < nm; k++) std::fprintf(routes, " %d", m[k]);
        std::fputc('\n', routes);
      }
    for (uint32_t i = 0; i < n && !compare; i++)
      std::printf("position %u token %d: %.1f ms (before %.2f, Engram %.2f, records %.1f, CPU %.1f, end %.2f); CPU %llu of %llu "
                  "routings, %llu from the file, %llu swaps\n",
                  t + i, tokens[t + i], ms, p.before, p.engram, p.records, p.answers, p.after, (unsigned long long)c.routings,
                  (unsigned long long)D41_LAYERS * D41_USED * n, (unsigned long long)c.ssd_jobs, (unsigned long long)model.swaps_done());
    for (uint32_t i = 0; i < n && compare; i++) {
      double worst = 0;
      int worst_layer = -1;
      bool worst_ffn = false;
      for (const auto &[key, v] : streams[i]) {
        if (!probes) break;
        EXPECT(v.size() == D41_HC_DIM, "RefModel's streams hold %zu values", v.size());
        const double e = rel_rms(model.probe((uint32_t)key.second, key.first) + (size_t)i * D41_HC_DIM, v.data(), v.size());
        EXPECT(!std::isnan(e), "position %u, layer %d: a stream is not a number", t + i, key.second);
        if (e > worst) worst = e, worst_layer = key.second, worst_ffn = key.first;
        if (std::getenv("DS41_STREAMS"))
          std::printf("  position %u layer %d %s: %.2e\n", t + i, key.second, key.first ? "ffn" : "attention", e);
      }
      EXPECT(!probes || streams[i].size() == 2 * layers, "RefModel gave %zu streams", streams[i].size());
      worst_stream = std::max(worst_stream, worst);

      const std::vector<float> &wanted = probes ? want[i] : ahead[t + i];
      const std::vector<float> got = model.logits(i);
      EXPECT(got.size() == vocab, "the logits hold %zu values", got.size());
      const double kl = kl_logits(wanted, got);
      const auto top_want = (int32_t)(std::max_element(wanted.begin(), wanted.end()) - wanted.begin());
      const auto top_got = (int32_t)(std::max_element(got.begin(), got.end()) - got.begin());
      EXPECT(!cands[i].empty() && cands[i][0].token == top_got, "the first candidate is the largest logit");
      kl_sum += kl;
      kls.push_back(kl);
      same += top_want == top_got;
      std::printf("position %u token %d: %.1f ms (before %.2f, Engram %.2f, records %.1f, CPU %.1f, end %.2f), streams worst %.2e (layer %d %s), "
                  "logits %.2e, KL %.5f, choice %d %s; CPU %llu of %llu routings, %llu from the file, %llu swaps\n",
                  t + i, tokens[t + i], ms, p.before, p.engram, p.records, p.answers, p.after, worst, worst_layer,
                  worst_ffn ? "ffn" : "attention", rel_rms(got.data(), wanted.data(), vocab), kl, top_got,
                  top_want == top_got ? "same" : "differs", (unsigned long long)c.routings,
                  (unsigned long long)D41_LAYERS * D41_USED * n, (unsigned long long)c.ssd_jobs, (unsigned long long)model.swaps_done());
    }
    std::fflush(stdout);
    t += n;
  }
  if (routes) std::fclose(routes);
  std::sort(kls.begin(), kls.end());
  const double median = compare ? kls[kls.size() / 2] : 0.0, after = step_count ? step_count : 1;
  const auto loose = (uint32_t)(kls.end() - std::upper_bound(kls.begin(), kls.end(), kLooseKl));
  if (compare)
    std::printf("%u positions: KL median %.5f, mean %.5f, %u above %.1f; %u first choices the same; streams worst %.2e\n", steps,
                median, kl_sum / steps, loose, kLooseKl, same, worst_stream);
  std::printf("a step after the first: %.1f ms; before the lists %.1f (Engram %.1f), waiting for records %.1f, the CPU's experts %.1f, to the lists' end %.1f\n",
              ms_sum / after, parts.before / after, parts.engram / after, parts.records / after, parts.answers / after,
              parts.after / after);
  if (!step_ms.empty()) {
    // The last quarter apart: with DS41_SWAPS the experts have settled by then.
    const auto at = [](std::vector<double> v, double q) {
      std::sort(v.begin(), v.end());
      return v[(size_t)(q * (double)(v.size() - 1))];
    };
    const std::vector<double> tail(step_ms.end() - (std::ptrdiff_t)std::max<size_t>(step_ms.size() / 4, 1), step_ms.end());
    std::printf("steps: median %.1f ms, a tenth below %.1f, a tenth above %.1f; the last quarter: median %.1f, a tenth above %.1f\n",
                at(step_ms, 0.5), at(step_ms, 0.1), at(step_ms, 0.9), at(tail, 0.5), at(tail, 0.9));
  }
  if (opt.adapt_swaps) {
    const SwapStats w = model.swap_stats();
    const double nb = (double)std::max<uint64_t>(w.batches, 1), nr = (double)std::max<uint64_t>(w.refills, 1);
    std::printf("swap batches: %llu of %.1f pairs, mapped %.1f steps after the choice; ms a batch: slots written %.1f (building %.1f, upload %.1f, "
                "both cards summed), mapped %.1f\n",
                (unsigned long long)w.batches, (double)w.pairs / nb, (double)w.steps_to_map / nb, w.to_loaded / nb, w.make / nb,
                w.upload / nb, w.to_mapped / nb);
    std::printf("refills: %llu, %.1f a batch; ms a refill: read %.2f, write %.2f, from queued to placed %.1f\n", (unsigned long long)w.refills,
                (double)w.refills / nb, w.refill_read / nr, w.refill_write / nr, w.refill_wait / nr);
  }
  if (std::getenv("DS41_LAYERS") && !step_ms.empty()) {
    const size_t from = layer_records.size() - std::max<size_t>(layer_records.size() / 4, 1), count = layer_records.size() - from;
    for (uint32_t l = 0; l < D41_LAYERS; l++) {
      double r = 0, a = 0;
      for (size_t i = from; i < layer_records.size(); i++) r += layer_records[i][l], a += layer_answers[i][l];
      std::printf("  layer %u, the last quarter: records %.3f ms, the CPU's experts %.3f\n", l, r / (double)count, a / (double)count);
    }
  }
  for (uint32_t c = 0; c < 2 && opt.trace; c++) {
    std::vector<TraceRow> rows = model.trace(c);
    double sum = 0;
    for (const TraceRow &r : rows) sum += r.ms;
    std::sort(rows.begin(), rows.end(), [](const TraceRow &a, const TraceRow &b) { return a.ms > b.ms; });
    std::printf("card %u, the last step's list: %.2f ms\n", c, sum);
    for (const TraceRow &r : rows) std::printf("  %-40s %4u launches %7.3f ms\n", r.kernel.c_str(), r.launches, r.ms);
  }
  if (!compare) return test::skip("DS41_NO_REF: nothing was compared");
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
