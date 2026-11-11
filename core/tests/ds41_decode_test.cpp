// deepseek41 writing sampled text on the cards with its drafter. A round: the drafter drafts a block after the token
// sampled last, one step runs that token and the first drafts whose keep chance reaches the cut, and a draft is kept
// when the request's sampler picks it from the row before it, as the engine's rounds keep one
// (docs/spec/STRATA_DECODE.md). Then the same tokens run one row a step in a second slot, and each position's
// candidates are compared: the rows of a round, kept or not, must leave what those steps leave.
// Needs two cards, DS41_MODEL and DS41_DRAFTER; without the drafter the text is written one row a step, nothing is
// compared and the test skips. DS41_PROMPTS: a file of prompts, a line each, a name then its tokens; DS41_PROMPT_IDS:
// the names to run, in order (all of them); DS41_NEW: new tokens a prompt; DS41_TEMP, DS41_SEED: the sampler's;
// DS41_CUT: the keep chance a draft needs to be run; DS41_DRAFTS: drafts run a round at most; DS41_NO_REF: no second
// pass, and the test skips; DS41_DUMP: a file that gets a line a round, with the candidates of its drafts and rows;
// DS41_POSITIONS: a line a compared position;
// DS41_ARENA_GIB, DS41_CARD_EXPERTS, DS41_PROFILE, DS41_SWAPS: as ds41_model_test.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "api/sampling.h"
#include "models/deepseek41/model.h"
#include "test_util.h"

using namespace engine;
using namespace engine::deepseek41;

namespace {

// A position's candidates from a round against the one-row step's, in nats over that step's candidates: half the
// positions within kMedianKl. A wider step sums an expert's rows in another order, and rounding to 8-bit activations
// and the routers' choices carry that last bit on. Where the model is unsure one position moves far, so the sampler's
// picks are bounded and not the worst position.
constexpr double kMedianKl = 0.002;
// Of the round's tokens, those the sampler also picks from the one-row step's candidates, in 100.
constexpr uint32_t kSamePicks = 85;
// The file's end-of-sentence token.
constexpr int32_t kEnd = 1;

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

uint32_t env_u32(const char *name, uint32_t otherwise) {
  const char *e = std::getenv(name);
  return e ? (uint32_t)std::strtoul(e, nullptr, 10) : otherwise;
}
double env_f(const char *name, double otherwise) {
  const char *e = std::getenv(name);
  return e ? std::atof(e) : otherwise;
}

struct Prompt {
  std::string id;
  std::vector<int32_t> tokens;
};

std::vector<Prompt> prompts() {
  const char *file = std::getenv("DS41_PROMPTS");
  // A user's turn in the model's chat template: after a bare start of text the model's odds are too flat to bound
  // the sampler's picks.
  if (!file)
    return {{"chat",
             {0, 128803, 65106, 1192, 260, 19657, 6403, 43209, 1234, 18222, 396, 3155, 295, 270, 1975, 36013, 16, 128804,
              128822}}};
  std::vector<Prompt> all;
  std::ifstream in(file);
  EXPECT(in.good(), "cannot read %s", file);
  for (std::string line; std::getline(in, line);) {
    std::istringstream s(line);
    Prompt p;
    if (!(s >> p.id)) continue;
    for (int32_t t; s >> t;) p.tokens.push_back(t);
    EXPECT(!p.tokens.empty(), "%s: prompt %s has no token", file, p.id.c_str());
    all.push_back(std::move(p));
  }
  const char *ids = std::getenv("DS41_PROMPT_IDS");
  if (!ids) return all;
  std::vector<Prompt> chosen;
  std::istringstream s(ids);
  for (std::string id; s >> id;) {
    const auto p = std::find_if(all.begin(), all.end(), [&](const Prompt &x) { return x.id == id; });
    EXPECT(p != all.end(), "no prompt %s in %s", id.c_str(), file);
    chosen.push_back(*p);
  }
  return chosen;
}

// KL(want || got) in nats over want's candidates. A token got lacks takes got's smallest logit, which is above its own.
double kl_candidates(const std::vector<Candidate> &want, const std::vector<Candidate> &got) {
  std::vector<double> w, g;
  for (const Candidate &c : want) {
    const auto at = std::find_if(got.begin(), got.end(), [&](const Candidate &x) { return x.token == c.token; });
    w.push_back(c.logit);
    g.push_back(at == got.end() ? got.back().logit : at->logit);
  }
  const auto log_sum = [](const std::vector<double> &x) {
    const double top = *std::max_element(x.begin(), x.end());
    double z = 0;
    for (double t : x) z += std::exp(t - top);
    return top + std::log(z);
  };
  const double zw = log_sum(w), zg = log_sum(g);
  double kl = 0;
  for (size_t i = 0; i < w.size(); i++) kl += std::exp(w[i] - zw) * ((w[i] - zw) - (g[i] - zg));
  return kl;
}

// One round, for the dump.
struct Round {
  uint32_t pos = 0, run = 0, kept = 0;
  int32_t anchor = 0, next = 0;  // the token it started from, and the one sampled after its kept drafts
  std::vector<CardDraft> drafts;
  std::vector<std::vector<Candidate>> rows;
};

// A prompt's text and how it was written.
struct Text {
  std::vector<int32_t> tokens;                // the prompt, then the new tokens
  std::vector<std::vector<Candidate>> from;  // what each new token was sampled from
  std::vector<std::array<uint32_t, 2>> how;  // and where: its round's drafts run, and its row of the round's step
  std::vector<Round> rounds;
  uint32_t written = 0;  // the new tokens after the first, which the rounds wrote
  uint32_t drafts_run = 0, drafts_kept = 0, blocks = 0;
  std::array<uint32_t, D41_DSP_BLOCK> reached{}, kept_at{};
  // The steps by the drafts they ran: their time, its parts waiting for the cards' records and for the CPU's experts,
  // and the routings the CPU answered, those from the file apart.
  std::array<double, D41_DSP_BLOCK + 1> step_ms{}, records_ms{}, cpu_ms{};
  std::array<uint64_t, D41_DSP_BLOCK + 1> steps{}, cpu_routings{}, file_routings{};
  double ms = 0, block_ms = 0, pick_ms = 0;
};

struct Rule {
  engine_settings settings{};
  bool drafting = false;
  double cut = 0.6;
  uint32_t most = D41_DSP_BLOCK, n_new = 48;
};

// The prompt's rows, as many a step as a step takes; returns the last row's candidates.
std::vector<Candidate> read_prompt(deepseek41::Model &model, uint32_t slot, const std::vector<int32_t> &tokens) {
  std::vector<Candidate> last;
  for (uint32_t t = 0; t < tokens.size();) {
    const uint32_t n = std::min<uint32_t>(STEP_MAX_ROWS, (uint32_t)tokens.size() - t);
    std::vector<StepIn> rows;
    for (uint32_t i = 0; i < n; i++) rows.push_back(row_at(slot, t + i, tokens[t + i], tokens));
    auto out = model.step(rows.data(), n);
    last = std::move(out.back());
    t += n;
  }
  return last;
}

Text write(deepseek41::Model &model, uint32_t slot, const Prompt &prompt, const Rule &rule) {
  const std::vector<int32_t> none;
  const engine_settings &st = rule.settings;
  Text w;
  w.tokens = prompt.tokens;
  const uint32_t n = (uint32_t)prompt.tokens.size();
  w.from.push_back(read_prompt(model, slot, w.tokens));
  w.how.push_back({0, 0});
  w.tokens.push_back(sample_p(w.from.back(), st, n, none).first);

  const auto t_start = Clock::now();
  while (w.tokens.back() != kEnd && w.tokens.size() - n < rule.n_new) {
    const uint32_t p = (uint32_t)w.tokens.size() - 1, left = rule.n_new - ((uint32_t)w.tokens.size() - n);
    Round r;
    r.pos = p;
    r.anchor = w.tokens[p];
    uint32_t k = 0;
    if (rule.drafting && left > 1) {
      const deepseek41::Model::DraftBlock block{slot, p, w.tokens[p], D41_DSP_BLOCK};
      auto t0 = Clock::now();
      model.df_pass(nullptr, 0, &block, 1);
      w.block_ms += ms_since(t0);
      t0 = Clock::now();
      // A draft is the sampler's pick with its position's number, so the model keeps it when it picks the same.
      r.drafts = model.drafts([&](uint32_t i, const std::vector<Candidate> &c) { return sample_p(c, st, p + 1 + i, none).first; },
                              (float)rule.cut, std::min(rule.most, left - 1));
      w.pick_ms += ms_since(t0);
      w.blocks++;
      k = (uint32_t)r.drafts.size();
    }
    std::vector<StepIn> rows{row_at(slot, p, w.tokens[p], w.tokens)};
    for (uint32_t j = 0; j < k; j++) rows.push_back(row_after(rows.back(), r.drafts[j].token));
    const auto t0 = Clock::now();
    r.rows = model.step(rows.data(), (uint32_t)rows.size());
    w.step_ms[k] += ms_since(t0);
    w.steps[k]++;
    w.records_ms[k] += model.step_times().records;
    w.cpu_ms[k] += model.step_times().answers;
    w.cpu_routings[k] += model.miss_counts().routings;
    w.file_routings[k] += model.miss_counts().ssd_jobs;
    EXPECT(r.rows.size() == rows.size(), "the step gave %zu rows of candidates", r.rows.size());
    uint32_t a = 0;
    std::vector<int32_t> picks;
    for (;; a++) {
      EXPECT(!r.rows[a].empty(), "position %u: the step gave no candidate", p + a);
      picks.push_back(sample_p(r.rows[a], st, p + a + 1, none).first);
      if (a == k || r.drafts[a].token != picks[a]) break;
    }
    r.run = k;
    r.kept = a;
    r.next = picks[a];
    w.drafts_run += k;
    w.drafts_kept += a;
    for (uint32_t j = 0; j < k && j <= a; j++) w.reached[j]++, w.kept_at[j] += j < a;
    for (uint32_t j = 0; j <= a; j++) {
      w.tokens.push_back(picks[j]);
      w.from.push_back(r.rows[j]);
      w.how.push_back({k, j});
      w.written++;
      if (picks[j] == kEnd) break;
    }
    w.rounds.push_back(std::move(r));
  }
  w.ms = ms_since(t_start);
  return w;
}

void dump_candidates(FILE *f, const std::vector<Candidate> &c) {
  std::fprintf(f, "[");
  for (size_t i = 0; i < c.size(); i++) std::fprintf(f, "%s[%d, %.4f]", i ? ", " : "", c[i].token, c[i].logit);
  std::fprintf(f, "]");
}

void dump(FILE *f, const std::string &id, const Text &w) {
  for (const Round &r : w.rounds) {
    std::fprintf(f, "{\"id\": \"%s\", \"pos\": %u, \"anchor\": %d, \"run\": %u, \"kept\": %u, \"next\": %d, \"drafts\": [",
                 id.c_str(), r.pos, r.anchor, r.run, r.kept, r.next);
    for (size_t i = 0; i < r.drafts.size(); i++) std::fprintf(f, "%s%d", i ? ", " : "", r.drafts[i].token);
    std::fprintf(f, "], \"keep\": [");
    for (size_t i = 0; i < r.drafts.size(); i++) std::fprintf(f, "%s%.4f", i ? ", " : "", r.drafts[i].keep);
    std::fprintf(f, "], \"q\": [");
    for (size_t i = 0; i < r.drafts.size(); i++) {
      if (i) std::fprintf(f, ", ");
      dump_candidates(f, r.drafts[i].candidates);
    }
    std::fprintf(f, "], \"p\": [");
    for (size_t i = 0; i < r.rows.size(); i++) {
      if (i) std::fprintf(f, ", ");
      dump_candidates(f, r.rows[i]);
    }
    std::fprintf(f, "]}\n");
  }
}

int run() {
  if (test::gpus() < 2) return test::skip("deepseek41's step needs two cards");
  const char *path = std::getenv("DS41_MODEL");
  if (!path) return test::skip("DS41_MODEL is not set: the first shard of DeepSeek-V4.1-Flash-1017's file");
  const char *drafter = std::getenv("DS41_DRAFTER");
  const bool compare = drafter && std::getenv("DS41_NO_REF") == nullptr;
  const std::vector<Prompt> all = prompts();

  Rule rule;
  rule.settings.temperature = (float)env_f("DS41_TEMP", 1.0);
  rule.settings.seed = env_u32("DS41_SEED", 1);
  rule.settings.penalty_repeat = 1.0f;
  rule.drafting = drafter != nullptr;
  rule.cut = env_f("DS41_CUT", rule.cut);
  rule.most = std::min<uint32_t>(env_u32("DS41_DRAFTS", D41_DSP_BLOCK), D41_DSP_BLOCK);
  rule.n_new = env_u32("DS41_NEW", rule.n_new);
  EXPECT(rule.n_new > 0, "DS41_NEW is at least 1");

  Options opt;
  opt.slots = 2;
  opt.max_ctx = 1024;
  opt.arena_bytes = (uint64_t)env_u32("DS41_ARENA_GIB", 0) << 30;
  opt.hugetlb_arena = std::getenv("ENGINE_HUGETLB_ARENA") != nullptr;
  opt.experts_per_layer = std::getenv("DS41_CARD_EXPERTS") ? std::atoi(std::getenv("DS41_CARD_EXPERTS")) : -1;
  if (const char *profile = std::getenv("DS41_PROFILE")) opt.routing_profile = profile;
  opt.adapt_swaps = std::getenv("DS41_SWAPS") != nullptr;
  if (drafter) opt.mtp_dir = drafter;
  for (const Prompt &p : all) EXPECT(p.tokens.size() + rule.n_new < opt.max_ctx, "prompt %s is too long", p.id.c_str());
  const auto t_load = Clock::now();
  deepseek41::Model model(path, opt);
  std::printf("loaded in %.1f s: %u experts a layer on the cards, arena %.1f GiB; temperature %.2f, seed %llu, %s\n",
              ms_since(t_load) / 1e3, model.info().experts_per_layer, (double)model.info().arena_bytes / (1ull << 30),
              (double)rule.settings.temperature, (unsigned long long)rule.settings.seed,
              rule.drafting ? "drafts run while the keep chance reaches the cut" : "no drafter: one row a step");
  for (uint32_t s = 0; s < opt.slots; s++) {
    std::vector<uint32_t> pages(model.pages_per_slot());
    for (uint32_t i = 0; i < pages.size(); i++) pages[i] = s * (uint32_t)pages.size() + i;
    model.set_pages(s, 0, pages.data(), (uint32_t)pages.size());
  }

  FILE *out = std::getenv("DS41_DUMP") ? std::fopen(std::getenv("DS41_DUMP"), "w") : nullptr;
  const std::vector<int32_t> none;
  Text sum;
  std::vector<double> kls, one_ms;
  uint32_t same = 0, compared = 0;
  for (const Prompt &prompt : all) {
    const Text w = write(model, 0, prompt, rule);
    const uint32_t n = (uint32_t)prompt.tokens.size();
    const double rounds = (double)std::max<size_t>(w.rounds.size(), 1);
    std::printf("%s: %u tokens in %.0f ms, %.1f a second; %zu rounds, %.2f drafts run and %.2f kept a round; a block %.2f ms, "
                "choosing %.2f ms\n",
                prompt.id.c_str(), w.written, w.ms, w.written * 1e3 / std::max(w.ms, 1e-9), w.rounds.size(),
                w.drafts_run / rounds, w.drafts_kept / rounds, w.block_ms / std::max(w.blocks, 1u),
                w.pick_ms / std::max(w.blocks, 1u));
    if (out) dump(out, prompt.id, w);
    sum.written += w.written, sum.ms += w.ms, sum.block_ms += w.block_ms, sum.pick_ms += w.pick_ms, sum.blocks += w.blocks;
    sum.drafts_run += w.drafts_run, sum.drafts_kept += w.drafts_kept;
    sum.rounds.resize(sum.rounds.size() + w.rounds.size());
    for (uint32_t j = 0; j < D41_DSP_BLOCK; j++) sum.reached[j] += w.reached[j], sum.kept_at[j] += w.kept_at[j];
    for (uint32_t j = 0; j <= D41_DSP_BLOCK; j++) {
      sum.step_ms[j] += w.step_ms[j], sum.steps[j] += w.steps[j], sum.records_ms[j] += w.records_ms[j];
      sum.cpu_ms[j] += w.cpu_ms[j], sum.cpu_routings[j] += w.cpu_routings[j], sum.file_routings[j] += w.file_routings[j];
    }
    std::fflush(stdout);
    if (!compare) continue;

    // The same tokens, one row a step, in a slot no round wrote.
    read_prompt(model, 1, prompt.tokens);
    double worst = 0;
    for (uint32_t t = n; t + 1 < w.tokens.size(); t++) {
      const StepIn row = row_at(1, t, w.tokens[t], w.tokens);
      const auto t0 = Clock::now();
      const auto c = model.step(&row, 1);
      one_ms.push_back(ms_since(t0));
      const double kl = kl_candidates(c[0], w.from[t + 1 - n]);
      EXPECT(!std::isnan(kl), "%s, position %u: the candidates are not numbers", prompt.id.c_str(), t);
      kls.push_back(kl);
      worst = std::max(worst, kl);
      const bool picked = sample_p(c[0], rule.settings, t + 1, none).first == w.tokens[t + 1];
      same += picked;
      compared++;
      if (std::getenv("DS41_POSITIONS"))
        std::printf("  position %u: row %u of a step with %u drafts, KL %.6f, the first candidate %d and %d, logits %.4f and %.4f, pick %s\n",
                    t + 1, w.how[t + 1 - n][1], w.how[t + 1 - n][0], kl, c[0][0].token, w.from[t + 1 - n][0].token,
                    (double)c[0][0].logit, (double)w.from[t + 1 - n][0].logit, picked ? "same" : "differs");
    }
    std::printf("%s: against one row a step, the candidates' worst KL %.5f\n", prompt.id.c_str(), worst);
    std::fflush(stdout);
  }
  if (out) std::fclose(out);

  const double rounds = (double)std::max<size_t>(sum.rounds.size(), 1);
  std::printf("%zu prompts: %u tokens in %.0f ms, %.1f a second; %zu rounds, %.2f drafts run and %.2f kept a round; a block %.2f ms, "
              "choosing %.2f ms\n",
              all.size(), sum.written, sum.ms, sum.written * 1e3 / std::max(sum.ms, 1e-9), sum.rounds.size(),
              sum.drafts_run / rounds, sum.drafts_kept / rounds, sum.block_ms / std::max(sum.blocks, 1u),
              sum.pick_ms / std::max(sum.blocks, 1u));
  for (uint32_t j = 0; j < D41_DSP_BLOCK && sum.reached[j]; j++)
    std::printf("  draft %u: kept %u of %u checked\n", j + 1, sum.kept_at[j], sum.reached[j]);
  for (uint32_t j = 0; j <= D41_DSP_BLOCK; j++) {
    if (!sum.steps[j]) continue;
    const double n = (double)sum.steps[j], rows = n * (j + 1);
    std::printf("  a step with %u drafts: %.1f ms (waiting for the cards' records %.1f, the CPU's experts %.1f), %llu steps; the CPU "
                "answered %.1f routings a row, %.2f of them from the file\n",
                j, sum.step_ms[j] / n, sum.records_ms[j] / n, sum.cpu_ms[j] / n, (unsigned long long)sum.steps[j],
                (double)sum.cpu_routings[j] / rows, (double)sum.file_routings[j] / rows);
  }
  if (!drafter) return test::skip("DS41_DRAFTER is not set: no draft was checked");
  if (!compare) return test::skip("DS41_NO_REF: nothing was compared");

  EXPECT(compared > 0, "no position was compared");
  std::sort(kls.begin(), kls.end());
  std::sort(one_ms.begin(), one_ms.end());
  const double median = kls[kls.size() / 2];
  std::printf("against one row a step, %u positions: the candidates' KL median %.5f, a tenth above %.5f, worst %.5f; the sampler "
              "picks the round's token at %u; a step of one row: median %.1f ms\n",
              compared, median, kls[kls.size() * 9 / 10], kls.back(), same, one_ms[one_ms.size() / 2]);
  EXPECT(sum.drafts_run > 0, "no draft reached the cut, so no round checked one");
  EXPECT(median <= kMedianKl, "the candidates' median KL against one row a step is %.5f, above %.3f", median, kMedianKl);
  EXPECT(same * 100 >= compared * kSamePicks, "the sampler picks the round's token at %u of %u positions", same, compared);
  std::printf("ds41_decode_test: ok\n");
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
