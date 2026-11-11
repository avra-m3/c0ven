// deepseek41's drafter on the cards against RefDrafter on the CPU, on the real files. RefDrafter is fed the cards' own
// features, the model's streams as its step left them, so the two differ by the drafter's rounding alone: the keys a
// step stores, then blocks drafted from the same context: the streams after each sublayer, the rows before the output
// norm, the logits, the keep chances and the drafts. The model's own step is compared in ds41_model_test. Needs two
// cards, DS41_MODEL (the model's first shard) and DS41_DRAFTER (the drafter's file). DS41_STEPS: positions run;
// DS41_TOKENS: a dump of upstream's whose tokens start the sequence; DS41_ARENA_GIB, DS41_CARD_EXPERTS, DS41_PROFILE:
// the model's tiers; DS41_TRACE: the last block's kernels; DS41_BLOCK: rows a block; DS41_REF: RefModel runs too and
// feeds a second RefDrafter, and the share of the cards' drafts that are its drafts is printed, not judged.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "models/deepseek41/drafter.h"
#include "models/deepseek41/model.h"
#include "models/deepseek41/reference.h"
#include "test_util.h"

using namespace engine;
using namespace engine::deepseek41;

namespace {

// Against RefDrafter fed the same features. The cards round activations to 8 bits a product and the router's weights
// to bf16, RefDrafter does not: a row whose third and fourth experts are near a tie may route otherwise, and a block
// with such a row is judged by its drafts alone.
constexpr double kKeyLimit = 0.01, kStreamLimit = 0.05;
// Logits in nats where the routings are the same, and the keep chance.
constexpr double kMedianKl = 0.003, kWorstKl = 0.02, kKeepLimit = 0.02;
constexpr double kBiasLimit = 1e-4;  // in logits
constexpr uint32_t kFirstBlock = 4;  // positions run before the first block

double rel_rms(const float *got, const float *want, size_t n) {
  double d = 0, w = 0;
  for (size_t i = 0; i < n; i++) {
    d += ((double)got[i] - want[i]) * ((double)got[i] - want[i]);
    w += (double)want[i] * want[i];
  }
  return std::sqrt(d / std::max(w, 1e-30));
}

// KL(want || got) of two rows of logits, in nats.
double kl_logits(const float *want, const float *got, size_t n) {
  const auto log_sum = [n](const float *x) {
    const double top = *std::max_element(x, x + n);
    double z = 0;
    for (size_t i = 0; i < n; i++) z += std::exp((double)x[i] - top);
    return top + std::log(z);
  };
  const double zw = log_sum(want), zg = log_sum(got);
  double kl = 0;
  for (size_t i = 0; i < n; i++) kl += std::exp(want[i] - zw) * ((want[i] - zw) - (got[i] - zg));
  return kl;
}

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

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v.empty() ? 0.0 : v[v.size() / 2];
}

int run() {
  if (test::gpus() < 2) return test::skip("deepseek41's step needs two cards");
  const char *path = std::getenv("DS41_MODEL");
  if (!path) return test::skip("DS41_MODEL is not set: the first shard of DeepSeek-V4.1-Flash-1017's file");
  const char *draft_path = std::getenv("DS41_DRAFTER");
  if (!draft_path) return test::skip("DS41_DRAFTER is not set: the drafter's file of DeepSeek-V4.1-Flash-1017");
  const uint32_t steps = env_u32("DS41_STEPS", 16), nblock = env_u32("DS41_BLOCK", D41_DSP_BLOCK);
  EXPECT(steps > kFirstBlock && nblock >= 1 && nblock <= D41_DSP_BLOCK, "DS41_STEPS above %u, DS41_BLOCK 1 to %d", kFirstBlock,
         D41_DSP_BLOCK);

  std::vector<int32_t> tokens = {0, 9211, 1418, 344, 31032, 5106};
  if (const char *dump = std::getenv("DS41_TOKENS")) tokens = dump_tokens(dump);
  const bool with_ref = std::getenv("DS41_REF") != nullptr;
  RefModel ref(path, RefOptions{16, steps});
  RefDrafter refd(draft_path, ref, 16);
  std::unique_ptr<RefDrafter> ref_fed = with_ref ? std::make_unique<RefDrafter>(draft_path, ref, 16) : nullptr;
  const uint32_t vocab = ref.vocab();

  Options opt;
  opt.slots = 1;
  opt.max_ctx = 1024;
  opt.arena_bytes = (uint64_t)env_u32("DS41_ARENA_GIB", 0) << 30;
  opt.hugetlb_arena = std::getenv("ENGINE_HUGETLB_ARENA") != nullptr;
  opt.experts_per_layer = std::getenv("DS41_CARD_EXPERTS") ? std::atoi(std::getenv("DS41_CARD_EXPERTS")) : -1;
  if (const char *profile = std::getenv("DS41_PROFILE")) opt.routing_profile = profile;
  opt.trace = std::getenv("DS41_TRACE") != nullptr;
  opt.mtp_dir = draft_path;
  const auto t_load = std::chrono::steady_clock::now();
  deepseek41::Model model(path, opt, Debug{true});
  std::printf("loaded in %.1f s: %u experts a layer of the model on the cards, %.1f and %.1f GiB in use\n",
              std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load).count(),
              model.info().experts_per_layer, (double)model.info().device_bytes[0] / (1ull << 30),
              (double)model.info().device_bytes[1] / (1ull << 30));
  EXPECT(model.drafting() && model.dflash() && model.noise_token() == refd.noise_token(), "the drafter's noise token");
  std::vector<uint32_t> pages(model.pages_per_slot());
  for (uint32_t i = 0; i < pages.size(); i++) pages[i] = i;
  model.set_pages(0, 0, pages.data(), (uint32_t)pages.size());

  using Values = std::map<std::pair<std::string, int>, std::vector<float>>;
  std::vector<float> want(vocab), features(D41_DSP_FEATURES);
  std::vector<double> kls, block_ms, chain_us, keep_diff, step_ms;
  double worst_key = 0, worst_input = 0, worst_stream = 0, worst_out = 0, worst_bias = 0, worst_first = 0;
  uint32_t drafts = 0, same = 0, outside = 0, first_same = 0, blocks = 0, ref_drafts = 0, ref_same = 0;
  uint32_t routings = 0, reroutes = 0;
  double worst_rerouted = 0;  // the streams of a block with a row routed to other experts
  double worst_kl = 0;        // of the blocks without one
  for (uint32_t t = 0; t < steps; t++) {
    Values fed;
    if (with_ref) {
      ref.decode(tokens[t], want.data());
      ref_fed->feed();
    }
    const StepIn row = row_at(0, t, tokens[t], tokens);
    const auto t0 = std::chrono::steady_clock::now();
    const auto step = model.step(&row, 1);
    if (t) step_ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    EXPECT(step.size() == 1 && !step[0].empty(), "position %u: the step gave no candidate", t);
    if (tokens.size() <= t + 1) tokens.push_back(step[0][0].token);
    // The features as the cards' step had them: the mean of the streams that enter each target layer.
    for (uint32_t k = 0; k < D41_DSP_TARGETS; k++) {
      const float *X = model.probe(D41_DSP_TARGET0 + k - 1, true);
      for (uint32_t i = 0; i < D41_EMBD; i++)
        features[(size_t)k * D41_EMBD + i] =
            0.25f * (X[i] + X[D41_EMBD + i] + X[2 * D41_EMBD + i] + X[3 * D41_EMBD + i]);
    }
    refd.feed(t, features.data(),
              [&](const char *name, int layer, const float *v, size_t count) { fed[{name, layer}].assign(v, v + count); });
    const double input = rel_rms(model.draft_input(), fed.at({"main_x", -1}).data(), D41_EMBD);
    double key = 0;
    for (uint32_t il = 0; il < D41_DSP_LAYERS; il++)
      key = std::max(key, rel_rms(model.draft_key(il), fed.at({"kv_injected", (int)il}).data(), D41_HD));
    EXPECT(!std::isnan(input) && !std::isnan(key), "position %u: a stored value is not a number", t);
    worst_input = std::max(worst_input, input);
    worst_key = std::max(worst_key, key);
    if (t + 1 < kFirstBlock) continue;

    // A block after position t: its first row is the next token.
    const int32_t anchor = tokens[t + 1];
    Values got;
    const std::vector<Draft> wanted =
        refd.draft(anchor, t + 1, nblock, nullptr,
                   [&](const char *name, int layer, const float *v, size_t count) {
                     const std::string s = name;
                     if (s == "hc_attn_post" || s == "l_out" || s == "hc_out" || s == "result_output" || s == "dsp_logits")
                       got[{s, layer}].assign(v, v + count);
                   });
    const deepseek41::Model::DraftBlock block{0, t + 1, anchor, nblock};
    const auto t1 = std::chrono::steady_clock::now();
    const auto cands = model.df_pass(nullptr, 0, &block, 1);
    const auto t2 = std::chrono::steady_clock::now();
    const std::vector<CardDraft> made = model.drafts();
    const auto t3 = std::chrono::steady_clock::now();
    if (blocks) {
      block_ms.push_back(std::chrono::duration<double, std::milli>(t2 - t1).count());
      chain_us.push_back(std::chrono::duration<double, std::micro>(t3 - t2).count());
    }
    blocks++;
    if (with_ref) {
      const std::vector<Draft> theirs = ref_fed->draft(anchor, t + 1, nblock);
      for (uint32_t i = 0; i < nblock; i++) {
        ref_drafts++;
        ref_same += made[i].token == theirs[i].token;
        if (made[i].token != theirs[i].token) break;
      }
    }
    EXPECT(cands.size() == 1 && cands[0].empty() && made.size() == nblock, "the block's rows");
    EXPECT(!made[0].candidates.empty(), "the first draft's candidates");

    double stream = 0;
    int stream_layer = -1;
    for (uint32_t il = 0; il < D41_DSP_LAYERS; il++)
      for (const bool ffn : {false, true}) {
        const std::vector<float> &v = got.at({ffn ? "l_out" : "hc_attn_post", (int)il});
        EXPECT(v.size() == (size_t)nblock * D41_HC_DIM, "RefDrafter's streams hold %zu values", v.size());
        const double e = rel_rms(model.probe(D41_LAYERS + il, ffn), v.data(), v.size());
        EXPECT(!std::isnan(e), "position %u, drafter layer %u: a stream is not a number", t, il);
        if (std::getenv("DS41_STREAMS")) std::printf("  block at %u, layer %u %s: %.2e\n", t + 1, il, ffn ? "ffn" : "attention", e);
        if (e > stream) stream = e, stream_layer = (int)il;
      }
    // A row routed to other experts than RefDrafter's has other streams from that layer on.
    uint32_t rerouted = 0;
    for (uint32_t il = 0; il < D41_DSP_LAYERS; il++)
      for (uint32_t i = 0; i < nblock; i++) {
        std::vector<int32_t> a(model.draft_route(il, i), model.draft_route(il, i) + D41_DSP_USED);
        std::vector<int32_t> b(refd.route(il, i), refd.route(il, i) + D41_DSP_USED);
        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());
        rerouted += a != b;
      }
    routings += D41_DSP_LAYERS * nblock;
    reroutes += rerouted;
    (rerouted ? worst_rerouted : worst_stream) = std::max(rerouted ? worst_rerouted : worst_stream, stream);
    const std::vector<float> &out = got.at({"hc_out", -1}), &logits = got.at({"result_output", -1});
    const std::vector<float> &biased = got.at({"dsp_logits", -1});
    double out_err = 0, kl_block = 0;
    std::string line;
    for (uint32_t i = 0; i < nblock; i++) {
      out_err = std::max(out_err, rel_rms(model.draft_out(i), out.data() + (size_t)i * D41_EMBD, D41_EMBD));
      const std::vector<float> have = model.logits(i);
      const float *base = logits.data() + (size_t)i * vocab;
      const double kl = kl_logits(base, have.data(), vocab);
      kls.push_back(kl);
      kl_block = std::max(kl_block, kl);
      // A draft differs from RefDrafter's only by the rows' values while the drafts before it are the same.
      bool before = true;
      for (uint32_t j = 0; j < i; j++) before = before && made[j].token == wanted[j].token;
      if (before) {
        // The cards' bias is RefDrafter's for the same token before, and their first candidate is the largest of
        // their own logits with it.
        const float *theirs = biased.data() + (size_t)i * vocab;
        double best = -1e30;
        for (uint32_t v = 0; v < vocab; v++) best = std::max(best, (double)have[v] + theirs[v] - base[v]);
        for (const Candidate &c : made[i].candidates)
          worst_bias = std::max(worst_bias, std::fabs((double)c.logit - have[c.token] - ((double)theirs[c.token] - base[c.token])));
        const Candidate &c0 = made[i].candidates[0];
        worst_first = std::max(worst_first, best - ((double)have[c0.token] + theirs[c0.token] - base[c0.token]));
        drafts++;
        same += made[i].token == wanted[i].token;
        first_same += i == 0 && made[i].token == wanted[i].token;
        keep_diff.push_back(std::fabs((double)made[i].keep - wanted[i].keep));
        const auto &c = made[i].candidates;
        outside += std::none_of(c.begin(), c.end(), [&](const Candidate &x) { return x.token == wanted[i].token; });
      }
      line += " " + std::to_string(made[i].token) + (made[i].token == wanted[i].token ? "" : "!=" + std::to_string(wanted[i].token));
    }
    if (!rerouted) worst_out = std::max(worst_out, out_err), worst_kl = std::max(worst_kl, kl_block);
    std::printf("position %u: stored input %.2e, keys %.2e; block: streams worst %.2e (layer %d), rows before the norm "
                "%.2e, logits KL worst %.5f, %u rows routed otherwise, drafts%s\n",
                t, input, key, stream, stream_layer, out_err, kl_block, rerouted, line.c_str());
    std::fflush(stdout);
  }

  double keep_mean = 0;
  for (double d : keep_diff) keep_mean += d / (double)std::max<size_t>(keep_diff.size(), 1);
  const double kl_median = median(kls), kl_worst = worst_kl;
  std::printf("%u blocks of %u rows: logits KL median %.5f, worst %.5f where the routings are the same; %u of %u drafts are RefDrafter's, %u of %u first "
              "drafts, %u of RefDrafter's outside the candidates; keep chance differs by %.4f on average\n",
              blocks, nblock, kl_median, kl_worst, same, drafts, first_same, blocks, outside, keep_mean);
  std::printf("stored: input worst %.2e, keys worst %.2e; block: streams worst %.2e, rows before the norm worst %.2e\n",
              worst_input, worst_key, worst_stream, worst_out);
  std::printf("%u of %u routings of a block's rows are not RefDrafter's; the streams of a block with one differ by up to %.2e\n",
              reroutes, routings, worst_rerouted);
  std::printf("the bias of the token before: worst %.2e from RefDrafter's; a first candidate at most %.2e below the largest\n",
              worst_bias, worst_first);
  std::printf("a block after the first: median %.2f ms on the cards, %.0f us for the drafts' chain; a step of one row: "
              "median %.1f ms\n",
              median(block_ms), median(chain_us), median(step_ms));
  if (with_ref)
    std::printf("against RefDrafter fed by RefModel: %u of %u drafts the same, up to each block's first that differs\n",
                ref_same, ref_drafts);
  for (uint32_t c = 0; c < 2 && opt.trace; c++) {
    std::vector<TraceRow> rows = model.trace(c);
    double sum = 0;
    for (const TraceRow &r : rows) sum += r.ms;
    std::sort(rows.begin(), rows.end(), [](const TraceRow &a, const TraceRow &b) { return a.ms > b.ms; });
    std::printf("card %u, the last block's list: %.2f ms\n", c, sum);
    for (const TraceRow &r : rows) std::printf("  %-40s %4u launches %7.3f ms\n", r.kernel.c_str(), r.launches, r.ms);
  }
  EXPECT(worst_input <= kKeyLimit && worst_key <= kKeyLimit, "what a step stores for the drafter: input %.2e, keys %.2e",
         worst_input, worst_key);
  EXPECT(worst_stream <= kStreamLimit && worst_out <= kStreamLimit, "the block's streams differ by %.2e, its rows by %.2e",
         worst_stream, worst_out);
  EXPECT(worst_bias <= kBiasLimit && worst_first <= kBiasLimit, "the bias differs by %.2e, a first candidate by %.2e",
         worst_bias, worst_first);
  EXPECT(reroutes * 10 <= routings, "%u of %u routings are not RefDrafter's", reroutes, routings);
  EXPECT(outside == 0, "%u of RefDrafter's drafts are outside the cards' candidates", outside);
  EXPECT(kl_median <= kMedianKl && kl_worst <= kWorstKl, "the block's logits: KL median %.5f, worst %.5f", kl_median, kl_worst);
  EXPECT(same * 10 >= drafts * 8, "%u of %u drafts are RefDrafter's", same, drafts);
  EXPECT(keep_mean <= kKeepLimit, "the keep chance differs by %.4f on average", keep_mean);
  std::printf("ds41_card_drafter_test: ok\n");
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
