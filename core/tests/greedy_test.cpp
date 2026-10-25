// Level 5 (docs/TESTING.md): the engine's greedy choice against llama.cpp's greedy tokens for the reference prompts
// (reference/, docs/measurements/REFERENCE_OUTPUTS.md), teacher-forced: at each position after the prompt the engine's top token
// is compared with the reference's next. Gate: at least 95% of positions on prompts of up to 512 tokens; longer ones
// are reported only. A chunk size reads the prompt through the prompt path in chunks, then one token per step.
// greedy_test <reference dir> [max prompt tokens, default 1024] [cards, default 2] [prompt chunk, default 0: none]
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <sstream>

#include "models/qwen4exp/model.h"
#include "test_util.h"

using namespace engine;

namespace {

struct Ref {
  std::string id;
  std::vector<int32_t> prompt, out;
};

std::vector<int32_t> ints_after(const std::string &line, const std::string &key) {
  std::vector<int32_t> v;
  size_t p = line.find("\"" + key + "\":[");
  if (p == std::string::npos) return v;
  p = line.find('[', p) + 1;
  while (line[p] != ']') {
    size_t q;
    v.push_back(std::stoi(line.substr(p), &q));
    p += q;
    if (line[p] == ',') p++;
  }
  return v;
}

std::string str_after(const std::string &line, const std::string &key) {
  size_t p = line.find("\"" + key + "\":\"");
  if (p == std::string::npos) return "";
  p += key.size() + 4;
  return line.substr(p, line.find('"', p) - p);
}

}  // namespace

int main(int argc, char **argv) {
  if (test::gpus() == 0) return test::no_gpu();
  std::string path = test::model_path();
  if (path.empty()) return test::skip("model not found under FLASH_MODEL_DIR");
  EXPECT(argc >= 2, "usage: greedy_test <reference dir> [max prompt tokens] [cards]");
  std::string dir = argv[1];
  size_t max_prompt = argc > 2 ? std::stoul(argv[2]) : 1024;
  uint32_t cards = argc > 3 ? (uint32_t)std::stoul(argv[3]) : 2;
  uint32_t chunk = argc > 4 ? (uint32_t)std::stoul(argv[4]) : 0;

  std::vector<Ref> refs;
  {
    std::ifstream pf(dir + "/prompts.jsonl"), gf(dir + "/llamacpp-3cf0325-IQ3_S-greedy.jsonl");
    EXPECT(pf && gf, "cannot read the reference files in %s", dir.c_str());
    std::string line;
    std::vector<std::pair<std::string, std::vector<int32_t>>> prompts;
    while (std::getline(pf, line)) prompts.push_back({str_after(line, "id"), ints_after(line, "tokens")});
    while (std::getline(gf, line)) {
      std::string id = str_after(line, "id");
      if (id.empty()) continue;
      for (auto &p : prompts)
        if (p.first == id) refs.push_back({id, p.second, ints_after(line, "tokens")});
    }
  }
  EXPECT(!refs.empty(), "no reference prompts read");

  qwen4exp::Options opt;
  opt.max_cards = cards;
  opt.max_ctx = 8192;
  opt.prompt_chunk = chunk ? std::max<uint32_t>(64, (chunk + 63) / 64 * 64) : 0;
  // FLASH_SWAPS=1: adaptive swaps on, a round every 16 steps (and after each prompt), as the engine runs them.
  const bool swaps = std::getenv("FLASH_SWAPS") != nullptr;
  opt.adapt_swaps = swaps;
  // FLASH_GROUPS=n: the cards in n groups (Options::groups).
  if (const char *g = std::getenv("FLASH_GROUPS")) opt.groups = (uint32_t)std::stoul(g);
  qwen4exp::Model model(path, opt);
  const auto &info = model.info();
  std::printf("loaded in %.1f s: %u card(s) in %u group(s), %u experts per layer cached, %.2f / %.2f GiB on the cards, "
              "%.2f GiB arena\n",
              info.seconds, info.cards, info.groups, info.experts_per_layer, info.device_bytes[0] / 1073741824.0,
              info.device_bytes[1] / 1073741824.0, info.arena_bytes / 1073741824.0);

  uint64_t gated_ok = 0, gated_n = 0;
  for (auto &r : refs) {
    if (r.prompt.size() > max_prompt) {
      std::printf("%-16s skipped: %zu prompt tokens\n", r.id.c_str(), r.prompt.size());
      continue;
    }
    std::vector<int32_t> seq = r.prompt;
    seq.insert(seq.end(), r.out.begin(), r.out.end());
    model.reset_slot(0);
    uint32_t ok = 0, n = 0, misses = 0;
    std::string first_diff;
    auto t0 = std::chrono::steady_clock::now();
    double prompt_s = 0;
    size_t first = 0;  // positions before it were read by the prompt path
    std::vector<qwen4exp::Candidate> last;
    if (chunk) {
      size_t n = r.prompt.size();
      size_t pieces = (n + chunk - 1) / chunk, at = 0;
      for (size_t k = 0; k < pieces; k++) {
        size_t len = (n - at) / (pieces - k);
        int32_t p1 = at >= 1 ? seq[at - 1] : qwen4exp::kNoToken, p2 = at >= 2 ? seq[at - 2] : qwen4exp::kNoToken;
        last = model.prompt(0, (uint32_t)at, seq.data() + at, (uint32_t)len, p1, p2);
        at += len;
        if (swaps) {
          model.note_prompt_routing();
          model.swap_round();
        }
      }
      first = n;
      prompt_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
    for (size_t i = first ? first - 1 : 0; i + 1 < seq.size(); i++) {
      int32_t p1 = i >= 1 ? seq[i - 1] : qwen4exp::kNoToken, p2 = i >= 2 ? seq[i - 2] : qwen4exp::kNoToken;
      std::vector<qwen4exp::Candidate> cand;
      if (first && i == first - 1) {
        cand = last;
      } else {
        cand = model.step(0, (uint32_t)i, seq[i], p1, p2);
        misses += model.last_stats().misses;
        if (swaps && i % 16 == 15) model.swap_round();
      }
      if (i + 1 >= r.prompt.size()) {
        n++;
        if (!cand.empty() && cand[0].token == seq[i + 1]) {
          ok++;
        } else if (first_diff.empty()) {
          char buf[128];
          std::snprintf(buf, sizeof buf, " first difference at output %zu: %d (%.3f) vs %d", i + 1 - r.prompt.size(),
                        cand.empty() ? -1 : cand[0].token, cand.empty() ? 0.0f : cand[0].logit, seq[i + 1]);
          first_diff = buf;
        }
      }
    }
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    size_t decoded = seq.size() - 1 - (first ? first - 1 : 0);
    std::printf("%-16s %4zu prompt + %3zu: %3u/%3u agree (%.1f%%), prompt %.0f tokens/s, decode %.1f ms/token, "
                "%.2f misses/token%s\n", r.id.c_str(), r.prompt.size(), r.out.size(), ok, n, 100.0 * ok / n,
                first ? first / prompt_s : 0.0, 1000 * (s - prompt_s) / decoded, (double)misses / decoded,
                first_diff.c_str());
    std::fflush(stdout);
    if (r.prompt.size() <= 512) gated_ok += ok, gated_n += n;
  }
  double rate = gated_n ? (double)gated_ok / gated_n : 0;
  if (swaps) std::printf("%llu experts swapped in\n", (unsigned long long)model.swaps());
  std::printf("prompts of up to 512 tokens: %llu/%llu = %.2f%% agree\n", (unsigned long long)gated_ok,
              (unsigned long long)gated_n, 100 * rate);
  EXPECT(rate >= 0.95, "below the 95%% gate");
  std::printf("PASS\n");
  return 0;
}
