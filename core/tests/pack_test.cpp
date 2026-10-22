// Level 6 (docs/TESTING.md): a prompt chunk of several sequences gives each what it gets alone, bit for bit: each
// sequence's 64 candidates after its prompt and after each of 16 greedy tokens. The chunks: the four gpqa prompts;
// three of them, one from position 101; and four runs whose later ones hold an image's block (the encoder's file is
// FLASH_MODEL_DIR's mmproj) or start deep, past the indexer's 2,051 cells and past p_attn_tiled's 12,288 positions.
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "models/qwen4exp/model.h"
#include "test_util.h"

using namespace engine;
using qwen4exp::Candidate;
using qwen4exp::kNoToken;

namespace {

std::vector<int32_t> ints_after(const std::string &line, const std::string &key) {
  std::vector<int32_t> v;
  size_t p = line.find("\"" + key + "\":[");
  if (p == std::string::npos) p = line.find("\"" + key + "\": [");
  if (p == std::string::npos) return v;
  p = line.find('[', p) + 1;
  while (line[p] != ']') {
    size_t q;
    v.push_back(std::stoi(line.substr(p), &q));
    p += q;
    if (line[p] == ',') p++;
    if (line[p] == ' ') p++;
  }
  return v;
}

// A line's "id", with or without a space after the colon.
std::string id_of(const std::string &line) {
  size_t p = line.find("\"id\":");
  if (p == std::string::npos) return "";
  p = line.find('"', p + 5) + 1;
  return line.substr(p, line.find('"', p) - p);
}

bool same(const std::vector<Candidate> &a, const std::vector<Candidate> &b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); i++)
    if (a[i].token != b[i].token || std::memcmp(&a[i].logit, &b[i].logit, 4) != 0) return false;
  return true;
}

constexpr int32_t kImagePad = 248056;  // <|image_pad|>, as core/api/engine_api.cpp writes an image's block

}  // namespace

int main(int argc, char **argv) {
  if (test::gpus() == 0) return test::no_gpu();
  std::string path = test::model_path();
  if (path.empty()) return test::skip("model not found under FLASH_MODEL_DIR");
  EXPECT(argc >= 2, "usage: pack_test <reference dir>");
  const char *md = std::getenv("FLASH_MODEL_DIR");
  const std::string vision = std::string(md ? md : "/models") + "/mmproj-Qwen3.8-Flash-Next-BF16.gguf";
  if (!std::ifstream(vision)) return test::skip("no " + vision);
  const uint32_t S = 4, D = 16, CUT = 101, MID = 3000, DEEP = 12416, RUN = 320;
  std::map<std::string, std::vector<int32_t>> prompts;
  {
    std::ifstream pf(std::string(argv[1]) + "/prompts.jsonl");
    std::string line;
    while (std::getline(pf, line)) prompts[id_of(line)] = ints_after(line, "tokens");
  }
  std::vector<std::vector<int32_t>> seqs(S);
  for (uint32_t s = 0; s < S; s++) {
    seqs[s] = prompts["gpqa-" + std::to_string(s + 1)];
    EXPECT(seqs[s].size() > CUT, "reference prompt gpqa-%u not found", s + 1);
  }
  EXPECT(prompts["agent-turn4"].size() >= MID + RUN, "reference prompt agent-turn4 not found");
  EXPECT(prompts["doc-32k"].size() >= DEEP + RUN, "reference prompt doc-32k not found");
  qwen4exp::Options opt;
  opt.slots = S;
  opt.max_ctx = 16384;
  opt.vision_path = vision;
  qwen4exp::Model model(path, opt);
  auto run = [&](uint32_t s, uint32_t from) {
    const auto &q = seqs[s];
    return qwen4exp::Model::PromptRun{s, from, q.data() + from, (uint32_t)q.size() - from,
                                      from ? q[from - 1] : kNoToken, from > 1 ? q[from - 2] : kNoToken};
  };
  // The prompt's candidates, then D greedy tokens' each.
  auto decode = [&](uint32_t s, const std::vector<Candidate> &first) {
    const auto &q = seqs[s];
    std::vector<std::vector<Candidate>> out{first};
    int32_t p1 = q[q.size() - 1], p2 = q[q.size() - 2];
    for (uint32_t i = 0; i < D; i++) {
      const int32_t tok = out.back()[0].token;
      qwen4exp::StepIn r{s, (uint32_t)q.size() + i, tok, p1, p2};
      out.push_back(model.step(&r, 1)[0]);
      p2 = p1;
      p1 = tok;
    }
    return out;
  };
  auto differ = [&](const std::vector<std::vector<Candidate>> &a, const std::vector<std::vector<Candidate>> &b) {
    uint32_t bad = 0;
    for (size_t i = 0; i < a.size(); i++) bad += !same(a[i], b[i]);
    return bad;
  };
  // Alone, each prompt whole; and gpqa-2 in two pieces.
  std::vector<std::vector<std::vector<Candidate>>> alone(S);
  for (uint32_t s = 0; s < S; s++) {
    model.reset_slot(s);
    auto r = run(s, 0);
    alone[s] = decode(s, model.prompt(r.slot, r.pos0, r.tokens, r.n, r.p1, r.p2));
  }
  model.reset_slot(1);
  model.prompt(1, 0, seqs[1].data(), CUT, kNoToken, kNoToken);
  auto rest = run(1, CUT);
  auto alone_cut = decode(1, model.prompt(rest.slot, rest.pos0, rest.tokens, rest.n, rest.p1, rest.p2));
  int failures = 0;
  {
    for (uint32_t s = 0; s < S; s++) model.reset_slot(s);
    std::vector<qwen4exp::Model::PromptRun> runs;
    for (uint32_t s = 0; s < S; s++) runs.push_back(run(s, 0));
    auto cand = model.prompt_runs(runs.data(), S);
    uint32_t bad = 0;
    for (uint32_t s = 0; s < S; s++) bad += differ(decode(s, cand[s]), alone[s]);
    std::printf("four prompts in one chunk: %u of %u candidate lists differ from alone\n", bad, S * (D + 1));
    failures += bad != 0;
  }
  {
    for (uint32_t s = 0; s < 3; s++) model.reset_slot(s);
    model.prompt(1, 0, seqs[1].data(), CUT, kNoToken, kNoToken);
    std::vector<qwen4exp::Model::PromptRun> runs{run(2, 0), run(1, CUT), run(0, 0)};
    auto cand = model.prompt_runs(runs.data(), 3);
    uint32_t bad = differ(decode(2, cand[0]), alone[2]) + differ(decode(1, cand[1]), alone_cut) +
                   differ(decode(0, cand[2]), alone[0]);
    std::printf("three runs, one from position %u: %u of %u candidate lists differ from alone\n", CUT, bad,
                3 * (D + 1));
    failures += bad != 0;
  }
  {
    // Slot 1: gpqa-2 with an image's block of 64 tokens (16 x 16 patches of fixed pseudo-random pixels) at index 24.
    // Slots 2 and 3: the deep prompts, cut where their run ends.
    qwen4exp::Image im;
    im.first = 24;
    im.grid_h = im.grid_w = 16;
    std::vector<uint8_t> pixels((size_t)im.grid_h * im.grid_w * 768);
    uint32_t x = 12345;
    for (auto &p : pixels) p = (uint8_t)((x = x * 1664525u + 1013904223u) >> 24);
    im.pixels = pixels.data();
    seqs[1] = prompts["gpqa-2"];
    seqs[1].insert(seqs[1].begin() + im.first, im.grid_h / 2 * (im.grid_w / 2), kImagePad);
    seqs[2].assign(prompts["agent-turn4"].begin(), prompts["agent-turn4"].begin() + MID + RUN);
    seqs[3].assign(prompts["doc-32k"].begin(), prompts["doc-32k"].begin() + DEEP + RUN);
    const uint32_t from[S] = {0, 0, MID, DEEP};
    // A slot made ready for its run: reset, its image set, its positions before the run read alone in pieces.
    auto ready = [&](uint32_t s) {
      model.reset_slot(s);
      if (s == 1) model.set_images(1, &im, 1);
      const auto &q = seqs[s];
      for (uint32_t p = 0; p < from[s];) {
        const uint32_t n = std::min(model.prompt_chunk(), from[s] - p);
        model.prompt(s, p, q.data() + p, n, p ? q[p - 1] : kNoToken, p > 1 ? q[p - 2] : kNoToken);
        p += n;
      }
    };
    std::vector<std::vector<std::vector<Candidate>>> deep_alone(S);
    deep_alone[0] = alone[0];
    for (uint32_t s = 1; s < S; s++) {
      ready(s);
      auto r = run(s, from[s]);
      deep_alone[s] = decode(s, model.prompt(r.slot, r.pos0, r.tokens, r.n, r.p1, r.p2));
    }
    std::vector<qwen4exp::Model::PromptRun> runs;
    model.reset_slot(0);
    for (uint32_t s = 1; s < S; s++) ready(s);
    for (uint32_t s = 0; s < S; s++) runs.push_back(run(s, from[s]));
    auto cand = model.prompt_runs(runs.data(), S);
    uint32_t bad = 0;
    for (uint32_t s = 0; s < S; s++) {
      const uint32_t b = differ(decode(s, cand[s]), deep_alone[s]);
      if (b) std::printf("  the run of slot %u, from position %u: %u differ\n", s, from[s], b);
      bad += b;
    }
    std::printf("four runs, an image's block in the second, the third from position %u, the fourth from %u: %u of "
                "%u candidate lists differ from alone\n",
                MID, DEEP, bad, S * (D + 1));
    failures += bad != 0;
  }
  EXPECT(failures == 0, "runs of a prompt chunk differ from their prompts read alone");
  std::printf("PASS\n");
  return 0;
}
