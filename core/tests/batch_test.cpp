// Level 6 (docs/TESTING.md): a decode step of several rows gives each row exactly what it gets alone. Eight
// sequences (the reference prompts' first 24 tokens) run one token per step, alone, then eight and four per step;
// then one as windows of 4 positions, one as draft windows with wrong drafts, committing the rows before the first
// wrong one, and three as draft windows in one step, each committing its own. Gate: every row's 64 candidates
// bit-identical to its single-row run (each row is computed alone in every kernel, the CPU's misses are per row, and
// a commit replays the accepted rows' state).
#include <algorithm>
#include <cstring>
#include <fstream>

#include "models/qwen4exp/model.h"
#include "test_util.h"

using namespace engine;

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
  }
  return v;
}

bool same(const std::vector<qwen4exp::Candidate> &a, const std::vector<qwen4exp::Candidate> &b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); i++)
    if (a[i].token != b[i].token || std::memcmp(&a[i].logit, &b[i].logit, 4) != 0) return false;
  return true;
}

}  // namespace

int main(int argc, char **argv) {
  if (test::gpus() == 0) return test::no_gpu();
  std::string path = test::model_path();
  if (path.empty()) return test::skip("model not found under FLASH_MODEL_DIR");
  EXPECT(argc >= 2, "usage: batch_test <reference dir>");
  const uint32_t S = 8, N = 24;
  std::vector<std::vector<int32_t>> seqs;
  {
    std::ifstream pf(std::string(argv[1]) + "/prompts.jsonl");
    std::string line;
    while (std::getline(pf, line) && seqs.size() < S) {
      auto t = ints_after(line, "tokens");
      if (t.size() >= N) seqs.push_back(std::vector<int32_t>(t.begin(), t.begin() + N));
    }
  }
  EXPECT(seqs.size() == S, "fewer than %u reference prompts of %u tokens", S, N);
  qwen4exp::Options opt;
  opt.slots = S;
  opt.max_ctx = 1024;
  opt.prompt_chunk = 0;
  if (const char *g = std::getenv("FLASH_GROUPS")) opt.groups = (uint32_t)std::stoul(g);  // the cards in groups
  qwen4exp::Model model(path, opt);
  auto row = [&](uint32_t s, uint32_t i) {
    const auto &q = seqs[s];
    return qwen4exp::StepIn{s, i, q[i], i ? q[i - 1] : qwen4exp::kNoToken, i > 1 ? q[i - 2] : qwen4exp::kNoToken};
  };
  // Alone.
  std::vector<std::vector<std::vector<qwen4exp::Candidate>>> alone(S);
  for (uint32_t s = 0; s < S; s++) {
    model.reset_slot(s);
    for (uint32_t i = 0; i < N; i++) {
      auto r = row(s, i);
      alone[s].push_back(model.step(&r, 1)[0]);
    }
  }
  int failures = 0;
  // Rows of several sequences.
  for (uint32_t width : {8u, 4u}) {
    for (uint32_t s = 0; s < width; s++) model.reset_slot(s);
    uint32_t bad = 0;
    for (uint32_t i = 0; i < N; i++) {
      std::vector<qwen4exp::StepIn> rows;
      for (uint32_t s = 0; s < width; s++) rows.push_back(row(s, i));
      auto out = model.step(rows.data(), width);
      for (uint32_t s = 0; s < width; s++) bad += !same(out[s], alone[s][i]);
    }
    std::printf("%u sequences per step: %u of %u rows differ from alone\n", width, bad, width * N);
    failures += bad != 0;
  }
  // Windows of 4 consecutive positions of one sequence.
  {
    model.reset_slot(3);
    uint32_t bad = 0;
    for (uint32_t i = 0; i < N; i += 4) {
      std::vector<qwen4exp::StepIn> rows;
      for (uint32_t k = 0; k < 4; k++) rows.push_back(row(3, i + k));
      auto out = model.step(rows.data(), 4);
      for (uint32_t k = 0; k < 4; k++) bad += !same(out[k], alone[3][i + k]);
    }
    std::printf("windows of 4 positions: %u of %u rows differ from alone\n", bad, N);
    failures += bad != 0;
  }
  // Draft windows: rows of 4 consecutive positions run as a window (state not committed), with every third draft
  // wrong; the rows up to the first wrong draft are committed, and the next window starts after them. Every committed
  // row must match its single-row run, and so must everything after.
  {
    model.reset_slot(5);
    uint32_t bad = 0, windows = 0, drafts = 0;
    for (uint32_t i = 0; i < N;) {
      uint32_t w = std::min<uint32_t>(4, N - i);
      std::vector<qwen4exp::StepIn> rows;
      uint32_t accept = w;  // rows committed: up to the first wrong draft
      for (uint32_t k = 0; k < w; k++) {
        auto r = row(5, i + k);
        if (k > 0 && (drafts++ % 3) == 2) {
          r.token = (r.token + 7) % 1000 + 100;  // a wrong draft
          if (accept == w) accept = k;
        }
        if (k > 0) r.p1 = rows[k - 1].token, r.p2 = k > 1 ? rows[k - 2].token : r.p2;
        rows.push_back(r);
      }
      auto out = model.step(rows.data(), w, true);
      for (uint32_t k = 0; k < accept; k++) bad += !same(out[k], alone[5][i + k]);
      model.commit(5, i, accept);
      i += accept;
      windows++;
    }
    std::printf("draft windows with wrong drafts: %u windows, %u committed rows differ from alone\n", windows, bad);
    failures += bad != 0;
  }
  // Draft windows of three sequences in one step: runs of 3, 2 and 3 rows, wrong drafts at different places in each,
  // each run's rows before its first wrong draft committed together.
  {
    const uint32_t seq[3] = {5, 6, 7}, len[3] = {3, 2, 3};
    for (uint32_t s : seq) model.reset_slot(s);
    uint32_t at[3] = {0, 0, 0}, bad = 0, windows = 0, drafts = 0, committed = 0;
    while (at[0] < N || at[1] < N || at[2] < N) {
      std::vector<qwen4exp::StepIn> rows;
      std::vector<CommitRun> runs;
      for (uint32_t k = 0; k < 3; k++) {
        const uint32_t s = seq[k], w = std::min(len[k], N - std::min(at[k], N));
        if (w == 0) continue;
        CommitRun run{w, at[k], s, (uint32_t)rows.size()};
        for (uint32_t j = 0; j < w; j++) {
          auto r = row(s, at[k] + j);
          if (j > 0 && (drafts++ % 4) == 1 + k % 2) {
            r.token = (r.token + 11) % 1000 + 100;
            if (run.n == w) run.n = j;
          }
          if (j > 0) r.p1 = rows.back().token, r.p2 = j > 1 ? rows[rows.size() - 2].token : r.p2;
          rows.push_back(r);
        }
        runs.push_back(run);
      }
      auto out = model.step(rows.data(), (uint32_t)rows.size(), true);
      for (const CommitRun &run : runs) {
        const uint32_t k = run.slot - seq[0];
        for (uint32_t j = 0; j < run.n; j++) bad += !same(out[run.row0 + j], alone[run.slot][run.pos0 + j]);
        at[k] += run.n;
        committed += run.n;
      }
      model.commit_runs(runs.data(), (uint32_t)runs.size());
      windows++;
    }
    std::printf("draft windows of three sequences in one step: %u windows, %u of %u committed rows differ from alone\n",
                windows, bad, committed);
    failures += bad != 0;
  }
  EXPECT(failures == 0, "rows of a step differ from their single-row runs");
  std::printf("PASS\n");
  return 0;
}
