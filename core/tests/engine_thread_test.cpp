// Level 1 (docs/TESTING.md): the C ABI's engine thread on a model of the test's own. No card. The model notes which
// thread makes each call and whether two calls overlap, which the model's lists do not allow
// (docs/adr/0077-one-thread-uses-the-cards.md). It skips on a CPU that lacks a feature of the tests' ggml build, as
// open_engine refuses one (cpu_features_test).
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "api/model.h"
#include "api/open_engine.h"
#include "cpu/features.h"
#include "engine_api.h"

#define EXPECT(cond, ...)                                       \
  do {                                                          \
    if (!(cond)) {                                              \
      std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
      std::fprintf(stderr, __VA_ARGS__);                        \
      std::fprintf(stderr, "\n");                               \
      std::exit(1);                                             \
    }                                                           \
  } while (0)

namespace {

using namespace engine;

struct Call {
  std::string what;
  uint32_t slot;
  std::thread::id thread;
};

// Every call that does work notes itself and holds the model for a while, so a call from another thread overlaps it.
class FakeModel : public Model {
 public:
  explicit FakeModel(const Options &o) : opt_(o) {
    info_.cards = 1;
    info_.vocab = 16;
  }

  std::vector<Call> calls() {
    std::lock_guard<std::mutex> lk(mu_);
    return calls_;
  }
  void clear() {
    std::lock_guard<std::mutex> lk(mu_);
    calls_.clear();
  }
  std::atomic<bool> overlap{false};
  std::atomic<bool> fail_reset{false};

  void reset_slot(uint32_t slot) override {
    In in(*this, "reset", slot, 200);
    if (fail_reset) throw std::runtime_error("reset failed");
  }
  void set_images(uint32_t slot, const Image *, uint32_t) override { In in(*this, "images", slot); }
  uint32_t image_tokens_max() const override { return 0; }
  ImageShape image_shape() const override { return {}; }

  std::vector<std::vector<Candidate>> step(const StepIn *rows, uint32_t n, bool, bool) override {
    In in(*this, "step", rows[0].slot, 300);
    {
      std::lock_guard<std::mutex> lk(mu_);
      last_rows_.assign(rows, rows + n);
      step_sizes_.push_back(n);
    }
    return std::vector<std::vector<Candidate>>(n, std::vector<Candidate>{Candidate{7, 1.0f}});
  }
  TokenScores read_logits(uint32_t row, const int32_t *tokens, uint32_t n) override {
    In in(*this, "read_logits", row, 50);
    std::lock_guard<std::mutex> lk(mu_);
    EXPECT(row < last_rows_.size(), "read_logits row %u out of range (%zu)", row, last_rows_.size());
    const StepIn r = last_rows_[row];
    TokenScores out;
    out.logits.resize(n);
    // Deterministic fake logits that depend on the last row's position and token, including low values outside top-k.
    for (uint32_t i = 0; i < n; i++) out.logits[i] = -10.0f + (float)tokens[i] + 0.25f * (float)r.pos;
    out.logsumexp = 8.5 + 0.25 * (double)r.pos;
    return out;
  }
  void commit(uint32_t, uint32_t, uint32_t) override { In in(*this, "commit", 0); }
  void commit_runs(const CommitRun *, uint32_t) override { In in(*this, "commit", 0); }

  bool drafting() const override { return false; }
  bool dflash() const override { return false; }
  uint32_t draft_max_default() const override { return 0; }
  uint32_t draft_max_cap() const override { return 0; }
  std::vector<std::vector<Candidate>> mtp_catchup(const StepIn *, uint32_t, const std::vector<uint32_t> &,
                                                  const uint32_t *) override {
    return {};
  }
  std::vector<Candidate> mtp_draft(uint32_t, uint32_t, int32_t, uint32_t) override { return {}; }
  std::vector<std::vector<Candidate>> mtp_drafts(const StepIn *, uint32_t, const uint32_t *) override { return {}; }
  std::vector<std::vector<Candidate>> mtp_catchup_prompt(const StepIn *, uint32_t, uint32_t,
                                                         const std::vector<uint32_t> &) override {
    return {};
  }
  std::vector<std::vector<std::vector<Candidate>>> df_pass(const StepIn *, uint32_t, const DraftBlock *,
                                                           uint32_t) override {
    return {};
  }
  void df_context_prompt(const StepIn *, uint32_t, uint32_t) override {}
  void draft_start(uint32_t, uint32_t) override {}

  std::vector<std::vector<Candidate>> prompt_runs(const PromptRun *, uint32_t, const int32_t *, uint32_t) override {
    throw std::logic_error("no prompt path");
  }
  uint32_t prompt_chunk() const override { return prompt_chunk_val_; }
  uint32_t fetch_rows() const override { return 0; }
  uint32_t decode_weight() const override { return 0; }
  std::vector<uint32_t> step_sizes() {
    std::lock_guard<std::mutex> lk(mu_);
    return step_sizes_;
  }
  uint32_t prompt_chunk_val_ = 0;

  uint64_t slot_state_bytes() override { return 16; }
  void save_slot(uint32_t slot, uint8_t *) override { In in(*this, "save", slot); }
  void load_slot(uint32_t slot, const uint8_t *) override { In in(*this, "load", slot); }
  void *alloc_pinned(size_t bytes) override { return new uint8_t[bytes]; }
  void free_pinned(void *p) override { delete[] static_cast<uint8_t *>(p); }

  void sleep() override { In in(*this, "sleep", 0); }
  void wake() override { In in(*this, "wake", 0); }

  uint32_t kv_pages() const override { return opt_.slots * pages_per_slot(); }
  uint32_t pages_per_slot() const override { return opt_.max_ctx / KV_PAGE_CELLS; }
  void set_pages(uint32_t slot, uint32_t, const uint32_t *, uint32_t) override { In in(*this, "pages", slot); }
  void copy_cells(uint32_t, uint32_t, uint32_t) override { In in(*this, "cells", 0); }
  void kv_need(uint32_t, uint32_t) override { In in(*this, "kv", 0); }
  void kv_ahead() override { In in(*this, "kv", 0); }
  void kvs_drop(uint32_t slot, uint32_t) override { In in(*this, "kv", slot); }

  void note_prompt_routing(bool) override {}
  uint32_t swap_round(uint32_t, bool) override { return 0; }
  bool swaps_ready() const override { return true; }
  bool swaps_settled() const override { return true; }

  const LoadInfo &info() const override { return info_; }
  const StepStats &last_stats() const override { return stats_; }
  uint32_t experts_cached() const override { return 0; }
  uint32_t experts_floor() const override { return 0; }
  NgramStats ngram_stats() const override { return {}; }

 private:
  struct In {
    FakeModel &m;
    In(FakeModel &model, const char *what, uint32_t slot, int hold_us = 50) : m(model) {
      if (m.inside_.fetch_add(1) != 0) m.overlap = true;
      {
        std::lock_guard<std::mutex> lk(m.mu_);
        m.calls_.push_back({what, slot, std::this_thread::get_id()});
      }
      std::this_thread::sleep_for(std::chrono::microseconds(hold_us));
    }
    ~In() { m.inside_.fetch_sub(1); }
  };

  Options opt_;
  LoadInfo info_;
  StepStats stats_;
  std::atomic<int> inside_{0};
  std::mutex mu_;
  std::vector<Call> calls_;
  std::vector<StepIn> last_rows_;
  std::vector<uint32_t> step_sizes_;
};

FakeModel *fake = nullptr;

std::unique_ptr<Model> open_fake(const std::string &, const Options &opt) {
  auto m = std::make_unique<FakeModel>(opt);
  fake = m.get();
  return m;
}

// Plans that ended before the wait that counts them asked for them: a poll may return more events than the wait wants.
uint32_t carried_ended = 0, carried_errors = 0;

engine_ctx *open_test_engine(uint32_t slots, uint32_t max_ctx, uint32_t checkpoints) {
  engine_config c{};
  c.model_path = "fake";
  c.slots = slots;
  c.max_ctx = max_ctx;
  c.checkpoints = checkpoints;
  carried_ended = carried_errors = 0;
  engine_ctx *e = nullptr;
  EXPECT(open_engine(&c, nullptr, &e, open_fake) == ENGINE_OK, "open: %s", engine_last_error(nullptr));
  return e;
}

void admit(engine_ctx *e, uint32_t slot, const std::vector<int32_t> &prompt, uint32_t max_tokens) {
  engine_settings s{};
  s.max_tokens = max_tokens;
  EXPECT(engine_admit(e, slot, &s, prompt.data(), (uint32_t)prompt.size()) == ENGINE_OK, "admit %u: %s", slot,
         engine_last_error(e));
  EXPECT(engine_admit_images(e, slot, nullptr, 0) == ENGINE_OK, "admit_images %u: %s", slot, engine_last_error(e));
}

// A plan that decodes `slot` and maps its logical pages [0, map) to pages first, first + 1, ...
void submit(engine_ctx *e, uint32_t slot, uint32_t map = 0, uint32_t first = 0) {
  std::vector<engine_page_change> pages;
  for (uint32_t i = 0; i < map; i++) pages.push_back({slot, i, first + i, 0, 0});
  engine_step_plan p{};
  p.nslots = 1;
  p.slots[0] = slot;
  p.npages = (uint32_t)pages.size();
  p.pages = pages.data();
  EXPECT(engine_submit(e, &p) == ENGINE_OK, "submit: %s", engine_last_error(e));
}

// Waits for n plans to end, counting those an earlier wait read past its n; returns how many of them ended with an
// error.
uint32_t wait_plans(engine_ctx *e, uint32_t n) {
  uint32_t ended = 0, errors = 0;
  const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (ended < n) {
    if (carried_ended) {
      const uint32_t take = std::min(carried_ended, n - ended);
      const uint32_t bad = std::min(carried_errors, take);
      ended += take;
      errors += bad;
      carried_ended -= take;
      carried_errors -= bad;
      continue;
    }
    EXPECT(std::chrono::steady_clock::now() < until, "no plan ended in 30 s");
    engine_wait(e, 100);
    engine_event ev[64];
    uint32_t got = 0;
    EXPECT(engine_poll(e, ev, 64, &got) == ENGINE_OK, "poll");
    for (uint32_t i = 0; i < got; i++) {
      if (ev[i].kind != ENGINE_EVENT_STEP && ev[i].kind != ENGINE_EVENT_ERROR) continue;
      carried_ended++;
      carried_errors += ev[i].kind == ENGINE_EVENT_ERROR;
    }
  }
  return errors;
}

// The index of the first call `what` on slot at or after `from`; -1 when none.
int find_call(const std::vector<Call> &calls, const char *what, uint32_t slot, size_t from = 0) {
  for (size_t i = from; i < calls.size(); i++)
    if (calls[i].what == what && calls[i].slot == slot) return (int)i;
  return -1;
}

// Admit hands the reset to the engine thread, which does it, then the images, before the slot's first step.
void test_reset_before_first_plan() {
  engine_ctx *e = open_test_engine(2, 64, 0);
  const std::thread::id caller = std::this_thread::get_id();
  admit(e, 0, {1, 2, 3}, 2);
  EXPECT(find_call(fake->calls(), "reset", 0) < 0, "admit reset the slot on the caller's thread");
  submit(e, 0, 2);
  EXPECT(wait_plans(e, 1) == 0, "the plan failed: %s", engine_last_error(e));
  const std::vector<Call> calls = fake->calls();
  const int r = find_call(calls, "reset", 0), im = find_call(calls, "images", 0), st = find_call(calls, "step", 0);
  EXPECT(r >= 0 && im > r && st > im, "reset %d, images %d, step %d: not in that order", r, im, st);
  EXPECT(calls[r].thread != caller && calls[r].thread == calls[st].thread, "the reset ran off the engine thread");
  EXPECT(calls[im].thread == calls[st].thread, "the images were set off the engine thread");
  submit(e, 0);
  EXPECT(wait_plans(e, 1) == 0, "the second plan failed");
  EXPECT(find_call(fake->calls(), "reset", 0, (size_t)r + 1) < 0, "the slot was reset twice");
  engine_close(e);
}

// A reset that fails ends the next plan with ENGINE_EVENT_ERROR, which the scheduler takes as the engine's failure.
void test_reset_error_is_an_event() {
  engine_ctx *e = open_test_engine(1, 64, 0);
  fake->fail_reset = true;
  admit(e, 0, {1, 2}, 2);
  submit(e, 0, 1);
  EXPECT(wait_plans(e, 1) == 1, "a failed reset did not end its plan with an error");
  EXPECT(std::string(engine_last_error(e)) == "reset failed", "the error says \"%s\"", engine_last_error(e));
  fake->fail_reset = false;
  EXPECT(engine_release(e, 0) == ENGINE_OK, "release");
  admit(e, 0, {1, 2}, 2);
  submit(e, 0, 1);
  EXPECT(wait_plans(e, 1) == 0, "the plan after a failed reset failed: %s", engine_last_error(e));
  engine_close(e);
}

// Admits while the engine thread runs plans for another slot: no model call overlaps another, and the caller's thread
// makes none.
void test_admits_beside_plans() {
  engine_ctx *e = open_test_engine(2, 4096, 0);
  const std::thread::id caller = std::this_thread::get_id();
  admit(e, 1, {1}, 100000);
  submit(e, 1, 4096 / KV_PAGE_CELLS);
  submit(e, 1);
  const uint32_t rounds = 200;
  for (uint32_t i = 0; i < rounds; i++) {
    admit(e, 0, {1, 2}, 2);
    submit(e, 1);
    EXPECT(wait_plans(e, 1) == 0, "a plan failed: %s", engine_last_error(e));
    EXPECT(engine_release(e, 0) == ENGINE_OK, "release");
  }
  EXPECT(wait_plans(e, 2) == 0, "a plan failed: %s", engine_last_error(e));
  EXPECT(!fake->overlap, "two model calls overlapped");
  uint32_t resets = 0;
  for (const Call &c : fake->calls()) {
    EXPECT(c.thread != caller, "the caller's thread called %s", c.what.c_str());
    resets += c.what == "reset" && c.slot == 0;
  }
  EXPECT(resets > 0 && resets <= rounds, "%u resets of slot 0 in %u admits", resets, rounds);
  engine_close(e);
}

// Restore waits for the engine thread, then does the slot's handed-over work itself before it loads the checkpoint, so
// the reset never overwrites what the checkpoint gave.
void test_restore_sets_up_first() {
  engine_ctx *e = open_test_engine(2, 64, 1);
  const std::thread::id caller = std::this_thread::get_id();
  admit(e, 1, {1, 2, 3, 4, 5}, 1);
  submit(e, 1, 2);
  submit(e, 1);
  submit(e, 1);
  EXPECT(wait_plans(e, 3) == 0, "a plan failed: %s", engine_last_error(e));
  EXPECT(engine_checkpoint(e, 1, 0) == ENGINE_OK, "checkpoint: %s", engine_last_error(e));
  admit(e, 0, {1, 2, 3, 9, 9}, 2);
  fake->clear();
  EXPECT(engine_restore(e, 0, 0) == ENGINE_OK, "restore: %s", engine_last_error(e));
  std::vector<Call> calls = fake->calls();
  const int r = find_call(calls, "reset", 0), ld = find_call(calls, "load", 0);
  EXPECT(r >= 0 && ld > r, "reset %d, load %d: not in that order", r, ld);
  EXPECT(calls[r].thread == caller, "restore left the reset to the engine thread, after the load");
  submit(e, 0, 2, 2);
  EXPECT(wait_plans(e, 1) == 0, "the plan after restore failed: %s", engine_last_error(e));
  EXPECT(find_call(fake->calls(), "reset", 0, (size_t)ld + 1) < 0, "the slot was reset after the restore");
  EXPECT(!fake->overlap, "two model calls overlapped");
  engine_close(e);
}

// Score-only admit reads exact choice logits at prompt end, emits no token, validates choice ids, and batches
// concurrent short score-only pieces into multi-slot windows.
void test_score_only_and_batched_windows() {
  engine_ctx *e = open_test_engine(2, 64, 0);
  // Validation: score-only needs at least one token, and every token id must be in vocab [0, 16).
  engine_settings bad{};
  bad.flags = ENGINE_SETTINGS_SCORE_ONLY;
  const int32_t p[] = {1, 2, 3};
  EXPECT(engine_admit(e, 0, &bad, p, 3) == ENGINE_BAD_ARGUMENT, "score-only with 0 tokens accepted");
  bad.n_score_tokens = 1;
  bad.score_tokens[0] = 16;
  EXPECT(engine_admit(e, 0, &bad, p, 3) == ENGINE_BAD_ARGUMENT, "out-of-vocab score token accepted");
  bad.score_tokens[0] = 2;
  bad.flags = 99u;
  EXPECT(engine_admit(e, 0, &bad, p, 3) == ENGINE_BAD_ARGUMENT, "unknown settings flag accepted");

  // Single score-only slot through decode() path (prompt_chunk == 0).
  engine_settings s{};
  s.flags = ENGINE_SETTINGS_SCORE_ONLY;
  s.n_score_tokens = 3;
  s.score_tokens[0] = 1;
  s.score_tokens[1] = 4;
  s.score_tokens[2] = 15;
  EXPECT(engine_admit(e, 0, &s, p, 3) == ENGINE_OK, "admit score-only");
  engine_slot_scores out{};
  EXPECT(engine_read_slot_scores(e, 0, &out) == ENGINE_BAD_ARGUMENT, "scores ready before prompt finished");
  submit(e, 0, 1);
  submit(e, 0);
  submit(e, 0);
  EXPECT(wait_plans(e, 3) == 0, "score-only decode failed: %s", engine_last_error(e));
  EXPECT(engine_read_slot_scores(e, 0, &out) == ENGINE_OK, "read_slot_scores failed");
  EXPECT(out.n == 3, "got %u scores, want 3", out.n);
  EXPECT(out.logits[0] == -8.5f && out.logits[1] == -5.5f && out.logits[2] == 5.5f, "unexpected logits");
  EXPECT(out.logsumexp == 9.0, "unexpected logsumexp %f", out.logsumexp);
  EXPECT(engine_release(e, 0) == ENGINE_OK, "release");

  // Two short score-only pieces in one plan batch together in prompt_score_windows (up to STEP_MAX_ROWS = 16).
  fake->prompt_chunk_val_ = 64;
  // Re-open with prompt_chunk enabled via FakeModel.
  engine_close(e);
  auto open_with_chunk = [](const std::string &, const Options &opt) -> std::unique_ptr<Model> {
    auto m = std::make_unique<FakeModel>(opt);
    m->prompt_chunk_val_ = 64;
    fake = m.get();
    return m;
  };
  engine_config cfg{};
  cfg.model_path = "fake";
  cfg.slots = 2;
  cfg.max_ctx = 64;
  carried_ended = carried_errors = 0;
  EXPECT(open_engine(&cfg, nullptr, &e, open_with_chunk) == ENGINE_OK, "open chunked");
  const std::vector<int32_t> p0 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const std::vector<int32_t> p1 = {1, 2, 3, 4, 5, 6, 7, 8, 11, 12};
  EXPECT(engine_admit(e, 0, &s, p0.data(), (uint32_t)p0.size()) == ENGINE_OK, "admit 0");
  EXPECT(engine_admit(e, 1, &s, p1.data(), (uint32_t)p1.size()) == ENGINE_OK, "admit 1");
  std::vector<engine_page_change> pages;
  for (uint32_t i = 0; i < 3; i++) pages.push_back({0, i, i, 0, 0});
  for (uint32_t i = 0; i < 3; i++) pages.push_back({1, i, 3 + i, 0, 0});
  engine_step_plan sp{};
  sp.npieces = 2;
  sp.pieces[0] = {0, 10};
  sp.pieces[1] = {1, 10};
  sp.npages = (uint32_t)pages.size();
  sp.pages = pages.data();
  sp.window_rows = 64;
  EXPECT(engine_submit(e, &sp) == ENGINE_OK, "submit batched score windows");
  EXPECT(wait_plans(e, 1) == 0, "batched score windows failed: %s", engine_last_error(e));
  const auto sizes = fake->step_sizes();
  EXPECT(sizes.size() == 2 && sizes[0] == 16 && sizes[1] == 4, "expected 16 then 4 batched rows, got %zu steps",
         sizes.size());
  engine_slot_scores out0{}, out1{};
  EXPECT(engine_read_slot_scores(e, 0, &out0) == ENGINE_OK && engine_read_slot_scores(e, 1, &out1) == ENGINE_OK,
         "read batched scores");
  EXPECT(out0.n == 3 && out1.n == 3, "batched score count");
  EXPECT(out0.logits[0] == -10.0f + 1.0f + 0.25f * 9.0f, "slot 0 logit at pos 9");
  EXPECT(out1.logits[2] == -10.0f + 15.0f + 0.25f * 9.0f, "slot 1 logit at pos 9");
  engine_close(e);
}

}  // namespace

int main() {
  if (const std::string missing = engine::missing_cpu_features(ENGINE_CPU_FEATURES); !missing.empty()) {
    std::printf("SKIP: this CPU lacks %s, which the tests' ggml build uses\n", missing.c_str());
    return 77;
  }
  test_reset_before_first_plan();
  test_reset_error_is_an_event();
  test_admits_beside_plans();
  test_restore_sets_up_first();
  test_score_only_and_batched_windows();
  std::printf("engine_thread_test: ok\n");
  return 0;
}
