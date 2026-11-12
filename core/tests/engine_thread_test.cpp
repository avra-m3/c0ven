// Level 1 (docs/TESTING.md): the C ABI's engine thread on a model of the test's own. No card. The model notes which
// thread makes each call and whether two calls overlap, which the model's lists do not allow
// (docs/adr/0075-one-thread-uses-the-cards.md).
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
    return std::vector<std::vector<Candidate>>(n, std::vector<Candidate>{Candidate{7, 1.0f}});
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
  uint32_t prompt_chunk() const override { return 0; }  // a prompt is read a token a step
  uint32_t fetch_rows() const override { return 0; }
  uint32_t decode_weight() const override { return 0; }

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

  void note_prompt_routing() override {}
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
};

FakeModel *fake = nullptr;

std::unique_ptr<Model> open_fake(const std::string &, const Options &opt) {
  auto m = std::make_unique<FakeModel>(opt);
  fake = m.get();
  return m;
}

engine_ctx *open_test_engine(uint32_t slots, uint32_t max_ctx, uint32_t checkpoints) {
  engine_config c{};
  c.model_path = "fake";
  c.slots = slots;
  c.max_ctx = max_ctx;
  c.checkpoints = checkpoints;
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

// Waits for n plans to end; returns how many of them ended with an error.
uint32_t wait_plans(engine_ctx *e, uint32_t n) {
  uint32_t ended = 0, errors = 0;
  const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (ended < n) {
    EXPECT(std::chrono::steady_clock::now() < until, "no plan ended in 30 s");
    engine_wait(e, 100);
    engine_event ev[64];
    uint32_t got = 0;
    EXPECT(engine_poll(e, ev, 64, &got) == ENGINE_OK, "poll");
    for (uint32_t i = 0; i < got; i++) {
      ended += ev[i].kind == ENGINE_EVENT_STEP || ev[i].kind == ENGINE_EVENT_ERROR;
      errors += ev[i].kind == ENGINE_EVENT_ERROR;
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

}  // namespace

int main() {
  test_reset_before_first_plan();
  test_reset_error_is_an_event();
  test_admits_beside_plans();
  test_restore_sets_up_first();
  std::printf("engine_thread_test: ok\n");
  return 0;
}
