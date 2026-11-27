// Host-only checks for split-decode host-event cleanup on mid-step failures.
#include <stdexcept>
#include <string>
#include <vector>

#include "models/qwen4exp/split_decode.h"
#include "test_util.h"

using namespace engine::qwen4exp;

namespace {

struct Harness {
  uint32_t ncards = 2;
  size_t nmiss = 3;
  std::vector<std::vector<int>> execs;
  std::vector<std::vector<int>> syncs;
  std::vector<int> miss_signals;
  int ngram_signals = 0;
  bool stuck = false;
  bool throw_exec = false;
  uint32_t throw_exec_c = 0, throw_exec_part = 0;
  bool throw_sync = false;
  uint32_t throw_sync_c = 0, throw_sync_part = 0;
  bool throw_ngram = false;
  bool throw_miss = false;
  size_t throw_miss_i = 0;
  bool fail_cleanup_signal = false;
  bool throw_cleanup_sync = false;
  bool sync_timeout_sticks = false;

  Harness(uint32_t cards = 2, size_t misses = 3)
      : ncards(cards), nmiss(misses), execs(cards), syncs(cards), miss_signals(misses, 0) {}

  void run() {
    run_split_decode(
        ncards, nmiss,
        [&](uint32_t c, uint32_t part) {
          if (throw_exec && c == throw_exec_c && part == throw_exec_part)
            throw std::runtime_error("injected execute failure");
          execs[c].push_back(static_cast<int>(part));
        },
        [&](uint32_t c, uint32_t part) {
          syncs[c].push_back(static_cast<int>(part));
          if (throw_sync && c == throw_sync_c && part == throw_sync_part) {
            if (sync_timeout_sticks) stuck = true;
            throw std::runtime_error("injected sync failure");
          }
          if (throw_cleanup_sync && stuck == false && part > 0) {
            throw std::runtime_error("cleanup sync failure");
          }
        },
        [&]() {
          if (throw_ngram) throw std::runtime_error("injected ngram failure");
          ngram_signals++;
        },
        [&]() noexcept {
          if (fail_cleanup_signal) return false;
          ngram_signals++;
          return true;
        },
        [&](size_t mi) {
          if (throw_miss && mi == throw_miss_i) throw std::runtime_error("injected miss compute failure");
        },
        [&](size_t mi) { miss_signals[mi]++; },
        [&](size_t mi) noexcept {
          if (fail_cleanup_signal) return false;
          miss_signals[mi]++;
          return true;
        },
        [&]() noexcept { return stuck; }, [&]() noexcept { stuck = true; });
  }
};

template <class Fn>
std::string catch_what(Fn &&fn) {
  try {
    fn();
  } catch (const std::exception &e) {
    return e.what();
  }
  return "";
}

}  // namespace

int main() {
  {
    Harness h(2, 3);
    h.run();
    EXPECT(!h.stuck, "normal run leaves stuck false");
    EXPECT(h.ngram_signals == 1, "ngram signaled once");
    for (int s : h.miss_signals) EXPECT(s == 1, "each miss event signaled once");
    EXPECT((h.execs[1] == std::vector<int>{0, 1, 2, 3}), "card 1 executed all parts");
    EXPECT((h.execs[0] == std::vector<int>{0, 1, 2, 3}), "card 0 executed all parts");
  }

  {
    Harness h(2, 3);
    h.throw_ngram = true;
    EXPECT(catch_what([&] { h.run(); }) == "injected ngram failure", "ngram failure propagated");
    EXPECT(!h.stuck, "symmetric cleanup keeps stuck false");
    EXPECT(h.ngram_signals == 1, "cleanup signals ngram event");
    EXPECT(h.miss_signals[0] == 1 && h.miss_signals[1] == 0 && h.miss_signals[2] == 0,
           "only part 1's miss event signaled");
    EXPECT((h.syncs[0] == std::vector<int>{1}) && (h.syncs[1] == std::vector<int>{1}),
           "both cards drain submitted part 1");
  }

  {
    Harness h(2, 3);
    h.throw_miss = true;
    h.throw_miss_i = 0;
    EXPECT(catch_what([&] { h.run(); }) == "injected miss compute failure", "miss failure propagated");
    EXPECT(!h.stuck, "miss failure cleanup keeps stuck false");
    EXPECT(h.ngram_signals == 1, "ngram signaled once");
    EXPECT(h.miss_signals[0] == 1 && h.miss_signals[1] == 1 && h.miss_signals[2] == 0,
           "cleanup signals miss events for both submitted waiting parts 1 and 2");
    EXPECT((h.syncs[0] == std::vector<int>{0, 2}) && (h.syncs[1] == std::vector<int>{2}),
           "both cards drain part 2 after miss 0 fails");
  }

  {
    Harness h(2, 3);
    h.throw_exec = true;
    h.throw_exec_c = 1;
    h.throw_exec_part = 2;
    EXPECT(catch_what([&] { h.run(); }) == "injected execute failure", "execute failure propagated");
    EXPECT(!h.stuck, "failure before part 2 starts leaves both cards symmetric at part 1");
    EXPECT(h.miss_signals[0] == 1 && h.miss_signals[1] == 0, "miss event 0 signaled for part 1");
    EXPECT((h.syncs[0] == std::vector<int>{0, 1}) && (h.syncs[1] == std::vector<int>{1}),
           "both cards drain part 1");
  }

  {
    Harness h(2, 3);
    h.throw_exec = true;
    h.throw_exec_c = 0;
    h.throw_exec_part = 2;
    EXPECT(catch_what([&] { h.run(); }) == "injected execute failure", "asymmetric execute failure propagated");
    EXPECT(h.stuck, "asymmetric submission marks queue stuck instead of draining");
    EXPECT(h.miss_signals[0] == 0 && h.miss_signals[1] == 0, "no extra signals when stuck");
    EXPECT(h.syncs[1].empty(), "card 1 not drained when submission is asymmetric");
  }

  {
    Harness h(2, 2);
    h.throw_sync = true;
    h.throw_sync_c = 0;
    h.throw_sync_part = 0;
    h.sync_timeout_sticks = true;
    EXPECT(catch_what([&] { h.run(); }) == "injected sync failure", "sync failure propagated");
    EXPECT(h.stuck, "timeout keeps stuck true");
    EXPECT(h.miss_signals[0] == 0, "no cleanup signals when already stuck");
    EXPECT(h.syncs[1].empty(), "no cleanup sync when already stuck");
  }

  {
    Harness h(2, 2);
    h.throw_ngram = true;
    h.throw_cleanup_sync = true;
    EXPECT(catch_what([&] { h.run(); }) == "injected ngram failure", "original error wins over cleanup sync error");
    EXPECT(h.stuck, "failed cleanup drain marks queue stuck");
  }
  return 0;
}
