// Host-event split-decode runner with unwind cleanup so partial submissions never hang waiting on unsignaled events.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace engine::qwen4exp {

// Runs a split decode step: submits parts 0 and 1, gathers n-gram rows and signals the n-gram event, then alternates
// draining part mi, submitting part mi + 2, computing miss layer mi on the host, and signaling miss event mi. If an
// exception unwinds after partial submission, signals pending host events and drains or marks queues stuck.
template <class ExecPart, class SyncPart, class RunNgram, class CleanupNgram, class RunMiss, class SignalMiss,
          class CleanupMiss, class IsStuck, class MarkStuck>
void run_split_decode(uint32_t ncards, size_t nmiss, ExecPart &&execute_part, SyncPart &&sync_part, RunNgram &&run_ngram,
                      CleanupNgram &&signal_ngram_cleanup, RunMiss &&run_miss, SignalMiss &&signal_miss,
                      CleanupMiss &&signal_miss_cleanup, IsStuck &&is_stuck, MarkStuck &&mark_stuck) {
  std::vector<int32_t> submitted(ncards, -1);
  bool ngram_signaled = false;
  size_t next_miss_signal = 0;
  bool finished = false;

  struct Guard {
    uint32_t ncards;
    size_t nmiss;
    std::vector<int32_t> &submitted;
    bool &ngram_signaled;
    size_t &next_miss_signal;
    bool &finished;
    SyncPart &sync_part;
    CleanupNgram &signal_ngram_cleanup;
    CleanupMiss &signal_miss_cleanup;
    IsStuck &is_stuck;
    MarkStuck &mark_stuck;

    ~Guard() {
      if (finished || is_stuck() || submitted.empty()) return;
      const auto [lo_it, hi_it] = std::minmax_element(submitted.begin(), submitted.end());
      const int32_t lo = *lo_it, hi = *hi_it;
      if (hi < 0) return;
      if (lo != hi) {
        mark_stuck();
        return;
      }
      bool ok = true;
      if (!ngram_signaled) ok = signal_ngram_cleanup();
      for (size_t mi = next_miss_signal; ok && mi < static_cast<size_t>(hi) && mi < nmiss; mi++)
        ok = signal_miss_cleanup(mi);
      if (!ok) {
        mark_stuck();
        return;
      }
      try {
        for (uint32_t c = 0; c < ncards; c++) sync_part(c, static_cast<uint32_t>(hi));
      } catch (...) {
        mark_stuck();
      }
    }
  } guard{ncards, nmiss, submitted, ngram_signaled, next_miss_signal, finished, sync_part, signal_ngram_cleanup,
          signal_miss_cleanup, is_stuck, mark_stuck};

  auto submit_all = [&](uint32_t part) {
    for (uint32_t c = ncards; c-- > 0;) {
      execute_part(c, part);
      submitted[c] = static_cast<int32_t>(part);
    }
  };

  submit_all(0);
  submit_all(1);
  run_ngram();
  ngram_signaled = true;

  for (size_t mi = 0; mi < nmiss; mi++) {
    sync_part(0, static_cast<uint32_t>(mi));
    if (mi + 1 < nmiss) submit_all(static_cast<uint32_t>(mi + 2));
    run_miss(mi);
    signal_miss(mi);
    next_miss_signal = mi + 1;
  }
  for (uint32_t c = 0; c < ncards; c++) sync_part(c, static_cast<uint32_t>(nmiss));
  finished = true;
}

}  // namespace engine::qwen4exp
