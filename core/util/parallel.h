// A fixed pool of worker threads that run a function over a range of indices, with the calling thread taking part.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace engine {

class Pool {
 public:
  explicit Pool(unsigned workers);
  ~Pool();
  Pool(const Pool &) = delete;
  Pool &operator=(const Pool &) = delete;

  // Calls f(i) for every i in [0, n), spread over the workers and the caller; returns when all are done.
  void run(uint64_t n, const std::function<void(uint64_t)> &f);
  unsigned size() const { return (unsigned)threads_.size() + 1; }
  // How many times run has been called. For the thread that calls run, between runs.
  uint64_t runs_started() const { return generation_; }

 private:
  // next and done on lines of their own: every thread takes from next for each index, and the caller polls done.
  struct Job {
    const std::function<void(uint64_t)> *fn;
    uint64_t n;
    alignas(64) std::atomic<uint64_t> next{0};
    alignas(64) std::atomic<uint64_t> done{0};
  };
  void work();
  static void drain(Job &job);

  std::vector<std::thread> threads_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::shared_ptr<Job> job_;
  uint64_t generation_ = 0;
  bool stop_ = false;
};

}  // namespace engine
