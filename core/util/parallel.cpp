#include "parallel.h"

#include <immintrin.h>

namespace engine {

Pool::Pool(unsigned workers) {
  for (unsigned i = 0; i < workers; i++) threads_.emplace_back([this] { work(); });
}

Pool::~Pool() {
  {
    std::lock_guard<std::mutex> lk(mu_);
    stop_ = true;
  }
  cv_.notify_all();
  for (auto &t : threads_) t.join();
}

void Pool::drain(Job &job) {
  for (uint64_t i; (i = job.next.fetch_add(1)) < job.n;) {
    (*job.fn)(i);
    job.done.fetch_add(1, std::memory_order_release);
  }
}

void Pool::work() {
  uint64_t seen = 0;
  for (;;) {
    std::shared_ptr<Job> job;
    {
      std::unique_lock<std::mutex> lk(mu_);
      cv_.wait(lk, [&] { return stop_ || generation_ != seen; });
      if (stop_) return;
      seen = generation_;
      job = job_;
    }
    drain(*job);
  }
}

void Pool::run(uint64_t n, const std::function<void(uint64_t)> &f) {
  auto job = std::make_shared<Job>();
  job->fn = &f;
  job->n = n;
  {
    std::lock_guard<std::mutex> lk(mu_);
    job_ = job;
    generation_++;
  }
  cv_.notify_all();
  drain(*job);
  // Spin until every index has finished: the jobs are short and their workers are already running.
  while (job->done.load(std::memory_order_acquire) < n) _mm_pause();
}

}  // namespace engine
