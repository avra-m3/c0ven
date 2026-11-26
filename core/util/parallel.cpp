#include "parallel.h"

#include <immintrin.h>
#include <pthread.h>
#include <sched.h>

#include <fstream>
#include <set>
#include <string>

namespace engine {

namespace {

// The first CPU of each physical core this process may run on, in CPU order.
std::vector<int> core_cpus() {
  cpu_set_t set;
  CPU_ZERO(&set);
  if (sched_getaffinity(0, sizeof set, &set) != 0) return {};
  std::vector<int> out;
  std::set<std::string> seen;
  for (int cpu = 0; cpu < CPU_SETSIZE; cpu++) {
    if (!CPU_ISSET(cpu, &set)) continue;
    std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/thread_siblings_list");
    std::string siblings;
    if (!std::getline(f, siblings)) return {};
    if (seen.insert(siblings).second) out.push_back(cpu);
  }
  return out;
}

}  // namespace

Pool::Pool(unsigned workers, bool pin) {
  std::vector<int> cpus = pin ? core_cpus() : std::vector<int>{};
  const bool pinned = cpus.size() > workers;
  for (unsigned i = 0; i < workers; i++) threads_.emplace_back([this, cpu = pinned ? cpus[i + 1] : -1] { work(cpu); });
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
  uint64_t ran = 0;
  for (uint64_t i; (i = job.next.fetch_add(1, std::memory_order_relaxed)) < job.n; ran++) (*job.fn)(i);
  if (ran) job.done.fetch_add(ran, std::memory_order_release);
}

void Pool::work(int cpu) {
  if (cpu >= 0) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    pthread_setaffinity_np(pthread_self(), sizeof set, &set);
  }
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
