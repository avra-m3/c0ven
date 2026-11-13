#include "parallel.h"

#include <immintrin.h>
#include <pthread.h>
#include <sched.h>

#include <fstream>
#include <string>

namespace engine {

namespace {

// The cores this process may run on, each as its hardware threads (thread_siblings_list), in CPU order.
std::vector<cpu_set_t> cores_allowed() {
  cpu_set_t allowed;
  CPU_ZERO(&allowed);
  std::vector<cpu_set_t> cores;
  if (sched_getaffinity(0, sizeof allowed, &allowed) != 0) return cores;
  std::vector<bool> seen(CPU_SETSIZE, false);
  for (int c = 0; c < CPU_SETSIZE; c++) {
    if (!CPU_ISSET(c, &allowed) || seen[c]) continue;
    cpu_set_t core;
    CPU_ZERO(&core);
    std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/thread_siblings_list");
    std::string list;
    if (f >> list) {
      // "0,16" or "0-1" or both forms, comma-separated.
      for (size_t i = 0; i < list.size();) {
        size_t j = list.find(',', i);
        if (j == std::string::npos) j = list.size();
        const std::string part = list.substr(i, j - i);
        const size_t dash = part.find('-');
        const int lo = std::stoi(part), hi = dash == std::string::npos ? lo : std::stoi(part.substr(dash + 1));
        for (int k = lo; k <= hi && k < CPU_SETSIZE; k++)
          if (CPU_ISSET(k, &allowed)) CPU_SET(k, &core);
        i = j + 1;
      }
    }
    if (CPU_COUNT(&core) == 0) CPU_SET(c, &core);
    for (int k = 0; k < CPU_SETSIZE; k++)
      if (CPU_ISSET(k, &core)) seen[k] = true;
    cores.push_back(core);
  }
  return cores;
}

}  // namespace

Pool::Pool(unsigned workers, bool own_cores) {
  const std::vector<cpu_set_t> cores = own_cores ? cores_allowed() : std::vector<cpu_set_t>{};
  for (unsigned i = 0; i < workers; i++) {
    threads_.emplace_back([this] { work(); });
    // Worker i takes core i + 1: the first core is left to the caller and the process's other threads.
    if (!cores.empty())
      pthread_setaffinity_np(threads_.back().native_handle(), sizeof(cpu_set_t), &cores[(i + 1) % cores.size()]);
  }
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
