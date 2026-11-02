// What deepseek41's kernel tests share: buffers on a card, an error check, and timing after warm-up with the Sysman
// clock of every card (main calls init_clocks before the first zeInit).
#pragma once

#include <level_zero/zes_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "test_util.h"
#include "ze/library.h"

namespace ds41test {

using namespace engine;

// Sysman's handles for each card, in device order; call before the first zeInit, since reading the clock through a
// core device handle gives nothing here.
inline std::vector<zes_freq_handle_t> &clock_domains() {
  static std::vector<zes_freq_handle_t> v;
  return v;
}
inline void init_clocks() {
  if (zesInit(0) != ZE_RESULT_SUCCESS) return;
  uint32_t nd = 0;
  if (zesDriverGet(&nd, nullptr) != ZE_RESULT_SUCCESS || !nd) return;
  std::vector<zes_driver_handle_t> drivers(nd);
  zesDriverGet(&nd, drivers.data());
  for (auto drv : drivers) {
    uint32_t n = 0;
    if (zesDeviceGet(drv, &n, nullptr) != ZE_RESULT_SUCCESS) continue;
    std::vector<zes_device_handle_t> devs(n);
    zesDeviceGet(drv, &n, devs.data());
    for (auto d : devs) {
      uint32_t nf = 0;
      if (zesDeviceEnumFrequencyDomains(d, &nf, nullptr) != ZE_RESULT_SUCCESS || !nf) continue;
      std::vector<zes_freq_handle_t> h(nf);
      zesDeviceEnumFrequencyDomains(d, &nf, h.data());
      for (auto f : h) {
        zes_freq_properties_t p = {ZES_STRUCTURE_TYPE_FREQ_PROPERTIES};
        if (zesFrequencyGetProperties(f, &p) == ZE_RESULT_SUCCESS && p.type == ZES_FREQ_DOMAIN_GPU)
          clock_domains().push_back(f);
      }
    }
  }
}

// Each card's GPU clock, as Sysman reads it now.
inline void log_clocks(const char *when) {
  std::printf("clock %s:", when);
  if (clock_domains().empty()) std::printf(" Sysman gave no frequency domain");
  for (size_t c = 0; c < clock_domains().size(); c++) {
    zes_freq_state_t s = {ZES_STRUCTURE_TYPE_FREQ_STATE};
    const bool ok = zesFrequencyGetState(clock_domains()[c], &s) == ZE_RESULT_SUCCESS;
    std::printf(" card %zu %.0f MHz", c, ok ? s.actual : -1.0);
  }
  std::printf("\n");
}

// A list's launches timed on its card: at least 50 ms of the same work first, then the median of many runs of the
// recorded list (regular, a barrier after each launch), in microseconds a launch. rec appends one launch.
inline double time_launch(ze::Runtime &rt, uint32_t card, const std::function<void(ze::List &)> &rec) {
  constexpr uint32_t N = 100, RUNS = 41;
  auto *ts = static_cast<uint64_t *>(rt.alloc_host(2 * sizeof(uint64_t)));
  double us;
  {
    ze::List L(rt, card, ze::List::regular, false, false);
    L.timestamp(ts);
    for (uint32_t i = 0; i < N; i++) rec(L);
    L.timestamp(ts + 1);
    L.close();
    const auto t0 = std::chrono::steady_clock::now();
    do {
      L.execute();
      L.synchronize();
    } while (std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(50));
    std::vector<double> v;
    const ze::Card &c = rt.cards()[card];
    for (uint32_t i = 0; i < RUNS; i++) {
      L.execute();
      L.synchronize();
      v.push_back((double)((ts[1] - ts[0]) & c.timestamp_mask) / (double)c.timer_hz * 1e6 / N);
    }
    std::sort(v.begin(), v.end());
    us = v[RUNS / 2];
  }
  rt.free(ts);
  return us;
}

// ds_wait's arguments (kernels/models/deepseek41/layer.cl); want and stuck lie on the card.
struct Wait {
  uint32_t *raise = nullptr;
  const uint32_t *flush = nullptr, *flag = nullptr, *want = nullptr, *nmiss = nullptr;
  uint32_t W = 0;
  uint32_t *stuck = nullptr;
  uint64_t *start = nullptr;
  uint64_t *stamp = nullptr;
};
inline void launch_wait(ze::List &L, ze::Library &lib, const Wait &w) {
  L.launch(lib.get(L.card(), "deepseek41", "ds_wait"),
           {w.raise, w.flush, w.flag, w.want, w.nmiss, w.W, w.stuck, w.start, w.stamp}, 1, 1, 1, 1);
}

// A figure from /proc: pswpout from vmstat, MemAvailable in GiB from meminfo. Timed runs fail when pages were swapped
// out, and load threads stop short of a timed run's floor.
inline uint64_t proc_value(const char *path, const char *key) {
  uint64_t v = 0;
  if (FILE *f = std::fopen(path, "r")) {
    char line[256];
    const size_t n = std::strlen(key);
    while (std::fgets(line, sizeof line, f))
      if (!std::strncmp(line, key, n) && (line[n] == ':' || line[n] == ' ')) v = std::strtoull(line + n + 1, nullptr, 10);
    std::fclose(f);
  }
  return v;
}
inline uint64_t pswpout() { return proc_value("/proc/vmstat", "pswpout"); }
inline double mem_available_gib() { return (double)proc_value("/proc/meminfo", "MemAvailable") / 1048576.0; }

// Card resets since construction, by /sys/class/devcoredump's devcd entries. The kernel removes an entry minutes after it
// appears and does not reuse a name, so names are compared, and a run calls fresh() between its long parts to catch each.
class CoreDumps {
 public:
  CoreDumps() : before_(names()) {}
  size_t before() const { return before_.size(); }
  size_t fresh() {
    for (const std::string &n : names())
      if (!before_.count(n)) fresh_.insert(n);
    return fresh_.size();
  }

 private:
  static std::set<std::string> names() {
    std::set<std::string> v;
    std::error_code ec;
    for (auto it = std::filesystem::directory_iterator("/sys/class/devcoredump", ec); !ec && it != std::filesystem::end(it);
         it.increment(ec))
      if (const std::string n = it->path().filename().string(); n.rfind("devcd", 0) == 0) v.insert(n);
    return v;
  }
  std::set<std::string> before_, fresh_;
};

// A card's buffers: device memory, written and read back through one immediate list.
struct Io {
  ze::Runtime &rt;
  uint32_t card;
  ze::List L;
  Io(ze::Runtime &r, uint32_t c) : rt(r), card(c), L(r, c, ze::List::immediate) {}
  template <class T>
  T *dev(size_t n) {
    return static_cast<T *>(rt.alloc_device(card, n * sizeof(T), 4096));
  }
  template <class T>
  void put(T *dst, const T *src, size_t n) {
    L.copy(dst, src, n * sizeof(T));
    L.synchronize();
  }
  template <class T>
  void put(T *dst, const std::vector<T> &src) {
    put(dst, src.data(), src.size());
  }
  template <class T>
  std::vector<T> get(const T *src, size_t n) {
    std::vector<T> v(n);
    L.copy(v.data(), src, n * sizeof(T));
    L.synchronize();
    return v;
  }
};

// The largest error of each named check, and the failures.
struct Checker {
  int failures = 0;
  struct Worst {
    const char *name;
    double err, tol;
  };
  std::vector<Worst> worst;
  void note(const char *name, double err, double tol) {
    failures += !(err <= tol);
    for (Worst &w : worst)
      if (!std::strcmp(w.name, name)) {
        if (!(err <= w.err)) w.err = err;
        return;
      }
    worst.push_back({name, err, tol});
  }
  // Largest |got - ref| over the largest |ref|.
  template <class G, class R>
  void rel(const char *name, const G *got, const R *ref, size_t n, double tol) {
    double e = 0, m = 0;
    for (size_t i = 0; i < n; i++) {
      const double d = std::fabs((double)got[i] - (double)ref[i]);
      e = std::isnan(d) ? INFINITY : std::max(e, d);
      m = std::max(m, std::fabs((double)ref[i]));
    }
    note(name, m > 0 ? e / m : e, tol);
  }
  template <class G, class R>
  void abs(const char *name, const G *got, const R *ref, size_t n, double tol) {
    double e = 0;
    for (size_t i = 0; i < n; i++) {
      const double d = std::fabs((double)got[i] - (double)ref[i]);
      e = std::isnan(d) ? INFINITY : std::max(e, d);
    }
    note(name, e, tol);
  }
  // The count of bytes that differ; the tolerance is 0.
  void exact(const char *name, const void *got, const void *ref, size_t bytes) {
    size_t diff = 0;
    for (size_t i = 0; i < bytes; i++) diff += ((const uint8_t *)got)[i] != ((const uint8_t *)ref)[i];
    note(name, (double)diff, 0.0);
  }
  int report() const {
    for (const Worst &w : worst) std::printf("%-44s largest error %.2e (limit %.0e) %s\n", w.name, w.err, w.tol, w.err <= w.tol ? "ok" : "FAIL");
    std::printf("%d failures\n", failures);
    return failures ? 1 : 0;
  }
};

inline std::vector<float> randn(std::mt19937 &rng, size_t n, float scale = 1.0f) {
  std::normal_distribution<float> nd(0.0f, scale);
  std::vector<float> v(n);
  for (float &x : v) x = nd(rng);
  return v;
}

// a8 (kernels/include/a8.h) of a row of D41_EMBD values on the host.
inline std::vector<uint8_t> a8_row(const float *x, size_t K) {
  std::vector<uint8_t> a(K + K / 8);
  for (size_t u = 0; u < K / 32; u++) {
    float amax = 0;
    for (int i = 0; i < 32; i++) amax = std::fmax(amax, std::fabs(x[32 * u + i]));
    const float d = amax / 127.0f, id = d != 0 ? 1.0f / d : 0.0f;
    for (int i = 0; i < 32; i++) a[32 * u + i] = (uint8_t)(int8_t)std::round(x[32 * u + i] * id);
    std::memcpy(a.data() + K + 4 * u, &d, 4);
  }
  return a;
}

// The sum of squares in double, as ref_ops takes it.
inline double sum_squares(const float *x, size_t n) {
  double s = 0;
  for (size_t i = 0; i < n; i++) s += (double)x[i] * x[i];
  return s;
}

inline const uint32_t WIDTHS[] = {1, 2, 3, 4, 6, 8};
inline const uint32_t TIMED_WIDTHS[] = {1, 2, 4, 8};

}  // namespace ds41test
