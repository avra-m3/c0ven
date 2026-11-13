// Level 3 (docs/TESTING.md): deepseek41's exchange, ds_exch_send then ds_wait on both cards, through host memory and the
// other card's. X' equal on both cards bit for bit in every exchange; the stores' rate; times alone and under
// compute_many on random IQ3_S weights and 16 threads reading Flash-Next's IQ3_S file with O_DIRECT; a soak under that
// load. Host memory stays above a timed run's floor. Needs two cards; without the file or the memory it runs the
// unloaded part and skips. Its times are docs/measurements/MEASUREMENTS.md#2026-11-02-deepseek41-exchange.
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <map>
#include <memory>
#include <thread>
#include <tuple>

#include "cpu/experts.h"
#include "ds41_kernel_test.h"
#include "ggml.h"
#include "models/deepseek41/layout.h"
#include "util/parallel.h"

using namespace engine;
using namespace ds41test;

namespace {

using Clock = std::chrono::steady_clock;
constexpr size_t E = D41_EMBD, D = D41_HC_DIM;
constexpr uint32_t WMAX = STEP_MAX_ROWS;
constexpr uint32_t KS = 16;       // patterns a card sends, each exchange in a list another
constexpr uint32_t NX = 82;       // exchanges a timed list or a soak's: a token's
constexpr uint32_t RUNS = 100;    // runs a timed case
constexpr uint32_t LINE = 16;     // a line of host memory in words
constexpr double SOAK_SECONDS = 30;  // a soak case
constexpr double FLOOR_GIB = 24;  // the load stops here, above a timed run's memory floor
const void *const none = nullptr;

enum Path { host, peer, peer_read };
enum Mode { send_only, sublayer, soak_run };  // sublayer: each exchange read by the sublayer's end; soak_run: and compared
const char *const PATH_NAME[] = {"host memory", "the other card's memory", "the other card's memory, read back"};

// The CPU's misses and the SSD's reads, as many as they can.
class Load {
 public:
  static constexpr uint32_t ROWS = 16, EXPERTS = 256, TOK = 4;
  // Returns false, having started nothing, when there is no file or the arena would pass the floor; why_not() says which.
  bool start(const std::string &path) {
    if (path.empty()) {
      why_not_ = "the IQ3_S file is not in FLASH_MODEL_DIR; no loaded cases";
      return false;
    }
    const size_t grow = ggml_row_size(GGML_TYPE_IQ3_S, E), drow = ggml_row_size(GGML_TYPE_IQ3_S, D41_FF);
    G_ = grow * D41_FF;
    D_ = drow * E;
    S_ = 2 * G_ + D_;
    const size_t bytes = EXPERTS * S_;
    const double avail = mem_available_gib();
    if (avail - (double)bytes / (1ull << 30) < FLOOR_GIB + 2) {
      char why[128];
      std::snprintf(why, sizeof why, "MemAvailable %.1f GiB leaves too little for a %.1f GiB arena; no loaded cases", avail,
                    (double)bytes / (1ull << 30));
      why_not_ = why;
      return false;
    }
    arena_ = static_cast<uint8_t *>(std::aligned_alloc(1 << 21, (bytes + (1 << 21) - 1) >> 21 << 21));
    EXPECT(arena_, "no arena");
    madvise(arena_, bytes, MADV_HUGEPAGE);
    // Valid blocks, quantised once for each matrix's rows and repeated: what the CPU reads is the point, not its value.
    std::mt19937 rng(7);
    ggml_quantize_init(GGML_TYPE_IQ3_S);
    const auto fg = randn(rng, ROWS * E, 0.5f), fd = randn(rng, ROWS * D41_FF, 0.05f);
    std::vector<uint8_t> qg(ROWS * grow), qd(ROWS * drow);
    ggml_quantize_chunk(GGML_TYPE_IQ3_S, fg.data(), qg.data(), 0, ROWS, E, nullptr);
    ggml_quantize_chunk(GGML_TYPE_IQ3_S, fd.data(), qd.data(), 0, ROWS, D41_FF, nullptr);
    pool_ = std::make_unique<Pool>(15);
    pool_->run(EXPERTS, [&](uint64_t e) {
      uint8_t *p = arena_ + e * S_;
      for (size_t i = 0; i < 2 * G_; i += qg.size()) std::memcpy(p + i, qg.data(), std::min(qg.size(), 2 * G_ - i));
      for (size_t i = 0; i < D_; i += qd.size()) std::memcpy(p + 2 * G_ + i, qd.data(), std::min(qd.size(), D_ - i));
    });
    threads_.emplace_back([this] { cpu_loop(); });
    for (uint32_t i = 0; i < 16; i++) threads_.emplace_back([this, i, path] { read_loop(path, i); });
    threads_.emplace_back([this] {
      while (!stop_) {
        if (mem_available_gib() < FLOOR_GIB) {
          floor_hit_ = true;
          stop_ = true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
    });
    std::printf("load: compute_many on %u threads over %u experts of IQ3_S, %.2f GiB, random weights; 16 threads read %s "
                "with O_DIRECT, 1 MiB a read\n",
                pool_->size(), EXPERTS, (double)bytes / (1ull << 30), path.c_str());
    return true;
  }
  void stop() {
    stop_ = true;
    for (auto &t : threads_) t.join();
    threads_.clear();
    pool_.reset();
    std::free(arena_);
    arena_ = nullptr;
  }
  bool floor_hit() const { return floor_hit_; }
  const std::string &why_not() const { return why_not_; }
  // The rates since the last call.
  void print_rates(const char *when) {
    const auto now = Clock::now();
    const double s = std::chrono::duration<double>(now - t_).count();
    const uint64_t c = cpu_.load(), d = ssd_.load();
    std::printf("load %s: compute_many %.1f GB/s, SSD %.2f GB/s, MemAvailable %.1f GiB\n", when, (double)(c - c0_) / s / 1e9,
                (double)(d - d0_) / s / 1e9, mem_available_gib());
    c0_ = c, d0_ = d, t_ = now;
  }

 private:
  void cpu_loop() {
    CpuExperts cpu(*pool_);
    std::mt19937 rng(9);
    const ExpertShape shape{GGML_TYPE_IQ3_S, GGML_TYPE_IQ3_S, (uint32_t)E, D41_FF, D41_SWIGLU_LIMIT};
    const auto x = randn(rng, TOK * E);
    std::vector<float> y(TOK * E);
    std::vector<CpuExperts::Job> jobs;
    while (!stop_) {
      jobs.clear();
      for (uint32_t t = 0; t < TOK; t++)
        for (uint32_t k = 0; k < D41_USED; k++) {
          const uint8_t *g = arena_ + (rng() % EXPERTS) * S_;
          jobs.push_back({ExpertWeights{g, g + G_, g + 2 * G_, 1}, t, 0.25f});
        }
      cpu.compute_many(shape, jobs, TOK, x.data(), E, y.data(), E);
      cpu_ += jobs.size() * S_;
    }
  }
  void read_loop(const std::string &path, uint32_t i) {
    const int fd = open(path.c_str(), O_RDONLY | O_DIRECT);
    if (fd < 0) {
      std::printf("FAIL: cannot open %s with O_DIRECT\n", path.c_str());
      read_failed_ = true;
      return;
    }
    constexpr size_t MIB = 1 << 20;
    const uint64_t blocks = (uint64_t)lseek(fd, 0, SEEK_END) / MIB;
    void *buf = std::aligned_alloc(4096, MIB);
    std::mt19937_64 rng(100 + i);
    while (!stop_) {
      const ssize_t n = pread(fd, buf, MIB, (off_t)((rng() % blocks) * MIB));
      if (n <= 0) {
        read_failed_ = true;
        break;
      }
      ssd_ += (uint64_t)n;
    }
    std::free(buf);
    close(fd);
  }
  std::string why_not_;
  std::unique_ptr<Pool> pool_;
  std::vector<std::thread> threads_;
  uint8_t *arena_ = nullptr;
  size_t G_ = 0, D_ = 0, S_ = 0;
  std::atomic<bool> stop_{false}, floor_hit_{false};
  std::atomic<uint64_t> cpu_{0}, ssd_{0};
  uint64_t c0_ = 0, d0_ = 0;
  Clock::time_point t_ = Clock::now();

 public:
  std::atomic<bool> read_failed_{false};
};

struct Plan {
  uint32_t step, stuck;
  uint64_t start;
  uint32_t bad, pad;
};

struct CardMem {
  Io io;
  float *X, *y, *in[2], *Xo, *ss, *gold;
  float *raw, *scale, *base;  // the raw mixes, the scales and bases the kernel makes the mixes from
  D41HcMix *mix;
  Plan *plan;
  uint64_t *stamp;
  CardMem(ze::Runtime &rt, uint32_t c) : io(rt, c) {
    X = io.dev<float>(WMAX * D);
    y = io.dev<float>(KS * WMAX * E);
    for (float *&p : in) p = io.dev<float>(WMAX * E);
    Xo = io.dev<float>(KS * WMAX * D);
    ss = io.dev<float>(KS * WMAX);
    gold = io.dev<float>(KS * WMAX * D);
    raw = io.dev<float>(WMAX * D41_MIX);
    scale = io.dev<float>(3);
    base = io.dev<float>(D41_MIX);
    mix = io.dev<D41HcMix>(WMAX);
    plan = io.dev<Plan>(1);
    stamp = io.dev<uint64_t>(2 * (NX + 2));
  }
};

// What a run leaves in host memory for one card.
struct HostSide {
  Plan plan;
  uint32_t stuck, bad;
  uint64_t ts[2];
  uint64_t stamp[2 * (NX + 2)];
};

struct Stat {
  double p50, p99, mean, max;
};
Stat stats(std::vector<double> &v) {
  std::sort(v.begin(), v.end());
  double sum = 0;
  for (double x : v) sum += x;
  return {v[v.size() / 2], v[v.size() * 99 / 100], sum / (double)v.size(), v.back()};
}

struct Bench {
  ze::Runtime &rt;
  ze::Library &lib;
  std::mt19937 rng{1};
  Checker ck;
  CardMem *cards[2];
  HostSide *side[2];
  float *host_in[2][2];  // what card i receives through host memory, by exchange as in CardMem::in
  uint32_t *flags;    // [2][NX], a line each
  uint32_t step = 0;
  std::vector<float> ys[2];         // the patterns, [KS][WMAX][E]
  std::vector<float> gold[2];       // [KS][WMAX][D]: X' from each pattern, the partial sums put in place by the CPU
  uint64_t wrong = 0;
  CoreDumps dumps;
  const std::vector<float> canary = std::vector<float>(WMAX * E, -123.5f);
  using Key = std::tuple<uint32_t, int, int>;
  std::map<Key, std::unique_ptr<ze::List>> lists[2];

  Bench(ze::Runtime &r, ze::Library &l) : rt(r), lib(l) {
    for (uint32_t c = 0; c < 2; c++) {
      cards[c] = new CardMem(rt, c);
      side[c] = static_cast<HostSide *>(rt.alloc_host(sizeof(HostSide), 4096));
      for (float *&p : host_in[c]) p = static_cast<float *>(rt.alloc_host(WMAX * E * sizeof(float), 4096));
    }
    flags = static_cast<uint32_t *>(rt.alloc_host(2 * NX * LINE * sizeof(uint32_t), 4096));
    std::memset(flags, 0, 2 * NX * LINE * sizeof(uint32_t));
  }
  uint32_t *flag(uint32_t c, uint32_t k) { return flags + (c * NX + k) * LINE; }
  float *y_at(uint32_t c, uint32_t s) { return cards[c]->y + (size_t)s * WMAX * E; }
  float *Xo_at(uint32_t c, uint32_t s) { return cards[c]->Xo + (size_t)s * WMAX * D; }
  float *ss_at(uint32_t c, uint32_t s) { return cards[c]->ss + (size_t)s * WMAX; }
  // Even and odd exchanges use different buffers: after raising its flag a card may read one while the other card's
  // next send is on its way, and that send's own wait holds the one after it back until the reading is done.
  float *received(uint32_t c, Path p, uint32_t k) { return p == host ? host_in[c][k % 2] : cards[c]->in[k % 2]; }

  void launch_post_norm(ze::List &L, uint32_t c, uint32_t W, uint32_t s, const float *y1, uint32_t y_host) {
    CardMem &m = *cards[c];
    L.launch(lib.get(c, "deepseek41", "ds_hc_post_norm"),
             {m.X, y_at(c, s), y1, y_host, m.raw, m.scale, m.base, m.mix, none, Xo_at(c, s), ss_at(c, s), none, none, none,
              none},
             W, 1, 1, D41_HC_SGS * 16);
  }
  void launch_send(ze::List &L, uint32_t c, uint32_t W, const float *y, const float *dst, uint32_t sgs = D41_EX_SGS) {
    L.launch(lib.get(c, "deepseek41", "ds_exch_send"), {y, dst}, W, 1, 1, sgs * 16);
  }
  void launch_check(ze::List &L, uint32_t c, uint32_t W, uint32_t s, const float *got) {
    L.launch(lib.get(c, "deepseek41_check", "ds_check_same"), {got, cards[c]->gold + (size_t)s * WMAX * D, &cards[c]->plan->bad},
             W, 1, 1, D41_HC_SGS * 16);
  }

  // The inputs, and what each pattern must give on each card.
  void prepare() {
    const auto X = randn(rng, WMAX * D);
    const auto raw = randn(rng, WMAX * D41_MIX, 0.5f);
    for (uint32_t c = 0; c < 2; c++) {
      cards[c]->io.put(cards[c]->X, X);
      cards[c]->io.put(cards[c]->raw, raw);
      cards[c]->io.put(cards[c]->scale, std::vector<float>{1, 1, 1});
      cards[c]->io.put(cards[c]->base, std::vector<float>(D41_MIX, 0.0f));
      ys[c] = randn(rng, KS * WMAX * E);
      cards[c]->io.put(cards[c]->y, ys[c]);
    }
    for (uint32_t c = 0; c < 2; c++) {
      CardMem &m = *cards[c];
      for (uint32_t s = 0; s < KS; s++) {
        m.io.put(m.in[0], &ys[1 - c][(size_t)s * WMAX * E], WMAX * E);
        launch_post_norm(m.io.L, c, WMAX, s, m.in[0], 0);
        m.io.L.synchronize();
      }
      gold[c] = m.io.get(m.Xo, KS * WMAX * D);
    }
    ck.exact("X' on both cards, sums in place", gold[0].data(), gold[1].data(), gold[0].size() * 4);
    for (uint32_t c = 0; c < 2; c++) {
      cards[c]->io.put(cards[c]->gold, gold[c]);
      check_the_check(c);
    }
  }

  // The soak's comparison counts one wrong bit in a word, and only in the rows it is given.
  void check_the_check(uint32_t c) {
    CardMem &m = *cards[c];
    std::vector<float> o(gold[c].begin(), gold[c].begin() + WMAX * D);
    for (size_t i : {(size_t)0, D - 1, (WMAX - 1) * D + 5}) {
      uint32_t u;
      std::memcpy(&u, &o[i], 4);
      u ^= 1;
      std::memcpy(&o[i], &u, 4);
    }
    m.io.put(Xo_at(c, 0), o);
    for (uint32_t W : {1u, WMAX}) {
      const uint32_t z = 0;
      m.io.put(&m.plan->bad, &z, 1);
      launch_check(m.io.L, c, W, 0, Xo_at(c, 0));
      m.io.L.synchronize();
      ck.note("the card's comparison counts each word", std::fabs((double)m.io.get(&m.plan->bad, 1)[0] - (W == 1 ? 2 : 3)), 0);
    }
  }

  // Each card's list: the plan's copy, then n exchanges. Exchange k sends pattern k % KS. Stamps of ds_wait with no flag
  // on either side turn ticks into time for the run.
  ze::List &list(uint32_t c, uint32_t W, Path p, Mode mode) {
    auto &slot = lists[c][Key(W, p, mode)];
    if (slot) return *slot;
    slot = std::make_unique<ze::List>(rt, c, ze::List::regular, false, false);
    ze::List &L = *slot;
    CardMem &m = *cards[c];
    const uint32_t other = 1 - c, n = mode == sublayer ? KS : NX;
    L.copy(m.plan, &side[c]->plan, sizeof(Plan));
    L.timestamp(&side[c]->ts[0]);
    Wait cal;
    cal.want = &m.plan->step, cal.stuck = &m.plan->stuck, cal.start = &m.plan->start, cal.stamp = m.stamp;
    launch_wait(L, lib, cal);
    for (uint32_t k = 0; k < n; k++) {
      float *dst = received(other, p, k);
      launch_send(L, c, W, y_at(c, k % KS), dst);
      Wait w;
      w.raise = flag(c, k), w.flag = flag(other, k), w.want = &m.plan->step, w.stuck = &m.plan->stuck;
      w.start = &m.plan->start;
      if (p == peer_read) w.flush = reinterpret_cast<const uint32_t *>(dst + (size_t)W * E - 1);
      w.stamp = m.stamp + 2 * (k + 1);
      launch_wait(L, lib, w);
      if (mode != send_only) launch_post_norm(L, c, W, k % KS, received(c, p, k), p == host);
      if (mode == soak_run) launch_check(L, c, W, k % KS, Xo_at(c, k % KS));
    }
    cal.stamp = m.stamp + 2 * (n + 1);
    launch_wait(L, lib, cal);
    L.timestamp(&side[c]->ts[1]);
    L.copy(side[c]->stamp, m.stamp, 2 * (n + 2) * sizeof(uint64_t));
    L.copy(&side[c]->stuck, &m.plan->stuck, sizeof(uint32_t));
    L.copy(&side[c]->bad, &m.plan->bad, sizeof(uint32_t));
    L.close();
    return L;
  }

  void run_once(ze::List &a, ze::List &b) {
    step++;
    for (uint32_t c = 0; c < 2; c++) side[c]->plan = {step, 0, 0, 0, 0};
    a.execute();
    b.execute();
    a.synchronize();
    b.synchronize();
  }

  // The count of values that differ, printed at the first in this run.
  uint64_t diff(const char *what, uint32_t c, uint32_t s, const float *got, const float *want, size_t n) {
    uint64_t bad = 0;
    for (size_t i = 0; i < n; i++)
      if (std::memcmp(&got[i], &want[i], 4)) {
        if (!bad)
          std::printf("WRONG %s: card %u pattern %u value %zu: %.8e against %.8e\n", what, c, s, i, got[i], want[i]);
        bad++;
      }
    return bad;
  }

  // Every exchange's outcome in this run, against what the CPU's in-place partial sums gave.
  uint64_t verify(uint32_t W, Path p, bool consume) {
    uint64_t bad = 0;
    const uint32_t n = consume ? KS : NX;
    for (uint32_t c = 0; c < 2; c++) {
      CardMem &m = *cards[c];
      bad += side[c]->stuck != 0;
      if (side[c]->stuck) std::printf("WRONG: card %u stuck at W=%u\n", c, W);
      if (consume) {
        for (uint32_t s = 0; s < KS; s++) {
          auto o = m.io.get(Xo_at(c, s), (size_t)WMAX * D);
          bad += diff("X'", c, s, o.data(), &gold[c][(size_t)s * WMAX * D], W * D);
        }
      } else {
        const uint32_t last = (n - 1) % KS;
        std::vector<float> o(WMAX * E);
        const float *got = received(c, p, n - 1);
        if (p == host) std::memcpy(o.data(), got, o.size() * 4);
        else o = m.io.get(got, WMAX * E);
        bad += diff("partial sum", c, last, o.data(), &ys[1 - c][(size_t)last * WMAX * E], W * E);
        // Nothing past the rows sent: the send stores whole rows and no more.
        bad += diff("canary", c, last, o.data() + W * E, canary.data(), (WMAX - W) * E);
      }
    }
    return bad;
  }

  void put_canary() {
    for (uint32_t c = 0; c < 2; c++) {
      for (uint32_t i = 0; i < 2; i++) {
        cards[c]->io.put(cards[c]->in[i], canary);
        std::memcpy(host_in[c][i], canary.data(), canary.size() * 4);
      }
    }
  }

  void timed_case(const char *load, uint32_t W, Path p, bool consume) {
    const Mode mode = consume ? sublayer : send_only;
    ze::List &a = list(0, W, p, mode), &b = list(1, W, p, mode);
    put_canary();
    const auto t0 = Clock::now();
    do run_once(a, b);
    while (Clock::now() - t0 < std::chrono::milliseconds(50));
    std::vector<double> v[2];
    const uint32_t n = consume ? KS : NX;
    double rate[2] = {};
    for (uint32_t run = 0; run < RUNS; run++) {
      run_once(a, b);
      wrong += verify(W, p, consume);
      for (uint32_t c = 0; c < 2; c++) {
        const ze::Card &card = rt.cards()[c];
        const HostSide &h = *side[c];
        const double us = (double)((h.ts[1] - h.ts[0]) & card.timestamp_mask) / (double)card.timer_hz * 1e6;
        const double tus = (double)(h.stamp[2 * (n + 1) + 1] - h.stamp[1]) / us;
        rate[c] += tus;
        for (uint32_t k = 2; k <= n; k++) v[c].push_back((double)(h.stamp[2 * k + 1] - h.stamp[2 * (k - 1) + 1]) / tus);
      }
    }
    for (uint32_t c = 0; c < 2; c++) {
      const Stat s = stats(v[c]);
      std::printf("exchange %s %s%s W=%u card %u: p50 %.2f p99 %.2f mean %.2f max %.2f us, %zu samples, %.0f ticks a us\n",
                  load, PATH_NAME[p], consume ? ", with the sublayer's end" : "", W, c, s.p50, s.p99, s.mean, s.max,
                  v[c].size(), rate[c] / RUNS);
    }
    dumps.fresh();
  }

  void table(const char *load) {
    // The first case warms the cards, so the clocks read here are those of the work.
    ze::List &a = list(0, 1, host, send_only), &b = list(1, 1, host, send_only);
    const auto t0 = Clock::now();
    do run_once(a, b);
    while (Clock::now() - t0 < std::chrono::milliseconds(50));
    log_clocks("before the exchange table");
    for (bool consume : {false, true})
      for (uint32_t W : WIDTHS)
        for (Path p : {host, peer, peer_read}) timed_case(load, W, p, consume);
    log_clocks("after the exchange table");
    std::fflush(stdout);
  }

  // Each run is a token's exchanges, each compared with the CPU's in-place sums on the card. One run a second is also
  // compared on the CPU, in case the card's comparison goes blind.
  void soak(Load &load) {
    struct Case {
      uint32_t W;
      Path p;
    };
    const Case cases[] = {{1, host}, {1, peer}, {2, peer}, {3, peer}, {4, peer}, {6, peer}, {8, peer}, {4, host}};
    for (const Case &c : cases) {
      ze::List &a = list(0, c.W, c.p, soak_run), &b = list(1, c.W, c.p, soak_run);
      const ze::Card &card = rt.cards()[0];
      const auto t0 = Clock::now();
      uint64_t bad = 0, stuck = 0, cpu_runs = 0, cpu_bad = 0, runs = 0;
      double busy = 0, seconds = 0, next_cpu = 0;
      while (seconds < SOAK_SECONDS && !load.floor_hit()) {
        run_once(a, b);
        runs++;
        const uint32_t seen = side[0]->bad + side[1]->bad;
        bad += seen;
        stuck += side[0]->stuck + side[1]->stuck;
        busy += (double)((side[0]->ts[1] - side[0]->ts[0]) & card.timestamp_mask) / (double)card.timer_hz;
        seconds = std::chrono::duration<double>(Clock::now() - t0).count();
        if (seen || seconds >= next_cpu) {
          cpu_bad += verify(c.W, c.p, true);
          cpu_runs++;
          next_cpu = seconds + 1;
        }
      }
      wrong += bad + stuck + cpu_bad;
      std::printf("soak under load, %s W=%u: %llu exchanges a card; X' values compared on both cards %llu, wrong %llu; "
                  "stuck %llu; runs also compared on the CPU %llu, wrong %llu; lists ran for %.1f%% of %.1f s\n",
                  PATH_NAME[c.p], c.W, (unsigned long long)(runs * NX), (unsigned long long)(2 * runs * NX * c.W * D),
                  (unsigned long long)bad, (unsigned long long)stuck, (unsigned long long)cpu_runs,
                  (unsigned long long)cpu_bad, busy / seconds * 100, seconds);
      dumps.fresh();
      load.print_rates("in the soak");
      std::fflush(stdout);
    }
  }

  // ds_exch_send on card 0, the other card idle. A store under 64 bytes across PCIe is far slower and gives wrong sums
  // (docs/measurements/MEASUREMENTS.md#2026-10-12-card-to-card), so this rate is the check.
  void kernel_times() {
    const float *y = cards[0]->y;
    float *pdst = cards[1]->in[0], *hdst = host_in[1][0];
    launch_send(cards[0]->io.L, 0, WMAX, y, pdst);
    cards[0]->io.L.synchronize();
    time_launch(rt, 0, [&](ze::List &L) { launch_send(L, 0, WMAX, y, pdst); });  // warms the card
    log_clocks("before timing");
    for (uint32_t sgs : {4u, 8u, 16u, 20u, 40u})
      for (uint32_t W : {1u, 8u}) {
        std::printf("time ds_exch_send %u sub-groups into the other card W=%u: %.2f us\n", sgs, W, time_launch(rt, 0, [&](ze::List &L) { launch_send(L, 0, W, y, pdst, sgs); }));
        std::printf("time ds_exch_send %u sub-groups into host memory W=%u: %.2f us\n", sgs, W, time_launch(rt, 0, [&](ze::List &L) { launch_send(L, 0, W, y, hdst, sgs); }));
      }
    for (uint32_t W : TIMED_WIDTHS) {
      const double tp = time_launch(rt, 0, [&](ze::List &L) { launch_send(L, 0, W, y, pdst); });
      const double th = time_launch(rt, 0, [&](ze::List &L) { launch_send(L, 0, W, y, hdst); });
      std::printf("time ds_exch_send into the other card W=%u: %.2f us, %.1f GB/s\n", W, tp, W * E * 4 / tp / 1e3);
      std::printf("time ds_exch_send into host memory W=%u: %.2f us, %.1f GB/s\n", W, th, W * E * 4 / th / 1e3);
      if (W == WMAX) {
        ck.note("stores into the other card under 8 GB/s", (double)(W * E * 4 / tp / 1e3 < 8), 0);
        ck.note("stores into host memory under 8 GB/s", (double)(W * E * 4 / th / 1e3 < 8), 0);
      }
    }
    for (uint32_t W : TIMED_WIDTHS)
      std::printf("time ds_check_same W=%u: %.2f us\n", W, time_launch(rt, 0, [&](ze::List &L) { launch_check(L, 0, W, 0, cards[0]->gold); }));
    log_clocks("after timing");
  }
};

}  // namespace

int main() {
  init_clocks();
  if (test::gpus() == 0) return test::no_gpu();
  ze::Runtime rt;
  std::printf("cards visible: %u\n", rt.ncards());
  if (const char *e = std::getenv("ENGINE_EXPECT_GPUS"))
    EXPECT(rt.ncards() == (uint32_t)std::atoi(e), "%u cards ran, ENGINE_EXPECT_GPUS=%s", rt.ncards(), e);
  if (rt.ncards() < 2) return test::skip("an exchange needs two cards");
  ze::Library lib(rt);
  const uint64_t swapped = pswpout();
  std::printf("MemAvailable %.1f GiB at the start\n", mem_available_gib());
  Bench b(rt, lib);
  std::printf("core dumps already there: %zu\n", b.dumps.before());
  b.prepare();
  b.kernel_times();
  auto phase = [&](const char *name) {
    std::printf("after %s: %llu pages swapped out so far, MemAvailable %.1f GiB\n", name,
                (unsigned long long)(pswpout() - swapped), mem_available_gib());
  };
  phase("the kernel times");
  b.table("alone");
  phase("the table alone");
  Load load;
  const bool loaded = load.start(test::model_path());
  if (loaded) {
    std::this_thread::sleep_for(std::chrono::seconds(3));
    load.print_rates("while it settled");
    phase("the arena's fill");
    b.table("loaded");
    load.print_rates("in the table");
    phase("the table under load");
    b.soak(load);
    phase("the soak");
    load.stop();
    b.ck.note("the load stopped at the host memory floor", (double)load.floor_hit(), 0);
    b.ck.note("SSD reads failed", (double)load.read_failed_, 0);
  }
  std::printf("MemAvailable %.1f GiB at the end, %llu pages swapped out, %zu new core dumps\n", mem_available_gib(),
              (unsigned long long)(pswpout() - swapped), b.dumps.fresh());
  b.ck.note("wrong values in any exchange", (double)b.wrong, 0);
  b.ck.note("pages swapped out", (double)(pswpout() - swapped), 0);
  b.ck.note("new core dumps", (double)b.dumps.fresh(), 0);
  const int failed = b.ck.report();
  return failed || loaded ? failed : test::skip(load.why_not());
}
