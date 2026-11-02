// Level 3 (docs/TESTING.md): deepseek41's wait wrapper, ds_wait (kernels/models/deepseek41/layer.cl). 100 of 100 rounds
// with the write 0-1000 us late, in every direction between the CPU and the cards, the payload exact. No answer, no
// exchange flag and no Engram flag each return within 2.4e9 ticks with *stuck set, and what follows still runs. Late waits
// stop at the list's 5.4e9 ticks. The long waits run only once shorter ones showed the clock fast enough to stay inside
// xe's 5 s job timeout. A new core dump fails the run. No model is needed.
#include <chrono>

#include "ds41_kernel_test.h"
#include "models/deepseek41/layout.h"

using namespace engine;
using namespace ds41test;

namespace {

using Clock = std::chrono::steady_clock;
constexpr uint32_t ROUNDS = 100, LINE = 16;  // LINE: 64 bytes in words, each flag on its own line of host memory
constexpr size_t E = D41_EMBD;
constexpr double SLACK = 1.01;     // of the bound: the last poll and the clock reads around it
constexpr double MAX_SECONDS = 4;  // wall time of a stuck wait; a case past it ends the long waits

void spin_for(double us) {
  const auto end = Clock::now() + std::chrono::nanoseconds((int64_t)(us * 1e3));
  while (Clock::now() < end) {
  }
}

uint32_t read_flag(const uint32_t *f) { return __atomic_load_n(f, __ATOMIC_ACQUIRE); }
void raise_flag(uint32_t *f, uint32_t v) { __atomic_store_n(f, v, __ATOMIC_RELEASE); }

struct CardMem {
  Io io;
  uint32_t *want, *stuck, *nmiss;
  uint64_t *start, *stamp;
  float *y, *in, *out;
  CardMem(ze::Runtime &rt, uint32_t c) : io(rt, c) {
    want = io.dev<uint32_t>(1);
    stuck = io.dev<uint32_t>(1);
    nmiss = io.dev<uint32_t>(STEP_MAX_ROWS);
    start = io.dev<uint64_t>(1);
    stamp = io.dev<uint64_t>(8);
    y = io.dev<float>(E);
    in = io.dev<float>(E);
    out = io.dev<float>(E);
  }
  void reset(uint32_t step) {
    const uint32_t z = 0;
    const uint64_t z64 = 0;
    io.put(want, &step, 1);
    io.put(stuck, &z, 1);
    io.put(start, &z64, 1);
  }
  uint32_t stuck_now() { return io.get(stuck, 1)[0]; }
};

struct Bench {
  ze::Runtime &rt;
  ze::Library &lib;
  std::vector<CardMem *> cards;
  uint32_t *flags;    // host memory, a line each
  float *payload;     // host memory, E values
  std::mt19937 rng{1};
  Checker ck;
  uint32_t step = 0;
  double tick_rate = 0;  // the lowest ticks a microsecond seen so far, in a long wait
  CoreDumps dumps;

  Bench(ze::Runtime &r, ze::Library &l) : rt(r), lib(l) {
    for (uint32_t c = 0; c < rt.ncards(); c++) cards.push_back(new CardMem(rt, c));
    flags = static_cast<uint32_t *>(rt.alloc_host(8 * LINE * sizeof(uint32_t), 4096));
    payload = static_cast<float *>(rt.alloc_host(E * sizeof(float), 4096));
    std::memset(flags, 0, 8 * LINE * sizeof(uint32_t));
  }
  ze::List *list(uint32_t c) { return new ze::List(rt, c, ze::List::regular, false, false); }

  // Time between the two stamps of a wait, in ticks; the second word is the exit.
  uint64_t ticks(CardMem &m) {
    auto s = m.io.get(m.stamp, 2);
    return s[1] - s[0];
  }

  // The CPU's write reaches a card that is already polling; the list then copies the payload to the card.
  void cpu_to_card(uint32_t c, const std::vector<double> &delays, const char *name) {
    CardMem &m = *cards[c];
    uint32_t *flag = flags;
    Wait w;
    w.flag = flag, w.want = m.want, w.stuck = m.stuck, w.start = m.start, w.stamp = m.stamp;
    ze::List *L = list(c);
    launch_wait(*L, lib, w);
    L->copy(m.out, payload, E * sizeof(float));
    L->close();
    double seen = 0, too_short = 0;
    for (double d : delays) {
      step++;
      m.reset(step);
      auto v = randn(rng, E);
      L->execute();
      spin_for(d);
      std::memcpy(payload, v.data(), E * sizeof(float));
      raise_flag(flag, step);
      L->synchronize();
      auto o = m.io.get(m.out, E);
      const bool same = !std::memcmp(o.data(), v.data(), E * sizeof(float)), free_of_stuck = m.stuck_now() == 0;
      seen += same && free_of_stuck;
      const double t = (double)ticks(m);
      // A card that did not poll for most of a long delay read the flag late or the clock wrong.
      if (d >= 400 && t < (d - 200) * 400) too_short++;
      if (d >= 100) tick_rate = tick_rate == 0 ? t / d : std::min(tick_rate, t / d);
    }
    std::printf("%s: %.0f of %zu seen with the payload exact\n", name, seen, delays.size());
    ck.note(name, (double)delays.size() - seen, 0);
    ck.note("polled for the delay", too_short, 0);
    delete L;
  }

  // A card raises its flag late, with the payload copied to host memory before it; the CPU spins.
  void card_to_cpu(uint32_t c, const std::vector<double> &delays, const char *name) {
    CardMem &m = *cards[c];
    uint32_t *flag = flags + LINE;
    Wait w;
    w.raise = flag, w.want = m.want, w.stuck = m.stuck, w.start = m.start;
    ze::List *L = list(c);
    L->copy(payload, m.y, E * sizeof(float));
    launch_wait(*L, lib, w);
    L->close();
    double seen = 0;
    for (double d : delays) {
      step++;
      m.reset(step);
      auto v = randn(rng, E);
      m.io.put(m.y, v);
      const auto t0 = Clock::now();
      const auto start = t0 + std::chrono::nanoseconds((int64_t)(d * 1e3));
      bool started = false, got = false;
      while (Clock::now() - t0 < std::chrono::seconds(2)) {
        if (!started && Clock::now() >= start) {
          L->execute();
          started = true;
        }
        if (read_flag(flag) == step) {
          got = true;
          break;
        }
      }
      L->synchronize();
      seen += got && !std::memcmp(payload, v.data(), E * sizeof(float));
    }
    std::printf("%s: %.0f of %zu seen with the payload exact\n", name, seen, delays.size());
    ck.note(name, (double)delays.size() - seen, 0);
    delete L;
  }

  // Card a sends a row into b's memory and raises the flag late; b has been polling, then copies what arrived.
  void card_to_card(uint32_t a, uint32_t b, const std::vector<double> &delays, const char *name) {
    CardMem &ma = *cards[a], &mb = *cards[b];
    uint32_t *flag = flags + 2 * LINE;
    Wait wb;
    wb.flag = flag, wb.want = mb.want, wb.stuck = mb.stuck, wb.start = mb.start;
    ze::List *Lb = list(b);
    launch_wait(*Lb, lib, wb);
    Lb->copy(mb.out, mb.in, E * sizeof(float));
    Lb->close();
    Wait wa;
    wa.raise = flag, wa.want = ma.want, wa.stuck = ma.stuck, wa.start = ma.start;
    ze::List *La = list(a);
    La->launch(lib.get(a, "deepseek41", "ds_exch_send"), {ma.y, mb.in}, 1, 1, 1, D41_EX_SGS * 16);
    launch_wait(*La, lib, wa);
    La->close();
    double seen = 0;
    for (double d : delays) {
      step++;
      ma.reset(step);
      mb.reset(step);
      auto v = randn(rng, E);
      ma.io.put(ma.y, v);
      Lb->execute();
      spin_for(d);
      La->execute();
      La->synchronize();
      Lb->synchronize();
      auto o = mb.io.get(mb.out, E);
      seen += !std::memcmp(o.data(), v.data(), E * sizeof(float)) && mb.stuck_now() == 0;
    }
    std::printf("%s: %.0f of %zu seen with the payload exact\n", name, seen, delays.size());
    ck.note(name, (double)delays.size() - seen, 0);
    delete La;
    delete Lb;
  }

  // Waits that must not spin: the flag already holds the step, no row missed, an earlier wait stuck. Then a list past its
  // bound: the flag is still read once, and if it is not raised the wait gives up at once.
  void at_once() {
    CardMem &m = *cards[0];
    uint32_t *never = flags + 3 * LINE;
    const std::vector<uint32_t> none(STEP_MAX_ROWS, 0);
    m.io.put(m.nmiss, none);
    uint64_t earlier = 0;
    for (int which = 0; which < 5; which++) {
      step++;
      m.reset(step);
      Wait w;
      w.flag = never, w.want = m.want, w.stuck = m.stuck, w.start = m.start, w.stamp = m.stamp;
      if (which == 0 || which == 3) raise_flag(never, step);
      if (which == 1) w.nmiss = m.nmiss, w.W = STEP_MAX_ROWS;
      if (which == 2) {
        const uint32_t one = 1;
        m.io.put(m.stuck, &one, 1);
      }
      if (which >= 3) {
        // The clock can read under the list's bound, so no fixed start is long ago. This one's bound, in wrapping
        // arithmetic, ends at the last case's entry.
        const uint64_t long_ago = earlier - D41_LIST_TICKS;
        m.io.put(m.start, &long_ago, 1);
      }
      launch_wait(m.io.L, lib, w);
      m.io.L.synchronize();
      const uint32_t want_stuck = which == 2 || which == 4;
      const auto s = m.io.get(m.stamp, 2);
      earlier = s[0];
      std::printf("at once: case %d, %llu ticks\n", which, (unsigned long long)(s[1] - s[0]));
      ck.note("at once: stuck flag as it was", (double)(m.stuck_now() != want_stuck), 0);
      // Well under a millisecond at any clock: no poll loop ran to its bound.
      ck.note("at once: ticks under 2,000,000", (double)(ticks(m) >= 2000000), 0);
    }
  }

  // One stuck wait: rec records the list, the wait stamped in the card's stamp. Returns false to end the long waits.
  bool stuck_case(uint32_t c, const char *name, const std::function<void(ze::List &)> &rec,
                  const std::function<void()> &check) {
    CardMem &m = *cards[c];
    step++;
    m.reset(step);
    ze::List *L = list(c);
    rec(*L);
    L->close();
    log_clocks("before a stuck wait");
    const auto t0 = Clock::now();
    L->execute();
    L->synchronize();
    const double secs = std::chrono::duration<double>(Clock::now() - t0).count();
    log_clocks("after a stuck wait");
    delete L;
    const uint64_t t = ticks(m);
    const uint32_t st = m.stuck_now();
    std::printf("%s: card %u, %.3f s, %llu ticks, *stuck %u, %.1f ticks a us\n", name, c, secs, (unsigned long long)t, st,
                (double)t / (secs * 1e6));
    ck.note("stuck flag set", (double)(st != 1), 0);
    ck.note("ticks within the bound", (double)(t > D41_WAIT_TICKS * SLACK || t < D41_WAIT_TICKS), 0);
    ck.note("stuck wait's seconds under 4", (double)(secs >= MAX_SECONDS), 0);
    check();
    return secs < MAX_SECONDS && !dumps.fresh();
  }

  void stuck_cases() {
    for (uint32_t c = 0; c < rt.ncards(); c++) {
      CardMem &m = *cards[c];
      uint32_t *never = flags + 3 * LINE;  // holds no step of a stuck case: each has a new one
      Wait w;
      w.flag = never, w.want = m.want, w.stuck = m.stuck, w.start = m.start, w.stamp = m.stamp;
      // No answer: a row missed, the CPU never answers.
      std::vector<uint32_t> miss(STEP_MAX_ROWS, 0);
      miss[1] = 1;
      m.io.put(m.nmiss, miss);
      Wait answer = w;
      answer.nmiss = m.nmiss, answer.W = STEP_MAX_ROWS;
      if (!stuck_case(c, "no answer", [&](ze::List &L) { launch_wait(L, lib, answer); }, [] {})) return;

      // No exchange flag: this card still raises its own, and the kernel after the wait still runs.
      uint32_t *own = flags + LINE;
      Wait ex = w;
      ex.raise = own;
      auto v = randn(rng, E);
      m.io.put(m.y, v);
      if (!stuck_case(
              c, "no exchange flag",
              [&](ze::List &L) {
                launch_wait(L, lib, ex);
                L.launch(lib.get(c, "deepseek41", "ds_exch_send"), {m.y, m.out}, 1, 1, 1, D41_EX_SGS * 16);
              },
              [&] {
                ck.note("exchange: own flag raised", (double)(read_flag(own) != step), 0);
                auto o = m.io.get(m.out, E);
                ck.exact("exchange: the kernel after the wait ran", o.data(), v.data(), E * sizeof(float));
              }))
        return;

      // No Engram flag: the copy of g still runs.
      auto g = randn(rng, E);
      std::memcpy(payload, g.data(), E * sizeof(float));
      if (!stuck_case(
              c, "no Engram flag",
              [&](ze::List &L) {
                launch_wait(L, lib, w);
                L.copy(m.out, payload, E * sizeof(float));
              },
              [&] {
                auto o = m.io.get(m.out, E);
                ck.exact("Engram: the copy of g ran", o.data(), g.data(), E * sizeof(float));
              }))
        return;

      // A wait after a stuck one finds *stuck set and does not poll. The stamp is the second's, so it is short.
      Wait first = w, second = w;
      first.stamp = nullptr;
      step++;
      m.reset(step);
      ze::List *L = list(c);
      launch_wait(*L, lib, first);
      launch_wait(*L, lib, second);
      L->close();
      const auto t0 = Clock::now();
      L->execute();
      L->synchronize();
      delete L;
      const double secs = std::chrono::duration<double>(Clock::now() - t0).count();
      std::printf("two waits: card %u, %.3f s, second wait %llu ticks\n", c, secs, (unsigned long long)ticks(m));
      ck.note("two waits: the second does not spin", (double)(ticks(m) >= 2000000), 0);
      ck.note("stuck wait's seconds under 4", (double)(secs >= MAX_SECONDS), 0);
      if (secs >= MAX_SECONDS || dumps.fresh() || !list_bound(c)) return;
    }
  }

  // Late waits, none past its own bound, stop at the list's. The CPU answers two, 1.7e9 ticks apart; the third gives up
  // when the list's bound passes, and the fourth does not poll. Returns false to end the long waits.
  bool list_bound(uint32_t c) {
    if (tick_rate < D41_LIST_TICKS / (MAX_SECONDS * 1e6)) {
      std::printf("FAIL: the device clock counts %.1f ticks a us, too slow to run the list's bound inside %.0f s\n", tick_rate,
                  MAX_SECONDS);
      ck.note("the clock fast enough for the list's bound", 1, 0);
      return false;
    }
    CardMem &m = *cards[c];
    uint32_t *f = flags + 4 * LINE;
    step++;
    m.reset(step);
    ze::List *L = list(c);
    for (uint32_t i = 0; i < 4; i++) {
      Wait w;
      w.flag = f + i * LINE, w.want = m.want, w.stuck = m.stuck, w.start = m.start, w.stamp = m.stamp + 2 * i;
      launch_wait(*L, lib, w);
    }
    L->close();
    const double late = 1.7e9 / tick_rate;
    log_clocks("before the list's bound");
    const auto t0 = Clock::now();
    L->execute();
    spin_for(late);
    raise_flag(f, step);
    spin_for(late);
    raise_flag(f + LINE, step);
    L->synchronize();
    const double secs = std::chrono::duration<double>(Clock::now() - t0).count();
    log_clocks("after the list's bound");
    delete L;
    const auto s = m.io.get(m.stamp, 8);
    const uint64_t total = s[7] - s[0];
    std::printf("list's bound: card %u, %.3f s, %llu ticks; waits %llu %llu %llu %llu ticks; *stuck %u\n", c, secs,
                (unsigned long long)total, (unsigned long long)(s[1] - s[0]), (unsigned long long)(s[3] - s[2]),
                (unsigned long long)(s[5] - s[4]), (unsigned long long)(s[7] - s[6]), m.stuck_now());
    ck.note("list's bound: stuck flag set", (double)(m.stuck_now() != 1), 0);
    ck.note("list's bound: ticks within it", (double)(total > D41_LIST_TICKS * SLACK || total < D41_LIST_TICKS), 0);
    for (uint32_t i = 0; i < 3; i++)
      ck.note("list's bound: no wait past its own", (double)(s[2 * i + 1] - s[2 * i] >= D41_WAIT_TICKS), 0);
    ck.note("list's bound: the last wait does not spin", (double)(s[7] - s[6] >= 2000000), 0);
    ck.note("list's bound: seconds under 4", (double)(secs >= MAX_SECONDS), 0);
    return secs < MAX_SECONDS && !dumps.fresh();
  }

  void bench() {
    CardMem &m = *cards[0];
    step++;
    m.reset(step);
    uint32_t *set = flags + 3 * LINE;
    raise_flag(set, step);
    const std::vector<uint32_t> none(STEP_MAX_ROWS, 0);
    m.io.put(m.nmiss, none);
    Wait read;
    read.flag = set, read.want = m.want, read.stuck = m.stuck, read.start = m.start;
    time_launch(rt, 0, [&](ze::List &L) { launch_wait(L, lib, read); });  // warms the card
    log_clocks("before timing");
    for (uint32_t w : TIMED_WIDTHS) {
      std::printf("time ds_wait flag already set W=%u: %.2f us\n", w, time_launch(rt, 0, [&](ze::List &L) { launch_wait(L, lib, read); }));
      Wait skip = read;
      skip.nmiss = m.nmiss, skip.W = w;
      std::printf("time ds_wait no row missed W=%u: %.2f us\n", w, time_launch(rt, 0, [&](ze::List &L) { launch_wait(L, lib, skip); }));
    }
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
  ze::Library lib(rt);
  Bench b(rt, lib);
  const uint64_t swapped = pswpout();
  std::printf("core dumps already there: %zu\n", b.dumps.before());

  std::vector<double> delays;
  for (uint32_t i = 0; i < ROUNDS; i++) delays.push_back(i * 1000.0 / (ROUNDS - 1));
  std::shuffle(delays.begin(), delays.end(), b.rng);

  b.cpu_to_card(0, delays, "CPU to card 0");
  b.card_to_cpu(0, delays, "card 0 to CPU");
  if (rt.ncards() > 1) {
    b.cpu_to_card(1, delays, "CPU to card 1");
    b.card_to_cpu(1, delays, "card 1 to CPU");
    b.card_to_card(0, 1, delays, "card 0 to card 1");
    b.card_to_card(1, 0, delays, "card 1 to card 0");
  } else {
    std::printf("SKIP: the directions between the cards need two cards\n");
  }
  b.at_once();
  b.bench();

  // The clock in long waits the CPU ends, so a stuck wait's time is known before one runs: 100 and 500 ms.
  b.tick_rate = 0;
  b.cpu_to_card(0, {100000.0, 500000.0}, "CPU to card 0, 100 and 500 ms late");
  if (rt.ncards() > 1) b.cpu_to_card(1, {100000.0, 500000.0}, "CPU to card 1, 100 and 500 ms late");
  std::printf("the lowest clock in a long wait: %.1f ticks a us\n", b.tick_rate);
  // 600 ticks a us makes the bound 4 s, under xe's 5 s. The clock is read here, not assumed.
  if (b.tick_rate < 600) {
    std::printf("FAIL: the device clock counts %.1f ticks a us, too slow to run the stuck waits inside 5 s\n", b.tick_rate);
    b.ck.note("the clock fast enough for the stuck waits", 1, 0);
  } else {
    b.stuck_cases();
  }
  const uint64_t swapped_now = pswpout();
  std::printf("pages swapped out in this run: %llu\n", (unsigned long long)(swapped_now - swapped));
  b.ck.note("pages swapped out", (double)(swapped_now - swapped), 0);
  std::printf("new core dumps: %zu\n", b.dumps.fresh());
  b.ck.note("new core dumps", (double)b.dumps.fresh(), 0);
  return b.ck.report();
}
