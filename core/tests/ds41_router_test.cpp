// Level 3 (docs/TESTING.md): deepseek41's MoE kernels between the products (kernels/models/deepseek41/layer.cl): ds_router,
// ds_swiglu_clamp_a8 and ds_moe_combine, against ref_ops and ggml at the model's shapes, and each one's time at every
// recorded width with every card visible. Limits: 1e-6 on weights, scales and sums; exact on selection, groups and
// MissJobs; a8 codes within their rounding. No model is needed.
#include <atomic>
#include <cmath>
#include <map>

#include "ds41_kernel_test.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "models/deepseek41/layout.h"
#include "models/deepseek41/ref_ops.h"

using namespace engine;
using namespace ds41test;
namespace ref = engine::deepseek41::ref;

namespace {

constexpr size_t E = D41_EMBD, N = D41_EXPERTS, U = D41_USED;
constexpr uint32_t WMAX = STEP_MAX_ROWS;
constexpr float LIMIT = 1e-6f;
const void *const none = nullptr;

// Graph values from ggml, an oracle independent of ref_ops. fn builds the result from input a (and b, when given).
template <class Fn>
std::vector<float> ggml_eval(const std::vector<float> &a, const std::vector<float> &b, size_t cols, Fn fn) {
  ggml_init_params ip = {a.size() * 16 + (32u << 20), nullptr, false};
  ggml_context *ctx = ggml_init(ip);
  ggml_tensor *ta = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int64_t)cols, (int64_t)(a.size() / cols));
  std::memcpy(ta->data, a.data(), a.size() * 4);
  ggml_tensor *tb = nullptr;
  if (!b.empty()) {
    tb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int64_t)cols, (int64_t)(b.size() / cols));
    std::memcpy(tb->data, b.data(), b.size() * 4);
  }
  ggml_tensor *o = fn(ctx, ta, tb);
  ggml_cgraph *gf = ggml_new_graph(ctx);
  ggml_build_forward_expand(gf, o);
  EXPECT(ggml_graph_compute_with_ctx(ctx, gf, 1) == GGML_STATUS_SUCCESS, "ggml's graph failed");
  std::vector<float> v(a.size());
  std::memcpy(v.data(), o->data, a.size() * 4);
  ggml_free(ctx);
  return v;
}

// ---- the router ----

enum class Scores {
  random, clustered, narrow, flat, paired, large, huge, tiny, lowest, zero_s, same_rows, biased, one_lane
};

struct RouteData {
  std::vector<float> logits, bias;  // [W][N], [N]
  bool exact_ties = false;          // built ties: the order must be exact; else ranks within a kernel's rounding may swap
};

RouteData make_route(std::mt19937 &rng, uint32_t W, Scores kind) {
  RouteData d;
  d.logits = randn(rng, W * N, 4.0f);
  d.bias = randn(rng, N, 0.5f);
  std::uniform_real_distribution<float> uni(0.0f, 1.0f);
  for (uint32_t r = 0; r < W; r++)
    for (size_t e = 0; e < N; e++) {
      float &l = d.logits[r * N + e];
      switch (kind) {
        case Scores::random: break;
        case Scores::clustered: l = (uni(rng) < 0.02f ? 6.0f : -3.0f) + 0.05f * l; break;
        case Scores::narrow: l = 2.0f + 0.02f * l; break;
        case Scores::flat: l = 1.5f; break;
        case Scores::paired: l = (e / 2 % 20 == 7 ? 5.0f : e / 2 % 20 == 3 ? 4.0f : -2.0f) + 0.01f * (float)(e / 2 % 7); break;
        // Crosses softplus's turn at 20, where both forms agree in float.
        case Scores::large: l = 15.0f + 2.0f * l; break;
        // Past exp's overflow near 88.7, where only softplus's turn at 20 keeps s finite.
        case Scores::huge: l = 90.0f + 2.0f * l; break;
        case Scores::tiny: l = -30.0f + 0.5f * l; break;
        // exp stays normal and s is near 4e-18, so the 1e-20 moves a weight by about 1e-4: a clamp or a larger epsilon
        // fails.
        case Scores::lowest: l = -80.0f + 0.25f * l; break;
        case Scores::zero_s: l = -200.0f; break;  // exp underflows: s is 0, and only the 1e-20 keeps the weight finite
        case Scores::same_rows: l = d.logits[e]; break;
        case Scores::biased: break;
        // 7 equal top keys in one lane of the sub-group and the rest equal: its queue of 3 empties and rt_top refills
        // it.
        case Scores::one_lane: l = e % 16 == r % 16 && e / 16 < 7 ? 5.0f : 1.0f; break;
      }
    }
  for (size_t e = 0; e < N; e++) {
    if (kind == Scores::flat || kind == Scores::paired || kind == Scores::zero_s || kind == Scores::one_lane)
      d.bias[e] = 0.0f;
    if (kind == Scores::biased) d.bias[e] = e % 64 == 9 ? 50.0f : -std::fabs(d.bias[e]);
  }
  d.exact_ties = kind == Scores::flat || kind == Scores::paired || kind == Scores::zero_s || kind == Scores::biased ||
                 kind == Scores::one_lane;
  return d;
}

// Which experts are cached: slot pairs 0 .. n - 1 for n random experts, the others -1.
std::vector<int32_t> make_map(std::mt19937 &rng, uint32_t n) {
  std::vector<int32_t> map(N, -1), order(N);
  for (size_t e = 0; e < N; e++) order[e] = (int32_t)e;
  std::shuffle(order.begin(), order.end(), rng);
  for (uint32_t k = 0; k < n; k++) map[order[k]] = (int32_t)k;
  return map;
}

struct WantGroup {
  int32_t slot;
  uint32_t n, cls;
  std::vector<uint32_t> asg;
};

// Groups as ds_router must give them, numbered by their first assignment.
std::vector<WantGroup> want_groups(const std::vector<int32_t> &slots, uint32_t split, uint32_t flip) {
  std::vector<WantGroup> g;
  std::map<int32_t, size_t> at;
  for (size_t a = 0; a < slots.size(); a++) {
    const int32_t s = slots[a];
    if (s < 0) continue;
    auto it = at.find(s);
    if (it == at.end()) {
      const bool upper = (uint32_t)s >= split;
      it = at.emplace(s, g.size()).first;
      g.push_back({s - (upper ? (int32_t)split : 0), 0, (uint32_t)upper ^ flip, {}});
    }
    g[it->second].n++;
    g[it->second].asg.push_back((uint32_t)a);
  }
  return g;
}

// What one launch of ds_router left on a card and in host memory.
struct RouteOut {
  std::vector<int32_t> ids, slots;
  std::vector<float> w;
  std::vector<uint32_t> nmiss;
  std::vector<D41Group> groups;
  uint32_t ngroups = 0;
  std::vector<uint8_t> headers;  // 128 B a row
};

struct RouteMem {
  Io io;
  float *logits, *bias, *w;
  int32_t *map, *ids, *slots;
  uint32_t *plan, *nmiss, *ngroups;  // plan: real rows, step
  D41Group *groups;
  RouteMem(ze::Runtime &rt, uint32_t c) : io(rt, c) {
    logits = io.dev<float>(WMAX * N);
    bias = io.dev<float>(N);
    w = io.dev<float>(WMAX * U);
    map = io.dev<int32_t>(N);
    ids = io.dev<int32_t>(WMAX * U);
    slots = io.dev<int32_t>(WMAX * U);
    plan = io.dev<uint32_t>(2);
    nmiss = io.dev<uint32_t>(WMAX);
    ngroups = io.dev<uint32_t>(1);
    groups = io.dev<D41Group>(WMAX * U);
  }
};

struct Bench {
  ze::Runtime &rt;
  ze::Library &lib;
  std::mt19937 rng{7};
  Checker ck;
  std::vector<RouteMem *> rm;
  D41MissJob *jobs;
  uint32_t near_ties = 0, shown = 0, weights_skipped = 0;
  Bench(ze::Runtime &r, ze::Library &l) : rt(r), lib(l) {
    for (uint32_t c = 0; c < std::min(rt.ncards(), 2u); c++) rm.push_back(new RouteMem(rt, c));
    jobs = static_cast<D41MissJob *>(rt.alloc_host(WMAX * sizeof(D41MissJob), 4096));
    EXPECT((reinterpret_cast<uintptr_t>(jobs) & 63) == 0, "a MissJob must be 64 B aligned");
  }

  void launch_router(ze::List &L, RouteMem &m, uint32_t W, uint32_t split, uint32_t flip, const void *job,
                     uint32_t sgs = 0) {
    L.launch(lib.get(m.io.card, "deepseek41", "ds_router"),
             {m.logits, m.bias, m.map, W, split, flip, m.plan, m.plan + 1, m.ids, m.w, m.slots, m.nmiss, job, m.groups,
              m.ngroups},
             1, 1, 1, 16 * W * (sgs ? sgs : D41_RT_SGS(W)));
  }

  // Marks everything the kernel may leave alone, so that what it writes shows.
  void reset_jobs(uint32_t W) {
    std::memset((void *)jobs, 0xAB, W * sizeof(D41MissJob));
  }
  void mark_results(RouteMem &m) {
    const std::vector<uint32_t> words(WMAX * U, 0xABABABABu);
    const std::vector<uint8_t> bytes(WMAX * U * sizeof(D41Group), 0xAB);
    m.io.put(reinterpret_cast<uint32_t *>(m.ids), words);
    m.io.put(reinterpret_cast<uint32_t *>(m.slots), words);
    m.io.put(reinterpret_cast<uint32_t *>(m.w), words);
    m.io.put(m.nmiss, words.data(), WMAX);
    m.io.put(m.ngroups, words.data(), 1);
    m.io.put(reinterpret_cast<uint8_t *>(m.groups), bytes);
  }

  RouteOut route(RouteMem &m, const RouteData &d, const std::vector<int32_t> &map, uint32_t W, uint32_t real,
                 uint32_t split, uint32_t flip, uint32_t step, bool with_jobs) {
    m.io.put(m.logits, d.logits.data(), W * N);
    m.io.put(m.bias, d.bias);
    m.io.put(m.map, map);
    const uint32_t plan[2] = {real, step};
    m.io.put(m.plan, plan, 2);
    if (with_jobs) reset_jobs(W);
    mark_results(m);
    launch_router(m.io.L, m, W, split, flip, with_jobs ? (const void *)jobs : none);
    m.io.L.synchronize();
    RouteOut o;
    o.ids = m.io.get(m.ids, W * U);
    o.slots = m.io.get(m.slots, W * U);
    o.w = m.io.get(m.w, W * U);
    o.nmiss = m.io.get(m.nmiss, W);
    o.ngroups = m.io.get(m.ngroups, 1)[0];
    o.groups = m.io.get(m.groups, WMAX * U);
    if (with_jobs) {
      o.headers.resize(W * 128u);
      for (uint32_t r = 0; r < W; r++) std::memcpy(&o.headers[r * 128u], (const void *)&jobs[r], 128);
    }
    return o;
  }

  void check_groups(const RouteOut &o, const std::vector<WantGroup> &want) {
    ck.note("ds_router group count", std::fabs((double)o.ngroups - (double)want.size()), 0);
    if (o.ngroups != want.size()) return;
    for (size_t g = 0; g < want.size(); g++) {
      const D41Group &h = o.groups[g];
      const WantGroup &w = want[g];
      ck.note("ds_router group slot within its class", std::fabs((double)h.slot - w.slot), 0);
      ck.note("ds_router group slot class", std::fabs((double)h.cls - w.cls), 0);
      ck.note("ds_router group members", std::fabs((double)h.n - w.n), 0);
      for (uint32_t i = 0; i < w.n && i < STEP_MAX_ROWS; i++)
        ck.note("ds_router group assignment", std::fabs((double)h.asg[i] - w.asg[i]), 0);
    }
  }

  // Every word of the header, so also that the stores reach all of its 128 bytes (the marks in pad go).
  void check_header(const RouteOut &o, uint32_t r, uint32_t step, const int32_t *ids, const float *w,
                    const int32_t *slots) {
    D41MissJob h;
    std::memcpy((void *)&h, &o.headers[r * 128u], 128);
    uint32_t n = 0;
    int32_t want_ids[U];
    float want_w[U];
    std::fill(want_ids, want_ids + U, -1);
    std::fill(want_w, want_w + U, 0.0f);
    for (size_t k = 0; k < U; k++)
      if (ids[k] >= 0 && slots[k] < 0) {
        want_ids[n] = ids[k];
        want_w[n] = w[k];
        n++;
      }
    ck.note("MissJob step", std::fabs((double)h.seq - step), 0);
    ck.note("MissJob count", std::fabs((double)h.nmiss - n), 0);
    ck.exact("MissJob ids", h.ids, want_ids, sizeof want_ids);
    ck.exact("MissJob weights, the card's bits", h.w, want_w, sizeof want_w);
    ck.exact("MissJob routed ids", h.all, ids, U * 4);
    const uint32_t zero[12] = {};
    ck.exact("MissJob pad: the 128 B all stored", h.pad, zero, sizeof zero);
  }

  void run_route_case(Scores kind, uint32_t W, uint32_t cached, uint32_t split) {
    const RouteData d = make_route(rng, W, kind);
    const auto map = make_map(rng, cached);
    const uint32_t step = rng() >> 1;
    RouteMem &m = *rm[0];
    RouteOut o = route(m, d, map, W, W, split, 0, step, true);

    // ggml's softplus loses a small s, so it stands as an oracle on rows whose selected logits are not small. ref_ops
    // holds DeepSeek's log1p form.
    const auto gs = ggml_eval(d.logits, {}, N, [](ggml_context *c, ggml_tensor *a, ggml_tensor *) {
      return ggml_sqrt(c, ggml_softplus(c, a));
    });
    for (uint32_t r = 0; r < W; r++) {
      const ref::Route want = ref::router(&d.logits[r * N], d.bias.data());
      const float *lr = &d.logits[r * N];
      auto sb = [&](int32_t e) { return (double)ref_s(lr[e]) + d.bias[e]; };
      int32_t ids[U];
      bool valid = true;
      for (size_t k = 0; k < U; k++) {
        const int32_t got = o.ids[r * U + k];
        ids[k] = got;
        if (got == want.id[k]) continue;
        if (got < 0 || got >= (int32_t)N) {
          ck.note("ds_router selection", 1, 0);
          valid = false;
          continue;
        }
        // Only a kernel's rounding of two scores within 1e-6 may swap ranks, and never on built ties.
        const bool near = !d.exact_ties && std::fabs(sb(got) - sb(want.id[k])) < 1e-6;
        near_ties += near;
        ck.note("ds_router selection", near ? 0 : 1, 0);
        if (!near && shown++ < 12)
          std::printf("selection: kind %d W=%u row %u rank %zu got %d (%.9g) want %d (%.9g)\n", (int)kind, W, r, k, got,
                      sb(got), want.id[k], sb(want.id[k]));
      }
      if (!valid) continue;
      double sum_g = 0;
      bool ggml_serves = true;
      for (size_t k = 0; k < U; k++) {
        sum_g += gs[r * N + ids[k]];
        ggml_serves = ggml_serves && lr[ids[k]] >= -2.0f;
      }
      for (size_t k = 0; k < U; k++) {
        // ref_ops' weight of the same expert, as ranks swapped by rounding hold the same experts. A swap into the sixth
        // rank changes the sum, so that row's expert has no such weight.
        const int32_t *at = std::find(want.id, want.id + U, ids[k]);
        if (at == want.id + U) {
          weights_skipped++;
        } else {
          ck.note("ds_router weight against ref_ops", std::fabs((double)o.w[r * U + k] - want.w[at - want.id]), LIMIT);
        }
        const double wg = gs[r * N + ids[k]] / (sum_g + 1e-20) * 1.5;
        if (ggml_serves) ck.note("ds_router weight against ggml", std::fabs((double)o.w[r * U + k] - wg), LIMIT);
      }
      uint32_t nmiss = 0;
      for (size_t k = 0; k < U; k++) {
        ck.note("ds_router slot map", std::fabs((double)o.slots[r * U + k] - (map[ids[k]] < 0 ? -1 : map[ids[k]])), 0);
        nmiss += map[ids[k]] < 0;
      }
      ck.note("ds_router miss count", std::fabs((double)o.nmiss[r] - nmiss), 0);
      check_header(o, r, step, ids, &o.w[r * U], &o.slots[r * U]);
    }
    check_groups(o, want_groups(o.slots, split, 0));
    for (uint32_t r = 0; r < W; r++) {
      const float *x = jobs[r].x;
      const uint32_t mark = 0xABABABABu;
      ck.exact("the router leaves MissJob x alone", &x[0], &mark, 4);
    }

    // Card 1 routes the same and holds the other class of each slot pair (ADR-0062 item 2).
    if (rm.size() > 1) {
      RouteOut q = route(*rm[1], d, map, W, W, split, 1, step, false);
      ck.exact("ds_router both cards' ids", q.ids.data(), o.ids.data(), q.ids.size() * 4);
      ck.exact("ds_router both cards' weights", q.w.data(), o.w.data(), q.w.size() * 4);
      ck.exact("ds_router both cards' slots", q.slots.data(), o.slots.data(), q.slots.size() * 4);
      ck.exact("ds_router both cards' miss counts", q.nmiss.data(), o.nmiss.data(), W * 4);
      check_groups(q, want_groups(o.slots, split, 1));
    }
  }

  // 5 real rows on the 6-row list against the same rows on the 5-row list: only row 6 differs, and it holds nothing.
  void run_padded() {
    for (Scores kind : {Scores::random, Scores::same_rows}) {
      const RouteData d = make_route(rng, 6, kind);
      const auto map = make_map(rng, 104);
      RouteMem &m = *rm[0];
      RouteOut five = route(m, d, map, 5, 5, 52, 0, 41, true);
      RouteOut pad = route(m, d, map, 6, 5, 52, 0, 41, true);
      ck.exact("padded step: ids", pad.ids.data(), five.ids.data(), 5 * U * 4);
      ck.exact("padded step: weights", pad.w.data(), five.w.data(), 5 * U * 4);
      ck.exact("padded step: slots", pad.slots.data(), five.slots.data(), 5 * U * 4);
      ck.exact("padded step: misses", pad.nmiss.data(), five.nmiss.data(), 5 * 4);
      ck.exact("padded step: MissJobs", pad.headers.data(), five.headers.data(), 5 * 128);
      ck.note("padded step: group count", std::fabs((double)pad.ngroups - five.ngroups), 0);
      for (uint32_t g = 0; g < five.ngroups && g < pad.ngroups; g++) {
        const D41Group &a = pad.groups[g], &b = five.groups[g];
        ck.note("padded step: group slot and class", std::fabs((double)a.slot - b.slot) + std::fabs((double)a.cls - b.cls), 0);
        ck.note("padded step: group members", std::fabs((double)a.n - b.n), 0);
        ck.exact("padded step: group assignments", a.asg, b.asg, b.n * 4);
        for (uint32_t i = 0; i < a.n && i < STEP_MAX_ROWS; i++)
          ck.note("padded step: no assignment of row 6", a.asg[i] >= 5 * U, 0);
      }
      const int32_t minus[U] = {-1, -1, -1, -1, -1, -1};
      const float zero[U] = {};
      ck.exact("padded row: ids", &pad.ids[5 * U], minus, sizeof minus);
      ck.exact("padded row: slots", &pad.slots[5 * U], minus, sizeof minus);
      ck.exact("padded row: weights", &pad.w[5 * U], zero, sizeof zero);
      ck.note("padded row: nmiss", (double)pad.nmiss[5], 0);
      check_header(pad, 5, 41, minus, zero, minus);
    }
  }

  // The host sees a job's header whole as soon as it sees the step: 300 rounds on two data sets that differ everywhere.
  void run_seq_order() {
    RouteMem &m = *rm[0];
    const uint32_t W = WMAX;
    const RouteData d[2] = {make_route(rng, W, Scores::random), make_route(rng, W, Scores::clustered)};
    const auto map = make_map(rng, 104);
    m.io.put(m.bias, d[0].bias);
    m.io.put(m.map, map);
    RouteOut want[2];
    for (int i : {0, 1}) want[i] = route(m, d[i], map, W, W, 52, 0, 1, true);
    uint32_t wrong = 0;
    for (uint32_t round = 0; round < 300; round++) {
      const RouteData &dr = d[round & 1];
      m.io.put(m.logits, dr.logits.data(), W * N);
      m.io.put(m.bias, dr.bias);
      const uint32_t plan[2] = {W, round + 2};
      m.io.put(m.plan, plan, 2);
      launch_router(m.io.L, m, W, 52, 0, jobs);
      for (uint32_t r = 0; r < W; r++) {
        auto *seq = reinterpret_cast<volatile uint32_t *>(&jobs[r].seq);
        const auto t0 = std::chrono::steady_clock::now();
        while (*seq != round + 2)
          if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(5)) {
            wrong++;
            break;
          }
        std::atomic_thread_fence(std::memory_order_acquire);
        uint8_t got[128];
        std::memcpy(got, (const void *)&jobs[r], 128);
        wrong += std::memcmp(got + 4, &want[round & 1].headers[r * 128u + 4], 124) != 0;
      }
      m.io.L.synchronize();
    }
    ck.note("MissJob whole when the host sees its step, 300 rounds", (double)wrong, 0);
  }

  void bench_router() {
    RouteMem &m = *rm[0];
    const RouteData d = make_route(rng, WMAX, Scores::random);
    const auto map = make_map(rng, 104);
    m.io.put(m.logits, d.logits);
    m.io.put(m.bias, d.bias);
    m.io.put(m.map, map);
    for (uint32_t W : WIDTHS) {
      const uint32_t plan[2] = {W, 3};
      m.io.put(m.plan, plan, 2);
      for (uint32_t sgs : {4u, 6u, 8u, 12u, 16u, 24u})
        if (W * sgs <= 64) std::printf("time ds_router no jobs, %u sub-groups a row W=%u: %.2f us\n", sgs, W,
                    time_launch(rt, 0, [&](ze::List &L) { launch_router(L, m, W, 52, 0, none, sgs); }));
      const double t0 = time_launch(rt, 0, [&](ze::List &L) { launch_router(L, m, W, 52, 0, none); });
      const double t1 = time_launch(rt, 0, [&](ze::List &L) { launch_router(L, m, W, 52, 0, jobs); });
      std::printf("time ds_router no jobs W=%u: %.2f us\n", W, t0);
      std::printf("time ds_router jobs in host memory W=%u: %.2f us\n", W, t1);
    }
    // A padded row skips its routing: 1 real row on the 8-row list.
    const uint32_t plan[2] = {1, 3};
    m.io.put(m.plan, plan, 2);
    std::printf("time ds_router 1 real row on the 8-row list, jobs: %.2f us\n",
                time_launch(rt, 0, [&](ze::List &L) { launch_router(L, m, 8, 52, 0, jobs); }));
  }

  // A store under 64 bytes across PCIe is slow and gives wrong values
  // (docs/measurements/MEASUREMENTS.md#2026-10-15-prompt-exchange-stores), yet a few in a router launch would not show
  // beside its fences. So the header's stores alone, at a rate, with two-byte stores to show that the limit can fail.
  void bench_job_stores() {
    constexpr uint32_t HEADERS = 1024;
    auto *h = static_cast<uint32_t *>(rt.alloc_host(HEADERS * 128, 4096));
    auto rate = [&](uint32_t kind) {
      const double us = time_launch(rt, 0, [&](ze::List &L) {
        L.launch(lib.get(0, "deepseek41", "ds_job_store_rate"), {h, HEADERS, kind}, 1, 1, 1, 16 * 16);
      });
      return HEADERS * 192.0 / us / 1e3;
    };
    const double wide = rate(0), word = rate(1), two = rate(2);
    std::printf("MissJob header stores into host memory: %.1f GB/s; a word a lane %.1f, two bytes a lane %.1f\n", wide,
                word, two);
    ck.note("MissJob header stores under 8 GB/s", wide < 8, 0);
    ck.note("two-byte stores not under 8 GB/s: the limit cannot fail", two >= 8, 0);
    rt.free(h);
  }

  // ---- the shared expert's SwiGLU ----

  void check_a8(const char *name_scale, const char *name_code, const uint8_t *row, const float *want, uint32_t ff) {
    for (uint32_t u = 0; u < ff / 32; u++) {
      float amax = 0;
      for (uint32_t i = 0; i < 32; i++) amax = std::fmax(amax, std::fabs(want[32 * u + i]));
      const float wd = amax / 127.0f;
      float d;
      std::memcpy(&d, row + ff + 4 * u, 4);
      ck.note(name_scale, wd > 0 ? std::fabs((double)d - wd) / wd : std::fabs((double)d), LIMIT);
      for (uint32_t i = 0; i < 32; i++) {
        // A code rounds want / d: it may stand off by the half plus what the kernel's float rounding moves it.
        const double code = (double)(int8_t)row[32 * u + i], exact = wd > 0 ? want[32 * u + i] / (double)wd : 0.0;
        ck.note(name_code, std::max(0.0, std::fabs(code - exact) - 0.5), 1e-3);
      }
    }
  }

  void run_swiglu() {
    const uint32_t c = 0;
    Io io(rt, c);
    const size_t FF = D41_FF;
    auto *g = io.dev<float>(WMAX * FF);
    auto *u = io.dev<float>(WMAX * FF);
    const size_t row_max = FF + FF / 8;
    auto *ha = io.dev<uint8_t>(WMAX * row_max + 64);
    auto *one = io.dev<float>(WMAX * 2 * FF);  // gate and up as one matvec launch writes them, [W][2 ff]
    auto launch_at = [&](ze::List &L, uint32_t W, const float *pg, const float *pu, uint32_t stride, uint32_t ff,
                         uint32_t sgs) {
      L.launch(lib.get(c, "deepseek41", "ds_swiglu_clamp_a8"), {pg, pu, stride, ff, ha}, W, 1, 1, sgs * 16);
    };
    auto launch = [&](ze::List &L, uint32_t W, uint32_t ff, uint32_t sgs) { launch_at(L, W, g, u, ff, ff, sgs); };
    for (uint32_t ff : {1280u, 1024u, 2304u})
      for (uint32_t W : WIDTHS) {
        // Values on both sides of the clamp, the clamp itself, and units all at or near 0.
        std::vector<float> vg = randn(rng, W * ff, 6.0f), vu = randn(rng, W * ff, 6.0f);
        for (size_t i = 0; i < 64 && i < vg.size(); i++) {
          const float e[8] = {10.0f, -10.0f, 40.0f, -40.0f, 1e5f, -1e5f, 0.0f, -0.0f};
          vg[i] = e[i % 8];
          vu[i] = e[(i / 8) % 8];
        }
        for (size_t i = 64; i < 96 && i < vg.size(); i++) vu[i] = 0.0f;
        for (size_t i = 96; i < 128 && i < vg.size(); i++) vg[i] = -1e5f;
        io.put(g, vg);
        io.put(u, vu);
        std::vector<uint8_t> marks(W * (ff + ff / 8) + 64, 0xAB);
        io.put(ha, marks);
        launch(io.L, W, ff, D41_SWIGLU_SGS);
        io.L.synchronize();
        auto got = io.get(ha, marks.size());
        std::vector<float> want(W * ff);
        for (size_t i = 0; i < want.size(); i++) want[i] = ref::swiglu(vg[i], vu[i]);
        auto viag = ggml_eval(vg, vu, ff, [](ggml_context *ctx, ggml_tensor *a, ggml_tensor *b) {
          return ggml_swiglu_clamp(ctx, a, b, D41_SWIGLU_LIMIT);
        });
        ck.abs("ref_ops' SwiGLU against ggml", want.data(), viag.data(), want.size(), LIMIT * 10);
        for (uint32_t r = 0; r < W; r++) {
          check_a8("ds_swiglu_clamp_a8 scale against ref_ops", "ds_swiglu_clamp_a8 code against ref_ops",
                   &got[r * (ff + ff / 8u)], &want[r * ff], ff);
          check_a8("ds_swiglu_clamp_a8 scale against ggml", "ds_swiglu_clamp_a8 code against ggml",
                   &got[r * (ff + ff / 8u)], &viag[r * ff], ff);
        }
        ck.exact("ds_swiglu_clamp_a8 writes nothing past its rows", &got[W * (ff + ff / 8u)], &marks[0], 64);
        // The same values as rows of one matrix: the same bytes.
        std::vector<float> rows(W * 2 * (size_t)ff);
        for (uint32_t r = 0; r < W; r++) {
          std::copy_n(&vg[r * ff], ff, &rows[r * 2 * ff]);
          std::copy_n(&vu[r * ff], ff, &rows[r * 2 * ff + ff]);
        }
        io.put(one, rows);
        launch_at(io.L, W, one, one + ff, 2 * ff, ff, D41_SWIGLU_SGS);
        io.L.synchronize();
        auto got_one = io.get(ha, W * (ff + ff / 8u));
        ck.exact("ds_swiglu_clamp_a8 rows of one matrix", got_one.data(), got.data(), got_one.size());
      }
    // A time for each width of the shared part, Q5_K's cut, and the sub-groups the kernel is pinned to.
    io.put(g, randn(rng, WMAX * FF, 3.0f));
    io.put(u, randn(rng, WMAX * FF, 3.0f));
    for (uint32_t ff : {1280u, 1024u})
      for (uint32_t sgs : {8u, 16u, 20u, 24u, 36u})
        for (uint32_t W : TIMED_WIDTHS)
          std::printf("time ds_swiglu_clamp_a8 %u rows, %u sub-groups W=%u: %.2f us\n", ff, sgs, W,
                      time_launch(rt, c, [&](ze::List &L) { launch(L, W, ff, sgs); }));
    io.put(one, randn(rng, WMAX * 2 * FF, 3.0f));
    for (uint32_t ff : {1280u, 1024u})
      for (uint32_t W : TIMED_WIDTHS)
        std::printf("time ds_swiglu_clamp_a8 %u rows of one matrix W=%u: %.2f us\n", ff, W,
                    time_launch(rt, c, [&](ze::List &L) { launch_at(L, W, one, one + ff, 2 * ff, ff, D41_SWIGLU_SGS); }));
    rt.free(g);
    rt.free(u);
    rt.free(one);
    rt.free(ha);
  }

  // ---- the combine ----

  void run_combine() {
    const uint32_t c = 0;
    Io io(rt, c);
    auto *y = io.dev<float>(WMAX * U * E);
    auto *sh = io.dev<float>(WMAX * E);
    auto *w = io.dev<float>(WMAX * U + 16);
    auto *slots = io.dev<int32_t>(WMAX * U + 16);
    auto *nmiss = io.dev<uint32_t>(WMAX);
    auto *out = io.dev<float>(WMAX * E);
    auto *inb = io.dev<float>(WMAX * E);  // the other part's memory on this card, with one card visible
    float *peer_card = rt.ncards() > 1 ? static_cast<float *>(rt.alloc_device(1, WMAX * E * 4, 4096)) : inb;
    auto *cpu = static_cast<float *>(rt.alloc_host(WMAX * E * 4, 4096));
    auto *peer_host = static_cast<float *>(rt.alloc_host(WMAX * E * 4, 4096));
    Io io1(rt, rt.ncards() > 1 ? 1 : 0);

    auto launch = [&](ze::List &L, uint32_t W, const void *pcpu, const void *ppeer, uint32_t sgs) {
      L.launch(lib.get(c, "deepseek41", "ds_moe_combine"), {y, w, slots, sh, pcpu, nmiss, out, ppeer}, W, 1, 1, sgs * 16);
    };
    const float qnan = std::nanf("");
    for (uint32_t W : WIDTHS)
      for (uint32_t pattern = 0; pattern < 4; pattern++) {
        std::vector<float> vy = randn(rng, W * U * E), vs = randn(rng, W * E), vc = randn(rng, W * E), vw(W * U);
        std::vector<int32_t> vslot(W * U);
        std::vector<uint32_t> vmiss(W);
        std::uniform_real_distribution<float> uni(0.0f, 1.0f);
        for (uint32_t r = 0; r < W; r++) {
          const bool padded = pattern == 3 && r == W - 1 && W > 1;
          uint32_t n = 0;
          for (uint32_t k = 0; k < U; k++) {
            const bool cached = !padded && (pattern == 0 || (pattern != 1 && uni(rng) < 0.5f));
            vslot[r * U + k] = cached ? (int32_t)(r * 7 + k) : -1;
            vw[r * U + k] = padded ? 0.0f : uni(rng) * 0.5f;
            n += !cached && !padded;
            // A rank that is not cached is never read, nor a CPU answer of a row with no miss.
            if (!cached) std::fill(&vy[(r * U + k) * E], &vy[(r * U + k + 1) * E], qnan);
          }
          vmiss[r] = n;
          if (!n) std::fill(&vc[r * E], &vc[(r + 1) * E], qnan);
        }
        io.put(y, vy);
        io.put(sh, vs);
        io.put(w, vw);
        io.put(slots, vslot);
        io.put(nmiss, vmiss);
        std::memcpy(cpu, vc.data(), vc.size() * 4);
        // In the fixed order of the kernel, in float.
        auto expect = [&](bool with_cpu) {
          std::vector<float> o(W * E);
          for (uint32_t r = 0; r < W; r++)
            for (size_t i = 0; i < E; i++) {
              float acc = 0;
              for (uint32_t k = 0; k < U; k++)
                if (vslot[r * U + k] >= 0) acc += vw[r * U + k] * vy[(r * U + k) * E + i];
              acc += vs[r * E + i];
              if (with_cpu && vmiss[r]) acc += vc[r * E + i];
              o[r * E + i] = acc;
            }
          return o;
        };
        // No peer, no CPU answer: the other card's part.
        launch(io.L, W, none, none, D41_CMB_SGS(W));
        io.L.synchronize();
        auto got = io.get(out, W * E);
        auto want = expect(false);
        ck.rel("ds_moe_combine cached + shared", got.data(), want.data(), want.size(), LIMIT);
        // The posting card, through each way of sending.
        want = expect(true);
        std::memset(peer_host, 0xAB, W * E * 4);
        launch(io.L, W, cpu, peer_host, D41_CMB_SGS(W));
        io.L.synchronize();
        got = io.get(out, W * E);
        ck.rel("ds_moe_combine with the CPU's answer", got.data(), want.data(), want.size(), LIMIT);
        ck.exact("ds_moe_combine sends to host memory", peer_host, got.data(), W * E * 4);
        launch(io.L, W, cpu, peer_card, D41_CMB_SGS(W));
        io.L.synchronize();
        auto gotp = io1.get(peer_card, W * E);
        ck.exact("ds_moe_combine sends to the other card", gotp.data(), got.data(), W * E * 4);
      }

    // The CPU's answer written after the list has started and after the card has read the old answer's lines, 0-1000 us
    // late. As a recorded list meets it: ds_wait, then the combine. Then the same reads with no ds_wait's fences and
    // with a kernel's end or none between them. No end and no fence is the control: it shows a stale line can appear.
    {
      std::vector<float> vy(WMAX * U * E, 0.0f), vs(WMAX * E, 0.0f), vw(WMAX * U, 0.0f);
      std::vector<int32_t> vslot(WMAX * U, -1);
      const std::vector<uint32_t> vmiss(WMAX, 6);
      io.put(y, vy);
      io.put(sh, vs);
      io.put(w, vw);
      io.put(slots, vslot);
      io.put(nmiss, vmiss);
      auto *read_first = io.dev<float>(WMAX * E);
      auto *want = io.dev<uint32_t>(1);
      auto *stuck = io.dev<uint32_t>(1);
      auto *start = io.dev<uint64_t>(1);
      auto *flag = static_cast<volatile uint32_t *>(rt.alloc_host(64, 64));
      auto *stamp = static_cast<volatile uint64_t *>(rt.alloc_host(2 * sizeof(uint64_t), 64));
      *flag = 0;
      const std::vector<float> answers[2] = {randn(rng, WMAX * E), randn(rng, WMAX * E)};
      uint32_t step = 0;
      enum Between { by_wait, by_hold, none_between };
      auto late = [&](uint32_t W, Between how, bool fence) {
        constexpr uint32_t ROUNDS = 100, HOLD_TICKS = 4800000;  // 2 ms at the top clock, past the longest delay
        const double delays_us[] = {0, 1, 10, 100, 1000};
        const char *kernel = fence ? "ds_moe_combine" : "ds_moe_combine_no_fence";
        const uint32_t lx = D41_CMB_SGS(W) * 16;
        ze::List L(rt, c, ze::List::regular, false, false);
        uint64_t *pstamp = const_cast<uint64_t *>(stamp);
        if (how == none_between) {
          const uint32_t on = fence;
          L.launch(lib.get(c, "deepseek41", "ds_moe_combine_read_twice"),
                   {y, w, slots, sh, cpu, nmiss, out, HOLD_TICKS, pstamp, on}, W, 1, 1, lx);
        } else {
          L.launch(lib.get(c, "deepseek41", "ds_moe_combine"), {y, w, slots, sh, cpu, nmiss, read_first, none}, W, 1, 1,
                   lx);
          if (how == by_wait) {
            Wait wt;
            wt.flag = const_cast<const uint32_t *>(flag), wt.want = want, wt.stuck = stuck, wt.start = start, wt.stamp = pstamp;
            launch_wait(L, lib, wt);
          } else {
            L.launch(lib.get(c, "deepseek41", "ds_hold"), {HOLD_TICKS, pstamp}, 1, 1, 1, 1);
          }
          L.launch(lib.get(c, "deepseek41", kernel), {y, w, slots, sh, cpu, nmiss, out, none}, W, 1, 1, lx);
        }
        L.close();
        uint32_t wrong = 0;
        for (uint32_t round = 0; round < ROUNDS; round++) {
          step++;
          const float *fresh = answers[step & 1].data();
          std::memcpy(cpu, answers[(step + 1) & 1].data(), W * E * 4);
          std::atomic_thread_fence(std::memory_order_seq_cst);
          const uint32_t no = 0;
          const uint64_t not_yet = 0;
          io.put(want, &step, 1);
          io.put(stuck, &no, 1);
          io.put(start, &not_yet, 1);
          stamp[0] = 0;
          L.execute();
          const auto t0 = std::chrono::steady_clock::now();
          while (stamp[0] == 0 && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(5)) {
          }
          const auto t1 = std::chrono::steady_clock::now();
          while (std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t1).count() <
                 delays_us[round % 5]) {
          }
          std::memcpy(cpu, fresh, W * E * 4);
          std::atomic_thread_fence(std::memory_order_seq_cst);
          *flag = step;
          L.synchronize();
          auto got = io.get(out, W * E);
          wrong += std::memcmp(got.data(), fresh, W * E * 4) != 0 || io.get(stuck, 1)[0] != 0;
        }
        const char *names[] = {"ds_wait", "a kernel's end", "nothing"};
        std::printf("late CPU answer, W=%u, %s between the reads, %s: %u wrong of %u\n", W, names[how],
                    fence ? "fence" : "no fence", wrong, ROUNDS);
        return (double)wrong;
      };
      for (uint32_t W : WIDTHS) {
        ck.note("ds_moe_combine late CPU answer behind ds_wait", late(W, by_wait, true), 0);
        ck.note("ds_moe_combine late CPU answer, fence in the launch", late(W, none_between, true), 0);
        ck.note("no stale line without the fence: the test cannot fail", late(W, none_between, false) == 0, 0);
        late(W, by_wait, false);
        late(W, by_hold, false);
      }
      rt.free(read_first);
      rt.free(want);
      rt.free(stuck);
      rt.free(start);
      rt.free(const_cast<uint32_t *>(flag));
      rt.free(const_cast<uint64_t *>(stamp));
    }

    // Times: 3 of 6 ranks cached, a miss in every row, as at about 50% hits.
    std::vector<float> vy = randn(rng, WMAX * U * E), vs = randn(rng, WMAX * E), vc = randn(rng, WMAX * E);
    std::vector<float> vw(WMAX * U, 0.2f);
    std::vector<int32_t> vslot(WMAX * U);
    for (size_t a = 0; a < vslot.size(); a++) vslot[a] = a % 2 ? -1 : (int32_t)a;
    std::vector<uint32_t> vmiss(WMAX, 3);
    io.put(y, vy);
    io.put(sh, vs);
    io.put(w, vw);
    io.put(slots, vslot);
    io.put(nmiss, vmiss);
    std::memcpy(cpu, vc.data(), vc.size() * 4);
    for (uint32_t W : WIDTHS) {
      for (uint32_t sgs : {4u, 8u, 16u}) {
        std::printf("time ds_moe_combine posting, into the other card, %u sub-groups W=%u: %.2f us\n", sgs, W,
                    time_launch(rt, c, [&](ze::List &L) { launch(L, W, cpu, peer_card, sgs); }));
        std::printf("time ds_moe_combine not posting, into the other card, %u sub-groups W=%u: %.2f us\n", sgs, W,
                    time_launch(rt, c, [&](ze::List &L) { launch(L, W, none, peer_card, sgs); }));
      }
      const uint32_t sgs = D41_CMB_SGS(W);
      const double ptop = time_launch(rt, c, [&](ze::List &L) { launch(L, W, cpu, peer_card, sgs); });
      const double phost = time_launch(rt, c, [&](ze::List &L) { launch(L, W, cpu, peer_host, sgs); });
      const double thost = time_launch(rt, c, [&](ze::List &L) { launch(L, W, none, peer_host, sgs); });
      const double tcard = time_launch(rt, c, [&](ze::List &L) { launch(L, W, none, peer_card, sgs); });
      const double tnone = time_launch(rt, c, [&](ze::List &L) { launch(L, W, none, none, sgs); });
      std::printf("time ds_moe_combine posting, into the other card W=%u: %.2f us\n", W, ptop);
      std::printf("time ds_moe_combine posting, into host memory W=%u: %.2f us\n", W, phost);
      std::printf("time ds_moe_combine not posting, into host memory W=%u: %.2f us\n", W, thost);
      std::printf("time ds_moe_combine not posting, into the other card W=%u: %.2f us\n", W, tcard);
      std::printf("time ds_moe_combine not posting, no send W=%u: %.2f us\n", W, tnone);
      // Bytes sent over the time the send adds; a store under 64 bytes across PCIe is far slower.
      if (W == WMAX) {
        ck.note("ds_moe_combine stores into the other card under 8 GB/s", (double)(W * E * 4) / (tcard - tnone) / 1e3 < 8, 0);
        ck.note("ds_moe_combine stores into host memory under 8 GB/s", (double)(W * E * 4) / (thost - tnone) / 1e3 < 8, 0);
      }
    }
  }

  static float ref_s(float x) { return std::sqrt(x > 20.0f ? x : std::log1p(std::exp(x))); }
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
  for (Scores k : {Scores::random, Scores::clustered, Scores::narrow, Scores::flat, Scores::paired, Scores::large,
                   Scores::huge, Scores::tiny, Scores::lowest, Scores::zero_s, Scores::same_rows, Scores::biased,
                   Scores::one_lane})
    for (uint32_t W : WIDTHS) {
      b.run_route_case(k, W, 104, 52);      // two classes
      b.run_route_case(k, W, 384, 192);     // every expert cached
      b.run_route_case(k, W, 0, 0);         // every expert missed
      b.run_route_case(k, W, 80, 1000);     // one class, whole experts
    }
  b.run_padded();
  b.run_seq_order();
  std::printf("near ties swapped by rounding: %u; weights left unchecked by a swap into rank 6: %u\n", b.near_ties,
              b.weights_skipped);
  log_clocks("before timing");
  b.bench_router();
  b.bench_job_stores();
  b.run_swiglu();
  b.run_combine();
  log_clocks("after timing");
  return b.ck.report();
}
