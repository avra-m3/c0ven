// Level 3 (docs/TESTING.md): deepseek41's routed experts on the cards (kernels/quant/moe.cl with -DMOE_CLASSES), with
// ds_swiglu_clamp_a8 and ds_moe_combine, on random weights at the model's shapes. Each of the file's type pairs runs at
// 1, 2, 3, 4, 6 and 8 rows, its experts cut into part A on one card and part B on the other, parts mixed in one launch,
// and whole. Limits: 1e-5 of the largest |reference| against ggml's weights in double, a8 scales and codes within their
// rounding and that, as matvec_test's. The CPU's answer arrives late, through ds_wait and the combine's fence. Then each
// kernel's time and read rate, swept by lanes a row, rows a sub-group and the codebook's place.
#include <atomic>
#include <cmath>
#include <map>
#include <memory>
#include <thread>

#include "cpu/experts.h"
#include "cpu/features.h"
#include "ds41_kernel_test.h"
#include "ggml.h"
#include "load/gguf.h"
#include "load/runs.h"
#include "models/deepseek41/layout.h"
#include "moe_tile.h"
#include "quant.h"
#include "util/parallel.h"

using namespace engine;
using namespace ds41test;

namespace {

constexpr uint32_t E = D41_EMBD, FF = D41_FF, U = D41_USED, PA = D41_PART_A, PB = D41_PART_B;
constexpr uint32_t GU_BPR = E / 256, X_STRIDE = E + E / 8;
constexpr uint32_t N_EXPERTS = 8, N_CACHED = 6;  // experts a test case routes to; the rest are the CPU's
constexpr double LIMIT = 1e-5, COMBINE_LIMIT = 1e-6;
constexpr double CPU_LIMIT = 5e-2;  // the CPU rounds an activation in blocks of 256, the cards in units of 32
constexpr uint32_t SG = 16;
constexpr uint64_t COLD_BYTES = 40ull << 20;  // past the card's 18 MiB cache
const void *const none = nullptr;

uint64_t align256(uint64_t x) { return (x + 255) & ~255ull; }

std::string lower(std::string s) {
  for (auto &c : s) c = (char)std::tolower(c);
  return s;
}

// Local size of a work-group of the class kernels (MOE_WG_ROWS rows), by moe.cl's CL_RS.
uint32_t local_size(uint32_t rows_a_sub_group, uint32_t lanes) {
  return SG * MOE_WG_ROWS / std::max(rows_a_sub_group, SG / lanes);
}

template <class F>
void in_threads(uint64_t n, const F &f) {
  const uint32_t nt = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
  std::vector<std::thread> ts;
  for (uint32_t t = 0; t < nt; t++) ts.emplace_back([&, t] { f(n * t / nt, n * (t + 1) / nt); });
  for (auto &t : ts) t.join();
}

// An expert in the file's layout.
struct Expert {
  std::vector<uint8_t> gate, up, down;
};

Expert make_expert(uint32_t gt, uint32_t dt, std::mt19937 &rng) {
  return {test::random_blocks(gt, (uint64_t)FF * GU_BPR, rng), test::random_blocks(gt, (uint64_t)FF * GU_BPR, rng),
          test::random_blocks(dt, (uint64_t)E * (FF / 256), rng)};
}

// The class of parts of rows rows, its first slot at base.
D41Class make_class(uint32_t gt, uint32_t dt, uint32_t rows, uint64_t base) {
  const uint64_t g = align256(runs_bytes(gt, (uint64_t)rows * GU_BPR));
  const uint64_t d = align256(runs_bytes(dt, (uint64_t)E * (rows / 256)));
  return {base, 2 * g + d, g, 2 * g, rows, {}};
}

// The expert's rows r0 .. r0 + rows - 1 of gate and up and the same columns of down, into a slot of class c.
void fill_slot(uint8_t *s, const Expert &e, uint32_t gt, uint32_t dt, const D41Class &c, uint32_t r0) {
  const uint64_t rb = (uint64_t)GU_BPR * quant_block_bytes(gt), db = (uint64_t)(FF / 256) * quant_block_bytes(dt);
  to_runs(gt, {{e.gate.data(), rb, r0, c.rows, 0, GU_BPR}}, s);
  to_runs(gt, {{e.up.data(), rb, r0, c.rows, 0, GU_BPR}}, s + c.up_off);
  to_runs(dt, {{e.down.data(), db, 0, E, r0 / 256, c.rows / 256}}, s + c.down_off);
}

double a8_value(const uint8_t *row, uint32_t K, uint32_t i) {
  float d;
  std::memcpy(&d, row + K + 4 * (i / 32), 4);
  return (double)d * (int8_t)row[i];
}

double swiglu(double g, double u) {
  const double gc = std::min(g, (double)D41_SWIGLU_LIMIT);
  const double uc = std::clamp(u, -(double)D41_SWIGLU_LIMIT, (double)D41_SWIGLU_LIMIT);
  return gc / (1.0 + std::exp(-gc)) * uc;
}

// One launch's assignments: rows W, each routed to U distinct experts, ids[a] and w[a] for a = row * U + rank.
struct Route {
  uint32_t W = 0;
  std::vector<uint32_t> ids;
  std::vector<float> w;
  std::vector<float> x;      // [W][E]
  std::vector<uint8_t> xa8;  // [W][X_STRIDE]
  std::vector<double> xd;    // the a8 rows' values
};

Route make_route(uint32_t W, std::mt19937 &rng) {
  Route r;
  r.W = W;
  r.ids.resize(W * U);
  r.w.resize(W * U);
  r.x = randn(rng, W * E);
  r.xd.resize(W * E);
  std::uniform_real_distribution<float> uni(0.05f, 0.5f);
  for (uint32_t t = 0; t < W; t++) {
    const auto a = a8_row(r.x.data() + t * E, E);
    r.xa8.insert(r.xa8.end(), a.begin(), a.end());
    for (uint32_t i = 0; i < E; i++) r.xd[t * E + i] = a8_value(a.data(), E, i);
    std::vector<uint32_t> order(N_EXPERTS);
    for (uint32_t e = 0; e < N_EXPERTS; e++) order[e] = e;
    // Expert 0 in every row, so a group fills up at every width.
    std::shuffle(order.begin() + 1, order.end(), rng);
    std::shuffle(order.begin(), order.begin() + U, rng);
    for (uint32_t k = 0; k < U; k++) {
      r.ids[t * U + k] = order[k];
      r.w[t * U + k] = uni(rng);
    }
  }
  return r;
}

// Grouped by expert in order of first use, cached experts only; place gives an expert's class and slot on this card.
template <class Place>
std::vector<D41Group> make_groups(const Route &r, const Place &place) {
  std::vector<D41Group> g;
  std::map<uint32_t, size_t> of;
  for (uint32_t a = 0; a < r.ids.size(); a++) {
    const uint32_t e = r.ids[a];
    if (e >= N_CACHED) continue;
    auto it = of.find(e);
    if (it == of.end()) {
      D41Group d;
      std::memset(&d, 0, sizeof d);
      const auto [cls, slot] = place(e);
      d.cls = cls;
      d.slot = (int32_t)slot;
      it = of.emplace(e, g.size()).first;
      g.push_back(d);
    }
    D41Group &d = g[it->second];
    d.asg[d.n++] = a;
  }
  return g;
}

struct Bench {
  ze::Runtime &rt;
  ze::Library &lib;
  Pool pool;
  CpuExperts cpu;
  Checker ck;
  std::mt19937 rng{20261102};
  const bool two = rt.ncards() > 1;
  uint32_t step = 0;

  Bench(ze::Runtime &r, ze::Library &l) : rt(r), lib(l), pool(8), cpu(pool) {}

  uint32_t card_of(uint32_t c) const { return two ? c : 0; }

  // A card's buffers for one type pair.
  struct Card {
    Io io;
    uint8_t *grids, *cache, *x, *ha;
    D41Class *classes;
    D41Group *groups;
    uint32_t *ngroups;
    float *y, *g_s, *u_s, *y_s, *out;
    uint8_t *ha_s, *shared;
    float *w;
    int32_t *slots;
    uint32_t *nmiss;
    Card(ze::Runtime &rt, uint32_t card) : io(rt, card) {}
  };

  static const char *name(uint32_t t) { return ggml_type_label(t); }

  void product_checks(uint32_t gt, uint32_t dt) {
    std::printf("---- gate and up %s, down %s\n", name(gt), name(dt));
    std::vector<Expert> ex;
    for (uint32_t e = 0; e < N_EXPERTS; e++) ex.push_back(make_expert(gt, dt, rng));
    const size_t ha_stride_max = PA + PA / 8;
    const uint32_t WMAX = STEP_MAX_ROWS;

    // Each card: part-A slots, then part-B slots. Experts 0-2 hold part A on card 0, the others on card 1.
    const uint32_t per_class = N_CACHED / 2;
    D41Class ca = make_class(gt, dt, PA, 0);
    D41Class cb = make_class(gt, dt, PB, per_class * ca.slot_bytes);
    const uint64_t cache_bytes = per_class * (ca.slot_bytes + cb.slot_bytes);
    D41Class table[2] = {ca, cb};
    auto part_on = [&](uint32_t c, uint32_t e) { return (e < per_class) == (c == 0) ? 0u : 1u; };
    auto place_on = [&](uint32_t c) {
      return [=, &part_on](uint32_t e) {
        return std::pair<uint32_t, uint32_t>(part_on(c, e), e < per_class ? e : e - per_class);
      };
    };

    // The shared expert: the larger part on the card that does not post, card 1 here.
    std::vector<uint8_t> sdown = test::random_blocks(QT_Q5_K, (uint64_t)E * (FF / 256), rng);
    const uint32_t ff_s[2] = {PB, PA}, k0_s[2] = {PA, 0};

    std::vector<std::unique_ptr<Card>> cards;
    for (uint32_t c = 0; c < 2; c++) {
      auto cd = std::make_unique<Card>(rt, card_of(c));
      Io &io = cd->io;
      cd->grids = io.dev<uint8_t>(QGRID_BYTES);
      io.put(cd->grids, codebooks().data(), QGRID_BYTES);
      cd->cache = io.dev<uint8_t>(cache_bytes);
      std::vector<uint8_t> host(cache_bytes);
      for (uint32_t e = 0; e < N_CACHED; e++) {
        const auto [cls, slot] = place_on(c)(e);
        const D41Class &t = table[cls];
        fill_slot(host.data() + t.base + slot * t.slot_bytes, ex[e], gt, dt, t, cls ? PA : 0);
      }
      io.put(cd->cache, host.data(), host.size());
      cd->classes = io.dev<D41Class>(2);
      io.put(cd->classes, table, 2);
      cd->x = io.dev<uint8_t>(WMAX * X_STRIDE);
      cd->ha = io.dev<uint8_t>(WMAX * U * ha_stride_max);
      cd->groups = io.dev<D41Group>(WMAX * U);
      cd->ngroups = io.dev<uint32_t>(1);
      cd->y = io.dev<float>(WMAX * U * E);
      cd->g_s = io.dev<float>(WMAX * ff_s[c]);
      cd->u_s = io.dev<float>(WMAX * ff_s[c]);
      cd->y_s = io.dev<float>(WMAX * E);
      cd->out = io.dev<float>(WMAX * E);
      cd->ha_s = io.dev<uint8_t>(WMAX * (ff_s[c] + ff_s[c] / 8) + 64);
      const uint32_t sbpr = ff_s[c] / 256;
      std::vector<uint8_t> sruns(runs_bytes(QT_Q5_K, (uint64_t)E * sbpr));
      to_runs(QT_Q5_K, {{sdown.data(), (uint64_t)(FF / 256) * quant_block_bytes(QT_Q5_K), 0, E, k0_s[c] / 256, sbpr}},
              sruns.data());
      cd->shared = io.dev<uint8_t>(sruns.size());
      io.put(cd->shared, sruns);
      cd->w = io.dev<float>(WMAX * U + 16);
      cd->slots = io.dev<int32_t>(WMAX * U + 16);
      cd->nmiss = io.dev<uint32_t>(WMAX);
      cards.push_back(std::move(cd));
    }
    // Whole experts on card 0 (2,304 rows, one class).
    D41Class cw = make_class(gt, dt, FF, 0);
    Io &io0 = cards[0]->io;
    auto *whole = io0.dev<uint8_t>(N_CACHED * cw.slot_bytes);
    {
      std::vector<uint8_t> host(N_CACHED * cw.slot_bytes);
      for (uint32_t e = 0; e < N_CACHED; e++) fill_slot(host.data() + e * cw.slot_bytes, ex[e], gt, dt, cw, 0);
      io0.put(whole, host.data(), host.size());
    }
    auto *wclass = io0.dev<D41Class>(1);
    io0.put(wclass, &cw, 1);
    const uint32_t ws = FF + FF / 8;
    auto *wha = io0.dev<uint8_t>(WMAX * U * ws);
    auto *wy = io0.dev<float>(WMAX * U * E);
    auto *wg = io0.dev<D41Group>(WMAX * U);
    auto *wn = io0.dev<uint32_t>(1);

    auto *cpu_ans = static_cast<float *>(rt.alloc_host(WMAX * E * 4, 4096));
    auto *flag = static_cast<uint32_t *>(rt.alloc_host(64, 64));
    auto *want = io0.dev<uint32_t>(1);
    auto *stuck = io0.dev<uint32_t>(1);
    auto *start = io0.dev<uint64_t>(1);
    auto *stamp = static_cast<uint64_t *>(rt.alloc_host(16, 16));

    const ExpertShape shape{gt, dt, E, FF, D41_SWIGLU_LIMIT};
    const std::string missing = missing_cpu_features(ENGINE_CPU_FEATURES);
    if (!missing.empty()) std::printf("the CPU lacks %s: its answer not checked\n", missing.c_str());

    auto gu_traits = ggml_get_type_traits((ggml_type)gt);
    auto d_traits = ggml_get_type_traits((ggml_type)dt);
    const uint32_t gbytes = quant_block_bytes(gt) * GU_BPR, dbytes = quant_block_bytes(dt) * (FF / 256);

    for (uint32_t W : WIDTHS) {
      const std::string gmod = "deepseek41_moe_gate_up_" + lower(name(gt)) + "_g" + std::to_string(W);
      const std::string dmod = "deepseek41_moe_down_" + lower(name(dt)) + "_g" + std::to_string(W);
      const uint32_t lxg = local_size(D41_GU_ROWS(W), D41_GU_LANES(gt, W)), lxd = local_size(D41_DOWN_ROWS, D41_DOWN_LANES(dt));
      Route r = make_route(W, rng);
      const uint32_t nasg = W * U;

      // Gate and up by the reference: hw[a][r], whole, in double, for cached assignments.
      std::vector<double> hw((size_t)nasg * FF);
      size_t clamped = 0;
      for (uint32_t a = 0; a < nasg; a++) {
        if (r.ids[a] >= N_CACHED) continue;
        const Expert &e = ex[r.ids[a]];
        const double *xd = r.xd.data() + (a / U) * E;
        std::atomic<size_t> over{0};
        in_threads(FF, [&](uint64_t r0, uint64_t r1) {
          std::vector<float> g(E), u(E);
          for (uint64_t i = r0; i < r1; i++) {
            gu_traits->to_float(&e.gate[i * gbytes], g.data(), E);
            gu_traits->to_float(&e.up[i * gbytes], u.data(), E);
            double gv = 0, uv = 0;
            for (uint32_t j = 0; j < E; j++) {
              gv += (double)g[j] * xd[j];
              uv += (double)u[j] * xd[j];
            }
            hw[(size_t)a * FF + i] = swiglu(gv, uv);
            over += gv > D41_SWIGLU_LIMIT || std::fabs(uv) > D41_SWIGLU_LIMIT;
          }
        });
        clamped += over;
      }

      // Each card's launches with parts mixed; then a8 and y against the reference.
      std::vector<double> y_ref((size_t)nasg * E, 0.0);
      std::vector<std::vector<uint8_t>> ha_host(2);
      std::vector<std::vector<float>> y_host(2);
      size_t largest = 0;
      for (uint32_t c = 0; c < 2; c++) {
        Card &cd = *cards[c];
        Io &io = cd.io;
        const uint32_t cn = card_of(c);
        const std::vector<D41Group> groups = make_groups(r, place_on(c));
        const uint32_t ng = (uint32_t)groups.size();
        for (const D41Group &g : groups) largest = std::max<size_t>(largest, g.n);
        io.put(cd.x, r.xa8);
        if (ng) io.put(cd.groups, groups);
        io.put(cd.ngroups, &ng, 1);
        std::vector<uint8_t> blank((size_t)nasg * ha_stride_max, 0xAB);
        io.put(cd.ha, blank);
        std::vector<float> nans((size_t)nasg * E, std::nanf(""));
        io.put(cd.y, nans);
        const uint32_t per = U, xs = X_STRIDE, hs = (uint32_t)ha_stride_max, K = E, rows = E;
        io.L.launch(lib.get(cn, gmod, "moe_gate_up_c"),
                    {cd.cache, cd.classes, cd.groups, cd.ngroups, cd.x, K, xs, per, cd.ha, hs, cd.grids}, PA / MOE_WG_ROWS,
                    nasg, 1, lxg);
        io.L.synchronize();
        io.L.launch(lib.get(cn, dmod, "moe_down_c"),
                    {cd.cache, cd.classes, cd.groups, cd.ngroups, rows, cd.ha, hs, cd.y, cd.grids}, E / MOE_WG_ROWS, nasg,
                    1, lxd);
        io.L.synchronize();
        ha_host[c] = io.get(cd.ha, (size_t)nasg * ha_stride_max);
        y_host[c] = io.get(cd.y, (size_t)nasg * E);

        for (const D41Group &g : groups) {
          const D41Class &t = table[g.cls];
          const uint32_t j0 = g.cls ? PA : 0, K_ = t.rows;
          for (uint32_t i = 0; i < g.n; i++) {
            const uint32_t a = g.asg[i];
            const uint8_t *row = ha_host[c].data() + (size_t)a * ha_stride_max;
            check_a8(g.cls ? "a8 h, part B" : "a8 h, part A", row, &hw[(size_t)a * FF + j0], K_);
            // Nothing past the row's codes and scales.
            for (size_t b = K_ + K_ / 8; b < ha_stride_max; b++)
              if (row[b] != 0xAB) ck.note("a8 h stores past its part", 1, 0);
            // down . h, with this card's a8 h.
            const Expert &e = ex[r.ids[a]];
            std::vector<double> yr(E);
            std::vector<double> hv(K_);
            for (uint32_t j = 0; j < K_; j++) hv[j] = a8_value(row, K_, j);
            in_threads(E, [&](uint64_t r0, uint64_t r1) {
              std::vector<float> d(FF);
              for (uint64_t ri = r0; ri < r1; ri++) {
                d_traits->to_float(&e.down[ri * dbytes], d.data(), FF);
                double acc = 0;
                for (uint32_t j = 0; j < K_; j++) acc += (double)d[j0 + j] * hv[j];
                yr[ri] = acc;
              }
            });
            for (uint32_t ri = 0; ri < E; ri++) y_ref[(size_t)a * E + ri] += yr[ri];
            ck.rel(g.cls ? "down, part B" : "down, part A", y_host[c].data() + (size_t)a * E, yr.data(), E, LIMIT);
          }
        }
        // Nothing stored for an assignment of no group.
        for (uint32_t a = 0; a < nasg; a++)
          if (r.ids[a] >= N_CACHED)
            for (uint32_t i = 0; i < E; i++)
              if (!std::isnan(y_host[c][(size_t)a * E + i])) {
                ck.note("down stores for a missed expert", 1, 0);
                break;
              }
      }

      // Whole experts: the a8 rows are the parts' bytes, down the parts' sum.
      {
        const std::vector<D41Group> groups =
            make_groups(r, [](uint32_t e) { return std::pair<uint32_t, uint32_t>(0, e); });
        const uint32_t ng = (uint32_t)groups.size();
        if (ng) io0.put(wg, groups);
        io0.put(wn, &ng, 1);
        const uint32_t per = U, xs = X_STRIDE, K = E, rows = E;
        const uint8_t *x0 = cards[0]->x, *g0 = cards[0]->grids;
        io0.L.launch(lib.get(0, gmod, "moe_gate_up_c"), {whole, wclass, wg, wn, x0, K, xs, per, wha, ws, g0},
                     FF / MOE_WG_ROWS, nasg, 1, lxg);
        io0.L.synchronize();
        io0.L.launch(lib.get(0, dmod, "moe_down_c"), {whole, wclass, wg, wn, rows, wha, ws, wy, g0}, E / MOE_WG_ROWS, nasg, 1,
                     lxd);
        io0.L.synchronize();
        const auto ha_w = io0.get(wha, (size_t)nasg * ws);
        const auto y_w = io0.get(wy, (size_t)nasg * E);
        for (const D41Group &g : groups)
          for (uint32_t i = 0; i < g.n; i++) {
            const uint32_t a = g.asg[i], e = r.ids[a];
            const uint32_t ca_ = e < per_class ? 0 : 1, cb_ = 1 - ca_;  // the card holding each part
            const uint8_t *row = ha_w.data() + (size_t)a * ws;
            const uint8_t *pa = ha_host[ca_].data() + (size_t)a * ha_stride_max;
            const uint8_t *pb = ha_host[cb_].data() + (size_t)a * ha_stride_max;
            check_a8("a8 h, whole", row, &hw[(size_t)a * FF], FF);
            ck.exact("a8 h, whole, codes against the parts'", row, pa, PA);
            ck.exact("a8 h, whole, codes against the parts'", row + PA, pb, PB);
            ck.exact("a8 h, whole, scales against the parts'", row + FF, pa + PA, PA / 8);
            ck.exact("a8 h, whole, scales against the parts'", row + FF + PA / 8, pb + PB, PB / 8);
            std::vector<double> sum(E);
            for (uint32_t ri = 0; ri < E; ri++)
              sum[ri] = (double)y_host[ca_][(size_t)a * E + ri] + y_host[cb_][(size_t)a * E + ri];
            ck.rel("down, whole against the parts' sum", y_w.data() + (size_t)a * E, sum.data(), E, LIMIT);
            ck.rel("down, whole against the reference", y_w.data() + (size_t)a * E, &y_ref[(size_t)a * E], E, LIMIT);
          }
      }

      // The shared expert's SwiGLU on each card, stored as a8, and its part of down.
      std::vector<float> shared_ref[2];
      std::vector<float> y_s_host[2];
      for (uint32_t c = 0; c < 2; c++) {
        Card &cd = *cards[c];
        const uint32_t ff = ff_s[c], cn = card_of(c), stride = ff + ff / 8;
        const auto gv = randn(rng, W * ff, 4.0f), uv = randn(rng, W * ff, 4.0f);
        cd.io.put(cd.g_s, gv);
        cd.io.put(cd.u_s, uv);
        cd.io.L.launch(lib.get(cn, "deepseek41", "ds_swiglu_clamp_a8"), {cd.g_s, cd.u_s, ff, ff, cd.ha_s}, W, 1, 1,
                       D41_SWIGLU_SGS * 16);
        cd.io.L.synchronize();
        const auto ha_s = cd.io.get(cd.ha_s, (size_t)W * stride);
        std::vector<double> rs(ff);
        for (uint32_t t = 0; t < W; t++) {
          for (uint32_t i = 0; i < ff; i++) rs[i] = swiglu(gv[t * ff + i], uv[t * ff + i]);
          check_a8("ds_swiglu_clamp_a8", ha_s.data() + (size_t)t * stride, rs.data(), ff);
        }
        const uint32_t srows = E, bpr = ff / 256, nt = W, ys = E;
        cd.io.L.launch(lib.get(cn, "matvec_q5_k_l4", "matvec"),
                       {cd.shared, srows, bpr, cd.ha_s, ff, stride, nt, cd.y_s, ys, cd.grids}, E / 16, 1, 1, 64);
        cd.io.L.synchronize();
        y_s_host[c] = cd.io.get(cd.y_s, (size_t)W * E);
        // Against the reference.
        std::vector<double> yr((size_t)W * E);
        auto s_traits = ggml_get_type_traits((ggml_type)QT_Q5_K);
        const uint32_t sb = (FF / 256) * quant_block_bytes(QT_Q5_K);
        in_threads(E, [&](uint64_t r0, uint64_t r1) {
          std::vector<float> d(FF);
          for (uint64_t ri = r0; ri < r1; ri++) {
            s_traits->to_float(&sdown[ri * sb], d.data(), FF);
            for (uint32_t t = 0; t < W; t++) {
              double acc = 0;
              const uint8_t *row = ha_s.data() + (size_t)t * stride;
              for (uint32_t j = 0; j < ff; j++) acc += (double)d[k0_s[c] + j] * a8_value(row, ff, j);
              yr[(size_t)t * E + ri] = acc;
            }
          }
        });
        ck.rel("shared down", y_s_host[c].data(), yr.data(), yr.size(), LIMIT);
        shared_ref[c].assign(yr.begin(), yr.end());
      }

      // The CPU's answer, then each card's partial sum. Card 0 posts: its list waits on the flag, and the answer comes
      // after that wait has started.
      std::vector<float> answer((size_t)W * E, 0.0f);
      std::vector<uint32_t> nmiss(W, 0);
      if (missing.empty()) {
        std::vector<CpuExperts::Job> jobs;
        for (uint32_t a = 0; a < nasg; a++)
          if (r.ids[a] >= N_CACHED) {
            const Expert &e = ex[r.ids[a]];
            jobs.push_back({{e.gate.data(), e.up.data(), e.down.data(), 1}, a / U, r.w[a]});
            nmiss[a / U]++;
          }
        if (!jobs.empty()) cpu.compute_many(shape, jobs, W, r.x.data(), E, answer.data(), E);
      }
      std::vector<float> outs[2];
      std::vector<int32_t> slots(nasg);
      for (uint32_t a = 0; a < nasg; a++) slots[a] = r.ids[a] < N_CACHED ? (int32_t)r.ids[a] : -1;
      for (uint32_t c = 0; c < 2; c++) {
        Card &cd = *cards[c];
        cd.io.put(cd.w, r.w);
        cd.io.put(cd.slots, slots);
        cd.io.put(cd.nmiss, nmiss);
      }
      {
        Card &cd = *cards[1];
        const uint32_t cn = card_of(1);
        cd.io.L.launch(lib.get(cn, "deepseek41", "ds_moe_combine"),
                       {cd.y, cd.w, cd.slots, cd.y_s, none, cd.nmiss, cd.out, none}, W, 1, 1, D41_CMB_SGS(W) * 16);
        cd.io.L.synchronize();
        outs[1] = cd.io.get(cd.out, (size_t)W * E);
      }
      {
        Card &cd = *cards[0];
        ze::List L(rt, card_of(0), ze::List::regular, false, false);
        Wait wt;
        wt.flag = flag, wt.want = want, wt.stuck = stuck, wt.start = start, wt.stamp = stamp;
        launch_wait(L, lib, wt);
        L.launch(lib.get(card_of(0), "deepseek41", "ds_moe_combine"),
                 {cd.y, cd.w, cd.slots, cd.y_s, cpu_ans, cd.nmiss, cd.out, none}, W, 1, 1, D41_CMB_SGS(W) * 16);
        L.close();
        step++;
        const uint32_t no = 0;
        const uint64_t not_yet = 0;
        cd.io.put(want, &step, 1);
        cd.io.put(stuck, &no, 1);
        cd.io.put(start, &not_yet, 1);
        stamp[0] = 0;
        std::memset(cpu_ans, 0xFF, (size_t)W * E * 4);  // a stale answer reads as NaN
        std::atomic_thread_fence(std::memory_order_seq_cst);
        L.execute();
        const auto t0 = std::chrono::steady_clock::now();
        while (stamp[0] == 0 && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(5)) {
        }
        const auto t1 = std::chrono::steady_clock::now();
        while (std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t1).count() < 100.0) {
        }
        std::memcpy(cpu_ans, answer.data(), (size_t)W * E * 4);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        *flag = step;
        L.synchronize();
        ck.note("ds_wait gave up on the CPU's answer", (double)cd.io.get(stuck, 1)[0], 0);
        outs[0] = cd.io.get(cd.out, (size_t)W * E);
      }
      // Each card's sum, in the kernel's order in float; the two together against the reference.
      std::vector<float> want0((size_t)W * E), want1((size_t)W * E);
      std::vector<double> total((size_t)W * E);
      for (uint32_t t = 0; t < W; t++)
        for (uint32_t i = 0; i < E; i++) {
          float a0 = 0, a1 = 0;
          double td = 0;
          for (uint32_t k = 0; k < U; k++) {
            const size_t a = (size_t)t * U + k;
            if (slots[a] < 0) continue;
            a0 += r.w[a] * y_host[0][a * E + i];
            a1 += r.w[a] * y_host[1][a * E + i];
            td += (double)r.w[a] * y_ref[a * E + i];
          }
          a0 += y_s_host[0][(size_t)t * E + i];
          a1 += y_s_host[1][(size_t)t * E + i];
          td += (double)shared_ref[0][(size_t)t * E + i] + shared_ref[1][(size_t)t * E + i];
          if (nmiss[t]) {
            a0 += answer[(size_t)t * E + i];
            td += answer[(size_t)t * E + i];
          }
          want0[(size_t)t * E + i] = a0;
          want1[(size_t)t * E + i] = a1;
          total[(size_t)t * E + i] = td;
        }
      ck.rel("ds_moe_combine with the CPU's answer", outs[0].data(), want0.data(), want0.size(), COMBINE_LIMIT);
      ck.rel("ds_moe_combine, the other card", outs[1].data(), want1.data(), want1.size(), COMBINE_LIMIT);
      std::vector<double> both((size_t)W * E);
      for (size_t i = 0; i < both.size(); i++) both[i] = (double)outs[0][i] + outs[1][i];
      ck.rel("both cards' sums against the reference", both.data(), total.data(), total.size(), LIMIT);

      // The cached experts through the CPU, for comparison: its rounding of activations differs, so a wider limit.
      if (missing.empty()) {
        std::vector<CpuExperts::Job> jobs;
        std::vector<float> by_cpu((size_t)W * E);
        for (uint32_t a = 0; a < nasg; a++)
          if (r.ids[a] < N_CACHED) {
            const Expert &e = ex[r.ids[a]];
            jobs.push_back({{e.gate.data(), e.up.data(), e.down.data(), 1}, a / U, r.w[a]});
          }
        cpu.compute_many(shape, jobs, W, r.x.data(), E, by_cpu.data(), E);
        std::vector<double> on_cards((size_t)W * E, 0.0);
        for (uint32_t a = 0; a < nasg; a++)
          if (r.ids[a] < N_CACHED)
            for (uint32_t i = 0; i < E; i++)
              on_cards[(size_t)(a / U) * E + i] +=
                  (double)r.w[a] * ((double)y_host[0][(size_t)a * E + i] + y_host[1][(size_t)a * E + i]);
        ck.rel("cards' experts against the CPU's", on_cards.data(), by_cpu.data(), by_cpu.size(), CPU_LIMIT);
      }
      std::printf("%s and %s, W=%u: %zu rows' values past the SwiGLU limit, largest group %zu, %u misses in row 0\n",
                  name(gt), name(dt), W, clamped, largest, nmiss[0]);
    }

    for (void *b : std::vector<void *>{whole, wclass, wha, wy, wg, wn, cpu_ans, flag, want, stuck, start, stamp}) rt.free(b);
    for (auto &cd : cards)
      for (void *b : std::vector<void *>{cd->grids, cd->cache, cd->x, cd->ha, cd->classes, cd->groups, cd->ngroups, cd->y,
                                         cd->g_s, cd->u_s, cd->y_s, cd->out, cd->ha_s, cd->shared, cd->w, cd->slots,
                                         cd->nmiss})
        rt.free(b);
  }

  // a8 of a row of K values against the reference: the scale, and each value within half a step and the limit.
  void check_a8(const char *what, const uint8_t *row, const double *ref, uint32_t K) {
    double m = 0;
    for (uint32_t i = 0; i < K; i++) m = std::max(m, std::fabs(ref[i]));
    double es = 0, ev = 0;
    for (uint32_t u = 0; u < K / 32; u++) {
      double amax = 0;
      for (uint32_t i = 0; i < 32; i++) amax = std::max(amax, std::fabs(ref[32 * u + i]));
      const double wd = amax / 127.0;
      float d;
      std::memcpy(&d, row + K + 4 * u, 4);
      es = std::max(es, std::fabs((double)d - wd));
      for (uint32_t i = 0; i < 32; i++) {
        const double got = (double)d * (int8_t)row[32 * u + i];
        ev = std::max(ev, std::fabs(got - ref[32 * u + i]) - 0.5 * (double)d);
      }
    }
    const std::string scale = std::string(what) + ", scale", value = std::string(what) + ", value";
    ck.note(keep(scale), m > 0 ? es / m : es, LIMIT);
    ck.note(keep(value), m > 0 ? std::max(ev, 0.0) / m : ev, LIMIT);
  }

  // Checker keeps names by pointer.
  std::map<std::string, std::string> names;
  const char *keep(const std::string &s) { return names.emplace(s, s).first->second.c_str(); }

  // ---- times ----

  // Gate and up of rows rows x 5,120, or down of 5,120 rows x K (rows here), group size 1, by lanes a row, the codebook's
  // place and rows a sub-group; copies in turn when cold.
  void sweep(uint32_t type, bool down, uint32_t part) {
    Io io(rt, 0);
    auto *grids = io.dev<uint8_t>(QGRID_BYTES);
    io.put(grids, codebooks().data(), QGRID_BYTES);
    const uint32_t rows = down ? E : part, K = down ? part : E, bpr = K / 256;
    const uint64_t n = (uint64_t)rows * bpr;
    const uint64_t mat = align256(runs_bytes(type, n)), slot = down ? mat : 2 * mat;
    const uint64_t bytes = (down ? 1 : 2) * runs_bytes(type, n);
    const uint32_t copies = (uint32_t)std::max<uint64_t>(2, (COLD_BYTES + slot - 1) / slot);
    std::vector<uint8_t> host(slot);
    const uint64_t rb = (uint64_t)bpr * quant_block_bytes(type);
    const auto b1 = test::random_blocks(type, n, rng);
    to_runs(type, {{b1.data(), rb, 0, rows, 0, bpr}}, host.data());
    if (!down) {
      const auto b2 = test::random_blocks(type, n, rng);
      to_runs(type, {{b2.data(), rb, 0, rows, 0, bpr}}, host.data() + mat);
    }
    auto *cache = io.dev<uint8_t>(copies * slot);
    for (uint32_t i = 0; i < copies; i++) io.put(cache + i * slot, host.data(), host.size());
    const D41Class cl{0, slot, down ? 0 : mat, 0, part, {}};
    auto *classes = io.dev<D41Class>(1);
    io.put(classes, &cl, 1);
    std::vector<D41Group> groups(copies);
    for (uint32_t i = 0; i < copies; i++) {
      std::memset(&groups[i], 0, sizeof(D41Group));
      groups[i].slot = (int32_t)i;
      groups[i].n = 1;
    }
    auto *dg = io.dev<D41Group>(copies);
    io.put(dg, groups);
    auto *dn = io.dev<uint32_t>(1);
    const uint32_t one = 1;
    io.put(dn, &one, 1);
    const auto xf = randn(rng, K);
    const auto a = a8_row(xf.data(), K);
    auto *act = io.dev<uint8_t>(a.size());
    io.put(act, a);
    const uint32_t stride = (uint32_t)a.size();
    auto *ha = io.dev<uint8_t>(FF + FF / 8);
    auto *y = io.dev<float>(E);
    const uint32_t hs = FF + FF / 8, per = 1;
    const struct {
      uint32_t lanes, rs;
      const char *suffix;
    } shapes[] = {{16, 4, "local"}, {16, 4, "constant"}, {16, 2, "r2"},         {16, 2, "r2_constant"},
                  {16, 1, "r1"},    {8, 4, "local"},     {8, 4, "constant"},    {8, 2, "r2"},
                  {8, 2, "r2_constant"}, {4, 4, "local"}, {4, 4, "constant"}};
    for (const auto &sh : shapes) {
      const std::string mod = "deepseek41_moe_" + lower(name(type)) + "_l" + std::to_string(sh.lanes) + "_" + sh.suffix;
      const uint32_t lx = local_size(sh.rs, sh.lanes);
      auto rec = [&](ze::List &L, uint32_t i) {
        const D41Group *g = dg + i;
        if (down)
          L.launch(lib.get(0, mod, "moe_down_c"), {cache, classes, g, dn, rows, act, stride, y, grids},
                   rows / MOE_WG_ROWS, 1, 1, lx);
        else
          L.launch(lib.get(0, mod, "moe_gate_up_c"), {cache, classes, g, dn, act, K, stride, per, ha, hs, grids},
                   rows / MOE_WG_ROWS, 1, 1, lx);
      };
      const double hot = time_launch(rt, 0, [&](ze::List &L) { rec(L, 0); });
      uint32_t at = 0;
      const double cold = time_launch(rt, 0, [&](ze::List &L) { rec(L, at++ % copies); });
      std::printf("sweep %s %s %u x %u lanes %2u %-8s: hot %.2f us %.1f GB/s, cold (%u copies) %.2f us %.1f GB/s\n",
                  down ? "down" : "gate+up", name(type), rows, K, sh.lanes, sh.suffix, hot, (double)bytes / hot / 1e3,
                  copies, cold, (double)bytes / cold / 1e3);
    }
    for (void *b : std::vector<void *>{grids, cache, classes, dg, dn, act, ha, y}) rt.free(b);
  }

  // Each kernel's time at W rows with the modules a model list would use: one group of each class, a group of W
  // assignments, and 6 W expert parts half of each class, every weight cold.
  void kept_times(uint32_t type, bool down) {
    Io io(rt, 0);
    auto *grids = io.dev<uint8_t>(QGRID_BYTES);
    io.put(grids, codebooks().data(), QGRID_BYTES);
    const uint32_t SLOTS = 24;
    // Only the kernel's own matrices count.
    const uint32_t dt = type, gt = type;
    D41Class ca = make_class(gt, dt, PA, 0);
    D41Class cb = make_class(gt, dt, PB, SLOTS * ca.slot_bytes);
    const uint64_t total = SLOTS * (ca.slot_bytes + cb.slot_bytes);
    const Expert e = make_expert(gt, dt, rng);
    auto *cache = io.dev<uint8_t>(total);
    {
      std::vector<uint8_t> sa(ca.slot_bytes), sb(cb.slot_bytes);
      fill_slot(sa.data(), e, gt, dt, ca, 0);
      fill_slot(sb.data(), e, gt, dt, cb, PA);
      for (uint32_t i = 0; i < SLOTS; i++) {
        io.put(cache + ca.base + i * ca.slot_bytes, sa.data(), sa.size());
        io.put(cache + cb.base + i * cb.slot_bytes, sb.data(), sb.size());
      }
    }
    D41Class table[2] = {ca, cb};
    auto *classes = io.dev<D41Class>(2);
    io.put(classes, table, 2);
    const uint32_t hs = PA + PA / 8, WMAX = STEP_MAX_ROWS;
    auto *ha = io.dev<uint8_t>(WMAX * U * hs);
    auto *y = io.dev<float>(WMAX * U * E);
    auto *x = io.dev<uint8_t>(WMAX * X_STRIDE);
    for (uint32_t t = 0; t < WMAX; t++) {
      const auto xf = randn(rng, E);
      io.put(x + t * X_STRIDE, a8_row(xf.data(), E));
    }
    // h as a8 for down, in class A's stride: even assignments take part A's K, odd part B's.
    for (uint32_t a = 0; a < WMAX * U; a++) {
      const auto hf = randn(rng, a % 2 ? PB : PA);
      io.put(ha + (size_t)a * hs, a8_row(hf.data(), hf.size()));
    }
    auto *dn = io.dev<uint32_t>(4);
    const uint32_t one = 1;
    log_clocks("before the kept shape's times");
    const uint32_t per = U, K = E, rows = E, xs = X_STRIDE;
    for (uint32_t W : TIMED_WIDTHS) {
      const std::string mod = std::string("deepseek41_moe_") + (down ? "down_" : "gate_up_") + lower(name(type)) + "_g" +
                              std::to_string(W);
      const uint32_t lx = down ? local_size(D41_DOWN_ROWS, D41_DOWN_LANES(type)) : local_size(D41_GU_ROWS(W), D41_GU_LANES(type, W));
      const uint32_t grid = down ? E / MOE_WG_ROWS : PA / MOE_WG_ROWS;
      auto *dg = io.dev<D41Group>(2 * SLOTS * U * WMAX);
      auto rec = [&](ze::List &L, const D41Group *g, const uint32_t *nn, uint32_t gy) {
        if (down)
          L.launch(lib.get(0, mod, "moe_down_c"), {cache, classes, g, nn, rows, ha, hs, y, grids}, grid, gy, 1, lx);
        else
          L.launch(lib.get(0, mod, "moe_gate_up_c"), {cache, classes, g, nn, x, K, xs, per, ha, hs, grids}, grid, gy, 1, lx);
      };
      const uint64_t abytes = (down ? 1 : 2) * runs_bytes(type, down ? (uint64_t)E * 5 : (uint64_t)PA * GU_BPR);
      const uint64_t bbytes = (down ? 1 : 2) * runs_bytes(type, down ? (uint64_t)E * 4 : (uint64_t)PB * GU_BPR);
      io.put(dn, &one, 1);
      // One group of one assignment, of each class, slots in turn.
      for (uint32_t cls : {0u, 1u}) {
        std::vector<D41Group> gs(SLOTS);
        for (uint32_t i = 0; i < SLOTS; i++) {
          std::memset(&gs[i], 0, sizeof(D41Group));
          gs[i].slot = (int32_t)i, gs[i].n = 1, gs[i].cls = cls, gs[i].asg[0] = cls;
        }
        io.put(dg, gs);
        uint32_t at = 0;
        const double us = time_launch(rt, 0, [&](ze::List &L) { rec(L, dg + at++ % SLOTS, dn, W * U); });
        std::printf("time %s %s part %c, one group of 1, W=%u list: %.2f us %.1f GB/s\n", down ? "down" : "gate+up",
                    name(type), cls ? 'B' : 'A', W, us, (double)(cls ? bbytes : abytes) / us / 1e3);
      }
      // One group of W assignments, part A.
      {
        std::vector<D41Group> gs(SLOTS);
        for (uint32_t i = 0; i < SLOTS; i++) {
          std::memset(&gs[i], 0, sizeof(D41Group));
          gs[i].slot = (int32_t)i, gs[i].n = W;
          for (uint32_t t = 0; t < W; t++) gs[i].asg[t] = t * U;
        }
        io.put(dg, gs);
        uint32_t at = 0;
        const double us = time_launch(rt, 0, [&](ze::List &L) { rec(L, dg + at++ % SLOTS, dn, W * U); });
        std::printf("time %s %s part A, one group of %u, W=%u: %.2f us\n", down ? "down" : "gate+up", name(type), W, W, us);
      }
      // 6 W parts, half of each class, one assignment each, slots in turn.
      {
        const uint32_t ng = W * U;
        const uint32_t rounds = 2 * SLOTS;
        std::vector<D41Group> gs(rounds * ng);
        for (uint32_t i = 0; i < rounds * ng; i++) {
          std::memset(&gs[i], 0, sizeof(D41Group));
          gs[i].cls = i & 1;
          gs[i].slot = (int32_t)((i / 2) % SLOTS);
          gs[i].n = 1;
          gs[i].asg[0] = i % ng;
        }
        io.put(dg, gs);
        io.put(dn, &ng, 1);
        uint32_t at = 0;
        const double us = time_launch(rt, 0, [&](ze::List &L) { rec(L, dg + (at++ % rounds) * ng, dn, ng); });
        const uint64_t mixb = (uint64_t)(ng / 2) * (abytes + bbytes);
        std::printf("time %s %s %u parts mixed, W=%u: %.2f us, %.2f us a part, %.1f GB/s\n", down ? "down" : "gate+up",
                    name(type), ng, W, us, us / ng, (double)mixb / us / 1e3);
        // No group at all: what a launch of empty groups costs.
        const uint32_t zero = 0;
        io.put(dn, &zero, 1);
        const double empty = time_launch(rt, 0, [&](ze::List &L) { rec(L, dg, dn, ng); });
        std::printf("time %s %s no groups, %u in the grid, W=%u: %.2f us\n", down ? "down" : "gate+up", name(type), ng, W,
                    empty);
      }
      rt.free(dg);
    }
    log_clocks("after the kept shape's times");
    for (void *b : std::vector<void *>{grids, cache, classes, ha, y, x, dn}) rt.free(b);
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
  // The file's pairs of gate and up with down that cover each routed type (engine/models/deepseek41/tensors.tsv).
  const uint32_t pairs[][2] = {{QT_IQ1_S, QT_IQ1_S}, {QT_IQ2_XXS, QT_IQ1_M}, {QT_IQ1_M, QT_IQ1_S}, {QT_IQ1_S, QT_IQ2_S},
                               {QT_IQ1_S, QT_IQ2_XXS}};
  if (std::getenv("DS41_MOE_TIMES_ONLY") == nullptr)
    for (const auto &p : pairs) b.product_checks(p[0], p[1]);
  log_clocks("before the sweep");
  for (uint32_t t : {QT_IQ1_S, QT_IQ2_XXS, QT_IQ1_M})
    for (uint32_t part : {FF, PA, PB}) b.sweep(t, false, part);
  for (uint32_t t : {QT_IQ1_S, QT_IQ1_M, QT_IQ2_XXS, QT_IQ2_S})
    for (uint32_t part : {FF, PA, PB}) b.sweep(t, true, part);
  log_clocks("after the sweep");
  for (uint32_t t : {QT_IQ1_S, QT_IQ2_XXS, QT_IQ1_M}) b.kept_times(t, false);
  for (uint32_t t : {QT_IQ1_S, QT_IQ1_M, QT_IQ2_XXS, QT_IQ2_S}) b.kept_times(t, true);
  return b.ck.report();
}
