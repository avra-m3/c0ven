// Level 3 (docs/TESTING.md): deepseek41's hyper-connection kernels (kernels/models/deepseek41/layer.cl) on random data
// at the model's shapes, against ref_ops and the pinned ggml's hyper-connection operations: each kernel alone at 1-8
// rows, a chain of all four, and each one's time at 1, 2, 4 and 8 rows with every card visible. Limits: 1e-5 of the
// largest |reference|, comb 1e-6 absolute, a8 and bf16 exact from the kernel's own x. No model is needed.
#include <cmath>

#include "ds41_kernel_test.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "models/deepseek41/layout.h"
#include "models/deepseek41/ref_ops.h"

using namespace engine;
using namespace ds41test;
namespace ref = engine::deepseek41::ref;

namespace {

constexpr size_t E = D41_EMBD, D = D41_HC_DIM;
constexpr uint32_t WMAX = STEP_MAX_ROWS;
constexpr uint32_t HC_LOCAL = D41_HC_SGS * 16;
constexpr uint32_t KSPLIT = D41_HC_KSPLIT;
const void *const none = nullptr;

// ggml's graph over one output tensor; gives it back as floats.
std::vector<float> compute(ggml_context *ctx, ggml_tensor *r) {
  ggml_cgraph *gf = ggml_new_graph(ctx);
  ggml_build_forward_expand(gf, r);
  EXPECT(ggml_graph_compute_with_ctx(ctx, gf, 1) == GGML_STATUS_SUCCESS, "ggml's graph failed");
  return std::vector<float>((const float *)r->data, (const float *)r->data + ggml_nelements(r));
}

ggml_tensor *tensor(ggml_context *ctx, const float *v, int64_t a, int64_t b = 1, int64_t c = 1) {
  ggml_tensor *t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, a, b, c);
  std::memcpy(t->data, v, ggml_nbytes(t));
  return t;
}

// ggml_dsv4_hc_comb over W rows of m: rows of D41_HC * D41_HC, in HcMix.comb's order.
std::vector<float> ggml_comb(const float *m, uint32_t W, const float *scale, const float *base) {
  ggml_init_params ip = {64u << 20, nullptr, false};
  ggml_context *ctx = ggml_init(ip);
  ggml_tensor *r = ggml_dsv4_hc_comb(ctx, tensor(ctx, m, D41_MIX, W), tensor(ctx, scale, 3),
                                     tensor(ctx, base, D41_MIX), D41_HC_EPS, D41_SINKHORN);
  std::vector<float> y = compute(ctx, r);
  ggml_free(ctx);
  return y;
}

// ggml_dsv4_hc_post, then ggml_dsv4_hc_pre of its result: X' [W][4][E] and x before the norm [W][E].
void ggml_post_pre(const float *X, const float *y, const std::vector<ref::HcMix> &mix, uint32_t W, std::vector<float> &Xo,
                   std::vector<float> &x) {
  std::vector<float> post, pre, comb;
  for (uint32_t r = 0; r < W; r++) {
    post.insert(post.end(), mix[r].post, mix[r].post + D41_HC);
    pre.insert(pre.end(), mix[r].pre, mix[r].pre + D41_HC);
    comb.insert(comb.end(), mix[r].comb, mix[r].comb + D41_HC * D41_HC);
  }
  ggml_init_params ip = {256u << 20, nullptr, false};
  ggml_context *ctx = ggml_init(ip);
  ggml_tensor *o = ggml_dsv4_hc_post(ctx, tensor(ctx, y, E, W), tensor(ctx, X, E, D41_HC, W),
                                     tensor(ctx, post.data(), D41_HC, W), tensor(ctx, comb.data(), D41_HC, D41_HC, W));
  ggml_tensor *p = ggml_dsv4_hc_pre(ctx, o, tensor(ctx, pre.data(), D41_HC, W));
  ggml_cgraph *gf = ggml_new_graph(ctx);
  ggml_build_forward_expand(gf, o);
  ggml_build_forward_expand(gf, p);
  EXPECT(ggml_graph_compute_with_ctx(ctx, gf, 1) == GGML_STATUS_SUCCESS, "ggml's graph failed");
  Xo.assign((const float *)o->data, (const float *)o->data + ggml_nelements(o));
  x.assign((const float *)p->data, (const float *)p->data + ggml_nelements(p));
  ggml_free(ctx);
}

D41HcMix to_mix(const ref::HcMix &m) {
  D41HcMix o;
  std::memcpy(&o, &m, sizeof o);
  return o;
}

struct Bench {
  ze::Runtime &rt;
  ze::Library &lib;
  Io &io;
  std::mt19937 rng{41};
  Checker ck;
  // Card memory for 8 rows.
  float *X, *Xo, *ss, *ss2, *fn, *m, *scale, *base, *norm, *e, *y0, *y1, *xf;
  D41HcMix *mix;
  uint32_t *count;
  uint8_t *xa;
  uint16_t *xb;
  // Host memory: the MissJobs and a partial sum, as at one row.
  D41MissJob *jobs;
  float *y1h;
  std::vector<float> hX, hfn, hnorm, he;

  Bench(ze::Runtime &r, ze::Library &l, Io &i) : rt(r), lib(l), io(i) {
    X = io.dev<float>(WMAX * D);
    Xo = io.dev<float>(WMAX * D);
    ss = io.dev<float>(WMAX);
    ss2 = io.dev<float>(WMAX);
    fn = io.dev<float>(D41_MIX * D);
    m = io.dev<float>((size_t)WMAX * D41_MIX);
    scale = io.dev<float>(3);
    base = io.dev<float>(D41_MIX);
    norm = io.dev<float>(E);
    e = io.dev<float>(WMAX * E);
    y0 = io.dev<float>(WMAX * E);
    y1 = io.dev<float>(WMAX * E);
    xf = io.dev<float>(WMAX * E);
    mix = io.dev<D41HcMix>(WMAX);
    count = io.dev<uint32_t>(1);
    io.put(count, std::vector<uint32_t>(1, 0));
    xa = io.dev<uint8_t>((size_t)WMAX * D41_A8_ROW);
    xb = io.dev<uint16_t>(WMAX * E);
    jobs = static_cast<D41MissJob *>(rt.alloc_host(WMAX * sizeof(D41MissJob)));
    y1h = static_cast<float *>(rt.alloc_host(WMAX * E * sizeof(float), 4096));
    hX = randn(rng, WMAX * D);
    hfn = randn(rng, D41_MIX * D, 0.02f);
    hnorm = randn(rng, E, 0.5f);
    he = randn(rng, WMAX * E, 0.3f);
    io.put(X, hX);
    io.put(fn, hfn);
    io.put(norm, hnorm);
    io.put(e, he);
  }
  ze_kernel_handle_t k(const char *name) { return lib.get(io.card, "deepseek41", name); }
  void run(const std::function<void(ze::List &)> &rec) {
    rec(io.L);
    io.L.synchronize();
  }

  void launch_init(ze::List &L, uint32_t W, const void *pxa, const void *pxb, const void *pxf) {
    L.launch(k("ds_hc_init"), {e, norm, Xo, ss2, pxa, pxb, pxf}, W, 1, 1, HC_LOCAL);
  }
  void launch_mix(ze::List &L, uint32_t W, uint32_t ks) {
    L.launch(k("ds_hc_mix"), {fn, X, ss, W, m, scale, base, mix, count}, D41_MIX, 1, 1, 16 * ks);
  }
  void launch_post(ze::List &L, uint32_t W, const void *py1, const void *pnorm, const void *pxa, const void *pxb,
                   const void *pxf, const void *pjob) {
    const uint32_t y_host = py1 == y1h;
    L.launch(k("ds_hc_post_norm"), {X, y0, py1, y_host, mix, pnorm, Xo, ss2, pxa, pxb, pxf, pjob}, W, 1, 1, HC_LOCAL);
  }

  // Check x (f32 from the kernel) against ref_ops, and its a8 and bf16 forms exactly against it.
  void check_x(const char *fname, const char *aname, const char *bname, const std::vector<float> &want, uint32_t W,
               bool a8, bool bf16) {
    auto gx = io.get(xf, W * E);
    ck.rel(fname, gx.data(), want.data(), W * E, 1e-5);
    if (a8) {
      auto ga = io.get(xa, (size_t)W * D41_A8_ROW);
      for (uint32_t r = 0; r < W; r++) {
        auto wa = a8_row(gx.data() + r * E, E);
        ck.exact(aname, ga.data() + (size_t)r * D41_A8_ROW, wa.data(), wa.size());
      }
    }
    if (bf16) {
      auto gb = io.get(xb, W * E);
      std::vector<uint16_t> wb(W * E);
      for (size_t i = 0; i < wb.size(); i++) wb[i] = ggml_fp32_to_bf16(gx[i]).bits;
      ck.exact(bname, gb.data(), wb.data(), wb.size() * 2);
    }
  }

  void test_init() {
    const std::vector<float> pre0 = {1, 0, 0, 0};
    for (uint32_t W : WIDTHS) {
      run([&](ze::List &L) { launch_init(L, W, xa, xb, xf); });
      auto gX = io.get(Xo, W * D);
      auto gs = io.get(ss2, W);
      std::vector<float> wX(W * D), wx(W * E), ws(W);
      for (uint32_t r = 0; r < W; r++) {
        for (uint32_t c = 0; c < D41_HC; c++) std::copy(&he[r * E], &he[r * E] + E, &wX[r * D + c * E]);
        ref::hc_collapse_norm(&wX[r * D], pre0.data(), hnorm.data(), &wx[r * E]);
        ws[r] = (float)sum_squares(&wX[r * D], D);
      }
      ck.exact("ds_hc_init X", gX.data(), wX.data(), wX.size() * 4);
      ck.rel("ds_hc_init ss", gs.data(), ws.data(), W, 1e-5);
      check_x("ds_hc_init x", "ds_hc_init x a8", "ds_hc_init x bf16", wx, W, true, true);
    }
  }

  // One launch gives the raw mixes and, from them, the mixes: both against ref_ops, the comb against ggml too.
  void test_mix() {
    struct Case {
      float s0, s1, s2, bspread;
    };
    std::vector<float> ws(WMAX);
    for (uint32_t r = 0; r < WMAX; r++) ws[r] = (float)sum_squares(&hX[r * D], D);
    io.put(ss, ws);
    std::vector<float> wm(WMAX * D41_MIX);
    for (uint32_t r = 0; r < WMAX; r++) ref::hc_project(&hX[r * D], hfn.data(), &wm[(size_t)r * D41_MIX]);
    for (const Case &c : {Case{1, 1, 1, 0}, Case{0.7f, 1.3f, 1.5f, 5}, Case{2, 2, 20, 0.1f}})
      for (uint32_t ks : {4u, 8u, 16u, 32u})
        for (uint32_t W : WIDTHS) {
          std::uniform_real_distribution<float> ub(-c.bspread, c.bspread);
          std::vector<float> hb(D41_MIX), hs = {c.s0, c.s1, c.s2};
          for (float &v : hb) v = c.bspread > 0 ? ub(rng) : 0.0f;
          io.put(base, hb);
          io.put(scale, hs);
          run([&](ze::List &L) { launch_mix(L, W, ks); });
          auto g = io.get(m, (size_t)W * D41_MIX);
          ck.rel("ds_hc_mix m", g.data(), wm.data(), g.size(), 1e-5);
          // The mixes from the raw mixes the card made: a scale of 20 turns the raw mixes' last bits into 1e-6 of comb.
          auto gm = io.get(mix, W);
          auto gg = ggml_comb(g.data(), W, hs.data(), hb.data());
          for (uint32_t r = 0; r < W; r++) {
            const ref::HcMix w = ref::hc_split(&g[(size_t)r * D41_MIX], hs.data(), hb.data());
            ck.abs("ds_hc_mix comb against ref_ops", gm[r].comb, w.comb, 16, 1e-6);
            ck.abs("ds_hc_mix comb against ggml", gm[r].comb, &gg[(size_t)r * 16], 16, 1e-6);
            ck.rel("ds_hc_mix pre", gm[r].pre, w.pre, 4, 1e-5);
            ck.rel("ds_hc_mix post", gm[r].post, w.post, 4, 1e-5);
          }
          EXPECT(io.get(count, 1)[0] == 0, "ds_hc_mix left its count at %u", io.get(count, 1)[0]);
        }
  }

  void test_post_norm() {
    for (uint32_t W : WIDTHS) {
      auto py0 = randn(rng, W * E, 0.7f), py1 = randn(rng, W * E, 0.7f);
      std::vector<float> y(W * E);
      for (size_t i = 0; i < y.size(); i++) y[i] = py0[i] + py1[i];
      std::vector<ref::HcMix> rm(W);
      std::vector<D41HcMix> mm(W);
      float hs[3] = {1, 1, 1}, hb[D41_MIX] = {};
      for (uint32_t r = 0; r < W; r++) {
        auto rmx = randn(rng, D41_MIX, 2.0f);
        rm[r] = ref::hc_split(rmx.data(), hs, hb);
        mm[r] = to_mix(rm[r]);
      }
      io.put(y0, py0);
      io.put(y1, py1);
      io.put(mix, mm);
      std::memcpy(y1h, py1.data(), py1.size() * 4);
      std::vector<float> wX(W * D), wx(W * E), ws(W), raw(W * E);
      for (uint32_t r = 0; r < W; r++) {
        ref::hc_post(&hX[r * D], &y[r * E], rm[r], &wX[r * D]);
        ref::hc_collapse_norm(&wX[r * D], rm[r].pre, hnorm.data(), &wx[r * E]);
        ws[r] = (float)sum_squares(&wX[r * D], D);
      }
      std::vector<float> oX, ox;
      ggml_post_pre(hX.data(), y.data(), rm, W, oX, ox);

      // The attention sublayer's end: a8, f32 and the MissJob's x in host memory.
      std::memset(jobs, 0, W * sizeof(D41MissJob));
      run([&](ze::List &L) { launch_post(L, W, y1, norm, xa, none, xf, jobs); });
      auto gX = io.get(Xo, W * D);
      auto gs = io.get(ss2, W);
      ck.rel("ds_hc_post_norm X'", gX.data(), wX.data(), gX.size(), 1e-5);
      ck.rel("ds_hc_post_norm X' against ggml", gX.data(), oX.data(), gX.size(), 1e-5);
      ck.rel("ds_hc_post_norm ss", gs.data(), ws.data(), W, 1e-5);
      check_x("ds_hc_post_norm x", "ds_hc_post_norm x a8", "", wx, W, true, false);
      auto gx = io.get(xf, W * E);
      for (uint32_t r = 0; r < W; r++) ck.exact("ds_hc_post_norm MissJob x", (const float *)jobs[r].x, &gx[r * E], E * 4);
      for (uint32_t r = 0; r < W; r++) {
        std::vector<float> nx(E);
        ref::rms_norm(&ox[r * E], hnorm.data(), E, nx.data());
        ck.rel("ds_hc_post_norm x against ggml", &gx[r * E], nx.data(), E, 1e-5);
      }

      // The partials swapped give the same bits, so both cards hold the same streams.
      run([&](ze::List &L) { L.launch(k("ds_hc_post_norm"), {X, y1, y0, 0u, mix, norm, Xo, ss2, xa, none, xf, none}, W, 1, 1, HC_LOCAL); });
      auto sX = io.get(Xo, W * D);
      ck.exact("ds_hc_post_norm partials swapped", sX.data(), gX.data(), gX.size() * 4);

      // The FFN sublayer's end: a8 and bf16.
      run([&](ze::List &L) { launch_post(L, W, y1, norm, xa, xb, xf, none); });
      check_x("ds_hc_post_norm x", "ds_hc_post_norm x a8", "ds_hc_post_norm x bf16", wx, W, true, true);

      // The other partial in host memory, as at one row.
      run([&](ze::List &L) { launch_post(L, W, y1h, norm, xa, xb, xf, none); });
      auto hhX = io.get(Xo, W * D);
      ck.exact("ds_hc_post_norm y1 in host memory", hhX.data(), gX.data(), gX.size() * 4);

      // X only, before an Engram layer: no norm, no x.
      run([&](ze::List &L) { launch_post(L, W, y1, none, none, none, none, none); });
      auto oXo = io.get(Xo, W * D);
      auto oss = io.get(ss2, W);
      ck.exact("ds_hc_post_norm X only", oXo.data(), gX.data(), gX.size() * 4);
      ck.exact("ds_hc_post_norm X only ss", oss.data(), gs.data(), W * 4);

      // One card: y0 holds the whole sum.
      io.put(y0, y);
      run([&](ze::List &L) { launch_post(L, W, none, norm, none, none, xf, none); });
      auto oneX = io.get(Xo, W * D);
      ck.rel("ds_hc_post_norm one card", oneX.data(), wX.data(), oneX.size(), 1e-5);
    }
  }

  // ds_hc_init, then the mixes, Sinkhorn and the post from the streams it wrote, against the same steps in ref_ops.
  void test_chain() {
    auto hs = randn(rng, 3, 0.5f);
    auto hb = randn(rng, D41_MIX, 0.5f);
    io.put(scale, hs);
    io.put(base, hb);
    const uint32_t W = WMAX;
    auto py = randn(rng, W * E, 0.7f);
    io.put(y0, py);
    run([&](ze::List &L) {
      L.launch(k("ds_hc_init"), {e, norm, X, ss, xa, none, xf}, W, 1, 1, HC_LOCAL);
      L.barrier();
      launch_mix(L, W, KSPLIT);
      L.barrier();
      launch_post(L, W, none, norm, xa, xb, xf, none);
    });
    std::vector<float> wX(W * D), wx(W * E), r0(W * D);
    std::vector<ref::HcMix> rm(W);
    for (uint32_t r = 0; r < W; r++) {
      for (uint32_t c = 0; c < D41_HC; c++) std::copy(&he[r * E], &he[r * E] + E, &r0[r * D + c * E]);
      rm[r] = ref::hc_mixes(&r0[r * D], hfn.data(), hs.data(), hb.data());
      ref::hc_post(&r0[r * D], &py[r * E], rm[r], &wX[r * D]);
      ref::hc_collapse_norm(&wX[r * D], rm[r].pre, hnorm.data(), &wx[r * E]);
    }
    auto gX = io.get(Xo, W * D);
    ck.rel("chain X'", gX.data(), wX.data(), gX.size(), 1e-5);
    auto gx = io.get(xf, W * E);
    ck.rel("chain x", gx.data(), wx.data(), gx.size(), 1e-5);
    // The chain leaves X overwritten with the init's streams; later steps need the random ones.
    io.put(X, hX);
  }

  void bench() {
    // Data the kernels can run on: X and y as above, mixes from ref_ops.
    std::vector<float> ws(WMAX);
    for (uint32_t r = 0; r < WMAX; r++) ws[r] = (float)sum_squares(&hX[r * D], D);
    io.put(ss, ws);
    auto hs = randn(rng, 3, 0.5f);
    auto hb = randn(rng, D41_MIX, 0.5f);
    io.put(scale, hs);
    io.put(base, hb);
    auto py = randn(rng, WMAX * E, 0.7f);
    io.put(y0, py);
    io.put(y1, py);
    run([&](ze::List &L) { launch_mix(L, WMAX, KSPLIT); });
    // A first timing warms the card, so that the clock logged next is the one the timings run at.
    time_launch(rt, 0, [&](ze::List &L) { launch_init(L, 1, xa, xb, none); });
    log_clocks("before timing");
    const double fn_gb = (double)D41_MIX * D * 4 / 1e9;
    for (uint32_t W : TIMED_WIDTHS) std::printf("time ds_hc_init W=%u: %.2f us\n", W, time_launch(rt, 0, [&](ze::List &L) { launch_init(L, W, xa, xb, none); }));
    for (uint32_t ks : {4u, 8u, 16u, 32u})
      for (uint32_t W : TIMED_WIDTHS) {
        const double us = time_launch(rt, 0, [&](ze::List &L) { launch_mix(L, W, ks); });
        std::printf("time ds_hc_mix ksplit=%u W=%u: %.2f us, weights at %.0f GB/s (estimate 367)\n", ks, W, us, fn_gb / (us * 1e-6));
      }
    // A step reads a different fn for each of its 80 sublayers, 158 MB, more than the card's cache holds. The launches
    // above reread one, so its weights come from the cache.
    constexpr uint32_t NFN = 80;
    auto *fns = static_cast<float *>(rt.alloc_device(io.card, (size_t)NFN * D41_MIX * D * 4, 4096));
    for (uint32_t i = 0; i < NFN; i++) io.put(fns + (size_t)i * D41_MIX * D, hfn);
    for (uint32_t ks : {4u, 8u, 16u, 32u})
      for (uint32_t W : TIMED_WIDTHS) {
        uint32_t next = 0;
        const double us = time_launch(rt, 0, [&](ze::List &L) {
          float *f = fns + (size_t)(next++ % NFN) * D41_MIX * D;
          L.launch(k("ds_hc_mix"), {f, X, ss, W, m, scale, base, mix, count}, D41_MIX, 1, 1, 16 * ks);
        });
        std::printf("time ds_hc_mix 80 matrices in turn ksplit=%u W=%u: %.2f us, weights at %.0f GB/s\n", ks, W, us, fn_gb / (us * 1e-6));
      }
    rt.free(fns);
    for (uint32_t W : TIMED_WIDTHS) {
      std::printf("time ds_hc_post_norm attention's end W=%u: %.2f us\n", W, time_launch(rt, 0, [&](ze::List &L) { launch_post(L, W, y1, norm, xa, none, xf, jobs); }));
      std::printf("time ds_hc_post_norm FFN's end W=%u: %.2f us\n", W, time_launch(rt, 0, [&](ze::List &L) { launch_post(L, W, y1, norm, xa, xb, none, none); }));
      std::printf("time ds_hc_post_norm X only W=%u: %.2f us\n", W, time_launch(rt, 0, [&](ze::List &L) { launch_post(L, W, y1, none, none, none, none, none); }));
    }
    std::printf("time ds_hc_post_norm y1 in host memory W=1: %.2f us\n", time_launch(rt, 0, [&](ze::List &L) { launch_post(L, 1, y1h, norm, xa, xb, none, none); }));
    log_clocks("after timing");
  }
};

}  // namespace

int main() {
  init_clocks();
  if (test::gpus() == 0) return test::no_gpu();
  ze::Runtime rt;
  std::printf("cards visible: %u\n", rt.ncards());
  ze::Library lib(rt);
  Io io(rt, 0);
  Bench b(rt, lib, io);
  b.test_mix();
  b.test_post_norm();
  b.test_init();
  b.test_chain();
  b.bench();
  return b.ck.report();
}
