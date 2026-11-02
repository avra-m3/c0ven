// Level 3 (docs/TESTING.md): deepseek41's Engram kernels (kernels/models/deepseek41/layer.cl), ds_engram_gate and
// ds_engram_apply, on random data at the model's shapes against ref_ops, and each kernel's time at 1, 2, 4 and 8 rows
// with every card visible. Limit: 1e-5 of the largest |reference|; the cards' streams equal bit for bit; a8 and bf16
// exact from the kernel's own x. With one card visible, or --one-card, both parts of the split run on card 0 and the
// two-card timings skip with a message. No model is needed.
#include <cmath>

#include "ds41_kernel_test.h"
#include "ggml.h"
#include "models/deepseek41/layout.h"
#include "models/deepseek41/ref_ops.h"

using namespace engine;
using namespace ds41test;
namespace ref = engine::deepseek41::ref;

namespace {

constexpr size_t E = D41_EMBD, D = D41_HC_DIM, KV = D41_ENG_KV;
constexpr uint32_t WMAX = STEP_MAX_ROWS;
constexpr uint32_t HC_LOCAL = D41_HC_SGS * 16;
const void *const none = nullptr;

// What the injection reads: the streams, kv and engram_gate's rows, per case.
struct Data {
  std::vector<float> X, kv, w;  // [W][4][E], [W][KV], [4][E]
};

enum class Kind { random, tiny_weights, zero_weights, orthogonal, key_is_stream, key_is_minus_stream, negative_weights };

Data make(std::mt19937 &rng, uint32_t W, Kind kind) {
  Data d;
  d.X = randn(rng, W * D);
  d.kv = randn(rng, W * KV, 0.4f);
  d.w = randn(rng, 4 * E, 0.35f);
  for (float &v : d.w) {
    if (kind == Kind::tiny_weights) v *= 1e-9f;
    if (kind == Kind::zero_weights) v = 0.0f;
    if (kind == Kind::key_is_stream || kind == Kind::key_is_minus_stream) v = 1.0f;
    if (kind == Kind::negative_weights) v = -std::fabs(v);
  }
  for (uint32_t r = 0; r < W; r++)
    for (size_t i = 0; i < D; i++) {
      float &x = d.X[r * D + i], &k = d.kv[r * KV + i];
      if (kind == Kind::orthogonal) {
        // The streams on even dimensions, the keys on odd: every product is 0.
        if (i % 2) x = 0.0f;
        else k = 0.0f;
        d.w[i] = 1.0f;
      }
      if (kind == Kind::key_is_stream) k = x;
      if (kind == Kind::key_is_minus_stream) k = -x;
      if (kind == Kind::negative_weights) k = std::fabs(k), x = std::fabs(x);  // every product at or below 0
    }
  return d;
}

// A card's own memory for an Engram layer.
struct CardMem {
  Io io;
  float *X, *kv, *w, *ss, *norm, *xf;
  uint8_t *xa;
  uint16_t *xb;
  D41HcMix *mix;
  D41EngramSend *own, *in;  // its gates; what the other card writes here
  CardMem(ze::Runtime &rt, uint32_t c) : io(rt, c) {
    X = io.dev<float>(WMAX * D);
    kv = io.dev<float>(WMAX * KV);
    w = io.dev<float>(4 * E);
    ss = io.dev<float>(WMAX);
    norm = io.dev<float>(E);
    xf = io.dev<float>(WMAX * E);
    xa = io.dev<uint8_t>((size_t)WMAX * D41_A8_ROW);
    xb = io.dev<uint16_t>(WMAX * E);
    mix = io.dev<D41HcMix>(WMAX);
    own = io.dev<D41EngramSend>(WMAX);
    in = io.dev<D41EngramSend>(WMAX);
  }
};

struct Part {
  uint32_t c0, c1, v0, v1;
  size_t stride() const { return (c1 - c0) * E + (v1 - v0); }
};

struct Bench {
  ze::Runtime &rt;
  ze::Library &lib;
  std::mt19937 rng{1};
  Checker ck;
  std::vector<CardMem *> cards;
  D41EngramSend *host[2];  // host[i]: what card i reads, written by the other
  bool two_cards;
  Bench(ze::Runtime &r, ze::Library &l, bool two) : rt(r), lib(l), two_cards(two) {
    // The split takes only parameters, so without a second card two sets of buffers on card 0 run it.
    for (uint32_t c : {0u, two ? 1u : 0u}) cards.push_back(new CardMem(rt, c));
    for (auto &h : host) h = static_cast<D41EngramSend *>(rt.alloc_host(WMAX * sizeof(D41EngramSend), 4096));
  }

  void put_part(CardMem &m, const Data &d, const Part &p, uint32_t W, const std::vector<D41HcMix> &mix,
                const std::vector<float> &norm) {
    m.io.put(m.X, d.X);
    std::vector<float> k(W * p.stride());
    for (uint32_t r = 0; r < W; r++) {
      float *o = &k[r * p.stride()];
      std::copy(&d.kv[r * KV + p.c0 * E], &d.kv[r * KV + p.c0 * E] + (p.c1 - p.c0) * E, o);
      std::copy(&d.kv[r * KV + D + p.v0], &d.kv[r * KV + D + p.v0] + (p.v1 - p.v0), o + (p.c1 - p.c0) * E);
    }
    m.io.put(m.kv, k);
    m.io.put(m.w, &d.w[p.c0 * E], (p.c1 - p.c0) * E);
    m.io.put(m.mix, mix);
    m.io.put(m.norm, norm);
  }

  void launch_gate(ze::List &L, CardMem &m, const Part &p, const void *peer, uint32_t W) {
    const uint32_t stride = (uint32_t)p.stride();
    L.launch(lib.get(m.io.card, "deepseek41", "ds_engram_gate"),
             {m.X, m.kv, stride, m.w, p.c0, p.c1, p.v0, p.v1, m.own, peer}, p.c1 - p.c0, W, 1, 256);
  }
  void launch_apply(ze::List &L, CardMem &m, const Part &p, const void *peer, uint32_t W, const void *pxa,
                    const void *pxb, const void *pxf) {
    const uint32_t stride = (uint32_t)p.stride(), peer_host = peer == host[0] || peer == host[1];
    L.launch(lib.get(m.io.card, "deepseek41", "ds_engram_apply"),
             {m.X, m.kv, stride, p.c0, p.c1, p.v0, p.v1, m.own, peer, peer_host, m.mix, m.norm, m.ss, pxa, pxb, pxf}, W,
             1, 1, HC_LOCAL);
  }

  void check_result(CardMem &m, const Data &d, const std::vector<ref::HcMix> &rm,
                    const std::vector<float> &norm, uint32_t W) {
    std::vector<float> wX = d.X, wx(W * E), ws(W), wgate(4 * W);
    for (uint32_t r = 0; r < W; r++) {
      ref::engram_gate(&wX[r * D], &d.kv[r * KV], d.w.data(), &wgate[r * 4]);
      ref::hc_collapse_norm(&wX[r * D], rm[r].pre, norm.data(), &wx[r * E]);
      ws[r] = (float)sum_squares(&wX[r * D], D);
    }
    auto gX = m.io.get(m.X, W * D);
    auto gx = m.io.get(m.xf, W * E);
    auto gs = m.io.get(m.ss, W);
    ck.rel("ds_engram_apply X", gX.data(), wX.data(), gX.size(), 1e-5);
    ck.rel("ds_engram_apply x", gx.data(), wx.data(), gx.size(), 1e-5);
    ck.rel("ds_engram_apply ss", gs.data(), ws.data(), W, 1e-5);
    auto ga = m.io.get(m.xa, (size_t)W * D41_A8_ROW);
    auto gb = m.io.get(m.xb, W * E);
    std::vector<uint16_t> wb(W * E);
    for (size_t i = 0; i < wb.size(); i++) wb[i] = ggml_fp32_to_bf16(gx[i]).bits;
    ck.exact("ds_engram_apply x bf16", gb.data(), wb.data(), wb.size() * 2);
    for (uint32_t r = 0; r < W; r++) {
      auto wa = a8_row(&gx[r * E], E);
      ck.exact("ds_engram_apply x a8", &ga[(size_t)r * D41_A8_ROW], wa.data(), wa.size());
    }
  }

  // Gates as the card holds them: own for its streams.
  void check_gates(CardMem &m, const Part &p, const Data &d, uint32_t W) {
    std::vector<float> wX = d.X, wgate(4 * W);
    for (uint32_t r = 0; r < W; r++) ref::engram_gate(&wX[r * D], &d.kv[r * KV], d.w.data(), &wgate[r * 4]);
    auto own = m.io.get(m.own, W);
    for (uint32_t r = 0; r < W; r++)
      for (uint32_t c = p.c0; c < p.c1; c++) {
        ck.rel("ds_engram_gate gate", &own[r].gate[c][0], &wgate[r * 4 + c], 1, 1e-5);
        for (uint32_t l = 1; l < 16; l++) ck.exact("ds_engram_gate 64-byte record", &own[r].gate[c][l], &own[r].gate[c][0], 4);
      }
  }

  std::vector<ref::HcMix> random_mixes(uint32_t W) {
    std::vector<ref::HcMix> rm(W);
    float hs[3] = {1, 1, 1}, hb[D41_MIX] = {};
    for (auto &x : rm) {
      auto m = randn(rng, D41_MIX, 2.0f);
      x = ref::hc_split(m.data(), hs, hb);
    }
    return rm;
  }

  void run_case(Kind kind, uint32_t W) {
    const Data d = make(rng, W, kind);
    const auto rm = random_mixes(W);
    std::vector<D41HcMix> mm(W);
    for (uint32_t r = 0; r < W; r++) std::memcpy(&mm[r], &rm[r], sizeof mm[r]);
    const auto norm = randn(rng, E, 0.5f);

    // All four streams on one card.
    const Part whole = {0, 4, 0, (uint32_t)E};
    CardMem &a = *cards[0];
    put_part(a, d, whole, W, mm, norm);
    launch_gate(a.io.L, a, whole, none, W);
    a.io.L.synchronize();
    check_gates(a, whole, d, W);
    launch_apply(a.io.L, a, whole, none, W, a.xa, a.xb, a.xf);
    a.io.L.synchronize();
    check_result(a, d, rm, norm, W);

    // Streams 0-1 and 2-3 on two parts, exchanging through the other part's memory and through host memory.
    CardMem &b = *cards[1];
    const Part pa = {0, 2, 0, 2560}, pb = {2, 4, 2560, 5120};
    for (bool through_host : {false, true}) {
      put_part(a, d, pa, W, mm, norm);
      put_part(b, d, pb, W, mm, norm);
      const void *to_b = through_host ? (const void *)host[1] : (const void *)b.in;
      const void *to_a = through_host ? (const void *)host[0] : (const void *)a.in;
      launch_gate(a.io.L, a, pa, to_b, W);
      a.io.L.synchronize();
      launch_gate(b.io.L, b, pb, to_a, W);
      b.io.L.synchronize();
      check_gates(a, pa, d, W);
      check_gates(b, pb, d, W);
      launch_apply(a.io.L, a, pa, to_a, W, a.xa, a.xb, a.xf);
      a.io.L.synchronize();
      launch_apply(b.io.L, b, pb, to_b, W, b.xa, b.xb, b.xf);
      b.io.L.synchronize();
      check_result(a, d, rm, norm, W);
      check_result(b, d, rm, norm, W);
      auto xa0 = a.io.get(a.X, W * D), xa1 = b.io.get(b.X, W * D);
      ck.exact("ds_engram_apply both cards' X", xa0.data(), xa1.data(), xa0.size() * 4);
      auto fa = a.io.get(a.xf, W * E), fb = b.io.get(b.xf, W * E);
      ck.exact("ds_engram_apply both cards' x", fa.data(), fb.data(), fa.size() * 4);
    }
  }

  void bench() {
    CardMem &a = *cards[0];
    const uint32_t W = WMAX;
    const Data d = make(rng, W, Kind::random);
    const Part whole0 = {0, 4, 0, (uint32_t)E};
    const auto rm = random_mixes(W);
    std::vector<D41HcMix> mm(W);
    for (uint32_t r = 0; r < W; r++) std::memcpy(&mm[r], &rm[r], sizeof mm[r]);
    const auto norm = randn(rng, E, 0.5f);
    put_part(a, d, whole0, W, mm, norm);
    time_launch(rt, 0, [&](ze::List &L) { launch_gate(L, a, whole0, none, W); });  // warms the card
    log_clocks("before timing");
    const Part pa = {0, 2, 0, 2560}, pb = {2, 4, 2560, 5120}, whole = {0, 4, 0, (uint32_t)E};
    if (!two_cards) std::printf("SKIP: the two-card timings of ds_engram_gate and ds_engram_apply need two cards\n");
    if (two_cards) {
      CardMem &b = *cards[1];
      put_part(a, d, pa, W, mm, norm);
      put_part(b, d, pb, W, mm, norm);
      launch_gate(b.io.L, b, pb, a.in, W);
      b.io.L.synchronize();
      launch_gate(b.io.L, b, pb, host[0], W);
      b.io.L.synchronize();
      for (uint32_t w : TIMED_WIDTHS) {
        std::printf("time ds_engram_gate two streams, send to the other card W=%u: %.2f us\n", w, time_launch(rt, 0, [&](ze::List &L) { launch_gate(L, a, pa, b.in, w); }));
        std::printf("time ds_engram_gate two streams, send to host memory W=%u: %.2f us\n", w, time_launch(rt, 0, [&](ze::List &L) { launch_gate(L, a, pa, host[1], w); }));
        launch_gate(a.io.L, a, pa, b.in, w);
        a.io.L.synchronize();
        std::printf("time ds_engram_apply peer in card memory W=%u: %.2f us\n", w, time_launch(rt, 0, [&](ze::List &L) { launch_apply(L, a, pa, a.in, w, a.xa, a.xb, none); }));
        std::printf("time ds_engram_apply peer in host memory W=%u: %.2f us\n", w, time_launch(rt, 0, [&](ze::List &L) { launch_apply(L, a, pa, host[0], w, a.xa, a.xb, none); }));
      }
    }
    put_part(a, d, whole, W, mm, norm);
    for (uint32_t w : TIMED_WIDTHS) {
      std::printf("time ds_engram_gate all four, no send W=%u: %.2f us\n", w, time_launch(rt, 0, [&](ze::List &L) { launch_gate(L, a, whole, none, w); }));
      launch_gate(a.io.L, a, whole, none, w);
      a.io.L.synchronize();
      std::printf("time ds_engram_apply all four W=%u: %.2f us\n", w, time_launch(rt, 0, [&](ze::List &L) { launch_apply(L, a, whole, none, w, a.xa, a.xb, none); }));
    }
    log_clocks("after timing");
  }
};

}  // namespace

int main(int argc, char **argv) {
  init_clocks();
  if (test::gpus() == 0) return test::no_gpu();
  ze::Runtime rt;
  std::printf("cards visible: %u\n", rt.ncards());
  const bool one_card = argc > 1 && !std::strcmp(argv[1], "--one-card");
  if (const char *e = std::getenv("ENGINE_EXPECT_GPUS"))
    EXPECT(rt.ncards() == (uint32_t)std::atoi(e), "%u cards ran, ENGINE_EXPECT_GPUS=%s", rt.ncards(), e);
  ze::Library lib(rt);
  const bool two = rt.ncards() > 1 && !one_card;
  if (!two) std::printf("one card in use: streams 0-1 and 2-3 run as two parts on card 0, not across cards\n");
  Bench b(rt, lib, two);
  for (Kind k : {Kind::random, Kind::tiny_weights, Kind::zero_weights, Kind::orthogonal, Kind::key_is_stream,
                 Kind::key_is_minus_stream, Kind::negative_weights})
    for (uint32_t W : {1u, 2u, 4u, 8u}) b.run_case(k, W);
  // x after the injection at layers 1 and 14: the same arithmetic on each layer's own random data.
  for (uint32_t layer : {1u, 14u}) {
    b.rng.seed(layer);
    b.run_case(Kind::random, WMAX);
  }
  b.bench();
  return b.ck.report();
}
