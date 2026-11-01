// Level 3 (docs/TESTING.md): deepseek41's ds_qkv_norm (kernels/models/deepseek41/attn.cl) on random data at the model's
// shapes, against ref_ops and pinned ggml's rope_ext on a view at the rotated dimensions: both rotary bases at the
// positions where the window ring wraps, 1-8 rows, rows of several slots, a padded row; and its time at 1, 2, 4 and 8
// rows with every card visible. Limits: 1e-5 of the largest |reference|; a8, bf16 and the ring's f16 exact from the
// kernel's own f32, and the ring's other cells untouched. No model is needed.
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

constexpr size_t Q = D41_QLR, H = D41_HD;
constexpr uint32_t WMAX = STEP_MAX_ROWS, SLOTS = 3;
constexpr size_t RING_VALUES = (size_t)SLOTS * D41_RING * H;
constexpr uint32_t LOCAL = D41_QKV_SGS * 16;
constexpr uint16_t UNTOUCHED = 0xabab;
constexpr uint32_t POSITIONS[] = {0, 1, 127, 128, 135, 136, 137, 4095, 16383};
const void *const none = nullptr;

// The rotated dimensions of each of W rows of H values, by ggml's rope_ext on a view at the offset, as upstream
// llama.cpp calls it for the layer kind (src/models/deepseek41.cpp, build_attention41).
std::vector<float> ggml_tail(const float *x, const std::vector<int32_t> &pos, bool yarn) {
  const uint32_t W = (uint32_t)pos.size();
  ggml_init_params ip = {64u << 20, nullptr, false};
  ggml_context *ctx = ggml_init(ip);
  ggml_tensor *a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, H, 1, W);
  std::memcpy(a->data, x, ggml_nbytes(a));
  ggml_tensor *tail = ggml_view_3d(ctx, a, D41_ROT, 1, W, a->nb[1], a->nb[2], D41_NOPE * sizeof(float));
  ggml_tensor *p = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, W);
  std::memcpy(p->data, pos.data(), W * sizeof(int32_t));
  const float freq_scale = yarn ? 1.0f / D41_YARN_FACTOR : 1.0f, ext = yarn ? 1.0f : 0.0f;
  const float attn = yarn ? 1.0f / (1.0f + 0.1f * logf(1.0f / freq_scale)) : 1.0f;
  ggml_tensor *r = ggml_rope_ext(ctx, tail, p, nullptr, D41_ROT, GGML_ROPE_TYPE_NORMAL, yarn ? D41_YARN_ORIG : 0,
                                 yarn ? D41_ROPE_BASE_C : D41_ROPE_BASE, freq_scale, ext, attn,
                                 yarn ? D41_YARN_BETA_FAST : 0.0f, yarn ? D41_YARN_BETA_SLOW : 0.0f);
  ggml_cgraph *gf = ggml_new_graph(ctx);
  ggml_build_forward_expand(gf, r);
  EXPECT(ggml_graph_compute_with_ctx(ctx, gf, 1) == GGML_STATUS_SUCCESS, "ggml's graph failed");
  std::vector<float> y((size_t)W * D41_ROT);
  for (uint32_t k = 0; k < W; k++)
    std::memcpy(&y[(size_t)k * D41_ROT], (const char *)r->data + k * r->nb[2], D41_ROT * sizeof(float));
  ggml_free(ctx);
  return y;
}

struct Bench {
  ze::Runtime &rt;
  ze::Library &lib;
  Io &io;
  std::mt19937 rng{41};
  Checker ck;
  float *qr, *kv, *q_norm, *kv_norm, *qf, *kvf;
  uint8_t *qa;
  uint16_t *qb, *ring;
  D41Row *rows;
  std::vector<float> hqr, hkv, hq_norm, hkv_norm;

  Bench(ze::Runtime &r, ze::Library &l, Io &i) : rt(r), lib(l), io(i) {
    qr = io.dev<float>(WMAX * Q);
    kv = io.dev<float>(WMAX * H);
    q_norm = io.dev<float>(Q);
    kv_norm = io.dev<float>(H);
    qf = io.dev<float>(WMAX * Q);
    kvf = io.dev<float>(WMAX * H);
    qa = io.dev<uint8_t>((size_t)WMAX * D41_QR_A8_ROW);
    qb = io.dev<uint16_t>(WMAX * Q);
    ring = io.dev<uint16_t>(RING_VALUES);
    rows = io.dev<D41Row>(WMAX);
    hqr = randn(rng, WMAX * Q, 0.8f);
    hkv = randn(rng, WMAX * H, 1.3f);
    hq_norm = randn(rng, Q, 0.5f);
    hkv_norm = randn(rng, H, 0.5f);
    io.put(qr, hqr);
    io.put(kv, hkv);
    io.put(q_norm, hq_norm);
    io.put(kv_norm, hkv_norm);
  }
  ze_kernel_handle_t k() { return lib.get(io.card, "deepseek41_attn", "ds_qkv_norm"); }
  void launch(ze::List &L, uint32_t W, uint32_t yarn, const void *pb, const void *pqf, const void *pkvf) {
    L.launch(k(), {qr, q_norm, kv, kv_norm, rows, yarn, qa, pb, pqf, ring, pkvf}, 2, W, 1, LOCAL);
  }

  // W rows at pos, pos + 1, ..., in slots round robin from the first; the rows from npad on are padded.
  std::vector<D41Row> make_rows(uint32_t W, uint32_t pos, uint32_t nslots, uint32_t npad) {
    std::vector<D41Row> v(WMAX);
    for (uint32_t r = 0; r < W; r++) {
      const uint32_t p = pos + r;
      v[r] = {r < npad ? r % nslots : D41_NO_ROW, p, p % D41_RING, D41_NO_ROW};
    }
    return v;
  }

  void check(uint32_t W, uint32_t pos, uint32_t yarn, uint32_t nslots, uint32_t npad) {
    const auto hrows = make_rows(W, pos, nslots, npad);
    io.put(rows, hrows);
    std::vector<uint16_t> model(RING_VALUES, UNTOUCHED);
    io.put(ring, model);
    io.L.fill(qa, &UNTOUCHED, sizeof UNTOUCHED, (size_t)W * D41_QR_A8_ROW);
    io.L.barrier();
    launch(io.L, W, yarn, qb, qf, kvf);
    io.L.synchronize();

    auto gq = io.get(qf, W * Q);
    auto ga = io.get(qa, (size_t)W * D41_QR_A8_ROW);
    auto gb = io.get(qb, W * Q);
    auto gk = io.get(kvf, W * H);
    std::vector<int32_t> ppos;
    for (uint32_t r = 0; r < W; r++) ppos.push_back((int32_t)hrows[r].pos);
    std::vector<float> normed(W * H);
    for (uint32_t r = 0; r < W; r++) {
      std::vector<float> wq(Q), wk(H);
      ref::rms_norm(&hqr[r * Q], hq_norm.data(), Q, wq.data());
      ck.rel("qr f32", &gq[r * Q], wq.data(), Q, 1e-5);
      auto wa = a8_row(&gq[r * Q], Q);
      ck.exact("qr a8 from its own f32", &ga[(size_t)r * D41_QR_A8_ROW], wa.data(), wa.size());
      std::vector<uint16_t> wb(Q);
      for (size_t i = 0; i < Q; i++) wb[i] = ggml_fp32_to_bf16(gq[r * Q + i]).bits;
      ck.exact("qr bf16 from its own f32", &gb[r * Q], wb.data(), Q * 2);

      ref::rms_norm(&hkv[r * H], hkv_norm.data(), H, wk.data());
      std::copy(wk.begin(), wk.end(), &normed[r * H]);
      ck.rel("kv f32 before the rotated dimensions", &gk[r * H], wk.data(), D41_NOPE, 1e-5);
      ref::rope_tail(wk.data(), H, hrows[r].pos, yarn, false);
      ck.rel(yarn ? "kv f32 against ref_ops, base 160000" : "kv f32 against ref_ops, base 10000", &gk[r * H], wk.data(), H, 1e-5);
      // The ring's f16 comes from the kernel's own f32.
      if (hrows[r].slot != D41_NO_ROW)
        for (size_t d = 0; d < H; d++)
          model[((size_t)hrows[r].slot * D41_RING + hrows[r].ring) * H + d] = ggml_fp32_to_fp16(gk[r * H + d]);
    }
    auto gtail = ggml_tail(normed.data(), ppos, yarn);
    for (uint32_t r = 0; r < W; r++)
      ck.rel(yarn ? "kv f32 against ggml, base 160000" : "kv f32 against ggml, base 10000", &gk[r * H + D41_NOPE], &gtail[(size_t)r * D41_ROT], D41_ROT, 1e-5);
    auto gr = io.get(ring, RING_VALUES);
    ck.exact("ring f16, its other cells untouched", gr.data(), model.data(), RING_VALUES * 2);
  }

  void test() {
    for (uint32_t yarn : {0u, 1u})
      for (uint32_t pos : POSITIONS)
        for (uint32_t W : WIDTHS) {
          check(W, pos, yarn, 1, W);
          if (W > 1) check(W, pos, yarn, SLOTS, W - 1);
        }
    // No optional output: the a8 and ring are the same with bf16 and the f32 copies left out.
    const uint32_t W = WMAX;
    const auto hrows = make_rows(W, 133, SLOTS, W);
    io.put(rows, hrows);
    io.L.fill(ring, &UNTOUCHED, sizeof UNTOUCHED, RING_VALUES * 2);
    io.L.barrier();
    launch(io.L, W, 1, qb, qf, kvf);
    io.L.synchronize();
    auto a0 = io.get(qa, (size_t)W * D41_QR_A8_ROW);
    auto r0 = io.get(ring, RING_VALUES);
    io.L.fill(ring, &UNTOUCHED, sizeof UNTOUCHED, RING_VALUES * 2);
    io.L.fill(qa, &UNTOUCHED, sizeof UNTOUCHED, (size_t)W * D41_QR_A8_ROW);
    io.L.barrier();
    launch(io.L, W, 1, none, none, none);
    io.L.synchronize();
    auto a1 = io.get(qa, (size_t)W * D41_QR_A8_ROW);
    auto r1 = io.get(ring, RING_VALUES);
    ck.exact("no optional output: a8", a1.data(), a0.data(), a0.size());
    ck.exact("no optional output: ring", r1.data(), r0.data(), r0.size() * 2);
  }

  void bench() {
    time_launch(rt, 0, [&](ze::List &L) { launch(L, 1, 1, none, none, none); });
    log_clocks("before timing");
    for (uint32_t pos : {100u, 16383u}) {
      for (uint32_t W : TIMED_WIDTHS) {
        // Rows of one slot, as a draft window is, the last at pos; the rotary work is the same for each row.
        const auto hrows = make_rows(W, pos + 1 - W, 1, W);
        io.put(rows, hrows);
        std::printf("time ds_qkv_norm a8 only, base 160000, last pos %u, W=%u: %.2f us\n", pos, W, time_launch(rt, 0, [&](ze::List &L) { launch(L, W, 1, none, none, none); }));
        std::printf("time ds_qkv_norm a8 and bf16, base 160000, last pos %u, W=%u: %.2f us\n", pos, W, time_launch(rt, 0, [&](ze::List &L) { launch(L, W, 1, qb, none, none); }));
        std::printf("time ds_qkv_norm a8 only, base 10000, last pos %u, W=%u: %.2f us\n", pos, W, time_launch(rt, 0, [&](ze::List &L) { launch(L, W, 0, none, none, none); }));
      }
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
  ze::Library lib(rt);
  Io io(rt, 0);
  Bench b(rt, lib, io);
  b.test();
  b.bench();
  return b.ck.report();
}
