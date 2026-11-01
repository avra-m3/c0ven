// Level 3 (docs/TESTING.md): deepseek41's ds_attn_split, ds_attn_merge and ds_output_groups on random data at the model's
// shapes, against ref_ops and ggml's soft_max with sinks: head ranges 0-31, 32-63 and 0-63, the ring filling and wrapping,
// n_vis 0, ratios 0, 1 and 2, shuffled and absent pages, large sinks, padded rows; output groups 0-3, 4-7 and 0-7 in Q8_0
// and Q6_K, then attn_output_b's columns through matvec; each kernel's time at 1, 2, 4 and 8 rows with every card visible.
// Limits: 2e-3 of the largest |reference| for attention, 1e-5 for the products; a8 exact from the kernel's own f32.
#include <cmath>
#include <numeric>
#include <string>

#include "ds41_kernel_test.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "load/gguf.h"
#include "load/runs.h"
#include "models/deepseek41/layout.h"
#include "models/deepseek41/ref_ops.h"
#include "models/deepseek41/rope.h"
#include "quant.h"

using namespace engine;
using namespace ds41test;
namespace ref = engine::deepseek41::ref;

namespace {

constexpr size_t H = D41_HD, CR = D41_COMP_ROW, Q_ALL = (size_t)D41_H * H;
constexpr uint32_t WMAX = STEP_MAX_ROWS, SLOTS = 8, NPP = D41_CTX_CAP / KV_PAGE_CELLS, NPAGES = SLOTS * NPP;
constexpr size_t RING_VALUES = (size_t)SLOTS * D41_RING * H, POOL_VALUES = (size_t)NPAGES * KV_PAGE_CELLS * CR;
constexpr size_t A8_ROW = D41_OG_A8_ROW;
constexpr uint32_t UNTOUCHED = 0xabababab;
constexpr double ATT_LIMIT = 2e-3, MV_LIMIT = 1e-5;
const void *const none = nullptr;

// Where row j lies in the pool, stated apart from d41_comp_index, which the kernel calls: a page holds 4 positions.
constexpr u64 pool_index(uint32_t page, uint32_t j, uint32_t ratio) {
  return (u64)page * (4 / ratio) + (j * ratio) % 4 / ratio;
}
static_assert(pool_index(5, 3, 2) == d41_comp_index(5, 3, 2) && pool_index(5, 7, 1) == d41_comp_index(5, 7, 1));

struct Range {
  uint32_t h0, nh;
};
constexpr Range RANGES[] = {{0, 32}, {32, 32}, {0, 64}};

enum class Sinks { spread, big30, big60, big200, mixed, low };

std::vector<float> make_sinks(std::mt19937 &rng, Sinks kind) {
  std::vector<float> s = randn(rng, D41_H);
  for (uint32_t h = 0; h < D41_H; h++) {
    if (kind == Sinks::big30) s[h] += 30.0f;
    if (kind == Sinks::big60) s[h] += 60.0f;
    if (kind == Sinks::big200) s[h] += 200.0f;
    if (kind == Sinks::mixed) s[h] += h % 3 == 0 ? 30.0f : h % 3 == 1 ? -30.0f : 0.0f;
    if (kind == Sinks::low) s[h] -= 200.0f;
  }
  return s;
}

// o for every head from ggml's softmax with sinks: scores of the rotated q on nk keys, then the probabilities on the keys.
std::vector<float> ggml_attn(const float *q, const float *keys, uint32_t nk, const float *sinks) {
  ggml_init_params ip = {(size_t)D41_H * nk * 8 + (32u << 20), nullptr, false};
  ggml_context *ctx = ggml_init(ip);
  ggml_tensor *s = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, nk, 1, D41_H);
  ggml_tensor *sk = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D41_H);
  std::memcpy(sk->data, sinks, D41_H * sizeof(float));
  float *sd = (float *)s->data;
  for (uint32_t h = 0; h < D41_H; h++)
    for (uint32_t i = 0; i < nk; i++) {
      double v = 0;
      for (size_t d = 0; d < H; d++) v += (double)q[(size_t)h * H + d] * keys[(size_t)i * H + d];
      sd[(size_t)h * nk + i] = (float)(v * (1.0 / std::sqrt((double)H)));
    }
  ggml_tensor *p = ggml_soft_max(ctx, s);
  ggml_soft_max_add_sinks(p, sk);
  ggml_cgraph *gf = ggml_new_graph(ctx);
  ggml_build_forward_expand(gf, p);
  EXPECT(ggml_graph_compute_with_ctx(ctx, gf, 1) == GGML_STATUS_SUCCESS, "ggml's graph failed");
  std::vector<float> o(Q_ALL);
  const float *pd = (const float *)p->data;
  for (uint32_t h = 0; h < D41_H; h++)
    for (size_t d = 0; d < H; d++) {
      double acc = 0;
      for (uint32_t i = 0; i < nk; i++) acc += (double)pd[(size_t)h * nk + i] * keys[(size_t)i * H + d];
      o[(size_t)h * H + d] = (float)acc;
    }
  ggml_free(ctx);
  return o;
}

double a8_value(const uint8_t *a, size_t K, size_t i) {
  float d;
  std::memcpy(&d, a + K + 4 * (i / 32), 4);
  return (double)d * (int8_t)a[i];
}

std::string module(const char *prefix, uint32_t type, uint32_t lanes) {
  std::string s = std::string(prefix) + ggml_type_label(type) + "_l" + std::to_string(lanes);
  for (auto &c : s) c = (char)std::tolower(c);
  return s;
}

struct Bench {
  ze::Runtime &rt;
  ze::Library &lib;
  Io &io;
  std::mt19937 rng{41};
  Checker ck;
  float *q, *part, *sinks, *of, *af, *yb;
  uint16_t *ring, *pool;
  uint8_t *oa, *aa, *grids;
  uint32_t *pt, *idx;
  D41Row *rows;
  std::vector<uint16_t> hring, hpool;
  std::vector<uint32_t> shuffled;

  Bench(ze::Runtime &r, ze::Library &l, Io &i) : rt(r), lib(l), io(i) {
    q = io.dev<float>(WMAX * Q_ALL);
    part = io.dev<float>((size_t)WMAX * D41_H * D41_ATT_PART);
    sinks = io.dev<float>(D41_H);
    of = io.dev<float>(WMAX * Q_ALL);
    af = io.dev<float>((size_t)WMAX * D41_OG * D41_OLR);
    yb = io.dev<float>((size_t)WMAX * D41_EMBD);
    ring = io.dev<uint16_t>(RING_VALUES);
    pool = io.dev<uint16_t>(POOL_VALUES);
    oa = io.dev<uint8_t>((size_t)D41_OG * WMAX * A8_ROW);
    aa = io.dev<uint8_t>((size_t)WMAX * (D41_OG * D41_OLR + D41_OG * D41_OLR / 8));
    grids = io.dev<uint8_t>(QGRID_BYTES);
    pt = io.dev<uint32_t>(NPAGES);
    idx = io.dev<uint32_t>(WMAX * D41_TOPK);
    rows = io.dev<D41Row>(WMAX);
    io.put(grids, codebooks().data(), QGRID_BYTES);
    // Every cell and row random, so a key read from a wrong place changes the result.
    hring.resize(RING_VALUES);
    hpool.resize(POOL_VALUES);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto &v : hring) v = ggml_fp32_to_fp16(nd(rng));
    for (auto &v : hpool) v = ggml_fp32_to_fp16(nd(rng));
    io.put(ring, hring);
    io.put(pool, hpool);
    shuffled.resize(NPAGES);
    std::iota(shuffled.begin(), shuffled.end(), 0u);
    std::shuffle(shuffled.begin(), shuffled.end(), rng);
    io.put(pt, shuffled);
  }

  ze_kernel_handle_t k_split() { return lib.get(io.card, "deepseek41_attn", "ds_attn_split"); }
  ze_kernel_handle_t k_merge() { return lib.get(io.card, "deepseek41_attn", "ds_attn_merge"); }
  static uint32_t nsplits(uint32_t ratio) { return ratio ? D41_ATT_SPLITS : 1; }
  void launch_split(ze::List &L, uint32_t W, uint32_t nh, uint32_t ratio, const void *ppt) {
    L.launch(k_split(), {q, rows, ratio, ring, ppt, NPP, pool, idx, part}, nh / D41_ATT_HEADS, W, nsplits(ratio),
             D41_ATT_WG);
  }
  void launch_merge(ze::List &L, uint32_t W, uint32_t h0, uint32_t nh, uint32_t ratio, const void *pof) {
    L.launch(k_merge(), {part, sinks, h0, rows, ratio, oa, pof}, nh, W, 1, 256);
  }

  std::vector<D41Row> make_rows(uint32_t W, uint32_t last, uint32_t nslots, uint32_t npad) {
    std::vector<D41Row> v(WMAX);
    for (uint32_t r = 0; r < W; r++) {
      const uint32_t p = last + 1 >= W ? last + 1 - W + r : r;
      v[r] = {r < npad ? r % nslots : D41_NO_ROW, p, p % D41_RING, D41_NO_ROW};
    }
    return v;
  }

  // Each row's top-k numbers: chosen at random, increasing, the entries after them left unwritten.
  std::vector<uint32_t> make_idx(const std::vector<D41Row> &hrows, uint32_t W, uint32_t ratio) {
    std::vector<uint32_t> v((size_t)WMAX * D41_TOPK, UNTOUCHED);
    if (!ratio) return v;
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    for (uint32_t r = 0; r < W; r++) {
      if (hrows[r].slot == D41_NO_ROW) continue;
      const uint32_t n = (hrows[r].pos + 1) / ratio;
      std::vector<float> sc(n);
      for (auto &x : sc) x = u(rng);
      const auto top = ref::top_k(sc.data(), n, D41_TOPK);
      std::copy(top.begin(), top.end(), &v[(size_t)r * D41_TOPK]);
    }
    return v;
  }

  // A row's keys as f32: the window oldest first, then its compressed rows.
  std::vector<float> row_keys(const D41Row &row, uint32_t ratio, const uint32_t *ids, bool pages, uint32_t *nk) {
    const uint32_t vis = std::min(row.pos + 1, (uint32_t)D41_WIN);
    const uint32_t comp = ratio ? std::min((row.pos + 1) / ratio, (uint32_t)D41_TOPK) : 0;
    std::vector<float> k((size_t)(vis + comp) * H);
    *nk = vis + comp;
    for (uint32_t i = 0; i < vis; i++) {
      const uint32_t p = row.pos + 1 - vis + i;
      const uint16_t *c = &hring[((size_t)row.slot * D41_RING + p % D41_RING) * H];
      for (size_t d = 0; d < H; d++) k[(size_t)i * H + d] = ggml_fp16_to_fp32(c[d]);
    }
    for (uint32_t i = 0; i < comp; i++) {
      const uint32_t j = ids[i], own = row.slot * NPP + j * ratio / KV_PAGE_CELLS, page = pages ? shuffled[own] : own;
      const uint16_t *c = &hpool[pool_index(page, j, ratio) * CR];
      for (size_t d = 0; d < H; d++) k[(size_t)(vis + i) * H + d] = ggml_fp16_to_fp32(c[d]);
    }
    return k;
  }

  void check(uint32_t W, uint32_t last, uint32_t ratio, uint32_t nslots, uint32_t npad, Sinks sk, float qscale, bool pages) {
    const auto hrows = make_rows(W, last, nslots, npad);
    io.put(rows, hrows);
    const auto hidx = make_idx(hrows, W, ratio);
    io.put(idx, hidx);
    const auto hq = randn(rng, W * Q_ALL, qscale);
    const auto hs = make_sinks(rng, sk);
    io.put(sinks, hs);

    // The reference for every head of every real row.
    std::vector<std::vector<float>> want(W), viag(W);
    for (uint32_t r = 0; r < W; r++) {
      if (hrows[r].slot == D41_NO_ROW) continue;
      uint32_t nk;
      const auto keys = row_keys(hrows[r], ratio, &hidx[(size_t)r * D41_TOPK], pages, &nk);
      std::vector<float> rq(&hq[r * Q_ALL], &hq[r * Q_ALL] + Q_ALL);
      for (uint32_t h = 0; h < D41_H; h++) ref::rope_tail(&rq[(size_t)h * H], H, hrows[r].pos, ratio != 0, false);
      std::vector<int32_t> all(nk);
      std::iota(all.begin(), all.end(), 0);
      want[r].resize(Q_ALL);
      ref::sparse_attn(rq.data(), keys.data(), H, all.data(), nk, hs.data(), want[r].data());
      for (uint32_t h = 0; h < D41_H; h++) ref::rope_tail(&want[r][(size_t)h * H], H, hrows[r].pos, ratio != 0, true);
      if (r == 0) {
        viag[r] = ggml_attn(rq.data(), keys.data(), nk, hs.data());
        for (uint32_t h = 0; h < D41_H; h++) ref::rope_tail(&viag[r][(size_t)h * H], H, hrows[r].pos, ratio != 0, true);
      }
    }

    for (Range range : RANGES) {
      const uint32_t nh = range.nh, h0 = range.h0, ng = nh / D41_OG_HEADS;
      const size_t row_values = (size_t)nh * H;
      std::vector<float> lq((size_t)W * row_values);
      for (uint32_t r = 0; r < W; r++)
        std::copy_n(&hq[(size_t)r * Q_ALL + (size_t)h0 * H], row_values, &lq[(size_t)r * row_values]);
      io.put(q, lq);
      io.L.fill(of, &UNTOUCHED, 4, (size_t)W * row_values * 4);
      io.L.fill(oa, &UNTOUCHED, 4, (size_t)ng * W * A8_ROW);
      io.L.barrier();
      launch_split(io.L, W, nh, ratio, pages ? (const void *)pt : none);
      io.L.barrier();
      launch_merge(io.L, W, h0, nh, ratio, of);
      io.L.synchronize();

      const auto go = io.get(of, (size_t)W * row_values);
      const auto ga = io.get(oa, (size_t)ng * W * A8_ROW);
      for (uint32_t r = 0; r < W; r++) {
        const float *gr = &go[(size_t)r * row_values];
        if (hrows[r].slot == D41_NO_ROW) {
          std::vector<uint32_t> blank(row_values, UNTOUCHED);
          ck.exact("a padded row's output untouched", gr, blank.data(), row_values * 4);
          std::vector<uint8_t> a8blank(A8_ROW);
          for (size_t i = 0; i < A8_ROW; i++) a8blank[i] = 0xab;
          for (uint32_t g = 0; g < ng; g++)
            ck.exact("a padded row's a8 untouched", &ga[((size_t)g * W + r) * A8_ROW], a8blank.data(), A8_ROW);
          continue;
        }
        const char *nm = ratio == 0 ? "o against ref_ops, ratio 0" : ratio == 1 ? "o against ref_ops, ratio 1" : "o against ref_ops, ratio 2";
        ck.rel(nm, gr, &want[r][(size_t)h0 * H], row_values, ATT_LIMIT);
        if (r == 0) ck.rel("o against ggml's soft_max with sinks", gr, &viag[r][(size_t)h0 * H], row_values, ATT_LIMIT);
        for (uint32_t g = 0; g < ng; g++) {
          const auto wa = a8_row(gr + (size_t)g * D41_OG_K, D41_OG_K);
          ck.exact("o a8 by group, from its own f32", &ga[((size_t)g * W + r) * A8_ROW], wa.data(), A8_ROW);
        }
      }
    }
  }

  void test_attn() {
    const uint32_t lasts[] = {0, 1, 5, 126, 127, 128, 134, 135, 136, 137, 255, 256, 257, 511, 1023, 1024, 1025, 16383};
    for (uint32_t ratio : {0u, 2u, 1u})
      for (uint32_t last : lasts)
        for (uint32_t W : {1u, 3u, 8u}) {
          check(W, last, ratio, 1, W, Sinks::spread, 1.0f, true);
          if (W > 1) check(W, last, ratio, 3, W - 1, Sinks::spread, 4.0f, false);
        }
    for (uint32_t W : WIDTHS) check(W, 16383, 1, std::min(W, SLOTS), W, Sinks::spread, 2.0f, true);
    for (Sinks s : {Sinks::big30, Sinks::big60, Sinks::big200, Sinks::mixed, Sinks::low}) {
      check(2, 0, 2, 1, 2, s, 1.0f, true);  // n_vis 0 and 1
      check(2, 100, 0, 1, 2, s, 1.0f, true);
      check(4, 16383, 1, 2, 4, s, 1.0f, true);
    }
  }

  // With and without the optional f32 output: the same a8.
  void check_no_of() {
    const auto hrows = make_rows(WMAX, 16383, 2, WMAX);
    io.put(rows, hrows);
    io.put(idx, make_idx(hrows, WMAX, 1));
    const auto hq = randn(rng, WMAX * Q_ALL);
    io.put(q, hq);
    launch_split(io.L, WMAX, D41_H, 1, pt);
    io.L.barrier();
    launch_merge(io.L, WMAX, 0, D41_H, 1, of);
    io.L.synchronize();
    const auto a0 = io.get(oa, (size_t)D41_OG * WMAX * A8_ROW);
    io.L.fill(oa, &UNTOUCHED, 4, a0.size());
    io.L.barrier();
    launch_merge(io.L, WMAX, 0, D41_H, 1, none);
    io.L.synchronize();
    const auto a1 = io.get(oa, a0.size());
    ck.exact("no optional output: o a8", a1.data(), a0.data(), a0.size());
  }

  // ---------------------------------------------------------------- output groups
  struct OutWeights {
    uint32_t type, bpr_a, bpr_b;
    std::vector<float> wa, wb;       // the dequantised weights
    std::vector<uint8_t> qa, qb;     // the file's layout
    uint64_t rowa, rowb;
  };

  OutWeights make_out(uint32_t type) {
    OutWeights o;
    o.type = type;
    const uint64_t RA = (uint64_t)D41_OG * D41_OLR, KA = D41_OG_K, RB = D41_EMBD, KB = RA;
    o.rowa = ggml_row_size((ggml_type)type, KA);
    o.rowb = ggml_row_size((ggml_type)type, KB);
    o.bpr_a = (uint32_t)(KA / quant_block_values(type));
    o.bpr_b = (uint32_t)(KB / quant_block_values(type));
    auto fa = randn(rng, RA * KA, 0.02f);
    o.qa.resize(RA * o.rowa);
    ggml_quantize_chunk((ggml_type)type, fa.data(), o.qa.data(), 0, RA, KA, nullptr);
    o.wa.resize(RA * KA);
    ggml_get_type_traits((ggml_type)type)->to_float(o.qa.data(), o.wa.data(), (int64_t)(RA * KA));
    fa = randn(rng, RB * KB, 0.02f);
    o.qb.resize(RB * o.rowb);
    ggml_quantize_chunk((ggml_type)type, fa.data(), o.qb.data(), 0, RB, KB, nullptr);
    o.wb.resize(RB * KB);
    ggml_get_type_traits((ggml_type)type)->to_float(o.qb.data(), o.wb.data(), (int64_t)(RB * KB));
    return o;
  }

  uint8_t *upload_slice(uint32_t type, const uint8_t *base, uint64_t row_bytes, uint64_t r0, uint64_t nrows, uint64_t b0, uint64_t nb) {
    std::vector<uint8_t> runs(runs_bytes(type, nrows * nb));
    to_runs(type, {{base, row_bytes, r0, nrows, b0, nb}}, runs.data());
    auto *d = io.dev<uint8_t>(runs.size());
    io.put(d, runs);
    return d;
  }

  void launch_groups(ze::List &L, uint32_t type, uint32_t lanes, const uint8_t *dW, uint32_t bpr, uint32_t ntok,
                     uint32_t ng, const void *paf) {
    L.launch(lib.get(io.card, module("deepseek41_og_", type, lanes), "ds_output_groups"), {dW, bpr, oa, ntok, aa, paf},
             D41_OLR / D41_OG_WG_ROWS, ng, 1, D41_OG_WG_ROWS / (16 / lanes) * 16);
  }

  // attn_output_a's groups [g0, g0 + ng) on tokens of random o, then attn_output_b's columns of those groups.
  void check_out(const OutWeights &o, uint32_t g0, uint32_t ng, uint32_t W, uint32_t lanes, uint8_t *dA, uint8_t *dB) {
    const uint32_t K = ng * D41_OLR;
    const uint32_t bprb = K / quant_block_values(o.type);
    std::vector<uint8_t> act((size_t)ng * W * A8_ROW);
    std::vector<float> dec((size_t)W * Q_ALL, 0.0f);
    for (uint32_t g = 0; g < ng; g++)
      for (uint32_t t = 0; t < W; t++) {
        const auto x = randn(rng, D41_OG_K, 1.0f + (float)t);
        const auto a8 = a8_row(x.data(), D41_OG_K);
        std::copy(a8.begin(), a8.end(), &act[((size_t)g * W + t) * A8_ROW]);
        for (size_t i = 0; i < D41_OG_K; i++)
          dec[(size_t)t * Q_ALL + (size_t)(g0 + g) * D41_OG_K + i] = (float)a8_value(a8.data(), D41_OG_K, i);
      }
    io.put(oa, act);
    const size_t arow = (size_t)K + K / 8;
    launch_groups(io.L, o.type, lanes, dA, o.bpr_a, W, ng, none);
    io.L.synchronize();
    const auto gb0 = io.get(aa, (size_t)W * arow);
    launch_groups(io.L, o.type, lanes, dA, o.bpr_a, W, ng, af);
    io.L.synchronize();
    const auto ga = io.get(af, (size_t)W * K);
    const auto gb = io.get(aa, (size_t)W * arow);
    ck.exact("no optional output: a a8", gb0.data(), gb.data(), gb.size());
    const bool q8 = o.type == QT_Q8_0;
    for (uint32_t t = 0; t < W; t++) {
      // ref_ops takes all 8 groups; the others are zero here.
      std::vector<float> a((size_t)D41_OG * D41_OLR), out(D41_EMBD);
      ref::grouped_output(&dec[(size_t)t * Q_ALL], o.wa.data(), o.wb.data(), a.data(), out.data());
      ck.rel(q8 ? "a f32 against ref_ops, Q8_0" : "a f32 against ref_ops, Q6_K", &ga[(size_t)t * K],
             &a[(size_t)g0 * D41_OLR], K, MV_LIMIT);
      const auto wa8 = a8_row(&ga[(size_t)t * K], K);
      ck.exact("a a8, from its own f32", &gb[(size_t)t * arow], wa8.data(), arow);
    }
    if (dB) {
      const uint32_t rps = 16 / lanes;
      const uint32_t rows_b = D41_EMBD, stride = (uint32_t)arow, ytok = rows_b;
      io.L.launch(lib.get(io.card, module("matvec_", o.type, lanes), "matvec"),
                  {dB, rows_b, bprb, aa, K, stride, W, yb, ytok, grids}, (rows_b + 4 * rps - 1) / (4 * rps), 1, 1, 64);
      io.L.synchronize();
      const auto gy = io.get(yb, (size_t)W * rows_b);
      const uint64_t b0 = (uint64_t)g0 * D41_OLR;
      for (uint32_t t = 0; t < W; t++) {
        std::vector<double> wy(rows_b);
        const uint8_t *ar = &gb[(size_t)t * arow];
        std::vector<double> av(K);
        for (uint32_t i = 0; i < K; i++) av[i] = a8_value(ar, K, i);
        for (uint32_t i = 0; i < rows_b; i++) {
          double s = 0;
          const float *wr = &o.wb[(size_t)i * D41_OG * D41_OLR + b0];
          for (uint32_t c = 0; c < K; c++) s += (double)wr[c] * av[c];
          wy[i] = s;
        }
        ck.rel(q8 ? "attn_output_b's columns, Q8_0" : "attn_output_b's columns, Q6_K", &gy[(size_t)t * rows_b], wy.data(), rows_b, MV_LIMIT);
      }
    }
  }

  void test_and_time_output() {
    for (uint32_t type : {(uint32_t)QT_Q8_0, (uint32_t)QT_Q6_K}) {
      const auto o = make_out(type);
      struct Part {
        uint32_t g0, ng;
        uint8_t *dA, *dB;
      };
      std::vector<Part> parts;
      for (Range r : RANGES) {
        const uint32_t g0 = r.h0 / D41_OG_HEADS, ng = r.nh / D41_OG_HEADS;
        const uint32_t vb = quant_block_values(type);
        parts.push_back({g0, ng, upload_slice(type, o.qa.data(), o.rowa, (uint64_t)g0 * D41_OLR, (uint64_t)ng * D41_OLR, 0, o.bpr_a),
                         upload_slice(type, o.qb.data(), o.rowb, 0, D41_EMBD, (uint64_t)g0 * D41_OLR / vb, (uint64_t)ng * D41_OLR / vb)});
      }
      for (const Part &p : parts)
        for (uint32_t lanes : {16u, 8u, 4u})
          for (uint32_t W : {1u, 2u, 3u, 8u}) check_out(o, p.g0, p.ng, W, lanes, p.dA, p.dB);
      // The card's 4 groups, timed.
      const Part &p = parts[0];
      time_launch(rt, 0, [&](ze::List &L) { launch_groups(L, type, 16, p.dA, o.bpr_a, 1, p.ng, none); });
      log_clocks("before timing the output products");
      for (uint32_t lanes : {16u, 8u, 4u})
        for (uint32_t W : TIMED_WIDTHS) {
          std::vector<uint8_t> act((size_t)p.ng * W * A8_ROW);
          for (uint32_t i = 0; i < p.ng * W; i++) {
            const auto x = randn(rng, D41_OG_K);
            const auto a8 = a8_row(x.data(), D41_OG_K);
            std::copy(a8.begin(), a8.end(), &act[(size_t)i * A8_ROW]);
          }
          io.put(oa, act);
          std::printf("time ds_output_groups %s lanes %u, 4 groups, W=%u: %.2f us\n", ggml_type_label(type), lanes, W,
                      time_launch(rt, 0, [&](ze::List &L) { launch_groups(L, type, lanes, p.dA, o.bpr_a, W, p.ng, none); }));
        }
      // Two slices in turn, as a step reads weights: 38 MB of attn_output_a in Q8_0, 28 MB in Q6_K, and more of
      // attn_output_b, against the card's 18 MiB cache.
      uint32_t at = 0;
      for (uint32_t lanes : {16u, 8u, 4u})
        for (uint32_t W : TIMED_WIDTHS)
          std::printf("time ds_output_groups %s lanes %u, 4 groups, weights cold, W=%u: %.2f us\n", ggml_type_label(type), lanes, W,
                      time_launch(rt, 0, [&](ze::List &L) {
                        launch_groups(L, type, lanes, parts[at++ % 2].dA, o.bpr_a, W, 4, none);
                      }));
      const uint32_t K = p.ng * D41_OLR, rows_b = D41_EMBD, bprb = K / quant_block_values(type);
      const uint32_t stride = K + K / 8, ytok = rows_b;
      const auto launch_b = [&](ze::List &L, uint32_t lanes, const uint8_t *dB, uint32_t W) {
        const uint32_t rps = 16 / lanes;
        L.launch(lib.get(io.card, module("matvec_", type, lanes), "matvec"), {dB, rows_b, bprb, aa, K, stride, W, yb, ytok, grids},
                 (rows_b + 4 * rps - 1) / (4 * rps), 1, 1, 64);
      };
      for (uint32_t lanes : {16u, 8u, 4u})
        for (uint32_t W : TIMED_WIDTHS) {
          std::printf("time matvec attn_output_b's 4,096 columns %s lanes %u, W=%u: %.2f us\n", ggml_type_label(type), lanes, W,
                      time_launch(rt, 0, [&](ze::List &L) { launch_b(L, lanes, p.dB, W); }));
          std::printf("time matvec attn_output_b's 4,096 columns %s lanes %u, weights cold, W=%u: %.2f us\n", ggml_type_label(type), lanes, W,
                      time_launch(rt, 0, [&](ze::List &L) { launch_b(L, lanes, parts[at++ % 2].dB, W); }));
        }
      log_clocks("after timing the output products");
      for (Part &pp : parts) {
        rt.free(pp.dA);
        rt.free(pp.dB);
      }
    }
  }

  // ---------------------------------------------------------------- attention timing
  void bench_attn() {
    struct Case {
      const char *what;
      uint32_t last, ratio;
      bool one_slot;
    };
    const Case cases[] = {{"640 keys, ratio 1, one slot's keys", 16383, 1, true},
                          {"640 keys, ratio 1, each row's slot", 16383, 1, false},
                          {"640 keys, ratio 2, one slot's keys", 16383, 2, true},
                          {"128 keys, window only, ratio 0", 16383, 0, true},
                          {"192 keys, ratio 2, n_vis 64", 127, 2, true}};
    const auto hs = randn(rng, D41_H);
    io.put(sinks, hs);
    {
      const auto hq = randn(rng, WMAX * Q_ALL);
      io.put(q, hq);
    }
    auto set = [&](const Case &c, uint32_t W) {
      const auto hrows = make_rows(W, c.last, c.one_slot ? 1 : W, W);
      io.put(rows, hrows);
      io.put(idx, make_idx(hrows, W, c.ratio));
    };
    set(cases[0], 1);
    time_launch(rt, 0, [&](ze::List &L) { launch_split(L, 1, 32, 1, pt); });
    log_clocks("before timing attention");
    for (const Case &c : cases)
      for (uint32_t W : TIMED_WIDTHS) {
        set(c, W);
        launch_split(io.L, W, 32, c.ratio, pt);
        io.L.synchronize();
        std::printf("time ds_attn_split 32 heads, %s, W=%u: %.2f us\n", c.what, W,
                    time_launch(rt, 0, [&](ze::List &L) { launch_split(L, W, 32, c.ratio, pt); }));
        std::printf("time ds_attn_merge 32 heads, %s, W=%u: %.2f us\n", c.what, W,
                    time_launch(rt, 0, [&](ze::List &L) { launch_merge(L, W, 0, 32, c.ratio, none); }));
      }
    for (uint32_t W : TIMED_WIDTHS) {
      set(cases[0], W);
      std::printf("time ds_attn_split 64 heads, 640 keys, W=%u: %.2f us\n", W,
                  time_launch(rt, 0, [&](ze::List &L) { launch_split(L, W, 64, 1, pt); }));
    }
    log_clocks("after timing attention");
    // A launch from cold keys: both slices of weights, 27.5 MB, read before each launch, against the card's 18 MiB cache.
    // The weights' own time is timed apart and taken off.
    const auto w = make_out(QT_Q6_K);
    uint8_t *slices[2];
    for (uint32_t g = 0; g < 2; g++)
      slices[g] = upload_slice(QT_Q6_K, w.qa.data(), w.rowa, (uint64_t)g * 4 * D41_OLR, (uint64_t)4 * D41_OLR, 0, w.bpr_a);
    const auto evict = [&](ze::List &L) {
      for (const uint8_t *slice : slices) launch_groups(L, QT_Q6_K, 8, slice, w.bpr_a, 1, 4, none);
    };
    log_clocks("before timing attention from cold keys");
    for (const Case &c : {cases[0], cases[1]})
      for (uint32_t W : TIMED_WIDTHS) {
        set(c, W);
        const double base = time_launch(rt, 0, evict);
        const double both = time_launch(rt, 0, [&](ze::List &L) {
          evict(L);
          launch_split(L, W, 32, 1, pt);
        });
        std::printf("time ds_attn_split 32 heads, %s, cold keys, W=%u: %.2f us (%.2f us with the weights, %.2f us alone)\n",
                    c.what, W, both - base, both, base);
      }
    for (uint32_t g = 0; g < 2; g++) rt.free(slices[g]);
    log_clocks("after timing attention from cold keys");
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
  b.test_attn();
  b.check_no_of();
  b.bench_attn();
  b.test_and_time_output();
  return b.ck.report();
}
