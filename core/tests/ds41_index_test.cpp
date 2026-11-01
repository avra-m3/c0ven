// Level 3 (docs/TESTING.md): deepseek41's ds_idx_score and ds_idx_topk (kernels/models/deepseek41/attn.cl) on random data
// against ref_ops and ggml's lightning indexer and top-k: n from 1 to 16,384 at ratios 1 and 2, shuffled, absent and
// missing pages, padded and mixed rows, built ties, clustered and narrow scores, last rows past a multiple of 4 that
// rank at the top; and each kernel's time at 1, 2, 4 and 8 rows with every card visible. Limits: 1e-4 of the largest
// |reference| for scores; exact sets where the k-th score is clear of the next by more than the scores' error, and
// otherwise only swaps within it.
#include <cmath>
#include <numeric>

#include "ds41_kernel_test.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "models/deepseek41/layout.h"
#include "models/deepseek41/ref_ops.h"

using namespace engine;
using namespace ds41test;
namespace ref = engine::deepseek41::ref;

namespace {

constexpr size_t D = D41_IDX_D, QV = D41_IDX_H * D, CR = D41_COMP_ROW, CAP = D41_CTX_CAP;
constexpr uint32_t WMAX = STEP_MAX_ROWS, SLOTS = 16, NPP = CAP / KV_PAGE_CELLS, NPAGES = SLOTS * NPP;
constexpr size_t POOL_VALUES = (size_t)NPAGES * KV_PAGE_CELLS * CR;
constexpr float SCORE_UNTOUCHED = -7.5f;
constexpr uint32_t IDX_UNTOUCHED = 0xabababab;
constexpr double SCORE_LIMIT = 1e-4;
// Scores closer than this share of the largest may swap between f32 and double sums.
constexpr double SET_BAND = 1e-5;
const void *const none = nullptr;

// Where row j lies in the pool, stated apart from d41_comp_index, which the kernel calls: a page holds 4 positions.
constexpr u64 pool_index(uint32_t page, uint32_t j, uint32_t ratio) { return (u64)page * (4 / ratio) + (j * ratio) % 4 / ratio; }
static_assert(pool_index(5, 3, 2) == d41_comp_index(5, 3, 2) && pool_index(5, 7, 1) == d41_comp_index(5, 7, 1));

// How one row's keys and weights are drawn.
enum class Kind { spread, ties, clustered, narrow, zero_weights };

std::vector<float> ggml_scores(const float *iq_rotated, const float *w, const uint16_t *keys, uint32_t n) {
  ggml_init_params ip = {(size_t)n * 512 + (64u << 20), nullptr, false};
  ggml_context *ctx = ggml_init(ip);
  ggml_tensor *q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, D41_IDX_H, 1, 1);
  ggml_tensor *k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, 1, n, 1);
  ggml_tensor *tw = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D41_IDX_H, 1, 1, 1);
  ggml_tensor *m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n, 1, 1, 1);
  std::memcpy(q->data, iq_rotated, QV * 4);
  std::memcpy(k->data, keys, (size_t)n * D * 2);
  for (uint32_t h = 0; h < D41_IDX_H; h++) ((float *)tw->data)[h] = w[h] * (1.0f / 64.0f);  // ggml's weights are prescaled
  std::memset(m->data, 0, (size_t)n * 2);
  ggml_tensor *out = ggml_lightning_indexer(ctx, q, k, tw, m);
  ggml_cgraph *gf = ggml_new_graph(ctx);
  ggml_build_forward_expand(gf, out);
  EXPECT(ggml_graph_compute_with_ctx(ctx, gf, 1) == GGML_STATUS_SUCCESS, "ggml's graph failed");
  std::vector<float> v((const float *)out->data, (const float *)out->data + n);
  ggml_free(ctx);
  return v;
}

// ggml's k largest, increasing. Its order among equal values is not fixed, so only tie-free data compares with it.
std::vector<uint32_t> ggml_top(const float *s, uint32_t n, uint32_t k) {
  ggml_init_params ip = {(size_t)n * 64 + (16u << 20), nullptr, false};
  ggml_context *ctx = ggml_init(ip);
  ggml_tensor *a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
  std::memcpy(a->data, s, (size_t)n * 4);
  ggml_tensor *out = ggml_top_k(ctx, a, (int)k);
  ggml_cgraph *gf = ggml_new_graph(ctx);
  ggml_build_forward_expand(gf, out);
  EXPECT(ggml_graph_compute_with_ctx(ctx, gf, 1) == GGML_STATUS_SUCCESS, "ggml's graph failed");
  std::vector<uint32_t> v((const uint32_t *)out->data, (const uint32_t *)out->data + k);
  ggml_free(ctx);
  std::sort(v.begin(), v.end());
  return v;
}

struct Bench {
  ze::Runtime &rt;
  ze::Library &lib;
  Io &io;
  std::mt19937 rng{41};
  Checker ck;
  float *iq, *w, *score;
  uint16_t *pool;
  uint32_t *pt, *idx;
  D41Row *rows;
  std::vector<uint32_t> shuffled;
  std::vector<uint16_t> hpool;
  // Rows checked with a tie-free top-k boundary, and with one inside the scores' error.
  uint32_t exact_rows = 0, band_rows = 0;

  Bench(ze::Runtime &r, ze::Library &l, Io &i) : rt(r), lib(l), io(i) {
    iq = io.dev<float>(WMAX * QV);
    w = io.dev<float>(WMAX * D41_IDX_H);
    score = io.dev<float>(WMAX * CAP);
    pool = io.dev<uint16_t>(POOL_VALUES);
    pt = io.dev<uint32_t>(NPAGES);
    idx = io.dev<uint32_t>(WMAX * D41_TOPK);
    rows = io.dev<D41Row>(SLOTS + WMAX);
    hpool.resize(POOL_VALUES);
    shuffled.resize(NPAGES);
    std::iota(shuffled.begin(), shuffled.end(), 0u);
    std::shuffle(shuffled.begin(), shuffled.end(), rng);
  }
  ze_kernel_handle_t k_score() { return lib.get(io.card, "deepseek41_attn", "ds_idx_score"); }
  ze_kernel_handle_t k_topk() { return lib.get(io.card, "deepseek41_attn", "ds_idx_topk"); }
  void launch_score(ze::List &L, uint32_t W, uint32_t ratio, const void *ppt, const D41Row *prows = nullptr) {
    L.launch(k_score(), {iq, w, prows ? prows : rows, ratio, ppt, NPP, pool, score}, D41_IDX_WGS, W, 1, D41_IDX_WG_ROWS);
  }
  void launch_topk(ze::List &L, uint32_t W, uint32_t ratio, const D41Row *prows = nullptr) {
    L.launch(k_topk(), {score, prows ? prows : rows, ratio, idx}, W, 1, 1, 1024);
  }

  // A slot's n rows of index keys, as f16.
  std::vector<uint16_t> make_keys(Kind kind, uint32_t n) {
    std::vector<uint16_t> keys((size_t)n * D);
    auto put = [&](size_t j, const float *v) {
      for (size_t d = 0; d < D; d++) keys[j * D + d] = ggml_fp32_to_fp16(v[d]);
    };
    if (kind == Kind::spread || kind == Kind::zero_weights) {
      const auto v = randn(rng, (size_t)n * D);
      for (uint32_t j = 0; j < n; j++) put(j, &v[j * D]);
    } else if (kind == Kind::ties) {
      // Rows drawn from few distinct keys, so equal scores are built, not chance.
      const auto base = randn(rng, 32 * D);
      std::uniform_int_distribution<uint32_t> pick(0, 31);
      for (uint32_t j = 0; j < n; j++) put(j, &base[pick(rng) * D]);
    } else {
      const uint32_t centres = kind == Kind::clustered ? 8 : 1;
      const float noise = kind == Kind::clustered ? 0.02f : 0.002f;
      const auto base = randn(rng, centres * D);
      std::uniform_int_distribution<uint32_t> pick(0, centres - 1);
      std::normal_distribution<float> nd(0.0f, noise);
      std::vector<float> v(D);
      for (uint32_t j = 0; j < n; j++) {
        const float *c = &base[pick(rng) * D];
        for (size_t d = 0; d < D; d++) v[d] = c[d] + nd(rng);
        put(j, v.data());
      }
    }
    return keys;
  }

  // Places a slot's keys at its pages in the host's copy of the pool.
  void place_keys(uint32_t slot, uint32_t ratio, bool pages, const std::vector<uint16_t> &keys, uint32_t n) {
    for (uint32_t j = 0; j < n; j++) {
      const uint32_t own = slot * NPP + j * ratio / KV_PAGE_CELLS, page = pages ? shuffled[own] : own;
      std::copy_n(&keys[(size_t)j * D], D, &hpool[pool_index(page, j, ratio) * CR + D41_HD]);
    }
  }

  // Makes rows from 4 * (n / 4) on rank at the top, as chance rarely does: idx_select loads them one at a time. Copies
  // of the best row tie with it under Kind::ties and otherwise scale it, as a score is linear in its key.
  void raise_tail(std::vector<uint16_t> &keys, uint32_t n, const float *q_rot, const float *row_w, Kind kind) {
    std::vector<float> k32((size_t)n * D), s(n);
    for (size_t i = 0; i < k32.size(); i++) k32[i] = ggml_fp16_to_fp32(keys[i]);
    ref::index_score(q_rot, row_w, k32.data(), D, n, s.data());
    const size_t best = std::max_element(s.begin(), s.end()) - s.begin();
    for (uint32_t j = n / 4 * 4; j < n; j++) {
      const float scale = kind == Kind::ties ? 1.0f : 1.25f + 0.25f * (float)(j % 4);
      for (size_t d = 0; d < D; d++) keys[j * D + d] = ggml_fp32_to_fp16(k32[best * D + d] * scale);
    }
  }

  // One launch of both kernels over rows with these slots and counts. n[r] of 0 at ratio 2 is position 0; a slot of
  // D41_NO_ROW pads. Each row's keys are drawn anew; top_tail raises their last rows.
  void run(uint32_t ratio, bool pages, bool absent_pages, const std::vector<uint32_t> &slots,
           const std::vector<uint32_t> &ns, Kind kind, bool top_tail = false) {
    const uint32_t W = (uint32_t)slots.size();
    std::vector<D41Row> hrows(WMAX);
    std::vector<std::vector<uint16_t>> keys(W);
    auto hq = randn(rng, W * QV, 0.3f);
    auto hw = randn(rng, W * D41_IDX_H, 2.0f);
    if (kind == Kind::zero_weights) std::fill(hw.begin(), hw.end(), 0.0f);
    std::vector<uint32_t> table(NPAGES, D41_NO_PAGE);
    for (uint32_t p = 0; p < NPAGES; p++) table[p] = pages ? shuffled[p] : p;
    for (uint32_t r = 0; r < W; r++) {
      if (slots[r] == D41_NO_ROW) {
        hrows[r] = {D41_NO_ROW, 0, 0, D41_NO_ROW};
        continue;
      }
      const uint32_t n = ns[r];
      // Even and odd positions give the same n at ratio 2; both are used. n = 0 only at ratio 2, position 0.
      const uint32_t pos = n == 0 ? 0 : ratio == 2 && n * 2 < CAP && (r & 1) ? n * 2 : n * ratio - 1;
      hrows[r] = {slots[r], pos, pos % D41_RING, D41_NO_ROW};
      keys[r] = make_keys(kind, n);
      if (top_tail) {
        std::vector<float> q_rot(&hq[r * QV], &hq[r * QV] + QV);
        for (uint32_t h = 0; h < D41_IDX_H; h++) ref::rope_tail(&q_rot[h * D], D, pos, true, false);
        raise_tail(keys[r], n, q_rot.data(), &hw[r * D41_IDX_H], kind);
      }
      place_keys(slots[r], ratio, pages, keys[r], n);
      if (absent_pages)
        for (uint32_t p = (n * ratio + KV_PAGE_CELLS - 1) / KV_PAGE_CELLS; p < NPP; p++) table[slots[r] * NPP + p] = D41_NO_PAGE;
    }
    io.put(pool, hpool.data(), POOL_VALUES);
    const bool use_pt = pages || absent_pages;
    if (use_pt) io.put(pt, table);
    io.put(rows, hrows);
    io.put(iq, hq);
    io.put(w, hw);
    std::vector<float> sfill(WMAX * CAP, SCORE_UNTOUCHED);
    std::vector<uint32_t> ifill(WMAX * D41_TOPK, IDX_UNTOUCHED);
    io.put(score, sfill);
    io.put(idx, ifill);
    launch_score(io.L, W, ratio, use_pt ? (const void *)pt : none);
    io.L.barrier();
    launch_topk(io.L, W, ratio);
    io.L.synchronize();
    const auto gs = io.get(score, WMAX * CAP);
    const auto gi = io.get(idx, WMAX * D41_TOPK);

    for (uint32_t r = 0; r < WMAX; r++) {
      const float *gr = &gs[r * CAP];
      const uint32_t *ir = &gi[(size_t)r * D41_TOPK];
      const bool real = r < W && slots[r] != D41_NO_ROW;
      const uint32_t n = real ? ns[r] : 0;
      // Nothing past the last sub-group holding a row, and nothing of a padded or absent row.
      const size_t written = (n + 15u) / 16u * 16u;
      ck.exact("score past the rows, and padded rows, untouched", gr + written, &sfill[written], (CAP - written) * 4);
      const uint32_t k = std::min<uint32_t>(D41_TOPK, n);
      ck.exact("top-k entries past k, and padded rows, untouched", ir + k, &ifill[k], (D41_TOPK - k) * 4);
      if (n == 0) continue;

      std::vector<float> q_rot(&hq[r * QV], &hq[r * QV] + QV), k32((size_t)n * D), want(n);
      for (uint32_t h = 0; h < D41_IDX_H; h++) ref::rope_tail(&q_rot[h * D], D, hrows[r].pos, true, false);
      for (size_t i = 0; i < k32.size(); i++) k32[i] = ggml_fp16_to_fp32(keys[r][i]);
      ref::index_score(q_rot.data(), &hw[r * D41_IDX_H], k32.data(), D, n, want.data());
      ck.rel(n < D41_TOPK ? "score against ref_ops, n below 512" : "score against ref_ops, n from 512", gr, want.data(), n, SCORE_LIMIT);
      // ggml once a launch: it is the same sum as ref_ops, and the sizes matter more than the rows.
      std::vector<float> want_g;
      if (r == 0) {
        want_g = ggml_scores(q_rot.data(), &hw[r * D41_IDX_H], keys[r].data(), n);
        ck.rel(n < D41_TOPK ? "score against ggml, n below 512" : "score against ggml, n from 512", gr, want_g.data(), n, SCORE_LIMIT);
        // The references' own gap: weights of both signs can cancel a lone score far below its terms.
        ck.rel(n < D41_TOPK ? "ggml against ref_ops, n below 512" : "ggml against ref_ops, n from 512", want_g.data(), want.data(), n, SCORE_LIMIT);
      }

      // The kernel's own scores give exactly ref_ops' choice, ties to the lower index.
      const std::vector<uint32_t> own = ref::top_k(gr, n, D41_TOPK);
      ck.exact("top-k from the kernel's scores: ties to the lower index", ir, own.data(), k * 4);
      for (uint32_t i = 1; i < k; i++) ck.note("top-k indices increase", ir[i] > ir[i - 1] ? 0 : 1, 0);
      if (top_tail && kind != Kind::ties) {
        const uint32_t tail = n % 4;
        double missing = tail;
        for (uint32_t i = k - tail; i < k; i++) missing -= ir[i] == n - k + i;
        ck.note("top-k holds the raised last rows", missing, 0);
      }

      // Against the references' scores: the same set, except for rows the sums' rounding can swap.
      const std::vector<uint32_t> rs = ref::top_k(want.data(), n, D41_TOPK);
      double mx = 0;
      for (uint32_t j = 0; j < n; j++) mx = std::max(mx, std::fabs((double)want[j]));
      const double band = SET_BAND * mx;
      if (n > k) {
        std::vector<float> sorted(want);
        std::nth_element(sorted.begin(), sorted.begin() + k, sorted.end(), std::greater<float>());
        const float below = sorted[k];  // the (k + 1)-th largest
        const float kth = *std::min_element(sorted.begin(), sorted.begin() + k);
        std::vector<uint8_t> in_got(n, 0), in_ref(n, 0);
        for (uint32_t i = 0; i < k; i++) {
          in_got[ir[i]] = 1;
          in_ref[rs[i]] = 1;
        }
        double out_of_band = 0, swapped = 0;
        for (uint32_t j = 0; j < n; j++)
          if (in_got[j] != in_ref[j]) {
            swapped++;
            if (std::fabs((double)want[j] - (double)kth) > band) out_of_band++;
          }
        ck.note("top-k against ref_ops' scores: rows swapped outside the scores' error", out_of_band, 0);
        if (kind == Kind::ties || (double)kth - (double)below > band) {
          exact_rows++;
          ck.note("top-k against ref_ops' scores: rows swapped, tie-free", swapped, 0);
        } else {
          band_rows++;
        }
        if (r == 0 && kind == Kind::spread && (double)kth - (double)below > band) {
          const std::vector<uint32_t> gt = ggml_top(want_g.data(), n, k);
          ck.exact("top-k against ggml's top-k, tie-free", ir, gt.data(), k * 4);
        }
      } else {
        ck.exact("top-k with n at most k is every row", ir, rs.data(), k * 4);
      }
    }
  }

  void test() {
    // 4,096 is the first n with idx_select's threshold, 8,193 the first with wide passes, and 12,345 mixes loads of 4
    // values and of one in its final scans, runs of 13.
    const uint32_t NS[] = {1, 511, 512, 513, 4096, 8192, 8193, 12345, 16384};
    for (uint32_t ratio : {1u, 2u})
      for (uint32_t n : NS) {
        if (n * ratio > CAP) continue;
        const bool big = n > 1024;
        for (Kind kind : {Kind::spread, Kind::ties, Kind::clustered, Kind::narrow, Kind::zero_weights}) {
          run(ratio, true, false, {0}, {n}, kind);
          if (!big || kind == Kind::spread) run(ratio, false, false, {1, 2}, {n, n}, kind);
        }
        run(ratio, true, true, {3, 4, 5}, {n, n, n}, Kind::spread);
      }
    for (uint32_t ratio : {1u, 2u})
      for (uint32_t n : {513u, 4097u, 8193u, 12345u, 16383u}) {
        if (n * ratio > CAP) continue;
        for (Kind kind : {Kind::spread, Kind::ties}) run(ratio, true, false, {0}, {n}, kind, true);
      }
    // Mixed rows in one launch: counts on both sides of the group and k boundaries, padding, and n = 0 at ratio 2.
    std::uniform_int_distribution<uint32_t> any(1, 4000);
    for (uint32_t ratio : {1u, 2u})
      for (uint32_t W : WIDTHS)
        for (bool pages : {false, true}) {
          std::vector<uint32_t> s, n;
          for (uint32_t r = 0; r < W; r++) {
            s.push_back(r + 8);
            n.push_back(r == 0 ? 63 + (W & 1) * 2 : r == 1 ? 64 : any(rng));
          }
          run(ratio, pages, false, s, n, Kind::spread);
          if (W > 1) {
            s[W - 1] = D41_NO_ROW;
            if (ratio == 2) n[0] = 0;
            run(ratio, pages, false, s, n, Kind::clustered);
          }
        }
    std::printf("rows checked for an exact top-k set: %u; rows whose k-th score lies within the error of the next: %u\n", exact_rows, band_rows);
  }

  // Every slot filled to the cap, for the timings.
  void fill_for_timing() {
    const auto base = randn(rng, CAP * D);
    std::vector<uint16_t> hb(CAP * D);
    for (size_t i = 0; i < hb.size(); i++) hb[i] = ggml_fp32_to_fp16(base[i]);
    for (uint32_t s = 0; s < SLOTS; s++) place_keys(s, 1, true, hb, CAP);
    io.put(pool, hpool.data(), POOL_VALUES);
    io.put(pt, shuffled);
    io.put(iq, randn(rng, WMAX * QV, 0.3f));
    io.put(w, randn(rng, WMAX * D41_IDX_H, 2.0f));
  }

  void set_rows(uint32_t n, uint32_t ratio, bool one_slot = false) {
    std::vector<D41Row> hrows(SLOTS + WMAX);
    for (uint32_t r = 0; r < SLOTS + WMAX; r++) hrows[r] = {one_slot ? 0u : r % SLOTS, n * ratio - 1, (n * ratio - 1) % D41_RING, D41_NO_ROW};
    io.put(rows, hrows);
  }

  void bench() {
    fill_for_timing();
    set_rows(CAP, 1);
    time_launch(rt, 0, [&](ze::List &L) { launch_score(L, 1, 1, pt); });
    log_clocks("before timing");
    struct Case {
      const char *what;
      uint32_t n, ratio;
      bool one_slot;
    };
    // Each row its own slot, reread each launch, as layers 24, 28, 32 and 36 reread layer 20's: 8 rows at the cap are
    // 33.5 MB of keys, more than the card's cache holds (docs/measurements/LEVEL_ZERO.md row 27). Every row on one slot
    // is 4 MiB, so those keys stay cached at any W.
    const Case score_cases[] = {{"n=16384, ratio 1, one slot's keys", 16384, 1, true},
                                {"n=16384, ratio 1, each row's slot reread", 16384, 1, false},
                                {"n=8192, ratio 2, one slot's keys", 8192, 2, true},
                                {"n=8192, ratio 2, each row's slot reread", 8192, 2, false},
                                {"n=512, ratio 1", 512, 1, false},
                                {"n=1, ratio 1", 1, 1, false}};
    for (const Case &c : score_cases) {
      set_rows(c.n, c.ratio, c.one_slot);
      for (uint32_t W : TIMED_WIDTHS)
        std::printf("time ds_idx_score %s W=%u: %.2f us\n", c.what, W, time_launch(rt, 0, [&](ze::List &L) { launch_score(L, W, c.ratio, pt); }));
    }
    // A new slot each launch: 67 MB of keys a cycle, more than the card's cache holds, so the keys come from its memory.
    set_rows(CAP, 1);
    for (uint32_t W : TIMED_WIDTHS) {
      uint32_t next = 0;
      const double us = time_launch(rt, 0, [&](ze::List &L) {
        launch_score(L, W, 1, pt, rows + next);
        next = (next + W) % SLOTS;
      });
      std::printf("time ds_idx_score n=16384, ratio 1, a new slot each launch W=%u: %.2f us\n", W, us);
    }
    // Scores as the kernel leaves them, then narrow ones.
    set_rows(CAP, 1);
    launch_score(io.L, WMAX, 1, pt);
    io.L.synchronize();
    const Case topk_cases[] = {{"n=16384, ratio 1", 16384, 1, false}, {"n=8192, ratio 2", 8192, 2, false}, {"n=513, ratio 1", 513, 1, false}, {"n=512, ratio 1", 512, 1, false}};
    for (const Case &c : topk_cases) {
      set_rows(c.n, c.ratio);
      for (uint32_t W : TIMED_WIDTHS)
        std::printf("time ds_idx_topk %s W=%u: %.2f us\n", c.what, W, time_launch(rt, 0, [&](ze::List &L) { launch_topk(L, W, c.ratio); }));
    }
    {
      std::normal_distribution<float> nd(1000.0f, 0.01f);
      std::vector<float> narrow(WMAX * CAP);
      for (float &x : narrow) x = nd(rng);
      io.put(score, narrow);
      set_rows(CAP, 1);
      for (uint32_t W : TIMED_WIDTHS)
        std::printf("time ds_idx_topk n=16384, narrow scores W=%u: %.2f us\n", W, time_launch(rt, 0, [&](ze::List &L) { launch_topk(L, W, 1); }));
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
