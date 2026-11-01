// Level 3 (docs/TESTING.md): deepseek41's ds_compress (kernels/models/deepseek41/attn.cl) on random data at the model's
// shapes against ref_ops: streams of steps of 1-8 rows over 64 positions at ratios 2 and 1, from even and odd starts,
// rows of one slot and of several, padded rows, rejected drafts, shuffled pages and positions near the context's cap;
// and its time at 1, 2, 4 and 8 rows with every card visible. Limits: 1e-4 of the largest |reference|; the pool's f16
// exact from the kernel's own f32, the ring and every other pool row exactly as the host's model. No model is needed.
#include <cmath>
#include <map>

#include "ds41_kernel_test.h"
#include "ggml.h"
#include "models/deepseek41/layout.h"
#include "models/deepseek41/ref_ops.h"

using namespace engine;
using namespace ds41test;
namespace ref = engine::deepseek41::ref;

namespace {

constexpr size_t H = D41_HD, CR = D41_COMP_ROW, CRF = D41_COMP_ROWF;
constexpr uint32_t WMAX = STEP_MAX_ROWS, SLOTS = WMAX, NPP = 16384 / KV_PAGE_CELLS;
constexpr uint32_t NPAGES = SLOTS * NPP;
constexpr size_t POOL_VALUES = (size_t)NPAGES * KV_PAGE_CELLS * CR;  // enough at ratio 1, twice enough at ratio 2
constexpr size_t RING_VALUES = (size_t)SLOTS * D41_CRING * D41_CRING_CELL;
constexpr uint32_t LOCAL = 256;
constexpr uint16_t UNTOUCHED = 0xabab;
const void *const none = nullptr;

// Where the row at position first lies in the pool, stated apart from d41_comp_index, which the kernel calls: a page is
// 4 positions, so it holds 4 / ratio rows, in order of position.
constexpr u64 pool_index(uint32_t page, uint32_t first, uint32_t ratio) {
  return (u64)page * (4 / ratio) + first % 4 / ratio;
}
static_assert(pool_index(5, 6, 2) == 11 && pool_index(5, 7, 1) == 23 && pool_index(5, 4, 1) == 20);
static_assert(d41_comp_index(5, 3, 2) == 11 && d41_comp_index(5, 2, 2) == 10);
static_assert(d41_comp_index(5, 7, 1) == 23 && d41_comp_index(5, 4, 1) == 20);

// One position's inputs.
struct Token {
  std::vector<float> kv, sc;
};

struct Bench {
  ze::Runtime &rt;
  ze::Library &lib;
  Io &io;
  std::mt19937 rng{41};
  Checker ck;
  float *kv, *sc, *cr, *norm, *k_norm, *rowf;
  uint16_t *attn_k, *pool;
  uint32_t *pt;
  D41Row *rows;
  std::vector<float> hnorm, hk_norm, hak;
  std::vector<uint32_t> shuffled;

  Bench(ze::Runtime &r, ze::Library &l, Io &i) : rt(r), lib(l), io(i) {
    kv = io.dev<float>(WMAX * H);
    sc = io.dev<float>(WMAX * H);
    cr = io.dev<float>(RING_VALUES);
    norm = io.dev<float>(H);
    k_norm = io.dev<float>(D41_IDX_D);
    rowf = io.dev<float>(WMAX * CRF);
    attn_k = io.dev<uint16_t>(D41_IDX_D * H);
    pool = io.dev<uint16_t>(POOL_VALUES);
    pt = io.dev<uint32_t>(NPAGES);
    rows = io.dev<D41Row>(WMAX);
    hnorm = randn(rng, H, 0.5f);
    hk_norm = randn(rng, D41_IDX_D, 0.5f);
    // attn_k as the file holds it, in bf16; ref_ops reads the same rounded weights.
    hak = randn(rng, D41_IDX_D * H, 0.05f);
    std::vector<uint16_t> hb(hak.size());
    for (size_t j = 0; j < hak.size(); j++) {
      hb[j] = ggml_fp32_to_bf16(hak[j]).bits;
      hak[j] = ggml_bf16_to_fp32(ggml_bf16_t{hb[j]});
    }
    io.put(attn_k, hb);
    io.put(norm, hnorm);
    io.put(k_norm, hk_norm);
    shuffled.resize(NPAGES);
    for (uint32_t p = 0; p < NPAGES; p++) shuffled[p] = p;
    std::shuffle(shuffled.begin(), shuffled.end(), rng);
    io.put(pt, shuffled);
  }
  ze_kernel_handle_t k() { return lib.get(io.card, "deepseek41_attn", "ds_compress"); }
  void launch(ze::List &L, uint32_t W, uint32_t ratio, const void *ppt, const void *prow) {
    const void *psc = ratio == 2 ? (const void *)sc : none, *pcr = ratio == 2 ? (const void *)cr : none;
    L.launch(k(), {kv, psc, pcr, rows, ratio, norm, attn_k, k_norm, ppt, NPP, pool, prow}, W, 1, 1, LOCAL);
  }

  // A stream of steps, with the host's model of the pool and the ring.
  struct Stream {
    uint32_t ratio;
    bool pages;
    std::vector<uint16_t> pool;
    std::vector<float> ring;
    std::map<uint64_t, Token> tokens;  // by slot << 32 | position
  };

  Stream open(uint32_t ratio, bool pages) {
    Stream st{ratio, pages, std::vector<uint16_t>(POOL_VALUES, UNTOUCHED), std::vector<float>(RING_VALUES, -1.0f), {}};
    io.L.fill(pool, &UNTOUCHED, sizeof UNTOUCHED, POOL_VALUES * 2);
    io.L.barrier();
    io.put(cr, st.ring);
    return st;
  }

  // One step: positions[r] of slots[r]; D41_NO_ROW in slots pads. Data for a position already seen is replaced, as a
  // rejected draft's position is computed again.
  void step(Stream &st, const std::vector<uint32_t> &slots, const std::vector<uint32_t> &poss, float spread) {
    const uint32_t W = (uint32_t)slots.size();
    std::vector<D41Row> hrows(WMAX);
    std::vector<float> hkv(W * H), hsc(W * H);
    for (uint32_t r = 0; r < W; r++) {
      uint32_t prev = D41_NO_ROW;
      for (uint32_t q = 0; q < r; q++)
        if (slots[r] != D41_NO_ROW && slots[q] == slots[r] && poss[q] + 1 == poss[r]) prev = q;
      hrows[r] = {slots[r], poss[r], poss[r] % D41_RING, prev};
      Token t{randn(rng, H, 1.1f), randn(rng, H, spread)};
      std::copy(t.kv.begin(), t.kv.end(), &hkv[r * H]);
      std::copy(t.sc.begin(), t.sc.end(), &hsc[r * H]);
      if (slots[r] != D41_NO_ROW) st.tokens[(uint64_t)slots[r] << 32 | poss[r]] = std::move(t);
    }
    io.put(rows, hrows);
    io.put(kv, hkv);
    io.put(sc, hsc);
    launch(io.L, W, st.ratio, st.pages ? (const void *)pt : none, rowf);
    io.L.synchronize();

    auto grow = io.get(rowf, W * CRF);
    for (uint32_t r = 0; r < W; r++) {
      if (slots[r] == D41_NO_ROW) continue;
      const uint32_t slot = slots[r], p = poss[r];
      if (st.ratio == 2 && !(p & 1)) {
        const Token &t = st.tokens[(uint64_t)slot << 32 | p];
        size_t at = ((size_t)slot * D41_CRING + p % D41_CRING) * D41_CRING_CELL;
        std::copy(t.kv.begin(), t.kv.end(), &st.ring[at]);
        std::copy(t.sc.begin(), t.sc.end(), &st.ring[at + H]);
        continue;
      }
      const uint32_t first = st.ratio == 2 ? p & ~1u : p;
      std::vector<float> ikv, isc, lat(H), want(CR), want_key(CR);
      for (uint32_t t = 0; t < st.ratio; t++) {
        const Token &tk = st.tokens.at((uint64_t)slot << 32 | (first + t));
        ikv.insert(ikv.end(), tk.kv.begin(), tk.kv.end());
        isc.insert(isc.end(), tk.sc.begin(), tk.sc.end());
      }
      ref::compress(ikv.data(), isc.data(), st.ratio, hnorm.data(), lat.data());
      ref::compressed_row(lat.data(), hak.data(), hk_norm.data(), first, want.data());
      const char *ln = st.ratio == 2 ? "latent, ratio 2" : "latent, ratio 1";
      const char *kn = st.ratio == 2 ? "index key, ratio 2" : "index key, ratio 1";
      const float *got = &grow[r * CRF];
      ck.rel(ln, got, want.data(), H, 1e-4);
      // The key from the kernel's own latent: a value next to a bf16 rounding boundary may round either way between
      // kernel and ref_ops, and one such step of one activation moves the key by about the limit.
      ref::compressed_row(got + CR, hak.data(), hk_norm.data(), first, want_key.data());
      ck.rel(kn, got + H, &want_key[H], D41_IDX_D, 1e-4);
      const uint32_t page = st.pages ? shuffled[slot * NPP + first / KV_PAGE_CELLS] : slot * NPP + first / KV_PAGE_CELLS;
      const u64 idx = pool_index(page, first, st.ratio);
      for (size_t d = 0; d < CR; d++) st.pool[idx * CR + d] = ggml_fp32_to_fp16(got[d]);
    }
  }

  void close(Stream &st) {
    auto gp = io.get(pool, POOL_VALUES);
    ck.exact("pool f16 from its own f32, other rows untouched", gp.data(), st.pool.data(), POOL_VALUES * 2);
    auto gr = io.get(cr, RING_VALUES);
    ck.exact("compressor ring", gr.data(), st.ring.data(), RING_VALUES * 4);
  }

  // Steps of width W over one slot from start, through at least 64 positions.
  void one_slot(uint32_t ratio, bool pages, uint32_t W, uint32_t start, float spread) {
    Stream st = open(ratio, pages);
    // An odd start has its even partner from an earlier step.
    if (ratio == 2 && (start & 1)) step(st, {1}, {start - 1}, spread);
    for (uint32_t p = start; p < start + 64; p += W) {
      std::vector<uint32_t> s(W, 1), q(W);
      for (uint32_t r = 0; r < W; r++) q[r] = p + r;
      step(st, s, q, spread);
    }
    close(st);
  }

  // 8 rows of one slot, of which only some are accepted; the next step starts just past those, overwriting the rest.
  void rejected_drafts(uint32_t ratio, bool pages, uint32_t start) {
    Stream st = open(ratio, pages);
    std::uniform_int_distribution<uint32_t> accepted(1, 8);
    if (ratio == 2 && (start & 1)) step(st, {2}, {start - 1}, 3.0f);
    for (uint32_t p = start; p < start + 96;) {
      std::vector<uint32_t> s(8, 2), q(8);
      for (uint32_t r = 0; r < 8; r++) q[r] = p + r;
      step(st, s, q, 3.0f);
      p += accepted(rng);
    }
    close(st);
  }

  // Each step holds each slot's next rows, a different count a slot, then padded rows; a slot may have none.
  void several_slots(uint32_t ratio, bool pages) {
    Stream st = open(ratio, pages);
    constexpr uint32_t USED = 3;
    uint32_t next[USED] = {0, 1000, 16200};
    if (ratio == 2) step(st, {0, 1, 2}, {next[0]++, next[1]++, next[2]++}, 5.0f);
    std::uniform_int_distribution<uint32_t> take(0, 3);
    for (uint32_t it = 0; it < 40; it++) {
      std::vector<uint32_t> s, q;
      for (uint32_t c = 0; c < USED; c++) {
        uint32_t n = take(rng);
        while (n-- && s.size() < 7) {
          s.push_back(c);
          q.push_back(next[c]++);
        }
      }
      if (s.empty()) continue;
      const uint32_t real = (uint32_t)s.size();
      uint32_t W = 1;
      for (uint32_t w : WIDTHS)
        if (w >= real) {
          W = w;
          break;
        }
      while (s.size() < W) {
        s.push_back(D41_NO_ROW);
        q.push_back(0);
      }
      step(st, s, q, 5.0f);
    }
    close(st);
  }

  void test() {
    for (bool pages : {false, true}) {
      for (uint32_t W : WIDTHS) {
        one_slot(2, pages, W, 0, 3.0f);
        one_slot(2, pages, W, 1, 3.0f);
        one_slot(1, pages, W, 0, 3.0f);
      }
      // Near the cap, where the rotation's angles round; scores close together and far apart.
      one_slot(2, pages, 8, 16310, 0.1f);
      one_slot(2, pages, 8, 16311, 20.0f);
      one_slot(1, pages, 8, 16310, 3.0f);
      rejected_drafts(2, pages, 0);
      rejected_drafts(2, pages, 16261);
      rejected_drafts(1, pages, 0);
      several_slots(2, pages);
      several_slots(1, pages);
    }
  }

  // Timed rows of distinct slots at positions of one parity, or one slot's consecutive rows.
  void timed(const char *what, uint32_t ratio, uint32_t base, bool one_window) {
    for (uint32_t W : TIMED_WIDTHS) {
      std::vector<D41Row> hrows(WMAX);
      for (uint32_t r = 0; r < W; r++) {
        const uint32_t p = one_window ? base + r : base;
        hrows[r] = {one_window ? 0 : r, p, p % D41_RING, one_window && r ? r - 1 : D41_NO_ROW};
      }
      io.put(rows, hrows);
      std::printf("time ds_compress %s W=%u: %.2f us\n", what, W, time_launch(rt, 0, [&](ze::List &L) { launch(L, W, ratio, pt, none); }));
    }
  }

  void bench() {
    auto hk = randn(rng, WMAX * H), hs = randn(rng, WMAX * H, 3.0f);
    io.put(kv, hk);
    io.put(sc, hs);
    io.put(cr, randn(rng, RING_VALUES));
    {
      std::vector<D41Row> hrows(WMAX);
      hrows[0] = {0, 16001, 16001 % D41_RING, D41_NO_ROW};
      io.put(rows, hrows);
    }
    time_launch(rt, 0, [&](ze::List &L) { launch(L, 1, 2, pt, none); });
    log_clocks("before timing");
    timed("ratio 2, even positions: ring only", 2, 16000, false);
    timed("ratio 2, odd positions: all complete, partner in the ring", 2, 16001, false);
    timed("ratio 2, one slot's rows from an odd position", 2, 16001, true);
    timed("ratio 1: all complete", 1, 16000, false);
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
