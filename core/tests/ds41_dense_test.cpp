// Level 3 (docs/TESTING.md): deepseek41's dense products at the file's types and the shapes one card holds, on random
// weights: attn_kv and attn_q_a whole in Q8_0, on both cards with the same bits; attn_q_b, engram_kv, the head and the
// shared experts in Q5_K, the shared experts cut 1,280/1,024. Each against the dequantised weights' product at 1, 2, 3
// and 8 rows, by lanes a row and K split in the work-group, with each one's time at 1, 2, 4 and 8 rows and its time with
// the weights cold, every card visible. Then the router's product, ffn_gate_inp in BF16 on the stream's bf16 form, by K
// split, against ggml's MUL_MAT. Limits: 1e-5 of the largest |reference|, the router's 1e-4 as for the other BF16
// products. No model is needed.
#include <cmath>
#include <deque>
#include <string>
#include <thread>

#include "ds41_kernel_test.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "load/gguf.h"
#include "load/runs.h"
#include "models/deepseek41/layout.h"
#include "quant.h"

using namespace engine;
using namespace ds41test;

namespace {

constexpr uint32_t WMAX = STEP_MAX_ROWS;
constexpr double MV_LIMIT = 1e-5, BF16_LIMIT = 1e-4;
constexpr uint64_t COLD_BYTES = 40ull << 20;  // more than the card's 18 MiB cache
constexpr uint32_t THREADS = 16;

void in_parallel(uint64_t n, const std::function<void(uint64_t, uint64_t)> &f) {
  std::vector<std::thread> pool;
  for (uint32_t t = 0; t < THREADS; t++)
    pool.emplace_back([&, t] { f(n * t / THREADS, n * (t + 1) / THREADS); });
  for (auto &th : pool) th.join();
}

// A tensor's rows in the file's layout, of random weights.
struct Matrix {
  uint32_t type;
  uint64_t rows, K, row_bytes;
  std::vector<uint8_t> q;
};

Matrix make_matrix(uint32_t type, uint64_t rows, uint64_t K, uint32_t seed) {
  Matrix m{type, rows, K, ggml_row_size((ggml_type)type, K), {}};
  m.q.resize(rows * m.row_bytes);
  in_parallel(rows, [&](uint64_t r0, uint64_t r1) {
    std::mt19937 rng(seed + (uint32_t)r0);
    const auto f = randn(rng, (r1 - r0) * K, 0.02f);
    ggml_quantize_chunk((ggml_type)type, f.data(), &m.q[r0 * m.row_bytes], 0, (int64_t)(r1 - r0), (int64_t)K, nullptr);
  });
  return m;
}

// Rows of a matrix; a product's slices are gathered into one on the card.
struct Slice {
  const Matrix *m;
  uint64_t row0, rows;
};

struct Product {
  std::string name;
  uint32_t type;
  std::vector<Slice> slices;
  uint64_t c0, K;   // the columns, at a block's edge
  bool ks;          // a K split is swept: few rows
  bool both_cards;  // whole on both cards, so the bits must agree
};

std::string module(uint32_t type, uint32_t lanes) {
  std::string s = std::string("matvec_") + ggml_type_label(type) + "_l" + std::to_string(lanes);
  for (auto &c : s) c = (char)std::tolower(c);
  return s;
}

struct Bench {
  ze::Runtime &rt;
  ze::Library &lib;
  std::mt19937 rng{41};
  Checker ck;
  std::deque<std::string> names;  // Checker keeps a check's name by pointer
  Bench(ze::Runtime &r, ze::Library &l) : rt(r), lib(l) {}
  const char *name(const std::string &s) { return names.emplace_back(s).c_str(); }

  static uint64_t total_rows(const Product &p) {
    uint64_t n = 0;
    for (const Slice &s : p.slices) n += s.rows;
    return n;
  }

  std::vector<uint8_t> runs_of(const Product &p) {
    const uint64_t vb = quant_block_values(p.type), nb = p.K / vb, b0 = p.c0 / vb;
    std::vector<SliceSpec> specs;
    for (const Slice &s : p.slices) specs.push_back({s.m->q.data(), s.m->row_bytes, s.row0, s.rows, b0, nb});
    std::vector<uint8_t> runs(runs_bytes(p.type, total_rows(p) * nb));
    to_runs(p.type, specs, runs.data());
    return runs;
  }

  // y[t][r] for W activation rows, from the weights as ggml dequantises them.
  static std::vector<double> reference(const Product &p, const std::vector<double> &acts, uint32_t W) {
    const uint64_t rows = total_rows(p), K = p.K;
    std::vector<double> y(W * rows);
    auto traits = ggml_get_type_traits((ggml_type)p.type);
    in_parallel(rows, [&](uint64_t r0, uint64_t r1) {
      std::vector<float> row(p.slices[0].m->K);
      uint64_t at = 0;
      for (const Slice &s : p.slices) {
        for (uint64_t i = 0; i < s.rows; i++) {
          const uint64_t r = at + i;
          if (r < r0 || r >= r1) continue;
          traits->to_float(&s.m->q[(s.row0 + i) * s.m->row_bytes], row.data(), (int64_t)s.m->K);
          for (uint32_t t = 0; t < W; t++) {
            double acc = 0;
            for (uint64_t c = 0; c < K; c++) acc += (double)row[p.c0 + c] * acts[t * K + c];
            y[t * rows + r] = acc;
          }
        }
        at += s.rows;
      }
    });
    return y;
  }

  // lanes a row; ks 0: matvec, else matvec_ks with K split ks.
  void launch(ze::List &L, const Product &p, uint32_t card, const uint8_t *dW, const uint8_t *act, uint32_t W, float *y,
              const uint8_t *grids, uint32_t lanes, uint32_t ks) {
    const uint32_t rows = (uint32_t)total_rows(p), K = (uint32_t)p.K;
    const uint32_t bpr = K / quant_block_values(p.type), stride = K + K / 8, rps = 16 / lanes;
    if (ks == 0)
      L.launch(lib.get(card, module(p.type, lanes), "matvec"), {dW, rows, bpr, act, K, stride, W, y, rows, grids},
               (rows + 4 * rps - 1) / (4 * rps), 1, 1, 64);
    else
      L.launch(lib.get(card, module(p.type, lanes), "matvec_ks"),
               {dW, rows, bpr, act, K, stride, W, y, rows, grids, ks}, (rows + rps - 1) / rps, 1, 1, 16 * ks);
  }

  void run(const Product &p) {
    const uint64_t rows = total_rows(p), K = p.K, row_a8 = K + K / 8;
    const auto runs = runs_of(p);
    std::printf("%s: %s, %llu x %llu, %.1f MiB\n", p.name.c_str(), ggml_type_label(p.type), (unsigned long long)rows,
                (unsigned long long)K, (double)runs.size() / 1048576.0);
    struct Shape {
      uint32_t lanes, ks;
    };
    std::vector<Shape> shapes = {{16, 0}, {8, 0}, {4, 0}};
    if (p.ks)
      for (uint32_t lanes : {16u, 8u, 4u})
        for (uint32_t ks : {2u, 4u, 8u})
          if (K / 32 / lanes >= ks) shapes.push_back({lanes, ks});

    Io io0(rt, 0);
    uint8_t *grids = io0.dev<uint8_t>(QGRID_BYTES);
    io0.put(grids, codebooks().data(), QGRID_BYTES);
    uint8_t *dW = io0.dev<uint8_t>(runs.size());
    io0.put(dW, runs);
    uint8_t *act = io0.dev<uint8_t>(WMAX * row_a8);
    float *y = io0.dev<float>(WMAX * rows);

    std::vector<uint8_t> ha(WMAX * row_a8);
    std::vector<double> acts(WMAX * K);
    for (uint32_t t = 0; t < WMAX; t++) {
      const auto x = randn(rng, K, 1.0f + (float)t);
      const auto a8 = a8_row(x.data(), K);
      std::copy(a8.begin(), a8.end(), &ha[t * row_a8]);
      for (uint64_t c = 0; c < K; c++) {
        float d;
        std::memcpy(&d, &a8[K + 4 * (c / 32)], 4);
        acts[t * K + c] = (double)d * (int8_t)a8[c];
      }
    }
    io0.put(act, ha);
    const auto want = reference(p, acts, WMAX);

    const char *against = name(p.name);
    for (const Shape &s : shapes)
      for (uint32_t W : {1u, 2u, 3u, 8u}) {
        launch(io0.L, p, 0, dW, act, W, y, grids, s.lanes, s.ks);
        io0.L.synchronize();
        const auto got = io0.get(y, W * rows);
        for (uint32_t t = 0; t < W; t++)
          ck.rel(against, &got[t * rows], &want[t * rows], rows, MV_LIMIT);
      }

    if (p.both_cards && rt.ncards() > 1) {
      Io io1(rt, 1);
      uint8_t *grids1 = io1.dev<uint8_t>(QGRID_BYTES);
      io1.put(grids1, codebooks().data(), QGRID_BYTES);
      uint8_t *dW1 = io1.dev<uint8_t>(runs.size());
      io1.put(dW1, runs);
      uint8_t *act1 = io1.dev<uint8_t>(WMAX * row_a8);
      io1.put(act1, ha);
      float *y1 = io1.dev<float>(WMAX * rows);
      const char *same = name(p.name + ", both cards' bits");
      for (const Shape &s : shapes) {
        launch(io0.L, p, 0, dW, act, WMAX, y, grids, s.lanes, s.ks);
        io0.L.synchronize();
        launch(io1.L, p, 1, dW1, act1, WMAX, y1, grids1, s.lanes, s.ks);
        io1.L.synchronize();
        const auto g0 = io0.get(y, WMAX * rows), g1 = io1.get(y1, WMAX * rows);
        ck.exact(same, g0.data(), g1.data(), g0.size() * 4);
      }
      for (void *b : std::vector<void *>{grids1, dW1, act1, y1}) rt.free(b);
    }

    // Copies in turn, as a step reads each layer's own weights, so each launch misses the card's cache.
    const uint32_t copies = (uint32_t)std::max<uint64_t>(1, (COLD_BYTES + runs.size() - 1) / runs.size());
    std::vector<uint8_t *> cold;
    for (uint32_t i = 0; i < copies; i++) {
      cold.push_back(io0.dev<uint8_t>(runs.size()));
      io0.put(cold.back(), runs);
    }
    time_launch(rt, 0, [&](ze::List &L) { launch(L, p, 0, dW, act, 1, y, grids, 8, 0); });
    log_clocks("before timing");
    for (const Shape &s : shapes)
      for (uint32_t W : TIMED_WIDTHS) {
        const double hot = time_launch(rt, 0, [&](ze::List &L) { launch(L, p, 0, dW, act, W, y, grids, s.lanes, s.ks); });
        uint32_t at = 0;
        const double cool = time_launch(rt, 0, [&](ze::List &L) {
          launch(L, p, 0, cold[at++ % copies], act, W, y, grids, s.lanes, s.ks);
        });
        std::printf("time %s lanes %u, K split %u, W=%u: %.2f us, weights cold (%u copies) %.2f us\n", p.name.c_str(),
                    s.lanes, s.ks, W, hot, copies, cool);
      }
    log_clocks("after timing");
    for (uint8_t *c : cold) rt.free(c);
    for (void *b : std::vector<void *>{grids, dW, act, y}) rt.free(b);
  }
};

// ffn_gate_inp . x by ggml's MUL_MAT, as upstream's graph has it: BF16 weights and F32 rows, which ggml converts.
std::vector<float> ggml_router(const std::vector<uint16_t> &w, const std::vector<float> &x, uint32_t W) {
  ggml_init_params ip = {(size_t)(64u << 20) + w.size() * 2 + x.size() * 8, nullptr, false};
  ggml_context *ctx = ggml_init(ip);
  ggml_tensor *tw = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, D41_EMBD, D41_EXPERTS);
  ggml_tensor *tx = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D41_EMBD, W);
  std::copy(w.begin(), w.end(), (uint16_t *)tw->data);
  std::copy(x.begin(), x.begin() + (size_t)W * D41_EMBD, (float *)tx->data);
  ggml_tensor *r = ggml_mul_mat(ctx, tw, tx);
  ggml_cgraph *gf = ggml_new_graph(ctx);
  ggml_build_forward_expand(gf, r);
  EXPECT(ggml_graph_compute_with_ctx(ctx, gf, 1) == GGML_STATUS_SUCCESS, "ggml's graph failed");
  std::vector<float> y((const float *)r->data, (const float *)r->data + (size_t)W * D41_EXPERTS);
  ggml_free(ctx);
  return y;
}

// The router's product, whole on each card. mv_xmx_bf16 reads all 8 rows of xb, also past W.
struct Router {
  ze::Runtime &rt;
  ze::Library &lib;
  Checker &ck;
  std::mt19937 rng{7};
  static constexpr uint32_t N = D41_EXPERTS, K = D41_EMBD;

  void launch(ze::List &L, uint32_t card, const uint16_t *dw, const uint16_t *xb, const uint32_t *ntok, float *y,
              uint32_t ks) {
    const uint32_t zero = 0;
    const float one = 1.0f;
    const void *const none = nullptr;
    L.launch(lib.get(card, "common", "mv_xmx_bf16"), {dw, N, K, K, xb, ntok, y, N, zero, one, zero, ks, none}, N / 16, 1,
             1, 16 * ks);
  }

  void run() {
    std::vector<uint16_t> hw((size_t)N * K);
    {
      const auto f = randn(rng, hw.size(), 0.02f);
      for (size_t i = 0; i < hw.size(); i++) hw[i] = ggml_fp32_to_bf16(f[i]).bits;
    }
    struct Mem {
      Io io;
      uint16_t *w, *xb;
      uint32_t *ntok;
      float *y;
      Mem(ze::Runtime &r, uint32_t c, const std::vector<uint16_t> &hw_) : io(r, c) {
        w = io.dev<uint16_t>(hw_.size());
        io.put(w, hw_);
        xb = io.dev<uint16_t>((size_t)WMAX * K);
        ntok = io.dev<uint32_t>(1);
        y = io.dev<float>((size_t)WMAX * N);
      }
    };
    std::vector<Mem *> mem;
    for (uint32_t c = 0; c < rt.ncards(); c++) mem.push_back(new Mem(rt, c, hw));

    const uint32_t splits[] = {1, 2, 4, 8, 16, 32, 64};
    const float scales[] = {1.0f, 1e3f, 1e-3f};
    std::vector<float> marks((size_t)WMAX * N, 0.0f);
    std::memset(marks.data(), 0xab, marks.size() * 4);
    for (float scale : scales)
      for (uint32_t W : WIDTHS) {
        // The rows past W hold other values, which the product reads and must not use.
        const auto x = randn(rng, (size_t)WMAX * K, scale);
        std::vector<uint16_t> hb(x.size());
        for (size_t i = 0; i < x.size(); i++) hb[i] = ggml_fp32_to_bf16(x[i]).bits;
        const auto viag = ggml_router(hw, x, W);
        // The same product in double, on the rounded weights and rows.
        std::vector<double> dbl((size_t)W * N);
        in_parallel(N, [&](uint64_t n0, uint64_t n1) {
          for (uint64_t n = n0; n < n1; n++)
            for (uint32_t t = 0; t < W; t++) {
              double acc = 0;
              for (uint32_t c = 0; c < K; c++)
                acc += (double)ggml_bf16_to_fp32(ggml_bf16_t{hw[n * K + c]}) *
                       ggml_bf16_to_fp32(ggml_bf16_t{hb[(size_t)t * K + c]});
              dbl[t * N + n] = acc;
            }
        });
        ck.rel("ggml's router product against the same in double", viag.data(), dbl.data(), viag.size(), BF16_LIMIT);
        for (Mem *m : mem) {
          m->io.put(m->xb, hb);
          m->io.put(m->ntok, &W, 1);
        }
        for (uint32_t ks : splits) {
          std::vector<std::vector<float>> got;
          for (Mem *m : mem) {
            m->io.put(m->y, marks);
            launch(m->io.L, m->io.card, m->w, m->xb, m->ntok, m->y, ks);
            m->io.L.synchronize();
            got.push_back(m->io.get(m->y, (size_t)WMAX * N));
          }
          for (uint32_t t = 0; t < W; t++) {
            ck.rel("router against ggml's MUL_MAT", &got[0][t * N], &viag[t * N], N, BF16_LIMIT);
            ck.rel("router against the same in double", &got[0][t * N], &dbl[t * N], N, BF16_LIMIT);
          }
          if (got.size() > 1)
            ck.exact("router, both cards' bits", got[0].data(), got[1].data(), got[0].size() * 4);
          ck.exact("router writes nothing past W", &got[0][(size_t)W * N], &marks[(size_t)W * N],
                   (size_t)(WMAX - W) * N * 4);
        }
      }

    // Times, with the weights hot and cold (copies in turn, 40 MiB at least).
    Mem &m = *mem[0];
    const auto x = randn(rng, (size_t)WMAX * K);
    std::vector<uint16_t> hb(x.size());
    for (size_t i = 0; i < x.size(); i++) hb[i] = ggml_fp32_to_bf16(x[i]).bits;
    m.io.put(m.xb, hb);
    const uint32_t copies = (uint32_t)((COLD_BYTES + hw.size() * 2 - 1) / (hw.size() * 2));
    std::vector<uint16_t *> cold;
    for (uint32_t i = 0; i < copies; i++) {
      cold.push_back(m.io.dev<uint16_t>(hw.size()));
      m.io.put(cold.back(), hw);
    }
    time_launch(rt, 0, [&](ze::List &L) { launch(L, 0, m.w, m.xb, m.ntok, m.y, D41_RT_KS); });
    log_clocks("before timing the router's product");
    for (uint32_t ks : splits)
      for (uint32_t W : TIMED_WIDTHS) {
        m.io.put(m.ntok, &W, 1);
        const double hot = time_launch(rt, 0, [&](ze::List &L) { launch(L, 0, m.w, m.xb, m.ntok, m.y, ks); });
        uint32_t at = 0;
        const double cool = time_launch(rt, 0, [&](ze::List &L) { launch(L, 0, cold[at++ % copies], m.xb, m.ntok, m.y, ks); });
        std::printf("time mv_xmx_bf16 ffn_gate_inp, K split %u, W=%u: %.2f us, weights cold (%u copies) %.2f us\n", ks, W,
                    hot, copies, cool);
      }
    log_clocks("after timing the router's product");
    for (uint16_t *c : cold) rt.free(c);
    for (Mem *p : mem) delete p;
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

  const Matrix kv = make_matrix(QT_Q8_0, 512, D41_EMBD, 1);
  b.run({"attn_kv", QT_Q8_0, {{&kv, 0, 512}}, 0, D41_EMBD, true, true});
  const Matrix qa = make_matrix(QT_Q8_0, D41_QLR, D41_EMBD, 2);
  b.run({"attn_q_a", QT_Q8_0, {{&qa, 0, D41_QLR}}, 0, D41_EMBD, true, true});
  // The card's half of attn_q_b's heads.
  const Matrix qb = make_matrix(QT_Q5_K, D41_H * D41_HD, D41_QLR, 3);
  b.run({"attn_q_b, 32 heads", QT_Q5_K, {{&qb, D41_H * D41_HD / 2, D41_H * D41_HD / 2}}, 0, D41_QLR, false, false});
  // A card's rows of engram_kv: its two streams' keys, then its half of the value rows.
  const Matrix ek = make_matrix(QT_Q5_K, D41_ENG_KV, D41_ENG_G, 4);
  b.run({"engram_kv, keys 0-1 and values 0-2559", QT_Q5_K, {{&ek, 0, 2 * D41_EMBD}, {&ek, D41_HC_DIM, D41_EMBD / 2}}, 0,
         D41_ENG_G, false, false});
  b.run({"engram_kv, keys 2-3 and values 2560-5119", QT_Q5_K,
         {{&ek, 2 * D41_EMBD, 2 * D41_EMBD}, {&ek, D41_HC_DIM + D41_EMBD / 2, D41_EMBD / 2}}, 0, D41_ENG_G, false, false});
  // A card's half of the head's rows.
  const Matrix head = make_matrix(QT_Q5_K, D41_VOCAB / 2, D41_EMBD, 5);
  b.run({"output, half the vocabulary", QT_Q5_K, {{&head, 0, D41_VOCAB / 2}}, 0, D41_EMBD, false, false});
  // Shared gate and up: a card's rows as one matrix, gate's then up's; down by columns. The cut is 5 and 4 of the 9
  // Q5_K blocks in 2,304 values.
  const Matrix sg = make_matrix(QT_Q5_K, D41_FF, D41_EMBD, 6), su = make_matrix(QT_Q5_K, D41_FF, D41_EMBD, 7);
  const Matrix sd = make_matrix(QT_Q5_K, D41_EMBD, D41_FF, 8);
  b.run({"shared gate and up, part A", QT_Q5_K, {{&sg, 0, 1280}, {&su, 0, 1280}}, 0, D41_EMBD, true, false});
  b.run({"shared gate and up, part B", QT_Q5_K, {{&sg, 1280, 1024}, {&su, 1280, 1024}}, 0, D41_EMBD, true, false});
  b.run({"shared down, part A", QT_Q5_K, {{&sd, 0, D41_EMBD}}, 0, 1280, false, false});
  b.run({"shared down, part B", QT_Q5_K, {{&sd, 0, D41_EMBD}}, 1280, 1024, false, false});
  Router{rt, lib, b.ck}.run();
  return b.ck.report();
}
