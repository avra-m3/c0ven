// The DFlash 2 drafter (docs/MODEL_FILES.md, "The DFlash 2 drafter"; ADR-0050): five Qwen3-shaped layers that read
// the main model's hidden states at five layers and draft a block of positions in one pass. Split by heads and MLP
// rows as the draft layer is (docs/DESIGN.md, section 5), with two exchanges a layer; each slot keeps the drafter's K
// and V of its last DF_RING positions. Its BF16 weights come from the files tools/dflash_fetch.py writes.
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

#include "ggml-cpu.h"
#include "model_impl.h"

namespace engine::qwen4exp {

using namespace detail;

namespace {

float bf16f(uint16_t h) {
  uint32_t u = (uint32_t)h << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// One tensor of the drafter's directory: raw little-endian BF16, `elements` of them.
std::vector<uint16_t> df_tensor(const std::string &dir, const std::string &name, size_t elements) {
  std::string path = dir + "/" + name;
  int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) throw std::runtime_error("cannot open the DFlash 2 tensor " + path);
  std::vector<uint16_t> v(elements);
  size_t got = 0;
  while (got < elements * 2) {
    ssize_t n = ::read(fd, reinterpret_cast<uint8_t *>(v.data()) + got, elements * 2 - got);
    if (n <= 0) break;
    got += (size_t)n;
  }
  bool longer = false;
  if (got == elements * 2) {
    uint8_t extra;
    longer = ::read(fd, &extra, 1) > 0;
  }
  ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);  // read once a load: its pages back to the host (ADR-0032)
  ::close(fd);
  if (got != elements * 2 || longer) throw std::runtime_error("the DFlash 2 tensor " + name + " has the wrong size");
  return v;
}

}  // namespace

bool Model::init_dflash() {
  if (::access((opt_.mtp_dir + "/dflash-manifest.json").c_str(), R_OK) != 0) return false;
  df_on_ = true;
  dfcplan_ = static_cast<DraftPlan *>(rt_->alloc_host(sizeof(DraftPlan)));
  dfbplan_ = static_cast<DraftPlan *>(rt_->alloc_host(sizeof(DraftPlan)));
  std::memset(dfcplan_, 0, sizeof(DraftPlan));
  std::memset(dfbplan_, 0, sizeof(DraftPlan));
  dfembed_ = static_cast<float *>(rt_->alloc_host(STEP_MAX_ROWS * Q_EMBD * 4));
  dfex_ = static_cast<float *>(rt_->alloc_host((size_t)2 * DF_LAYERS * ncards_ * STEP_MAX_ROWS * Q_EMBD * 4));
  dtop_idx_ = static_cast<uint32_t *>(rt_->alloc_host((size_t)ncards_ * STEP_MAX_ROWS * opt_.candidates * 4));
  dtop_val_ = static_cast<float *>(rt_->alloc_host((size_t)ncards_ * STEP_MAX_ROWS * opt_.candidates * 4));
  dtop_n_ = static_cast<uint32_t *>(rt_->alloc_host(ncards_ * STEP_MAX_ROWS * 4));
  // The exchanges' events, then one a card: its context pass is done (each list runs on a queue of its own).
  devents_ = std::make_unique<ze::EventPool>(*rt_, (2 * DF_LAYERS + 1) * ncards_);
  const size_t pf = (size_t)PROMPT_MAX_RUNS * STEP_MAX_ROWS * DF_FEAT * 2;
  pfeat_ = static_cast<uint16_t *>(rt_->alloc_host(pf));
  std::memset(pfeat_, 0, pf);
  dffirst_.assign(opt_.slots, 0);
  return true;
}

void Model::load_dflash(Card &card) {
  const std::string dir = opt_.mtp_dir;
  card.df = std::make_unique<DfCard>();
  DfCard &D = *card.df;
  const uint32_t c = card.c, C = ncards_, E = Q_EMBD;
  if (DF_H % C || DF_KV % C || DF_FF % C) throw std::runtime_error("DFlash 2 does not split over these cards");
  D.nq = DF_H / C;
  D.nkv = DF_KV / C;
  D.ff = DF_FF / C;
  auto dev16 = [&](const std::vector<uint16_t> &v) { return upload<uint16_t>(card, v.data(), v.size() * 2); };
  auto devf = [&](const std::vector<uint16_t> &h) {
    std::vector<float> f(h.size());
    for (size_t i = 0; i < h.size(); i++) f[i] = bf16f(h[i]);
    return upload<float>(card, f.data(), f.size() * 4);
  };
  auto tensor = [&](const std::string &n, size_t el) { return df_tensor(dir, n, el); };
  auto rows = [&](const std::vector<uint16_t> &w, size_t K, size_t r0, size_t n) {
    return std::vector<uint16_t>(w.begin() + r0 * K, w.begin() + (r0 + n) * K);
  };
  auto cols = [&](const std::vector<uint16_t> &w, size_t nrows, size_t K, size_t k0, size_t n) {
    std::vector<uint16_t> v(nrows * n);
    for (size_t r = 0; r < nrows; r++) std::memcpy(&v[r * n], &w[r * K + k0], n * 2);
    return v;
  };
  const size_t qd = (size_t)D.nq * DF_D, kd = (size_t)D.nkv * DF_D;
  D.fc = dev16(tensor("fc.weight", (size_t)E * DF_FEAT));
  D.hnorm = devf(tensor("hidden_norm.weight", E));
  D.norm = devf(tensor("norm.weight", E));
  for (uint32_t l = 0; l < DF_LAYERS; l++) {
    DfCard::Layer &Ly = D.layers[l];
    const std::string p = "layers." + std::to_string(l) + ".";
    Ly.in_norm = devf(tensor(p + "input_layernorm.weight", E));
    Ly.post_norm = devf(tensor(p + "post_attention_layernorm.weight", E));
    Ly.qn = devf(tensor(p + "self_attn.q_norm.weight", DF_D));
    Ly.kn = devf(tensor(p + "self_attn.k_norm.weight", DF_D));
    Ly.q = dev16(rows(tensor(p + "self_attn.q_proj.weight", (size_t)DF_H * DF_D * E), E, c * qd, qd));
    Ly.k = dev16(rows(tensor(p + "self_attn.k_proj.weight", (size_t)DF_KV * DF_D * E), E, c * kd, kd));
    Ly.v = dev16(rows(tensor(p + "self_attn.v_proj.weight", (size_t)DF_KV * DF_D * E), E, c * kd, kd));
    Ly.o = dev16(cols(tensor(p + "self_attn.o_proj.weight", (size_t)E * DF_H * DF_D), E, DF_H * DF_D, c * qd, qd));
    Ly.gate = dev16(rows(tensor(p + "mlp.gate_proj.weight", (size_t)DF_FF * E), E, (size_t)c * D.ff, D.ff));
    Ly.up = dev16(rows(tensor(p + "mlp.up_proj.weight", (size_t)DF_FF * E), E, (size_t)c * D.ff, D.ff));
    Ly.down = dev16(cols(tensor(p + "mlp.down_proj.weight", (size_t)E * DF_FF), E, DF_FF, (size_t)c * D.ff, D.ff));
    Ly.abase = devf(tensor(p + "attention_conv.base_kernel", 4 * E));
    Ly.akp = dev16(tensor(p + "attention_conv.kernel_projection.weight", (size_t)DF_KP * E));
    Ly.mbase = devf(tensor(p + "mlp_conv.base_kernel", 4 * E));
    Ly.mkp = dev16(tensor(p + "mlp_conv.kernel_projection.weight", (size_t)DF_KP * E));
    const size_t ring = (size_t)opt_.slots * DF_RING * kd * 4;
    Ly.rk = upload<float>(card, nullptr, ring);
    Ly.rv = upload<float>(card, nullptr, ring);
  }
  constexpr size_t W = STEP_MAX_ROWS;
  auto fbuf = [&](size_t n) { return upload<float>(card, nullptr, W * n * 4); };
  // The bf16 rows mv_xmx_bf16 reads are zeroed once: it reads all 8, also past a pass's rows.
  auto bbuf = [&](size_t n) {
    std::vector<uint16_t> z(W * n, 0);
    return upload<uint16_t>(card, z.data(), z.size() * 2);
  };
  D.feat = bbuf(DF_FEAT);
  D.ctx = fbuf(E);
  D.ctxb = bbuf(E);
  D.h = fbuf(E);
  D.x = fbuf(E);
  D.xb = bbuf(E);
  D.xc = fbuf(E);
  D.xcb = bbuf(E);
  D.dyn = fbuf(DF_KP);
  D.q = fbuf(qd);
  D.k = fbuf(kd);
  D.v = fbuf(kd);
  // df_attn's splits: [row][query head][DF_SPLITS][DF_D + 2] (kernels/models/qwen4exp/dflash.cl).
  constexpr size_t splits = (DF_WINDOW + STEP_MAX_ROWS + 63) / 64;
  D.part = fbuf(qd / DF_D * splits * (DF_D + 2));
  D.o = fbuf(qd);
  D.ob = bbuf(qd);
  D.g = fbuf(D.ff);
  D.u = fbuf(D.ff);
  D.gub = bbuf(D.ff);
  D.hn = fbuf(E);
  D.cplan = upload<DraftPlan>(card, nullptr, sizeof(DraftPlan));
  D.bplan = upload<DraftPlan>(card, nullptr, sizeof(DraftPlan));
}

// A context pass of W rows (the last decode step's features into each slot's ring) or, with block, a block pass.
void Model::record_df(Card &card, uint32_t W, bool block) {
  DfCard &D = *card.df;
  auto list = std::make_unique<ze::List>(*rt_, card.c, ze::List::regular);
  ze::List &L = *list;
  const uint32_t c = card.c, C = ncards_, E = Q_EMBD;
  auto df = [&](const char *name) { return kernel(c, "qwen4exp_dflash", name); };
  const uint32_t zero = 0, nq = D.nq, nkv = D.nkv, qd = nq * DF_D, kd = nkv * DF_D;
  const float one = 1.0f;
  DraftPlan *pl = block ? D.bplan : D.cplan;
  const uint32_t *nrows = &pl->nrows;
  uint16_t *no_yb = nullptr;
  // The BF16 products on the XMX units with bf16 activations (xb: 8 rows of K), K split in 4, as the draft layer's.
  auto bf16mv = [&](const uint16_t *Wt, uint32_t rows, uint32_t K, const uint16_t *xb, float *y,
                    ze_event_handle_t signal = nullptr) {
    if (opt_.trace) L.label("mv_xmx_bf16 " + std::to_string(rows) + "x" + std::to_string(K));
    L.launch(kernel(c, "common", "mv_xmx_bf16"), {Wt, rows, K, K, xb, nrows, y, rows, zero, one, rows, 4u, no_yb},
             (rows + 15) / 16, 1, 1, 16 * 4, 1, 1, signal);
  };
  L.copy(pl, block ? dfbplan_ : dfcplan_, sizeof(DraftPlan));
  if (!block) {
    // The context: fc and its norm, then each layer's keys and values of it (the layers' inputs never see it).
    bf16mv(D.fc, E, DF_FEAT, D.feat, D.ctx);
    float *no_out = nullptr;
    L.launch(df("df_norm"), {D.ctx, D.hnorm, pl, no_out, D.ctxb}, W, 1, 1, 256);
    const float *no_q = nullptr;
    for (uint32_t l = 0; l < DF_LAYERS; l++) {
      DfCard::Layer &Ly = D.layers[l];
      bf16mv(Ly.k, kd, E, D.ctxb, D.k);
      bf16mv(Ly.v, kd, E, D.ctxb, D.v);
      L.launch(df("df_prep"), {no_q, D.k, D.v, Ly.qn, Ly.kn, zero, nkv, Ly.rk, Ly.rv, pl}, nkv, W, 1, DF_D);
    }
    L.signal((*devents_)[2 * DF_LAYERS * C + c]);
    L.close();
    D.ctx_lists[W] = std::move(list);
    return;
  }
  if (opt_.trace) {
    D.trace[W] = static_cast<uint64_t *>(rt_->alloc_host(8192 * 8));
    L.trace(D.trace[W], 8192);
  }
  L.wait({(*devents_)[2 * DF_LAYERS * C + c]});  // the ring holds the context's keys and values
  L.copy(D.h, dfembed_, (size_t)W * E * 4);
  // Exchange x, as the draft layer's: each card's product writes its partial sums into its block of dfex_ in host
  // memory and signals; each card waits for the other's, then adds both.
  uint32_t x = 0;
  auto partial = [&](uint32_t ex, uint32_t from) { return dfex_ + ((size_t)ex * C + from) * STEP_MAX_ROWS * E; };
  auto event = [&]() { return C > 1 ? (*devents_)[x * C + c] : nullptr; };
  auto exchange = [&](const float *base) {
    if (C > 1) L.wait({(*devents_)[x * C + (1 - c)]});
    L.launch(df("df_finish"), {partial(x, 0), C, D.dyn, base, pl, D.h}, E / 64, W, 1, 64);
    x++;
  };
  constexpr uint32_t splits = (DF_WINDOW + STEP_MAX_ROWS + 63) / 64;
  for (uint32_t l = 0; l < DF_LAYERS; l++) {
    DfCard::Layer &Ly = D.layers[l];
    // Attention: the norm, the convolution before it, q k v, the queries and keys normed and turned, the block's
    // keys and values into the ring, attention over the ring, o; then the convolution after it.
    L.launch(df("df_norm"), {D.h, Ly.in_norm, pl, D.x, D.xb}, W, 1, 1, 256);
    bf16mv(Ly.akp, DF_KP, E, D.xb, D.dyn);
    L.launch(df("df_conv"), {D.x, D.dyn, Ly.abase, pl, D.xc, D.xcb}, E / 64, W, 1, 64);
    bf16mv(Ly.q, qd, E, D.xcb, D.q);
    bf16mv(Ly.k, kd, E, D.xcb, D.k);
    bf16mv(Ly.v, kd, E, D.xcb, D.v);
    L.launch(df("df_prep"), {D.q, D.k, D.v, Ly.qn, Ly.kn, nq, nkv, Ly.rk, Ly.rv, pl}, nq + nkv, W, 1, DF_D);
    L.launch(df("df_attn"), {D.q, Ly.rk, Ly.rv, nq, nkv, pl, D.part}, nkv, W, splits, DF_D);
    L.launch(df("df_merge"), {nq, pl, D.part, D.o, D.ob}, nq, W, 1, DF_D);
    bf16mv(Ly.o, E, qd, D.ob, partial(x, c), event());
    exchange(Ly.abase);
    // The MLP, between its own convolutions.
    L.launch(df("df_norm"), {D.h, Ly.post_norm, pl, D.x, D.xb}, W, 1, 1, 256);
    bf16mv(Ly.mkp, DF_KP, E, D.xb, D.dyn);
    L.launch(df("df_conv"), {D.x, D.dyn, Ly.mbase, pl, D.xc, D.xcb}, E / 64, W, 1, 64);
    bf16mv(Ly.gate, D.ff, E, D.xcb, D.g);
    bf16mv(Ly.up, D.ff, E, D.xcb, D.u);
    L.launch(df("df_swiglu"), {D.g, D.u, D.ff, pl, D.gub}, (D.ff + 63) / 64, W, 1, 64);
    bf16mv(Ly.down, E, D.ff, D.gub, partial(x, c), event());
    exchange(Ly.mbase);
  }
  // The final norm, then the main model's head and its largest logits per row.
  uint16_t *no_b = nullptr;
  L.launch(df("df_norm"), {D.h, D.norm, pl, D.hn, no_b}, W, 1, 1, 256);
  const uint32_t units = E / 32 * W, gs = E;
  L.launch(kernel(c, "common", "quantize_a8"), {D.hn, gs, gs, W, card.xa8}, (units + 3) / 4, 1, 1, 64);
  matvec(L, card, card.head, card.xa8, card.logits, W);
  uint32_t *ti = dtop_idx_ + (size_t)c * STEP_MAX_ROWS * opt_.candidates;
  float *tv = dtop_val_ + (size_t)c * STEP_MAX_ROWS * opt_.candidates;
  L.launch(kernel(c, "common", "topk_rows"), {card.logits, card.vocab_n, card.vocab_n, opt_.candidates, card.vocab0,
                                              nrows, ti, tv, dtop_n_ + (size_t)c * STEP_MAX_ROWS},
           W, 1, 1, 1024);
  L.close();
  D.block_lists[W] = std::move(list);
}

std::vector<std::vector<std::vector<Candidate>>> Model::df_pass(const StepIn *ctx, uint32_t nctx,
                                                                const DraftBlock *blocks, uint32_t nblocks) {
  if (!df_on_) throw std::runtime_error("DFlash 2 is not loaded");
  if (nctx > STEP_MAX_ROWS) throw std::runtime_error("a context pass takes at most 8 rows");
  uint32_t nb = 0;
  for (uint32_t b = 0; b < nblocks; b++) {
    if (blocks[b].n == 0 || blocks[b].slot >= opt_.slots) throw std::runtime_error("a block is empty or past the slots");
    for (uint32_t q = 0; q < b; q++)
      if (blocks[q].slot == blocks[b].slot) throw std::runtime_error("a pass takes one block a sequence");
    nb += blocks[b].n;
  }
  if (nb > STEP_MAX_ROWS) throw std::runtime_error("a pass's blocks take at most 8 rows");
  auto width = [](uint32_t n) {
    uint32_t W = 1;
    while (W < n) W *= 2;
    return W;
  };
  if (nctx) {
    dfcplan_->nrows = nctx;
    for (uint32_t r = 0; r < nctx; r++) {
      if (ctx[r].slot >= opt_.slots) throw std::runtime_error("slot out of range");
      dfcplan_->rows[r] = {ctx[r].slot, ctx[r].pos, 0, 0, 0, {0, 0, 0}};
    }
  }
  if (nb) {
    const GgufTensor &te = gguf_.tensor("token_embd.weight");
    auto to_float = ggml_get_type_traits((ggml_type)te.type)->to_float;
    uint32_t r = 0;
    for (uint32_t b = 0; b < nblocks; b++) {
      const DraftBlock &bk = blocks[b];
      for (uint32_t j = 0; j < bk.n; j++, r++) {
        const uint32_t pos = bk.pos + j;
        uint32_t lo = std::max(dffirst_[bk.slot], pos + 1 > DF_WINDOW ? pos + 1 - DF_WINDOW : 0u);
        lo = std::min(lo, bk.pos);
        dfbplan_->rows[r] = {bk.slot, pos, r - j, bk.n, lo, {0, 0, 0}};
        const int32_t tok = j == 0 ? bk.token : DF_MASK;
        to_float(te.data + (uint64_t)tok * te.row_bytes(), dfembed_ + (size_t)r * Q_EMBD, Q_EMBD);
      }
    }
    dfbplan_->nrows = nb;
  }
  for (uint32_t i = 0; i < devents_->size(); i++) ZE_CHECK(zeEventHostReset((*devents_)[i]));
  if (!nctx)
    for (uint32_t c = 0; c < ncards_; c++) ZE_CHECK(zeEventHostSignal((*devents_)[2 * DF_LAYERS * ncards_ + c]));
  for (uint32_t c = ncards_; c-- > 0;) {
    DfCard &D = *cards_[c]->df;
    if (nctx) D.ctx_lists.at(width(nctx))->execute();
    if (nb) D.block_lists.at(D.last = width(nb))->execute();
  }
  for (uint32_t c = 0; c < ncards_; c++) {
    DfCard &D = *cards_[c]->df;
    if (nb)
      D.block_lists.at(width(nb))->synchronize();
    else if (nctx)
      D.ctx_lists.at(width(nctx))->synchronize();
  }
  std::vector<std::vector<std::vector<Candidate>>> out(nblocks);
  uint32_t r = 0;
  for (uint32_t b = 0; b < nblocks; b++) {
    out[b].resize(blocks[b].n - 1);
    for (uint32_t j = 0; j < blocks[b].n; j++, r++) {
      if (j == 0) continue;
      auto &o = out[b][j - 1];
      for (uint32_t c = 0; c < ncards_; c++) {
        size_t base = ((size_t)c * STEP_MAX_ROWS + r) * opt_.candidates;
        for (uint32_t i = 0; i < dtop_n_[c * STEP_MAX_ROWS + r]; i++)
          o.push_back({(int32_t)dtop_idx_[base + i], dtop_val_[base + i]});
      }
      std::sort(o.begin(), o.end(), [](const Candidate &a, const Candidate &b) {
        return a.logit != b.logit ? a.logit > b.logit : a.token < b.token;
      });
      if (o.size() > opt_.candidates) o.resize(opt_.candidates);
    }
  }
  return out;
}

void Model::df_context_prompt(const StepIn *rows, uint32_t n, uint32_t end) {
  if (!df_on_) throw std::runtime_error("DFlash 2 is not loaded");
  if (n == 0 || n > STEP_MAX_ROWS) throw std::runtime_error("a prompt context takes 1 to 8 rows");
  // The run that ends at row end: p_df_feat kept its last rows' features.
  for (uint32_t r = 0; r < PROMPT_MAX_RUNS; r++) {
    const PromptPlan &pp = pplan_[1 + r];
    if (pp.n == 0 || pp.row0 + pp.n != end) continue;
    // Only the run's rows in its last chunk have features: a shorter run gives fewer context rows.
    const uint32_t m = std::min<uint32_t>(pp.n, STEP_MAX_ROWS);
    if (n > m) {
      rows += n - m;
      n = m;
    }
    const uint16_t *src = pfeat_ + ((size_t)r * STEP_MAX_ROWS + m - n) * DF_FEAT;
    for (auto &card : cards_) card->up->copy(card->df->feat, src, (size_t)n * DF_FEAT * 2);
    for (auto &card : cards_) card->up->synchronize();
    df_pass(rows, n, nullptr, 0);
    return;
  }
  throw std::runtime_error("no prompt run ends at that row");
}

void Model::draft_start(uint32_t slot, uint32_t first) {
  if (df_on_) {
    if (slot >= opt_.slots) throw std::runtime_error("slot out of range");
    dffirst_[slot] = first;
    return;
  }
  mtp_start(slot, first);
}

std::vector<TraceRow> Model::df_trace(uint32_t c) const {
  const DfCard *D = df_on_ ? cards_.at(c)->df.get() : nullptr;
  if (!D || !D->trace.count(D->last)) return {};
  return trace_rows(*D->block_lists.at(D->last), D->trace.at(D->last), rt_->cards()[c]);
}

std::vector<uint16_t> Model::df_features(uint32_t row) {
  if (!df_on_ || row >= STEP_MAX_ROWS) throw std::runtime_error("no such DFlash 2 features");
  std::vector<uint16_t> v(DF_FEAT);
  Card &card = *cards_[0];
  card.up->copy(v.data(), card.df->feat + (size_t)row * DF_FEAT, DF_FEAT * 2);
  card.up->synchronize();
  return v;
}

std::vector<float> Model::df_hidden() {
  if (!df_on_) throw std::runtime_error("DFlash 2 is not loaded");
  std::vector<float> v((size_t)STEP_MAX_ROWS * Q_EMBD);
  Card &card = *cards_[0];
  card.up->copy(v.data(), card.df->hn, v.size() * 4);
  card.up->synchronize();
  return v;
}

}  // namespace engine::qwen4exp
