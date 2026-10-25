// The prompt path: a chunk of up to prompt_chunk tokens, in runs of consecutive tokens of up to PROMPT_MAX_RUNS
// sequences, in one pass on the cards, a group after another (docs/DEVELOPING.md, "How a prompt chunk runs"). Each
// card records lists per chunk size (the powers of two from 64, and prompt_chunk) at load, one for a run and one for
// several; the kernels read the plans (PromptPlan) in host memory, so a list serves every chunk up to its size. The
// kernels that follow a sequence run once per run. The cards of a group exchange fp16 partial sums twice per layer
// (ADR-0030).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <thread>

#include "model_impl.h"

namespace engine::qwen4exp {

using namespace detail;

namespace {

// The events' kinds.
constexpr uint32_t kEx = 0, kStaged = 1, kFree = 2, kNgram = 3, kIdx = 4, kMix = 5, kLast = 6, kRouted = 7, kHand = 8;

uint32_t bucket(uint32_t n, uint32_t tmax) {
  uint32_t b = 64;
  while (b < n && b < tmax) b *= 2;
  return std::min(b, tmax);
}

// The most tiles the experts' assignments can take: every expert's last tile may be partial.
uint32_t max_tiles(uint32_t T) { return (T * Q_USED + MOE_TILE - 1) / MOE_TILE + Q_EXPERTS; }

// The hyper-connections' down and inject products in one: 324 values per row, stored 336 apart (64-byte rows).
constexpr uint32_t kLriW = 336;

}  // namespace

// kind kEx and kMix: (layer, exchange or block 0 or 1 as layer * 2 + which, rank); kStaged, kFree and kRouted:
// (layer, rank); kNgram: the one event the host signals when the chunk's n-gram rows are in; kLast: (0, rank); kHand:
// (0, card), the residuals in the card's from the group before. A layer is one group's, so its ranks name its cards.
uint32_t Model::pev(uint32_t kind, uint32_t layer, uint32_t c) const {
  const uint32_t C = ncards_, mix = 4 * Q_LAYERS * C + 1 + kAttn * C;
  if (kind == kEx) return layer * C + c;  // layer here is 2 l + which
  if (kind == kNgram) return 4 * Q_LAYERS * C;
  if (kind == kIdx) return 4 * Q_LAYERS * C + 1 + layer * C + c;  // layer here is the attention layer's index
  if (kind == kMix) return mix + layer * C + c;
  if (kind == kLast) return mix + 2 * Q_LAYERS * C + c;
  if (kind == kRouted) return mix + 2 * Q_LAYERS * C + C + layer * C + c;
  if (kind == kHand) return mix + 3 * Q_LAYERS * C + C + c;
  return 2 * Q_LAYERS * C + (kind - 1) * Q_LAYERS * C + layer * C + c;
}

void Model::load_prompt(Card &card) {
  const uint16_t zero16 = 0;
  card.pc = std::make_unique<PromptCard>();
  PromptCard &P = *card.pc;
  const uint64_t T = tmax_;
  auto dev = [&](uint64_t bytes) { return static_cast<uint8_t *>(rt_->alloc_device(card.dev, std::max<uint64_t>(bytes, 256))); };
  auto f32 = [&](uint64_t n) { return reinterpret_cast<float *>(dev(n * 4)); };
  auto u16 = [&](uint64_t n) { return reinterpret_cast<uint16_t *>(dev(n * 2)); };
  auto u32 = [&](uint64_t n) { return reinterpret_cast<uint32_t *>(dev(n * 4)); };
  P.R = f32(T * Q_HC_DIM);
  P.inj[0] = f32(T * Q_HC);
  P.inj[1] = f32(T * Q_HC);
  P.lr = f32(T * kLriW);
  P.rr = f32(T * Q_HC);
  P.lrb = u16(T * Q_HC_LR);
  P.x16 = u16(T * Q_EMBD);
  P.xb = u16(T * Q_EMBD);
  P.ab = f32(T * 2 * card.nvh);
  P.gb = f32(T * 2 * card.nvh);
  const uint64_t ow = std::max(card.nvh * Q_GDN_D, card.nq * Q_ATT_D);
  P.o16 = u16(T * ow);
  // Region A: the mix (n, nb, a), the experts (xg, hg, yg), or the PLE module (pk, N).
  {
    uint64_t mix = align256(T * Q_HC_DIM * 2) * 2;
    uint64_t moe = align256(T * Q_USED * Q_EMBD * 2) * 2 + align256(T * Q_USED * card.ff * 2);
    uint64_t ple = align256(T * Q_HC_DIM * 4) * 2;
    P.reg_bytes[0] = std::max({mix, moe, ple});
    uint8_t *r = P.reg[0] = dev(P.reg_bytes[0]);
    P.a = reinterpret_cast<uint16_t *>(r);
    P.nb = reinterpret_cast<uint16_t *>(r + align256(T * Q_HC_DIM * 2));
    P.xg = reinterpret_cast<uint16_t *>(r);
    P.yg = reinterpret_cast<uint16_t *>(r + align256(T * Q_USED * Q_EMBD * 2));
    P.hg = reinterpret_cast<uint16_t *>(r + 2 * align256(T * Q_USED * Q_EMBD * 2));
    P.pk = reinterpret_cast<float *>(r);
    P.N = reinterpret_cast<float *>(r + align256(T * Q_HC_DIM * 4));
  }
  // Region B: DeltaNet (m, z, u, raw) or attention (Q, K, V, iq, scores, blocks, bscores, cells).
  {
    // The scores of this card's blocks only, as p_idx_score writes them: rows nbl apart.
    const uint64_t nbl = (opt_.max_ctx / Q_IDX_BLOCK + card.C - 1) / card.C;
    uint64_t gdn[] = {T * card.nch * 4, T * card.nvh * Q_GDN_D * 4, T * card.nch * 4, T * card.nvh * Q_GDN_D * 4};
    uint64_t att[] = {T * card.nq * 2 * Q_ATT_D * 4, T * card.nkv * Q_ATT_D * 4, T * card.nkv * Q_ATT_D * 4,
                      T * (Q_IDX_H + 1) * Q_IDX_D * 4, T * nbl * 4, T * (Q_IDX_BLOCKS + 1) * 4,
                      T * (Q_IDX_BLOCKS + 1) * 4, T * Q_IDX_CELLS * 4};
    uint64_t sg = 0, sa = 0;
    for (uint64_t b : gdn) sg += align256(b);
    for (uint64_t b : att) sa += align256(b);
    P.reg_bytes[1] = std::max(sg, sa);
    uint8_t *r = P.reg[1] = dev(P.reg_bytes[1]);
    uint8_t *at = r;
    float **gp[] = {&P.m, &P.z, &P.u, &P.raw};
    for (int i = 0; i < 4; i++) *gp[i] = reinterpret_cast<float *>(at), at += align256(gdn[i]);
    at = r;
    void **ap[] = {(void **)&P.Q, (void **)&P.K, (void **)&P.V, (void **)&P.iq, (void **)&P.scores,
                   (void **)&P.blocks, (void **)&P.bscores, (void **)&P.cells};
    for (int i = 0; i < 8; i++) *ap[i] = at, at += align256(att[i]);
  }
  P.ncells = u32(T);
  P.cand = u32(T * kIdxCand);
  P.cand_peer = u32(T * kIdxCand);
  P.merged = f32(T * 2 * (Q_IDX_BLOCKS + 1));
  P.kh = u16((uint64_t)opt_.max_ctx * card.nkv * Q_ATT_D);
  P.vh = u16((uint64_t)opt_.max_ctx * card.nkv * Q_ATT_D);
  P.hp = (card.nq + 7) / 8 * 8;
  P.qt = u16(T * Q_ATT_D * P.hp);
  P.q16 = u16(T * card.nq * Q_ATT_D);
  P.nbw = (opt_.max_ctx / Q_IDX_BLOCK + 31) / 32;
  P.sel = u32(T * P.nbw);
  P.wbx = u32(T);
  card.up->fill(P.qt, &zero16, 2, T * Q_ATT_D * P.hp * 2);
  P.rl = f32(T * (Q_EXPERTS + 1));
  P.ids = reinterpret_cast<int32_t *>(u32(T * Q_USED));
  P.w = f32(T * Q_USED);
  P.cnt = u32(Q_EXPERTS);
  P.base = u32(Q_EXPERTS);
  P.cursor = u32(Q_EXPERTS);
  P.inv = u32(T * Q_USED);
  P.ntiles_c = u32(1);
  P.ntiles_s = u32(1);
  P.tiles_c = reinterpret_cast<MoeTile *>(dev(max_tiles((uint32_t)T) * sizeof(MoeTile)));
  P.tiles_s = reinterpret_cast<MoeTile *>(dev(max_tiles((uint32_t)T) * sizeof(MoeTile)));
  P.shg = f32(T * card.ff);
  P.shu = f32(T * card.ff);
  P.shh = u16(T * card.ff);
  P.s = f32(T * Q_EMBD);
  P.part = u16(T * Q_EMBD);
  P.recv[0] = u16(T * Q_EMBD);
  P.recv[1] = u16(T * Q_EMBD);
  P.ub = u16(T * Q_EMBD);
  P.pv = f32(T * Q_EMBD);
  P.gates = f32(T * Q_HC);
  uint64_t w16 = 0;
  for (auto &L : card.layers)
    for (const Mat *m : {&L.qkv, &L.z, &L.out, &L.q, &L.k, &L.v, &L.o, &L.sh_gate, &L.sh_up, &L.sh_down})
      w16 = std::max<uint64_t>(w16, (uint64_t)m->rows * m->K);
  P.w16 = u16(w16);
  if ((opt_.profile_routing || opt_.adapt_swaps) && card.c == 0) {  // its group's layers
    const uint32_t z = 0;
    P.prof = u32(Q_LAYERS * Q_EXPERTS);
    card.up->fill(P.prof, &z, 4, Q_LAYERS * Q_EXPERTS * 4);
    card.up->synchronize();
  }
  const uint32_t zero = 0;
  card.up->fill(P.cnt, &zero, 4, Q_EXPERTS * 4);
  card.up->synchronize();
}

// The staging buffer, the experts' staging maps, and the copy list that streams the experts a chunk of more than
// kFetchRows needs: every layer of the card's group, its uncached experts, the card's share of gate and up and its
// region of down, into slot l % 2, after the chunk's list has freed that slot (event kFree of layer l - 2); event
// kStaged of layer l says they are in. A slot holds every expert the cache may leave out of the group's largest layer
// (HostLayer::pieces). Shorter chunks fetch their experts on the fetch list (fetch_experts).
void Model::record_prompt_copy(Card &card) {
  PromptCard &P = *card.pc;
  for (uint32_t l = card.l0; l < card.l1; l++)
    P.stage_slot = std::max<uint64_t>(P.stage_slot, host_layers_[l]->pieces.size() * host_layers_[l]->stg_bytes);
  if (P.stage_slot) P.stage = static_cast<uint8_t *>(rt_->alloc_device(card.dev, 2 * P.stage_slot));
  for (uint32_t l = 0; l < Q_LAYERS; l++)
    P.stage_map.push_back(upload<int32_t>(card, host_layers_[l]->stage_of.data(), Q_EXPERTS * 4));
  P.copy = prompt_copies(card, staging_now(), copy_count_);
  if (P.stage_slot) {
    P.fetch = std::make_unique<ze::List>(*rt_, card.dev, ze::List::immediate, true);
    P.routed = static_cast<uint32_t *>(rt_->alloc_host((size_t)Q_LAYERS * Q_EXPERTS * 4));
    // The staging buffer written once through the fetch list, in copies of the size it makes: the runtime's first
    // CPU copy into a page of a card's memory takes several times as long as the next ones.
    const HostLayer &hl = *host_layers_[card.l0];
    const uint64_t piece = hl.stg_up_off;
    for (uint64_t off = 0; off < 2 * P.stage_slot; off += piece)
      P.fetch->copy(P.stage + off, hl.pieces[0], std::min(piece, 2 * P.stage_slot - off));
    P.fetch->synchronize();
  }
}

// The chunk's list copies each layer's routing counts to P.routed and signals kRouted after p_moe_count; it waits
// on kStaged before the staged experts' products. Each layer's experts go into slot l % 2 once kFree of layer l - 2
// says the chunk is done with it. Every expert the chunk routes to is staged before kStaged, so the products are the
// recorded list's; which experts go ahead of the router changes only the time.
void Model::fetch_experts(uint32_t dev, const std::vector<std::vector<uint32_t>> &order) {
  Card &card = *cards_[dev];
  const uint32_t c = card.c;
  PromptCard &P = *card.pc;
  ze::List &F = *P.fetch;
  const uint64_t f0 = (uint64_t)c * card.ff;
  // Expert e of layer l into its staging position: the same three copies as prompt_copies' for a run of one. From
  // the anonymous arena the runtime makes these copies with the CPU and returns when they are done
  // (docs/measurements/MEASUREMENTS.md#2026-10-21-short-chunk-fetch).
  auto one = [&](const HostLayer &hl, uint8_t *slot, uint32_t e) {
    const ExpertWeights &w = hl.weights[e];
    const uint64_t half = card.ff * hl.gate_row_bytes, dpart = hl.down_bytes / hl.down_parts;
    const uint64_t d0 = hl.down_parts > 1 ? c * dpart : 0;
    uint8_t *dst = slot + (uint64_t)hl.stage_of[e] * hl.stg_bytes;
    F.copy(dst, w.gate + f0 * hl.gate_row_bytes, half);
    F.copy(dst + hl.stg_up_off, w.up + f0 * hl.gate_row_bytes, half);
    F.copy(dst + hl.stg_down_off, w.down + d0, dpart);
  };
  // Bounded as ze::List::synchronize is: a list whose engine was reset never signals.
  auto wait = [](ze_event_handle_t e) {
    const ze_result_t r = zeEventHostSynchronize(e, 60'000'000'000ull);
    if (r == ZE_RESULT_NOT_READY) throw std::runtime_error("fetch_experts: the chunk's list stopped signalling");
    ZE_CHECK(r);
  };
  std::vector<uint8_t> done(Q_EXPERTS);
  for (uint32_t l = card.l0; l < card.l1; l++) {
    const HostLayer &hl = *host_layers_[l];
    uint8_t *slot = P.stage + (l % 2) * P.stage_slot;
    const ze_event_handle_t routed = (*pevents_)[pev(kRouted, l, c)];
    if (l >= card.l0 + 2) wait((*pevents_)[pev(kFree, l - 2, c)]);
    std::fill(done.begin(), done.end(), 0);
    for (uint32_t e : order[l]) {
      if (zeEventQueryStatus(routed) == ZE_RESULT_SUCCESS) break;
      one(hl, slot, e);
      done[e] = 1;
    }
    wait(routed);
    const uint32_t *cnt = P.routed + (size_t)l * Q_EXPERTS;
    for (uint32_t e : hl.uncached)
      if (cnt[e] && !done[e]) one(hl, slot, e);
    F.synchronize();
    ZE_CHECK(zeEventHostSignal((*pevents_)[pev(kStaged, l, c)]));
  }
}

Model::Staging Model::staging_now() const {
  Staging st(Q_LAYERS);
  for (uint32_t l = 0; l < Q_LAYERS; l++)
    for (uint32_t e : host_layers_[l]->uncached) {
      // From the arena piece, not the weights: a swap's outgoing expert reads from the piece the swap thread filled
      // until its move into this one is settled (settle_moves).
      const HostLayer &hl = *host_layers_[l];
      uint8_t *p = hl.arena[e];
      st[l].push_back({p, p + hl.gate_bytes, p + 2 * hl.gate_bytes, hl.down_parts});
    }
  return st;
}

// The copy list that stages each layer's experts st names, from their arena pieces. A swap leaves each staging slot's
// piece where it was (swap_maps), so the list is recorded again only when the uncached experts grow or shrink
// (kv_share.cpp). copies: the list's copies.
std::unique_ptr<ze::List> Model::prompt_copies(Card &card, const Staging &st, uint32_t &copies) {
  PromptCard &P = *card.pc;
  auto list = std::make_unique<ze::List>(*rt_, card.dev, ze::List::regular, true);
  ze::List &L = *list;
  const uint64_t f0 = (uint64_t)card.c * card.ff;
  copies = 0;
  for (uint32_t l = card.l0; l < card.l1; l++) {
    const HostLayer &hl = *host_layers_[l];
    const std::vector<ExpertWeights> &sw = st[l];
    if (l >= card.l0 + 2) L.wait({(*pevents_)[pev(kFree, l - 2, card.c)]});
    uint8_t *slot = P.stage + (l % 2) * P.stage_slot;
    const uint64_t half = card.ff * hl.gate_row_bytes, dpart = hl.down_bytes / hl.down_parts;
    const uint64_t d0 = hl.down_parts > 1 ? card.c * dpart : 0;
    // Experts adjacent in the arena (load lays them out so) take one region copy a matrix: each copy has a fixed
    // cost, higher from the anonymous arena (docs/measurements/MEASUREMENTS.md#2026-10-17-anon-arena-default). A run ends at a
    // piece's end or where a swap put an expert in a spare piece.
    const uint64_t stride = align256(2 * hl.gate_bytes + hl.down_bytes);
    for (size_t i = 0, n; i < sw.size(); i += n) {
      const ExpertWeights &w = sw[i];
      for (n = 1; i + n < sw.size(); n++) {
        const ExpertWeights &x = sw[i + n];
        if (x.gate != w.gate + n * stride || x.up != w.up + n * stride || x.down != w.down + n * stride) break;
      }
      uint8_t *dst = slot + i * hl.stg_bytes;
      L.copy_rows(dst, hl.stg_bytes, w.gate + f0 * hl.gate_row_bytes, stride, half, n);
      L.copy_rows(dst + hl.stg_up_off, hl.stg_bytes, w.up + f0 * hl.gate_row_bytes, stride, half, n);
      L.copy_rows(dst + hl.stg_down_off, hl.stg_bytes, w.down + d0, stride, dpart, n);
      copies += 3;
    }
    L.signal((*pevents_)[pev(kStaged, l, card.c)]);
  }
  L.close();
  return list;
}

// The list for chunks of up to T rows in up to `runs` runs (1 or PROMPT_MAX_RUNS).
ze::List &Model::prompt_list(Card &card, uint32_t T, uint32_t runs) {
  PromptCard &P = *card.pc;
  const uint32_t key = runs << 16 | T;
  auto it = P.lists.find(key);
  if (it != P.lists.end()) return *it->second;
  auto list = std::make_unique<ze::List>(*rt_, card.dev, ze::List::regular);
  ze::List &L = *list;
  if (opt_.trace) {
    const size_t cap = 1 << 16;
    P.trace[key] = static_cast<uint64_t *>(rt_->alloc_host(cap * 8));
    L.trace(P.trace[key], cap);
  }
  const uint32_t c = card.c, C = card.C, dev = card.dev;
  auto q4 = [&](const char *name) { return kernel(dev, "qwen4exp_prompt", name); };
  const PromptPlan *pp = pplan_;
  // Run r's plan and positions, for the kernels that follow a sequence.
  auto run = [&](uint32_t r) { return pplan_ + 1 + r; };
  auto pos = [&](uint32_t r) { return ppos_ + (size_t)r * ppos_run(); };
  const float one = 1.0f;
  const uint32_t zero = 0;
  const uint32_t nb_max = opt_.max_ctx / Q_IDX_BLOCK, mt = max_tiles(T);
  // tm tiles of 8 tokens and tn tiles of 16 rows a sub-group, sg sub-groups a work-group (kernels/gemm.cl): 8, 2
  // and 4, or for bf16 4, 2 or 1 and 4 (modules gemm_bf16_t4n<tn>), or 4, 2 and 11 (gemm_bf16_t4n2s11). y2: a second
  // fp16 output, or null. nt and rows, for x and y from a row other than 0: where the count of tokens is, and the most.
  // Weights larger than the cards' 18 MiB cache (docs/measurements/MEASUREMENTS.md#2026-10-11-level-zero-report) are taken in
  // bands of 4 blocks of rows.
  auto gemm = [&](bool bf16, const uint16_t *W, uint32_t N, uint32_t K, const uint16_t *x, uint32_t ldx, void *y,
                  uint32_t ldy, bool yhalf, uint32_t mode, float scale, ze_event_handle_t sig = nullptr,
                  uint32_t tm = 8, uint32_t tn = 2, uint16_t *y2 = nullptr, uint32_t sg = 4,
                  const uint32_t *nt = nullptr, uint32_t rows = 0) {
    uint32_t yh = yhalf ? 1 : 0;
    const uint32_t *n = nt ? nt : &pplan_->n;
    if (!rows) rows = T;
    std::string mod = bf16 ? "gemm_bf16" : "gemm_f16";
    if (tm != 8 || tn != 2) mod += "_t" + std::to_string(tm) + "n" + std::to_string(tn);
    if (sg != 4) mod += "s" + std::to_string(sg);
    if (opt_.trace) L.label(std::string(bf16 ? "gemm_bf16 " : "gemm_f16 ") + std::to_string(N) + "x" + std::to_string(K));
    const uint32_t band = (uint64_t)N * K * 2 > (18u << 20) ? 4 : 0;
    L.launch(kernel(dev, mod, "gemm"), {W, N, K, K, x, ldx, n, y, ldy, yh, mode, scale, y2, band},
             (N + kSgSize * tn * sg - 1) / (kSgSize * tn * sg), (rows + 8 * tm - 1) / (8 * tm), 1, kSgSize * sg, 1,
             1, sig);
  };
  auto dense = [&](const Mat &m, const uint16_t *x, uint32_t ldx, void *y, uint32_t ldy, bool yhalf,
                   uint16_t *y2 = nullptr, ze_event_handle_t sig = nullptr) {
    std::string mod = std::string("qgemm_") + ggml_type_label(m.type);
    for (auto &ch : mod) ch = (char)std::tolower(ch);
    const uint32_t units = m.K / 32;
    L.launch(kernel(dev, mod, "dequant_f16"), {m.p, m.rows, m.bpr, zero, units, P.w16, m.K, card.grids},
             (m.rows * units + 63) / 64, 1, 1, 64);
    gemm(false, P.w16, m.rows, m.K, x, ldx, y, ldy, yhalf, 0, 1.0f, sig, 8, 2, y2);
  };
  // An exchange: the kernel that makes each card's fp16 partial also writes it into the other card's memory as it
  // computes, and signals (docs/measurements/MEASUREMENTS.md#2026-10-15-prompt-exchange-stores); both cards add the two partials
  // in card order, so their residuals stay the same. Exchange e = 2 l + which uses recv[e % 2]: a card writes e + 2
  // only after the other card has sent e + 1, which it does after reading e. The other card is the group's.
  PromptCard *peer = C > 1 ? group_card(card.g, 1 - c).pc.get() : nullptr;
  // From layer 1's PLE module on, a chunk of kHcSplitRows or more splits the hyper-connections' rows between the
  // cards: card c keeps R for rows r0 .. r1 - 1 (prows_[c] of them) and sends the other card its rows of the mixed x
  // (event kMix). Before, both cards compute every row, as the PLE module's convolution reads across rows.
  const bool split = C > 1 && T >= kHcSplitRows;
  const bool fetch = T <= kFetchRows && P.fetch;  // the host fetches the uncached experts (fetch_experts)
  const uint32_t half = split ? T / 2 : T;
  uint32_t r0 = 0, r1 = T;  // this card's rows, all of them until the split
  const uint32_t *nown = &pplan_->n;
  // next_norm: the norm of the block that follows, done with the combine (p_hc_combine_norm), or null.
  auto exchange = [&](uint32_t l, uint32_t which, const float *inj, const float *next_norm) {
    const uint16_t *p0 = P.part, *p1 = nullptr;
    if (C > 1) {
      uint16_t *mine = P.recv[(2 * l + which) % 2];
      L.wait({(*pevents_)[pev(kEx, 2 * l + which, 1 - c)]});
      p0 = c == 0 ? P.part : mine;
      p1 = c == 0 ? mine : P.part;
    }
    if (next_norm)
      L.launch(q4("p_hc_combine_norm"), {p0, p1, inj, next_norm, pp, P.R, P.rr, P.nb, r0, r1}, Q_HC, r1 - r0, 1,
               256);
    else
      L.launch(q4("p_hc_combine"), {p0, p1, inj, pp, P.R, r0, r1}, Q_EMBD / 64, r1 - r0, 1, 64);
  };
  // normed: the combine before it already did the norm.
  // block: 2 l + i, for the event that says this card's rows of x are in the other card's.
  auto hc_mix = [&](const Layer &Ly, uint32_t i, bool normed, uint32_t block) {
    const uint32_t rows = r1 - r0;
    if (!normed) L.launch(q4("p_hc_norm"), {P.R, Ly.hc_norm[i], pp, P.rr, P.nb, r0, r1}, Q_HC, rows, 1, 256);
    // Tiles of 4 tokens: faster with K or N short (docs/measurements/MEASUREMENTS.md#2026-10-15-prompt-gemm-tiles). From 1,024
    // rows one work-group takes all 324 columns, reading its tokens once (docs/measurements/MEASUREMENTS.md#2026-10-17-hc-lowrank).
    const uint16_t *nb = P.nb + (size_t)r0 * Q_HC_DIM;
    float *lr = P.lr + (size_t)r0 * kLriW;
    if (rows >= 1024)
      gemm(true, Ly.hc_down[i], Q_HC_LR + Q_HC, Q_HC_DIM, nb, Q_HC_DIM, lr, kLriW, false, 0, one, nullptr, 4, 2,
           nullptr, 11, nown, rows);
    else
      gemm(true, Ly.hc_down[i], Q_HC_LR + Q_HC, Q_HC_DIM, nb, Q_HC_DIM, lr, kLriW, false, 0, one, nullptr, 4, 1,
           nullptr, 4, nown, rows);
    L.launch(q4("p_hc_lr"), {P.lr, kLriW, pp, P.lrb, P.inj[i], r0, r1}, (Q_HC_LR + Q_HC + 63) / 64, rows, 1, 64);
    gemm(true, Ly.hc_up[i], Q_HC_DIM, Q_HC_LR, P.lrb + (size_t)r0 * Q_HC_LR, Q_HC_LR, P.a + (size_t)r0 * Q_HC_DIM,
         Q_HC_DIM, true, 0, one, nullptr, 4, 2, nullptr, 4, nown, rows);
    const bool send = r1 - r0 < T;
    L.launch(q4("p_hc_mix"), {P.R, P.rr, Ly.hc_norm[i], P.a, pp, P.x16, P.xb, r0, r1, send ? peer->x16 : nullptr,
                              send ? peer->xb : nullptr},
             Q_EMBD / 256, rows, 1, 64, 1, 1, send ? (*pevents_)[pev(kMix, block, c)] : nullptr);
    if (send) L.wait({(*pevents_)[pev(kMix, block, 1 - c)]});
  };

  // The first group starts from the embeddings; the others from the residuals the group before copies in, each card
  // its rows when the rows split.
  if (card.first()) {
    L.launch(q4("p_hc_init"), {pembed_, pp, P.R}, Q_HC_DIM / 64, T, 1, 64);
  } else {
    L.wait({(*pevents_)[pev(kHand, 0, dev)]});
    if (split) r0 = c * half, r1 = r0 + half, nown = prows_ + c;
  }
  for (uint32_t l = card.l0; l < card.l1; l++) {
    Layer &Ly = card.layers[l];
    HostLayer &hl = *host_layers_[l];
    if (l == 1) {  // the PLE module
      L.wait({(*pevents_)[pev(kNgram, 0, 0)]});
      // The rows as bf16, from host memory in one copy: a kernel reading them there is slow.
      L.copy(P.ub, pngram16_, (size_t)T * Q_EMBD * 2);
      gemm(true, card.ple_k, Q_HC_DIM, Q_EMBD, P.ub, Q_EMBD, P.pk, Q_HC_DIM, false, 0, one);
      gemm(true, card.ple_v, Q_EMBD, Q_EMBD, P.ub, Q_EMBD, P.pv, Q_EMBD, false, 0, one);
      L.launch(q4("p_ple_gate"), {P.pk, P.R, card.ple_gk, card.ple_gq, pp, P.gates}, Q_HC, T, 1, 256);
      L.launch(q4("p_ple_norm"), {P.pv, P.gates, card.ple_gc, pp, P.N}, Q_HC, T, 1, 256);
      for (uint32_t r = 0; r < runs; r++)
        L.launch(q4("p_ple_conv"), {P.N, P.pv, P.gates, card.ple_conv, card.ple_hist, card.ple_stride, run(r), P.R},
                 Q_HC_DIM / 64, T, 1, 64);
      for (uint32_t r = 0; r < runs; r++)
        L.launch(q4("p_ple_save"), {P.N, card.ple_hist, card.ple_stride, run(r)}, Q_HC_DIM / 64, 1, 1, 64);
      if (split) r0 = c * half, r1 = r0 + half, nown = prows_ + c;
    }
    // After layer 0 the PLE module stands between the combine and the norm; a group's first layer has no combine
    // before it on its cards.
    hc_mix(Ly, 0, l >= 2 && l != card.l0, 2 * l);
    // Where this card's partial of exchange `which` also goes, and the event that says it is there.
    auto peer_part = [&](uint32_t which) { return C > 1 ? peer->recv[(2 * l + which) % 2] : nullptr; };
    auto sent = [&](uint32_t which) { return C > 1 ? (*pevents_)[pev(kEx, 2 * l + which, c)] : nullptr; };
    if (!Ly.attn) {
      const uint32_t nch = card.nch, nvo = card.nvh * Q_GDN_D;
      dense(Ly.qkv, P.x16, Q_EMBD, P.m, nch, false);
      dense(Ly.z, P.x16, Q_EMBD, P.z, nvo, false);
      gemm(true, Ly.ab, 2 * card.nvh, Q_EMBD, P.xb, Q_EMBD, P.ab, 2 * card.nvh, false, 0, one, nullptr, 4, 1);
      for (uint32_t r = 0; r < runs; r++)
        L.launch(q4("p_gdn_conv"), {P.m, Ly.conv, Ly.conv_state, Ly.conv_stride, nch, run(r), P.u}, (nch + 63) / 64,
                 T, 1, 64);
      for (uint32_t r = 0; r < runs; r++)
        L.launch(q4("p_gdn_conv_save"), {P.m, Ly.conv_state, Ly.conv_stride, nch, run(r)}, (nch + 63) / 64, 1, 1, 64);
      L.launch(q4("p_gdn_l2norm"), {P.u, nch, pp}, 2 * card.nkh, T, 1, Q_GDN_D);
      L.launch(q4("p_gdn_gates"), {P.ab, Ly.A, Ly.dt, card.nvh, pp, P.gb}, 1, T, 1, 64);
      for (uint32_t r = 0; r < runs; r++)
        L.launch(kernel(dev, "qwen4exp_gdn", "p_gdn_recur"),
                 {P.u, nch, P.gb, Ly.S, Ly.S_stride, card.nkh, card.nvh, run(r), P.raw}, card.nvh * 8, 1, 1, 64);
      L.launch(q4("p_gdn_out"), {P.raw, P.z, Ly.ssm_norm, card.nvh, pp, P.o16}, card.nvh, T, 1, Q_GDN_D);
      dense(Ly.out, P.o16, nvo, P.part, Q_EMBD, true, peer_part(0), sent(0));
    } else {
      const uint32_t nqo = card.nq * Q_ATT_D;
      dense(Ly.q, P.x16, Q_EMBD, P.Q, card.nq * 2 * Q_ATT_D, false);
      dense(Ly.k, P.x16, Q_EMBD, P.K, card.nkv * Q_ATT_D, false);
      dense(Ly.v, P.x16, Q_EMBD, P.V, card.nkv * Q_ATT_D, false);
      gemm(true, Ly.idx_w, (Q_IDX_H + 1) * Q_IDX_D, Q_EMBD, P.xb, Q_EMBD, P.iq, (Q_IDX_H + 1) * Q_IDX_D, false, 0, one);
      for (uint32_t r = 0; r < runs; r++)
        L.launch(q4("p_attn_prep"), {P.Q, P.K, P.V, Ly.qn, Ly.kn, card.nq, card.nkv, Ly.kc, Ly.vc, card.ptab, npp_,
                                     run(r), pos(r), P.qt, P.hp, P.q16},
                 card.nq + card.nkv, T, 1, Q_ATT_D);
      // The indexer, split by position: each card scores its own blocks, the cards swap their best, both merge.
      const uint32_t nbl = (nb_max + C - 1) / C;
      for (uint32_t r = 0; r < runs; r++) {
        L.launch(q4("p_idx_prep"), {P.iq, Ly.iqn, Ly.raw, card.ptab, npp_, run(r), pos(r), c, C}, Q_IDX_H + 1, T, 1,
                 Q_IDX_D);
        L.launch(q4("p_idx_pool"), {Ly.raw, card.ptab, npp_, Ly.ikn, Ly.pooled, run(r), pos(r), c, C}, T / 4 + 1, 1,
                 1, Q_IDX_D);
        L.launch(q4("p_idx_score"), {P.iq, Ly.pooled, card.ptab, npp_, nbl, run(r), P.scores, c, C}, (nbl + 63) / 64,
                 (T + kIdxScoreRows - 1) / kIdxScoreRows, 1, 64);
        L.launch(q4("p_idx_cand"), {P.scores, nbl, run(r), P.blocks, P.bscores, P.cand, c, C}, T, 1, 1, 1024);
      }
      const uint32_t *c0 = P.cand, *c1 = nullptr;
      if (C > 1) {
        L.copy(peer->cand_peer, P.cand, (size_t)T * kIdxCand * 4, (*pevents_)[pev(kIdx, Ly.idx, c)]);
        L.wait({(*pevents_)[pev(kIdx, Ly.idx, 1 - c)]});
        c0 = c == 0 ? P.cand : P.cand_peer;
        c1 = c == 0 ? P.cand_peer : P.cand;
      }
      for (uint32_t r = 0; r < runs; r++)
        L.launch(q4("p_idx_merge"),
                 {c0, c1, run(r), P.merged, P.blocks, P.bscores, P.cells, P.ncells, P.sel, P.nbw, P.wbx}, T, 1, 1, 1024);
      // kh and vh hold one sequence: each run converts its own, then attends.
      for (uint32_t r = 0; r < runs; r++) {
        L.launch(q4("p_kv16"), {Ly.kc, Ly.vc, card.ptab, npp_, card.nkv, run(r), P.kh, P.vh},
                 (2 * card.nkv * Q_ATT_D / 32 + 63) / 64, opt_.max_ctx, 1, 64);
        L.launch(kernel(dev, "qwen4exp_attn", "p_attn_xmx"),
                 {P.qt, P.Q, P.kh, P.vh, card.nq, P.hp, card.nkv, P.cells, P.ncells, run(r), P.o16}, 1,
                 (T + kAttXmxRows - 1) / kAttXmxRows, 1, 16 * (P.hp / 8) * kAttXmxRows);
        L.launch(kernel(dev, "qwen4exp_attn", "p_attn_tiled"),
                 {P.q16, P.Q, P.kh, P.vh, card.nq, card.nkv, P.sel, P.nbw, P.wbx, run(r), P.o16}, card.nq, (T + 7) / 8,
                 1, 16);
      }
      dense(Ly.o, P.o16, nqo, P.part, Q_EMBD, true, peer_part(0), sent(0));
    }
    exchange(l, 0, P.inj[0], Ly.hc_norm[1]);

    // The MoE block.
    hc_mix(Ly, 1, true, 2 * l + 1);
    gemm(true, Ly.router, Q_EXPERTS + 1, Q_EMBD, P.xb, Q_EMBD, P.rl, Q_EXPERTS + 1, false, 0, one);
    L.launch(q4("p_router"), {P.rl, pp, P.ids, P.w}, (T + 3) / 4, 1, 1, 128);
    const uint32_t nasg = T * Q_USED;
    L.launch(q4("p_moe_count"), {P.ids, pp, P.cnt}, (nasg + 63) / 64, 1, 1, 64);
    if (fetch) L.copy(P.routed + (size_t)l * Q_EXPERTS, P.cnt, Q_EXPERTS * 4, (*pevents_)[pev(kRouted, l, c)]);
    L.launch(q4("p_moe_scan"), {P.cnt, Ly.slot_map, P.stage_map[l], P.base, P.cursor, P.tiles_c, P.ntiles_c,
                                P.tiles_s, P.ntiles_s, P.prof ? P.prof + l * Q_EXPERTS : nullptr}, 1, 1, 1, Q_EXPERTS);
    L.launch(q4("p_moe_fill"), {P.ids, pp, P.base, P.cursor, P.inv}, (nasg + 63) / 64, 1, 1, 64);
    const uint32_t xsig = quant_pairs_sigma(Ly.gu_type), hsig = quant_pairs_sigma(Ly.down_type);
    L.launch(q4("p_moe_scatter"), {P.x16, P.inv, pp, xsig, P.xg}, 1, T, 1, Q_EMBD / 8);
    const uint32_t gbpr = Q_EMBD / quant_block_values(Ly.gu_type), dqk = quant_block_values(Ly.down_type);
    std::string gm = std::string("qgemm_") + ggml_type_label(Ly.gu_type), dm = std::string("qgemm_") +
                                                                                ggml_type_label(Ly.down_type);
    for (auto &ch : gm) ch = (char)std::tolower(ch);
    for (auto &ch : dm) ch = (char)std::tolower(ch);
    const uint32_t E = Q_EMBD, ff = card.ff;
    uint8_t *stage = P.stage + (l % 2) * P.stage_slot;
    // The cached experts and the shared expert first: they give the staging its last moments. Each expert's rows of
    // hg and yg are its own, so the order changes no result.
    if (nslots_cached_ > 0) {
      L.launch(kernel(dev, gm, "moe_gate_up_t"), {Ly.cache, Ly.slot_bytes, P.tiles_c, P.ntiles_c, ff, gbpr, Ly.up_off,
                                                P.xg, E, E, P.hg, hsig, card.grids}, ff / 64, mt, 1, 64);
      uint32_t dbpr = ff / dqk;
      L.launch(kernel(dev, dm, "moe_down_t"), {Ly.cache, Ly.slot_bytes, Ly.down_off, P.tiles_c, P.ntiles_c, E, dbpr, zero,
                                             P.hg, ff, P.yg, card.grids}, E / 64, mt, 1, 64);
    }
    dense(Ly.sh_gate, P.x16, Q_EMBD, P.shg, ff, false);
    dense(Ly.sh_up, P.x16, Q_EMBD, P.shu, ff, false);
    L.launch(q4("p_swiglu"), {P.shg, P.shu, ff, pp, P.shh}, (ff + 63) / 64, T, 1, 64);
    dense(Ly.sh_down, P.shh, ff, P.s, Q_EMBD, false);
    L.wait({(*pevents_)[pev(kStaged, l, c)]});
    if (!hl.pieces.empty())
      L.launch(kernel(dev, gm + "_file", "moe_gate_up_t"), {stage, hl.stg_bytes, P.tiles_s, P.ntiles_s, ff, gbpr,
                                                          hl.stg_up_off, P.xg, E, E, P.hg, hsig, card.grids}, ff / 64, mt, 1, 64);
    if (!hl.pieces.empty()) {
      // The staged down holds this card's blocks of each row only, unless the arena keeps it whole.
      const bool part = hl.down_parts > 1;
      uint32_t dbpr = (part ? ff : Q_FF) / dqk, b0 = part ? 0 : c * ff / dqk;
      L.launch(kernel(dev, dm + "_file", "moe_down_t"), {stage, hl.stg_bytes, hl.stg_down_off, P.tiles_s, P.ntiles_s, E,
                                                       dbpr, b0, P.hg, ff, P.yg, card.grids}, E / 64, mt, 1, 64);
    }
    L.signal((*pevents_)[pev(kFree, l, c)]);
    L.launch(q4("p_moe_combine"), {P.yg, P.inv, P.w, P.s, P.rl, pp, P.part, peer_part(1)}, Q_EMBD / 128, T, 1, 64, 1,
             1, sent(1));
    exchange(l, 1, P.inj[1], l >= 1 && l + 1 < card.l1 ? card.layers[l + 1].hc_norm[0] : nullptr);
  }
  if (!card.last()) {  // the residuals to the next group's card of this rank
    Card &next = group_card(card.g + 1, c);
    L.copy(next.pc->R, P.R, (size_t)T * Q_HC_DIM * 4, (*pevents_)[pev(kHand, 0, next.dev)]);
  } else {
    if (split) {  // each run's rows the head and the draft layer read, on both cards
      for (uint32_t r = 0; r < runs; r++)
        L.launch(q4("p_hc_share"), {P.R, run(r), r0, r1, peer->R}, Q_HC_DIM / 64, STEP_MAX_ROWS, 1, 64, 1, 1,
                 r + 1 == runs ? (*pevents_)[pev(kLast, 0, c)] : nullptr);
      L.wait({(*pevents_)[pev(kLast, 0, 1 - c)]});
    }
    // Each run's last row's head, as in the decode step: run r's in row r.
    for (uint32_t r = 0; r < runs; r++)
      L.launch(q4("p_take_last"), {P.R, run(r), card.R + (size_t)r * Q_HC_DIM}, Q_HC_DIM / 64, 1, 1, 64);
    record_head(L, card, head_plan_, runs);
  }
  L.close();
  ze::List &ref = *list;
  P.lists[key] = std::move(list);
  return ref;
}

std::vector<Candidate> Model::prompt(uint32_t slot, uint32_t pos0, const int32_t *tokens, uint32_t n, int32_t p1,
                                     int32_t p2, const int32_t *next, uint32_t next_n) {
  const PromptRun run{slot, pos0, tokens, n, p1, p2};
  return std::move(prompt_runs(&run, 1, next, next_n)[0]);
}

std::vector<std::vector<Candidate>> Model::prompt_runs(const PromptRun *runs, uint32_t nruns, const int32_t *next,
                                                       uint32_t next_n) {
  if (tmax_ == 0) throw std::runtime_error("the prompt path is off");
  if (nruns == 0 || nruns > PROMPT_MAX_RUNS) throw std::runtime_error("a prompt chunk takes 1 to 4 runs");
  uint32_t n = 0;
  for (uint32_t r = 0; r < nruns; r++) {
    const PromptRun &q = runs[r];
    if (q.slot >= opt_.slots) throw std::runtime_error("slot out of range");
    if (q.n == 0 || q.n > tmax_) throw std::runtime_error("prompt chunk out of range");
    if (q.pos0 + q.n > opt_.max_ctx) throw std::runtime_error("prompt past the context");
    for (uint32_t k = 0; k < r; k++)
      if (runs[k].slot == q.slot) throw std::runtime_error("a prompt chunk takes one run a sequence");
    n += q.n;
  }
  if (n > tmax_) throw std::runtime_error("prompt chunk out of range");
  if (nruns > 1) next = nullptr;
  wait_commit();
  if (move_bg_.joinable()) move_bg_.join();  // the staging copies read arena places the last swap round still fills
  if (copies_dirty_) {
    const auto t0 = std::chrono::steady_clock::now();
    const Staging st = staging_now();
    for (auto &card : cards_) {
      retired_.push_back(std::move(card->pc->copy));
      card->pc->copy = prompt_copies(*card, st, copy_count_);
    }
    copies_dirty_ = false;
    std::fprintf(stderr, "engine: the prompt path's staging copies recorded again in %.1f ms, %u copies a card\n",
                 std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(), copy_count_);
  }
  const uint32_t T = bucket(n, tmax_), R = nruns > 1 ? PROMPT_MAX_RUNS : 1;
  const bool fetch = T <= kFetchRows && cards_[0]->pc->fetch;
  std::vector<ze::List *> lists;
  for (auto &card : cards_) {
    lists.push_back(&prompt_list(*card, T, R));
    card->pc->last = R << 16 | T;
  }

  ++plan_->step;
  pplan_[0] = {n, runs[0].pos0, runs[0].slot, 0};
  prows_[0] = std::min(n, T / 2), prows_[1] = n - prows_[0];  // each card's rows when they split (prompt_list)
  head_plan_->nrows = nruns;
  const GgufTensor &te = gguf_.tensor("token_embd.weight");
  auto to_float = ggml_get_type_traits((ggml_type)te.type)->to_float;
  std::vector<uint32_t> row0(nruns);
  for (uint32_t r = 0, at = 0; r < PROMPT_MAX_RUNS; r++) {
    if (r >= nruns) {
      pplan_[1 + r] = {0, 0, 0, 0};
      continue;
    }
    const PromptRun &q = runs[r];
    row0[r] = at;
    pplan_[1 + r] = {q.n, q.pos0, q.slot, at};
    // The positions of the run's cells, from its first block's first cell (step.h).
    uint32_t *ppos = ppos_ + (size_t)r * ppos_run();
    const uint32_t base = q.pos0 / Q_IDX_BLOCK * Q_IDX_BLOCK;
    for (uint32_t e = 0; e < q.pos0 + q.n - base; e++) {
      const auto p = position(q.slot, base + e);
      for (uint32_t k = 0; k < 3; k++) ppos[4 * e + k] = p[k];
    }
    pool_->run(q.n, [&](uint64_t t) {
      to_float(te.data + (uint64_t)q.tokens[t] * te.row_bytes(), pembed_ + (at + t) * Q_EMBD, Q_EMBD);
    });
    place_images(q.slot, q.pos0, q.n, at);
    at += q.n;
  }
  for (uint32_t i = 0; i < pevents_->size(); i++) ZE_CHECK(zeEventHostReset((*pevents_)[i]));
  // A short chunk's uncached experts, each layer's hottest first by the swap counts (then by id): a thread a card
  // fetches them (fetch_experts). A thread that fails signals its card's remaining kStaged, so the lists end, and
  // its error is thrown once they have.
  std::vector<std::vector<uint32_t>> order;
  std::vector<std::exception_ptr> fetch_err(ncards_);
  std::vector<std::thread> fetchers;
  struct JoinAll {
    std::vector<std::thread> &t;
    ~JoinAll() {
      for (auto &x : t)
        if (x.joinable()) x.join();
    }
  } join_fetchers{fetchers};
  if (fetch) {
    order.resize(Q_LAYERS);
    for (uint32_t l = 0; l < Q_LAYERS; l++) {
      const float *u = &use_[(size_t)l * Q_EXPERTS];
      order[l] = host_layers_[l]->uncached;
      std::sort(order[l].begin(), order[l].end(),
                [&](uint32_t a, uint32_t b) { return u[a] != u[b] ? u[a] > u[b] : a < b; });
    }
    for (uint32_t c = 0; c < ncards_; c++)
      fetchers.emplace_back([this, c, &order, &fetch_err] {
        try {
          fetch_experts(c, order);
        } catch (...) {
          fetch_err[c] = std::current_exception();
          const Card &card = *cards_[c];
          for (uint32_t l = card.l0; l < card.l1; l++) zeEventHostSignal((*pevents_)[pev(kStaged, l, card.c)]);
        }
      });
  }
  for (uint32_t c = ncards_; c-- > 0;) {
    if (!fetch) cards_[c]->pc->copy->execute();
    lists[c]->execute();
  }
  // The n-gram rows, read while the cards run layer 0; they wait for the event just before layer 1 (ADR-0015). A
  // chunk the last one named as next has its rows already, read during that run, since a gather can outlast layer 0
  // (docs/measurements/MEASUREMENTS.md#2026-10-17-ngram-ahead).
  const PromptRun &q0 = runs[0];
  const bool ahead = nruns == 1 && ahead_.valid && ahead_.slot == q0.slot && ahead_.pos0 == q0.pos0 &&
                     ahead_.p1 == q0.p1 && ahead_.p2 == q0.p2 && ahead_.tokens.size() == n &&
                     std::equal(q0.tokens, q0.tokens + n, ahead_.tokens.begin());
  ahead_.valid = false;
  const float *rows = ahead ? ahead_.rows.data() : pngram_;
  if (!ahead)
    for (uint32_t r = 0; r < nruns; r++)
      ngram_->gather_many(runs[r].tokens, runs[r].n, runs[r].p1, runs[r].p2, pngram_ + (size_t)row0[r] * Q_EMBD);
  pool_->run(n, [&](uint64_t t) {
    for (uint32_t i = 0; i < Q_EMBD; i++) pngram16_[t * Q_EMBD + i] = bf16_bits(rows[t * Q_EMBD + i]);
  });
  ZE_CHECK(zeEventHostSignal((*pevents_)[pev(kNgram, 0, 0)]));
  // The next chunk's rows, on a thread of its own while the host waits for the cards; it ends before this call does,
  // so the n-gram cache has one user at a time.
  std::thread gather_next;
  bool gathered = false;
  struct Join {
    std::thread &t;
    ~Join() {
      if (t.joinable()) t.join();
    }
  } join{gather_next};
  if (next && next_n > 0 && next_n <= tmax_) {
    ahead_.slot = q0.slot;
    ahead_.pos0 = q0.pos0 + n;
    ahead_.p1 = q0.tokens[n - 1];
    ahead_.p2 = n >= 2 ? q0.tokens[n - 2] : q0.p1;
    ahead_.tokens.assign(next, next + next_n);
    ahead_.rows.resize((size_t)tmax_ * Q_EMBD);
    gather_next = std::thread([this, &gathered] {
      try {  // a failed read is the next chunk's to report, when it reads its rows itself
        ngram_->gather_many(ahead_.tokens.data(), (uint32_t)ahead_.tokens.size(), ahead_.p1, ahead_.p2,
                            ahead_.rows.data());
        gathered = true;
      } catch (...) {
      }
    });
  }
  for (uint32_t c = 0; c < ncards_; c++) {
    lists[c]->synchronize();
    if (!fetch) cards_[c]->pc->copy->synchronize();
  }
  for (auto &x : fetchers) x.join();
  for (auto &e : fetch_err)
    if (e) std::rethrow_exception(e);
  if (gather_next.joinable()) {
    gather_next.join();
    ahead_.valid = gathered;
  }
  std::vector<std::vector<Candidate>> out(nruns);
  for (uint32_t r = 0; r < nruns; r++) {
    auto &o = out[r];
    for (uint32_t c = 0; c < width_; c++) {  // the last group's ranks
      const size_t base = ((size_t)c * STEP_MAX_ROWS + r) * opt_.candidates;
      for (uint32_t i = 0; i < top_n_[c * STEP_MAX_ROWS + r]; i++) o.push_back({(int32_t)top_idx_[base + i], top_val_[base + i]});
    }
    std::sort(o.begin(), o.end(), [](const Candidate &a, const Candidate &b) {
      return a.logit != b.logit ? a.logit > b.logit : a.token < b.token;
    });
    if (o.size() > opt_.candidates) o.resize(opt_.candidates);
  }
  return out;
}

// The sum of each group's first card's counts, which hold its own layers'.
std::vector<uint32_t> Model::routing_counts() const {
  std::vector<uint32_t> v(Q_LAYERS * Q_EXPERTS, 0), w(v.size());
  for (uint32_t g = 0; g < ngroups_; g++) {
    Card &card = group_card(g, 0);
    const PromptCard *P = card.pc.get();
    if (!P || !P->prof) return v;
    card.up->copy(w.data(), P->prof, w.size() * 4);
    card.up->synchronize();
    for (size_t i = 0; i < v.size(); i++) v[i] += w[i];
  }
  return v;
}

std::vector<TraceRow> trace_rows(const ze::List &L, const uint64_t *ts, const ze::Card &card) {
  std::map<std::string, TraceRow> by;
  for (size_t i = 0; i < L.trace_names().size(); i++) {
    uint64_t d = (ts[i + 1] - ts[i]) & card.timestamp_mask;
    TraceRow &r = by[L.trace_names()[i]];
    r.kernel = L.trace_names()[i];
    r.launches++;
    r.ms += 1e3 * (double)d / (double)card.timer_hz;
  }
  std::vector<TraceRow> rows;
  for (auto &[k, r] : by) rows.push_back(r);
  std::sort(rows.begin(), rows.end(), [](const TraceRow &a, const TraceRow &b) { return a.ms > b.ms; });
  return rows;
}

std::vector<TraceRow> Model::prompt_trace(uint32_t c) const {
  const PromptCard &P = *cards_.at(c)->pc;
  auto it = P.trace.find(P.last);
  if (it == P.trace.end()) return {};
  return trace_rows(*P.lists.at(P.last), it->second, rt_->cards()[c]);
}

std::vector<TraceRow> Model::step_trace(uint32_t c) const {
  const Card &card = *cards_.at(c);
  auto it = card.trace.find(last_width_);
  if (it == card.trace.end()) return {};
  return trace_rows(*card.lists.at(last_width_), it->second, rt_->cards()[c]);
}

}  // namespace engine::qwen4exp
