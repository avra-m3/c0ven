// deepseek41's prompt chunk path: up to Options::prompt_chunk rows of one sequence through the layers at once, on both
// cards. A layer runs in two lists a card, with the host between them: the first ends at the router, the host then
// makes the tile lists of the experts the cards hold and has the CPU answer the others, and the second runs the
// experts, the shared expert, the cross-card sum and the next layer's input. The cards split the work as the decode
// step does (heads, output groups, expert parts) and exchange partial sums through each other's memory.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <immintrin.h>

#include <stdexcept>

#include "ggml.h"
#include "models/deepseek41/prompt.h"
#include "quant.h"

namespace engine::deepseek41 {

namespace {

constexpr uint32_t E = D41_EMBD, D = D41_HC_DIM;
constexpr uint32_t kHeads = D41_H / 2;                     // a card's query heads
constexpr uint32_t kGroups = D41_OG / 2;                   // and its output groups
constexpr uint32_t kEngA8Row = D41_ENG_G + D41_ENG_G / 8;  // a token's gathered Engram rows as a8
constexpr uint32_t kEngKvRows = 2 * E + E / 2;             // a card's rows of engram_kv
constexpr uint32_t kTopkLocal = 1024;
constexpr uint32_t kQKV = kHeads * D41_HD;                 // a card's q values a row
constexpr uint32_t kOutA = kGroups * D41_OLR;              // and out_a's
const void *const none = nullptr;

uint32_t ratio_of(uint32_t l) { return (uint32_t)D41_RATIO((int)l); }
bool kv_source(uint32_t l) { return D41_KV_SRC((int)l) == (int)l; }
bool idx_source(uint32_t l) { return ratio_of(l) != 0 && D41_IDX_SRC((int)l) == (int)l; }
int engram_of(uint32_t l) { return D41_ENGRAM((int)l); }

std::string lower(std::string s) {
  for (char &ch : s) ch = (char)std::tolower((unsigned char)ch);
  return s;
}

// A row of K values as a8 (kernels/include/a8.h).
void a8_row(const float *x, uint32_t K, uint8_t *a) {
  for (uint32_t u = 0; u < K / 32; u++) {
    float amax = 0;
    for (uint32_t i = 0; i < 32; i++) amax = std::fmax(amax, std::fabs(x[32 * u + i]));
    const float d = amax / 127.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
    for (uint32_t i = 0; i < 32; i++) a[32 * u + i] = (uint8_t)(int8_t)std::round(x[32 * u + i] * id);
    std::memcpy(a + K + 4 * u, &d, 4);
  }
}

// A card's half of a staged expert in the file's layout: the gate and up rows of the card's class and the down blocks of
// those rows' columns, each matrix after the last at a 256-byte boundary. The cards take the two classes in turn, so the
// two halves of an expert are the cached pairs' two halves.
struct StageLayout {
  uint32_t rows = 0, nb = 0;             // gate and up rows, and the down blocks a row of the part
  uint64_t rb = 0, bb = 0;               // bytes of a gate row, and of a down block
  uint64_t r0 = 0, b0 = 0;               // the first gate row, and the first down block of the part
  uint64_t gate_bytes = 0, up_off = 0, down_off = 0, bytes = 0;
  bool part_b = false;
};

StageLayout stage_layout(const LayerExperts &le, uint32_t l, uint32_t c) {
  StageLayout s;
  const uint32_t gt = le.shape.gate_type, dt = le.shape.down_type;
  s.part_b = (c ^ (l & 1)) != 0;
  s.rows = s.part_b ? D41_PART_B : D41_PART_A;
  s.rb = (E / quant_block_values(gt)) * quant_block_bytes(gt);
  s.bb = quant_block_bytes(dt);
  s.nb = s.rows / quant_block_values(dt);
  s.r0 = s.part_b ? D41_PART_A : 0;
  s.b0 = s.r0 / quant_block_values(dt);
  s.gate_bytes = s.rows * s.rb;
  s.up_off = (s.gate_bytes + 255) / 256 * 256;
  s.down_off = 2 * s.up_off;
  s.bytes = (s.down_off + (uint64_t)E * s.nb * s.bb + 255) / 256 * 256;
  return s;
}

using Clock = std::chrono::steady_clock;
double ms(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }

}  // namespace

Model::Card::~Card() = default;

// What one chunk keeps while it runs, and the host's plan for the layer in hand.
struct Model::ChunkRun {
  uint32_t n = 0, slot = 0, pos0 = 0, step = 0;
  bool logits = false;
  uint32_t groups() const { return (n + kChunkGroup - 1) / kChunkGroup; }
  uint32_t group_rows(uint32_t g) const { return std::min(kChunkGroup, n - g * kChunkGroup); }
  // The layer's cached assignments in expert order, by card and class.
  uint32_t nasg = 0;
  std::vector<MoeTile> tiles[2][2];
  std::vector<MoeTile> stiles[2];  // the staged experts' tiles by card
};

void Model::alloc_chunk(Card &card) {
  const uint32_t cap = opt_.prompt_chunk;
  if (cap % 64 || cap > D41_CHUNK_MAX) throw std::runtime_error("deepseek41's prompt chunk is a multiple of 64, at most 4,352");
  if (opt_.slots * D41_RING_WIDE * 1ull * D41_HD * 2 > (1ull << 31)) throw std::runtime_error("too many slots for a prompt chunk");
  card.ck = std::make_unique<ChunkCard>();
  ChunkCard &k = *card.ck;
  k.cap = cap;
  const size_t R = cap, R8 = cap + 8, G = kChunkGroup, A = (size_t)R * D41_USED;
  const auto dv = [&](size_t bytes) { return rt_->alloc_device(card.c, bytes, 4096); };
#define DEV(T, n) static_cast<T *>(dv((size_t)(n) * sizeof(T)))
  k.words = DEV(uint32_t, 16);
  k.real = DEV(uint32_t, cap / 8 + 2);
  k.e = DEV(float, R * E);
  k.X = DEV(float, R * D);
  k.X1 = DEV(float, R * D);
  k.ss = DEV(float, R8);
  k.m = DEV(float, R8 * D41_MIX);
  k.amix = DEV(D41HcMix, R8);
  k.fmix = DEV(D41HcMix, R8);
  k.xf = DEV(float, R * E);
  k.xh = DEV(uint16_t, R * E);
  k.xb = DEV(uint16_t, R * E);
  k.xa = DEV(uint8_t, D41_A8_ROW);
  k.qr = DEV(float, R * D41_QLR);
  k.kv = DEV(float, R * D41_HD);
  k.qf = DEV(float, R * D41_QLR);
  k.qh = DEV(uint16_t, R * D41_QLR);
  k.qb = DEV(uint16_t, R * D41_QLR);
  k.qa = DEV(uint8_t, R * D41_QR_A8_ROW);
  k.ckv = DEV(float, R * D41_HD);
  k.csc = DEV(float, R * D41_HD);
  k.idx = DEV(uint32_t, R * D41_TOPK + 16);
  k.rows = DEV(D41Row, R8);
  k.rows_c = DEV(D41Row, R8);
  k.q = DEV(float, G * kQKV);
  k.iq = DEV(float, G * D41_IDX_H * D41_IDX_D);
  k.iw = DEV(float, G * D41_IDX_H);
  k.score = DEV(float, G * D41_CTX_CAP);
  k.part = DEV(float, G * kHeads * D41_ATT_PART);
  k.of = DEV(float, G * kQKV);
  k.oa = DEV(uint8_t, kGroups * G * D41_OG_A8_ROW);
  k.oh = DEV(uint16_t, G * kQKV);
  k.aa = DEV(uint16_t, R * kOutA);
  k.y = DEV(float, R * E);
  for (auto &b : k.in) b = DEV(float, R * E);
  k.w16a = DEV(uint16_t, 20971520);
  k.w16b = DEV(uint16_t, 20971520);
  k.ring = DEV(uint16_t, (size_t)opt_.slots * D41_RING_WIDE * D41_HD);
  k.eng_a8 = DEV(uint8_t, R * kEngA8Row);
  k.ekv = DEV(float, R * kEngKvRows);
  k.eown = DEV(D41EngramSend, R);
  k.ein = DEV(D41EngramSend, R);
  k.rlogits = DEV(float, R8 * D41_EXPERTS);
  k.w = DEV(float, (R8)*D41_USED);
  k.sgu = DEV(float, R * 2 * D41_PART_A);
  k.shared = DEV(float, R * E);
  k.out = DEV(float, R * E);
  k.ids = DEV(int32_t, R8 * D41_USED);
  k.slots = DEV(int32_t, R8 * D41_USED);
  k.nmiss = DEV(uint32_t, R8);
  k.ngroups = DEV(uint32_t, 64);
  k.groups = DEV(D41Group, 8 * D41_USED);
  k.xg = DEV(uint16_t, A * E);
  k.hbuf = DEV(uint16_t, A * D41_PART_A);
  k.ybuf = DEV(uint16_t, A * E);
  k.sh = DEV(uint16_t, R * D41_PART_A);
  k.yidx = DEV(int32_t, A);
  k.rowidx = DEV(uint32_t, A);
  for (auto &t : k.tiles) t = DEV(MoeTile, A);
  k.tiles_s = DEV(MoeTile, A);
#undef DEV
  for (uint32_t l = 0; l < D41_LAYERS; l++) k.stage_slot = std::max(k.stage_slot, stage_layout(experts_[l], l, card.c).bytes);
  // The file layout's kernels read a few bytes past the last block.
  // The CPU writes the staged experts through a mapping of the buffer (ADR-0065 of the other model's path): the driver
  // writes the whole buffer first, then each page is touched through the mapping, as pages touched before the driver's
  // first copy fault again after it.
  {
    const size_t bytes = (size_t)kStageBuffers * k.stage_slot + QUANT_FILE_PAD;
    k.stage = static_cast<uint8_t *>(rt_->alloc_device_mapped(card.c, bytes, k.stage_cpu));
    constexpr size_t piece = 4u << 20;
    uint8_t *zero = static_cast<uint8_t *>(rt_->alloc_host(piece));
    std::memset(zero, 0, piece);
    ze::List warm(*rt_, card.c, ze::List::immediate, true);
    for (size_t off = 0; off < bytes; off += piece) warm.copy(k.stage + off, zero, std::min(piece, bytes - off));
    warm.synchronize();
    rt_->free(zero);
    for (size_t off = 0; off < bytes; off += 4096) *(volatile uint8_t *)(k.stage_cpu + off) = 0;
    _mm_sfence();
  }
  if (card.c != 0) return;
  chunk_cap_ = cap;
  const auto host = [&](size_t bytes) {
    void *p = rt_->alloc_host(bytes);
    std::memset(p, 0, bytes);
    return p;
  };
  cembed_ = static_cast<float *>(host(R * E * 4));
  crows_ = static_cast<D41Row *>(host(R8 * sizeof(D41Row)));
  crowsc_ = static_cast<D41Row *>(host(R8 * sizeof(D41Row)));
  cwords_ = static_cast<uint32_t *>(host(128));
  creal_ = static_cast<uint32_t *>(host((cap / 8 + 2) * 4));
  cjobs_ = static_cast<D41MissJob *>(host((R8) * sizeof(D41MissJob)));
  cpu_cy_ = static_cast<float *>(host(R8 * E * 4));
  ceng_a8_ = static_cast<uint8_t *>(host(R * kEngA8Row));
  crowidx_ = static_cast<uint32_t *>(host(A * 4));
  cyidx_ = static_cast<int32_t *>(host(A * 4));
  cnmiss_ = static_cast<uint32_t *>(host(R8 * 4));
  for (auto &c : ctiles_)
    for (auto &t : c) t = static_cast<MoeTile *>(host(A * sizeof(MoeTile)));
  for (auto &t : ctiles_s_) t = static_cast<MoeTile *>(host(A * sizeof(MoeTile)));
  if (debug_.probes) cprobe_ = static_cast<float *>(host((size_t)2 * D41_LAYERS * kChunkProbeRows * D * 4));
}

std::vector<uint32_t> Model::chunk_probe_rows(uint32_t n) const {
  std::vector<uint32_t> rows;
  for (uint32_t i = 0; i < kChunkProbeRows && i < n; i++)
    rows.push_back(n <= kChunkProbeRows ? i : (uint32_t)((uint64_t)i * (n - 1) / (kChunkProbeRows - 1)));
  return rows;
}

const float *Model::chunk_probe(uint32_t layer, bool ffn, uint32_t i) const {
  if (!cprobe_ || layer >= D41_LAYERS || i >= kChunkProbeRows) throw std::runtime_error("no such chunk probe");
  return cprobe_ + (((size_t)(ffn ? D41_LAYERS : 0) + layer) * kChunkProbeRows + i) * D;
}

std::vector<uint8_t> Model::pool_page(uint32_t layer, uint32_t page) {
  Card &card = *cards_[0];
  const Layer &ly = card.layers.at(layer);
  if (!ly.pool || page >= kv_pages_) throw std::runtime_error("no such pool page");
  const size_t bytes = (size_t)(KV_PAGE_CELLS / ratio_of(layer)) * D41_COMP_ROW_BYTES;
  std::vector<uint8_t> v(bytes);
  card.up->copy(v.data(), reinterpret_cast<const uint8_t *>(ly.pool) + page * bytes, bytes);
  card.up->synchronize();
  return v;
}

// A token's three predecessors hash with it into the Engram rows, gathered and made a8 for the chunk.
void Model::chunk_engram_rows(uint32_t table, const int32_t *tokens, const int32_t *prev, uint32_t n, uint8_t *dst) {
  std::vector<uint64_t> ids((size_t)n * D41_ENG_COLS);
  for (uint32_t r = 0; r < n; r++) {
    const auto back = [&](uint32_t k) { return r >= k ? tokens[r - k] : prev[k - r - 1]; };
    const auto h = hash_->rows(tokens[r], back(1), back(2), back(3), table);
    std::copy(h.begin(), h.end(), ids.begin() + (size_t)r * D41_ENG_COLS);
  }
  std::vector<float> f((size_t)n * D41_ENG_G);
  engram_->gather(ids.data(), (uint32_t)ids.size(), f.data());
  for (uint32_t r = 0; r < n; r++) a8_row(f.data() + (size_t)r * D41_ENG_G, D41_ENG_G, dst + (size_t)r * kEngA8Row);
}

// The matrix as f16 rows of K into out.
void Model::chunk_dequant(ze::List &L, Card &card, const Mat &m, uint16_t *out) {
  const std::string mod = "qgemm_" + lower(ggml_type_label(m.type));
  const uint32_t units = m.K / 32, zero = 0;
  L.label("dequant " + lower(ggml_type_label(m.type)));
  L.launch(kernel(card, mod.c_str(), "dequant_f16"), {m.p, m.rows, m.bpr, zero, units, out, m.K, card.grids},
           (m.rows * units + 63) / 64, 1, 1, 64);
}

// y[t][n] = w[n] . x[t] for t < rows, w: N rows of K in 16 bits. ntok: a word on the card with the count of rows x holds.
void Model::chunk_gemm(ze::List &L, Card &card, bool bf16, const uint16_t *w, uint32_t N, uint32_t K, const void *x,
                       uint32_t ldx, const uint32_t *ntok, uint32_t rows, void *y, uint32_t ldy, bool yhalf,
                       const char *name) {
  // Weights larger than the cards' cache are taken in bands of 4 blocks of rows.
  const uint32_t yh = yhalf, mode = 0, band = (uint64_t)N * K * 2 > (18u << 20) ? 4 : 0, ldw = K;
  const float scale = 1.0f;
  const void *y2 = nullptr;
  L.label(name);
  L.launch(kernel(card, bf16 ? "gemm_bf16" : "gemm_f16", "gemm"),
           {w, N, K, ldw, x, ldx, ntok, y, ldy, yh, mode, scale, y2, band}, (N + 127) / 128, (rows + 63) / 64, 1, 64);
}

void Model::chunk_to_f16(ze::List &L, Card &card, const float *in, uint16_t *out, uint64_t n) {
  L.label("to f16");
  L.launch(kernel(card, "deepseek41_prompt", "dp_to_f16"), {in, out, n}, (uint32_t)((n / 4 + 255) / 256), 1, 1, 256);
}

// The sublayer's raw mixes of every row, eight rows to a launch, then its mixes.
void Model::chunk_hc_mix(ze::List &L, Card &card, const float *fn, const ChunkRun &run) {
  ChunkCard &k = *card.ck;
  L.launch(kernel(card, "deepseek41", "ds_hc_mix"), {fn, k.X, k.ss, run.n, k.m}, D41_MIX, (run.n + STEP_MAX_ROWS - 1) / STEP_MAX_ROWS, 1,
           16 * D41_HC_KSPLIT);
}

// Engram's injection before layer l's attention, as record_engram does it for a step.
void Model::chunk_engram(ze::List &L, Card &card, uint32_t l, const ChunkRun &run) {
  ChunkCard &k = *card.ck;
  Layer &ly = card.layers[l];
  Card &other = *cards_[1 - card.c];
  const uint32_t e = (uint32_t)engram_of(l), fk = 2 * kAllLayers + e;
  const uint32_t c0 = 2 * card.c, c1 = c0 + 2, v0 = card.c * (E / 2), v1 = v0 + E / 2, stride = kEngKvRows;
  const uint32_t in_card = 0, n = run.n;
  L.copy(k.eng_a8, ceng_a8_, (size_t)n * kEngA8Row);
  for (uint32_t g0 = 0; g0 < n; g0 += STEP_MAX_ROWS)
    matvec(L, card, "engram_kv", ly.engram_kv, k.eng_a8 + (size_t)g0 * kEngA8Row, k.ekv + (size_t)g0 * kEngKvRows,
           std::min<uint32_t>(STEP_MAX_ROWS, n - g0), 8);
  L.launch(kernel(card, "deepseek41", "ds_engram_gate"),
           {k.X, k.ekv, stride, ly.engram_gate, c0, c1, v0, v1, k.eown, other.ck->ein}, c1 - c0, n, 1, 256);
  wait(L, card, "wait: other card, Engram", flag(card.c, fk), flag(other.c, fk));
  L.launch(kernel(card, "deepseek41", "ds_engram_apply"),
           {k.X, k.ekv, stride, c0, c1, v0, v1, k.eown, k.ein, in_card, k.fmix, ly.attn_norm, k.ss, none, k.xb, k.xf}, n,
           1, 1, D41_HC_SGS * 16);
}

// Layer l's attention for the whole chunk, from the input in xf and xb to the next sublayer's input.
void Model::chunk_attention(ze::List &L, Card &card, uint32_t l, const ChunkRun &run) {
  ChunkCard &k = *card.ck;
  Layer &ly = card.layers[l];
  Card &other = *cards_[1 - card.c];
  const uint32_t n = run.n, ratio = ratio_of(l), yarn = ratio != 0, first_head = card.c * kHeads;
  const bool kv_src = kv_source(l), idx_src = idx_source(l);
  uint16_t *pool = card.layers[ratio ? (uint32_t)D41_KV_SRC((int)l) : 2].pool;
  const auto attn = [&](const char *name) { return kernel(card, "deepseek41_attn_wide", name); };
  const uint32_t *ntok = k.words;  // n
  const uint32_t window = D41_WIN - 1;

  chunk_to_f16(L, card, k.xf, k.xh, (uint64_t)n * E);
  chunk_hc_mix(L, card, ly.attn_fn, run);
  L.launch(kernel(card, "deepseek41", "ds_hc_sinkhorn"), {k.m, ly.attn_scale, ly.attn_base, k.amix}, n, 1, 1, 16);

  // The 127 positions before the chunk, into the chunk's cells.
  const uint32_t nin = std::min(run.pos0, window);
  if (nin)
    L.launch(kernel(card, "deepseek41_prompt", "dp_ring_move"),
             {ly.ring, (uint32_t)D41_RING, k.ring, (uint32_t)D41_RING_WIDE, run.slot, run.pos0 - nin}, nin, 1, 1, 256);

  chunk_dequant(L, card, ly.q_a, k.w16a);
  chunk_gemm(L, card, false, k.w16a, D41_QLR, E, k.xh, E, ntok, n, k.qr, D41_QLR, false, "gemm q_a");
  chunk_dequant(L, card, ly.kv, k.w16a);
  chunk_gemm(L, card, false, k.w16a, D41_HD, E, k.xh, E, ntok, n, k.kv, D41_HD, false, "gemm kv");
  if (kv_src) {
    chunk_gemm(L, card, true, ly.comp_kv, D41_HD, E, k.xb, E, ntok, n, k.ckv, D41_HD, false, "gemm compressor_kv");
    if (ratio == 2)
      chunk_gemm(L, card, true, ly.comp_gate, D41_HD, E, k.xb, E, ntok, n, k.csc, D41_HD, false, "gemm compressor_gate");
  }
  const void *qb = idx_src ? k.qb : nullptr;
  L.launch(attn("ds_qkv_norm"), {k.qr, ly.q_norm, k.kv, ly.kv_norm, k.rows, yarn, k.qa, qb, k.qf, k.ring, none}, 2, n, 1,
           D41_QKV_SGS * 16);
  chunk_to_f16(L, card, k.qf, k.qh, (uint64_t)n * D41_QLR);
  if (kv_src) {
    const void *sc = ratio == 2 ? k.csc : nullptr, *cr = ly.cr;
    L.launch(attn("ds_compress"), {k.ckv, sc, cr, ratio == 2 ? k.rows_c : k.rows, ratio, ly.comp_norm, ly.attn_k, ly.k_norm, card.pt, npp_, ly.pool, none},
             n, 1, 1, 256);
  }

  chunk_dequant(L, card, ly.q_b, k.w16a);
  chunk_dequant(L, card, ly.out_a, k.w16b);
  for (uint32_t g = 0; g < run.groups(); g++) {
    const uint32_t g0 = g * kChunkGroup, Gn = run.group_rows(g);
    const uint32_t *gtok = k.words + (Gn == kChunkGroup ? 1 : 2);
    const D41Row *rows = k.rows + g0;
    uint32_t *idx = k.idx + (size_t)g0 * D41_TOPK;
    chunk_gemm(L, card, false, k.w16a, kQKV, D41_QLR, k.qh + (size_t)g0 * D41_QLR, D41_QLR, gtok, Gn, k.q, kQKV, false,
               "gemm q_b");
    if (idx_src) {
      chunk_gemm(L, card, true, ly.idx_q_b, D41_IDX_H * D41_IDX_D, D41_QLR, k.qb + (size_t)g0 * D41_QLR, D41_QLR, gtok, Gn, k.iq,
                 D41_IDX_H * D41_IDX_D, false, "gemm indexer q_b");
      chunk_gemm(L, card, true, ly.idx_proj, D41_IDX_H, E, k.xb + (size_t)g0 * E, E, gtok, Gn, k.iw, D41_IDX_H, false,
                 "gemm indexer proj");
      L.launch(attn("ds_idx_score"), {k.iq, k.iw, rows, ratio, card.pt, npp_, pool, k.score}, D41_IDX_WGS, Gn, 1, D41_IDX_WG_ROWS);
      L.launch(attn("ds_idx_topk"), {k.score, rows, ratio, idx}, Gn, 1, 1, kTopkLocal);
    }
    L.launch(attn("ds_attn_split"), {k.q, rows, ratio, k.ring, card.pt, npp_, pool, idx, k.part, none},
             kHeads / D41_ATT_HEADS, Gn, ratio ? D41_ATT_SPLITS : 1, D41_ATT_WG);
    L.launch(attn("ds_attn_merge"), {k.part, ly.sinks, first_head, rows, ratio, k.oa, k.of, none}, kHeads, Gn, 1, 256);
    chunk_to_f16(L, card, k.of, k.oh, (uint64_t)Gn * kQKV);
    for (uint32_t gi = 0; gi < kGroups; gi++)
      chunk_gemm(L, card, false, k.w16b + (size_t)gi * D41_OLR * D41_OG_K, D41_OLR, D41_OG_K, k.oh + (size_t)gi * D41_OG_K, kQKV,
                 gtok, Gn, k.aa + (size_t)g0 * kOutA + gi * D41_OLR, kOutA, true, "gemm out_a");
  }
  // The window the next positions read: the last 127, back into the persistent ring.
  {
    const uint32_t end = run.pos0 + n, nout = std::min(end, window);
    L.launch(kernel(card, "deepseek41_prompt", "dp_ring_move"),
             {k.ring, (uint32_t)D41_RING_WIDE, ly.ring, (uint32_t)D41_RING, run.slot, end - nout}, nout, 1, 1, 256);
  }
  chunk_dequant(L, card, ly.out_b, k.w16a);
  chunk_gemm(L, card, false, k.w16a, E, kOutA, k.aa, kOutA, ntok, n, k.y, E, false, "gemm out_b");

  const uint32_t fk = 2 * l;
  L.launch(kernel(card, "deepseek41", "ds_exch_send"), {k.y, other.ck->in[0]}, n, 1, 1, D41_EX_SGS * 16);
  wait(L, card, "wait: other card, attention", flag(card.c, fk), flag(other.c, fk));
  const void *job = l % 2 == card.c ? cjobs_ : nullptr;
  const uint32_t in_card = 0;
  L.launch(kernel(card, "deepseek41", "ds_hc_post_norm"),
           {k.X, k.y, k.in[0], in_card, k.amix, ly.ffn_norm, k.X1, k.ss, none, k.xb, k.xf, job}, n, 1, 1, D41_HC_SGS * 16);
  chunk_probe_copy(L, card, false, l, k.X1, run);
}

// Layer l's MoE block up to the router: the mixes, the router's product, and its choices of every row, eight rows to a
// launch. The posting card also leaves the CPU's records in cjobs_.
void Model::chunk_router(ze::List &L, Card &card, uint32_t l, const ChunkRun &run) {
  ChunkCard &k = *card.ck;
  Layer &ly = card.layers[l];
  const uint32_t n = run.n;
  const bool posts = l % 2 == card.c;
  chunk_to_f16(L, card, k.xf, k.xh, (uint64_t)n * E);
  L.launch(kernel(card, "deepseek41", "ds_hc_mix"), {ly.ffn_fn, k.X1, k.ss, n, k.m}, D41_MIX, (n + STEP_MAX_ROWS - 1) / STEP_MAX_ROWS, 1,
           16 * D41_HC_KSPLIT);
  L.launch(kernel(card, "deepseek41", "ds_hc_sinkhorn"), {k.m, ly.ffn_scale, ly.ffn_base, k.fmix}, n, 1, 1, 16);
  chunk_gemm(L, card, true, ly.gate_inp, D41_EXPERTS, E, k.xb, E, k.words, n, k.rlogits, D41_EXPERTS, false, "gemm router");
  const uint32_t eight = STEP_MAX_ROWS;
  for (uint32_t g0 = 0; g0 < n; g0 += STEP_MAX_ROWS) {
    const void *jobs = posts ? cjobs_ + g0 : nullptr;
    L.launch(kernel(card, "deepseek41", "ds_router"),
             {k.rlogits + (size_t)g0 * D41_EXPERTS, ly.bias, ly.slot_map, eight, ly.split, ly.flip, k.real + g0 / 8,
              &card.plan->step, k.ids + (size_t)g0 * D41_USED, k.w + (size_t)g0 * D41_USED, k.slots + (size_t)g0 * D41_USED,
              k.nmiss + g0, jobs, k.groups, k.ngroups},
             1, 1, 1, 16 * eight * D41_RT_SGS(eight));
  }
}

// The cached experts' tiles and the shared expert's part; the copies of the staged experts and the CPU's answers are
// still on their way while these run.
void Model::chunk_experts_a(ze::List &L, Card &card, uint32_t l, const ChunkRun &run) {
  ChunkCard &k = *card.ck;
  Layer &ly = card.layers[l];
  const uint32_t n = run.n;
  const LayerExperts &le = layer_experts(l);
  const uint32_t gt = le.shape.gate_type, dt = le.shape.down_type;
  const uint32_t xsig = quant_pairs_sigma(gt), hsig = quant_pairs_sigma(dt);
  const uint32_t zero = 0;

  if (run.nasg) {
    L.copy(k.rowidx, crowidx_, (size_t)run.nasg * 4);
    L.launch(kernel(card, "deepseek41_prompt", "dp_gather"), {k.xh, k.rowidx, k.xg, xsig}, run.nasg, 1, 1, 256);
    for (uint32_t ci = 0; ci < 2; ci++) {
      const size_t nt = run.tiles[card.c][ci].size();
      if (!nt) continue;
      const D41Class &cls = tables_[l][card.c][ci];
      const uint8_t *base = ly.cache + cls.base;
      const uint32_t rows = (uint32_t)cls.rows, bpr_gate = E / quant_block_values(gt);
      const uint32_t bpr_down = rows / quant_block_values(dt), ldx = E, kx = E, edim = E;
      const uint64_t slot_bytes = cls.slot_bytes, up_off = cls.up_off, down_off = cls.down_off;
      const uint32_t *count = k.words + 3 + ci;
      L.label("expert gate and up");
      L.launch(kernel(card, ("qgemm_" + lower(ggml_type_label(gt)) + "_limit10").c_str(), "moe_gate_up_t"),
               {base, slot_bytes, k.tiles[ci], count, rows, bpr_gate, up_off, k.xg, kx, ldx, k.hbuf, hsig, card.grids},
               (rows + 63) / 64, (uint32_t)nt, 1, 64);
      L.label("expert down");
      L.launch(kernel(card, ("qgemm_" + lower(ggml_type_label(dt))).c_str(), "moe_down_t"),
               {base, slot_bytes, down_off, k.tiles[ci], count, edim, bpr_down, zero, k.hbuf, rows, k.ybuf, card.grids},
               (E + 63) / 64, (uint32_t)nt, 1, 64);
    }
  }

  // The shared expert's part.
  const uint32_t ff = ly.sh_ff, two_ff = 2 * ff;
  chunk_dequant(L, card, ly.sh_gu, k.w16a);
  chunk_gemm(L, card, false, k.w16a, two_ff, E, k.xh, E, k.words, n, k.sgu, two_ff, false, "gemm shared gate and up");
  L.launch(kernel(card, "deepseek41_prompt", "dp_swiglu_f16"), {k.sgu, two_ff, ff, k.sh}, n, 1, 1, 256);
  chunk_dequant(L, card, ly.sh_down, k.w16b);
  chunk_gemm(L, card, false, k.w16b, E, ff, k.sh, ff, k.words, n, k.shared, E, false, "gemm shared down");
}

// The staged experts' tiles, the card's partial sum, the exchange and the next layer's input.
void Model::chunk_experts_b(ze::List &L, Card &card, uint32_t l, const ChunkRun &run) {
  ChunkCard &k = *card.ck;
  Card &other = *cards_[1 - card.c];
  const uint32_t n = run.n;
  const bool posts = l % 2 == card.c;
  const LayerExperts &le = layer_experts(l);
  const uint32_t gt = le.shape.gate_type, dt = le.shape.down_type;
  const uint32_t hsig = quant_pairs_sigma(dt);
  const uint32_t zero = 0;

  if (const size_t nt = run.stiles[card.c].size()) {
    const StageLayout sl = stage_layout(le, l, card.c);
    const uint32_t bpr_gate = E / quant_block_values(gt), ldx = E, kx = E, edim = E;
    const uint32_t *count = k.words + 5;
    L.label("staged expert gate and up");
    L.launch(kernel(card, ("qgemm_" + lower(ggml_type_label(gt)) + "_file_limit10").c_str(), "moe_gate_up_t"),
             {k.stage, k.stage_slot, k.tiles_s, count, sl.rows, bpr_gate, sl.up_off, k.xg, kx, ldx, k.hbuf, hsig, card.grids},
             (sl.rows + 63) / 64, (uint32_t)nt, 1, 64);
    L.label("staged expert down");
    L.launch(kernel(card, ("qgemm_" + lower(ggml_type_label(dt)) + "_file").c_str(), "moe_down_t"),
             {k.stage, k.stage_slot, sl.down_off, k.tiles_s, count, edim, sl.nb, zero, k.hbuf, sl.rows, k.ybuf, card.grids},
             (E + 63) / 64, (uint32_t)nt, 1, 64);
  }

  const void *cpu = posts ? cpu_cy_ : nullptr;
  L.launch(kernel(card, "deepseek41_prompt", "dp_combine"),
           {k.ybuf, k.yidx, k.w, k.shared, cpu, k.nmiss, k.out, other.ck->in[1]}, n, 1, 1, 256);
  const uint32_t fk = 2 * l + 1, in_card = 0;
  wait(L, card, "wait: other card, MoE", flag(card.c, fk), flag(other.c, fk));

  const bool last = l + 1 == D41_LAYERS, engram = !last && engram_of(l + 1) >= 0;
  if (!last) {
    const void *norm = engram ? nullptr : card.layers[l + 1].attn_norm;
    const void *xb = engram ? nullptr : k.xb, *xf = engram ? nullptr : k.xf;
    L.launch(kernel(card, "deepseek41", "ds_hc_post_norm"),
             {k.X1, k.out, k.in[1], in_card, k.fmix, norm, k.X, k.ss, none, xb, xf, none}, n, 1, 1, D41_HC_SGS * 16);
  } else {
    // Only the last row's output is read, and only when the caller wants its candidates.
    if (n > 1)
      L.launch(kernel(card, "deepseek41", "ds_hc_post_norm"),
               {k.X1, k.out, k.in[1], in_card, k.fmix, none, k.X, k.ss, none, none, none, none}, n - 1, 1, 1,
               D41_HC_SGS * 16);
    const size_t at = n - 1;
    const void *norm = run.logits ? card.out_norm : nullptr, *xa = run.logits ? k.xa : nullptr;
    L.launch(kernel(card, "deepseek41", "ds_hc_post_norm"),
             {k.X1 + at * D, k.out + at * E, k.in[1] + at * E, in_card, k.fmix + at, norm, k.X + at * D, k.ss + at, xa, none,
              none, none},
             1, 1, 1, D41_HC_SGS * 16);
    if (run.logits) {
      matvec(L, card, "output", card.head, k.xa, card.logits, 1, 4);
      L.launch(kernel(card, "common", "topk_rows"),
               {card.logits, card.vocab_n, card.vocab_n, opt_.candidates, card.vocab0, &card.plan->real,
                top_idx_ + (size_t)card.c * STEP_MAX_ROWS * opt_.candidates,
                top_val_ + (size_t)card.c * STEP_MAX_ROWS * opt_.candidates, top_n_ + (size_t)card.c * STEP_MAX_ROWS},
               1, 1, 1, kTopkLocal);
    }
  }
  chunk_probe_copy(L, card, true, l, k.X, run);
}

// Puts the card's half of expert w into staging slot `slot` in host memory: its rows of gate and up, and its part of
// down. The arena keeps down as part A's blocks of every row and then part B's; the file's down is whole, and a part of
// it is a block range of every row.
void Model::chunk_stage_fill(Card &card, uint32_t l, uint32_t slot, const ExpertWeights &w, bool whole_down) {
  const LayerExperts &le = layer_experts(l);
  const StageLayout sl = stage_layout(le, l, card.c);
  const uint64_t dq = quant_block_values(le.shape.down_type), whole_row = (D41_FF / dq) * sl.bb, part_row = sl.nb * sl.bb;
  uint8_t *dst = card.ck->stage_cpu + (uint64_t)slot * card.ck->stage_slot;
  std::memcpy(dst, w.gate + sl.r0 * sl.rb, sl.gate_bytes);
  std::memcpy(dst + sl.up_off, w.up + sl.r0 * sl.rb, sl.gate_bytes);
  if (whole_down) {
    for (uint32_t r = 0; r < E; r++)
      std::memcpy(dst + sl.down_off + (uint64_t)r * part_row, w.down + (uint64_t)r * whole_row + sl.b0 * sl.bb, part_row);
  } else {
    const uint8_t *down = sl.part_b ? w.down + (uint64_t)E * kPartABlocks * sl.bb : w.down;
    std::memcpy(dst + sl.down_off, down, (uint64_t)E * part_row);
  }
}

void Model::chunk_probe_copy(ze::List &L, Card &card, bool ffn, uint32_t l, const float *from, const ChunkRun &run) {
  if (!cprobe_ || card.c != 0) return;
  const std::vector<uint32_t> rows = chunk_probe_rows(run.n);
  for (uint32_t i = 0; i < rows.size(); i++)
    L.copy(cprobe_ + (((size_t)(ffn ? D41_LAYERS : 0) + l) * kChunkProbeRows + i) * D, from + (size_t)rows[i] * D, (size_t)D * 4);
}

std::vector<Candidate> Model::chunk_forward(uint32_t slot, const int32_t *seq, uint32_t pos, uint32_t n, bool logits) {
  int32_t prev[3];
  for (uint32_t k = 1; k <= 3; k++) prev[k - 1] = pos >= k ? seq[pos - k] : kNoToken;
  return chunk_forward(slot, seq + pos, prev, pos, n, logits);
}

std::vector<Candidate> Model::chunk_forward(uint32_t slot, const int32_t *tokens, const int32_t *prev, uint32_t pos, uint32_t n,
                                            bool logits) {
  if (!chunk_cap_) throw std::runtime_error("deepseek41 has no prompt chunk: Options::prompt_chunk is 0");
  if (n == 0 || n > chunk_cap_) throw std::runtime_error("a prompt chunk takes 1 to Options::prompt_chunk rows");
  if (slot >= opt_.slots) throw std::runtime_error("slot out of range");
  if (pos + n > opt_.max_ctx) throw std::runtime_error("position past the context");
  for (uint32_t q = pos; q < pos + n; q++) {
    if (tokens[q - pos] < 0 || (uint32_t)tokens[q - pos] >= vocab_) throw std::runtime_error("a token outside the vocabulary");
    if (pt_[(size_t)slot * npp_ + q / KV_PAGE_CELLS] == D41_NO_PAGE) throw std::runtime_error("a row's page is not set");
  }
  ctimes_ = {};
  const auto t_start = Clock::now();
  ChunkRun run;
  run.n = n;
  run.slot = slot;
  run.pos0 = pos;
  run.logits = logits;

  // The embeddings, the rows' records and the counts the lists read.
  const GgufTensor &te = gguf_.tensor("token_embd.weight");
  const auto to_float = ggml_get_type_traits((ggml_type)te.type)->to_float;
  for (uint32_t r = 0; r < n; r++) {
    crows_[r] = {slot, pos + r, (pos + r) % D41_RING_WIDE, r ? r - 1 : D41_NO_ROW};
    // An even row only stores its entry in the ratio-2 compressor's ring of 16, and one more than 16 rows from the end
    // would be overwritten by a later row of the same cell, in no fixed order: it stores nothing.
    crowsc_[r] = crows_[r];
    if (!((pos + r) & 1) && r + 16 < n) crowsc_[r].slot = D41_NO_ROW;
    to_float(te.data + (uint64_t)tokens[r] * te.row_bytes(), cembed_ + (size_t)r * E, E);
  }
  for (uint32_t g = 0; g * 8 < n; g++) creal_[g] = std::min<uint32_t>(8, n - g * 8);
  std::memset(cwords_, 0, 128);
  for (uint32_t c = 0; c < 2; c++) {
    cwords_[16 * c] = n;
    cwords_[16 * c + 1] = kChunkGroup;
    cwords_[16 * c + 2] = run.group_rows(run.groups() - 1);
  }
  ctimes_.prepare += ms(t_start, Clock::now());

  // With tracing, the card's timestamps after every launch of list `which` (0: to the routers, 1 and 2: the second list's
  // parts), added to the kernel's row.
  constexpr size_t kTraceCap = 16384;
  const auto trace_on = [&](Card &cp, ze::List &L, uint32_t which) {
    if (!chunk_tuning.trace) return;
    if (!cp.ck->ts[which]) cp.ck->ts[which] = static_cast<uint64_t *>(rt_->alloc_host(kTraceCap * 8));
    L.trace(cp.ck->ts[which], kTraceCap);
  };
  const auto trace_add = [&](Card &cp, ze::List &L, uint32_t which) {
    if (!chunk_tuning.trace) return;
    const ze::Card &dev = rt_->cards()[cp.c];
    const uint64_t *stamp = cp.ck->ts[which];
    auto &rows = ctrace_[cp.c];
    const std::vector<std::string> &names = L.trace_names();
    for (size_t i = 0; i < names.size(); i++) {
      auto row = std::find_if(rows.begin(), rows.end(), [&](const TraceRow &r) { return r.kernel == names[i]; });
      if (row == rows.end()) row = rows.insert(rows.end(), TraceRow{names[i], 0, 0});
      row->launches++;
      row->ms += (double)((stamp[i + 1] - stamp[i]) & dev.timestamp_mask) * 1e3 / (double)dev.timer_hz;
    }
  };
  for (auto &t : ctrace_) t.clear();

  const auto start_lists = [&](bool second) {
    plan_->step = ++step_;
    plan_->real = 1;
    plan_->stuck = 0;
    plan_->start = 0;
    for (auto &cp : cards_) {
      auto &list = second ? cp->ck->p2 : cp->ck->p1;
      if (!list) list = std::make_unique<ze::List>(*rt_, cp->c, ze::List::regular);
      list->reset_list();
      trace_on(*cp, *list, second ? 1 : 0);
      list->copy(cp->plan, plan_, sizeof(StepWords));
      list->copy(cp->ck->words, cwords_ + 16 * cp->c, 64);
    }
  };
  const auto run_lists = [&](bool second) {
    for (auto &cp : cards_) {
      ze::List &L = *(second ? cp->ck->p2 : cp->ck->p1);
      L.copy(stuck_ + 16 * cp->c, &cp->plan->stuck, 4);
      L.close();
    }
    for (auto &cp : cards_) (second ? cp->ck->p2 : cp->ck->p1)->execute();
    for (auto &cp : cards_) (second ? cp->ck->p2 : cp->ck->p1)->synchronize();
    for (auto &cp : cards_) trace_add(*cp, *(second ? cp->ck->p2 : cp->ck->p1), second ? 1 : 0);
    for (uint32_t c = 0; c < 2; c++)
      if (stuck_[16 * c]) throw std::runtime_error("card " + std::to_string(c) + " gave up a wait of the prompt chunk");
  };

  std::vector<int32_t> ids((size_t)(n + 8) * D41_USED), slots(ids.size());
  std::vector<uint32_t> nmiss(n + 8);
  for (uint32_t l = 0; l < D41_LAYERS; l++) {
    auto t0 = Clock::now();
    if (engram_of(l) >= 0) chunk_engram_rows((uint32_t)engram_of(l), tokens, prev, n, ceng_a8_);
    ctimes_.prepare += ms(t0, Clock::now());

    // The first list: to the router.
    t0 = Clock::now();
    start_lists(false);
    for (auto &cp : cards_) {
      ze::List &L = *cp->ck->p1;
      ChunkCard &k = *cp->ck;
      if (l == 0) {
        L.copy(k.e, cembed_, (size_t)n * E * 4);
        L.copy(k.rows, crows_, (size_t)n * sizeof(D41Row));
        L.copy(k.rows_c, crowsc_, (size_t)n * sizeof(D41Row));
        L.copy(k.real, creal_, (size_t)((n + 7) / 8) * 4);
        L.launch(kernel(*cp, "deepseek41", "ds_hc_init"), {k.e, cp->layers[0].attn_norm, k.X, k.ss, none, k.xb, k.xf}, n, 1,
                 1, D41_HC_SGS * 16);
      }
      if (engram_of(l) >= 0) chunk_engram(L, *cp, l, run);
      chunk_attention(L, *cp, l, run);
      chunk_router(L, *cp, l, run);
    }
    run_lists(false);
    ctimes_.attention += ms(t0, Clock::now());

    // The host: which experts the cards hold, which the copies bring to them, and which the CPU answers.
    t0 = Clock::now();
    {
      Card &c0 = *cards_[0];
      c0.up->copy(ids.data(), c0.ck->ids, (size_t)n * D41_USED * 4);
      c0.up->copy(slots.data(), c0.ck->slots, (size_t)n * D41_USED * 4);
      c0.up->copy(nmiss.data(), c0.ck->nmiss, (size_t)n * 4);
      c0.up->synchronize();
    }
    const LayerExperts &le = layer_experts(l);
    const std::vector<uint32_t> &pairs = pairs_[l];
    const uint32_t npairs = (uint32_t)pairs.size();
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> by_pair(npairs);  // (row, rank)
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> by_miss(D41_EXPERTS);
    for (uint32_t r = 0; r < n; r++)
      for (uint32_t j = 0; j < D41_USED; j++) {
        const int32_t p = slots[(size_t)r * D41_USED + j];
        if (p >= 0)
          by_pair[(uint32_t)p].push_back({r, j});
        else
          by_miss[(uint32_t)ids[(size_t)r * D41_USED + j]].push_back({r, j});
      }

    // The experts the cards lack that enough rows route to, the ones with most rows first, up to the staging slots.
    const ChunkTuning &tu = chunk_tuning;
    struct Staged {
      uint32_t e;
      bool arena;
    };
    std::vector<Staged> staged;
    const uint32_t stage_rows = tu.stage_rows == ChunkTuning::kStageAuto ? (n < 128 ? 4 : 1) : tu.stage_rows;
    if (stage_rows) {
      std::vector<std::pair<uint32_t, uint32_t>> cand;  // (rows, expert)
      for (uint32_t e = 0; e < D41_EXPERTS; e++) {
        if (by_miss[e].size() < stage_rows) continue;
        const bool in_arena = arena_->weights(l, e).gate != nullptr;
        if (tu.source == ChunkTuning::arena_only && !in_arena) continue;
        cand.push_back({(uint32_t)by_miss[e].size(), e});
      }
      std::stable_sort(cand.begin(), cand.end(), [](auto &a, auto &b) { return a.first > b.first; });
      if (cand.size() > kStageSlots) cand.resize(kStageSlots);
      std::sort(cand.begin(), cand.end(), [](auto &a, auto &b) { return a.second < b.second; });
      for (auto &c : cand)
        staged.push_back({c.second, tu.source != ChunkTuning::file_only && arena_->weights(l, c.second).gate != nullptr});
      // The arena's first: the copies of slots run from the start, the file's after them.
      std::stable_sort(staged.begin(), staged.end(), [](const Staged &a, const Staged &b) { return a.arena > b.arena; });
    }

    // The arena's experts take the slots from 0 and the file's those from the next batch, so that a batch of slots comes
    // from one source.
    uint32_t arena_n = 0;
    while (arena_n < staged.size() && staged[arena_n].arena) arena_n++;
    const uint32_t ssd_base = (arena_n + kStageBatch - 1) / kStageBatch * kStageBatch;
    const auto stage_slot = [&](uint32_t s) { return s < arena_n ? s : ssd_base + (s - arena_n); };
    for (auto &c : run.tiles)
      for (auto &t : c) t.clear();
    for (auto &t : run.stiles) t.clear();
    std::fill(cyidx_, cyidx_ + (size_t)n * D41_USED, -1);
    uint32_t at = 0;
    for (uint32_t p = 0; p < npairs; p++) {
      const auto &list = by_pair[p];
      for (size_t i = 0; i < list.size(); i++) {
        crowidx_[at + i] = list[i].first;
        cyidx_[(size_t)list[i].first * D41_USED + list[i].second] = (int32_t)(at + i);
      }
      for (uint32_t c = 0; c < 2; c++) {
        const uint32_t ci = (uint32_t)(p >= split_[l]) ^ c ^ (l & 1);
        const uint32_t slot_in_class = p < split_[l] ? p : p - split_[l];
        for (size_t i = 0; i < list.size(); i += MOE_TILE)
          run.tiles[c][ci].push_back({pairs[p], slot_in_class, (uint32_t)(at + i), (uint32_t)std::min<size_t>(MOE_TILE, list.size() - i)});
      }
      at += (uint32_t)list.size();
    }
    run.nasg = at;
    ctimes_.cached_routings += at;
    std::vector<uint8_t> is_staged(D41_EXPERTS, 0);
    for (uint32_t s = 0; s < staged.size(); s++) {
      const auto &list = by_miss[staged[s].e];
      is_staged[staged[s].e] = 1;
      for (size_t i = 0; i < list.size(); i++) {
        crowidx_[at + i] = list[i].first;
        cyidx_[(size_t)list[i].first * D41_USED + list[i].second] = (int32_t)(at + i);
      }
      for (uint32_t c = 0; c < 2; c++)
        for (size_t i = 0; i < list.size(); i += MOE_TILE)
          run.stiles[c].push_back({staged[s].e, stage_slot(s), (uint32_t)(at + i), (uint32_t)std::min<size_t>(MOE_TILE, list.size() - i)});
      at += (uint32_t)list.size();
      ctimes_.staged_routings += (uint32_t)list.size();
    }
    run.nasg = at;
    // The CPU keeps the others: each row's record loses the staged experts.
    for (uint32_t r = 0; r < n; r++) {
      D41MissJob &job = cjobs_[r];
      uint32_t kept = 0;
      for (uint32_t m = 0; m < job.nmiss; m++) {
        if (is_staged[(uint32_t)job.ids[m]]) continue;
        job.ids[kept] = job.ids[m];
        job.w[kept] = job.w[m];
        kept++;
      }
      job.nmiss = kept;
      cnmiss_[r] = kept;
    }
    ctimes_.plan += ms(t0, Clock::now());

    t0 = Clock::now();
    // The second list in two parts: the cached experts' tiles and the shared expert now, the rest when the copies and
    // the CPU are done.
    for (uint32_t c = 0; c < 2; c++) {
      for (uint32_t ci = 0; ci < 2; ci++) cwords_[16 * c + 3 + ci] = (uint32_t)run.tiles[c][ci].size();
      cwords_[16 * c + 5] = (uint32_t)run.stiles[c].size();
    }
    start_lists(true);
    for (auto &cp : cards_) {
      ChunkCard &k = *cp->ck;
      ze::List &L = *k.p2;
      L.copy(k.yidx, cyidx_, (size_t)n * D41_USED * 4);
      L.copy(k.nmiss, cnmiss_, (size_t)n * 4);
      for (uint32_t ci = 0; ci < 2; ci++) {
        const auto &t = run.tiles[cp->c][ci];
        if (t.empty()) continue;
        std::memcpy(ctiles_[cp->c][ci], t.data(), t.size() * sizeof(MoeTile));
        L.copy(k.tiles[ci], ctiles_[cp->c][ci], t.size() * sizeof(MoeTile));
      }
      const auto &st = run.stiles[cp->c];
      if (!st.empty()) {
        std::memcpy(ctiles_s_[cp->c], st.data(), st.size() * sizeof(MoeTile));
        L.copy(k.tiles_s, ctiles_s_[cp->c], st.size() * sizeof(MoeTile));
      }
      chunk_experts_a(L, *cp, l, run);
      L.close();
      if (!k.p3) k.p3 = std::make_unique<ze::List>(*rt_, cp->c, ze::List::regular);
      k.p3->reset_list();
      trace_on(*cp, *k.p3, 2);
      chunk_experts_b(*k.p3, *cp, l, run);
      k.p3->copy(stuck_ + 16 * cp->c, &cp->plan->stuck, 4);
      k.p3->close();
    }
    const auto t_p2 = Clock::now();
    for (auto &cp : cards_) cp->ck->p2->execute();
    ctimes_.record += ms(t0, t_p2);

    // The staged experts: the arena's first, then the file's. The CPU puts each expert's two halves into host memory
    // and the copy engines take them to the cards, a batch of experts at a time.
    t0 = Clock::now();
    ctimes_.staged_arena += arena_n;
    ctimes_.staged_ssd += (uint32_t)staged.size() - arena_n;
    const auto stage_batch = [&](uint32_t s0, uint32_t count, const std::vector<ExpertWeights> &w, bool whole) {
      const auto t_f = Clock::now();
      pool_->run((uint64_t)count * 2, [&](uint64_t t) {
        const uint32_t i = (uint32_t)(t >> 1);
        chunk_stage_fill(*cards_[t & 1], l, s0 + i, w[i], whole);
      });
      ctimes_.fill += ms(t_f, Clock::now());
    };
    // The file's experts in rounds the ring holds, the first read while the arena's are copied.
    std::vector<std::vector<uint32_t>> ssd_ids;
    for (uint32_t s = arena_n; s < staged.size(); s++) {
      if (ssd_ids.empty() || ssd_ids.back().size() == kStageRound) ssd_ids.emplace_back();
      ssd_ids.back().push_back(staged[s].e);
    }
    std::vector<ExpertWeights> ssd_w;
    if (!ssd_ids.empty()) ssd_->start(0, le, ssd_ids[0], ssd_w);
    for (uint32_t s0 = 0; s0 < arena_n; s0 += kStageBatch) {
      const uint32_t count = std::min(arena_n - s0, kStageBatch);
      std::vector<ExpertWeights> w;
      for (uint32_t s = s0; s < s0 + count; s++) w.push_back(arena_->weights(l, staged[s].e));
      stage_batch(s0, count, w, false);
    }
    for (size_t round = 0; round < ssd_ids.size(); round++) {
      const auto t_w = Clock::now();
      ssd_->wait();
      ctimes_.ssd_wait += ms(t_w, Clock::now());
      std::vector<ExpertWeights> w = std::move(ssd_w);
      ssd_w.clear();
      if (round + 1 < ssd_ids.size()) ssd_->start((round + 1) & 1, le, ssd_ids[round + 1], ssd_w);
      const uint32_t s0 = ssd_base + (uint32_t)round * kStageRound;
      stage_batch(s0, (uint32_t)ssd_ids[round].size(), w, true);
      ssd_->release(round & 1);
    }
    // A full fence sends the writes out of the CPU's write-combining buffers, and a read through the mapping returns only
    // after they have arrived.
    const auto t_cw = Clock::now();
    _mm_mfence();
    for (auto &cp : cards_) (void)*(volatile uint8_t *)(cp->ck->stage_cpu);
    ctimes_.copy_wait += ms(t_cw, Clock::now());
    ctimes_.stage += ms(t0, Clock::now());

    // The CPU answers the experts that stayed with it, rows in slices whose SSD reads fit the ring.
    t0 = Clock::now();
    {
      const D41MissJob *jobs = cjobs_;
      uint32_t r0 = 0;
      while (r0 < n) {
        uint32_t r1 = r0, ssd = 0;
        std::vector<uint32_t> seen;
        while (r1 < n && r1 - r0 < 64) {
          std::vector<uint32_t> add;
          for (uint32_t k = 0; k < jobs[r1].nmiss; k++) {
            const uint32_t id = (uint32_t)jobs[r1].ids[k];
            if (id < D41_EXPERTS && !arena_->weights(l, id).gate &&
                std::find(seen.begin(), seen.end(), id) == seen.end() && std::find(add.begin(), add.end(), id) == add.end())
              add.push_back(id);
          }
          if (r1 > r0 && ssd + add.size() > 40) break;
          ssd += (uint32_t)add.size();
          seen.insert(seen.end(), add.begin(), add.end());
          r1++;
        }
        MissAsk ask;
        ask.layer = l;
        ask.rows = r1 - r0;
        ask.real_rows = r1 - r0;
        ask.step = plan_->step;
        ask.jobs = jobs + r0;
        ask.answer = cpu_cy_ + (size_t)r0 * E;
        const MissCounts c = miss_->run_layer(ask);
        ctimes_.cpu_routings += c.routings;
        r0 = r1;
      }
    }
    ctimes_.cpu += ms(t0, Clock::now());

    // The second list's end.
    t0 = Clock::now();
    for (auto &cp : cards_) cp->ck->p3->execute();
    for (auto &cp : cards_) cp->ck->p2->synchronize();
    for (auto &cp : cards_) cp->ck->p3->synchronize();
    for (auto &cp : cards_) {
      trace_add(*cp, *cp->ck->p2, 1);
      trace_add(*cp, *cp->ck->p3, 2);
    }
    for (uint32_t c = 0; c < 2; c++)
      if (stuck_[16 * c]) throw std::runtime_error("card " + std::to_string(c) + " gave up a wait of the prompt chunk");
    ctimes_.tiles += (uint32_t)(run.tiles[0][0].size() + run.tiles[0][1].size() + run.stiles[0].size());
    ctimes_.experts += ms(t_p2, Clock::now());
  }

  std::vector<Candidate> out;
  if (logits) {
    for (uint32_t c = 0; c < 2; c++) {
      const size_t base = (size_t)c * STEP_MAX_ROWS * opt_.candidates;
      for (uint32_t i = 0; i < top_n_[c * STEP_MAX_ROWS]; i++) out.push_back({(int32_t)top_idx_[base + i], top_val_[base + i]});
    }
    std::sort(out.begin(), out.end(), [](const Candidate &a, const Candidate &b) {
      return a.logit != b.logit ? a.logit > b.logit : a.token < b.token;
    });
    if (out.size() > opt_.candidates) out.resize(opt_.candidates);
  }
  ctimes_.total = ms(t_start, Clock::now());
  return out;
}

}  // namespace engine::deepseek41
