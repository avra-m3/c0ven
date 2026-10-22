// Images in a slot's prompt (docs/spec/MODEL_VISION.md; ADR-0031): the positions they give every later cell (section
// 4.2), what the decode step reads of them, and the encoder from an image's pixels to its rows of the text model's
// input (sections 2 and 3). The encoder runs just before the prompt chunk that holds part of an image's block, its
// weights streamed a stage at a time into two slots, its activations in the prompt path's regions A and B. With two
// cards each takes half the merge windows and sends its rows of each layer's K and V to the other card.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "model_impl.h"

namespace engine::qwen4exp {

using namespace detail;

namespace {

constexpr uint32_t kVD = 1152, kVFF = 4304, kVFFP = 4320, kVLayers = 27, kVHeads = 16, kVHdp = 80, kVPatch = 768;
constexpr uint32_t kVQkvW = kVHeads * kVHdp;  // fp16 values per row of q, k and v (vision.cl's V_QKV_W)
constexpr uint32_t kMerge = 4 * kVD, kVOut = Q_EMBD;
constexpr uint32_t kStages = kVLayers + 2;  // the layers, then the merger's mm.0 and mm.2
constexpr uint32_t kTokensMax = 4096;       // section 4.3
constexpr uint32_t kFramesMax = 2;          // a pair of video frames (section 6)

// The four products of a layer in its stage: qkv, out, up, down (rows of 4,320, the last 16 zero).
struct LayerOffsets {
  uint64_t qkv, out, up, down, bytes;
};
LayerOffsets layer_offsets() {
  LayerOffsets o{};
  o.qkv = 0;
  o.out = align256(o.qkv + (uint64_t)3 * kVD * kVD * 2);
  o.up = align256(o.out + (uint64_t)kVD * kVD * 2);
  o.down = align256(o.up + (uint64_t)kVFF * kVD * 2);
  o.bytes = align256(o.down + (uint64_t)kVD * kVFFP * 2);
  return o;
}

// The event of kind k (0 staged, 1 free: per stage; 2 kv sent, 3 attention done: per layer) for card c.
uint32_t vev(uint32_t k, uint32_t i, uint32_t c, uint32_t C) {
  const uint32_t base[] = {0, kStages, 2 * kStages, 2 * kStages + kVLayers};
  return (base[k] + i) * C + c;
}

// The activations' sizes for an image of up to W windows on C cards: per card, rows R = 4 ceil(W / C), and the
// pixels of R patches of two frames.
struct Act {
  uint64_t x, y, a, q, o, h, kv, pix;
};
Act act_bytes(uint32_t W, uint32_t C) {
  const uint64_t R = 4ull * ((W + C - 1) / C), N = 4ull * W;
  return {R * kVD * 4, R * kVFF * 4, R * kVD * 2, R * kVQkvW * 2, R * kVD * 2, R * kVFFP * 2, N * kVQkvW * 2,
          R * kVPatch * kFramesMax};
}

// Places the buffers in the two regions, first fit, the largest first; false when they do not fit.
// at, when given, receives y, h, k, v, x, q, a, o and pix in that order.
bool carve(const Act &s, uint8_t *const reg[2], const uint64_t reg_bytes[2], std::array<void **, 9> *at) {
  struct Want {
    uint64_t bytes;
    void **at;
  };
  const uint64_t bytes[9] = {s.y, s.h, s.kv, s.kv, s.x, s.q, s.a, s.o, s.pix};
  std::vector<Want> w;
  for (uint32_t i = 0; i < 9; i++) w.push_back({bytes[i], at ? (*at)[i] : nullptr});
  std::stable_sort(w.begin(), w.end(), [](const Want &p, const Want &q) { return p.bytes > q.bytes; });
  uint64_t used[2] = {0, 0};
  for (const Want &e : w) {
    const uint64_t b = align256(e.bytes);
    int r = used[0] + b <= reg_bytes[0] ? 0 : used[1] + b <= reg_bytes[1] ? 1 : -1;
    if (r < 0) return false;
    if (e.at) *e.at = reg[r] + used[r];
    used[r] += b;
  }
  return true;
}

// ggml's GELU table on the CPU (section 3.4), as ggml-cpu.c fills it: by the bits of an fp16 x, the fp16 value of the
// tanh form computed in f32 with tanhf.
std::vector<uint16_t> gelu_table() {
  std::vector<uint16_t> t(1 << 16);
  for (uint32_t i = 0; i < t.size(); i++) {
    const float x = ggml_fp16_to_fp32((ggml_fp16_t)i);
    t[i] = ggml_fp32_to_fp16(0.5f * x * (1.0f + tanhf(0.79788456080286535587989211986876f * x * (1.0f + 0.044715f * x * x))));
  }
  return t;
}

}  // namespace

void Model::set_images(uint32_t slot, const Image *images, uint32_t n) {
  if (slot >= opt_.slots) throw std::runtime_error("slot out of range");
  std::vector<Image> ims;
  std::vector<ImageBlock> blocks;
  for (uint32_t i = 0; i < n; i++) {
    const Image &im = images[i];
    if (im.frames != 1 && im.frames != kFramesMax) throw std::runtime_error("an image has one frame or two");
    if (im.grid_h == 0 || im.grid_w == 0 || im.grid_h % 2 || im.grid_w % 2)
      throw std::runtime_error("an image's grid is even and not empty");
    ImageBlock b{im.first, im.grid_h, im.grid_w};
    if (i > 0 && im.first < blocks.back().first + blocks.back().tokens())
      throw std::runtime_error("images in order, their blocks apart");
    if ((uint64_t)im.first + b.tokens() > opt_.max_ctx) throw std::runtime_error("an image past the context");
    if (im.pixels && b.tokens() > vmax_tokens_) throw std::runtime_error("an image larger than image_tokens_max()");
    ims.push_back(im);
    blocks.push_back(b);
  }
  images_.at(slot) = std::move(ims);
  blocks_.at(slot) = std::move(blocks);
  slot_gen_.at(slot) = ++vgen_;
}

std::array<uint32_t, 3> Model::position(uint32_t slot, uint32_t cell) const {
  return mrope_position(blocks_.at(slot), cell);
}

void Model::set_rope(uint32_t slot, uint32_t cell, bool check) {
  const auto &blocks = blocks_[slot];
  if (check && mrope_image_at(blocks, cell)) throw std::runtime_error("an image's tokens go through the prompt path");
  const uint32_t blk = cell / Q_IDX_BLOCK * Q_IDX_BLOCK;
  const auto p = mrope_position(blocks, blk);
  rope_[slot] = {mrope_delta(blocks, cell), blk, p[0], p[1], p[2], {0, 0, 0}};
}

// first: the host side, once (the file, the weights' stages, the pixels' and the encoding's buffers, the largest
// image); always: each card's side, which a sleep frees (ADR-0032).
void Model::load_vision(bool first) {
  if (opt_.vision_path.empty()) return;
  if (!tmax_) throw std::runtime_error("images need the prompt path (prompt_chunk)");
  if (first) vgguf_ = std::make_unique<GgufModel>(opt_.vision_path);
  const GgufModel &g = *vgguf_;
  if (g.get_string("general.architecture") != "clip" || g.get_string("clip.projector_type") != "qwen3vl_merger")
    throw std::runtime_error("not a qwen3vl_merger encoder");
  if (g.get_int("clip.vision.block_count") != kVLayers || g.get_int("clip.vision.embedding_length") != kVD ||
      g.get_int("clip.vision.feed_forward_length") != kVFF || g.get_int("clip.vision.attention.head_count") != kVHeads ||
      g.get_int("clip.vision.patch_size") != 16 || g.get_int("clip.vision.projection_dim") != kVOut)
    throw std::runtime_error("an encoder of another shape");
  auto tensor = [&](const std::string &name, uint32_t type, uint64_t elements) -> const GgufTensor & {
    const GgufTensor &t = g.tensor(name);
    if (t.type != type || t.elements() != elements) throw std::runtime_error("unexpected tensor " + name);
    return t;
  };
  auto vblk = [](uint32_t l, const char *name) { return "v.blk." + std::to_string(l) + "." + name; };

  const uint32_t C = ncards_;
  if (first) {
    // The 16-bit weights, stage by stage, into pinned host memory.
    const LayerOffsets lo = layer_offsets();
    for (uint32_t s = 0; s < kVLayers; s++) vstage_len_.push_back(lo.bytes);
    vstage_len_.push_back(align256((uint64_t)kMerge * kMerge * 2));
    vstage_len_.push_back(align256((uint64_t)kVOut * kMerge * 2));
    uint64_t total = 0;
    for (uint64_t b : vstage_len_) {
      vstage_off_.push_back(total);
      total += b;
      vstage_bytes_ = std::max(vstage_bytes_, b);
    }
    vw_ = static_cast<uint8_t *>(rt_->alloc_host(total));
    std::memset(vw_, 0, total);
    for (uint32_t l = 0; l < kVLayers; l++) {
      uint8_t *st = vw_ + vstage_off_[l];
      std::memcpy(st + lo.qkv, tensor(vblk(l, "attn_qkv.weight"), QT_BF16, 3ull * kVD * kVD).data, 3ull * kVD * kVD * 2);
      std::memcpy(st + lo.out, tensor(vblk(l, "attn_out.weight"), QT_BF16, 1ull * kVD * kVD).data, 1ull * kVD * kVD * 2);
      std::memcpy(st + lo.up, tensor(vblk(l, "ffn_up.weight"), QT_BF16, 1ull * kVFF * kVD).data, 1ull * kVFF * kVD * 2);
      const uint8_t *down = tensor(vblk(l, "ffn_down.weight"), QT_BF16, 1ull * kVD * kVFF).data;
      for (uint32_t r = 0; r < kVD; r++)
        std::memcpy(st + lo.down + (uint64_t)r * kVFFP * 2, down + (uint64_t)r * kVFF * 2, kVFF * 2);
    }
    std::memcpy(vw_ + vstage_off_[kVLayers], tensor("mm.0.weight", QT_BF16, 1ull * kMerge * kMerge).data,
                1ull * kMerge * kMerge * 2);
    std::memcpy(vw_ + vstage_off_[kVLayers + 1], tensor("mm.2.weight", QT_BF16, 1ull * kMerge * kVOut).data,
                1ull * kMerge * kVOut * 2);

    // The largest image the regions hold, on every card.
    uint32_t W = std::min(kTokensMax, tmax_ - 2);
    for (;; W -= 4) {
      bool fits = true;
      for (auto &card : cards_) fits = fits && carve(act_bytes(W, C), card->pc->reg, card->pc->reg_bytes, nullptr);
      if (fits) break;
      if (W <= 64) throw std::runtime_error("the prompt regions hold no image");
    }
    vmax_tokens_ = W;
    vpix_ = static_cast<uint8_t *>(rt_->alloc_host((size_t)4 * W * kVPatch * kFramesMax));
    venc_ = static_cast<float *>(rt_->alloc_host((size_t)W * kVOut * 4));
    vcount_ = static_cast<uint32_t *>(rt_->alloc_host(64 * C));
    vevents_ = std::make_unique<ze::EventPool>(*rt_, (2 * kStages + 2 * kVLayers) * C);
  }

  // The patch embedding, transposed to [768][1,152]: for a still image the two frames' kernels summed (section 2.3),
  // for a pair of video frames each frame's own, frame 0's first (section 6).
  std::vector<float> wt((size_t)kVPatch * kVD), wt2((size_t)kFramesMax * kVPatch * kVD);
  {
    const float *w0 = reinterpret_cast<const float *>(tensor("v.patch_embd.weight", QT_F32, 1ull * kVPatch * kVD).data);
    const float *w1 = reinterpret_cast<const float *>(tensor("v.patch_embd.weight.1", QT_F32, 1ull * kVPatch * kVD).data);
    const size_t f1 = (size_t)kVPatch * kVD;
    for (uint32_t o = 0; o < kVD; o++)
      for (uint32_t k = 0; k < kVPatch; k++) {
        const float a = w0[(size_t)o * kVPatch + k], b = w1[(size_t)o * kVPatch + k];
        wt[(size_t)k * kVD + o] = a + b;
        wt2[(size_t)k * kVD + o] = a;
        wt2[f1 + (size_t)k * kVD + o] = b;
      }
  }

  const std::vector<uint16_t> gelu = gelu_table();

  for (auto &card : cards_) {
    card->vc = std::make_unique<VisionCard>();
    VisionCard &V = *card->vc;
    auto f32 = [&](const std::string &name, uint64_t n) {
      return upload<float>(*card, tensor(name, QT_F32, n).data, n * 4);
    };
    V.wt = upload<float>(*card, wt.data(), wt.size() * 4);
    V.wt2 = upload<float>(*card, wt2.data(), wt2.size() * 4);
    V.gelu = upload<uint16_t>(*card, gelu.data(), gelu.size() * 2);
    V.pbias = f32("v.patch_embd.bias", kVD);
    V.pos = f32("v.position_embd.weight", 48ull * 48 * kVD);
    for (uint32_t l = 0; l < kVLayers; l++)
      V.layers.push_back({f32(vblk(l, "ln1.weight"), kVD), f32(vblk(l, "ln1.bias"), kVD),
                          f32(vblk(l, "attn_qkv.bias"), 3 * kVD), f32(vblk(l, "attn_out.bias"), kVD),
                          f32(vblk(l, "ln2.weight"), kVD), f32(vblk(l, "ln2.bias"), kVD),
                          f32(vblk(l, "ffn_up.bias"), kVFF), f32(vblk(l, "ffn_down.bias"), kVD)});
    V.post_w = f32("v.post_ln.weight", kVD);
    V.post_b = f32("v.post_ln.bias", kVD);
    V.b0 = f32("mm.0.bias", kMerge);
    V.b2 = f32("mm.2.bias", kVOut);
    std::array<void **, 9> at = {(void **)&V.y, (void **)&V.h, (void **)&V.k, (void **)&V.v,  (void **)&V.x,
                                 (void **)&V.q, (void **)&V.a, (void **)&V.o, (void **)&V.pix};
    if (!carve(act_bytes(vmax_tokens_, C), card->pc->reg, card->pc->reg_bytes, &at))
      throw std::runtime_error("the prompt regions no longer hold the largest image");
    V.list = std::make_unique<ze::List>(*rt_, card->c, ze::List::immediate);
    V.copy = std::make_unique<ze::List>(*rt_, card->c, ze::List::immediate, true);
  }
  for (const auto &[name, t] : g.tensors()) g.release(t);
}

// The weights' two slots: the prompt path's staging slots when each holds the largest stage, else two of the
// encoder's own (ADR-0031, item 4). After record_prompt_copy, which sizes them.
void Model::vision_slots() {
  if (!vmax_tokens_) return;
  for (auto &card : cards_) {
    VisionCard &V = *card->vc;
    const PromptCard &P = *card->pc;
    if (P.stage && P.stage_slot >= vstage_bytes_) {
      V.slot[0] = P.stage;
      V.slot[1] = P.stage + P.stage_slot;
    } else {
      for (auto &s : V.slot) s = static_cast<uint8_t *>(rt_->alloc_device(card->c, vstage_bytes_));
    }
  }
}

void Model::encode_image(const Image &im, float *out) {
  if (!vmax_tokens_) throw std::runtime_error("no image encoder (Options::vision_path)");
  const ImageBlock b{im.first, im.grid_h, im.grid_w};
  const uint32_t W = b.tokens(), N = 4 * W;
  const uint32_t F = im.frames;
  if (!im.pixels || (F != 1 && F != kFramesMax) || !W || im.grid_h % 2 || im.grid_w % 2 || W > vmax_tokens_)
    throw std::runtime_error("an image the encoder cannot take");
  wait_commit();
  // Two cards when each has windows to take.
  const uint32_t C = W >= 2 ? ncards_ : 1, Call = ncards_;
  std::memcpy(vpix_, im.pixels, (size_t)N * kVPatch * F);
  for (uint32_t i = 0; i < vevents_->size(); i++) ZE_CHECK(zeEventHostReset((*vevents_)[i]));
  const LayerOffsets lo = layer_offsets();
  const float one = 1.0f, scale = 1.0f / std::sqrt(72.0f);
  const uint32_t zero = 0, gh = im.grid_h, gw = im.grid_w;
  // The cards' commands are appended layer by layer, the cards in turn, so that every event a list waits for is
  // signalled by a command appended before the wait: an immediate list's append can block until the card has run
  // what came before, which would never happen if the card waited for a command still to be appended.
  auto ev = [&](uint32_t k, uint32_t i, uint32_t cc) { return (*vevents_)[vev(k, i, cc, Call)]; };
  auto vk = [&](uint32_t c, const char *name) { return kernel(c, "qwen4exp_vision", name); };
  auto gemm = [&](uint32_t c, const uint8_t *w, uint32_t n, uint32_t k, uint32_t ldw, const uint16_t *x, uint32_t ldx,
                  const uint32_t *ntok, uint32_t rows, float *y) {
    const uint32_t band = (uint64_t)n * k * 2 > (18u << 20) ? 4 : 0;
    cards_[c]->vc->list->launch(kernel(c, "gemm_bf16", "gemm"),
                                {w, n, k, ldw, x, ldx, ntok, y, n, zero, zero, one, (uint16_t *)nullptr, band},
                                (n + 127) / 128, (rows + 63) / 64, 1, 64);
  };
  // Card c takes windows [w0[c], w0[c + 1]): R[c] rows from row r0[c].
  std::vector<uint32_t> w0(C + 1), R(C), r0(C);
  for (uint32_t c = 0; c <= C; c++) w0[c] = W * c / C;
  for (uint32_t c = 0; c < C; c++) {
    R[c] = 4 * (w0[c + 1] - w0[c]);
    r0[c] = 4 * w0[c];
    vcount_[16 * c] = R[c];
    vcount_[16 * c + 1] = R[c] / 4;
  }
  // A stage's weights into its slot, once the stage two before has finished with it.
  auto stage = [&](uint32_t c, uint32_t s) {
    VisionCard &V = *cards_[c]->vc;
    if (s >= 2) V.copy->wait({ev(1, s - 2, c)});
    V.copy->copy(V.slot[s % 2], vw_ + vstage_off_[s], vstage_len_[s], ev(0, s, c));
  };
  for (uint32_t c = 0; c < C; c++) {
    VisionCard &V = *cards_[c]->vc;
    stage(c, 0);
    stage(c, 1);
    V.list->copy(V.pix, vpix_ + (size_t)r0[c] * kVPatch * F, (size_t)R[c] * kVPatch * F);
    V.list->launch(vk(c, "v_embed"), {V.pix, R[c], r0[c], gh, gw, F, F == 1 ? V.wt : V.wt2, V.pbias, V.pos, V.x},
                   (R[c] + 7) / 8, 1, 1, 128);
  }
  for (uint32_t l = 0; l < kVLayers; l++) {
    // To each card's K and V for the layer, and on to the other card's.
    for (uint32_t c = 0; c < C; c++) {
      VisionCard &V = *cards_[c]->vc;
      ze::List &L = *V.list;
      const VisionCard::Layer &Ly = V.layers[l];
      const uint8_t *st = V.slot[l % 2];
      const size_t kv0 = (size_t)r0[c] * kVQkvW;
      L.wait({ev(0, l, c)});
      L.launch(vk(c, "v_norm"), {V.x, R[c], Ly.ln1w, Ly.ln1b, V.a}, R[c], 1, 1, 16);
      gemm(c, st + lo.qkv, 3 * kVD, kVD, kVD, V.a, kVD, &vcount_[16 * c], R[c], V.y);
      L.launch(vk(c, "v_qkv"), {V.y, Ly.bqkv, R[c], r0[c], gw, V.q, V.k + kv0, V.v + kv0}, kVHeads * 36 / 64, R[c], 1,
               64);
      if (C > 1) {
        // Into the other card's K and V once it has finished with the last layer's.
        VisionCard &peer = *cards_[1 - c]->vc;
        if (l > 0) L.wait({ev(3, l - 1, 1 - c)});
        L.copy(peer.k + kv0, V.k + kv0, (size_t)R[c] * kVQkvW * 2);
        L.copy(peer.v + kv0, V.v + kv0, (size_t)R[c] * kVQkvW * 2, ev(2, l, c));
      }
    }
    // The rest of the layer, and the stage two ahead into the slot it frees.
    for (uint32_t c = 0; c < C; c++) {
      VisionCard &V = *cards_[c]->vc;
      ze::List &L = *V.list;
      const VisionCard::Layer &Ly = V.layers[l];
      const uint8_t *st = V.slot[l % 2];
      const uint32_t *nrows = &vcount_[16 * c];
      if (C > 1) L.wait({ev(2, l, 1 - c)});
      L.launch(kernel(c, "qwen4exp_vision_attn", "v_attn"), {V.q, R[c], V.k, V.v, N, scale, V.o}, kVHeads,
               (R[c] + 15) / 16, 1, 16);
      if (C > 1) L.signal(ev(3, l, c));
      gemm(c, st + lo.out, kVD, kVD, kVD, V.o, kVD, nrows, R[c], V.y);
      L.launch(vk(c, "v_residual"), {V.x, V.y, Ly.bout, R[c]}, kVD / 64, R[c], 1, 64);
      L.launch(vk(c, "v_norm"), {V.x, R[c], Ly.ln2w, Ly.ln2b, V.a}, R[c], 1, 1, 16);
      gemm(c, st + lo.up, kVFF, kVD, kVD, V.a, kVD, nrows, R[c], V.y);
      L.launch(vk(c, "v_bias_gelu"), {V.y, kVFF, Ly.bup, kVFF, R[c], V.h, kVFFP, V.gelu}, (kVFFP + 63) / 64, R[c], 1,
               64);
      gemm(c, st + lo.down, kVD, kVFF, kVFFP, V.h, kVFFP, nrows, R[c], V.y);
      L.launch(vk(c, "v_residual"), {V.x, V.y, Ly.bdown, R[c]}, kVD / 64, R[c], 1, 64);
      L.signal(ev(1, l, c));
      stage(c, l + 2);
    }
  }
  // The merger: each window's four normed patches are one row of 4,608 (section 3.3).
  for (uint32_t c = 0; c < C; c++) {
    VisionCard &V = *cards_[c]->vc;
    ze::List &L = *V.list;
    const uint32_t T = R[c] / 4, *nwin = &vcount_[16 * c + 1];
    L.launch(vk(c, "v_norm"), {V.x, R[c], V.post_w, V.post_b, V.a}, R[c], 1, 1, 16);
    L.wait({ev(0, kVLayers, c)});
    gemm(c, V.slot[kVLayers % 2], kMerge, kMerge, kMerge, V.a, kMerge, nwin, T, V.y);
    L.signal(ev(1, kVLayers, c));
    L.launch(vk(c, "v_bias_gelu"), {V.y, kMerge, V.b0, kMerge, T, V.h, kMerge, V.gelu}, kMerge / 64, T, 1, 64);
    L.wait({ev(0, kVLayers + 1, c)});
    gemm(c, V.slot[(kVLayers + 1) % 2], kVOut, kMerge, kMerge, V.h, kMerge, nwin, T, V.y);
    L.signal(ev(1, kVLayers + 1, c));
    L.launch(vk(c, "v_bias"), {V.y, V.b2, kVOut, T, V.y}, kVOut / 64, T, 1, 64);
    L.copy(out + (size_t)w0[c] * kVOut, V.y, (size_t)T * kVOut * 4);
  }
  for (uint32_t c = 0; c < C; c++) {
    cards_[c]->vc->list->synchronize();
    cards_[c]->vc->copy->synchronize();
  }
}

void Model::place_images(uint32_t slot, uint32_t pos0, uint32_t n, uint32_t row0) {
  const auto &ims = images_.at(slot);
  for (size_t i = 0; i < ims.size(); i++) {
    const Image &im = ims[i];
    const ImageBlock &b = blocks_[slot][i];
    const uint32_t lo = std::max(pos0, b.first), hi = std::min(pos0 + n, b.first + b.tokens());
    if (lo >= hi) continue;
    if (!im.pixels) throw std::runtime_error("an image's block without its pixels");
    if (venc_slot_ != (int64_t)slot || venc_gen_ != slot_gen_[slot] || venc_index_ != i) {
      venc_slot_ = -1;
      encode_image(im, venc_);
      venc_slot_ = slot;
      venc_gen_ = slot_gen_[slot];
      venc_index_ = i;
    }
    std::memcpy(pembed_ + (size_t)(row0 + lo - pos0) * Q_EMBD, venc_ + (size_t)(lo - b.first) * kVOut,
                (size_t)(hi - lo) * kVOut * 4);
  }
}

}  // namespace engine::qwen4exp
