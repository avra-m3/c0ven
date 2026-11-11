// What deepseek41's prompt chunk path keeps on a card: buffers for up to Options::prompt_chunk rows, and the lists of a
// layer's two phases.
#pragma once

#include "models/deepseek41/model.h"
#include "moe_tile.h"

namespace engine::deepseek41 {

constexpr uint32_t kStageSlots = 256;  // experts a layer's copies hold on a card
constexpr uint32_t kStageBatch = 16;   // experts one pool run writes to the cards
constexpr uint32_t kStageRound = 32;   // experts one round of SSD reads brings, a multiple of kStageBatch
constexpr uint32_t kStageBuffers = kStageSlots + kStageBatch;  // slots a card holds: the file's start at a batch
constexpr uint32_t kChunkRows = 2048;  // the most rows a chunk takes, whatever Options::prompt_chunk asks
constexpr uint32_t kChunkGroup = 256;  // rows the attention takes at a time: its scores and partial sums are per row

struct Model::ChunkCard {
  uint32_t cap = 0;
  std::unique_ptr<ze::List> p1, p2;
  uint32_t *words = nullptr;  // cwords_ on the card
  uint32_t *real = nullptr;   // creal_ on the card
  float *e = nullptr;
  float *X = nullptr, *X1 = nullptr, *ss = nullptr, *m = nullptr;
  D41HcMix *amix = nullptr, *fmix = nullptr;
  float *xf = nullptr;
  uint16_t *xh = nullptr, *xb = nullptr;  // the normed input as f16 and as bf16
  uint8_t *xa = nullptr;                  // the last row's, as a8, for the output head
  float *qr = nullptr, *kv = nullptr, *qf = nullptr, *ckv = nullptr, *csc = nullptr;
  uint16_t *qh = nullptr, *qb = nullptr;
  uint8_t *qa = nullptr;
  uint32_t *idx = nullptr;
  D41Row *rows = nullptr, *rows_c = nullptr;
  // Attention, kChunkGroup rows at a time.
  float *q = nullptr, *iq = nullptr, *iw = nullptr, *score = nullptr, *part = nullptr, *of = nullptr;
  uint8_t *oa = nullptr;
  uint16_t *oh = nullptr;
  uint16_t *aa = nullptr;  // out_a's result as f16, [row][the card's groups' D41_OLR]
  float *y = nullptr, *in[2] = {nullptr, nullptr};
  uint16_t *w16a = nullptr, *w16b = nullptr;  // dequantised weights
  uint16_t *ring = nullptr;                   // the window cells of the chunk: [slot][D41_RING_WIDE][D41_HD]
  uint8_t *eng_a8 = nullptr;
  float *ekv = nullptr;
  D41EngramSend *eown = nullptr, *ein = nullptr;
  // The router and the experts.
  float *rlogits = nullptr, *w = nullptr, *sgu = nullptr, *shared = nullptr, *out = nullptr;
  int32_t *ids = nullptr, *slots = nullptr;
  uint32_t *nmiss = nullptr, *ngroups = nullptr;
  D41Group *groups = nullptr;
  uint16_t *xg = nullptr, *hbuf = nullptr, *ybuf = nullptr, *sh = nullptr;
  int32_t *yidx = nullptr;
  uint32_t *rowidx = nullptr;
  MoeTile *tiles[2] = {nullptr, nullptr}, *tiles_s = nullptr;
  uint8_t *stage = nullptr;      // staging slots of stage_slot bytes: a card's halves of experts in the file's layout
  uint8_t *stage_cpu = nullptr;  // the same bytes, where the CPU writes them
  uint64_t stage_slot = 0;
  std::unique_ptr<ze::List> p3;              // the layer's closing list
  uint64_t *ts[3] = {nullptr, nullptr, nullptr};  // the lists' launch timestamps, when tracing
};

}  // namespace engine::deepseek41
