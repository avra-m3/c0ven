// The load check (docs/DESIGN.md, section 7): the decode kernels timed at each card's share of a layer that the
// cards allow, as the step-time model's terms need them. Card 0 runs moe_gate_up_g and moe_down_g on experts in the
// cache slot layout and matvec on attn_qkv's row slice, a peer write of a reduce goes from card 0 to card 1, and a
// row goes there and back for the hand-off. Each launch or copy is timed by its own timestamp event. The weights are
// the file's, read with O_DIRECT into device memory the check frees again.
#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>

#include "load/runs.h"
#include "model_impl.h"

namespace engine::qwen4exp {

using namespace detail;

namespace {

constexpr uint32_t kCheckLaunches = 20;   // timed a configuration, after as many untimed
constexpr uint32_t kCheckExperts = 64;    // in the check's cache: a launch's experts are not the last launch's
constexpr uint64_t kColdBytes = 96ull << 20;  // a dense slice's copies at least this much, so each launch reads cold

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

// Seconds of the command that signalled the event, by its timestamps.
double seconds(const ze::Card &card, ze_event_handle_t e) {
  ze_kernel_timestamp_result_t ts{};
  ZE_CHECK(zeEventQueryKernelTimestamp(e, &ts));
  return (double)((ts.global.kernelEnd - ts.global.kernelStart) & card.timestamp_mask) / (double)card.timer_hz;
}

uint64_t fnv(uint64_t h, const void *p, size_t n) {
  for (size_t i = 0; i < n; i++) h = (h ^ static_cast<const uint8_t *>(p)[i]) * 0x100000001b3ull;
  return h;
}

}  // namespace

std::string Model::check_key() const {
  std::string key;
  for (uint32_t c = 0; c < ncards_; c++) key += (c ? "+" : "") + rt_->cards()[c].uuid;
  char buf[64];
  std::snprintf(buf, sizeof buf, "/%x/", rt_->cards()[0].driver_version);
  key += buf;
  std::vector<std::filesystem::path> files;
  for (const auto &f : std::filesystem::directory_iterator(ENGINE_KERNEL_DIR))
    if (f.path().extension() == ".bin") files.push_back(f.path());
  std::sort(files.begin(), files.end());
  uint64_t h = 0xcbf29ce484222325ull;
  std::vector<char> bytes;
  for (const auto &f : files) {
    const std::string name = f.filename().string();
    h = fnv(h, name.data(), name.size());
    std::ifstream in(f, std::ios::binary);
    bytes.assign(std::istreambuf_iterator<char>(in), {});
    h = fnv(h, bytes.data(), bytes.size());
  }
  std::snprintf(buf, sizeof buf, "%016llx", (unsigned long long)h);
  return key + buf;
}

CheckResult Model::check() {
  const auto t0 = std::chrono::steady_clock::now();
  CheckResult r;
  Card &card = *cards_[0];
  const ze::Card &zc = rt_->cards()[0];
  ze::List L(*rt_, 0, ze::List::immediate);
  ze::EventPool pool(*rt_, kCheckLaunches, true);
  auto reset = [&] {
    for (uint32_t i = 0; i < pool.size(); i++) ZE_CHECK(zeEventHostReset(pool[i]));
  };
  // A launch untimed (i < kCheckLaunches) or timed by event i - kCheckLaunches; the median of the timed.
  auto timed = [&](const std::function<void(uint32_t, ze_event_handle_t)> &launch) {
    reset();
    for (uint32_t i = 0; i < 2 * kCheckLaunches; i++) launch(i, i < kCheckLaunches ? nullptr : pool[i - kCheckLaunches]);
    L.synchronize();
    std::vector<double> s;
    for (uint32_t i = 0; i < kCheckLaunches; i++) s.push_back(seconds(zc, pool[i]));
    return median(s);
  };

  // The file's weights: the first layer with IQ3_S gate and up (or layer 0), the first with IQ4_NL down (or layer
  // 0), each the first kCheckExperts experts; and blk.0.attn_qkv.
  uint32_t lg = 0, ld = 0;
  for (uint32_t l = Q_LAYERS; l-- > 0;) {
    if (gguf_.tensor(blk(l, "ffn_gate_exps.weight")).type == QT_IQ3_S) lg = l;
    if (gguf_.tensor(blk(l, "ffn_down_exps.weight")).type == QT_IQ4_NL) ld = l;
  }
  DirectReader reader(gguf_.files());
  struct Read {
    GgufTensor t;
    AlignedBuffer buf;
  };
  auto read = [&](const std::string &name, uint64_t bytes) {
    Read x{gguf_.tensor(name), aligned_buffer(DirectReader::span(gguf_.tensor(name).offset, bytes))};
    x.t.data = reader.read(x.t.shard, x.t.offset, bytes, x.buf.get());
    return x;
  };
  const uint64_t pe_g = gguf_.tensor(blk(lg, "ffn_gate_exps.weight")).bytes / Q_EXPERTS;
  const uint64_t pe_d = gguf_.tensor(blk(ld, "ffn_down_exps.weight")).bytes / Q_EXPERTS;
  Read g = read(blk(lg, "ffn_gate_exps.weight"), kCheckExperts * pe_g);
  Read u = read(blk(lg, "ffn_up_exps.weight"), kCheckExperts * pe_g);
  Read d = read(blk(ld, "ffn_down_exps.weight"), kCheckExperts * pe_d);
  Read q = read("blk.0.attn_qkv.weight", gguf_.tensor("blk.0.attn_qkv.weight").bytes);
  r.gate_up_type = g.t.type, r.down_type = d.t.type, r.dense_type = q.t.type;

  // The clock at its top before the first timing, and kept there between: dependent matvec launches on attn_qkv's
  // whole matrix.
  const uint32_t qbv = quant_block_values(q.t.type), qbpr = (uint32_t)(q.t.shape[0] / qbv);
  const uint32_t qrows = (uint32_t)q.t.shape[1];
  auto *act = static_cast<uint8_t *>(rt_->alloc_device(0, a8_bytes(Q_EMBD) * STEP_MAX_ROWS * Q_USED));
  auto *y = static_cast<float *>(rt_->alloc_device(0, (size_t)qrows * 4 * STEP_MAX_ROWS * Q_USED));
  const uint8_t zero = 0;
  L.fill(act, &zero, 1, a8_bytes(Q_EMBD) * STEP_MAX_ROWS * Q_USED);
  std::vector<uint8_t *> copies;
  Mat qm{q.t.type, qrows, (uint32_t)Q_EMBD, qbpr, nullptr};
  auto load_dense = [&](uint32_t rows) {
    const uint64_t bytes = runs_bytes(q.t.type, (uint64_t)rows * qbpr);
    std::vector<uint8_t> slice(bytes);
    to_runs(q.t.type, {{q.t.data, q.t.row_bytes(), 0, rows, 0, qbpr}}, slice.data());
    for (auto *p : copies) rt_->free(p);
    copies.assign(std::max<uint64_t>(4, (kColdBytes + bytes - 1) / bytes), nullptr);
    for (auto &p : copies) {
      p = static_cast<uint8_t *>(rt_->alloc_device(0, bytes));
      L.copy(p, slice.data(), bytes);
    }
    L.synchronize();
    qm.rows = rows;
    return bytes;
  };
  auto warm = [&](uint32_t ms) {
    const auto w0 = std::chrono::steady_clock::now();
    Mat m = qm;
    m.p = copies[0];
    while (std::chrono::steady_clock::now() - w0 < std::chrono::milliseconds(ms)) {
      for (int i = 0; i < 50; i++) matvec(L, card, m, act, y, 1);
      L.synchronize();
    }
  };
  load_dense(qrows);
  warm(200);

  for (uint32_t wi = 0; wi < 4; wi++) {
    const uint32_t w = 1u << wi;
    if (w > ncards_) break;
    // Dense: attn_qkv's row slice at the width, one row, cycling through the copies.
    const uint64_t qbytes = load_dense(qrows / w);
    warm(30);
    const double qs = timed([&](uint32_t i, ze_event_handle_t e) {
      Mat m = qm;
      m.p = copies[i % copies.size()];
      matvec(L, card, m, act, y, 1, e);
    });
    r.dense_gbs[wi] = qbytes / qs / 1e9;

    // Experts: the slot layout of Model::load_experts at the width, card 0's rows of gate and up and columns of down.
    const uint32_t ff = Q_FF / w, gbpr = Q_EMBD / quant_block_values(g.t.type);
    const uint32_t dbv = quant_block_values(d.t.type), dbpr = ff % dbv ? 0 : ff / dbv;
    const uint64_t gb = runs_bytes(g.t.type, (uint64_t)ff * gbpr), db = runs_bytes(d.t.type, (uint64_t)Q_EMBD * dbpr);
    const uint64_t up_off = align256(gb), down_off = up_off + align256(gb), slot_bytes = down_off + align256(db);
    std::vector<uint8_t> host(kCheckExperts * slot_bytes);
    reader.pool().run(kCheckExperts, [&](uint64_t e) {
      uint8_t *s = host.data() + e * slot_bytes;
      to_runs(g.t.type, {{g.t.data + e * pe_g, g.t.row_bytes(), 0, ff, 0, gbpr}}, s);
      to_runs(u.t.type, {{u.t.data + e * pe_g, u.t.row_bytes(), 0, ff, 0, gbpr}}, s + up_off);
      if (dbpr) to_runs(d.t.type, {{d.t.data + e * pe_d, d.t.row_bytes(), 0, Q_EMBD, 0, dbpr}}, s + down_off);
    });
    auto *cache = static_cast<uint8_t *>(rt_->alloc_device(0, host.size()));
    L.copy(cache, host.data(), host.size());
    L.synchronize();
    warm(80);  // the card idled through the copy; its clock is back at the top after this, as before the first timing
    const uint32_t max_asg = 4 * Q_USED;
    auto *groups = static_cast<MoeGroup *>(rt_->alloc_device(0, (size_t)2 * kCheckLaunches * max_asg * sizeof(MoeGroup)));
    auto *ngroups = static_cast<uint32_t *>(rt_->alloc_device(0, (size_t)2 * kCheckLaunches * 4));
    auto *ha = static_cast<uint8_t *>(rt_->alloc_device(0, (size_t)max_asg * (align256(ff) + align256(ff) / 8)));
    for (uint32_t ai = 0; ai < 3; ai++) {
      const uint32_t A = 1u << ai, nasg = A * Q_USED;
      // Launch i's groups: nasg distinct experts, one assignment each, taken in turn from the cache's.
      std::vector<MoeGroup> gs((size_t)2 * kCheckLaunches * max_asg);
      std::vector<uint32_t> ns(2 * kCheckLaunches, nasg);
      for (uint32_t i = 0, next = 0; i < 2 * kCheckLaunches; i++)
        for (uint32_t k = 0; k < nasg; k++) {
          MoeGroup &G = gs[(size_t)i * max_asg + k];
          std::memset(&G, 0, sizeof G);
          G.slot = (int32_t)(next++ % kCheckExperts);
          G.n = 1;
          G.asg[0] = k;
        }
      L.copy(groups, gs.data(), gs.size() * sizeof(MoeGroup));
      L.copy(ngroups, ns.data(), ns.size() * 4);
      L.synchronize();
      const std::string gm = moe_module(g.t.type, A, false), dm = dbpr ? moe_module(d.t.type, A, true) : "";
      if (!gm.empty()) {
        warm(30);
        const uint32_t lx = kSgSize * MOE_WG_ROWS / MOE_GU_ROWS(A), K = Q_EMBD, xs = (uint32_t)a8_bytes(Q_EMBD);
        const uint32_t per = Q_USED;
        const double s = timed([&](uint32_t i, ze_event_handle_t e) {
          L.launch(kernel(0, gm, "moe_gate_up_g"),
                   {cache, slot_bytes, groups + (size_t)i * max_asg, ngroups + i, ff, gbpr, up_off, act, K, xs, per, y,
                    card.grids, ha},
                   (ff + MOE_WG_ROWS - 1) / MOE_WG_ROWS, nasg, 1, lx, 1, 1, e);
        });
        r.gate_up_gbs[wi][ai] = 2.0 * gb * nasg / s / 1e9;
      }
      if (!dm.empty()) {
        warm(30);
        const uint32_t lx = kSgSize * MOE_WG_ROWS / MOE_DOWN_ROWS(d.t.type, A), K = ff, xs = (uint32_t)a8_bytes(ff);
        const uint32_t rows = Q_EMBD;
        const double s = timed([&](uint32_t i, ze_event_handle_t e) {
          L.launch(kernel(0, dm, "moe_down_g"),
                   {cache, slot_bytes, groups + (size_t)i * max_asg, ngroups + i, rows, dbpr, down_off, act, K, xs, y,
                    card.grids},
                   (rows + MOE_WG_ROWS - 1) / MOE_WG_ROWS, nasg, 1, lx, 1, 1, e);
        });
        r.down_gbs[wi][ai] = (double)db * nasg / s / 1e9;
      }
    }
    rt_->free(cache), rt_->free(groups), rt_->free(ngroups), rt_->free(ha);
  }
  for (auto *p : copies) rt_->free(p);

  if (ncards_ > 1) {
    // A reduce's peer write: rows of partial sums from card 0's memory to card 1's, as send_peer copies them.
    const size_t most = (size_t)4 * Q_EMBD * 4;
    auto *to = static_cast<uint8_t *>(rt_->alloc_device(1, most));
    for (uint32_t ai = 0; ai < 3; ai++) {
      const size_t bytes = (size_t)(1u << ai) * Q_EMBD * 4;
      r.exchange_us[ai] = 1e6 * timed([&](uint32_t, ze_event_handle_t e) { L.copy(to, y, bytes, e); });
    }
    // The hand-off: a row to card 1 and back, each side waiting for the other's copy. Card 0's clock times the trip
    // from its copy's end to the row's return, card 1's its own copy; less that copy, the half is one. The waits take
    // events without timestamps, as the engine's own between the cards do: a timestamp event waited on by the other
    // card loses the device.
    ze::List L1(*rt_, 1, ze::List::immediate);
    ze::EventPool flags(*rt_, 2);
    auto *ts = static_cast<uint64_t *>(rt_->alloc_host(4 * 8));
    const ze::Card &zc1 = rt_->cards()[1];
    std::vector<double> half;
    for (uint32_t i = 0; i < 2 * kCheckLaunches; i++) {
      ze_event_handle_t there = flags[0], back = flags[1];
      for (auto e : {there, back}) ZE_CHECK(zeEventHostReset(e));
      L.copy(to, y, Q_EMBD * 4, there);
      L.timestamp(ts);
      L1.wait({there});
      L1.timestamp(ts + 2);
      L1.copy(y, to, Q_EMBD * 4, back);
      L1.timestamp(ts + 3);
      L.wait({back});
      L.timestamp(ts + 1);
      L.synchronize();
      L1.synchronize();
      const double trip = (double)((ts[1] - ts[0]) & zc.timestamp_mask) / (double)zc.timer_hz;
      const double copy1 = (double)((ts[3] - ts[2]) & zc1.timestamp_mask) / (double)zc1.timer_hz;
      if (i >= kCheckLaunches) half.push_back((trip - copy1) / 2);
    }
    r.handoff_us = 1e6 * median(half);
    rt_->free(ts), rt_->free(to);
  }
  rt_->free(act), rt_->free(y);
  r.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return r;
}

}  // namespace engine::qwen4exp
