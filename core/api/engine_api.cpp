// The C ABI over the core (include/engine_api.h). One engine thread runs the queued step plans, in order: a plan's
// page changes, its prompt pieces through the prompt path, then one decode step for its decode slots, batched. The
// caller's thread only copies requests in, queues plans, takes events out, asks for checkpoints and puts the engine to
// sleep and wakes it.
#include "engine_api.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstring>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "api/lookup.h"
#include "api/sampling.h"
#include "api/swap_cadence.h"
#include "models/qwen4exp/model.h"

using engine::penalised;
using engine::sample_p;
using engine::qwen4exp::Candidate;
using engine::qwen4exp::kNoToken;
using engine::qwen4exp::StepIn;

namespace {

constexpr int32_t kImageToken = 248056;  // <|image_pad|> (docs/spec/MODEL_VISION.md, section 4.1)
constexpr int32_t kVideoToken = 248057;  // <|video_pad|>: a pair of video frames (section 6)

// An image or frame pair of a slot's prompt; its pixels are shared with the checkpoints taken while it was in the slot.
struct SlotImage {
  uint32_t first = 0, grid_h = 0, grid_w = 0, frames = 1;
  std::shared_ptr<const std::vector<uint8_t>> pixels;
  bool operator==(const SlotImage &o) const {
    return first == o.first && grid_h == o.grid_h && grid_w == o.grid_w && frames == o.frames &&
           (pixels == o.pixels || *pixels == *o.pixels);
  }
};

// The images whose blocks start before position end.
std::vector<SlotImage> images_before(const std::vector<SlotImage> &ims, uint32_t end) {
  std::vector<SlotImage> v;
  for (const SlotImage &im : ims)
    if (im.first < end) v.push_back(im);
  return v;
}

struct Slot {
  bool used = false;
  engine_settings settings{};
  std::vector<int32_t> prompt;
  std::vector<int32_t> tokens;  // everything fed so far, prompt first
  uint32_t fed = 0;             // positions computed
  int32_t next = kNoToken;      // the token sampled last, fed at the next step
  uint32_t generated = 0;
  bool done = false;
  std::vector<int32_t> drafts;  // the draft layer's proposals for the positions after next
  std::vector<SlotImage> images;
  bool stalled = false;         // STALLED was sent, and no page has been mapped since
  uint32_t windows_end = UINT32_MAX;  // where the last prompt piece read as decode windows ended
  engine::Lookup lookup;        // the slot's own tokens, for drafts from earlier runs of them
  engine::DraftOdds odds;
  bool lookup_round = false;    // drafts holds the lookup's drafts
  engine_slot_stats stats{};
  uint64_t last_ns = 0;         // when a plan last computed it
  bool resident = false;        // its pages count for the streamed pool's ring (kv_resident)
};

// A used slot no plan has computed for this long leaves the streamed pool's resident ring (kv_resident): the cards'
// memory goes to the experts the requests in progress need.
constexpr uint64_t kIdleResident = 300ull * 1000000000ull;

// A prompt piece of at most this many tokens is read as decode windows of forced rows: a prompt chunk stages every
// uncached expert whatever its size, so a short piece is faster as windows
// (docs/measurements/MEASUREMENTS.md#2026-10-17-prompt-windows).
constexpr uint32_t kWindowPrompt = 64;

// The most drafts each of several writing slots makes (docs/measurements/MEASUREMENTS.md#2026-10-19-draft-rounds-several-writers).
constexpr uint32_t kSharedDrafts = 2;

struct Checkpoint {
  uint8_t *data = nullptr;      // pinned host memory: a slot's fixed state
  uint64_t bytes = 0;
  std::vector<int32_t> tokens;  // the positions it holds
  std::vector<SlotImage> images;  // the images whose blocks start among them
  bool full = false;
};

// A queued plan, with its page changes.
struct Queued {
  engine_step_plan plan;
  std::vector<engine_page_change> pages;
};

thread_local std::string open_error;

uint64_t now_ns() {
  return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
}

bool media(int32_t tok) { return tok == kImageToken || tok == kVideoToken; }

// The last tokens of hist the penalties count, oldest first, without image and video tokens, which llama.cpp's server
// leaves out too (docs/spec/MODEL_IO.md, section 2.4); empty without penalties.
std::vector<int32_t> penalty_window(const std::vector<int32_t> &hist, const engine_settings &s) {
  std::vector<int32_t> w;
  if (!penalised(s)) return w;
  for (size_t i = hist.size(); i > 0 && w.size() < s.penalty_last_n; i--)
    if (!media(hist[i - 1])) w.push_back(hist[i - 1]);
  std::reverse(w.begin(), w.end());
  return w;
}

constexpr size_t kTimingRing = 4096;

}  // namespace

struct engine_ctx {
  std::unique_ptr<engine::qwen4exp::Model> model;
  engine_info info{};
  std::vector<Slot> slots;
  std::vector<Checkpoint> checkpoints;
  uint32_t draft_max = 3;
  double draft_min_p = 0.5;
  bool swaps = false;  // ENGINE_CONFIG_SWAPS
  std::mutex mu;
  std::condition_variable cv, events_cv, idle_cv;
  std::deque<Queued> plans;
  // The page tables (include/engine_api.h): each slot's row as the engine thread has applied it, and how many of its
  // pages are mapped then (mapped) and once every queued plan has run (queued_mapped). Under mu, but table only on
  // the engine thread.
  std::vector<std::vector<uint32_t>> table;
  std::vector<uint32_t> mapped, queued_mapped;
  std::vector<uint32_t> top;  // per slot: one past its highest mapped page, as applied; engine thread
  std::deque<engine_event> events;
  std::deque<engine_timing> timing;
  uint32_t plans_done = 0, plans_taken = 0;
  bool running = false;  // a plan is being run
  bool prompt_ended = false;  // the plan being run read some slot's last prompt token
  uint32_t writers = 0;  // the slots the plan being run decoded
  engine::WindowTimes window_times;  // one slot's decode steps and rounds, by their rows
  engine::SwapCadence swap_cadence;  // when the next swap round is due; under mu
  engine_stats stats{};  // the counters (include/engine_api.h); the model's are copied in at each plan's end
  uint64_t t_open = 0;
  bool stop = false;
  bool asleep = false;
  uint64_t checkpoint_bytes = 0;
  std::string error;
  std::thread thread;

  // Under mu.
  void emit(engine_event ev) {
    events.push_back(ev);
    events_cv.notify_all();
  }
  void record(engine_timing t) {
    if (timing.size() == kTimingRing) timing.pop_front();
    timing.push_back(t);
  }

  // The positions slot s has pages for. Under mu.
  uint32_t mapped_cells(uint32_t s) const { return mapped[s] * KV_PAGE_CELLS; }

  // Slot s needs a page for position pos: it waits, and says so once. Under mu.
  void stall(uint32_t s, uint32_t pos) {
    Slot &sl = slots[s];
    if (sl.stalled) return;
    sl.stalled = true;
    stats.stalls++;
    emit({ENGINE_EVENT_STALLED, s, kNoToken, pos});
  }

  // A plan's page changes, in order: copies between pages, then the slots' rows written to the cards. Runs on the
  // engine thread; submit checked them.
  void apply_pages(const std::vector<engine_page_change> &changes) {
    std::vector<std::pair<uint32_t, uint32_t>> dirty(slots.size(), {UINT32_MAX, 0});  // logical pages [lo, hi)
    for (const engine_page_change &c : changes) {
      if (c.page != ENGINE_NO_PAGE) {
        if (c.cells) model->copy_cells(c.source, c.page, c.cells);
        table[c.slot][c.logical] = c.page;
        dirty[c.slot].first = std::min(dirty[c.slot].first, c.logical);
        dirty[c.slot].second = std::max(dirty[c.slot].second, c.logical + 1);
      }
      std::lock_guard<std::mutex> lk(mu);
      if (c.page == ENGINE_NO_PAGE) {
        mapped[c.slot] = std::min(mapped[c.slot], c.logical);
      } else {
        if (c.logical == mapped[c.slot]) mapped[c.slot]++;
        slots[c.slot].stalled = false;
      }
    }
    for (uint32_t s = 0; s < dirty.size(); s++)
      if (dirty[s].first < dirty[s].second)
        model->set_pages(s, dirty[s].first, table[s].data() + dirty[s].first, dirty[s].second - dirty[s].first);
    std::vector<bool> changed(slots.size(), false);
    for (const engine_page_change &c : changes) changed[c.slot] = true;
    std::lock_guard<std::mutex> lk(mu);
    for (uint32_t s = 0; s < slots.size(); s++) {
      if (!changed[s]) continue;
      top[s] = 0;
      for (uint32_t i = 0; i < mapped[s]; i++) top[s] = std::max(top[s], table[s][i] + 1);
    }
  }

  // The pages a plan may touch (Model::kv_need): the caller's in use, the pages its changes name, and every page
  // its slots map.
  uint32_t kv_in_use(const engine_step_plan &plan, const std::vector<engine_page_change> &changes) const {
    const uint32_t P = model->kv_pages();
    uint32_t n = P - std::min(plan.kv_free, P);
    for (const engine_page_change &c : changes) {
      if (c.page != ENGINE_NO_PAGE) n = std::max(n, c.page + 1);
      if (c.cells) n = std::max(n, c.source + 1);
    }
    for (uint32_t i = 0; i < plan.nslots && i < ENGINE_MAX_STEP_SLOTS; i++) n = std::max(n, top[plan.slots[i]]);
    for (uint32_t i = 0; i < plan.npieces && i < ENGINE_MAX_PROMPT_PIECES; i++) n = std::max(n, top[plan.pieces[i].slot]);
    return std::min(n, P);
  }

  // The pages the streamed pool's ring keeps resident (Model::kv_need): what the plan's slots and the slots a plan
  // computed within kIdleResident have mapped, with this plan's changes. A used slot idle longer leaves the ring
  // (drops: the slot and its mapped pages, for Model::kvs_drop) and no longer counts; a released one only stops
  // counting, as its pages may be another slot's by now.
  uint32_t kv_resident(const engine_step_plan &plan, const std::vector<engine_page_change> &changes, uint64_t now,
                       std::vector<std::pair<uint32_t, uint32_t>> &drops) {
    std::lock_guard<std::mutex> lk(mu);
    std::vector<bool> in(slots.size(), false);
    for (uint32_t i = 0; i < plan.nslots && i < ENGINE_MAX_STEP_SLOTS; i++)
      if (plan.slots[i] < slots.size()) in[plan.slots[i]] = true;
    for (uint32_t i = 0; i < plan.npieces && i < ENGINE_MAX_PROMPT_PIECES; i++)
      if (plan.pieces[i].slot < slots.size()) in[plan.pieces[i].slot] = true;
    for (const engine_page_change &c : changes)
      if (c.slot < slots.size() && c.page != ENGINE_NO_PAGE) in[c.slot] = true;
    uint32_t n = 0;
    for (uint32_t s = 0; s < slots.size(); s++) {
      Slot &sl = slots[s];
      if (in[s]) {
        sl.last_ns = now;
        sl.resident = true;
      } else if (!sl.used) {
        sl.resident = false;
      } else if (sl.resident && now - sl.last_ns > kIdleResident) {
        drops.push_back({s, mapped[s]});
        sl.resident = false;
      }
      if (!sl.resident) continue;
      uint32_t m = mapped[s];
      for (const engine_page_change &c : changes)
        if (c.slot == s && c.page != ENGINE_NO_PAGE) m = std::max(m, c.logical + 1);
      n += m;
    }
    return n;
  }

  // Emits tok, the slot's token for position pos, and ends the request if it must. Returns whether it ended. Under
  // mu.
  bool emit_token(uint32_t s, int32_t tok, uint32_t pos) {
    Slot &sl = slots[s];
    sl.next = tok;
    sl.generated++;
    emit({ENGINE_EVENT_TOKEN, s, tok, pos});
    bool stop_tok = false;
    for (uint32_t i = 0; i < sl.settings.n_stop && i < ENGINE_MAX_STOP; i++) stop_tok |= sl.settings.stop[i] == tok;
    if (stop_tok || sl.generated >= sl.settings.max_tokens) {
      sl.done = true;
      emit({ENGINE_EVENT_DONE, s, tok, stop_tok ? (uint32_t)ENGINE_DONE_STOP : (uint32_t)ENGINE_DONE_LENGTH});
    } else if (pos >= info.max_ctx) {
      sl.done = true;
      emit({ENGINE_EVENT_DONE, s, tok, ENGINE_DONE_CONTEXT});
    }
    return sl.done;
  }

  // After a slot computed position fed - 1 and its candidates are known (the last prompt position or a decode row):
  // samples, emits and ends the request if it must. Under mu.
  void produce(uint32_t s, const std::vector<Candidate> &cand) {
    Slot &sl = slots[s];
    emit_token(s, sample_p(cand, sl.settings, sl.fed, penalty_window(sl.tokens, sl.settings)).first, sl.fed);
  }

  // The tokens before a slot's first draft, for its penalties: what it has fed, then the token sampled last. Under mu.
  std::vector<int32_t> draft_history(const Slot &sl) const {
    std::vector<int32_t> h = penalty_window(sl.tokens, sl.settings);
    if (penalised(sl.settings)) h.push_back(sl.next);
    return h;
  }

  // The slot's drafts, up to draft_max, from a catch-up's last row's candidates (row src of its pass); pos is the
  // first draft's position and hist the tokens before it (draft_history). A draft is the request's sampler's pick with
  // its position's uniform, so the main model accepts it when it samples the same; drafting stops below draft_min_p.
  // The draft after position q reads the draft layer's cell q - 1 (mtp.cpp). Runs on the engine thread.
  std::vector<int32_t> make_drafts(uint32_t s, const engine_settings &st, uint32_t pos, const std::vector<Candidate> &first,
                                   uint32_t src, std::vector<int32_t> hist) {
    std::vector<int32_t> d;
    std::pair<int32_t, double> pick = sample_p(first, st, pos, hist);
    if (first.empty() || pick.second < draft_min_p) return d;
    d.push_back(pick.first);
    while (d.size() < draft_max && pos + d.size() + 1 < info.max_ctx) {
      auto c = model->mtp_draft(s, pos + (uint32_t)d.size() - 2, d.back(), src);
      src = 0;
      if (penalised(st)) hist.push_back(d.back());
      pick = sample_p(c, st, pos + (uint32_t)d.size(), hist);
      if (pick.second < draft_min_p) break;
      d.push_back(pick.first);
    }
    return d;
  }

  // One slot's drafting from a pass's row: its sampler, the first draft's position, the row's candidates, the row,
  // the most drafts it may make, and the tokens before the first draft (draft_history).
  struct DraftSeed {
    uint32_t slot;
    engine_settings settings;
    uint32_t pos;
    std::vector<Candidate> first;
    uint32_t src;
    uint32_t most;
    std::vector<int32_t> hist;
  };

  // make_drafts for several slots at once, each further draft of every slot still drafting in one pass of the draft
  // layer. Runs on the engine thread.
  std::vector<std::vector<int32_t>> make_drafts_many(std::vector<DraftSeed> &seeds) {
    const size_t n = seeds.size();
    std::vector<std::vector<int32_t>> d(n);
    std::vector<uint32_t> src(n), live;
    for (size_t i = 0; i < n; i++) {
      DraftSeed &sd = seeds[i];
      if (sd.most == 0 || sd.first.empty()) continue;
      std::pair<int32_t, double> pick = sample_p(sd.first, sd.settings, sd.pos, sd.hist);
      if (pick.second < draft_min_p) continue;
      d[i].push_back(pick.first);
      src[i] = sd.src;
      if (d[i].size() < sd.most && sd.pos + 2 < info.max_ctx) live.push_back((uint32_t)i);
    }
    while (!live.empty()) {
      std::vector<StepIn> rows;
      std::vector<uint32_t> from;
      for (uint32_t i : live) {
        rows.push_back({seeds[i].slot, seeds[i].pos + (uint32_t)d[i].size() - 2, d[i].back(), kNoToken, kNoToken});
        from.push_back(src[i]);
      }
      auto c = model->mtp_drafts(rows.data(), (uint32_t)rows.size(), from.data());
      std::vector<uint32_t> still;
      for (size_t k = 0; k < live.size(); k++) {
        const uint32_t i = live[k];
        DraftSeed &sd = seeds[i];
        src[i] = (uint32_t)k;
        if (penalised(sd.settings)) sd.hist.push_back(d[i].back());
        std::pair<int32_t, double> pick = sample_p(c[k], sd.settings, sd.pos + (uint32_t)d[i].size(), sd.hist);
        if (pick.second < draft_min_p) continue;
        d[i].push_back(pick.first);
        if (d[i].size() < sd.most && sd.pos + d[i].size() + 1 < info.max_ctx) still.push_back(i);
      }
      live = std::move(still);
    }
    return d;
  }

  // One slot's DFlash 2 block: its sampler, the first draft's position, the token before it (the anchor, the
  // slot's token sampled last), the most drafts it may make, and the tokens before the first draft (draft_history).
  struct DfSeed {
    uint32_t slot;
    engine_settings settings;
    uint32_t pos;
    int32_t anchor;
    uint32_t most;
    std::vector<int32_t> hist;
  };

  // DFlash 2 (dflash.cpp): the context pass over ctx, rows of the last step whose features the cards hold, then one
  // block pass for every seed. Each seed's drafts are picked as make_drafts picks them: the request's sampler with
  // each position's uniform, stopping below draft_min_p. Runs on the engine thread.
  std::vector<std::vector<int32_t>> df_drafts(const StepIn *ctx, uint32_t nctx, std::vector<DfSeed> &seeds) {
    std::vector<std::vector<int32_t>> d(seeds.size());
    std::vector<engine::qwen4exp::Model::DraftBlock> blocks;
    std::vector<size_t> of;
    for (size_t i = 0; i < seeds.size(); i++) {
      const DfSeed &sd = seeds[i];
      const uint32_t most = std::min<uint32_t>(sd.most, STEP_MAX_ROWS - 1);
      if (most == 0 || sd.pos == 0 || sd.pos >= info.max_ctx) continue;
      blocks.push_back({sd.slot, sd.pos - 1, sd.anchor, 1 + most});
      of.push_back(i);
    }
    if (nctx == 0 && blocks.empty()) return d;
    auto c = model->df_pass(ctx, nctx, blocks.data(), (uint32_t)blocks.size());
    for (size_t b = 0; b < blocks.size(); b++) {
      DfSeed &sd = seeds[of[b]];
      for (uint32_t k = 0; k < c[b].size(); k++) {
        if (k > 0 && sd.pos + k + 1 >= info.max_ctx) break;
        std::pair<int32_t, double> pick = sample_p(c[b][k], sd.settings, sd.pos + k, sd.hist);
        if (c[b][k].empty() || pick.second < draft_min_p) break;
        d[of[b]].push_back(pick.first);
        if (penalised(sd.settings)) sd.hist.push_back(pick.first);
      }
    }
    return d;
  }

  // The drafts the i-th of n writing slots may make: the window's rows (STEP_MAX_ROWS) shared out, each slot's row
  // first, and at most draft_max, or kSharedDrafts with several slots.
  uint32_t draft_share(uint32_t i, uint32_t n) const {
    if (n >= STEP_MAX_ROWS) return 0;
    const uint32_t spare = STEP_MAX_ROWS - n, most = n > 1 ? std::min(draft_max, kSharedDrafts) : draft_max;
    return std::min(most, spare / n + (i < spare % n ? 1u : 0u));
  }

  // A draft round's window of n rows that took ns, and a slot's run in it: k drafts verified, a accepted, from the
  // lookup or the draft layer. Under mu.
  void count_round(size_t n, uint64_t ns, const engine::qwen4exp::StepStats &ss) {
    stats.rounds++;
    stats.round_rows += n;
    stats.round_ns += ns;
    stats.round_misses += ss.misses;
    stats.round_layers_missed += ss.layers_missed;
    round_ns = ns;
  }
  void count_run(Slot &sl, bool lookup, uint32_t k, uint32_t a) {
    sl.stats.steps++;
    sl.stats.rounds++;
    sl.stats.drafts += k;
    sl.stats.accepted += a;
    if (lookup) {
      sl.stats.lookup_rounds++;
      sl.stats.lookup_drafts += k;
      sl.stats.lookup_accepted += a;
      stats.lookup_runs++;
      stats.lookup_drafts += k;
      stats.lookup_accepted += a;
      stats.lookup_ns += round_ns;
    } else {
      stats.layer_runs++;
      stats.layer_drafts += k;
      stats.layer_accepted += a;
      stats.layer_ns += round_ns;
    }
    round_ns = 0;  // a round of several slots counts its time once, to the layer
  }
  uint64_t round_ns = 0;  // the round being counted

  // Whether a slot still has prompt to read. Under mu.
  bool reading() const {
    for (const Slot &sl : slots)
      if (sl.used && !sl.done && sl.fed < sl.prompt.size()) return true;
    return false;
  }

  // A prompt piece as read from its slot before it runs.
  struct Piece {
    uint32_t slot = 0, pos0 = 0, n = 0;
    std::vector<int32_t> toks, next;  // next: the slot's piece expected after this one, for its n-gram rows
    int32_t p1 = kNoToken, p2 = kNoToken;
    bool windowed = false, restart = false;
    bool writer = false;  // a writing slot's next token, read beside the prompt pieces (add_writers)
  };

  // Reads the piece's tokens from its slot. False when there is nothing to read: the slot is gone or done, or its next
  // cell has no page (it stalls). Runs on the engine thread.
  bool prepare_piece(engine_prompt_piece pc, Piece &pi) {
    std::lock_guard<std::mutex> lk(mu);
    if (pc.slot >= slots.size()) return false;
    Slot &sl = slots[pc.slot];
    if (!sl.used || sl.done || sl.fed >= sl.prompt.size() || info.prompt_chunk == 0) return false;
    const uint32_t pos0 = sl.fed, left = (uint32_t)sl.prompt.size() - pos0;
    uint32_t n = std::min(left, info.prompt_chunk);
    if (pc.tokens) n = std::min(n, pc.tokens);
    if (mapped_cells(pc.slot) <= pos0) {
      stall(pc.slot, pos0);
      return false;
    }
    n = std::min(n, mapped_cells(pc.slot) - pos0);
    pi.slot = pc.slot;
    pi.pos0 = pos0;
    pi.n = n;
    pi.toks.assign(sl.prompt.begin() + pos0, sl.prompt.begin() + pos0 + n);
    pi.p1 = pos0 >= 1 ? sl.tokens[pos0 - 1] : kNoToken;
    pi.p2 = pos0 >= 2 ? sl.tokens[pos0 - 2] : kNoToken;
    // A decode step takes no cell of an image's block: such a piece goes by the prompt path.
    bool image = false;
    for (const SlotImage &im : sl.images)
      image |= im.first < pos0 + n && im.first + (im.grid_h / 2) * (im.grid_w / 2) > pos0;
    pi.windowed = n <= kWindowPrompt && !image;
    pi.restart = sl.windows_end != pos0;
    // The piece expected next, of the same size, so its n-gram rows are read while this one runs. A short one is
    // read as windows, which gather their own rows.
    const uint32_t nn = std::min(left - n, pc.tokens ? std::min(pc.tokens, info.prompt_chunk) : info.prompt_chunk);
    if (!pi.windowed && nn > kWindowPrompt)
      pi.next.assign(sl.prompt.begin() + pos0 + n, sl.prompt.begin() + pos0 + n + nn);
    return true;
  }

  // Writing slots whose next token joins the chunk as a run of one row, while the chunk has runs to spare: they get a
  // token from the chunk instead of waiting it out, and the chunk has staged every expert their rows route to, so the
  // CPU computes none. A short run attends from the 8-bit cache as the decode step does (prompt_attn.cl, p_attn_rows).
  // Under mu.
  void add_writers(std::vector<Piece> &pieces) {
    for (uint32_t s = 0; s < slots.size() && pieces.size() < PROMPT_MAX_RUNS; s++) {
      const Slot &sl = slots[s];
      if (!sl.used || sl.done || sl.fed < sl.prompt.size() || sl.fed >= info.max_ctx || sl.fed >= mapped_cells(s))
        continue;
      if (std::any_of(pieces.begin(), pieces.end(), [&](const Piece &pi) { return pi.slot == s; })) continue;
      Piece pi;
      pi.slot = s;
      pi.pos0 = sl.fed;
      pi.n = 1;
      pi.toks = {sl.next};
      pi.p1 = sl.fed >= 1 ? sl.tokens[sl.fed - 1] : kNoToken;
      pi.p2 = sl.fed >= 2 ? sl.tokens[sl.fed - 2] : kNoToken;
      pi.writer = true;
      pieces.push_back(std::move(pi));
    }
  }

  // A writer's row the prompt path has read, at row end - 1 of the chunk: as a decode step's row, its token joins the
  // slot and the next is sampled; the draft layer reads the row so its cells stay whole, and makes no drafts while a
  // prompt is read, as for several writers in decode. Runs on the engine thread.
  void writer_read(const Piece &pi, const std::vector<Candidate> &cand, uint32_t end) {
    StepIn row;
    {
      std::lock_guard<std::mutex> lk(mu);
      Slot &sl = slots[pi.slot];
      if (!sl.used || sl.done) return;
      sl.tokens.push_back(pi.toks[0]);
      sl.fed = pi.pos0 + 1;
      sl.stats.steps++;
      produce(pi.slot, cand);
      sl.drafts.clear();
      if (!model->drafting()) return;
      row = {pi.slot, pi.pos0, sl.next, kNoToken, kNoToken};
    }
    if (model->dflash())
      model->df_context_prompt(&row, 1, end);
    else
      model->mtp_catchup_prompt(&row, 1, end, {});  // writes its cell; no drafts while a prompt is read
  }

  // Pieces of different slots through the prompt path as one chunk: the experts are staged once for all their rows
  // (Model::prompt_runs). Writing slots join it while it has runs to spare (add_writers). Runs on the engine thread.
  void prompt_chunk(uint32_t plan, std::vector<Piece> &pieces) {
    {
      std::lock_guard<std::mutex> lk(mu);
      add_writers(pieces);
    }
    std::vector<engine::qwen4exp::Model::PromptRun> runs;
    uint32_t n = 0, wrote = 0;
    for (const Piece &pi : pieces) {
      runs.push_back({pi.slot, pi.pos0, pi.toks.data(), pi.n, pi.p1, pi.p2});
      n += pi.n;
      wrote += pi.writer;
    }
    const Piece &first = pieces[0];
    const bool one = pieces.size() == 1 && !first.next.empty();
    uint64_t t0 = now_ns();
    auto cand = model->prompt_runs(runs.data(), (uint32_t)runs.size(), one ? first.next.data() : nullptr,
                                   one ? (uint32_t)first.next.size() : 0);
    {
      std::lock_guard<std::mutex> lk(mu);
      const uint64_t t1 = now_ns();
      record({ENGINE_TIMING_PROMPT, plan, n, 0, t0, t1});
      stats.prompt_chunks++;
      stats.prompt_rows += n - wrote;
      stats.prompt_ns += t1 - t0;
    }
    for (uint32_t r = 0, end = 0; r < pieces.size(); r++) {
      end += pieces[r].n;
      if (pieces[r].writer)
        writer_read(pieces[r], cand[r], end);
      else
        piece_read(pieces[r], cand[r], end);
    }
    pieces.clear();
  }

  // A piece the prompt path has read, its last row at row end - 1 of the chunk: its tokens join the slot, and at the
  // prompt's end its first token is chosen and the draft layer drafts. Runs on the engine thread.
  void piece_read(const Piece &pi, const std::vector<Candidate> &cand, uint32_t end) {
    std::lock_guard<std::mutex> lk(mu);
    Slot &sl = slots[pi.slot];
    if (!sl.used) return;  // released while it ran
    sl.tokens.insert(sl.tokens.end(), pi.toks.begin(), pi.toks.end());
    if (model->drafting()) sl.lookup.index(sl.tokens);  // a piece at a time, not all at the first round
    sl.fed = pi.pos0 + pi.n;
    emit({ENGINE_EVENT_PROMPT, pi.slot, kNoToken, sl.fed});
    if (sl.fed != sl.prompt.size()) return;
    prompt_ended = true;
    produce(pi.slot, cand);
    if (!model->drafting() || sl.done) return;
    // The draft layer reads the prompt's last rows (its cells start there), and drafts.
    uint32_t k = std::min<uint32_t>(pi.n, STEP_MAX_ROWS), fed = sl.fed;
    std::vector<StepIn> mr;
    for (uint32_t j = 0; j < k; j++) {
      uint32_t pos = fed - k + j;
      mr.push_back({pi.slot, pos, pos + 1 < fed ? sl.tokens[pos + 1] : sl.next, kNoToken, kNoToken});
    }
    model->draft_start(pi.slot, fed - k);
    if (model->dflash()) {
      model->df_context_prompt(mr.data(), k, end);
      std::vector<DfSeed> seeds{{pi.slot, sl.settings, fed + 1, sl.next, draft_share(0, 1), draft_history(sl)}};
      sl.drafts = std::move(df_drafts(nullptr, 0, seeds)[0]);  // under mu, as below
      return;
    }
    auto dc = model->mtp_catchup_prompt(mr.data(), k, end, {k - 1});
    sl.drafts = make_drafts(pi.slot, sl.settings, fed + 1, dc[0], 0, draft_history(sl));  // under mu: admit and release wait for it
  }

  // A plan's prompt pieces. Those of different slots that the prompt path reads go as one chunk, up to
  // PROMPT_MAX_RUNS of them and prompt_chunk rows; a short one goes as windows when it comes. Runs on the engine
  // thread.
  void prompt_pieces(uint32_t plan, const engine_step_plan &sp) {
    std::vector<Piece> chunk;
    uint32_t rows = 0;
    auto flush = [&] {
      if (chunk.empty()) return;
      prompt_chunk(plan, chunk);
      rows = 0;
      if (swaps) model->note_prompt_routing();
    };
    for (uint32_t i = 0; i < sp.npieces && i < ENGINE_MAX_PROMPT_PIECES; i++) {
      // A piece of a slot already in the chunk starts where that one ends.
      for (const Piece &pi : chunk)
        if (pi.slot == sp.pieces[i].slot) {
          flush();
          break;
        }
      Piece pi;
      if (!prepare_piece(sp.pieces[i], pi)) continue;
      if (pi.windowed) {  // its slot is not in the chunk, which can wait for it
        prompt_windows(plan, pi.slot, pi.pos0, pi.toks, pi.p1, pi.p2, pi.restart);
        if (swaps) model->note_prompt_routing();
        continue;
      }
      if (chunk.size() == PROMPT_MAX_RUNS || rows + pi.n > info.prompt_chunk) flush();
      rows += pi.n;
      chunk.push_back(std::move(pi));
    }
    flush();
  }

  // A short prompt piece as decode windows of up to STEP_MAX_ROWS forced rows, each committed whole. The draft layer
  // reads every row, from pos0 on when restart (its cells before were never computed). Then as piece_read. Runs on
  // the engine thread.
  void prompt_windows(uint32_t plan, uint32_t s, uint32_t pos0, const std::vector<int32_t> &toks, int32_t p1,
                      int32_t p2, bool restart) {
    const uint32_t n = (uint32_t)toks.size();
    const bool mtp = model->drafting();
    int32_t after;  // the prompt's token after the piece, or kNoToken when the piece ends the prompt
    {
      std::lock_guard<std::mutex> lk(mu);
      const Slot &sl = slots[s];
      after = pos0 + n < sl.prompt.size() ? sl.prompt[pos0 + n] : kNoToken;
    }
    uint64_t t0 = now_ns();
    if (mtp && restart) model->draft_start(s, pos0);
    std::vector<Candidate> cand;
    std::vector<StepIn> mr;  // the last window's rows for the draft layer, each with the token after it
    for (uint32_t i = 0; i < n; i += STEP_MAX_ROWS) {
      const uint32_t w = std::min<uint32_t>(STEP_MAX_ROWS, n - i);
      std::vector<StepIn> rows;
      for (uint32_t k = i; k < i + w; k++) {
        const int32_t a = k >= 1 ? toks[k - 1] : p1, b = k >= 2 ? toks[k - 2] : k == 1 ? p1 : p2;
        rows.push_back({s, pos0 + k, toks[k], a, b});
      }
      auto out = model->step(rows.data(), w, true, true);
      model->commit(s, pos0 + i, w);
      cand = out[w - 1];
      mr.clear();
      for (uint32_t k = i; k < i + w; k++)
        mr.push_back({s, pos0 + k, k + 1 < n ? toks[k + 1] : after, kNoToken, kNoToken});
      // The last row's next token is sampled below when the piece ends the prompt; DFlash 2's context needs no token.
      if (mtp && model->dflash()) {
        std::vector<DfSeed> none;
        df_drafts(mr.data(), w, none);
      } else if (mtp && mr.back().token != kNoToken) {
        model->mtp_catchup(mr.data(), w, {});
      }
    }
    std::lock_guard<std::mutex> lk(mu);
    const uint64_t t1 = now_ns();
    record({ENGINE_TIMING_PROMPT, plan, n, 0, t0, t1});
    stats.window_pieces++;
    stats.window_rows += n;
    stats.window_ns += t1 - t0;
    Slot &sl = slots[s];
    if (!sl.used) return;  // released while it ran
    sl.tokens.insert(sl.tokens.end(), toks.begin(), toks.end());
    if (mtp) sl.lookup.index(sl.tokens);  // as piece_read
    sl.fed = pos0 + n;
    sl.windows_end = sl.fed;
    emit({ENGINE_EVENT_PROMPT, s, kNoToken, sl.fed});
    if (sl.fed != sl.prompt.size()) return;
    prompt_ended = true;
    produce(s, cand);
    if (!mtp || sl.done) return;
    if (model->dflash()) {
      std::vector<DfSeed> seeds{{s, sl.settings, sl.fed + 1, sl.next, draft_share(0, 1), draft_history(sl)}};
      sl.drafts = std::move(df_drafts(nullptr, 0, seeds)[0]);  // under mu, as below
      return;
    }
    mr.back().token = sl.next;
    auto dc = model->mtp_catchup(mr.data(), (uint32_t)mr.size(), {(uint32_t)mr.size() - 1});
    sl.drafts = make_drafts(s, sl.settings, sl.fed + 1, dc[0], 0, draft_history(sl));  // under mu, as above
  }

  // One decode step for the plan's slots, batched. Runs on the engine thread.
  void decode(uint32_t plan, const engine_step_plan &p) {
    std::vector<StepIn> rows;
    std::vector<uint32_t> who;
    bool all_writing = true;  // no row reads a prompt token
    {
      std::lock_guard<std::mutex> lk(mu);
      for (uint32_t i = 0; i < p.nslots && i < ENGINE_MAX_STEP_SLOTS; i++) {
        uint32_t s = p.slots[i];
        if (s >= slots.size() || std::find(who.begin(), who.end(), s) != who.end()) continue;
        Slot &sl = slots[s];
        if (!sl.used || sl.done) continue;
        if (sl.fed >= info.max_ctx) {
          sl.done = true;
          emit({ENGINE_EVENT_DONE, s, kNoToken, ENGINE_DONE_CONTEXT});
          continue;
        }
        if (sl.fed >= mapped_cells(s)) {
          stall(s, sl.fed);
          continue;
        }
        int32_t tok = sl.fed < sl.prompt.size() ? sl.prompt[sl.fed] : sl.next;
        all_writing = all_writing && sl.fed >= sl.prompt.size();
        uint32_t pos = sl.fed;
        rows.push_back({s, pos, tok, pos >= 1 ? sl.tokens[pos - 1] : kNoToken, pos >= 2 ? sl.tokens[pos - 2] : kNoToken});
        who.push_back(s);
      }
    }
    writers = (uint32_t)rows.size();
    if (rows.empty()) return;
    {
      std::unique_lock<std::mutex> lk(mu);
      Slot &sl = slots[who[0]];
      if (rows.size() == 1 && model->drafting() && sl.fed >= sl.prompt.size()) {
        // A lone writing slot's own tokens may give better drafts than the draft layer
        // (docs/adr/0041-drafts-from-the-slots-own-tokens.md).
        std::vector<int32_t> look = sl.lookup.drafts(sl.tokens, rows[0].token, STEP_MAX_ROWS - 1, media);
        const uint32_t take = engine::choose_drafts(sl.odds, (uint32_t)sl.drafts.size(), draft_max,
                                                    (uint32_t)look.size(), STEP_MAX_ROWS, window_times);
        sl.lookup_round = take > 0;
        if (take) look.resize(take), sl.drafts = std::move(look);
      }
      if (rows.size() == 1 && model->drafting() && sl.fed >= sl.prompt.size() && !sl.drafts.empty()) {
        std::vector<int32_t> d = sl.drafts;
        lk.unlock();
        round(plan, rows[0], d);
        return;
      }
      // Several slots, all writing, some with drafts: their windows in one pass. Not while a prompt is read, nor for a
      // slot's second token, which already waits for the prompt steps after its first: a round there would lengthen
      // the step that ends the wait (docs/measurements/MEASUREMENTS.md#2026-10-19-draft-rounds-several-writers).
      bool writing = model->drafting() && rows.size() > 1 && !reading(), any = false;
      std::vector<std::vector<int32_t>> ds;
      for (uint32_t s : who) {
        writing = writing && slots[s].fed >= slots[s].prompt.size() && slots[s].generated > 1;
        any = any || !slots[s].drafts.empty();
        ds.push_back(slots[s].drafts);
      }
      if (writing && any) {
        lk.unlock();
        rounds(plan, rows, ds);
        return;
      }
    }
    uint64_t t0 = now_ns();
    auto out = model->step(rows.data(), (uint32_t)rows.size());
    std::vector<StepIn> mr;  // the draft layer's catch-up: each row with the token after it
    {
      std::lock_guard<std::mutex> lk(mu);
      const uint64_t t1 = now_ns();
      const engine::qwen4exp::StepStats &ss = model->last_stats();
      record({ENGINE_TIMING_DECODE, plan, (uint32_t)rows.size(), ss.misses, t0, t1});
      if (rows.size() == 1) window_times.add(1, (t1 - t0) * 1e-6);
      swap_cadence.step((uint32_t)rows.size(), ss.layers_missed, (t1 - t0) * 1e-6);
      stats.steps++;
      stats.step_rows += rows.size();
      stats.step_ns += t1 - t0;
      stats.step_misses += ss.misses;
      stats.step_layers_missed += ss.layers_missed;
      if (all_writing && model->drafting()) stats.undrafted_steps++;
      for (size_t i = 0; i < rows.size(); i++) {
        uint32_t s = who[i];
        Slot &sl = slots[s];
        if (!sl.used) continue;
        sl.tokens.push_back(rows[i].token);
        sl.fed = rows[i].pos + 1;
        if (sl.fed < sl.prompt.size()) {
          if (sl.fed % 512 == 0) emit({ENGINE_EVENT_PROMPT, s, kNoToken, sl.fed});
          mr.push_back({s, rows[i].pos, sl.prompt[sl.fed], kNoToken, kNoToken});
          continue;
        }
        if (sl.fed == sl.prompt.size()) emit({ENGINE_EVENT_PROMPT, s, kNoToken, sl.fed});
        if (sl.fed > sl.prompt.size()) sl.stats.steps++;  // the row was one of its own tokens
        produce(s, out[i]);
        sl.drafts.clear();
        mr.push_back({s, rows[i].pos, sl.next, kNoToken, kNoToken});
      }
    }
    // The draft layer reads every computed row, so its cells stay whole; writing slots get drafts.
    if (!model->drafting() || mr.empty()) return;
    if (model->dflash()) {
      std::vector<DfSeed> seeds;
      {
        std::lock_guard<std::mutex> lk(mu);
        const uint32_t n = (uint32_t)mr.size();
        for (uint32_t i = 0; i < n && (n == 1 || !reading()); i++) {
          const Slot &sl = slots[mr[i].slot];
          if (sl.used && !sl.done && sl.fed >= sl.prompt.size())
            seeds.push_back({mr[i].slot, sl.settings, sl.fed + 1, sl.next, draft_share(i, n), draft_history(sl)});
        }
      }
      auto d = df_drafts(mr.data(), (uint32_t)mr.size(), seeds);
      std::lock_guard<std::mutex> lk(mu);
      for (size_t i = 0; i < seeds.size(); i++) {
        Slot &sl = slots[seeds[i].slot];
        if (sl.used && !sl.done) sl.drafts = d[i];
      }
      return;
    }
    std::vector<uint32_t> keep;  // the rows that may draft
    {
      std::lock_guard<std::mutex> lk(mu);
      const bool read = reading();
      for (uint32_t i = 0; i < mr.size(); i++) {
        const Slot &sl = slots[mr[i].slot];
        if (sl.used && !sl.done && (mr.size() == 1 ? sl.fed >= sl.prompt.size() : !read)) keep.push_back(i);
      }
    }
    auto dc = model->mtp_catchup(mr.data(), (uint32_t)mr.size(), keep);
    if (mr.size() != 1) {
      std::vector<DraftSeed> seeds;
      {
        std::lock_guard<std::mutex> lk(mu);
        if (reading()) return;
        for (uint32_t j = 0; j < keep.size(); j++) {
          const uint32_t i = keep[j];
          const Slot &sl = slots[mr[i].slot];
          if (sl.used && !sl.done)
            seeds.push_back({mr[i].slot, sl.settings, sl.fed + 1, dc[j], j, draft_share(i, (uint32_t)mr.size()),
                             draft_history(sl)});
        }
      }
      auto d = make_drafts_many(seeds);
      std::lock_guard<std::mutex> lk(mu);
      for (size_t i = 0; i < seeds.size(); i++) {
        Slot &sl = slots[seeds[i].slot];
        if (sl.used && !sl.done) sl.drafts = d[i];
      }
      return;
    }
    std::unique_lock<std::mutex> lk(mu);
    Slot &sl = slots[mr[0].slot];
    if (!sl.used || sl.done || sl.fed < sl.prompt.size() || dc.empty()) return;
    const uint32_t s = mr[0].slot, pos = sl.fed + 1;
    const engine_settings st = sl.settings;
    std::vector<int32_t> h = draft_history(sl);
    lk.unlock();
    auto d = make_drafts(s, st, pos, dc[0], 0, std::move(h));
    lk.lock();
    if (slots[s].used && !slots[s].done) slots[s].drafts = d;
  }

  // A draft round for one slot (docs/spec/STRATA_DECODE.md, sections 1-3): row 0 (the token sampled last) and the
  // drafts verified in one window; the drafts accepted while each equals the token sampled after the row before it,
  // and the next token after them; the accepted rows committed; the draft layer catches up and drafts again.
  void round(uint32_t plan, StepIn row0, const std::vector<int32_t> &drafts) {
    const uint32_t s = row0.slot, p = row0.pos;
    uint32_t k = (uint32_t)std::min<size_t>(drafts.size(), info.max_ctx - 1 - p);
    k = std::min<uint32_t>(k, STEP_MAX_ROWS - 1);
    {
      std::lock_guard<std::mutex> lk(mu);
      k = std::min(k, mapped_cells(s) - 1 - p);  // the window's rows only where the slot has pages (decode checked p)
    }
    std::vector<StepIn> rows{row0};
    for (uint32_t j = 0; j < k; j++) {
      const StepIn &b = rows.back();
      rows.push_back({s, p + 1 + j, drafts[j], b.token, b.p1});
    }
    uint64_t t0 = now_ns();
    auto out = model->step(rows.data(), (uint32_t)rows.size(), true);
    std::vector<int32_t> picks;
    engine_settings settings;
    std::vector<int32_t> hist;  // the tokens before the row being sampled, for the penalties
    {
      std::lock_guard<std::mutex> lk(mu);
      settings = slots[s].settings;
      hist = penalty_window(slots[s].tokens, settings);
    }
    const bool pen = penalised(settings);
    uint32_t a = 0;
    for (; a <= k; a++) {
      if (pen) hist.push_back(rows[a].token);
      picks.push_back(sample_p(out[a], settings, p + a + 1, hist).first);
      if (a == k || drafts[a] != picks[a]) break;
    }
    model->commit(s, p, a + 1);
    bool done;
    {
      std::lock_guard<std::mutex> lk(mu);
      const uint64_t t1 = now_ns();
      const engine::qwen4exp::StepStats &ss = model->last_stats();
      record({ENGINE_TIMING_ROUND, plan, (uint32_t)rows.size(), ss.misses, t0, t1});
      window_times.add((uint32_t)rows.size(), (t1 - t0) * 1e-6);
      swap_cadence.step((uint32_t)rows.size(), ss.layers_missed, (t1 - t0) * 1e-6);
      Slot &sl = slots[s];
      if (!sl.used) return;
      count_round(rows.size(), t1 - t0, ss);
      count_run(sl, sl.lookup_round, k, a);
      if (sl.lookup_round)
        sl.odds.lookup_round(a, k);
      else if (k)
        sl.odds.layer_round(a);
      sl.lookup_round = false;
      for (uint32_t j = 0; j <= a; j++) sl.tokens.push_back(rows[j].token);
      sl.fed = p + a + 1;
      done = false;
      for (uint32_t j = 0; j <= a && !done; j++) done = emit_token(s, picks[j], p + j + 1);
      sl.drafts.clear();
    }
    if (done) return;
    std::vector<StepIn> mr;
    for (uint32_t j = 0; j <= a; j++) mr.push_back({s, p + j, picks[j], kNoToken, kNoToken});
    if (pen) hist.push_back(picks[a]);
    std::vector<int32_t> d;
    if (model->dflash()) {
      std::vector<DfSeed> seeds{{s, settings, p + a + 2, picks[a], draft_share(0, 1), std::move(hist)}};
      d = std::move(df_drafts(mr.data(), (uint32_t)mr.size(), seeds)[0]);
    } else {
      auto dc = model->mtp_catchup(mr.data(), (uint32_t)mr.size(), {a});
      d = make_drafts(s, settings, p + a + 2, dc[0], 0, std::move(hist));
    }
    std::lock_guard<std::mutex> lk(mu);
    if (slots[s].used && !slots[s].done) slots[s].drafts = d;
  }

  // A draft round for several slots in one window (round, slot by slot): each slot's run of rows, its token sampled
  // last and its drafts, as many as the window's rows allow. Each run is accepted and committed as round's one, then
  // the draft layer catches up on each run's accepted rows and drafts again for every slot. Runs on the engine thread.
  void rounds(uint32_t plan, const std::vector<StepIn> &row0, const std::vector<std::vector<int32_t>> &drafts) {
    const uint32_t ns = (uint32_t)row0.size();
    std::vector<uint32_t> start(ns), k(ns);
    std::vector<StepIn> rows;
    {
      std::lock_guard<std::mutex> lk(mu);
      for (uint32_t i = 0; i < ns; i++) {
        const uint32_t s = row0[i].slot, p = row0[i].pos;
        uint32_t ki = (uint32_t)std::min<size_t>(drafts[i].size(), info.max_ctx - 1 - p);
        ki = std::min(ki, mapped_cells(s) - 1 - p);
        ki = std::min<uint32_t>(ki, STEP_MAX_ROWS - (uint32_t)rows.size() - (ns - i));
        start[i] = (uint32_t)rows.size();
        k[i] = ki;
        rows.push_back(row0[i]);
        for (uint32_t j = 0; j < ki; j++) {
          const StepIn &b = rows.back();
          rows.push_back({s, p + 1 + j, drafts[i][j], b.token, b.p1});
        }
      }
    }
    const uint32_t n = (uint32_t)rows.size();
    uint64_t t0 = now_ns();
    auto out = model->step(rows.data(), n, true);
    std::vector<engine_settings> settings(ns);
    std::vector<std::vector<int32_t>> hist(ns);  // each slot's tokens before the row being sampled, for the penalties
    {
      std::lock_guard<std::mutex> lk(mu);
      for (uint32_t i = 0; i < ns; i++) {
        settings[i] = slots[row0[i].slot].settings;
        hist[i] = penalty_window(slots[row0[i].slot].tokens, settings[i]);
      }
    }
    std::vector<std::vector<int32_t>> picks(ns);
    std::vector<uint32_t> acc(ns);
    std::vector<CommitRun> runs;
    for (uint32_t i = 0; i < ns; i++) {
      const uint32_t p = row0[i].pos;
      const bool pen = penalised(settings[i]);
      uint32_t a = 0;
      for (; a <= k[i]; a++) {
        if (pen) hist[i].push_back(rows[start[i] + a].token);
        picks[i].push_back(sample_p(out[start[i] + a], settings[i], p + a + 1, hist[i]).first);
        if (a == k[i] || drafts[i][a] != picks[i][a]) break;
      }
      acc[i] = a;
      runs.push_back({a + 1, p, row0[i].slot, start[i]});
    }
    model->commit_runs(runs.data(), ns);
    std::vector<bool> live(ns, false);
    {
      std::lock_guard<std::mutex> lk(mu);
      const uint64_t t1 = now_ns();
      const engine::qwen4exp::StepStats &ss = model->last_stats();
      record({ENGINE_TIMING_ROUND, plan, n, ss.misses, t0, t1});
      swap_cadence.step(n, ss.layers_missed, (t1 - t0) * 1e-6);
      count_round(n, t1 - t0, ss);
      for (uint32_t i = 0; i < ns; i++) {
        const uint32_t s = row0[i].slot, p = row0[i].pos, a = acc[i];
        Slot &sl = slots[s];
        if (!sl.used) continue;
        count_run(sl, false, k[i], a);
        for (uint32_t j = 0; j <= a; j++) sl.tokens.push_back(rows[start[i] + j].token);
        sl.fed = p + a + 1;
        bool done = false;
        for (uint32_t j = 0; j <= a && !done; j++) done = emit_token(s, picks[i][j], p + j + 1);
        sl.drafts.clear();
        live[i] = !done;
      }
    }
    std::vector<StepIn> mr;
    for (uint32_t i = 0; i < ns; i++)
      for (uint32_t j = 0; j <= k[i]; j++) {
        const StepIn &r = rows[start[i] + j];
        mr.push_back({r.slot, r.pos, j < acc[i] ? rows[start[i] + j + 1].token : picks[i][acc[i]], kNoToken, kNoToken});
      }
    if (model->dflash()) {
      bool read;
      {
        std::lock_guard<std::mutex> lk(mu);
        read = reading();  // a request admitted while the round ran: no drafts
      }
      std::vector<DfSeed> seeds;
      for (uint32_t i = 0; i < ns && !read; i++)
        if (live[i]) {
          if (penalised(settings[i])) hist[i].push_back(picks[i][acc[i]]);
          seeds.push_back({row0[i].slot, settings[i], row0[i].pos + acc[i] + 2, picks[i][acc[i]], draft_share(i, ns),
                           std::move(hist[i])});
        }
      auto d = df_drafts(mr.data(), n, seeds);
      std::lock_guard<std::mutex> lk(mu);
      for (size_t i = 0; i < seeds.size(); i++) {
        Slot &sl = slots[seeds[i].slot];
        if (sl.used && !sl.done) sl.drafts = d[i];
      }
      return;
    }
    // The draft layer needs no rejected row: its cell is written again before any row reads it. Each live slot's
    // last accepted row runs the whole layer.
    std::vector<StepIn> ma;
    std::vector<uint32_t> from, keep;  // each row's row of the window
    for (uint32_t i = 0; i < ns; i++) {
      for (uint32_t j = 0; j <= acc[i]; j++) {
        ma.push_back(mr[start[i] + j]);
        from.push_back(start[i] + j);
      }
      if (live[i]) keep.push_back((uint32_t)ma.size() - 1);
    }
    auto dc = model->mtp_catchup(ma.data(), (uint32_t)ma.size(), keep, from.data());
    {
      std::lock_guard<std::mutex> lk(mu);
      if (reading()) return;  // a request admitted while the round ran
    }
    std::vector<DraftSeed> seeds;
    for (uint32_t i = 0, j = 0; i < ns; i++)
      if (live[i]) {
        if (penalised(settings[i])) hist[i].push_back(picks[i][acc[i]]);
        seeds.push_back({row0[i].slot, settings[i], row0[i].pos + acc[i] + 2, dc[j], j, draft_share(i, ns),
                         std::move(hist[i])});
        j++;
      }
    auto d = make_drafts_many(seeds);
    std::lock_guard<std::mutex> lk(mu);
    for (size_t i = 0; i < seeds.size(); i++) {
      Slot &sl = slots[seeds[i].slot];
      if (sl.used && !sl.done) sl.drafts = d[i];
    }
  }

  void run() {
    for (;;) {
      engine_step_plan plan;
      std::vector<engine_page_change> pages;
      uint32_t seq;
      {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return stop || !plans.empty(); });
        if (stop) return;
        plan = plans.front().plan;
        pages = std::move(plans.front().pages);
        plans.pop_front();
        seq = ++plans_taken;
        running = true;
        prompt_ended = false;
        writers = 0;
      }
      const uint64_t t_plan = now_ns();
      try {
        std::vector<std::pair<uint32_t, uint32_t>> drops;
        const uint32_t resident = kv_resident(plan, pages, t_plan, drops);
        for (auto [s, n] : drops) model->kvs_drop(s, n);
        model->kv_need(kv_in_use(plan, pages), resident);
        apply_pages(pages);
        prompt_pieces(seq, plan);
        decode(seq, plan);
        // A swap round after a plan that ended a prompt, and otherwise once the time the CPU's misses have cost since
        // the last round reaches that round's time, within 4 to 16 plans of it (api/swap_cadence.h); 4 and 16 until
        // both are known, and 16 while several slots write. Never while a slot reads its prompt
        // (docs/measurements/MEASUREMENTS.md#2026-10-16-drafting-and-swaps). While several slots write, the round's swaps are
        // shared out: the cache follows several requests' routing at once.
        bool due;
        {
          std::lock_guard<std::mutex> lk(mu);
          swap_cadence.plan();
          due = prompt_ended || swap_cadence.due(writers);
        }
        uint64_t swapped = 0, swap_ns = 0;
        if (swaps && due && !prompt_reading()) {
          const uint64_t t0 = now_ns();
          swapped = model->swap_round(64 / std::max<uint32_t>(1, writers));
          swap_ns = now_ns() - t0;
        }
        model->kv_ahead();
        const engine::qwen4exp::NgramStats ng = model->ngram_stats();
        std::lock_guard<std::mutex> lk(mu);
        stats.plans++;
        stats.busy_ns += now_ns() - t_plan;
        if (swap_ns) {
          swap_cadence.round(swap_ns * 1e-6);
          stats.swap_rounds++;
          stats.swaps += swapped;
          stats.swap_ns += swap_ns;
        }
        stats.ngram_rows = ng.rows;
        stats.ngram_hits = ng.hits;
        stats.ngram_reads = ng.reads;
        stats.experts_cached = model->experts_cached();
        stats.experts_floor = model->experts_floor();
        emit({ENGINE_EVENT_STEP, 0, kNoToken, ++plans_done});
      } catch (const std::exception &ex) {
        std::lock_guard<std::mutex> lk(mu);
        error = ex.what();
        plans_done++;
        emit({ENGINE_EVENT_ERROR, 0, kNoToken, ENGINE_ERROR});
      }
      std::lock_guard<std::mutex> lk(mu);
      running = false;
      idle_cv.notify_all();
    }
  }

  // Whether a slot is still reading its prompt.
  bool prompt_reading() {
    std::lock_guard<std::mutex> lk(mu);
    for (const Slot &sl : slots)
      if (sl.used && !sl.done && sl.fed < sl.prompt.size()) return true;
    return false;
  }

  // Waits until every queued plan has run. Under the lock lk.
  void drain(std::unique_lock<std::mutex> &lk) {
    idle_cv.wait(lk, [&] { return plans.empty() && !running; });
  }

  // Whether the engine sleeps, with the error saying so. Under mu.
  bool sleeping() {
    if (asleep) error = "the engine is asleep";
    return asleep;
  }

  // The checkpoints' pinned memory, each a slot's fixed state (docs/DESIGN.md, section 6): at open and at a wake, never
  // on the token path. Under mu.
  void pin_checkpoints() {
    for (auto &ck : checkpoints) {
      if (!ck.data) ck.data = static_cast<uint8_t *>(model->alloc_pinned(checkpoint_bytes));
      ck.bytes = checkpoint_bytes;
    }
  }
  void unpin_checkpoints() {
    for (auto &ck : checkpoints) {
      if (ck.data) model->free_pinned(ck.data);
      ck = Checkpoint{};
    }
  }
};

extern "C" {

engine_status engine_open(const engine_config *config, engine_info *info, engine_ctx **out) {
  if (!config || !config->model_path || !out) return ENGINE_BAD_ARGUMENT;
  try {
    engine::qwen4exp::Options opt;
    if (config->slots) opt.slots = config->slots;
    if (config->max_ctx) opt.max_ctx = config->max_ctx;
    if (config->max_cards) opt.max_cards = config->max_cards;
    if (config->flags & ENGINE_CONFIG_EXPERTS_SET) opt.experts_per_layer = config->experts_per_layer;
    if (config->cpu_threads) opt.cpu_threads = config->cpu_threads;
    if (config->reserve_bytes) opt.reserve_bytes = config->reserve_bytes;
    if (config->prompt_chunk || (config->flags & ENGINE_CONFIG_PROMPT_SET)) opt.prompt_chunk = config->prompt_chunk;
    if (config->routing_profile) opt.routing_profile = config->routing_profile;
    if (config->mtp_dir) opt.mtp_dir = config->mtp_dir;
    if (config->vision_path) opt.vision_path = config->vision_path;
    opt.kv_pages = config->kv_pages;
    opt.kv_resident = config->kv_resident_cells;
    opt.adapt_swaps = (config->flags & ENGINE_CONFIG_SWAPS) != 0;
    opt.share_kv = (config->flags & ENGINE_CONFIG_SHARE_KV) != 0;
    auto e = std::make_unique<engine_ctx>();
    e->model = std::make_unique<engine::qwen4exp::Model>(config->model_path, opt);
    const auto &li = e->model->info();
    e->info.api_version = ENGINE_API_VERSION;
    e->info.cards = li.cards;
    e->info.slots = opt.slots;
    e->info.max_ctx = opt.max_ctx;
    e->info.experts_per_layer = li.experts_per_layer;
    e->info.vocab = li.vocab;
    e->info.device_bytes[0] = li.device_bytes[0];
    e->info.device_bytes[1] = li.device_bytes[1];
    e->info.arena_bytes = li.arena_bytes;
    e->info.load_seconds = li.seconds;
    e->info.prompt_chunk = e->model->prompt_chunk();
    e->info.max_step_rows = ENGINE_MAX_STEP_SLOTS;
    e->info.drafting = e->model->drafting() ? 1 : 0;
    e->info.image_tokens_max = e->model->image_tokens_max();
    e->info.kv_pages = e->model->kv_pages();
    e->info.pages_per_slot = e->model->pages_per_slot();
    if (config->draft_max) e->draft_max = std::min<uint32_t>(config->draft_max, ENGINE_MAX_STEP_SLOTS - 1);
    else if (e->model->dflash()) e->draft_max = STEP_MAX_ROWS - 1;  // one pass drafts a whole window
    if (config->draft_min_p > 0) e->draft_min_p = config->draft_min_p;
    e->swaps = opt.adapt_swaps;
    e->t_open = now_ns();
    e->stats.experts_cached = e->model->experts_cached();
    e->stats.experts_floor = e->model->experts_floor();
    // The checkpoints, each a slot's fixed state, in pinned host memory: allocated now, never on the token path
    // (docs/DESIGN.md, section 6).
    uint32_t nck = config->checkpoints ? config->checkpoints : 2;
    e->checkpoint_bytes = e->model->slot_state_bytes();
    e->checkpoints.resize(nck);
    e->pin_checkpoints();
    e->info.checkpoints = nck;
    e->slots.resize(opt.slots);
    e->table.assign(opt.slots, std::vector<uint32_t>(e->info.pages_per_slot, 0));
    e->mapped.assign(opt.slots, 0);
    e->queued_mapped.assign(opt.slots, 0);
    e->top.assign(opt.slots, 0);
    e->thread = std::thread([p = e.get()] { p->run(); });
    if (info) *info = e->info;
    *out = e.release();
    return ENGINE_OK;
  } catch (const std::exception &ex) {
    open_error = ex.what();
    return ENGINE_ERROR;
  }
}

engine_status engine_admit(engine_ctx *e, uint32_t slot, const engine_settings *settings, const int32_t *prompt,
                           uint32_t n_prompt) {
  if (!e || !settings || (!prompt && n_prompt) || n_prompt == 0) return ENGINE_BAD_ARGUMENT;
  std::lock_guard<std::mutex> lk(e->mu);
  if (slot >= e->slots.size()) return ENGINE_BAD_ARGUMENT;
  if (n_prompt > e->info.max_ctx) return ENGINE_BAD_ARGUMENT;
  if (!(settings->min_p >= 0.0f && settings->min_p <= 1.0f)) return ENGINE_BAD_ARGUMENT;
  if (penalised(*settings) && !(settings->penalty_repeat > 0.0f)) return ENGINE_BAD_ARGUMENT;
  if (e->sleeping()) return ENGINE_ERROR;
  Slot &sl = e->slots[slot];
  if (sl.used) return ENGINE_BUSY;
  try {
    // Nothing runs on the slot: the engine thread skips unused slots, and the slot was free.
    e->model->reset_slot(slot);
  } catch (const std::exception &ex) {
    e->error = ex.what();
    return ENGINE_ERROR;
  }
  sl = Slot{};
  sl.used = true;
  sl.settings = *settings;
  e->mapped[slot] = e->queued_mapped[slot] = 0;
  sl.prompt.assign(prompt, prompt + n_prompt);
  return ENGINE_OK;
}

engine_status engine_admit_images(engine_ctx *e, uint32_t slot, const engine_image *images, uint32_t n) {
  if (!e || (!images && n)) return ENGINE_BAD_ARGUMENT;
  std::lock_guard<std::mutex> lk(e->mu);
  if (slot >= e->slots.size()) return ENGINE_BAD_ARGUMENT;
  if (e->sleeping()) return ENGINE_ERROR;
  Slot &sl = e->slots[slot];
  if (!sl.used || sl.fed != 0) return ENGINE_BAD_ARGUMENT;
  if (n && !e->info.image_tokens_max) {
    e->error = "no image encoder: open with vision_path";
    return ENGINE_BAD_ARGUMENT;
  }
  std::vector<SlotImage> ims;
  std::vector<engine::qwen4exp::Image> mi;
  uint32_t end = 0;
  for (uint32_t i = 0; i < n; i++) {
    const engine_image &im = images[i];
    const uint64_t tokens = (uint64_t)(im.grid_h / 2) * (im.grid_w / 2);
    if (!im.pixels || !tokens || im.grid_h % 2 || im.grid_w % 2 || (im.frames != 1 && im.frames != 2) ||
        tokens > e->info.image_tokens_max || im.first < end || im.first + tokens > sl.prompt.size()) {
      e->error = "image " + std::to_string(i) + ": out of range, out of order, too large or not 1 or 2 frames";
      return ENGINE_BAD_ARGUMENT;
    }
    const int32_t pad = im.frames == 1 ? kImageToken : kVideoToken;
    for (uint64_t t = im.first; t < im.first + tokens; t++)
      if (sl.prompt[t] != pad) {
        e->error = "image " + std::to_string(i) + ": the prompt's block is not " +
                   (im.frames == 1 ? "image" : "video") + " tokens";
        return ENGINE_BAD_ARGUMENT;
      }
    end = (uint32_t)(im.first + tokens);
    auto px = std::make_shared<const std::vector<uint8_t>>(im.pixels, im.pixels + tokens * 4 * 768 * im.frames);
    ims.push_back({im.first, im.grid_h, im.grid_w, im.frames, px});
    mi.push_back({im.first, im.frames, im.grid_h, im.grid_w, px->data()});
  }
  try {
    // Nothing runs on the slot: it has read nothing yet, and the engine thread reads a slot's images only for it.
    e->model->set_images(slot, mi.data(), n);
  } catch (const std::exception &ex) {
    e->error = ex.what();
    return ENGINE_BAD_ARGUMENT;
  }
  sl.images = std::move(ims);
  return ENGINE_OK;
}

engine_status engine_release(engine_ctx *e, uint32_t slot) {
  if (!e) return ENGINE_BAD_ARGUMENT;
  std::lock_guard<std::mutex> lk(e->mu);
  if (slot >= e->slots.size()) return ENGINE_BAD_ARGUMENT;
  e->slots[slot].used = false;
  return ENGINE_OK;
}

engine_status engine_submit(engine_ctx *e, const engine_step_plan *plan) {
  if (!e || !plan || plan->nslots > ENGINE_MAX_STEP_SLOTS || plan->npieces > ENGINE_MAX_PROMPT_PIECES ||
      (plan->npages && !plan->pages))
    return ENGINE_BAD_ARGUMENT;
  std::lock_guard<std::mutex> lk(e->mu);
  if (e->sleeping()) return ENGINE_ERROR;
  if (e->plans.size() >= 64) return ENGINE_BUSY;
  // The page changes against the tables as the plans queued before leave them, all or none.
  const uint32_t C = e->info.cards, kvp = e->info.kv_pages, npp = e->info.pages_per_slot;
  std::vector<uint32_t> after = e->queued_mapped;
  for (uint32_t i = 0; i < plan->npages; i++) {
    const engine_page_change &c = plan->pages[i];
    std::string bad;
    if (c.slot >= e->slots.size() || c.logical > after[c.slot] || c.logical >= npp)
      bad = "logical page out of order or past the slot's table";
    else if (c.page != ENGINE_NO_PAGE && (c.page >= kvp || c.page % C != c.logical % C))
      bad = "page out of the pool, or on the wrong card for its logical page";
    else if (c.page != ENGINE_NO_PAGE && c.cells &&
             (c.cells >= KV_PAGE_CELLS || c.source >= kvp || c.source % C != c.page % C))
      bad = "copy from a page out of the pool or on the wrong card, or of a whole page";
    if (!bad.empty()) {
      e->error = "page change " + std::to_string(i) + " (slot " + std::to_string(c.slot) + ", logical " +
                 std::to_string(c.logical) + "): " + bad;
      return ENGINE_BAD_ARGUMENT;
    }
    if (c.page == ENGINE_NO_PAGE)
      after[c.slot] = c.logical;
    else if (c.logical == after[c.slot])
      after[c.slot]++;
  }
  e->queued_mapped = std::move(after);
  e->plans.push_back({*plan, std::vector<engine_page_change>(plan->pages, plan->pages + plan->npages)});
  e->plans.back().plan.pages = nullptr;
  e->cv.notify_all();
  return ENGINE_OK;
}

engine_status engine_poll(engine_ctx *e, engine_event *buf, uint32_t cap, uint32_t *n) {
  if (!e || (!buf && cap) || !n) return ENGINE_BAD_ARGUMENT;
  std::lock_guard<std::mutex> lk(e->mu);
  uint32_t k = 0;
  while (k < cap && !e->events.empty()) {
    buf[k++] = e->events.front();
    e->events.pop_front();
  }
  *n = k;
  return ENGINE_OK;
}

engine_status engine_wait(engine_ctx *e, uint32_t timeout_ms) {
  if (!e) return ENGINE_BAD_ARGUMENT;
  std::unique_lock<std::mutex> lk(e->mu);
  bool ready = e->events_cv.wait_for(lk, std::chrono::milliseconds(timeout_ms), [&] { return !e->events.empty(); });
  return ready ? ENGINE_OK : ENGINE_TIMEOUT;
}

engine_status engine_checkpoint(engine_ctx *e, uint32_t slot, uint32_t id) {
  if (!e) return ENGINE_BAD_ARGUMENT;
  std::unique_lock<std::mutex> lk(e->mu);
  if (slot >= e->slots.size() || id >= e->checkpoints.size()) return ENGINE_BAD_ARGUMENT;
  if (e->sleeping()) return ENGINE_ERROR;
  e->drain(lk);
  Slot &sl = e->slots[slot];
  if (!sl.used) return ENGINE_BAD_ARGUMENT;
  Checkpoint &ck = e->checkpoints[id];
  try {
    e->model->save_slot(slot, ck.data);
  } catch (const std::exception &ex) {
    e->error = ex.what();
    return ENGINE_ERROR;
  }
  ck.tokens.assign(sl.tokens.begin(), sl.tokens.begin() + sl.fed);
  ck.images = images_before(sl.images, sl.fed);
  ck.full = true;
  e->stats.checkpoints_taken++;
  return ENGINE_OK;
}

engine_status engine_restore(engine_ctx *e, uint32_t slot, uint32_t id) {
  if (!e) return ENGINE_BAD_ARGUMENT;
  std::unique_lock<std::mutex> lk(e->mu);
  if (slot >= e->slots.size() || id >= e->checkpoints.size()) return ENGINE_BAD_ARGUMENT;
  if (e->sleeping()) return ENGINE_ERROR;
  e->drain(lk);
  Slot &sl = e->slots[slot];
  Checkpoint &ck = e->checkpoints[id];
  if (!sl.used || !ck.full || sl.fed != 0) return ENGINE_BAD_ARGUMENT;
  // The checkpoint's tokens must start the slot's prompt, and leave at least one to read (its last row's logits are
  // not kept).
  if (ck.tokens.size() >= sl.prompt.size() || !std::equal(ck.tokens.begin(), ck.tokens.end(), sl.prompt.begin()))
    return ENGINE_BAD_ARGUMENT;
  if (images_before(sl.images, (uint32_t)ck.tokens.size()) != ck.images) return ENGINE_BAD_ARGUMENT;
  try {
    e->model->load_slot(slot, ck.data);
  } catch (const std::exception &ex) {
    e->error = ex.what();
    return ENGINE_ERROR;
  }
  sl.tokens = ck.tokens;
  sl.fed = (uint32_t)ck.tokens.size();
  e->stats.restores++;
  return ENGINE_OK;
}

engine_status engine_drop(engine_ctx *e, uint32_t id) {
  if (!e) return ENGINE_BAD_ARGUMENT;
  std::lock_guard<std::mutex> lk(e->mu);
  if (id >= e->checkpoints.size()) return ENGINE_BAD_ARGUMENT;
  e->checkpoints[id].full = false;
  e->checkpoints[id].tokens.clear();
  e->checkpoints[id].images.clear();
  return ENGINE_OK;
}

engine_status engine_read_timing(engine_ctx *e, engine_timing *buf, uint32_t cap, uint32_t *n) {
  if (!e || (!buf && cap) || !n) return ENGINE_BAD_ARGUMENT;
  std::lock_guard<std::mutex> lk(e->mu);
  uint32_t k = 0;
  while (k < cap && !e->timing.empty()) {
    buf[k++] = e->timing.front();
    e->timing.pop_front();
  }
  *n = k;
  return ENGINE_OK;
}

engine_status engine_read_stats(engine_ctx *e, engine_stats *out) {
  if (!e || !out) return ENGINE_BAD_ARGUMENT;
  std::lock_guard<std::mutex> lk(e->mu);
  *out = e->stats;
  out->up_ns = now_ns() - e->t_open;
  return ENGINE_OK;
}

engine_status engine_read_slot_stats(engine_ctx *e, uint32_t slot, engine_slot_stats *out) {
  if (!e || !out) return ENGINE_BAD_ARGUMENT;
  std::lock_guard<std::mutex> lk(e->mu);
  if (slot >= e->slots.size() || !e->slots[slot].used) return ENGINE_BAD_ARGUMENT;
  *out = e->slots[slot].stats;
  return ENGINE_OK;
}

engine_status engine_sleep(engine_ctx *e) {
  if (!e) return ENGINE_BAD_ARGUMENT;
  std::unique_lock<std::mutex> lk(e->mu);
  if (e->asleep) return ENGINE_OK;
  for (uint32_t s = 0; s < e->slots.size(); s++)
    if (e->slots[s].used) {
      e->error = "slot " + std::to_string(s) + " is admitted";
      return ENGINE_BUSY;
    }
  e->drain(lk);
  try {
    e->model->sleep();
    e->unpin_checkpoints();
  } catch (const std::exception &ex) {
    e->error = ex.what();
    return ENGINE_ERROR;
  }
  e->asleep = true;
  e->info.device_bytes[0] = e->info.device_bytes[1] = 0;
  return ENGINE_OK;
}

engine_status engine_wake(engine_ctx *e, engine_info *info) {
  if (!e) return ENGINE_BAD_ARGUMENT;
  std::lock_guard<std::mutex> lk(e->mu);
  if (e->asleep) {
    try {
      e->pin_checkpoints();
      e->model->wake();
    } catch (const std::exception &ex) {
      e->error = ex.what();
      e->unpin_checkpoints();
      return ENGINE_ERROR;
    }
    e->asleep = false;
    // The model woke with its own table (each slot its own run of pages); Python maps every slot afresh.
    for (auto &row : e->table) std::fill(row.begin(), row.end(), 0u);
    std::fill(e->mapped.begin(), e->mapped.end(), 0u);
    std::fill(e->queued_mapped.begin(), e->queued_mapped.end(), 0u);
    const auto &li = e->model->info();
    e->info.experts_per_layer = li.experts_per_layer;
    e->info.device_bytes[0] = li.device_bytes[0];
    e->info.device_bytes[1] = li.device_bytes[1];
    e->info.load_seconds = li.seconds;
  }
  if (info) *info = e->info;
  return ENGINE_OK;
}

const char *engine_last_error(const engine_ctx *e) { return e ? e->error.c_str() : open_error.c_str(); }

void engine_close(engine_ctx *e) {
  if (!e) return;
  {
    std::lock_guard<std::mutex> lk(e->mu);
    e->stop = true;
  }
  e->cv.notify_all();
  if (e->thread.joinable()) e->thread.join();
  delete e;
}

uint32_t engine_abi_layout(engine_abi_row *table, uint32_t cap) {
#define ROW(s, f) {#s "." #f, (uint32_t)offsetof(s, f), (uint32_t)sizeof(((s *)nullptr)->f)}
#define SIZE(s) {#s, 0, (uint32_t)sizeof(s)}
  static const engine_abi_row rows[] = {
      ROW(engine_config, model_path), ROW(engine_config, slots), ROW(engine_config, max_ctx),
      ROW(engine_config, max_cards), ROW(engine_config, experts_per_layer), ROW(engine_config, cpu_threads),
      ROW(engine_config, flags), ROW(engine_config, reserve_bytes), ROW(engine_config, prompt_chunk),
      ROW(engine_config, checkpoints), ROW(engine_config, routing_profile), ROW(engine_config, mtp_dir),
      ROW(engine_config, draft_max), ROW(engine_config, draft_min_p), ROW(engine_config, vision_path),
      ROW(engine_config, kv_pages), ROW(engine_config, kv_resident_cells), SIZE(engine_config),
      ROW(engine_info, api_version), ROW(engine_info, cards), ROW(engine_info, slots), ROW(engine_info, max_ctx),
      ROW(engine_info, experts_per_layer), ROW(engine_info, vocab), ROW(engine_info, device_bytes),
      ROW(engine_info, arena_bytes), ROW(engine_info, load_seconds), ROW(engine_info, prompt_chunk),
      ROW(engine_info, checkpoints), ROW(engine_info, max_step_rows), ROW(engine_info, drafting),
      ROW(engine_info, image_tokens_max), ROW(engine_info, kv_pages), ROW(engine_info, pages_per_slot),
      SIZE(engine_info),
      ROW(engine_settings, temperature), ROW(engine_settings, top_k), ROW(engine_settings, top_p),
      ROW(engine_settings, max_tokens), ROW(engine_settings, seed), ROW(engine_settings, n_stop),
      ROW(engine_settings, stop), ROW(engine_settings, min_p), ROW(engine_settings, penalty_last_n),
      ROW(engine_settings, penalty_repeat), ROW(engine_settings, penalty_freq), ROW(engine_settings, penalty_present),
      SIZE(engine_settings),
      ROW(engine_page_change, slot), ROW(engine_page_change, logical), ROW(engine_page_change, page),
      ROW(engine_page_change, source), ROW(engine_page_change, cells), SIZE(engine_page_change),
      ROW(engine_prompt_piece, slot), ROW(engine_prompt_piece, tokens), SIZE(engine_prompt_piece),
      ROW(engine_step_plan, nslots), ROW(engine_step_plan, slots), ROW(engine_step_plan, npieces),
      ROW(engine_step_plan, pieces), ROW(engine_step_plan, npages), ROW(engine_step_plan, pages),
      ROW(engine_step_plan, kv_free), SIZE(engine_step_plan),
      ROW(engine_event, kind), ROW(engine_event, slot), ROW(engine_event, token), ROW(engine_event, value),
      SIZE(engine_event),
      ROW(engine_timing, kind), ROW(engine_timing, plan), ROW(engine_timing, rows), ROW(engine_timing, misses),
      ROW(engine_timing, start_ns), ROW(engine_timing, end_ns), SIZE(engine_timing),
      ROW(engine_image, first), ROW(engine_image, grid_h), ROW(engine_image, grid_w), ROW(engine_image, frames),
      ROW(engine_image, pixels), SIZE(engine_image),
      ROW(engine_stats, up_ns), ROW(engine_stats, plans), ROW(engine_stats, busy_ns), ROW(engine_stats, prompt_chunks),
      ROW(engine_stats, prompt_rows), ROW(engine_stats, prompt_ns), ROW(engine_stats, window_pieces),
      ROW(engine_stats, window_rows), ROW(engine_stats, window_ns), ROW(engine_stats, steps), ROW(engine_stats, step_rows),
      ROW(engine_stats, step_ns), ROW(engine_stats, step_misses), ROW(engine_stats, step_layers_missed),
      ROW(engine_stats, undrafted_steps), ROW(engine_stats, rounds), ROW(engine_stats, round_rows),
      ROW(engine_stats, round_ns), ROW(engine_stats, round_misses), ROW(engine_stats, round_layers_missed),
      ROW(engine_stats, layer_runs), ROW(engine_stats, layer_drafts), ROW(engine_stats, layer_accepted),
      ROW(engine_stats, layer_ns), ROW(engine_stats, lookup_runs), ROW(engine_stats, lookup_drafts),
      ROW(engine_stats, lookup_accepted), ROW(engine_stats, lookup_ns), ROW(engine_stats, swap_rounds),
      ROW(engine_stats, swaps), ROW(engine_stats, swap_ns), ROW(engine_stats, stalls),
      ROW(engine_stats, checkpoints_taken), ROW(engine_stats, restores), ROW(engine_stats, ngram_rows),
      ROW(engine_stats, ngram_hits), ROW(engine_stats, ngram_reads), ROW(engine_stats, experts_cached),
      ROW(engine_stats, experts_floor), SIZE(engine_stats),
      ROW(engine_slot_stats, steps), ROW(engine_slot_stats, rounds), ROW(engine_slot_stats, drafts),
      ROW(engine_slot_stats, accepted), ROW(engine_slot_stats, lookup_rounds), ROW(engine_slot_stats, lookup_drafts),
      ROW(engine_slot_stats, lookup_accepted), SIZE(engine_slot_stats),
      ROW(engine_abi_row, name), ROW(engine_abi_row, offset), ROW(engine_abi_row, size), SIZE(engine_abi_row),
  };
#undef ROW
#undef SIZE
  uint32_t n = sizeof rows / sizeof rows[0];
  if (table)
    for (uint32_t i = 0; i < n && i < cap; i++) table[i] = rows[i];
  return n;
}

}  // extern "C"
