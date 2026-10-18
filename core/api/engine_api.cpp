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

#include "models/qwen4exp/model.h"

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
};

// A prompt piece of at most this many tokens is read as decode windows of forced rows: a prompt chunk stages every
// uncached expert whatever its size, so a short piece is faster as windows
// (docs/measurements/2026-10-17-prompt-windows.md).
constexpr uint32_t kWindowPrompt = 64;

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

// A uniform number in [0, 1) from the request's seed and the position sampled at: a token depends only on these, not
// on how the work was split into steps or windows (splitmix64).
double uniform(uint64_t seed, uint32_t pos) {
  uint64_t z = seed + 0x9e3779b97f4a7c15ull * (pos + 1ull);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  z ^= z >> 31;
  return (double)(z >> 11) * (1.0 / 9007199254740992.0);
}

// Sampling over the engine's candidates (the largest logits, highest first): temperature, top-k, top-p. Returns the
// token and its probability under that distribution (greedy: the top candidate's under the plain softmax).
std::pair<int32_t, double> sample_p(const std::vector<Candidate> &cand, const engine_settings &s, uint32_t pos) {
  if (cand.empty()) return {kNoToken, 0.0};
  if (s.temperature <= 0.0f) {
    double sum = 0;
    for (auto &x : cand) sum += std::exp((double)x.logit - cand[0].logit);
    return {cand[0].token, 1.0 / sum};
  }
  size_t k = s.top_k == 0 ? cand.size() : std::min<size_t>(s.top_k, cand.size());
  std::vector<double> p(k);
  double sum = 0;
  for (size_t i = 0; i < k; i++) sum += p[i] = std::exp((cand[i].logit - cand[0].logit) / s.temperature);
  if (s.top_p > 0.0f && s.top_p < 1.0f) {
    double acc = 0;
    size_t keep = k;
    for (size_t i = 0; i < k; i++) {
      acc += p[i] / sum;
      if (acc >= s.top_p) {
        keep = i + 1;
        break;
      }
    }
    k = keep;
    p.resize(k);
    sum = 0;
    for (double v : p) sum += v;
  }
  double r = uniform(s.seed, pos) * sum;
  for (size_t i = 0; i < k; i++) {
    r -= p[i];
    if (r <= 0) return {cand[i].token, p[i] / sum};
  }
  return {cand[k - 1].token, p[k - 1] / sum};
}

int32_t sample(const std::vector<Candidate> &cand, const engine_settings &s, uint32_t pos) {
  return sample_p(cand, s, pos).first;
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
  std::deque<engine_event> events;
  std::deque<engine_timing> timing;
  uint32_t plans_done = 0, plans_taken = 0;
  bool running = false;  // a plan is being run
  bool prompt_ended = false;  // the plan being run read some slot's last prompt token
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
    emit_token(s, sample(cand, sl.settings, sl.fed), sl.fed);
  }

  // The slot's drafts, up to draft_max, from a catch-up's last row's candidates (row src of its pass); pos is the
  // first draft's position. A draft is the request's sampler's pick with its position's uniform, so the main model
  // accepts it when it samples the same; drafting stops below draft_min_p. The draft after position q reads the draft
  // layer's cell q - 1 (mtp.cpp). Runs on the engine thread.
  std::vector<int32_t> make_drafts(uint32_t s, const engine_settings &st, uint32_t pos, const std::vector<Candidate> &first,
                                   uint32_t src) {
    std::vector<int32_t> d;
    std::pair<int32_t, double> pick = sample_p(first, st, pos);
    if (first.empty() || pick.second < draft_min_p) return d;
    d.push_back(pick.first);
    while (d.size() < draft_max && pos + d.size() + 1 < info.max_ctx) {
      auto c = model->mtp_draft(s, pos + (uint32_t)d.size() - 2, d.back(), src);
      src = 0;
      pick = sample_p(c, st, pos + (uint32_t)d.size());
      if (pick.second < draft_min_p) break;
      d.push_back(pick.first);
    }
    return d;
  }

  // A prompt piece through the prompt path. Runs on the engine thread.
  void prompt_piece(uint32_t plan, engine_prompt_piece pc) {
    std::vector<int32_t> toks, next;
    uint32_t pos0, n;
    int32_t p1, p2;
    bool windowed, restart;
    {
      std::lock_guard<std::mutex> lk(mu);
      if (pc.slot >= slots.size()) return;
      Slot &sl = slots[pc.slot];
      if (!sl.used || sl.done || sl.fed >= sl.prompt.size() || info.prompt_chunk == 0) return;
      pos0 = sl.fed;
      uint32_t left = (uint32_t)sl.prompt.size() - pos0;
      n = std::min(left, info.prompt_chunk);
      if (pc.tokens) n = std::min(n, pc.tokens);
      if (mapped_cells(pc.slot) <= pos0) {
        stall(pc.slot, pos0);
        return;
      }
      n = std::min(n, mapped_cells(pc.slot) - pos0);
      toks.assign(sl.prompt.begin() + pos0, sl.prompt.begin() + pos0 + n);
      p1 = pos0 >= 1 ? sl.tokens[pos0 - 1] : kNoToken;
      p2 = pos0 >= 2 ? sl.tokens[pos0 - 2] : kNoToken;
      // A decode step takes no cell of an image's block: such a piece goes by the prompt path.
      bool image = false;
      for (const SlotImage &im : sl.images)
        image |= im.first < pos0 + n && im.first + (im.grid_h / 2) * (im.grid_w / 2) > pos0;
      windowed = n <= kWindowPrompt && !image;
      restart = sl.windows_end != pos0;
      // The piece expected next, of the same size, so its n-gram rows are read while this one runs. A short one is
      // read as windows, which gather their own rows.
      const uint32_t nn = std::min(left - n, pc.tokens ? std::min(pc.tokens, info.prompt_chunk) : info.prompt_chunk);
      if (!windowed && nn > kWindowPrompt)
        next.assign(sl.prompt.begin() + pos0 + n, sl.prompt.begin() + pos0 + n + nn);
    }
    if (windowed) {
      prompt_windows(plan, pc.slot, pos0, toks, p1, p2, restart);
      return;
    }
    uint64_t t0 = now_ns();
    auto cand = model->prompt(pc.slot, pos0, toks.data(), n, p1, p2, next.empty() ? nullptr : next.data(),
                              (uint32_t)next.size());
    std::lock_guard<std::mutex> lk(mu);
    record({ENGINE_TIMING_PROMPT, plan, n, 0, t0, now_ns()});
    Slot &sl = slots[pc.slot];
    if (!sl.used) return;  // released while it ran
    sl.tokens.insert(sl.tokens.end(), toks.begin(), toks.end());
    sl.fed = pos0 + n;
    emit({ENGINE_EVENT_PROMPT, pc.slot, kNoToken, sl.fed});
    if (sl.fed != sl.prompt.size()) return;
    prompt_ended = true;
    produce(pc.slot, cand);
    if (!model->drafting() || sl.done) return;
    // The draft layer reads the prompt's last rows (its cells start there), and drafts.
    uint32_t k = std::min<uint32_t>(n, STEP_MAX_ROWS), fed = sl.fed;
    std::vector<StepIn> mr;
    for (uint32_t j = 0; j < k; j++) {
      uint32_t pos = fed - k + j;
      mr.push_back({pc.slot, pos, pos + 1 < fed ? sl.tokens[pos + 1] : sl.next, kNoToken, kNoToken});
    }
    model->mtp_start(pc.slot, fed - k);
    auto dc = model->mtp_catchup_prompt(mr.data(), k, n);
    sl.drafts = make_drafts(pc.slot, sl.settings, fed + 1, dc[k - 1], k - 1);  // under mu: admit and release wait for it
  }

  // A short prompt piece as decode windows of up to STEP_MAX_ROWS forced rows, each committed whole. The draft layer
  // reads every row, from pos0 on when restart (its cells before were never computed). Then as prompt_piece. Runs on
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
    if (mtp && restart) model->mtp_start(s, pos0);
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
      // The last row's next token is sampled below when the piece ends the prompt.
      if (mtp && mr.back().token != kNoToken) model->mtp_catchup(mr.data(), w);
    }
    std::lock_guard<std::mutex> lk(mu);
    record({ENGINE_TIMING_PROMPT, plan, n, 0, t0, now_ns()});
    Slot &sl = slots[s];
    if (!sl.used) return;  // released while it ran
    sl.tokens.insert(sl.tokens.end(), toks.begin(), toks.end());
    sl.fed = pos0 + n;
    sl.windows_end = sl.fed;
    emit({ENGINE_EVENT_PROMPT, s, kNoToken, sl.fed});
    if (sl.fed != sl.prompt.size()) return;
    prompt_ended = true;
    produce(s, cand);
    if (!mtp || sl.done) return;
    mr.back().token = sl.next;
    auto dc = model->mtp_catchup(mr.data(), (uint32_t)mr.size());
    sl.drafts = make_drafts(s, sl.settings, sl.fed + 1, dc.back(), (uint32_t)mr.size() - 1);  // under mu, as above
  }

  // One decode step for the plan's slots, batched. Runs on the engine thread.
  void decode(uint32_t plan, const engine_step_plan &p) {
    std::vector<StepIn> rows;
    std::vector<uint32_t> who;
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
        uint32_t pos = sl.fed;
        rows.push_back({s, pos, tok, pos >= 1 ? sl.tokens[pos - 1] : kNoToken, pos >= 2 ? sl.tokens[pos - 2] : kNoToken});
        who.push_back(s);
      }
    }
    if (rows.empty()) return;
    {
      std::unique_lock<std::mutex> lk(mu);
      Slot &sl = slots[who[0]];
      if (rows.size() == 1 && model->drafting() && sl.fed >= sl.prompt.size() && !sl.drafts.empty()) {
        std::vector<int32_t> d = sl.drafts;
        lk.unlock();
        round(plan, rows[0], d);
        return;
      }
    }
    uint64_t t0 = now_ns();
    auto out = model->step(rows.data(), (uint32_t)rows.size());
    std::vector<StepIn> mr;  // the draft layer's catch-up: each row with the token after it
    {
      std::lock_guard<std::mutex> lk(mu);
      record({ENGINE_TIMING_DECODE, plan, (uint32_t)rows.size(), model->last_stats().misses, t0, now_ns()});
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
        produce(s, out[i]);
        sl.drafts.clear();
        mr.push_back({s, rows[i].pos, sl.next, kNoToken, kNoToken});
      }
    }
    // The draft layer reads every computed row, so its cells stay whole; a slot decoding alone gets drafts.
    if (!model->drafting() || mr.empty()) return;
    auto dc = model->mtp_catchup(mr.data(), (uint32_t)mr.size());
    if (mr.size() != 1) return;
    std::unique_lock<std::mutex> lk(mu);
    Slot &sl = slots[mr[0].slot];
    if (!sl.used || sl.done || sl.fed < sl.prompt.size()) return;
    const uint32_t s = mr[0].slot, pos = sl.fed + 1;
    const engine_settings st = sl.settings;
    lk.unlock();
    auto d = make_drafts(s, st, pos, dc[0], 0);
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
    {
      std::lock_guard<std::mutex> lk(mu);
      settings = slots[s].settings;
    }
    uint32_t a = 0;
    for (; a <= k; a++) {
      picks.push_back(sample(out[a], settings, p + a + 1));
      if (a == k || drafts[a] != picks[a]) break;
    }
    model->commit(s, p, a + 1);
    bool done;
    {
      std::lock_guard<std::mutex> lk(mu);
      record({ENGINE_TIMING_ROUND, plan, (uint32_t)rows.size(), model->last_stats().misses, t0, now_ns()});
      Slot &sl = slots[s];
      if (!sl.used) return;
      for (uint32_t j = 0; j <= a; j++) sl.tokens.push_back(rows[j].token);
      sl.fed = p + a + 1;
      done = false;
      for (uint32_t j = 0; j <= a && !done; j++) done = emit_token(s, picks[j], p + j + 1);
      sl.drafts.clear();
    }
    if (done) return;
    std::vector<StepIn> mr;
    for (uint32_t j = 0; j <= a; j++) mr.push_back({s, p + j, picks[j], kNoToken, kNoToken});
    auto dc = model->mtp_catchup(mr.data(), (uint32_t)mr.size());
    auto d = make_drafts(s, settings, p + a + 2, dc[a], a);
    std::lock_guard<std::mutex> lk(mu);
    if (slots[s].used && !slots[s].done) slots[s].drafts = d;
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
      }
      try {
        apply_pages(pages);
        for (uint32_t i = 0; i < plan.npieces && i < ENGINE_MAX_PROMPT_PIECES; i++) {
          prompt_piece(seq, plan.pieces[i]);
          if (swaps) model->note_prompt_routing();
        }
        decode(seq, plan);
        // A swap round after a plan that ended a prompt, and after every 16th plan. Never while a slot reads its
        // prompt (docs/measurements/2026-10-16-drafting-and-swaps.md).
        if (swaps && (prompt_ended || seq % 16 == 0) && !prompt_reading()) model->swap_round();
        std::lock_guard<std::mutex> lk(mu);
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
    opt.adapt_swaps = (config->flags & ENGINE_CONFIG_SWAPS) != 0;
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
    if (config->draft_min_p > 0) e->draft_min_p = config->draft_min_p;
    e->swaps = opt.adapt_swaps;
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
      ROW(engine_config, kv_pages), SIZE(engine_config),
      ROW(engine_info, api_version), ROW(engine_info, cards), ROW(engine_info, slots), ROW(engine_info, max_ctx),
      ROW(engine_info, experts_per_layer), ROW(engine_info, vocab), ROW(engine_info, device_bytes),
      ROW(engine_info, arena_bytes), ROW(engine_info, load_seconds), ROW(engine_info, prompt_chunk),
      ROW(engine_info, checkpoints), ROW(engine_info, max_step_rows), ROW(engine_info, drafting),
      ROW(engine_info, image_tokens_max), ROW(engine_info, kv_pages), ROW(engine_info, pages_per_slot),
      SIZE(engine_info),
      ROW(engine_settings, temperature), ROW(engine_settings, top_k), ROW(engine_settings, top_p),
      ROW(engine_settings, max_tokens), ROW(engine_settings, seed), ROW(engine_settings, n_stop),
      ROW(engine_settings, stop), SIZE(engine_settings),
      ROW(engine_page_change, slot), ROW(engine_page_change, logical), ROW(engine_page_change, page),
      ROW(engine_page_change, source), ROW(engine_page_change, cells), SIZE(engine_page_change),
      ROW(engine_prompt_piece, slot), ROW(engine_prompt_piece, tokens), SIZE(engine_prompt_piece),
      ROW(engine_step_plan, nslots), ROW(engine_step_plan, slots), ROW(engine_step_plan, npieces),
      ROW(engine_step_plan, pieces), ROW(engine_step_plan, npages), ROW(engine_step_plan, pages),
      SIZE(engine_step_plan),
      ROW(engine_event, kind), ROW(engine_event, slot), ROW(engine_event, token), ROW(engine_event, value),
      SIZE(engine_event),
      ROW(engine_timing, kind), ROW(engine_timing, plan), ROW(engine_timing, rows), ROW(engine_timing, misses),
      ROW(engine_timing, start_ns), ROW(engine_timing, end_ns), SIZE(engine_timing),
      ROW(engine_image, first), ROW(engine_image, grid_h), ROW(engine_image, grid_w), ROW(engine_image, frames),
      ROW(engine_image, pixels), SIZE(engine_image),
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
