// The model as the C ABI's code drives it (api/engine_api.cpp): the values passed both ways, and the calls made. Each
// architecture's class in models/<arch>/ derives from Model, and open_model picks the class by the file's
// general.architecture (ADR-0061).
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "step.h"

namespace engine {

constexpr int32_t kNoToken = -1;

struct Options {
  uint32_t slots = 1;                       // sequences resident at once
  uint32_t max_ctx = 32768;                 // positions per sequence, a multiple of KV_PAGE_CELLS
  uint32_t kv_pages = 0;                    // the KV cache's pool, in pages of KV_PAGE_CELLS cells; 0: every slot's
                                            // whole context
  uint32_t kv_resident = 0;                 // K and V cells per slot kept in VRAM, the rest in host memory (ADR-0008
                                            // item 3); 0: every cell
  uint32_t max_cards = 2;                   // 1 runs the whole model on card 0
  uint64_t reserve_bytes = 1ull << 30;      // device memory the engine does not count on: the driver's own, and some free
  int32_t experts_per_layer = -1;           // experts cached per layer; -1: as many as fit
  unsigned cpu_threads = 16;                // the CPU's expert workers, with the host thread
  uint32_t candidates = 64;                 // logits returned per step, the largest
  uint32_t prompt_chunk = 4352;             // tokens per prompt pass at most, a multiple of 64; 0: one token per
                                            // step
  bool trace = false;                       // the model times every kernel of its lists
  bool profile_routing = false;             // the prompt path counts each expert's rows
  bool adapt_swaps = false;                 // count the routing, for Model::swap_round
  // On one card, the CPU computes a short chunk's rarest uncached experts (ADR-0072). Its rows round differently from
  // the card's, so a run's numbers then depend on what shares its chunk: off where they must repeat exactly.
  bool rare_on_cpu = true;
  // The pages of the KV pool not in use lend their memory to the expert cache; needs adapt_swaps.
  bool share_kv = false;
  // The arena in the process's own memory, which the copy engines read in place (ADR-0035). false, or a card without
  // shared system USM: the driver's buffers, which xe moves out of GTT under reclaim and so stalls the CPU's experts
  // (docs/measurements/MEASUREMENTS.md#2026-10-17-decode-stalls-eviction).
  bool anon_arena = true;
  // The arena's pieces from the host's reserved huge pages (vm.nr_hugepages) while they last, then as anon_arena's:
  // pages the kernel never moves, so it never stops a card to bind one again. Needs anon_arena.
  bool hugetlb_arena = false;
  bool ngram_in_ram = false;                // the whole n-gram table in RAM
  // Bytes of the host's reserved huge pages for the n-gram table: with ngram_in_ram its first part, without it the row
  // cache (models/qwen4exp/ngram.h). A multiple of the huge page size; 0: none.
  uint64_t ngram_huge_bytes = 0;
  // A profile to choose the cached experts by (tools/route_profile.cpp); empty: the first experts of each layer;
  // none: the model's own.
  std::optional<std::string> routing_profile;
  // The drafter's files (docs/MODEL_FILES.md); empty: no drafting.
  std::string mtp_dir;
  // The image encoder's file (docs/spec/MODEL_VISION.md, section 1); empty: no images. Needs prompt_chunk.
  std::string vision_path;
};

// One kernel's share of a traced list's last run.
struct TraceRow {
  std::string kernel;
  uint32_t launches = 0;
  double ms = 0;  // from the end of the launch before it, so waits on events count to the kernel after them
};

struct Candidate {
  int32_t token;
  float logit;
};

struct LoadInfo {
  uint32_t cards = 0;
  uint32_t experts_per_layer = 0;
  uint32_t vocab = 0;
  uint64_t device_bytes[2] = {0, 0};
  uint64_t arena_bytes = 0;  // host memory holding the experts not on the cards
  uint64_t expert_budget[2] = {0, 0};  // device memory each card had left for the expert cache with the whole KV pool
                                       // backed
  bool arena_anon = false;   // the arena is the process's own memory (Options::anon_arena)
  double seconds = 0;
  double host_read_bytes_per_s = 0;  // the CPU workers' read of the arena at the first load; 0: not measured
  unsigned loader_threads = 0;       // the loader's threads, from DirectReader::probe_threads
  double loader_bytes_per_s = 0;     // their read rate in the probe
  bool ngram_in_ram = false;
  double ngram_read_us = 0;          // one 4 KiB row read from the file, when the table is not in RAM
  uint64_t ngram_cache_rows = 0;     // rows the n-gram cache holds
  uint64_t ngram_huge_bytes = 0;     // the n-gram table's or its cache's bytes in reserved huge pages
  double copy_bytes_per_s[2] = {0, 0};  // each card's copy engine reading pinned host memory
};

// One row of a decode step: a token of a sequence at its position; p1, p2 and p3 are the three tokens before it, p1
// the nearest (kNoToken when absent).
struct StepIn {
  uint32_t slot = 0, pos = 0;
  int32_t token = 0, p1 = -1, p2 = -1, p3 = -1;
};

// The row of token at position pos of a sequence whose tokens before it are tokens[0 .. pos).
inline StepIn row_at(uint32_t slot, uint32_t pos, int32_t token, const std::vector<int32_t> &tokens) {
  auto back = [&](uint32_t k) { return pos >= k ? tokens[pos - k] : kNoToken; };
  return {slot, pos, token, back(1), back(2), back(3)};
}

// The row after row b of the same sequence, holding token.
inline StepIn row_after(const StepIn &b, int32_t token) { return {b.slot, b.pos + 1, token, b.token, b.p1, b.p2}; }

// How a prompt holds an image (include/engine_api.h, engine_image): the token that fills an image's block, the one
// that fills a pair of video frames' (kNoToken: no video), and the patches. One token is merge x merge patches.
struct ImageShape {
  int32_t image_token = kNoToken;
  int32_t video_token = kNoToken;
  uint32_t patch_bytes = 0;  // one frame of one patch, 8-bit RGB
  uint32_t merge = 1;
};

// Whether tok is the shape's image or video token.
inline bool is_media(const ImageShape &sh, int32_t tok) {
  return tok != kNoToken && (tok == sh.image_token || tok == sh.video_token);
}

// The tokens of the block of an image of grid_h x grid_w patches and frames frames; 0 when the shape cannot hold it: a
// side not a multiple of merge, or frames not 1, or 2 with a video token.
inline uint64_t image_block_tokens(const ImageShape &sh, uint32_t frames, uint32_t grid_h, uint32_t grid_w) {
  if (grid_h % sh.merge || grid_w % sh.merge || (frames != 1 && !(frames == 2 && sh.video_token != kNoToken)))
    return 0;
  return (uint64_t)(grid_h / sh.merge) * (grid_w / sh.merge);
}

// The bytes of Image::pixels for an image of that many tokens and frames.
inline uint64_t image_bytes(const ImageShape &sh, uint64_t tokens, uint32_t frames) {
  return tokens * sh.merge * sh.merge * sh.patch_bytes * frames;
}

// An image in a slot's prompt (docs/spec/MODEL_VISION.md): its block of (grid_h / merge) (grid_w / merge) tokens
// starts at prompt index first; grid_h and grid_w are its size in patches; frames is 2 for a pair of video frames.
// pixels: its patches in the encoder's order (section 2.2), each frames x patch_bytes bytes of 8-bit RGB, frame,
// channel, row, column; null when only the positions are wanted.
struct Image {
  uint32_t first = 0;
  uint32_t frames = 1, grid_h = 0, grid_w = 0;
  const uint8_t *pixels = nullptr;
};

struct StepStats {
  uint32_t misses = 0;          // routed experts the CPU computed
  uint32_t layers_missed = 0;   // layers with at least one miss
};

struct NgramStats {
  uint64_t rows = 0;    // rows asked for
  uint64_t hits = 0;    // served by the cache
  uint64_t reads = 0;   // read from the SSD
};

// The calls the C ABI's code makes on an open model. They run on one thread at a time; "between steps" means no step,
// prompt chunk or draft pass is running.
class Model {
 public:
  virtual ~Model() = default;

  // Zeroes a slot's state, before a new sequence, and forgets its images.
  virtual void reset_slot(uint32_t slot) = 0;

  // The images of the prompt a slot reads, in order, their blocks apart. The pixels are read when the first prompt
  // chunk that holds part of an image's block runs, so they must live until then. Throws for an image it refuses.
  virtual void set_images(uint32_t slot, const Image *images, uint32_t n) = 0;
  // The most tokens an image may have: 0 when the model reads no images.
  virtual uint32_t image_tokens_max() const = 0;
  virtual ImageShape image_shape() const = 0;

  // One step of n rows (1 to STEP_MAX_ROWS) of several sequences, or consecutive positions of one, in order. Returns
  // each row's largest logits, highest first. window: one run of positions a sequence, whose state waits for commit to
  // keep the run's first rows (the accepted drafts and the row before them); the rest's KV is overwritten later.
  // prompt: the rows are prompt tokens, counted for adaptive swaps as the prompt path counts them.
  virtual std::vector<std::vector<Candidate>> step(const StepIn *rows, uint32_t n, bool window = false,
                                                   bool prompt = false) = 0;
  // Whether step takes these rows now. A step of up to STEP_MAX_RUN rows always fits; a wider one may need more of the
  // cards' memory than the model holds for it.
  virtual bool step_fits(const StepIn *rows, uint32_t n) const {
    (void)rows;
    return n > 0 && n <= STEP_MAX_ROWS;
  }
  // Return before the cards finish; the next call on a slot's state waits for them. commit: one run, the window's
  // first n rows. commit_runs: n runs of the window, each its own sequence's.
  virtual void commit(uint32_t slot, uint32_t pos0, uint32_t n) = 0;
  virtual void commit_runs(const CommitRun *runs, uint32_t n) = 0;

  // Drafting: drafting() says whether the model drafts at all, dflash() whether by blocks (df_pass) rather than one
  // draft a pass (the mtp_ calls). draft_max_default is the drafts a round makes at most when engine_config.draft_max
  // does not say; draft_max_cap the most it may say.
  virtual bool drafting() const = 0;
  virtual bool dflash() const = 0;
  virtual uint32_t draft_max_default() const = 0;
  virtual uint32_t draft_max_cap() const = 0;

  // The draft layer: rows r = (slot, position i, the token at i + 1), the hidden input the main model's final residual
  // of the last step's row r (row from[r] with from). Every row writes its cell; only the rows in keep run the rest of
  // the layer, and it returns their drafts (for i + 2) in keep's order.
  virtual std::vector<std::vector<Candidate>> mtp_catchup(const StepIn *rows, uint32_t n,
                                                          const std::vector<uint32_t> &keep,
                                                          const uint32_t *from = nullptr) = 0;
  // One more draft for such a row, the hidden input the draft layer's own residual of row src_row of its last pass
  // (whose cell was i - 1). A draft from keep[j] reads src_row j.
  virtual std::vector<Candidate> mtp_draft(uint32_t slot, uint32_t pos, int32_t token, uint32_t src_row) = 0;
  // mtp_draft for rows of distinct sequences in one pass, row r's input row src_rows[r] of the last pass.
  virtual std::vector<std::vector<Candidate>> mtp_drafts(const StepIn *rows, uint32_t n, const uint32_t *src_rows) = 0;
  // mtp_catchup over the n rows before row end of the last prompt chunk (a run's last rows), from the prompt path's
  // residuals.
  virtual std::vector<std::vector<Candidate>> mtp_catchup_prompt(const StepIn *rows, uint32_t n, uint32_t end,
                                                                 const std::vector<uint32_t> &keep) = 0;

  // The block drafter's block: n rows from position pos of a slot (its token there, then the drafter's own) that read
  // the slot's drafter state from the position draft_start set.
  struct DraftBlock {
    uint32_t slot, pos;
    int32_t token;
    uint32_t n;
  };
  // Context rows first, row r = (slot, position) from the last decode step's row r into the slot's drafter state; then
  // the blocks, of distinct slots, at most STEP_MAX_ROWS rows in all. Returns each block's candidates for positions
  // pos + 1 .. pos + n - 1. A pass may have no context rows or no blocks.
  virtual std::vector<std::vector<std::vector<Candidate>>> df_pass(const StepIn *ctx, uint32_t nctx,
                                                                   const DraftBlock *blocks, uint32_t nblocks) = 0;
  // df_pass's context rows only, from the n rows before row end of the last prompt chunk (a run's last rows).
  virtual void df_context_prompt(const StepIn *rows, uint32_t n, uint32_t end) = 0;
  // The slot's first position the drafter reads; earlier ones were never computed.
  virtual void draft_start(uint32_t slot, uint32_t first) = 0;

  // One sequence's run of a prompt chunk: n tokens from position pos0; p1, p2 and p3 are the three tokens before
  // pos0, p1 the nearest (kNoToken when absent).
  struct PromptRun {
    uint32_t slot, pos0;
    const int32_t *tokens;
    uint32_t n;
    int32_t p1, p2, p3 = kNoToken;
  };
  // Runs of 1 to PROMPT_MAX_RUNS sequences, each of them once, in one pass, at most prompt_chunk() rows. Returns each
  // run's last token's largest logits. next: with one run, the tokens of the chunk expected to follow (next_n of them),
  // or null; the model may read ahead for them.
  virtual std::vector<std::vector<Candidate>> prompt_runs(const PromptRun *runs, uint32_t nruns,
                                                          const int32_t *next = nullptr, uint32_t next_n = 0) = 0;
  // The most rows a prompt chunk takes; 0: no prompt path, one token a step.
  virtual uint32_t prompt_chunk() const = 0;
  // The most rows a chunk may have and fetch only the uncached experts its router picks; a longer one copies every
  // uncached expert. 0 when the cards cache every expert, and every chunk is staged one way.
  virtual uint32_t fetch_rows() const = 0;
  // What a decode token is worth in prompt rows when the scheduler sizes a prompt piece beside writers.
  virtual uint32_t decode_weight() const = 0;

  // A slot's fixed state as one block: what the KV pages do not hold. Its size, and copies to and from host memory
  // (the block must be host memory every card reaches: alloc_pinned).
  virtual uint64_t slot_state_bytes() = 0;
  virtual void save_slot(uint32_t slot, uint8_t *dst) = 0;
  virtual void load_slot(uint32_t slot, const uint8_t *src) = 0;
  virtual void *alloc_pinned(size_t bytes) = 0;
  virtual void free_pinned(void *p) = 0;

  // Sleep (ADR-0032): sleep frees all the model holds on the cards and keeps the runtime, kernels, host buffers and
  // arena. wake loads the cards again, and throws, leaving the model asleep, when they no longer fit. Slots are empty
  // after a wake. Neither runs during a step, and while asleep no other call may run.
  virtual void sleep() = 0;
  virtual void wake() = 0;

  // The KV cache's pages (include/engine_api.h): the pool holds kv_pages() pages of KV_PAGE_CELLS cells, and a
  // slot's row of the page table has pages_per_slot() entries. set_pages maps n logical pages of a slot from first;
  // copy_cells copies the first cells cells (fewer than a page) of page from into page to. Between steps.
  virtual uint32_t kv_pages() const = 0;
  virtual uint32_t pages_per_slot() const = 0;
  virtual void set_pages(uint32_t slot, uint32_t first, const uint32_t *pages, uint32_t n) = 0;
  virtual void copy_cells(uint32_t from, uint32_t to, uint32_t cells) = 0;
  // The pool's pages from in_use up hold nothing to read again: kv_need backs every page below in_use, and the
  // pages streamed slots read (resident) in VRAM; kv_ahead readies the next pages behind the steps; kvs_drop takes a
  // slot's first `pages` logical pages out of the resident ones. Between steps.
  virtual void kv_need(uint32_t in_use, uint32_t resident = UINT32_MAX) = 0;
  virtual void kv_ahead() = 0;
  virtual void kvs_drop(uint32_t slot, uint32_t pages) = 0;

  // Adaptive swaps of the cached experts: note_prompt_routing adds the last prompt chunk's routing to the counts, or
  // with add false drops it;
  // swap_round makes up to max_swaps swaps where the counts say so and returns the swaps made (decay false: the
  // counts have not changed since the last round). swaps_ready: a round would not wait for the last one's work.
  // swaps_settled: the last round chose nothing for the next. Between steps.
  virtual void note_prompt_routing(bool add) = 0;
  virtual uint32_t swap_round(uint32_t max_swaps = 64, bool decay = true) = 0;
  virtual bool swaps_ready() const = 0;
  virtual bool swaps_settled() const = 0;

  virtual const LoadInfo &info() const = 0;
  virtual const StepStats &last_stats() const = 0;
  // Experts cached per layer now, the mean over the layers, and the count the load keeps room for whatever the KV
  // pool holds. Between steps.
  virtual uint32_t experts_cached() const = 0;
  virtual uint32_t experts_floor() const = 0;
  // The n-gram table's counters; zeros without one. Between steps.
  virtual NgramStats ngram_stats() const = 0;
};

// Opens the model file at path (its first shard) as its general.architecture says. Throws for an architecture the
// engine does not run, naming it.
std::unique_ptr<Model> open_model(const std::string &path, const Options &opt);

}  // namespace engine
