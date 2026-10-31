/* The engine core's C ABI: the one boundary between the Python control plane and the C++ core (ADR-0022,
 * docs/DESIGN.md section 3). Rules:
 *   - C++ never calls Python: no callbacks.
 *   - Only the plain structs below and buffers cross; every call returns an engine_status; no exception crosses.
 *   - C++ owns all memory and threads; Python owns every decision.
 * engine/api/abi.py mirrors this file by hand; engine_abi_layout() lets a test check every size and offset.
 * Symbols carry the prefix engine_ (ADR-0024, item 2). */
#ifndef ENGINE_API_H
#define ENGINE_API_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(ENGINE_API_EXPORT)
#define ENGINE_API __attribute__((visibility("default")))
#else
#define ENGINE_API
#endif

#define ENGINE_API_VERSION 10

typedef enum {
  ENGINE_OK = 0,
  ENGINE_ERROR = 1,         /* see engine_last_error */
  ENGINE_BAD_ARGUMENT = 2,
  ENGINE_BUSY = 3,          /* the slot is in use, the plan queue is full, or a slot is admitted at a sleep */
  ENGINE_TIMEOUT = 4
} engine_status;

typedef struct engine_ctx engine_ctx;

/* What open loads and allocates. Zero means the default given here. */
typedef struct {
  const char *model_path;      /* the first shard of the GGUF */
  uint32_t slots;              /* sequences resident at once; default 1, at most 8 decode together */
  uint32_t max_ctx;            /* positions per sequence, a multiple of 4; default 32768 */
  uint32_t max_cards;          /* 1 or 2; default 2 */
  int32_t experts_per_layer;   /* experts cached on the cards per layer; negative or 0 with the flag: as many as fit */
  uint32_t cpu_threads;        /* the CPU's expert workers; default 16 */
  uint32_t flags;              /* ENGINE_CONFIG_* */
  uint64_t reserve_bytes;      /* device memory left free per card; default 512 MiB */
  uint32_t prompt_chunk;       /* tokens per prompt pass at most, a multiple of 64; default 4352; with the flag, 0
                                  turns the prompt path off */
  uint32_t checkpoints;        /* slot checkpoints held in host memory, each a slot's fixed state; default 2 */
  const char *routing_profile; /* the profile that chooses the cached experts; null: the engine's own */
  const char *mtp_dir;         /* the draft (MTP) layer's tensors; null: no drafting */
  uint32_t draft_max;          /* drafts per round at most, 1-7; default 3 */
  float draft_min_p;           /* a draft is the request's sampler's pick from the draft layer's candidates; drafting
                                  stops at one whose probability under the sampler is below this; default 0.5 */
  const char *vision_path;     /* the image encoder's file (the mmproj GGUF); null: no images. Needs the prompt path */
  uint32_t kv_pages;           /* the KV cache's pages of 4 cells, on each card, rounded up to a multiple of the cards;
                                  default: slots x the context's */
  uint32_t kv_resident_cells;  /* K and V cells per slot kept in VRAM, the rest in host memory (ADR-0008 item 3);
                                  0: every cell */
} engine_config;

#define ENGINE_CONFIG_EXPERTS_SET 1u  /* experts_per_layer is meaningful, 0 included */
#define ENGINE_CONFIG_PROMPT_SET 2u   /* prompt_chunk is meaningful, 0 included */
#define ENGINE_CONFIG_SWAPS 4u        /* adaptive swaps: the cached experts follow the routing, a round once the CPU's
                                         misses have cost as much time as the last round took, 4 to 16 plans apart (4
                                         before both are known; 16 while several slots write); changes which
                                         experts the CPU computes, so their rounding */
#define ENGINE_CONFIG_SHARE_KV 8u     /* with SWAPS and the experts not set: the KV pages the plans leave free (kv_free)
                                         lend their memory to the expert cache, so it holds more experts while the
                                         pool holds less */

/* What open reports. */
typedef struct {
  uint32_t api_version;
  uint32_t cards;
  uint32_t slots;
  uint32_t max_ctx;
  uint32_t experts_per_layer;
  uint32_t vocab;
  uint64_t device_bytes[2];
  uint64_t arena_bytes;        /* host memory holding the experts the cards do not cache */
  double load_seconds;
  uint32_t prompt_chunk;       /* 0: prompts go one token per step */
  uint32_t checkpoints;
  uint32_t max_step_rows;      /* decode rows per step at most */
  uint32_t drafting;           /* 1: a plan whose decode slots all write runs a round of drafts verified in a window */
  uint32_t image_tokens_max;   /* the most tokens an image may have; 0: no images */
  uint32_t kv_pages;           /* the KV cache's pages */
  uint32_t pages_per_slot;     /* logical pages a slot can map: the context's, rounded up to a multiple of the cards */
} engine_info;

/* A request's sampling and stopping, copied once by admit. The core samples in llama.cpp's order: penalties, top-k,
 * top-p, min-p, temperature (docs/spec/MODEL_IO.md, section 2.4). A sampled token depends only on the seed, its
 * position and the tokens before it, never on how the work was split into steps. */
#define ENGINE_MAX_STOP 8
typedef struct {
  float temperature;           /* 0: greedy */
  uint32_t top_k;              /* 0 or above the engine's candidates (64): 64 */
  float top_p;                 /* 0 or 1: off */
  uint32_t max_tokens;         /* new tokens at most */
  uint64_t seed;
  uint32_t n_stop;
  int32_t stop[ENGINE_MAX_STOP];  /* token ids that end the request (kept as its last token) */
  float min_p;                 /* keep the tokens at least min_p times as likely as the top one; 0: off */
  uint32_t penalty_last_n;     /* the penalties count the last this many tokens before the one sampled, the prompt's
                                  included and image and video tokens not; 0: no penalties */
  float penalty_repeat;        /* a counted token's logit divided by it (multiplied when <= 0); 1: off */
  float penalty_freq;          /* subtracted from a counted token's logit once per occurrence; 0: off */
  float penalty_present;       /* subtracted from a counted token's logit once; 0: off */
} engine_settings;

/* The KV cache is paged, and the caller keeps the page table (docs/DESIGN.md, section 4). A page is 4 cells: logical
 * page i of a slot holds its positions 4i .. 4i + 3. Each card has kv_pages pages; a page id names the same page on
 * both cards. On two cards logical page i maps only to a page p with p % 2 == i % 2 (the indexer's keys are split
 * by position). A slot's pages are mapped from its logical page 0 up, with no gap; admit unmaps them all. A slot reads
 * and writes only through its mapped pages: one that reaches a position past them waits (ENGINE_EVENT_STALLED) until
 * a plan maps more, and drafts stop short of them. A page may be mapped in several slots, for a shared prefix: a slot
 * writes the cells of the positions it computes, so a page is shared only once its 4 cells are computed. */
#define ENGINE_NO_PAGE 0xffffffffu
typedef struct {
  uint32_t slot;
  uint32_t logical;            /* at most the slot's mapped pages: the next one, or one mapped already */
  uint32_t page;               /* the page it maps to; ENGINE_NO_PAGE: unmap logical and every page after it */
  uint32_t source;             /* with cells: the page to copy from first */
  uint32_t cells;              /* 0-3: copy the first cells cells of source into page, then map it (the partial last
                                  page of a shared prefix); source and page alike modulo the cards */
} engine_page_change;

/* One plan: page changes, applied in order before anything else; some slots' prompt pieces, each of up to 64 tokens
 * with no image as decode windows, a longer one through the prompt path; then one decode step for the decode slots,
 * batched (up to max_step_rows). A slot still reading its prompt may be in decode: it then reads its next prompt token
 * in the step. The scheduler decides the mix and the pieces' sizes (ADR-0017). With drafting, a plan whose decode
 * slots are all writing runs a round: each slot's drafts verified in one window, and up to draft_max + 1 tokens out
 * for a lone slot, 3 for each of several. The slots share the window's max_step_rows rows, so each drafts fewer as
 * there are more of them. Several slots run a round only while no slot has prompt left to read, and not for one's
 * second token. */
#define ENGINE_MAX_STEP_SLOTS 8
#define ENGINE_MAX_PROMPT_PIECES 4
typedef struct {
  uint32_t slot;
  uint32_t tokens;             /* prompt tokens to read now, at most prompt_chunk; 0: as many as remain, up to it */
} engine_prompt_piece;

typedef struct {
  uint32_t nslots;             /* decode slots */
  uint32_t slots[ENGINE_MAX_STEP_SLOTS];
  uint32_t npieces;
  engine_prompt_piece pieces[ENGINE_MAX_PROMPT_PIECES];
  uint32_t npages;
  const engine_page_change *pages;  /* npages changes, copied by submit */
  uint32_t kv_free;            /* the pages at the top of the pool the caller holds nothing in, after the changes:
                                  ids kv_pages - kv_free and up; 0 says none. With ENGINE_CONFIG_SHARE_KV their memory
                                  may serve the expert cache until a plan maps them again */
} engine_step_plan;

typedef enum {
  ENGINE_EVENT_TOKEN = 1,      /* slot sampled token; value: its position */
  ENGINE_EVENT_PROMPT = 2,     /* slot read prompt tokens; value: how many so far */
  ENGINE_EVENT_DONE = 3,       /* slot finished; value: ENGINE_DONE_* */
  ENGINE_EVENT_STEP = 4,       /* a submitted plan finished; value: the plan's sequence number from 1 */
  ENGINE_EVENT_ERROR = 5,      /* value: an engine_status; engine_last_error says more */
  ENGINE_EVENT_STALLED = 6     /* slot waits for a page; value: the position it needs. Once until pages are mapped */
} engine_event_kind;

#define ENGINE_DONE_STOP 1     /* a stop token */
#define ENGINE_DONE_LENGTH 2   /* max_tokens */
#define ENGINE_DONE_CONTEXT 3  /* the slot's context is full */

typedef struct {
  uint32_t kind;
  uint32_t slot;
  int32_t token;
  uint32_t value;
} engine_event;

/* One timing record (docs/DESIGN.md, section 7): a prompt piece or a decode step, as the engine ran it. */
typedef enum {
  ENGINE_TIMING_PROMPT = 1,
  ENGINE_TIMING_DECODE = 2,
  ENGINE_TIMING_ROUND = 3      /* a draft round: rows = the window's rows, misses as decode */
} engine_timing_kind;

typedef struct {
  uint32_t kind;
  uint32_t plan;               /* the plan's sequence number */
  uint32_t rows;               /* prompt tokens, or decode rows */
  uint32_t misses;             /* experts the CPU computed (decode) */
  uint64_t start_ns;           /* steady clock */
  uint64_t end_ns;
} engine_timing;

/* The engine's counters since open (docs/DESIGN.md, section 7): sums over everything the engine thread ran. A reader
 * takes two snapshots and differences them. Times are the steady clock's nanoseconds, as the timing records'. */
typedef struct {
  uint64_t up_ns;              /* since open */
  uint64_t plans;              /* plans run */
  uint64_t busy_ns;            /* the engine thread's time in plans */
  uint64_t prompt_chunks;      /* prompt chunks through the prompt path */
  uint64_t prompt_rows;
  uint64_t prompt_ns;
  uint64_t window_pieces;      /* prompt pieces read as decode windows */
  uint64_t window_rows;
  uint64_t window_ns;
  uint64_t steps;              /* decode steps that verified no drafts */
  uint64_t step_rows;
  uint64_t step_ns;
  uint64_t step_misses;        /* experts the CPU computed in them */
  uint64_t step_layers_missed; /* layers with a miss, summed over them */
  uint64_t undrafted_steps;    /* of the steps, with drafting on, those whose rows were all writing: no slot had a draft */
  uint64_t rounds;             /* draft rounds: windows that verified drafts, of one slot or several */
  uint64_t round_rows;
  uint64_t round_ns;
  uint64_t round_misses;
  uint64_t round_layers_missed;
  uint64_t layer_runs;         /* a slot's run in a round with the draft layer's drafts */
  uint64_t layer_drafts;       /* drafts verified in them */
  uint64_t layer_accepted;
  uint64_t layer_ns;           /* the rounds' time, by the source of a lone slot's drafts; several slots' to the layer */
  uint64_t lookup_runs;        /* a slot's run with drafts from its own tokens (ADR-0041) */
  uint64_t lookup_drafts;
  uint64_t lookup_accepted;
  uint64_t lookup_ns;
  uint64_t swap_rounds;        /* swap rounds run (ENGINE_CONFIG_SWAPS) */
  uint64_t swaps;              /* experts swapped in them */
  uint64_t swap_ns;            /* the engine thread's time in them */
  uint64_t stalls;             /* ENGINE_EVENT_STALLED sent */
  uint64_t checkpoints_taken;
  uint64_t restores;
  uint64_t ngram_rows;         /* the n-gram table's rows asked for (ADR-0015) */
  uint64_t ngram_hits;         /* of them, served by its cache */
  uint64_t ngram_reads;        /* rows read from the SSD */
  uint64_t experts_cached;     /* experts cached per layer at the last plan's end, the mean over the layers */
  uint64_t experts_floor;      /* the fewest the load keeps room for, whatever the KV pool holds (ENGINE_CONFIG_SHARE_KV) */
} engine_stats;

/* One slot's counters since admit: its own decode, not its prompt. */
typedef struct {
  uint64_t steps;              /* decode steps and rounds that fed it one of its own tokens */
  uint64_t rounds;             /* of them, draft rounds */
  uint64_t drafts;             /* drafts verified for it */
  uint64_t accepted;
  uint64_t lookup_rounds;      /* of the rounds, those with drafts from its own tokens */
  uint64_t lookup_drafts;
  uint64_t lookup_accepted;
} engine_slot_stats;

/* Loads the model and allocates everything. Blocks until ready. */
ENGINE_API engine_status engine_open(const engine_config *config, engine_info *info, engine_ctx **out);

/* Copies a request into a free slot: its settings and prompt ids. The slot's state is reset and its pages unmapped;
 * no queued plan may name the slot, in its decode slots, pieces or page changes. ENGINE_BAD_ARGUMENT for min_p outside
 * [0, 1], or a penalty with penalty_repeat <= 0. */
ENGINE_API engine_status engine_admit(engine_ctx *e, uint32_t slot, const engine_settings *settings, const int32_t *prompt,
                           uint32_t n_prompt);

/* An image, or a pair of a video's frames, in an admitted prompt (docs/spec/MODEL_VISION.md). Its block of
 * grid_h * grid_w / 4 tokens starts at prompt index first; the prompt holds the image token there, or the video token
 * for a pair of frames. pixels: grid_h x grid_w patches in the encoder's order (merge windows row-major, each window's
 * four patches row-major), each frames x 768 bytes of 8-bit RGB, frame, channel, row, column. */
typedef struct {
  uint32_t first;
  uint32_t grid_h, grid_w;     /* in patches of 16 pixels, both even */
  uint32_t frames;             /* 1: a still image; 2: a pair of a video's frames (section 6) */
  const uint8_t *pixels;
} engine_image;

/* Gives an admitted slot the images and frame pairs of its prompt, in order, their blocks apart: they set the positions of every
 * prompt token from the first image on, and the encoder turns each into its block's rows when the prompt reaches
 * it. Copies the pixels. Before the slot's first plan, and before restore; at most image_tokens_max tokens an image. */
ENGINE_API engine_status engine_admit_images(engine_ctx *e, uint32_t slot, const engine_image *images, uint32_t n);

/* Frees a slot. A slot named in a plan still queued is skipped. */
ENGINE_API engine_status engine_release(engine_ctx *e, uint32_t slot);

/* Queues a step plan. Does not wait. BAD_ARGUMENT if a page change breaks the rules above, counting the changes of
 * the plans queued before it. */
ENGINE_API engine_status engine_submit(engine_ctx *e, const engine_step_plan *plan);

/* Moves up to cap events into buf; *n receives how many. Does not wait. */
ENGINE_API engine_status engine_poll(engine_ctx *e, engine_event *buf, uint32_t cap, uint32_t *n);

/* Waits until an event is ready or timeout_ms passes (ENGINE_TIMEOUT). */
ENGINE_API engine_status engine_wait(engine_ctx *e, uint32_t timeout_ms);

/* Copies slot's fixed state (its recurrent state: not the KV, which stays in its pages) into checkpoint id
 * (< checkpoints), with its token history, so a later request sharing the prefix restores it instead of reading it
 * again. The caller keeps the pages of the checkpoint's positions. Waits for the plans queued before it. */
ENGINE_API engine_status engine_checkpoint(engine_ctx *e, uint32_t slot, uint32_t id);

/* Copies checkpoint id into slot, which must be admitted: the slot then holds the checkpoint's positions, and its own
 * prompt continues after them (the checkpoint's tokens must be a prefix of the slot's prompt, and the images whose
 * blocks start inside it the same images, pixel for pixel). The caller maps the checkpoint's pages into the slot, the
 * last one copied when the checkpoint ends inside it, in a plan before the slot's next. */
ENGINE_API engine_status engine_restore(engine_ctx *e, uint32_t slot, uint32_t id);

/* Empties checkpoint id. */
ENGINE_API engine_status engine_drop(engine_ctx *e, uint32_t id);

/* Moves up to cap timing records into buf, oldest first; *n receives how many. A full ring drops the oldest. */
ENGINE_API engine_status engine_read_timing(engine_ctx *e, engine_timing *buf, uint32_t cap, uint32_t *n);

/* Copies the counters into *out. Awake or asleep. */
ENGINE_API engine_status engine_read_stats(engine_ctx *e, engine_stats *out);

/* Copies an admitted slot's counters into *out; ENGINE_BAD_ARGUMENT for a slot not admitted. */
ENGINE_API engine_status engine_read_slot_stats(engine_ctx *e, uint32_t slot, engine_slot_stats *out);

/* Sleep (ADR-0032): frees everything the engine holds on the cards, the image encoder's included, and the
 * checkpoints, and keeps the host's copy of the experts the cards do not cache and of the encoder's weights, so a
 * wake reads only the cards' part of the model again. Waits for the queued plans; ENGINE_BUSY while a slot is
 * admitted. While asleep, admit, admit_images, submit, checkpoint and restore fail with ENGINE_ERROR; the checkpoints
 * are empty. Sleeping twice does nothing. */
ENGINE_API engine_status engine_sleep(engine_ctx *e);

/* Loads the cards again, with the experts they cached when the engine slept, and pins the checkpoints. Blocks until
 * ready; info, if not null, receives what open reported, with this load's device bytes and seconds. On failure the
 * engine stays asleep. Waking an engine that is awake only fills info. */
ENGINE_API engine_status engine_wake(engine_ctx *e, engine_info *info);

/* The last error's message, valid until the next call on e (or for open's failures, with e null). */
ENGINE_API const char *engine_last_error(const engine_ctx *e);

/* Frees everything. */
ENGINE_API void engine_close(engine_ctx *e);

/* For the ABI test: one row per field of every struct above, name "struct.field" with its offset and size, then
 * one row per struct, name "struct", with offset 0 and its size. Returns the number of rows; table may be null. */
typedef struct {
  const char *name;
  uint32_t offset;
  uint32_t size;
} engine_abi_row;
ENGINE_API uint32_t engine_abi_layout(engine_abi_row *table, uint32_t cap);

#ifdef __cplusplus
}
#endif

#endif
