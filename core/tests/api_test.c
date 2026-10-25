/* The C ABI end to end, from C: chat-hashmap's greedy tokens against llama.cpp's (reference/). Also a second request
 * after release, a stop token, short pieces as decode windows, a longer prompt by the prompt path, two requests in one
 * step, checkpoints restored into a slot that shares the first slot's pages, one token after a restore, the timing
 * records, drafting, adaptive swaps, the pool and the cache sharing memory, a sleep and a wake. Each slot maps its own
 * pages unless it shares. api_test <first shard> */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "engine_api.h"

#define CHECK(cond, ...)                                    \
  do {                                                      \
    if (!(cond)) {                                          \
      fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
      fprintf(stderr, __VA_ARGS__);                         \
      fprintf(stderr, "\n");                                \
      exit(1);                                              \
    }                                                       \
  } while (0)

/* chat-hashmap from reference/prompts.jsonl, and the first 32 of its greedy tokens. */
static const int32_t prompt[] = {
    248045, 846, 198, 814, 20139, 1204, 264, 5010, 2336, 13081, 45776, 13, 17227, 71159, 321, 1724, 26153, 11, 321,
    1910, 948, 1754, 369, 279, 2577, 5576, 13, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271};
static const int32_t expect[] = {32,   5010, 2336, 318,  269,   5010,  1898, 8,     13640, 6679, 310,
                                 2663, 1608, 264,  5010, 709,   13,    64446, 11,   1396,  1328,  13640,
                                 310,  264,  4752, 1857, 303,   449,   16045, 1287, 13,    4213};

/* chat-arith from reference/prompts.jsonl, 108 tokens, and the first 16 of llama.cpp's greedy tokens for it: a
 * prompt longer than the 64 tokens the engine reads as decode windows (engine_api.cpp, kWindowPrompt). */
static const int32_t prompt_long[] = {
    248045, 8678, 198, 24342, 286, 4879, 369, 716, 310, 830, 11553, 13, 5044, 1683, 15060, 1472, 279, 3274, 11,
    9307, 1328, 30800, 11, 2814, 47675, 25605, 11, 321, 60445, 55404, 11, 27224, 11, 321, 30246, 303, 279, 1534,
    4087, 13, 248046, 198, 248045, 846, 198, 32, 5257, 10583, 264, 7803, 506, 220, 15, 24, 25, 19, 15, 321, 33060,
    220, 16, 20, 21, 12754, 506, 264, 23271, 220, 22, 17, 12754, 7294, 13, 1049, 1179, 45369, 220, 16, 17, 4276,
    321, 33060, 2361, 220, 24, 15, 12754, 506, 220, 21, 15, 12754, 7294, 13, 2332, 1092, 854, 1503, 424, 16821, 30,
    248046, 198, 248045, 74455, 198, 248068, 198};
static const int32_t expect_long[] = {
    1596, 1144, 4087, 4145, 33633, 13, 14235, 12204, 13, 25291, 10583, 220, 15, 24, 25, 19};
static const uint32_t nl = sizeof prompt_long / sizeof prompt_long[0];

static uint32_t npp;      /* pages per slot */
static uint32_t kv_free;  /* every plan's kv_free: the pool's top pages the test holds nothing in */

/* Submits a plan of page changes only, and waits for it. */
static void submit_pages(engine_ctx *e, const engine_page_change *changes, uint32_t n) {
  engine_step_plan plan;
  memset(&plan, 0, sizeof plan);
  plan.npages = n;
  plan.pages = changes;
  plan.kv_free = kv_free;
  CHECK(engine_submit(e, &plan) == ENGINE_OK, "submit pages: %s", engine_last_error(e));
  for (int step_done = 0; !step_done;) {
    CHECK(engine_wait(e, 60000) == ENGINE_OK, "no event within 60 s");
    engine_event ev[16];
    uint32_t got = 0;
    CHECK(engine_poll(e, ev, 16, &got) == ENGINE_OK, "poll");
    for (uint32_t i = 0; i < got; i++) {
      CHECK(ev[i].kind != ENGINE_EVENT_ERROR, "engine error: %s", engine_last_error(e));
      if (ev[i].kind == ENGINE_EVENT_STEP) step_done = 1;
    }
  }
}

/* Maps slot's logical pages from first to the end of its row to the slot's own run: page slot npp + i. */
static void own_pages(engine_ctx *e, uint32_t slot, uint32_t first) {
  engine_page_change *c = calloc(npp, sizeof *c);
  CHECK(c != NULL, "calloc");
  for (uint32_t i = first; i < npp; i++) c[i - first] = (engine_page_change){slot, i, slot * npp + i, 0, 0};
  submit_pages(e, c, npp - first);
  free(c);
}

static engine_status admit(engine_ctx *e, uint32_t slot, const engine_settings *s, const int32_t *p, uint32_t n) {
  engine_status st = engine_admit(e, slot, s, p, n);
  if (st == ENGINE_OK) own_pages(e, slot, 0);
  return st;
}

/* Runs slot to its end; with chunk, the prompt goes through the prompt path first, in pieces of that many tokens. */
static int run_with(engine_ctx *e, uint32_t slot, uint32_t chunk, int32_t *out, uint32_t cap) {
  uint32_t got = 0, done = 0, prompt_done = chunk == 0;
  while (!done) {
    engine_step_plan plan;
    memset(&plan, 0, sizeof plan);
    plan.kv_free = kv_free;
    if (!prompt_done) {
      plan.npieces = 1;
      plan.pieces[0].slot = slot;
      plan.pieces[0].tokens = chunk;
    } else {
      plan.nslots = 1;
      plan.slots[0] = slot;
    }
    CHECK(engine_submit(e, &plan) == ENGINE_OK, "submit");
    for (int step_done = 0; !step_done;) {
      CHECK(engine_wait(e, 60000) == ENGINE_OK, "no event within 60 s");
      engine_event ev[16];
      uint32_t n = 0;
      CHECK(engine_poll(e, ev, 16, &n) == ENGINE_OK, "poll");
      for (uint32_t i = 0; i < n; i++) {
        CHECK(ev[i].kind != ENGINE_EVENT_ERROR, "engine error: %s", engine_last_error(e));
        if (ev[i].kind == ENGINE_EVENT_TOKEN && got < cap) out[got++] = ev[i].token;
        if (ev[i].kind == ENGINE_EVENT_TOKEN) prompt_done = 1;
        if (ev[i].kind == ENGINE_EVENT_DONE) done = 1;
        if (ev[i].kind == ENGINE_EVENT_STEP) step_done = 1;
      }
    }
  }
  return (int)got;
}

static int run(engine_ctx *e, uint32_t slot, const engine_settings *s, int32_t *out, uint32_t cap) {
  (void)s;
  return run_with(e, slot, 0, out, cap);
}

/* Two slots decoding together, one step for both, each to its end. */
static void run_two(engine_ctx *e, int32_t *out0, int32_t *out1, int *n0, int *n1) {
  int done0 = 0, done1 = 0;
  *n0 = *n1 = 0;
  while (!done0 || !done1) {
    engine_step_plan plan;
    memset(&plan, 0, sizeof plan);
    if (!done0) plan.slots[plan.nslots++] = 0;
    if (!done1) plan.slots[plan.nslots++] = 1;
    CHECK(engine_submit(e, &plan) == ENGINE_OK, "submit");
    for (int step_done = 0; !step_done;) {
      CHECK(engine_wait(e, 60000) == ENGINE_OK, "no event within 60 s");
      engine_event ev[16];
      uint32_t n = 0;
      CHECK(engine_poll(e, ev, 16, &n) == ENGINE_OK, "poll");
      for (uint32_t i = 0; i < n; i++) {
        CHECK(ev[i].kind != ENGINE_EVENT_ERROR, "engine error: %s", engine_last_error(e));
        if (ev[i].kind == ENGINE_EVENT_TOKEN && ev[i].slot == 0 && *n0 < 64) out0[(*n0)++] = ev[i].token;
        if (ev[i].kind == ENGINE_EVENT_TOKEN && ev[i].slot == 1 && *n1 < 64) out1[(*n1)++] = ev[i].token;
        if (ev[i].kind == ENGINE_EVENT_DONE) *(ev[i].slot ? &done1 : &done0) = 1;
        if (ev[i].kind == ENGINE_EVENT_STEP) step_done = 1;
      }
    }
  }
}

/* Whether no token of out[0..n) is among the last_n tokens before it, the prompt's included. */
static int no_repeat(const int32_t *p, uint32_t np, const int32_t *out, int n, uint32_t last_n) {
  for (int i = 0; i < n; i++)
    for (uint32_t back = 1; back <= last_n && back <= np + (uint32_t)i; back++) {
      int32_t before = back <= (uint32_t)i ? out[i - (int)back] : p[np + (uint32_t)i - back];
      if (before == out[i]) return 0;
    }
  return 1;
}

int main(int argc, char **argv) {
  CHECK(argc == 2, "usage: api_test <first shard>");
  FILE *f = fopen(argv[1], "rb");
  if (!f) {
    printf("SKIP: model not found\n");
    return 77;
  }
  fclose(f);
  engine_config cfg;
  memset(&cfg, 0, sizeof cfg);
  cfg.model_path = argv[1];
  cfg.max_ctx = 1024;
  cfg.slots = 2;
  if (getenv("FLASH_GROUPS")) cfg.groups = (uint32_t)atoi(getenv("FLASH_GROUPS"));  /* the cards in groups */
  engine_info info;
  engine_ctx *e = NULL;
  engine_status st = engine_open(&cfg, &info, &e);
  CHECK(st == ENGINE_OK, "open: %s", engine_last_error(NULL));
  CHECK(info.api_version == ENGINE_API_VERSION && info.vocab == 248320, "info");
  printf("open: %u card(s) in %u group(s), %u experts per layer, %.1f s\n", info.cards, info.groups,
         info.experts_per_layer, info.load_seconds);
  CHECK(info.groups == (cfg.groups ? cfg.groups : 1), "groups: %u", info.groups);
  npp = info.pages_per_slot;
  CHECK(info.kv_pages == cfg.slots * npp && npp * 4 >= cfg.max_ctx, "pages: %u, %u a slot", info.kv_pages, npp);

  engine_settings s;
  memset(&s, 0, sizeof s);
  s.max_tokens = 32;
  uint32_t np = sizeof prompt / sizeof prompt[0];
  int32_t out[64];
  CHECK(admit(e, 0, &s, prompt, np) == ENGINE_OK, "admit");
  CHECK(engine_admit(e, 0, &s, prompt, np) == ENGINE_BUSY, "a used slot must be refused");
  int n = run(e, 0, &s, out, 64);
  CHECK(n == 32, "%d tokens, expected 32", n);
  /* llama.cpp's token 19 is a near tie that rounding can flip, so only the tokens before it are checked. This run's
   * 32 tokens are the reference for the cases below, which check that the paths agree with each other. */
  for (int i = 0; i < 19; i++) CHECK(out[i] == expect[i], "token %d: %d, expected %d", i, out[i], expect[i]);
  int32_t ref[32];
  memcpy(ref, out, sizeof ref);
  printf("32 greedy tokens, the first 19 equal llama.cpp's\n");

  /* The same slot again, ending at a stop token: the 5th expected token. */
  CHECK(engine_release(e, 0) == ENGINE_OK, "release");
  s.n_stop = 1;
  s.stop[0] = expect[4];
  CHECK(admit(e, 0, &s, prompt, np) == ENGINE_OK, "admit again");
  n = run(e, 0, &s, out, 64);
  CHECK(n == 5 && out[4] == expect[4], "stop token: %d tokens", n);
  for (int i = 0; i < 5; i++) CHECK(out[i] == expect[i], "after release, token %d: %d, expected %d", i, out[i], expect[i]);
  printf("a second request in the slot gives the same tokens and stops at the stop token\n");

  /* The prompt in pieces of 16. A piece of up to 64 tokens is read as decode windows of forced rows, which round as
   * decoding does; the first 16 tokens are checked. */
  CHECK(engine_release(e, 0) == ENGINE_OK, "release");
  s.n_stop = 0;
  s.max_tokens = 16;
  CHECK(admit(e, 0, &s, prompt, np) == ENGINE_OK, "admit for the windows");
  n = run_with(e, 0, 16, out, 64);
  CHECK(n == 16, "windows: %d tokens", n);
  for (int i = 0; i < 16; i++) CHECK(out[i] == expect[i], "windows, token %d: %d, expected %d", i, out[i], expect[i]);
  printf("the prompt in pieces of 16, read as decode windows: the first 16 greedy tokens equal llama.cpp's\n");

  /* A longer prompt, in a piece of 80 by the prompt path and one of 28 as windows. The prompt path's numbers differ a
   * little from one token per step (fp16 products), so the first 16 tokens are checked. */
  CHECK(engine_release(e, 0) == ENGINE_OK, "release");
  CHECK(admit(e, 0, &s, prompt_long, nl) == ENGINE_OK, "admit for the prompt path");
  n = run_with(e, 0, 80, out, 64);
  CHECK(n == 16, "prompt path: %d tokens", n);
  for (int i = 0; i < 16; i++)
    CHECK(out[i] == expect_long[i], "prompt path, token %d: %d, expected %d", i, out[i], expect_long[i]);
  printf("a 108-token prompt by the prompt path (80) and as windows (28): the first 16 greedy tokens equal llama.cpp's\n");

  /* Two requests in one step each. */
  CHECK(engine_release(e, 0) == ENGINE_OK, "release");
  s.max_tokens = 32;
  CHECK(admit(e, 0, &s, prompt, np) == ENGINE_OK, "admit slot 0");
  CHECK(admit(e, 1, &s, prompt, np) == ENGINE_OK, "admit slot 1");
  int32_t out1[64];
  int n1;
  run_two(e, out, out1, &n, &n1);
  CHECK(n == 32 && n1 == 32, "two slots: %d and %d tokens", n, n1);
  for (int i = 0; i < 32; i++) CHECK(out[i] == ref[i] && out1[i] == ref[i], "two slots, token %d", i);
  printf("two requests decoding in one step: both give the 32 tokens\n");

  /* A checkpoint of slot 0 after 22 prompt positions, restored into slot 1 for the same prompt: slot 1 maps slot 0's
   * first 5 pages, the 6th copied (its first 2 cells), then its own. */
  CHECK(engine_release(e, 0) == ENGINE_OK && engine_release(e, 1) == ENGINE_OK, "release both");
  CHECK(admit(e, 0, &s, prompt, np) == ENGINE_OK, "admit slot 0");
  for (int i = 0; i < 22; i++) {
    engine_step_plan plan;
    memset(&plan, 0, sizeof plan);
    plan.nslots = 1;
    CHECK(engine_submit(e, &plan) == ENGINE_OK, "submit");
  }
  CHECK(engine_checkpoint(e, 0, 0) == ENGINE_OK, "checkpoint: %s", engine_last_error(e));
  CHECK(engine_admit(e, 1, &s, prompt, np) == ENGINE_OK, "admit slot 1");
  CHECK(engine_restore(e, 1, 0) == ENGINE_OK, "restore: %s", engine_last_error(e));
  {
    engine_page_change c[6];
    for (uint32_t i = 0; i < 5; i++) c[i] = (engine_page_change){1, i, i, 0, 0};
    c[5] = (engine_page_change){1, 5, npp + 5, 5, 2};
    submit_pages(e, c, 6);
    own_pages(e, 1, 6);
    engine_page_change bad = {1, 9, 0, 0, 0};  /* logical 9 on page 0: the wrong card, on two cards */
    engine_step_plan plan;
    memset(&plan, 0, sizeof plan);
    plan.npages = 1;
    plan.pages = &bad;
    CHECK(info.cards == 1 || engine_submit(e, &plan) == ENGINE_BAD_ARGUMENT, "a page on the wrong card");
  }
  n = run(e, 1, &s, out, 64);
  CHECK(n == 32, "restored: %d tokens", n);
  for (int i = 0; i < 32; i++) CHECK(out[i] == ref[i], "restored, token %d: %d, expected %d", i, out[i], ref[i]);
  CHECK(engine_drop(e, 0) == ENGINE_OK, "drop");
  printf("a checkpoint taken after 22 positions, restored into another slot sharing its pages: the 32 tokens\n");

  /* A checkpoint of the whole prompt but its last token (35 positions), restored, and that token read as a prompt
   * piece of one, a decode window of one row: slot 1 maps slot 0's first 8 pages, the 9th copied (its first 3
   * cells), then its own. */
  CHECK(engine_release(e, 0) == ENGINE_OK && engine_release(e, 1) == ENGINE_OK, "release both");
  CHECK(admit(e, 0, &s, prompt, np) == ENGINE_OK, "admit slot 0");
  for (uint32_t i = 0; i + 1 < np; i++) {
    engine_step_plan plan;
    memset(&plan, 0, sizeof plan);
    plan.nslots = 1;
    CHECK(engine_submit(e, &plan) == ENGINE_OK, "submit");
  }
  CHECK(engine_checkpoint(e, 0, 0) == ENGINE_OK, "checkpoint: %s", engine_last_error(e));
  CHECK(engine_admit(e, 1, &s, prompt, np) == ENGINE_OK, "admit slot 1");
  CHECK(engine_restore(e, 1, 0) == ENGINE_OK, "restore: %s", engine_last_error(e));
  {
    engine_page_change c[9];
    for (uint32_t i = 0; i < 8; i++) c[i] = (engine_page_change){1, i, i, 0, 0};
    c[8] = (engine_page_change){1, 8, npp + 8, 8, 3};
    submit_pages(e, c, 9);
    own_pages(e, 1, 9);
  }
  n = run_with(e, 1, 1, out, 64);
  CHECK(n == 32, "one token after a restore: %d tokens", n);
  CHECK(out[0] == expect[0], "one token after a restore: %d, expected llama.cpp's %d", out[0], expect[0]);
  for (int i = 0; i < 32; i++) CHECK(out[i] == ref[i], "one token after a restore, token %d: %d, expected %d", i, out[i], ref[i]);
  CHECK(engine_drop(e, 0) == ENGINE_OK, "drop");
  printf("a checkpoint of the prompt but its last token, that token read as a window of one: the 32 tokens\n");

  engine_timing tr[256];
  uint32_t nt = 0;
  CHECK(engine_read_timing(e, tr, 256, &nt) == ENGINE_OK && nt > 0, "timing records");
  engine_stats cs;
  CHECK(engine_read_stats(e, &cs) == ENGINE_OK, "stats");
  CHECK(cs.plans > 0 && cs.steps > 0 && cs.step_rows >= cs.steps && cs.busy_ns > 0, "stats: %llu plans, %llu steps",
        (unsigned long long)cs.plans, (unsigned long long)cs.steps);
  CHECK(cs.prompt_chunks > 0 && cs.window_pieces > 0 && cs.prompt_rows + cs.window_rows >= nl,
        "stats: %llu chunks of %llu rows, %llu windowed pieces of %llu rows", (unsigned long long)cs.prompt_chunks,
        (unsigned long long)cs.prompt_rows, (unsigned long long)cs.window_pieces, (unsigned long long)cs.window_rows);
  CHECK(cs.restores > 0 && cs.checkpoints_taken > 0, "stats: checkpoints");
  CHECK(cs.rounds == 0 && cs.undrafted_steps == 0, "stats: rounds without drafting");
  CHECK(cs.experts_cached == info.experts_per_layer && cs.experts_floor <= cs.experts_cached,
        "stats: %llu experts cached (floor %llu), info says %u", (unsigned long long)cs.experts_cached,
        (unsigned long long)cs.experts_floor, info.experts_per_layer);
  engine_slot_stats sst;
  CHECK(engine_read_slot_stats(e, info.slots, &sst) == ENGINE_BAD_ARGUMENT, "slot stats of a slot out of range");
  for (uint32_t i = 0; i < nt; i++) CHECK(tr[i].end_ns >= tr[i].start_ns && tr[i].rows > 0, "timing record %u", i);
  printf("%u timing records\n", nt);

  /* A sleep and a wake (ADR-0032). Asleep, nothing is admitted or run; awake, the cards cache the experts they
   * cached before, so the same 32 tokens, and the longer prompt's first 16 by the prompt path. */
  engine_step_plan none;
  memset(&none, 0, sizeof none);
  CHECK(engine_sleep(e) == ENGINE_BUSY, "a sleep with slots admitted must be refused");
  CHECK(engine_release(e, 0) == ENGINE_OK && engine_release(e, 1) == ENGINE_OK, "release both");
  CHECK(engine_sleep(e) == ENGINE_OK, "sleep: %s", engine_last_error(e));
  CHECK(engine_sleep(e) == ENGINE_OK, "a second sleep");
  CHECK(engine_admit(e, 0, &s, prompt, np) == ENGINE_ERROR, "an admit while asleep must fail");
  CHECK(engine_submit(e, &none) == ENGINE_ERROR, "a plan while asleep must fail");
  CHECK(engine_wake(e, &info) == ENGINE_OK, "wake: %s", engine_last_error(e));
  CHECK(info.device_bytes[0] > 0 && info.checkpoints == 2, "info after the wake");
  printf("wake: %.1f s\n", info.load_seconds);
  CHECK(admit(e, 0, &s, prompt, np) == ENGINE_OK, "admit after the wake");
  n = run(e, 0, &s, out, 64);
  CHECK(n == 32, "after the wake: %d tokens", n);
  for (int i = 0; i < 32; i++) CHECK(out[i] == ref[i], "after the wake, token %d: %d, expected %d", i, out[i], ref[i]);
  CHECK(engine_release(e, 0) == ENGINE_OK, "release");
  s.max_tokens = 16;
  CHECK(admit(e, 0, &s, prompt_long, nl) == ENGINE_OK, "admit for the prompt path");
  n = run_with(e, 0, 80, out, 64);
  CHECK(n == 16, "prompt path after the wake: %d tokens", n);
  for (int i = 0; i < 16; i++) CHECK(out[i] == expect_long[i], "prompt path after the wake, token %d", i);
  CHECK(engine_release(e, 0) == ENGINE_OK, "release");
  printf("after a sleep and a wake: the 32 tokens, and the prompt path's 16 of the longer prompt\n");

  /* A presence penalty that no counted token survives: greedy then repeats no token of the last 64, prompt included,
   * where the plain tokens repeat one (5010). */
  s.max_tokens = 32;
  s.penalty_last_n = 64;
  s.penalty_repeat = 1.0f;
  s.penalty_present = 1000.0f;
  CHECK(admit(e, 0, &s, prompt, np) == ENGINE_OK, "admit with a penalty");
  n = run(e, 0, &s, out, 64);
  CHECK(n == 32, "penalised: %d tokens", n);
  CHECK(no_repeat(prompt, np, out, n, 64), "penalised: a token of the last 64 came again");
  CHECK(memcmp(out, ref, sizeof ref) != 0, "penalised: the plain tokens");
  CHECK(engine_release(e, 0) == ENGINE_OK, "release");
  s.penalty_last_n = 0;
  s.penalty_present = 0.0f;
  printf("a presence penalty: 32 tokens, none among the 64 before it\n");

  engine_abi_row rows[256];
  uint32_t nr = engine_abi_layout(rows, 256);
  CHECK(nr > 0 && nr <= 256, "layout rows");
  engine_close(e);

  /* With drafting, exactly the tokens of plain decoding: drafts change how many tokens a step gives, never which.
   * The draft layer's memory leaves fewer experts cached, so the CPU rounds others and token 19 may flip: only the
   * first 16 are checked against llama.cpp's. Two slots draft in one window, and give the same tokens; batch_test
   * checks that a window's rows are those of plain steps. */
  if (getenv("FLASH_MTP_DIR") || access("/models/mtp/mtp-manifest.json", R_OK) == 0) {
    cfg.mtp_dir = getenv("FLASH_MTP_DIR") ? getenv("FLASH_MTP_DIR") : "/models/mtp";
    CHECK(engine_open(&cfg, &info, &e) == ENGINE_OK, "open with drafting: %s", engine_last_error(NULL));
    CHECK(info.drafting == 1, "drafting not on");
    s.max_tokens = 32;
    s.n_stop = 0;
    int32_t plain[64];
    CHECK(admit(e, 0, &s, prompt, np) == ENGINE_OK && admit(e, 1, &s, prompt, np) == ENGINE_OK, "admit");
    run_two(e, plain, out1, &n, &n1);
    CHECK(n == 32 && n1 == 32, "two slots drafting: %d and %d tokens", n, n1);
    for (int i = 0; i < 16; i++)
      CHECK(plain[i] == expect[i], "two slots, token %d: %d, expected %d", i, plain[i], expect[i]);
    for (int i = 0; i < 32; i++) CHECK(out1[i] == plain[i], "two slots, token %d: %d and %d", i, plain[i], out1[i]);
    uint32_t rounds2 = 0, nt3 = 0;
    CHECK(engine_read_timing(e, tr, 256, &nt3) == ENGINE_OK, "timing");
    for (uint32_t i = 0; i < nt3; i++) rounds2 += tr[i].kind == ENGINE_TIMING_ROUND;
    CHECK(rounds2 > 0, "no draft round ran for two slots");
    CHECK(engine_release(e, 0) == ENGINE_OK && engine_release(e, 1) == ENGINE_OK, "release both");
    CHECK(admit(e, 0, &s, prompt, np) == ENGINE_OK, "admit");
    n = run(e, 0, &s, out, 64);
    CHECK(n == 32, "drafting: %d tokens", n);
    for (int i = 0; i < 32; i++)
      CHECK(out[i] == plain[i], "drafting, token %d: %d, with two slots %d", i, out[i], plain[i]);
    CHECK(engine_read_slot_stats(e, 0, &sst) == ENGINE_OK, "slot stats");
    CHECK(sst.rounds > 0 && sst.rounds <= sst.steps && sst.accepted <= sst.drafts && sst.steps + sst.accepted + 1 >= 32,
          "slot stats: %llu steps, %llu rounds, %llu of %llu drafts accepted", (unsigned long long)sst.steps,
          (unsigned long long)sst.rounds, (unsigned long long)sst.accepted, (unsigned long long)sst.drafts);
    CHECK(sst.lookup_accepted <= sst.lookup_drafts && sst.lookup_drafts <= sst.drafts, "slot stats: lookup");
    CHECK(engine_read_stats(e, &cs) == ENGINE_OK, "stats");
    CHECK(cs.rounds > 0 && cs.layer_runs + cs.lookup_runs >= cs.rounds && cs.round_ns > 0, "stats: %llu rounds",
          (unsigned long long)cs.rounds);
    CHECK(cs.layer_accepted <= cs.layer_drafts && cs.lookup_accepted <= cs.lookup_drafts, "stats: accepted");
    CHECK(cs.layer_ns + cs.lookup_ns == cs.round_ns, "stats: the rounds' time by source");
    printf("drafting: %llu rounds, %llu of %llu layer drafts and %llu of %llu lookup drafts accepted\n",
           (unsigned long long)cs.rounds, (unsigned long long)cs.layer_accepted, (unsigned long long)cs.layer_drafts,
           (unsigned long long)cs.lookup_accepted, (unsigned long long)cs.lookup_drafts);
    CHECK(engine_release(e, 0) == ENGINE_OK, "release");
    /* With penalties, each window row and draft counts the rows and drafts before it: the same tokens again. */
    s.penalty_last_n = 64;
    s.penalty_repeat = 1.3f;
    s.penalty_freq = 0.5f;
    CHECK(admit(e, 0, &s, prompt, np) == ENGINE_OK && admit(e, 1, &s, prompt, np) == ENGINE_OK, "admit");
    run_two(e, plain, out1, &n, &n1);
    CHECK(n == 32 && n1 == 32, "penalised, plain: %d and %d tokens", n, n1);
    CHECK(engine_release(e, 0) == ENGINE_OK && engine_release(e, 1) == ENGINE_OK, "release both");
    CHECK(admit(e, 0, &s, prompt, np) == ENGINE_OK, "admit");
    n = run(e, 0, &s, out, 64);
    CHECK(n == 32, "penalised, drafting: %d tokens", n);
    for (int i = 0; i < 32; i++)
      CHECK(out[i] == plain[i], "penalised, drafting, token %d: %d, plainly %d", i, out[i], plain[i]);
    CHECK(engine_release(e, 0) == ENGINE_OK, "release");
    s.penalty_last_n = 0;
    s.penalty_repeat = 1.0f;
    s.penalty_freq = 0.0f;
    CHECK(admit(e, 0, &s, prompt, np) == ENGINE_OK, "admit");
    n = run_with(e, 0, 64, out, 64);
    CHECK(n == 32, "drafting after windows: %d tokens", n);
    for (int i = 0; i < 16; i++) CHECK(out[i] == expect[i], "drafting after windows, token %d", i);
    CHECK(engine_release(e, 0) == ENGINE_OK, "release");
    CHECK(admit(e, 0, &s, prompt_long, nl) == ENGINE_OK, "admit");
    n = run_with(e, 0, 80, out, 64);
    CHECK(n == 32, "drafting after the prompt path: %d tokens", n);
    for (int i = 0; i < 16; i++) CHECK(out[i] == expect_long[i], "drafting after the prompt path, token %d", i);
    uint32_t rounds = 0, nt2 = 0;
    CHECK(engine_read_timing(e, tr, 256, &nt2) == ENGINE_OK, "timing");
    for (uint32_t i = 0; i < nt2; i++) rounds += tr[i].kind == ENGINE_TIMING_ROUND;
    CHECK(rounds > 0, "no draft round ran");
    printf("drafting: the 32 tokens of two slots drafting together, %u and %u draft rounds\n", rounds2, rounds);
    engine_close(e);
  }

  /* With adaptive swaps. Swaps change which experts the CPU computes, so their rounding: the first 16 greedy tokens
   * are checked, through three requests. */
  cfg.mtp_dir = NULL;
  cfg.flags |= ENGINE_CONFIG_SWAPS;
  CHECK(engine_open(&cfg, &info, &e) == ENGINE_OK, "open with swaps: %s", engine_last_error(NULL));
  s.max_tokens = 48;
  for (int r = 0; r < 3; r++) {
    CHECK(admit(e, 0, &s, prompt, np) == ENGINE_OK, "admit");
    n = r == 1 ? run(e, 0, &s, out, 64) : run_with(e, 0, 16, out, 64);
    CHECK(n == 48, "swaps: %d tokens", n);
    for (int i = 0; i < 16; i++) CHECK(out[i] == expect[i], "swaps, request %d, token %d: %d, expected %d", r, i, out[i], expect[i]);
    CHECK(engine_release(e, 0) == ENGINE_OK, "release");
  }
  printf("adaptive swaps: three requests, the first 16 tokens each\n");
  /* After swaps, a wake caches the experts the swaps chose. */
  CHECK(engine_sleep(e) == ENGINE_OK, "sleep: %s", engine_last_error(e));
  CHECK(engine_wake(e, &info) == ENGINE_OK, "wake: %s", engine_last_error(e));
  CHECK(admit(e, 0, &s, prompt, np) == ENGINE_OK, "admit");
  n = run_with(e, 0, 16, out, 64);
  CHECK(n == 48, "swaps after the wake: %d tokens", n);
  for (int i = 0; i < 16; i++) CHECK(out[i] == expect[i], "swaps after the wake, token %d", i);
  CHECK(engine_release(e, 0) == ENGINE_OK, "release");
  printf("adaptive swaps, after a sleep and a wake: the first 16 tokens\n");
  engine_close(e);

  /* The KV pool and the cache sharing memory, on a pool larger than its fixed part: slot 0's pages, then slot 1's
   * too, take memory from the cache; with slot 1's free again, the rounds give it back. */
  cfg.max_ctx = 65536;
  cfg.flags |= ENGINE_CONFIG_SHARE_KV;
  CHECK(engine_open(&cfg, &info, &e) == ENGINE_OK, "open sharing: %s", engine_last_error(NULL));
  npp = info.pages_per_slot;
  uint32_t loaded = info.experts_per_layer;
  for (int r = 0; r < 4; r++) {
    /* Request 2 through the prompt path, after the cache changed. */
    const uint32_t slot = r == 1 ? 1 : 0;
    const int32_t *p = r == 2 ? prompt_long : prompt, *want = r == 2 ? expect_long : expect;
    kv_free = info.kv_pages - (r == 1 ? 2 : 1) * npp;
    CHECK(admit(e, slot, &s, p, r == 2 ? nl : np) == ENGINE_OK, "admit");
    n = run_with(e, slot, r == 2 ? 80 : 0, out, 64);
    CHECK(n == 48, "sharing: %d tokens", n);
    for (int i = 0; i < 16; i++) CHECK(out[i] == want[i], "sharing, request %d, token %d", r, i);
    CHECK(engine_release(e, slot) == ENGINE_OK, "release");
  }
  CHECK(engine_sleep(e) == ENGINE_OK, "sleep: %s", engine_last_error(e));
  CHECK(engine_wake(e, &info) == ENGINE_OK, "wake: %s", engine_last_error(e));
  CHECK(admit(e, 0, &s, prompt_long, nl) == ENGINE_OK, "admit");
  n = run_with(e, 0, 80, out, 64);
  CHECK(n == 48, "sharing after the wake: %d tokens", n);
  for (int i = 0; i < 16; i++) CHECK(out[i] == expect_long[i], "sharing after the wake, token %d", i);
  printf("sharing: %u experts a layer at the load, %u after the wake; four requests and one after the wake\n", loaded,
         info.experts_per_layer);
  engine_close(e);
  printf("PASS\n");
  return 0;
}
