// The step plan: what the cards compute in one step. The host writes it into host memory, and the kernels read it
// where it lies (docs/DESIGN.md, sections 2 and 3). Fixed size: a recorded step list reads the same address every
// step.
#ifndef SHARED_STEP_H
#define SHARED_STEP_H

#include "types.h"

#define STEP_MAX_ROWS 8

// One row: one token of one sequence.
typedef struct {
  u32 slot;   // the sequence's slot: which state and KV it reads and writes
  u32 pos;    // the token's position in its sequence
  u32 token;  // the token id
  u32 flags;  // STEP_ROW_*
} StepRow;

#define STEP_ROW_LOGITS 1u  // the row's logits are wanted

typedef struct {
  u32 nrows;
  u32 step;    // counts steps, for the timing records
  u32 window;  // 1: the rows are windows of consecutive positions, one run of rows per sequence, and the recurrent
               // state is not committed; a commit (CommitPlan) then writes the accepted rows' state
  u32 pad;
  StepRow rows[STEP_MAX_ROWS];
} StepPlan;

// A pass of the DFlash 2 drafter (docs/MODEL_FILES.md, "The DFlash 2 drafter"). A context pass's row r takes the
// features of the last decode step's row r into the slot's ring at pos. A block pass's rows are blocks, one a slot,
// each the rows row0 .. row0 + n - 1 at positions pos0 .. pos0 + n - 1; a row attends to its slot's ring from
// position lo through its block's last.
typedef struct {
  u32 slot;
  u32 pos;
  u32 row0;  // the block's first row in the pass (a block pass)
  u32 n;     // the block's rows (a block pass)
  u32 lo;    // the first position the row attends to (a block pass)
  u32 pad[3];
} DraftRow;

typedef struct {
  u32 nrows;
  u32 pad[3];
  DraftRow rows[STEP_MAX_ROWS];
} DraftPlan;

SHARED_ASSERT(sizeof(DraftRow) == 32, "DraftRow layout");
SHARED_ASSERT(sizeof(DraftPlan) == 16 + 32 * STEP_MAX_ROWS, "DraftPlan layout");

// After a window: in each of nruns runs, the n rows from window row row0 (the accepted ones) commit their recurrent
// state, from position pos0 of slot.
typedef struct {
  u32 n;
  u32 pos0;
  u32 slot;
  u32 row0;
} CommitRun;

typedef struct {
  u32 nruns;
  u32 pad[3];
  CommitRun runs[STEP_MAX_ROWS];
} CommitPlan;

SHARED_ASSERT(sizeof(CommitRun) == 16, "CommitRun layout");
SHARED_ASSERT(sizeof(CommitPlan) == 16 + 16 * STEP_MAX_ROWS, "CommitPlan layout");

// A prompt chunk, computed in one pass (the prompt path): runs of up to PROMPT_MAX_RUNS sequences, one after
// another in its rows. A run is n consecutive tokens of one sequence from position pos0, in rows row0 .. row0 + n - 1.
// The row-wise kernels read the chunk's plan (row0 0, n all its rows), the kernels that follow a sequence one run's.
// The recorded chunk lists read the plans where they lie, so one list serves any n up to the size it was recorded for.
#define PROMPT_MAX_RUNS 4
// A run of at most this many rows (a decode row and its drafts) attends from the 8-bit cache, as the decode step
// does, over the cells the indexer chose; longer runs convert their sequence to fp16 and attend on the XMX units.
#define PROMPT_SHORT_RUN 8

typedef struct {
  u32 n;
  u32 pos0;
  u32 slot;
  u32 row0;
} PromptPlan;

SHARED_ASSERT(sizeof(PromptPlan) == 16, "PromptPlan layout");

SHARED_ASSERT(sizeof(StepRow) == 16, "StepRow layout");
SHARED_ASSERT(sizeof(StepPlan) == 16 + 16 * STEP_MAX_ROWS, "StepPlan layout");

// The KV cache's page, for every model: a row of the page table maps a sequence's cells this many at a time
// (include/engine_api.h, docs/DESIGN.md section 4).
#define KV_PAGE_CELLS 4

#endif
