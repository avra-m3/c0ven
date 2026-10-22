# ADR-0017: scheduling several requests - time slices, packed prompts, one producer per shared prefix

**Status:** accepted, 2026-10-09. It makes items 5 and 6 of [ADR-0010](0010-several-requests.md) precise. Item 1's
piece size is superseded in part by [ADR-0044](0044-a-short-prompt-is-read-whole.md) (draft).
**Revisit when:** measuring several sequences at once shows that a count of chunks serves decode as well as a time
slice, or that packing does not save the expert streaming it should.

## Context

- **A prompt chunk's cost is not constant.** Each row's indexer scores every earlier pooled key, so a chunk near the
  end of a 262K prompt costs more than one near its start.
- **A chunk is long against a window.** A 2,048-token chunk takes ~1 s on a B60 (*estimate*), and a decode window
  ~56 ms. "One chunk, then one decode pass" would leave the writing requests ~5% of the time.
- **A chunk's expert streaming costs about the same at any length.** With 10 of 512 experts per token, 200 tokens
  already reach ~98% of a layer's experts (*estimate*, with independent routing), so a short prompt pays nearly a
  full chunk's streaming.
- **Agents started together send the same system prompt at the same moment.** A 30K-token system prompt takes tens of
  seconds.
- **The chat template decides what is shared.** It puts the reasoning-effort sentence first in the system message,
  and with tools the whole tool list before the system prompt's own text.
- **[Splash](https://github.com/incoai/splash) does all of the below** (`runtime/engine/Scheduler.cpp`,
  `DEVELOPMENT.md`):
  - prompt commands aimed at 500 ms while others write;
  - decode owed half of each prompt command's time;
  - up to 4 short prompts packed into one command;
  - one producer for a cold shared prefix;
  - decode at one fixed shape per batch width.

## Decision

1. **Prompt work is scheduled by time, not by chunks.**
   - While requests of the same priority are writing, a prompt command aims at a time budget. Its row budget shrinks
     until the measured cost per row fits.
   - After a prompt command of t ms, decode runs until it has had a set share of t.
   - The budget and the share are start options. With nobody writing, a prompt takes whole chunks.
2. **Short prompts are packed into one chunk,** shortest first, with a bound on how often a prompt may be passed
   over. A prompt alone may end its chunk at its last row.
3. **One producer for a shared prefix.**
   - A request whose prompt starts with a prefix another request is computing waits for that request's checkpoint at
     the end of the shared part.
   - While it waits it holds no slot and no KV.
   - A higher-priority request never waits for a lower one.
4. **Shared prefixes, and the kept system-prompt checkpoint, are keyed by the rendered tokens.** In practice they hold
   per pair of reasoning effort and tool list.
5. **One recorded shape per batch width.**
   - The window size is a function of the batch width, with fewer drafts as the batch grows.
   - Each card keeps one set of recorded command lists per width.
   - A decode pass takes the sequences that are ready, and never waits to fill a width.
6. **Checkpoints are taken where Strata takes them:**
   - at the start of each assistant turn;
   - every 16K prompt tokens;
   - at the end of a system prompt of 2,048 tokens or more, which is kept.

## Options not chosen

| Option | Why not |
| --- | --- |
| A fixed number of prompt chunks per decode pass | a chunk's cost grows with its position and dwarfs a window; the writers' share of time would swing with the prompt's length |
| Short chunks for short prompts, unpacked | each pays nearly a full chunk's expert streaming |
| Checkpoints every 4,096 tokens (Splash) | at ~118 MB each, a 262K prompt would hold ~7.5 GB of them if kept |
| Sharing prefixes by system prompt text alone | the template puts the effort sentence and the tool list first, so equal system prompts often render to different tokens |

## Consequences

- The server reports the prompt's and decode's share of time per slot.
- When prompt and decode work run on different cards, the time slice is a limit per card.
- Recording a set of command lists per batch width has a cost in time and memory, which is measured.
- The checks for several requests include N agents with the same 30K system prompt arriving together, and a short
  request's first token during a 128K prompt.
