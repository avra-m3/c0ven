# ADR-0076: An admission evicts only checkpoints that free pages

**Status:** draft. 2026-11-10.
**Revisit when:** a replay of served requests shows a rule that weighs the tokens a checkpoint saves against the
pages it frees reads fewer prompt tokens than least recently used.

## Context

- [ADR-0068](0068-kv-pages-are-reserved-as-a-sequence-grows.md) item 3 evicts, for a slot's growth, only
  checkpoints that hold the last reference to a page of the class that is short. `plan_admission` kept the older
  rule: the least recently used checkpoint of all, until the request's pages can be reserved.
- A checkpoint holds one reference to each page of its prefix. A slot holds the pages of its conversation, and a
  longer checkpoint of the same conversation holds the shorter one's pages too. Evicting a checkpoint whose pages
  are all held elsewhere frees nothing.
- A conversation's checkpoints are last used when its turn is admitted. A turn that writes for minutes has the
  least recently used checkpoints by the time another admission needs pages, and its slot holds every page of
  them. The older rule evicted them first and freed nothing. When that turn ended its pages were freed, and its
  next turn read the conversation again.
- On the served engine a 228,249-token prompt that shared no prefix was admitted into a full pool. The older rule
  evicted 24 checkpoints for it, and a conversation of 99,104 tokens that was writing read 104,154 tokens on its
  next turn ([the record](../measurements/MEASUREMENTS.md#2026-11-10-admission-evictions)).

## Decision

1. An admission makes room as a slot's growth does: it evicts only checkpoints that hold the last reference to a
   page of a class still short, least recently used first, kept ones last, never the one it resumes from or one a
   slot is restoring from.
2. When no checkpoint holds a last reference and the pool is still short, a checkpoint goes whose last page is held
   only by checkpoints that can be evicted. Two checkpoints that end on the same page free it only together.
3. Only the last page of each class is checked. A checkpoint's pages are a prefix of its conversation's, and a slot
   or checkpoint that holds one of them holds those before it, so the references never rise along the prefix.

## Why

Replaying the 695 requests of that session through `engine/seqstate/`, the older rule gives every request its
logged cached length. With this rule the admission evicts 19 checkpoints, the writing conversation resumes at
99,099 tokens, and the session reads 1,442,259 prompt tokens in place of 1,526,665
([the record](../measurements/MEASUREMENTS.md#2026-11-10-admission-evictions)).

A conversation that waits between turns now loses its end first, since only its longest checkpoint holds pages of
its own, and its shorter checkpoints still serve its next turn.

Checking every page of every checkpoint costs time on the scheduler's thread at each admission into a full pool:
32 checkpoints of a 200,000-token conversation hold 1.6 million references. Item 3 makes the check a few pages a
checkpoint.

## Options not chosen

- **Evict the checkpoint that frees the most pages.** It takes the longest conversation first, which costs the
  most to read again.
- **Refuse an admission that cannot fit even with every such checkpoint evicted, and evict nothing.** It needs the
  count of pages no slot holds. It remains to be done, as ADR-0068's consequences say.

## Consequences

- `SeqState._make_room` serves admission and growth, and `SeqState._victims` gives the checkpoints an eviction may
  take.
- A conversation that waits can still lose its checkpoints to a long prompt admitted while it waits: in the
  replay a 79,170-token conversation did, under both rules.
- `tests/host/test_seqstate.py` checks item 3 against a scan of every page over random admissions, checkpoints
  and releases.
