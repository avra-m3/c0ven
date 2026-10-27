# ADR-0051: checkpoints stay spread back from each conversation's end

**Status:** draft, 2026-10-28; not accepted.
**Revisit when:** a client is seen resuming from a point that the count rule dropped while an older checkpoint of
the same conversation stayed, or room, not the count, becomes what drops most checkpoints.

## Context

- The engine process keeps 8 checkpoints by default, each about 118 MB of host memory
  ([CACHING.md](../CACHING.md#what-load-does-to-checkpoints)). A request takes one where its answer starts, one every
  16K tokens and one at the end of a long system prompt ([ADR-0017](0017-scheduling-several-requests.md), item 6).
- A new checkpoint past the count pushed out the least recently used. A conversation that has just moved on still
  holds the checkpoint it started from, which is newer than a paused conversation's only checkpoint, so the paused
  one went first. Agents run several sessions and subagents at once, and each comes back after the others have
  taken turns.
- Least recent use also keeps only the newest points of a conversation. After a few short turns they all sit in the
  last few thousand tokens, and a client that rewrites its history further back (an agent that prunes old tool outputs
  more than 40K tokens back) matches none of them and reads the whole conversation again.
- Strata drops checkpoints by the gap they leave, as an option (`--prompt-cache-spread`, Strata commit
  [a7d4b754](https://github.com/mach10x/strata/commit/a7d4b75443ff249cad776fb44a831ea87002b7ea)), for one
  conversation's chain of checkpoints.

## Decision

1. A checkpoint records its parent: the longest other checkpoint whose tokens are a strict prefix of its own. Adding
   and removing checkpoints keeps the links.
2. A new checkpoint past the count pushes out, of the checkpoints that another extends (the new one counted), the
   one whose removal leaves the smallest gap for its distance from the end: (the nearest checkpoint after it − its
   parent) / (the longest checkpoint after it − its parent), its parent's length 0 when it has none. A tie goes to
   the least recently used. A kept checkpoint and one being restored from are not chosen.
3. When no checkpoint qualifies, and when a request needs room, the least recently used goes, kept ones last, as
   before.

## Why

Replayed through `engine/seqstate/` on 37,881 requests of 1,022 agent sessions, 8 checkpoints read
248.4 M prompt tokens again with least recent use and 100.0 M with this rule, fewer than least recent use with 64
checkpoints (103.0 M). Of the requests that continued their conversation, 4,273 did not resume where their last
turn started with least recent use, and 216 with this rule. On one conversation of 200 turns of 1,000 tokens, a
rewrite 42,000 tokens back reads 47,000 tokens again instead of 200,000
([measurements](../measurements/MEASUREMENTS.md#2026-10-28-checkpoints-spread)).

Room stays with least recent use because a checkpoint that a longer one extends shares all its pages with that one:
dropping it frees no room.

## Options not chosen

- Least recent use among the checkpoints another extends: the same 103 M on the replay as this rule without the new
  checkpoint counted, but on one long conversation it keeps the last seven turns and reads 200,000 tokens again for a
  rewrite 42,000 back.
- Strata's rule as written, which never drops the first checkpoint of a chain: 117.8 M on the replay. A chain's first
  checkpoint here is often a turn's start, not the system prompt, which `kept` already protects.
- More checkpoints: 16 with least recent use read 180.8 M, and each costs 118 MB of pinned host memory.

## Consequences

- Adding a checkpoint compares its tokens with each other checkpoint's, a check of the last token first: at most
  5.5 ms with eight related checkpoints of 150-246K tokens, 0.4 ms for unrelated ones (*estimate*, timed once on a
  laptop).
- With more conversations taking turns than checkpoints, every newest checkpoint is a leaf and least recent use
  still drops the one about to be used.
