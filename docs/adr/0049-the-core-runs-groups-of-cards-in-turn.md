# ADR-0049: the core runs the cards in groups, one group after another

**Status:** draft, 2026-10-26; not accepted.
**Revisit when:** rows from different requests should pipeline across the groups, or the machine has more than two
cards.

## Context

- [ADR-0023](0023-two-cards-tensor-split.md) splits every layer over both cards. It superseded
  [ADR-0009](0009-two-cards.md)'s ranges of layers a card, which this core never ran.
- The step-time model (`engine/step_time.py`, [DESIGN.md](../DESIGN.md#the-step-time-model)) costs every layout
  `groups(g, w)`, and the chooser takes the cheapest the core runs. The core ran `groups(1, n)` only, so the model's
  costs for any other layout could not be checked against a step.
- A layout is fixed at load: moving weights between cards reads them from the file again.

## Decision

1. `engine_config.groups` (C ABI version 11) and `Options::groups` put the cards in that many groups. Each group
   holds an equal range of the layers, in order, and each card of a group its share of them as ADR-0023 splits
   them. The groups divide the cards and the layers; the default is 1.
2. Within a decode step and a prompt chunk the groups run in turn. The first starts from the embedding. After its
   last layer a group copies its residuals into the next group's card of the same rank, card to card, and signals
   an event the next group's list waits on. No rows pipeline across the groups.
3. The first group runs the PLE module. The last runs the final mixer, the vocabulary head and the draft layer.
   A card holds only its own layers' weights, KV pages, DeltaNet state and cached experts.
4. The first card of each group hands its layers' misses to the CPU. The host loop does not change.
5. With [ADR-0045](0045-the-kv-pool-and-the-expert-cache-share-each-cards-memory.md), each group shares its own
   cards' memory between its own layers' KV and caches.
6. The image encoder runs over every card in every layout.
7. The engine process opens the core with the groups `--layout` names, one group for `auto` and `tensor`. The
   chooser takes the layout the core opened with, and the step-time model costs it with no pipelining.

## Options not chosen

- Choosing the groups after the load check, and loading again when `auto` prefers another layout: a second load
  takes tens of seconds, and on two cards the model puts `groups(1, 2)` first at every row count it is tested at.
- Pipelining in the same change: it needs a step a group in flight and a scheduler that feeds them. Running the
  groups in turn first checks the model's other terms alone.
- A hand-off through host memory, as ADR-0009 had it: the residuals go card to card, as a window's exchanges do
  ([ADR-0030](0030-exchanges-into-the-other-card.md)).

## Consequences

- On two cards `--layout layers` (or `groups:2x1`) runs, and the step log sets its prediction against its steps.
  Its one-row step is about a third longer than the tensor split's, and the model predicts it within 4%
  ([measured](../measurements/MEASUREMENTS.md#2026-10-26-groups-in-turn)).
- `auto` keeps the tensor split, and logs the other layouts' costs.
- Pipelining across the groups is a later change; until then `step_time`'s `pipelined` stays false.
