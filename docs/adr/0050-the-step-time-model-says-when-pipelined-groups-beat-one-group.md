# ADR-0050: the step-time model says when pipelined groups beat one group; the core keeps one group

**Status:** draft, 2026-10-26.
**Revisit when:** a machine serves more requests at once than the model's break-even, or has more than two cards.

## Context

- Two ways to use n cards: one group of every card, each layer split over them ([ADR-0023](0023-two-cards-tensor-split.md)),
  or several groups, each holding a range of the layers, with requests pipelined across the groups. The layout is
  fixed at load: moving weights reads them from the file again.
- `engine/step_time.py` costs a layout `groups(g, w)` from the model's products and measured terms
  ([DESIGN.md](../DESIGN.md#the-step-time-model)). Its steps match the measured ones within 15% for one group and
  for two groups run in turn ([measured](../measurements/MEASUREMENTS.md#2026-10-26-groups-in-turn)).
- A probe that pipelines two groups of one card measured a token's time at 2 to 8 requests on both layouts
  ([measured](../measurements/MEASUREMENTS.md#2026-10-26-pipelined-groups)).
- Much of a step does not shrink with the width: F, the router, the norms, the DeltaNet recurrence and the
  hyper-connections, runs whole on every card of a group.

## Decision

1. The model gives each group's part of a step: the weight reads of its layers (the last group's with the head),
   its share of F, the reduces and M by its layers, and a hand-off unless it is last.
2. `token_time` gives a token's time with a number of requests writing. With several groups and at least as many
   requests, the slowest group's part sets the pace. This is the formula for when pipelined groups beat one group.
3. A step costs the cards what a step of its list's rows costs: the core records lists for 1, 2, 4 and 8 rows, and
   a step runs the smallest of at least its rows.
4. At load the engine process logs every allowed layout's time a token at 1, 2, 4 and 8 requests. `auto` takes the
   layout the core runs with the least time for one request.
5. The core runs one group of every card. It does not pipeline groups.

## Why

On two cards the model puts the break-even between 4 and 6 requests writing, and the probe at 6, where the two
layouts' runs overlap. The model is within 15% of the probe at every point. Both make one group faster at 4
requests, by 9% in the model and 14% in the probe. One request on two groups takes about a third longer than on one
group, whatever the pipelining, and the deployed engine serves 4 requests at once. Code to pipeline the groups would
not be chosen here.

Without decision 3, the model's times at 6 requests were 8-16% below the measured ones: a 6-row step runs the 8-row
list, and a pipelined set of 3 rows the 4-row list.

## Options not chosen

- Building the pipelined groups now: a step a group in flight, two sets of step buffers and lists, both groups'
  misses answered at once, drafting set aside; for a layout `auto` would not take on two cards.
- Choosing the layout from the rows of the last life's steps: the layout's cost at one request decides latency,
  and with one layout the core runs there was nothing to choose between.

## Consequences

- The pipelining rule assumes the CPU answers both groups' misses as fast as one's, and the host's work for two
  steps in flight costs no more than for one. The probe, one host thread serving both groups, holds it within 15%.
- Groups in the core wait for a machine where the model puts the break-even inside the load it serves.
