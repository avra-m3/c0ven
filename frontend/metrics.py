"""The engine's counters as `/metrics` reports them (DESIGN.md, section 7): the STATS line's values as numbers, and
the ratios read from them. The core counts; this file divides."""

from __future__ import annotations

from collections.abc import Mapping


def numbers(report: Mapping[str, str]) -> dict[str, int | float | str]:
    """Each value as an int or a float where it reads as one."""
    out: dict[str, int | float | str] = {}
    for key, value in report.items():
        try:
            out[key] = int(value)
        except ValueError:
            try:
                out[key] = float(value)
            except ValueError:
                out[key] = value
    return out


def derive(report: Mapping[str, str]) -> dict[str, float]:
    """What the counters say about the engine since it started; a ratio whose divisor is zero is left out.

    `step_ms` and `round_ms`: a decode step without drafts, and a draft round, on average. `misses_per_row`: experts
    the CPU computed per decode row, steps and rounds together. `tokens_per_s` for `step`, `layer` and `lookup`: tokens
    out per second of the cards' time in plain steps, in rounds with the draft layer's drafts and in rounds with drafts
    from the slot's own tokens, each counting the one token a run always gives and the drafts accepted. Side by side
    they say which source pays. `accepted_per_run` for `layer` and `lookup`: drafts accepted each time a slot drafted.
    `undrafted_share`: of the plain steps, those with drafting on where every row wrote and no slot had a draft.
    `reading_round_share`: of the writers' rounds and undrafted steps run while another slot read its prompt, the
    rounds.
    `busy`: the engine thread's time in plans over the time since open; `swap_share` the swap rounds' part of it.
    `prompt_rows_per_s` and `window_rows_per_s`: prompt rows through the prompt path and as decode windows.
    `ngram_hit_rate`: the n-gram table's rows served from its cache.
    `miss_wait_ms_per_step` and `miss_wait_ms_per_round`: the engine thread's time from a layer's records with a miss
    being published to its answer, per plain step and per round; `miss_wait_step_share` and `miss_wait_round_share`
    their part of the steps' and the rounds' time; `miss_wait_ms_per_layer` per layer with a miss, both together.
    `draft_pass_ms`: what a plan's decode did after its step or round, the draft layer's pass with the sampling, on
    average; `draft_pass_share` its part of the engine thread's time in plans.
    `layer_accepted_at_<n>`, n from 1 to 7: of the slot-runs that verified n or more of the draft layer's drafts, those
    whose first n were accepted. `layer_stopped_at_max`, `_share`, `_p` and `_choice`: of the slot-runs with the draft
    layer's drafts, those whose drafting reached `draft_max`, reached the fewer drafts the round's share gave the slot,
    stopped at a draft below `draft_min_p`, or reached the fewer drafts the core chose for a round of several writing
    slots."""
    n = numbers(report)

    def count(key: str) -> float:
        v = n.get(key, 0)
        return float(v) if isinstance(v, (int, float)) else 0.0

    out: dict[str, float] = {}

    def ratio(name: str, a: float, b: float, scale: float = 1.0) -> None:
        if b > 0:
            out[name] = round(a / b * scale, 4)

    ratio("step_ms", count("step_ns"), count("steps"), 1e-6)
    ratio("round_ms", count("round_ns"), count("rounds"), 1e-6)
    ratio("misses_per_row", count("step_misses") + count("round_misses"), count("step_rows") + count("round_rows"))
    ratio(
        "layers_missed_per_step",
        count("step_layers_missed") + count("round_layers_missed"),
        count("steps") + count("rounds"),
    )
    ratio("step_tokens_per_s", count("step_rows"), count("step_ns"), 1e9)
    ratio("layer_tokens_per_s", count("layer_runs") + count("layer_accepted"), count("layer_ns"), 1e9)
    ratio("lookup_tokens_per_s", count("lookup_runs") + count("lookup_accepted"), count("lookup_ns"), 1e9)
    ratio("layer_accepted_per_run", count("layer_accepted"), count("layer_runs"))
    ratio("lookup_accepted_per_run", count("lookup_accepted"), count("lookup_runs"))
    ratio("undrafted_share", count("undrafted_steps"), count("steps"))
    ratio("reading_round_share", count("reading_rounds"), count("reading_rounds") + count("reading_steps"))
    ratio("busy", count("busy_ns"), count("up_ns"))
    ratio("swap_share", count("swap_ns"), count("busy_ns"))
    ratio("prompt_rows_per_s", count("prompt_rows"), count("prompt_ns"), 1e9)
    ratio("window_rows_per_s", count("window_rows"), count("window_ns"), 1e9)
    ratio("ngram_hit_rate", count("ngram_hits"), count("ngram_rows"))
    ratio("miss_wait_ms_per_step", count("step_miss_ns"), count("steps"), 1e-6)
    ratio("miss_wait_ms_per_round", count("round_miss_ns"), count("rounds"), 1e-6)
    ratio("miss_wait_step_share", count("step_miss_ns"), count("step_ns"))
    ratio("miss_wait_round_share", count("round_miss_ns"), count("round_ns"))
    ratio(
        "miss_wait_ms_per_layer",
        count("step_miss_ns") + count("round_miss_ns"),
        count("step_layers_missed") + count("round_layers_missed"),
        1e-6,
    )
    ratio("draft_pass_ms", count("draft_ns"), count("draft_passes"), 1e-6)
    ratio("draft_pass_share", count("draft_ns"), count("busy_ns"))
    for depth in range(1, 8):
        ratio(f"layer_accepted_at_{depth}", count(f"layer_accepted_{depth}"), count(f"layer_drafts_{depth}"))
    for why in ("max", "share", "p", "choice"):
        ratio(f"layer_stopped_at_{why}", count(f"layer_stop_{why}"), count("layer_runs"))
    return out
