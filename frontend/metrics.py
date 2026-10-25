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
    `busy`: the engine thread's time in plans over the time since open; `swap_share` the swap rounds' part of it.
    `prompt_rows_per_s` and `window_rows_per_s`: prompt rows through the prompt path and as decode windows.
    `ngram_hit_rate`: the n-gram table's rows served from its cache."""
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
    ratio("busy", count("busy_ns"), count("up_ns"))
    ratio("swap_share", count("swap_ns"), count("busy_ns"))
    ratio("prompt_rows_per_s", count("prompt_rows"), count("prompt_ns"), 1e9)
    ratio("window_rows_per_s", count("window_rows"), count("window_ns"), 1e9)
    ratio("ngram_hit_rate", count("ngram_hits"), count("ngram_rows"))
    return out
