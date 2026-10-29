# ADR-0056: the aim is the fastest performance on Battlemage

**Status:** draft, 2026-10-29; not accepted. Would supersede [ADR-0019](0019-performance-target.md), item 1 of
[ADR-0006](0006-scope.md) in part (one machine), and item 2 of ADR-0006 (one model family, from ISTA-DASLab's files).
**Revisit when:** another engine runs this model faster than C0ven on the same Battlemage cards with the same file;
Qwen releases the Qwen4 models; the test machine gains cards; Intel ships a card generation after Battlemage.

## Context

- [ADR-0019](0019-performance-target.md) set a target for one request on both B60s with the IQ3_S file: 30 tokens/s
  decode without drafting and 2,000 tokens/s prompt reading, at 4K. The engine passed both: decode of doc-4k-qa at
  17.255 ms a token (58 tokens/s) without drafting ([record](../measurements/MEASUREMENTS.md#2026-10-15-sub-bytes-words)),
  and its prompt at 2,135 tokens/s ([record](../measurements/MEASUREMENTS.md#2026-10-15-prompt-router)). A target
  that is met no longer says which work comes next.
- On the same two B60s and the same file, the README's figures give C0ven 2,072 tokens/s prompt reading and 102
  tokens/s decode at 4k, against llama.cpp's 357 and 17
  ([C0ven](../measurements/MEASUREMENTS.md#2026-10-27-readme-benchmarks),
  [llama.cpp](../measurements/MEASUREMENTS.md#2026-10-18-against-llama-cpp)).
- [ADR-0006](0006-scope.md) item 1 names one machine. The engine already runs on one card as well as two
  ([record](../measurements/MEASUREMENTS.md#2026-10-19-one-card)), and more than two cards are planned
  ([#26](https://github.com/avra-m3/c0ven/issues/26)).
- The engine's output on a file matches llama.cpp's on the same file: mean KL 0.158 against 0.160 from the BF16
  checkpoint on IQ3_S ([record](../measurements/MEASUREMENTS.md#2026-10-29-kl-iq3s-llamacpp)).
- ADR-0006 item 2 limits the engine to ISTA-DASLab's files, and [ADR-0038](0038-the-other-quants-load.md) needed an
  ADR to add that repository's other sizes. Other publishers' quants of the same model use the same types, or a few
  more, and the engine reads each tensor by its type.

## Decision

1. **The project's aim is the fastest inference of the models it supports (item 5) on Intel Battlemage cards.** No
   fixed target bounds the work: a change is worth making while it makes the engine measurably faster.
2. **Faster is judged as [ADR-0016](0016-how-a-change-is-judged.md) has it,** by ABBA on fresh processes, on the
   test machine. The README compares C0ven with the other engines that run this model on the same cards and file.
3. **Speed does not cost quality.** A change keeps `kl_test`'s mean KL within its error of the code before it
   ([ADR-0054](0054-the-bf16-checkpoint-is-the-end-to-end-reference.md)). The engine does not cut precision below
   what llama.cpp computes on the same file.
4. **Any number of Battlemage cards in one Linux machine with the `xe` driver is in scope.** The two B60s stay the
   test machine. Kernels are written for Battlemage's units; there is still no portability work for other GPUs.
5. **One architecture is built today: `qwen4_exp`,** Qwen3.8-Flash-Next (`qwen4exp` in a GGUF's metadata). Support
   for the Qwen4 models is planned for their release.
6. **Any quant of a built architecture may load, from any publisher, without an ADR of its own.** A pull request adds
   the file's tensor table, the kernels for any type the engine does not yet read, and the file's `kl_test` and speed
   figures. Every speed figure names its file.
7. **Benchmarks and tests stay on IQ3_S** until the primary target changes or the test machine gains cards. IQ3_S
   stays the default, and the README's figures, the references and the tests' gates are taken with it.

## Options not chosen

| Option | Why not |
| --- | --- |
| A new fixed target | a target is met and then stops guiding the work, as ADR-0019's was |
| Trade quality for speed (approximate kernels, lower-precision sums) | users choose a file for its quality; the engine should give each file's quality at its best speed |

## Consequences

- Plans are ranked by the speed they are expected to gain on Battlemage, with the measurement that would show it.
- ADR-0019's stop rule no longer applies. Its numbers stay as the record of the first target.
- ADR-0006's other items stand: the full 262,144-token context, several requests at once, Docker behind a
  model-swapping proxy.
- [ADR-0024](0024-repository-layout.md) keeps model code in `models/<arch>/`, so Qwen4 gets its own directory beside
  `qwen4exp`.
- The README's comparison with other engines is kept current: it is the evidence for item 1.
