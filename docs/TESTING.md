# Testing

This page says how each part of the engine is checked, against what, where, and how long a check may take.
[ADR-0012](adr/0012-correctness-reference.md) fixes the reference: llama.cpp's CPU backend at `3cf0325`, through
the reference outputs ([REFERENCE_OUTPUTS.md](REFERENCE_OUTPUTS.md)). The tools are
[ADR-0004](adr/0004-write-or-reuse.md)'s: `pytest` for every test, `numpy` to compare tensors, and plain executables
under `ctest` for the core's own unit tests.

Level Zero has no CPU device, so every test of a kernel runs on a B60 ([ADR-0001](adr/0001-level-zero-from-scratch.md)).
"The machine" below is the test machine of [MEASUREMENTS.md#the-test-machine](MEASUREMENTS.md#the-test-machine).

## The levels

| Level | What it checks | Reference | Where it runs | Time budget |
| --- | --- | --- | --- | --- |
| 1. Host | Python: the scheduler, admission, sequence-state bookkeeping, the memory plan, the GGUF tensor table, the protocol, the front end against the mock engine ([ADR-0022](adr/0022-two-processes-python-and-cpp.md)). C++: layouts shared across the C ABI, ring buffers, the arena's free list | expected values written in the test; the C ABI's struct sizes and offsets checked from both sides | anywhere, no card | 30 s |
| 2. Wrapper | each function of [ADR-0003](adr/0003-kernel-language.md)'s header of Intel builtins (block reads, DPAS, sub-group shuffles), once | expected results written in the test, as the kernel spike checked them ([record](MEASUREMENTS.md#2026-10-11-kernel-spike)) | the machine | 1 min |
| 3. Kernel | each kernel against reference values for the same inputs: every expert type's decode, the dense products, DeltaNet, attention, the indexer, norms, sampling | ggml's CPU implementation of the same operation (its `to_float` dequantisation and dot products, called from the test through the pinned ggml), or a scalar reference in double precision inside the test. A generator kept in the repository is test code, owned like engine code | the machine | 5 min |
| 4. Layer | a whole block of the model fed fixed inputs; the image encoder is the only one checked | llama.cpp's rows for the same images ([record](MEASUREMENTS.md#2026-10-17-encoder-reference)) | the machine | 10 min |
| 5. End to end | the tokenizer; greedy agreement, teacher-forced and free-running; mean KL and top-1 agreement on the KL text | the reference prompts, greedy tokens and KL base ([REFERENCE_OUTPUTS.md](REFERENCE_OUTPUTS.md)) | the machine | 30 min |
| 6. Determinism | a request's tokens are identical alone, in a window, batched, and packed with other prompts, with [ADR-0016](adr/0016-how-a-change-is-judged.md)'s switches set: no PCIe share of the misses, no adaptive swaps, no prefetching, no prompt cache. Also two runs of the same request, and runs with the CPU's misses on 8 and 16 workers | the engine's own run alone | the machine | 30 min |
| 7. Performance | accepted tokens per window and wall time per window ([ADR-0016](adr/0016-how-a-change-is-judged.md)); decode tokens/s and prompt tokens/s at 4K against [ADR-0019](adr/0019-performance-target.md)'s target and the latest measurements | [MEASUREMENTS.md](MEASUREMENTS.md); before and after alternate (ABBA) on fresh processes, 3 rounds, medians and ranges ([MEASUREMENTS.md#how-the-numbers-were-taken](MEASUREMENTS.md#how-the-numbers-were-taken)) | the machine, in the measurement state of MEASUREMENTS.md#how-the-numbers-were-taken step 1 | ~30 min |

The tests that exist, by level (each is described in [DEVELOPING.md](DEVELOPING.md#tests)):

- Level 1: the Python host tests in `tests/host/`, and `gguf_test` and `mrope_test`. No ring-buffer or free-list test.
- Level 2: `ze_smoke_test`, `topk_test` and `gemm_test`'s DPAS layout check. Not every builtin has a test.
- Level 3: `matvec_test`, `moe_test`, `gemm_test`, `qgemm_test`, `attn_test` and `vision_test`.
- Level 4: `encoder_test`. No test of a text layer exists.
- Level 5: `greedy_test` (teacher-forced), `api_test` and the tokenizer check in `tests/end_to_end/`. No free-running
  or KL check.
- Level 6: `batch_test`, and `api_test`'s drafting and swap cases. The packed, two-run and worker-count cases are not
  built.

Levels 1-3 run on every change. Levels 4-6 run on every change to a kernel, the decode loop, the memory system or the
scheduler. Level 7 runs on every change that claims speed.

Level 7 is valid only after level 6 passes: a faster run that changes tokens is a different computation. So level 7
fails a change that is faster but lowers acceptance. A change fails if its median wall time per window is worse than
before by more than the range of the "before" runs, or if accepted tokens per window fall at all with the switches
set. Its record says which of the two failed.

## Tolerances

The error measure for a tensor is the largest absolute difference divided by the largest absolute reference value,
unless a row says otherwise.

| Path | Tolerance | Why |
| --- | --- | --- |
| Integer work: decoding a quantised block to its integers, int8 dot products before scaling | exact | Integer arithmetic has one right answer |
| FP32 results from quantised or BF16 weights (the decode kernels, with the same activations the reference is given) | 1e-5 | Only the order of the float sums differs from the reference. The kernel spike's kernels met 1.0e-7 to 1.4e-7 for all eight types ([record](MEASUREMENTS.md#2026-10-11-kernel-spike)) |
| FP16 products on the XMX units (the prompt path) | 2e-3 | fp16 carries 11 bits (a rounding of 4.9e-4 per value). The prompt kernels of the design-inputs record, the fp16 ones included, met 1e-7 to 3e-4 ([design-inputs record, part 8](MEASUREMENTS.md#2026-10-13-design-inputs)) |
| BF16 outputs (the image encoder's kernels that write a product's next input, [MODEL_VISION.md](spec/MODEL_VISION.md)) | 1e-2 | bf16 carries 8 bits (a rounding of 3.9e-3 per value), on top of the fp16 error where the inputs are fp16 |
| The DeltaNet recurrence over 2,048 tokens, FP32 state | 1e-5 | Checked against double precision in the same run of the design-inputs record; the state does not grow (its gate is below 1) |
| End to end, the KL text | mean KL at most 0.02, top-1 agreement at least 95% | 0.02 is ADR-0012's gate. The reference against itself gives 99.99% top-1, and 0.035 KL / 91% when only llama.cpp's batch size changes ([REFERENCE_OUTPUTS.md](REFERENCE_OUTPUTS.md#how-stable-the-reference-is)), so the engine feeds the KL text in 512-token pieces for this check. *Provisional*: to be revisited with the first measurement |
| End to end, greedy, teacher-forced | at least 95% of positions, on the prompts of up to 512 tokens | Those are the most reliable references ([REFERENCE_OUTPUTS.md](REFERENCE_OUTPUTS.md#how-stable-the-reference-is)); longer prompts are reported, not gated |

## Test data

| Data | Where | How a test finds it |
| --- | --- | --- |
| Small vectors (a few KB): wrapper expectations, hand-made kernel inputs | committed, beside the tests | relative path |
| The reference prompts and greedy tokens | committed, `reference/` | relative path |
| The KL base (8.1 GB) | not distributed; made as [REFERENCE_OUTPUTS.md](REFERENCE_OUTPUTS.md) says | `FLASH_REF_DIR` |
| The encoder's reference rows | the machine, `$FLASH_REF_DIR/vision` ([record](MEASUREMENTS.md#2026-10-17-encoder-reference)) | `FLASH_REF_DIR` |
| The model | the machine, `/models` in the container ([MODEL_FILES.md](MODEL_FILES.md)) | `FLASH_MODEL_DIR`, defaulting to `/models` |

**A test whose data is missing, or whose sha256 does not match, skips with a message naming the file. It never
passes** ([ADR-0012](adr/0012-correctness-reference.md), decision 5). A run that skipped tests reports how many.

## Where tests run

- Anywhere, without a card: level 1, and compiling every kernel with `ocloc` when the
  container is at hand.
- On a machine with two B60s, in the engine image ([ADR-0025](adr/0025-the-engine-image.md)): levels 2-7.
  GPU tests carry the ctest label `gpu`. A GPU test skips when Level Zero lists no GPU, and fails instead when
  `ENGINE_EXPECT_GPUS` is set, so a run on the cards sets it.
- CI, on GitHub's hosted runners ([ADR-0028](adr/0028-ci-without-a-card.md)), on every push: it builds the image,
  the engine and every kernel, runs the Python checks and level 1, and checks that every GPU test of the core skips
  without a card and fails without one when `ENGINE_EXPECT_GPUS` is set. No self-hosted runner with the cards is set
  up, so levels 2-6 run by hand.

## The rule for every kernel change

A change that adds or changes a kernel:

1. adds its level-3 test, against ggml or a scalar reference, at the shapes the model uses, and with the tolerance of
   its type above;
2. runs the level 4-6 tests that exist and says they passed;
3. if it claims speed, runs level 7 and links its row in [MEASUREMENTS.md](MEASUREMENTS.md);
4. shows in its commit message the commands run, the counts passed and skipped, and the largest error per test.
