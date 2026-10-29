# Testing

How each part of the engine is checked, against what, where, and in how long.

End to end, the reference is Qwen's BF16 checkpoint read by vLLM on the KL set
([ADR-0054](adr/0054-the-bf16-checkpoint-is-the-end-to-end-reference.md)); for layers and kernels it is llama.cpp's CPU
backend at `3cf0325` and ggml ([ADR-0012](adr/0012-correctness-reference.md)). Both are described in the
[reference outputs](measurements/REFERENCE_OUTPUTS.md). Tests use `pytest`, `numpy` to compare tensors, and plain
executables under `ctest` for the core's unit tests ([ADR-0004](adr/0004-write-or-reuse.md)).

Level Zero has no CPU device, so every kernel test needs a B60 ([ADR-0001](adr/0001-level-zero-from-scratch.md)).
"The machine" means [the test machine](measurements/MEASUREMENTS.md#the-test-machine).

## The levels

| Level | Checks | Against | Runs on | Time limit |
| --- | --- | --- | --- | --- |
| 1. Host | Python: the scheduler, admission, sequence state, the memory plan, the tensor table, the protocol, the front end on the mock engine ([ADR-0022](adr/0022-two-processes-python-and-cpp.md)). C++: layouts shared across the C ABI, ring buffers, the arena's free list | values written in the test; the C ABI's struct sizes and offsets, from both sides | anywhere, no card | 30 s |
| 2. Wrapper | each Intel builtin in [ADR-0003](adr/0003-kernel-language.md)'s header (block reads, DPAS, sub-group shuffles) | results written in the test ([record](measurements/MEASUREMENTS.md#2026-10-11-kernel-spike)) | the machine | 1 min |
| 3. Kernel | each kernel on fixed inputs: every expert type's decode, the dense products, DeltaNet, attention, the indexer, norms, sampling | ggml's CPU code for the same operation, through the pinned ggml, or a double-precision reference in the test | the machine | 5 min |
| 4. Layer | one whole block of the model on fixed inputs. Only the image encoder has one | llama.cpp's rows for the same images ([record](measurements/MEASUREMENTS.md#2026-10-17-encoder-reference)) | the machine | 10 min |
| 5. End to end | the tokenizer; top-1 agreement, mean and p99 KL and perplexity on the KL set, teacher-forced | the reference prompts and the BF16 base dump ([REFERENCE_OUTPUTS.md](measurements/REFERENCE_OUTPUTS.md#the-kl-set)) | the machine | 30 min |
| 6. Determinism | a request gives the same tokens alone, in a window, batched and packed with other prompts, with [ADR-0016](adr/0016-how-a-change-is-judged.md)'s switches set (no PCIe share of the misses, no adaptive swaps, no prefetching, no prompt cache). Also two runs of one request, and the CPU's misses on 8 and 16 workers | the same request run alone | the machine | 30 min |
| 7. Performance | accepted tokens and wall time per window ([ADR-0016](adr/0016-how-a-change-is-judged.md)); decode and prompt tokens/s at 4K against [ADR-0019](adr/0019-performance-target.md)'s target | the latest [measurements](measurements/MEASUREMENTS.md), before and after alternating (ABBA) on fresh processes, 3 rounds, medians and ranges ([method](measurements/MEASUREMENTS.md#how-the-numbers-were-taken)) | the machine, set up as the method's step 1 says | about 30 min |

Levels 1-3 run on every change. Levels 4-6 run on every change to a kernel, the decode loop, the memory system or the
scheduler. Level 7 runs on every change that claims to be faster.

Level 7 counts only once level 6 passes, because a faster run that changes the tokens computes something else. A
change fails level 7 if its median wall time per window is worse than before by more than the spread of the "before"
runs, or if accepted tokens per window drop at all with the switches set. Its record says which.

The tests that exist ([DEVELOPING.md](DEVELOPING.md#tests) describes each):

| Level | Tests | Missing |
| --- | --- | --- |
| 1 | `tests/host/`, `gguf_test`, `mrope_test` | ring buffers, the free list |
| 2 | `ze_smoke_test`, `topk_test`, `gemm_test`'s DPAS layout check | some builtins |
| 3 | `matvec_test`, `moe_test`, `gemm_test`, `qgemm_test`, `attn_test`, `vision_test` | |
| 4 | `encoder_test` | a text layer |
| 5 | `kl_test`, `api_test`, the tokenizer check in `tests/end_to_end/` | free-running |
| 6 | `batch_test`, `pack_test`, `api_test`'s drafting and swap cases | two runs, worker counts |

## Tolerances

A tensor's error is its largest absolute difference divided by the largest absolute reference value, unless a row
says otherwise.

| Path | Tolerance | Why |
| --- | --- | --- |
| Integer work: decoding a quantised block, int8 dot products before scaling | exact | integers have one right answer |
| FP32 results from quantised or BF16 weights (the decode kernels, given the reference's activations) | 1e-5 | only the order of float sums differs. Measured: 1.0e-7 to 1.4e-7 for all eight types ([record](measurements/MEASUREMENTS.md#2026-10-11-kernel-spike)) |
| FP16 products on the XMX units (the prompt path) | 2e-3 | fp16 has 11 bits, a rounding of 4.9e-4 per value. Measured: 1e-7 to 3e-4 ([record, part 8](measurements/MEASUREMENTS.md#2026-10-13-design-inputs)) |
| BF16 outputs (the image encoder's kernels that write a product's next input, [MODEL_VISION.md](spec/MODEL_VISION.md)) | 1e-2 | bf16 has 8 bits, a rounding of 3.9e-3 per value, plus the fp16 error where inputs are fp16 |
| The DeltaNet recurrence over 2,048 tokens, FP32 state | 1e-5 | checked against double precision in the same record; the state does not grow, as its gate is below 1 |
| End to end, the KL set against the BF16 checkpoint | mean KL at most 0.170, top-1 agreement at least 83.5% | the IQ3_S file read by the engine gives 0.158 and 84.7% ([record](measurements/MEASUREMENTS.md#2026-10-29-kl-bf16)); the gate catches breakage. A change that moves the distribution is judged by `kl_test --ab`, the change in mean KL with its 95% interval ([ADR-0054](adr/0054-the-bf16-checkpoint-is-the-end-to-end-reference.md)) |

## Test data

| Data | Where | Found by |
| --- | --- | --- |
| Small vectors (a few KB): wrapper results, hand-made kernel inputs | committed, beside the tests | relative path |
| The reference prompts and greedy tokens | committed, `reference/` | relative path |
| The BF16 base dump, `kl-base-bf16.bin` (41 MB) | not in Git; [REFERENCE_OUTPUTS.md](measurements/REFERENCE_OUTPUTS.md#the-kl-set) says how to make it and gives its sha256 | `FLASH_REF_DIR` |
| llama.cpp's KL base (8.1 GB) | not distributed, and read by no check; [REFERENCE_OUTPUTS.md](measurements/REFERENCE_OUTPUTS.md) says how to make it | `FLASH_REF_DIR` |
| The encoder's reference rows | the machine, `$FLASH_REF_DIR/vision` ([record](measurements/MEASUREMENTS.md#2026-10-17-encoder-reference)) | `FLASH_REF_DIR` |
| The model | the machine, `/models` in the container ([MODEL_FILES.md](MODEL_FILES.md)) | `FLASH_MODEL_DIR`, default `/models` |

**A test whose data is missing, or fails its sha256 check, skips and names the file. It never passes**
([ADR-0012](adr/0012-correctness-reference.md), decision 5). A run reports how many tests it skipped.

## Where tests run

Level 1 runs anywhere, without a card. With the container, every kernel can also be compiled with `ocloc`.

Levels 2-7 run on a machine with two B60s, in the engine image ([ADR-0025](adr/0025-the-engine-image.md)). GPU tests
have the ctest label `gpu`. A GPU test skips when Level Zero finds no GPU, and fails instead when `ENGINE_EXPECT_GPUS`
is set. Runs on the cards set it.

CI runs on GitHub's hosted runners on every push ([ADR-0028](adr/0028-ci-without-a-card.md)). It builds the image,
the engine and every kernel, runs the Python checks and level 1, and checks that each GPU test of the core skips
without a card, and fails when `ENGINE_EXPECT_GPUS` is set. No runner has the cards, so levels 2-6 run by hand.

## The rule for every kernel change

A change that adds or changes a kernel:

1. adds a level-3 test against ggml or a scalar reference, at the model's shapes, with its type's tolerance;
2. runs the existing level 4-6 tests and says they passed;
3. if it claims to be faster, runs level 7 and links its row in [MEASUREMENTS.md](measurements/MEASUREMENTS.md);
4. lists in its commit message the commands run, the counts passed and skipped, and the largest error per test.
