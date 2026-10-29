# Developing

This page covers building the C++ core and its kernels and running its tests. From
[the Python half](#the-python-half) on, it covers the front end, the control plane and the tools.
[Releases](#releases) says how a version is made.
[ADR-0024](adr/0024-repository-layout.md) says where code goes, and [TESTING.md](TESTING.md) says how a change is
checked.

## Build

The core builds in the image that the [`Dockerfile`](../Dockerfile) defines
([ADR-0025](adr/0025-the-engine-image.md)). It needs the Level Zero headers, `ocloc` and the llama.cpp tree at
`/opt/llama.cpp`, and the image provides all three. Build the image with
`docker build --target engine -t c0ven/engine:<tag> .`; CI does not publish it
([ADR-0043](adr/0043-publish-only-the-serving-image.md)). Then start a container with the cards, the model and the checkout. The
render group's id is the group of `/dev/dri/renderD128` on the host (`stat -c %g /dev/dri/renderD128`):

```bash
docker run -d --name engine-dev --label project=c0ven --device /dev/dri --group-add <render gid> --ulimit memlock=-1:-1 \
  -v <model dir>:/models:ro -v c0ven-cache:/home/ubuntu/.cache -v <checkout>:/src -w /src \
  c0ven/engine:<tag> sleep infinity
docker exec engine-dev bash -c 'cmake -S /src -B /src/build -G Ninja -DENGINE_WERROR=ON && ninja -C /src/build'
```

The build makes:

| Output | What it is |
| --- | --- |
| `build/kernels/*.bin` | the kernel modules, compiled ahead of time by `ocloc` for `bmg-g21` ([ADR-0003](adr/0003-kernel-language.md)), one per `engine_add_kernel_module` in [`kernels/CMakeLists.txt`](../kernels/CMakeLists.txt): the decode step's `matvec_<type>_l<lanes>` and `moe_<type>_g<group>`, the prompt path's `gemm_*`, `qgemm_<type>[_file]` and `qwen4exp_prompt`/`_attn`/`_gdn`, the image encoder's `qwen4exp_vision*`, the model's own `qwen4exp`, and `common` and `smoke` |
| `build/core/libengine.so` | the C ABI of [`include/engine_api.h`](../include/engine_api.h), the library the control plane loads. It exports only the `engine_` symbols |
| `build/core/libengine_core.a` | the core itself, which the tests link |
| `build/core/tests/*` | the core's tests |
| `build/tools/route_profile` | builds the routing profile the expert cache starts from, by running the reference prompts through the prompt path |

ggml's CPU backend is built from `/opt/llama.cpp` with the reference build's CPU flags
([cmake/ggml.cmake](../cmake/ggml.cmake)). `-DENGINE_GGML_DIR=` points at another tree at the same commit.

## Tests

```bash
docker exec engine-dev bash -c 'cd /src/build && ctest -LE gpu'                       # no card needed
docker exec engine-dev bash -c 'cd /src/build && ENGINE_EXPECT_GPUS=2 ctest -L gpu'   # on the cards
```

| Test | Level ([TESTING.md](TESTING.md)) | What it checks |
| --- | --- | --- |
| `gguf_test` | 1 | the GGUF reader against the census of the IQ3_S file |
| `mrope_test` | 1 | the M-RoPE positions an image gives a prompt's cells, against transformers' rule; no card |
| `ze_smoke_test` | 2 | a kernel runs on every card, and one card waits on another's event |
| `matvec_test` | 3 | the weight products of all ten quantised types against ggml's dequantisation, 1e-5 |
| `moe_test` | 3 | the decode step's grouped expert products of `moe.cl` equal `matvec.cl`'s bit for bit, for every expert type and group size; Q8_0 down on random weights quantised by ggml when the file has none |
| `topk_test` | 2 | `topk_rows` against the host: the k largest of rows of up to 248,320 values, ties to the lower indices, both of its paths |
| `gemm_test` | 2, 3 | the prompt path's fp16 and bf16 products on the XMX units, and the DPAS register layout, 2e-3 |
| `qgemm_test` | 3 | the prompt path's quantised products: dense weights dequantised to fp16, the experts decoded on the fly in both layouts, 2e-3; Q8_0 down experts on random weights when the file has none |
| `kl_test` | 5 | the engine's distribution against the BF16 checkpoint's on the KL set, teacher-forced one token a step: mean KL at most 0.170, top-1 agreement at least 83.5%. Needs `$FLASH_REF_DIR/kl-base-bf16.bin` |
| `attn_test` | 3 | the prompt path's attention kernel against double precision: dense and sparse rows, rows past 12,288 positions, keys that grow along the positions, both cards' head layouts; 2e-3 |
| `vision_test` | 3 | the image encoder's kernels against double precision: 1e-5 for f32 outputs, 2e-3 for fp16, 1e-2 for bf16 |
| `encoder_test` | 4 | the image encoder against llama.cpp's rows for the same images and video frame pairs ([images](measurements/MEASUREMENTS.md#2026-10-17-encoder-reference), [pairs](measurements/MEASUREMENTS.md#2026-10-17-video-pair-reference)): mean cosine at least 0.999, least at least 0.95. Needs `$FLASH_REF_DIR/vision` and the `mmproj` file |
| `batch_test` | 6 | a decode step of several rows (several sequences, a window of one, or windows of several in one step, each committed alone) gives each row exactly its single-row result |
| `pack_test` | 6 | a prompt chunk of several sequences' runs, one of them from the middle of its prompt, gives each sequence exactly the candidates and the greedy tokens of its prompt read alone; so do runs after a chunk's first that hold an image's block or start past the indexer's 2,051 cells and past 12,288 positions |
| `api_test` | 5 | the C ABI from C: 32 greedy tokens, the first 19 equal to llama.cpp's (token 19 is a near tie), release and admit again, a stop token, short prompt pieces read as decode windows, a 108-token prompt by the prompt path, two requests in one step, a checkpoint restored into another slot that shares the first slot's pages and copies the page the checkpoint ends inside, one token read after a restore, the timing records; with the draft layer's tensors (`FLASH_MTP_DIR`, by default `/models/mtp`), drafting gives the same tokens alone and with two slots drafting in one window; with adaptive swaps, three requests keep their first 16 tokens |

A test whose model is missing skips (exit code 77). Tests find the model at `FLASH_MODEL_DIR`, by default
`/models`; ctest gives `api_test` the path `/models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf`.
`FLASH_MODEL` names another file's first shard instead: `matvec_test`, `qgemm_test` and `moe_test` then check the
kernels of every quantised type that file has (`gguf_test` and `kl_test` hold the IQ3_S file's own numbers).
A GPU test also skips when Level Zero lists no GPU, unless `ENGINE_EXPECT_GPUS` is set; then it fails. `api_test`
checks only for the model, since the C ABI does not say why `engine_open` failed.

`kl_test <set> <base dump> [cards] [prompt tokens] [dump to write]` also runs by hand. Given a number of prompt
tokens, it reads each segment's first ones through the prompt path, in chunks of up to 512, and scores the positions
after them. Given a path, it writes the engine's dump there. `kl_test --compare <base dump> <dump> [set]` compares two
dumps without a card and without the gate. `kl_test --ab <base dump> <before> <after> [set]` compares a change's two
dumps with the base and gives the change in mean KL with its 95% interval, the figure a change is judged by
([ADR-0054](adr/0054-the-bf16-checkpoint-is-the-end-to-end-reference.md)). A dump named without a directory is read
from `FLASH_REF_DIR`.

`ENGINE_MTP_CACHE_DIR` keeps the draft layer's quantised experts between loads. When it is set, a load reads them
from that directory instead of quantising the BF16 tensors again (about 43 s for both cards,
[record](measurements/MEASUREMENTS.md#2026-10-16-load-phases)). If the file is missing or does not match, the load quantises and
writes it there. There is one file per card, named by a key built from the MTP directory's manifest, the expert type
and the card split. Point the variable at a volume, such as the dev container's cache volume
(`/home/ubuntu/.cache/c0ven/mtp`), since the models are mounted read-only. When it is unset, every load
quantises.

A load locks the arena in memory and stops if it cannot, so the container needs `--ulimit memlock=-1:-1`. A Docker
daemon that cannot raise the limit, such as one inside another container, refuses that flag. There, set
`ENGINE_UNLOCKED_ARENA=1` to load with the arena unlocked. The kernel may then swap out or move arena pages the copy
engines read, and under the host's reclaim a copy can reset a card's copy engine
([ADR-0035](adr/0035-the-arena-in-the-engines-own-memory.md)). Do not time runs that way.

`ZE_AFFINITY_MASK=0` limits a run to card 0. The model then runs whole on one card, with fewer experts cached.


## How a step runs

[DESIGN.md](DESIGN.md) is the architecture. In the code:

- `core/models/qwen4exp/model.cpp` loads each card's half of every layer and records the decode step as one regular
  command list per card for each width (1, 2, 3, 4, 6 and 8 rows), once. The step's first command copies the step plan
  ([`shared/step.h`](../shared/step.h)) from host memory to the card, and the kernels read that copy, so the same list
  runs every step. The host computes the rows' embeddings and resets the step's events before it submits.
- The cards exchange partial sums twice per layer. A one-row step goes through host memory; a window of two rows or
  more writes into the other card's memory ([ADR-0030](adr/0030-exchanges-into-the-other-card.md)). Each card
  signals an event when its partial sum is written and waits on the other's. Regular lists are used because an
  immediate list blocks the host when it waits on an event whose signal is not yet appended.
- Card 0 publishes each layer's routed experts and misses in host memory (`MissJob`). The host thread spins on them,
  computes the layer's misses with ggml ([`core/cpu/experts.cpp`](../core/cpu/experts.cpp)), writes the sum and
  raises a flag in host memory. Card 0's `cpu_wait` kernel spins on that flag, and returns at once when no row
  missed. This is the first form of [ADR-0011](adr/0011-cpu-gpu-signalling.md): flags and spinning waits.
- Each card returns its share of the vocabulary's 64 largest logits; the host merges them and samples.

## How a prompt chunk runs

`core/models/qwen4exp/prompt.cpp` reads up to `prompt_chunk` (4,352 by default) tokens in one pass, as runs of
consecutive tokens of up to four sequences (`Model::prompt_runs`):

- Each card records a list per chunk size (powers of two from 64, and `prompt_chunk`) for a single run at load, and
  one for several runs from 256 rows; a smaller list for several runs is recorded when first used. The kernels read
  the chunk's plan and its runs' (`PromptPlan` in [`shared/step.h`](../shared/step.h)) from host memory, so a list
  serves any chunk up to its size. The row-wise kernels take all the chunk's rows at once. The kernels that follow a
  sequence (the convolutions, the DeltaNet recurrence, attention and its indexer, the PLE history, the rows the head
  reads) also run once for all the runs: each work-group finds its run from its row, or takes the run of its group
  index. The fp16 copy of a sequence's K and V (`p_kv16`) and the attention that reads it run once per run, as the
  copy's buffer holds one sequence.
- A run of up to 8 rows (`PROMPT_SHORT_RUN`) attends as the decode step does, from the 8-bit cache over the cells the
  indexer chose (`p_attn_rows`), with no fp16 copy. The core adds writing slots' next tokens to a chunk with runs to
  spare, as runs of one row ([ADR-0053](adr/0053-writers-rows-ride-in-a-prompt-chunk.md)): they get a token from the
  chunk instead of waiting for it, and the CPU computes none of their experts, since the chunk stages every expert.
- Products run on the XMX units: BF16 weights directly, quantised dense weights dequantised to fp16 just before
  their product ([`kernels/gemm.cl`](../kernels/gemm.cl), [`kernels/quant/qgemm.cl`](../kernels/quant/qgemm.cl)).
  The experts' assignments are grouped by expert on the card, and each expert's tile of up to 64 rows is computed
  with its weights decoded on the fly.
- Each card stages the experts it does not cache into a staging buffer on the card, two layers' worth in turn, so
  the CPU computes nothing for a prompt. For a chunk of more than 1,024 rows a list recorded at load on the card's
  copy engine copies every uncached expert from the arena. For a chunk of 1,024 rows or fewer a thread per card
  copies them through an immediate list (`fetch_experts`): each layer's uncached experts, hottest first by the swap
  counts, until the layer's router has run, then the rest the router picked. The chunk's list waits for them after
  the cached experts' and the shared expert's products. The runtime makes these copies from the anonymous arena with
  the CPU ([measurements/LEVEL_ZERO.md](measurements/LEVEL_ZERO.md) row 31; [the record](measurements/MEASUREMENTS.md#2026-10-21-short-chunk-fetch);
  [ADR-0046](adr/0046-a-short-chunk-fetches-the-experts-its-router-picks.md)).
- The cards exchange fp16 partial sums. The kernel that makes a card's partial sum also writes it into the other
  card's memory while it computes, 64 bytes per store. Stores of 32 bytes or less cross at a tenth of the rate
  ([the record](measurements/MEASUREMENTS.md#2026-10-15-prompt-exchange-stores)).
- `Options::trace` times every kernel of the chunk's lists; `Model::prompt_trace` sums them per kernel.

The C ABI reads a prompt piece of up to 64 tokens with no image in it as decode windows instead: up to 8 forced
rows a window, each committed whole, with the draft layer reading every row (`prompt_windows` in
[`core/api/engine_api.cpp`](../core/api/engine_api.cpp)). Such a piece is faster as windows than as a chunk that
stages every uncached expert ([the record](measurements/MEASUREMENTS.md#2026-10-17-prompt-windows)); against a chunk that fetches
its experts, as above, it is not measured. The other pieces of a plan, of different slots, go as one chunk, up to
four of them and `prompt_chunk` rows (`prompt_pieces`).

## The Python half

The front end, the engine's control plane, the tools and their tests. How the parts fit is
[DESIGN.md](DESIGN.md).

### Tools

[uv](https://docs.astral.sh/uv/) installs Python and the packages. The project pins uv itself
(`required-version` in [`pyproject.toml`](../pyproject.toml)), Python in [`.python-version`](../.python-version),
and every package in [`uv.lock`](../uv.lock). The library versions are [ADR-0004](adr/0004-write-or-reuse.md)'s.

```bash
uv sync
```

`uv sync` makes `.venv/` with the runtime and development packages. A uv older than the pin refuses to run, but any
installed uv can run the pinned one:

```bash
uv tool run --from uv==0.12.22 uv sync
```

`gguf` comes from llama.cpp's repository at the reference commit, not from PyPI ([ADR-0026](adr/0026-python-tooling.md)).

### Checks

Every change passes all four:

```bash
uv run ruff format
```

```bash
uv run ruff check
```

```bash
uv run basedpyright
```

```bash
uv run pytest
```

[ruff](https://docs.astral.sh/ruff/) formats and lints. Its rules are in `pyproject.toml`; `ruff check --fix`
applies the safe fixes.

[basedpyright](https://docs.basedpyright.com/) checks types in strict mode, over the code and the tests
([ADR-0026](adr/0026-python-tooling.md) has the choice and its reasons). Strict mode refuses values of unknown type,
so annotate every function's parameters and returns. A line that must break a rule carries
`# pyright: ignore[rule]`, naming the rule.

<a id="tests-1"></a>

### Tests

`uv run pytest` runs every level that the computer at hand can run ([TESTING.md](TESTING.md)). A test whose data is
missing skips, with a message that names what is missing: the core library (`build/core/libengine.so`, or
`ENGINE_LIBRARY`) or the model (`FLASH_MODEL_DIR`, default `/models`). `-m "not model"` leaves out the tests that
need the model.

The host tests need nothing but the repository. They run:

- the scheduler against the mock engine, which checks every plan's page tables and checkpoints as it runs them;
- the native backend's adapter against a fake of the core ([ADR-0027](adr/0027-the-c-abi-as-the-core-was-optimised.md));
- the front end's APIs against a fake engine;
- the engine process and its client over the line protocol.

The check of the ABI against the core loads `build/core/libengine.so`, so it runs where the core was built: in the
engine image.

### Running without the cards

The engine runs with three backends ([DESIGN.md, section 3](DESIGN.md#3-the-c-abi-between-python-and-c)): `native` (the
core, on the cards), `mock` (made-up tokens, real plans) and `reference` (replays the greedy tokens of the reference
prompts, [measurements/REFERENCE_OUTPUTS.md](measurements/REFERENCE_OUTPUTS.md)). The last two need no core and no model file, so the front end
runs against them anywhere, given the model's `tokenizer.json` and `chat_template.jinja`:

```bash
uv run python -m frontend --engine-backend mock --tokenizer tokenizer.json --chat-template chat_template.jinja --context 8192 --slots 2
```

It serves the OpenAI and Anthropic APIs on `127.0.0.1:8080`. `--engine-arg` passes a flag to the engine as is,
for example `--engine-arg=--realtime`, with which the mock takes as long per step as it reports.

The engine runs alone too, speaking the line protocol ([`protocol.py`](../protocol.py)) on stdin and stdout:

```bash
uv run python -m engine --backend mock --arch qwen4exp --context 4096
```

`GEN 1 8 1,2,3` asks for 8 tokens after the prompt 1, 2, 3, and the engine answers with `T` lines and a `DONE`.
`QUIT` ends it at once, with any request unfinished.

### The memory plan

`tools/plan.py` prints the plan for a context and a number of sequences, from the committed tensor table, with no
model at hand:

```bash
uv run python -m tools.plan --context 262144 --slots 1
```

### The timeline

`tools/attach.py` reads a running engine's timing rings into a Perfetto timeline. It runs in the engine's
container. The engine's `INFO` line gives its pid and the timing region's fd, and the tool's docstring has the
command. The tool expects a shared timing region that no backend provides: both report `timing_fd=-1`, and the core
hands out timing records through `engine_read_timing` instead
([DESIGN.md, section 7](DESIGN.md#7-timing-and-debugging)).

## Releases

The version is in [`VERSION`](../VERSION), and nowhere else ([ADR-0042](adr/0042-version-numbers.md)). Each pull
request raises it, and its author picks which number: the minor number for a change to the server's options, its
API or the model files it accepts, the patch number for anything else. The `version` workflow fails a pull request
whose `VERSION` is not above its base's.

Each merge to `main` is a release: CI pushes the serving image tagged with the version beside `main` and
`sha-<commit>`. It fails instead if that version is already published from another commit, as when two pull requests
raised `VERSION` to the same number.
