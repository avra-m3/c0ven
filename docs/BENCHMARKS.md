# Benchmarks

How the chart in the [README](../README.md#benchmarks) was measured. Its numbers, with their ranges and the time to
the first token at each prompt size, are in MEASUREMENTS.md:
[C0ven on two cards and one](measurements/MEASUREMENTS.md#2026-10-27-readme-benchmarks) and
[C0ven against llama.cpp](measurements/MEASUREMENTS.md#2026-10-18-against-llama-cpp).

Each server took one request at a time through its `/v1/chat/completions`: the reference prompts `doc-4k-qa` (4,159
tokens) and `doc-32k` (32,010 tokens), and a 255,017-token prompt that, with its answer, nearly fills the
262,144-token context. Greedy, 512 tokens out. Prompt reading is the prompt's tokens over the time to the first token,
decode the tokens after the first over the time to the last, both timed at the client. Each number is the median of
six runs, each a fresh server process.

C0ven 0.5.0 ran with its defaults and DFlash 2's files beside the model (2026-10-27). On one card
(`ZE_AFFINITY_MASK=0`) the engine runs the whole model on that card, with 129 experts a layer in VRAM against two
cards' 356, and the rest in RAM; it needs about 45 GiB of free host memory to load. With the 255,017-token prompt the
first token comes after 150 s on two cards and 279 s on one.

llama.cpp release b11719 ran with its SYCL backend on the same two B60s and the same IQ3_S file (2026-10-18). Its own
placement (`--fit`) keeps layer 31's feed-forward and the experts of layers 32-48 on the CPU. It was not run with the
255,017-token prompt.
