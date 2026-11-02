# Benchmarks

How the chart in the [README](../README.md#benchmarks) was measured. Its numbers, with their ranges and the time to
the first token at each prompt size, are in
[MEASUREMENTS.md](measurements/MEASUREMENTS.md#2026-11-01-against-strata-and-llama-cpp).

Each server took one request at a time through its `/v1/chat/completions`: the reference prompts `doc-4k-qa` (4,159
tokens) and `doc-32k` (32,010 tokens), and a 255,017-token prompt that, with its answer, nearly fills the
262,144-token context. Greedy, 512 tokens out. Prompt reading is the prompt's tokens over the time to the first token,
decode the tokens after the first over the time to the last, both timed at the client. Each number is the median of
six runs, each a fresh server process, on the same IQ3_S file and the same two B60s (2026-11-01). Answers to `doc-32k`
end at 110 tokens in some runs and at 512 in others, and the two lengths decode at different rates, so its decode
numbers compare best through the ranges in MEASUREMENTS.md.

The four-request row sends `agent-turn1` to `agent-turn4` together on two cards, with no prefix shared between them.
Reading there is all prompt tokens over the time to the last first token. The decode panel shows the aggregate rate:
all tokens out over the time from the first send to the last token, which counts the wait for the prompts too.

C0ven 0.12.2 ran with DFlash 2's files beside the model. On one card (`ZE_AFFINITY_MASK=0`) the engine runs the whole
model on that card, with 129 experts a layer in VRAM against two cards' 355, and the rest in RAM; it needs about 45 GiB
of free host memory to load. Four requests used `--slots 4 --kv-tokens 524288 --default-max-tokens 32768`.

Strata 0.2.9 with engine 0.1.55 ran with its SYCL backend, from what its `sycl/setup_intel.py` writes for these cards,
with four changes. The setup's two-card settings gave text that turned to `!!!!` after a few tokens, so
`STRATA_VERIFY_NO_HOST=1` is set, which also turns its adaptive swaps off. That setting needs `layer_split` 32 in place of
the automatic split and `STRATA_MIRROR_MIB=20000`, and `--ple-io mmap` replaces `ram` to stay within the host's memory.
Its prompt cache is off. Four requests add `--batch 4`. Other settings may do better or worse on these cards. Its long
prompt counts 255,198 tokens, and each of its prompts with tools 108 fewer than the other two. One start of 19 stalled
during its warm-up and was run again.

llama.cpp release b12032 ran with its SYCL backend, `-ub 1024`, its prompt cache off and its own placement, which keeps
some tensors on the CPU. Four requests used `-np 4 --kv-unified`. Its 255,017-token numbers are approximate: they come from
two runs each, because a first token takes 21-25 minutes.

vLLM was not run because the model does not fit. Its smallest checkpoint is 102.2 GB (`nvidia/Qwen3.8-Flash-Next-FP4-Local`),
and the two B60s hold 48 GB. The others range from 130.8 GB to 360 GB.
