# Which model file to pick

C0ven runs one model, Qwen3.8-Flash-Next, from any of four files that ISTA-DASLab publishes or from Unsloth's
UD-IQ4_XS. This page explains what sets them apart. Each file's size, RAM and speed are in
[Getting started](GETTING_STARTED.md#1-pick-a-model), and every file's source and layout are in
[MODEL_FILES.md](MODEL_FILES.md).

## The short answer

Start with IQ3_S, the default. Every other figure in these docs is taken with it.

Pick IQ3_XXS if you want replies about 15% faster or have about 8 GiB less RAM to spare. Its benchmark scores are
within a few questions of IQ3_S's.

Pick IQ2_XS or Q2_0 if your machine is short of RAM, or if speed matters more to you than the last few points
of accuracy. They score about 4 points lower on hard reasoning and coding tests.

Pick UD-IQ4_XS if you want answers closer to the original model and can spare about 11 GiB more RAM. Replies
stream about a fifth slower than IQ3_S's.

## What the four files are

The model has billions of numbers, its weights. A *quant* stores each weight in fewer bits than the original 16, so
the file is smaller and faster to read, and the answers drift a little from the original model's. The name says
roughly how many bits a weight gets: IQ3_S and IQ3_XXS about 3, IQ2_XS and Q2_0 about 2. ISTA-DASLab chose the type
tensor by tensor, so each file gives more bits to some parts of the model than to others.

The model is a mixture of experts: for each word it uses a few of its 512 experts a layer. The experts used most
stay in the cards' memory (VRAM) and the rest wait in the computer's memory (RAM), where the CPU computes them. A
smaller file fits more experts on the cards, so less work falls to the CPU. That is why the smaller files are faster
and need less RAM.

## Your hardware decides first

| File | Experts a layer on the cards |
| --- | ---: |
| IQ3_S | [307 of 512](measurements/MEASUREMENTS.md#2026-10-19-other-quants) |
| IQ3_XXS | [369](measurements/MEASUREMENTS.md#2026-10-19-other-quants) |
| IQ2_XS | [460](measurements/MEASUREMENTS.md#2026-10-19-other-quants) |
| Q2_0 | [481](measurements/MEASUREMENTS.md#2026-10-19-other-quants) |
| UD-IQ4_XS | [244](measurements/MEASUREMENTS.md#2026-10-29-ud-iq4xs-speed) |

### Disk

Each of ISTA-DASLab's files comes in two parts. The second part, 28.8 GB, is the same in all four, so keeping a
second size costs only its first part ([MODEL_FILES.md](MODEL_FILES.md#the-other-quants)). UD-IQ4_XS comes in three
parts and shares none of them. The draft layer and the image
encoder are extra and the same for every file.

### RAM

The more experts stay in RAM, the more RAM a file needs, and the [table of files](GETTING_STARTED.md#1-pick-a-model)
gives the figure for each. If the machine starts to swap, every reply slows down. Each further slot that holds a full
262,144-token conversation adds about 3.2 GiB ([measurements/MEMORY_BUDGET.md](measurements/MEMORY_BUDGET.md)).

### Cards

All the figures on this page are for two B60s. On one card the engine runs IQ3_S at less than half the
speed ([record](measurements/MEASUREMENTS.md#2026-11-01-against-strata-and-llama-cpp)). The smaller files have not
been measured on one card.

## Then speed

| File | Wait before the reply starts to a 246,571-token prompt (s) |
| --- | ---: |
| IQ3_S | [137.4](measurements/MEASUREMENTS.md#2026-10-19-other-quants) |
| IQ3_XXS | [132.0](measurements/MEASUREMENTS.md#2026-10-19-other-quants) |
| IQ2_XS | [127.9](measurements/MEASUREMENTS.md#2026-10-19-other-quants) |
| Q2_0 | [128.1](measurements/MEASUREMENTS.md#2026-10-19-other-quants) |
| UD-IQ4_XS | [145.5](measurements/MEASUREMENTS.md#2026-10-29-ud-iq4xs-speed), against IQ3_S's 139.9 |

A token is a piece of a word: English text runs at about four tokens to three words. Reply speed, in the
[table of files](GETTING_STARTED.md#1-pick-a-model), is how fast the answer streams once it starts. All four stream several times faster than anyone reads,
so the difference shows in long answers, code and programs that call the model many times. The wait before the first
token grows with the prompt. A 4,159-token prompt, about 3,000 words, starts in about two seconds on every file; a
prompt that fills the whole context, such as a large codebase, takes over two minutes on
every file, and the smaller files save 5-10 s of it.

The smaller files are faster because more experts fit on the cards and each expert is fewer bytes to read. After a
very long prompt the gap mostly closes: IQ2_XS replied at 84.8 tokens/s against IQ3_S's 79.3, Q2_0 at the same speed
as IQ3_S, and IQ3_XXS a little slower than IQ3_S in its own test, 68.5 against 72.5
([record](measurements/MEASUREMENTS.md#2026-10-19-other-quants)).

## Benchmarks as an anchor

How much accuracy does each file give up? ISTA-DASLab scored each file on three hard tests and compared it with the
original 16-bit model (BF16), on [the files' model card](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF).
These are their figures, not C0ven's:

| File | AIME25, maths (30 problems) | GPQA-Diamond, science (198 questions) | LiveCodeBench v6, coding |
| --- | ---: | ---: | ---: |
| BF16, the original | 100 | 91.92 | 87.43 |
| IQ3_S | 100 | 92.93 | 86.86 |
| IQ3_XXS | 100 | 91.41 | 86.29 |
| IQ2_XS | 96.67 | 87.37 | 83.43 |
| Q2_0 | 96.67 | 89.39 | 81.14 |

The scores are percent correct. An average above the original's is chance: the tests are small, and one question of GPQA-Diamond
is half a point. Read the 3-bit files as matching the original and the 2-bit files as about 4 points behind it, losing
one maths problem in 30 and a few science and coding questions. Q2_0 does a little better at science and IQ2_XS at
coding.

These tests are hard on purpose: long chains of reasoning and code that has to run, where small errors add up.
Nobody has measured the files on everyday questions, summaries or chat.

Unsloth does not publish these scores for UD-IQ4_XS. C0ven's `kl_test` instead measures how far a file's next-word
choices are from the original model's, on 5,000 tokens of code, docs, English, four other languages and agent turns.
Lower is closer:

| File | Mean KL | Same top choice as the original |
| --- | ---: | ---: |
| IQ3_S | [0.158](measurements/MEASUREMENTS.md#2026-10-29-kl-bf16) | 84.7% |
| UD-IQ4_XS | [0.112](measurements/MEASUREMENTS.md#2026-10-29-kl-ud-iq4xs) | 87.4% |

## Choosing

| Your situation | File |
| --- | --- |
| Two B60s, RAM to spare, you want the best answers | IQ3_S |
| You want faster replies and do not want to give up accuracy | IQ3_XXS |
| About 15-25 GiB of RAM free for the engine | IQ3_XXS |
| Less than about 15 GiB free, or you want the fastest replies | IQ2_XS, or Q2_0 for the least RAM |
| Mostly code | IQ3_S or IQ3_XXS; of the 2-bit files, IQ2_XS |
| A low disk budget | Q2_0, 66.4 GB |
| The answers closest to the original, at about a fifth less reply speed | UD-IQ4_XS |

Changing your mind costs only a download: point `--model` at the other file's first part and restart
([MODEL_FILES.md](MODEL_FILES.md#pointing-the-engine-at-the-files)). The engine's tests, references and the README's
figures stay on IQ3_S ([ADR-0038](adr/0038-the-other-quants-load.md), draft).
