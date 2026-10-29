# Which model file to pick

C0ven runs one model, Qwen3.8-Flash-Next, from any of four files that ISTA-DASLab publishes or from Unsloth's
UD-IQ4_XS. This page is for choosing between them when you have not done it before. Where to download the file you pick and how to lay it out is
in [MODEL_FILES.md](MODEL_FILES.md).

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

| File | Download | RAM the engine takes | Experts a layer on the cards |
| --- | ---: | ---: | ---: |
| IQ3_S | 83.6 GB | [22.6 GiB](measurements/MEASUREMENTS.md#2026-10-19-other-quants) | [307 of 512](measurements/MEASUREMENTS.md#2026-10-19-other-quants) |
| IQ3_XXS | 75.8 GB | [14.6 GiB](measurements/MEASUREMENTS.md#2026-10-19-other-quants) | [369](measurements/MEASUREMENTS.md#2026-10-19-other-quants) |
| IQ2_XS | 68.0 GB | [7.1 GiB](measurements/MEASUREMENTS.md#2026-10-19-other-quants) | [460](measurements/MEASUREMENTS.md#2026-10-19-other-quants) |
| Q2_0 | 66.4 GB | about 5.6 GiB (*estimate*) | [481](measurements/MEASUREMENTS.md#2026-10-19-other-quants) |
| UD-IQ4_XS | 93.7 GB | about 32 GiB, against IQ3_S's 21 in the same test ([record](measurements/MEASUREMENTS.md#2026-10-29-ud-iq4xs-speed)) | [244](measurements/MEASUREMENTS.md#2026-10-29-ud-iq4xs-speed) |

### Disk

Each of ISTA-DASLab's files comes in two parts. The second part, 28.8 GB, is the same in all four, so keeping a
second size costs only its first part ([MODEL_FILES.md](MODEL_FILES.md#the-other-quants)). UD-IQ4_XS comes in three
parts and shares none of them. The draft layer and the image
encoder are extra and the same for every file.

### RAM

The figures are the most the engine's process took on two cards with one request slot and a 262,144-token
conversation: the experts it keeps in RAM plus that conversation's memory. Q2_0's is an estimate: its measured
expert store (1.91 GiB) plus the 3.7 GiB the other files added to theirs. Each further slot that holds a full
262,144-token conversation adds about 3.2 GiB ([measurements/MEMORY_BUDGET.md](measurements/MEMORY_BUDGET.md)). Leave room beyond these for
the system and your other programs: if the machine starts to swap, every reply slows down.

### Cards

All the figures on this page are for two B60s. On one card the engine runs IQ3_S at less than half the
speed and needs about 45 GiB of free RAM ([record](measurements/MEASUREMENTS.md#2026-10-19-one-card)). The smaller files have not
been measured on one card. They should need far less RAM there, for the same reason as on two.

## Then speed

| File | Reply speed after a short prompt (tokens/s) | Wait before the reply starts to a 246,571-token prompt (s) |
| --- | ---: | ---: |
| IQ3_S | [85](measurements/MEASUREMENTS.md#2026-10-18-against-llama-cpp) | [137.4](measurements/MEASUREMENTS.md#2026-10-19-other-quants) |
| IQ3_XXS | [98.3](measurements/MEASUREMENTS.md#2026-10-19-other-quants) | [132.0](measurements/MEASUREMENTS.md#2026-10-19-other-quants) |
| IQ2_XS | [100.8](measurements/MEASUREMENTS.md#2026-10-19-other-quants) | [127.9](measurements/MEASUREMENTS.md#2026-10-19-other-quants) |
| Q2_0 | [104.4](measurements/MEASUREMENTS.md#2026-10-19-other-quants) | [128.1](measurements/MEASUREMENTS.md#2026-10-19-other-quants) |
| UD-IQ4_XS | [81.5](measurements/MEASUREMENTS.md#2026-10-29-ud-iq4xs-speed), against IQ3_S's 99.7 in the same test | [145.5](measurements/MEASUREMENTS.md#2026-10-29-ud-iq4xs-speed), against IQ3_S's 139.9 |

A token is a piece of a word: English text runs at about four tokens to three words. Reply speed is how fast the
answer streams once it starts, after a 4,159-token prompt. All four stream several times faster than anyone reads,
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

| File | AIME25, maths (30 problems) | GPQA-Diamond, science (198 questions) | LiveCodeBench v6, coding | Average, as a share of BF16's |
| --- | ---: | ---: | ---: | ---: |
| BF16, the original | 100 | 91.92 | 87.43 | 100% |
| IQ3_S | 100 | 92.93 | 86.86 | 100.2% |
| IQ3_XXS | 100 | 91.41 | 86.29 | 99.4% |
| IQ2_XS | 96.67 | 87.37 | 83.43 | 95.7% |
| Q2_0 | 96.67 | 89.39 | 81.14 | 95.7% |

The scores are percent correct. A share above 100% is chance: the tests are small, and one question of GPQA-Diamond
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
