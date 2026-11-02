# Choosing a model

C0ven runs one model, Qwen3.8-Flash-Next, and you can download it at several sizes. Each size is a file you choose by
name. A smaller file needs less RAM, and its answers can be a little less accurate.

## The short answer

| Your situation | File |
| --- | --- |
| Not sure, and you have the RAM for it | IQ3_S, the file we test the most |
| You want to use less RAM with almost no loss of accuracy | IQ3_XXS |
| You are short of RAM | IQ2_XS, or Q2_0 for the least RAM |
| You mostly write code and are short of RAM | IQ2_XS |
| You want answers closest to the original model, and have RAM to spare | UD-IQ4_XS |

Then go to the [quick start](../README.md#3-download-it) and use the file's name in the download command. You can
change your mind later: download another file and start the server with it.

## Check what your machine has

Free RAM is the `available` column of this command:

```bash
free -h
```

Free disk is the `Avail` column of this one. You need the file's download size plus about 2 GB for the parts that go
with it:

```bash
df -h ~
```

## Files we have tested

| File | Tested on | Download | RAM, one B60 | RAM, two B60s |
| --- | --- | ---: | ---: | ---: |
| IQ3_S | one and two B60s | 83.6 GB | [43 GiB](measurements/MEASUREMENTS.md#2026-10-19-one-card) | [22.6 GiB](measurements/MEASUREMENTS.md#2026-10-19-other-quants) |
| IQ3_XXS | two B60s | 75.8 GB | about 36 GiB | [14.6 GiB](measurements/MEASUREMENTS.md#2026-10-19-other-quants) |
| IQ2_XS | two B60s | 68.0 GB | about 29 GiB | [7.1 GiB](measurements/MEASUREMENTS.md#2026-10-19-other-quants) |
| Q2_0 | two B60s | 66.4 GB | about 27 GiB | about 5.6 GiB |
| UD-IQ4_XS | two B60s | 93.7 GB | about 53 GiB | [about 32 GiB](measurements/MEASUREMENTS.md#2026-10-29-ud-iq4xs-speed) |

The [benchmarks](../README.md#benchmarks) show IQ3_S's speed on one and two cards.

## Files we expect to work

These are [Unsloth's other files](MODEL_FILES.md#unsloths-other-files). The engine has code for every kind of weight
in them, but none has been run on the engine. If you try one, please
[tell us how it went](https://github.com/avra-m3/c0ven/issues).

| File | Download | RAM, one B60 | RAM, two B60s |
| --- | ---: | ---: | ---: |
| UD-IQ1_S | 72.5 GB | about 33 GiB | about 12 GiB |
| UD-IQ1_M | 74.5 GB | about 35 GiB | about 14 GiB |
| UD-Q2_K_XL | 78.9 GB | about 39 GiB | about 18 GiB |
| UD-IQ3_XXS | 82.0 GB | about 42 GiB | about 21 GiB |
| UD-Q3_K_XL | 90.0 GB | about 49 GiB | about 28 GiB |
| UD-Q4_K_XL | 111.3 GB | about 69 GiB | about 48 GiB |
| UD-Q5_K_XL | 158.3 GB | about 89 GiB | about 68 GiB |
| UD-Q6_K_XL | 169.2 GB | about 99 GiB | about 78 GiB |
| Q8_0 | 188.2 GB | about 117 GiB | about 96 GiB |

## What the RAM figures mean

RAM is what the engine takes with one request at a time and a 262,144-token context. Each further request slot adds
about 3.2 GiB. Your system and other programs need room beyond this: if the machine starts to swap, every reply slows
down.

A figure with "about" is an *estimate*, from the engine's memory plan for that file plus 4 GiB, the amount the plan
fell short of IQ3_S's measured peak. Q2_0's on two B60s is its measured expert memory plus the 3.7 GiB the other files
added. UD-IQ4_XS's on two B60s is a measured figure, set against IQ3_S's 21 GiB in the same test.

## How accurate each file is

ISTA-DASLab, who publish the first four files, scored them on three hard tests against the original 16-bit model.
These are [their figures](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF), in percent correct:

| File | AIME25, maths | GPQA-Diamond, science | LiveCodeBench v6, coding | Average, as a share of the original's |
| --- | ---: | ---: | ---: | ---: |
| The original | 100 | 91.92 | 87.43 | 100% |
| IQ3_S | 100 | 92.93 | 86.86 | 100.2% |
| IQ3_XXS | 100 | 91.41 | 86.29 | 99.4% |
| IQ2_XS | 96.67 | 87.37 | 83.43 | 95.7% |
| Q2_0 | 96.67 | 89.39 | 81.14 | 95.7% |

Read IQ3_S and IQ3_XXS as matching the original, and IQ2_XS and Q2_0 as about 4 points behind it. A score above the
original's is chance, because the tests are small. Nobody has measured these files on everyday questions, summaries or
chat.

Unsloth publishes no such scores for its files. Our own test compares each file's choice of next word with the original
model's. UD-IQ4_XS makes [the same top choice 87.4%](measurements/MEASUREMENTS.md#2026-10-29-kl-ud-iq4xs) of the time,
and IQ3_S [84.7%](measurements/MEASUREMENTS.md#2026-10-29-kl-bf16).

## Disk space when you keep several files

IQ3_S, IQ3_XXS, IQ2_XS and Q2_0 share a 28.8 GB second part. The download reuses it when you already have one of them
in the folder, so a second file costs 28.8 GB less. Unsloth's files share nothing with them.
