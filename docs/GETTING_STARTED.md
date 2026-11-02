# Getting started

You need Docker, the `xe` kernel driver and one model file. These steps take you from nothing to a running server.

## 1. Pick a model

If you are not sure, pick IQ3_S, the file we test the most. If you have less RAM than it needs, pick the largest
tested file that fits. [CHOOSING_A_MODEL.md](CHOOSING_A_MODEL.md) explains what sets the files apart.

All of these are the same model, Qwen3.8-Flash-Next, stored at different sizes. A smaller file needs less RAM and
replies faster, and its answers can be a little less accurate.

### Tested

We have run these files on the engine.

| File | Tested on | Download | Test scores | RAM, one B60 | Speed, one B60 | RAM, two B60s | Speed, two B60s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| IQ3_S | one and two B60s | 83.6 GB | 100.2% | [43 GiB](measurements/MEASUREMENTS.md#2026-10-19-one-card) | [49](measurements/MEASUREMENTS.md#2026-11-01-against-strata-and-llama-cpp) | [22.6 GiB](measurements/MEASUREMENTS.md#2026-10-19-other-quants) | [102](measurements/MEASUREMENTS.md#2026-11-01-against-strata-and-llama-cpp) |
| IQ3_XXS | two B60s | 75.8 GB | 99.4% | about 36 GiB | not measured | [14.6 GiB](measurements/MEASUREMENTS.md#2026-10-19-other-quants) | [98, against IQ3_S's 85](measurements/MEASUREMENTS.md#2026-10-19-other-quants) |
| IQ2_XS | two B60s | 68.0 GB | 95.7% | about 29 GiB | not measured | [7.1 GiB](measurements/MEASUREMENTS.md#2026-10-19-other-quants) | [101, against IQ3_S's 85](measurements/MEASUREMENTS.md#2026-10-19-other-quants) |
| Q2_0 | two B60s | 66.4 GB | 95.7% | about 27 GiB | not measured | about 5.6 GiB | [104, against IQ3_S's 86](measurements/MEASUREMENTS.md#2026-10-19-other-quants) |
| UD-IQ4_XS | two B60s | 93.7 GB | not published | about 53 GiB | not measured | [about 32 GiB](measurements/MEASUREMENTS.md#2026-10-29-ud-iq4xs-speed) | [81, against IQ3_S's 100](measurements/MEASUREMENTS.md#2026-10-29-ud-iq4xs-speed) |

Speed is how fast a reply streams after a 4,159-token prompt, in tokens per second. Speeds with "against" come from
one test series, run on an older version of the engine, so compare a file with the IQ3_S figure beside it.

Test scores are the publisher's, from
[ISTA-DASLab's model card](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF): the average of three
hard tests (AIME25, GPQA-Diamond and LiveCodeBench v6), as a share of the original 16-bit model's. Unsloth publishes no
scores for each file. The [scores test by test](CHOOSING_A_MODEL.md#benchmarks-as-an-anchor) are on the page that
compares the files.

### Expected to work, not tested

These are [Unsloth's other files](MODEL_FILES.md#unsloths-other-files). The engine has code for every weight type in
them, but none has been loaded, so we have no speed figures. If you try one, please
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

### About the RAM figures

RAM is what the engine takes with one request at a time and a 262,144-token context. Each further request slot adds
about 3.2 GiB, and your system and other programs need room beyond this.

A figure with "about" is an *estimate*: the engine's memory plan for that file, plus the 4 GiB by which the plan fell
short of IQ3_S's measured peak on one card. Q2_0's on two B60s is its measured expert store plus the 3.7 GiB the
other files added to theirs. UD-IQ4_XS's on two B60s is
[measured](measurements/MEASUREMENTS.md#2026-10-29-ud-iq4xs-speed) against IQ3_S's 21 GiB in the same test.

## 2. Download it

This downloads the model into `~/c0ven-models`, with the image encoder and the DFlash 2 drafter. Replace `IQ3_S` with
your file's name:

```bash
mkdir -p ~/c0ven-models
```

```bash
docker run --rm --user "$(id -u):$(id -g)" -v ~/c0ven-models:/models --entrypoint python ghcr.io/avra-m3/c0ven/serve:main -m tools.model_fetch IQ3_S
```

The download checks each file. If it stops, run the same command again and it carries on. When it finishes it prints
the `--model` value for the next step.

Keep the directory on a filesystem such as ext4 or xfs, which allows the direct reads the engine uses.
[MODEL_FILES.md](MODEL_FILES.md) lists every file, in case you want to fetch them yourself.

## 3. Start the server

Find the render group's ID on your machine:

```bash
stat -c %g /dev/dri/renderD128
```

Start the server. Replace `993` with your group ID and the file name after `--model` with the one the download printed:

```bash
docker run --rm --device /dev/dri --group-add 993 --ulimit memlock=-1:-1 --read-only -p 127.0.0.1:8080:8080 -v ~/c0ven-models:/models:ro -v c0ven-cache:/cache -e ENGINE_CACHE_DIR=/cache ghcr.io/avra-m3/c0ven/serve:main --model /models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --slots 1 --context 262144 --host 0.0.0.0 --port 8080
```

What the `docker run` flags do:

| Flag | Why |
| --- | --- |
| `--device /dev/dri`, `--group-add 993` | the cards, and the render group that may open them |
| `--ulimit memlock=-1:-1` | lets the engine lock its experts in RAM; a load stops without it |
| `--read-only` | the image writes nothing to its root filesystem; it may be left out |
| `-p 127.0.0.1:8080:8080` | the API, on this machine only. The server inside listens on `--host 0.0.0.0 --port 8080` |
| `-v ~/c0ven-models:/models:ro` | the model files, read-only |
| `-v c0ven-cache:/cache`, `-e ENGINE_CACHE_DIR=/cache` | keeps what a load works out, so the next start can reuse it |

The arguments after the image name are the server's [options](OPTIONS.md). To run it in the background, replace
`--rm` with `-d --name c0ven`. `docker logs -f c0ven` follows it, and `docker stop c0ven` stops it.

## 4. Send a request

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

Your tools can now use the server through its [OpenAI and Anthropic APIs](API.md).
