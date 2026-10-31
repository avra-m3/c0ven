# Model files

C0ven runs one model, Qwen3.8-Flash-Next, from ISTA-DASLab's GSQ-RCO IQ3_S GGUF by default
([ADR-0006](adr/0006-scope.md), [ADR-0018](adr/0018-one-file-iq3s.md)), or from
[another file](#the-other-quants) of the same model. This page says which files to get, how to
check them, how to lay them out, and how to point the engine at them.

| Part | Files | Size | Needed for |
| --- | --- | ---: | --- |
| The model | the IQ3_S GGUF, 2 shards | 83.6 GB | everything |
| The DFlash 2 drafter | 78 tensors from incoai's BF16 checkpoint, in `dflash/` | 0.9 GB | drafting; the engine prefers it to the MTP layer |
| The draft (MTP) layer | 31 tensors from Qwen's BF16 checkpoint, in `mtp/` | 5.2 GB | drafting without `dflash/`; with neither, the engine decodes one token a step |
| The image encoder | `mmproj-Qwen3.8-Flash-Next-BF16.gguf` | 0.9 GB | images and video in requests |

## The model directory

Put every file in one directory:

```
<model dir>/
  Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
  Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf
  mmproj-Qwen3.8-Flash-Next-BF16.gguf
  mtp/
    mtp-manifest.json
    mtp.fc_embedding.weight
    ... (one file per tensor, 31 in all)
```

Keep the shards' names: the engine finds each shard from the others by the `-0000N-of-0000M` suffix. Keep the
directory on a filesystem that allows `O_DIRECT` reads, such as ext4 or xfs. The engine reads the weights and the
n-gram table that way, and a filesystem that refuses them stops it ([ADR-0015](adr/0015-ngram-table-on-the-ssd.md),
[ADR-0032](adr/0032-sleep-the-front-end-stays-the-engine-keeps-the-arena.md)). The engine only reads the directory, so
it can be mounted read-only.

## Pointing the engine at the files

`--model` takes the path of any shard. From that path the engine finds the rest:

| Part | Where the engine looks | To change it |
| --- | --- | --- |
| The other shards | beside it, by name | - |
| The drafter | `dflash/` beside the model when it holds `dflash-manifest.json`, else `mtp/` when it holds `mtp-manifest.json` | `--engine-arg=--draft-dir=<dir>`; `--engine-arg=--no-draft` decodes without it |
| The image encoder | the one `mmproj-*.gguf` beside the model; with two or more, none | `--engine-arg=--vision=<file>`; `--engine-arg=--no-vision` reads no images |
| The tokenizer and chat template | the GGUF's metadata | `--tokenizer <tokenizer.json>`, `--chat-template <chat_template.jinja>` |

A missing drafter or encoder is not an error: the engine runs without it. The core takes a directory that holds
`dflash-manifest.json` as DFlash 2's and any other as the MTP layer's.

At each load the engine quantises the draft layer's experts from BF16, which takes tens of seconds. Set
`ENGINE_MTP_CACHE_DIR` to a writable directory to keep the result between loads. The cache is keyed by the bytes of
`mtp-manifest.json`, the expert type and the card split, and a file that does not match is written again
([DEVELOPING.md](DEVELOPING.md)).

With Docker, mount the model directory at `/models` and give a volume for the cache, as in the README's
[quick start](../README.md#quick-start):

```bash
docker run ... -v <model dir>:/models:ro -v c0ven-mtp:/cache -e ENGINE_MTP_CACHE_DIR=/cache/mtp \
  ghcr.io/avra-m3/c0ven/serve:main --model /models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf ...
```

The tests find the same directory through `FLASH_MODEL_DIR`, by default `/models`, and skip when it is missing
([TESTING.md](TESTING.md)).

## Getting the files

### The model

[`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
at revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09`, the revision Strata's `setup.py` pins. At that revision the
shards are in `IQ3_S/`:

```bash
base=https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/ed59f92082b1e93c0e96d60a8b11aab089b52f09/IQ3_S
wget -c "$base/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf" "$base/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf"
sha256sum Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-0000*-of-00002.gguf
```

| Shard | Bytes | sha256 (= the Hub's LFS oid) |
| --- | ---: | --- |
| `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` | 54,817,524,224 | `4c1eb2ceb4915e1192f4f386021897bde56a97f40a0bb78bb86465e0f7d2aca3` |
| `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` | 28,800,138,432 | `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113` |

The Hub gives the same values at `GET /api/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/<revision>/IQ3_S`
(`lfs.oid`, `lfs.size`). Together the shards are 83,617,662,656 B (83.6 GB).

#### The other quants

The same repository's `IQ3_XXS/`, `IQ2_XS/` and `Q2_0/` folders load too, at the same revision
([ADR-0038](adr/0038-the-other-quants-load.md), draft). Their second shard, the n-gram table, is byte for byte IQ3_S's
(the sha256 above), so a link to it under the quant's name serves. Put the image encoder's file and `mtp/` beside them
as for IQ3_S; without the encoder's file the engine leaves room for more experts and its speed is not comparable.
Their speed and agreement with IQ3_S are in [measurements/MEASUREMENTS.md](measurements/MEASUREMENTS.md#other-quants).

| First shard | Bytes | sha256 |
| --- | ---: | --- |
| `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf` | 47,039,860,096 | `219ea929900dfa9ef091f3aa473fdba6874b65fcb36526d7d851ac9e95856d15` |
| `Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf` | 39,225,954,592 | `92cee27ae5bbadcd732416a0f7a7f0acc092399dbbe8f5a5efa707c2ec0a49d7` |
| `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf` | 37,623,740,192 | `69820c02ec7d0b45ef2ebb19d6620299db749fe2aded7f39f93c6b88b199b720` |

#### Unsloth's UD-IQ4_XS

[`unsloth/Qwen3.8-Flash-Next-GGUF`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) at revision
`766911a6b7369840a91dbcd95f9f997acaab6cd6`, the three shards in `UD-IQ4_XS/`. It is 10.1 GB larger than IQ3_S and
closer to the BF16 checkpoint ([CHOOSING_A_MODEL.md](CHOOSING_A_MODEL.md)). The engine's tests, references and the
README's figures stay on IQ3_S. No other file from that repository has been loaded
([below](#unsloths-other-files)). Put the
image encoder's file, `dflash/` and `mtp/` beside it as for IQ3_S; Unsloth's own MTP files are not read.

```bash
base=https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/766911a6b7369840a91dbcd95f9f997acaab6cd6/UD-IQ4_XS
wget -c "$base/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf" "$base/Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf" "$base/Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00003.gguf"
sha256sum Qwen3.8-Flash-Next-UD-IQ4_XS-0000*-of-00003.gguf
```

| Shard | Bytes | sha256 (= the Hub's LFS oid) |
| --- | ---: | --- |
| `Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf` | 10,946,624 | `5ce89370720f8bf90890f439361282104c1aa1482d4013bb9a50923e758e71a4` |
| `Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf` | 49,835,229,856 | `577a38a2392b40ca2193cea502e1d92f60b8cd370675d308e0ec21885d9daaa7` |
| `Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00003.gguf` | 43,836,407,744 | `d4634e6d84f0ebb0940be15c90d3790bf6464e3dea3a1cddc567dc0e83ad8833` |

Together the shards are 93,682,584,224 B (93.7 GB). Every tensor and its type is in
[tensors-UD-IQ4_XS.tsv](model/tensors-UD-IQ4_XS.tsv).

#### Unsloth's other files

The engine has kernels for every type in the repository's other folders at the same revision, but none of the files
has been loaded. The kernels for the expert types only these files use (IQ1_S and Q5_1, and Q4_K, Q5_K, Q6_K and Q8_0
for gate and up) are not yet tested. The experts' types, read from the files' headers, with the number of layers
(gate and up have the same type in each layer):

| Folder | Gate and up | Down |
| --- | --- | --- |
| `UD-IQ1_S/` | IQ1_S 34, IQ2_XXS 14 | IQ4_NL |
| `UD-IQ1_M/` | IQ1_M 24, IQ2_XXS 24 | IQ4_NL |
| `UD-Q2_K_XL/` | IQ2_XS 47, IQ3_XXS 1 | IQ4_NL |
| `UD-IQ3_XXS/` | IQ2_S 47, IQ3_S 1 | IQ4_NL |
| `UD-Q3_K_XL/` | IQ3_XXS 47, IQ4_XS 1 | IQ4_NL 43, Q8_0 5 |
| `UD-Q4_K_XL/` | Q4_K 47, Q5_K 1 | Q5_1 43, Q8_0 5 |
| `UD-Q5_K_XL/` | Q5_K 47, Q6_K 1 | Q8_0 |
| `UD-Q6_K_XL/` | Q6_K 47, Q8_0 1 | Q8_0 |
| `Q8_0/` | Q8_0 | Q8_0 |

Their other tensors are Q4_K, Q5_K, Q6_K, Q8_0, IQ4_NL, BF16 and F32, which the engine reads. `Q8_0/` also stores
`ssm_alpha`, `ssm_beta` and the hyper-connections' inject weights as Q8_0, where the other files have F32; the engine
holds those weights as BF16 and converts them from either type.

### The image encoder

The same repository at revision `1c04b8102ca5346f1faf4d9914503e378d713021`
([ADR-0031](adr/0031-images-through-the-prompt-path.md)), at its top level:

```bash
wget -c https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/1c04b8102ca5346f1faf4d9914503e378d713021/mmproj-Qwen3.8-Flash-Next-BF16.gguf
```

It is 907,543,008 B, with sha256 `b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0`. What it holds
is in [spec/MODEL_VISION.md](spec/MODEL_VISION.md).

### The MTP layer

The GSQ-RCO GGUF has no draft (MTP) head: it has no `mtp` or `nextn` tensor. The head comes
from Qwen's BF16 checkpoint, [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
at revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`, as in Strata's `tools/mtp_fetch.py` (at `1678de3`). The
checkpoint is 131 safetensors shards (360 GB), and its 31 `mtp.*` tensors are spread over 28 of them. Range requests
fetch only those 5.2 GB:

1. Read `model.safetensors.index.json` at the revision, and keep the names that start with `mtp.`.
2. For each shard that holds one, request its first 8 bytes (the header's length), then the JSON header.
3. For each tensor, request its byte range: `8 + header length + data_offsets`. Check that each answer is `206`,
   with a `Content-Range` equal to the range asked for.
4. Write each tensor's bytes, as they are, to `mtp/<tensor name>`: raw little-endian BF16, with no header.
5. Check each file's sha256 against the sums in Strata's `tools/mtp_fetch.py` at `1678de3`, which are for the same
   revision. The Hub's sha256 covers whole shards, so it cannot check a byte range.
6. Write `mtp/mtp-manifest.json`, listing for each tensor its dtype, shape, source shard, byte range and sha256.
   The engine needs the file to find the directory, and keys its quantised cache by the file's bytes.

### The DFlash 2 drafter

DFlash 2 is a separate small model trained to draft for Qwen3.8-Flash-Next: five dense layers that read the main
model's hidden states and propose up to 7 tokens in one pass. It comes from
[`incoai/Qwen3.8-Flash-Next-DFlash2`](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2) at revision
`a22a1744e85c17364dcc54a6a420af7135d4c7f6`, one file, `model.safetensors`, of 1,206,140,008 B with sha256
`d6723f2a359eaf33ec0c95f22ae8edda896774020c552296a85e1913b4d3186f`. From a c0ven checkout:

```bash
python -m tools.dflash_fetch --out <model dir>/dflash
```

The tool downloads the file, checks its size and sha256, writes each tensor's bytes, as they are, to
`dflash/<tensor name>` (raw little-endian BF16, no header), and writes `dflash/dflash-manifest.json` last, with each
tensor's dtype, shape, byte range and sha256. It leaves out the tensors whose names start with `candidate_selector.`:
the engine does not use the checkpoint's candidate selector. The temporary download is deleted.

The engine reads the drafter's shape from its code, not from `config.json`: hidden size 2,560, 32 query and 8 KV
heads of 128, MLP 7,680, RMS norms with eps 1e-6 and no +1, RoPE base 10,000, a block of 8 rows, mask token 248,077,
and the main model's layers 5, 15, 25, 35 and 45 as its input (`core/models/qwen4exp/layout.h`, `model_impl.h`).
