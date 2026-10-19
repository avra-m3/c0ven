# Model files

C0ven runs one model, Qwen3.8-Flash-Next, from ISTA-DASLab's GSQ-RCO IQ3_S GGUF
([ADR-0006](adr/0006-scope.md), [ADR-0018](adr/0018-one-file-iq3s.md)). This page says which files to get, how to
check them, how to lay them out, and how to point the engine at them. Its second half lists what the files hold.

| Part | Files | Size | Needed for |
| --- | --- | ---: | --- |
| The model | the IQ3_S GGUF, 2 shards | 83.6 GB | everything |
| The draft (MTP) layer | 31 tensors from Qwen's BF16 checkpoint, in `mtp/` | 5.2 GB | drafting; without it the engine decodes one token a step |
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

Keep the shards' names: the engine finds each shard from the other by the `-0000N-of-00002` suffix. Keep the
directory on a filesystem that allows `O_DIRECT` reads, such as ext4 or xfs. The engine reads the weights and the
n-gram table that way, and a filesystem that refuses them stops it ([ADR-0015](adr/0015-ngram-table-on-the-ssd.md),
[ADR-0032](adr/0032-sleep-the-front-end-stays-the-engine-keeps-the-arena.md)). The engine only reads the directory, so
it can be mounted read-only.

## Pointing the engine at the files

`--model` takes the path of either shard. From that path the engine finds the rest:

| Part | Where the engine looks | To change it |
| --- | --- | --- |
| The other shard | beside it, by name | - |
| The draft layer | `mtp/` beside the model, when it holds `mtp-manifest.json` | `--engine-arg=--draft-dir=<dir>`; `--engine-arg=--no-draft` decodes without it |
| The image encoder | the one `mmproj-*.gguf` beside the model; with two or more, none | `--engine-arg=--vision=<file>`; `--engine-arg=--no-vision` reads no images |
| The tokenizer and chat template | the GGUF's metadata ([Metadata keys](#metadata-keys)) | `--tokenizer <tokenizer.json>`, `--chat-template <chat_template.jinja>` |

A missing draft layer or encoder is not an error: the engine runs without it.

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
Their speed and agreement with IQ3_S are in [MEASUREMENTS.md](MEASUREMENTS.md#other-quants).

| First shard | Bytes | sha256 |
| --- | ---: | --- |
| `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf` | 47,039,860,096 | `219ea929900dfa9ef091f3aa473fdba6874b65fcb36526d7d851ac9e95856d15` |
| `Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf` | 39,225,954,592 | `92cee27ae5bbadcd732416a0f7a7f0acc092399dbbe8f5a5efa707c2ec0a49d7` |
| `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf` | 37,623,740,192 | `69820c02ec7d0b45ef2ebb19d6620299db749fe2aded7f39f93c6b88b199b720` |

### The image encoder

The same repository at revision `1c04b8102ca5346f1faf4d9914503e378d713021`
([ADR-0031](adr/0031-images-through-the-prompt-path.md)), at its top level:

```bash
wget -c https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/1c04b8102ca5346f1faf4d9914503e378d713021/mmproj-Qwen3.8-Flash-Next-BF16.gguf
```

It is 907,543,008 B, with sha256 `b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0`. What it holds
is in [spec/MODEL_VISION.md](spec/MODEL_VISION.md).

### The MTP layer

The GSQ-RCO GGUF has no draft (MTP) head: it has no `mtp` or `nextn` tensor
([What the file holds](#what-the-file-holds)). The head comes from Qwen's BF16 checkpoint, [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
at revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`, as in Strata's `tools/mtp_fetch.py` (at `1678de3`). The
checkpoint is 131 safetensors shards (360 GB), and its 31 `mtp.*` tensors are spread over 28 of them. Range requests
fetch only those 5.2 GB:

1. Read `model.safetensors.index.json` at the revision, and keep the names that start with `mtp.`.
2. For each shard that holds one, request its first 8 bytes (the header's length), then the JSON header.
3. For each tensor, request its byte range: `8 + header length + data_offsets`. Check that each answer is `206`,
   with a `Content-Range` equal to the range asked for.
4. Write each tensor's bytes, as they are, to `mtp/<tensor name>`: raw little-endian BF16, with no header.
5. Check each file's sha256 against the table below. The Hub's sha256 covers whole shards, so it cannot check a
   byte range. The table's sums are those in Strata's `tools/mtp_fetch.py` at `1678de3`, for the same revision.
6. Write `mtp/mtp-manifest.json`, listing for each tensor its dtype, shape, source shard, byte range and sha256.
   The engine needs the file to find the directory, and keys its quantised cache by the file's bytes.

Every tensor is BF16. The total is 5,214,301,696 B (5.21 GB), of which the experts take 5.03 GB.

| Tensor | Shape | Bytes | Source shard | Offset in shard | sha256 |
| --- | --- | ---: | --- | ---: | --- |
| `mtp.fc_embedding.weight` | 2560 x 2560 | 13,107,200 | `model-00120-of-00131` | 1,935,412,640 | `241d6e87ded317a61b1aeb3ca4e28c53f1caba09ad5be66092e91fe5e3e14751` |
| `mtp.fc_hidden.weight` | 2560 x 2560 | 13,107,200 | `model-00124-of-00131` | 1,716,599,928 | `97c8e427ad0177e2c713ed24a397d815b523eda15ef453f1ae6463f6c70a8f96` |
| `mtp.hyper_connection_mixer.hc_norm.weight` | 10240 | 20,480 | `model-00037-of-00131` | 1,742,326,680 | `ce948a398e15aeaa403c424cda34e6eed296ab2570d591bb5ff3da642b255d64` |
| `mtp.hyper_connection_mixer.input_mix_weight_down.weight` | 320 x 10240 | 6,553,600 | `model-00041-of-00131` | 1,935,412,688 | `c633d2841fda463e82045e0b93fb8cb8d511ec0c9eda616020f01aca18d71947` |
| `mtp.hyper_connection_mixer.input_mix_weight_up.weight` | 10240 x 320 | 6,553,600 | `model-00043-of-00131` | 1,832,518,416 | `6565285f0ba2923703252763cfc11d8a9bda40ab42e7548e12464b04dbdbec13` |
| `mtp.layers.0.attn_hyper_connection.block_inject_weight.weight` | 4 x 10240 | 81,920 | `model-00047-of-00131` | 1,716,599,984 | `c8afff64e7f54412d02120097c4255033c55245e1c1eed84cc630d428ccb864c` |
| `mtp.layers.0.attn_hyper_connection.hc_norm.weight` | 10240 | 20,480 | `model-00049-of-00131` | 1,935,412,672 | `1fb619aeaaf637ccd70f46f1f8f8116e6a46448cb43026b246de8ee8286ed809` |
| `mtp.layers.0.attn_hyper_connection.input_mix_weight_down.weight` | 320 x 10240 | 6,553,600 | `model-00053-of-00131` | 1,832,518,424 | `e80ac8bf3e270e9d5a949ec7c913329dc8de017d69851334820fae7b8c86a7f5` |
| `mtp.layers.0.attn_hyper_connection.input_mix_weight_up.weight` | 10240 x 320 | 6,553,600 | `model-00057-of-00131` | 1,935,412,680 | `f80f1fd44e1183e7764ea496b11ff89ead9dd758f861d4e79a770bf54956b1a5` |
| `mtp.layers.0.mlp.experts.down_proj` | 512 x 2560 x 640 | 1,677,721,600 | `model-00062-of-00131` | 1,832,518,400 | `539c46234f73bac70450f731f40e715f77ac32faf77def0f3f800e58aaabbbc0` |
| `mtp.layers.0.mlp.experts.gate_up_proj` | 512 x 1280 x 2560 | 3,355,443,200 | `model-00060-of-00131` | 115,918,328 | `839e98f9642ebc884b178abd2090b6dc8b0bcd81ed92d7f7a08314c6a68f96d5` |
| `mtp.layers.0.mlp.gate.weight` | 512 x 2560 | 2,621,440 | `model-00066-of-00131` | 1,716,599,952 | `f0d35fa41b360a08bc0c5b1e5766aa42df960291716ee189208b5d2599462c20` |
| `mtp.layers.0.mlp.shared_expert.down_proj.weight` | 2560 x 640 | 3,276,800 | `model-00076-of-00131` | 1,935,412,680 | `d2ced8c81367cc494599d1a8cdc279c3dd6579b628ded05acd7495e99cee01d8` |
| `mtp.layers.0.mlp.shared_expert.gate_proj.weight` | 640 x 2560 | 3,276,800 | `model-00072-of-00131` | 1,832,518,528 | `f4a81109165853b3b64f36d809f98094b599763364070f5fa079200e1c386d4c` |
| `mtp.layers.0.mlp.shared_expert.up_proj.weight` | 640 x 2560 | 3,276,800 | `model-00072-of-00131` | 1,835,795,328 | `172586861eea53efbff10afc43ded5b6e49c83879e9efc01e089c652bbb0a156` |
| `mtp.layers.0.mlp.shared_expert_gate.weight` | 1 x 2560 | 5,120 | `model-00070-of-00131` | 1,832,518,400 | `acdd306d9996d356e4667a16ff63af6fef086fc7f570f1bc00bbe882a2a93f6e` |
| `mtp.layers.0.mlp_hyper_connection.block_inject_weight.weight` | 4 x 10240 | 81,920 | `model-00078-of-00131` | 1,832,518,424 | `c5782dd92b479a1f8ef43d683fe9c917fd9e8fba81f23c1aae8817bfe4b476a6` |
| `mtp.layers.0.mlp_hyper_connection.hc_norm.weight` | 10240 | 20,480 | `model-00082-of-00131` | 1,935,412,656 | `21a504876b5e7b788301557c68dfb757999430eba36c66ec8ab96a63604dbfd1` |
| `mtp.layers.0.mlp_hyper_connection.input_mix_weight_down.weight` | 320 x 10240 | 6,553,600 | `model-00086-of-00131` | 1,935,412,696 | `b71240036dea954e034d5ae04e097bcd7ac51cb28a44e4c93d198af3447278a1` |
| `mtp.layers.0.mlp_hyper_connection.input_mix_weight_up.weight` | 10240 x 320 | 6,553,600 | `model-00088-of-00131` | 1,832,518,424 | `40756866bf2a5e82c28e407f717b6fa07614d593f78e79c7ceb2cd4ffe6dda94` |
| `mtp.layers.0.self_attn.indexer.index_qk_proj.weight` | 640 x 2560 | 3,276,800 | `model-00090-of-00131` | 1,832,518,416 | `2befafae6e944c1fc651fd73d4d527eb6f138a6c1679065b003128ee3150e859` |
| `mtp.layers.0.self_attn.indexer.k_layernorm.weight` | 128 | 256 | `model-00094-of-00131` | 1,935,412,672 | `2db621f6a08b4085756fc69d60a4a1bd1d7fbf5d406456d2477075842efaae54` |
| `mtp.layers.0.self_attn.indexer.q_layernorm.weight` | 128 | 256 | `model-00100-of-00131` | 1,716,599,968 | `afaf59f68e251939970ffdeacad0667dffc657a522bea62cdc44a52816fc49d6` |
| `mtp.layers.0.self_attn.k_norm.weight` | 256 | 512 | `model-00102-of-00131` | 1,935,412,648 | `8f924b0357cf936e14fc34518205d000e1423aa60626827df81b0804fae8cdfd` |
| `mtp.layers.0.self_attn.k_proj.weight` | 512 x 2560 | 2,621,440 | `model-00108-of-00131` | 1,832,518,624 | `1a295dc2d6e633e808aeb19e00c97856a1817bbfcc83f5b610e2bfcacfcb8c38` |
| `mtp.layers.0.self_attn.o_proj.weight` | 2560 x 6144 | 31,457,280 | `model-00106-of-00131` | 1,832,518,400 | `9d2fcf0ac6ac19de87f638f710cfd1af6ad14141e75b7d3d5d29fbd06f374219` |
| `mtp.layers.0.self_attn.q_norm.weight` | 256 | 512 | `model-00112-of-00131` | 1,935,412,664 | `57957aa1e7c5a3ac628bff0ec5522ad55a39b8c0b8bd2d324ae640e3771617ef` |
| `mtp.layers.0.self_attn.q_proj.weight` | 12288 x 2560 | 62,914,560 | `model-00108-of-00131` | 1,835,140,064 | `5220aef3d288f40759f7356af9ef453e5ac0256a38749a9bb803a4e9db58150d` |
| `mtp.layers.0.self_attn.v_proj.weight` | 512 x 2560 | 2,621,440 | `model-00108-of-00131` | 1,898,054,624 | `39cb12f5424e965fa71fb81db0ee230c701b7b1e0b72c36641fde7e0e41bff41` |
| `mtp.pre_fc_norm_embedding.weight` | 2560 | 5,120 | `model-00114-of-00131` | 1,832,518,392 | `04c4a570850e06f2d8913da8220d54d4c7f87db6eb6d45480b938e8ba41d6a86` |
| `mtp.pre_fc_norm_hidden.weight` | 10240 | 20,480 | `model-00118-of-00131` | 1,716,599,944 | `b933b44559879449b1baf2af8385c5c237347307e3fd452394870cb53e6eab6d` |


## What the file holds

The tensor census of the IQ3_S file. [model/tensors-IQ3_S.tsv](model/tensors-IQ3_S.tsv) lists every tensor, one row
each. It was read with llama.cpp's `gguf-py` at commit `1d8de7c1b`, which knows type 42 (Q2_0,
`gguf-py/gguf/constants.py`), in `python:3.12-slim` with numpy and the files mounted read-only:

```
python gguf-py/gguf/scripts/gguf_dump.py --no-tensors <shard>     # metadata
python gguf-py/gguf/scripts/gguf_dump.py --data-offset <shard>    # start of the data section
```

The per-tensor rows come from `gguf.GGUFReader`: `name`, `tensor_type`, `shape`, `n_bytes` and `data_offset`.
`data_offset` is absolute: the data section's start plus the tensor's own offset.

### The two shards

| | Shard 1 | Shard 2 |
| --- | ---: | ---: |
| GGUF version | 3 | 3 |
| Metadata keys | 67 | 3 (`split.*` only) |
| Tensors | 1,223 | 1 (`per_layer_token_embd.weight`, the n-gram table) |
| Data section starts at | 11,024,384 | 192 |
| Alignment | 32 | 32 |
| Sum of tensor bytes | 54,806,499,840 | 28,800,138,240 |
| Header + tensor bytes | 54,817,524,224 | 28,800,138,432 |
| File size | 54,817,524,224 | 28,800,138,432 |
| Difference | 0 | 0 |

The last tensor in each shard ends exactly at the end of the file. There is no padding between tensors: every size is
a multiple of 32.

The n-gram table (`per_layer_token_embd.weight`) is IQ4_NL, shape 160 x 320,001,536, 28,800,138,240 B. Its data
starts at byte 192 of shard 2 and ends at 28,800,138,432, the shard's last byte. One row is 160 values,
90 B in IQ4_NL (5 blocks of 18 B), so row `r` lies at `192 + 90 r`.

### Metadata keys

Shard 1, in file order. Arrays give their length; the long ones are not shown in full.

- `general.*`: `architecture` = `qwen4exp`, `type`, `sampling.top_k` = 20, `sampling.top_p` = 0.95,
  `sampling.temp` = 1.0, `name`, `description`, `size_label` = `512x56B`, `quantized_by`, `repo_url`,
  `file_type` = 26, `quantization_version` = 2, `tags` (7).
- `qwen4exp.*`:
  - Sizes: `block_count` 48, `context_length` 262,144, `embedding_length` 2,560,
    `embedding_length_per_layer_input` 160.
  - Attention: `attention.head_count` 24, `attention.head_count_kv` 2, `attention.key_length` 256,
    `attention.value_length` 256, `attention.layer_norm_rms_epsilon` 1e-6, `attention.compress_ratios` [48].
  - RoPE: `rope.dimension_sections` [11, 11, 10, 0], `rope.freq_base` 1e7, `rope.dimension_count` 64.
  - Experts: `expert_count` 512, `expert_used_count` 10, `expert_feed_forward_length` 640,
    `expert_shared_feed_forward_length` 640.
  - Recurrent layers: `ssm.conv_kernel` 4, `ssm.state_size` 128, `ssm.group_count` 16, `ssm.time_step_rank` 48,
    `ssm.inner_size` 6,144, `full_attention_interval` 4.
  - Hyper-connections: `hyper_connection.count` 4, `hyper_connection.low_rank` 320.
  - Indexer: `attention.indexer.head_count` 4, `attention.indexer.key_length` 128, `attention.indexer.top_k` 2,048.
  - PLE: `ple.layers` [1] = [1], `ple.ngram_size` 3, `ple.heads_per_ngram` 8, `ple.conv_kernel` 4,
    `ple.eos_token_id` 248,044, `ple.layer_multipliers` [3], `ple.head_offsets` [16], `ple.head_vocab_sizes` [16],
    `ple.image_token_id` 248,056.
- `tokenizer.*`: `ggml.model` = `gpt2`, `ggml.pre` = `qwen35`, `ggml.tokens` [248,320], `ggml.token_type`
  [248,320], `ggml.merges` [247,587], `ggml.eos_token_id` 248,046, `ggml.padding_token_id` 248,044,
  `ggml.bos_token_id` 248,044, `ggml.add_bos_token` false, `chat_template` (a Jinja string).
- `quantize.imatrix.*`: `file`, `dataset`, `entries_count` 926, `chunks_count` 1,000.
- `split.*`: `no` 0, `count` 2, `tensors.count` 1,224.

Shard 2 has `split.no` 1, `split.count` 2 and `split.tensors.count` 1,224.

### Bytes per type and part

Parts by tensor name. The patterns are tried in order:

| Part | Pattern |
| --- | --- |
| experts | `blk.N.ffn_(gate\|up\|down)_exps.weight` (the routed experts, 512 per tensor) |
| n-gram table | `per_layer_token_embd.weight` |
| embedding | `token_embd.weight` |
| head | `output*` (`output.weight`, `output_hc_down`, `output_hc_norm`, `output_hc_up`) |
| PLE | `blk.N.ple_*` (layer 1 only) |
| dense | every other `blk.N.*`: attention, indexer, recurrent (`ssm_*`), router (`ffn_gate_inp*`), shared expert (`*_shexp`), hyper-connections (`hc_*`), norms |
| other | anything else: none |

| Part | Type | Tensors | Bytes |
| --- | --- | ---: | ---: |
| experts | IQ2_S | 40 | 10,747,904,000 |
| experts | IQ3_XXS | 34 | 10,918,297,600 |
| experts | IQ3_S | 20 | 7,208,960,000 |
| experts | IQ4_XS | 2 | 891,289,600 |
| experts | IQ4_NL | 39 | 18,402,508,800 |
| experts | Q2_0 | 9 | 2,123,366,400 |
| **experts** | | **144** | **50,292,326,400** |
| dense | BF16 | 480 | 1,449,246,720 |
| dense | Q6_K | 128 | 1,356,364,800 |
| dense | Q5_K | 35 | 346,931,200 |
| dense | Q4_K | 47 | 198,696,960 |
| dense | IQ4_XS | 42 | 169,902,080 |
| dense | IQ4_NL | 47 | 43,315,200 |
| dense | F32 | 288 | 9,899,520 |
| dense | Q8_0 | 1 | 1,740,800 |
| **dense** | | **1,068** | **3,576,097,280** |
| head | Q6_K, BF16, F32 | 4 | 534,620,160 |
| embedding | IQ4_XS | 1 | 337,715,200 |
| PLE | BF16, F16, F32 | 6 | 65,740,800 |
| **shard 1** | | **1,223** | **54,806,499,840** |
| n-gram table (shard 2) | IQ4_NL | 1 | 28,800,138,240 |
| **both shards** | | **1,224** | **83,606,638,080** |

The dense part splits further, by name pattern:

| Dense tensors | Bytes |
| --- | ---: |
| attention (`attn_*`) | 1,527,474,176 |
| hyper-connections (`hc_*`) | 1,270,087,680 |
| recurrent (`ssm_*`) | 467,590,656 |
| shared expert (`*_shexp`) | 145,536,000 |
| router (`ffn_gate_inp*`) | 126,074,880 |
| indexer (`indexer.*`) | 39,333,888 |
| **dense** | **3,576,097,280** |

The same totals by type, both shards:

| Type | ggml id | Tensors | Bytes |
| --- | ---: | ---: | ---: |
| IQ4_NL | 20 | 87 | 47,245,962,240 |
| IQ3_XXS | 18 | 34 | 10,918,297,600 |
| IQ2_S | 22 | 40 | 10,747,904,000 |
| IQ3_S | 21 | 20 | 7,208,960,000 |
| Q2_0 | 42 | 9 | 2,123,366,400 |
| Q6_K | 14 | 129 | 1,877,836,800 |
| BF16 | 30 | 484 | 1,527,889,920 |
| IQ4_XS | 23 | 45 | 1,398,906,880 |
| Q5_K | 13 | 35 | 346,931,200 |
| Q4_K | 12 | 47 | 198,696,960 |
| F32 | 0 | 292 | 10,063,360 |
| Q8_0 | 8 | 1 | 1,740,800 |
| F16 | 1 | 1 | 81,920 |

### The routed experts, layer by layer

Each layer has three expert tensors: gate and up are 2,560 x 640 x 512, and down is 640 x 2,560 x 512. Gate and up
always share a type.

| Gate and up | Layers |
| --- | --- |
| IQ2_S (20) | 1-6, 8-11, 13, 15, 19, 23, 24, 26, 32, 34, 36, 39 |
| IQ3_XXS (17) | 0, 7, 12, 14, 16, 18, 20, 22, 30, 31, 33, 37, 38, 41, 42, 43, 46 |
| IQ3_S (10) | 17, 21, 25, 27, 28, 29, 35, 40, 44, 45 |
| IQ4_XS (1) | 47 |

| Down | Layers |
| --- | --- |
| Q2_0 (9) | 1, 8, 10, 12, 13, 20, 21, 30, 36 |
| IQ4_NL (39) | the other 39 |

### Against the allocation file

The Hub repository's `tensor-allocation/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.rco-allocation.txt`, at the
same revision, lists 1,223 tensors and a type for each. The census matches it exactly:

- No tensor is in one list and not the other.
- No tensor's type differs.

The allocation file covers shard 1 only, so it does not list the n-gram table. Its type counts line
(`BF16=484, F16=1, F32=292, IQ2_S=40, IQ3_S=20, IQ3_XXS=34, IQ4_NL=86, IQ4_XS=45, Q2_0=9, Q4_K=47, Q5_K=35,
Q6_K=129, Q8_0=1`) equals the census less the n-gram table. The census agrees with the expert table in
[ADR-0018](adr/0018-one-file-iq3s.md).
