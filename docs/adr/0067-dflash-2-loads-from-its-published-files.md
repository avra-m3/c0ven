# ADR-0067: DFlash 2 loads from its published files

**Status:** draft, 2026-11-08; not accepted. Supersedes items 1 and 6 of
[ADR-0050](0050-dflash-2-drafts.md) in part (what the draft directory holds and how it is fetched).
**Revisit when:** incoai publishes the drafter in another form, or a second drafter needs a format of its own.

## Context

- [ADR-0050](0050-dflash-2-drafts.md) has `tools/dflash_fetch.py` download the drafter's `model.safetensors` and
  `config.json`, write each tensor to a file of its own, and write `dflash-config.txt` and `dflash-manifest.json`. The
  core reads those, and takes the directory as DFlash 2's by the manifest.
- A deployment that fetches files with its own downloader has to run that tool, or copy its output, to get the
  drafter. The published files are enough: the safetensors header gives each tensor's dtype, shape and byte range, and
  `config.json` gives every value `dflash-config.txt` held.
- The serve image does not ship `docs/`, and with a DFlash 2 drafter the plan read `docs/model/mtp-manifest.json`
  for the draft ring's width, so 0.20.0 could not start with `dflash/` at all.

## Decision

1. The draft directory holds DFlash 2 as incoai publishes it: `model.safetensors` and `config.json`. The core takes
   a directory with both as DFlash 2's, and refuses one with `dflash-manifest.json` and without them.
2. The core reads each tensor by its byte range from the safetensors header, and the sizes, the mask token and the
   target layers from `config.json` (`core/util/json.h` reads both). The candidate selector's tensors are not read.
3. The memory plan reads the drafter's shapes from the same header. A served start reads nothing under `docs/`: the
   draft ring it counts for DFlash 2, or for no drafter, takes its width from the model file's attention layer, which
   the MTP layer's equals. Only the mock and the reference, which have no model file, read
   `docs/model/mtp-manifest.json`.
4. `tools/model_fetch.py` downloads the two files at the pinned revision and checks their size and sha256.
   `tools/dflash_fetch.py` is removed.

## Options not chosen

- Keeping the split layout beside the published one. Two layouts to read and test for the same bytes.
- Reading the files in the control plane and passing the tensors to the core. The C ABI would change, and the core
  already reads its other files itself.

## Consequences

- A `dflash/` fetched by `tools/dflash_fetch.py` stops the load with a message saying what to put there. The MTP
  layer's directory does not change.
- The directory is 1.2 GB against 0.9 GB: the file keeps the candidate selector's tensors.
