# ADR-0082: support Intel Arc Alchemist cards alongside Battlemage

**Status:** draft, 2026-11-26. Extends [ADR-0057](0057-the-engine-reads-the-host-it-runs-on.md),
[ADR-0065](0065-a-short-chunk-writes-through-a-mapping.md) and [ADR-0080](0080-peer-exchanges-fall-back-to-host-memory.md);
supersedes [ADR-0006](0006-scope.md) item 1, [ADR-0011](0011-cpu-gpu-signalling.md) item 2 and
[ADR-0056](0056-the-aim-is-the-fastest-performance-on-battlemage.md) item 4 in part.
**Revisit when:** Xe-HPG kernels show regressions on a card, or the `xe` driver aligns Alchemist dma-buf mappings and
2D copy pitches with Battlemage.

## Context

Intel Arc A-series (Alchemist, Xe-HPG, IP major 12: `12.55.8`, `12.56.5`, `12.57.0`) differs from Battlemage (Xe2, IP
major 20) in five ways that break unmodified Battlemage kernels and runtime paths:

- Alchemist's XMX matrix units use an 8-lane sub-group (`K = 16`, 8 rows by 8 columns per `dpas` instruction) instead of
  Battlemage's 16-lane form, and lack 2D block read and write builtins. Unannotated OpenCL kernels may also compile at
  SIMD8, which overflows sub-group reductions or exceeds thread counts at 256 GRFs per thread.
- EU polling of host-written completion flags (`cpu_wait` in `layer.cl`) can stall on Alchemist when host writes and GPU
  polling interact under long-running command queues.
- Mapped dma-buf staging buffers (`ze::Runtime::alloc_device_mapped`) may be write-back mapped on Alchemist, leaving CPU
  stores and readbacks in CPU caches instead of PCIe.
- Alchemist's 2D copy engine truncates large source pitches when `L.copy_rows` copies slices from the host expert arena
  into card staging buffers.
- Two-card consumer host bridges can pass `zeDeviceCanAccessPeer` while dropping EU stores to peer memory and faulting
  on copy-engine BAR-to-BAR copies.

## Decision

1. Build ahead of time for `bmg-g21;bmg-g31;acm-g10;acm-g11;acm-g12` in `ENGINE_AOT_DEVICES`, compiling Xe-HPG targets
   with `-DENGINE_HPG` and keeping a dedicated Xe-HPG SPIR-V fallback directory (`ENGINE_KERNEL_HPG_SPIRV_DIR`).
2. Keep shared kernels in single files via `kernels/include/xmx.h` (`xmx_xe2.h` for 16-lane XMX and `xmx_hpg.h` for
   8-lane XMX), add `REQD_SG` on non-XMX kernels whose SLM or work-group size assumes 16 lanes, and use per-target
   `SOURCE_HPG` sources in `engine_add_kernel_module` for 2D-block XMX kernels (`gemm_hpg.cl`, `prompt_attn_hpg.cl`
   sharing scalar rows with `prompt_attn_common.h`, and `vision_attn_hpg.cl`).
3. Gate Alchemist runtime behaviors on `ze::Card::is_hpg()` (`ip_version >> 22 == 12`) with environment overrides for
   testing:
   - `host_flag_polling = false` (`ENGINE_NO_HOST_POLL`): record regular decode command lists on one shared `ze::Queue`
     per card (`Card::step_queue`) split around each miss layer (`ze::List::split`), compute misses on the host after
     each router part finishes, and signal a Level Zero host event before `moe_combine`.
   - `writeback_mapping = true` (`ENGINE_STAGE_FLUSH`): flush CPU cache lines (`_mm_clflushopt` or `_mm_clflush` plus
     `_mm_sfence`) across written staging ranges, during the initial page-fault touch in `record_prompt_copy`, and
     before the ordering PCIe readback byte.
   - `copy_2d_arena = false` (`ENGINE_1D_STAGE_COPY`): return `tmax_` from `Model::fetch_rows()` so every prompt chunk
     stages uncached experts through the CPU mapping (`fetch_experts`) without recording `Model::prompt_copies`, keeping
     one-card `rare_experts` capped at `kFetchRows` and 1D `L.copy` transfers as the fallback when no CPU mapping
     exists.
   - `ze::check` marks the queue stuck (`ze::queue_stuck()`) on `ZE_RESULT_ERROR_DEVICE_LOST` so teardown exits without
     hanging on the lost context.
4. Run the `smoke` kernel peer-write check before the 4 KiB blitter copy check in `ze::Runtime::peer_access` so a board
   that drops peer stores selects host-memory exchanges ([ADR-0080](0080-peer-exchanges-fall-back-to-host-memory.md))
   before issuing peer copy-engine transfers.

## Options not chosen

- Preprocessor `#ifdef ENGINE_HPG` branches inside `gemm.cl`, `prompt_attn.cl` and `vision_attn.cl`. Separate
  `SOURCE_HPG` sources keep both Battlemage and Alchemist hot loops readable and free of conditional dead paths.
- Disabling P2P unconditionally for all Xe-HPG cards or disabling the vision encoder. Swapping the probe order keeps
  working two-card Alchemist configurations on peer exchanges while avoiding copy-engine faults on broken bridges, and
  `vision_attn_hpg.cl` supports image and video inputs natively.

## Consequences

- Both Battlemage and Alchemist GPUs build from one CMake configuration and run all model paths including sparse prompt
  attention, DFlash 2 and vision.
- Battlemage generated binaries, command lists and runtime paths remain identical unless an override environment variable
  is set.
