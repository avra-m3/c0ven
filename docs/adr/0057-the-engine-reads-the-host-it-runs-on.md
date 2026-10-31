# ADR-0057: the engine reads the host it runs on

**Status:** draft, 2026-10-31; not accepted.
**Revisit when:** a value the engine reads is wrong on a machine someone reports; or the engine gains a setting that
depends on the host and is not read.

## Context

- Several of the engine's numbers were the test machine's, written as constants: each card's usable memory, 16 CPU
  workers, the build's CPU features, two host memory thresholds, 8 loader threads and one kernel binary
  ([#21](https://github.com/avra-m3/c0ven/issues/21)).
- Where the engine reads the hardware it adapts: the core sizes the expert cache from each card's memory. Where it does
  not, another machine fails late or with no message. A cgroup memory limit under the arena kills the engine during
  its load; a CPU without AVX-512 stops on an illegal instruction at the first miss.
- [ADR-0006](0006-scope.md) item 1 named one machine and said to revisit when the machine changes.
  [ADR-0056](0056-the-aim-is-the-fastest-performance-on-battlemage.md) (draft) widens the scope to any number of
  Battlemage cards in one machine.

## Decision

1. **The engine reads the host before its memory plan, and never writes it.** `engine/host.py` reads the host's
   available memory, the cgroup's `memory.max`, `memory.swap.max` and `cpu.max`, `RLIMIT_MEMLOCK`, the CPUs the process
   may use and their cores, and the CPU's features.
2. **The plan takes each card's memory and the card count from the cards.** The core lists the cards before open
   (`engine_cards`, ABI version 11): name, memory, IP version, Xe cores and PCIe link. The smallest card bounds the
   plan, and one card holds every weight whole. The test machine's card is the plan's card only for the mock and the
   reference.
3. **Kernels are built for each chip in `ENGINE_AOT_DEVICES`** (`bmg-g21` and `bmg-g31` by default), each set in a
   directory named for the chip's IP version, and the core loads the set that matches each card. A card with no set
   compiles the modules' SPIR-V at load and logs that its chip is untested. `ENGINE_KERNEL_TARGET=spirv` forces that
   path on any card, for testing.
4. **The CPU's expert workers are one per core the process can keep busy:** the CPUs it may run on, cut to the
   cgroup's `cpu.max` and to their physical cores. `ENGINE_CPU_WORKERS` overrides the count. The core refuses to open
   on a CPU without a feature the ggml build uses, and names it. At the first load the workers read the arena once,
   and the fit report gives the rate.
5. **A plan that does not fit is refused before the load.** The message names the figure that does not fit and gives
   options that do. `--ram-max` sets a host memory limit of the user's own.
6. **Each load prints a fit report** beside its plan figures and the core's figures after the load, and sends it in
   INFO, so `/metrics` carries it.
7. **The test machine's constants stay as test fixtures** for the mock and the reference backends, which run without a
   card. The two B60s remain the machine every measurement is taken on.

## Options not chosen

- Constants with a flag for each: a user on another machine must first find out which constants are wrong, which is
  the failure this decision removes.
- Probing in the core only: the plan is Python's, and Python would then need every figure back through the ABI before
  it plans.

## Consequences

- The memory plan's two host thresholds go; the plan checks its host total against what the host can give.
- Inside a container only the container's own cgroup is visible. A limit on a parent cgroup is not read, and a load
  over it is still killed.
- A build compiles every kernel once for each chip, and the image holds each set.
- The plan's card figures for the draft layer and working memory remain estimates. The fit report notes a core
  expert count more than 5% from the plan's.
- A worker past the CPU quota waits for it, so under a quota the default is the quota's CPUs. The ggml build stays AVX-512 only;
  an AVX2 build chosen at run time would be a decision of its own.
- The storage and links slice adds its rows to the fit report ([#25](https://github.com/avra-m3/c0ven/issues/25)).
