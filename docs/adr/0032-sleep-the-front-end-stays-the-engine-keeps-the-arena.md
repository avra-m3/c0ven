# ADR-0032: sleep: the front end stays, the engine gives the cards back and keeps the arena

**Status:** accepted, 2026-10-17, provisional: a sleep and a wake end to end beside another model on the cards, and
the eviction they may cause, confirm or revise it. Its numbers are from the
[sleep and wake record](../measurements/MEASUREMENTS.md#2026-10-17-sleep-and-wake).
**Revisit when:** a wake swaps the host's other memory out or takes far longer than its estimate; Level Zero can
import host memory that both cards read; or the arena's RAM is needed back while the engine sleeps.

## Context

The engine must give both cards back so that another model can run on them. Three ways exist:

1. a proxy such as llama-swap stops the front end and the engine;
2. the front end stays up, keeps the model's port, and the proxy asks it to sleep and wake;
3. the engine keeps its processes and gives back memory through Level Zero's virtual memory calls.

While asleep the arena can stay in host memory, so that a wake reads only the cards' part of the model and does not
pin 20.49 GiB again on a host that another model may have left fragmented.

If the engine child exits on sleep, something else must hold the arena. The front end could hold it as shared memory
that each child imports, but that import does not work: anonymous memory imported into Level Zero serves only the
context's first card, and a card that reads imported memory it cannot see resets with a lost device
([host memory record](../measurements/MEASUREMENTS.md#2026-10-12-host-memory), Q6). The arena has to stay the memory
of the process that loaded it: a `zeMemAllocHost` allocation of its context or, with `Options::anon_arena`, its own
anonymous memory.

An idle process that holds the runtime and every kernel keeps 200 MiB of VRAM a card and 127-129 MiB of RSS, and two
processes can share the cards ([VRAM record](../measurements/MEASUREMENTS.md#2026-10-16-vram-give-back)). Recording
every command list again takes 0.18-0.20 s ([load record](../measurements/MEASUREMENTS.md#2026-10-16-load-phases)).

A load that reads the GGUF through its mapping fills the page cache and pushes the host's other memory into swap,
and most of its time is one thread copying the arena out of the mapped file (load record). `O_DIRECT` reads the
first shard at 6.49-6.64 GB/s from 8 threads of 1 MiB, and `to_runs` rearranges the experts at 12.26-13.04 GB/s on
16 threads, so the CPU does not limit a load that rearranges what it reads
([load rates record](../measurements/MEASUREMENTS.md#2026-10-16-load-read-rates)).

## Decision

1. The front end runs as a service of its own, outside the proxy, and keeps the model's port. It starts the engine
   as its child. It serves `POST /admin/wake`, `POST /admin/sleep` and `POST /admin/lease` on an admin port of their
   own, which only the host's loopback reaches unless `--admin-host` says otherwise.
2. Asleep, the engine child stays alive and gives back only the cards: every device allocation, the image
   encoder's included, its recorded lists, and the checkpoints. It keeps its Level Zero context, its kernels, its host
   buffers, the encoder's 16-bit weights and the arena. The core's ABI, version 4, has `engine_sleep` and
   `engine_wake`; the line protocol has `SLEEP [<drain_ms>]`, `WAKE` and `ASLEEP <ms>`.
3. A wake loads the cards' part of the model again, with the experts they cached when they slept, swaps included, and
   the encoder's part, records the lists and pins the checkpoints. The arena is filled once, by the first load. With
   `--sleep-mode exit` the child exits instead, which gives the arena's RAM back and makes the next wake a full load.
4. Three guards hold the engine to the proxy's view. `/health` answers 503 unless the engine is ready. From a
   sleep's start until a wake's `READY`, every request gets 503 before it reaches the engine, so no request wakes it.
   A lease fails closed: once the last lease's connection ends, for any reason, the front end sleeps on its own.
5. A sleep refuses the queued requests with `ERR`, lets the running ones go on for up to its drain, and ends the rest
   with finish `cancel`. The drain is `--drain-timeout`, 30 s by default, and the front end's own shutdown waits no
   longer.
6. With llama-swap, the model's entry proxies to the front end's port and keeps `checkEndpoint: /health`. Its `cmd`
   holds a lease, `POST /admin/lease`, until llama-swap stops it; its `cmdStop` calls `POST /admin/sleep` and returns
   once the engine sleeps. `python -m frontend.lease` does both, and `curl` does too.
7. The core reads the weights with `O_DIRECT` and never through the page cache. Each card's dense tensors are read
   whole into staging, a layer at a time. The experts are read one at a time into a staging buffer of two batches
   of 64, by 8 threads, while the CPU threads rearrange the batch before for both cards and copy the arena's experts
   into their pieces. The draft layer's files are dropped from the page cache once read. The GGUF stays mapped for
   the token embedding, which each step reads, and for the swaps, which read experts from it while the engine runs.
   The encoder's file is read through its mapping: its 16-bit weights once, into the host memory a sleep keeps, and
   its 32-bit tensors at each load, for the cards.
8. Each load reports its seconds and the host's `pswpout` and `compact_stall` across it in `INFO`.
   `--swapout-limit-mib` fails a load that made the host swap out more than that, with `ERR -`.
9. Sequence state does not survive a sleep: the checkpoints are emptied.

## Why

Keeping the child is the only known way to keep an arena both cards can read, and it costs 0.31-0.32 GiB of VRAM a
card while asleep ([sleep and wake record](../measurements/MEASUREMENTS.md#2026-10-17-sleep-and-wake)). It also keeps
the kernels loaded and the runtime set up, though those cost only about 0.1 s (VRAM record). Recording the lists
again is cheap, so the virtual memory calls of way 3, which crash the driver with recorded lists above 16 GiB a card,
are not needed.

A separate admin port keeps the sleep and wake calls away from the clients that reach the API through the proxy. The
lease makes the proxy's own start and stop the only path to the cards. Failing closed matters because an idle engine
left awake beside another model might have its VRAM evicted into host RAM, which sleep exists to prevent (not
measured).

Reading with `O_DIRECT` takes the page cache out of every load: the page cache rises by at most 0.79 GiB across one.
The pipeline overlaps the reads with the CPU's rearranging, and a start loads in 19.9-54.3 s. It still reads at only
1.3-3.5 GB/s, short of the SSD's 6.5: 58.95 GB in 17.0-44.1 s, against 9.2 s at the SSD's rate. A wake reads only the
cards' part, 38.30 GB, in 11.9-18.1 s, and takes 14.2-20.7 s in all (sleep and wake record).

## Options not chosen

- Way 1, the proxy stopping the front end as well: each wake pays the front end's 5.3-5.6 s start
  ([start and quit record](../measurements/MEASUREMENTS.md#2026-10-16-engine-start-and-quit)), and the port goes away.
- The front end holding the arena as a `memfd` that each child imports: Level Zero's import serves one card (Q6).
- The child exiting on every sleep: each wake pins the arena again after the other model, which may stall in
  compaction or push the host's other memory into swap. It stays available as `--sleep-mode exit`.
- A copy of the weights in the cards' layout: about 60 GB more on the disk, and the CPU already rearranges faster
  than the SSD reads.

## Consequences

- Asleep, the engine holds the arena (20.49 GiB at 288 experts a layer, 13.63 GiB at 363), its host buffers and
  0.31-0.32 GiB of VRAM a card. The child's RSS, Python's included, is 806-889 MiB (sleep and wake record).
- Each card reads every dense tensor it loads whole, so a load reads the dense weights twice: 4.18 GB more than
  once (the 48 layers' dense bytes, the head and final mixer, and the PLE, from
  [MEMORY_BUDGET.md](../measurements/MEMORY_BUDGET.md#weights)). It costs about 1.3-2.0 s (*estimate*: 4.18 GB at the
  2.1-3.2 GB/s a wake reads at). Reading each card's rows only, or loading both cards from one read, would remove it.
- The staging is about 0.5 GB of ordinary memory at its peak during a load, freed when the load ends (*estimate*:
  the head's 521 MB, read whole for each card in turn; for the experts, two batches of 64 of the largest, 2,662,400 B
  each and 2,678,784 B once widened for `O_DIRECT`, and each card's half of one batch rearranged, 513 MB in all).
- A filesystem that refuses `O_DIRECT` stops the engine, as it does for the n-gram table
  ([ADR-0015](0015-ngram-table-on-the-ssd.md)).
- The proxy's entry, the front end's service and its admin port belong to the deployment, not the engine.
- [DESIGN.md, section 1](../DESIGN.md#1-the-parts) describes the front end as a service the proxy does not start.
- Asleep with an encoder, the engine also holds the encoder's 16-bit weights, 0.90 GB of pinned host memory
  ([ADR-0031](0031-images-through-the-prompt-path.md)). This ADR's numbers were measured without one.
