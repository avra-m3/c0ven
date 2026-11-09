# Server options

The arguments after the image name in `docker run` are the server's options. The
[quick start](../README.md#quick-start) shows them in use.

| Flag | Default | What it does |
| --- | --- | --- |
| `--model` | | The GGUF file (any shard). |
| `--slots` | `1` | How many requests can run at once. |
| `--context` | the model file's | Context length in tokens. |
| `--default-max-tokens` | the rest of the context | What a request without `max_tokens` may write, in tokens (`32k` is 32,768). |
| `--kv-tokens` | slots x context | The KV cache the slots share, in tokens (`512k` is 524,288), at least `--context`. The tokens the requests do not hold lend their VRAM to the expert cache; with `--deterministic`, each token of it takes VRAM that would otherwise cache experts. See below. |
| `--host` | `127.0.0.1` | Address to listen on. |
| `--port` | `8080` | Port to listen on. |
| `--name` | | The model's name in the APIs. |
| `--engine-arg` | | Passed to the engine as is. Repeatable. |
| `--start` | `awake` | `asleep` waits for the first wake to load the model. |
| `--sleep-mode` | `keep` | `exit` has the engine exit on sleep: the RAM comes back too, and each wake is a full load. |
| `--admin-host` | `127.0.0.1` | Address for the admin routes. |
| `--admin-port` | | Serve `/admin/wake`, `/admin/sleep` and `/admin/lease` on this port. Without it there are none. |
| `--drain-timeout` | `30` | Seconds a sleep lets running requests go on before it cancels them. |
| `--log-level` | `INFO` | Log level. |
| `--metrics-log` | `0` | Seconds between log lines of the `/metrics` counters. Each request's record is logged too. `0`: none. |
| `--version` | | Print the version and exit. |

`--kv-tokens` sets the KV cache's size apart from `--slots`. With `--slots 4 --kv-tokens 512k`, four requests run
at once, and their prompts and answers together may hold up to 524,288 tokens. A request holds room for its prompt
when it starts, and for its answer as it writes it, so a large `max_tokens` holds nothing until it is written. A
request waits while the cache cannot hold its prompt. When the running answers outgrow the cache, the newest request
pauses and starts again from where it was once there is room, rereading its prompt and the tokens it has written
([CACHING.md](CACHING.md#the-shared-room-for-the-token-cache)). One that reaches `max_tokens`, or
`--default-max-tokens` without it, ends with the finish reason `length`.

The KV cache streams: each request keeps up to 32,768 tokens of K and V in VRAM, the cells its steps read, and the
whole of it in host memory, and the VRAM the rest would take caches experts. The host copy is
pinned at load: 13,056 bytes a token of the KV cache, 13.7 GB for four 262,144-token slots.

`ZE_AFFINITY_MASK` limits which cards the engine sees. Useful engine flags through `--engine-arg`: `--no-draft`,
`--no-vision`, `--deterministic`, `--checkpoints N`, `--ram-max GiB`, which refuses a configuration that would take
more host memory, `--ngram-ram on|off`, which holds the n-gram table (28.8 GB) in RAM or leaves it in the file; by
default it is in RAM when the host has room for it, `--hugepages`, which puts the expert arena in the host's reserved
huge pages (see below), and `--card-memory GiB`, which limits the memory the engine takes on
each card, for a card that also drives a desktop (`python -m engine --help` lists them all).

## Reserved huge pages

The cards read the expert arena in place. When the kernel moves one of its small pages, a card stops until it can
read that page again. `--engine-arg --hugepages` takes the arena from the host's reserved huge pages, which the
kernel never moves or swaps ([ADR-0066](adr/0066-the-arena-may-take-reserved-huge-pages.md)). Reserve them on the
host before the start: the arena's size is in the fit report as "host arena", and with 2 MiB pages each GiB needs
512 of them.

```bash
sudo sysctl vm.nr_hugepages=20480
```

That reserves 40 GiB, which the host cannot use for anything else while it is reserved. The arena takes 1 GiB pieces
from the pool while it lasts, and the rest from ordinary memory; the log says how many pieces are in reserved huge
pages. A container needs no extra flag, unless its cgroup limits huge pages (`hugetlb.2MB.max`).


## Long prompt pieces beside writers

While a request is writing, a prompt is read in pieces of at most 1,021 rows, each aimed at `--prompt-budget-ms`
(500 ms by default), and after each piece the writers get half its time (`--decode-share`). A writer's longest gap
between tokens is then 0.71-0.87 s ([the record](measurements/MEASUREMENTS.md#2026-11-01-full-prompt-pieces)).

Where nobody watches tokens stream, as for an agent's tool calls, a larger budget reads prompts faster beside writers
at the cost of longer gaps:

```bash
--engine-arg=--prompt-budget-ms=2000
```

When a whole chunk costs at most 1.6 times the budget, each piece is a whole chunk of 4,349 rows, which leaves room
for three writers' rows, or the rest of the prompt ([ADR-0069](adr/0069-a-prompt-piece-may-be-a-whole-chunk-when-the-budget-fits-one.md)). A whole chunk takes
1,645-1,680 ms at position 0 and 2,748 ms at position 239,360
([the record](measurements/MEASUREMENTS.md#2026-10-17-prefill-timings)), so 2,000 ms (3,200 ms at 1.6 times) fits
one with room to spare. A writer's longest gap is then a whole chunk's time plus up to 85 ms
([the record](measurements/MEASUREMENTS.md#2026-11-01-whole-rows-and-stretch)): about 1.7 s at position 0 and
2.8 s near position 239k. The writers still get half of each piece's time. With a smaller budget, pieces stay at
1,021 rows wherever a whole chunk does not fit 1.6 times it, so set the budget for the deepest position served.
