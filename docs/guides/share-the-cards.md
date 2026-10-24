# Share the cards with other models through llama-swap

This guide has [llama-swap](https://github.com/mostlygeek/llama-swap) wake C0ven when a request for its model
arrives and put it to sleep when another model needs the cards.

Asleep, the engine gives back the cards' memory and keeps its RAM, so a wake loads only the cards. A sleep took
0.38-0.72 s and a wake 14.2-20.7 s ([the record](../measurements/MEASUREMENTS.md#2026-10-17-sleep-and-wake)).
The design is [ADR-0032](../adr/0032-sleep-the-front-end-stays-the-engine-keeps-the-arena.md).

## You need

- The model files, laid out as in [MODEL_FILES.md](../MODEL_FILES.md).
- A running llama-swap that can reach the machine with the cards.

## 1. Start C0ven asleep, with the admin routes on a second port

Replace `993` with your render group's ID and `/path/to/models` with your model directory.

```bash
docker run -d --name c0ven --device /dev/dri --group-add 993 --ulimit memlock=-1:-1 --read-only -p 127.0.0.1:8090:8080 -p 127.0.0.1:8091:8081 -v /path/to/models:/models:ro -v c0ven-mtp:/cache -e ENGINE_MTP_CACHE_DIR=/cache/mtp ghcr.io/avra-m3/c0ven/serve:main --model /models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --slots 4 --context 262144 --host 0.0.0.0 --port 8080 --start asleep --admin-host 0.0.0.0 --admin-port 8081
```

## 2. Add the model to llama-swap's config

The model's `cmd` holds a lease, which wakes the engine and keeps it awake. Its `cmdStop` puts it to sleep. Use the
address at which llama-swap reaches the two ports:

```yaml
models:
  qwen3.8-flash-next:
    proxy: http://<c0ven>:8090
    checkEndpoint: /health
    cmd: curl -sfN -X POST http://<c0ven>:8091/admin/lease
    cmdStop: curl -sf -X POST http://<c0ven>:8091/admin/sleep
```

Where C0ven's code is installed, `python -m frontend.lease http://<c0ven>:8091` does what the `cmd` does, and the
same with `--sleep` what the `cmdStop` does.

## 3. Check it

Request the model through llama-swap. The first request waits for the wake. Then request another model: the cards
go to it, and `docker logs c0ven` shows the sleep.

## If it does not work

- While the engine sleeps or loads, `/health` and every request get 503, so only a lease wakes it.
- If the lease's connection ends without a sleep, the engine goes to sleep on its own.
- A sleep lets running requests go on for `--drain-timeout` seconds (30 by default), then cancels them.
- Asleep, the engine still holds about 20 GiB of RAM and 0.32 GiB of VRAM a card. `--sleep-mode exit` gives the
  RAM back, and each wake is then a full load. The [known limits](../../README.md#known-limits) have the rest.
