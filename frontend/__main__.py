"""The front end, which serves the APIs (DESIGN.md, section 1). It starts the engine as its child.

python -m frontend --model /models/<file>-00001-of-00002.gguf --slots 4 --context 262144 --port 8080
python -m frontend --tokenizer tokenizer.json --chat-template chat_template.jinja --engine-backend mock

Under llama-swap (ADR-0032) the front end runs on its own, starts asleep and serves the admin routes
on a port of their own: `--start asleep --admin-port 8081`. llama-swap's command for the model is then
`python -m frontend.lease http://127.0.0.1:8081`, and its cmdStop the same with `--sleep`.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import socket
import sys
import tempfile
from pathlib import Path

import uvicorn

import protocol
from frontend.admin import Admin
from frontend.app import create_app
from frontend.engine_client import EngineClient, SleepMode
from frontend.model import load

log = logging.getLogger("frontend")


def version() -> str:
    """The project's version, from the `VERSION` file beside the packages (ADR-0042)."""
    return (Path(__file__).resolve().parent.parent / "VERSION").read_text().strip()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m frontend", description=(__doc__ or "").splitlines()[0])
    p.add_argument("--version", action="version", version=f"c0ven {version()}")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--model", type=Path, help="the GGUF file (any shard)")
    p.add_argument("--tokenizer", type=Path, help="the model's tokenizer.json, instead of the GGUF's vocabulary")
    p.add_argument("--chat-template", type=Path, help="the model's chat_template.jinja, instead of the GGUF's")
    p.add_argument("--name", help="the model's name in the APIs")
    p.add_argument("--engine-backend", choices=("native", "mock", "reference"), default="native")
    p.add_argument("--arch", default="qwen4exp", help="the architecture, for the mock and the reference")
    p.add_argument("--slots", type=int, default=1)
    p.add_argument("--context", type=int, default=262_144)
    p.add_argument(
        "--kv-tokens",
        type=protocol.tokens,
        default=0,
        help="the KV pool the slots share, in tokens (512k = 524,288), at least --context; default: slots x context",
    )
    p.add_argument(
        "--kv-stream",
        action="store_true",
        help="stream the KV cache: 32,768 cells of K and V per slot stay in VRAM, the rest in host memory, and the "
        "room caches experts (ADR-0008 item 3)",
    )
    p.add_argument(
        "--default-max-tokens",
        type=protocol.tokens,
        default=0,
        help="what a request without max_tokens may write, in tokens (32k = 32,768); default: the rest of the context",
    )
    p.add_argument("--engine-arg", action="append", default=[], help="passed to the engine as is (repeatable)")
    p.add_argument("--start", choices=("awake", "asleep"), default="awake", help="asleep: load at the first wake")
    p.add_argument(
        "--sleep-mode",
        choices=[m.value for m in SleepMode],
        default=SleepMode.KEEP.value,
        help="keep: the engine frees the cards and keeps the arena in host memory; exit: it exits",
    )
    p.add_argument("--admin-host", default="127.0.0.1")
    p.add_argument("--admin-port", type=int, help="serve /admin/wake, /admin/sleep and /admin/lease on this port")
    p.add_argument(
        "--drain-timeout",
        type=float,
        default=30.0,
        help="seconds a sleep lets the running requests go on before it cancels them; a shutdown waits as long",
    )
    p.add_argument("--log-level", default="INFO")
    p.add_argument(
        "--metrics-log",
        type=float,
        default=0.0,
        help="seconds between JSON log lines of the /metrics counters, which also logs each request's record; 0: none",
    )
    return p.parse_args(argv)


def engine_argv(args: argparse.Namespace, arch: str = "", table_cache: Path | None = None) -> list[str]:
    """The engine's command line. arch: the model's, read by the front end already; table_cache: where every engine
    this front end starts keeps the model's tensor table, so only the first reads it from the GGUF."""
    argv = [
        sys.executable,
        "-m",
        "engine",
        "--backend",
        args.engine_backend,
        "--slots",
        str(args.slots),
        "--context",
        str(args.context),
        *(["--kv-tokens", str(args.kv_tokens)] if args.kv_tokens else []),
        *(["--kv-stream"] if args.kv_stream else []),
        "--log-level",
        args.log_level,
    ]
    if args.engine_backend == "native":
        if args.model is None:
            raise SystemExit("the native engine needs --model")
        argv += ["--model", str(args.model)]
        if arch:
            argv += ["--arch", arch]
        if table_cache is not None:
            argv += ["--table-cache", str(table_cache)]
    else:
        argv += ["--arch", args.arch]
    return argv + list(args.engine_arg)


def table_cache(stack: contextlib.ExitStack) -> Path | None:
    """A directory for the engine's tensor table, removed at exit; None where nothing is writable, as in a container
    with a read-only root, and then each engine start reads the table from the GGUF."""
    try:
        return Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="c0ven-")))
    except OSError as exc:
        log.warning("no directory for the tensor table (%s); each engine start reads it from the GGUF", exc)
        return None


def listen(host: str, port: int) -> socket.socket:
    return socket.create_server((host, port), family=socket.AF_INET6 if ":" in host else socket.AF_INET)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(name)s %(message)s")
    model = load(args.model, args.tokenizer, args.chat_template, args.name)
    sockets = [listen(args.host, args.port)]
    admin = None
    if args.admin_port is not None:
        sockets.append(listen(args.admin_host, args.admin_port))
        admin = Admin(drain_ms=round(args.drain_timeout * 1e3), port=args.admin_port)
    with contextlib.ExitStack() as stack:
        engine = EngineClient(
            engine_argv(args, model.arch, table_cache(stack)),
            mode=SleepMode(args.sleep_mode),
            asleep=args.start == "asleep",
        )
        app = create_app(model, engine, admin, args.default_max_tokens, args.metrics_log)
        # A shutdown waits for open streams, a held lease among them, no longer than a sleep's drain.
        graceful = max(1, round(args.drain_timeout))
        config = uvicorn.Config(app, log_level=args.log_level.lower(), timeout_graceful_shutdown=graceful)
        uvicorn.Server(config).run(sockets=sockets)


if __name__ == "__main__":
    main()
