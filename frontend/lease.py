"""The lease llama-swap runs as the model's command (ADR-0032).

python -m frontend.lease http://127.0.0.1:8081            # cmd: wakes the engine and holds it awake
python -m frontend.lease http://127.0.0.1:8081 --sleep    # cmdStop: sleeps it, once its running requests drain

The first holds `POST /admin/lease` open. It exits 0 when a sleep ends the lease, and 1 when the wake fails or the
connection breaks. Stopped by a signal, it drops the connection, and the front end then sleeps on its own. The
second exits once the engine is asleep. Standard library only, so it starts at once.
"""

from __future__ import annotations

import argparse
import http.client
import json
import sys
import urllib.parse


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m frontend.lease", description=(__doc__ or "").splitlines()[0])
    p.add_argument("url", help="the front end's admin address, such as http://127.0.0.1:8081")
    p.add_argument("--sleep", action="store_true", help="sleep the engine instead of holding it awake")
    p.add_argument("--drain-ms", type=int, help="with --sleep: how long running requests may go on")
    args = p.parse_args(argv)
    url = urllib.parse.urlsplit(str(args.url))
    path = "/admin/lease"
    if args.sleep:
        path = "/admin/sleep" if args.drain_ms is None else f"/admin/sleep?drain_ms={args.drain_ms}"
    conn = http.client.HTTPConnection(url.hostname or "127.0.0.1", url.port or 80, timeout=None)
    try:
        conn.request("POST", path)
        response = conn.getresponse()
        if response.status != 200:
            print(f"{path}: {response.status} {response.read().decode(errors='replace')}", file=sys.stderr)
            return 1
        last: dict[str, object] = {}
        for line in response:  # the lease: "ready", a blank line every few seconds, then "asleep"
            if line.strip():
                print(line.decode().strip(), flush=True)
                last = json.loads(line)
        return 0 if last.get("status") == "asleep" else 1
    except (OSError, http.client.HTTPException, json.JSONDecodeError) as exc:
        print(f"{path}: {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
