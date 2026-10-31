"""Check a change on the cards: GPU tests, served runs against main in ABBA order, the edge cases, and a verdict.

    python3 tools/check.py --a <main checkout> --model <first shard> --out <dir> [--claim decode|prompt|near-full]

Run from the change's checkout (B); both checkouts are built (`build/`) and have their `.venv`. The steps, scenarios
and verdicts are TESTING.md's, section "Checking a change". It stops at the first failure and writes `results.jsonl`
and `summary.md` to `--out` as it goes. Standard library only, so any Python 3.14 runs it.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import http.client
import itertools
import json
import os
import re
import signal
import statistics
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

type Body = dict[str, Any]
type Rec = dict[str, Any]

DEPLOYED = ("--slots", "4", "--context", "262144", "--kv-tokens", "524288", "--default-max-tokens", "32768")
ONE_SLOT = ("--slots", "1", "--context", "262144")
# name: (flags, extra environment). Same-tokens turns checkpoints off, so a prompt sent again is read again.
CONFIGS: dict[str, tuple[tuple[str, ...], dict[str, str]]] = {
    "deployed": (DEPLOYED, {}),
    "one-slot": (ONE_SLOT, {}),
    "one-card": (ONE_SLOT, {"ZE_AFFINITY_MASK": "0"}),
    "same-tokens": ((*DEPLOYED, "--engine-arg=--deterministic", "--engine-arg=--checkpoints=0"), {}),
}
# The long prompts are test data under FLASH_REF_DIR (TESTING.md, test data): file name and sha256.
LONG_FILES = {
    "long": ("check/long-255k.json", "447e3776e994ed6a8e8dd3ede881ebf290a0cec1db1311e57ab01f6633ee8485"),
    "long-two": ("check/prompts-2x240k.json", "e9e00097ce7be14f0198a23483584893c89cae34e78777bb3d8929a4100ae35f"),
}
QUICK = ("short", "four", "mixed")
EDGE_TESTS = "api_test|vision_test|encoder_test"
ERROR = re.compile(r"\berror\b|traceback|the engine failed|did not (publish|finish)", re.IGNORECASE)
GAP_S = 60.0  # no token for this long while a request is open fails the run
# Each quick-round scenario's run-to-run spread on main, as a share of its time (TESTING.md). Four's time depends on
# how many tokens its tool calls take, which varies from run to run.
SPREAD = {"short": 0.01, "four": 0.10, "mixed": 0.04}
LONG_SLOWER = 0.01  # long's first token repeats within 0.1 s of about 120 s; later than A's by more fails B
# Host state that moves a run's time, and how far it may move before main's stored runs no longer stand for main.
DRIFT = {"mem_avail_gib": 6.0, "swap_used_gib": 2.0, "card0_free_gib": 1.0, "card1_free_gib": 1.0}
# The share of the CPU busy over 5 s while the check runs nothing: above it, another job is on the CPU, which computes
# the expert misses. A load average would count the check's own GPU tests.
BUSY_MAX = 0.25
SWAP_PAGES = 1 << 18  # 1 GiB of 4 KiB pages: more swapped out than the A run fails the B run
FINISHES = ("stop", "length", "tool_calls")


@dataclass
class Run:
    """One fresh server: its requests, and what made it fail."""

    n: int
    arm: str
    config: str
    scenarios: tuple[str, ...]
    start: str = ""
    requests: list[Rec] = field(default_factory=list[Rec])
    failures: list[str] = field(default_factory=list[str])
    swapped_pages: int = 0
    mem_low_gib: float = 0.0
    load_s: float = 0.0

    def seconds(self, scenario: str) -> float:
        """The scenario's time: from its first request sent to its last request's end, the slowest request's time."""
        return max(r["seconds"] for r in self.requests if r["scenario"] == scenario)

    def first_token(self, scenario: str) -> float:
        return max(r["ttft"] or 0.0 for r in self.requests if r["scenario"] == scenario)

    def decode(self, scenario: str) -> float:
        tps = [r["decode_tps"] for r in self.requests if r["scenario"] == scenario and r["decode_tps"]]
        return statistics.fmean(tps) if tps else 0.0

    def stored(self) -> Rec:
        """What a stored run keeps: enough for the verdicts, not the text."""
        keep = ("scenario", "seconds", "ttft", "decode_tps")
        return {
            "n": self.n,
            "arm": self.arm,
            "config": self.config,
            "scenarios": list(self.scenarios),
            "start": self.start,
            "swapped_pages": self.swapped_pages,
            "requests": [{k: r[k] for k in keep} for r in self.requests],
        }

    @staticmethod
    def from_stored(rec: Rec) -> Run:
        run = Run(rec["n"], rec["arm"], rec["config"], tuple(rec["scenarios"]), start=rec["start"])
        run.swapped_pages, run.requests = rec["swapped_pages"], rec["requests"]
        return run


# The verdicts, kept apart from the runs so the host tests check them.


def against_a(a: Run, b: Run) -> list[str]:
    """The ways B fails against its A run: a scenario over twice A's time, or over 1 GiB more swapped out."""
    out = [
        f"{s}: {b.seconds(s):.1f} s against A's {a.seconds(s):.1f} s"
        for s in b.scenarios
        if s in a.scenarios and b.seconds(s) > 2 * a.seconds(s)
    ]
    if b.swapped_pages - a.swapped_pages > SWAP_PAGES:
        out.append(f"swapped out {b.swapped_pages} pages against A's {a.swapped_pages}")
    return out


def quick_stop(a1: Run, b1: Run) -> list[str]:
    """The scenarios where B1 is slower than A1 by more than twice their usual spread."""
    return [
        f"{s}: {b1.seconds(s):.2f} s against {a1.seconds(s):.2f} s"
        for s in QUICK
        if b1.seconds(s) > a1.seconds(s) * (1 + 2 * SPREAD[s])
    ]


def slower(a: Sequence[Run], b: Sequence[Run], scenarios: Sequence[str]) -> list[str]:
    """The scenarios where every B run is slower than every A run."""
    return [s for s in scenarios if apart([r.seconds(s) for r in a], [r.seconds(s) for r in b]) < 0]


def past_stored(stored: Sequence[Run], b: Sequence[Run]) -> list[str]:
    """The scenarios where every B run is slower than main's slowest stored run by more than the scenario's spread,
    which the host drifts by between a stored run and now."""
    return [s for s in QUICK if min(r.seconds(s) for r in b) > max(r.seconds(s) for r in stored) * (1 + SPREAD[s])]


def drift(old: Rec, new: Rec) -> list[str]:
    """How the host now differs from when main's runs were stored, past what `DRIFT` allows; empty when they stand."""
    out = [
        f"{k} {old[k]} then, {new[k]} now" for k, limit in DRIFT.items() if abs(new.get(k, 0) - old.get(k, 0)) > limit
    ]
    if old.get("governor") != new.get("governor"):
        out.append(f"governor {old.get('governor')} then, {new.get('governor')} now")
    if new.get("cpu_busy", 0) > BUSY_MAX:
        out.append(f"CPU {new['cpu_busy']:.0%} busy with the check idle")
    return out


def apart(a: Sequence[float], b: Sequence[float]) -> int:
    """1 when every B time beats every A time, -1 when every A time beats every B time, else 0 (lower is better)."""
    if max(b) < min(a):
        return 1
    if min(b) > max(a):
        return -1
    return 0


def far_apart(a: Sequence[float], b: Sequence[float]) -> int:
    """As `apart`, for one round: the medians differ by more than ten times the larger arm's range."""
    gap = statistics.median(a) - statistics.median(b)
    if abs(gap) <= 10 * max(max(a) - min(a), max(b) - min(b)):
        return 0
    return 1 if gap > 0 else -1


def gain(signs: dict[str, int]) -> bool:
    """A claim holds when some scenario is faster and none slower."""
    return any(s > 0 for s in signs.values()) and all(s >= 0 for s in signs.values())


# Requests.


def reference_body(messages: dict[str, Rec], pid: str, tag: str, max_tokens: int) -> Body:
    """A row of reference/messages.jsonl as a chat body. `tag` starts the first message, so one scenario's prompts
    share no prefix with another's; it is the same in both arms."""
    m = messages[pid]
    msgs: list[Rec] = copy.deepcopy(m["messages"])
    msgs[0]["content"] = f"Request {tag}.\n" + msgs[0]["content"]
    for x in msgs:  # the file keeps tool calls as {name, arguments}; the API takes them under `function`
        if x.get("tool_calls"):
            x["tool_calls"] = [c if "function" in c else {"type": "function", "function": c} for c in x["tool_calls"]]
    body: Body = {"messages": msgs, "max_tokens": max_tokens}
    if m.get("tools"):
        body["tools"] = [t if "type" in t else {"type": "function", "function": t} for t in m["tools"]]
    kwargs: Rec = {}
    if not m.get("thinking"):
        kwargs["enable_thinking"] = False
    if m.get("effort"):
        kwargs["reasoning_effort"] = m["effort"]
    if kwargs:
        body["chat_template_kwargs"] = kwargs
    return body


def file_body(row: Rec, max_tokens: int) -> Body:
    body: Body = {"messages": row["messages"], "max_tokens": max_tokens}
    if row.get("tools"):
        body["tools"] = row["tools"]
    if row.get("kwargs"):
        body["chat_template_kwargs"] = row["kwargs"]
    return body


class Client:
    """Streams requests to one server and records each as a line of results.jsonl."""

    def __init__(self, port: int, run: Run, write: Callable[[Rec], None]) -> None:
        self.port, self.run, self.write = port, run, write
        self.lock = threading.Lock()

    def get(self, path: str, timeout: float = 30) -> tuple[int, Any]:
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        try:
            c.request("GET", path)
            r = c.getresponse()
            data = r.read()
        finally:
            c.close()
        try:
            return r.status, json.loads(data)
        except ValueError:
            return r.status, None

    def send(self, scenario: str, mode: str, pid: str, body: Body, first: threading.Event | None = None) -> Rec:
        """One greedy streamed request. `first` is set at its first token, or when it ends without one."""
        body = body | {"temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
        sent = time.perf_counter()
        arrivals: list[float] = []
        text: list[str] = []
        usage: Rec = {}
        finish, status, err = None, None, None
        try:
            c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=900)
            c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
            r = c.getresponse()
            status = r.status
            if status != 200:
                err = r.read()[:500].decode(errors="replace")
            else:
                for raw in r:
                    line = raw.strip()
                    if not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        break
                    chunk: Rec = json.loads(data)
                    usage = chunk.get("usage") or usage
                    choices: list[Rec] = chunk.get("choices") or []
                    for choice in choices:
                        delta: Rec = choice.get("delta") or {}
                        piece = (delta.get("reasoning_content") or "") + (delta.get("content") or "")
                        if delta.get("tool_calls"):
                            piece += json.dumps(delta["tool_calls"], sort_keys=True)
                        if piece:
                            arrivals.append(time.perf_counter() - sent)
                            text.append(piece)
                            if first is not None:
                                first.set()
                        finish = choice.get("finish_reason") or finish
            c.close()
        except (OSError, http.client.HTTPException, ValueError) as e:
            err = f"{type(e).__name__}: {e}"
        if first is not None:
            first.set()
        gaps = [y - x for x, y in itertools.pairwise(arrivals)]
        made = int(usage.get("completion_tokens") or 0)
        span = arrivals[-1] - arrivals[0] if len(arrivals) > 1 else 0.0
        rec: Rec = {
            "run": self.run.n,
            "arm": self.run.arm,
            "config": self.run.config,
            "scenario": scenario,
            "mode": mode,
            "id": pid,
            "status": status,
            "error": err,
            "ttft": arrivals[0] if arrivals else None,
            "seconds": time.perf_counter() - sent,
            "finish": finish,
            "prompt_tokens": usage.get("prompt_tokens"),
            "tokens": made,
            "max_gap": max(gaps) if gaps else None,
            "decode_tps": (made - 1) / span if span else None,
            "sha": hashlib.sha256("".join(text).encode()).hexdigest()[:16],
        }
        bad: list[str] = []
        if status != 200 or err:
            bad.append(f"status {status} {err or ''}".strip())
        if finish not in FINISHES:
            bad.append(f"finish {finish}")
        if gaps and max(gaps) > GAP_S:
            bad.append(f"no token for {max(gaps):.0f} s")
        rec["bad"] = bad
        with self.lock:
            if scenario != "warm-up":
                self.run.requests.append(rec)
                self.run.failures.extend(f"{scenario} {pid}: {x}" for x in bad)
            self.write(rec)
            ttft = f"{rec['ttft']:.2f}" if rec["ttft"] is not None else "-"
            note = f" FAIL {'; '.join(bad)}" if bad else ""
            say(
                f"  {scenario} {mode} {pid}: prompt {rec['prompt_tokens']}, {made} out, first token {ttft} s, "
                f"decode {rec['decode_tps'] or 0:.1f} tokens/s, {rec['seconds']:.1f} s, {finish}{note}"
            )
        return rec

    def together(self, scenario: str, mode: str, reqs: Sequence[tuple[str, Body]]) -> list[Rec]:
        out: list[Rec] = [{} for _ in reqs]

        def go(i: int) -> None:
            out[i] = self.send(scenario, mode, *reqs[i])

        threads = [threading.Thread(target=go, args=(i,)) for i in range(len(reqs))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return out

    def after_first(
        self, scenario: str, mode: str, writers: Sequence[tuple[str, Body]], late: tuple[str, Body]
    ) -> list[Rec]:
        """The writers at once, and `late` once each writer has its first token."""
        events = [threading.Event() for _ in writers]
        out: list[Rec] = [{} for _ in writers]

        def go(i: int) -> None:
            out[i] = self.send(scenario, mode, *writers[i], events[i])

        threads = [threading.Thread(target=go, args=(i,)) for i in range(len(writers))]
        for t in threads:
            t.start()
        for e in events:
            e.wait()
        last = self.send(scenario, mode, *late)
        for t in threads:
            t.join()
        return [*out, last]


def run_scenario(name: str, client: Client, messages: dict[str, Rec], ref_dir: Path) -> None:
    def ref(pid: str, tag: str = name) -> tuple[str, Body]:
        return pid, reference_body(messages, pid, tag, 512)

    if name == "short":
        client.send(name, "alone", *ref("gpqa-1"))
    elif name == "four":
        client.together(name, "at once", [ref(f"agent-turn{i}") for i in range(1, 5)])
    elif name == "mixed":
        client.after_first(name, "writers then one", [ref("gpqa-2"), ref("gpqa-3"), ref("gpqa-4")], ref("agent-turn2"))
    elif name == "long":
        row: Rec = json.loads((ref_dir / LONG_FILES[name][0]).read_text())
        client.send(name, "alone", str(row["id"]), file_body(row, 256))
    elif name == "long-two":
        rows: list[Rec] = json.loads((ref_dir / LONG_FILES[name][0]).read_text())
        client.together(name, "at once", [(str(r["id"]), file_body(r, 512)) for r in rows])
    elif name == "one-card":
        client.send(name, "alone", *ref("gpqa-1", f"{name}-a"))
        client.together(name, "at once", [ref("gpqa-1", f"{name}-b"), ref("edit-py")])
    elif name == "same-tokens":
        g, e = ref("gpqa-1"), ref("edit-py")
        recs = [client.send(name, "alone", *g), client.send(name, "alone", *e)]
        recs += client.after_first(name, "together", [g], e)
        recs += client.together(name, "at once", [g, e])
        for pid in ("gpqa-1", "edit-py"):
            shas = {r["mode"]: r["sha"] for r in recs if r["id"] == pid}
            if len(set(shas.values())) != 1:
                client.run.failures.append(f"same-tokens {pid}: different text {shas}")
    else:
        raise ValueError(f"unknown scenario {name}")


# Processes and the host.


def say(line: str) -> None:
    print(f"{datetime.now(UTC):%H:%M:%SZ} {line}", flush=True)


def utc() -> str:
    return f"{datetime.now(UTC):%Y-%m-%d %H:%M:%SZ}"


def meminfo_gib(key: str = "MemAvailable") -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith(key + ":"):
            return int(line.split()[1]) / (1 << 20)
    return 0.0


def cpu_busy(seconds: float) -> float:
    """The share of all CPUs busy over `seconds`, from /proc/stat."""

    def sample() -> tuple[int, int]:
        ticks = [int(x) for x in Path("/proc/stat").read_text().split("\n", 1)[0].split()[1:]]
        return sum(ticks), ticks[3] + ticks[4]  # idle and iowait

    total0, idle0 = sample()
    time.sleep(seconds)
    total1, idle1 = sample()
    return round(1 - (idle1 - idle0) / max(total1 - total0, 1), 3)


def swapped_out() -> int:
    for line in Path("/proc/vmstat").read_text().splitlines():
        if line.startswith("pswpout "):
            return int(line.split()[1])
    return 0


def descendants(pid: int) -> list[int]:
    children: dict[int, list[int]] = {}
    for d in Path("/proc").iterdir():
        if d.name.isdigit():
            try:
                ppid = int((d / "stat").read_text().rsplit(")", 1)[1].split()[1])
            except OSError, IndexError, ValueError:
                continue
            children.setdefault(ppid, []).append(int(d.name))
    out: list[int] = []
    todo = [pid]
    while todo:
        for c in children.get(todo.pop(), []):
            out.append(c)
            todo.append(c)
    return out


def alive(pid: int) -> bool:
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


def stop_tree(proc: subprocess.Popen[bytes]) -> None:
    tree = descendants(proc.pid)
    proc.terminate()
    try:
        proc.wait(90)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    deadline = time.monotonic() + 30
    while (left := [p for p in tree if alive(p)]) and time.monotonic() < deadline:
        time.sleep(1)
    for p in left:
        os.kill(p, signal.SIGKILL)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def git(checkout: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", "-c", "safe.directory=*", "-C", str(checkout), *args], check=False, capture_output=True, text=True
    )
    return out.stdout.strip()


def head(checkout: Path) -> str:
    out = subprocess.run(
        ["git", "-c", "safe.directory=*", "-C", str(checkout), "log", "--oneline", "-1"],
        check=False,
        capture_output=True,
        text=True,
    )
    return out.stdout.strip()


# The sequence.


class Stop(Exception):
    """The change fails a step; the message says why."""


@dataclass
class Step:
    name: str
    ran: str
    verdict: str


class Check:
    def __init__(self, a: argparse.Namespace) -> None:
        self.args = a
        self.b: Path = Path.cwd()
        self.a: Path = a.a.resolve()
        self.out: Path = a.out.resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.ref_dir = Path(os.environ.get("FLASH_REF_DIR", "."))
        self.skip: dict[str, str] = {k: v for k, _, v in (str(s).partition("=") for s in a.skip)}
        self.runs: list[Run] = []
        self.steps: list[Step] = []
        self.started = utc()
        self.verdict = "running"
        lines = (self.b / "reference/messages.jsonl").read_text().splitlines()
        self.messages: dict[str, Rec] = {(r := json.loads(x))["id"]: r for x in lines if x.strip()}
        self.missing = {
            s: f"{p} missing or not its sha256"
            for s, (p, digest) in LONG_FILES.items()
            if not (self.ref_dir / p).is_file() or sha256(self.ref_dir / p) != digest
        }
        self.host: Rec = {}
        self.key: Rec = {
            "a": git(self.a, "rev-parse", "HEAD"),
            "model": str(a.model),
            "messages": sha256(self.b / "reference/messages.jsonl"),
            "scenarios": hashlib.sha256(repr((CONFIGS, LONG_FILES, QUICK)).encode()).hexdigest(),
            "extra": a.baseline_key,
        }
        self.store: Path | None = a.baselines / f"{self.key['a']}.json" if a.baselines else None
        self.base: list[Run] = []

    def fingerprint(self) -> Rec:
        info = {x.split(":")[0]: int(x.split()[1]) for x in Path("/proc/meminfo").read_text().splitlines()}
        gov = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
        host: Rec = {
            "mem_avail_gib": round(info["MemAvailable"] / (1 << 20), 1),
            "swap_used_gib": round((info["SwapTotal"] - info["SwapFree"]) / (1 << 20), 1),
            "cpu_busy": cpu_busy(5),
            "governor": gov.read_text().strip() if gov.exists() else "",
        }
        if self.args.vram_cmd:
            out = subprocess.run([self.args.vram_cmd], check=False, capture_output=True, text=True, timeout=60)
            for m in re.finditer(r"card (\d+): ([\d.]+) GiB free", out.stdout):
                host[f"card{m[1]}_free_gib"] = float(m[2])
        say(f"host: {host}")
        return host

    def load_base(self) -> list[Run]:
        """Main's stored runs at this commit, when the host has not moved since; otherwise none."""
        if not self.store or not self.store.is_file():
            return []
        rec: Rec = json.loads(self.store.read_text())
        if rec["key"] != self.key:
            say("main's stored runs were taken with other inputs: not used")
            return []
        if moved := drift(rec["host"], self.host):
            say(f"main's stored runs not used, the host has moved: {'; '.join(moved)}")
            return []
        say(f"main's stored runs from {rec['made']} stand for main")
        return [Run.from_stored(r) for r in rec["runs"]]

    def save_base(self, runs: Sequence[Run]) -> None:
        """Adds main's runs that passed to the store. Runs taken on a moved host replace the stored ones."""
        if not self.store:
            return
        new = [r for r in runs if r.arm == "A" and not r.failures]
        self.base += [r for r in new if r not in self.base]
        rec = {"key": self.key, "host": self.host, "made": self.started, "runs": [r.stored() for r in self.base]}
        self.store.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.store.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, indent=1) + "\n")
        tmp.replace(self.store)

    def write(self, rec: Rec) -> None:
        with (self.out / "results.jsonl").open("a") as f:
            f.write(json.dumps(rec) + "\n")

    def tree(self, arm: str) -> Path:
        return self.a if arm == "A" else self.b

    def ctest(self, name: str, args: Sequence[str]) -> str:
        log = self.out / f"ctest-{name}.log"
        env = os.environ | {"ENGINE_EXPECT_GPUS": "2"}
        with log.open("wb") as f:
            cmd = ["ctest", *args, "--output-on-failure"]
            rc = subprocess.run(cmd, check=False, cwd=self.b / "build", env=env, stdout=f, stderr=subprocess.STDOUT)
            rc = rc.returncode
        text = log.read_text(errors="replace")
        line = next((x.strip() for x in text.splitlines() if "tests passed" in x), f"rc {rc}")
        say(f"  {line}")
        self.write({"step": name, "rc": rc, "summary": line})
        if rc:
            raise Stop(f"{line} ({log})")
        return line

    def served(self, arm: str, config: str, scenarios: Sequence[str]) -> Run:
        run = Run(len(self.runs) + 1, arm, config, tuple(scenarios), start=utc())
        self.runs.append(run)
        tree = self.tree(arm)
        flags, extra = CONFIGS[config]
        env = os.environ | extra | {"ENGINE_LIBRARY": str(tree / "build/core/libengine.so")}
        cmd = [
            str(tree / ".venv/bin/python"),
            "-m",
            "frontend",
            "--model",
            str(self.args.model),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.args.port),
            *flags,
        ]
        log = self.out / f"server-{run.n:02d}-{arm}-{config}.log"
        say(f"run {run.n}: {arm} {config}: {', '.join(scenarios)}")
        swap0 = swapped_out()
        low = [meminfo_gib()]
        done = threading.Event()
        with log.open("wb") as f:
            # The server stays in this process group, so a guard that kills the check kills the server too.
            proc = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, cwd=tree)

        def watch() -> None:
            # Below the floor the server is stopped and the run fails, so the check reports it and goes on.
            while not done.wait(1):
                low[0] = min(low[0], meminfo_gib())
                if low[0] < self.args.mem_floor:
                    run.failures.append(f"MemAvailable {low[0]:.1f} GiB, under {self.args.mem_floor} GiB")
                    proc.terminate()
                    return

        threading.Thread(target=watch, daemon=True).start()
        client = Client(self.args.port, run, self.write)
        t0 = time.monotonic()
        try:
            while client_ready(client, proc) is False:
                if time.monotonic() - t0 > 1500:
                    run.failures.append("not ready after 1,500 s")
                    break
                time.sleep(1)
            if proc.poll() is None and not run.failures:
                run.load_s = time.monotonic() - t0
                say(f"  loaded in {run.load_s:.1f} s")
                client.send(
                    "warm-up",
                    "alone",
                    "warm-up",
                    {
                        "messages": [{"role": "user", "content": "What is the capital of France?"}],
                        "max_tokens": 16,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                )
                for s in scenarios:
                    run_scenario(s, client, self.messages, self.ref_dir)
                    if run.failures:
                        break
            elif proc.poll() is not None and not run.failures:
                run.failures.append(f"server exited with {proc.returncode} while loading")
        finally:
            stop_tree(proc)
            done.set()
        errors = [x for x in log.read_text(errors="replace").splitlines() if ERROR.search(x)]
        if errors:
            run.failures.append(f"{len(errors)} error lines in {log.name}, first: {errors[0][:160]}")
        run.swapped_pages = swapped_out() - swap0
        run.mem_low_gib = low[0]
        self.write(
            {
                "run": run.n,
                "arm": arm,
                "config": config,
                "scenario": "end",
                "start": run.start,
                "load_s": run.load_s,
                "swapped_pages": run.swapped_pages,
                "mem_low_gib": run.mem_low_gib,
                "failures": run.failures,
            }
        )
        say(
            f"  swapped out {run.swapped_pages} pages, MemAvailable low {run.mem_low_gib:.1f} GiB"
            + (f"; FAIL {'; '.join(run.failures)}" if run.failures else "")
        )
        self.summary()
        time.sleep(self.args.pause)
        return run

    def round(
        self,
        config: str,
        scenarios: Sequence[str],
        after_b1: Callable[[Run, Run], None] | None = None,
    ) -> list[Run]:
        """One ABBA round. Each B run fails against the A run before it; `after_b1` may stop the round early."""
        out: list[Run] = []
        for arm in "ABBA":
            run = self.served(arm, config, scenarios)
            out.append(run)
            if arm == "B":
                a = out[0]
                bad = run.failures + (against_a(a, run) if not a.failures else [])
                if bad:
                    raise Stop(f"run {run.n} (B): {'; '.join(bad)}")
                if len(out) == 2 and after_b1:
                    after_b1(a, run)
            elif run.failures:
                raise Stop(f"run {run.n} (A, main): {'; '.join(run.failures)}")
        return out

    def times(self, runs: Sequence[Run], arm: str, scenario: str, first_token: bool = False) -> list[float]:
        return [r.first_token(scenario) if first_token else r.seconds(scenario) for r in runs if r.arm == arm]

    def step2(self) -> Step:
        return Step(
            "2. GPU tests", f"`ctest -L gpu -E '{EDGE_TESTS}'`", self.ctest("gpu", ["-L", "gpu", "-E", EDGE_TESTS])
        )

    def b_against(self, run: Run, stored: Sequence[Run]) -> Run:
        """A B run measured against main's stored runs: it fails as it would against the best of them."""
        bad = run.failures or min((against_a(a, run) for a in stored), key=len)
        if bad:
            raise Stop(f"run {run.n} (B): {'; '.join(bad)}")
        return run

    def slow(self, runs: Sequence[Run]) -> list[str]:
        return slower([r for r in runs if r.arm == "A"], [r for r in runs if r.arm == "B"], QUICK)

    def step3(self) -> tuple[Step, list[Run]]:
        def early(a1: Run, b1: Run) -> None:
            if slow := quick_stop(a1, b1):
                raise Stop(f"needs work: B1 slower than A1 by more than twice the spread: {'; '.join(slow)}")

        # The host is read here, after the GPU tests and with no server up, so the check's own work does not count.
        self.host = self.fingerprint()
        self.base = self.load_base()
        screen = ""
        if stored := [r for r in self.base if set(QUICK) <= set(r.scenarios)]:
            b = [self.b_against(self.served("B", "deployed", QUICK), stored) for _ in range(2)]
            slow = past_stored(stored, b)
            if not slow:
                ran = f"B B, deployed: short, four, mixed, against main's {len(stored)} stored runs"
                return Step("3. Quick round", ran, "pass"), []
            say(f"slower than main's stored runs in {', '.join(slow)}: fresh ABBA rounds decide")
            screen = f"B B against main's stored runs, slower in {', '.join(slow)}; then "
        runs = self.round("deployed", QUICK, early)
        if self.slow(runs):  # after one round, chance alone gives this 1 time in 6
            runs += self.round("deployed", QUICK)
        self.save_base(runs)
        if (slow := self.slow(runs)) and len(runs) >= 8:
            raise Stop(f"slower than main after 2 rounds in {', '.join(slow)}")
        rounds = "1 ABBA round" if len(runs) == 4 else f"{len(runs) // 4} ABBA rounds"
        return Step("3. Quick round", f"{screen}{rounds}, deployed: short, four, mixed", "pass"), runs

    def step4(self, quick: list[Run]) -> list[Step]:
        claim = self.args.claim
        steps: list[Step] = []
        if claim == "decode":
            runs = list(quick)
            while len(runs) < 8:
                runs += self.round("deployed", QUICK)
            signs = {s: apart(self.times(runs, "A", s), self.times(runs, "B", s)) for s in QUICK}
            if not all(signs.values()):
                runs += self.round("deployed", QUICK)
                signs = {s: apart(self.times(runs, "A", s), self.times(runs, "B", s)) for s in QUICK}
            rounds = len(runs) // 4
            ran = f"decode, {rounds} rounds deployed: short, four, mixed"
            steps.append(self.claim_step(ran, signs, runs, QUICK))
        elif claim == "prompt":
            runs = self.needs("long", lambda: self.round("one-slot", ["long"]))
            self.save_base(runs)
            signs = {"long": far_apart(self.times(runs, "A", "long", True), self.times(runs, "B", "long", True))}
            steps.append(self.claim_step("prompt, one round one slot: long (first token)", signs, runs, ["long"]))
        elif claim == "near-full":
            runs = self.needs("long-two", lambda: self.round("deployed", ["long-two"]))
            signs = {"long-two": far_apart(self.times(runs, "A", "long-two"), self.times(runs, "B", "long-two"))}
            steps.append(self.claim_step("near full, one round deployed: long-two", signs, runs, ["long-two"]))
        if self.args.rounding:
            steps.append(self.kl_ab())
        return steps

    def needs(self, scenario: str, then: Callable[[], list[Run]]) -> list[Run]:
        if scenario in self.missing:
            raise Stop(f"{scenario} cannot run: {self.missing[scenario]}")
        return then()

    def claim_step(self, ran: str, signs: dict[str, int], runs: Sequence[Run], scenarios: Sequence[str]) -> Step:
        words = {1: "faster", 0: "level", -1: "slower"}
        verdict = ", ".join(f"{s} {words[signs[s]]}" for s in scenarios)
        self.write({"step": "claim", "claim": self.args.claim, "signs": signs, "runs": [r.n for r in runs]})
        if not gain(signs):
            raise Stop(f"no gain: {verdict}")
        return Step("4. Claim", ran, verdict)

    def kl_ab(self) -> Step:
        base = self.ref_dir / "kl-base-bf16.bin"
        dumps = {arm: self.out / f"kl-{arm}.bin" for arm in "AB"}
        for arm, dump in dumps.items():
            say(f"kl_test dump, {arm}")
            log = self.out / f"kl-{arm}.log"
            with log.open("wb") as f:
                cmd = [
                    str(self.tree(arm) / "build/core/tests/kl_test"),
                    "../reference/kl-set.jsonl",
                    str(base),
                    "2",
                    "0",
                    str(dump),
                ]
                rc = subprocess.run(cmd, check=False, cwd=self.tree(arm) / "build", stdout=f, stderr=subprocess.STDOUT)
                rc = rc.returncode
            if rc:
                raise Stop(f"kl_test on {arm} rc {rc} ({log})")
        out = subprocess.run(
            [str(self.b / "build/core/tests/kl_test"), "--ab", str(base), str(dumps["A"]), str(dumps["B"])],
            check=False,
            capture_output=True,
            text=True,
        )
        (self.out / "kl-ab.log").write_text(out.stdout + out.stderr)
        last = (out.stdout.strip().splitlines() or [f"rc {out.returncode}"])[-1]
        say(f"  {last}")
        if out.returncode:
            raise Stop(f"kl_test --ab: {last}")
        return Step("4. Rounding", "`kl_test --ab` against main's dump", last)

    def step5(self, ran: set[str]) -> Step:
        results: list[str] = []
        not_run: list[str] = []
        say("edges: api_test, vision_test, encoder_test")
        results.append(f"tests {self.ctest('edges', ['-R', EDGE_TESTS])}")
        # One-card last: its load is the one most likely to take the host's memory.
        for scenario, config in (
            ("same-tokens", "same-tokens"),
            ("long", "one-slot"),
            ("long-two", "deployed"),
            ("one-card", "one-card"),
        ):
            if scenario in ran:
                continue
            why = self.skip.get(scenario) or self.missing.get(scenario)
            if why:
                not_run.append(f"{scenario} ({why})")
                say(f"edges: {scenario} not run: {why}")
                continue
            run = self.served("B", config, [scenario])
            if run.failures:
                raise Stop(f"{scenario}: {'; '.join(run.failures)}")
            results.append(self.long_against_main(run) if scenario == "long" else f"{scenario} pass")
        verdict = "pass" + (f"; not run: {', '.join(not_run)}" if not_run else "")
        return Step("5. Edges", ", ".join(results), verdict)

    def long_against_main(self, b: Run) -> str:
        """Long's first token against main's: a stored run when B is not later than it, else a fresh A run."""
        late = b.first_token("long")
        if stored := [r.first_token("long") for r in self.base if "long" in r.scenarios]:
            if late <= max(stored) * (1 + LONG_SLOWER):
                return f"long pass, first token {late:.1f} s against main's stored {max(stored):.1f} s"
            say(f"long's first token {late:.1f} s, later than main's stored {max(stored):.1f} s: a fresh A run decides")
        a = self.served("A", "one-slot", ["long"])
        if a.failures:
            raise Stop(f"long on main: {'; '.join(a.failures)}")
        self.save_base([a])
        if late > a.first_token("long") * (1 + LONG_SLOWER):
            raise Stop(f"long: first token {late:.1f} s against main's {a.first_token('long'):.1f} s")
        return f"long pass, first token {late:.1f} s against main's {a.first_token('long'):.1f} s"

    def main(self) -> int:
        say(f"check of {head(self.b)} against {head(self.a)}")
        try:
            self.steps.append(self.step2())
            quick: list[Run] = []
            if not self.args.no_served:
                step, quick = self.step3()
                self.steps.append(step)
                ran: set[str] = set()
                if self.args.claim or self.args.rounding:
                    self.steps.extend(self.step4(quick))
                    ran = {"prompt": {"long"}, "near-full": {"long-two"}}.get(self.args.claim or "", set())
                self.steps.append(self.step5(ran))
            not_run = any("not run" in s.verdict for s in self.steps)
            self.verdict = "pass, with edges not run" if not_run else "pass"
        except Stop as e:
            self.verdict = f"fail: {e}"
        say(self.verdict)
        self.summary()
        return 0 if self.verdict.startswith("pass") else 1

    def summary(self) -> None:
        """summary.md: the tables a record and a pull request take, rewritten after every run."""
        lines = [
            f"# Check of `{head(self.b)}` against `{head(self.a)}`",
            "",
            f"{self.started} to {utc()}. Verdict: **{self.verdict}**.",
            "",
            "| Step | What ran | Verdict |",
            "| --- | --- | --- |",
        ]
        lines += [f"| {s.name} | {s.ran.replace('|', r'\|')} | {s.verdict} |" for s in self.steps]
        lines += [
            "",
            "| Run | Arm | Config | Start (UTC) | Scenario | Time, s | First token, s | Decode, tokens/s | "
            "Swapped pages | Failures |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for r in self.runs:
            for s in r.scenarios:
                if not any(q["scenario"] == s for q in r.requests):
                    lines.append(
                        f"| {r.n} | {r.arm} | {r.config} | {r.start} | {s} | | | | {r.swapped_pages} | "
                        f"{'; '.join(r.failures)} |"
                    )
                    continue
                lines.append(
                    f"| {r.n} | {r.arm} | {r.config} | {r.start} | {s} | {r.seconds(s):.2f} | "
                    f"{r.first_token(s):.2f} | {r.decode(s):.1f} | {r.swapped_pages} | "
                    f"{'; '.join(r.failures)} |"
                )
        lines += ["", "Time is the scenario's slowest request, first token its latest, decode its requests' mean."]
        (self.out / "summary.md").write_text("\n".join(lines) + "\n")


def client_ready(client: Client, proc: subprocess.Popen[bytes]) -> bool | None:
    """True once /health answers 200; None when the server has exited; False while it loads."""
    if proc.poll() is not None or client.run.failures:
        return None
    try:
        return client.get("/health", 5)[0] == 200
    except OSError:
        return False


def main() -> None:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("--a", type=Path, required=True, help="main's checkout, built, the A arm")
    p.add_argument("--model", type=Path, default=os.environ.get("FLASH_MODEL"), help="the first shard (FLASH_MODEL)")
    p.add_argument("--out", type=Path, required=True, help="directory for results.jsonl, summary.md and the logs")
    p.add_argument("--claim", choices=("decode", "prompt", "near-full"), help="the speed-up the change claims")
    p.add_argument("--rounding", action="store_true", help="the change moves rounding: add kl_test --ab")
    p.add_argument(
        "--skip",
        action="append",
        default=[],
        metavar="EDGE=REASON",
        help="leave out an edge case and say why; the verdict says it did not run",
    )
    p.add_argument("--no-served", action="store_true", help="step 2 only")
    p.add_argument("--baselines", type=Path, help="directory of main's stored runs, one file per main commit")
    p.add_argument("--baseline-key", default="", help="more inputs the stored runs depend on, such as the image id")
    p.add_argument("--vram-cmd", help="a command that prints `card N: X GiB free` lines, for the host's state")
    p.add_argument("--port", type=int, default=8455)
    p.add_argument("--pause", type=float, default=10, help="seconds between runs, for the cards to settle")
    p.add_argument("--mem-floor", type=float, default=22, help="GiB of MemAvailable under which a run is stopped")
    a = p.parse_args()
    if a.model is None:
        p.error("--model or FLASH_MODEL is needed")
    raise SystemExit(Check(a).main())


if __name__ == "__main__":
    main()
