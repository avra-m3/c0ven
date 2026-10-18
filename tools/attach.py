"""The attach tool: stream a running engine's timing records into a Perfetto timeline (DESIGN.md, section 7).

The engine's timing rings live in a memfd. From the same container, this tool maps it through
`/proc/<pid>/fd/<fd>` (both are in the engine's INFO line), sets the detail level the engine reads once per step,
reads every ring for a while, and writes the records as Chrome trace events, which Perfetto opens. On exit it sets
the level back to 0, and the engine returns to its plain lists.

    python -m tools.attach --pid 1234 --fd 7 --seconds 10 --level 1 --out trace.json

The rings have one writer each and this tool as their only reader; it keeps its own read position per ring. Loads
on x86 are not reordered with other loads, so reading `head` before the records it covers is enough.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import mmap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from engine.api import abi
from engine.api.interface import TIMING_DTYPE, TimingRecords, TimingStage

RECORD = TIMING_DTYPE.itemsize
HEADER = ctypes.sizeof(abi.CTimingRegion)
RING = ctypes.sizeof(abi.CTimingRing)


class NotARegion(ValueError):
    pass


@dataclass(slots=True)
class Region:
    buffer: mmap.mmap
    header: abi.CTimingRegion
    rings: list[abi.CTimingRing]
    records: list[TimingRecords]

    @classmethod
    def map(cls, path: Path) -> Region:
        with path.open("r+b") as f:
            buffer = mmap.mmap(f.fileno(), 0)
        return cls.over(buffer)

    @classmethod
    def over(cls, buffer: mmap.mmap) -> Region:
        header = abi.CTimingRegion.from_buffer(buffer)
        if header.magic != abi.TIMING_MAGIC:
            raise NotARegion("not an engine timing region")
        if header.abi_version != abi.TIMING_REGION_VERSION:
            raise NotARegion(f"the region has ABI {header.abi_version}, this tool {abi.TIMING_REGION_VERSION}")
        rings: list[abi.CTimingRing] = []
        records: list[TimingRecords] = []
        stride = RING + header.ring_records * RECORD
        for i in range(header.n_rings):
            offset = HEADER + i * stride
            rings.append(abi.CTimingRing.from_buffer(buffer, offset))
            records.append(np.frombuffer(buffer, TIMING_DTYPE, header.ring_records, offset + RING))
        return cls(buffer, header, rings, records)

    def set_detail(self, level: int) -> None:
        self.header.detail = level


@dataclass(slots=True)
class Reader:
    region: Region
    tails: list[int]
    lost: int = 0

    @classmethod
    def at_head(cls, region: Region) -> Reader:
        return cls(region, [ring.head for ring in region.rings])

    def read(self) -> list[TimingRecords]:
        """New records per ring since the last read."""
        out: list[TimingRecords] = []
        size = self.region.header.ring_records
        for i, ring in enumerate(self.region.rings):
            head = ring.head
            tail = self.tails[i]
            if head - tail > size:
                self.lost += head - tail - size
                tail = head - size
            first, last = tail % size, head % size
            records = self.region.records[i]
            if head == tail:
                out.append(records[:0].copy())
            elif first < last:
                out.append(records[first:last].copy())
            else:
                out.append(np.concatenate([records[first:], records[:last]]))
            self.tails[i] = head
        return out


def trace_events(rings: list[TimingRecords], threads: list[int]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for records, thread in zip(rings, threads, strict=True):
        for r in records:
            stage = TimingStage(int(r["stage"])) if int(r["stage"]) in TimingStage else None
            events.append(
                {
                    "name": stage.name.lower() if stage is not None else f"stage {int(r['stage'])}",
                    "ph": "X",
                    "ts": int(r["t"]) / 1e3,
                    "dur": int(r["duration"]) / 1e3,
                    "pid": 0,
                    "tid": thread,
                    "args": {"step": int(r["step"]), "aux": int(r["aux"])},
                }
            )
    return events


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m tools.attach", description=(__doc__ or "").splitlines()[0])
    p.add_argument("--pid", type=int, required=True)
    p.add_argument("--fd", type=int, required=True)
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--level", type=int, default=1)
    p.add_argument("--interval", type=float, default=0.05)
    p.add_argument("--out", type=Path, default=Path("trace.json"))
    args = p.parse_args(argv)
    region = Region.map(Path(f"/proc/{args.pid}/fd/{args.fd}"))
    reader = Reader.at_head(region)
    threads = [ring.thread for ring in region.rings]
    events: list[dict[str, Any]] = []
    region.set_detail(args.level)
    try:
        end = time.monotonic() + args.seconds
        while time.monotonic() < end:
            time.sleep(args.interval)
            events += trace_events(reader.read(), threads)
    finally:
        region.set_detail(0)
    events += trace_events(reader.read(), threads)
    args.out.write_text(json.dumps({"traceEvents": events, "displayTimeUnit": "ns"}))
    print(f"{len(events)} records to {args.out}; {reader.lost} lost to full rings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
