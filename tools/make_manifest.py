#!/usr/bin/env python3
"""Build manifest.jsonl for a TRIBE batch and assign videos to workers.

Each line describes one video:

    {"video_id": "3f2a...", "path": "viral/clip.mp4", "source_name": "clip",
     "duration_s": 41.2, "size_bytes": 1234567, "worker": 2, "num_workers": 4}

``video_id`` is derived from file content, so re-running the tool on the same
library yields the same IDs (and a worker's resume check still matches).

Workers are assigned by total source duration (greedy longest-first), not
round-robin, so one shard doesn't end up with all the long clips.

Duration comes from ffprobe when available, otherwise from a pure-Python
MP4/MOV ``mvhd`` reader (enough for the formats short-form platforms export).

Usage:
    python tools/make_manifest.py --videos-root /workspace/tribe-job/videos \
        --workers 4 --out /workspace/tribe-job/manifest.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi"}
# TRIBE's get_events_dataframe accepts only these; anything else must be remuxed.
TRIBE_SUFFIXES = {".mp4", ".avi", ".mkv", ".mov", ".webm"}


def content_id(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()[:16]


def _iter_atoms(f, start: int, end: int):
    pos = start
    while pos + 8 <= end:
        f.seek(pos)
        header = f.read(8)
        if len(header) < 8:
            return
        size, kind = struct.unpack(">I4s", header)
        header_len = 8
        if size == 1:
            size = struct.unpack(">Q", f.read(8))[0]
            header_len = 16
        elif size == 0:
            size = end - pos
        if size < header_len:
            return
        yield kind, pos + header_len, pos + size
        pos += size


def mvhd_duration(path: Path) -> float | None:
    """Duration from the moov/mvhd atom of an MP4/MOV file, or None."""
    with path.open("rb") as f:
        f.seek(0, 2)
        file_end = f.tell()
        for kind, body, end in _iter_atoms(f, 0, file_end):
            if kind != b"moov":
                continue
            for sub, sub_body, _ in _iter_atoms(f, body, end):
                if sub != b"mvhd":
                    continue
                f.seek(sub_body)
                version = f.read(1)[0]
                f.read(3)  # flags
                if version == 1:
                    f.read(16)  # creation + modification time
                    timescale, duration = struct.unpack(">IQ", f.read(12))
                else:
                    f.read(8)
                    timescale, duration = struct.unpack(">II", f.read(8))
                return duration / timescale if timescale else None
    return None


def ffprobe_duration(path: Path) -> float | None:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(out.stdout.strip())
    except ValueError:
        return None


def probe_duration(path: Path) -> float | None:
    if shutil.which("ffprobe"):
        d = ffprobe_duration(path)
        if d:
            return d
    if path.suffix.lower() in {".mp4", ".mov", ".m4v"}:
        return mvhd_duration(path)
    return None


def assign_workers(durations: list[float], num_workers: int) -> list[int]:
    """Greedy longest-processing-time assignment. Returns worker index per item."""
    if num_workers < 1:
        raise ValueError("num_workers must be >= 1")
    heap = [(0.0, w) for w in range(num_workers)]
    assignment = [0] * len(durations)
    for i in sorted(range(len(durations)), key=lambda i: -durations[i]):
        load, w = heapq.heappop(heap)
        assignment[i] = w
        heapq.heappush(heap, (load + durations[i], w))
    return assignment


def build(videos_root: Path, num_workers: int) -> tuple[list[dict], list[str]]:
    rows, problems, seen = [], [], {}
    for path in sorted(p for p in videos_root.rglob("*") if p.is_file()):
        suffix = path.suffix.lower()
        if suffix not in VIDEO_SUFFIXES:
            continue
        rel = path.relative_to(videos_root).as_posix()
        if suffix not in TRIBE_SUFFIXES:
            problems.append(f"{rel}: {suffix} not accepted by TRIBE; remux to .mp4")
            continue
        duration = probe_duration(path)
        if not duration or duration <= 0:
            problems.append(f"{rel}: could not determine duration")
            continue
        vid = content_id(path)
        if vid in seen:
            problems.append(f"{rel}: duplicate content of {seen[vid]}; skipped")
            continue
        seen[vid] = rel
        rows.append({
            "video_id": vid,
            "path": rel,
            "source_name": path.stem,
            "duration_s": round(duration, 3),
            "size_bytes": path.stat().st_size,
        })
    for row, w in zip(rows, assign_workers([r["duration_s"] for r in rows], num_workers)):
        row["worker"] = w
        row["num_workers"] = num_workers
    return rows, problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--videos-root", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows, problems = build(args.videos_root, args.workers)
    for p in problems:
        print(f"WARN {p}", file=sys.stderr)
    if not rows:
        print("no usable videos found", file=sys.stderr)
        return 1

    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    with tmp.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    tmp.replace(args.out)

    loads = [0.0] * args.workers
    counts = [0] * args.workers
    for r in rows:
        loads[r["worker"]] += r["duration_s"]
        counts[r["worker"]] += 1
    total = sum(loads)
    print(f"{len(rows)} videos, {total / 60:.1f} min total -> {args.out}")
    for w in range(args.workers):
        print(f"  worker-{w}: {counts[w]:4d} videos  {loads[w] / 60:6.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
