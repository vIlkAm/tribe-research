#!/usr/bin/env python3
"""Slice the study set into ordered run batches for one pod.

    .venv/bin/python tools/make_batches.py --gpus 4          # re-run whenever the GPU count changes

Reads results/study/{pilot,manifest,deep_dive}.jsonl and writes, per batch,
``<out>/<name>/manifest.jsonl`` (workers reassigned for ``--gpus``) and
``<out>/<name>/videos/`` (hard links) for ``tools/pod.sh push-videos``:

    b00_pilot       the 10 pilot clips (timing, transcription check)
    b01_frontend    edge cases (shortest, longest) + the next clips in study order,
                    so b00+b01 is the first real bundle set for the frontend
    b02...          the rest of the study set in its stored order, --size each
    dNN_deep_dive   the account deep-dive add-on, last

The study manifest is interleaved by deal and stratum, so every batch is a small
representative sample and any finished prefix can be analysed on its own.
Batches are disjoint; all of them can share one ``$JOB/outputs`` on the pod
(the worker skips a video already done in any worker dir).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from make_manifest import assign_workers  # noqa: E402


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()] if path.exists() else []


def plan(pilot: list[dict], main: list[dict], deep: list[dict], frontend_n: int, size: int,
         edge_n: int = 2) -> list[tuple[str, list[dict]]]:
    """Ordered, disjoint batches. Order within a batch follows the study order."""
    used = {r["video_id"] for r in pilot}
    rest = [r for r in main if r["video_id"] not in used]
    by_len = sorted(rest, key=lambda r: r["duration_s"])
    edges = {r["video_id"] for r in by_len[:edge_n] + by_len[-edge_n:]}
    front = [r for r in rest if r["video_id"] in edges]
    front += [r for r in rest if r["video_id"] not in edges][: max(0, frontend_n - len(front))]
    used |= {r["video_id"] for r in front}
    rest = [r for r in rest if r["video_id"] not in used]
    batches = [("b00_pilot", pilot), ("b01_frontend", front)]
    batches += [(f"b{i + 2:02d}", rest[j:j + size]) for i, j in enumerate(range(0, len(rest), size))]
    used |= {r["video_id"] for r in rest}
    deep = [r for r in deep if r["video_id"] not in used]
    batches += [(f"d{i:02d}_deep_dive", deep[j:j + size]) for i, j in enumerate(range(0, len(deep), size))]
    return [(n, b) for n, b in batches if b]


def write_batch(name: str, rows: list[dict], out: Path, staging: Path, gpus: int) -> dict:
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    workers = assign_workers([r["duration_s"] for r in rows], gpus)
    with (d / "manifest.jsonl").open("w") as f:
        for r, w in zip(rows, workers):
            f.write(json.dumps({**r, "worker": w, "num_workers": gpus}) + "\n")
    keep = {d / "videos" / r["path"] for r in rows}
    for old in (d / "videos").rglob("*.mp4") if (d / "videos").exists() else []:
        if old not in keep:
            old.unlink()
    for r in rows:
        dst = d / "videos" / r["path"]
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        src = staging / r["path"]
        try:
            os.link(src, dst)
        except OSError:
            shutil.copyfile(src, dst)
    return {"batch": name, "clips": len(rows), "source_min": round(sum(r["duration_s"] for r in rows) / 60, 1),
            "gb": round(sum(r.get("size_bytes") or 0 for r in rows) / 1e9, 2), "gpus": gpus}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--study", type=Path, default=ROOT / "results/study")
    ap.add_argument("--out", type=Path, default=ROOT / "results/batches")
    ap.add_argument("--gpus", type=int, required=True, help="GPUs on the pod = workers per batch")
    ap.add_argument("--frontend", type=int, default=40, help="clips in b01_frontend")
    ap.add_argument("--size", type=int, default=200, help="clips per later batch")
    args = ap.parse_args()

    pilot, main_rows, deep = (read(args.study / f) for f in ("pilot.jsonl", "manifest.jsonl", "deep_dive.jsonl"))
    if not main_rows:
        ap.error(f"no study manifest in {args.study}; run tools/select_study_set.py")
    batches = plan(pilot, main_rows, deep, args.frontend, args.size)
    names = {n for n, _ in batches}
    for stale in args.out.glob("*") if args.out.exists() else []:
        if stale.is_dir() and stale.name not in names:
            shutil.rmtree(stale)  # links/manifests from an earlier slicing only
    summary = [write_batch(n, b, args.out, args.study / "staging", args.gpus) for n, b in batches]
    with (args.out / "batches.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    cum = 0.0
    for s in summary:
        cum += s["source_min"]
        print(f"{s['batch']:>16}  {s['clips']:4d} clips  {s['source_min']:6.1f} min  {s['gb']:5.2f} GB"
              f"  (cumulative {cum / 60:.1f} h)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
