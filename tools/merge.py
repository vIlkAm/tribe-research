#!/usr/bin/env python3
"""Collect per-worker outputs into one index and report gaps against the manifest.

Writes ``<out-root>/index.jsonl`` (one metadata row per completed video, with a
``preds_file`` path relative to out-root) and ``<out-root>/benchmark.json``
(throughput per worker and overall). Raw predictions stay in the per-video
.npz files; nothing is concatenated.

Exit code is non-zero while any manifest video is missing or failed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    args = ap.parse_args()

    manifest = [json.loads(l) for l in args.manifest.read_text().splitlines() if l.strip()]
    done, failed = {}, {}
    for meta_path in sorted(args.out_root.glob("worker-*/*.json")):
        if meta_path.name.endswith(".error.json"):
            err = json.loads(meta_path.read_text())
            failed[err["video_id"]] = err
            continue
        meta = json.loads(meta_path.read_text())
        meta["preds_file"] = meta_path.with_suffix(".npz").relative_to(args.out_root).as_posix()
        done[meta["video_id"]] = meta

    missing = [r for r in manifest if r["video_id"] not in done]
    per_worker: dict[int, dict] = {}
    for m in done.values():
        w = per_worker.setdefault(m["worker_id"], {"videos": 0, "source_s": 0.0, "compute_s": 0.0})
        w["videos"] += 1
        w["source_s"] += m["duration_s"]
        w["compute_s"] += m["timing_s"]["total"]
    for w in per_worker.values():
        w["source_s_per_compute_s"] = round(w["source_s"] / w["compute_s"], 4) if w["compute_s"] else None

    src = sum(w["source_s"] for w in per_worker.values())
    wall = max((w["compute_s"] for w in per_worker.values()), default=0.0)
    categories: dict[str, int] = {}
    for e in failed.values():
        categories[e.get("category", "other")] = categories.get(e.get("category", "other"), 0) + 1
    benchmark = {
        "failures_by_category": categories,
        "videos_done": len(done),
        "videos_in_manifest": len(manifest),
        "source_minutes": round(src / 60, 2),
        "slowest_worker_compute_minutes": round(wall / 60, 2),
        "per_worker": {f"worker-{k}": v for k, v in sorted(per_worker.items())},
        "note": "compute excludes model load and setup; add pod billing time separately",
    }

    with (args.out_root / "index.jsonl").open("w") as f:
        for r in manifest:
            if r["video_id"] in done:
                f.write(json.dumps(done[r["video_id"]]) + "\n")
    (args.out_root / "benchmark.json").write_text(json.dumps(benchmark, indent=2) + "\n")

    print(json.dumps(benchmark, indent=2))
    for r in missing:
        err = failed.get(r["video_id"])
        status = f"FAILED [{err.get('category', '?')}] {err['error']}" if err else "missing"
        print(f"{status}: worker-{r['worker']} {r['path']}", file=sys.stderr)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
