#!/usr/bin/env python3
"""Plan a full TRIBE run: every clip on this server, each unique content once.

Inputs are the read-only export from ``tools/export_metrics.sh`` (no live DB
query here). Clip files come from two places on this server, both under the
media archive root: the 2026-09-26 backfill (``media_backfill_results``) and the
regular archive (``media_archive_assets``). Nothing is only on Drive.

Unique content:
  * identical files (same sha256) are one content;
  * the same clip posted on several platforms (``media_cross_platform_links``)
    is one content. Suspicious links (more than 3 members, or two members on
    one platform) are not merged; their members stay separate.
The representative file is the largest one (best bitrate); TRIBE runs on it
once and the outcome of every member joins back through ``members.csv``.

Order: deals are interleaved round-robin so a partial run covers every deal;
inside a deal, content whose members have >= 7 days of tracked history comes
first. The run is cut into chunks in that order; workers are balanced by
duration inside each chunk, so pushing chunk k lets every GPU start on it.

Output (``--out``, default results/run_full):
    manifest.jsonl   worker.py manifest rows + chunk, deal_id, n_members
    members.csv      video_id, vp_id, platform, deal_id, representative
    staging/<chunk>/<deal_slug>/<video_id>.mp4   hard links (copies across filesystems)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from make_manifest import assign_workers, probe_duration  # noqa: E402

csv.field_size_limit(1 << 30)
HOME = Path.home()
DEFAULT_ARCHIVE = HOME / "services/clipping-cartel/media_archive_worker/archive"
CONTAINER_ARCHIVE = "/archive/"


def read(metrics: Path, name: str) -> list[dict]:
    with (metrics / f"{name}.csv").open(newline="") as f:
        return list(csv.DictReader(f))


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (s or "unknown").lower()).strip("-")[:40] or "unknown"


class DSU:
    def __init__(self):
        self.p: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def collect_files(metrics: Path, archive_root: Path) -> dict[str, dict]:
    """vp_id -> {path, sha256, size, platform, deal_id}; backfill wins over archive."""
    files: dict[str, dict] = {}
    for r in read(metrics, "media_archive_assets"):
        lp = r["local_archive_path"]
        if lp.startswith(CONTAINER_ARCHIVE) and not r["local_deleted_at"] and r["sha256"]:
            files[r["video_performance_id"]] = {
                "src": archive_root / lp[len(CONTAINER_ARCHIVE):], "sha256": r["sha256"],
                "size": int(r["file_size_bytes"] or 0), "platform": r["platform"], "deal_id": r["deal_id"]}
    latest: dict[str, dict] = {}
    for r in read(metrics, "media_backfill_results"):
        if r["status"] == "uploaded" and r["sha256"] and r["local_path"].startswith(CONTAINER_ARCHIVE):
            prev = latest.get(r["video_performance_id"])
            if prev is None or r["updated_at"] > prev["updated_at"]:
                latest[r["video_performance_id"]] = r
    for vp, r in latest.items():
        files[vp] = {"src": archive_root / r["local_path"][len(CONTAINER_ARCHIVE):], "sha256": r["sha256"],
                     "size": int(r["file_size_bytes"] or 0), "platform": r["platform"], "deal_id": r["deal_id"]}
    return files


def suspicious_links(members: list[dict], platform_of: dict[str, str]) -> set[str]:
    by_link: dict[str, list[str]] = defaultdict(list)
    for m in members:
        by_link[m["link_id"]].append(m["video_performance_id"])
    bad = set()
    for link, vps in by_link.items():
        plats = [platform_of.get(v) for v in vps]
        if len(vps) > 3 or len(plats) != len(set(plats)):
            bad.add(link)
    return bad


def history_days(metrics: Path) -> dict[str, float]:
    """vp_id -> days between upload and last tracked observation (cheap, from video_performances)."""
    out = {}
    for r in read(metrics, "video_performances"):
        try:
            up = date.fromisoformat(r["upload_date"])
            last = datetime.fromisoformat(r["last_updated"].replace("Z", "+00:00")).date()
            out[r["id"]] = (last - up).days
        except (ValueError, TypeError):
            continue
    return out


def interleave(groups: dict[str, list]) -> list:
    order, queues = [], [list(v) for _, v in sorted(groups.items(), key=lambda kv: -len(kv[1]))]
    while any(queues):
        for q in queues:
            if q:
                order.append(q.pop(0))
    return order


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--metrics", type=Path, default=ROOT / "results/metrics")
    ap.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE)
    ap.add_argument("--out", type=Path, default=ROOT / "results/run_full")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunks", type=int, default=6)
    ap.add_argument("--min-duration", type=float, default=2.0)
    ap.add_argument("--max-duration", type=float, default=600.0)
    ap.add_argument("--no-stage", action="store_true", help="plan only, don't create staging links")
    args = ap.parse_args()

    files = collect_files(args.metrics, args.archive_root)
    deal_name = {r["id"]: r["name"] for r in read(args.metrics, "deals")}
    members = read(args.metrics, "cross_platform_members")
    bad = suspicious_links(members, {v: f["platform"] for v, f in files.items()})

    dsu = DSU()
    by_sha: dict[str, str] = {}
    for vp, f in files.items():
        dsu.find(vp)
        if f["sha256"] in by_sha:
            dsu.union(vp, by_sha[f["sha256"]])
        else:
            by_sha[f["sha256"]] = vp
    link_first: dict[str, str] = {}
    for m in members:
        vp = m["video_performance_id"]
        if m["link_id"] in bad or vp not in files:
            continue
        if m["link_id"] in link_first:
            dsu.union(vp, link_first[m["link_id"]])
        else:
            link_first[m["link_id"]] = vp

    groups: dict[str, list[str]] = defaultdict(list)
    for vp in files:
        groups[dsu.find(vp)].append(vp)

    # Representative: largest file; probe its duration (in parallel).
    reps = {g: max(vps, key=lambda v: (files[v]["size"], v)) for g, vps in groups.items()}
    missing = [g for g, v in reps.items() if not files[v]["src"].is_file()]
    for g in missing:
        alt = [v for v in groups[g] if files[v]["src"].is_file()]
        if alt:
            reps[g] = max(alt, key=lambda v: (files[v]["size"], v))
        else:
            reps.pop(g)
    with ThreadPoolExecutor(max_workers=min(48, os.cpu_count() or 8)) as ex:
        durs = dict(zip(reps, ex.map(lambda g: probe_duration(files[reps[g]]["src"]), reps)))

    hist = history_days(args.metrics)
    per_deal: dict[str, list[str]] = defaultdict(list)
    skipped = defaultdict(int)
    for g, rep in reps.items():
        d = durs.get(g)
        if not d:
            skipped["no_duration"] += 1
            continue
        if not (args.min_duration <= d <= args.max_duration):
            skipped["duration_out_of_range"] += 1
            continue
        per_deal[files[rep]["deal_id"]].append(g)
    for deal, gs in per_deal.items():
        gs.sort(key=lambda g: (-(max(hist.get(v, -1) for v in groups[g]) >= 7), -len(groups[g]), g))
    order = interleave(per_deal)

    args.out.mkdir(parents=True, exist_ok=True)
    rows, member_rows = [], []
    n = len(order)
    for i, g in enumerate(order):
        rep = reps[g]
        f = files[rep]
        vid = f["sha256"][:16]
        chunk = i * args.chunks // n
        deal = f["deal_id"]
        rows.append({"video_id": vid, "path": f"{slug(deal_name.get(deal, deal))}/{vid}.mp4",
                     "source_name": rep, "duration_s": round(durs[g], 3), "size_bytes": f["size"],
                     "chunk": chunk, "deal_id": deal, "n_members": len(groups[g]), "_src": str(f["src"])})
        for v in sorted(groups[g]):
            member_rows.append({"video_id": vid, "vp_id": v, "platform": files[v]["platform"],
                                "deal_id": files[v]["deal_id"], "representative": v == rep})
    for c in range(args.chunks):
        idx = [i for i, r in enumerate(rows) if r["chunk"] == c]
        for i, w in zip(idx, assign_workers([rows[i]["duration_s"] for i in idx], args.workers)):
            rows[i]["worker"], rows[i]["num_workers"] = w, args.workers

    if not args.no_stage:
        for r in rows:
            dst = args.out / "staging" / f"chunk{r['chunk']}" / r["path"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                continue
            try:
                os.link(r["_src"], dst)
            except OSError:
                shutil.copyfile(r["_src"], dst)
    with (args.out / "manifest.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps({k: v for k, v in r.items() if k != "_src"}) + "\n")
    with (args.out / "members.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(member_rows[0]))
        w.writeheader()
        w.writerows(member_rows)

    total_min = sum(r["duration_s"] for r in rows) / 60
    print(f"{len(files)} clip files -> {len(groups)} unique contents; {len(rows)} planned "
          f"({total_min:.0f} min of source, {sum(r['size_bytes'] for r in rows) / 1e9:.1f} GB), "
          f"{len(bad)} suspicious links not merged, skipped {dict(skipped)}", file=sys.stderr)
    for c in range(args.chunks):
        cr = [r for r in rows if r["chunk"] == c]
        print(f"  chunk {c}: {len(cr)} clips, {sum(r['duration_s'] for r in cr) / 60:.0f} min", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
