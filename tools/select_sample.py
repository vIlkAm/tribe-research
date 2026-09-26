#!/usr/bin/env python3
"""Copy a stratified sample of backfilled clips into a TRIBE videos folder.

Reads the backfill's ``*.results.jsonl`` logs (file-based: no database query),
keeps rows whose clip is under ``<archive-root>/backfill-20260926/``, and
samples round-robin across (deal, platform, duration bucket) strata so one big
deal or platform can't dominate. Files are **copied**, never symlinked (rsync
would push a symlink as a broken link), and each copy's sha256 must match the
recorded one. Identical content under two ids is sent once.

Output::

    <out>/<platform>/<vp_id>.mp4     -> make_manifest.py source_name = vp_id
    <out>/_sample.jsonl              vp_id, deal_id, platform, duration_s, sha256, bucket

Usage:
    .venv/bin/python tools/select_sample.py --n 60 --out videos
    tools/pod.sh push-videos videos
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from make_manifest import probe_duration  # noqa: E402

HOME = Path.home()
DEFAULT_RESULTS = str(HOME / "projects/media-backfill-20260926/logs/*.results.jsonl")
DEFAULT_ARCHIVE = HOME / "services/clipping-cartel/media_archive_worker/archive"
CONTAINER_ARCHIVE = "/archive/"
ALLOWED_PREFIX = "backfill-20260926/"  # AGENTS.md: read-only copies from this folder only
BUCKETS = [(0, 15, "<15s"), (15, 30, "15-30s"), (30, 60, "30-60s"), (60, float("inf"), ">60s")]


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def bucket(d: float) -> str:
    return next(name for lo, hi, name in BUCKETS if lo <= d < hi)


def load_candidates(results_glob: str, archive_root: Path) -> list[dict]:
    """Latest uploaded/downloaded row per vp_id whose file sits in the allowed folder."""
    latest: dict[str, dict] = {}
    for f in sorted(glob.glob(results_glob)):
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("status") not in ("uploaded", "downloaded") or not r.get("sha256"):
                continue
            lp = r.get("local_path") or ""
            if not lp.startswith(CONTAINER_ARCHIVE + ALLOWED_PREFIX):
                continue
            latest[r["vp_id"]] = {
                "vp_id": r["vp_id"], "deal_id": r.get("deal_id"), "platform": r.get("platform"),
                "sha256": r["sha256"], "src": archive_root / lp[len(CONTAINER_ARCHIVE):],
            }
    return list(latest.values())


def stratified(cands: list[dict], n: int, seed: int) -> list[dict]:
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for c in cands:
        strata[(c["deal_id"], c["platform"], c["bucket"])].append(c)
    rng = random.Random(seed)
    keys = sorted(strata)
    for k in keys:
        rng.shuffle(strata[k])
    rng.shuffle(keys)
    picked, seen_sha = [], set()
    while len(picked) < n and any(strata[k] for k in keys):
        for k in keys:
            while strata[k]:
                c = strata[k].pop()
                if c["sha256"] not in seen_sha:
                    seen_sha.add(c["sha256"])
                    picked.append(c)
                    break
            if len(picked) >= n:
                break
    return picked


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--results", default=DEFAULT_RESULTS, help="glob of backfill results.jsonl files")
    ap.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE)
    ap.add_argument("--platforms", nargs="*", default=None)
    ap.add_argument("--min-duration", type=float, default=3.0)
    ap.add_argument("--max-duration", type=float, default=90.0, help="skip longer clips (cost control)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, copy nothing")
    args = ap.parse_args()

    cands = load_candidates(args.results, args.archive_root)
    if args.platforms:
        cands = [c for c in cands if c["platform"] in set(args.platforms)]
    usable, missing = [], 0
    for c in cands:
        if not c["src"].is_file():
            missing += 1
            continue
        d = probe_duration(c["src"])
        if d and args.min_duration <= d <= args.max_duration:
            c["duration_s"], c["bucket"] = round(d, 3), bucket(d)
            usable.append(c)
    print(f"{len(cands)} candidates, {missing} missing on disk, {len(usable)} within "
          f"{args.min_duration:g}-{args.max_duration:g}s", file=sys.stderr)

    picked = stratified(usable, args.n, args.seed)
    strata = {(c["deal_id"], c["platform"], c["bucket"]) for c in picked}
    total = sum(c["duration_s"] for c in picked)
    print(f"picked {len(picked)} clips, {total / 60:.1f} min of source, {len(strata)} strata, "
          f"{len({c['deal_id'] for c in picked})} deals", file=sys.stderr)
    if args.dry_run:
        for c in picked:
            print(c["platform"], c["bucket"], c["deal_id"], c["vp_id"], c["duration_s"])
        return 0

    rows, bad = [], []
    for c in picked:
        dst = args.out / c["platform"] / f"{c['vp_id']}.mp4"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_symlink():
            dst.unlink()
        if not (dst.is_file() and sha256(dst) == c["sha256"]):
            shutil.copyfile(c["src"], dst)
            if sha256(dst) != c["sha256"]:
                dst.unlink()
                bad.append(c["vp_id"])
                continue
        rows.append({k: c[k] for k in ("vp_id", "deal_id", "platform", "duration_s", "sha256", "bucket")})
    with (args.out / "_sample.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"copied/verified {len(rows)} -> {args.out}" + (f"; sha256 mismatch, skipped: {bad}" if bad else ""),
          file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
