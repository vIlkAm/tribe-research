#!/usr/bin/env python3
"""Stage 2 of docs/PREREGISTRATION.md: the rest of the eligible contents, plus the lockbox extension.

    .venv/bin/python tools/select_rest.py                # -> results/study/rest.jsonl, lockbox_ext.csv, rest/*
    .venv/bin/python tools/select_rest.py --gpus 2 --size 400

1. Checks the pre-registered input hashes (outcomes, run manifest, members, selection). A mismatch stops
   the draw: the extension must come from the pinned files.
2. Eligible contents are the ones ``select_study_set.py`` counts as eligible, minus everything already in
   the study set, the account deep-dive or the pilot.
3. **Lockbox extension:** 15 % of each deal's new eligible contents, uniform, seed 20260926. Candidates
   linked by ``content_group`` to a content already selected (train or lockbox) are not drawn: their twin
   is already in the study set, so they could never be a clean hold-out. Every remaining content linked to
   an extension content is dropped from the rest-train, so no twin of a hold-out clip is trained on.
4. Writes ``results/study/lockbox_ext.csv`` (the sealed list, with the input hashes) and ordered batches
   ``results/batches/rNN/`` (manifest + hard-linked videos, same layout as ``make_batches.py``). Batches are
   interleaved by deal in a seeded random order, so any finished prefix is a representative sample.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import select_study_set as sss  # noqa: E402
from make_manifest import assign_workers  # noqa: E402

SEED = 20260926
PINNED = {  # docs/PREREGISTRATION.md, stage 2
    "results/outcomes.parquet": "bc679801467603ea5b3bb9608ff057e6a9702685b72be65ccc9337b4e1d5a02c",
    "results/run_full/manifest.jsonl": "9f778476862eea2a95584fb212492da74052a6ab82ca8ceccf9d73df86a91397",
    "results/run_full/members.csv": "85c1d1eb8752e682d07667252ac1608bdc0aebd952261efda4f7f6ad66a7cae1",
    "results/study/selection.csv": "dc47ba561d64e48a725d252afd489a65e0d26493c4cf21a4ddf3753ceed599cb",
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def content_groups(members: pd.DataFrame, outcomes: pd.DataFrame) -> dict[str, set]:
    """video_id -> the outcomes content_groups of its posts (reposts of one content share a group)."""
    if "content_group" not in outcomes:
        return {}
    cg = members.merge(outcomes[["id", "content_group"]], left_on="vp_id", right_on="id", how="inner")
    cg = cg.dropna(subset=["content_group"])
    return cg.groupby("video_id")["content_group"].agg(set).to_dict()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gpus", type=int, default=2, help="workers per pod (reassign later with the pod's count)")
    ap.add_argument("--size", type=int, default=400, help="clips per batch")
    ap.add_argument("--lockbox", type=float, default=0.15)
    ap.add_argument("--no-stage", action="store_true", help="plan only: no batch directories")
    args = ap.parse_args()

    bad = {p: h for p, h in PINNED.items() if sha256(ROOT / p) != h}
    if bad:
        print(f"pinned inputs changed, refusing to draw: {sorted(bad)}", file=sys.stderr)
        return 2

    sargs = argparse.Namespace(run=ROOT / "results/run_full", outcomes=ROOT / "results/outcomes.parquet",
                               metrics=ROOT / "results/metrics", min_duration=5.0, max_duration=90.0)
    c, _ = sss.load(sargs)
    c = sss.eligibility(c, sargs)
    study = ROOT / "results/study"
    taken = set()
    for f in ("manifest.jsonl", "deep_dive.jsonl", "pilot.jsonl"):
        if (study / f).exists():
            taken |= set(pd.read_json(study / f, lines=True)["video_id"])
    el = c[(c["excluded"] == "") & ~c["video_id"].isin(taken)].copy()

    members = pd.read_csv(sargs.run / "members.csv")
    outcomes = pd.read_parquet(sargs.outcomes)
    groups = content_groups(members, outcomes)
    taken_groups = set().union(*(groups.get(v, set()) for v in taken)) if taken else set()
    el["_linked_to_study"] = el["video_id"].map(lambda v: bool(groups.get(v, set()) & taken_groups))

    rng = np.random.default_rng(SEED)
    lock = []
    for deal, g in el.sort_values("video_id").groupby("deal_id", sort=True):
        n = int(round(args.lockbox * len(g)))
        pool = g.loc[~g["_linked_to_study"], "video_id"].to_numpy()
        lock += list(rng.permutation(pool)[: min(n, len(pool))])
    lock = set(lock)
    lock_groups = set().union(*(groups.get(v, set()) for v in lock)) if lock else set()
    el["split"] = np.where(el["video_id"].isin(lock), "lockbox", "train")
    twin = (el["split"] == "train") & el["video_id"].map(lambda v: bool(groups.get(v, set()) & lock_groups))
    dropped = int(twin.sum())
    el = el[~twin].copy()

    # seeded deal-interleaved order: round-robin over deals, random within a deal
    el["_r"] = rng.random(len(el))
    el = el.sort_values(["deal_id", "_r"])
    el["_k"] = el.groupby("deal_id").cumcount() / el.groupby("deal_id")["video_id"].transform("size")
    el = el.sort_values(["_k", "_r"]).reset_index(drop=True)
    el["stratum"] = np.where(el["split"] == "lockbox", "uniform", "rest")
    el["incl_prob"] = 1.0

    hashes = {p: PINNED[p] for p in PINNED}
    lb = el[el["split"] == "lockbox"][["video_id", "path", "deal_id", "duration_s"]].copy()
    lb.to_csv(study / "lockbox_ext.csv", index=False)
    (study / "lockbox_ext.meta.json").write_text(json.dumps({
        "seed": SEED, "share_per_deal": args.lockbox, "n_lockbox_ext": int(len(lb)),
        "n_rest_train": int((el["split"] == "train").sum()), "dropped_twins_of_lockbox": dropped,
        "not_drawable_linked_to_study": int(el["_linked_to_study"].sum()),
        "pinned_inputs_sha256": hashes, "rule": "docs/PREREGISTRATION.md stage 2",
    }, indent=1))

    extra = ["split", "stratum", "incl_prob", "deal_id", "n_members", "platforms"]
    rows = [{k: (None if isinstance(r[k], float) and np.isnan(r[k]) else (r[k].item() if hasattr(r[k], "item")
             else r[k])) for k in ["video_id", "path", "source_name", "duration_s", "size_bytes", "chunk"] + extra}
            for _, r in el.iterrows()]
    with (study / "rest.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"rest: {len(el)} contents ({len(lb)} lockbox ext, {dropped} twins dropped), "
          f"{el['duration_s'].sum() / 3600:.1f} h of source", file=sys.stderr)
    if args.no_stage:
        return 0
    stats = []
    for i, j in enumerate(range(0, len(rows), args.size)):
        chunk = rows[j:j + args.size]
        d = ROOT / "results/batches" / f"r{i:02d}"
        d.mkdir(parents=True, exist_ok=True)
        workers = assign_workers([r["duration_s"] for r in chunk], args.gpus)
        with (d / "manifest.jsonl").open("w") as f:
            for r, w in zip(chunk, workers):
                f.write(json.dumps({**{k: v for k, v in r.items() if k != "chunk"},
                                    "worker": w, "num_workers": args.gpus}) + "\n")
        for r in chunk:
            dst = d / "videos" / r["path"]
            if dst.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            src = sargs.run / "staging" / f"chunk{int(r['chunk'])}" / r["path"]
            try:
                os.link(src, dst)
            except OSError:
                shutil.copyfile(src, dst)
        stats.append({"batch": d.name, "clips": len(chunk),
                      "source_min": round(sum(r["duration_s"] for r in chunk) / 60, 1),
                      "gb": round(sum(r.get("size_bytes") or 0 for r in chunk) / 1e9, 2)})
    for s in stats:
        print(json.dumps(s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
