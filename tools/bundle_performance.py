#!/usr/bin/env python3
"""Add a ``performance.json`` (tools/predict.py) next to each bundle's ``analysis.json``, plus the per-clip
index metadata tools/handoff.py puts into ``index.json``.

    .venv/bin/python tools/bundle_performance.py --analyses results/handoff/frontend40/analyses \\
        --out-root results/runs/frontend40-bf16/outputs --featurizer results/features/prelim/features.featurizer.json \\
        --model-dir results/models/prelim-v0 --dest results/handoff/frontend40_v2/analyses --allow-preliminary

The source bundles are copied (only the files handoff ships, plus ``_static/``) into ``--dest``; the source
directory is never modified, so the v0.2 ``analysis.json`` stays byte-identical.

Posting context, one post per clip: the bundle's ``source_name`` (the batch manifest's ``source_name``, which is
the content's ``representative`` post in ``members.csv``). From that post in ``outcomes.parquet`` only explicit
context columns are read (deal_id, platform, social_account_id, follower_count, upload_date, video_link), never a
label; width, height and audio level come from ``selection.csv``. Lockbox clips (``split == lockbox`` in the
selection, or in the stage-2 extension) are flagged ``is_lockbox``; no observed outcome of any clip is written.

``deal_label`` is a neutral ``Deal <first 8 hex of deal_id>``: client/deal names never enter a bundle.
Writes ``<dest>/../clip_meta.json`` (``--clip-meta`` to override) for ``tools/handoff.py --clip-meta``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_features  # noqa: E402
import handoff  # noqa: E402
import predict  # noqa: E402

CONTEXT_COLS = ["id", "deal_id", "platform", "social_account_id", "follower_count", "upload_date", "video_link"]


def deal_label(deal_id: str) -> str:
    return f"Deal {str(deal_id).replace('-', '')[:8]}"


def find_output(roots: list[Path], vid: str) -> Path | None:
    for r in roots:
        hit = sorted(r.glob(f"worker-*/{vid}.npz"))
        if hit:
            return hit[0]
    return None


def lockbox_set(selection: pd.DataFrame, ext_path: Path | None) -> set[str]:
    ids = set(selection.loc[selection["split"].astype(str) == "lockbox", "video_id"].astype(str))
    if ext_path is not None and ext_path.exists():
        ids |= set(pd.read_csv(ext_path, dtype={"video_id": str})["video_id"])
    return ids


def post_context(vid: str, vp_id: str, posts: pd.DataFrame, members: pd.DataFrame,
                 sel: pd.DataFrame) -> tuple[dict, str | None]:
    """The clip's own post: ``vp_id`` must be one of the content's posts in members.csv."""
    m = members[(members["video_id"] == vid) & (members["vp_id"] == vp_id)]
    if m.empty:
        raise SystemExit(f"{vid}: source_name {vp_id} is not one of its posts in members.csv")
    if vp_id not in posts.index:
        raise SystemExit(f"{vid}: post {vp_id} is not in outcomes")
    p = posts.loc[vp_id]
    s = sel.loc[vid] if vid in sel.index else None
    up = p["upload_date"]
    ctx = {"deal_id": str(p["deal_id"]), "deal_label": deal_label(p["deal_id"]), "platform": str(p["platform"]),
           "account_id": None if pd.isna(p["social_account_id"]) else str(p["social_account_id"]),
           "follower_count": None if pd.isna(p["follower_count"]) else float(p["follower_count"]),
           "posted_at": None if pd.isna(up) else pd.Timestamp(up).isoformat(), "prescale": "s384"}
    if s is not None:
        for k in ("width", "height", "audio_mean_db"):
            if k in s and pd.notna(s[k]):
                ctx[k] = float(s[k])
    return ctx, str(p["video_link"]) if pd.notna(p["video_link"]) else None


def run(args) -> dict:
    sel = pd.read_csv(args.selection, dtype={"video_id": str})
    lock = lockbox_set(sel, args.lockbox_ext)
    sel = sel.set_index("video_id")
    members = pd.read_csv(args.members, dtype=str)
    posts = pd.read_parquet(args.outcomes, columns=CONTEXT_COLS).astype({"id": str}).set_index("id")
    # predict.py reads the featurizer only to score (a GO model, or --allow-preliminary); none without a model
    fz = build_features.load_featurizer(args.featurizer) if args.featurizer else None
    roots = [Path(r) for r in args.out_root]
    dest = args.dest
    if dest.exists() and any(dest.iterdir()):
        raise SystemExit(f"{dest} is not empty")
    dest.mkdir(parents=True, exist_ok=True)
    if (args.analyses / "_static").exists():
        shutil.copytree(args.analyses / "_static", dest / "_static")
    meta, counts = [], {}
    for ap in sorted(args.analyses.glob("*/analysis.json")):
        a = json.loads(ap.read_text())
        vid = a["video_id"]
        npz = find_output(roots, vid)
        if npz is None:
            raise SystemExit(f"{vid}: no worker output under {', '.join(map(str, roots))}")
        ctx, link = post_context(vid, str(a["source_name"]), posts, members, sel)
        blk = predict.predict_performance(npz, ctx, fz, args.model_dir, allow_preliminary=args.allow_preliminary)
        is_lock = vid in lock
        if is_lock != (blk["clip_in_training"] == "lockbox"):
            raise SystemExit(f"{vid}: lockbox flag {is_lock} vs predict.py clip_in_training "
                             f"{blk['clip_in_training']!r}: the model's lockbox list and the selection disagree")
        out = dest / vid
        out.mkdir()
        for f in sorted(ap.parent.iterdir()):
            if f.name in handoff.SHIP and f.name != "performance.json":
                shutil.copy2(f, out / f.name)
        (out / "performance.json").write_text(json.dumps(blk, indent=2, ensure_ascii=False) + "\n")
        meta.append({"video_id": vid, "platform": ctx["platform"], "video_link": link, "is_lockbox": is_lock,
                     "vp_id": str(a["source_name"]), "deal_id": ctx["deal_id"], "deal_label": ctx["deal_label"]})
        key = f"{blk['model_status']}/{blk['clip_in_training']}"
        counts[key] = counts.get(key, 0) + 1
    clip_meta = args.clip_meta or dest.parent / "clip_meta.json"
    clip_meta.write_text(json.dumps(meta, indent=1) + "\n")
    return {"bundles": len(meta), "status/clip_in_training": counts, "clip_meta": str(clip_meta), "dest": str(dest)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--analyses", type=Path, required=True, help="source bundles (<video_id>/analysis.json)")
    ap.add_argument("--out-root", type=Path, required=True, action="append", help="worker outputs (repeatable)")
    ap.add_argument("--featurizer", type=Path, default=None, help="needed whenever a model will score")
    ap.add_argument("--model-dir", type=Path, default=None, help="absent: every block is not_trained")
    ap.add_argument("--dest", type=Path, required=True, help="new analyses dir (must be empty or absent)")
    ap.add_argument("--clip-meta", type=Path, default=None)
    ap.add_argument("--selection", type=Path, default=ROOT / "results/study/selection.csv")
    ap.add_argument("--lockbox-ext", type=Path, default=ROOT / "results/study/lockbox_ext.csv")
    ap.add_argument("--members", type=Path, default=ROOT / "results/run_full/members.csv")
    ap.add_argument("--outcomes", type=Path, default=ROOT / "results/outcomes.parquet")
    ap.add_argument("--allow-preliminary", action="store_true", help="passed to predict.py")
    args = ap.parse_args(argv)
    from threadpoolctl import threadpool_limits

    with threadpool_limits(4):
        print(json.dumps(run(args), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
