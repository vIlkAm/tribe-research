#!/usr/bin/env python3
"""Pick an internal "fell short" / "beat expectations" example pair in the hero's stratum, by a fixed rule.

    .venv/bin/python tools/select_example_pair.py --out results/demo/example_pair.json

INTERNAL ONLY: reads observed outcomes (training split) and names source files. Never goes into a release, and
never shown for a lockbox clip. The pair is an illustration for the tailnet walkthrough, not evidence: stage 1
(M4) found no reliable difference in the predicted brain response between clips that beat or missed
expectations.

Rule (RULE below): the eligible clips of ``tools/select_demo_clips.py`` (train outside both lockboxes, English,
20-60 s, >= 1.5 words/s, >= -30 dB, worker output ok) whose own post is in the hero's (deal, platform) stratum.
Each clip is labelled by its own post (the manifest ``source_name``): the ``log_interactions_rate`` residual after
the arm-A out-of-fold prediction (content scheme), cut into thirds over every OOF post in that stratum.
"Fell short": bottom third and ``reach_rel_local`` < 0; "beat expectations": top third and ``reach_rel_local``
> 0, so the views panel agrees with the label. Both need empty ``dq_flags`` and a full-resolution source file on
this server. Order: sha256("20260926:<video_id>"), first of each; the most extreme residual is not used.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import select_demo_clips as sdc  # noqa: E402

TARGET = "log_interactions_rate"
RULE = (
    "Eligible as in tools/select_demo_clips.py; own post (manifest source_name) in the hero's deal|platform "
    "stratum. Residual = y - pred_A_stack for log_interactions_rate (content-scheme OOF), thirds over every OOF "
    "post in that stratum. fell_short: bottom third and reach_rel_local < 0; beat_expectations: top third and "
    "reach_rel_local > 0; both with empty dq_flags and a full-resolution source file present. Order: "
    "sha256('20260926:<video_id>') ascending, first of each. Internal only: reads observed outcomes.")
OBSERVED_COLS = ["id", "platform", "video_link", "upload_date", "views_final", "likes", "comments", "shares",
                 "saves", "engagement_rate_reported", "reach_rel_local", "local_baseline_n", "age_days_at_last_obs",
                 "dq_flags"]


def label_posts(oof: Path, stratum: str) -> pd.DataFrame:
    """Every OOF post of one stratum with its residual and third (-1 bottom, 0 middle, +1 top)."""
    o = pd.read_csv(oof, dtype={"video_id": str, "vp_id": str, "stratum": str})
    o = o[(o["scheme"] == "content") & (o["target"] == TARGET) & (o["stratum"] == stratum)].copy()
    if len(o) < 30:
        raise SystemExit(f"stratum {stratum}: {len(o)} OOF posts, need >= 30 for thirds")
    o["residual"] = o["y"] - o["pred_A_stack"]
    lo, hi = np.quantile(o["residual"], [1 / 3, 2 / 3])
    o["third"] = np.where(o["residual"] <= lo, -1, np.where(o["residual"] >= hi, 1, 0))
    o["residual_pct"] = o["residual"].rank(pct=True)
    return o.set_index("vp_id")


def source_file(staging: Path, source_path: str) -> Path | None:
    hit = sorted(staging.glob(f"chunk*/{source_path}"))
    return hit[0] if hit else None


def observed_block(row: pd.Series) -> dict:
    """What the clip got on the platform (observed), for the internal view only. No account or post id."""
    def num(k):
        v = row.get(k)
        return None if v is None or pd.isna(v) else float(v)
    return {
        "platform": row["platform"], "video_link": row["video_link"],
        "upload_date": None if pd.isna(row["upload_date"]) else str(pd.Timestamp(row["upload_date"]).date()),
        "views": num("views_final"), "likes": num("likes"), "comments": num("comments"), "shares": num("shares"),
        "saves": num("saves"), "engagement_rate_pct": num("engagement_rate_reported"),
        "views_vs_account_usual_log": num("reach_rel_local"),
        "views_vs_account_usual_x": None if pd.isna(row["reach_rel_local"]) else float(np.exp(row["reach_rel_local"])),
        "account_usual_n_posts": num("local_baseline_n"), "age_days_at_last_obs": num("age_days_at_last_obs"),
    }


def pick(elig: list[dict], hero: dict, posts: pd.DataFrame, outcomes: pd.DataFrame, lockbox: set[str],
         staging: Path) -> dict:
    stratum = f"{hero['deal_id']}|{hero['platform']}"
    cands, counts = [], {"eligible_in_stratum": 0}
    for e in elig:
        if (e["deal_id"], e["platform"]) != (hero["deal_id"], hero["platform"]):
            continue
        counts["eligible_in_stratum"] += 1
        if e["video_id"] in lockbox:
            raise SystemExit(f"{e['video_id']}: lockbox clip among the eligible clips")
        vp = e["source_name"]
        if vp not in posts.index or vp not in outcomes.index:
            counts["no_label"] = counts.get("no_label", 0) + 1
            continue
        p, oc = posts.loc[vp], outcomes.loc[vp]
        if isinstance(oc["dq_flags"], str) and oc["dq_flags"]:
            counts["dq_flags"] = counts.get("dq_flags", 0) + 1
            continue
        rr = oc["reach_rel_local"]
        role = ("fell_short" if p["third"] == -1 and pd.notna(rr) and rr < 0 else
                "beat_expectations" if p["third"] == 1 and pd.notna(rr) and rr > 0 else None)
        if role is None:
            counts["not_in_a_group"] = counts.get("not_in_a_group", 0) + 1
            continue
        src = source_file(staging, e["source_path"])
        if src is None:
            counts["no_source_file"] = counts.get("no_source_file", 0) + 1
            continue
        cands.append({"role": role, **e, "source_file": str(src), "residual": float(p["residual"]),
                      "residual_pct_in_stratum": float(p["residual_pct"]), "reach_rel_local": float(rr),
                      "observed": observed_block(oc)})
        counts[f"candidates_{role}"] = counts.get(f"candidates_{role}", 0) + 1
    picks = []
    for role in ("fell_short", "beat_expectations"):
        group = sorted((c for c in cands if c["role"] == role), key=lambda c: c["rank_key"])
        if group:
            picks.append(group[0])
    return {"stratum": stratum, "n_oof_posts_in_stratum": int(len(posts)), "counts": counts, "picks": picks}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--demo-selection", type=Path, default=ROOT / "results/demo/selection.json")
    ap.add_argument("--oof", type=Path, default=ROOT / "results/models/stage1-eval/oof_predictions.csv")
    ap.add_argument("--outcomes", type=Path, default=ROOT / "results/outcomes.parquet")
    ap.add_argument("--staging", type=Path, default=ROOT / "results/run_full/staging")
    ap.add_argument("--selection", type=Path, default=ROOT / "results/study/selection.csv")
    ap.add_argument("--lockbox-ext", type=Path, default=ROOT / "results/study/lockbox_ext.csv")
    ap.add_argument("--members", type=Path, default=ROOT / "results/run_full/members.csv")
    ap.add_argument("--batches-dir", type=Path, default=ROOT / "results/batches_s384")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    demo = json.loads(args.demo_selection.read_text())
    hero = next(p for p in demo["picks"] if p["role"] == "hero")
    roots = [Path(b["out_root"]) for b in demo["batches_used"]]
    logs = [ROOT / "results/runs/study-bf16/study2-logs"]
    elig, _, _, _ = sdc.eligible(roots, args.selection, args.lockbox_ext, args.members, args.batches_dir, logs)
    sel = pd.read_csv(args.selection, usecols=["video_id", "split"], dtype=str)
    lockbox = set(sel.loc[sel["split"] == "lockbox", "video_id"])
    if args.lockbox_ext.exists():
        lockbox |= set(pd.read_csv(args.lockbox_ext, usecols=["video_id"], dtype=str)["video_id"])
    posts = label_posts(args.oof, f"{hero['deal_id']}|{hero['platform']}")
    outcomes = pd.read_parquet(args.outcomes, columns=OBSERVED_COLS).astype({"id": str}).set_index("id")
    res = pick(elig, hero, posts, outcomes, lockbox, args.staging)
    res.update({"rule": RULE, "internal_only": True, "hero": hero["video_id"],
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "complete": len(res["picks"]) == 2})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps({"counts": res["counts"], "picks": [
        {k: p[k] for k in ("role", "video_id", "duration_s", "residual_pct_in_stratum", "reach_rel_local")}
        | {"views": p["observed"]["views"]} for p in res["picks"]]}, indent=1))
    return 0 if res["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
