#!/usr/bin/env python3
"""Pick the demo clips (hero + fallback A + fallback B) deterministically, without reading any outcome.

    .venv/bin/python tools/select_demo_clips.py --out results/demo/selection.json

Reads only: ``selection.csv`` columns video_id/path/deal_id/split/duration_s/audio_mean_db,
``lockbox_ext.csv`` video_id, ``members.csv`` video_id/vp_id/platform/deal_id, the batch manifests
(video_id/path/source_name/duration_s) and the worker ``<vid>.json`` files (status, transcript language, word
count). Never ``outcomes.parquet``, never a label or performance column, never a model.

Eligibility (RULE below): study-set train clip outside both lockboxes, worker output ok in a complete batch
(``done-<b>`` marker, or every manifest clip accounted for: ok, worker error or prep failure), whisper language ``en`` with probability
>= 0.9, 20-60 s, >= 1.5 words/s, audio mean level >= -30 dB.

Draw: each eligible clip's stratum is (deal, platform of its own post: the manifest ``source_name``); a stratum is
preferred when it has >= 30 study train contents (counted from members.csv + selection.csv, no labels), so a
rank can exist. Clips are ordered by sha256("<seed>:<video_id>") (seed 20260926), preferred strata first; the
first clip of each new deal is taken until three deals are chosen: hero, fallback A, fallback B. Hash order keeps
a pick stable when more batches land unless a new clip hashes ahead of it.

The output JSON holds the rule, the parameters, the inputs' sha256 and per pick the video_id, deal, platform,
duration, source filename (manifest ``path``) and ``source_name``. Source filenames name the deal folder: keep this
file internal (B8 media mapping), not in a release.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_features  # noqa: E402

SEED = 20260926
PARAMS = {"seed": SEED, "language": "en", "min_language_probability": 0.9, "min_duration_s": 20.0,
          "max_duration_s": 60.0, "min_words_per_s": 1.5, "min_audio_mean_db": -30.0, "min_stratum_contents": 30,
          "n_picks": 3}
ROLES = ("hero", "fallback_a", "fallback_b")
RULE = (
    "Eligible: split=train in selection.csv and not in lockbox_ext.csv (outside both lockboxes); worker output "
    "status ok in a complete batch (done-<b> marker, or every manifest clip is ok, a worker error or a prep "
    "failure); whisper detected_language == 'en' with language_probability >= 0.9; "
    "duration 20-60 s; words per second (transcript n_words / duration) >= 1.5; selection.csv audio_mean_db >= "
    "-30 dB. Stratum = (deal, platform of the clip's own post, the manifest source_name); preferred when it holds "
    ">= 30 study train contents counted from members.csv (no labels). Order: preferred strata first, then "
    "sha256('20260926:<video_id>') ascending; take the first clip of each new deal until 3 distinct deals: hero, "
    "fallback A, fallback B. No outcome, label, performance column or model is read.")
SEL_COLS = ["video_id", "path", "deal_id", "split", "duration_s", "audio_mean_db"]
MEMBER_COLS = ["video_id", "vp_id", "platform", "deal_id"]
MANIFEST_KEYS = ("video_id", "path", "source_name", "duration_s")


def hash_rank(vid: str, seed: int = SEED) -> str:
    return hashlib.sha256(f"{seed}:{vid}".encode()).hexdigest()


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def batch_complete(out_root: Path, batches_dir: Path, logs: list[Path]) -> tuple[bool, str]:
    """A100 queue: ``done-<b>`` in a logs dir. Otherwise (L40S/4090) every manifest clip must be accounted for:
    a completed output (npz + emb.npz), a worker ``.error.json`` or a prep failure in ``prep.json``. With no
    failures that is "npz and emb counts equal the manifest"."""
    b = out_root.name.removeprefix("outputs-")
    if any((d / f"done-{b}").exists() for d in logs):
        return True, f"done-{b}"
    man = batches_dir / b / "manifest.jsonl"
    if not man.exists():
        return False, "no manifest"
    rows = [json.loads(line) for line in man.read_text().splitlines() if line.strip()]
    npz = {p.stem for p in out_root.glob("worker-*/*.npz") if not p.name.endswith(".emb.npz")}
    emb = {p.name.removesuffix(".emb.npz") for p in out_root.glob("worker-*/*.emb.npz")}
    err = {p.name.removesuffix(".error.json") for p in out_root.glob("worker-*/*.error.json")}
    prep = batches_dir / b / "prep.json"
    prep_failed = set((json.loads(prep.read_text()).get("failed") or {}) if prep.exists() else ())
    ok_ids = npz & emb
    missing = [r["video_id"] for r in rows
               if r["video_id"] not in ok_ids and r["video_id"] not in err and r.get("path") not in prep_failed]
    why = (f"accounted: {len(ok_ids & {r['video_id'] for r in rows})} ok + {len(err)} worker errors + "
           f"{len(prep_failed)} prep failures of {len(rows)}; {len(missing)} missing")
    return not missing, why


def load_manifests(batches_dir: Path, names: list[str]) -> dict[str, dict]:
    rows = {}
    for b in names:
        p = batches_dir / b / "manifest.jsonl"
        if p.exists():
            for line in p.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    rows[r["video_id"]] = {k: r.get(k) for k in MANIFEST_KEYS} | {"batch": b}
    return rows


def eligible(out_roots: list[Path], selection: Path, lockbox_ext: Path | None, members: Path, batches_dir: Path,
             logs: list[Path], params: dict = PARAMS) -> tuple[list[dict], dict, list[dict], list[dict]]:
    """Every clip passing the eligibility rule (no outcome read): (eligible, counts, used roots, skipped roots)."""
    sel = pd.read_csv(selection, usecols=SEL_COLS, dtype={"video_id": str, "deal_id": str})
    ext = set(pd.read_csv(lockbox_ext, usecols=["video_id"], dtype=str)["video_id"]) \
        if lockbox_ext is not None and lockbox_ext.exists() else set()
    mem = pd.read_csv(members, usecols=MEMBER_COLS, dtype=str)
    train = sel[(sel["split"] == "train") & ~sel["video_id"].isin(ext)].set_index("video_id")

    batches, used_roots, skipped = [], [], []
    for r in out_roots:
        ok, why = batch_complete(r, batches_dir, logs)
        (used_roots if ok else skipped).append({"out_root": str(r), "evidence": why})
        if ok:
            batches.append(r.name.removeprefix("outputs-"))
    manifest = load_manifests(batches_dir, batches)
    items, _ = build_features.discover_all([Path(u["out_root"]) for u in used_roots])

    tm = mem[mem["video_id"].isin(train.index)]
    stratum_n = tm.drop_duplicates(["video_id", "deal_id", "platform"]).groupby(["deal_id", "platform"]).size()

    counts = {"completed_in_complete_batches": len(items)}
    elig = []
    for vid, meta_path, _npz in items:
        why = None
        if vid not in train.index:
            why = "not a study train clip outside both lockboxes"
        m = json.loads(Path(meta_path).read_text())
        tr = m.get("transcript") or {}
        dur = float(m.get("duration_s") or 0.0)
        s = train.loc[vid] if why is None else None
        mrow = manifest.get(vid)
        if why is None and mrow is None:
            why = "not in a batch manifest"
        if why is None and ((m.get("runtime") or {}).get("dry_run")):
            why = "dry run"
        if why is None and (tr.get("detected_language") != params["language"]
                            or float(tr.get("language_probability") or 0) < params["min_language_probability"]):
            why = "language"
        if why is None and not (params["min_duration_s"] <= dur <= params["max_duration_s"]):
            why = "duration"
        wps = float(tr.get("n_words") or 0) / dur if dur > 0 else 0.0
        if why is None and wps < params["min_words_per_s"]:
            why = "words_per_s"
        if why is None and not (pd.notna(s["audio_mean_db"]) and float(s["audio_mean_db"]) >= params["min_audio_mean_db"]):
            why = "audio_level"
        if why is None:
            own = mem[(mem["video_id"] == vid) & (mem["vp_id"] == str(mrow["source_name"]))]
            if own.empty:
                why = "source_name not among the content's posts"
        if why is not None:
            counts[f"excluded_{why}"] = counts.get(f"excluded_{why}", 0) + 1
            continue
        deal, plat = str(own.iloc[0]["deal_id"]), str(own.iloc[0]["platform"])
        n = int(stratum_n.get((deal, plat), 0))
        elig.append({"video_id": vid, "deal_id": deal, "platform": plat, "stratum_train_contents": n,
                     "preferred_stratum": n >= params["min_stratum_contents"], "duration_s": round(dur, 3),
                     "words_per_s": round(wps, 3), "language_probability": float(tr["language_probability"]),
                     "audio_mean_db": float(s["audio_mean_db"]), "source_path": mrow["path"],
                     "source_name": mrow["source_name"], "batch": mrow["batch"],
                     "rank_key": hash_rank(vid, params["seed"])})
    counts["eligible"] = len(elig)
    return elig, counts, used_roots, skipped


def select(out_roots: list[Path], selection: Path, lockbox_ext: Path | None, members: Path, batches_dir: Path,
           logs: list[Path], params: dict = PARAMS) -> dict:
    elig, counts, used_roots, skipped = eligible(out_roots, selection, lockbox_ext, members, batches_dir, logs,
                                                 params)
    counts["eligible_in_preferred_strata"] = sum(e["preferred_stratum"] for e in elig)
    order = sorted(elig, key=lambda e: (not e["preferred_stratum"], e["rank_key"]))
    picks, deals = [], set()
    for e in order:
        if e["deal_id"] in deals:
            continue
        deals.add(e["deal_id"])
        picks.append({"role": ROLES[len(picks)], **e})
        if len(picks) == params["n_picks"]:
            break
    return {
        "rule": RULE, "params": params, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs": {"selection": {"path": str(selection), "sha256": _sha(selection), "columns": SEL_COLS},
                   "lockbox_ext": {"path": str(lockbox_ext), "sha256": _sha(lockbox_ext)}
                   if lockbox_ext is not None and lockbox_ext.exists() else None,
                   "members": {"path": str(members), "sha256": _sha(members), "columns": MEMBER_COLS},
                   "outcomes_read": False},
        "batches_used": used_roots, "batches_skipped_incomplete": skipped, "counts": counts,
        "complete": len(picks) == params["n_picks"], "picks": picks,
        "ids": [p["video_id"] for p in picks],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-root", type=Path, action="append", default=None,
                    help="worker outputs (repeatable; default results/runs/study-bf16/outputs-b0*)")
    ap.add_argument("--selection", type=Path, default=ROOT / "results/study/selection.csv")
    ap.add_argument("--lockbox-ext", type=Path, default=ROOT / "results/study/lockbox_ext.csv")
    ap.add_argument("--members", type=Path, default=ROOT / "results/run_full/members.csv")
    ap.add_argument("--batches-dir", type=Path, default=ROOT / "results/batches_s384")
    ap.add_argument("--logs", type=Path, action="append", default=None,
                    help="dirs with done-<b> markers (default results/runs/study-bf16/study2-logs)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    roots = args.out_root or sorted((ROOT / "results/runs/study-bf16").glob("outputs-b0*"))
    logs = args.logs or [ROOT / "results/runs/study-bf16/study2-logs"]
    res = select(roots, args.selection, args.lockbox_ext, args.members, args.batches_dir, logs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({"counts": res["counts"], "batches_used": [b["out_root"] for b in res["batches_used"]],
                      "skipped": res["batches_skipped_incomplete"],
                      "picks": [{k: p[k] for k in ("role", "video_id", "deal_id", "platform", "duration_s",
                                                    "stratum_train_contents", "source_path")} for p in res["picks"]]},
                     indent=1))
    return 0 if res["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
