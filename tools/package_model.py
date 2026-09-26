#!/usr/bin/env python3
"""Package a saved performance model (``fit_models.py --save-model``) + its featurizer for a GitHub release.

    .venv/bin/python tools/package_model.py --tag model-prelim-v0 --model-dir results/models/prelim-v0 \\
        --featurizer results/features/prelim/features.featurizer.json --out results/releases/model-prelim-v0

Writes into ``--out``: ``<tag>.tar.gz`` (``<tag>/model/``, ``<tag>/featurizer/``, ``<tag>/roi_map/`` and
``<tag>/RELEASE_MANIFEST.json``), the same ``RELEASE_MANIFEST.json`` next to it and ``NOTES.md`` (the release
note, no numbers). ``tools/publish_model.sh`` runs this and ``gh release create``.

The manifest records the model_version, git, the training clip count, the batches (worker-output roots) and the
clips each contributed, the lockbox exclusions, the CV metrics **labelled preliminary unless the pre-registered
stage-1 GO rule passed**, and the sha256 of every file in the tarball. Refuses to package when: a media file
would be included, a lockbox clip appears in any reference table or in the training ids, or a file differs
from the model manifest's sha256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_features  # noqa: E402
import predict  # noqa: E402

MEDIA_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg",
             ".opus", ".wma"}
LICENCE = ("Research use only. TRIBE v2 is CC-BY-NC-4.0 (non-commercial); this artefact is derived from its "
           "predictions. No footage or audio is included.")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def status_label(man: dict) -> tuple[str, str | None]:
    """The model's own status under predict.py's rules; ``preliminary`` when the GO rule did not pass."""
    tm = man["targets"][predict.PRIMARY]
    st, why = predict.model_status_from(tm["metrics"], man["feature_set"], man["model"])
    return ("preliminary" if st == "not_trained" else st), why


def batches_used(features_meta: dict, train_ids: set[str], lock_ids: set[str]) -> list[dict]:
    roots = features_meta.get("out_root")
    roots = roots if isinstance(roots, list) else [roots]
    seen: set[str] = set()
    out = []
    for r in roots:
        items, failed = build_features.discover(Path(r))
        vids = [v for v, *_ in items if v not in seen]
        seen |= set(vids)
        out.append({"out_root": str(Path(r).relative_to(ROOT)) if Path(r).is_absolute() and
                    Path(r).is_relative_to(ROOT) else str(r),
                    "clips": len(vids), "failed": len(failed), "train_clips": len(set(vids) & train_ids),
                    "lockbox_clips": len(set(vids) & lock_ids)})
    return out


def check_no_lockbox(model_dir: Path, man: dict, lock_ids: set[str], train_ids: set[str]) -> dict:
    clash = sorted(train_ids & lock_ids)
    if clash:
        raise SystemExit(f"lockbox clips among the training ids: {clash[:3]}")
    found = {}
    for t, tm in man["targets"].items():
        for key in ("reference", "reference_accounts"):
            df = pd.read_csv(model_dir / tm["files"][key], dtype={"video_id": str})
            hit = sorted(set(df.get("video_id", pd.Series(dtype=str))) & lock_ids)
            if hit:
                raise SystemExit(f"{tm['files'][key]}: lockbox clips {hit[:3]} in a reference table")
            found[f"{t}/{key}"] = int(len(df))
    return found


def build(args) -> dict:
    model_dir, fz_json = args.model_dir, args.featurizer
    man = json.loads((model_dir / "manifest.json").read_text())
    for t, tm in man["targets"].items():
        for k, f in tm["files"].items():
            if sha256(model_dir / f) != tm["sha256"][k]:
                raise SystemExit(f"{model_dir / f}: sha256 differs from the model manifest")
    fz = json.loads(fz_json.read_text())
    if fz.get("features_sha256") != (man["inputs"].get("features") or {}).get("sha256"):
        raise SystemExit("the featurizer was built from a different feature table than the model was fit on")
    stem = fz_json.name.removesuffix(".featurizer.json")
    fmeta_path = fz_json.with_name(f"{stem}.meta.json")
    fmeta = json.loads(fmeta_path.read_text())
    roi = Path(args.roi_map or fz["roi_map"])
    if sha256(roi) != fz["roi_map_sha256"]:
        raise SystemExit(f"{roi}: ROI map sha256 differs from the featurizer's")
    lock_ids = set(json.loads((model_dir / man["lockbox_ids"]).read_text()))
    train_ids = set(json.loads((model_dir / "train_video_ids.json").read_text()))
    status, why = status_label(man)
    refs = check_no_lockbox(model_dir, man, lock_ids, train_ids)

    tag = args.tag
    out = args.out
    if out.exists():
        shutil.rmtree(out)
    stage = out / "stage" / tag
    shutil.copytree(model_dir, stage / "model")
    (stage / "featurizer").mkdir(parents=True)
    for f in (fz_json, fz_json.with_name(fz["npz"]), fmeta_path):
        shutil.copy2(f, stage / "featurizer" / f.name)
    (stage / "roi_map").mkdir()
    shutil.copy2(roi, stage / "roi_map" / roi.name)
    bad = [p for p in stage.rglob("*") if p.suffix.lower() in MEDIA_EXT]
    if bad:
        raise SystemExit(f"media files would ship: {bad[:3]}")

    prim = man["targets"][predict.PRIMARY]
    metrics = {t: tm["metrics"] for t, tm in man["targets"].items()}
    label = ("PRELIMINARY - n={} training clips. Not the pre-registered stage-1 analysis (that runs on the full "
             "study set) and implies nothing about its decision; noisy, not a result, do not quote."
             .format(prim["n_train_contents"])
             if status == "preliminary" else f"{status}: see docs/PREREGISTRATION.md before quoting")
    lock_in_table = len(lock_ids & set(pd.read_parquet(Path(man["inputs"]["features"]["path"]),
                                                       columns=["video_id"])["video_id"].astype(str))) \
        if (man["inputs"].get("features") or {}).get("path") and Path(man["inputs"]["features"]["path"]).exists() \
        else None
    rel = {
        "release_tag": tag, "model_version": man["model_version"], "status": status,
        "status_reason": ("the GO gate in predict.py is not met by this fit, so it may only be served with "
                          "--allow-preliminary; this is not the stage-1 analysis") if why else None,
        "predict_flag": "--allow-preliminary" if status == "preliminary" else None,
        "licence": LICENCE, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_git": man["git"], "packaged_from_git": _git_head(),
        "feature_set": man["feature_set"], "model": man["model"], "artefact_version": man["artefact_version"],
        "training": {t: {"n_train_contents": tm["n_train_contents"], "n_train_posts": tm["n_train_posts"],
                         "n_reference_contents": tm["n_reference_contents"], "n_deals": len(tm["deals"]),
                         "n_strata": len(tm["strata"])} for t, tm in man["targets"].items()},
        "batches": batches_used(fmeta, train_ids, lock_ids),
        "lockbox": {"n_lockbox_ids_in_model": len(lock_ids), "lockbox_clips_in_feature_table": lock_in_table,
                    "pca_fit_excluded": fz["pca_exclude"], "pca_fit_counts": {k: {"n_fit": b["n_fit"],
                    "n_excluded_from_fit": b["n_excluded_from_fit"]} for k, b in fz["blocks"].items()},
                    "lockbox_scored": bool(man["config"].get("score_lockbox")),
                    "lockbox_in_training_or_references": 0, "reference_rows_checked": refs},
        "cv_metrics": {"label": label, "per_target": metrics},
        "runtime": fz.get("training_runtime"),
        "contents_note": ("model/reference_*.csv hold the training contents' out-of-fold predictions, observed "
                          "within-stratum percentiles/labels and account ids (internal; the frontend never needs "
                          "this tarball: predictions ship per clip as performance.json). No lockbox clip is in "
                          "them."),
        "use": ("python tools/predict.py <vid>.npz --featurizer featurizer/" + fz_json.name + " --model-dir model "
                "--roi-map roi_map/" + roi.name + " --deal-id D --platform P"
                + (" --allow-preliminary" if status == "preliminary" else "")),
    }
    files = {str(p.relative_to(stage)): sha256(p) for p in sorted(stage.rglob("*")) if p.is_file()}
    rel["files_sha256"] = files
    (stage / "RELEASE_MANIFEST.json").write_text(json.dumps(rel, indent=2, default=str) + "\n")
    tgz = out / f"{tag}.tar.gz"
    with tarfile.open(tgz, "w:gz") as tar:
        tar.add(stage, arcname=tag)
    names = tarfile.open(tgz).getnames()
    bad = [n for n in names if Path(n).suffix.lower() in MEDIA_EXT]
    if bad:
        raise SystemExit(f"media files in the tarball: {bad[:3]}")
    shutil.copy2(stage / "RELEASE_MANIFEST.json", out / "RELEASE_MANIFEST.json")
    (out / "NOTES.md").write_text(notes(rel, tag) + "\n")
    shutil.rmtree(out / "stage")
    return {"tarball": str(tgz), "bytes": tgz.stat().st_size, "files": len(names), "status": status,
            "model_version": man["model_version"]}


def notes(rel: dict, tag: str) -> str:
    n = rel["training"][predict.PRIMARY]["n_train_contents"]
    head = ("**Preliminary, not validated.** Trained on {} clips. It is not the pre-registered stage-1 analysis "
            "and implies nothing about its outcome; its numbers only show the format.".format(n)
            if rel["status"] == "preliminary" else f"Status: {rel['status']}.")
    batches = ", ".join(f"`{b['out_root']}` ({b['clips']})" for b in rel["batches"])
    return "\n".join([
        f"Performance model `{rel['model_version']}` ({tag}).", "", head, "",
        "- Research only. TRIBE v2 is CC-BY-NC-4.0; nothing here is for a paid product or client deliverable.",
        "- No footage or audio: ridge models, PCA components, reference tables and the ROI map only.",
        "- Private: model/reference_*.csv hold training clips' observed percentiles and account ids.",
        f"- Batches (worker outputs, bf16 fast loop): {batches}.",
        "- The frontend does not need this tarball: predictions ship per clip as `performance.json` in the data "
        "release. See docs/MODEL.md.",
        f"- Every file's sha256 is in `RELEASE_MANIFEST.json` (git {str((rel['model_git'] or {}).get('commit'))[:7]}).",
    ])


def _git_head() -> dict:
    import subprocess

    try:
        c = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                           check=True).stdout.strip()
        return {"commit": c}
    except Exception:  # noqa: BLE001
        return {"commit": None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--featurizer", type=Path, required=True, help="<stem>.featurizer.json")
    ap.add_argument("--roi-map", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    print(json.dumps(build(args), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
