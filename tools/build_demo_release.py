#!/usr/bin/env python3
"""Build the small static demo data release (``data-demo-stage1-v1``): three bundles, no footage, no weights.

    # after the stage-1 fit (model dir from fit_models.py --save-model):
    .venv/bin/python tools/build_demo_release.py --selection results/demo/selection.json \\
        --model-dir results/models/stage1 --featurizer results/features/stage1/features.featurizer.json \\
        --model-release model-stage1-v1 --out results/handoff/data-demo-stage1-v1
    # no model (GO failed and nothing is released, or a dry run):
    .venv/bin/python tools/build_demo_release.py --selection ... --not-trained --out /tmp/demo/release

Steps: ``tools/brain_report.py`` renders each selected clip's v0.2 bundle from its worker output (analysis.json,
brain_proxy.jpg, brain_vertex.jpg, shared ``_static/``; shot cuts from the pre-scaled batch clip when it is on this
server); ``tools/bundle_performance.py`` adds ``performance.json`` (clip's own post as context, never a label);
``tools/handoff.py`` validates everything (schemas, assets resolve, one model_version, clip meta) and writes
``<out>/<release>.tar.gz`` with ``index.json`` (hero first, then fallback A and B) and ``RELEASE_MANIFEST.json``
(sha256 of every file, roles, selection rule).

The status is mechanical (``predict.py``): a model that passes the pre-registered stage-1 GO rule gives
``research_preview`` (``validated: false``; training clips get their out-of-fold prediction, ranks stay null below
30 reference contents); a model that fails it gives ``not_trained`` with no numbers, drivers or reach; so does
``--not-trained``. ``preliminary`` is never requested and refused if it appears, as are ``validated``,
``out_of_scope`` and lockbox clips. Nothing is published: ``gh release create`` is a separate, manual step.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bundle_performance  # noqa: E402
import handoff  # noqa: E402
import package_model  # noqa: E402

ALLOWED = {"research_preview", "not_trained"}
LICENCE = package_model.LICENCE
DEFAULT_ROI = ROOT / "tribe_research/assets/roi_map_roi_groups_v0.npz"
NUMERIC_KEYS = ("engagement", "reach", "drivers")


def render_bundle(vid: str, out_root: Path, roi_map: Path, work: Path, videos_root: Path | None,
                  synthetic: bool = False) -> Path:
    """One clip's v0.2 bundle via brain_report.py (subprocess: the same code path as every earlier handoff)."""
    rep = work / f"report-{vid}"
    cmd = [sys.executable, str(ROOT / "tools/brain_report.py"), "--out-root", str(out_root), "--report-dir",
           str(rep), "--roi-map", str(roi_map), "--analysis", "--only", vid, "--jobs", "1"]
    if videos_root is not None:
        cmd += ["--videos-root", str(videos_root)]
    if synthetic:
        cmd.append("--synthetic")
    subprocess.run(cmd, check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    d = rep / "analyses" / vid
    if not (d / "analysis.json").exists():
        raise SystemExit(f"{vid}: brain_report wrote no analysis.json under {rep}")
    return rep / "analyses"


def videos_root_for(pick: dict, batches_dir: Path | None) -> Path | None:
    if batches_dir is None or not pick.get("batch") or not pick.get("source_path"):
        return None
    root = batches_dir / pick["batch"] / "videos"
    return root if (root / pick["source_path"]).exists() else None


def scrub_blocks(dest: Path, ids: list[str], reason: str | None) -> None:
    """Browser-facing files carry no account or post ID; a not_trained reason carries no digits (no CV numbers)."""
    if reason is not None and any(c.isdigit() for c in reason):
        raise SystemExit(f"--not-trained-reason must not contain numbers: {reason!r}")
    for vid in ids:
        f = dest / vid / "performance.json"
        blk = json.loads(f.read_text())
        blk["context"]["account_id"] = None
        if blk["model_status"] == "not_trained":
            if reason is not None:
                blk["reason"] = reason
            if any(c.isdigit() for c in blk["reason"]):
                raise SystemExit(f"{vid}: not_trained reason carries numbers: {blk['reason']!r}")
        f.write_text(json.dumps(blk, indent=2, ensure_ascii=False) + "\n")
        a = dest / vid / "analysis.json"
        if a.exists():  # source_name is the internal post id used for the outcome join; nullable in the contract
            ana = json.loads(a.read_text())
            ana["source_name"] = None
            a.write_text(json.dumps(ana, ensure_ascii=False) + "\n")


def check_blocks(dest: Path, ids: list[str], not_trained: bool) -> dict:
    statuses = {}
    for vid in ids:
        blk = json.loads((dest / vid / "performance.json").read_text())
        st = blk["model_status"]
        if st not in ALLOWED:
            raise SystemExit(f"{vid}: model_status {st!r} is not allowed in the demo release "
                             f"(only {sorted(ALLOWED)}; preliminary/validated/out_of_scope are refused)")
        if not_trained and st != "not_trained":
            raise SystemExit(f"{vid}: --not-trained but the block says {st!r}")
        if blk["clip_in_training"] == "lockbox":
            raise SystemExit(f"{vid}: lockbox clip in the demo release")
        if blk["validated"]:
            raise SystemExit(f"{vid}: validated is true; stage 1 is never validated")
        if st == "not_trained" and any(blk.get(k) for k in NUMERIC_KEYS):
            raise SystemExit(f"{vid}: not_trained block carries {[k for k in NUMERIC_KEYS if blk.get(k)]}")
        statuses[vid] = {"model_status": st, "clip_in_training": blk["clip_in_training"],
                         "brain_claim": blk["brain_claim"], "model_version": blk["model_version"]}
    if len({s["model_status"] for s in statuses.values()}) > 1:
        raise SystemExit(f"mixed statuses in one release: {statuses}")
    return statuses


def build(args) -> dict:
    sel = json.loads(args.selection.read_text())
    picks = sel["picks"]
    if len(picks) != 3 or not sel.get("complete"):
        raise SystemExit(f"{args.selection}: need a complete selection of 3 picks, got {len(picks)}")
    if args.not_trained == bool(args.model_dir):
        raise SystemExit("give exactly one of --model-dir or --not-trained")
    if args.model_dir and not args.featurizer:
        raise SystemExit("--model-dir needs --featurizer")
    ids = [p["video_id"] for p in picks]
    roots = [Path(r) for r in (args.out_root or [b["out_root"] for b in sel["batches_used"]])]

    out = args.out
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    work = out / "work"
    src = work / "analyses"
    src.mkdir(parents=True)
    for p in picks:
        npz = bundle_performance.find_output(roots, p["video_id"])
        if npz is None:
            raise SystemExit(f"{p['video_id']}: no worker output under the given roots")
        ana = render_bundle(p["video_id"], npz.parent.parent, args.roi_map, work,
                            videos_root_for(p, args.batches_dir))
        shutil.copytree(ana / p["video_id"], src / p["video_id"])
        if not (src / "_static").exists():
            shutil.copytree(ana / "_static", src / "_static")

    dest = work / "bundles"
    bp = argparse.Namespace(
        analyses=src, out_root=roots, featurizer=args.featurizer, model_dir=args.model_dir, dest=dest,
        clip_meta=work / "clip_meta.json", selection=args.study_selection, lockbox_ext=args.lockbox_ext,
        members=args.members, outcomes=args.outcomes, allow_preliminary=False)
    bundle_performance.run(bp)
    scrub_blocks(dest, ids, args.not_trained_reason if args.not_trained else None)
    statuses = check_blocks(dest, ids, args.not_trained)
    meta = json.loads(bp.clip_meta.read_text())
    if any(m["is_lockbox"] for m in meta):
        raise SystemExit("a lockbox clip reached the clip meta")
    status = next(iter(statuses.values()))["model_status"]
    model_version = next(iter(statuses.values()))["model_version"]

    by_id = {m["video_id"]: m for m in meta}
    extra = {
        "licence": LICENCE, "status": status, "validated": False,
        "status_rule": ("mechanical from tools/predict.py: research_preview only if the saved model passes the "
                        "pre-registered stage-1 GO rule, else not_trained; preliminary is refused"),
        "packaged_from_git": package_model._git_head(),
        "model_dir": str(args.model_dir) if args.model_dir else None,
        "model_manifest_sha256": package_model.sha256(args.model_dir / "manifest.json") if args.model_dir else None,
        "featurizer_sha256": package_model.sha256(args.featurizer) if args.featurizer else None,
        "selection": {"rule": sel["rule"], "params": sel["params"], "ids": ids,
                      "selection_json_sha256": package_model.sha256(args.selection),
                      "batches_used": [Path(b["out_root"]).name for b in sel["batches_used"]]},
        "clips": [{"role": p["role"], "video_id": p["video_id"], "deal_label": by_id[p["video_id"]]["deal_label"],
                   "platform": by_id[p["video_id"]]["platform"], "duration_s": p["duration_s"],
                   "video_link": by_id[p["video_id"]]["video_link"], "is_lockbox": False,
                   **statuses[p["video_id"]]} for p in picks],
        "contents_note": ("No footage, audio, model weights, encoders, reference tables, raw vertex arrays or "
                          "observed outcomes. Source filenames for the owner's media mapping are in the internal "
                          "selection JSON, not here."),
    }
    rm = work / "release_manifest_extra.json"
    rm.write_text(json.dumps(extra, indent=1, default=str))
    tgz = out / f"{args.release}.tar.gz"
    rc = handoff.main(["--analyses", str(dest), "--out", str(tgz), "--expect-real", "--clip-meta",
                       str(bp.clip_meta), "--require-performance", "--release", args.release,
                       "--release-manifest", str(rm), "--order", *ids]
                      + (["--model-release", args.model_release] if args.model_release else []))
    if rc != 0:
        raise SystemExit("handoff validation failed (see above)")
    names = tarfile.open(tgz).getnames()
    bad = [n for n in names if Path(n).suffix.lower() in package_model.MEDIA_EXT | {".npz", ".npy", ".joblib"}]
    if bad:
        raise SystemExit(f"forbidden files in the tarball: {bad[:3]}")
    (out / "CONTENTS.txt").write_text("\n".join(names) + "\n")
    if not args.keep_work:
        shutil.rmtree(work)
    return {"tarball": str(tgz), "sha256": package_model.sha256(tgz), "files": len(names), "status": status,
            "model_version": model_version, "ids": ids,
            "manifest": str(tgz.with_name(f"{args.release}.RELEASE_MANIFEST.json")),
            "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--selection", type=Path, required=True, help="tools/select_demo_clips.py output")
    ap.add_argument("--out-root", type=Path, action="append", default=None,
                    help="worker outputs (default: the selection's batches_used)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--model-dir", type=Path, default=None, help="fit_models.py --save-model dir")
    g.add_argument("--not-trained", action="store_true", help="no model: every block is not_trained")
    ap.add_argument("--not-trained-reason", default=None,
                    help="reason shown in every not_trained block (no digits), e.g. the stage-1 no-GO sentence")
    ap.add_argument("--featurizer", type=Path, default=None, help="<stem>.featurizer.json (with --model-dir)")
    ap.add_argument("--model-release", default=None, help="private model release tag, e.g. model-stage1-v1")
    ap.add_argument("--release", default="data-demo-stage1-v1")
    ap.add_argument("--out", type=Path, required=True, help="empty dir for the tarball + manifest")
    ap.add_argument("--roi-map", type=Path, default=DEFAULT_ROI)
    ap.add_argument("--batches-dir", type=Path, default=ROOT / "results/batches_s384",
                    help="pre-scaled clips for shot cuts (read locally, never shipped)")
    ap.add_argument("--study-selection", type=Path, default=ROOT / "results/study/selection.csv")
    ap.add_argument("--lockbox-ext", type=Path, default=ROOT / "results/study/lockbox_ext.csv")
    ap.add_argument("--members", type=Path, default=ROOT / "results/run_full/members.csv")
    ap.add_argument("--outcomes", type=Path, default=ROOT / "results/outcomes.parquet",
                    help="read for the post context columns only (bundle_performance.CONTEXT_COLS)")
    ap.add_argument("--keep-work", action="store_true")
    args = ap.parse_args(argv)
    print(json.dumps(build(args), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
