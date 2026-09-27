#!/usr/bin/env python3
"""Stage the INTERNAL tailnet demo set under frontend/public/ (git-ignored): showcase + example pair.

    .venv/bin/python tools/stage_demo_local.py            # uses results/demo/showcase.json + example_pair.json

Writes ``frontend/public/demo-stage1/`` (index.json in demo order: spike, flat, beat_expectations, fell_short;
per clip analysis.json, performance.json = not_trained, library.json; observed.json for the pair only;
examples.json) and ``frontend/public/clips/<id>.mp4`` (full-resolution source). Bundle dirs and clips not in the
new index are removed from public/ so they cannot be picked on stage. The release ``data-demo-stage1-v1``
tarball is not touched. Observed numbers only for the pair (train split), never a lockbox clip; nothing here is
ever published.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_demo_release as bdr  # noqa: E402
import bundle_performance  # noqa: E402

ORDER = ("spike", "flat", "beat_expectations", "fell_short")
CAPTION_OBS = ("Observed on platform (the post's own numbers, last snapshot). Internal view only; never shown for "
               "sealed test clips or in a release.")
CAPTION_NULL = "The response scores above did not predict these numbers: stage 1 found no reliable link (no-GO)."
PLAIN = {
    "fell_short": ("Fell short of what the metadata-only model expected for this account and platform (bottom third "
                   "of interactions, fewer views than the account's recent usual)."),
    "beat_expectations": ("Beat what the metadata-only model expected for this account and platform (top third of "
                          "interactions, more views than the account's recent usual)."),
}
CAVEAT = ("One illustrative pair picked by a fixed rule, not evidence. Across 1,155 training clips, those that beat "
          "expectations and those that fell short show the same predicted brain response on average (see What the "
          "model learned). Any difference between these two clips is a property of these two clips.")


def plan(showcase: dict, pair: dict) -> list[dict]:
    """Ordered demo clips: first spike, first flat, then the pair (beat, fell short); one entry per video."""
    by_role = {"spike": showcase["spike"][0], "flat": showcase["flat"][0]}
    by_role.update({p["role"]: p for p in pair["picks"]})
    missing = [r for r in ORDER if r not in by_role]
    if missing:
        raise SystemExit(f"no pick for {missing}")
    out = []
    for role in ORDER:
        p = by_role[role]
        if any(o["video_id"] == p["video_id"] for o in out):
            raise SystemExit(f"{p['video_id']} would fill two demo roles")
        m = None
        if "window_ms" in p:
            m = {"start_ms": int(p["window_ms"][0]), "end_ms": int(p["window_ms"][1]), "label": p["label"]}
        out.append({"role": role, "video_id": p["video_id"], "batch": p["batch"], "source_path": p["source_path"],
                    "source_file": p["source_file"], "demo_moment": m, "observed": p.get("observed"),
                    "residual_pct_in_stratum": p.get("residual_pct_in_stratum")})
    return out


def render(vid: str, batch: str, roots: list[Path], work: Path, reason: str) -> Path:
    npz = bundle_performance.find_output(roots, vid)
    if npz is None:
        raise SystemExit(f"{vid}: no worker output")
    vroot = ROOT / "results/batches_s384" / batch / "videos"
    ana = bdr.render_bundle(vid, npz.parent.parent, bdr.DEFAULT_ROI, work / vid, vroot if vroot.exists() else None)
    src = work / vid / "src"
    shutil.copytree(ana / vid, src / vid)
    shutil.copytree(ana / "_static", src / "_static")
    dest = work / vid / "bundles"
    bp = argparse.Namespace(
        analyses=src, out_root=roots, featurizer=None, model_dir=None, dest=dest, clip_meta=work / vid / "meta.json",
        selection=ROOT / "results/study/selection.csv", lockbox_ext=ROOT / "results/study/lockbox_ext.csv",
        members=ROOT / "results/run_full/members.csv", outcomes=ROOT / "results/outcomes.parquet",
        allow_preliminary=False)
    bundle_performance.run(bp)
    bdr.scrub_blocks(dest, [vid], reason)
    bdr.check_blocks(dest, [vid], True)
    if any(m["is_lockbox"] for m in json.loads(bp.clip_meta.read_text())):
        raise SystemExit(f"{vid}: lockbox")
    return dest


def index_entry(pub: Path, c: dict, link: str | None, platform: str | None) -> dict:
    a = json.loads((pub / c["video_id"] / "analysis.json").read_text())
    e = {"video_id": c["video_id"], "analysis_id": a.get("analysis_id"), "path": f"{c['video_id']}/analysis.json",
         "duration_ms": a.get("duration_ms"), "status": a.get("status"), "synthetic": False,
         "n_channels": len(a.get("channels") or []), "n_moments": len(a.get("moments") or []),
         "has_words": bool((a.get("events") or {}).get("words")), "n_warnings": len((a.get("quality") or {}).get("warnings") or []),
         "performance_path": f"{c['video_id']}/performance.json",
         "performance": {"model_status": "not_trained", "validated": False, "clip_in_training": "no"},
         "platform": platform, "video_link": link, "is_lockbox": False, "demo_role": c["role"]}
    if c["demo_moment"]:
        e["demo_moment"] = c["demo_moment"]
    if c["role"] in ("fell_short", "beat_expectations"):
        e["internal_example"] = c["role"]
    return e


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--showcase", type=Path, default=ROOT / "results/demo/showcase.json")
    ap.add_argument("--pair", type=Path, default=ROOT / "results/demo/example_pair.json")
    ap.add_argument("--demo-selection", type=Path, default=ROOT / "results/demo/selection.json")
    ap.add_argument("--public", type=Path, default=ROOT / "frontend/public")
    ap.add_argument("--work", type=Path, default=Path("/tmp/stage_demo_local"))
    args = ap.parse_args(argv)

    clips = plan(json.loads(args.showcase.read_text()), json.loads(args.pair.read_text()))
    roots = [Path(b["out_root"]) for b in json.loads(args.demo_selection.read_text())["batches_used"]]
    pub, clip_dir = args.public / "demo-stage1", args.public / "clips"
    old_idx = json.loads((pub / "index.json").read_text())
    reason = json.loads((pub / old_idx["bundles"][0]["performance_path"]).read_text())["reason"]
    if args.work.exists():
        shutil.rmtree(args.work)
    clip_dir.mkdir(parents=True, exist_ok=True)

    for c in clips:
        vid = c["video_id"]
        if not (pub / vid / "analysis.json").exists():
            dest = render(vid, c["batch"], roots, args.work, reason)
            shutil.copytree(dest / vid, pub / vid)
            if not (pub / "_static").exists():
                shutil.copytree(dest / "_static", pub / "_static")
        (pub / vid / "observed.json").unlink(missing_ok=True)
        if not (clip_dir / f"{vid}.mp4").exists():
            shutil.copyfile(c["source_file"], clip_dir / f"{vid}.mp4")

    ids = [c["video_id"] for c in clips]
    subprocess.run([sys.executable, str(ROOT / "tools/build_library_profile.py"), "--clips", *ids], check=True,
                   cwd=ROOT, stdout=subprocess.DEVNULL)
    for vid in ids:
        shutil.copyfile(ROOT / f"results/library/{vid}.library.json", pub / vid / "library.json")

    import pandas as pd
    posts = pd.read_parquet(ROOT / "results/outcomes.parquet", columns=["id", "video_link", "platform"])
    members = pd.read_csv(ROOT / "results/run_full/members.csv", dtype=str)
    entries = []
    for c in clips:
        link = platform = None
        if c["observed"]:
            link, platform = c["observed"]["video_link"], c["observed"]["platform"]
            obs = dict(c["observed"]) | {
                "schema": "nvi.observed.v0", "internal_only": True, "label": "observed on platform",
                "caption": CAPTION_OBS, "caption_null": CAPTION_NULL,
                "vs_expectation": {"role": c["role"], "plain": PLAIN[c["role"]],
                                   "residual_pct_in_stratum": round(c["residual_pct_in_stratum"], 3)}}
            (pub / c["video_id"] / "observed.json").write_text(json.dumps(obs, indent=1) + "\n")
        else:
            vp = members[(members["video_id"] == c["video_id"])]
            rep = vp[vp["representative"].astype(str) == "True"] if "representative" in vp else vp
            row = posts[posts["id"].isin((rep if len(rep) else vp)["vp_id"])]
            if len(row):
                link, platform = row.iloc[0]["video_link"], row.iloc[0]["platform"]
        entries.append(index_entry(pub, c, link, platform))

    pair = {c["role"]: c["video_id"] for c in clips if c["role"] in PLAIN}
    (pub / "examples.json").write_text(json.dumps({
        "schema": "nvi.examples.v0", "internal_only": True,
        "title": "A clip that fell short vs one that beat expectations",
        "pairs": [{"fell_short": pair["fell_short"], "beat_expectations": pair["beat_expectations"],
                   "same": "same account group and platform"}],
        "rule": json.loads(args.pair.read_text())["rule"], "caveat": CAVEAT, "caption_null": CAPTION_NULL},
        indent=1) + "\n")
    idx = {k: v for k, v in old_idx.items() if k not in ("bundles", "count", "performance_status_counts")}
    idx.update({"count": len(entries), "performance_status_counts": {"not_trained": len(entries)}, "bundles": entries,
                "local_note": ("Local internal demo set (showcase + example pair); the data-demo-stage1-v1 release "
                               "tarball is unchanged.")})
    (pub / "index.json").write_text(json.dumps(idx, indent=1) + "\n")

    keep = set(ids) | {"_static"}
    for d in pub.iterdir():
        if d.is_dir() and d.name not in keep:
            shutil.rmtree(d)
    for f in clip_dir.glob("*.mp4"):
        if f.stem not in ids:
            f.unlink()
    print(json.dumps([{k: e.get(k) for k in ("demo_role", "video_id", "duration_ms", "demo_moment", "platform")}
                      for e in entries], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
