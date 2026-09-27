#!/usr/bin/env python3
"""Stage the INTERNAL tailnet demo set under frontend/public/ (git-ignored): 9 clips, 3 per performance tier.

    .venv/bin/python tools/stage_demo_local.py      # uses results/demo/library_demo.json (tools/library_tiers.py)

Writes ``frontend/public/demo-stage1/`` (index.json ordered great, typical, bad; per clip analysis.json,
performance.json = not_trained, library.json, observed.json with the tier; examples.json = default compare pair)
plus ``frontend/public/library_patterns.json`` and ``frontend/public/clips/<id>.mp4`` (full-resolution source).
Bundle dirs and clips not in the new index are removed from public/ so they cannot be picked on stage. The release
``data-demo-stage1-v1`` tarball is not touched. Training clips only, never a lockbox clip; never published.
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

ORDER = ("great", "typical", "bad")
TIER_LABEL = {"great": "Did great", "typical": "Typical", "bad": "Did badly"}
PLATFORM = {"youtube": "YouTube", "tiktok": "TikTok", "instagram": "Instagram"}
CAPTION_OBS = ("Observed on platform (the post's own numbers, last snapshot). Internal view only; never shown for "
               "sealed test clips or in a release.")
CAPTION_NULL = ("The predicted brain response is not a views forecast: in the pre-registered test it did not improve "
                "predictions beyond basic information (no-GO).")
CAVEAT_OWNER = ("Hand-picked examples from the library, chosen to show how the second-by-second reading works. "
                "They are not typical of their group; see Library for what holds across all clips.")
CAVEAT = ("Two clips from the library, picked at random within their performance group by a fixed rule. What they "
          "show is a property of these two clips; see Library for what holds across all clips.")


def plan(demo: dict) -> list[dict]:
    """Ordered demo clips (great, typical, bad; hash order within each); one entry per video."""
    out, seen = [], set()
    for tier in ORDER:
        for p in demo["picks"].get(tier, []):
            if p["video_id"] in seen:
                raise SystemExit(f"{p['video_id']} is in two tiers")
            seen.add(p["video_id"])
            m = p.get("demo_moment")
            out.append({"role": tier, "video_id": p["video_id"], "batch": p["batch"],
                        "source_path": p["source_path"], "source_file": p["source_file"], "demo_moment": m,
                        "observed": p["observed"], "deal_label": p["deal_label"], "platform": p["platform"],
                        "views_pct": p["views_pct"], "n_ref": p["n_ref"]})
    if not out:
        raise SystemExit("no demo picks")
    return out


VIEWS_RULE_PLAIN = {"great": "300,000+ views", "typical": "5,000-15,000 views",
                    "bad": "under 700 views and below this account's usual"}


def tier_plain(c: dict, basis: str = "relative") -> str:
    x = c["observed"]["views_vs_account_usual_x"]
    if basis == "views":
        v = c["observed"]["views"]
        return (f"{v:,.0f} views ({VIEWS_RULE_PLAIN[c['role']]}); {x:.2g}× this account's recent usual."
                if x < 10 else f"{v:,.0f} views ({VIEWS_RULE_PLAIN[c['role']]}); {x:.0f}× this account's recent usual.")
    side = "More" if x > 1 else "Fewer"
    poss = "'" if c["deal_label"].endswith("s") else "'s"
    xs = f"{x:.0f}" if x >= 10 else f"{x:.2g}"
    return (f"{side} views than this account's recent usual ({xs}×). That jump over its usual is bigger than "
            f"{c['views_pct']:.0f}% of {c['deal_label']}{poss} {PLATFORM.get(c['platform'], c['platform'])} posts "
            f"({c['n_ref']:,} posts).")


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


def index_entry(pub: Path, c: dict) -> dict:
    a = json.loads((pub / c["video_id"] / "analysis.json").read_text())
    e = {"video_id": c["video_id"], "analysis_id": a.get("analysis_id"), "path": f"{c['video_id']}/analysis.json",
         "duration_ms": a.get("duration_ms"), "status": a.get("status"), "synthetic": False,
         "n_channels": len(a.get("channels") or []), "n_moments": len(a.get("moments") or []),
         "has_words": bool((a.get("events") or {}).get("words")),
         "n_warnings": len((a.get("quality") or {}).get("warnings") or []),
         "performance_path": f"{c['video_id']}/performance.json",
         "performance": {"model_status": "not_trained", "validated": False, "clip_in_training": "no"},
         "platform": c["platform"], "video_link": c["observed"]["video_link"], "is_lockbox": False,
         "demo_role": c["role"], "deal_label": c["deal_label"]}
    if c["demo_moment"]:
        e["demo_moment"] = c["demo_moment"]
    return e


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--demo", type=Path, default=ROOT / "results/demo/library_demo.json")
    ap.add_argument("--patterns", type=Path, default=ROOT / "results/library/patterns.json")
    ap.add_argument("--demo-selection", type=Path, default=ROOT / "results/demo/selection.json")
    ap.add_argument("--public", type=Path, default=ROOT / "frontend/public")
    ap.add_argument("--work", type=Path, default=Path("/tmp/stage_demo_local"))
    args = ap.parse_args(argv)

    demo = json.loads(args.demo.read_text())
    clips = plan(demo)
    selection = demo.get("selection", "rule")
    owner = selection in ("owner", "candidates")
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
        if not (clip_dir / f"{vid}.mp4").exists():
            shutil.copyfile(c["source_file"], clip_dir / f"{vid}.mp4")

    ids = [c["video_id"] for c in clips]
    subprocess.run([sys.executable, str(ROOT / "tools/build_library_profile.py"), "--clips", *ids], check=True,
                   cwd=ROOT, stdout=subprocess.DEVNULL)
    entries = []
    for c in clips:
        vid = c["video_id"]
        shutil.copyfile(ROOT / f"results/library/{vid}.library.json", pub / vid / "library.json")
        obs = dict(c["observed"])
        if not obs.get("shares"):
            obs["shares"] = None  # 0 = not reported (YouTube 100%, Instagram 99.9% zero in the export)
        obs = obs | {
            "schema": "nvi.observed.v0", "internal_only": True, "label": "observed on platform",
            "caption": CAPTION_OBS, "caption_null": CAPTION_NULL, "tier": c["role"],
            "tier_label": TIER_LABEL[c["role"]], "tier_plain": tier_plain(c, demo.get("tier_basis", "relative")),
            "tier_basis": demo.get("tier_basis", "relative"),
            "views_pct_in_deal_platform": round(c["views_pct"], 1), "n_ref_posts": c["n_ref"]}
        (pub / vid / "observed.json").write_text(json.dumps(obs, indent=1) + "\n")
        entries.append(index_entry(pub, c))

    first = {t: next((c["video_id"] for c in clips if c["role"] == t), None) for t in ORDER}
    (pub / "examples.json").write_text(json.dumps({
        "schema": "nvi.examples.v1", "internal_only": True,
        "default_pair": {"a": first["great"], "b": first["bad"]}, "caveat": CAVEAT_OWNER if owner else CAVEAT,
        "selection": selection, "caption_null": CAPTION_NULL},
        indent=1) + "\n")
    shutil.copyfile(args.patterns, args.public / "library_patterns.json")
    idx = {k: v for k, v in old_idx.items() if k not in ("bundles", "count", "performance_status_counts")}
    idx.update({"count": len(entries), "performance_status_counts": {"not_trained": len(entries)}, "bundles": entries,
                "selection": selection,
                "demo_source": str(args.demo.resolve().relative_to(ROOT)),
                "local_note": (({"owner": "Local internal demo set, hand-picked by the owner",
                                 "candidates": "Local internal browse set: every demo-eligible clip per tier"}[selection]
                                if owner else
                                "Local internal demo set (3 clips per performance tier, hash order)")
                               + " (tools/library_tiers.py); the data-demo-stage1-v1 release tarball is unchanged.")})
    (pub / "index.json").write_text(json.dumps(idx, indent=1) + "\n")

    keep = set(ids) | {"_static"}
    for d in pub.iterdir():
        if d.is_dir() and d.name not in keep:
            shutil.rmtree(d)
    for f in clip_dir.glob("*.mp4"):
        if f.stem not in ids:
            f.unlink()
    print(json.dumps([{k: e.get(k) for k in ("demo_role", "video_id", "deal_label", "platform", "duration_ms")}
                      for e in entries], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
