#!/usr/bin/env python3
"""Turn worker outputs into ROI features and brain visuals.

For each completed video in ``--out-root`` (worker-N/<video_id>.npz/.json):

    <report-dir>/features/<video_id>.roi.npz        ROI curves [G, T] + seg_start (neural_timeseries_feature)
    <report-dir>/features/summaries.jsonl           one row per (video, ROI group) (video_feature_summary)
    <report-dir>/analyses/<video_id>/analysis.json  frontend contract nvi.analysis.v0.2 + brain sprites (--analysis)
    <report-dir>/analyses/<video_id>/summary.png    surface views + ROI curves     (--png)
    <report-dir>/analyses/<video_id>/demo.mp4       source | brain | readout demo  (--video)
    <report-dir>/analyses/_static/                  region hover map + legend

Runs on CPU; use it on this server after pulling results, or on the pod.

    python tools/brain_report.py --out-root results/run1 --report-dir results/run1/report \
        --roi-map tribe_research/assets/roi_map_roi_groups_v0.npz --analysis --png --video --videos-root videos
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tribe_research.brain.features import FEATURE_VERSION, RoiMap, video_features  # noqa: E402
from tribe_research.brain.proxies import ProxySpec  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--report-dir", type=Path, required=True)
    ap.add_argument("--roi-map", type=Path, required=True)
    ap.add_argument("--videos-root", type=Path, default=None, help="source clips, for --video")
    ap.add_argument("--analysis", action="store_true", help="analysis.json + brain sprites per video")
    ap.add_argument("--png", action="store_true", help="also a summary PNG (implies --analysis)")
    ap.add_argument("--video", action="store_true", help="also a demo MP4 (implies --analysis)")
    ap.add_argument("--no-research-vertex", action="store_true", help="skip the per-vertex research sprite")
    ap.add_argument("--only", nargs="*", default=None, help="video_ids to include")
    ap.add_argument("--synthetic", action="store_true", help="label visuals as synthetic (dry-run data)")
    ap.add_argument("--jobs", type=int, default=8, help="parallel processes (this host also runs production)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="keep bundles whose analysis.json is newer than the prediction (incremental pulls)")
    args = ap.parse_args()

    roi = RoiMap.load(args.roi_map)
    spec = ProxySpec.load()
    feat_dir, ana_dir = args.report_dir / "features", args.report_dir / "analyses"
    feat_dir.mkdir(parents=True, exist_ok=True)
    bundles = args.analysis or args.png or args.video

    metas = [json.loads(p.read_text()) for p in sorted(args.out_root.glob("worker-*/*.json"))
             if not p.name.endswith(".error.json")]
    if args.only:
        metas = [m for m in metas if m["video_id"] in set(args.only)]
    if not metas:
        print("no completed videos found", file=sys.stderr)
        return 1

    if bundles:
        from tribe_research.brain import bundle

        region_map = bundle.write_static(ana_dir, roi, spec)
    else:
        region_map = None
    job = dict(out_root=args.out_root, feat_dir=feat_dir, ana_dir=ana_dir, roi_path=args.roi_map,
               videos_root=args.videos_root, bundles=bundles, synthetic=args.synthetic,
               research_vertex=not args.no_research_vertex, png=args.png, video=args.video,
               region_map=region_map, skip_existing=args.skip_existing)
    rows = []
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max(1, args.jobs), initializer=_init, initargs=(job,)) as ex:
        for vid_rows, line in ex.map(_one, metas, chunksize=1):
            rows.extend(vid_rows)
            print(line, flush=True)

    with (feat_dir / "summaries.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(metas)} videos -> {args.report_dir}")
    return 0


_JOB: dict = {}


def _init(job: dict) -> None:
    os.nice(5)  # yield to production services on this host
    _JOB.clear()
    _JOB.update(job, roi=RoiMap.load(job["roi_path"]), spec=ProxySpec.load())


def _one(m: dict) -> tuple[list[dict], str]:
    """Features (always, cheap) and, if asked, the bundle for one video."""
    j = _JOB
    roi, vid = j["roi"], m["video_id"]
    npz = j["out_root"] / f"worker-{m['worker_id']}" / f"{vid}.npz"
    t, curves, summaries = video_features(npz, roi, m["tr_s"])
    np.savez_compressed(
        j["feat_dir"] / f"{vid}.roi.npz", seg_start=t, curves=curves.astype(np.float16),
        group_names=np.array(roi.group_names), feature_version=FEATURE_VERSION,
        roi_provenance=json.dumps(roi.provenance),
    )
    rows = [{"video_id": vid, "source_name": m.get("source_name"), **s,
             "roi_groups_version": roi.provenance.get("groups_version"),
             "tribe_commit": m.get("tribe_commit")} for s in summaries]
    line = f"{vid}  {m.get('source_name')}  {len(t)} steps"
    if j["bundles"]:
        done = j["ana_dir"] / vid / "analysis.json"
        if j["skip_existing"] and done.exists() and done.stat().st_mtime >= npz.stat().st_mtime:
            return rows, line + "  (bundle kept)"
        from tribe_research.brain import bundle

        src = j["videos_root"] / m["path"] if j["videos_root"] else None
        if src is not None and not src.exists():
            src = None
        a = bundle.write_bundle(j["ana_dir"], m, npz, roi, j["spec"], j["region_map"], source_video=src,
                                synthetic=j["synthetic"], research_vertex=j["research_vertex"],
                                png=j["png"], video=j["video"])
        line += f"  {len(a['moments'])} moments, {len(a['quality']['warnings'])} warnings"
    return rows, line


if __name__ == "__main__":
    sys.exit(main())
