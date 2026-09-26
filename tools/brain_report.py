#!/usr/bin/env python3
"""Turn worker outputs into ROI features and brain visuals.

For each completed video in ``--out-root`` (worker-N/<video_id>.npz/.json):

    <report-dir>/features/<video_id>.roi.npz        ROI curves [G, T] + seg_start (neural_timeseries_feature)
    <report-dir>/features/summaries.jsonl           one row per (video, ROI group) (video_feature_summary)
    <report-dir>/visuals/<video_id>.png             surface views + ROI curves     (--png)
    <report-dir>/visuals/<video_id>.mp4             source | brain | timeline demo  (--video)

Runs on CPU; use it on this server after pulling results, or on the pod.

    python tools/brain_report.py --out-root results/run1 --report-dir results/run1/report \
        --roi-map tribe_research/assets/roi_map_roi_groups_v0.npz --png --video --videos-root videos
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tribe_research.brain.features import FEATURE_VERSION, RoiMap, video_features  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--report-dir", type=Path, required=True)
    ap.add_argument("--roi-map", type=Path, required=True)
    ap.add_argument("--videos-root", type=Path, default=None, help="source clips, for --video")
    ap.add_argument("--png", action="store_true")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--only", nargs="*", default=None, help="video_ids to include")
    ap.add_argument("--synthetic", action="store_true", help="label visuals as synthetic (dry-run data)")
    args = ap.parse_args()

    roi = RoiMap.load(args.roi_map)
    labels = [g.replace("_", " ") for g in roi.group_names]
    feat_dir, vis_dir = args.report_dir / "features", args.report_dir / "visuals"
    feat_dir.mkdir(parents=True, exist_ok=True)
    if args.png or args.video:
        vis_dir.mkdir(parents=True, exist_ok=True)

    metas = [json.loads(p.read_text()) for p in sorted(args.out_root.glob("worker-*/*.json"))
             if not p.name.endswith(".error.json")]
    if args.only:
        metas = [m for m in metas if m["video_id"] in set(args.only)]
    if not metas:
        print("no completed videos found", file=sys.stderr)
        return 1

    rows = []
    for m in metas:
        vid = m["video_id"]
        npz = args.out_root / f"worker-{m['worker_id']}" / f"{vid}.npz"
        t, curves, summaries = video_features(npz, roi, m["tr_s"])
        np.savez_compressed(
            feat_dir / f"{vid}.roi.npz", seg_start=t, curves=curves.astype(np.float16),
            group_names=np.array(roi.group_names), feature_version=FEATURE_VERSION,
            roi_provenance=json.dumps(roi.provenance),
        )
        for s in summaries:
            rows.append({"video_id": vid, "source_name": m.get("source_name"), **s,
                         "roi_groups_version": roi.provenance.get("groups_version"),
                         "tribe_commit": m.get("tribe_commit")})

        if args.png or args.video:
            from tribe_research.brain import render

            preds = np.load(npz)["preds"]
            title = m.get("source_name") or vid
            if args.png:
                render.render_summary_png(vis_dir / f"{vid}.png", preds, t, curves, labels,
                                          title + (" [SYNTHETIC]" if args.synthetic else ""))
            if args.video:
                src = args.videos_root / m["path"] if args.videos_root else None
                if src is not None and not src.exists():
                    src = None
                render.render_video(vis_dir / f"{vid}.mp4", preds, t, curves, labels,
                                    source_video=src, title=title, duration_s=m["duration_s"],
                                    synthetic=args.synthetic)
        print(f"{vid}  {m.get('source_name')}  {len(t)} steps")

    with (feat_dir / "summaries.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(metas)} videos -> {args.report_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
