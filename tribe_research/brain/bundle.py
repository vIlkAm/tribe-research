"""One analysis bundle per video: analysis.json plus the assets it references.

Layout (every ``src`` in ``analysis.assets`` is relative to the bundle folder)::

    <analyses>/_static/region_idmap.png, region_legend.png   per ROI-map version
    <analyses>/<video_id>/analysis.json
    <analyses>/<video_id>/brain_proxy.jpg                     proxy-mode sprite
    <analyses>/<video_id>/brain_vertex.jpg                    research-mode sprite
    <analyses>/<video_id>/summary.png, demo.mp4               optional
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tribe_research.brain import render
from tribe_research.brain.analysis import build_analysis
from tribe_research.brain.events import detect_shots, words_lane
from tribe_research.brain.features import RoiMap, roi_curves
from tribe_research.brain.proxies import ProxySpec, channel_masks, channel_raw, zscore_excluding_onset

PROXY_VMAX, PROXY_THRESHOLD = 2.5, 0.5
VERTEX_VMAX, VERTEX_THRESHOLD = 3.0, 1.0


def _save_jpg(path: Path, img: np.ndarray) -> None:
    import imageio.v2 as iio

    iio.imwrite(path, img, quality=85)


def _sprite(mode, src, frames, t, dur, vmax, threshold) -> dict:
    return {
        "mode": mode, "src": src, "tile_w": int(frames.shape[2]), "tile_h": int(frames.shape[1]),
        "cols": None, "rows": None,
        "frame_start_ms": [int(round(s * 1000)) for s in t],
        "frame_end_ms": [int(round((s + d) * 1000)) for s, d in zip(t, dur)],
        "views": list(render.VIEWS), "colormap": {
            "name": render.COLORMAP, "vmin": -vmax, "vmax": vmax, "threshold": threshold, "unit": "z_within_clip"},
    }


def write_static(analyses_dir: Path, roi: RoiMap, spec: ProxySpec) -> dict:
    """Region idmap + legend (same for every clip of one ROI-map version)."""
    d = analyses_dir / "_static"
    d.mkdir(parents=True, exist_ok=True)
    colors = [c.color for c in spec.channels]
    if len(set(c.lower() for c in colors)) != len(colors) or "#000000" in colors:
        raise ValueError("channel colours must be unique and not black (they double as idmap ids)")
    import imageio.v2 as iio

    iio.imwrite(d / "region_idmap.png", render.render_region_idmap(channel_masks(spec, roi), colors))
    render.render_legend(d / "region_legend.png",
                         [{"label": c.label, "color": c.color, "direction": c.direction} for c in spec.channels],
                         PROXY_VMAX)
    return {"idmap_src": "../_static/region_idmap.png", "legend_src": "../_static/region_legend.png",
            "ids": {c.color.lower(): c.key for c in spec.channels}}


def write_bundle(
    analyses_dir: Path, meta: dict, npz: Path, roi: RoiMap, spec: ProxySpec, region_map: dict, *,
    source_video: Path | None = None, synthetic: bool = False, research_vertex: bool = True,
    png: bool = False, video: bool = False,
) -> dict:
    vid = meta["video_id"]
    d = analyses_dir / vid
    d.mkdir(parents=True, exist_ok=True)
    data = np.load(npz)
    preds = data["preds"]
    t = data["seg_start"].astype(np.float64)
    dur = data["seg_duration"].astype(np.float64) if "seg_duration" in data else np.full(len(t), meta["tr_s"])
    shots = detect_shots(source_video) if source_video is not None else None

    masks = channel_masks(spec, roi)
    z, _ = zscore_excluding_onset(t, channel_raw(preds, masks))
    proxy_frames = render.render_brain_frames(render.paint_channels(z, masks), vmax=PROXY_VMAX,
                                              threshold=PROXY_THRESHOLD, px=render.TILE_PX)
    assets: dict = {"region_map": region_map}
    sheet, cols, rows = render.sprite_sheet(proxy_frames)
    _save_jpg(d / "brain_proxy.jpg", sheet)
    assets["brain_map"] = {**_sprite("proxy", "brain_proxy.jpg", proxy_frames, t, dur, PROXY_VMAX, PROXY_THRESHOLD),
                           "cols": cols, "rows": rows}
    if research_vertex:
        vf = render.render_brain_frames(render.vertex_z(preds), vmax=VERTEX_VMAX, threshold=VERTEX_THRESHOLD,
                                        px=render.TILE_PX)
        sheet, cols, rows = render.sprite_sheet(vf)
        _save_jpg(d / "brain_vertex.jpg", sheet)
        assets["research_vertex_map"] = {
            **_sprite("research_vertex", "brain_vertex.jpg", vf, t, dur, VERTEX_VMAX, VERTEX_THRESHOLD),
            "cols": cols, "rows": rows}
    title = meta.get("source_name") or vid
    if png:
        render.render_summary_png(d / "summary.png", preds, t, roi_curves(preds, roi),
                                  [g.replace("_", " ") for g in roi.group_names],
                                  title + (" [SYNTHETIC]" if synthetic else ""))
        assets["summary_png"] = "summary.png"
    if video:
        assets["demo_mp4"] = "demo.mp4"

    analysis = build_analysis(meta=meta, preds=preds, seg_start=t, seg_duration=dur, roi=roi, spec=spec,
                              words=words_lane(meta.get("words")), shots_ms=shots, assets=assets,
                              synthetic=synthetic)
    if video:
        bm = assets["brain_map"]
        render.render_video(d / "demo.mp4", analysis, proxy_frames, bm["frame_start_ms"], bm["frame_end_ms"],
                            source_video=source_video, title=title)
    (d / "analysis.json").write_text(json.dumps(analysis, indent=1) + "\n")
    return analysis
