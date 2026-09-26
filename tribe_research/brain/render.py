"""Headless cortical-surface rendering for TRIBE outputs.

Draws the fsaverage5 inflated surface (bundled with nilearn, no download) as
2-D orthographic projections with matplotlib/Agg, so it runs on a server with
no GPU/OpenGL. Two outputs:

* ``render_summary_png``: 4 surface views of the clip-level response pattern +
  ROI group curves. For reports.
* ``render_video``: source video | brain views per TRIBE step | ROI timelines
  with a playhead, muxed with the source audio. The demo artifact.

Every image is captioned as a model prediction for an average subject; it is
a research proxy, not measured brain activity.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402

FS5 = 10242
CAPTION = "TRIBE v2 prediction · average subject · research proxy, not measured brain activity"

# (hemisphere, camera direction from brain toward viewer)
VIEWS = {
    "left lateral": ("left", np.array([-1.0, 0, 0])),
    "right lateral": ("right", np.array([1.0, 0, 0])),
    "left medial": ("left", np.array([1.0, 0, 0])),
    "right medial": ("right", np.array([-1.0, 0, 0])),
}


@dataclass
class ProjectedView:
    name: str
    offset: int  # vertex offset into the 20484 vector
    polys: np.ndarray  # [F, 3, 2] projected triangles, back-to-front
    faces: np.ndarray  # [F, 3] vertex indices (hemisphere-local), same order
    shade: np.ndarray  # [F] 0..1 sulcal shading


@lru_cache(maxsize=1)
def _views() -> tuple[ProjectedView, ...]:
    from nilearn import datasets, surface

    fs = datasets.fetch_surf_fsaverage("fsaverage5")
    out = []
    for name, (hemi, cam) in VIEWS.items():
        coords, faces = surface.load_surf_mesh(fs[f"infl_{hemi}"])
        sulc = surface.load_surf_data(fs[f"sulc_{hemi}"])
        coords = coords - coords.mean(axis=0)
        up = np.array([0, 0, 1.0])
        right = np.cross(-cam, up)
        xy = np.stack([coords @ right, coords @ up], axis=1)
        depth = (coords @ cam)[faces].mean(axis=1)
        order = np.argsort(depth)  # far first (painter's algorithm)
        faces = faces[order]
        s = sulc[faces].mean(axis=1)
        shade = 1 - (s - s.min()) / (np.ptp(s) or 1)
        out.append(ProjectedView(name, 0 if hemi == "left" else FS5, xy[faces], faces, shade))
    return tuple(out)


def vertex_z(preds: np.ndarray) -> np.ndarray:
    """Per-vertex z-score across time: highlights change within the clip."""
    p = preds.astype(np.float32)
    sd = p.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1
    return (p - p.mean(axis=0, keepdims=True)) / sd


def _face_colors(view: ProjectedView, values: np.ndarray, vmax: float, threshold: float, cmap) -> np.ndarray:
    v = values[view.offset: view.offset + FS5][view.faces].mean(axis=1)
    gray = 0.35 + 0.45 * view.shade
    rgba = np.stack([gray, gray, gray, np.ones_like(gray)], axis=1)
    over = cmap(np.clip((v / vmax + 1) / 2, 0, 1))
    alpha = np.clip((np.abs(v) - threshold) / max(vmax - threshold, 1e-6), 0, 1)[:, None]
    rgba[:, :3] = rgba[:, :3] * (1 - alpha) + over[:, :3] * alpha
    return rgba


def render_brain_frames(
    values: np.ndarray, *, vmax: float = 2.5, threshold: float = 1.0, px: tuple[int, int] = (640, 480)
) -> np.ndarray:
    """values [T, 20484] -> RGB uint8 [T, H, W, 3], one 2x2 panel image per step."""
    cmap = plt.get_cmap("RdBu_r")
    views = _views()
    dpi = 100
    fig, axes = plt.subplots(2, 2, figsize=(px[0] / dpi, px[1] / dpi), dpi=dpi)
    fig.patch.set_facecolor("black")
    colls = []
    for ax, view in zip(axes.flat, views):
        coll = PolyCollection(view.polys, edgecolors="none", antialiaseds=False)
        ax.add_collection(coll)
        ax.set_xlim(view.polys[..., 0].min(), view.polys[..., 0].max())
        ax.set_ylim(view.polys[..., 1].min(), view.polys[..., 1].max())
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(view.name, color="#bbbbbb", fontsize=8, pad=2)
        colls.append(coll)
    fig.subplots_adjust(0, 0, 1, 0.95, 0.02, 0.08)
    frames = []
    for row in values:
        for coll, view in zip(colls, views):
            coll.set_facecolor(_face_colors(view, row, vmax, threshold, cmap))
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    return np.stack(frames)


def render_summary_png(
    out: str | Path, preds: np.ndarray, seg_start: np.ndarray, curves: np.ndarray,
    group_labels: list[str], title: str,
) -> Path:
    """Clip-level pattern (mean |vertex z| over time, signed by peak) + ROI curves."""
    z = vertex_z(preds)
    peak = z[np.abs(z).argmax(axis=0), np.arange(z.shape[1])]
    brain = render_brain_frames(peak[None], px=(800, 600))[0]

    fig = plt.figure(figsize=(12, 9), dpi=100)
    fig.patch.set_facecolor("white")
    ax_b = fig.add_axes([0.05, 0.42, 0.9, 0.52])
    ax_b.imshow(brain)
    ax_b.axis("off")
    ax_b.set_title(f"{title}\npeak within-clip deviation per vertex", fontsize=11)
    ax_t = fig.add_axes([0.07, 0.07, 0.72, 0.3])
    _plot_curves(ax_t, seg_start, curves, group_labels)
    fig.text(0.5, 0.01, CAPTION, ha="center", fontsize=8, color="#666666")
    out = Path(out)
    fig.savefig(out)
    plt.close(fig)
    return out


def _plot_curves(ax, t, curves, labels, dark=False):
    zc = (curves - curves.mean(axis=1, keepdims=True)) / np.maximum(curves.std(axis=1, keepdims=True), 1e-8)
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, max(len(labels), 2)))
    for i, lab in enumerate(labels):
        ax.step(t, zc[i], where="post", lw=1.2, color=colors[i], label=lab)
    ax.set_xlabel("time in clip (s)", color="#cccccc" if dark else "black", fontsize=8)
    ax.set_ylabel("ROI z (within clip)", color="#cccccc" if dark else "black", fontsize=8)
    ax.legend(fontsize=6, ncol=1, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False,
              labelcolor="#dddddd" if dark else "black")
    ax.tick_params(labelsize=7, colors="#cccccc" if dark else "black")
    if dark:
        ax.set_facecolor("#111111")
        for s in ax.spines.values():
            s.set_color("#444444")


def _read_video(path: Path):
    import imageio.v2 as iio

    reader = iio.get_reader(str(path), "ffmpeg")
    fps = float(reader.get_meta_data().get("fps", 30.0))
    return reader, fps


def render_video(
    out: str | Path, preds: np.ndarray, seg_start: np.ndarray, curves: np.ndarray,
    group_labels: list[str], *, source_video: str | Path | None = None, title: str = "",
    fps: float = 10.0, duration_s: float | None = None, synthetic: bool = False,
) -> Path:
    """Side-by-side demo MP4: source frame | brain (per TRIBE step) | ROI timelines + playhead."""
    import imageio.v2 as iio
    import imageio_ffmpeg

    out = Path(out)
    brains = render_brain_frames(vertex_z(preds))
    tr = float(np.median(np.diff(seg_start))) if len(seg_start) > 1 else 1.0
    end = duration_s or float(seg_start[-1] + tr)

    reader = src_fps = None
    if source_video is not None:
        reader, src_fps = _read_video(Path(source_video))

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    fig.patch.set_facecolor("black")
    ax_v = fig.add_axes([0.01, 0.30, 0.25, 0.64])
    ax_b = fig.add_axes([0.27, 0.30, 0.72, 0.64])
    ax_t = fig.add_axes([0.05, 0.10, 0.78, 0.17])
    for ax in (ax_v, ax_b):
        ax.axis("off")
    im_b = ax_b.imshow(brains[0])
    placeholder = np.full((16, 9, 3), 40, np.uint8)
    im_v = ax_v.imshow(placeholder)
    if reader is None:
        ax_v.text(0.5, 0.5, "synthetic\n(no source video)" if synthetic else "no source video",
                  color="#888888", ha="center", va="center", transform=ax_v.transAxes)
    _plot_curves(ax_t, seg_start, curves, group_labels, dark=True)
    playhead = ax_t.axvline(0, color="white", lw=1)
    fig.text(0.01, 0.97, title + ("  [SYNTHETIC DATA]" if synthetic else ""), color="white", fontsize=11, va="top")
    fig.text(0.5, 0.005, CAPTION, ha="center", fontsize=8, color="#888888")

    silent = out.with_name(out.stem + ".silent.mp4")
    writer = iio.get_writer(str(silent), fps=fps, codec="libx264", quality=7, macro_block_size=8)
    # Decode the source once, sequentially; random access re-seeks per frame.
    src_iter = iter(reader) if reader is not None else None
    src_idx, src_frame = -1, None
    try:
        for k in range(int(end * fps)):
            t = k / fps
            step = max(int(np.searchsorted(seg_start, t, side="right")) - 1, 0)
            im_b.set_data(brains[step])
            if src_iter is not None:
                want = int(t * src_fps)
                while src_idx < want:
                    nxt = next(src_iter, None)
                    if nxt is None:
                        break
                    src_frame, src_idx = nxt, src_idx + 1
                if src_frame is not None and im_v.get_array().shape != src_frame.shape:
                    im_v.set_data(src_frame)
                    h, w = src_frame.shape[:2]
                    im_v.set_extent((-0.5, w - 0.5, h - 0.5, -0.5))
                elif src_frame is not None:
                    im_v.set_data(src_frame)
            playhead.set_xdata([t, t])
            fig.canvas.draw()
            writer.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3])
    finally:
        writer.close()
        plt.close(fig)
        if reader is not None:
            reader.close()

    if source_video is not None:
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run(
            [ff, "-y", "-v", "error", "-i", str(silent), "-i", str(source_video),
             "-map", "0:v", "-map", "1:a?", "-c:v", "copy", "-c:a", "aac", "-shortest", str(out)],
            check=True,
        )
        silent.unlink()
    else:
        silent.replace(out)
    return out
