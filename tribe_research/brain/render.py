"""Headless cortical-surface rendering for TRIBE outputs.

Draws the fsaverage5 inflated surface (bundled with nilearn, no download) as
2-D orthographic projections with matplotlib/Agg, so it runs on a server with
no GPU/OpenGL. Outputs:

* ``render_brain_frames`` + ``sprite_sheet``: one 2x2 view tile per TRIBE step,
  either proxy mode (each channel's region painted with its z) or per-vertex
  research mode. The frontend's scrubbable brain panel.
* ``render_region_idmap`` / ``render_legend``: static hover map and legend.
* ``render_summary_png``: clip-level vertex pattern + ROI group curves.
* ``render_video``: source | proxy brain | readout over a timeline with the
  default channels, candidate moments and shot ticks, muxed with source audio.

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
COLORMAP = "RdBu_r"
TILE_PX = (480, 360)
SPRITE_MAX_COLS = 10

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


def _panel_figure(px: tuple[int, int], titles: bool = True):
    """2x2 surface-view figure; geometry is identical with or without titles (idmap alignment)."""
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
        ax.set_title(view.name, color="#bbbbbb" if titles else "black", fontsize=8, pad=2)
        colls.append(coll)
    fig.subplots_adjust(0, 0, 1, 0.95, 0.02, 0.08)
    return fig, colls, views


def _draw(fig) -> np.ndarray:
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def render_brain_frames(
    values: np.ndarray, *, vmax: float = 2.5, threshold: float = 1.0, px: tuple[int, int] = (640, 480)
) -> np.ndarray:
    """values [T, 20484] -> RGB uint8 [T, H, W, 3], one 2x2 panel image per step."""
    cmap = plt.get_cmap(COLORMAP)
    fig, colls, views = _panel_figure(px)
    frames = []
    for row in values:
        for coll, view in zip(colls, views):
            coll.set_facecolor(_face_colors(view, row, vmax, threshold, cmap))
        frames.append(_draw(fig))
    plt.close(fig)
    return np.stack(frames)


def paint_channels(z: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Channel z [C, T] + masks [C, V] -> [T, V]; each region carries its channel's value, rest 0."""
    out = np.zeros((z.shape[1], masks.shape[1]), np.float32)
    for c in range(masks.shape[0]):
        out[:, masks[c]] = z[c][:, None]
    return out


def render_region_idmap(masks: np.ndarray, id_colors: list[str], px: tuple[int, int] = TILE_PX) -> np.ndarray:
    """Flat, unshaded channel-id image with the same tile geometry as the brain sprite.

    A face takes a channel's colour when at least two of its vertices belong to
    it; everything else (and the background) is pure black.
    """
    vid = np.full(masks.shape[1], -1)
    for c in range(masks.shape[0]):
        vid[masks[c]] = c
    rgb = np.array([[int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)] for h in id_colors] + [[0.0, 0.0, 0.0]])
    fig, colls, views = _panel_figure(px, titles=False)
    for coll, view in zip(colls, views):
        f = vid[view.offset: view.offset + FS5][view.faces]
        a, b, c = f[:, 0], f[:, 1], f[:, 2]
        face = np.where((a == b) | (a == c), a, np.where(b == c, b, -1))
        colors = np.ones((len(face), 4))
        colors[:, :3] = rgb[face]  # -1 indexes the trailing black row
        coll.set_facecolor(colors)
    img = _draw(fig)
    plt.close(fig)
    return img


def sprite_sheet(frames: np.ndarray, max_cols: int = SPRITE_MAX_COLS) -> tuple[np.ndarray, int, int]:
    """[N, H, W, 3] -> one row-major sheet image, cols, rows (unused cells black)."""
    n, h, w, _ = frames.shape
    cols = min(n, max_cols)
    rows = -(-n // cols)
    sheet = np.zeros((rows * h, cols * w, 3), np.uint8)
    for i, f in enumerate(frames):
        r, c = divmod(i, cols)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = f
    return sheet, cols, rows


def render_legend(out: str | Path, channels: list[dict], vmax: float) -> Path:
    """Channel colours + the signed colormap the brain sprite uses."""
    n = len(channels)
    fig = plt.figure(figsize=(6.4, 1.3 + 0.3 * n), dpi=100)
    fig.patch.set_facecolor("black")
    ax = fig.add_axes([0.03, 0.9 / (1.3 + 0.3 * n), 0.94, 1 - 1.0 / (1.3 + 0.3 * n)])
    ax.set_xlim(0, 1)
    ax.set_ylim(n, 0)
    ax.axis("off")
    for i, ch in enumerate(channels):
        ax.add_patch(plt.Rectangle((0.0, i + 0.2), 0.05, 0.6, color=ch["color"]))
        ax.text(0.08, i + 0.5, ch["label"], color="white", fontsize=9, va="center")
        ax.text(0.55, i + 0.5, ch["direction"].replace("_", " "), color="#aaaaaa", fontsize=8, va="center")
    cb = fig.add_axes([0.08, 0.45 / (1.3 + 0.3 * n), 0.84, 0.18 / (1.3 + 0.3 * n)])
    cb.imshow(np.linspace(-vmax, vmax, 256)[None], aspect="auto", cmap=COLORMAP, extent=(-vmax, vmax, 0, 1))
    cb.set_yticks([])
    cb.tick_params(labelsize=7, colors="#cccccc")
    cb.set_xlabel("predicted response, z within this clip (lower | higher)", color="#cccccc", fontsize=7)
    for sp in cb.spines.values():
        sp.set_color("#444444")
    out = Path(out)
    fig.savefig(out, facecolor="black")
    plt.close(fig)
    return out


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


MOMENT_COLORS = {"attention_drop": "#E4572E", "broad_response": "#F2C14E"}


def _readout(analysis: dict, t: float, visible: list[dict]) -> tuple[list[tuple[str, str]], str]:
    """Lines (text, colour) for the right-hand readout at time t, plus the active moment title."""
    k = int(t * analysis["timing"]["display_hz"])
    lines = []
    w = max((len(ch["label"]) for ch in visible), default=0)
    for ch in visible:
        v = ch["values"][k] if k < len(ch["values"]) else None
        lines.append((f"{ch['label']:<{w}} {'  n/a' if v is None else f'{v:+.1f} z'}", ch["color"]))
    ms = t * 1000
    active = [m for m in analysis["moments"] if m["start_ms"] <= ms < m["end_ms"]]
    title = max(active, key=lambda m: m["severity"])["title"] if active else ""
    return lines, title


def render_video(
    out: str | Path, analysis: dict, frames: np.ndarray, frame_start_ms: list[int], frame_end_ms: list[int],
    *, source_video: str | Path | None = None, title: str = "", fps: float = 10.0,
) -> Path:
    """Demo MP4: source | proxy brain | readout, over a timeline of default channels + moments."""
    import imageio.v2 as iio
    import imageio_ffmpeg

    out = Path(out)
    synthetic = analysis["synthetic"]
    end = analysis["duration_ms"] / 1000
    hz = analysis["timing"]["display_hz"]
    visible = [c for c in analysis["channels"] if c["default_visible"]]
    starts, ends = np.array(frame_start_ms) / 1000, np.array(frame_end_ms) / 1000

    reader = src_fps = None
    if source_video is not None:
        reader, src_fps = _read_video(Path(source_video))

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    fig.patch.set_facecolor("black")
    ax_v = fig.add_axes([0.01, 0.33, 0.22, 0.60])
    ax_b = fig.add_axes([0.24, 0.33, 0.50, 0.60])
    ax_t = fig.add_axes([0.05, 0.07, 0.92, 0.21])
    for ax in (ax_v, ax_b):
        ax.axis("off")
    im_b = ax_b.imshow(frames[0])
    no_pred = ax_b.text(0.5, 0.5, "no prediction", color="#bbbbbb", fontsize=14, ha="center", va="center",
                        transform=ax_b.transAxes, visible=False)
    im_v = ax_v.imshow(np.full((16, 9, 3), 40, np.uint8))
    if reader is None:
        ax_v.text(0.5, 0.5, "synthetic\n(no source video)" if synthetic else "no source video",
                  color="#888888", ha="center", va="center", transform=ax_v.transAxes)

    # timeline: default channels on the display grid (gaps stay empty), moments, shots, onset
    grid = np.arange(len(visible[0]["values"])) / hz if visible else np.array([])
    for ch in visible:
        y = np.array([np.nan if v is None else v for v in ch["values"]], float)
        ax_t.step(grid, y, where="post", lw=1.4, color=ch["color"], label=ch["label"])
    for m in analysis["moments"]:
        col = MOMENT_COLORS.get(m["kind"])
        if col:
            ax_t.axvspan(m["start_ms"] / 1000, m["end_ms"] / 1000, color=col, alpha=0.18, lw=0)
    for g0, g1 in analysis["timing"]["gaps_ms"]:
        ax_t.axvspan(g0 / 1000, g1 / 1000, color="#555555", alpha=0.35, hatch="//", lw=0)
    o0, o1 = analysis["timing"]["onset_window_ms"]
    ax_t.axvspan(o0 / 1000, o1 / 1000, color="#ffffff", alpha=0.05, lw=0)
    for s in analysis["events"]["shots_ms"] or []:
        ax_t.axvline(s / 1000, ymin=0.92, ymax=1.0, color="#dddddd", lw=0.8)
    ax_t.axhline(0, color="#444444", lw=0.6)
    ax_t.set_xlim(0, end)
    ax_t.set_facecolor("#111111")
    ax_t.set_xlabel("time in clip (s) · ticks = shot changes · shaded = candidate moments · hatched = no prediction",
                    color="#aaaaaa", fontsize=7)
    ax_t.set_ylabel("z (this clip)", color="#aaaaaa", fontsize=7)
    ax_t.tick_params(labelsize=7, colors="#aaaaaa")
    for sp in ax_t.spines.values():
        sp.set_color("#444444")
    ax_t.legend(fontsize=7, ncol=len(visible) or 1, loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False,
                labelcolor="#dddddd")
    playhead = ax_t.axvline(0, color="white", lw=1)

    fig.text(0.76, 0.92, "Predicted response (this clip)", color="white", fontsize=10, va="top")
    read_txt = [fig.text(0.76, 0.86 - 0.045 * i, "", color=ch["color"], fontsize=8, va="top", family="monospace")
                for i, ch in enumerate(visible)]
    moment_txt = fig.text(0.76, 0.84 - 0.045 * len(visible), "", color="#F2C14E", fontsize=9, va="top", wrap=True)
    fig.text(0.76, 0.36, "Relative to this clip only.\nNot a score, not a behavior prediction.",
             color="#777777", fontsize=7, va="bottom")
    fig.text(0.01, 0.97, title + ("  [SYNTHETIC DATA]" if synthetic else ""),
             color="#ff6666" if synthetic else "white", fontsize=11, va="top")
    fig.text(0.5, 0.005, CAPTION, ha="center", fontsize=8, color="#888888")

    silent = out.with_name(out.stem + ".silent.mp4")
    writer = iio.get_writer(str(silent), fps=fps, codec="libx264", quality=7, macro_block_size=8)
    # Decode the source once, sequentially; random access re-seeks per frame.
    src_iter = iter(reader) if reader is not None else None
    src_idx, src_frame = -1, None
    try:
        for k in range(int(end * fps)):
            t = k / fps
            i = int(np.searchsorted(starts, t, side="right")) - 1
            in_pred = i >= 0 and t < ends[i]
            if in_pred:
                im_b.set_data(frames[i])
            im_b.set_alpha(1.0 if in_pred else 0.25)
            no_pred.set_visible(not in_pred)
            if src_iter is not None:
                want = int(t * src_fps)
                while src_idx < want:
                    nxt = next(src_iter, None)
                    if nxt is None:
                        break
                    src_frame, src_idx = nxt, src_idx + 1
                if src_frame is not None and im_v.get_array().shape != src_frame.shape:
                    h, w = src_frame.shape[:2]
                    im_v.set_extent((-0.5, w - 0.5, h - 0.5, -0.5))
                if src_frame is not None:
                    im_v.set_data(src_frame)
            lines, mom = _readout(analysis, t, visible)
            for txt, (line, _) in zip(read_txt, lines):
                txt.set_text(line)
            moment_txt.set_text(mom)
            playhead.set_xdata([t, t])
            writer.append_data(_draw(fig))
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
