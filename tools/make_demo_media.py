#!/usr/bin/env python3
"""Render the ViralBrain demo-video assets (charts, facts card, synced clip videos).

Every number shown is read from files under results/ at render time; nothing is typed in.
Outputs go ONLY to /tmp/viralbrain-media (never into this repo):

  01_views_vs_usual    exploratory: response curves for clips above / below their account's usual views
  02_views_split       exploratory: variance shares of views (swing around the usual vs account size)
  03_stage1_result     pre-registered stage-1 test (no-GO)
  04_tribe_paper       facts card about TRIBE v2 (text only, no paper figure)
  05_clip_<id>, 06_clip_<id>   clip + per-second predicted response, audio muxed

Each asset gets a 1920x1080 PNG (final frame) and a 1920x1080 30 fps H.264 yuv420p MP4.
Only the two named demo clips are opened (H.264/AV1 copies in /tmp/viralbrain-judges).

Regenerate:
  cd /home/tyler/projects/tribe-research && \
  PYTHONDONTWRITEBYTECODE=1 nice -n 10 .venv/bin/python tools/make_demo_media.py [--only 01,05]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = Path("/tmp/viralbrain-media")
CLIPS_DIR = Path("/tmp/viralbrain-judges/viralbrain-demo/clips")
# The only two clips this script may open (demo picks; never a lockbox clip).
CLIPS = (("05", "ef3335a5c1b282e3"), ("06", "b17bdcaf52dc2195"))

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
matplotlib.rcParams["animation.ffmpeg_path"] = FFMPEG
W, H, FPS, DPI = 1920, 1080, 30, 100

BG = "#090c11"
PANEL = "#11161e"
TEXT = "#eceef2"
MUTED = "#8b97a8"
GOLD = "#dfb67c"
ORANGE = "#ffac5c"  # above / better
BLUE = "#8eb8df"  # below / worse
GRID = "#232c38"
BAND_ALPHA = 0.25

FOOTER = ("TRIBE v2 prediction · average subject · ViralBrain research demo "
          "(non-commercial, TRIBE v2 CC BY-NC 4.0)")
PAPER_FOOTER = ("Source: d'Ascoli et al., 'A foundation model of vision, audition, and language for "
                "in-silico neuroscience', FAIR at Meta, 2026 · github.com/facebookresearch/tribev2")
FOOTER_VIEWS = "Platform views · ViralBrain research demo (non-commercial)"
FOOTER_STAGE1 = ("Platform engagement · pre-registered stage-1 test · ViralBrain research demo "
                 "(non-commercial, TRIBE v2 CC BY-NC 4.0)")
MINUS = "−"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": TEXT,
    "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.unicode_minus": True,
    "figure.facecolor": BG,
})

FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
NUMBERS: dict = {}  # every displayed number -> source, dumped to OUT/numbers_used.json


# ----------------------------------------------------------------------------- helpers

def pt(px: float) -> float:
    """Font size in points for a target pixel size at DPI."""
    return px * 72.0 / DPI


def sgn(v: float, nd: int = 2) -> str:
    return f"{v:+.{nd}f}".replace("-", MINUS)


def num(v: float, nd: int) -> str:
    return f"{v:.{nd}f}".replace("-", MINUS)


def load(rel: str) -> dict:
    return json.loads((REPO / rel).read_text())


def record(key: str, shown: str, source: str, raw) -> None:
    NUMBERS[key] = {"shown": shown, "source": source, "raw": raw}


def arr(xs) -> np.ndarray:
    return np.array([np.nan if v is None else float(v) for v in xs], dtype=float)


def ease(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return 1.0 - (1.0 - x) ** 3


def ramp(t: float, t0: float, t1: float) -> float:
    return min(max((t - t0) / (t1 - t0), 0.0), 1.0)


def new_fig():
    return plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=BG)


def overlay(fig):
    """Full-frame transparent axes in pixel coordinates (x right, y down)."""
    ov = fig.add_axes([0, 0, 1, 1])
    ov.set_xlim(0, W)
    ov.set_ylim(H, 0)
    ov.axis("off")
    ov.patch.set_visible(False)
    return ov


def px_rect(x0, y0, x1, y1):
    return [x0 / W, 1 - y1 / H, (x1 - x0) / W, (y1 - y0) / H]


def rbox(ov, x0, y0, x1, y1, fc=PANEL, ec="none", lw=0.0, r=18, alpha=1.0, z=0):
    p = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=z)
    ov.add_patch(p)
    return p


def txt(ov, x, y, s, px, color=TEXT, weight="normal", ha="left", va="top", alpha=1.0, z=5, **kw):
    return ov.text(x, y, s, fontsize=pt(px), color=color, fontweight=weight, ha=ha, va=va,
                   alpha=alpha, zorder=z, **kw)


def footer(ov, s=FOOTER, x=60):
    return txt(ov, x, H - 18, s, 18, MUTED, va="bottom")


def style_axes(ax, tick_px=24):
    ax.set_facecolor("none")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1.5)
    ax.tick_params(labelsize=pt(tick_px), length=0, pad=10)
    ax.grid(axis="y", color=GRID, lw=1.0)
    ax.set_axisbelow(True)


def signed_tick(v, _pos=None):
    if abs(v) < 1e-9:
        return "0"
    return sgn(v, 1)


def fig_rgb(fig) -> np.ndarray:
    fig.canvas.draw()
    a = np.asarray(fig.canvas.buffer_rgba())
    assert a.shape[:2] == (H, W), a.shape
    return a[:, :, :3].copy()


class Encoder:
    """Pipe RGB frames into the imageio-ffmpeg binary (libx264, yuv420p, faststart)."""

    def __init__(self, path: Path, audio_src: Path | None = None, tune: str | None = None):
        self.path = path
        self.err = tempfile.TemporaryFile()
        cmd = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-"]
        if audio_src is not None:
            cmd += ["-i", str(audio_src), "-map", "0:v:0", "-map", "1:a:0"]
        cmd += ["-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
                "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-profile:v", "high",
                "-pix_fmt", "yuv420p", "-r", str(FPS),
                "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
                "-color_range", "tv"]
        if tune:
            cmd += ["-tune", tune]
        if audio_src is not None:
            cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest"]
        cmd += ["-movflags", "+faststart", str(path)]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.err)
        self.n = 0

    def write(self, frame: np.ndarray):
        assert frame.shape == (H, W, 3) and frame.dtype == np.uint8
        self.p.stdin.write(frame.tobytes())
        self.n += 1

    def close(self):
        self.p.stdin.close()
        rc = self.p.wait()
        if rc != 0:
            self.err.seek(0)
            raise RuntimeError(f"ffmpeg failed for {self.path}: {self.err.read().decode(errors='replace')}")


def render_animation(name: str, fig, update, duration_s: float, static_after: float):
    """Call update(t) per frame, encode, save the final frame as PNG."""
    n = int(round(duration_s * FPS))
    enc = Encoder(OUT / f"{name}.mp4", tune="animation")
    cached = None
    last = None
    for k in range(n):
        t = k / FPS
        if cached is not None:
            frame = cached
        else:
            update(t)
            frame = fig_rgb(fig)
            if t >= static_after:
                cached = frame
        enc.write(frame)
        last = frame
    enc.close()
    plt.close(fig)
    Image.fromarray(last).save(OUT / f"{name}.png")
    return n


def title(ov, s, y=40):
    return txt(ov, 60, y, s, 44, TEXT, weight="bold", linespacing=1.15)


def tag(ov, s, x1=1860, y=46):
    t = txt(ov, x1 - 22, y + 12, s, 22, GOLD, weight="bold", ha="right")
    return t


def tag_box(fig, ov, t):
    """Draw a rounded outline around a tag text once its extent is known."""
    fig.canvas.draw()
    bb = t.get_window_extent()
    x0, x1 = bb.x0 - 18, bb.x1 + 18
    y0, y1 = H - bb.y1 - 10, H - bb.y0 + 10
    return rbox(ov, x0, y0, x1, y1, fc="none", ec=GOLD, lw=1.6, r=12, alpha=0.8, z=4)


# ----------------------------------------------------------------------------- 01

def chart_views_vs_usual():
    src = "results/library/learned.json:views_vs_usual"
    L = load("results/library/learned.json")["views_vs_usual"]
    secs = np.array(L["seconds"], dtype=float)
    xs = secs + 0.5  # second i covers [i, i+1); plot at its centre
    s = {k: {q: arr(L["index"][k][q]) for q in ("mean", "lo", "hi")} for k in ("top", "bottom")}
    for k in s:
        for q in s[k]:
            assert len(s[k][q]) == len(secs)

    # Summary numbers: take the 2-dp strings from the file's own result_plain (formatted there from
    # unrounded values), and verify each against the stored 3-dp summary. Fall back to 3 dp.
    summ = L["summary"]
    stored = [summ["whole_0_29"]["diff"], summ["whole_0_29"]["lo"], summ["whole_0_29"]["hi"],
              summ["opening_0_4"]["diff"], summ["opening_0_4"]["lo"], summ["opening_0_4"]["hi"]]
    found = re.findall(r"[+\-−]\d+\.\d+", L["result_plain"])
    if len(found) == 6 and all(abs(float(f.replace(MINUS, "-")) - v) <= 0.006 for f, v in zip(found, stored)):
        shown = [f.replace("-", MINUS) for f in found]
        how = "2 dp as written in views_vs_usual.result_plain (checked within 0.006 of summary)"
    else:
        shown = [sgn(v, 3) for v in stored]
        how = "3 dp from views_vs_usual.summary (result_plain parse failed)"
    names = ["whole_0_29.diff", "whole_0_29.lo", "whole_0_29.hi",
             "opening_0_4.diff", "opening_0_4.lo", "opening_0_4.hi"]
    for nm, sh, v in zip(names, shown, stored):
        record(f"01.summary.{nm}", sh, f"{src}.summary.{nm} ({how})", v)
    # summary key "opening_0_4" = the first 4 s (index slice [:4] = seconds [0, 4)); shade that span
    opening_s = int(re.fullmatch(r"opening_0_(\d+)", "opening_0_4").group(1))

    n_top, n_bot, n_c, n_d = L["n_top"], L["n_bottom"], L["n_contents"], L["n_deals"]
    for k, v in (("n_top", n_top), ("n_bottom", n_bot), ("n_contents", n_c), ("n_deals", n_d)):
        record(f"01.{k}", f"{v:,}", f"{src}.{k}", v)
    assert L.get("exploratory") is True

    fig = new_fig()
    ov = overlay(fig)
    title(ov, "Clips that beat their account's usual views\nhave a higher predicted brain response")
    tg = tag(ov, f"Exploratory · {n_c:,} clips · {n_d} clients")
    tag_box(fig, ov, tg)
    rbox(ov, 40, 178, 1880, 1012, fc=PANEL, r=22, z=0)

    # legend row
    for x, col, lab in ((100, ORANGE, f"Above their account's usual views ({n_top:,} clips)"),
                        (980, BLUE, f"Below their account's usual views ({n_bot:,} clips)")):
        ov.add_patch(Rectangle((x, 205), 56, 22, fc=col, alpha=BAND_ALPHA, lw=0, zorder=3))
        ov.plot([x, x + 56], [216, 216], color=col, lw=pt(6), solid_capstyle="round", zorder=4)
        txt(ov, x + 76, 216, lab, 26, TEXT, va="center")

    ax = fig.add_axes(px_rect(190, 262, 1840, 800))
    style_axes(ax)
    ymin = np.nanmin([np.nanmin(s["top"]["lo"]), np.nanmin(s["bottom"]["lo"])])
    ymax = np.nanmax([np.nanmax(s["top"]["hi"]), np.nanmax(s["bottom"]["hi"])])
    ax.set_xlim(0, len(secs))
    ax.set_ylim(math.floor(ymin / 0.05 - 0.4) * 0.05, math.ceil(ymax / 0.05 + 0.4) * 0.05)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(signed_tick))
    ax.set_xticks(np.arange(0, len(secs) + 1, 5))
    ax.set_xlabel("Seconds into the clip", fontsize=pt(26), labelpad=12)
    ax.set_ylabel("Predicted response vs library (sd)", fontsize=pt(26), labelpad=14)
    ax.axvspan(0, opening_s, color=GOLD, alpha=0.08, lw=0, zorder=0)
    ax.text(opening_s / 2, 0.97, "Opening", transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=pt(24), color=GOLD, fontweight="bold")
    ax.axhline(0, color=MUTED, lw=1.4, ls=(0, (6, 5)), alpha=0.8, zorder=1)

    dyn = []
    dots = {}
    for k, col in (("top", ORANGE), ("bottom", BLUE)):
        band = ax.fill_between(xs, s[k]["lo"], s[k]["hi"], color=col, alpha=BAND_ALPHA, lw=0, zorder=2)
        (ln,) = ax.plot(xs, s[k]["mean"], color=col, lw=pt(5), zorder=3, solid_capstyle="round")
        (dot,) = ax.plot([], [], "o", ms=pt(16), color=col, mec=PANEL, mew=pt(3), zorder=4)
        dyn += [band, ln]
        dots[k] = (dot, s[k]["mean"])

    ann_txt = (f"Gap between the lines · first 30 s: {shown[0]} sd (95% range {shown[1]} to {shown[2]})"
               f"  ·  opening: {shown[3]} sd ({shown[4]} to {shown[5]})")
    ann_box = rbox(ov, 90, 912, 1830, 982, fc="none", ec=GOLD, lw=1.6, r=14, z=3)
    ann = txt(ov, 960, 947, ann_txt, 27, TEXT, ha="center", va="center")
    footer(ov)

    t_draw0, t_draw1, t_ann0, t_ann1 = 0.4, 5.0, 5.1, 5.7
    x_end = float(len(secs))

    def update(t):
        p = ramp(t, t_draw0, t_draw1) * x_end
        clip = Rectangle((-1, -100), p + 1, 200, transform=ax.transData)
        for a in dyn:
            a.set_clip_path(clip)
        for dot, ys in dots.values():
            xp = min(max(p, xs[0]), xs[-1])
            yp = np.interp(xp, xs, ys) if 0 < p else np.nan
            vis = 0 < p < x_end - 0.05 and np.isfinite(yp)
            dot.set_data([xp], [yp]) if vis else dot.set_data([], [])
        a = ramp(t, t_ann0, t_ann1)
        ann.set_alpha(a)
        ann_box.set_alpha(0.8 * a)

    return render_animation("01_views_vs_usual", fig, update, duration_s=7.8, static_after=t_ann1 + 0.05)


# ----------------------------------------------------------------------------- 02

def chart_views_split():
    src = "results/library/theory.json:decomposition"
    T = load("results/library/theory.json")
    D = T["decomposition"]
    sv, sa, sc, n = D["share_video"], D["share_account"], D["share_covariance"], D["n_clips"]
    assert T.get("exploratory") is True
    assert abs(sv + sa + sc - 1.0) < 0.002, (sv, sa, sc)
    pv, pa, pc = (f"{100 * sv:.1f}%", f"{100 * sa:.1f}%", f"{100 * sc:.1f}%".replace("-", MINUS))
    # the 1-dp percentages must also sum to 100.0 on screen
    assert abs(round(100 * sv, 1) + round(100 * sa, 1) + round(100 * sc, 1) - 100.0) < 0.05
    record("02.share_video", pv, f"{src}.share_video", sv)
    record("02.share_account", pa, f"{src}.share_account", sa)
    record("02.share_covariance", pc, f"{src}.share_covariance", sc)
    record("02.n_clips", f"{n:,}", f"{src}.n_clips", n)
    record("02.n_accounts", f"{T['n_accounts']:,}", "results/library/theory.json:n_accounts", T["n_accounts"])

    fig = new_fig()
    ov = overlay(fig)
    title(ov, "What drives a clip's views within the same client and platform")
    txt(ov, 60, 118, "Share of the variation in views (log scale) within each client × platform",
        26, MUTED)
    tg = tag(ov, f"Exploratory · {n:,} clips · {T['n_accounts']:,} accounts", y=112)
    tag_box(fig, ov, tg)
    rbox(ov, 40, 186, 1880, 900, fc=PANEL, r=22, z=0)

    X0, FULL = 110, 1460  # 100% = FULL px
    rows = [
        ("Swing around the account's usual (luck, timing, algorithm, the video itself)", sv, pv, ORANGE, 250),
        ("Account size (the account's usual views)", sa, pa, BLUE, 520),
    ]
    bars, vals = [], []
    for lab, share, shown, col, y in rows:
        txt(ov, X0, y, lab, 30, TEXT)
        rbox(ov, X0, y + 58, X0 + FULL, y + 168, fc="#1a212c", r=12, z=1)  # 100% track
        bar = rbox(ov, X0, y + 58, X0 + 1, y + 168, fc=col, r=12, z=2)
        v = txt(ov, X0 + share * FULL + 26, y + 113, shown, 54, col, weight="bold", va="center", alpha=0)
        bars.append((bar, share))
        vals.append(v)
    txt(ov, X0 + FULL, 250 + 176, "100%", 20, MUTED, ha="right")
    cap = txt(ov, X0, 780,
              f"Overlap {pc} (shares add to 100% with the overlap). n = {n:,} clips.", 28, MUTED, alpha=0)
    swing = re.search(r"The swing includes[^.]*\.", T["interpreter_line"]).group(0)
    record("02.caption_sentence", swing, "results/library/theory.json:interpreter_line (verbatim sentence)", None)
    cap2 = txt(ov, X0, 830, swing, 24, MUTED, alpha=0)
    footer(ov, FOOTER_VIEWS)

    starts = (0.4, 1.1)

    def update(t):
        for (bar, share), v, t0 in zip(bars, vals, starts):
            g = ease(ramp(t, t0, t0 + 1.6))
            w = share * FULL * g
            bar.set_visible(w >= 4)  # a sub-radius rounded box draws as a bowtie; hide until it has width
            bar.set_boxstyle("round", pad=0, rounding_size=min(12.0, w / 2))
            bar.set_width(max(w, 1))
            v.set_alpha(ramp(t, t0 + 1.5, t0 + 1.9))
        a = ramp(t, 3.2, 3.8)
        cap.set_alpha(a)
        cap2.set_alpha(ramp(t, 3.6, 4.2))

    return render_animation("02_views_split", fig, update, duration_s=6.4, static_after=4.25)


# ----------------------------------------------------------------------------- 03

def chart_stage1():
    src = "results/library/learned.json:stage1"
    S = load("results/library/learned.json")["stage1"]
    rhos = [S["rho_A_metadata"], S["rho_B_brain"], S["rho_E_video"], S["rho_BE_brain_video"]]
    keys = ["rho_A_metadata", "rho_B_brain", "rho_E_video", "rho_BE_brain_video"]
    labels = ["Clip & account info", "+ brain response", "+ video/audio/text features", "+ both"]
    cols = [MUTED, GOLD, BLUE, ORANGE]
    be = S["BE_minus_A_content"]
    reach = S["reach_E_minus_A_content"]
    m = re.search(r">=\s*\+?(\d+\.\d+)", S["go_rule"])
    need = float(m.group(1))
    result = S["result"]
    for k, r in zip(keys, rhos):
        record(f"03.{k}", f"{r:.3f}", f"{src}.{k}", r)
    record("03.BE_minus_A_content.point", sgn(be["point"], 3), f"{src}.BE_minus_A_content.point", be["point"])
    record("03.BE_minus_A_content.ci95", f"{sgn(be['ci95'][0], 3)} to {sgn(be['ci95'][1], 3)}",
           f"{src}.BE_minus_A_content.ci95", be["ci95"])
    record("03.reach_E_minus_A_content.point", sgn(reach["point"], 3),
           f"{src}.reach_E_minus_A_content.point", reach["point"])
    record("03.reach_E_minus_A_content.ci95", f"{sgn(reach['ci95'][0], 3)} to {sgn(reach['ci95'][1], 3)}",
           f"{src}.reach_E_minus_A_content.ci95", reach["ci95"])
    record("03.go_rule_margin", f"+{need:.2f}", f"{src}.go_rule (parsed '>= +0.02')", need)
    record("03.pass_line", f"{rhos[0] + need:.3f}", f"{src}.rho_A_metadata + go_rule margin", rhos[0] + need)
    record("03.n_contents_primary", f"{S['n_contents_primary']:,}", f"{src}.n_contents_primary",
           S["n_contents_primary"])
    record("03.kish_n_eff", f"{S['kish_n_eff']:,}", f"{src}.kish_n_eff", S["kish_n_eff"])
    record("03.result", result, f"{src}.result", result)

    fig = new_fig()
    ov = overlay(fig)
    title(ov, "Pre-registered test: does the brain layer rank clips better?")
    txt(ov, 60, 116,
        f"Primary endpoint: engagement (interactions per view) · Spearman ρ within client × platform, "
        f"5-fold CV · {S['n_contents_primary']:,} clips in the primary fit (effective n {S['kish_n_eff']:,})",
        24, MUTED)
    rbox(ov, 40, 176, 1880, 1000, fc=PANEL, r=22, z=0)

    ax = fig.add_axes(px_rect(220, 222, 1560, 720))
    style_axes(ax)
    ax.set_ylim(0, 0.4)
    ax.set_xlim(-0.6, 3.6)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=pt(24), color=TEXT)
    ax.set_ylabel("Rank correlation with engagement\n(within client)", fontsize=pt(24), labelpad=14,
                  linespacing=1.25)
    bars = ax.bar(range(4), [0] * 4, width=0.62, color=cols, zorder=3)
    vlabels = [ax.text(i, 0.0, f"{r:.3f}", ha="center", va="top", fontsize=pt(32), color=BG,
                       fontweight="bold", alpha=0, zorder=4) for i, r in enumerate(rhos)]
    passy = rhos[0] + need
    pass_ln = ax.axhline(passy, color=GOLD, lw=2.2, ls=(0, (8, 6)), alpha=0, zorder=5)
    pass_tx = txt(ov, 1590, 222 + (0.4 - passy) / 0.4 * (720 - 222),
                  f"Pass line\n{rhos[0]:.3f} {'+'} {need:.2f} = {passy:.3f}", 22, GOLD, va="center",
                  alpha=0, linespacing=1.3)

    # stamp row
    stamp_tx = txt(ov, 110, 820, f"Result: {result}", 38, GOLD, weight="bold", va="center", alpha=0, z=6)
    fig.canvas.draw()
    bb = stamp_tx.get_window_extent()
    stamp_box = rbox(ov, bb.x0 - 22, 820 - 34, bb.x1 + 22, 820 + 34, fc="none", ec=GOLD, lw=3, r=10,
                     alpha=0, z=6)
    stamp_detail = txt(ov, bb.x1 + 50, 820,
                       f"brain + video features add {sgn(be['point'], 3)} "
                       f"(95% {sgn(be['ci95'][0], 3)} to {sgn(be['ci95'][1], 3)}); the rule needed "
                       f"+{need:.2f}", 28, TEXT, va="center", alpha=0)
    line2 = txt(ov, 110, 912,
                f"Video/audio/text features add {sgn(reach['point'], 3)} for views vs account's usual "
                f"(95% {sgn(reach['ci95'][0], 3)} to {sgn(reach['ci95'][1], 3)}) · secondary endpoint",
                27, MUTED, va="center", alpha=0)
    footer(ov, FOOTER_STAGE1)

    def update(t):
        for i, (b, r) in enumerate(zip(bars, rhos)):
            t0 = 0.4 + 0.45 * i
            b.set_height(r * ease(ramp(t, t0, t0 + 1.2)))
            vlabels[i].set_y(b.get_height() - 0.012)
            vlabels[i].set_alpha(ramp(t, t0 + 1.1, t0 + 1.45))
        a = ramp(t, 3.4, 3.9)
        pass_ln.set_alpha(0.9 * a)
        pass_tx.set_alpha(a)
        b = ramp(t, 4.0, 4.4)
        stamp_tx.set_alpha(b)
        stamp_box.set_alpha(b)
        stamp_detail.set_alpha(ramp(t, 4.2, 4.7))
        line2.set_alpha(ramp(t, 5.0, 5.5))

    return render_animation("03_stage1_result", fig, update, duration_s=7.6, static_after=5.55)


# ----------------------------------------------------------------------------- 04

TILES = [
    ("1,117 h of fMRI · 720 people", "evaluated across 8 datasets (trained on 25 people, 452 h)"),
    ("Video + audio + text", "reads V-JEPA2, w2v-BERT and Llama 3.2 features"),
    ("R ≈ 0.4 on the average brain",
     "on the 7T HCP set, about twice the median single person's scan as a predictor of the group average"),
    ("5 s brain delay", "predictions are aligned back to the moment in the video that caused them"),
]
BOTTOM_LINE = ("It models an average viewer's perception, not behaviour: it does not predict who keeps "
               "watching or shares.")


def card_tribe_paper():
    fig = new_fig()
    ov = overlay(fig)
    ttl = title(ov, "The brain model: TRIBE v2 (Meta FAIR, 2026)", y=52)
    GX0, GX1, GY0, GY1, GAP = 60, 1860, 170, 830, 40
    tw = (GX1 - GX0 - GAP) / 2
    th = (GY1 - GY0 - GAP) / 2
    groups = []
    for i, (head, sub) in enumerate(TILES):
        cx = GX0 + (i % 2) * (tw + GAP)
        cy = GY0 + (i // 2) * (th + GAP)
        box = rbox(ov, cx, cy, cx + tw, cy + th, fc=PANEL, r=22, alpha=0, z=1)
        acc = rbox(ov, cx, cy + 34, cx + 8, cy + th - 34, fc=GOLD, r=4, alpha=0, z=2)
        h = txt(ov, cx + 52, cy + 62, head, 46, TEXT, weight="bold", alpha=0)
        s = txt(ov, cx + 52, cy + 150, textwrap.fill(sub, 50), 30, MUTED, alpha=0, linespacing=1.35)
        groups.append((box, acc, h, s))
    bl = txt(ov, 960, 905, BOTTOM_LINE, 30, GOLD, ha="center", va="center", alpha=0)
    footer(ov, PAPER_FOOTER)

    def update(t):
        ttl.set_alpha(ramp(t, 0.0, 0.4))
        for i, (box, acc, h, s) in enumerate(groups):
            a = ramp(t, 0.6 + 1.0 * i, 1.3 + 1.0 * i)
            for art in (box, acc, h, s):
                art.set_alpha(a)
        bl.set_alpha(ramp(t, 4.7, 5.3))

    return render_animation("04_tribe_paper", fig, update, duration_s=7.5, static_after=5.35)


# ----------------------------------------------------------------------------- 05 / 06

CLIP_W, CLIP_H, CLIP_X, CLIP_Y = 562, 1000, 70, 40
RX0, RX1 = 700, 1860
AX = (840, 290, 1830, 735)  # chart axes px box
PLATFORM = {"youtube": "YouTube", "tiktok": "TikTok", "instagram": "Instagram"}


def fmt_x(x: float) -> str:
    if x >= 10:
        return f"{x:.0f}x"
    if x >= 1:
        return f"{x:.1f}x"
    return f"{x:.2f}x"


def fmt_t(t: float) -> str:
    s = int(t)
    return f"{s // 60}:{s % 60:02d}"


def find_pick(vid: str) -> dict:
    J = load("results/demo/judges_demo.json")
    hits = [(tier, e) for tier, es in J["picks"].items() for e in es if e.get("video_id") == vid]
    assert len(hits) == 1, (vid, len(hits))
    return hits[0][1]


def clip_ylim() -> float:
    m = 0.0
    for _, vid in CLIPS:
        idx = load(f"results/library/{vid}.library.json")["index"]
        for a in (idx["values"], idx["band"]["p25"], idx["band"]["p75"]):
            m = max(m, float(np.nanmax(np.abs(arr(a)))))
    return math.ceil(m * 1.12 / 0.25) * 0.25


def step_segments(vals, dur):
    """Coloured step line: horizontal run per second [i, min(i+1, dur)), vertical joins split at 0."""
    segs, cols = [], []

    def c(v):
        return ORANGE if v >= 0 else BLUE

    n = len(vals)
    for i in range(n):
        if i >= dur or not np.isfinite(vals[i]):
            continue
        x1 = min(i + 1.0, dur)
        segs.append([(i, vals[i]), (x1, vals[i])])
        cols.append(c(vals[i]))
        if i + 1 < n and i + 1 < dur and np.isfinite(vals[i + 1]):
            a, b = vals[i], vals[i + 1]
            if (a >= 0) == (b >= 0):
                segs.append([(i + 1, a), (i + 1, b)])
                cols.append(c(b))
            else:
                segs.append([(i + 1, a), (i + 1, 0.0)])
                cols.append(c(a))
                segs.append([(i + 1, 0.0), (i + 1, b)])
                cols.append(c(b))
    return segs, cols


def clip_video(prefix: str, vid: str, ylim: float):
    lib_src = f"results/library/{vid}.library.json"
    lib = load(lib_src)
    assert lib["video_id"] == vid
    pick = find_pick(vid)
    obs = pick["observed"]
    src_clip = CLIPS_DIR / f"{vid}.mp4"
    assert src_clip.is_file(), src_clip

    dur = lib["duration_ms"] / 1000.0
    idx = lib["index"]
    assert idx["hz"] == 1
    vals = arr(idx["values"])
    pcts = arr(idx["percentile"])
    p25, p75 = arr(idx["band"]["p25"]), arr(idx["band"]["p75"])
    n = len(vals)
    assert n == len(pcts) == len(p25) == len(p75) and n >= math.ceil(dur) - 1
    moments = sorted(lib["moments"], key=lambda m: m["start_ms"])
    n_ref = lib["reference"]["n_clips"]

    platform = PLATFORM.get(obs["platform"], obs["platform"])
    views = int(round(obs["views"]))
    xu = obs["views_vs_account_usual_x"]
    head1 = f"{pick['deal_label']} · {platform}"
    head2 = f"{views:,} views · {fmt_x(xu)} the account's usual"
    in_label = f"TRIBE v2 prediction · average subject · compared with {n_ref:,} library clips"
    record(f"{prefix}.header", f"{head1} · {head2}",
           "results/demo/judges_demo.json:picks[].{deal_label, observed.platform, observed.views, "
           "observed.views_vs_account_usual_x}", {"views": obs["views"], "x": xu})
    record(f"{prefix}.n_ref", f"{n_ref:,}", f"{lib_src}:reference.n_clips", n_ref)
    record(f"{prefix}.index", "per-second line, percentile readout (0 dp), p25-p75 band",
           f"{lib_src}:index.values / index.percentile / index.band.p25,p75", {"n_seconds": n, "duration_s": dur})
    record(f"{prefix}.moments", [m["plain"] for m in moments], f"{lib_src}:moments[].plain", len(moments))

    fig = new_fig()
    ov = overlay(fig)
    # clip frame placeholder border
    rbox(ov, CLIP_X - 3, CLIP_Y - 3, CLIP_X + CLIP_W + 3, CLIP_Y + CLIP_H + 3, fc=GRID, r=6, z=0)
    txt(ov, RX0, 34, head1, 46, TEXT, weight="bold")
    txt(ov, RX0, 100, head2, 34, TEXT)
    txt(ov, RX0, 160, in_label, 24, GOLD, weight="bold")
    rbox(ov, RX0, 205, RX1, 868, fc=PANEL, r=20, z=0)

    ax = fig.add_axes(px_rect(*AX))
    style_axes(ax)
    ax.set_xlim(0, dur)
    ax.set_ylim(-ylim, ylim)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(signed_tick))
    ax.set_xticks(np.arange(0, dur + 1e-9, 5))
    ax.set_xlabel("Seconds into the clip", fontsize=pt(24), labelpad=8)
    ax.set_ylabel("Predicted response (library sd)", fontsize=pt(24), labelpad=10)

    xs = np.arange(n, dtype=float)
    keep = xs < dur
    xstep = np.append(xs[keep], dur)
    ystep = lambda a: np.append(a[keep], a[keep][-1])  # noqa: E731
    ax.fill_between(xstep, ystep(p25), ystep(p75), step="post", color=MUTED, alpha=0.20, lw=0, zorder=1)
    ax.axhline(0, color=MUTED, lw=1.4, ls=(0, (6, 5)), alpha=0.9, zorder=2)

    v_step = ystep(vals)
    line_art = [
        ax.fill_between(xstep, 0, np.maximum(v_step, 0), step="post", color=ORANGE, alpha=BAND_ALPHA, lw=0,
                        zorder=3),
        ax.fill_between(xstep, np.minimum(v_step, 0), 0, step="post", color=BLUE, alpha=BAND_ALPHA, lw=0,
                        zorder=3),
    ]
    segs, cols = step_segments(vals, dur)
    lc = LineCollection(segs, colors=cols, linewidths=pt(5), capstyle="round", joinstyle="round", zorder=4)
    ax.add_collection(lc)
    line_art.append(lc)

    # legend row
    ly = 832
    ov.add_patch(Rectangle((RX0 + 40, ly - 11), 44, 22, fc=MUTED, alpha=0.20, lw=0, zorder=3))
    txt(ov, RX0 + 96, ly, "Library middle 50%", 24, MUTED, va="center")
    ov.add_line(Line2D([RX0 + 390, RX0 + 434], [ly, ly], color=MUTED, lw=pt(2), ls=(0, (4, 3)), zorder=3))
    txt(ov, RX0 + 446, ly, "Library average", 24, MUTED, va="center")
    ov.plot([RX0 + 690, RX0 + 718], [ly, ly], color=ORANGE, lw=pt(5), solid_capstyle="round", zorder=3)
    ov.plot([RX0 + 730, RX0 + 758], [ly, ly], color=BLUE, lw=pt(5), solid_capstyle="round", zorder=3)
    txt(ov, RX0 + 772, ly, "Above / below library average", 24, MUTED, va="center")

    # caveat (verbatim from the file) + footer
    txt(ov, RX0, 978, textwrap.fill(lib["caveat"], 118), 19, MUTED, linespacing=1.3)
    record(f"{prefix}.caveat", lib["caveat"], f"{lib_src}:caveat", None)
    footer(ov, x=RX0)

    # state-specific artists: moment box + highlighted span
    state_art = {}
    for k, m in enumerate(moments):
        s0, s1 = m["start_ms"] / 1000.0, m["end_ms"] / 1000.0
        span = ax.axvspan(s0, s1, color=GOLD, alpha=0.13, lw=0, zorder=0)
        box = rbox(ov, RX0, 884, RX1, 962, fc=PANEL, ec=GOLD, lw=1.8, r=14, z=1)
        tx = txt(ov, RX0 + 26, 923, textwrap.fill(m["plain"], 82), 24, TEXT, va="center", linespacing=1.3)
        state_art[k] = [span, box, tx]
    if not moments:
        txt(ov, RX0 + 26, 923, "No standout moments flagged for this clip.", 24, MUTED, va="center")

    def layer(state, with_line):
        for k, arts in state_art.items():
            for a in arts:
                a.set_visible(k == state)
        for a in line_art:
            a.set_visible(with_line)
        return fig_rgb(fig)

    states = [None] + list(range(len(moments)))
    layers = {s: (layer(s, False), layer(s, True)) for s in states}
    fig.canvas.draw()
    to_disp = ax.transData.transform
    ax_bb = ax.get_window_extent()
    ax_x0, ax_x1 = int(math.floor(ax_bb.x0)), int(math.ceil(ax_bb.x1))
    ax_y0, ax_y1 = int(math.floor(H - ax_bb.y1)), int(math.ceil(H - ax_bb.y0))
    plt.close(fig)

    def px(t, v):
        x, y = to_disp((t, v))
        return float(x), float(H - y)

    f_read = ImageFont.truetype(str(FONT_DIR / "DejaVuSans.ttf"), 26)
    f_read_b = ImageFont.truetype(str(FONT_DIR / "DejaVuSans-Bold.ttf"), 26)
    rgb = lambda h: tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))  # noqa: E731

    def readout(i):
        """Per-second readout; percentile wording mirrors the file's own 'higher/lower than X%' phrasing."""
        p = pcts[i]
        if not np.isfinite(p):
            return ""
        if p >= 50:
            return f"higher than {p:.0f}% of library clips at this second"
        return f"lower than {100 - p:.0f}% of library clips at this second"

    def compose(buf, t, state):
        base, full = layers[state]
        frame = base.copy()
        xp, _ = px(t, 0.0)
        xpi = int(round(xp))
        frame[ax_y0:ax_y1, ax_x0:xpi] = full[ax_y0:ax_y1, ax_x0:xpi]
        img = Image.fromarray(frame)
        d = ImageDraw.Draw(img)
        d.line([(xp, ax_y0), (xp, ax_y1)], fill=rgb(TEXT), width=3)
        i = min(int(t), n - 1)
        v = vals[i]
        if np.isfinite(v):
            _, yv = px(t, v)
            col = rgb(ORANGE if v >= 0 else BLUE)
            d.ellipse([xp - 11, yv - 11, xp + 11, yv + 11], fill=col, outline=rgb(PANEL), width=3)
            head = f"{sgn(v, 2)} sd"
            d.text((RX0 + 30, 222), head, font=f_read_b, fill=col)
            rest = readout(i)
            if rest:
                d.text((RX0 + 30 + f_read_b.getlength(head) + 14, 222), "·  " + rest, font=f_read,
                       fill=rgb(TEXT))
        d.text((RX1 - 30, 222), f"{fmt_t(t)} / {fmt_t(dur)}", font=f_read, fill=rgb(MUTED), anchor="ra")
        img.paste(Image.fromarray(np.frombuffer(buf, np.uint8).reshape(CLIP_H, CLIP_W, 3)), (CLIP_X, CLIP_Y))
        return np.asarray(img)

    dec = subprocess.Popen(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-i", str(src_clip),
         "-vf", f"fps={FPS},scale={CLIP_W}:{CLIP_H}:flags=bicubic", "-an",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    name = f"{prefix}_clip_{vid}"
    enc = Encoder(OUT / f"{name}.mp4", audio_src=src_clip)
    fsize = CLIP_W * CLIP_H * 3
    k = 0
    last_buf = None
    while True:
        buf = dec.stdout.read(fsize)
        if len(buf) < fsize:
            break
        t = min(k / FPS, dur)
        state = None
        for j, m in enumerate(moments):
            if m["start_ms"] / 1000.0 <= t < m["end_ms"] / 1000.0:
                state = j
        last_buf = buf
        enc.write(compose(buf, t, state))
        k += 1
    dec.stdout.close()
    if dec.wait() != 0:
        raise RuntimeError(f"decode failed for {src_clip}")
    enc.close()
    assert last_buf is not None, f"no frames decoded from {src_clip}"
    # Final-frame PNG: the last clip frame with the playhead at the clip's end (line fully drawn).
    last = compose(last_buf, dur, None)
    Image.fromarray(last).save(OUT / f"{name}.png")
    return k


# ----------------------------------------------------------------------------- main

def probe(path: Path) -> str:
    r = subprocess.run([FFMPEG, "-hide_banner", "-i", str(path)], capture_output=True, text=True)
    lines = [ln.strip() for ln in r.stderr.splitlines() if "Duration" in ln or "Stream #" in ln]
    return " | ".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="", help="comma list of asset prefixes, e.g. 01,05 (default: all)")
    args = ap.parse_args(argv)
    assert REPO not in OUT.resolve().parents and OUT.resolve() != REPO, "outputs must stay outside the repo"
    OUT.mkdir(parents=True, exist_ok=True)
    want = {s.strip() for s in args.only.split(",") if s.strip()}

    jobs = [("01", chart_views_vs_usual), ("02", chart_views_split), ("03", chart_stage1),
            ("04", card_tribe_paper)]
    ylim = clip_ylim()
    for prefix, vid in CLIPS:
        jobs.append((prefix, lambda p=prefix, v=vid: clip_video(p, v, ylim)))

    for prefix, fn in jobs:
        if want and prefix not in want:
            continue
        t0 = time.time()
        nframes = fn()
        print(f"[{prefix}] {nframes} frames in {time.time() - t0:.1f}s", flush=True)

    npath = OUT / "numbers_used.json"
    prev = json.loads(npath.read_text()) if (want and npath.exists()) else {}
    prev.update(NUMBERS)
    npath.write_text(json.dumps(prev, indent=1, ensure_ascii=False))
    for mp4 in sorted(OUT.glob("*.mp4")):
        print(f"{mp4.name}: {probe(mp4)}")


if __name__ == "__main__":
    sys.exit(main())
