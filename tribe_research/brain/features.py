"""ROI time series and per-video summary features (backend guide §05, §07).

Everything is computed on the real segment timeline (``seg_start`` seconds),
never on row index, because TRIBE drops empty segments and TR need not be 1 s.

Summaries are derived from the within-video z-scored curve unless the name
says ``raw``. Absolute model-space values are kept (``raw_mean``) but should not
be compared across videos until validated. See the guide: "Do not use absolute
activation as a score".
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

FEATURE_VERSION = "summary_v0"
MIN_POINTS = 6  # below this, a curve is flagged short and variance/transitions are unreliable


@dataclass(frozen=True)
class RoiMap:
    group_names: list[str]
    group_mask: np.ndarray  # bool [G, V]
    provenance: dict

    @classmethod
    def load(cls, path: str | Path) -> "RoiMap":
        z = np.load(path, allow_pickle=False)
        return cls(
            group_names=[str(g) for g in z["group_names"]],
            group_mask=z["group_mask"].astype(bool),
            provenance=json.loads(str(z["provenance"])),
        )


def roi_curves(preds: np.ndarray, roi_map: RoiMap) -> np.ndarray:
    """Mean prediction over each group's vertices: [T, V] -> [G, T] (float32)."""
    if preds.shape[1] != roi_map.group_mask.shape[1]:
        raise ValueError(f"preds has {preds.shape[1]} vertices, ROI map {roi_map.group_mask.shape[1]}")
    mask = roi_map.group_mask.astype(np.float32)
    counts = mask.sum(axis=1, keepdims=True)
    if (counts == 0).any():
        raise ValueError("ROI group with zero vertices")
    return (mask @ preds.astype(np.float32).T) / counts


def zscore(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return (x - x.mean()) / sd if sd > 1e-12 else np.zeros_like(x)


def _window(t: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (t >= lo) & (t < hi)


def _slope(t: np.ndarray, y: np.ndarray) -> float:
    if len(t) < 2 or np.ptp(t) == 0:
        return float("nan")
    return float(np.polyfit(t, y, 1)[0])


def count_transitions(z: np.ndarray, threshold: float = 0.5) -> int:
    """Hysteresis crossings between z > +thr and z < -thr (ignores jitter near 0)."""
    state, n = 0, 0
    for v in z:
        new = 1 if v > threshold else -1 if v < -threshold else state
        if state != 0 and new != state:
            n += 1
        state = new
    return n


@dataclass
class Summary:
    signal_key: str
    n_points: int
    short_clip: bool
    raw_mean: float
    mean_0_3s: float
    mean_0_5s: float
    peak_value: float
    peak_time_ms: int
    trough_value: float
    trough_time_ms: int
    slope_0_3s: float
    temporal_variance: float
    auc_normalized: float
    n_transitions: int


def summarize(signal_key: str, t: np.ndarray, y: np.ndarray, tr: float) -> Summary:
    """Summary features for one ROI curve ``y`` sampled at segment starts ``t`` (s)."""
    order = np.argsort(t)
    t, y = t[order], y[order]
    z = zscore(y)

    def wmean(lo, hi):
        m = _window(t, lo, hi)
        return float(z[m].mean()) if m.any() else float("nan")

    w3 = _window(t, 0, 3)
    span = (t[-1] - t[0]) + tr if len(t) else 0.0
    # left-Riemann AUC over real time: each sample holds for min(gap, TR)
    widths = np.minimum(np.diff(np.append(t, t[-1] + tr)), tr) if len(t) else np.array([])
    return Summary(
        signal_key=signal_key,
        n_points=int(len(t)),
        short_clip=bool(len(t) < MIN_POINTS),
        raw_mean=float(y.mean()),
        mean_0_3s=wmean(0, 3),
        mean_0_5s=wmean(0, 5),
        peak_value=float(z.max()),
        peak_time_ms=int(round(t[int(z.argmax())] * 1000)),
        trough_value=float(z.min()),
        trough_time_ms=int(round(t[int(z.argmin())] * 1000)),
        slope_0_3s=_slope(t[w3], z[w3]),
        temporal_variance=float(y.var()),
        auc_normalized=float((z * widths).sum() / span) if span > 0 else float("nan"),
        n_transitions=count_transitions(z),
    )


def video_features(npz_path: str | Path, roi_map: RoiMap, tr: float) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Load one worker output and return (seg_start, curves[G,T], summaries)."""
    z = np.load(npz_path)
    t = z["seg_start"].astype(np.float64)
    curves = roi_curves(z["preds"], roi_map)
    summaries = [
        {**asdict(summarize(g, t, curves[i], tr)), "feature_version": FEATURE_VERSION}
        for i, g in enumerate(roi_map.group_names)
    ]
    return t, curves, summaries
