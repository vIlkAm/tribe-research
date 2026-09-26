"""Neural proxy channels: ROI-group unions -> within-clip z curves on a display grid.

Timing: TRIBE v2 trains against fMRI 5 s later than the stimulus
(``grids/defaults.py`` neuro_extractor ``offset: 5``), so each prediction row
is already in *stimulus* time: ``seg_start`` lines up with the video frame.
No further shift is applied here.

Normalization: z-score within the clip, with the mean/sd estimated outside the
onset window (the first ``ONSET_S`` seconds carry a stimulus-onset transient
that would otherwise dominate the scale). Onset samples are still reported,
just flagged. Values are relative to this clip only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from tribe_research.brain.features import RoiMap

PROXIES_FILE = Path(__file__).with_name("proxies_v0.yaml")
DIRECTIONS = {"learned_positive", "learned_negative", "hypothesis_positive", "context_dependent", "no_monotonic"}
ONSET_S = 2.0
MIN_BASELINE_POINTS = 4  # need this many post-onset samples to exclude the onset from the scale
DISPLAY_HZ = 2


@dataclass(frozen=True)
class ProxyDef:
    key: str
    label: str
    roi_groups: list[str]
    regions_text: str
    direction: str
    direction_note: str
    default_visible: bool
    color: str
    copy: dict


@dataclass(frozen=True)
class ProxySpec:
    version: str
    channels: list[ProxyDef]
    unavailable: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path = PROXIES_FILE) -> "ProxySpec":
        raw = yaml.safe_load(Path(path).read_text())
        chans = [ProxyDef(key=k, **v) for k, v in raw["channels"].items()]
        for c in chans:
            if c.direction not in DIRECTIONS:
                raise ValueError(f"{c.key}: unknown direction {c.direction!r}")
        return cls(raw["version"], chans, raw.get("unavailable", {}))


def channel_masks(spec: ProxySpec, roi: RoiMap) -> np.ndarray:
    """bool [C, V]: union of each channel's ROI group masks."""
    idx = {g: i for i, g in enumerate(roi.group_names)}
    out = np.zeros((len(spec.channels), roi.group_mask.shape[1]), bool)
    for c, ch in enumerate(spec.channels):
        missing = [g for g in ch.roi_groups if g not in idx]
        if missing:
            raise ValueError(f"channel {ch.key}: ROI map has no group(s) {missing}")
        for g in ch.roi_groups:
            out[c] |= roi.group_mask[idx[g]]
    return out


def channel_raw(preds: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Vertex-weighted mean over each channel's region: [T, V] -> [C, T]."""
    m = masks.astype(np.float32)
    return (m @ preds.astype(np.float32).T) / m.sum(axis=1, keepdims=True)


def onset_mask(t: np.ndarray) -> np.ndarray:
    return t < ONSET_S


def zscore_excluding_onset(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, bool]:
    """z-score each row of ``y`` [C, T]; returns (z, onset_excluded_from_scale)."""
    base = ~onset_mask(t)
    excluded = int(base.sum()) >= MIN_BASELINE_POINTS
    ref = y[:, base] if excluded else y
    mu = ref.mean(axis=1, keepdims=True)
    sd = ref.std(axis=1, keepdims=True)
    sd[sd < 1e-12] = 1.0
    return (y - mu) / sd, excluded


def to_display_grid(t: np.ndarray, dur: np.ndarray, y: np.ndarray, duration_s: float,
                    hz: int = DISPLAY_HZ) -> tuple[np.ndarray, list[list[float | None]]]:
    """Sample-and-hold native segments onto a uniform grid; uncovered time -> None.

    TRIBE drops segments with no events, so gaps are real and must stay visible
    rather than being interpolated.
    """
    grid = np.arange(0, duration_s, 1.0 / hz)
    rows: list[list[float | None]] = [[None] * len(grid) for _ in range(y.shape[0])]
    for k, g in enumerate(grid):
        i = int(np.searchsorted(t, g, side="right")) - 1
        if i >= 0 and g < t[i] + dur[i]:
            for c in range(y.shape[0]):
                rows[c][k] = round(float(y[c, i]), 3)
    return grid, rows


def gaps(t: np.ndarray, dur: np.ndarray, duration_s: float) -> list[list[int]]:
    """[start_ms, end_ms] spans with no prediction."""
    out, cursor = [], 0.0
    for s, d in zip(t, dur):
        if s > cursor + 1e-6:
            out.append([int(round(cursor * 1000)), int(round(s * 1000))])
        cursor = max(cursor, s + d)
    if duration_s > cursor + 1e-6:
        out.append([int(round(cursor * 1000)), int(round(duration_s * 1000))])
    return out
