"""Within-group percentiles and the reach x engagement quadrant.

Percentiles are mid-rank: ``(average_rank - 0.5) / n`` in (0, 1), computed only
among rows with a value, and only when the group has at least ``min_n`` such
rows. The quadrant is a two-way label against the group medians; it is
deliberately *not* combined into a single score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

QUADRANTS = {
    (True, True): "high_reach_high_engagement",
    (True, False): "high_reach_low_engagement",
    (False, True): "low_reach_high_engagement",
    (False, False): "low_reach_low_engagement",
}


def group_percentile(values: np.ndarray, group: pd.Series, min_n: int = 5):
    """Returns ``(percentile, n)``; ``n`` = rows with a value in the group."""
    v = np.asarray(values, dtype="float64")
    df = pd.DataFrame({"g": group.to_numpy(dtype=object), "v": v})
    ok = df["g"].notna() & np.isfinite(v)
    sub = df[ok]
    pct = np.full(len(v), np.nan)
    if sub.empty:
        return pct, np.zeros(len(v), dtype=np.int64)
    gn = sub.groupby("g")["v"].transform("size").to_numpy()
    rk = sub.groupby("g")["v"].rank(method="average").to_numpy()
    p = (rk - 0.5) / gn
    p[gn < min_n] = np.nan
    pct[np.flatnonzero(ok.to_numpy())] = p
    # every row in a group sees the group's n, including rows without a value
    n = df["g"].map(sub.groupby("g").size()).fillna(0).to_numpy(dtype=np.int64)
    return pct, n


def quadrant(reach: np.ndarray, engagement: np.ndarray, group: pd.Series, min_n: int = 5):
    """Label each row by reach / engagement at-or-above vs below its group median.

    Medians are taken over rows that have both values; groups with fewer than
    ``min_n`` such rows get no label.
    """
    r = np.asarray(reach, dtype="float64")
    e = np.asarray(engagement, dtype="float64")
    df = pd.DataFrame({"g": group.to_numpy(dtype=object), "r": r, "e": e})
    ok = df["g"].notna() & np.isfinite(r) & np.isfinite(e)
    out = np.full(len(r), None, dtype=object)
    sub = df[ok]
    if sub.empty:
        return out
    grp = sub.groupby("g")
    size = grp["r"].transform("size").to_numpy()
    hi_r = sub["r"].to_numpy() >= grp["r"].transform("median").to_numpy()
    hi_e = sub["e"].to_numpy() >= grp["e"].transform("median").to_numpy()
    labels = np.array([QUADRANTS[(a, b)] for a, b in zip(hi_r, hi_e)], dtype=object)
    labels[size < min_n] = None
    out[np.flatnonzero(ok.to_numpy())] = labels
    return out
