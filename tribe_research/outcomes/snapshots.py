"""Snapshot cleaning and reach at fixed ages.

All work is on integer video codes (row positions in ``video_performances``)
with one sort of the snapshot frame; there are no per-video Python loops.
Times are float days since the Unix epoch (UTC).

Cleaning, in order:

1. drop snapshots of unknown videos, without a time, or with null/negative views;
2. dedupe on (video, snapshot time), keeping the largest views;
3. drop "broken zeros": views == 0 after an earlier non-zero snapshot;
4. enforce monotone non-decreasing cumulative views with a running maximum.
   Remaining decreases are counted; the largest relative drop is kept so a
   single spike (which the running max locks in) can be flagged downstream.

Reach at age A uses log-linear interpolation of ``log1p(views)`` in age
between the last snapshot at/before A and the first at/after A. Without a
bracket, a one-sided log-linear extrapolation from the two nearest snapshots is
allowed only when the nearest snapshot is within ``max_extrapolation_frac * A``
of A (0 disables it); otherwise the value is null.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BASIS_EXACT = "exact"
BASIS_INTERP = "interpolated"
BASIS_EXTRAP = "extrapolated"
BASIS_TOO_YOUNG = "too_young"
BASIS_NO_BRACKET = "no_bracket"
BASIS_NO_SNAPSHOTS = "no_snapshots"
BASIS_NO_UPLOAD = "no_upload_date"


def clean_snapshots(code: np.ndarray, t: np.ndarray, views: np.ndarray, n_videos: int):
    """Clean raw snapshots.

    ``code`` is the video row position (-1 = unknown video), ``t`` float days,
    ``views`` float. Returns ``(clean, stats)``: ``clean`` is a DataFrame with
    ``code, t, views`` sorted by (code, t) and monotone per code; ``stats`` is a
    dict of per-video numpy arrays of length ``n_videos`` plus scalar totals.
    """
    code = np.asarray(code, dtype=np.int64)
    t = np.asarray(t, dtype="float64")
    views = np.asarray(views, dtype="float64")

    known = code >= 0
    n_raw = np.bincount(code[known], minlength=n_videos)
    keep = known & np.isfinite(t) & np.isfinite(views) & (views >= 0)
    df = pd.DataFrame({"code": code[keep], "t": t[keep], "views": views[keep]})
    # one sort; ties on (code, t) end with the largest views -> keep="last" keeps it
    df = df.sort_values(["code", "t", "views"], kind="mergesort", ignore_index=True)

    dup = df.duplicated(["code", "t"], keep="last").to_numpy()
    n_dup = np.bincount(df["code"].to_numpy()[dup], minlength=n_videos)
    df = df.loc[~dup].reset_index(drop=True)

    g = df.groupby("code", sort=False)["views"]
    run_max = g.cummax().to_numpy()
    broken_zero = (df["views"].to_numpy() == 0) & (run_max > 0)
    n_broken_zero = np.bincount(df["code"].to_numpy()[broken_zero], minlength=n_videos)
    df = df.loc[~broken_zero].reset_index(drop=True)

    c = df["code"].to_numpy()
    v = df["views"].to_numpy()
    run_max = df.groupby("code", sort=False)["views"].cummax().to_numpy()
    prev_max = np.empty_like(run_max)
    if len(run_max):
        prev_max[0] = np.nan
        prev_max[1:] = run_max[:-1]
        first = np.r_[True, c[1:] != c[:-1]]
        prev_max[first] = np.nan
    decrease = np.isfinite(prev_max) & (v < prev_max)
    drop_frac = np.where(decrease & (prev_max > 0), (prev_max - v) / np.where(prev_max > 0, prev_max, 1), 0.0)
    n_decrease = np.bincount(c[decrease], minlength=n_videos)
    max_drop = np.zeros(n_videos)
    if decrease.any():
        np.maximum.at(max_drop, c[decrease], drop_frac[decrease])

    df["views"] = run_max
    n_clean = np.bincount(c, minlength=n_videos)
    last_idx = np.r_[c[1:] != c[:-1], True] if len(c) else np.zeros(0, bool)
    first_idx = np.r_[True, c[1:] != c[:-1]] if len(c) else np.zeros(0, bool)
    last_t = np.full(n_videos, np.nan)
    last_v = np.full(n_videos, np.nan)
    first_t = np.full(n_videos, np.nan)
    last_t[c[last_idx]] = df["t"].to_numpy()[last_idx]
    last_v[c[last_idx]] = run_max[last_idx]
    first_t[c[first_idx]] = df["t"].to_numpy()[first_idx]

    stats = {
        "n_snapshots_raw": n_raw,
        "n_snapshots": n_clean,
        "n_duplicates": n_dup,
        "n_broken_zeros": n_broken_zero,
        "n_decreases": n_decrease,
        "max_drop_frac": max_drop,
        "first_snapshot_t": first_t,
        "last_snapshot_t": last_t,
        "last_snapshot_views": last_v,
        "totals": {
            "rows_raw": int(len(code)),
            "rows_unknown_video": int((~known).sum()),
            "rows_invalid": int((known & ~keep).sum()),
            "rows_duplicate": int(dup.sum()),
            "rows_broken_zero": int(broken_zero.sum()),
            "rows_clean": int(len(df)),
        },
    }
    return df, stats


def _scatter(n: int, idx: np.ndarray, vals: np.ndarray) -> np.ndarray:
    out = np.full(n, np.nan)
    out[idx] = vals
    return out


def views_at_age(clean: pd.DataFrame, upload_t: np.ndarray, age_last_obs: np.ndarray,
                 age: float, max_extrapolation_frac: float = 0.2):
    """Views at ``age`` days for every video.

    ``clean`` comes from :func:`clean_snapshots`; ``upload_t`` is the per-video
    upload anchor (float days, NaN if unknown); ``age_last_obs`` is the per-video
    age at the last views observation. Returns ``(views, basis, span_days)``
    arrays of length ``len(upload_t)``; ``span_days`` is the width of the
    bracket (0 when exact, NaN when not interpolated).
    """
    n = len(upload_t)
    c = clean["code"].to_numpy()
    a = clean["t"].to_numpy() - upload_t[c]
    lv = np.log1p(clean["views"].to_numpy())
    ok = np.isfinite(a)
    c, a, lv = c[ok], a[ok], lv[ok]
    # neighbours within the same video (rows are sorted by code, then time/age)
    same_prev = np.r_[False, c[1:] == c[:-1]]
    same_next = np.r_[c[1:] == c[:-1], False]
    a_prev = np.r_[np.nan, a[:-1]]; lv_prev = np.r_[np.nan, lv[:-1]]
    a_next = np.r_[a[1:], np.nan]; lv_next = np.r_[lv[1:], np.nan]
    a_prev[~same_prev] = np.nan; lv_prev[~same_prev] = np.nan
    a_next[~same_next] = np.nan; lv_next[~same_next] = np.nan

    le = a <= age
    # last row at/before age per code: le and (next row is another code or next row > age)
    lo_row = le & (~same_next | ~np.r_[le[1:], False])
    ge = a >= age
    # first row at/after age per code: ge and (previous row is another code or previous < age)
    hi_row = ge & (~same_prev | ~np.r_[False, ge[:-1]])

    a_lo = _scatter(n, c[lo_row], a[lo_row]); lv_lo = _scatter(n, c[lo_row], lv[lo_row])
    a_lo2 = _scatter(n, c[lo_row], a_prev[lo_row]); lv_lo2 = _scatter(n, c[lo_row], lv_prev[lo_row])
    a_hi = _scatter(n, c[hi_row], a[hi_row]); lv_hi = _scatter(n, c[hi_row], lv[hi_row])
    a_hi2 = _scatter(n, c[hi_row], a_next[hi_row]); lv_hi2 = _scatter(n, c[hi_row], lv_next[hi_row])
    has_any = np.bincount(c, minlength=n) > 0

    out = np.full(n, np.nan)
    span = np.full(n, np.nan)
    basis = np.full(n, BASIS_NO_BRACKET, dtype=object)

    has_lo, has_hi = np.isfinite(a_lo), np.isfinite(a_hi)
    exact = (has_lo & (a_lo == age)) | (has_hi & (a_hi == age))
    both = has_lo & has_hi & ~exact
    with np.errstate(invalid="ignore", divide="ignore"):
        w = (age - a_lo) / (a_hi - a_lo)
        interp = lv_lo + (lv_hi - lv_lo) * w
    out[both] = interp[both]
    span[both] = (a_hi - a_lo)[both]
    basis[both] = BASIS_INTERP
    out[exact] = np.where(has_lo & (a_lo == age), lv_lo, lv_hi)[exact]
    span[exact] = 0.0
    basis[exact] = BASIS_EXACT

    tol = max_extrapolation_frac * age
    with np.errstate(invalid="ignore", divide="ignore"):
        fwd_slope = (lv_lo - lv_lo2) / (a_lo - a_lo2)
        bwd_slope = (lv_hi2 - lv_hi) / (a_hi2 - a_hi)
    fwd = has_lo & ~has_hi & np.isfinite(fwd_slope) & (age - a_lo <= tol) & (tol > 0)
    bwd = has_hi & ~has_lo & np.isfinite(bwd_slope) & (a_hi - age <= tol) & (tol > 0)
    out[fwd] = (lv_lo + fwd_slope * (age - a_lo))[fwd]
    # backward: never below zero views and never above the first observed value
    out[bwd] = np.clip(lv_hi - bwd_slope * (a_hi - age), 0.0, lv_hi)[bwd]
    basis[fwd | bwd] = BASIS_EXTRAP

    unresolved = ~(both | exact | fwd | bwd)
    too_young = unresolved & (age_last_obs < age)
    basis[too_young] = BASIS_TOO_YOUNG
    basis[unresolved & ~has_any] = BASIS_NO_SNAPSHOTS
    basis[np.isnan(upload_t)] = BASIS_NO_UPLOAD
    out[np.isnan(upload_t)] = np.nan
    return np.expm1(out), basis, span
