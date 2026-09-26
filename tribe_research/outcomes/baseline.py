"""Account baselines (leave-one-out medians) and follower counts near upload.

The baseline for a video is the median ``log1p(views@A)`` of the *other*
videos in its group. The leave-one-out median is computed for all groups at
once from one sort (rank-index trick), so there is no per-group Python loop.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def loo_median(group: np.ndarray, y: np.ndarray, min_others: int = 5):
    """Leave-one-out median of ``y`` within ``group``.

    ``group`` is an int code (-1 = no group). For a row with a finite ``y`` the
    median is over the other finite values in its group; for a row with a null
    ``y`` it is over all finite values of its group. Returns ``(median,
    n_others)``; ``median`` is NaN where ``n_others < min_others``.
    """
    group = np.asarray(group, dtype=np.int64)
    y = np.asarray(y, dtype="float64")
    n = len(y)
    med = np.full(n, np.nan)
    n_others = np.zeros(n, dtype=np.int64)
    if n == 0:
        return med, n_others
    valid = (group >= 0) & np.isfinite(y)
    n_groups = int(group.max()) + 1 if (group >= 0).any() else 0
    if n_groups == 0:
        return med, n_others

    vidx = np.flatnonzero(valid)
    order = np.lexsort((y[vidx], group[vidx]))
    sorted_rows = vidx[order]
    s = y[sorted_rows]
    if len(s) == 0:
        return med, n_others
    g_sorted = group[sorted_rows]
    counts = np.bincount(g_sorted, minlength=n_groups)
    start = np.r_[0, np.cumsum(counts)[:-1]]
    # position of each valid row inside its group's sorted block
    pos = np.full(n, np.iinfo(np.int64).max, dtype=np.int64)
    pos[sorted_rows] = np.arange(len(sorted_rows)) - start[g_sorted]

    has_group = group >= 0
    gi = np.where(has_group, group, 0)
    m = np.where(has_group, counts[gi] - valid.astype(np.int64), 0)
    n_others[:] = m
    use = has_group & (m >= max(min_others, 1))
    k = pos  # excluded position (int max for rows that exclude nothing)
    j2 = m // 2
    j1 = np.where(m % 2 == 1, j2, j2 - 1)

    def pick(j):
        jj = j + (j >= k)
        idx = start[gi] + jj
        return s[np.clip(idx, 0, len(s) - 1)]

    vals = 0.5 * (pick(j1) + pick(j2))
    med[use] = vals[use]
    return med, n_others


def followers_near(upload_t: np.ndarray, account: pd.Series, stats: pd.DataFrame,
                   stat_t: np.ndarray, tolerance_days: float):
    """Follower count of the account snapshot nearest to the upload anchor.

    Returns ``(follower_count, gap_days)`` where ``gap_days`` is snapshot minus
    upload (negative = snapshot before upload). Null when no snapshot within
    ``tolerance_days`` or the upload date / account is unknown.
    """
    n = len(upload_t)
    fc = np.full(n, np.nan)
    gap = np.full(n, np.nan)
    st = pd.DataFrame({"acct": stats["social_account_id"].to_numpy(dtype=object),
                       "t": stat_t, "followers": stats["follower_count"].to_numpy(dtype="float64")})
    # a follower_count of 0 is treated as "not reported" (most zero rows in the
    # export are provider misses, not accounts with no followers)
    st = st.dropna()
    st = st[st["followers"] > 0].drop_duplicates(["acct", "t"], keep="last")
    left = pd.DataFrame({"row": np.arange(n), "acct": account.to_numpy(dtype=object), "t": upload_t})
    left = left.dropna()
    if st.empty or left.empty:
        return fc, gap
    st["t_snap"] = st["t"]
    merged = pd.merge_asof(left.sort_values("t"), st.sort_values("t"), on="t", by="acct",
                           direction="nearest", tolerance=float(tolerance_days))
    merged = merged.dropna(subset=["followers"])
    rows = merged["row"].to_numpy(dtype=np.int64)
    fc[rows] = merged["followers"].to_numpy()
    gap[rows] = (merged["t_snap"] - merged["t"]).to_numpy()
    return fc, gap


def local_loo_median(group: np.ndarray, t: np.ndarray, y: np.ndarray, k: int = 10, min_others: int = 5):
    """Median ``y`` of the ``k`` other same-group rows nearest in time ``t``.

    Controls for account growth: an early clip is compared with the account's
    other early clips, not with its whole history. Candidates are rows of the
    same group with finite ``t`` and ``y``; the target itself is excluded. In
    one dimension the k nearest form a contiguous window around the target in
    time order, so the window is grown ``k`` times with two pointers (a loop
    over ``k``, vectorised over all rows). Ties in distance take the earlier
    video. Returns ``(median, n_used, max_gap_days)``; ``median`` is NaN where
    ``n_used < min_others``.
    """
    group = np.asarray(group, dtype=np.int64)
    t = np.asarray(t, dtype="float64")
    y = np.asarray(y, dtype="float64")
    n = len(y)
    med = np.full(n, np.nan)
    n_used = np.zeros(n, dtype=np.int64)
    max_gap = np.full(n, np.nan)
    has_g = (group >= 0) & np.isfinite(t)
    valid = has_g & np.isfinite(y)
    if not valid.any() or k <= 0:
        return med, n_used, max_gap

    vidx = np.flatnonzero(valid)
    rows = vidx[np.lexsort((vidx, t[vidx], group[vidx]))]
    gs, ts, ys = group[rows], t[rows], y[rows]
    counts = np.bincount(gs, minlength=int(group.max()) + 1)
    start = np.r_[0, np.cumsum(counts)[:-1]]
    end = start + counts
    # composite key (group, time) for insertion positions of rows without their own y
    t0 = float(ts.min())
    scale = float(np.nanmax(t[has_g]) - min(t0, float(np.nanmin(t[has_g])))) + 2.0
    key = gs * scale + (ts - t0)
    pos = np.full(n, -1, dtype=np.int64)
    pos[rows] = np.arange(len(rows))

    tgt = np.flatnonzero(has_g)
    gi, tt = group[tgt], t[tgt]
    is_v = valid[tgt]
    ins = np.searchsorted(key, gi * scale + (tt - t0), side="left")
    left = np.where(is_v, pos[tgt] - 1, ins - 1)
    right = np.where(is_v, pos[tgt] + 1, ins)
    lo, hi = start[gi], end[gi]
    last = len(ts) - 1
    vals = np.full((len(tgt), k), np.nan)
    gap = np.zeros(len(tgt))
    for step in range(k):
        can_l, can_r = left >= lo, right < hi
        dl = np.where(can_l, tt - ts[np.clip(left, 0, last)], np.inf)
        dr = np.where(can_r, ts[np.clip(right, 0, last)] - tt, np.inf)
        take_l = can_l & (dl <= dr)
        take_r = can_r & ~take_l
        take = take_l | take_r
        idx = np.where(take_l, left, right)
        vals[take, step] = ys[idx[take]]
        gap = np.where(take, np.maximum(gap, np.where(take_l, dl, dr)), gap)
        left = np.where(take_l, left - 1, left)
        right = np.where(take_r, right + 1, right)
    used = np.isfinite(vals).sum(axis=1)
    ok = used >= max(min_others, 1)
    n_used[tgt] = used
    max_gap[tgt] = np.where(used > 0, gap, np.nan)
    if ok.any():
        med[tgt[ok]] = np.nanmedian(vals[ok], axis=1)
    return med, n_used, max_gap
