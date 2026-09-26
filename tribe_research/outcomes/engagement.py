"""Empirical-Bayes (beta-binomial) shrinkage of engagement rates.

For a count ``k`` (likes, comments, ...) over ``n`` views, the prior
``Beta(mu * c, (1 - mu) * c)`` is fitted by the method of moments per
deal x platform on videos with ``n >= min_views``:

    p_i   = k_i / n_i
    mu    = mean(p_i)
    tau2  = var(p_i) - mean(mu * (1 - mu) / n_i)       # remove binomial noise
    c     = mu * (1 - mu) / tau2 - 1, clipped to [min_conc, max_conc]

(``tau2 <= 0``, i.e. no detectable between-video spread, gives ``max_conc``.)
Groups with fewer than ``min_videos`` usable videos fall back to the platform
prior, then to a global prior. The posterior is ``Beta(a + k, b + n - k)``:
mean ``(a + k) / (c + n)`` and an equal-tailed interval. Views are the
trials; counts above views are clipped to views and flagged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import betaincinv

LEVELS = ("deal_platform", "platform", "global")


def _fit_groups(key: pd.Series, k: np.ndarray, n: np.ndarray, min_views: float, min_videos: int,
                min_conc: float, max_conc: float) -> pd.DataFrame:
    use = np.isfinite(k) & np.isfinite(n) & (n >= min_views) & key.notna().to_numpy()
    p = np.clip(k[use] / n[use], 0.0, 1.0)
    df = pd.DataFrame({"key": key.to_numpy(dtype=object)[use], "p": p, "inv_n": 1.0 / n[use]})
    if df.empty:
        return pd.DataFrame(columns=["n_videos", "mu", "conc", "tau2"])
    agg = df.groupby("key").agg(n_videos=("p", "size"), mu=("p", "mean"), var=("p", "var"),
                                inv_n=("inv_n", "mean"))
    agg = agg[agg["n_videos"] >= min_videos].copy()
    mu = agg["mu"].clip(1e-6, 1 - 1e-6)
    tau2 = agg["var"] - mu * (1 - mu) * agg["inv_n"]
    with np.errstate(divide="ignore", invalid="ignore"):
        conc = np.where(tau2 > 0, mu * (1 - mu) / tau2 - 1.0, max_conc)
    agg["mu"] = mu
    agg["tau2"] = tau2
    agg["conc"] = np.clip(conc, min_conc, max_conc)
    return agg[["n_videos", "mu", "conc", "tau2"]]


def shrink(k: np.ndarray, n: np.ndarray, deal_platform: pd.Series, platform: pd.Series, *,
           min_views: float = 100, min_videos: int = 30, min_conc: float = 1.0,
           max_conc: float = 1e4, interval: float = 0.90):
    """Shrink ``k / n`` per row. Returns a dict of arrays plus the fitted priors.

    Keys: ``raw`` (clipped to 1), ``eb``, ``lo``, ``hi``, ``prior_level``,
    ``prior_mean``, ``gt1`` (count above views) and ``priors`` (list of dicts).
    """
    k = np.asarray(k, dtype="float64")
    n = np.asarray(n, dtype="float64")
    rows = len(k)
    gt1 = np.isfinite(k) & np.isfinite(n) & (k > n)
    kc = np.where(gt1, n, k)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = np.where(n > 0, kc / n, np.nan)

    glob_key = pd.Series("all", index=platform.index, dtype=object)
    fits = {
        "deal_platform": (deal_platform, _fit_groups(deal_platform, k, n, min_views, min_videos, min_conc, max_conc)),
        "platform": (platform, _fit_groups(platform, k, n, min_views, min_videos, min_conc, max_conc)),
        "global": (glob_key, _fit_groups(glob_key, k, n, min_views, min_videos, min_conc, max_conc)),
    }
    mu = np.full(rows, np.nan)
    conc = np.full(rows, np.nan)
    level = np.full(rows, "none", dtype=object)
    for lvl in reversed(LEVELS):  # finest level written last wins
        key, fit = fits[lvl]
        m = key.map(fit["mu"]).to_numpy(dtype="float64", na_value=np.nan)
        cc = key.map(fit["conc"]).to_numpy(dtype="float64", na_value=np.nan)
        hit = np.isfinite(m)
        mu[hit], conc[hit], level[hit] = m[hit], cc[hit], lvl

    has = np.isfinite(kc) & np.isfinite(n) & (n >= 0) & np.isfinite(mu)
    a = mu * conc
    b = (1 - mu) * conc
    kk = np.where(has, kc, 0.0)
    nn = np.where(has, n, 0.0)
    eb = np.where(has, (a + kk) / (conc + nn), np.nan)
    tail = (1 - interval) / 2
    pa = np.where(has, a + kk, 1.0)
    pb = np.where(has, b + nn - kk, 1.0)
    lo = np.where(has, betaincinv(pa, pb, tail), np.nan)
    hi = np.where(has, betaincinv(pa, pb, 1 - tail), np.nan)
    level = np.where(np.isfinite(kc) & np.isfinite(n), level, "none")

    priors = []
    for lvl in LEVELS:
        for key, r in fits[lvl][1].iterrows():
            priors.append({"level": lvl, "key": str(key), "n_videos": int(r["n_videos"]),
                           "mu": float(r["mu"]), "concentration": float(r["conc"]),
                           "alpha": float(r["mu"] * r["conc"]), "beta": float((1 - r["mu"]) * r["conc"]),
                           "tau2": float(r["tau2"])})
    return {"raw": raw, "eb": eb, "lo": lo, "hi": hi, "prior_level": level,
            "prior_mean": np.where(has, mu, np.nan), "gt1": gt1, "priors": priors}
