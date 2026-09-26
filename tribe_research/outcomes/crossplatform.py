"""Content groups: the same clip posted on several platforms.

``content_group`` is ``"link:<link_id>"`` for videos in a cross-platform link
and ``"video:<id>"`` otherwise. Spreads use only account-normalised or
within-deal x platform quantities (``reach_rel``, the engagement percentile),
because raw rates differ structurally between platforms.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PLATFORMS = ("tiktok", "instagram", "youtube")


def content_groups(video_ids: pd.Series, members: pd.DataFrame):
    """Returns ``(content_group, n_conflicting_memberships)``.

    A video should be in at most one link; if the export violates that, the
    first link listed wins and the conflict count is reported.
    """
    m = members[["link_id", "video_performance_id"]].dropna()
    conflicts = int(m.drop_duplicates().duplicated("video_performance_id").sum())
    m = m.drop_duplicates("video_performance_id", keep="first")
    link = video_ids.map(m.set_index("video_performance_id")["link_id"])
    group = np.where(link.notna(), "link:" + link.astype(str), "video:" + video_ids.astype(str))
    return pd.Series(group, index=video_ids.index, dtype=object), conflicts


def group_spread(group: pd.Series, platform: pd.Series, reach_rel: np.ndarray,
                 engagement_pct: np.ndarray) -> dict[str, np.ndarray]:
    """Per-group size, platform count and each member's offset from the group mean.

    Offsets are only defined for groups with at least two members that have
    the value. ``group_reach_rel_<platform>`` is the group's mean ``reach_rel``
    on that platform (same value on every row of the group; null for single-video
    groups).
    """
    df = pd.DataFrame({"g": group.to_numpy(dtype=object), "p": platform.to_numpy(dtype=object),
                       "r": np.asarray(reach_rel, dtype="float64"),
                       "e": np.asarray(engagement_pct, dtype="float64")})
    grp = df.groupby("g", sort=False)
    out = {
        "content_group_size": grp["g"].transform("size").to_numpy(dtype=np.int64),
        "content_group_n_platforms": grp["p"].transform("nunique").to_numpy(dtype=np.int64),
    }
    for col, name in (("r", "reach_rel"), ("e", "pct_interactions_deal_platform")):
        cnt = grp[col].transform("count").to_numpy()
        mean = grp[col].transform("mean").to_numpy()
        ok = cnt >= 2
        out[f"{name}_group_mean"] = np.where(ok, mean, np.nan)
        out[f"{name}_vs_group"] = np.where(ok, df[col].to_numpy() - mean, np.nan)
    rng = (grp["r"].transform("max") - grp["r"].transform("min")).to_numpy()
    out["reach_rel_group_range"] = np.where(grp["r"].transform("count").to_numpy() >= 2, rng, np.nan)
    per_platform = df.dropna(subset=["r"]).groupby(["g", "p"])["r"].mean().unstack("p")
    for plat in PLATFORMS:
        col = per_platform[plat] if plat in per_platform.columns else pd.Series(dtype="float64")
        vals = df["g"].map(col).to_numpy(dtype="float64", na_value=np.nan)
        out[f"group_reach_rel_{plat}"] = np.where(out["content_group_size"] >= 2, vals, np.nan)
    return out
