"""Assemble the one-row-per-video outcomes table from the coerced export tables."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from . import baseline as bl
from . import crossplatform as xp
from . import engagement as eng
from . import ranking as rk
from . import snapshots as sn
from .io import to_days

RATE_COMPONENTS = ("likes", "comments", "shares", "saves")
RATE_NAMES = {"likes": "like_rate", "comments": "comment_rate", "shares": "share_rate",
              "saves": "save_rate", "interactions": "interactions_rate"}
FLAG_COLUMNS = (
    "flag_missing_snapshots", "flag_missing_upload_date", "flag_upload_date_inconsistent",
    "flag_too_young", "flag_views_zero", "flag_views_missing", "flag_broken_history",
    "flag_rate_gt1", "flag_unknown_account", "flag_no_deal", "flag_history_truncated", "flag_video_gone",
)


@dataclass(frozen=True)
class OutcomeParams:
    ages: tuple[float, ...] = (1, 3, 7, 30)
    primary_age: float = 7                 # the reach used for baselines / reach_rel (added to ages)
    upload_anchor_hours: float = 12.0      # upload_date is a date; assume this UTC hour
    max_extrapolation_frac: float = 0.2    # 0 = strict brackets only
    baseline_min_others: int = 5
    local_baseline_k: int = 10             # time-local baseline: nearest-in-upload-time others
    follower_tolerance_days: float = 30.0
    eb_min_views: float = 100.0
    eb_min_videos: int = 30
    eb_min_concentration: float = 1.0
    eb_max_concentration: float = 1e4
    eb_interval: float = 0.90
    rank_min_n: int = 5
    broken_drop_frac: float = 0.5          # a raw drop of >50% below the running max
    final_below_snapshot_frac: float = 0.5  # vp.views < 50% of snapshot max
    inconsistent_upload_days: float = 1.0  # snapshot this many days before upload anchor


def _age_col(a: float) -> str:
    return f"{a:g}d"


def _codes(s: pd.Series) -> np.ndarray:
    codes, _ = pd.factorize(s, use_na_sentinel=True)
    return codes.astype(np.int64)


def build_outcomes(tables: dict[str, pd.DataFrame], params: OutcomeParams | None = None):
    """Return ``(outcomes, meta)``; ``tables`` must come from ``load_tables`` / ``coerce_tables``."""
    p = params or OutcomeParams()
    ages = tuple(sorted(set(float(a) for a in p.ages) | {float(p.primary_age)}))
    prim_col = f"views_{_age_col(p.primary_age)}"
    vp = tables["video_performances"]
    vp = vp[vp["id"].notna()].drop_duplicates("id", keep="last").reset_index(drop=True)
    n = len(vp)
    meta: dict = {"params": asdict(p), "n_videos": n}

    # --- account, platform, deal -------------------------------------------------
    acc = tables["social_accounts"].dropna(subset=["id"]).drop_duplicates("id", keep="last")
    acc = acc.set_index("id")
    deals = tables["deals"].dropna(subset=["id"]).drop_duplicates("id", keep="last").set_index("id")
    known_acct = vp["social_account_id"].isin(acc.index)
    platform = vp["social_account_id"].map(acc["platform"])
    platform = platform.map(lambda v: None if v is None or pd.isna(v) else str(v).strip().lower())
    deal_id = vp["social_account_id"].map(acc["deal_id"])
    deal_src = np.where(deal_id.notna(), "social_account", None).astype(object)
    fallback = deal_id.isna() & vp["deal_campaign_id"].isin(deals.index)
    deal_id = deal_id.where(~fallback, vp["deal_campaign_id"])
    deal_src[fallback.to_numpy()] = "deal_campaign_id"
    deal_name = deal_id.map(deals["name"])
    deal_platform = pd.Series(np.where(deal_id.notna() & platform.notna(),
                                       deal_id.astype(str) + "|" + platform.astype(str), None),
                              dtype=object)

    # --- time anchors ------------------------------------------------------------
    upload_t = to_days(vp["upload_date"]) + p.upload_anchor_hours / 24.0
    # when views_final was observed: provider_observed_at if exported, else last_updated
    observed_t = to_days(vp["provider_observed_at"])
    last_updated_t = np.where(np.isfinite(observed_t), observed_t, to_days(vp["last_updated"]))

    snaps = tables["video_snapshots"]
    code = pd.Index(vp["id"]).get_indexer(snaps["video_performance_id"])
    snap_t = to_days(snaps["snapshot_at"])
    date_t = to_days(snaps["snapshot_date"]) + 0.5
    snap_t = np.where(np.isfinite(snap_t), snap_t, date_t)
    clean, st = sn.clean_snapshots(code, snap_t, snaps["views"].to_numpy(dtype="float64", na_value=np.nan), n)
    meta["snapshots"] = st["totals"] | {"rows_time_from_snapshot_date": int(
        (~np.isfinite(to_days(snaps["snapshot_at"])) & np.isfinite(date_t)).sum())}

    last_obs_t = np.fmax(st["last_snapshot_t"], last_updated_t)
    age_last = last_obs_t - upload_t
    views_final = vp["views"].to_numpy(dtype="float64", na_value=np.nan)
    views_final_src = np.where(np.isfinite(views_final), "video_performances", None).astype(object)
    use_snap = ~np.isfinite(views_final) & np.isfinite(st["last_snapshot_views"])
    views_final = np.where(use_snap, st["last_snapshot_views"], views_final)
    views_final_src[use_snap] = "last_snapshot"

    out = pd.DataFrame({
        "id": vp["id"],
        "social_account_id": vp["social_account_id"],
        "platform": platform,
        "deal_id": deal_id,
        "deal_name": deal_name,
        "deal_id_source": deal_src,
        "deal_campaign_id": vp["deal_campaign_id"],
        "video_link": vp["video_link"],
        "upload_date": vp["upload_date"],
        "duration_seconds": vp["duration_seconds"],
        "is_campaign_video": vp["is_campaign_video"],
        "payout_eligible": vp["payout_eligible"],
        "views_final": views_final,
        "views_final_source": views_final_src,
        "likes": vp["likes"], "comments": vp["comments"], "shares": vp["shares"], "saves": vp["saves"],
        "engagement_rate_reported": vp["engagement_rate"],
        "n_snapshots_raw": st["n_snapshots_raw"],
        "n_snapshots": st["n_snapshots"],
        "n_snapshot_duplicates": st["n_duplicates"],
        "n_snapshot_broken_zeros": st["n_broken_zeros"],
        "n_snapshot_decreases": st["n_decreases"],
        "snapshot_max_drop_frac": st["max_drop_frac"],
        "first_snapshot_age_days": st["first_snapshot_t"] - upload_t,
        "age_days_at_last_obs": age_last,
    })

    # --- 1. reach at fixed ages -----------------------------------------------------
    for a in ages:
        v, basis, span = sn.views_at_age(clean, upload_t, age_last, float(a), p.max_extrapolation_frac)
        out[f"views_{_age_col(a)}"] = v
        out[f"views_{_age_col(a)}_basis"] = basis
        out[f"views_{_age_col(a)}_span_days"] = span
        meta.setdefault("views_age_basis_counts", {})[_age_col(a)] = pd.Series(basis).value_counts().to_dict()

    # --- 2. baseline & relative reach ----------------------------------------------
    y = np.log1p(out[prim_col].to_numpy())
    age_used = np.where(np.isfinite(y), float(p.primary_age), np.nan)
    reach_basis = np.where(np.isfinite(y), prim_col, None).astype(object)
    fb = ~np.isfinite(y) & np.isfinite(views_final) & (age_last >= p.primary_age)
    y = np.where(fb, np.log1p(views_final), y)
    age_used = np.where(fb, age_last, age_used)
    reach_basis[fb] = "views_final"
    # snapshots well before the upload date mean the ages are wrong: keep the
    # views_{A}d values (flagged) but keep these videos out of reach_log and of
    # every other video's baseline pool
    first_age = out["first_snapshot_age_days"].to_numpy()
    inconsistent = np.isfinite(first_age) & (first_age < -p.inconsistent_upload_days)
    y = np.where(inconsistent, np.nan, y)
    age_used = np.where(inconsistent, np.nan, age_used)
    reach_basis[inconsistent] = "upload_date_inconsistent"
    acct_codes = _codes(vp["social_account_id"].where(known_acct))
    med_acc, n_acc = bl.loo_median(acct_codes, y, p.baseline_min_others)
    med_dp, n_dp = bl.loo_median(_codes(deal_platform), y, p.baseline_min_others)
    use_acc = np.isfinite(med_acc)
    use_dp = ~use_acc & np.isfinite(med_dp)
    base = np.where(use_acc, med_acc, np.where(use_dp, med_dp, np.nan))
    out["views_age_basis"] = out[f"{prim_col}_basis"]
    out["reach_log"] = y
    out["reach_basis"] = reach_basis
    out["views_age_used"] = age_used
    out["baseline"] = base
    out["baseline_level"] = np.where(use_acc, "account", np.where(use_dp, "deal_platform", "none"))
    out["baseline_n"] = np.where(use_acc, n_acc, np.where(use_dp, n_dp, 0))
    out["reach_rel"] = y - base
    # same-basis neighbours only: a 7-day value is compared with the account's
    # other 7-day values, a final-views fallback with its other final-views values
    # (lifetime views run higher than 7-day views, so mixing them biases the ratio)
    acct_basis = pd.Series(acct_codes).astype(str).str.cat(pd.Series(reach_basis).astype(str), sep="|")
    acct_basis_codes = np.where(acct_codes >= 0, _codes(acct_basis), -1)
    med_loc, n_loc, gap_loc = bl.local_loo_median(acct_basis_codes, upload_t, y, p.local_baseline_k,
                                                  p.baseline_min_others)
    use_loc = np.isfinite(med_loc)
    use_dp_loc = ~use_loc & np.isfinite(med_dp)
    base_loc = np.where(use_loc, med_loc, np.where(use_dp_loc, med_dp, np.nan))
    out["local_baseline"] = base_loc
    out["local_baseline_level"] = np.where(use_loc, "account_local",
                                           np.where(use_dp_loc, "deal_platform", "none"))
    out["local_baseline_n"] = np.where(use_loc, n_loc, np.where(use_dp_loc, n_dp, 0))
    out["local_baseline_max_gap_days"] = np.where(use_loc, gap_loc, np.nan)
    out["reach_rel_local"] = y - base_loc

    stats = tables["social_account_stat_snapshots"]
    fc, gap = bl.followers_near(upload_t, vp["social_account_id"], stats, to_days(stats["snapshot_at"]),
                                p.follower_tolerance_days)
    out["follower_count"] = fc
    out["follower_snapshot_gap_days"] = gap

    # --- 3. engagement shrinkage ----------------------------------------------------
    counts = {c: vp[c].to_numpy(dtype="float64", na_value=np.nan) for c in RATE_COMPONENTS}
    stack = np.column_stack([counts[c] for c in RATE_COMPONENTS])
    avail = np.isfinite(stack)
    counts["interactions"] = np.where(avail.any(axis=1), np.nansum(stack, axis=1), np.nan)
    comp_names = np.array(RATE_COMPONENTS, dtype=object)
    out["interactions_components"] = ["+".join(comp_names[row]) or None for row in avail]
    priors, any_gt1 = {}, np.zeros(n, dtype=bool)
    for comp, name in RATE_NAMES.items():
        r = eng.shrink(counts[comp], views_final, deal_platform, platform,
                       min_views=p.eb_min_views, min_videos=p.eb_min_videos,
                       min_conc=p.eb_min_concentration, max_conc=p.eb_max_concentration,
                       interval=p.eb_interval)
        out[f"{name}_raw"] = r["raw"]
        out[f"{name}_eb"] = r["eb"]
        out[f"{name}_eb_lo90"] = r["lo"]
        out[f"{name}_eb_hi90"] = r["hi"]
        out[f"{name}_prior_mean"] = r["prior_mean"]
        out[f"{name}_prior_level"] = r["prior_level"]
        any_gt1 |= r["gt1"]
        priors[name] = r["priors"]
    meta["priors"] = priors

    # --- 4. per-deal ranking --------------------------------------------------------
    deal_key = deal_id.astype(object).where(deal_id.notna(), None)
    for suffix, key in (("deal", deal_key), ("deal_platform", deal_platform)):
        pr, nr = rk.group_percentile(out["reach_rel"].to_numpy(), key, p.rank_min_n)
        pe, ne = rk.group_percentile(out["interactions_rate_eb"].to_numpy(), key, p.rank_min_n)
        out[f"pct_reach_rel_{suffix}"] = pr
        out[f"n_reach_rel_{suffix}"] = nr
        out[f"pct_interactions_{suffix}"] = pe
        out[f"n_interactions_{suffix}"] = ne
    out["quadrant"] = rk.quadrant(out["reach_rel"].to_numpy(), out["interactions_rate_eb"].to_numpy(),
                                  deal_key, p.rank_min_n)
    out["quadrant_deal_platform"] = rk.quadrant(out["reach_rel"].to_numpy(),
                                                out["interactions_rate_eb"].to_numpy(),
                                                deal_platform, p.rank_min_n)

    # --- 5. cross-platform ---------------------------------------------------------
    group, conflicts = xp.content_groups(vp["id"], tables["cross_platform_members"])
    out["content_group"] = group
    for col, vals in xp.group_spread(group, platform, out["reach_rel"].to_numpy(),
                                     out["pct_interactions_deal_platform"].to_numpy()).items():
        out[col] = vals
    meta["cross_platform_membership_conflicts"] = conflicts

    # --- 6. data-quality flags ------------------------------------------------------
    vf = out["views_final"].to_numpy()
    out["flag_missing_snapshots"] = st["n_snapshots"] == 0
    out["flag_missing_upload_date"] = ~np.isfinite(upload_t)
    out["flag_upload_date_inconsistent"] = inconsistent
    out["flag_too_young"] = np.isfinite(age_last) & (age_last < p.primary_age)  # primary age
    out["flag_views_zero"] = np.isfinite(vf) & (vf == 0)
    out["flag_views_missing"] = ~np.isfinite(vf)
    with np.errstate(invalid="ignore"):
        final_low = (vp["views"].to_numpy(dtype="float64", na_value=np.nan)
                     < p.final_below_snapshot_frac * st["last_snapshot_views"])
    out["flag_broken_history"] = ((st["n_broken_zeros"] > 0)
                                  | (st["max_drop_frac"] > p.broken_drop_frac) | final_low)
    out["flag_rate_gt1"] = any_gt1
    out["flag_unknown_account"] = ~known_acct.to_numpy()
    out["flag_no_deal"] = deal_id.isna().to_numpy()
    out["flag_history_truncated"] = vp["history_truncated"].fillna(False).to_numpy(dtype=bool)
    out["flag_video_gone"] = vp["caar_video_gone_since"].notna().to_numpy()
    dq = np.full(n, "", dtype=object)
    for f in FLAG_COLUMNS:
        dq = dq + np.where(out[f].to_numpy(), f.removeprefix("flag_") + ";", "")
    out["dq_flags"] = [s.rstrip(";") or None for s in dq]
    meta["flag_counts"] = {f: int(out[f].sum()) for f in FLAG_COLUMNS}

    if len(out) != vp["id"].nunique() or not out["id"].is_unique:
        raise AssertionError("outcomes must have exactly one row per video_performance id")
    return out, meta
