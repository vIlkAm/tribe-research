"""Outcomes table: synthetic fixtures only (no export, no database)."""

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tribe_research.outcomes import OutcomeParams, build_outcomes, coerce_tables  # noqa: E402
from tribe_research.outcomes import baseline as bl  # noqa: E402
from tribe_research.outcomes import crossplatform as xp  # noqa: E402
from tribe_research.outcomes import ranking as rk  # noqa: E402
from tribe_research.outcomes.snapshots import clean_snapshots  # noqa: E402

UPLOAD = pd.Timestamp("2026-06-01", tz="UTC")
ANCHOR = UPLOAD + pd.Timedelta(hours=12)  # default upload_anchor_hours


def at(age_days: float) -> pd.Timestamp:
    return ANCHOR + pd.Timedelta(days=age_days)


def video(vid, account="a1", views=1000.0, likes=50.0, comments=5.0, shares=None, saves=None,
          upload=UPLOAD, last_age=60.0, **kw):
    row = {"id": vid, "social_account_id": account, "video_link": f"https://x/{vid}",
           "views": views, "likes": likes, "comments": comments, "shares": shares, "saves": saves,
           "engagement_rate": None, "upload_date": upload,
           "last_updated": at(last_age) if upload is not None else ANCHOR + pd.Timedelta(days=last_age),
           "duration_seconds": 30.0, "is_campaign_video": True, "payout_eligible": False,
           "deal_campaign_id": None}
    row.update(kw)
    return row


def snaps_for(vid, ages_views):
    return [{"video_performance_id": vid, "snapshot_at": at(a), "snapshot_date": None, "views": v,
             "likes": None, "comments": None, "shares": None, "saves": None, "source": "test"}
            for a, v in ages_views]


def tables(videos, snaps=(), accounts=None, deals=None, stats=None, members=None):
    accounts = accounts if accounts is not None else [
        {"id": "a1", "deal_id": "d1", "platform": "tiktok", "handle": "h", "user_id": "u1"}]
    deals = deals if deals is not None else [{"id": "d1", "name": "Deal One"}]
    snap_cols = ["video_performance_id", "snapshot_at", "snapshot_date", "views", "likes",
                 "comments", "shares", "saves", "source"]
    raw = {
        "video_performances": pd.DataFrame(videos),
        "video_snapshots": pd.DataFrame(list(snaps), columns=snap_cols),
        "social_accounts": pd.DataFrame(accounts),
        "deals": pd.DataFrame(deals),
        "social_account_stat_snapshots": None if stats is None else pd.DataFrame(stats),
        "cross_platform_members": None if members is None else pd.DataFrame(members),
    }
    return coerce_tables(raw)[0]


def run(videos, snaps=(), params=None, **kw):
    out, meta = build_outcomes(tables(videos, snaps, **kw), params)
    return out.set_index("id", drop=False), meta


# --- 1. reach at fixed ages --------------------------------------------------------

def test_log_linear_interpolation_and_exact():
    out, _ = run([video("v1"), video("v2")],
                 snaps_for("v1", [(0.5, 99), (5, 999), (9, 9999), (40, 20000)])
                 + snaps_for("v2", [(1, 10), (7, 500), (30, 900)]))
    expected = np.expm1(np.log(1000) + (np.log(10000) - np.log(1000)) * (7 - 5) / (9 - 5))
    assert out.loc["v1", "views_7d"] == pytest.approx(expected)
    assert out.loc["v1", "views_7d_basis"] == "interpolated"
    assert out.loc["v1", "views_7d_span_days"] == pytest.approx(4)
    assert out.loc["v2", "views_7d"] == pytest.approx(500)
    assert out.loc["v2", "views_7d_basis"] == "exact"
    assert out.loc["v2", "views_30d"] == pytest.approx(900)
    assert out.loc["v2", "views_1d_basis"] == "exact"


def test_no_extrapolation_beyond_tolerance():
    vids = [video("far", last_age=3.5), video("near", last_age=5.95),
            video("old_gap", last_age=40), video("back")]
    snaps = (snaps_for("far", [(1, 100), (3.5, 400)])
             + snaps_for("near", [(5, 1000), (5.95, 2000)])
             + snaps_for("old_gap", [(1, 100), (3, 300)])
             + snaps_for("back", [(1.1, 100), (2.1, 300)]))
    out, _ = run(vids, snaps)
    assert np.isnan(out.loc["far", "views_7d"]) and out.loc["far", "views_7d_basis"] == "too_young"
    assert out.loc["far", "flag_too_young"]
    slope = (np.log1p(2000) - np.log1p(1000)) / 0.95
    assert out.loc["near", "views_7d_basis"] == "extrapolated"
    assert out.loc["near", "views_7d"] == pytest.approx(np.expm1(np.log1p(2000) + slope * 1.05))
    assert np.isnan(out.loc["old_gap", "views_7d"]) and out.loc["old_gap", "views_7d_basis"] == "no_bracket"
    back = out.loc["back", "views_1d"]
    assert out.loc["back", "views_1d_basis"] == "extrapolated" and 0 <= back <= 100
    strict, _ = run(vids, snaps, params=OutcomeParams(max_extrapolation_frac=0.0))
    assert np.isnan(strict.loc["near", "views_7d"]) and strict.loc["near", "views_7d_basis"] == "too_young"
    assert np.isnan(strict.loc["back", "views_1d"])


# --- snapshot cleaning ---------------------------------------------------------------

def test_monotone_cleaning_dedupe_and_broken_zeros():
    code = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, -1])
    t = np.array([1.0, 2.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 1.0])
    views = np.array([100, 200, 150, 0, 180, 400, 1000, 100, 1200, 5], dtype=float)
    clean, st = clean_snapshots(code, t, views, 2)
    v0 = clean.loc[clean.code == 0, "views"].to_numpy()
    assert list(v0) == [100, 200, 200, 400]  # dup keeps max, zero dropped, dip lifted
    assert st["n_duplicates"][0] == 1 and st["n_broken_zeros"][0] == 1
    assert st["n_decreases"][0] == 1 and st["max_drop_frac"][0] == pytest.approx(0.1)
    assert st["max_drop_frac"][1] == pytest.approx(0.9)
    assert all(np.diff(clean.loc[clean.code == 1, "views"]) >= 0)
    assert st["totals"]["rows_unknown_video"] == 1

    out, _ = run([video("ok"), video("spiky")],
                 snaps_for("ok", [(1, 100), (2, 95), (8, 300)])
                 + snaps_for("spiky", [(1, 100), (2, 5000), (3, 150), (8, 300)]))
    assert not out.loc["ok", "flag_broken_history"]      # small noise is only cleaned
    assert out.loc["spiky", "flag_broken_history"]       # a 97% drop is suspicious


# --- 2. baselines --------------------------------------------------------------------

def test_loo_median_matches_brute_force():
    rng = np.random.default_rng(0)
    group = rng.integers(-1, 6, 300)
    y = rng.normal(size=300)
    y[rng.random(300) < 0.2] = np.nan
    med, n_others = bl.loo_median(group, y, min_others=5)
    for i in range(300):
        if group[i] < 0:
            assert np.isnan(med[i])
            continue
        others = [y[j] for j in range(300) if j != i and group[j] == group[i] and np.isfinite(y[j])]
        assert n_others[i] == len(others)
        if len(others) >= 5:
            assert med[i] == pytest.approx(np.median(others))
        else:
            assert np.isnan(med[i])


def _old_video(vid, account, views):
    return video(vid, account=account, views=views, likes=views * 0.05)


def test_baseline_levels_account_fallback_and_null_target():
    accounts = [{"id": "big", "deal_id": "d1", "platform": "tiktok", "handle": "b", "user_id": "1"},
                {"id": "small", "deal_id": "d1", "platform": "tiktok", "handle": "s", "user_id": "2"},
                {"id": "lonely", "deal_id": "d2", "platform": "youtube", "handle": "l", "user_id": "3"}]
    deals = [{"id": "d1", "name": "A"}, {"id": "d2", "name": "B"}]
    big = [10 ** k for k in (2, 3, 3, 4, 5, 6)]           # 6 videos: 5 others each
    vids = [_old_video(f"b{i}", "big", v) for i, v in enumerate(big)]
    vids += [_old_video(f"s{i}", "small", 500) for i in range(5)]  # 4 others each
    vids += [video("young", account="big", views=50, last_age=2)]  # no reach value yet
    vids += [_old_video("l0", "lonely", 10)]
    out, _ = run(vids, accounts=accounts, deals=deals)
    b0 = out.loc["b0"]
    assert b0["baseline_level"] == "account" and b0["baseline_n"] == 5
    assert b0["reach_basis"] == "views_final"
    assert b0["baseline"] == pytest.approx(np.median(np.log1p(big[1:])))
    assert b0["reach_rel"] == pytest.approx(np.log1p(100) - b0["baseline"])
    s0 = out.loc["s0"]
    assert s0["baseline_level"] == "deal_platform" and s0["baseline_n"] == 10
    assert s0["baseline"] == pytest.approx(np.median(np.log1p(big + [500] * 4)))
    y = out.loc["young"]
    assert y["baseline_level"] == "account" and y["baseline_n"] == 6
    assert y["baseline"] == pytest.approx(np.median(np.log1p(big)))
    assert np.isnan(y["reach_rel"]) and y["flag_too_young"]
    assert out.loc["l0", "baseline_level"] == "none" and np.isnan(out.loc["l0", "reach_rel"])


def test_local_loo_median_matches_brute_force():
    rng = np.random.default_rng(3)
    n = 400
    group = rng.integers(-1, 5, n)
    t = rng.uniform(0, 300, n)          # continuous: no distance ties
    t[rng.random(n) < 0.05] = np.nan
    y = rng.normal(size=n)
    y[rng.random(n) < 0.2] = np.nan
    med, used, gap = bl.local_loo_median(group, t, y, k=10, min_others=5)
    for i in range(n):
        if group[i] < 0 or not np.isfinite(t[i]):
            assert np.isnan(med[i]) and used[i] == 0
            continue
        cand = [j for j in range(n) if j != i and group[j] == group[i]
                and np.isfinite(t[j]) and np.isfinite(y[j])]
        near = sorted(cand, key=lambda j: abs(t[j] - t[i]))[:10]
        assert used[i] == len(near)
        if len(near) >= 5:
            assert med[i] == pytest.approx(np.median(y[near]))
            assert gap[i] == pytest.approx(max(abs(t[j] - t[i]) for j in near))
        else:
            assert np.isnan(med[i])


def test_local_baseline_controls_account_growth():
    vids = [video(f"g{i:02d}", views=float(np.round(1000 * np.exp(0.1 * i))),
                  upload=UPLOAD + pd.Timedelta(days=i), last_age=200) for i in range(40)]
    out, _ = run(vids)
    first, mid, last = out.loc["g00"], out.loc["g20"], out.loc["g39"]
    assert mid["local_baseline_level"] == "account_local" and mid["local_baseline_n"] == 10
    assert mid["local_baseline_max_gap_days"] == pytest.approx(5)
    assert abs(mid["reach_rel_local"]) < 0.01
    assert first["reach_rel"] < -1.5 and last["reach_rel"] > 1.5     # whole-history is confounded
    assert abs(first["reach_rel_local"]) < abs(first["reach_rel"]) / 2
    assert abs(last["reach_rel_local"]) < abs(last["reach_rel"]) / 2
    assert out["reach_rel"].notna().all() and out["reach_rel_local"].notna().all()


def test_local_baseline_compares_same_reach_basis_only():
    # 6 videos with a 7-day value, 6 older ones that fall back to (higher) final views,
    # interleaved in time: each is compared only with its own kind
    vids, snaps = [], []
    for i in range(12):
        up = UPLOAD + pd.Timedelta(days=i)
        if i % 2 == 0:
            vids.append(video(f"w{i}", views=5000.0, upload=up, last_age=200))
            snaps += [{**s, "snapshot_at": s["snapshot_at"] + pd.Timedelta(days=i)}
                      for s in snaps_for(f"w{i}", [(7, 1000)])]
        else:
            vids.append(video(f"f{i}", views=20000.0, upload=up, last_age=200))
    out, _ = run(vids, snaps)
    w, f = out.loc["w4"], out.loc["f5"]
    assert w["reach_basis"] == "views_7d" and f["reach_basis"] == "views_final"
    assert w["local_baseline_level"] == "account_local" and w["local_baseline_n"] == 5
    assert w["reach_rel_local"] == pytest.approx(0, abs=1e-9)
    assert f["reach_rel_local"] == pytest.approx(0, abs=1e-9)


def test_primary_age_is_selectable():
    vids = [video("s", last_age=60), video("old", last_age=60), video("kid", last_age=10)]
    snaps = snaps_for("s", [(10, 1000), (20, 3000)]) + snaps_for("kid", [(1, 10), (9, 90)])
    out, meta = run(vids, snaps, params=OutcomeParams(primary_age=14))
    assert "views_14d" in out.columns and "views_7d" in out.columns
    s = out.loc["s"]
    assert s["views_14d_basis"] == "interpolated" and s["views_age_basis"] == "interpolated"
    assert s["reach_basis"] == "views_14d" and s["views_age_used"] == 14
    assert s["reach_log"] == pytest.approx(np.log1p(1000) + (np.log1p(3000) - np.log1p(1000)) * 0.4)
    assert out.loc["old", "reach_basis"] == "views_final" and out.loc["old", "views_age_used"] == pytest.approx(60)
    kid = out.loc["kid"]
    assert kid["flag_too_young"] and np.isnan(kid["views_age_used"]) and kid["views_age_basis"] == "too_young"
    default, _ = run(vids, snaps)
    assert "views_14d" not in default.columns and default.loc["kid", "views_age_used"] == 7


def test_inconsistent_upload_date_is_kept_out_of_baselines():
    vids = [_old_video(f"b{i}", "a1", 1000 * (i + 1)) for i in range(6)] + [_old_video("bad", "a1", 10 ** 7)]
    snaps = snaps_for("bad", [(-20, 5000), (7, 9000)])
    out, _ = run(vids, snaps)
    bad = out.loc["bad"]
    assert bad["flag_upload_date_inconsistent"] and bad["views_7d_basis"] == "exact"
    assert bad["reach_basis"] == "upload_date_inconsistent" and np.isnan(bad["reach_rel"])
    assert out.loc["b0", "baseline_n"] == 5 and out.loc["b0", "local_baseline_n"] == 5
    assert out.loc["b0", "baseline"] == pytest.approx(np.median(np.log1p([2000, 3000, 4000, 5000, 6000])))


def test_follower_count_nearest_upload_with_tolerance():
    accounts = [{"id": "a1", "deal_id": "d1", "platform": "tiktok", "handle": "h", "user_id": "1"},
                {"id": "a2", "deal_id": "d1", "platform": "tiktok", "handle": "i", "user_id": "2"}]
    stats = [{"social_account_id": "a1", "follower_count": 100, "snapshot_at": ANCHOR - pd.Timedelta(days=10)},
             {"social_account_id": "a1", "follower_count": 200, "snapshot_at": ANCHOR + pd.Timedelta(days=2)},
             {"social_account_id": "a1", "follower_count": 0, "snapshot_at": ANCHOR},  # provider miss
             {"social_account_id": "a2", "follower_count": 999, "snapshot_at": ANCHOR + pd.Timedelta(days=90)}]
    out, _ = run([video("v1", account="a1"), video("v2", account="a2")], accounts=accounts, stats=stats)
    assert out.loc["v1", "follower_count"] == 200
    assert out.loc["v1", "follower_snapshot_gap_days"] == pytest.approx(2)
    assert np.isnan(out.loc["v2", "follower_count"])
    no_stats, _ = run([video("v1")])
    assert np.isnan(no_stats.loc["v1", "follower_count"])


# --- 3. engagement shrinkage ---------------------------------------------------------

def _engagement_fixture():
    rng = np.random.default_rng(1)
    views = np.round(np.exp(rng.uniform(np.log(500), np.log(1e5), 200)))
    rates = rng.beta(5, 95, 200)  # mean 0.05, concentration 100
    likes = rng.binomial(views.astype(int), rates)
    vids = [video(f"v{i}", views=float(v), likes=float(k)) for i, (v, k) in enumerate(zip(views, likes))]
    vids += [video("tiny", views=3, likes=3), video("huge", views=2_000_000, likes=200_000),
             video("zero", views=0, likes=0), video("over", views=10, likes=40)]
    accounts = [{"id": "a1", "deal_id": "d1", "platform": "tiktok", "handle": "h", "user_id": "1"},
                {"id": "a2", "deal_id": "d2", "platform": "tiktok", "handle": "i", "user_id": "2"}]
    vids += [video(f"o{i}", account="a2", views=1000, likes=100) for i in range(10)]
    deals = [{"id": "d1", "name": "A"}, {"id": "d2", "name": "B"}]
    return run(vids, accounts=accounts, deals=deals)


def test_shrinkage_small_samples_to_prior_large_keep_raw():
    out, meta = _engagement_fixture()
    tiny, huge = out.loc["tiny"], out.loc["huge"]
    prior = tiny["like_rate_prior_mean"]
    assert tiny["like_rate_prior_level"] == "deal_platform"
    assert 0.03 < prior < 0.08
    assert tiny["like_rate_raw"] == 1.0
    assert abs(tiny["like_rate_eb"] - prior) < 0.1 * abs(1.0 - prior)
    assert tiny["like_rate_eb_lo90"] < tiny["like_rate_eb"] < tiny["like_rate_eb_hi90"]
    assert huge["like_rate_eb"] == pytest.approx(0.1, rel=0.01)
    assert huge["like_rate_eb_hi90"] - huge["like_rate_eb_lo90"] < 0.002
    assert out.loc["zero", "like_rate_eb"] == pytest.approx(prior) and np.isnan(out.loc["zero", "like_rate_raw"])
    assert out.loc["zero", "flag_views_zero"]
    over = out.loc["over"]
    assert over["like_rate_raw"] == 1.0 and over["flag_rate_gt1"] and "rate_gt1" in over["dq_flags"]
    # a deal x platform group of 10 is too small -> platform prior
    assert out.loc["o0", "like_rate_prior_level"] == "platform"
    dp = [p for p in meta["priors"]["like_rate"] if p["level"] == "deal_platform"]
    assert len(dp) == 1 and 20 < dp[0]["concentration"] < 500
    # shares/saves are missing -> no rate, and interactions use the available parts only
    assert np.isnan(out.loc["v0", "share_rate_eb"]) and out.loc["v0", "share_rate_prior_level"] == "none"
    assert out.loc["v0", "interactions_components"] == "likes+comments"
    assert out.loc["v0", "interactions_rate_raw"] == pytest.approx(
        (out.loc["v0", "likes"] + 5) / out.loc["v0", "views_final"])


# --- 4. ranking ------------------------------------------------------------------------

def test_group_percentiles_and_quadrant():
    g = pd.Series(["d1"] * 10 + ["d2"] * 3 + [None])
    v = np.r_[np.arange(1, 11, dtype=float), [1.0, 2.0, 3.0], [5.0]]
    v[9] = np.nan
    pct, n = rk.group_percentile(v, g, min_n=5)
    assert list(n) == [9] * 10 + [3] * 3 + [0]
    assert pct[0] == pytest.approx(0.5 / 9) and pct[8] == pytest.approx(8.5 / 9)
    assert np.isnan(pct[9]) and np.isnan(pct[10:]).all()
    q = rk.quadrant(np.array([1, 2, 3, 4, 5.0]), np.array([5, 4, 3, 2, 1.0]), pd.Series(["d"] * 5), min_n=5)
    assert list(q) == ["low_reach_high_engagement", "low_reach_high_engagement",
                       "high_reach_high_engagement", "high_reach_low_engagement", "high_reach_low_engagement"]


def test_per_deal_ranking_in_table():
    out, _ = _engagement_fixture()
    d1 = out[out.deal_id == "d1"]
    assert (d1["n_interactions_deal"] == len(d1)).all()
    assert d1["pct_interactions_deal"].between(0, 1).all()
    assert out.loc["huge", "pct_interactions_deal"] > 0.9
    assert out["quadrant"].dropna().isin(list(rk.QUADRANTS.values())).all()
    assert "score" not in " ".join(out.columns)


# --- 5. cross-platform ---------------------------------------------------------------

def test_cross_platform_group_spread():
    ids = pd.Series(["t", "i", "y", "solo"])
    members = pd.DataFrame({"link_id": ["L1", "L1", "L1"], "video_performance_id": ["t", "i", "y"]})
    group, conflicts = xp.content_groups(ids, members)
    assert list(group) == ["link:L1"] * 3 + ["video:solo"] and conflicts == 0
    spread = xp.group_spread(group, pd.Series(["tiktok", "instagram", "youtube", "tiktok"]),
                             np.array([1.0, 0.0, -1.0, 2.0]), np.array([0.9, 0.5, np.nan, 0.1]))
    assert list(spread["content_group_size"]) == [3, 3, 3, 1]
    assert list(spread["content_group_n_platforms"]) == [3, 3, 3, 1]
    assert list(spread["reach_rel_vs_group"][:3]) == pytest.approx([1.0, 0.0, -1.0])
    assert np.isnan(spread["reach_rel_vs_group"][3])
    assert spread["reach_rel_group_range"][0] == pytest.approx(2.0)
    assert spread["pct_interactions_deal_platform_vs_group"][0] == pytest.approx(0.2)
    assert np.isnan(spread["pct_interactions_deal_platform_vs_group"][2])
    assert spread["group_reach_rel_youtube"][0] == pytest.approx(-1.0)
    assert np.isnan(spread["group_reach_rel_tiktok"][3])


# --- 6. nulls, flags and one row per video -----------------------------------------------

def test_null_handling_flags_and_one_row_per_video():
    vids = [video("nodate", upload=None, history_truncated="t", caar_video_gone_since=at(30)),
            video("noviews", views=None, likes=None, comments=None),
            video("ghost", account="missing"), video("fine"), video("fine", views=5000)]
    accounts = [{"id": "a1", "deal_id": "d1", "platform": "TikTok", "handle": "h", "user_id": "1"},
                {"id": "a1", "deal_id": "d1", "platform": "TikTok", "handle": "h", "user_id": "1"}]
    members = [{"link_id": "L1", "video_performance_id": "fine"},
               {"link_id": "L1", "video_performance_id": "fine"},
               {"link_id": "L2", "video_performance_id": "fine"}]
    snaps = snaps_for("nodate", [(1, 10), (8, 20)]) + snaps_for("fine", [(1, 10), (8, 20), (8, 20)])
    out, meta = run(vids, snaps, accounts=accounts, members=members)
    assert len(out) == 4 and out.index.is_unique
    assert meta["cross_platform_membership_conflicts"] == 1
    nd = out.loc["nodate"]
    assert nd["flag_missing_upload_date"] and nd["views_7d_basis"] == "no_upload_date"
    assert np.isnan(nd["age_days_at_last_obs"]) and np.isnan(nd["reach_rel"])
    assert nd["flag_history_truncated"] and nd["flag_video_gone"] and not out.loc["fine", "flag_video_gone"]
    nv = out.loc["noviews"]
    assert nv["flag_views_missing"] and nv["flag_missing_snapshots"]
    assert nv["views_7d_basis"] == "no_snapshots" and pd.isna(nv["interactions_components"])
    assert np.isnan(nv["like_rate_eb"])
    gh = out.loc["ghost"]
    assert gh["flag_unknown_account"] and gh["flag_no_deal"] and pd.isna(gh["platform"])
    fine = out.loc["fine"]
    assert fine["platform"] == "tiktok" and fine["views_final"] == 5000
    assert fine["content_group"] == "link:L1" and fine["n_snapshot_duplicates"] == 1
    assert pd.isna(fine["dq_flags"]) or "broken_history" not in fine["dq_flags"]


def test_all_upload_dates_missing_degrades_to_flags():
    accounts = [{"id": "a1", "deal_id": "d1", "platform": "tiktok", "handle": "h", "user_id": "1"},
                {"id": "a2", "deal_id": "d1", "platform": "youtube", "handle": "i", "user_id": "2"}]
    vids = [video(f"v{i}", account="a1" if i % 2 else "a2", upload=None) for i in range(8)]
    members = [{"link_id": "L1", "video_performance_id": "v0"}, {"link_id": "L1", "video_performance_id": "v1"}]
    out, _ = run(vids, snaps_for("v0", [(1, 10), (8, 20)]), accounts=accounts, members=members)
    assert len(out) == 8 and out.index.is_unique
    assert (out["baseline_level"] == "none").all() and out["reach_rel"].isna().all()
    assert out["flag_missing_upload_date"].all() and (out["views_7d_basis"] == "no_upload_date").all()
    assert out["group_reach_rel_tiktok"].isna().all() and out.loc["v0", "content_group_size"] == 2


def test_cli_reads_csv_export(tmp_path):
    m = tmp_path / "metrics"
    m.mkdir()
    (m / "video_performances.csv").write_text(
        "id,social_account_id,video_link,views,likes,comments,shares,saves,engagement_rate,upload_date,"
        "last_updated,duration_seconds,is_campaign_video,payout_eligible,deal_campaign_id\n"
        "101,7,https://x/1,1000,50,5,,,0.055,2026-06-01,2026-07-20 10:00:00.123+00,31.5,t,f,\n"
        "102,7,https://x/2,,,,,,,,2026-07-20 10:00:00+00,,f,,9\n")
    (m / "video_snapshots.csv").write_text(
        "video_performance_id,snapshot_at,snapshot_date,views,likes,comments,shares,saves,source\n"
        "101,2026-06-02 12:00:00+00,2026-06-02,100,,,,,a\n"
        "101,,2026-06-08,800,,,,,a\n"
        "101,2026-07-01 12:00:00.5+00,2026-07-01,1000,,,,,a\n"
        "999,2026-07-01 12:00:00+00,2026-07-01,5,,,,,a\n")
    (m / "social_accounts.csv").write_text("id,deal_id,platform,handle,user_id\n7,3,instagram,h,\n")
    (m / "deals.csv").write_text("id,name\n3,Deal Three\n9,Deal Nine\n")
    out = tmp_path / "outcomes.parquet"
    res = subprocess.run([sys.executable, str(ROOT / "tools" / "build_outcomes.py"),
                          "--metrics-dir", str(m), "--out", str(out)], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    df = pd.read_parquet(out).set_index("id")
    assert list(df.index) == ["101", "102"]
    assert df.loc["101", "views_1d_basis"] == "exact" and df.loc["101", "views_1d"] == pytest.approx(100)
    assert df.loc["101", "views_7d_basis"] == "exact"   # snapshot_date fallback -> 12:00 UTC
    assert df.loc["101", "is_campaign_video"] == True and df.loc["101", "payout_eligible"] == False  # noqa: E712
    assert df.loc["101", "deal_name"] == "Deal Three" and df.loc["101", "platform"] == "instagram"
    assert df.loc["102", "deal_id"] == "3" and df.loc["102", "deal_id_source"] == "social_account"
    assert df.loc["102", "flag_missing_upload_date"] and pd.isna(df.loc["102", "views_final"])
    meta = json.loads((tmp_path / "outcomes.parquet.meta.json").read_text())
    assert meta["snapshots"]["rows_unknown_video"] == 1
    assert set(meta["load"]["missing_optional_tables"]) == {"social_account_stat_snapshots",
                                                           "cross_platform_members"}
    assert meta["columns"] == list(pd.read_parquet(out).columns)


# --- performance -------------------------------------------------------------------------

def test_performance_1_3m_snapshots():
    rng = np.random.default_rng(2)
    n_v, per = 100_000, 13
    n_s = n_v * per
    vid = np.char.add("v", np.arange(n_v).astype(str))
    acct = rng.integers(0, 5000, n_v)
    upload = pd.Timestamp("2026-01-01", tz="UTC") + pd.to_timedelta(rng.integers(0, 200, n_v), unit="D")
    views = np.round(np.exp(rng.uniform(3, 14, n_v)))
    vp = pd.DataFrame({"id": vid, "social_account_id": np.char.add("a", acct.astype(str)),
                       "video_link": "x", "views": views, "likes": np.round(views * 0.05),
                       "comments": np.round(views * 0.004), "shares": None, "saves": None,
                       "engagement_rate": None, "upload_date": upload,
                       "last_updated": upload + pd.Timedelta(days=60), "duration_seconds": 30.0,
                       "is_campaign_video": True, "payout_eligible": False, "deal_campaign_id": None})
    s_code = np.repeat(np.arange(n_v), per)
    ages = np.sort(rng.uniform(0, 45, (n_v, per)), axis=1).ravel()
    frac = np.clip(ages / 10, 0, 1).reshape(n_v, per)
    s_views = (frac * views[:, None]).ravel() * rng.uniform(0.97, 1.0, n_s)  # noisy, non-monotone
    s_views[rng.random(n_s) < 0.01] = 0
    snaps = pd.DataFrame({"video_performance_id": vid[s_code],
                          "snapshot_at": upload[s_code] + pd.to_timedelta(ages, unit="D"),
                          "snapshot_date": None, "views": s_views, "likes": None, "comments": None,
                          "shares": None, "saves": None, "source": "x"})
    snaps = pd.concat([snaps, snaps.iloc[: n_s // 20]], ignore_index=True)  # duplicates
    accounts = pd.DataFrame({"id": np.char.add("a", np.arange(5000).astype(str)),
                             "deal_id": np.char.add("d", (np.arange(5000) % 50).astype(str)),
                             "platform": np.array(["tiktok", "instagram", "youtube"])[np.arange(5000) % 3],
                             "handle": "h", "user_id": "u"})
    deals = pd.DataFrame({"id": np.char.add("d", np.arange(50).astype(str)), "name": "n"})
    t0 = time.perf_counter()
    tabs, _ = coerce_tables({"video_performances": vp, "video_snapshots": snaps,
                             "social_accounts": accounts, "deals": deals})
    out, meta = build_outcomes(tabs)
    elapsed = time.perf_counter() - t0
    assert len(out) == n_v
    assert meta["snapshots"]["rows_raw"] > 1_300_000
    assert out["views_7d"].notna().mean() > 0.9
    # ~6 s on an idle host; the margin absorbs a heavily shared machine
    assert elapsed < 45, f"outcomes took {elapsed:.1f}s"
