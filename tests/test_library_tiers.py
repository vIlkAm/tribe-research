"""library_tiers: post tiers, weighted coin flip, BH, patterns and demo picks on synthetic inputs only."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import library_tiers as lt  # noqa: E402


def _posts(n=200, deal="d1", platform="tiktok", dq_every=0):
    x = np.linspace(-2, 2, n)
    dq = [("stale" if dq_every and i % dq_every == 0 else None) for i in range(n)]
    return pd.DataFrame({"id": [f"{deal}-{platform}-{i}" for i in range(n)], "deal_id": deal, "platform": platform,
                         "reach_rel_local": x, "dq_flags": pd.Series(dq, dtype=object),
                         "reach_log": np.log1p(np.linspace(10, 10000, n)), "reach_basis": "views_7d"})


def test_post_tiers_thresholds_and_sign_rule():
    t = lt.post_tiers(_posts())
    assert (t["n_ref"] == 200).all()
    assert set(t["tier"].dropna()) == {"great", "typical", "bad"}
    g, b = t[t["tier"] == "great"], t[t["tier"] == "bad"]
    assert (g["views_pct"] >= 80).all() and (g["reach_rel_local"] > 0).all()
    assert (b["views_pct"] <= 20).all() and (b["reach_rel_local"] < 0).all()
    assert t[t["tier"] == "typical"]["views_pct"].between(40, 60).all()
    # an all-positive stratum: the bottom 20% are "below the others" but above usual, so never "bad"
    p = _posts()
    p["reach_rel_local"] += 5
    assert "bad" not in set(lt.post_tiers(p)["tier"].dropna())


def test_post_tiers_min_ref_and_dq():
    small = lt.post_tiers(_posts(n=lt.MIN_REF - 1, deal="small"))
    assert small["tier"].isna().all()
    t = lt.post_tiers(_posts(dq_every=4))
    assert len(t) == 150 and (t["n_ref"] == 150).all()


def test_post_tiers_absolute_and_both():
    p = _posts()
    # absolute views run the opposite way from views-vs-usual: a small account's jump is not "great" under "both"
    p["reach_log"] = np.log1p(np.linspace(10000, 10, 200))
    a = lt.post_tiers(p, "absolute")
    assert (a[a["tier"] == "great"]["abs_pct"] >= 80).all() and (a[a["tier"] == "bad"]["abs_pct"] <= 20).all()
    assert (a[a["tier"] == "great"]["reach_rel_local"] < 0).all()
    b = lt.post_tiers(p, "both")
    assert set(b["tier"].dropna()) <= {"typical"}
    # posts without 7-day views never get an absolute or combined tier, and do not count in the absolute reference
    p = _posts()
    p.loc[:49, "reach_basis"] = "views_final"
    b = lt.post_tiers(p, "both")
    assert b.loc[p.loc[:49, "id"], "tier"].isna().all() and b["n_abs"].max() == 150
    assert lt.post_tiers(p, "relative").loc[p.loc[:49, "id"], "tier"].notna().any()
    with pytest.raises(SystemExit, match="tier basis"):
        lt.post_tiers(p, "views")


def test_wauc_coin_flip_and_weights():
    a = np.array([1.0, 2.0, 3.0])
    assert lt.wauc(a, np.ones(3), a, np.ones(3)) == pytest.approx(0.5)
    assert lt.wauc(a + 10, np.ones(3), a, np.ones(3)) == 1.0
    assert lt.wauc(np.array([0.0, 5.0]), np.array([1.0, 3.0]), np.array([1.0]), np.ones(1)) == pytest.approx(0.75)
    assert np.isnan(lt.wauc(np.array([np.nan]), np.ones(1), a, np.ones(3)))


def test_bh_monotone_and_capped():
    q = lt.bh(np.array([0.01, 0.04, 0.03, 0.9]))
    assert q.tolist() == pytest.approx([0.04, 0.16 / 3, 0.16 / 3, 0.9])
    assert (lt.bh(np.array([0.9, 0.95])) <= 1).all()


def _tab(rng, n=240, signal=True):
    tier = np.array(["great", "typical", "bad"] * (n // 3))
    tab = pd.DataFrame({"tier": tier, "stratum": np.where(np.arange(n) % 2, "s1", "s2"), "w": 1.0})
    for key, *_ in lt.FEATURES:
        tab[key] = rng.normal(0, 1, n)
    if signal:
        tab["brain_above_typical"] = np.where(tier == "great", 70.0, np.where(tier == "bad", 30.0, 50.0)) + \
            rng.normal(0, 5, n)
    return tab


def test_patterns_find_planted_signal_only():
    rows = {r["key"]: r for r in lt.patterns(_tab(np.random.default_rng(1)), np.random.default_rng(2), n_boot=300)}
    r = rows["brain_above_typical"]
    assert r["verdict"] == "weak_tendency" and r["diff_great_minus_bad"]["lo"] > 30
    assert r["coin_flip"]["point"] > 0.95 and "higher" in r["plain"] and "exploratory" in r["plain"]
    assert sum(v["verdict"] == "weak_tendency" for v in rows.values()) <= 2
    line = lt.interpreter_line(r)
    assert "70%" in line and "30%" in line and "50%" in line and "coin flip" in line


def test_patterns_null_gives_no_verdict():
    rows = lt.patterns(_tab(np.random.default_rng(3), signal=False), np.random.default_rng(4), n_boot=300)
    assert all(0 <= r["q"] <= 1 for r in rows)
    assert sum(r["verdict"] == "weak_tendency" for r in rows) <= 1


def _elig(vid, deal, key, dur=20.0):
    return {"video_id": vid, "source_name": f"p{vid}", "deal_id": deal, "duration_s": dur, "rank_key": key,
            "source_path": f"{vid}.mp4"}


def test_demo_picks_rules(tmp_path, monkeypatch):
    monkeypatch.setattr(lt.sep, "source_file", lambda staging, sp: tmp_path / sp)
    elig = [_elig("a", "d1", "1"), _elig("b", "d1", "2"), _elig("c", "d2", "3"), _elig("d", "d3", "4", dur=90),
            _elig("e", "d4", "5"), _elig("f", "d5", "6"), _elig("g", "d6", "0")]
    tiers = pd.DataFrame({"tier": ["great"] * 6 + ["bad"], "views_pct": [90.0] * 6 + [5.0], "n_ref": 500,
                          "reach_rel_local": [1.0] * 6 + [-1.0], "abs_pct": 60.0, "n_abs": 400, "views_7d_abs": 1000.0},
                         index=[f"p{v}" for v in "abcdefg"])
    lang = {v: {"clean": v != "e"} for v in "abcdefg"}
    picks, counts = lt.demo_picks(elig, tiers, lang, tmp_path, lockbox=set(), veto={"c"})
    assert [p["video_id"] for p in picks["great"]] == ["a", "f"]  # b same deal as a, c vetoed, d long, e language
    assert [p["video_id"] for p in picks["bad"]] == ["g"]
    assert counts == {"vetoed": 1, "too_long": 1, "language": 1}
    with pytest.raises(SystemExit, match="lockbox"):
        lt.demo_picks(elig, tiers, lang, tmp_path, lockbox={"a"}, veto=set())
    tiers.loc["pg", "reach_rel_local"] = 0.5
    with pytest.raises(SystemExit, match="tier bad"):
        lt.demo_picks(elig, tiers, lang, tmp_path, lockbox=set(), veto=set())


def test_owner_picks_keep_every_check(tmp_path, monkeypatch):
    monkeypatch.setattr(lt.sep, "source_file", lambda staging, sp: tmp_path / sp)
    elig = [_elig("a", "d1", "1"), _elig("b", "d2", "2"), _elig("c", "d3", "3")]
    tiers = pd.DataFrame({"tier": ["great", "bad", "typical"], "views_pct": [90.0, 5.0, 50.0], "n_ref": 500,
                          "reach_rel_local": [1.0, -1.0, 0.0], "abs_pct": [70.0, 10.0, 50.0], "n_abs": 400,
                          "views_7d_abs": [900.0, 20.0, 300.0]}, index=["pa", "pb", "pc"])
    lang = {v: {"clean": v != "c"} for v in "abc"}
    picks, counts = lt.owner_picks(["great=a", "bad=b"], elig, tiers, lang, tmp_path, lockbox=set())
    assert [p["video_id"] for p in picks["great"]] == ["a"] and picks["bad"][0]["selection"] == "owner"
    assert counts == {"owner_picks": 2}
    for spec, lock, msg in ((["bad=a"], set(), "computed tier is great"), (["typical=c"], set(), "language"),
                            (["great=a"], {"a"}, "lockbox"), (["great=zz"], set(), "not an eligible"),
                            (["best=a"], set(), "tier must be")):
        with pytest.raises(SystemExit, match=msg):
            lt.owner_picks(spec, elig, tiers, lang, tmp_path, lockbox=lock)
