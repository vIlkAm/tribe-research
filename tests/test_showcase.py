"""language_screen, select_showcase (spike/flat rules) and stage_demo_local.plan, on synthetic inputs only."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import language_screen as ls  # noqa: E402
import select_showcase as ss  # noqa: E402
import stage_demo_local as sdl  # noqa: E402


def test_language_tiers():
    assert ls.tier("You're") is None and ls.tier("assume") is None and ls.tier("hello") is None
    assert ls.tier("f***") == "strict" and ls.tier("Ass.") == "strict"
    assert ls.tier("Freaking") == "mild" and ls.tier("Hell!") == "mild"
    s = ls.screen([{"text": "great"}, {"text": "damn"}])
    assert s == {"n_strict": 0, "n_mild": 1, "n_words": 2, "clean": False}
    assert ls.screen([])["clean"]


def test_runs():
    assert ss.runs(np.array([0, 1, 1, 0, 1], bool)) == [(1, 2), (4, 4)]


def test_spike_needs_low_before_and_early_start():
    p = np.array([40, 40, 40, 40, 30, 95, 96, 97, 50, 50, 50], float)
    c = ss.spike_candidates(p, 11.0, cuts=[5.2], words=[])
    assert len(c) == 1 and c[0]["run_s"] == [5, 8] and c[0]["cut_or_speech_at_rise"]
    assert c[0]["window_ms"] == [3000, 9000]
    high_before = p.copy(); high_before[2:5] = 80
    assert ss.spike_candidates(high_before, 11.0, [], []) == []
    late = np.full(30, 40.0); late[20:23] = 95
    assert ss.spike_candidates(late, 30.0, [], []) == []


def test_flat_trims_edges_and_counts_cuts():
    p = np.full(20, 5.0)
    c = ss.flat_candidates(p, 20.0, cuts=[3.5, 19.5])
    assert len(c) == 1 and c[0]["run_s"] == [2, 18] and c[0]["n_cuts"] == 1
    assert c[0]["window_ms"] == [2000, 10000]
    short = np.full(20, 50.0); short[5:9] = 5
    assert ss.flat_candidates(short, 20.0, []) == []


def test_rank_one_per_clip_and_order():
    mk = lambda vid, role, **kw: {"video_id": vid, "role": role, "rank_key": vid, **kw}
    cands = [mk("a", "spike", rise=40.0, cut_or_speech_at_rise=True), mk("b", "spike", rise=55.0,
             cut_or_speech_at_rise=False), mk("b", "spike", rise=50.0, cut_or_speech_at_rise=True),
             mk("c", "flat", n_cuts=1, length_s=9, run_mean_pct=5.0), mk("d", "flat", n_cuts=0, length_s=5,
             run_mean_pct=9.0)]
    r = ss.rank(cands)
    assert [c["video_id"] for c in r["spike"]] == ["b", "a"]
    assert [c["video_id"] for c in r["flat"]] == ["d", "c"]


def _pick(vid, **kw):
    return {"video_id": vid, "batch": "b", "source_path": f"x/{vid}.mp4", "source_file": f"/s/{vid}.mp4", **kw}


def _tier_pick(vid, x, pct, deal="Deal"):
    return {"video_id": vid, "batch": "b", "source_path": "p", "source_file": "f", "deal_label": deal,
            "platform": "tiktok", "views_pct": pct, "n_ref": 1234,
            "observed": {"views_vs_account_usual_x": x, "video_link": "https://example.invalid"}}


def test_plan_tier_order_and_duplicates():
    demo = {"picks": {"bad": [_tier_pick("b1", 0.3, 10)], "great": [_tier_pick("g1", 2.4, 85)],
                      "typical": [_tier_pick("t1", 0.99, 49)]}}
    plan = sdl.plan(demo)
    assert [c["role"] for c in plan] == ["great", "typical", "bad"]
    assert plan[0]["demo_moment"] is None
    demo["picks"]["bad"].append(_tier_pick("g1", 0.3, 10))
    with pytest.raises(SystemExit, match="two tiers"):
        sdl.plan(demo)


def test_tier_plain_wording():
    assert sdl.tier_plain({**_tier_pick("g", 2.4, 85.2, "Enhanced Games")}) == (
        "More views than this account's recent usual (2.4×): better than 85% of Enhanced Games' TikTok posts "
        "(1,234 posts).")
    assert sdl.tier_plain(_tier_pick("g", 105.6, 97)).startswith("More views than this account's recent usual (106×)")
    assert sdl.tier_plain(_tier_pick("b", 0.34, 16, "Stan")).startswith("Fewer views") and "Stan's TikTok" in \
        sdl.tier_plain(_tier_pick("b", 0.34, 16, "Stan"))
