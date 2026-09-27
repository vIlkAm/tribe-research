"""build_library_profile: per-clip library percentiles and learned.json on synthetic clips (no outputs, no labels)."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import build_library_profile as blp  # noqa: E402
from tribe_research.brain import moments_pop as mp  # noqa: E402

KEYS = list(blp.CHANNEL_LABELS)
C = len(KEYS)
STATE = {"train_ids_sha256": "0" * 64}


def raw_curves(rng, T):
    shared = rng.normal(0, 1, T)
    return 0.7 * shared[None, :] + 0.7 * rng.normal(0, 1, (C, T))


def make_clip(rng, vid, T, raw=None, gap=None):
    t = np.arange(T, dtype=float)
    raw = raw_curves(rng, T) if raw is None else raw
    dur = np.ones(T)
    if gap is not None:
        keep = np.ones(T, bool)
        keep[gap[0]:gap[1]] = False
        t, dur, raw = t[keep], dur[keep], raw[:, keep]
    return mp.Clip(vid, t, dur, raw, float(T))


@pytest.fixture(scope="module")
def world():
    rng = np.random.default_rng(3)
    # 240 clips of 20-29 s (length bin 1) + 10 clips of 40 s (bin 2: thin, pools over lengths)
    clips = [make_clip(rng, f"c{i:03d}", int(rng.integers(20, 30))) for i in range(240)]
    clips += [make_clip(rng, f"L{i:02d}", 40) for i in range(10)]
    norms = mp.fit_norms(clips)
    lib = blp.Library([blp.entry_from_clip(c, norms) for c in clips], KEYS)
    return rng, norms, lib


def test_percentile_mid_rank():
    ref = np.array([1.0, 2.0, 2.0, 3.0])
    assert blp.percentile_of(ref, [0.0, 2.0, 5.0]).tolist() == [0.0, 50.0, 100.0]


def test_self_exclusion_drops_every_own_row(world):
    _, _, lib = world
    i = lib.index["c000"]
    e = lib.entries[i]
    sb = int(e.sb[0])
    with_self, _ = lib.cell(e.lb, sb, None)
    without, pooled = lib.cell(e.lb, sb, i)
    assert not pooled and len(with_self) - len(without) == 1
    assert len(without) == len(with_self) - np.isin(with_self, e.r[e.sb == sb]).sum()


def test_library_percentiles_are_roughly_uniform(world):
    _, _, lib = world
    allp = np.concatenate([p[np.isfinite(p)] for p in lib.pct])
    assert abs((allp < 20).mean() - 0.2) < 0.02
    hooks = [blp.profile(e, lib, i, STATE)["scores"][0]["percentile"] for i, e in enumerate(lib.entries[:120])]
    assert 42 < np.mean(hooks) < 58 and 22 < np.std(hooks) < 35


def test_profile_shape_caveat_and_self_excluded(world):
    _, _, lib = world
    prof = blp.profile(lib.entries[5], lib, 5, STATE)
    assert prof["schema_version"] == "nvi.library.v0"
    assert prof["caveat"] == blp.CAVEAT and "did not predict views" in prof["caveat"]
    assert prof["reference"]["self_excluded"] is True and prof["reference"]["n_clips"] == len(lib.entries) - 1
    assert [s["key"] for s in prof["scores"]] == ["hook", "hold", "peak", "dead_zones", "finish"]
    assert [c["label_plain"] for c in prof["channels"]] == [blp.CHANNEL_LABELS[k] for k in KEYS]
    assert len(prof["summary"]) == 3 and all(s.endswith(".") for s in prof["summary"])
    assert len(prof["index"]["values"]) == int(np.ceil(lib.entries[5].duration))
    json.dumps(prof, allow_nan=False)


def test_planted_strong_hook_and_weak_finish(world):
    rng, norms, lib = world
    raw = raw_curves(rng, 25)
    raw[:, :4] += 3.0
    raw[:, -3:] -= 3.0
    prof = blp.profile(blp.entry_from_clip(make_clip(rng, "new", 25, raw), norms), lib, None, STATE)
    s = {x["key"]: x for x in prof["scores"]}
    assert prof["reference"]["self_excluded"] is False and prof["reference"]["n_clips"] == len(lib.entries)
    assert s["hook"]["verdict"] == "strong" and s["hook"]["plain"].startswith("Higher predicted response in the opening than")
    assert "similar-length clips in your library" in s["hook"]["plain"]
    assert s["finish"]["verdict"] == "weak" and s["finish"]["plain"].startswith("Lower predicted response in the ending than")
    assert s["hook"]["window_ms"] == [0, 4000] and s["finish"]["window_ms"] == [22000, 25000]
    assert prof["summary"][0] == s["hook"]["plain"] and prof["summary"][2] == s["finish"]["plain"]
    assert any(m["kind"] == "standout_high" and m["start_ms"] == 0 for m in prof["moments"])


def test_standout_moment_channels_and_dead_zones(world):
    rng, norms, lib = world
    raw = raw_curves(rng, 26)
    raw[:, 10:14] += 2.5
    raw[KEYS.index("social"), 10:14] += 3.0
    raw[:, 16:24] -= 2.5
    raw[KEYS.index("value"), 16:24] -= 3.0
    prof = blp.profile(blp.entry_from_clip(make_clip(rng, "new", 26, raw), norms), lib, None, STATE)
    hi = [m for m in prof["moments"] if m["kind"] == "standout_high"]
    lo = [m for m in prof["moments"] if m["kind"] == "standout_low"]
    assert hi and hi[0]["start_ms"] == 10000 and hi[0]["end_ms"] == 14000 and hi[0]["channels"][0] == "social"
    assert lo and lo[0]["channels"][0] == "value" and "lowest in Reward / value" in lo[0]["plain"]
    assert len(prof["moments"]) <= blp.MAX_MOMENTS
    dz = next(s for s in prof["scores"] if s["key"] == "dead_zones")
    assert dz["value"] >= 8 and dz["verdict"] == "weak" and dz["plain"].startswith("More low-response seconds")


def test_gap_is_null_everywhere(world):
    rng, norms, lib = world
    e = blp.entry_from_clip(make_clip(rng, "gap", 24, gap=(8, 11)), norms)
    prof = blp.profile(e, lib, None, STATE)
    ix = prof["index"]
    for j in range(24):
        is_gap = 8 <= j < 11
        assert (ix["values"][j] is None) == is_gap
        assert (ix["percentile"][j] is None) == is_gap
        assert (ix["band"]["p50"][j] is None) == is_gap
    json.dumps(prof, allow_nan=False)


def test_thin_length_bin_pools_and_says_so(world):
    rng, norms, lib = world
    prof = blp.profile(blp.entry_from_clip(make_clip(rng, "long", 40), norms), lib, None, STATE)
    hook = prof["scores"][0]
    assert hook["plain"] is not None and "similar-length" not in hook["plain"]
    assert all(v is not None for v in prof["index"]["percentile"])


def test_short_clip_has_no_middle(world):
    rng, norms, lib = world
    prof = blp.profile(blp.entry_from_clip(make_clip(rng, "short", 6), norms), lib, None, STATE)
    hold = prof["scores"][1]
    assert hold["value"] is None and hold["verdict"] is None and hold["plain"] is None
    assert "too short for a separate middle" in prof["summary"][1] and len(prof["summary"]) == 3


def test_pc1_equal_mixture(world):
    _, _, lib = world
    pc = blp.pc1(lib)
    assert all(v > 0 for v in pc["loadings"].values()) and pc["cosine_with_equal_weights"] > 0.95


def test_learned_good_vs_bad_train_only(world, tmp_path):
    _, _, lib = world
    ids = [e.video_id for e in lib.entries]
    train = set(ids[:-5])
    oof = tmp_path / "oof.csv"
    rng = np.random.default_rng(0)
    with open(oof, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target", "scheme", "vp_id", "video_id", "y", "w", "stratum", "pred_A_stack"])
        for i, v in enumerate(ids):
            deal = f"d{i % 3}"
            w.writerow(["log_interactions_rate", "content", f"p{i}", v, rng.normal(), 1.0, f"{deal}|tiktok", 0.0])
            w.writerow(["reach_rel_local", "content", f"q{i}", v, 9.0, 1.0, f"{deal}|tiktok", 0.0])
    out = blp.learned(lib, STATE, oof, train)
    g = out["good_vs_bad"]
    assert out["schema_version"] == "nvi.learned.v0" and out["stage1"]["result"] == "no-GO"
    assert g["n_contents"] == len(train) and g["n_label_rows_before_train_filter"] == len(ids)
    assert g["n_top"] == g["n_bottom"] > 0
    assert len(g["index"]["top"]["mean"]) == blp.GVB_HORIZON_S and set(g["channels"]) == set(KEYS)
    lo, hi = g["index"]["top"]["lo"], g["index"]["top"]["hi"]
    assert all(a <= m <= b for a, m, b in zip(lo, g["index"]["top"]["mean"], hi) if m is not None)
    assert g["result_plain"].startswith("Clips with more and with fewer likes") and "pc1_loadings" in g
    assert json.dumps(out, allow_nan=False) == json.dumps(blp.learned(lib, STATE, oof, train), allow_nan=False)
