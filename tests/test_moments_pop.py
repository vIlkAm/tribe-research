"""moments_pop (prereg family 6, M1-M3): synthetic curves with known structure, no GPU, no outcomes."""

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

import build_moments_pop  # noqa: E402
from tribe_research.brain import moments_pop as mp  # noqa: E402
from tribe_research.brain.proxies import ProxySpec  # noqa: E402

KEYS = [c.key for c in ProxySpec.load().channels]
C = len(KEYS)
SENS = KEYS.index("sensory")


def ar1(rng, n, phi=0.7):
    e = rng.normal(0, np.sqrt(1 - phi ** 2), n)
    y = np.empty(n)
    y[0] = rng.normal()
    for i in range(1, n):
        y[i] = phi * y[i - 1] + e[i]
    return y


def make_clip(rng, vid, T=None, cut_lag=None, cut_amp=1.5, drop_at=None, hook=0.0, gap=None):
    """Raw channel curves: shared onset transient + AR(1) noise (+ planted structure)."""
    T = T or int(rng.integers(20, 50))
    t = np.arange(T, dtype=float)
    raw = np.stack([ar1(rng, T) for _ in range(C)]) + 2.0 * np.exp(-t / 2.0)  # every clip's onset transient
    raw[:, t < mp.HOOK_S] += hook
    shots = sorted(rng.choice(np.arange(3, T - 3), size=max(1, T // 8), replace=False).astype(float).tolist())
    if cut_lag is not None:
        for s in shots:
            i = int(s + cut_lag)
            if 0 <= i < T:
                raw[SENS, i] += cut_amp
    if drop_at is not None:
        raw[KEYS.index("attention"), drop_at:drop_at + 8] -= 4.0
    dur = np.ones(T)
    if gap is not None:
        keep = np.ones(T, bool)
        keep[gap[0]:gap[1]] = False
        t, dur, raw = t[keep], dur[keep], raw[:, keep]
    return mp.Clip(vid, t, dur, raw, float(T), shots, [1.0])


@pytest.fixture(scope="module")
def trained():
    rng = np.random.default_rng(1)
    clips = [make_clip(rng, f"c{i}", cut_lag=2) for i in range(200)]
    return clips, mp.fit(clips, KEYS, n_surr=10, n_boot=100)


def test_count_events_matches_event_runs():
    rng = np.random.default_rng(0)
    for _ in range(200):
        y = rng.normal(0, 1.5, int(rng.integers(3, 40)))
        t = np.arange(len(y), dtype=float)
        d = np.ones(len(y))
        want = [len(mp.event_runs(t, d, y, g, 1)) + len(mp.event_runs(t, d, y, g, -1)) for g in mp.THR_GRID]
        assert np.array_equal(mp.count_events(y), want)


def test_events_never_bridge_a_gap():
    t = np.array([0, 1, 2, 3, 6, 7, 8, 9], float)
    d = np.ones(8)
    assert mp.event_runs(t, d, np.full(8, 3.0), 2.0, +1) == []           # 4 s + 4 s, neither reaches 5 s
    assert mp.event_runs(np.arange(8.0), d, np.full(8, 3.0), 2.0, +1) == [(0.0, 8.0)]


def test_norms_remove_the_shared_onset_transient(trained):
    clips, state = trained
    hooks = [mp.clip_features(c, state)[f"mpop_hook_{KEYS[0]}"] for c in clips[:50]]
    assert abs(np.mean(hooks)) < 0.15                                    # within-clip z would not remove it
    rng = np.random.default_rng(7)
    boosted = mp.clip_features(make_clip(rng, "h", hook=3.0), state)
    assert np.mean([boosted[f"mpop_hook_{k}"] for k in KEYS]) > 2.0


def test_noise_gives_at_most_the_calibrated_false_event_rate(trained):
    _, state = trained
    rng = np.random.default_rng(99)
    events = []
    for i in range(300):
        c = make_clip(rng, f"n{i}", cut_lag=2)
        events.append(mp.clip_features(c, state)["mpop_m2_events_per_min_all"] * c.duration / 60)
    # primary budget: 0.5 surrogate events per clip per channel; real AR noise vs its own surrogates
    assert np.mean(events) <= 1.5 * mp.FALSE_EVENTS_PER_CLIP * C


def test_summed_budget_is_stricter_and_mostly_silent_on_noise(trained):
    clips, state = trained
    summed = mp.fit(clips, KEYS, n_surr=10, n_boot=20, budget_mode="summed")
    assert summed["budget_mode"] == "summed" and all(a > b for a, b in zip(summed["m2"]["thr"], state["m2"]["thr"]))
    rng = np.random.default_rng(98)
    events = []
    for i in range(300):
        c = make_clip(rng, f"s{i}", cut_lag=2)
        events.append(mp.clip_features(c, summed)["mpop_m2_events_per_min_all"] * c.duration / 60)
    # the planted cut responses are real structure, so allow some excess over the surrogate budget
    assert np.mean(events) <= 2 * mp.FALSE_EVENTS_PER_CLIP and np.mean(np.array(events) == 0) > 0.4
    with pytest.raises(ValueError):
        mp.calibrate([(clips[0].t, clips[0].dur, clips[0].raw)], 1, budget_mode="total")


def test_planted_drop_is_found_at_the_right_time(trained):
    _, state = trained
    rng = np.random.default_rng(5)
    c = make_clip(rng, "d", T=40, drop_at=20)
    row = mp.clip_features(c, state)
    assert row["mpop_m2_drop_per_min_attention"] > 0 and row["mpop_m2_first_drop_frac_attention"] <= 22 / 40
    k = KEYS.index("attention")
    drops = mp.event_runs(c.t, c.dur, mp.normed(c, state["norms"])[k], state["m2"]["thr"][k], -1)
    assert any(abs(s - 20) <= 3 and e >= 26 for s, e in drops)          # the planted 20-28 s drop


def test_positive_control_recovers_the_cut_lag(trained):
    _, state = trained
    ctl = state["m3"]["control"]
    assert ctl["passed"] and ctl["peak_lag_s"] == 2 and ctl["peak_boot_lo"] > 0
    rng = np.random.default_rng(3)
    late = [make_clip(rng, f"l{i}", cut_lag=7) for i in range(120)]
    none = [make_clip(rng, f"z{i}") for i in range(120)]
    for clips in (late, none):
        us = [mp.normed(c, state["norms"]) for c in clips]
        assert not mp.positive_control(clips, us, KEYS, n_boot=50)["passed"]


def test_m3_residual_removes_the_locked_response_and_keeps_the_level(trained):
    clips, state = trained
    c = clips[0]
    u = mp.normed(c, state["norms"])
    r = mp.residual(c, u, state["m3"]["kernels"])
    assert np.allclose(np.nanmean(r, 1), np.nanmean(u, 1))              # demeaned design: level untouched
    idx = [int(s + 2) for s in c.shots_s if s + 2 < len(c.t)]
    assert np.mean(r[SENS, idx]) < np.mean(u[SENS, idx]) - 0.5
    row = mp.clip_features(c, state)
    assert any(k.startswith("mpop_m3_") for k in row)


def test_m3_columns_absent_when_control_fails():
    rng = np.random.default_rng(4)
    clips = [make_clip(rng, f"q{i}") for i in range(80)]
    state = mp.fit(clips, KEYS, n_surr=3, n_boot=30)
    assert not state["m3"]["control"]["passed"]
    assert not any(k.startswith("mpop_m3_") for k in mp.clip_features(clips[0], state))


def test_gappy_clip_surrogates_and_features_run():
    rng = np.random.default_rng(6)
    clips = [make_clip(rng, f"g{i}", gap=(10, 13) if i % 3 == 0 else None, cut_lag=2) for i in range(60)]
    state = mp.fit(clips, KEYS, n_surr=3, n_boot=20)
    row = mp.clip_features(clips[0], state)
    assert np.isfinite(row["mpop_m2_events_per_min_all"])


def test_save_load_roundtrip(trained, tmp_path):
    clips, state = trained
    mp.save(state, tmp_path / "s")
    back = mp.load(tmp_path / "s")
    assert back["train_ids_sha256"] == state["train_ids_sha256"] and back["m2"]["thr"] == state["m2"]["thr"]
    assert mp.clip_features(clips[3], back) == pytest.approx(mp.clip_features(clips[3], state), nan_ok=True)


def test_contrast_finds_a_planted_window_and_stays_quiet_on_null():
    rng = np.random.default_rng(8)
    n, B = 300, 30
    strata = np.repeat(np.arange(10), n // 10).astype(str)
    y = rng.normal(size=n) + (strata.astype(int) * 3.0)                 # stratum level must not matter
    X = rng.normal(size=(n, B))
    ranks = np.argsort(np.argsort(y - strata.astype(int) * 3.0))
    X[:, 8:14] += 0.9 * (ranks[:, None] / n - 0.5) * 2                  # better clips higher over bins 8-13
    X[rng.random((n, B)) < 0.05] = np.nan
    res = mp.contrast(X, y, strata, n_perm=300)
    hit = [c for c in res["clusters"] if c["p_fwe"] < 0.05]
    assert hit and hit[0]["direction"] == "top_higher" and 7 <= hit[0]["start_bin"] <= 10 and hit[0]["end_bin"] >= 12
    null = mp.contrast(rng.normal(size=(n, B)), y, strata, n_perm=300)
    assert not [c for c in null["clusters"] if c["p_fwe"] < 0.05]


def test_m2_power_is_measured_and_grows_with_amplitude(trained):
    _, state = trained
    r = state["m2"]["power"]["detect_rate_by_amp_sd"]
    rates = [r[str(a)] for a in mp.POWER_AMPS_SD]
    assert all(0 <= x <= 1 for x in rates) and rates == sorted(rates) and rates[-1] > rates[0]


def test_contrast_labels_collapse_to_one_row_per_content():
    import pandas as pd

    lab = pd.DataFrame({"video_id": ["a", "a", "b"], "y": [1.0, 3.0, 0.0], "stratum": ["d1", "d1", "d2"],
                        "weight": [2.0, 4.0, 1.0]})
    out = build_moments_pop.one_row_per_content(lab)
    assert out.set_index("video_id")["y"].to_dict() == pytest.approx({"a": 14 / 6, "b": 0.0}) and len(out) == 2
    with pytest.raises(SystemExit, match="span several strata"):
        build_moments_pop.one_row_per_content(lab.assign(stratum=["d1:tiktok", "d1:instagram", "d2"]))


def test_labels_from_oof_follow_the_prereg(tmp_path):
    import pandas as pd

    o = pd.DataFrame({
        "target": ["log_interactions_rate"] * 4 + ["reach_rel_local"],
        "scheme": ["content", "content", "account", "content", "content"],
        "vp_id": list("pqrst"), "video_id": ["a", "a", "a", "b", "a"],
        "y": [2.0, 4.0, 9.0, 1.0, 5.0], "w": [1.0, 3.0, 1.0, 2.0, 1.0],
        "stratum": ["d1|tiktok", "d1|instagram", "d1|tiktok", None, "d1|tiktok"],
        "pred_A_stack": [1.0, 1.0, 0.0, 0.0, 0.0]})
    o.to_csv(tmp_path / "oof.csv", index=False)
    lab = build_moments_pop.one_row_per_content(build_moments_pop.labels_from_oof(tmp_path / "oof.csv"))
    # content scheme + primary target only, lockbox (no stratum) dropped, stratum = deal, w-weighted residual
    assert lab.to_dict("records") == [{"video_id": "a", "y": pytest.approx((1 * 1 + 3 * 3) / 4), "stratum": "d1",
                                       "weight": 2.0}]


def test_bh_and_split_half_replication():
    # sorted p .01 .03 .04 .5 -> raw .04 .06 .0533 .5 -> step-up min .04 .0533 .0533 .5
    assert mp.bh([0.01, 0.04, 0.03, 0.5]) == pytest.approx([0.04, 0.16 / 3, 0.16 / 3, 0.5])
    rng = np.random.default_rng(12)
    n = 400
    strata = np.array([f"d{i % 8}" for i in range(n)])
    y = rng.normal(size=n)
    X = rng.normal(size=(n, 10))
    X[:, 2:5] += y[:, None]
    assert mp.replicates(X, y, strata, None, 2, 5)["replicated"]
    h = mp.split_half_deals(strata)
    assert set(np.unique(h)) == {0, 1} and all(len(set(h[strata == d])) == 1 for d in np.unique(strata))


def test_tertiles_are_within_stratum():
    y = np.array([1, 2, 3, 100, 200, 300], float)
    lab = mp.tertile_labels(y, np.array(list("aaabbb")))
    assert lab.tolist() == [-1, 0, 1, -1, 0, 1]


# ── CLI ──────────────────────────────────────────────────────────────────


def _write_outputs(root: Path, roi: Path, rng, n: int, precision="bf16"):
    from test_features_models import channel_vertices, write_roi

    groups, v = write_roi(roi)
    d = root / "worker-0"
    d.mkdir(parents=True, exist_ok=True)
    ids = []
    for i in range(n):
        c = make_clip(rng, f"{precision}{i:03d}", cut_lag=2)
        p = rng.normal(0, 0.01, (len(c.t), v)).astype(np.float32)
        for k, key in enumerate(KEYS):
            vs = channel_vertices(groups, key)
            p[:, vs] += c.raw[k][:, None] / 1.0
        vid = c.video_id
        np.savez(d / f"{vid}.npz", preds=p.astype(np.float16), seg_start=c.t, seg_duration=c.dur)
        (d / f"{vid}.json").write_text(json.dumps({
            "video_id": vid, "duration_s": c.duration, "words": [{"start": 1.0, "duration": 0.3, "text": "w"}],
            "video_config": {"video_precision": precision}, "runtime": {"dry_run": False}}))
        ids.append((vid, c.shots_s))
    return ids


def test_cli_fit_features_contrast(tmp_path):
    rng = np.random.default_rng(11)
    roi, out = tmp_path / "roi.npz", tmp_path / "outputs"
    good = _write_outputs(out, roi, rng, 60)
    stock = _write_outputs(out, roi, rng, 5, precision="fp32")
    shots = tmp_path / "shots"
    shots.mkdir()
    for vid, s in good + stock:
        (shots / f"{vid}.json").write_text(json.dumps({"shots_ms": [int(x * 1000) for x in s]}))
    sel = tmp_path / "selection.csv"
    with open(sel, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "split"])
        for i, (vid, _) in enumerate(good + stock):
            w.writerow([vid, "lockbox" if i < 10 else "train"])
    stem = tmp_path / "state"
    base = ["--roi-map", str(roi), "--shots", str(shots)]
    assert build_moments_pop.main(["fit", str(out), *base, "--selection", str(sel), "--out", str(stem),
                                   "--n-surrogates", "3", "--n-boot", "30", "--min-clips", "20"]) == 0
    meta = json.loads(stem.with_suffix(".json").read_text())
    assert meta["n_train_clips"] == 50                                  # 10 lockbox + 5 fp32 refused
    assert meta["inputs"]["skipped_reasons"] == {"split=lockbox": 10, "precision": 5}
    trained_ids = stem.with_suffix(".train_ids.txt").read_text().split()
    assert not {vid for vid, _ in good[:10]} & set(trained_ids)
    feats = tmp_path / "mpop.csv"
    assert build_moments_pop.main(["features", str(out), *base, "--state", str(stem), "--out", str(feats)]) == 0
    import pandas as pd

    df = pd.read_csv(feats)
    assert len(df) == 65 and "mpop_hook_attention" in df and "edit_cuts_per_min" in df
    assert not [c for c in df.columns if c.startswith("brain_")]
    oof = tmp_path / "oof.csv"
    pd.DataFrame({"target": "log_interactions_rate", "scheme": "content", "video_id": [v for v, _ in good],
                  "y": rng.normal(size=60), "w": 1.0, "pred_A_stack": 0.0,
                  "stratum": [f"deal{i % 4}|tiktok" for i in range(60)]}).to_csv(oof, index=False)
    rep = tmp_path / "contrast.json"
    assert build_moments_pop.main(["contrast", str(out), *base, "--state", str(stem), "--oof", str(oof),
                                   "--selection", str(sel), "--prereg-commit", "test", "--n-perm", "50",
                                   "--out", str(rep)]) == 0
    r = json.loads(rep.read_text())
    assert r["n_clips"] == 50 and r["n_deals"] == 4 and set(r["views"]) >= {"onset_s", "fraction"}
    assert r["n_tests"] == 3 * C and all("test_q_bh" in v for view in r["views"].values() for v in view.values())


def test_contents_spanning_deals_are_refused_not_fatal():
    import pandas as pd

    lab = pd.DataFrame({"video_id": ["a", "a", "b"], "y": [1.0, 2.0, 3.0], "stratum": ["d1", "d2", "d1"],
                        "weight": [1.0, 1.0, 1.0]})
    out = build_moments_pop.one_row_per_content(build_moments_pop.refuse_multi_deal(lab))
    assert out["video_id"].tolist() == ["b"]
