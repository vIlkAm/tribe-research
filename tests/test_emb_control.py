"""Extractor-embedding control arm (E, BE) and the non-English clip rule, on small synthetic outputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

pytest.importorskip("sklearn")

import build_features  # noqa: E402
import fit_models  # noqa: E402
from test_features_models import QUIET, write_clip, write_roi  # noqa: E402

N = 40
DIMS = {"video": 12, "audio": 8, "text": 16}


def write_emb(d: Path, vid: str, rng, modalities=("video", "audio", "text")) -> None:
    arrs = {}
    for m in modalities:
        mean = rng.normal(0, 1, (2, DIMS[m]))
        arrs.update({f"{m}_mean": mean.astype(np.float16), f"{m}_sd": np.abs(mean).astype(np.float16),
                     f"{m}_bins": (mean[None] + rng.normal(0, 0.3, (4,) + mean.shape)).astype(np.float16), f"{m}_n": np.int32(20)})
    np.savez(d / f"{vid}.emb.npz", **arrs)


@pytest.fixture(scope="module")
def outputs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("emb")
    roi = tmp / "roi.npz"
    groups, v = write_roi(roi)
    d = tmp / "outputs" / "worker-0"
    d.mkdir(parents=True)
    rng = np.random.default_rng(3)
    for i in range(N):
        vid = f"e{i:03d}"
        write_clip(d, vid, groups, v, 1.0, 1.0, rng)
        if i < N - 4:
            write_emb(d, vid, rng, ("video", "audio") if i % 5 == 0 else ("video", "audio", "text"))
    (d / "e036.emb.npz").write_bytes(b"not a zip")  # broken export: the clip keeps its preds, loses emb
    meta = json.loads((d / "e001.json").read_text())
    meta["transcript"] = {"detected_language": "es", "language_probability": 0.93}
    (d / "e001.json").write_text(json.dumps(meta))
    meta = json.loads((d / "e002.json").read_text())
    meta["transcript"] = {"detected_language": "de", "language_probability": 0.2}  # too unsure to trust
    (d / "e002.json").write_text(json.dumps(meta))
    return tmp / "outputs", roi


def test_emb_pca_columns_lockbox_projected_and_broken_export_tolerated(outputs):
    out, roi = outputs
    lock = {"e003", "e004"}
    df, meta, _ = build_features.build(out, roi, n_jobs=1, n_pca=3, pca_exclude=lock, **QUIET)
    assert (df["status"] == "ok").all() and len(df) == N
    for m in ("video", "audio", "text"):
        cols = [f"emb_{m}_pca_{j:02d}" for j in (1, 2, 3)]
        assert set(cols) <= set(df.columns)
        info = meta["emb"]["per_modality"][m]
        assert info["dim"] == 2 * DIMS[m] and info["n_fit"] == info["n_clips"] - 2
    by = df.set_index("video_id")
    assert by.loc[["e036", "e037", "e038", "e039"], "emb_video_pca_01"].isna().all()  # no / broken export
    assert by.loc["e005", "emb_text_pca_01"] != by.loc["e005", "emb_text_pca_01"]  # text omitted -> NaN
    assert by.loc["e006", "qc_emb_modalities"] == 3 and by.loc["e005", "qc_emb_modalities"] == 2
    assert meta["columns"]["emb"] and all(c.startswith("emb_") for c in meta["columns"]["emb"])
    for m in ("video", "audio", "text"):  # the control sees time structure too, not only the time mean
        assert meta["emb"]["per_modality"][f"{m}_sd"]["dim"] == 2 * DIMS[m]
        assert meta["emb"]["per_modality"][f"{m}_bins"]["dim"] == 4 * 2 * DIMS[m]
        assert f"emb_{m}_sd_pca_01" in df.columns and f"emb_{m}_bins_pca_01" in df.columns
    assert meta["emb"]["version"] == "emb_features_v1"


def test_load_emb_bins_are_shape_around_the_mean_and_empty_quarters_carry_nothing(tmp_path):
    mean = np.array([[1.0, 2.0]])
    bins = np.array([[[0.0, 2.0]], [[2.0, 2.0]], [[np.nan, np.nan]], [[1.0, 2.0]]])
    np.savez(tmp_path / "v.emb.npz", video_mean=mean.astype(np.float16), video_sd=np.ones_like(mean),
             video_bins=bins.astype(np.float16), video_n=np.int32(8))
    e = build_features.load_emb(str(tmp_path / "v.npz"))
    assert set(e) == {"video", "video_sd", "video_bins"}
    np.testing.assert_allclose(e["video_bins"], [-1, 0, 1, 0, 0, 0, 0, 0])


def test_language_qc_flags_only_confident_non_english(outputs):
    out, roi = outputs
    df, _, _ = build_features.build(out, roi, n_jobs=1, n_pca=2, **QUIET)
    by = df.set_index("video_id")
    assert by.loc["e001", "qc_non_english"] == 1.0
    assert by.loc["e002", "qc_non_english"] == 0.0 and by.loc["e010", "qc_non_english"] == 0.0


def _design(n=60, with_emb=True, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"platform": rng.choice(["tiktok", "instagram"], n), "deal_id": rng.choice(["a", "b"], n),
                       "base_log_duration": rng.normal(3, 0.3, n), "brain_x": rng.normal(0, 1, n)})
    if with_emb:
        df["emb_video_pca_01"] = rng.normal(0, 1, n)
    return df


def test_feature_sets_add_control_arms_only_with_emb():
    fs = fit_models.feature_sets(_design())
    assert set(fs) == {"A", "B", "E", "BE"}
    assert "emb_video_pca_01" in fs["E"][1] and "brain_x" not in fs["E"][1]
    assert {"emb_video_pca_01", "brain_x"} <= set(fs["BE"][1])
    assert set(fit_models.feature_sets(_design(with_emb=False))) == {"A", "B"}


def test_capped_ui_moment_columns_stay_out_of_the_brain_block_by_default():
    df = _design()
    df["brain_moments_total_per_min"] = np.random.default_rng(3).normal(0, 1, len(df))
    fs = fit_models.feature_sets(df)
    assert "brain_x" in fs["BE"][1] and "brain_moments_total_per_min" not in fs["BE"][1] + fs["B"][1]
    assert "brain_moments_total_per_min" in fit_models.feature_sets(df, moment_cols=True)["BE"][1]


def test_bootstrap_reports_brain_beyond_inputs_and_report_table():
    rng = np.random.default_rng(1)
    n = 120
    y = rng.normal(0, 1, n)
    preds = {"A_ridge": rng.normal(0, 1, n), "B_ridge": y + rng.normal(0, 1, n),
             "E_ridge": y + rng.normal(0, 1, n), "BE_ridge": y + rng.normal(0, 0.5, n)}
    b = fit_models.bootstrap(y, preds, np.arange(n), np.repeat(["s1", "s2"], n // 2), None, 50, 0)
    assert {"B_ridge-A_ridge", "E_ridge-A_ridge", "BE_ridge-E_ridge", "B_ridge-E_ridge", "BE_ridge-A_ridge"} <= set(b["deltas"])
    L: list[str] = []
    fit_models._control_table(L, b, ["ridge"])
    text = "\n".join(L)
    assert "| ridge | BE − E |" in text and "| ridge | BE − A |" in text
    L2: list[str] = []
    fit_models._control_table(L2, fit_models.bootstrap(y, {"A_ridge": preds["A_ridge"], "B_ridge": preds["B_ridge"]},
                                                       np.arange(n), np.zeros(n), None, 20, 0), ["ridge"])
    assert L2 == []  # no E arm, no control table


def test_build_dataset_drops_confident_non_english_clips():
    feats = pd.DataFrame({"video_id": ["v1", "v2"], "status": "ok", "base_log_duration": [3.0, 3.1],
                          "qc_non_english": [1.0, 0.0]})
    members = pd.DataFrame({"video_id": ["v1", "v2"], "vp_id": ["p1", "p2"], "platform": "tiktok",
                            "deal_id": "d"})
    outcomes = pd.DataFrame({"id": ["p1", "p2"], "deal_id": "d", "platform": "tiktok", "social_account_id": "acc",
                             "reach_rel_local": [0.1, 0.2], "local_baseline_level": "account_local"})
    df, info, _ = fit_models.build_dataset(feats, members, outcomes, ["reach_rel_local"])
    assert list(df["video_id"]) == ["v2"] and info["clips_excluded_non_english"] == 1
    df, info, _ = fit_models.build_dataset(feats, members, outcomes, ["reach_rel_local"], exclude_non_english=False)
    assert len(df) == 2


def _stack_data(n=600, n_noise=60, signal=0.0, seed=0):
    rng = np.random.default_rng(seed)
    content = np.arange(n)
    a = rng.normal(0, 1, n)
    brain_sig = rng.normal(0, 1, n)
    y = 0.5 * a + signal * brain_sig + rng.normal(0, 1, n)
    df = pd.DataFrame({"platform": rng.choice(["tiktok", "instagram"], n), "deal_id": "d", "_content": content,
                       "social_account_id": rng.choice([f"acc{i}" for i in range(30)], n),
                       "base_log_duration": a, "brain_sig": brain_sig})
    for j in range(n_noise):
        df[f"brain_noise_{j}"] = rng.normal(0, 1, n)
    return df, y


def _held_out_corr(kind, cols, df, y):
    tr, te = np.arange(len(y)) < 400, np.arange(len(y)) >= 400
    _, p = fit_models.fit_predict(kind, cols, df[tr], y[tr], None, df[te], 0)
    return np.corrcoef(p, y[te])[0, 1], p


def test_stack_compresses_each_block_so_pure_noise_costs_little():
    df, y = _stack_data()
    noise = [c for c in df if c.startswith("brain_noise_")]
    a_cols = (["platform"], ["base_log_duration"])
    b_cols = (["platform"], ["base_log_duration"] + noise)
    r_a, p_a = _held_out_corr("stack", a_cols, df, y)
    r_stack, p_b = _held_out_corr("stack", b_cols, df, y)
    assert r_stack > r_a - 0.02 and np.corrcoef(p_a, p_b)[0, 1] > 0.95


def test_stack_uses_a_real_block_signal_and_scores_one_column_per_block():
    df, y = _stack_data(signal=0.6, n_noise=5)
    cols = (["platform"], ["base_log_duration", "brain_sig"] + [c for c in df if c.startswith("brain_noise_")])
    r_a, _ = _held_out_corr("stack", (["platform"], ["base_log_duration"]), df, y)
    r_b, _ = _held_out_corr("stack", cols, df, y)
    assert r_b > r_a + 0.1
    est = fit_models.make_model("stack", *cols).fit(df[fit_models.input_columns(cols)], y,
                                                   groups=df["_content"].to_numpy())
    assert est.score_cols_ == ["stack_brain"] and est.a_num_ == ["base_log_duration"]
