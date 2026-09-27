"""Arm X (prereg secondary): A + editing covariates from a separate clip table, opt-in via --extra-features.

The arm must reuse the primary run's rows, folds and bootstrap: adding it may not move any A/B/E prediction."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

pytest.importorskip("sklearn")

import fit_models  # noqa: E402
from test_features_models import QUIET, _one_thread, synth  # noqa: E402,F401  (module fixtures)

KW = dict(targets=["log_interactions_rate"], schemes=["content"], models=["stack", "ridge"], n_splits=3, n_boot=30,
          perm_repeats=0, niche=False, threads=1, **QUIET)


def _with_emb(synth):
    f = synth["features"].copy()  # a stand-in extractor column so the E arm (and E − X) exists
    f["emb_video_pca_01"] = np.random.default_rng(5).normal(0, 1, len(f))
    return f


def _edit_table(synth):
    rng = np.random.default_rng(11)
    vids = sorted(synth["amps"])
    return pd.DataFrame({"video_id": vids[1:],  # one clip missing: kept, imputed
                         "edit_cuts_per_min": [synth["amps"][v] + rng.normal(0, 0.2) for v in vids[1:]],
                         "edit_first_cut_s": rng.uniform(0, 5, len(vids) - 1),
                         "mpop_brain_derived": rng.normal(0, 1, len(vids) - 1)})  # other prefixes stay out


def test_arm_x_adds_x_without_moving_the_primary_arms(synth, tmp_path):
    feats = _with_emb(synth)
    res0 = fit_models.run(feats, synth["members"], synth["outcomes"], selection=synth["selection"],
                          out_dir=tmp_path / "base", **KW)
    merged, info = fit_models.merge_extra_features(feats, _edit_table(synth), "edit_")
    assert len(merged) == len(feats) and info["ok_clips_missing"] == 1
    assert info["columns"] == ["edit_cuts_per_min", "edit_first_cut_s"]
    res1 = fit_models.run(merged, synth["members"], synth["outcomes"], selection=synth["selection"],
                          out_dir=tmp_path / "x", extra=info, **KW)

    t = "log_interactions_rate"
    assert "X" not in res0["features"][t] and "extra_features" not in res0["config"]
    assert res1["config"]["extra_features"]["prefix"] == "edit_"
    fx = res1["features"][t]
    assert {"edit_cuts_per_min", "edit_first_cut_s"} <= set(fx["X"]) and "mpop_brain_derived" not in fx["X"]
    assert not any(c.startswith("edit_") for k in ("A", "B", "E", "BE") for c in fx[k])
    assert set(fx["X"]) - set(fx["A"]) == {"edit_cuts_per_min", "edit_first_cut_s"}

    o0 = pd.read_csv(tmp_path / "base" / "oof_predictions.csv")
    o1 = pd.read_csv(tmp_path / "x" / "oof_predictions.csv")
    assert len(o0) == len(o1) and list(o0["vp_id"]) == list(o1["vp_id"])
    for c in [c for c in o0.columns if c.startswith("pred_")]:
        np.testing.assert_array_equal(o0[c].to_numpy(), o1[c].to_numpy())  # same rows, folds, fits
    assert "pred_X_stack" in o1

    p0, p1 = (r["targets"][t]["schemes"]["content"]["pooled"] for r in (res0, res1))
    assert p0["n"] == p1["n"] and p0["models"]["A_stack"] == p1["models"]["A_stack"]
    assert p0["deltas"]["BE_stack-E_stack"] == p1["deltas"]["BE_stack-E_stack"]  # bootstrap draws unchanged
    for d in ("X_stack-A_stack", "B_stack-X_stack", "E_stack-X_stack"):
        assert d in p1["deltas"] and d not in p0["deltas"]
    assert p1["deltas"]["X_stack-A_stack"]["within_stratum_spearman"]["point"] > 0  # planted editing signal
    report = (tmp_path / "x" / "report.md").read_text()
    assert "X − A" in report and "Arm X" in report
    assert "Arm X" not in (tmp_path / "base" / "report.md").read_text()


def test_merge_extra_features_refuses_bad_tables(synth):
    f = synth["features"]
    ok = pd.DataFrame({"video_id": ["c0000"], "edit_x": [1.0]})
    with pytest.raises(SystemExit):
        fit_models.merge_extra_features(f, pd.concat([ok, ok]), "edit_")  # duplicate video_id
    with pytest.raises(SystemExit):
        fit_models.merge_extra_features(f, ok, "brain_")  # would leak into another block
    with pytest.raises(SystemExit):
        fit_models.merge_extra_features(f, ok, "cut_")  # no matching column
