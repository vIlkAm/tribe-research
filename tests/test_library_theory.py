"""library_theory on synthetic data only: split, centred correlation, planted effects found and confirmed."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import library_theory as th  # noqa: E402
import library_tiers as lt  # noqa: E402


def test_split_is_disjoint_and_balanced_per_stratum():
    acc = pd.Series({f"a{i}": ("s1" if i < 10 else "s2") for i in range(21)})
    h = th.split_accounts(acc)
    assert set(h) == set(acc.index) and set(h.values()) == {0, 1}
    for s in ("s1", "s2"):
        v = [h[a] for a in acc.index if acc[a] == s]
        assert abs(sum(v) - len(v) / 2) <= 0.5
    assert th.split_accounts(acc) == h


def test_centred_corr_ignores_group_levels():
    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(20), 10)
    x = rng.normal(size=200)
    y = 0.8 * x + rng.normal(scale=0.5, size=200)
    level = rng.normal(scale=50, size=20)[g]
    w = np.ones(200)
    assert th.centred_corr(x + level, y - level, w, g) > 0.7
    assert abs(th.centred_corr(level + rng.normal(size=200) * 1e-3, level, w, np.zeros(200, int))) > 0.99
    assert abs(th.centred_corr(level + rng.normal(size=200), rng.normal(size=200), w, g)) < 0.2


def _library(rng, effect=True, n_acc=120, per=6):
    acc = np.repeat([f"acc{i}" for i in range(n_acc)], per)
    n = len(acc)
    d = pd.DataFrame({"account": acc, "stratum": np.where(np.arange(n) // per % 2, "d1|tiktok", "d2|youtube"),
                      "w": 1.0})
    for k, *_ in lt.FEATURES:
        d[k] = rng.normal(size=n)
    size = rng.normal(scale=2, size=n_acc)[np.arange(n) // per]
    vid = (0.6 * d["brain_opening"] if effect else 0) + rng.normal(size=n)
    d["local_baseline"] = size
    d["reach_rel_local"] = vid
    d["reach_log"] = size + vid
    d["log_engagement"] = rng.normal(size=n)
    d["half"] = d["account"].map(th.split_accounts(d.groupby("account")["stratum"].first()))
    return d


def test_planted_within_account_effect_holds_and_null_does_not():
    res = th.run(_library(np.random.default_rng(1)), n_boot=200)
    by = {(p["feature"], p["outcome"]): p for p in res["pairs"]}
    assert by[("brain_opening", "video")]["verdict"] == "holds"
    assert by[("brain_opening", "account")]["verdict"] != "holds"
    assert sum(p["holds"] for p in res["pairs"]) <= 3
    null = th.run(_library(np.random.default_rng(2), effect=False), n_boot=200)
    assert sum(p["holds"] for p in null["pairs"]) <= 1


def test_decomposition_and_interpreter():
    d = _library(np.random.default_rng(3))
    dec = th.decompose(d)
    assert abs(dec["share_account"] + dec["share_video"] + dec["share_covariance"] - 1) < 1e-9
    assert dec["share_account"] > dec["share_video"]
    res = th.run(d, n_boot=200)
    line = th.interpreter(res, dec)
    assert "account size" in line and "holding on the other half" in line
    empty = {"pairs": [dict(p, holds=False) for p in res["pairs"]]}
    assert "not whether it will do well" in th.interpreter(empty, dec)
