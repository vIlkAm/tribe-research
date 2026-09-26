"""Feature table + model comparison + niche tuning on synthetic worker outputs with planted signals.

Kept small and single-threaded so it stays quick on a loaded shared host: a reduced vertex count for the
bulk synthetic clips (the full 20484-vertex production format is covered by one real dry-run clip),
few HGB iterations, few folds and bootstrap draws.
"""

from __future__ import annotations

import json
import re
import subprocess
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
import profile_account  # noqa: E402
from tribe_research.brain.proxies import ProxySpec  # noqa: E402

SPEC = ProxySpec.load()
V_FULL = 20484
FORBIDDEN = re.compile(r"\bfir(e|es|ed|ing)\b|\bcaus(e|es|ed|ing|al)\b|reads?\b.*\bmind|\bviral", re.I)
DEALS = ["deal-a", "deal-b", "deal-c", "deal-d", "deal-e", "deal-f"]
PLATFORMS = ["tiktok", "instagram", "youtube"]
N_CONTENT = 240
NICHE_DEAL = "deal-c"
DEAL_CYCLE = DEALS[:3] + [NICHE_DEAL] + DEALS[3:]  # the niche deal gets a double share of contents
QUIET = {"log": lambda *a: None}


@pytest.fixture(scope="module", autouse=True)
def _one_thread():
    from threadpoolctl import threadpool_limits

    with threadpool_limits(1):
        yield


def _run(*args):
    return subprocess.run([sys.executable, *map(str, args)], check=True, capture_output=True, text=True)


def write_roi(path: Path, v: int | None = None) -> tuple[list[str], int]:
    groups = sorted({g for c in SPEC.channels for g in c.roi_groups})
    v = v or (len(groups) + 1) * 100
    mask = np.zeros((len(groups), v), bool)
    for i in range(len(groups)):
        mask[i, i * 100:(i + 1) * 100] = True
    np.savez(path, group_names=np.array(groups), group_mask=mask,
             provenance=json.dumps({"groups_version": "SYNTHETIC", "synthetic": True}))
    return groups, v


def channel_vertices(groups: list[str], key: str) -> np.ndarray:
    ch = next(c for c in SPEC.channels if c.key == key)
    return np.concatenate([np.arange(groups.index(g) * 100, (groups.index(g) + 1) * 100) for g in ch.roi_groups])


def write_clip(d: Path, vid: str, groups, v: int, amp: float, amp2: float, rng, tr: float = 1.0) -> None:
    """Worker-format outputs; channel noise is coherent per region so it survives the ROI mean."""
    T = int(rng.integers(12, 30))
    t = np.arange(T) * tr
    p = rng.normal(0, 0.05, (T, v)).astype(np.float32)
    for c in SPEC.channels:
        p[:, channel_vertices(groups, c.key)] += rng.normal(0, 1.0, T)[:, None]
    p[:, channel_vertices(groups, "attention")] += (amp * (t < 2.0))[:, None]  # planted early attention bump
    p[:, channel_vertices(groups, "social")] += (amp2 * (t < 2.0))[:, None]  # rewarded in the niche deal only
    np.savez(d / f"{vid}.npz", preds=p.astype(np.float16), seg_start=t, seg_duration=np.full(T, tr))
    n_words = int(rng.integers(0, 30))
    words = [{"start": round(float(s), 3), "duration": 0.3, "text": "w"}
             for s in np.sort(rng.uniform(0, T * tr, n_words))]
    meta = {"video_id": vid, "path": f"{vid}.mp4", "source_name": f"vp-{vid}", "duration_s": float(T * tr),
            "n_segments": T, "n_vertices": v, "tr_s": tr, "words": words, "worker_id": int(d.name.split("-")[1]),
            "runtime": {"dry_run": True}}
    (d / f"{vid}.json").write_text(json.dumps(meta))


def write_failures(d0: Path, d1: Path) -> None:
    (d0 / "bad1.error.json").write_text(json.dumps({"video_id": "bad1", "error": "boom", "category": "gpu_oom"}))
    (d1 / "bad2.json").write_text(json.dumps({"video_id": "bad2", "duration_s": 5.0}))
    (d1 / "bad2.npz").write_bytes(b"not a zip")


def _z(a):
    a = np.asarray(a, float)
    return (a - a.mean()) / a.std()


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("fm")
    roi = tmp / "roi.npz"
    groups, v = write_roi(roi)
    out = tmp / "outputs"
    for w in range(3):
        (out / f"worker-{w}").mkdir(parents=True)
    rng = np.random.default_rng(7)
    amps = rng.uniform(0.0, 5.0, N_CONTENT)
    amps2 = rng.uniform(0.0, 5.0, N_CONTENT)
    vids = [f"c{i:04d}" for i in range(N_CONTENT)]
    for i, vid in enumerate(vids):
        write_clip(out / f"worker-{i % 3}", vid, groups, v, amps[i], amps2[i], rng)
    write_failures(out / "worker-0", out / "worker-1")
    df, meta, _ = build_features.build(out, roi, n_jobs=1, n_pca=5, **QUIET)

    # posts: 1-3 platforms per content, 1-2 accounts per platform (same-platform reposts for the ICC)
    rows, outcomes, selection = [], [], []
    z1, z2 = _z(amps), _z(amps2)
    p_off = {"tiktok": 0.4, "instagram": 0.0, "youtube": -0.4}
    for i, vid in enumerate(vids):
        deal = DEAL_CYCLE[i % len(DEAL_CYCLE)]
        # the niche deal rewards the social bump instead of the attention bump
        content_reach = 2.0 * z2[i] if deal == NICHE_DEAL else z1[i]
        lockbox = rng.random() < 0.25
        anchor = None
        for plat in rng.choice(PLATFORMS, int(rng.integers(1, 4)), replace=False):
            for acc in rng.choice(4, int(rng.integers(1, 3)), replace=False):
                account = f"acc-{deal}-{plat}-{acc}"
                anchor = anchor or account
                vp = f"vp-{vid}-{plat}-{acc}"
                rows.append({"video_id": vid, "vp_id": vp, "platform": plat, "deal_id": deal})
                outcomes.append({
                    "id": vp, "deal_id": deal, "platform": plat, "social_account_id": account,
                    "content_group": "linked" if vid in ("c0000", "c0004") else f"cg-{vid}",
                    "upload_date": pd.Timestamp("2026-06-01", tz="UTC") + pd.Timedelta(days=int(rng.integers(0, 60))),
                    "reach_rel_local": content_reach + p_off[plat] + rng.normal(0, 0.35),
                    "local_baseline_level": "account_local",
                    "interactions_rate_eb": float(np.exp(-4 + 0.4 * z1[i] + 0.2 * p_off[plat] + rng.normal(0, 0.3))),
                    "views_7d": 10 ** (4 + z1[i]),  # target-derived: must never be used as a feature
                    **{f: False for f in fit_models.EXCLUDE_IF_TRUE if f.startswith("flag_")},
                })
        selection.append({"video_id": vid, "deal_id": deal, "split": "lockbox" if lockbox else "train",
                          "incl_prob": 0.05 if lockbox else float(rng.uniform(0.3, 1.0)),
                          "anchor_account": anchor, "stratum": deal, "audio_mean_db": float(rng.normal(-20, 3)),
                          "width": 1080, "height": int(rng.choice([1920, 1350]))})
    outcomes[1]["flag_history_truncated"] = True
    outcomes[2]["local_baseline_level"] = "deal_platform"  # reach target unusable, engagement still usable
    return {"features": df, "meta": meta, "members": pd.DataFrame(rows), "outcomes": pd.DataFrame(outcomes),
            "selection": pd.DataFrame(selection), "amps": dict(zip(vids, amps)), "tmp": tmp, "out": out, "roi": roi}


@pytest.fixture(scope="module")
def fitted(synth, tmp_path_factory):
    out = tmp_path_factory.mktemp("models")
    mp = pytest.MonkeyPatch()
    mp.setitem(fit_models.HGB_PARAMS, "max_iter", 40)
    try:
        res = fit_models.run(synth["features"], synth["members"], synth["outcomes"], selection=synth["selection"],
                             targets=["reach_rel_local"], out_dir=out, n_splits=4, n_boot=60, perm_repeats=1, perm_per_feature=False,
                             min_deal_n=30, niche_base="ridge", score_lockbox=True, threads=1, **QUIET)
    finally:
        mp.undo()
    return res, out


def test_features_table(synth):
    df, meta = synth["features"], synth["meta"]
    st = df.set_index("video_id")["status"]
    assert (st == "ok").sum() == N_CONTENT
    assert st["bad1"] == "failed" and st["bad2"] == "error"
    brain, base = meta["columns"]["brain"], meta["columns"]["base"]
    for k in [c.key for c in SPEC.channels]:
        for s in ("mean_0_3s", "mean_0_5s", "mean_all", "peak", "trough", "slope_half",
                  "peak_time_frac", "stability", "early_minus_rest"):
            assert f"brain_{k}_{s}" in brain
    assert {f"brain_pca_{j:02d}" for j in range(1, 6)} <= set(brain)
    assert "brain_moments_attention_drop_per_min" in brain
    assert {"base_duration_s", "base_word_rate", "base_speech_coverage", "base_shots_per_s"} <= set(base)
    ok = df[df["status"] == "ok"].set_index("video_id")
    a = pd.Series(synth["amps"])
    rho = ok.loc[a.index, "brain_attention_mean_0_3s"].corr(a, method="spearman")
    assert rho > 0.8  # the planted bump is visible in the early-window feature
    assert ok["synthetic"].all()


def test_build_features_real_worker_format_and_cli(tmp_path):
    """One real pod/worker.py --dry-run output (production format, 20484 vertices) plus failures, via the CLI."""
    from test_pipeline import fake_mp4

    roi = tmp_path / "roi.npz"
    groups, v = write_roi(roi, V_FULL)
    out = tmp_path / "outputs"
    videos = tmp_path / "videos"
    videos.mkdir()
    fake_mp4(videos / "real.mp4", 9.0, b"r")
    manifest = tmp_path / "manifest.jsonl"
    _run(ROOT / "tools/make_manifest.py", "--videos-root", videos, "--workers", 1, "--out", manifest)
    _run(ROOT / "pod/worker.py", "--manifest", manifest, "--videos-root", videos,
         "--out-root", out, "--worker-id", 0, "--dry-run")
    real_vid = next((out / "worker-0").glob("*.json")).stem
    (out / "worker-1").mkdir()
    rng = np.random.default_rng(1)
    for i in range(3):
        write_clip(out / "worker-1", f"s{i}", groups, v, 1.0, 1.0, rng)
    write_failures(out / "worker-0", out / "worker-1")
    f = tmp_path / "f.parquet"
    _run(ROOT / "tools/build_features.py", "--out-root", out, "--out", f, "--roi-map", roi, "--n-jobs", 1,
         "--n-pca", 3)
    st = pd.read_parquet(f).set_index("video_id")["status"]
    assert st[real_vid] == "ok" and st["bad1"] == "failed" and st["bad2"] == "error" and (st == "ok").sum() == 4
    meta = json.loads((tmp_path / "f.meta.json").read_text())
    assert meta["pca"]["n_components"] == 3 and (tmp_path / "f.pca.npz").exists()
    # --limit counts completed clips, including the corrupt one
    df, _, _ = build_features.build(out, roi, n_jobs=1, n_pca=2, limit=3, **QUIET)
    assert df["status"].isin(["ok", "error"]).sum() == 3


def test_grouping_never_splits_content(synth):
    df, info, posts = fit_models.build_dataset(synth["features"], synth["members"], synth["outcomes"],
                                               list(fit_models.DEFAULT_TARGETS), synth["selection"])
    assert info["excluded_by_quality_flags"] == 1
    assert info["target_rows_failing_requirements"]["reach_rel_local"] == 1
    assert df["y_reach_rel_local"].isna().sum() <= 1 and df["y_log_interactions_rate"].notna().all()
    assert "views_7d" not in df.columns and "base_aspect" in df.columns
    assert np.allclose(df["_w"], 1 / df["incl_prob"])
    # the outcomes content_group link merges two different clips into one content component
    comp = df.groupby("video_id")["_content"].first()
    if {"c0000", "c0004"} <= set(comp.index):
        assert comp["c0000"] == comp["c0004"]
    assert df.groupby("_content")["split"].nunique().max() == 1  # no content component spans train/lockbox
    tr = df[df["split"] == "train"].reset_index(drop=True)
    for scheme in fit_models.SCHEMES:
        splits = fit_models.make_splits(tr, scheme, n_splits=5, seed=0, min_deal_n=30)
        assert splits, scheme
        covered = np.zeros(len(tr), bool)
        for _, a, b in splits:
            assert not set(a) & set(b)
            for col in ("video_id", "_content", "content_group"):
                assert not set(tr[col].iloc[a]) & set(tr[col].iloc[b]), (scheme, col)
            if scheme == "account":  # neither anchor nor posting accounts cross a fold (fix: posting accounts)
                assert not set(tr["anchor_account"].iloc[a]) & set(tr["anchor_account"].iloc[b])
                assert not set(tr["social_account_id"].iloc[a]) & set(tr["social_account_id"].iloc[b])
            if scheme == "lodo":
                assert tr["deal_id"].iloc[b].nunique() == 1
                assert tr["deal_id"].iloc[b].iloc[0] not in set(tr["deal_id"].iloc[a])
            covered[b] = True
        if scheme != "lodo":
            assert covered.all()


def test_weighted_metrics():
    rng = np.random.default_rng(0)
    y, p = rng.normal(size=300), rng.normal(size=300)
    codes = rng.integers(0, 3, 300)
    from scipy import stats

    # unweighted within-stratum Spearman equals the scipy per-stratum mean (equal-size-weighted)
    ref = [stats.spearmanr(y[codes == g], p[codes == g])[0] for g in range(3)]
    n = np.bincount(codes)
    assert np.isclose(fit_models.strat_spearman(y, p, codes), np.average(ref, weights=n))
    # integer weights == row duplication, ties included
    yt = np.round(y, 1)
    w = rng.integers(1, 4, 300).astype(float)
    rep = np.repeat(np.arange(300), w.astype(int))
    assert np.isclose(fit_models.spearman(yt, p, w), fit_models.spearman(yt[rep], p[rep]))
    assert np.isclose(fit_models.r2(y, p, w), fit_models.r2(y[rep], p[rep]))


def test_planted_signal_b_beats_a(fitted):
    res, out = fitted
    tr = res["targets"]["reach_rel_local"]
    content = tr["schemes"]["content"]["pooled"]
    for model in ("ridge", "hgb"):
        d = content["deltas"][f"B_{model}-A_{model}"]
        assert d["within_stratum_spearman"]["point"] > 0.1, model
    assert content["deltas"]["B_ridge-A_ridge"]["within_stratum_spearman"]["ci"][0] > 0
    assert set(tr["schemes"]) == {"content", "account", "lodo"}
    lodo = tr["schemes"]["lodo"]
    assert len(lodo["per_deal"]) == len(DEALS)
    # leave-one-deal-out resamples deals, not contents; the per-deal spread is reported
    assert lodo["bootstrap_unit"] == "deal" and lodo["pooled"]["n_units"] == len(DEALS)
    assert lodo["deal_spread"]["B_ridge-A_ridge"]["r2"]["n_deals"] == len(DEALS)
    assert lodo["per_deal"][NICHE_DEAL]["n_units"] > 10  # inside one deal: content units, real CIs
    fs = res["features"]["reach_rel_local"]
    assert fit_models.ACCOUNT_TE in fs["A"] and fit_models.ACCOUNT_TE in fs["B"]  # B - A isolates the brain columns
    # final numbers: the lockbox, fit once on train, unweighted
    lb = tr["lockbox"]["pooled"]
    assert lb["n"] == tr["n_lockbox"] > 30
    for model in ("ridge", "hgb"):  # few lockbox posts per deal×platform, so the pooled R² carries the check
        assert lb["deltas"][f"B_{model}-A_{model}"]["r2"]["ci"][0] > 0, model
    assert set(tr["lockbox"]["per_platform"]) <= set(PLATFORMS)
    # ICC ceilings from same-platform reposts / across platforms
    icc = tr["icc"]
    assert 0.3 < icc["same_platform"]["icc"] < 1 and icc["same_platform"]["groups"] > 10
    assert np.isfinite(icc["across_platforms"]["icc"])
    fams = tr["permutation_importance"]["families"]
    top = [f["name"] for f in fams if f["name"] != "family:all_brain"][:3]
    assert "family:attention" in top or "family:xch" in top
    assert "views_7d" not in res["features"]["reach_rel_local"]["B"]
    qt = pd.read_csv(out / "quartile_table_reach_rel_local.csv")
    assert {"rho", "p_account", "q_bh_within_deal", "q_bh_global"} <= set(qt.columns)
    report = (out / "report.md").read_text()
    assert "SYNTHETIC" in report and "within-stratum" in report and "lockbox" in report.lower()
    assert report.index("Final result: lockbox") < report.index("cross-validation on the train split")
    assert "ICC" in report and "Platform-centred R²" in report and "natural distribution" not in report
    assert "te_account_mean" in report and "account-clustered" in report
    # 6-12 posts per account never reach min_account_n: the report says the account layer was skipped
    al = tr["niche"]["account_layer"]
    assert al["accounts_qualified"] == 0 and al["accounts_considered"] > 0
    assert "Account-level layers were skipped" in report
    assert "BH q=" in report and "conditional on the deviation penalty" in report
    assert "stimulus-onset transient" in report
    assert not FORBIDDEN.search(report), FORBIDDEN.search(report)
    json.loads((out / "metrics.json").read_text())
    oof = pd.read_csv(out / "oof_predictions.csv")
    assert {"content", "account", "lodo", "lockbox"} <= set(oof["scheme"])


def test_niche_tuning_helps_only_the_planted_deal(fitted):
    res, _ = fitted
    ni = res["targets"]["reach_rel_local"]["niche"]
    assert set(ni["per_deal"]) == set(DEALS)
    for deal, r in ni["per_deal"].items():
        d_gen = r["deltas"]["tuned_C-general_B"]["r2"]
        d_wo = r["deltas"]["tuned_C-general_without_deal"]["r2"]
        if deal == NICHE_DEAL:
            assert d_gen["ci"][0] > 0 and d_wo["ci"][0] > 0, (deal, d_gen, d_wo)
            assert r["deltas"]["tuned_C-offset_only"]["r2"]["ci"][0] > 0  # the gain comes from the slopes
        else:
            assert d_gen["ci"][0] <= 0, (deal, d_gen)  # no reliable gain where there is no niche deviation
    curve = ni["learning_curve"][NICHE_DEAL]
    assert curve["all"]["r2"] > curve["0"]["r2"] and "10" in curve
    # verdicts are BH-adjusted over deals: q >= p, and the planted deal has the smallest q
    q = {d: r["q_bh_tuned_vs_general_r2"] for d, r in ni["per_deal"].items()}
    assert all(q[d] >= r["deltas"]["tuned_C-general_B"]["r2"]["p_boot"] - 1e-12 for d, r in ni["per_deal"].items())
    assert min(q, key=q.get) == NICHE_DEAL, q  # 6 deals x 60 boots: ranking, not a q threshold
    # profiles re-select their penalty on their own (globally standardised) scale. With ~50 niche contents the
    # 1-SE rule shrinks the deviations hard, so check direction, not size: the global slope on the planted social
    # feature is pulled up by the niche deal, so only the niche deal deviates upward from it (and reliably so)
    profs = res["targets"]["reach_rel_local"]["niche_profiles"]
    dev = {d: next(r for r in p["features"] if r["feature"] == "brain_social_mean_0_3s") for d, p in profs.items()}
    assert dev[NICHE_DEAL]["deviation"] > 0 and dev[NICHE_DEAL]["deviation_q"] < 0.05, dev[NICHE_DEAL]
    assert all(r["deviation"] < 0 for d, r in dev.items() if d != NICHE_DEAL), dev
    assert all(r["deviation_q"] >= r["deviation_p"] for p in profs.values() for r in p["features"])
    assert all(np.isfinite(p["alpha"]) and "deviation_q" in p["features"][0] for p in profs.values())


def test_fit_niche_and_profile_account(synth, tmp_path):
    df, info, _ = fit_models.build_dataset(synth["features"], synth["members"], synth["outcomes"],
                                           ["reach_rel_local"], synth["selection"])
    train = df[(df["split"] == "train") & df["y_reach_rel_local"].notna()].reset_index(drop=True)
    nm = fit_models.fit_niche(train, f"deal:{NICHE_DEAL}", "reach_rel_local", base="ridge", n_boot=40)
    m = fit_models.niche_mask(train, f"deal:{NICHE_DEAL}")
    assert nm.predict(train[m]).shape == (m.sum(),)
    assert nm.profile["n_posts"] == m.sum() and nm.adjustment.alpha < fit_models.ALPHAS_ADJ[-1]
    # an account niche: its contents' reposts on other accounts stay out of the general model
    acct = train["social_account_id"].value_counts().index[0]
    na = fit_models.fit_niche(train, f"account:{acct}", "reach_rel_local", base="ridge", n_boot=5)
    ma, content = fit_models.niche_mask(train, f"account:{acct}"), train["_content"].to_numpy()
    assert na.n_general == (~ma & ~np.isin(content, content[ma])).sum() < (~ma).sum()
    # CLI: files on disk, a deal and an account
    f = tmp_path / "features.parquet"
    synth["features"].to_parquet(f)
    synth["members"].to_csv(tmp_path / "members.csv", index=False)
    synth["outcomes"].to_parquet(tmp_path / "outcomes.parquet")
    synth["selection"].to_csv(tmp_path / "selection.csv", index=False)
    common = ["--features", f, "--members", tmp_path / "members.csv", "--outcomes", tmp_path / "outcomes.parquet",
              "--selection", tmp_path / "selection.csv", "--target", "reach_rel_local", "--out-dir", tmp_path / "p",
              "--base", "ridge", "--n-boot", 40, "--threads", 1]
    profile_account.main([*map(str, common), "--deal", NICHE_DEAL])
    md = (tmp_path / "p" / f"deal-{NICHE_DEAL}-reach_rel_local.md").read_text()
    js = json.loads((tmp_path / "p" / f"deal-{NICHE_DEAL}-reach_rel_local.json").read_text())
    assert "SYNTHETIC" in md and "Caveats" in md and not FORBIDDEN.search(md)
    assert js["quality"]["deltas"]["tuned-general_without_niche"]["r2"]["point"] > 0
    # "differs" is exactly the BH-significant deviations, and the planted social feature is among them
    q = {r["feature"]: r["deviation_q"] for r in js["profile"]["features"]}
    assert {r["feature"] for r in js["differs"]} == {f for f, v in q.items() if v < profile_account.DIFFERS_Q}
    assert "brain_social_mean_0_3s" in {r["feature"] for r in js["differs"]}
    assert "BH q" in md and "conditional on the deviation penalty" in md
    profile_account.main([*map(str, common), "--account", acct])
    assert (tmp_path / "p" / f"account-{acct}-reach_rel_local.md").exists()


# ── focused checks for the review fixes ───────────────────────────────────


def test_account_components_join_anchors_through_a_posting_account():
    # two contents with different anchors, reposted by one shared account: one component (fix 1)
    df = pd.DataFrame({"video_id": ["v1", "v2", "v3"], "social_account_id": ["a1", "shared", "shared"],
                       "anchor_account": ["a1", "a1", "a2"], "content_group": ["g1", "g1", "g3"]})
    df = pd.concat([df, pd.DataFrame({"video_id": ["v4"], "social_account_id": ["a4"], "anchor_account": ["a4"],
                                      "content_group": ["g4"]})], ignore_index=True)
    c = fit_models.account_components(df)
    assert c[0] == c[1] == c[2] and c[3] != c[0]


def test_engagement_target_uses_the_selector_label_rules():
    flags = {f: False for f in fit_models.EXCLUDE_IF_TRUE if f.startswith("flag_")}
    rows = [  # (rate, lo90, hi90, extra flags): fix 2
        (0.05, 0.04, 0.06, {}),                               # narrow: kept
        (0.05, 0.00, 0.06, {}),                               # width / rate = 1.2 >= 1: unknown
        (0.05, np.nan, np.nan, {}),                           # no interval: unknown
        (0.05, 0.04, 0.06, {"flag_views_zero": True}),        # engagement only
        (0.05, 0.04, 0.06, {"flag_rate_gt1": True}),          # engagement only
        (0.05, 0.04, 0.06, {"flag_missing_upload_date": True}),  # study-set label flag: whole row
    ]
    out = pd.DataFrame([{"id": f"p{i}", "deal_id": "d", "platform": "tiktok", "social_account_id": f"a{i}",
                         "reach_rel_local": 0.1, "local_baseline_level": "account_local", "interactions_rate_eb": r,
                         "interactions_rate_eb_lo90": lo, "interactions_rate_eb_hi90": hi, **flags, "flag_views_zero": False,
                         "flag_rate_gt1": False, **ex} for i, (r, lo, hi, ex) in enumerate(rows)])
    mem = pd.DataFrame({"video_id": [f"v{i}" for i in range(6)], "vp_id": [f"p{i}" for i in range(6)]})
    df, info = fit_models.load_posts(mem, out, fit_models.DEFAULT_TARGETS)
    df = df.set_index("vp_id")
    assert "p5" not in df.index and info["excluded_by_flag"]["flag_missing_upload_date"] == 1
    assert df["y_log_interactions_rate"].notna().tolist() == [True, False, False, False, False]
    assert df["y_reach_rel_local"].notna().all()  # the engagement-only rules leave reach alone
    assert info["target_rows_unknown"]["log_interactions_rate"] == 2
    assert info["target_rows_excluded_by_flag"]["log_interactions_rate"] == 2


def test_account_target_encoder_is_out_of_fold():
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame({"social_account_id": rng.choice([f"a{i}" for i in range(10)], n), "deal_id": "d",
                      "platform": rng.choice(["tiktok", "youtube"], n), "_content": [f"c{i // 2}" for i in range(n)]})
    y = rng.normal(size=n)
    enc = fit_models.AccountTargetEncoder(random_state=0)
    e0 = enc.fit_transform(X, y)[fit_models.ACCOUNT_TE].to_numpy()
    y2 = y.copy()
    y2[0] += 100.0  # a row's own outcome (and its content's) never enters its own feature (fix 3)
    e1 = fit_models.AccountTargetEncoder(random_state=0).fit_transform(X, y2)[fit_models.ACCOUNT_TE].to_numpy()
    same_content = (X["_content"] == X["_content"][0]).to_numpy()
    assert np.allclose(e0[same_content], e1[same_content]) and not np.allclose(e0, e1)
    # transform: an unseen account gets its deal x platform mean
    new = pd.DataFrame({"social_account_id": ["zz"], "deal_id": ["d"], "platform": ["tiktok"], "_content": ["x"]})
    assert np.isclose(enc.transform(new)[fit_models.ACCOUNT_TE].iloc[0], y[X["platform"] == "tiktok"].mean())


def test_lockbox_is_opt_in_and_features_are_per_target(synth, tmp_path):
    res = fit_models.run(synth["features"], synth["members"], synth["outcomes"], selection=synth["selection"],
                         targets=fit_models.DEFAULT_TARGETS, out_dir=tmp_path, schemes=["content"],
                         models=["ridge"], n_splits=3, n_boot=12, perm_repeats=0, niche=False, threads=1, **QUIET)
    assert res["config"]["score_lockbox"] is False  # fix 4
    assert all("lockbox" not in tr for tr in res["targets"].values())
    assert "lockbox" not in set(pd.read_csv(tmp_path / "oof_predictions.csv")["scheme"])
    report = (tmp_path / "report.md").read_text()
    assert "lockbox not scored" in report and "not scored" in report
    assert set(res["features"]) == set(fit_models.DEFAULT_TARGETS)  # per-target feature lists (minor)
    assert all(fit_models.ACCOUNT_TE in f["A"] for f in res["features"].values())
    assert res["config"]["niche_base"] == "ridge"  # follows --models
    assert fit_models.resolve_niche_base(None, ["ridge", "hgb"]) == "hgb"
    with pytest.raises(ValueError):
        fit_models.resolve_niche_base("hgb", ["ridge"])


def test_multiplicity_and_centred_helpers():
    # BH (fix 6): q >= p, NaN stays out of the family
    p = np.array([0.001, 0.02, np.nan, 0.04, 0.5])
    q = fit_models.bh(p)
    assert np.isnan(q[2]) and np.all(q[[0, 1, 3, 4]] >= p[[0, 1, 3, 4]])
    assert np.allclose(q[[0, 1, 3, 4]], fit_models.bh(p[[0, 1, 3, 4]]))
    rng = np.random.default_rng(0)
    assert fit_models.boot_p(2.0, rng.normal(2.0, 0.5, 200)) < 1e-3 < fit_models.boot_p(0.2, rng.normal(0.2, 1, 200))
    # platform-centred R² (fix 7): predicting each platform's mean scores 0; ordinary R² would credit it
    g = np.repeat(["t", "i", "y"], 50)
    y = np.repeat([1.0, 0.0, -1.0], 50) + rng.normal(0, 0.3, 150)
    mu = pd.Series(y).groupby(g).transform("mean").to_numpy()
    assert np.isclose(fit_models.r2_centred(y, mu, g), 0.0) and fit_models.r2(y, mu) > 0.8


def test_cluster_rank_p_does_not_count_reposts_as_evidence():
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=30), rng.normal(size=30)
    y = y + 0.35 * x
    cl = np.arange(30)
    rep = np.repeat(np.arange(30), 8)  # every post copied 8x within its account (fix 7)
    from scipy import stats

    p1, p8 = fit_models.cluster_rank_p(x, y, cl), fit_models.cluster_rank_p(x[rep], y[rep], cl[rep])
    naive8 = stats.spearmanr(x[rep], y[rep])[1]
    assert abs(np.log(p8 / p1)) < 0.5 and naive8 < 0.05 < p8  # copies do not manufacture significance
    assert np.isnan(fit_models.cluster_rank_p(x[:8], y[:8], np.arange(8) % 3))  # too few clusters


def test_profile_penalty_is_selected_on_the_matrix_it_is_used_with(synth, monkeypatch):
    df, _, _ = fit_models.build_dataset(synth["features"], synth["members"], synth["outcomes"],
                                        ["reach_rel_local"], synth["selection"])
    d = df[(df["split"] == "train") & df["y_reach_rel_local"].notna()].reset_index(drop=True)
    m = fit_models.niche_mask(d, f"deal:{NICHE_DEAL}")
    y, w = d["y_reach_rel_local"].to_numpy(float), d["_w"].to_numpy(float)
    cols = fit_models.profile_columns(fit_models.feature_sets(d)["B"][1])
    seen = {}

    def spy(Z, r, groups, w, **kw):
        seen["Z"] = Z
        return 123.0

    monkeypatch.setattr(fit_models, "select_alpha", spy)
    prof = fit_models.niche_profile(d, m, y, w, cols, n_boot=0)
    X = fit_models._linear_global(d, y, w, cols)[1]  # fix 6: the globally standardised niche rows
    assert prof["alpha"] == 123.0 and np.allclose(seen["Z"], X[m][:, :len(cols)])


def test_pca_fit_excludes_lockbox_rows(tmp_path):
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (30, 12)), rng.normal(5, 1, (10, 12))]).astype(np.float32)
    pca, Z = build_features.fit_pca(X, 3, n_fit=30)  # fix 8
    assert Z.shape == (40, 3) and np.allclose(pca.mean_, X[:30].mean(0), atol=1e-5)
    sel = tmp_path / "selection.csv"
    pd.DataFrame({"video_id": ["a", "b", "c"], "split": ["train", "lockbox", "pilot"]}).to_csv(sel, index=False)
    assert build_features.lockbox_ids(sel) == {"b"}


def test_build_features_pca_exclusion_and_pool_fail_fast(synth, tmp_path):
    df, meta, _ = build_features.build(synth["out"], synth["roi"], n_jobs=1, n_pca=3, limit=12,
                                       pca_exclude={"c0001", "c0002"}, **QUIET)
    ok = df[df["status"] == "ok"].set_index("video_id")
    assert meta["pca"]["n_excluded_from_fit"] == 2 and meta["pca"]["n_fit"] == len(ok) - 2
    assert ok.loc[["c0001", "c0002"], "brain_pca_01"].notna().all()  # held out of the fit, still transformed
    # a broken ROI map fails in the parent, before any worker pool is started (minor)
    bad = tmp_path / "bad.npz"
    bad.write_bytes(b"not an npz")
    r = subprocess.run([sys.executable, str(ROOT / "tools/build_features.py"), "--out-root", synth["out"], "--out",
                        tmp_path / "f.parquet", "--roi-map", bad, "--n-jobs", "2"], capture_output=True, text=True,
                       timeout=60)
    assert r.returncode != 0 and "_init" in r.stderr and "build_features.py" in r.stderr


def test_build_features_several_out_roots_equal_one_and_refuse_mixed_precision(synth, tmp_path):
    """Pulled batches live in separate roots: building over them equals building over one merged root (the
    first completed copy of a clip wins), and clips from different video runtimes are never mixed."""
    import shutil

    a, b = tmp_path / "a", tmp_path / "b"
    for w in range(3):
        shutil.copytree(synth["out"] / f"worker-{w}", (a if w < 2 else b) / f"worker-{w}")
    dup = b / "worker-7"
    dup.mkdir()
    for f in ("c0000.json", "c0000.npz"):  # the same clip pulled twice: counted once
        shutil.copy(a / "worker-0" / f, dup / f)
    one, meta1, _ = build_features.build(synth["out"], synth["roi"], n_jobs=1, n_pca=5, **QUIET)
    two, meta2, _ = build_features.build([a, b], synth["roi"], n_jobs=1, n_pca=5, **QUIET)
    cols = [c for c in one.columns if c != "worker_id"]
    pd.testing.assert_frame_equal(one[cols], two[cols])
    assert meta2["out_root"] == [str(a), str(b)] and meta1["out_root"] == str(synth["out"])
    assert meta2["status_counts"] == meta1["status_counts"]
    _run(ROOT / "tools/build_features.py", "--out-root", a, "--out-root", b, "--out", tmp_path / "f.parquet",
         "--roi-map", synth["roi"], "--n-jobs", 1, "--n-pca", 5)
    assert len(pd.read_parquet(tmp_path / "f.parquet")) == len(one)
    # one clip recorded as the bf16 fast loop among unrecorded (= stock fp32) clips: refused, API and CLI
    p = b / "worker-2" / "c0002.json"
    p.write_text(json.dumps({**json.loads(p.read_text()),
                             "video_config": {"video_precision": "bf16", "fast_video": True}}))
    with pytest.raises(SystemExit, match="mixed video runtimes"):
        build_features.build([a, b], synth["roi"], n_jobs=1, n_pca=5, **QUIET)
    r = subprocess.run([sys.executable, str(ROOT / "tools/build_features.py"), "--out-root", a, "--out-root", b,
                        "--out", tmp_path / "g.parquet", "--roi-map", synth["roi"], "--n-jobs", "1"],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode != 0 and "mixed video runtimes" in r.stderr and not (tmp_path / "g.parquet").exists()
