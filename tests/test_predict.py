"""Serving path, offline: featurizer state + featurize_one, fit_models --save-model, tools/predict.py.

Synthetic worker outputs with a planted early-attention signal (the fixtures of test_features_models /
test_emb_control), two deals x two platforms, one fit_models CLI evaluation with --save-model and one without.
"""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))

pytest.importorskip("sklearn")

import build_features  # noqa: E402
import fit_models  # noqa: E402
import predict  # noqa: E402
from test_emb_control import write_emb  # noqa: E402
from test_features_models import QUIET, write_clip, write_roi  # noqa: E402
from tribe_research.brain.proxies import ProxySpec  # noqa: E402

REF_COMMIT = "1207848"  # the last commit of tools/build_features.py before the featurizer refactor
DEALS = ["deal-a", "deal-b"]
PLATS = ["tiktok", "instagram"]
N = 200
TARGETS = ["log_interactions_rate", "reach_rel_local"]


@pytest.fixture(scope="module", autouse=True)
def _one_thread():
    from threadpoolctl import threadpool_limits

    with threadpool_limits(1):
        yield


def _z(a):
    a = np.asarray(a, float)
    return (a - a.mean()) / a.std()


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    return make_study(tmp_path_factory.mktemp("serve"))


def make_study(tmp: Path) -> dict:
    """The synthetic study (worker outputs, features + featurizer, members/outcomes/selection) in ``tmp``.
    A plain function so tools/make_performance_samples.py can build the same fixture outside pytest."""
    roi = tmp / "roi.npz"
    groups, v = write_roi(roi)
    out = tmp / "outputs"
    for w in range(2):
        (out / f"worker-{w}").mkdir(parents=True)
    rng = np.random.default_rng(11)
    amps = rng.uniform(0.0, 5.0, N)
    vids = [f"c{i:03d}" for i in range(N)]
    for i, vid in enumerate(vids):
        d = out / f"worker-{i % 2}"
        write_clip(d, vid, groups, v, amps[i], rng.uniform(0, 3), rng)
        write_emb(d, vid, rng)
    z = _z(amps)
    rows, outcomes, selection = [], [], []
    for i, vid in enumerate(vids):
        deal = DEALS[i % 2]
        lockbox = rng.random() < 0.2
        anchor = None
        for plat in rng.choice(PLATS, int(rng.integers(1, 3)), replace=False):
            acc = int(rng.integers(0, 2))
            account = f"acc-{deal}-{plat}-{acc}"
            anchor = anchor or account
            vp = f"vp-{vid}-{plat}"
            rows.append({"video_id": vid, "vp_id": vp, "platform": plat, "deal_id": deal})
            outcomes.append({
                "id": vp, "deal_id": deal, "platform": plat, "social_account_id": account,
                "content_group": f"cg-{vid}",
                "upload_date": pd.Timestamp("2026-06-01", tz="UTC") + pd.Timedelta(days=int(rng.integers(0, 60))),
                "reach_rel_local": 0.8 * z[i] + rng.normal(0, 0.4), "local_baseline_level": "account_local",
                "interactions_rate_eb": float(np.exp(-4 + 0.6 * z[i] + 0.1 * acc + rng.normal(0, 0.25))),
                **{f: False for f in fit_models.EXCLUDE_IF_TRUE if f.startswith("flag_")},
            })
        selection.append({"video_id": vid, "deal_id": deal, "split": "lockbox" if lockbox else "train",
                          "incl_prob": 0.05 if lockbox else float(rng.uniform(0.3, 1.0)),
                          "anchor_account": anchor, "stratum": deal, "audio_mean_db": float(rng.normal(-20, 3)),
                          "width": 1080, "height": int(rng.choice([1920, 1350]))})
    sel = pd.DataFrame(selection)
    lock = set(sel.loc[sel["split"] == "lockbox", "video_id"])
    state: dict = {}
    df, meta, _ = build_features.build(out, roi, n_jobs=1, n_pca=4, pca_exclude=lock, state_out=state, **QUIET)
    feats = tmp / "features.parquet"
    df.to_parquet(feats, index=False)
    build_features.write_featurizer(tmp / "features", state, feats)
    paths = {"features": feats, "members": tmp / "members.csv", "outcomes": tmp / "outcomes.parquet",
             "selection": tmp / "selection.csv"}
    pd.DataFrame(rows).to_csv(paths["members"], index=False)
    pd.DataFrame(outcomes).to_parquet(paths["outcomes"], index=False)
    sel.to_csv(paths["selection"], index=False)
    # clips outside the study: a new one, plus scope variants of it
    new = tmp / "new"
    new.mkdir()
    write_clip_in(new, "n000", groups, v, 4.5, rng)
    return {"tmp": tmp, "roi": roi, "out": out, "df": df, "state": state, "sel": sel, "lock": lock,
            "paths": paths, "featurizer": tmp / "features.featurizer.json", "new": new, "groups": groups, "v": v}


def write_clip_in(d: Path, vid: str, groups, v, amp, rng, meta_update: dict | None = None) -> Path:
    (d / "worker-9").mkdir(exist_ok=True)  # write_clip reads the worker id from the directory name
    write_clip(d / "worker-9", vid, groups, v, amp, 0.0, rng)
    write_emb(d / "worker-9", vid, rng)
    if meta_update:
        p = d / "worker-9" / f"{vid}.json"
        p.write_text(json.dumps({**json.loads(p.read_text()), **meta_update}))
    return d / "worker-9" / f"{vid}.npz"


def _fit_cli(study, out_dir: Path, *extra, score_lockbox: bool = True):
    p = study["paths"]
    cmd = [sys.executable, ROOT / "tools/fit_models.py", "--features", p["features"], "--members", p["members"],
           "--outcomes", p["outcomes"], "--selection", p["selection"], "--out-dir", out_dir, "--targets", *TARGETS,
           "--schemes", "content", "account", "--models", "stack", "--n-splits", 3, "--n-boot", 30,
           "--perm-repeats", 0, "--no-niche", "--threads", 1, "--min-account-n", 10,
           *(["--score-lockbox"] if score_lockbox else []), *extra]
    return subprocess.run(list(map(str, cmd)), check=True, capture_output=True, text=True)


@pytest.fixture(scope="module")
def served(study):
    model = make_served(study)
    _fit_cli(study, study["tmp"] / "eval_plain", "--lockbox-ext", study["tmp"] / "absent.csv")
    return model


def make_served(study, name: str = "model", *extra, score_lockbox: bool = True) -> Path:
    """fit_models.py CLI evaluation (``eval_<name>``, or ``eval_saved`` for the default) + ``--save-model``
    into ``study["tmp"] / name``; ``extra`` goes to the CLI after the fixture's flags."""
    tmp = study["tmp"]
    ext = tmp / "lockbox_ext.csv"
    pd.DataFrame({"video_id": ["x-not-in-study"], "path": ["x.mp4"], "deal_id": ["deal-a"],
                  "duration_s": [10.0]}).to_csv(ext, index=False)
    eval_dir = tmp / ("eval_saved" if name == "model" else f"eval_{name}")
    _fit_cli(study, eval_dir, "--save-model", tmp / name, "--lockbox-ext", ext, *extra, score_lockbox=score_lockbox)
    return tmp / name


def _ctx(study, vid=None, **kw):
    s = study["sel"].set_index("video_id")
    row = s.loc[vid] if vid in s.index else s.iloc[0]
    ctx = {"deal_id": row["deal_id"], "platform": "tiktok", "width": 1080, "height": 1920,
           "audio_mean_db": -20.0}
    ctx.update(kw)
    return ctx


def _clip(study, vid):
    return next(study["out"].glob(f"worker-*/{vid}.npz"))


# ── build_features: batch unchanged, featurize_one == batch row ───────────


def _reference_module(name="build_features"):
    try:
        src = subprocess.run(["git", "-C", str(ROOT), "show", f"{REF_COMMIT}:tools/{name}.py"],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"git or commit {REF_COMMIT} unavailable")
    mod = types.ModuleType(f"{name}_ref")
    mod.__file__ = str(ROOT / "tools" / f"_{name}_ref.py")  # its ROOT/defaults resolve like the real one
    sys.modules[mod.__name__] = mod  # dataclasses look their module up
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    return mod


def test_batch_table_byte_identical_to_pre_refactor(tmp_path):
    ref = _reference_module()
    roi = tmp_path / "roi.npz"
    groups, v = write_roi(roi)
    d = tmp_path / "out" / "worker-0"
    d.mkdir(parents=True)
    rng = np.random.default_rng(5)
    for i in range(24):
        write_clip(d, f"x{i:02d}", groups, v, rng.uniform(0, 3), rng.uniform(0, 3), rng)
        if i < 20:
            write_emb(d, f"x{i:02d}", rng, ("video", "audio") if i % 4 == 0 else ("video", "audio", "text"))
    (d / "x21.emb.npz").write_bytes(b"not a zip")
    for excl in (None, {"x03", "x07"}):
        a, ma, _ = ref.build(tmp_path / "out", roi, n_jobs=1, n_pca=4, pca_exclude=excl, **QUIET)
        b, mb, _ = build_features.build(tmp_path / "out", roi, n_jobs=1, n_pca=4, pca_exclude=excl,
                                        state_out={}, **QUIET)
        pd.testing.assert_frame_equal(a, b, check_exact=True)
        ba, bb = io.BytesIO(), io.BytesIO()
        a.to_parquet(ba, index=False)
        b.to_parquet(bb, index=False)
        assert ba.getvalue() == bb.getvalue()
        for k in ("columns", "pca", "emb", "status_counts"):
            assert json.dumps(ma.get(k), sort_keys=True, default=str) == json.dumps(mb.get(k), sort_keys=True,
                                                                                   default=str)


def test_featurize_one_matches_the_batch_row(study):
    st = build_features.load_featurizer(study["featurizer"])
    info = st["info"]
    assert info["pca_exclude"]["used"] and info["pca_exclude"]["n_ids"] == len(study["lock"])
    assert info["blocks"]["brain"]["n_fit"] == N - len(study["lock"])
    batch = study["df"].copy().set_index("video_id")
    pca = [c for c in info["feature_columns"] if "_pca_" in c]
    exact = [c for c in info["columns"] if c not in pca and c != "video_id"]
    train_vid = study["sel"].loc[study["sel"]["split"] == "train", "video_id"].iloc[0]
    for vid in (train_vid, sorted(study["lock"])[0]):
        one = build_features.featurize_one(_clip(study, vid), {}, st)
        assert list(one.columns) == info["columns"]
        want = batch.loc[[vid]].reset_index()[info["columns"]].copy()
        assert (one.dtypes == want.dtypes).all()
        pd.testing.assert_frame_equal(one[exact], want[exact], check_exact=True)
        # PCA: batch uses fit_transform (U*S) for fit rows and transform for the rest; float32 round-off only
        np.testing.assert_allclose(one[pca].to_numpy(float), want[pca].to_numpy(float), rtol=0, atol=1e-4)


def test_load_featurizer_refuses_a_changed_artefact(study, tmp_path):
    for f in ("features.featurizer.json", "features.featurizer.npz"):
        shutil.copy(study["tmp"] / f, tmp_path / f)
    info = json.loads((tmp_path / "features.featurizer.json").read_text())
    (tmp_path / "features.featurizer.json").write_text(json.dumps({**info, "npz_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="sha256"):
        build_features.load_featurizer(tmp_path / "features.featurizer.json")
    (tmp_path / "features.featurizer.json").write_text(json.dumps({**info, "features_version": "old"}))
    with pytest.raises(ValueError, match="features_version"):
        build_features.load_featurizer(tmp_path / "features.featurizer.json")


# ── fit_models --save-model ───────────────────────────────────────────────


def test_build_dataset_identical_to_pre_refactor(study, tmp_path):
    """The follower/post-time/aspect helpers the predictor shares, on their edge cases, vs the pre-refactor code;
    plus a whole tiny evaluation (oof_predictions.csv bytes, metrics without runtime)."""
    ref = _reference_module("fit_models")
    p = study["paths"]
    feats, members = fit_models.read_table(p["features"]), fit_models.read_table(p["members"])
    out = fit_models.read_table(p["outcomes"]).drop(columns="upload_date")
    rng = np.random.default_rng(4)
    n = len(out)
    out["follower_count"] = rng.choice([0, np.nan, -5, 1, 999, 12345678, 3.5e9], n)
    stamps = ["2026-06-01T13:45:00Z", "2026-06-02", "2026-06-03T08:10:00+02:00", "2026-06-04 23:59:59",
              "not a date", None, "2026-06-05T00:30:00-05:00"]
    out["posted_at"] = [stamps[i % len(stamps)] for i in range(n)]
    sel = fit_models.read_table(p["selection"])
    sel.loc[sel.index[:3], "width"] = 0
    sel.loc[sel.index[3:5], "height"] = np.nan
    a, ia, _ = ref.build_dataset(feats, members, out, TARGETS, sel)
    b, ib, _ = fit_models.build_dataset(feats, members, out, TARGETS, sel)
    assert {"base_follower_bucket", "base_post_weekday", "base_post_hour", "base_aspect"} <= set(b.columns)
    assert b["base_aspect"].isna().sum() >= 3 and b["base_follower_bucket"].isna().any()
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    assert json.dumps(ia, sort_keys=True, default=str) == json.dumps(ib, sort_keys=True, default=str)
    kw = dict(selection=sel, targets=TARGETS, schemes=["content"], models=["ridge"], n_splits=3, n_boot=20,
              perm_repeats=0, niche=False, threads=1, **QUIET)
    ra = ref.run(feats, members, out, out_dir=tmp_path / "a", **kw)
    rb = fit_models.run(feats, members, out, out_dir=tmp_path / "b", moment_cols=True, **kw)  # the pre-refactor brain block kept UI moments
    oa, ob = (pd.read_csv(tmp_path / d / "oof_predictions.csv", dtype=str) for d in ("a", "b"))
    assert list(ob.columns) == [*oa.columns[:6], "stratum", *oa.columns[6:]]  # 900001b added the stratum only
    pd.testing.assert_frame_equal(oa, ob.drop(columns="stratum"), check_exact=True)
    for r in (ra, rb):
        r.pop("runtime_s")
    assert rb["config"].pop("moment_cols") is True  # the only config key added since
    assert json.dumps(ra, sort_keys=True, default=str).replace(str(tmp_path / "a"), "") == \
        json.dumps(rb, sort_keys=True, default=str).replace(str(tmp_path / "b"), "")


def test_save_model_leaves_the_evaluation_unchanged(served, study):
    a, b = study["tmp"] / "eval_saved", study["tmp"] / "eval_plain"
    assert (a / "oof_predictions.csv").read_bytes() == (b / "oof_predictions.csv").read_bytes()
    ma, mb = (json.loads((d / "metrics.json").read_text()) for d in (a, b))
    for m in (ma, mb):
        m.pop("runtime_s")
    ma["config"].pop("out_dir", None), mb["config"].pop("out_dir", None)
    assert json.dumps(ma, sort_keys=True).replace(str(a), "") == json.dumps(mb, sort_keys=True).replace(str(b), "")
    assert not (b / "manifest.json").exists()


def test_saved_model_is_the_lockbox_scoring_model(served, study):
    """Refit on all train rows = the model --score-lockbox scored the lockbox with (same fit_predict)."""
    import joblib

    man = json.loads((served / "manifest.json").read_text())
    assert man["artefact_version"] == fit_models.ARTEFACT_VERSION and man["feature_set"] == "BE"
    p = study["paths"]
    df, _, _ = fit_models.build_dataset(fit_models.read_table(p["features"]), fit_models.read_table(p["members"]),
                                        fit_models.read_table(p["outcomes"]), TARGETS,
                                        fit_models.read_table(p["selection"]))
    oof = pd.read_csv(study["tmp"] / "eval_saved" / "oof_predictions.csv", dtype={"vp_id": str, "video_id": str})
    for t in TARGETS:
        tm = man["targets"][t]
        est = joblib.load(served / tm["files"]["model"])  # unpickles as fit_models.<Class> (CLI run as __main__)
        assert type(est).__module__ == "fit_models"
        lb = df[(df["split"] == "lockbox") & np.isfinite(df[f"y_{t}"])].reset_index(drop=True)
        got = pd.Series(est.predict(lb[tm["input_columns"]]), index=lb["vp_id"])
        want = oof[(oof["target"] == t) & (oof["scheme"] == "lockbox")].set_index("vp_id")["pred_BE_stack"]
        np.testing.assert_allclose(got.loc[want.index].to_numpy(), want.to_numpy(), rtol=1e-9, atol=1e-9)
        for k, f in tm["files"].items():
            assert fit_models._sha256(served / f) == tm["sha256"][k]
    assert man["inputs"]["features"]["sha256"] == fit_models._sha256(p["features"])
    lbc = man["targets"][predict.PRIMARY]["metrics"]["lockbox_contents"]
    assert lbc["n"] > 0 and lbc["includes_extension"] is False  # the fixture's extension id is not in the study
    assert man["inputs"]["featurizer"]["sha256"] == fit_models._sha256(study["featurizer"])


def test_reference_tables_never_hold_lockbox_rows(served, study):
    lock = set(json.loads((served / "lockbox_ids.json").read_text()))
    assert study["lock"] <= lock and "x-not-in-study" in lock  # selection lockbox + the extension
    train = set(json.loads((served / "train_video_ids.json").read_text()))
    assert not lock & train
    man = json.loads((served / "manifest.json").read_text())
    for t in TARGETS:
        files = man["targets"][t]["files"]
        ref = pd.read_csv(served / files["reference"], dtype={"video_id": str})
        acc = pd.read_csv(served / files["reference_accounts"], dtype={"video_id": str})
        assert len(ref) and not set(ref["video_id"]) & lock and set(ref["video_id"]) <= train
        assert not set(acc["video_id"]) & lock
        assert not ref.duplicated(["deal_id", "platform", "video_id"]).any()  # content-level
        assert ref["obs_pct"].between(0, 1).all()
        # weights are 1/incl_prob of the selection
        ip = study["sel"].set_index("video_id")["incl_prob"]
        np.testing.assert_allclose(ref["w"], 1 / ip.loc[ref["video_id"]].to_numpy())


def test_save_model_refuses_a_lockbox_extension_train_row(served, study, tmp_path):
    p = study["paths"]
    res = json.loads((study["tmp"] / "eval_saved" / "metrics.json").read_text())
    res["models"] = ["stack"]
    train_vid = study["sel"].loc[study["sel"]["split"] == "train", "video_id"].iloc[0]
    with pytest.raises(SystemExit, match="lockbox-extension"):
        fit_models.save_model(fit_models.read_table(p["features"]), fit_models.read_table(p["members"]),
                              fit_models.read_table(p["outcomes"]), res=res, eval_dir=study["tmp"] / "eval_saved",
                              model_dir=tmp_path / "m", selection=fit_models.read_table(p["selection"]),
                              lockbox_ext={train_vid}, log=lambda *a: None)
    assert not (tmp_path / "m" / "manifest.json").exists()


# ── predict.py ────────────────────────────────────────────────────────────


def _status(served):
    man = json.loads((served / "manifest.json").read_text())
    return predict.model_status_from(man["targets"][predict.PRIMARY]["metrics"], "BE", "stack")


def test_round_trip_new_clip_numbers_and_drivers(served, study):
    import joblib

    status, why = _status(served)
    assert status in ("research_preview", "validated"), why  # the planted signal passes the GO rule
    npz = study["new"] / "worker-9" / "n000.npz"
    ctx = _ctx(study, deal_id="deal-a", platform="tiktok", account_id="acc-deal-a-tiktok-0")
    blk = predict.predict_performance(npz, ctx, study["featurizer"], served)
    assert blk["model_status"] == status and blk["reason"] is None
    assert blk["clip_in_training"] == "no" and blk["retrospective"] is False
    e = blk["engagement"]
    assert e["target"] == predict.PRIMARY and e["reference_n"] >= 30
    assert 0 <= e["percentile_deal_platform"] <= 1
    lo, hi = e["likely_range"]
    assert 0 <= lo <= hi <= 1
    assert e["percentile_account"] is not None and blk["context"]["account_level"] is True
    assert e["validation"]["scheme"] in ("content_cv", "lockbox")
    # the fixture's scored lockbox has no extension contents: never "supported", however strong
    assert blk["brain_claim"] == "directional" and any("not the confirmatory set" in w for w in blk["warnings"])
    assert blk["model_version"].startswith("perf-stack-BE_v1+")
    assert blk["provenance"]["features_version"] == build_features.FEATURES_VERSION
    assert any("SYNTHETIC" in w for w in blk["warnings"])
    fams = {d["family"] for d in blk["drivers"]}
    assert fams == {"metadata", "account_history", "content_embedding", "brain_response"}
    brain = next(d for d in blk["drivers"] if d["family"] == "brain_response")
    assert {x["channel"] for x in brain["brain_detail"]} <= {c.key for c in ProxySpec.load().channels} | {"other"}
    assert abs(sum(x["contribution"] for x in brain["brain_detail"]) - brain["contribution"]) < 1e-4
    # drivers are an exact decomposition: they sum to prediction − reference-average prediction
    man = json.loads((served / "manifest.json").read_text())
    tm = man["targets"][predict.PRIMARY]
    st = build_features.load_featurizer(study["featurizer"])
    X = predict.design_row(build_features.featurize_one(npz, ctx, st), ctx)
    p = float(joblib.load(served / tm["files"]["model"]).predict(X[tm["input_columns"]])[0])
    rt = pd.read_csv(served / tm["files"]["reference_terms"]).set_index(["deal_id", "platform"])
    assert abs(sum(d["contribution"] for d in blk["drivers"]) - (p - rt.loc[("deal-a", "tiktok"), "pred_mean"])) < 1e-4
    # the percentile is the weighted rank of that prediction among the reference out-of-fold predictions
    ref = pd.read_csv(served / tm["files"]["reference"])
    ref = ref[(ref["deal_id"] == "deal-a") & (ref["platform"] == "tiktok")]
    assert e["reference_n"] == len(ref)
    assert e["percentile_deal_platform"] == round(predict.weighted_percentile(p, ref["pred"], ref["w"]), 4)
    # the CLI writes the same block (model_version etc. included)
    out = study["tmp"] / "perf.json"
    subprocess.run([sys.executable, str(ROOT / "tools/predict.py"), str(npz), "--featurizer", str(study["featurizer"]),
                    "--model-dir", str(served), "--deal-id", "deal-a", "--platform", "tiktok", "--account-id",
                    "acc-deal-a-tiktok-0", "--width", "1080", "--height", "1920", "--audio-mean-db", "-20",
                    "--out", str(out)], check=True, capture_output=True, text=True)
    assert json.loads(out.read_text()) == json.loads(json.dumps(blk))


def test_training_and_lockbox_clips(served, study):
    sel = study["sel"]
    ref = pd.read_csv(served / "reference_log_interactions_rate.csv", dtype={"video_id": str})
    single = ref.groupby("video_id")["platform"].transform("size") == 1  # posted on one platform only
    r = ref[single].iloc[0]
    blk = predict.predict_performance(_clip(study, r["video_id"]), _ctx(study, r["video_id"], platform=r["platform"]),
                                      study["featurizer"], served)
    assert blk["clip_in_training"] == "train_oof" and blk["retrospective"] is True
    assert blk["drivers"] == [] and any("drivers omitted" in w for w in blk["warnings"])
    here = ref[(ref["deal_id"] == r["deal_id"]) & (ref["platform"] == r["platform"])]
    assert blk["engagement"]["reference_n"] == len(here) - 1  # ranked without itself
    rest = here[here["video_id"] != r["video_id"]]
    assert blk["engagement"]["percentile_deal_platform"] == round(
        predict.weighted_percentile(r["pred"], rest["pred"], rest["w"]), 4)
    # a training clip asked about in a stratum it has no out-of-fold prediction in: out of scope, no number
    other = next(p for p in PLATS if p != r["platform"])
    blk = predict.predict_performance(_clip(study, r["video_id"]), _ctx(study, r["video_id"], platform=other),
                                      study["featurizer"], served)
    assert blk["model_status"] == "out_of_scope" and blk["engagement"] is None
    lvid = sorted(study["lock"])[0]
    blk = predict.predict_performance(_clip(study, lvid), _ctx(study, lvid), study["featurizer"], served)
    assert blk["clip_in_training"] == "lockbox" and blk["engagement"] is not None
    assert any("lockbox" in w.lower() for w in blk["warnings"])
    assert "y" not in json.dumps(blk["engagement"]).split('"')  # no observed outcome field
    assert sel.set_index("video_id").loc[lvid, "split"] == "lockbox"


def _scope(study, served, vid, amp=2.0, meta=None, **ctx):
    rng = np.random.default_rng(sum(map(ord, vid)))
    d = study["tmp"] / f"scope-{vid}"
    d.mkdir(exist_ok=True)
    npz = write_clip_in(d, vid, study["groups"], study["v"], amp, rng, meta)
    return predict.predict_performance(npz, _ctx(study, **{"deal_id": "deal-a", **ctx}), study["featurizer"], served)


@pytest.mark.parametrize("case,meta,ctx,match", [
    ("long", {"duration_s": 120.0}, {}, "Duration"),
    ("short", {"duration_s": 2.0}, {}, "Duration"),
    ("deal", None, {"deal_id": "deal-zzz"}, "Unknown deal"),
    ("plat", None, {"platform": "youtube"}, "no training clips for this deal on youtube"),
    ("mute", {"modalities": {"event_counts": {"Video": 12, "Word": 3}}}, {}, "No audio"),
    ("mutectx", None, {"has_audio": False}, "No audio"),
    ("lang", {"transcript": {"detected_language": "es", "language_probability": 0.95}}, {}, "non-English"),
    ("rt", {"tribe_commit": "other"}, {}, None),
])
def test_out_of_scope_never_has_numbers(served, study, case, meta, ctx, match):
    if _status(served)[0] == "not_trained":
        pytest.fail("fixture model did not pass the GO rule")
    blk = _scope(study, served, f"s-{case}", meta=meta, **ctx)
    if case == "rt":  # the training clips recorded no tribe_commit: parity unchecked, a warning, still scored
        assert blk["model_status"] != "out_of_scope" and any("tribe_commit" in w for w in blk["warnings"])
        return
    assert blk["model_status"] == "out_of_scope" and match in blk["reason"]
    assert blk["engagement"] is None and blk["reach"] is None and blk["drivers"] == []


def test_runtime_mismatch_is_out_of_scope(served, study, tmp_path):
    for f in ("features.featurizer.json", "features.featurizer.npz"):
        shutil.copy(study["tmp"] / f, tmp_path / f)
    fz = tmp_path / "features.featurizer.json"
    info = json.loads(fz.read_text())
    info["training_runtime"]["video_precision"] = {json.dumps("bf16"): 150}
    fz.write_text(json.dumps(info))
    st = build_features.load_featurizer(fz)
    rng = np.random.default_rng(3)
    npz = write_clip_in(tmp_path, "r000", study["groups"], study["v"], 2.0, rng,
                        {"video_config": {"video_precision": "fp32"}})
    blk = predict.predict_performance(npz, _ctx(study, deal_id="deal-a"), st, served)
    assert blk["model_status"] == "out_of_scope" and "video_precision" in blk["reason"]
    info["features_sha256"] = "0" * 64  # a featurizer from another feature table: refuse outright
    st["info"] = info
    with pytest.raises(ValueError, match="parity"):
        predict.predict_performance(npz, _ctx(study, deal_id="deal-a"), st, served)


def test_not_trained_has_no_numbers(served, study, tmp_path):
    npz = study["new"] / "worker-9" / "n000.npz"
    blk = predict.predict_performance(npz, _ctx(study, deal_id="deal-a"), study["featurizer"], tmp_path / "none")
    assert blk["model_status"] == "not_trained" and blk["model_version"] is None
    assert blk["engagement"] is None and blk["reach"] is None and blk["drivers"] == []
    failed = tmp_path / "failed"
    shutil.copytree(served, failed)
    man = json.loads((failed / "manifest.json").read_text())
    man["targets"][predict.PRIMARY]["metrics"]["go"]["content"] = {"point": 0.01, "ci": [-0.01, 0.03]}
    (failed / "manifest.json").write_text(json.dumps(man))
    blk = predict.predict_performance(npz, _ctx(study, deal_id="deal-zzz", has_audio=False), study["featurizer"],
                                      failed)
    assert blk["model_status"] == "not_trained" and "GO rule" in blk["reason"]  # before any scope check
    assert blk["engagement"] is None and blk["reach"] is None and blk["drivers"] == []
    assert blk["model_version"] == man["model_version"]


def _with_metrics(served, dst: Path, go=None, lockbox=None) -> Path:
    """A copy of the served model with its primary target's GO (and optionally lockbox) metrics replaced."""
    shutil.copytree(served, dst)
    man = json.loads((dst / "manifest.json").read_text())
    m = man["targets"][predict.PRIMARY]["metrics"]
    if go is not None:
        m["go"]["content"] = {"point": go[0], "ci": [go[1], go[2]]}
        m["go"]["account"] = {"point": 0.01, "ci": [None, None]}
    if lockbox is not None:
        m["served_lockbox"] = None if lockbox == "none" else {"point": lockbox[0], "ci": [lockbox[1], lockbox[2]]}
    (dst / "manifest.json").write_text(json.dumps(man))
    return dst


@pytest.mark.parametrize("go,lockbox,want", [
    ((0.01, -0.01, 0.03), "none", "not_trained"),  # GO fails -> preliminary only with the opt-in
    ((0.05, -0.01, 0.09), "none", "research_preview"),
    ((0.05, -0.01, 0.09), (0.2, 0.05, 0.3), "validated"),
])
@pytest.mark.parametrize("allow", [False, True])
def test_preliminary_only_with_the_opt_in_and_never_after_go(served, study, tmp_path, go, lockbox, want, allow):
    model = _with_metrics(served, tmp_path / "m", go=go, lockbox=lockbox)
    npz = study["new"] / "worker-9" / "n000.npz"
    ctx = _ctx(study, deal_id="deal-a", has_audio=True)
    blk = predict.predict_performance(npz, ctx, study["featurizer"], model, allow_preliminary=allow)
    man = json.loads((model / "manifest.json").read_text())
    n = man["targets"][predict.PRIMARY]["n_train_contents"]
    if want == "not_trained" and allow:
        assert blk["model_status"] == "preliminary" and blk["validated"] is False and blk["reason"] is None
        assert blk["n_train"] == n and blk["caption"] == predict.PRELIMINARY_CAPTION.format(n=n)
        assert f"trained on {n} clips, not validated" in blk["caption"] and "not a forecast" in blk["caption"]
        assert blk["brain_claim"] == "not_tested"  # the fixture's CV rule says directional; not for preliminary
        e = blk["engagement"]
        assert 0 <= e["percentile_deal_platform"] <= 1 and e["likely_range"] is not None and blk["drivers"]
        assert e["confidence"] == "low"
        assert e["validation"] == {"scheme": "none", "within_stratum_spearman": None, "ci95": [None, None]}
        assert blk["reach"] is None  # engagement only, whatever reach's own metrics say
        text = json.dumps(blk)
        assert "GO rule" not in text and "0.010" not in text  # no CV estimate travels with the clip
    else:
        assert blk["model_status"] == want and blk["model_status"] != "preliminary"
        assert blk["caption"] is None and blk["validated"] is (want == "validated")
        if want == "not_trained":
            assert blk["engagement"] is None and blk["n_train"] is None and "GO rule" in blk["reason"]
        else:
            assert blk["n_train"] == n and blk["engagement"]["validation"]["scheme"] != "none"


def test_preliminary_keeps_scope_checks_and_cli_flag(served, study, tmp_path):
    model = _with_metrics(served, tmp_path / "m", go=(0.0, -0.05, 0.05))
    npz = study["new"] / "worker-9" / "n000.npz"
    blk = predict.predict_performance(npz, _ctx(study, deal_id="deal-zzz", has_audio=True), study["featurizer"],
                                      model, allow_preliminary=True)
    assert blk["model_status"] == "out_of_scope" and "Unknown deal" in blk["reason"]
    assert blk["engagement"] is None and blk["n_train"] is None and blk["caption"] is None
    lvid = sorted(study["lock"])[0]
    blk = predict.predict_performance(_clip(study, lvid), _ctx(study, lvid, has_audio=True), study["featurizer"],
                                      model, allow_preliminary=True)
    assert blk["model_status"] == "preliminary" and blk["clip_in_training"] == "lockbox" and blk["retrospective"]
    out = tmp_path / "perf.json"
    args = [sys.executable, str(ROOT / "tools/predict.py"), str(npz), "--featurizer", str(study["featurizer"]),
            "--model-dir", str(model), "--deal-id", "deal-a", "--platform", "tiktok", "--width", "1080",
            "--height", "1920", "--audio-mean-db", "-20", "--out", str(out)]
    subprocess.run(args, check=True, capture_output=True, text=True)
    assert json.loads(out.read_text())["model_status"] == "not_trained"  # the default is unchanged
    subprocess.run([*args, "--allow-preliminary"], check=True, capture_output=True, text=True)
    assert json.loads(out.read_text())["model_status"] == "preliminary"


def test_a_preliminary_block_needs_n_train():
    with pytest.raises(ValueError, match="n_train"):
        predict._block("preliminary", None)
    assert predict._block("not_trained", "x", n_train=5)["n_train"] is None  # never a number without numbers


# ── rules on hand-built metrics ───────────────────────────────────────────


def _m(go=(0.05, -0.01, 0.09), acct=0.01, served_go=None, lockbox=None, brain=None, brain_lb=None, stage2=True):
    def e(t):
        return None if t is None else {"point": t[0], "ci": [t[1], t[2]]}

    return {"go": {"content": e(go), "account": None if acct is None else {"point": acct, "ci": [None, None]}},
            "served_go": {"content": e(served_go), "account": {"point": 0.0, "ci": [None, None]}},
            "served_lockbox": e(lockbox), "brain": {"content": e(brain), "lockbox": e(brain_lb)},
            "lockbox_contents": {"includes_extension": stage2}}


@pytest.mark.parametrize("m,fs,model,want", [
    (_m(), "BE", "stack", "research_preview"),
    (_m(go=(0.02, -0.029, 0.07)), "BE", "stack", "research_preview"),
    (_m(go=(0.019, 0.0, 0.05)), "BE", "stack", "not_trained"),  # point below +0.02
    (_m(go=(0.05, -0.03, 0.1)), "BE", "stack", "not_trained"),  # CI lower not > −0.03
    (_m(go=(0.05, None, None)), "BE", "stack", "not_trained"),  # no CI (too few bootstrap draws)
    (_m(acct=-0.001), "BE", "stack", "not_trained"),  # account guard
    (_m(acct=None), "BE", "stack", "not_trained"),  # account scheme not run
    (_m(lockbox=(0.2, 0.05, 0.3)), "BE", "stack", "validated"),
    (_m(lockbox=(0.2, 0.0, 0.3)), "BE", "stack", "research_preview"),
    (_m(go=(0.0, -0.1, 0.1), lockbox=(0.3, 0.2, 0.4)), "BE", "stack", "not_trained"),  # lockbox can't skip stage 1
    (_m(served_go=(0.05, 0.0, 0.1)), "E", "stack", "research_preview"),
    (_m(served_go=None), "E", "stack", "not_trained"),
    ({}, "BE", "stack", "not_trained"),
])
def test_model_status_rules(m, fs, model, want):
    assert predict.model_status_from(m, fs, model)[0] == want


@pytest.mark.parametrize("m,want", [
    (_m(), "not_tested"),
    (_m(brain=(0.01, -0.02, 0.04)), "directional"),
    (_m(brain=(-0.01, -0.04, 0.02)), "not_supported"),
    (_m(brain=(0.01, None, None)), "not_tested"),
    (_m(brain=(0.01, -0.02, 0.04), brain_lb=(0.05, 0.01, 0.09)), "supported"),
    (_m(brain=(0.01, -0.02, 0.04), brain_lb=(0.05, -0.01, 0.09)), "not_supported"),
    # the sealed 225 alone is not the confirmatory set: fall back to the stage-1 CV rule
    (_m(brain=(0.01, -0.02, 0.04), brain_lb=(0.05, 0.01, 0.09), stage2=False), "directional"),
    (_m(brain=(-0.01, -0.04, 0.02), brain_lb=(0.05, 0.01, 0.09), stage2=False), "not_supported"),
    ({}, "not_tested"),
])
def test_brain_claim_rules(m, want):
    assert predict.brain_claim_from(m) == want


def test_rank_helpers():
    ref, w = np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, 1.0, 1.0, 5.0])
    assert predict.weighted_percentile(0.0, ref, w) == 0.0
    assert predict.weighted_percentile(9.0, ref, w) == 1.0
    assert predict.weighted_percentile(2.0, ref, w) == pytest.approx(1.5 / 8)
    assert predict.weighted_percentile(3.5, ref, np.ones(4)) == 0.75
    lo, hi = predict.likely_range(0.0, np.linspace(0, 1, 100), np.linspace(0, 1, 100), np.ones(100))
    assert lo < hi <= 0.3  # the 30 nearest predictions are the lowest ones here
