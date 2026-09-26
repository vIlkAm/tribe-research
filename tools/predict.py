#!/usr/bin/env python3
"""Score one clip's worker output: the proposed ``performance`` block (docs/PRODUCT_PIPELINE.md, section 3).

Inputs: the clip's ``<vid>.npz`` / ``.json`` (``.emb.npz`` next to them, if exported), a context (deal and
platform at least), the featurizer state from tools/build_features.py (``<stem>.featurizer.json``) and a model
directory from ``tools/fit_models.py --save-model``. Output: the block as JSON (stdout or ``--out``).

    python tools/predict.py outputs/worker-0/<vid>.npz \\
        --featurizer results/features/features.featurizer.json --model-dir results/models/served \\
        --deal-id <deal> --platform tiktok [--account-id <id>] [--follower-count N] [--posted-at ISO] \\
        [--width W --height H] [--audio-mean-db DB] [--no-audio] [--context ctx.json] [--out perf.json]

Rules (the doc's; interpretations are marked in ``model_status_from`` / ``brain_claim_from``):

- ``not_trained`` (no model, or the pre-registered stage-1 GO rule not passed): no numbers at all.
- ``preliminary`` (opt-in only: ``--allow-preliminary`` / ``allow_preliminary=True``): a saved model that did NOT
  pass the GO rule is scored anyway so the frontend has real-format numbers to build against. ``validated`` is
  false, ``n_train`` and the mandatory ``caption`` say how little it saw, ``brain_claim`` is ``not_tested``, the
  validation numbers and the GO reason (which quotes CV estimates) are left out, confidence is ``low`` and reach
  is omitted. Without the opt-in the same model is ``not_trained``; a model that passes
  GO is never ``preliminary``. It never implies the stage-1 GO decision.
- ``out_of_scope`` (duration outside 5-90 s, unknown deal or deal x platform, no audio, confident non-English,
  a train/serve runtime mismatch, a training clip without an out-of-fold prediction here): no numbers, a reason.
- ``research_preview`` / ``validated``: the percentile is the weighted (1/incl_prob) mid-rank of the clip's
  predicted score among the deal x platform reference contents' out-of-fold predictions (null below 30); the
  likely range is the weighted P10-P90 of the observed percentiles of the reference contents predicted closest.
- A training clip gets its own out-of-fold prediction and is left out of its reference; its drivers are omitted
  (they would be in-sample). A lockbox clip is scored normally; no observed outcome is ever part of the block.

Every number is relative to one deal on one platform, a model prediction, and not a guarantee.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_features  # noqa: E402
import fit_models  # noqa: E402  (the saved models unpickle as fit_models.<Class>)

PLATFORMS = ("tiktok", "instagram", "youtube")
PRIMARY, REACH = "log_interactions_rate", "reach_rel_local"
MIN_DURATION_S, MAX_DURATION_S = 5.0, 90.0  # training scope (docs/PRODUCT_PIPELINE.md validation table)
GO_POINT, GO_CI_FLOOR = 0.02, -0.03  # docs/PREREGISTRATION.md stage-1 scaling rule (BE − A)
MIN_REFERENCE_N = 30
NEIGHBOUR_SHARE, MIN_NEIGHBOURS = 0.2, 30  # likely range: the closest max(30, 20 %) reference contents
LOW_CONFIDENCE_N = 100
SCORED = ("research_preview", "validated", "preliminary")
PRELIMINARY_CAPTION = ("Preliminary model — trained on {n} clips, not validated. Illustrative of the format, "
                       "not a forecast.")
FAMILY_LABELS = {"content_embedding": "Video, audio and text content", "brain_response": "Predicted brain response",
                 "metadata": "Duration, format, posting time", "account_history": "Deal/account typical level"}
ACCOUNT_TERMS = ("num__" + fit_models.ACCOUNT_TE, "num__base_follower_bucket")


# ── status rules ──────────────────────────────────────────────────────────


def _go(g: dict | None, label: str) -> tuple[bool, str | None]:
    """Stage-1 GO: content-CV Δ within-stratum ρ point >= +0.02 and CI lower > −0.03, and the account-scheme
    point >= 0. Missing pieces (scheme not run, no CI at <= 10 bootstrap draws) fail closed."""
    g = g or {}
    c, a = g.get("content") or {}, g.get("account") or {}
    ci = c.get("ci") or [None, None]
    if c.get("point") is None or ci[0] is None:
        return False, f"no stage-1 content-CV estimate with a CI for {label}"
    if c["point"] < GO_POINT or ci[0] <= GO_CI_FLOOR:
        return False, (f"stage-1 {label} = {c['point']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] does not pass the "
                       f"pre-registered GO rule (point >= {GO_POINT:+.2f}, CI lower > {GO_CI_FLOOR:+.2f})")
    if a.get("point") is None:
        return False, f"the account-scheme guard for {label} was not run"
    if a["point"] < 0:
        return False, f"account-scheme {label} = {a['point']:+.3f} < 0 (the GO guard)"
    return True, None


def model_status_from(metrics: dict, feature_set: str, model: str) -> tuple[str, str | None]:
    """not_trained unless the literal pre-registered BE_stack − A_stack passes the GO rule. Interpretation: a
    served model other than BE/stack must also pass the same rule on its own contrast (fail closed). validated
    additionally needs the served model's own lockbox within-stratum ρ CI lower bound > 0."""
    ok, why = _go(metrics.get("go"), "BE − A (stack)")
    if not ok:
        return "not_trained", why
    if (feature_set, model) != ("BE", "stack"):
        ok, why = _go(metrics.get("served_go"), f"{feature_set} − A ({model})")
        if not ok:
            return "not_trained", why
    lb = metrics.get("served_lockbox") or {}
    if (lb.get("ci") or [None])[0] is not None and lb["ci"][0] > 0:
        return "validated", None
    return "research_preview", None


def confirmatory_lockbox(metrics: dict) -> bool:
    """The scored lockbox is the prereg's stage-2 set (it contains lockbox-extension contents)."""
    return bool((metrics.get("lockbox_contents") or {}).get("includes_extension"))


def brain_claim_from(metrics: dict) -> str:
    """BE_stack − E_stack. Interpretation of the doc + prereg: a scored stage-2 lockbox (the sealed 225 plus the
    extension) -> supported iff its CI lower > 0, else not_supported. Otherwise (stage 1, or a lockbox without
    the extension, which the prereg does not accept as confirmatory) the CV rule: directional if the point is > 0
    and the CI upper >= 0, not_supported otherwise; no estimate with a CI (no E arm, too few draws) ->
    not_tested."""
    b = metrics.get("brain") or {}
    lb = b.get("lockbox") or {}
    if confirmatory_lockbox(metrics) and lb.get("point") is not None and (lb.get("ci") or [None])[0] is not None:
        return "supported" if lb["ci"][0] > 0 else "not_supported"
    c = b.get("content") or {}
    ci = c.get("ci") or [None, None]
    if c.get("point") is None or ci[1] is None:
        return "not_tested"
    return "directional" if c["point"] > 0 and ci[1] >= 0 else "not_supported"


# ── ranking helpers ───────────────────────────────────────────────────────


def weighted_percentile(p: float, ref: np.ndarray, w: np.ndarray) -> float:
    """Weighted mid-rank of ``p`` among ``ref`` (0..1): weight below plus half the tied weight, over the total.
    The same basis as fit_models._wranks with the new clip at weight 0."""
    ref, w = np.asarray(ref, float), np.asarray(w, float)
    tot = w.sum()
    return float((w[ref < p].sum() + 0.5 * w[ref == p].sum()) / tot) if tot > 0 else float("nan")


def weighted_quantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    """Quantile of the weighted empirical distribution (weights at their mid-cumulative positions)."""
    o = np.argsort(x, kind="stable")
    x, w = np.asarray(x, float)[o], np.asarray(w, float)[o]
    c = (np.cumsum(w) - 0.5 * w) / w.sum()
    return float(np.interp(q, c, x))


def likely_range(p: float, ref_pred: np.ndarray, ref_obs_pct: np.ndarray, w: np.ndarray) -> list[float]:
    """Weighted P10-P90 of the observed within-stratum percentiles of the reference contents predicted closest
    to ``p`` (the nearest max(30, 20 %) by predicted score). Expect it wide at realistic ρ."""
    k = min(len(ref_pred), max(MIN_NEIGHBOURS, int(math.ceil(NEIGHBOUR_SHARE * len(ref_pred)))))
    near = np.argsort(np.abs(np.asarray(ref_pred, float) - p), kind="stable")[:k]
    return [round(weighted_quantile(ref_obs_pct[near], w[near], q), 4) for q in (0.1, 0.9)]


# ── drivers ───────────────────────────────────────────────────────────────


def term_family(name: str) -> tuple[str, str | None]:
    """linear_terms name -> (driver family, source column or None)."""
    if name in ACCOUNT_TERMS:
        return "account_history", None
    if name.startswith("brain:") or name.startswith("num__brain_"):
        return "brain_response", name.split(":", 1)[1] if ":" in name else name[len("num__"):]
    if name.startswith("emb:") or name.startswith("num__emb_"):
        return "content_embedding", None
    return "metadata", None


def brain_channel(col: str, channels: list[str]) -> str:
    for k in sorted(channels, key=len, reverse=True):
        if col.startswith(f"brain_{k}_"):
            return k
    return "other"  # brain_pca_*, brain_moments_*, brain_xch_*


def drivers_from(terms: pd.Series, ref: pd.Series, channels: list[str]) -> list[dict]:
    """Per family: the clip's summed final-ridge terms minus the deal x platform reference average (training
    posts, weighted 1/incl_prob). Exact: the contributions sum to prediction − reference-average prediction."""
    fam: dict[str, float] = {}
    detail: dict[str, float] = {}
    for name, v in terms.items():
        f, col = term_family(name)
        c = float(v - ref.get(name, 0.0))
        fam[f] = fam.get(f, 0.0) + c
        if f == "brain_response" and col is not None:
            ch = brain_channel(col, channels)
            detail[ch] = detail.get(ch, 0.0) + c
    out = []
    for f, c in sorted(fam.items(), key=lambda kv: -abs(kv[1])):
        d = {"family": f, "label": FAMILY_LABELS[f], "contribution": round(c, 6)}
        if f == "brain_response" and detail:
            d["brain_detail"] = [{"channel": ch, "contribution": round(v, 6)}
                                 for ch, v in sorted(detail.items(), key=lambda kv: -abs(kv[1]))]
        out.append(d)
    return out


# ── the block ─────────────────────────────────────────────────────────────


def _block(status, reason, *, brain_claim="not_tested", model_version=None, context=None, clip_in_training="no",
           retrospective=False, warnings_=None, provenance=None, engagement=None, reach=None, drivers=None,
           n_train=None) -> dict:
    """``validated`` mirrors the status; ``n_train`` (training contents of the engagement model) is set only on a
    scored block; ``caption`` is set only (and always) on a preliminary one."""
    numbers = status in SCORED
    prelim = status == "preliminary"
    if prelim and not isinstance(n_train, int):
        raise ValueError("a preliminary block needs n_train")
    return {"model_status": status, "validated": status == "validated", "reason": reason,
            "brain_claim": "not_tested" if prelim else brain_claim, "model_version": model_version,
            "n_train": int(n_train) if numbers and n_train is not None else None,
            "caption": PRELIMINARY_CAPTION.format(n=int(n_train)) if prelim else None,
            "context": context, "clip_in_training": clip_in_training, "retrospective": bool(retrospective),
            "engagement": engagement if numbers else None, "reach": reach if numbers else None,
            "drivers": (drivers or []) if numbers else [], "warnings": list(warnings_ or []),
            "provenance": provenance or {}}


def _num(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _has_audio(meta: dict, ctx: dict) -> bool | None:
    if ctx.get("has_audio") is not None:
        return bool(ctx["has_audio"])
    counts = (meta.get("modalities") or {}).get("event_counts")
    if isinstance(counts, dict):
        return bool(counts.get("Audio", 0))
    return None


def design_row(feats: pd.DataFrame, ctx: dict) -> pd.DataFrame:
    """The featurised clip plus the context columns fit_models.build_dataset derives (shared helpers)."""
    row = feats.copy()
    row["platform"] = str(ctx["platform"])
    row["deal_id"] = str(ctx["deal_id"])
    row["social_account_id"] = str(ctx.get("account_id") or "missing")  # unknown -> deal x platform level
    row["_content"] = "__new__"  # the encoder's transform ignores it
    row["base_follower_bucket"] = fit_models.follower_bucket([ctx.get("follower_count")]).to_numpy()
    wd, hours = fit_models.post_time([ctx.get("posted_at")])
    row["base_post_weekday"], row["base_post_hour"] = wd.to_numpy(), hours.to_numpy()
    row["base_aspect"] = fit_models.aspect([ctx.get("height")], [ctx.get("width")]).to_numpy()
    row["base_audio_mean_db"] = pd.to_numeric(pd.Series([ctx.get("audio_mean_db")]), errors="coerce").to_numpy()
    return row


def load_model_dir(path) -> dict | None:
    p = Path(path) if path else None
    if p is None or not (p / "manifest.json").exists():
        return None
    man = json.loads((p / "manifest.json").read_text())
    if man.get("artefact_version") != fit_models.ARTEFACT_VERSION:
        raise ValueError(f"{p}: artefact_version {man.get('artefact_version')!r}, this code reads "
                         f"{fit_models.ARTEFACT_VERSION!r}")
    return {"dir": p, "manifest": man, "lockbox": set(json.loads((p / man["lockbox_ids"]).read_text())),
            "train": set(json.loads((p / "train_video_ids.json").read_text()))}


def _target_files(md: dict, t: str) -> dict:
    tm = md["manifest"]["targets"][t]
    for k, f in tm["files"].items():
        if fit_models._sha256(md["dir"] / f) != tm["sha256"][k]:
            raise ValueError(f"{md['dir'] / f}: sha256 differs from the manifest")
    return {k: md["dir"] / f for k, f in tm["files"].items()}


def score_target(md: dict, t: str, X: pd.DataFrame, vid: str, stratum: tuple[str, str], account: str | None,
                 status: str) -> dict:
    """One target's prediction, percentiles, likely range and validation; ``oof`` flags a training clip."""
    import joblib

    tm = md["manifest"]["targets"][t]
    files = _target_files(md, t)
    ref = pd.read_csv(files["reference"], dtype={"deal_id": str, "platform": str, "video_id": str})
    here = ref[(ref["deal_id"] == stratum[0]) & (ref["platform"] == stratum[1])]
    own = here[here["video_id"] == vid]
    if len(own):  # a training clip: its out-of-fold prediction, never an in-sample one, ranked without itself
        p, oof = float(own["pred"].iloc[0]), True
        here = here[here["video_id"] != vid]
    elif vid in md["train"]:  # trained on (another stratum, or a post without this target): never in-sample
        return {"error": "Training clip without an out-of-fold prediction for this deal and platform"}
    else:
        est = joblib.load(files["model"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p, oof = float(est.predict(X[tm["input_columns"]])[0]), False
    n = int(len(here))
    w = here["w"].to_numpy(float)
    pct = weighted_percentile(p, here["pred"].to_numpy(float), w) if n >= MIN_REFERENCE_N else None
    rng = likely_range(p, here["pred"].to_numpy(float), here["obs_pct"].to_numpy(float), w) \
        if n >= MIN_REFERENCE_N else None
    pct_acct = n_acct = None
    if account:
        ra = pd.read_csv(files["reference_accounts"], dtype={"deal_id": str, "platform": str, "video_id": str,
                                                              "social_account_id": str})
        ra = ra[(ra["deal_id"] == stratum[0]) & (ra["platform"] == stratum[1]) & (ra["social_account_id"] == account)
                & (ra["video_id"] != vid)]
        if len(ra):
            n_acct = int(len(ra))
            pct_acct = round(weighted_percentile(p, ra["pred"].to_numpy(float), ra["w"].to_numpy(float)), 4)
    m = tm["metrics"]
    if status == "preliminary":  # its CV estimate is noise at this n and not a result: never shipped per clip
        v, scheme = {"point": None, "ci": [None, None]}, "none"
    else:
        v = (m["served_lockbox"] if status == "validated" else m["served_cv"]) or {"point": None, "ci": [None, None]}
        scheme = "lockbox" if status == "validated" else "content_cv"
    return {
        "pred": p, "oof": oof,
        "block": {
            "target": t,
            "percentile_deal_platform": None if pct is None else round(pct, 4),
            "likely_range": rng, "reference_n": n,
            "percentile_account": pct_acct, "account_reference_n": n_acct,
            "confidence": "medium" if status == "validated" and n >= LOW_CONFIDENCE_N else "low",
            "validation": {"scheme": scheme,
                           "within_stratum_spearman": _num(v.get("point")),
                           "ci95": [_num(x) for x in (v.get("ci") or [None, None])]},
        },
    }


def predict_performance(worker_output, context: dict, featurizer, model_dir, *, roi_map=None,
                        analyses_dir=None, allow_preliminary: bool = False) -> dict:
    """The ``performance`` block for one clip. ``featurizer`` is a path or a ``load_featurizer`` result; it is
    only read when a model passed the GO rule (or ``allow_preliminary`` turns a failed one into ``preliminary``).
    Configuration errors (hash mismatches, unreadable outputs) raise."""
    ctx = dict(context or {})
    vid, meta_path, _ = build_features.resolve_worker_output(worker_output)
    meta = json.loads(Path(meta_path).read_text())
    platform = str(ctx.get("platform") or "").lower() or None
    ctx["platform"] = platform
    account = str(ctx["account_id"]) if ctx.get("account_id") not in (None, "") else None
    context_block = {"deal_id": ctx.get("deal_id"), "deal_label": ctx.get("deal_label") or ctx.get("deal_id"),
                     "platform": platform, "account_id": account, "account_level": False}
    rt = build_features.runtime_key(meta)
    prov = {"tribe_commit": rt["tribe_commit"], "pod_code": ctx.get("pod_code"),
            "precision": rt["video_precision"], "fast_video": rt["fast_video"], "emb_export": rt["emb_export"],
            "prescale": ctx.get("prescale"), "features_version": build_features.FEATURES_VERSION,
            "emb_version": build_features.EMB_VERSION, "video_id": vid}
    warn: list[str] = []
    if (meta.get("runtime") or {}).get("dry_run"):
        warn.append("SYNTHETIC: dry-run (stub) predictions; the numbers test the pipeline, not TRIBE.")
    md = load_model_dir(model_dir)
    if md is None:
        return _block("not_trained", "no saved performance model", context=context_block, warnings_=warn,
                      provenance=prov)
    man = md["manifest"]
    clip_in = "lockbox" if vid in md["lockbox"] else "train_oof" if vid in md["train"] else "no"
    retro = bool(ctx.get("retrospective")) or bool(ctx.get("posted_at")) or clip_in != "no"
    if clip_in == "lockbox":
        warn.append("Sealed lockbox clip: never show its observed outcome.")
    elif retro:
        warn.append("Already-posted clip: this is a retrospective prediction.")
    prov.update({"model_version": man["model_version"], "model_git": man["git"],
                 "training_features_sha256": (man["inputs"].get("features") or {}).get("sha256")})
    common = dict(model_version=man["model_version"], context=context_block, clip_in_training=clip_in,
                  retrospective=retro, warnings_=warn, provenance=prov)
    if PRIMARY not in man["targets"]:
        return _block("not_trained", f"no {PRIMARY} model in {md['dir']}", **common)
    tm = man["targets"][PRIMARY]
    status, reason = model_status_from(tm["metrics"], man["feature_set"], man["model"])
    if status == "not_trained" and not allow_preliminary:
        common["brain_claim"] = brain_claim_from(tm["metrics"])
        return _block(status, reason, **common)
    if status == "not_trained":  # explicit opt-in: the GO rule failed, show the format, claim nothing
        status, reason = "preliminary", None
        common["n_train"] = int(tm["n_train_contents"])
    else:
        common["n_train"] = int(tm["n_train_contents"])
        common["brain_claim"] = brain_claim_from(tm["metrics"])
        b = tm["metrics"].get("brain") or {}
        if common["brain_claim"] == "supported" and ((b.get("content") or {}).get("point") or 0) <= 0:
            warn.append("Brain claim: the lockbox and the content CV disagree in sign (prereg: report both).")
        if b.get("lockbox") and not confirmatory_lockbox(tm["metrics"]):
            warn.append("Brain claim from content CV only: the scored lockbox lacks the stage-2 extension, so it "
                        "is not the confirmatory set.")

    def oos(why):
        return _block("out_of_scope", why, **common)

    if platform not in PLATFORMS:
        return oos(f"platform {ctx.get('platform')!r} is not one of {', '.join(PLATFORMS)}")
    deal = str(ctx.get("deal_id") or "")
    if deal not in tm["deals"]:
        return oos("Unknown deal: the model has no training clips for it")
    if f"{deal}|{platform}" not in tm["strata"]:
        return oos(f"The model has no training clips for this deal on {platform}")
    dur = _num(meta.get("duration_s"))
    if dur is None or not MIN_DURATION_S <= dur <= MAX_DURATION_S:
        return oos(f"Duration {dur if dur is None else round(dur, 1)} s is outside the model's "
                   f"{MIN_DURATION_S:g}-{MAX_DURATION_S:g} s training scope")
    audio = _has_audio(meta, ctx)
    if audio is False:
        return oos("No audio track: outside the model's training scope")
    if audio is None:
        warn.append("Audio presence unknown (no modalities in the worker output and no has_audio in the context).")

    state = featurizer if isinstance(featurizer, dict) else build_features.load_featurizer(
        featurizer, roi_map=roi_map, analyses_dir=analyses_dir)
    fz = state["info"]
    want = (man["inputs"].get("features") or {}).get("sha256")
    if not want or fz.get("features_sha256") != want:
        raise ValueError("train/serve parity: the featurizer was built with a different feature table than the "
                         f"model was fit on ({fz.get('features_sha256')} vs {want})")
    prov.update({"proxies_version": fz["proxies_version"], "roi_map_sha256": fz["roi_map_sha256"],
                 "featurizer_npz_sha256": fz["npz_sha256"]})
    for k, v in rt.items():  # the clip must come from the same TRIBE/pod configuration as the training clips
        seen = {json.loads(s) for s in (fz.get("training_runtime") or {}).get(k, {})}
        known = sorted((x for x in seen if x is not None), key=json.dumps)
        if v is not None and known and v not in known:
            return oos(f"Train/serve mismatch: {k} {v!r} is not among the training clips' {known}")
        if v is None and known:
            warn.append(f"The worker output does not record {k}; the training clips did ({known}).")
        elif v is not None and not known and seen:
            warn.append(f"The training clips did not record {k}; parity with {v!r} is unchecked.")
    if not fz["pca_exclude"]["used"]:
        warn.append("The features' PCAs were fit including lockbox clips (built without --exclude-from-pca-fit).")
    feats = build_features.featurize_one(worker_output, ctx, state)
    if feats["status"].iloc[0] != "ok":
        raise ValueError(f"{vid}: could not featurise: {feats['error'].iloc[0]}")
    if float(feats["qc_non_english"].iloc[0] or 0) > 0:
        return oos("Confident non-English speech: TRIBE transcribes everything as English, and the model "
                   "excluded such clips")
    uses = set(tm["input_columns"])
    missing = [k for k, c in (("follower_count", "base_follower_bucket"), ("posted_at", "base_post_weekday"),
                              ("width/height", "base_aspect"), ("audio_mean_db", "base_audio_mean_db"))
               if c in uses and ctx.get(k.split("/")[0]) in (None, "")]
    if missing:
        warn.append(f"Context without {', '.join(missing)}: imputed with the training median.")
    if any(c.startswith("emb_") for c in uses) and not float(feats["qc_emb_modalities"].iloc[0] or 0):
        warn.append("No extractor-embedding export for this clip: its content features are imputed.")
    if (fz.get("training_has_shots_share") or 0) > 0 and not float(feats["base_has_shots"].iloc[0] or 0):
        warn.append("No shot boundaries for this clip; most training clips had them.")
    X = design_row(feats, ctx)
    eng = score_target(md, PRIMARY, X, vid, (deal, platform), account, status)
    if "error" in eng:
        return oos(eng["error"])
    if eng["block"]["reference_n"] < LOW_CONFIDENCE_N:
        warn.append("Low confidence: this deal has few reference clips on this platform.")
    if eng["block"]["percentile_deal_platform"] is None:
        warn.append(f"Fewer than {MIN_REFERENCE_N} reference clips for this deal on this platform: no percentile "
                    "or range.")
    reach = None
    if REACH in man["targets"] and status != "preliminary":  # preliminary: engagement only
        r_status, _ = model_status_from(man["targets"][REACH]["metrics"], man["feature_set"], man["model"])
        if r_status != "not_trained":
            r = score_target(md, REACH, X, vid, (deal, platform), account, r_status)
            if "error" not in r:
                reach = r["block"]
                warn.append("Reach is mostly driven by account, platform and timing.")
    context_block["account_level"] = eng["block"]["percentile_account"] is not None
    drivers: list[dict] = []
    if eng["oof"]:
        warn.append("Training clip: drivers omitted (they would come from a model that saw this clip).")
    else:
        import joblib

        files = _target_files(md, PRIMARY)
        est = joblib.load(files["model"])
        ref_terms = pd.read_csv(files["reference_terms"], dtype={"deal_id": str, "platform": str}).set_index(
            ["deal_id", "platform"])
        terms = fit_models.linear_terms(est, X[tm["input_columns"]]).iloc[0]
        channels = [c.key for c in state["clip"]["spec"].channels]
        drivers = drivers_from(terms, ref_terms.loc[(deal, platform)], channels)
        prov["artefact_sha256"] = tm["sha256"]["model"]
    prov.setdefault("artefact_sha256", tm["sha256"]["model"])
    return _block(status, None, engagement=eng["block"], reach=reach, drivers=drivers, **common)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("worker_output", help="<vid>.npz, <vid>.json, <vid>.emb.npz, or a directory with one clip")
    ap.add_argument("--featurizer", type=Path, required=True, help="<stem>.featurizer.json from build_features.py")
    ap.add_argument("--model-dir", type=Path, required=True, help="fit_models.py --save-model DIR")
    ap.add_argument("--context", type=Path, default=None, help="JSON object; the flags below override its keys")
    ap.add_argument("--deal-id")
    ap.add_argument("--deal-label")
    ap.add_argument("--platform", choices=PLATFORMS)
    ap.add_argument("--account-id")
    ap.add_argument("--follower-count", type=float)
    ap.add_argument("--posted-at", help="ISO timestamp; also marks the prediction retrospective")
    ap.add_argument("--width", type=float)
    ap.add_argument("--height", type=float)
    ap.add_argument("--audio-mean-db", type=float, help="tools/qc_files.py audio_mean_db")
    ap.add_argument("--no-audio", dest="has_audio", action="store_false", default=None)
    ap.add_argument("--retrospective", action="store_true", default=None)
    ap.add_argument("--pod-code", help="pod code commit the clip ran with (provenance)")
    ap.add_argument("--prescale", help="e.g. s384 (provenance)")
    ap.add_argument("--roi-map", type=Path, default=None, help="override the featurizer's ROI map path (hash-checked)")
    ap.add_argument("--analyses-dir", type=Path, default=None, help="brain_report analyses/ dir for shots_ms")
    ap.add_argument("--allow-preliminary", action="store_true",
                    help="score a saved model that did NOT pass the stage-1 GO rule as model_status "
                         "'preliminary' (validated false, caption, n_train) instead of not_trained")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    ctx = json.loads(args.context.read_text()) if args.context else {}
    for k in ("deal_id", "deal_label", "platform", "account_id", "follower_count", "posted_at", "width", "height",
              "audio_mean_db", "has_audio", "retrospective", "pod_code", "prescale"):
        if getattr(args, k) is not None:
            ctx[k] = getattr(args, k)
    try:
        block = predict_performance(args.worker_output, ctx, args.featurizer, args.model_dir,
                                    roi_map=args.roi_map, analyses_dir=args.analyses_dir,
                                    allow_preliminary=args.allow_preliminary)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    text = json.dumps(block, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
