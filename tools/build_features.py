#!/usr/bin/env python3
"""Per-clip feature table from TRIBE worker outputs (CPU, parallel over clips).

Scans ``<out-root>/worker-*/<video_id>.json|.npz`` (pod/worker.py format; ``--out-root`` may be given more
than once, e.g. one per pulled batch: the first root holding a completed clip wins) and writes one row per
``video_id``. Clips whose worker runs differ in video precision or frame loop (fp32 stock vs the bf16 fast
loop; unrecorded counts as stock fp32) are refused: the pre-registration never mixes them in one table.

    <out>                      features (.parquet, or .csv by suffix)
    <out stem>.meta.json       feature list, families, versions, PCA variance, counts
    <out stem>.pca.npz         PCA of the per-clip time-mean cortex map (mean, components)
    <out stem>.featurizer.npz  featurizer state: brain and every extractor-embedding PCA (mean, components)
    <out stem>.featurizer.json its sidecar: versions, column order, PCA fit counts, training runtime mix, sha256s
                               of the npz, the table, the ROI map and the proxies file (``load_featurizer``)

One new clip is featurised exactly like a training row with ``featurize_one`` (tools/predict.py): the same
per-clip code, then a projection on the saved PCAs. Nothing is imputed here; missing values stay NaN and the
model pipeline imputes them as it did in training.

Column prefixes decide how tools/fit_models.py uses a column:

    base_*   non-brain features available without TRIBE predictions
             (duration, speech from the saved words, shots if provided)
    brain_*  derived from TRIBE predictions
             brain_<channel>_*   per proxy channel (within-clip z unless ``raw``)
             brain_moments_*     rule-based candidate moments (tribe_research/brain/moments.py)
             brain_xch_*         cross-channel trajectory features
             brain_pca_NN        raw-cortex time-mean map projected on the top PCA components
    emb_*    the extractor features TRIBE's brain mapping consumes, without the mapping (the control arm):
             emb_<video|audio|text>_pca_NN       PCA of the clip's time-mean ``<vid>.emb.npz`` vector
             emb_<video|audio|text>_sd_pca_NN    PCA of its per-feature sd over time
             emb_<video|audio|text>_bins_pca_NN  PCA of its four quarter means minus the time mean
             (``emb_features_v1``, read from the pod's ``emb_pool_v1`` files: per layer group, flattened), each
             fit like brain_pca; absent when no clip has one. The sd/bins blocks give the control the same
             coarse temporal access the brain features have, so BE − E credits the brain mapping, not merely
             time-resolved pooling
    qc_*     quality/diagnostics; used by neither model
    status   ok | failed (worker .error.json) | error (could not featurize) | missing (manifest only)

Brain values are predictions for an average subject. Within-clip z features
are relative to the clip; ``raw`` and ``pca`` features compare model-space
levels across clips, which is exactly what the modelling step is testing.

    python tools/build_features.py --out-root outputs [--out-root more-outputs ...] \
        --out results/features/features.parquet \
        --roi-map tribe_research/assets/roi_map_roi_groups_v0.npz \
        [--manifest results/run_full/manifest.jsonl] [--analyses-dir report/analyses] [--n-jobs 40] \
        [--exclude-from-pca-fit results/study/selection.csv]

``--exclude-from-pca-fit`` keeps the selection's lockbox clips (``split == lockbox``) out of the PCA fit;
they are still projected, so the lockbox never shapes a feature it is scored on.

Onset: ``mean_all``, ``peak``/``trough``, ``raw_mean``/``raw_sd``, ``xch_broad_frac``/``xch_mean_abs_z``
and the PCA time-mean include the first ``ONSET_S`` seconds (stimulus-onset transient in the predictions).
The within-clip z scale and ``xch_synchrony`` exclude them; ``mean_0_3s``/``early_minus_rest`` cover them
by design.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tribe_research.brain.analysis import _stability  # noqa: E402
from tribe_research.brain.events import speech_spans, words_lane  # noqa: E402
from tribe_research.brain.features import FEATURE_VERSION, RoiMap  # noqa: E402
from tribe_research.brain.moments import (  # noqa: E402
    DROP_LOOKBACK_S, DROP_MIN_Z, ENTER_Z, _runs, detect_moments,
)
from tribe_research.brain.proxies import (  # noqa: E402
    ONSET_S, ProxySpec, channel_masks, channel_raw, zscore_excluding_onset,
)

FEATURES_VERSION = "clip_features_v1"
DEFAULT_ROI_MAP = ROOT / "tribe_research/assets/roi_map_roi_groups_v0.npz"
N_PCA = 20
PCA_BATCH = 1000
PCA_MAX_INMEM_BYTES = 8 * 1024 ** 3  # above this, fall back to IncrementalPCA batches
PCA_THREADS = 8  # BLAS threads for the PCA (the host is shared; more threads mostly spin)
MOMENT_KINDS = ("attention_drop", "broad_response", "proxy_rise", "proxy_fall")
EARLY_S = 2.0  # "first 2 s vs rest" trajectory window (= proxies.ONSET_S)
EMB_MODALITIES = ("video", "audio", "text")
EMB_VERSION = "emb_features_v1"  # how the features are derived; the pod file format is pod/emb_export.py VERSION
FEATURIZER_VERSION = "featurizer_v1"  # layout of <stem>.featurizer.npz/.json
EMB_STATS = ("", "_sd", "_bins")  # block key suffix per modality: time mean, sd over time, quarter shape
NON_ENGLISH_MIN_PROB = 0.5  # whisperx language probability above which a non-"en" detection is trusted

# per-process state (set by _init)
_STATE: dict = {}


# ── per-clip computation ─────────────────────────────────────────────────


def clip_state(roi_map_path: str, analyses_dir: str | None) -> dict:
    """What ``featurize`` needs per clip: proxy spec, normalised channel masks, vertex count, analyses dir."""
    spec = ProxySpec.load()
    roi = RoiMap.load(roi_map_path)
    masks = channel_masks(spec, roi).astype(np.float32)
    return dict(spec=spec, masks=masks / masks.sum(axis=1, keepdims=True), n_vertices=masks.shape[1],
                analyses_dir=Path(analyses_dir) if analyses_dir else None)


def _init(roi_map_path: str, analyses_dir: str | None) -> None:
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(1)
    except Exception:  # noqa: BLE001  (optional; env vars below also help)
        pass
    _STATE.update(clip_state(roi_map_path, analyses_dir))


def _wmean(t: np.ndarray, y: np.ndarray, lo: float, hi: float) -> float:
    m = (t >= lo) & (t < hi)
    return float(y[m].mean()) if m.any() else float("nan")


def _count_attention_drops(t, dur, z_att) -> int:
    """Uncapped count of attention drops, same rule as moments.detect_moments."""
    n = 0
    for r in _runs(t, dur, z_att, -1):
        look = (t >= r["start"] - DROP_LOOKBACK_S) & (t < r["start"])
        if look.any() and float(z_att[look].max() - r["peak_z"]) >= DROP_MIN_Z:
            n += 1
    return n


def _shots_for(meta: dict, state: dict | None = None) -> list[int] | None:
    if isinstance(meta.get("shots_ms"), list):
        return [int(s) for s in meta["shots_ms"]]
    d = (_STATE if state is None else state).get("analyses_dir")
    if d is not None:
        p = d / meta["video_id"] / "analysis.json"
        if p.exists():
            shots = json.loads(p.read_text()).get("events", {}).get("shots_ms")
            if isinstance(shots, list):
                return [int(s) for s in shots]
    return None


def base_features(meta: dict, duration: float, shots_ms: list[int] | None) -> dict:
    words = words_lane(meta.get("words") or [])
    spans = speech_spans(words)
    speech_s = sum(b - a for a, b in spans) / 1000.0
    dur = max(duration, 1e-6)
    f = {
        "base_duration_s": duration,
        "base_log_duration": math.log(dur),
        "base_has_words": float(bool(words)),
        "base_n_words": float(len(words)),
        "base_word_rate": len(words) / dur,
        "base_speech_coverage": min(1.0, speech_s / dur),
        "base_first_word_s": words[0]["start_ms"] / 1000.0 if words else float("nan"),
        "base_words_first_3s": float(sum(w["start_ms"] < 3000 for w in words)),
        "base_has_shots": float(shots_ms is not None),
    }
    if shots_ms is not None:
        s = sorted(x for x in shots_ms if x >= 0)
        f["base_n_shots"] = float(len(s))
        f["base_shots_per_s"] = len(s) / dur
        f["base_first_shot_s"] = s[0] / 1000.0 if s else float("nan")
        f["base_shots_first_3s"] = float(sum(x < 3000 for x in s))
    else:
        f.update(base_n_shots=float("nan"), base_shots_per_s=float("nan"),
                 base_first_shot_s=float("nan"), base_shots_first_3s=float("nan"))
    return f


def brain_features(t: np.ndarray, dur: np.ndarray, raw: np.ndarray, duration: float, spec: ProxySpec,
                   shots_ms, speech_ms) -> dict:
    z, onset_excluded = zscore_excluding_onset(t, raw)
    keys = [c.key for c in spec.channels]
    D = max(duration, 1e-6)
    half = D / 2.0
    f: dict[str, float] = {}
    for i, k in enumerate(keys):
        zi, ri = z[i], raw[i]
        p = f"brain_{k}_"
        f[p + "mean_0_3s"] = _wmean(t, zi, 0, 3)
        f[p + "mean_0_5s"] = _wmean(t, zi, 0, 5)
        f[p + "mean_all"] = float(zi.mean())
        f[p + "peak"] = float(zi.max())
        f[p + "trough"] = float(zi.min())
        f[p + "peak_time_frac"] = float(t[int(zi.argmax())] / D)
        f[p + "trough_time_frac"] = float(t[int(zi.argmin())] / D)
        f[p + "slope_half"] = _wmean(t, zi, half, math.inf) - _wmean(t, zi, -math.inf, half)
        f[p + "early_minus_rest"] = _wmean(t, zi, -math.inf, EARLY_S) - _wmean(t, zi, EARLY_S, math.inf)
        s = _stability(zi, t)
        f[p + "stability"] = float("nan") if s is None else float(s)
        f[p + "rises_per_min"] = 60.0 * len(_runs(t, dur, zi, +1)) / D
        f[p + "falls_per_min"] = 60.0 * len(_runs(t, dur, zi, -1)) / D
        f[p + "raw_mean"] = float(ri.mean())
        f[p + "raw_sd"] = float(ri.std())

    # moments as the UI shows them (capped at moments.MAX_MOMENTS, so also report uncapped drops)
    moments = detect_moments(t, dur, z, spec.channels, shots_ms, speech_ms)
    per_min = 60.0 / D
    for kind in MOMENT_KINDS:
        f[f"brain_moments_{kind}_per_min"] = per_min * sum(m["kind"] == kind for m in moments)
    f["brain_moments_total_per_min"] = per_min * len(moments)
    f["brain_moments_first_start_frac"] = (min(m["start_ms"] for m in moments) / 1000.0 / D
                                           if moments else float("nan"))
    if "attention" in keys:
        ia = keys.index("attention")
        n_drop = _count_attention_drops(t, dur, z[ia])
        f["brain_xch_attention_drops_per_min"] = per_min * n_drop
        f["brain_xch_any_attention_drop"] = float(n_drop > 0)

    # cross-channel trajectory
    post = t >= ONSET_S
    zp = z[:, post] if post.sum() >= 3 else z
    if zp.shape[1] >= 3:
        c = np.corrcoef(zp)
        iu = np.triu_indices(len(keys), 1)
        f["brain_xch_synchrony"] = float(np.nanmean(c[iu]))
    else:
        f["brain_xch_synchrony"] = float("nan")
    active = (z > ENTER_Z).sum(axis=0)
    f["brain_xch_broad_frac"] = float((active >= 3).mean())
    f["brain_xch_mean_abs_z"] = float(np.abs(z).mean())
    f["brain_xch_early_minus_rest_mean"] = float(np.nanmean(
        [f[f"brain_{k}_early_minus_rest"] for k in keys]))
    f["brain_xch_onset_excluded"] = float(onset_excluded)
    return f


def load_emb(npz_path: str) -> dict[str, np.ndarray]:
    """``<vid>.emb.npz`` next to the preds -> {block: flattened float32 vector}; {} if absent/unusable.

    Blocks per modality m: ``m`` time mean [G*D], ``m_sd`` sd over time [G*D], ``m_bins`` the four quarter means
    minus the time mean [4*G*D] (an empty quarter counts as the mean: no shape information). The export is
    best-effort on the pod, so a missing or broken file only drops this clip's emb_* values."""
    p = Path(npz_path[: -len(".npz")] + ".emb.npz")
    if not p.exists():
        return {}
    out = {}
    try:
        with np.load(p) as zf:
            for m in EMB_MODALITIES:
                if f"{m}_mean" not in zf.files:
                    continue
                mean = zf[f"{m}_mean"].astype(np.float32)
                if not mean.size or not np.isfinite(mean).all():
                    continue
                out[m] = mean.ravel()
                if f"{m}_sd" in zf.files:
                    sd = zf[f"{m}_sd"].astype(np.float32)
                    if sd.shape == mean.shape and np.isfinite(sd).all():
                        out[f"{m}_sd"] = sd.ravel()
                if f"{m}_bins" in zf.files:
                    bins = zf[f"{m}_bins"].astype(np.float32)
                    if bins.shape[1:] == mean.shape:
                        shape = np.where(np.isfinite(bins), bins - mean[None], 0.0)
                        out[f"{m}_bins"] = shape.ravel()
    except Exception:  # noqa: BLE001
        return {}
    return out


def language_qc(meta: dict) -> dict:
    """whisperx's language call for the clip. TRIBE transcribes as English, so a confident non-English
    detection means the text features describe a mistranscription."""
    tr = meta.get("transcript") or {}
    lang, prob = tr.get("detected_language"), tr.get("language_probability")
    non_en = bool(lang) and lang != "en" and prob is not None and float(prob) >= NON_ENGLISH_MIN_PROB
    return {"qc_language_prob": float(prob) if prob is not None else float("nan"),
            "qc_non_english": float(non_en)}


def runtime_key(meta: dict) -> dict:
    """The worker settings train/serve parity depends on (docs/PRODUCT_PIPELINE.md), None when not recorded."""
    vc = meta.get("video_config") or {}
    return {"tribe_commit": meta.get("tribe_commit"), "video_precision": vc.get("video_precision"),
            "fast_video": vc.get("fast_video"), "emb_export": (meta.get("emb") or {}).get("version")}


def featurize(item: tuple[str, str, str], state: dict | None = None, meta: dict | None = None
              ) -> tuple[str, dict, np.ndarray | None, dict, dict]:
    """(video_id, meta.json path, npz path) -> (video_id, row, time-mean vertex map or None, emb vectors,
    ``runtime_key``). The runtime settings stay out of the table; they go to the featurizer sidecar.

    ``state`` defaults to this process's pool state (``_init``); ``featurize_one`` passes a loaded featurizer
    and, to add request context such as ``shots_ms``, the already-read ``meta``."""
    st = _STATE if state is None else state
    vid, meta_path, npz_path = item
    try:
        meta = json.loads(Path(meta_path).read_text()) if meta is None else meta
        with np.load(npz_path) as zf:
            preds = zf["preds"]
            t = zf["seg_start"].astype(np.float64)
            dur = zf["seg_duration"].astype(np.float64)
        if preds.ndim != 2 or preds.shape[0] != len(t) or len(t) == 0:
            raise ValueError(f"bad preds shape {preds.shape} for {len(t)} segments")
        if preds.shape[1] != st["n_vertices"]:
            raise ValueError(f"preds has {preds.shape[1]} vertices, ROI map {st['n_vertices']}")
        order = np.argsort(t, kind="stable")
        t, dur, preds = t[order], dur[order], preds[order]
        p32 = preds.astype(np.float32)
        if not np.isfinite(p32).all():
            raise ValueError("non-finite predictions")
        duration = float(meta.get("duration_s") or (t[-1] + dur[-1]))
        raw = st["masks"] @ p32.T  # [C, T] vertex-weighted channel means
        shots = _shots_for(meta, st)
        words = words_lane(meta.get("words") or [])
        row = {
            "status": "ok", "error": None,
            "source_name": meta.get("source_name"),
            "worker_id": meta.get("worker_id"),
            "synthetic": bool((meta.get("runtime") or {}).get("dry_run", False)),
            "qc_n_segments": float(len(t)),
            "qc_tr_s": float(meta.get("tr_s") or np.median(dur)),
            "qc_segment_coverage": float(min(1.0, np.minimum(dur, np.diff(np.append(t, t[-1] + dur[-1]))).sum()
                                         / max(duration, 1e-6))),
            **language_qc(meta),
            **base_features(meta, duration, shots),
            **brain_features(t, dur, raw, duration, st["spec"], shots, speech_spans(words)),
        }
        emb = load_emb(npz_path)
        row["qc_emb_modalities"] = float(sum(m in emb for m in EMB_MODALITIES))
        return vid, row, p32.mean(axis=0), emb, runtime_key(meta)
    except Exception as exc:  # noqa: BLE001  one bad clip never stops the table
        return vid, {"status": "error", "error": f"{type(exc).__name__}: {exc}"[:500]}, None, {}, {}


# ── discovery ────────────────────────────────────────────────────────────


def discover_all(out_roots) -> tuple[list[tuple[str, str, str]], dict[str, dict]]:
    """``discover`` over several roots in order: the first completed copy of a clip wins, and a failure in one
    root is dropped when another root completed the clip."""
    done: dict[str, tuple[str, str, str]] = {}
    failed: dict[str, dict] = {}
    for root in out_roots:
        items, fails = discover(Path(root))
        for it in items:
            done.setdefault(it[0], it)
        for vid, err in fails.items():
            failed.setdefault(vid, err)
    for vid in done:
        failed.pop(vid, None)
    return sorted(done.values()), failed


def check_one_video_runtime(runtimes: dict[str, dict]) -> None:
    """Refuse a table whose clips ran with different video precision / frame loop (PREREGISTRATION.md, frozen
    pipeline, precision row). Outputs without ``video_config`` predate the fast loop, which always records it,
    so they count as the stock fp32 loop."""
    seen: dict[str, list[str]] = {}
    for vid, rt in runtimes.items():
        k = json.dumps({"video_precision": rt.get("video_precision") or "fp32",
                        "fast_video": bool(rt.get("fast_video"))}, sort_keys=True)
        seen.setdefault(k, []).append(vid)
    if len(seen) > 1:
        detail = "; ".join(f"{k}: {len(v)} clips (e.g. {sorted(v)[0]})" for k, v in sorted(seen.items()))
        raise SystemExit(f"mixed video runtimes in one feature table ({detail}); fp32/stock and bf16 outputs are "
                         "never mixed in one fit (PREREGISTRATION.md, precision)")


def discover(out_root: Path) -> tuple[list[tuple[str, str, str]], dict[str, dict]]:
    """Completed clips (json + npz) and worker failures; a success anywhere wins over an error."""
    done: dict[str, tuple[str, str, str]] = {}
    failed: dict[str, dict] = {}
    for p in sorted(out_root.glob("worker-*/*.json")):
        if p.name.endswith(".error.json"):
            try:
                err = json.loads(p.read_text())
                failed[err.get("video_id") or p.name[: -len(".error.json")]] = err
            except Exception:  # noqa: BLE001
                failed[p.name[: -len(".error.json")]] = {"error": "unreadable error.json"}
            continue
        vid = p.stem
        npz = p.with_suffix(".npz")
        if npz.exists() and vid not in done:  # re-sharded duplicates: first (sorted) complete copy
            done[vid] = (vid, str(p), str(npz))
        elif not npz.exists() and vid not in done:
            failed.setdefault(vid, {"error": "json present but npz missing", "category": "missing_npz"})
    for vid in done:
        failed.pop(vid, None)
    return sorted(done.values()), failed


# ── PCA over the time-mean vertex maps ───────────────────────────────────


def fit_pca(X: np.ndarray, n_components: int, seed: int = 0, threads: int = PCA_THREADS,
            n_fit: int | None = None):
    """Top-k PCA of the time-mean maps, fit on the first ``n_fit`` rows (default all), applied to every row.
    Randomized PCA on the in-memory matrix (14k clips × 20484 vertices ≈ 1.2 GB float32, seconds);
    IncrementalPCA in batches when the matrix would exceed PCA_MAX_INMEM_BYTES."""
    from sklearn.decomposition import PCA, IncrementalPCA
    from threadpoolctl import threadpool_limits

    n = X.shape[0]
    n_fit = n if n_fit is None else n_fit
    k = min(n_components, n_fit - 1, X.shape[1])
    if k < 1:
        return None, np.zeros((n, 0), np.float32)
    F = X[:n_fit]  # a view: callers put the fit rows first
    with threadpool_limits(threads):
        if X.nbytes <= PCA_MAX_INMEM_BYTES:
            pca = PCA(n_components=k, svd_solver="randomized", random_state=seed)
            Z = pca.fit_transform(X) if n_fit == n else pca.fit(F).transform(X)
            pca.method_ = "randomized"
            return pca, Z.astype(np.float32)
        n_batches = max(1, n_fit // max(PCA_BATCH, k))  # every batch has >= max(PCA_BATCH, k) rows
        ipca = IncrementalPCA(n_components=k)
        for b in np.array_split(np.arange(n_fit), n_batches):
            ipca.partial_fit(F[b])
        Z = np.vstack([ipca.transform(X[b]) for b in np.array_split(np.arange(n), max(1, n // PCA_BATCH))])
        ipca.method_ = "incremental"
        return ipca, Z.astype(np.float32)


# ── main ─────────────────────────────────────────────────────────────────


def lockbox_ids(path: Path) -> set[str]:
    """video_ids of the lockbox rows (``split == lockbox``) of a study selection (.csv or .jsonl manifest)."""
    sel = pd.read_json(path, lines=True) if path.suffix in (".jsonl", ".ndjson") else pd.read_csv(path)
    if not {"video_id", "split"} <= set(sel.columns):
        raise SystemExit(f"{path}: needs video_id and split columns")
    return set(sel.loc[sel["split"].astype(str) == "lockbox", "video_id"].astype(str))


def cast_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Every base_/brain_/emb_/qc_ column as float32 (the table's storage type; batch and featurize_one)."""
    for c in [c for c in df.columns if c.startswith(("base_", "brain_", "emb_", "qc_"))]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    return df


def build(out_root, roi_map: Path, *, n_jobs: int, manifest: Path | None = None,
          analyses_dir: Path | None = None, n_pca: int = N_PCA, limit: int | None = None,
          pca_exclude: set[str] | None = None, log=print,
          state_out: dict | None = None) -> tuple[pd.DataFrame, dict, object]:
    """Feature table, meta and the brain PCA. ``state_out``, when given, receives the featurizer state
    (``{"arrays": ..., "info": ...}``, see ``write_featurizer``) without changing the table. ``out_root`` is one
    path or a list of them (``discover_all``)."""
    t0 = time.perf_counter()
    runtimes: dict[str, dict] = {}
    pcas: dict[str, object] = {}
    roots = [Path(r) for r in out_root] if isinstance(out_root, (list, tuple)) else [Path(out_root)]
    items, failed = discover_all(roots)
    if limit is not None:
        items = items[:limit]
    where = roots[0] if len(roots) == 1 else f"{len(roots)} roots ({', '.join(map(str, roots))})"
    log(f"{len(items)} completed clips, {len(failed)} failed/incomplete under {where}")
    rows: dict[str, dict] = {}
    vecs: dict[str, np.ndarray] = {}
    embs: dict[str, dict[str, np.ndarray]] = {m + k: {} for m in EMB_MODALITIES for k in EMB_STATS}
    init_args = (str(roi_map), str(analyses_dir) if analyses_dir else None)
    _init(*init_args)  # here first: a failing pool initializer would make the pool respawn workers forever
    if n_jobs <= 1 or len(items) < 2:
        results = map(featurize, items)
        pool = None
    else:
        ctx = mp.get_context("spawn")  # fork after BLAS/OpenMP threads start can deadlock
        pool = ctx.Pool(n_jobs, initializer=_init, initargs=init_args)
        results = pool.imap_unordered(featurize, items, chunksize=max(1, min(32, len(items) // (n_jobs * 8) or 1)))
    try:
        for i, (vid, row, vec, emb, rt) in enumerate(results, 1):
            rows[vid] = row
            if rt:
                runtimes[vid] = rt
            if vec is not None:
                vecs[vid] = vec
            for m, v in emb.items():
                embs[m][vid] = v
            if i % 2000 == 0:
                log(f"  {i}/{len(items)} clips ({time.perf_counter() - t0:.0f}s)")
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    t_feat = time.perf_counter() - t0
    check_one_video_runtime(runtimes)

    for vid, err in failed.items():
        rows.setdefault(vid, {"status": "failed", "error": str(err.get("error"))[:500],
                              "failure_category": err.get("category", "other")})
    if manifest is not None:
        man = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
        for r in man:
            rows.setdefault(r["video_id"], {"status": "missing", "error": None})
            rows[r["video_id"]].update({"manifest_deal_id": r.get("deal_id"),
                                        "manifest_source_name": r.get("source_name"),
                                        "manifest_duration_s": r.get("duration_s"),
                                        "manifest_chunk": r.get("chunk")})

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "video_id"
    df = df.reset_index().sort_values("video_id", kind="stable").reset_index(drop=True)

    # PCA fit rows first (so the fit slice is a view), then the excluded (lockbox) rows, only projected
    held = sorted(v for v in vecs if v in (pca_exclude or set()))
    ok_ids = sorted(v for v in vecs if v not in (pca_exclude or set())) + held
    n_fit = len(ok_ids) - len(held)
    pca_info: dict = {"n_components": 0}
    ipca = None
    if ok_ids:
        X = np.stack([vecs[v] for v in ok_ids]).astype(np.float32)
        vecs.clear()
        ipca, Z = fit_pca(X, n_pca, threads=max(1, min(n_jobs, PCA_THREADS)), n_fit=n_fit)
        del X
        if ipca is not None:
            pcas["brain"] = ipca
            cols = [f"brain_pca_{j + 1:02d}" for j in range(Z.shape[1])]
            pcs = pd.DataFrame(Z, columns=cols)
            pcs.insert(0, "video_id", ok_ids)
            df = df.merge(pcs, on="video_id", how="left")
            pca_info = {"n_components": int(Z.shape[1]), "n_fit": n_fit, "n_excluded_from_fit": len(held),
                        "explained_variance_ratio": [round(float(x), 5) for x in ipca.explained_variance_ratio_],
                        "method": getattr(ipca, "method_", "pca"),
                        "input": "per-clip time-mean of raw predictions over all 20484-vertex rows "
                                 "(includes the onset window)",
                        "note": ("unsupervised (no outcome labels); fit without the excluded (lockbox) clips, "
                                 "which are only projected" if held else
                                 "unsupervised (no outcome labels), fit on all clips")}
    emb_info: dict = {}
    for m, ev in embs.items():
        dims = pd.Series([v.size for v in ev.values()]).value_counts()
        if dims.empty:
            continue
        d = int(dims.index[0])  # one extractor, one width; a different width is a broken export
        ev = {v: x for v, x in ev.items() if x.size == d}
        held_m = sorted(v for v in ev if v in (pca_exclude or set()))
        ids_m = sorted(v for v in ev if v not in (pca_exclude or set())) + held_m
        epca, Z = fit_pca(np.stack([ev[v] for v in ids_m]).astype(np.float32), n_pca,
                          threads=max(1, min(n_jobs, PCA_THREADS)), n_fit=len(ids_m) - len(held_m))
        if epca is None:
            continue
        pcas[f"emb_{m}"] = epca
        pcs = pd.DataFrame(Z, columns=[f"emb_{m}_pca_{j + 1:02d}" for j in range(Z.shape[1])])
        pcs.insert(0, "video_id", ids_m)
        df = df.merge(pcs, on="video_id", how="left")
        emb_info[m] = {"n_clips": len(ids_m), "dim": d, "n_components": int(Z.shape[1]),
                       "n_fit": len(ids_m) - len(held_m), "n_wrong_dim_dropped": int(dims.sum() - len(ids_m)),
                       "explained_variance_ratio": [round(float(x), 5) for x in epca.explained_variance_ratio_]}
    embs.clear()
    df = cast_feature_columns(df)

    counts = df["status"].value_counts().to_dict()
    meta = {
        "features_version": FEATURES_VERSION,
        "roi_feature_version": FEATURE_VERSION,
        "proxies_version": ProxySpec.load().version,
        "roi_map": str(roi_map),
        "out_root": str(roots[0]) if len(roots) == 1 else [str(r) for r in roots],
        "status_counts": {str(k): int(v) for k, v in counts.items()},
        "synthetic_rows": int(df.get("synthetic", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
        "columns": {
            "base": [c for c in df.columns if c.startswith("base_")],
            "brain": [c for c in df.columns if c.startswith("brain_")],
            "emb": [c for c in df.columns if c.startswith("emb_")],
            "qc": [c for c in df.columns if c.startswith("qc_")],
        },
        "pca": pca_info,
        "emb": {"version": EMB_VERSION, "input": "per-clip time mean, sd over time and quarter shape (quarter "
                "means minus the time mean) of the layer-group-averaged extractor features TRIBE feeds its brain "
                "mapping (<vid>.emb.npz), PCA per extractor and statistic like brain_pca",
                "per_modality": emb_info},
        "timing_s": {"featurize": round(t_feat, 2), "total": round(time.perf_counter() - t0, 2)},
        "n_jobs": n_jobs,
        "notes": [
            "brain_* are TRIBE model predictions for an average subject, not measured activity.",
            "brain_<channel>_* (except raw_*) are within-clip z-scores; raw_* and pca_* are model-space levels.",
            "brain_moments_*_per_min use the UI moment list (capped at 12 per clip); "
            "brain_xch_attention_drops_per_min is uncapped.",
            f"mean_all, peak/trough, raw_mean/raw_sd, xch_broad_frac/xch_mean_abs_z and the PCA time-mean include "
            f"the first {ONSET_S:g} s (stimulus-onset transient); the z scale and xch_synchrony exclude it.",
        ],
    }
    if state_out is not None:
        state_out.update(featurizer_state(df, meta, pcas, runtimes, roi_map, pca_exclude))
    return df, meta, ipca


# ── featurizer state: featurise one new clip exactly like a training row ──


def sha256_file(path: str | Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def featurizer_state(df: pd.DataFrame, meta: dict, pcas: dict, runtimes: dict, roi_map: Path,
                     pca_exclude: set[str] | None) -> dict:
    """Arrays (PCA mean/components per block, float32 as in the .pca.npz) and the json-able description."""
    from tribe_research.brain.proxies import PROXIES_FILE

    arrays, blocks = {}, {}
    for name, pca in pcas.items():
        arrays[f"{name}_mean"] = pca.mean_.astype(np.float32)
        arrays[f"{name}_components"] = pca.components_.astype(np.float32)
        info = meta["pca"] if name == "brain" else meta["emb"]["per_modality"][name[len("emb_"):]]
        blocks[name] = {"columns": [f"{name}_pca_{j + 1:02d}" for j in range(pca.components_.shape[0])],
                        "dim": int(pca.components_.shape[1]), "n_fit": int(info["n_fit"]),
                        "n_excluded_from_fit": int(info.get("n_excluded_from_fit",
                                                            info.get("n_clips", 0) - info["n_fit"]))}
    ok = df[df["status"] == "ok"] if "status" in df else df
    mix: dict[str, dict[str, int]] = {}
    for rt in (runtimes[v] for v in ok["video_id"] if v in runtimes):
        for k, v in rt.items():
            mix.setdefault(k, {})
            mix[k][json.dumps(v)] = mix[k].get(json.dumps(v), 0) + 1
    shots = pd.to_numeric(ok.get("base_has_shots", pd.Series(dtype=float)), errors="coerce")
    info = {
        "featurizer_version": FEATURIZER_VERSION,
        "features_version": FEATURES_VERSION, "emb_version": EMB_VERSION,
        "roi_feature_version": meta["roi_feature_version"], "proxies_version": meta["proxies_version"],
        "proxies_sha256": sha256_file(PROXIES_FILE),
        "roi_map": str(roi_map), "roi_map_sha256": sha256_file(roi_map),
        "columns": list(df.columns),
        "feature_columns": [c for c in df.columns if c.startswith(("base_", "brain_", "emb_", "qc_"))],
        "blocks": blocks,
        "pca_exclude": {"used": bool(pca_exclude), "n_ids": len(pca_exclude or ()),
                        "note": "PCA fit rows exclude these (lockbox) clips; they are only projected"
                        if pca_exclude else "no exclusion set: the PCAs were fit on every clip, lockbox "
                        "included (build without --exclude-from-pca-fit)"},
        "imputation": "none: missing values stay NaN; the model pipeline imputes them (tools/fit_models.py)",
        "training_runtime": mix,
        "training_has_shots_share": float(shots.mean()) if shots.notna().any() else None,
        "n_ok": int(len(ok)),
    }
    return {"arrays": arrays, "info": info}


def write_featurizer(stem: str | Path, state: dict, table_path: Path) -> dict:
    """``<stem>.featurizer.npz`` + ``.json``; the json pins the npz, the table and the inputs by sha256."""
    npz = Path(f"{stem}.featurizer.npz")
    np.savez_compressed(npz, **state["arrays"])
    info = {**state["info"], "npz": npz.name, "npz_sha256": sha256_file(npz),
            "features_table": Path(table_path).name, "features_sha256": sha256_file(table_path)}
    Path(f"{stem}.featurizer.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def load_featurizer(path: str | Path, roi_map: str | Path | None = None,
                    analyses_dir: str | Path | None = None) -> dict:
    """Load ``<stem>.featurizer.json`` (+ npz), verify every pinned sha256, and build the per-clip state.

    Refuses a mismatched npz, ROI map, proxies file or version: a silent train/serve skew is worse than none."""
    from tribe_research.brain.proxies import PROXIES_FILE

    path = Path(path)
    if path.suffix == ".npz":
        path = path.with_suffix(".json")
    info = json.loads(path.read_text())
    if info.get("featurizer_version") != FEATURIZER_VERSION:
        raise ValueError(f"{path}: featurizer_version {info.get('featurizer_version')!r}, "
                         f"this code reads {FEATURIZER_VERSION!r}")
    for key, want in (("features_version", FEATURES_VERSION), ("emb_version", EMB_VERSION),
                      ("roi_feature_version", FEATURE_VERSION), ("proxies_version", ProxySpec.load().version)):
        if info.get(key) != want:
            raise ValueError(f"{path}: {key} {info.get(key)!r} != this code's {want!r}")
    if sha256_file(PROXIES_FILE) != info["proxies_sha256"]:
        raise ValueError(f"{PROXIES_FILE} changed since the featurizer was built (sha256 mismatch)")
    npz = path.with_name(info["npz"])
    if sha256_file(npz) != info["npz_sha256"]:
        raise ValueError(f"{npz}: sha256 mismatch with {path.name}")
    roi = Path(roi_map or info["roi_map"])
    if sha256_file(roi) != info["roi_map_sha256"]:
        raise ValueError(f"{roi}: ROI map sha256 differs from the one the features were built with")
    with np.load(npz) as z:
        arrays = {k: z[k] for k in z.files}
    return {"info": info, "arrays": arrays, "path": str(path),
            "clip": clip_state(str(roi), str(analyses_dir) if analyses_dir else None)}


def resolve_worker_output(src: str | Path) -> tuple[str, str, str]:
    """A ``<vid>.npz`` / ``<vid>.json`` / ``<vid>.emb.npz`` path, or a directory holding one clip's outputs,
    -> the (video_id, json, npz) item ``featurize`` reads (the emb file is found next to the npz)."""
    p = Path(src)
    if p.is_dir():
        found = [q for q in sorted(p.glob("*.json")) if not q.name.endswith(".error.json")
                 and q.with_suffix(".npz").exists()]
        if len(found) != 1:
            raise ValueError(f"{p}: expected one <vid>.json + <vid>.npz, found {len(found)}")
        p = found[0]
    name = p.name
    for suf in (".emb.npz", ".npz", ".json"):
        if name.endswith(suf):
            vid = name[: -len(suf)]
            break
    else:
        raise ValueError(f"{p}: not a worker output (<vid>.npz, .json or .emb.npz)")
    j, n = p.with_name(f"{vid}.json"), p.with_name(f"{vid}.npz")
    if not (j.exists() and n.exists()):
        raise ValueError(f"{p}: needs both {j.name} and {n.name}")
    return vid, str(j), str(n)


def project(x: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    """PCA projection with the saved mean/components (what sklearn's PCA/IncrementalPCA.transform computes)."""
    return ((x.astype(np.float32) - mean) @ components.T).astype(np.float32)


def featurize_one(worker_output: str | Path, context: dict | None, state: dict) -> pd.DataFrame:
    """One clip -> a one-row frame with the training table's exact columns and dtypes.

    Runs the batch path's ``featurize`` on the clip, projects its time-mean map and emb vectors on the saved
    PCAs (an emb vector of the wrong width is dropped, as in the batch build), then orders and casts the
    columns like ``build``. ``context`` may carry ``shots_ms``; nothing is imputed."""
    context = context or {}
    vid, meta_path, npz_path = resolve_worker_output(worker_output)
    meta = None
    if isinstance(context.get("shots_ms"), list):  # read exactly like a worker json that carries shots_ms
        meta = {**json.loads(Path(meta_path).read_text()), "shots_ms": [int(s) for s in context["shots_ms"]]}
    _, row, vec, emb, rt = featurize((vid, meta_path, npz_path), state["clip"], meta)
    row = {"video_id": vid, **row}
    arrays, blocks = state["arrays"], state["info"]["blocks"]
    if row.get("status") == "ok":
        vectors = {"brain": vec, **{f"emb_{m}": v for m, v in emb.items()}}
        for name, b in blocks.items():
            v = vectors.get(name)
            if v is None or v.size != b["dim"]:
                continue
            z = project(v[None, :], arrays[f"{name}_mean"], arrays[f"{name}_components"])[0]
            row.update(zip(b["columns"], map(float, z)))
    df = pd.DataFrame([row]).reindex(columns=state["info"]["columns"])
    df.attrs["runtime"] = rt
    return cast_feature_columns(df)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-root", type=Path, required=True, action="append",
                    help="worker outputs root (contains worker-*/); repeat for several (first completed copy wins)")
    ap.add_argument("--out", type=Path, required=True, help="features file (.parquet or .csv)")
    ap.add_argument("--roi-map", type=Path, default=DEFAULT_ROI_MAP)
    ap.add_argument("--manifest", type=Path, default=None, help="optional: add manifest columns, mark missing clips")
    ap.add_argument("--analyses-dir", type=Path, default=None,
                    help="optional brain_report analyses/ dir: read events.shots_ms per clip")
    ap.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 2) - 8))
    ap.add_argument("--n-pca", type=int, default=N_PCA)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--exclude-from-pca-fit", type=Path, default=None, metavar="SELECTION",
                    help="study selection.csv (or manifest .jsonl): its lockbox clips are projected, not fit")
    args = ap.parse_args()

    excl = lockbox_ids(args.exclude_from_pca_fit) if args.exclude_from_pca_fit else None
    state: dict = {}
    roots = args.out_root if len(args.out_root) > 1 else args.out_root[0]
    df, meta, ipca = build(roots, args.roi_map, n_jobs=args.n_jobs, manifest=args.manifest,
                           analyses_dir=args.analyses_dir, n_pca=args.n_pca, limit=args.limit, pca_exclude=excl,
                           state_out=state)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix == ".csv":
        df.to_csv(args.out, index=False)
    else:
        df.to_parquet(args.out, index=False)
    stem = args.out.with_suffix("")
    Path(f"{stem}.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    if ipca is not None:
        np.savez_compressed(f"{stem}.pca.npz", mean=ipca.mean_.astype(np.float32),
                            components=ipca.components_.astype(np.float32),
                            explained_variance_ratio=ipca.explained_variance_ratio_)
    write_featurizer(stem, state, args.out)
    print(json.dumps({"out": str(args.out), "rows": len(df), **{k: meta[k] for k in ("status_counts", "timing_s")},
                      "n_base": len(meta["columns"]["base"]), "n_brain": len(meta["columns"]["brain"]),
                      "n_emb": len(meta["columns"]["emb"])}, indent=2))
    return 0 if meta["status_counts"].get("ok", 0) else 1


if __name__ == "__main__":
    sys.exit(main())
