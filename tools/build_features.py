#!/usr/bin/env python3
"""Per-clip feature table from TRIBE worker outputs (CPU, parallel over clips).

Scans ``<out-root>/worker-*/<video_id>.json|.npz`` (pod/worker.py format) and
writes one row per ``video_id``:

    <out>                      features (.parquet, or .csv by suffix)
    <out stem>.meta.json       feature list, families, versions, PCA variance, counts
    <out stem>.pca.npz         PCA of the per-clip time-mean cortex map (mean, components)

Column prefixes decide how tools/fit_models.py uses a column:

    base_*   non-brain features available without TRIBE predictions
             (duration, speech from the saved words, shots if provided)
    brain_*  derived from TRIBE predictions
             brain_<channel>_*   per proxy channel (within-clip z unless ``raw``)
             brain_moments_*     rule-based candidate moments (tribe_research/brain/moments.py)
             brain_xch_*         cross-channel trajectory features
             brain_pca_NN        raw-cortex time-mean map projected on the top PCA components
    qc_*     quality/diagnostics; used by neither model
    status   ok | failed (worker .error.json) | error (could not featurize) | missing (manifest only)

Brain values are predictions for an average subject. Within-clip z features
are relative to the clip; ``raw`` and ``pca`` features compare model-space
levels across clips, which is exactly what the modelling step is testing.

    python tools/build_features.py --out-root outputs --out results/features/features.parquet \
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

# per-process state (set by _init)
_STATE: dict = {}


# ── per-clip computation ─────────────────────────────────────────────────


def _init(roi_map_path: str, analyses_dir: str | None) -> None:
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(1)
    except Exception:  # noqa: BLE001  (optional; env vars below also help)
        pass
    spec = ProxySpec.load()
    roi = RoiMap.load(roi_map_path)
    masks = channel_masks(spec, roi).astype(np.float32)
    _STATE.update(spec=spec, masks=masks / masks.sum(axis=1, keepdims=True), n_vertices=masks.shape[1],
                  analyses_dir=Path(analyses_dir) if analyses_dir else None)


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


def _shots_for(meta: dict) -> list[int] | None:
    if isinstance(meta.get("shots_ms"), list):
        return [int(s) for s in meta["shots_ms"]]
    d = _STATE.get("analyses_dir")
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


def featurize(item: tuple[str, str, str]) -> tuple[str, dict, np.ndarray | None]:
    """(video_id, meta.json path, npz path) -> (video_id, row, time-mean vertex map or None)."""
    vid, meta_path, npz_path = item
    try:
        meta = json.loads(Path(meta_path).read_text())
        with np.load(npz_path) as zf:
            preds = zf["preds"]
            t = zf["seg_start"].astype(np.float64)
            dur = zf["seg_duration"].astype(np.float64)
        if preds.ndim != 2 or preds.shape[0] != len(t) or len(t) == 0:
            raise ValueError(f"bad preds shape {preds.shape} for {len(t)} segments")
        if preds.shape[1] != _STATE["n_vertices"]:
            raise ValueError(f"preds has {preds.shape[1]} vertices, ROI map {_STATE['n_vertices']}")
        order = np.argsort(t, kind="stable")
        t, dur, preds = t[order], dur[order], preds[order]
        p32 = preds.astype(np.float32)
        if not np.isfinite(p32).all():
            raise ValueError("non-finite predictions")
        duration = float(meta.get("duration_s") or (t[-1] + dur[-1]))
        raw = _STATE["masks"] @ p32.T  # [C, T] vertex-weighted channel means
        shots = _shots_for(meta)
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
            **base_features(meta, duration, shots),
            **brain_features(t, dur, raw, duration, _STATE["spec"], shots, speech_spans(words)),
        }
        return vid, row, p32.mean(axis=0)
    except Exception as exc:  # noqa: BLE001  one bad clip never stops the table
        return vid, {"status": "error", "error": f"{type(exc).__name__}: {exc}"[:500]}, None


# ── discovery ────────────────────────────────────────────────────────────


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


def build(out_root: Path, roi_map: Path, *, n_jobs: int, manifest: Path | None = None,
          analyses_dir: Path | None = None, n_pca: int = N_PCA, limit: int | None = None,
          pca_exclude: set[str] | None = None, log=print) -> tuple[pd.DataFrame, dict, object]:
    t0 = time.perf_counter()
    items, failed = discover(out_root)
    if limit is not None:
        items = items[:limit]
    log(f"{len(items)} completed clips, {len(failed)} failed/incomplete under {out_root}")
    rows: dict[str, dict] = {}
    vecs: dict[str, np.ndarray] = {}
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
        for i, (vid, row, vec) in enumerate(results, 1):
            rows[vid] = row
            if vec is not None:
                vecs[vid] = vec
            if i % 2000 == 0:
                log(f"  {i}/{len(items)} clips ({time.perf_counter() - t0:.0f}s)")
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    t_feat = time.perf_counter() - t0

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
    for c in [c for c in df.columns if c.startswith(("base_", "brain_", "qc_"))]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

    counts = df["status"].value_counts().to_dict()
    meta = {
        "features_version": FEATURES_VERSION,
        "roi_feature_version": FEATURE_VERSION,
        "proxies_version": ProxySpec.load().version,
        "roi_map": str(roi_map),
        "out_root": str(out_root),
        "status_counts": {str(k): int(v) for k, v in counts.items()},
        "synthetic_rows": int(df.get("synthetic", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
        "columns": {
            "base": [c for c in df.columns if c.startswith("base_")],
            "brain": [c for c in df.columns if c.startswith("brain_")],
            "qc": [c for c in df.columns if c.startswith("qc_")],
        },
        "pca": pca_info,
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
    return df, meta, ipca


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-root", type=Path, required=True, help="worker outputs root (contains worker-*/)")
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
    df, meta, ipca = build(args.out_root, args.roi_map, n_jobs=args.n_jobs, manifest=args.manifest,
                           analyses_dir=args.analyses_dir, n_pca=args.n_pca, limit=args.limit, pca_exclude=excl)
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
    print(json.dumps({"out": str(args.out), "rows": len(df), **{k: meta[k] for k in ("status_counts", "timing_s")},
                      "n_base": len(meta["columns"]["base"]), "n_brain": len(meta["columns"]["brain"])}, indent=2))
    return 0 if meta["status_counts"].get("ok", 0) else 1


if __name__ == "__main__":
    sys.exit(main())
