#!/usr/bin/env python3
"""Population-normed time-resolved features (prereg family 6, M1-M3): shots, fit, features, contrast.

    # 1. shot cuts for M3 (ffmpeg scene score on the pre-scaled clips; niced, few processes)
    tools/build_moments_pop.py shots --batches results/batches_s384 --out results/moments_pop/shots
    # 2. freeze norms, surrogate thresholds, FIR kernels + positive control from TRAIN bf16 clips
    tools/build_moments_pop.py fit OUT_ROOT [OUT_ROOT ...] --selection results/study/selection.csv \
        --shots results/moments_pop/shots --out results/moments_pop/state_v1
    # 3. one mpop_/edit_ row per clip (lockbox rows too: no label is read)
    tools/build_moments_pop.py features OUT_ROOT [...] --state results/moments_pop/state_v1 \
        --shots results/moments_pop/shots --out results/moments_pop/mpop_v1.csv
    # 4. M4 good vs bad (prereg 900001b): arm-A out-of-fold residuals from fit_models
    tools/build_moments_pop.py contrast OUT_ROOT [...] --state ... --shots ... \
        --oof results/models/stage1/oof_predictions.csv --selection results/study/selection.csv \
        --prereg-commit 900001b --out results/moments_pop/contrast_v1.json

``fit`` never reads an outcome and refuses lockbox clips and non-bf16 outputs. Commit the state
before anyone joins its features to outcomes (prereg family 6).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tribe_research.brain import moments_pop as mp  # noqa: E402
from tribe_research.brain.events import speech_spans, words_lane  # noqa: E402
from tribe_research.brain.features import RoiMap  # noqa: E402
from tribe_research.brain.proxies import ProxySpec, channel_masks  # noqa: E402

DEFAULT_ROI_MAP = ROOT / "tribe_research/assets/roi_map_roi_groups_v0.npz"
PRECISION = "bf16"
PRIMARY_TARGET = "log_interactions_rate"
BH_Q = 0.10


def masks_and_keys(roi_map: Path) -> tuple[np.ndarray, list[str]]:
    spec = ProxySpec.load()
    m = channel_masks(spec, RoiMap.load(roi_map)).astype(np.float32)
    return m / m.sum(1, keepdims=True), [c.key for c in spec.channels]


def discover(roots: list[Path]) -> dict[str, tuple[Path, Path]]:
    out: dict[str, tuple[Path, Path]] = {}
    for root in roots:
        for p in sorted(root.glob("worker-*/*.json")):
            if p.name.endswith(".error.json"):
                continue
            npz = p.with_suffix(".npz")
            if npz.exists():
                out.setdefault(p.stem, (p, npz))
    return out


def load_shots(shots_dir: Path | None, vid: str) -> list[float] | None:
    if shots_dir is None:
        return None
    p = shots_dir / f"{vid}.json"
    return [s / 1000 for s in json.loads(p.read_text())["shots_ms"]] if p.exists() else None


def load_clip(vid: str, meta_path: Path, npz_path: Path, masks: np.ndarray, shots_dir: Path | None
              ) -> tuple[mp.Clip, dict]:
    meta = json.loads(meta_path.read_text())
    with np.load(npz_path) as z:
        preds, t, dur = z["preds"], z["seg_start"].astype(float), z["seg_duration"].astype(float)
    o = np.argsort(t, kind="stable")
    t, dur, preds = t[o], dur[o], preds[o].astype(np.float32)
    if not np.isfinite(preds).all():
        raise ValueError("non-finite predictions")
    words = words_lane(meta.get("words") or [])
    onsets = [a / 1000 for a, _ in speech_spans(words)]
    clip = mp.Clip(vid, t, dur, masks @ preds.T, float(meta.get("duration_s") or t[-1] + dur[-1]),
                   load_shots(shots_dir, vid), onsets)
    return clip, meta


def read_split(selection: Path) -> dict[str, str]:
    with open(selection) as f:
        return {r["video_id"]: r["split"] for r in csv.DictReader(f)}


# ── shots ────────────────────────────────────────────────────────────────


def _shots_one(job: tuple[str, str, str]) -> str:
    vid, video, out = job
    from tribe_research.brain.events import SCENE_THRESHOLD, detect_shots
    try:
        shots = detect_shots(video)
        tmp = Path(out).with_suffix(".tmp")
        tmp.write_text(json.dumps({"video_id": vid, "shots_ms": shots, "scene_threshold": SCENE_THRESHOLD,
                                   "source": Path(video).name}))
        tmp.replace(out)
        return "ok"
    except Exception as exc:  # noqa: BLE001
        return f"{vid}: {type(exc).__name__}: {exc}"[:300]


def cmd_shots(a) -> int:
    os.nice(19)
    a.out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for man in sorted(a.batches.glob(f"{a.batch_glob}/manifest.jsonl")):
        for line in man.read_text().splitlines():
            r = json.loads(line)
            video = man.parent / "videos" / r["path"]
            out = a.out / f"{r['video_id']}.json"
            if video.exists() and not out.exists():
                jobs.append((r["video_id"], str(video), str(out)))
    print(f"shots: {len(jobs)} clips to scan with {a.n_jobs} niced processes", flush=True)
    bad = 0
    with ProcessPoolExecutor(a.n_jobs) as ex:
        for i, res in enumerate(ex.map(_shots_one, jobs, chunksize=4), 1):
            if res != "ok":
                bad += 1
                print(res, flush=True)
            if i % 100 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    print(f"shots: {len(jobs) - bad} written, {bad} failed")
    return 0 if not bad else 1


# ── fit ──────────────────────────────────────────────────────────────────


def cmd_fit(a) -> int:
    os.nice(10)
    split = read_split(a.selection)
    masks, keys = masks_and_keys(a.roi_map)
    clips, skipped = [], {}
    for vid, (mj, npz) in sorted(discover(a.out_roots).items()):
        if split.get(vid) != "train":
            skipped[vid] = f"split={split.get(vid, 'not in selection')}"
            continue
        clip, meta = load_clip(vid, mj, npz, masks, a.shots)
        prec = (meta.get("video_config") or {}).get("video_precision")
        if prec != PRECISION or (meta.get("runtime") or {}).get("dry_run"):
            skipped[vid] = f"precision={prec}"
            continue
        if not np.allclose(clip.dur, mp.TR_S):
            skipped[vid] = "non-uniform segments"
            continue
        clips.append(clip)
    if len(clips) < a.min_clips:
        print(f"fit: only {len(clips)} usable train clips (< --min-clips {a.min_clips})", file=sys.stderr)
        return 1
    state = mp.fit(clips, keys, n_surr=a.n_surrogates, n_boot=a.n_boot, budget_mode=a.budget)
    state["inputs"] = {"out_roots": [str(p) for p in a.out_roots], "selection": str(a.selection),
                       "n_skipped": len(skipped), "skipped_reasons": _count(skipped.values()),
                       "n_with_shots": sum(c.shots_s is not None for c in clips), "development_only": a.dev}
    npz, js = mp.save(state, a.out)
    ids = a.out.with_suffix(".train_ids.txt")
    ids.write_text("\n".join(sorted(c.video_id for c in clips)) + "\n")
    print(json.dumps({"n_train_clips": len(clips), "train_ids_sha256": state["train_ids_sha256"],
                      "m2_thr": state["m2"]["thr"], "m2_power": state["m2"]["power"], "m3_thr": state["m3"]["thr"],
                      "control": {k: v for k, v in state["m3"]["control"].items() if k != "kernel"},
                      "files": [str(npz), str(js), str(ids)]}, indent=1, default=mp._jsonable))
    return 0


def _count(vals) -> dict:
    out: dict = {}
    for v in vals:
        k = v.split("=")[0] if "=" in v else v
        out[v if k == "split" else k] = out.get(v if k == "split" else k, 0) + 1
    return out


# ── features ─────────────────────────────────────────────────────────────


def cmd_features(a) -> int:
    import pandas as pd

    state = mp.load(a.state)
    masks, keys = masks_and_keys(a.roi_map)
    if keys != state["channels"]:
        raise SystemExit(f"channel mismatch: state {state['channels']} vs spec {keys}")
    rows = []
    for vid, (mj, npz) in sorted(discover(a.out_roots).items()):
        try:
            clip, _ = load_clip(vid, mj, npz, masks, a.shots)
            rows.append({"video_id": vid, "mpop_version": mp.VERSION, "mpop_has_shots": clip.shots_s is not None,
                         **mp.clip_features(clip, state), **mp.edit_features(clip)})
        except Exception as exc:  # noqa: BLE001
            rows.append({"video_id": vid, "mpop_version": mp.VERSION, "mpop_error": f"{type(exc).__name__}: {exc}"})
    df = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out) if a.out.suffix == ".parquet" else df.to_csv(a.out, index=False)
    print(f"features: {len(df)} clips -> {a.out}; M3 {'on' if state['m3']['control'].get('passed') else 'OFF (control failed)'}")
    return 0


# ── contrast ─────────────────────────────────────────────────────────────


def labels_from_oof(path: Path, target: str = PRIMARY_TARGET):
    """M4 labels (prereg): content-scheme train rows, y - pred_A_stack, weight w, stratum = the deal."""
    import pandas as pd

    o = pd.read_csv(path, dtype={"video_id": str, "vp_id": str, "stratum": str})
    o = o[(o["scheme"] == "content") & (o["target"] == target) & o["stratum"].notna()]
    if o.empty or "pred_A_stack" not in o:
        raise SystemExit(f"{path}: no content-scheme {target} rows with pred_A_stack")
    return pd.DataFrame({"video_id": o["video_id"], "y": o["y"] - o["pred_A_stack"],
                         "stratum": o["stratum"].str.split("|").str[0], "weight": o["w"]})


def refuse_multi_deal(lab):
    """Prereg M4: a content spanning deals is refused, i.e. excluded from the contrast (reported, not re-assigned)."""
    mixed = lab.groupby("video_id")["stratum"].nunique()
    bad = mixed.index[mixed > 1]
    if len(bad):
        print(f"contrast: refused {len(bad)} contents that span several deals: {sorted(bad)}")
    return lab[~lab["video_id"].isin(bad)]


def one_row_per_content(lab):
    """Post-level labels -> one row per content: w-weighted mean y, mean weight; the stratum (deal) must be shared.

    A content cross-posted to several platforms has one curve; keeping one row per post would put that
    curve into several strata and narrow the permutation null."""
    import pandas as pd

    lab = lab.assign(weight=lab["weight"] if "weight" in lab else 1.0)
    mixed = lab.groupby("video_id")["stratum"].nunique()
    if (mixed > 1).any():
        raise SystemExit(f"{int((mixed > 1).sum())} contents span several strata; use stratum = deal "
                         "(platform-free) for the contrast, or pass one row per content")
    rows = [{"video_id": v, "y": float(np.average(g["y"], weights=g["weight"])), "stratum": g["stratum"].iloc[0],
             "weight": float(g["weight"].mean())} for v, g in lab.groupby("video_id", sort=True)]
    return pd.DataFrame(rows)


def cmd_contrast(a) -> int:
    import pandas as pd

    state = mp.load(a.state)
    masks, keys = masks_and_keys(a.roi_map)
    if (a.labels is None) == (a.oof is None):
        raise SystemExit("pass exactly one of --oof (fit_models oof_predictions.csv) or --labels")
    if a.oof is not None:
        lab = refuse_multi_deal(labels_from_oof(a.oof, a.target))
    else:
        lab = pd.read_csv(a.labels, dtype={"video_id": str})
        need = {"video_id", "y", "stratum"}
        if not need <= set(lab.columns):
            raise SystemExit(f"{a.labels}: needs columns {sorted(need)} (+ optional weight)")
    lab = one_row_per_content(lab)
    split = read_split(a.selection)
    n0 = len(lab)
    lab = lab[lab["video_id"].map(split).eq("train")]
    if len(lab) < n0:
        print(f"contrast: dropped {n0 - len(lab)} non-train label rows (lockbox stays sealed)")
    found = discover(a.out_roots)
    curves: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray | None]] = {}
    for vid in lab["video_id"].unique():
        if vid in found:
            clip, _ = load_clip(vid, *found[vid], masks, a.shots)
            u = mp.normed(clip, state["norms"])
            r = mp.residual(clip, u, np.asarray(state["m3"]["kernels"])) if state["m3"]["control"].get("passed") else None
            curves[vid] = (*mp.grid_curves(clip, u), None if r is None else mp.grid_curves(clip, r)[0])
    lab = lab[lab["video_id"].isin(curves)].reset_index(drop=True)
    y, strata = lab["y"].to_numpy(float), lab["stratum"].astype(str).to_numpy()
    w = lab["weight"].to_numpy(float) if "weight" in lab else None
    report = {"version": mp.VERSION, "prereg_commit": a.prereg_commit, "unit": "content", "stratum": "deal",
              "budget_mode": state.get("budget_mode"), "n_deals": int(lab["stratum"].nunique()), "n_rows": len(lab),
              "n_clips": int(lab["video_id"].nunique()), "state_train_ids_sha256": state["train_ids_sha256"],
              "views": {}}
    views = {"onset_s": 0, "fraction": 1, "onset_s_residual": 2}
    for name, i in views.items():
        for k, key in enumerate(keys):
            if curves and curves[lab["video_id"][0]][i] is None:
                continue
            X = np.stack([curves[v][i][k] for v in lab["video_id"]])
            res = mp.contrast(X, y, strata, w, n_perm=a.n_perm)
            for c in res["clusters"]:
                c.update(mp.replicates(X, y, strata, w, c["start_bin"], c["end_bin"]))
            report["views"].setdefault(name, {})[key] = {
                "diff": [None if not np.isfinite(x) else round(float(x), 4) for x in res["diff"]],
                "z": [None if not np.isfinite(x) else round(float(x), 3) for x in res["z"]],
                "clusters": res["clusters"], "n_top": res["n_top"], "n_bottom": res["n_bottom"]}
    # BH q = 0.10 across the channel x view tests; a test's p is its best cluster's FWE p (1 if none)
    tests = [(n, k) for n, v in report["views"].items() for k in v]
    ps = [min((c["p_fwe"] for c in report["views"][n][k]["clusters"]), default=1.0) for n, k in tests]
    for (n, k), p_, q in zip(tests, ps, mp.bh(ps)):
        r = report["views"][n][k]
        r["test_p"], r["test_q_bh"] = p_, q
        for c in r["clusters"]:
            c["counts"] = bool(q <= BH_Q and c["p_fwe"] <= p_ + 1e-12 and c["replicated"])
    report["n_tests"], report["bh_q"] = len(tests), BH_Q
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1, default=mp._jsonable) + "\n")
    for name, v in report["views"].items():
        for key, r in v.items():
            for c in r["clusters"]:
                print(f"{name:18s} {key:10s} bins {c['start_bin']}-{c['end_bin']} {c['direction']} "
                      f"p_fwe={c['p_fwe']:.3f} q={r['test_q_bh']:.3f} replicated={c['replicated']} "
                      f"counts={c['counts']}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("shots")
    s.add_argument("--batches", type=Path, default=ROOT / "results/batches_s384")
    s.add_argument("--out", type=Path, default=ROOT / "results/moments_pop/shots")
    s.add_argument("--batch-glob", default="[bd]*", help="batch dirs to scan (default: study + deep-dive)")
    s.add_argument("--n-jobs", type=int, default=4)
    for name in ("fit", "features", "contrast"):
        p = sub.add_parser(name)
        p.add_argument("out_roots", type=Path, nargs="+")
        p.add_argument("--roi-map", type=Path, default=DEFAULT_ROI_MAP)
        p.add_argument("--shots", type=Path, default=None, help="dir of <vid>.json from the shots step")
        p.add_argument("--out", type=Path, required=True)
        if name == "fit":
            p.add_argument("--selection", type=Path, required=True)
            p.add_argument("--n-surrogates", type=int, default=mp.N_SURROGATES)
            p.add_argument("--n-boot", type=int, default=mp.CONTROL_BOOT)
            p.add_argument("--min-clips", type=int, default=200)
            p.add_argument("--dev", action="store_true", help="mark the state as a development fit (not the frozen one)")
            p.add_argument("--budget", choices=mp.BUDGETS, default="per_channel",
                           help="surrogate false-event budget: per channel (primary) or summed (sensitivity)")
        else:
            p.add_argument("--state", type=Path, required=True, help="stem written by fit")
        if name == "contrast":
            p.add_argument("--oof", type=Path, default=None, help="fit_models oof_predictions.csv (the prereg M4 labels)")
            p.add_argument("--target", default=PRIMARY_TARGET)
            p.add_argument("--labels", type=Path, default=None, help="alternative: video_id, y (residualised), stratum[, weight]")
            p.add_argument("--selection", type=Path, required=True, help="drops non-train rows (lockbox stays sealed)")
            p.add_argument("--prereg-commit", required=True, help="commit that pre-registered this contrast")
            p.add_argument("--n-perm", type=int, default=mp.N_PERM)
    a = ap.parse_args(argv)
    return {"shots": cmd_shots, "fit": cmd_fit, "features": cmd_features, "contrast": cmd_contrast}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
