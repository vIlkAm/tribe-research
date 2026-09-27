#!/usr/bin/env python3
"""Per-clip "response profile against your library" + a global "what the model learned" file.

DESCRIBES the predicted average-viewer brain response of a clip relative to the library (the
1,264 bf16 train clips behind the frozen moments state ``state_v1``). It is NOT a performance
forecast: stage 1 was no-GO and M4 found no brain pattern separating better from worse clips.
Every output carries that caveat.

    # three demo profiles + learned.json + the self-excluded sanity check, one library load
    tools/build_library_profile.py --clips ef3335a5c1b282e3 39a56688d3b5b35e 90136a44fed3f0e6 \
        --learned --sanity 50

Definitions (fixed):

- ``u[c, t]``: ``moments_pop.normed`` with the frozen state norms (nothing is refit).
- Response index ``r[t]``: nan-aware mean of ``u`` over the 7 channels.
- Per-second percentile: ``r[t]`` vs the library's ``r`` in the same (length bin, second bin)
  cell; a cell with fewer than ``MIN_NORM_N`` clips pools over length bins. Mid-rank
  (``less + 0.5 * equal``). A clip in the library is excluded from its own reference (all of
  its rows, including the several it puts into the >= 60 s bin).
- Scores (hook, hold, peak, dead_zones, finish) and channel means: percentile vs the library's
  clip-level distribution of the same statistic, same length bin (pooled if thin).

Labels: ``learned.json`` reads train-split out-of-fold residuals (already used in stage 1) and
filters them to the state's train IDs; ``results/study/selection.csv`` is never opened, so no
lockbox label can be read. Outputs go to ``results/library/`` (git-ignored).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import build_moments_pop as bmp  # noqa: E402
from tribe_research.brain import moments_pop as mp  # noqa: E402

SCHEMA = "nvi.library.v0"
LEARNED_SCHEMA = "nvi.learned.v0"
DEFAULT_STATE = ROOT / "results/moments_pop/state_v1"
DEFAULT_OUT = ROOT / "results/library"
DEFAULT_OOF = ROOT / "results/models/stage1-eval/oof_predictions.csv"
FINISH_S = 3
PEAK_S = 3
DEAD_PCT = 20.0
STANDOUT_HI, STANDOUT_LO = 90.0, 10.0
STANDOUT_MIN_S = 2
MAX_MOMENTS = 3
STRONG, WEAK = 67.0, 33.0
GVB_HORIZON_S = 30
N_BOOT = 1000

CAVEAT = ("Describes how an average viewer's brain is predicted to respond, compared with your library. "
          "In our pre-registered test on 1,486 clips this did not predict views.")
REF_DESCRIPTION = "1,264 of your clips (study set, training split), same clip length and second"
CHANNEL_LABELS = {"attention": "Grabs attention", "social": "Social / people", "value": "Reward / value",
                  "control": "Mental effort", "self": "Personal relevance", "language": "Speech & meaning",
                  "sensory": "Visual & sound intensity"}
SCORE_LABELS = {"hook": "Opening (first 4 s)", "hold": "Middle", "peak": "Strongest 3 seconds",
                "dead_zones": "Low-response seconds", "finish": "Ending (last 3 s)"}
SCORE_NOUNS = {"hook": "opening", "hold": "middle", "peak": "peak moment (best 3 s)", "finish": "ending"}
INDEX_DEFINITION = ("Mean over the 7 channels of the predicted response, expressed relative to your library "
                    "clips of the same length at the same second (0 = library average, units of library sd). "
                    "One value per second.")


# ── library ──────────────────────────────────────────────────────────────


@dataclass
class Entry:
    video_id: str
    duration: float
    lb: int
    sec: np.ndarray      # [T] grid second of each segment (floor of its start)
    sb: np.ndarray       # [T] moments_pop second bin
    u: np.ndarray        # [C, T]
    r: np.ndarray        # [T]


def response_index(u: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"), _quiet():
        return np.nanmean(u, 0)


class _quiet:
    def __enter__(self):
        import warnings
        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore", category=RuntimeWarning)

    def __exit__(self, *a):
        self._w.__exit__(*a)


def entry_from_clip(clip: mp.Clip, norms: dict) -> Entry:
    u = mp.normed(clip, norms)
    return Entry(clip.video_id, float(clip.duration), mp.len_bin(clip.duration),
                 np.floor(clip.t / mp.TR_S + 1e-9).astype(int), mp.sec_bin(clip.t), u, response_index(u))


_MASKS = None


def _init_worker(roi_map: str) -> None:
    global _MASKS
    _MASKS = bmp.masks_and_keys(Path(roi_map))[0]


def _load_one(job):
    vid, mj, npz = job
    clip, _ = bmp.load_clip(vid, Path(mj), Path(npz), _MASKS, None)
    return clip


def load_library(state: dict, stem: Path, roi_map: Path, jobs: int, cache: Path | None) -> list[mp.Clip]:
    """The state's train clips (raw channel curves), from the state's own out_roots; hash-checked."""
    ids = [x for x in stem.with_suffix(".train_ids.txt").read_text().split() if x]
    key = hashlib.sha256((state["train_ids_sha256"] + stem.with_suffix(".json").read_text()).encode()).hexdigest()
    if cache is not None and cache.exists():
        with open(cache, "rb") as f:
            obj = pickle.load(f)
        if obj.get("key") == key:
            return obj["clips"]
    found = bmp.discover([ROOT / p if not Path(p).is_absolute() else Path(p) for p in state["inputs"]["out_roots"]])
    missing = [v for v in ids if v not in found]
    if missing:
        raise SystemExit(f"{len(missing)} library clips not found in the state's out_roots, e.g. {missing[:3]}")
    work = [(v, str(found[v][0]), str(found[v][1])) for v in ids]
    with ProcessPoolExecutor(jobs, initializer=_init_worker, initargs=(str(roi_map),)) as ex:
        clips = list(ex.map(_load_one, work, chunksize=8))
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump({"key": key, "clips": clips}, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(cache)
    return clips


class Library:
    """Reference distributions over library entries; every query can exclude one library clip."""

    def __init__(self, entries: list[Entry], keys: list[str]):
        self.entries, self.keys = entries, keys
        self.index = {e.video_id: i for i, e in enumerate(entries)}
        self.lbs = np.array([e.lb for e in entries])
        vals, owners, lbs, sbs = [], [], [], []
        for i, e in enumerate(entries):
            ok = np.isfinite(e.r)
            vals.append(e.r[ok]); owners.append(np.full(ok.sum(), i)); lbs.append(np.full(ok.sum(), e.lb))
            sbs.append(e.sb[ok])
        self._v, self._o = np.concatenate(vals), np.concatenate(owners)
        self._lb, self._sb = np.concatenate(lbs), np.concatenate(sbs)
        self._cells: dict = {}
        for sb in np.unique(self._sb):
            m = self._sb == sb
            self._cells[("all", int(sb))] = (self._v[m], self._o[m])
            for lb in np.unique(self._lb[m]):
                mm = m & (self._lb == lb)
                self._cells[(int(lb), int(sb))] = (self._v[mm], self._o[mm])
        # per-second percentiles of every library clip (self-excluded), then clip-level statistics
        self.pct = [self.second_percentiles(e, i)[0] for i, e in enumerate(entries)]
        self.stats = {k: np.array([clip_stats(e, p)[k][0] for e, p in zip(entries, self.pct)], float)
                      for k in SCORE_LABELS}
        with _quiet():
            self.chan = np.stack([np.nanmean(e.u, 1) for e in entries])     # [N, C]

    def cell(self, lb: int, sb: int, exclude: int | None) -> tuple[np.ndarray, bool]:
        """Reference r values for (lb, sb) without clip ``exclude``; pooled over lengths if < MIN_NORM_N clips."""
        v, o = self._cells.get((lb, sb), (np.empty(0), np.empty(0, int)))
        keep = o != exclude if exclude is not None else np.ones(len(o), bool)
        if len(np.unique(o[keep])) >= mp.MIN_NORM_N:
            return v[keep], False
        v, o = self._cells.get(("all", sb), (np.empty(0), np.empty(0, int)))
        keep = o != exclude if exclude is not None else np.ones(len(o), bool)
        return v[keep], True

    def second_percentiles(self, e: Entry, exclude: int | None, bands: bool = False):
        """Per segment: percentile of r in its cell, and (optionally) the cell's p25/p50/p75."""
        pct = np.full(len(e.r), np.nan)
        band = np.full((3, len(e.r)), np.nan)
        for sb in np.unique(e.sb):
            m = (e.sb == sb) & np.isfinite(e.r)
            if not m.any():
                continue
            ref, _ = self.cell(e.lb, int(sb), exclude)
            if not len(ref):
                continue
            pct[m] = percentile_of(ref, e.r[m])
            if bands:
                band[:, m] = np.percentile(ref, [25, 50, 75])[:, None]
        return pct, band

    def clip_ref(self, values: np.ndarray, lb: int, exclude: int | None) -> tuple[np.ndarray, bool]:
        keep = np.isfinite(values)
        if exclude is not None:
            keep[exclude] = False
        same = keep & (self.lbs == lb)
        if same.sum() >= mp.MIN_NORM_N:
            return values[same], False
        return values[keep], True


def percentile_of(ref: np.ndarray, x) -> np.ndarray:
    """Mid-rank percentile (0-100) of x within ref."""
    s = np.sort(np.asarray(ref, float))
    x = np.atleast_1d(np.asarray(x, float))
    lo = np.searchsorted(s, x, "left"); hi = np.searchsorted(s, x, "right")
    return 100.0 * (lo + 0.5 * (hi - lo)) / len(s)


# ── per-clip statistics ──────────────────────────────────────────────────


def n_seconds(e: Entry) -> int:
    return max(int(math.ceil(e.duration - 1e-6)), int(e.sec.max()) + 1 if len(e.sec) else 0, 1)


def to_grid(e: Entry, y: np.ndarray) -> np.ndarray:
    """Segment values -> 1 Hz grid (NaN where no prediction); [..., T] -> [..., n_seconds]."""
    y = np.asarray(y, float)
    out = np.full(y.shape[:-1] + (n_seconds(e),), np.nan)
    out[..., e.sec] = y
    return out


def _nanmean(x) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else float("nan")


def clip_stats(e: Entry, pct: np.ndarray) -> dict[str, tuple[float, list[int] | None]]:
    """(value, [start_ms, end_ms]) per score key; NaN value where the window has no prediction."""
    r, p = to_grid(e, e.r), to_grid(e, pct)
    n = len(r)
    dur_ms = int(round(e.duration * 1000))
    ms = lambda s: min(int(s * 1000), dur_ms)  # noqa: E731
    hook_n = int(mp.HOOK_S)
    out = {"hook": (_nanmean(r[:hook_n]), [0, ms(hook_n)])}
    a, b = hook_n, n - FINISH_S
    out["hold"] = (_nanmean(r[a:b]) if b > a else float("nan"), [ms(a), ms(b)] if b > a else None)
    best, at = float("nan"), None
    for i in range(0, n - PEAK_S + 1):
        w = r[i:i + PEAK_S]
        if np.isfinite(w).all() and not (w.mean() <= best):
            best, at = float(w.mean()), i
    out["peak"] = (best, [ms(at), ms(at + PEAK_S)] if at is not None else None)
    fin = np.isfinite(p)
    out["dead_zones"] = (float((p[fin] < DEAD_PCT).sum()) if fin.any() else float("nan"), [0, dur_ms])
    out["finish"] = (_nanmean(r[max(n - FINISH_S, 0):]), [ms(max(n - FINISH_S, 0)), dur_ms])
    return out


def verdict(p) -> str | None:
    if p is None or not np.isfinite(p):
        return None
    return "strong" if p >= STRONG else "weak" if p <= WEAK else "typical"


def score_plain(key: str, v: str | None, p: float | None, pooled: bool, value=None) -> str | None:
    if v is None:
        return None
    ref = "clips in your library" if pooled else "similar-length clips in your library"
    pi = int(round(p))
    if key == "dead_zones":
        tail = f" ({int(value)} s in the bottom fifth of your library at that point)"
        if v == "strong":
            return f"Fewer low-response seconds than {pi}% of {ref}{tail}."
        if v == "weak":
            return f"More low-response seconds than {100 - pi}% of {ref}{tail}."
        return f"A typical number of low-response seconds for {ref}{tail}."
    noun = SCORE_NOUNS[key]
    if v == "strong":
        return f"Higher predicted response in the {noun} than {pi}% of {ref}."
    if v == "weak":
        return f"Lower predicted response in the {noun} than {100 - pi}% of {ref}."
    return f"Typical predicted response in the {noun}: higher than {pi}% of {ref}."


def mmss(ms: int) -> str:
    s = int(ms // 1000)
    return f"{s // 60}:{s % 60:02d}"


def moments(e: Entry, pct: np.ndarray, band: np.ndarray, keys: list[str]) -> list[dict]:
    r, p, p50 = to_grid(e, e.r), to_grid(e, pct), to_grid(e, band[1])
    ug = to_grid(e, e.u)
    dur_ms = int(round(e.duration * 1000))
    found = []
    for kind, on in (("standout_high", np.nan_to_num(p, nan=-1) >= STANDOUT_HI),
                     ("standout_low", np.nan_to_num(p, nan=101) <= STANDOUT_LO)):
        i = 0
        while i < len(on):
            if not on[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(on) and on[j + 1]:
                j += 1
            if j - i + 1 >= STANDOUT_MIN_S:
                strength = _nanmean(np.abs(r[i:j + 1] - p50[i:j + 1]))
                with _quiet():
                    cm = np.nanmean(ug[:, i:j + 1], 1)
                order = np.argsort(-cm if kind == "standout_high" else cm, kind="stable")
                top = [keys[k] for k in order if np.isfinite(cm[k])][:2]
                a, b = i * 1000, min((j + 1) * 1000, dur_ms)
                names = " and ".join(CHANNEL_LABELS[k] for k in top)
                if kind == "standout_high":
                    plain = (f"{mmss(a)}–{mmss(b)}: predicted response in the top 10% of your library at this "
                             f"point, led by {names}.")
                else:
                    plain = (f"{mmss(a)}–{mmss(b)}: predicted response in the bottom 10% of your library at this "
                             f"point, lowest in {names}.")
                found.append({"start_ms": a, "end_ms": b, "kind": kind, "channels": top,
                              "channels_plain": [CHANNEL_LABELS[k] for k in top],
                              "strength": strength, "plain": plain})
            i = j + 1
    found.sort(key=lambda m: -m["strength"])
    return sorted(found[:MAX_MOMENTS], key=lambda m: m["start_ms"])


def summary(scores: dict[str, dict]) -> list[str]:
    h, hold, pk, fin = scores["hook"], scores["hold"], scores["peak"], scores["finish"]
    s1 = h["plain"] or "There is no prediction for the opening seconds."
    word = {"strong": "stronger than in most", "weak": "weaker than in most", "typical": "about the same as in most"}
    ref = "clips in your library" if hold.get("_pooled") else "similar-length clips"
    if pk["verdict"] is not None:
        pref = "clips in your library" if pk.get("_pooled") else "similar-length clips"
        tail = (f"its strongest 3 seconds (from {mmss(pk['window_ms'][0])}) are higher than in "
                f"{int(round(pk['percentile']))}% of {pref}.")
    else:
        tail = None
    if hold["verdict"] is not None:
        s2 = f"Through the middle the predicted response is {word[hold['verdict']]} {ref}"
        s2 += f"; {tail}" if tail else "."
    elif tail:
        s2 = f"The clip is too short for a separate middle; {tail}"
    else:
        s2 = "The clip is too short for a separate middle."
    s3 = fin["plain"] or "There is no prediction for the ending."
    return [s1, s2, s3]


def profile(e: Entry, lib: Library, exclude: int | None, state: dict) -> dict:
    keys = lib.keys
    pct, band = lib.second_percentiles(e, exclude, bands=True)
    st = clip_stats(e, pct)
    scores = {}
    for k in SCORE_LABELS:
        value, win = st[k]
        if np.isfinite(value):
            ref, pooled = lib.clip_ref(lib.stats[k], e.lb, exclude)
            p = float(percentile_of(ref, value)[0])
            if k == "dead_zones":
                p = 100.0 - p                     # higher = better (fewer low-response seconds)
        else:
            p, pooled = None, False
        v = verdict(p)
        scores[k] = {"key": k, "label": SCORE_LABELS[k], "value": value if np.isfinite(value) else None,
                     "percentile": p, "verdict": v, "plain": score_plain(k, v, p, pooled, value),
                     "window_ms": win, "_pooled": pooled}
    with _quiet():
        cm = np.nanmean(e.u, 1)
    channels = []
    for k, key in enumerate(keys):
        if np.isfinite(cm[k]):
            ref, _ = lib.clip_ref(lib.chan[:, k], e.lb, exclude)
            p = float(percentile_of(ref, cm[k])[0])
        else:
            p = None
        channels.append({"key": key, "label_plain": CHANNEL_LABELS[key], "percentile": p, "verdict": verdict(p)})
    summ = summary(scores)
    n_ref = len(lib.entries) - (exclude is not None)
    out = {
        "schema_version": SCHEMA, "video_id": e.video_id, "duration_ms": int(round(e.duration * 1000)),
        "reference": {"n_clips": n_ref, "description": REF_DESCRIPTION, "state": "moments_pop state_v1",
                      "state_train_ids_sha256": state["train_ids_sha256"], "self_excluded": exclude is not None},
        "caveat": CAVEAT,
        "index": {"definition": INDEX_DEFINITION, "hz": 1, "values": to_grid(e, e.r),
                  "percentile": to_grid(e, pct),
                  "band": {"p25": to_grid(e, band[0]), "p50": to_grid(e, band[1]), "p75": to_grid(e, band[2])}},
        "scores": [{k2: v2 for k2, v2 in s.items() if not k2.startswith("_")} for s in scores.values()],
        "channels": channels,
        "moments": [{k2: v2 for k2, v2 in m.items() if k2 != "strength"} for m in moments(e, pct, band, keys)],
        "summary": summ,
    }
    return clean(out)


def clean(o):
    """Round floats to 3 dp; NaN/inf -> None; numpy -> python."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return round(float(o), 3) if np.isfinite(o) else None
    return o


def write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, allow_nan=False) + "\n")
    tmp.replace(path)


# ── learned.json ─────────────────────────────────────────────────────────


def pc1(lib: Library) -> dict:
    U = np.concatenate([e.u.T for e in lib.entries])
    U = U[np.isfinite(U).all(1)]
    U = U - U.mean(0)
    _, s, vt = np.linalg.svd(U, full_matrices=False)
    load = vt[0] * (1 if vt[0].sum() >= 0 else -1)
    ev = s ** 2 / (s ** 2).sum()
    equal = 1 / np.sqrt(len(load))
    return {"channels": lib.keys, "loadings": {k: float(x) for k, x in zip(lib.keys, load)},
            "explained_variance_ratio": float(ev[0]), "n_rows": int(len(U)),
            "equal_weight_loading": float(equal),
            "cosine_with_equal_weights": float(load.sum() * equal),
            "note": "PC1 of the population-normed channel values over all library seconds. The response index "
                    "uses the plain channel mean, not PC1; this is a check that the mean is a fair summary."}


def boot_curves(X: np.ndarray, rng: np.random.Generator, n_boot: int = N_BOOT) -> tuple:
    """X [n, S] (NaN = not covered) -> mean, lo, hi per column and the bootstrap draws [n_boot, S]."""
    with _quiet():
        mean = np.nanmean(X, 0)
        idx = rng.integers(0, len(X), (n_boot, len(X)))
        draws = np.nanmean(X[idx], 1)
        lo, hi = np.nanpercentile(draws, [2.5, 97.5], 0)
    return mean, lo, hi, draws


def _gvb_curves(lib: Library, lab, rng: np.random.Generator) -> tuple:
    """Top vs bottom within-deal thirds of lab["y"] -> (lab, top, bot, R, index block, channel blocks)."""
    y, strata = lab["y"].to_numpy(float), lab["stratum"].astype(str).to_numpy()
    tl = mp.tertile_labels(y, strata)
    S = GVB_HORIZON_S

    def grid30(vid, which):
        e = lib.entries[lib.index[vid]]
        a = np.full(which.shape[:-1] + (S,), np.nan) if which.ndim > 1 else np.full(S, np.nan)
        m = e.sec < S
        a[..., e.sec[m]] = which[..., m]
        return a

    R = np.stack([grid30(v, lib.entries[lib.index[v]].r) for v in lab["video_id"]])          # [n, S]
    U = np.stack([grid30(v, lib.entries[lib.index[v]].u) for v in lab["video_id"]])          # [n, C, S]
    top, bot = tl == 1, tl == -1

    def block(Xt, Xb):
        mt, lt, ht, dt = boot_curves(Xt, rng)
        mb, lb_, hb, db = boot_curves(Xb, rng)
        with _quiet():
            dlo, dhi = np.nanpercentile(dt - db, [2.5, 97.5], 0)
        return {"top": {"mean": mt, "lo": lt, "hi": ht, "n_per_second": np.isfinite(Xt).sum(0)},
                "bottom": {"mean": mb, "lo": lb_, "hi": hb, "n_per_second": np.isfinite(Xb).sum(0)},
                "diff_top_minus_bottom": {"mean": mt - mb, "lo": dlo, "hi": dhi}}

    index = block(R[top], R[bot])
    channels = {key: {"label_plain": CHANNEL_LABELS[key], **block(U[top, k], U[bot, k])}
                for k, key in enumerate(lib.keys)}
    return tl, top, bot, R, index, channels


def _labels(oof: Path, train_ids: set[str], lib: Library, target: str, residual: bool):
    lab = bmp.labels_from_oof(oof, target)
    if not residual:  # raw target: labels_from_oof gives y - pred_A_stack; add the prediction back
        import pandas as pd
        o = pd.read_csv(oof, dtype={"video_id": str, "stratum": str})
        o = o[(o["scheme"] == "content") & (o["target"] == target) & o["stratum"].notna()]
        lab = lab.assign(y=o["y"].to_numpy(float))
    lab = bmp.refuse_multi_deal(lab)
    n0 = len(bmp.one_row_per_content(lab))
    lab = bmp.one_row_per_content(lab)
    lab = lab[lab["video_id"].isin(train_ids) & lab["video_id"].isin(list(lib.index))].reset_index(drop=True)
    return lab, n0


def good_vs_bad(lib: Library, oof: Path, train_ids: set[str]) -> dict:
    lab, n0 = _labels(oof, train_ids, lib, bmp.PRIMARY_TARGET, residual=True)
    S = GVB_HORIZON_S
    rng = np.random.default_rng(mp.SEED)
    tl, top, bot, R, index, channels = _gvb_curves(lib, lab, rng)

    out = {
        "definition": ("Within each deal, clips are split into thirds by how much better or worse they did than "
                       "the metadata-only model expected (log_interactions_rate residual after the arm-A "
                       "out-of-fold prediction, content scheme, training split only). Curves are the unweighted "
                       "mean response index per second since the start (0-29 s) for the top and bottom thirds; "
                       "bands are 95% bootstrap intervals over contents (seed 20260926, 1,000 draws). "
                       "The pre-registered M4 test itself was weighted and stratum-balanced."),
        "unit": "content", "stratum": "deal", "target": bmp.PRIMARY_TARGET, "seconds": list(range(S)),
        "n_contents": int(len(lab)), "n_label_rows_before_train_filter": int(n0),
        "n_top": int(top.sum()), "n_bottom": int(bot.sum()), "n_deals": int(lab["stratum"].nunique()),
        "weighting": "unweighted", "n_boot": N_BOOT, "seed": mp.SEED,
        "index": index,
        "channels": channels,
        "result_plain": ("Clips with more and with fewer likes and comments per view than expected show the same "
                         "predicted brain response over time; the pre-registered test (M4) found no reliable "
                         "difference. The groups are by engagement per view, not views: clips with fewer views "
                         "tend to have higher rates."),
        "m4": {"min_p_fwe": 0.134, "q_bh_all": 0.766, "clusters_surviving": 0, "n_tests": 21,
               "prereg_commit": "900001b"},
    }
    return out



def _group_diff(Xt: np.ndarray, Xb: np.ndarray, rng: np.random.Generator, n_boot: int = N_BOOT) -> dict:
    """Top minus bottom in the per-clip mean over the given seconds, bootstrap 95% interval over clips."""
    with _quiet():
        t, b = np.nanmean(Xt, 1), np.nanmean(Xb, 1)
    t, b = t[np.isfinite(t)], b[np.isfinite(b)]
    draws = np.array([rng.choice(t, len(t)).mean() - rng.choice(b, len(b)).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"diff": float(t.mean() - b.mean()), "lo": float(lo), "hi": float(hi), "top_mean": float(t.mean()),
            "bottom_mean": float(b.mean())}


def views_vs_usual(lib: Library, oof: Path, train_ids: set[str]) -> dict:
    """Learned chart grouped by views against the account's usual (design: LIBRARY_THEORY.md, 2026-09-27)."""
    lab, n0 = _labels(oof, train_ids, lib, "reach_rel_local", residual=False)
    S = GVB_HORIZON_S
    rng = np.random.default_rng(mp.SEED)
    tl, top, bot, R, index, channels = _gvb_curves(lib, lab, rng)
    whole = _group_diff(R[top], R[bot], rng)
    opening = _group_diff(R[top][:, :int(mp.HOOK_S)], R[bot][:, :int(mp.HOOK_S)], rng)
    d = index["diff_top_minus_bottom"]
    sig = [s for s, lo, hi in zip(range(S), d["lo"], d["hi"]) if np.isfinite(lo) and (lo > 0 or hi < 0)]

    def says(x):
        if x["lo"] > 0:
            return "higher"
        return "lower" if x["hi"] < 0 else "about the same"
    result = (f"Over the first 30 seconds, clips that beat their account's usual views had a {says(whole)} predicted "
              f"response than clips below it (difference {whole['diff']:+.2f} library sd, 95% range "
              f"{whole['lo']:+.2f} to {whole['hi']:+.2f}). In the opening 4 seconds it was {says(opening)} "
              f"({opening['diff']:+.2f}, {opening['lo']:+.2f} to {opening['hi']:+.2f}). Exploratory, not the "
              "pre-registered test.")
    return {
        "definition": ("Within each deal, clips are split into thirds by views against their account's usual "
                       "(reach_rel_local, content-scheme training rows, one row per content). Curves are the "
                       "unweighted mean response index per second (0-29 s) for the top and bottom thirds; bands are "
                       "95% bootstrap intervals over contents (seed 20260926, 1,000 draws). Design fixed before it was "
                       "computed: docs/LIBRARY_THEORY.md, 2026-09-27."),
        "unit": "content", "stratum": "deal", "target": "reach_rel_local", "seconds": list(range(S)),
        "n_contents": int(len(lab)), "n_label_rows_before_train_filter": int(n0),
        "n_top": int(top.sum()), "n_bottom": int(bot.sum()), "n_deals": int(lab["stratum"].nunique()),
        "weighting": "unweighted", "n_boot": N_BOOT, "seed": mp.SEED, "exploratory": True,
        "index": index, "channels": channels,
        "summary": {"whole_0_29": whole, "opening_0_4": opening, "seconds_ci_excludes_0": sig,
                    "n_seconds": S},
        "result_plain": result,
    }

STAGE1 = {
    "result": "no-GO",
    "lines": [
        "Metadata alone (platform, account, timing, length) ranks clips within each deal and platform at about "
        "0.34 (Spearman correlation with interactions rate).",
        "Adding the predicted brain features adds about nothing (+0.004; brain on top of video features +0.0004).",
        "Video features (the model's audio/visual/text embeddings) add a small real signal for reach (+0.07).",
        "The pre-registered rule needed +0.02 on the primary endpoint and a non-negative account check; both "
        "failed, so the result is no-GO and the product shows no performance number.",
    ],
    "primary_target": "log_interactions_rate",
    "rho_A_metadata": 0.335, "rho_B_brain": 0.339, "rho_E_video": 0.343, "rho_BE_brain_video": 0.344,
    "BE_minus_A_content": {"point": 0.009, "ci95": [-0.009, 0.024]},
    "BE_minus_A_account": {"point": -0.012, "ci95": [-0.030, 0.015]},
    "BE_minus_E_content": {"point": 0.0004, "ci95": [-0.010, 0.010]},
    "reach_E_minus_A_content": {"point": 0.073, "ci95": [0.025, 0.117]},
    "go_rule": "BE - A point >= +0.02 and CI lower > -0.03 (content), and account-scheme point >= 0",
    "n_clips_ok": 1486, "n_train_clips": 1264, "n_contents_primary": 1157, "kish_n_eff": 470,
    "source": "docs/STAGE1_RESULT.md",
}


def learned(lib: Library, state: dict, oof: Path, train_ids: set[str]) -> dict:
    pc = pc1(lib)
    gvb = good_vs_bad(lib, oof, train_ids)
    gvb["pc1_loadings"] = pc
    vvu = views_vs_usual(lib, oof, train_ids)
    return clean({
        "schema_version": LEARNED_SCHEMA, "internal": True,
        "caveat": CAVEAT + " This file is an internal walkthrough of what the study found.",
        "reference": {"n_clips": len(lib.entries), "description": REF_DESCRIPTION, "state": "moments_pop state_v1",
                      "state_train_ids_sha256": state["train_ids_sha256"]},
        "channels": [{"key": k, "label_plain": CHANNEL_LABELS[k]} for k in lib.keys],
        "stage1": STAGE1, "good_vs_bad": gvb, "views_vs_usual": vvu, "pc1_loadings": pc,
    })


# ── sanity ───────────────────────────────────────────────────────────────


def sanity(lib: Library, state: dict, n: int) -> dict:
    rng = np.random.default_rng(mp.SEED)
    pick = sorted(rng.choice(len(lib.entries), size=min(n, len(lib.entries)), replace=False).tolist())
    per = {k: [] for k in SCORE_LABELS}
    chan = []
    bad_nan = 0
    for i in pick:
        prof = profile(lib.entries[i], lib, i, state)
        for s in prof["scores"]:
            if s["percentile"] is not None:
                per[s["key"]].append(s["percentile"])
        chan += [c["percentile"] for c in prof["channels"] if c["percentile"] is not None]
        ix = prof["index"]
        for j, v in enumerate(ix["values"]):
            if (v is None) != (ix["percentile"][j] is None) or (v is None) != (ix["band"]["p50"][j] is None):
                bad_nan += 1
    allp = np.concatenate([p[np.isfinite(p)] for p in lib.pct])
    dz = lib.stats["dead_zones"]
    return clean({
        "n_clips": len(pick), "seed": mp.SEED,
        "score_percentiles": {k: {"n": len(v), "mean": float(np.mean(v)) if v else None,
                                  "sd": float(np.std(v)) if v else None} for k, v in per.items()},
        "channel_percentiles": {"n": len(chan), "mean": float(np.mean(chan)), "sd": float(np.std(chan))},
        "uniform_expectation": {"mean": 50.0, "sd": 28.87},
        "library_seconds": int(len(allp)),
        "library_frac_seconds_pct_below_20": float((allp < DEAD_PCT).mean()),
        "library_frac_seconds_pct_at_least_90": float((allp >= STANDOUT_HI).mean()),
        "dead_zones_frac_clips_zero": float(np.mean(dz[np.isfinite(dz)] == 0)),
        "null_mismatch_seconds": bad_nan,
    })


# ── main ─────────────────────────────────────────────────────────────────


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", nargs="*", default=[], help="video IDs to profile")
    ap.add_argument("--extra-roots", type=Path, nargs="*", default=[],
                    help="more worker output roots to find non-library clips in (reads brain outputs only)")
    ap.add_argument("--learned", action="store_true", help="write learned.json")
    ap.add_argument("--sanity", type=int, default=0, help="score N random library clips (self-excluded)")
    ap.add_argument("--state", type=Path, default=DEFAULT_STATE)
    ap.add_argument("--oof", type=Path, default=DEFAULT_OOF)
    ap.add_argument("--roi-map", type=Path, default=bmp.DEFAULT_ROI_MAP)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args(argv)
    if os.nice(0) < 10:
        os.nice(10 - os.nice(0))
    t0 = time.time()
    state = mp.load(a.state)
    masks, keys = bmp.masks_and_keys(a.roi_map)
    if keys != state["channels"]:
        raise SystemExit(f"channel mismatch: state {state['channels']} vs spec {keys}")
    cache = None if a.no_cache else a.out_dir / ".library_cache_v0.pkl"
    clips = load_library(state, a.state, a.roi_map, a.jobs, cache)
    if mp.train_ids_hash(c.video_id for c in clips) != state["train_ids_sha256"]:
        raise SystemExit("library IDs do not match the state's train_ids_sha256")
    print(f"library: {len(clips)} clips loaded in {time.time() - t0:.0f}s", flush=True)
    lib = Library([entry_from_clip(c, state["norms"]) for c in clips], keys)
    print(f"library: reference built in {time.time() - t0:.0f}s", flush=True)
    found = None
    for vid in a.clips:
        if vid in lib.index:
            i = lib.index[vid]
            e, excl = lib.entries[i], i
        else:
            if found is None:
                roots = [ROOT / p for p in state["inputs"]["out_roots"]] + list(a.extra_roots)
                found = bmp.discover(roots)
            if vid not in found:
                print(f"{vid}: no worker output found", file=sys.stderr)
                return 1
            clip, _ = bmp.load_clip(vid, *found[vid], masks, None)
            e, excl = entry_from_clip(clip, state["norms"]), None
        prof = profile(e, lib, excl, state)
        write_json(prof, a.out_dir / f"{vid}.library.json")
        print(json.dumps({"video_id": vid, "self_excluded": prof["reference"]["self_excluded"],
                          "summary": prof["summary"],
                          "scores": {s["key"]: [s["value"], s["percentile"], s["verdict"]] for s in prof["scores"]},
                          "moments": [m["plain"] for m in prof["moments"]]}, indent=1), flush=True)
    if a.learned:
        ids = {x for x in a.state.with_suffix(".train_ids.txt").read_text().split() if x}
        out = learned(lib, state, a.oof, ids)
        write_json(out, a.out_dir / "learned.json")
        g = out["good_vs_bad"]
        print(json.dumps({"n_top": g["n_top"], "n_bottom": g["n_bottom"], "n_contents": g["n_contents"],
                          "top_mean_r_0_29": _nanmean([x for x in g["index"]["top"]["mean"] if x is not None]),
                          "bottom_mean_r_0_29": _nanmean([x for x in g["index"]["bottom"]["mean"] if x is not None]),
                          "diff_ci_excludes_0_seconds": [s for s, lo, hi in zip(g["seconds"],
                                                         g["index"]["diff_top_minus_bottom"]["lo"],
                                                         g["index"]["diff_top_minus_bottom"]["hi"])
                                                         if lo is not None and (lo > 0 or hi < 0)],
                          "pc1": out["pc1_loadings"]}, indent=1), flush=True)
    if a.sanity:
        s = sanity(lib, state, a.sanity)
        write_json(s, a.out_dir / "sanity.json")
        print(json.dumps(s, indent=1), flush=True)
    print(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
