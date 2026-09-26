"""Population-normed, surrogate-calibrated time-resolved features (prereg exploratory family 6).

``moments.py`` z-scores each channel *within the clip*, so every clip has peaks
and every clip hits the moment cap. This module asks the question the other way
round: at this second of a clip of this length, is the predicted response higher
or lower than it usually is across the study clips, and is the departure longer
and larger than the clip's own autocorrelation would produce by chance?

Everything below is fitted on train contents only and never reads an outcome:

- **Norms**: per channel, clip-length bin and second since onset, mean and sd of
  the raw channel curve over train clips. ``u = (raw - mean) / sd``.
- **M1 hook**: mean ``u`` over the first ``HOOK_S`` seconds (Tong et al. 2020:
  the opening seconds carried the forecast in their data).
- **M2 events**: runs of ``+u`` (rise) or ``-u`` (drop) at or above a channel
  threshold lasting at least ``MIN_EVENT_S`` (one hemodynamic response). The
  threshold is the smallest one at which phase-randomised surrogates of the train
  clips (same amplitude spectrum and mean, per contiguous run, so gaps are never
  bridged) produce at most ``FALSE_EVENTS_PER_CLIP`` events per clip summed over
  channels. Zero events is a valid result.
- **M3 event-locked residuals**: a population FIR kernel per channel for shot
  cuts and speech onsets (lags ``FIR_LAGS``), fitted on the train clips' ``u``;
  M2 then runs on ``u - kernel * events`` with its own surrogate threshold. The
  positive control: the sensory channel's cut kernel must peak inside
  ``CONTROL_LAG_S`` with a clip-bootstrap lower bound > 0. Prediction rows are
  already in stimulus time (TRIBE trains against fMRI 5 s later, see
  ``proxies.py``), so the expected lag is near 0-3 s, not 5 s. If the control
  fails, M3 columns are not emitted.

The good-vs-bad contrast (``contrast``) is outcome-facing and lives behind its
own pre-registration row; it takes residualised labels as input and never
fetches them itself.

Research proxy only: these are predictions for an average viewer, not measured
brain activity, and nothing here is calibrated against retention.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

VERSION = "moments_pop_v1"
SEED = 20260926
TR_S = 1.0                                   # worker output: one prediction row per second
LEN_EDGES = (0.0, 15.0, 30.0, 60.0, np.inf)  # clip-length bins [lo, hi)
MAX_SEC_BIN = 60                             # seconds since onset 0..59 each get a bin; >= 60 share one
MIN_NORM_N = 20                              # fewer train rows in a (len, sec) cell -> pool over length bins
HOOK_S = 4.0
MIN_EVENT_S = 5.0
FALSE_EVENTS_PER_CLIP = 0.5                  # summed over channels, surrogate data
N_SURROGATES = 20
THR_GRID = np.round(np.arange(0.5, 6.01, 0.05), 2)
FIR_LAGS = tuple(range(-2, 9))               # seconds after the event
CONTROL_CHANNEL = "sensory"
CONTROL_LAG_S = (0.0, 3.0)
CONTROL_BOOT = 500
FIR_RIDGE = 1.0
POWER_AMPS_SD = (1.0, 1.5, 2.0)             # M2 power check: planted sustained drop, in norm sd
POWER_LEN_S = 7                              # ... lasting this long (> MIN_EVENT_S, one HRF plus a bit)
CONTRAST_HORIZON_S = 30                      # seconds-since-onset view
CONTRAST_FRAC_BINS = 20                      # fraction-of-clip view
CONTRAST_Z = 2.0                             # cluster-forming threshold (per-bin |z| from the permutation sd)
N_PERM = 2000


# ── curves ───────────────────────────────────────────────────────────────


@dataclass
class Clip:
    video_id: str
    t: np.ndarray          # [T] segment starts (s), sorted
    dur: np.ndarray        # [T] segment durations (s)
    raw: np.ndarray        # [C, T] vertex-weighted channel means (proxies.channel_raw)
    duration: float
    shots_s: list[float] | None = None
    speech_onsets_s: list[float] | None = None


def len_bin(duration: float) -> int:
    return int(np.searchsorted(LEN_EDGES, duration, side="right") - 1)


def sec_bin(t: np.ndarray) -> np.ndarray:
    return np.minimum(np.floor(t / TR_S).astype(int), MAX_SEC_BIN)


def runs_of_contiguous(t: np.ndarray, dur: np.ndarray) -> list[tuple[int, int]]:
    """[a, b) index ranges with no gap between consecutive segments (TRIBE drops event-free segments)."""
    out, a = [], 0
    for i in range(1, len(t)):
        if t[i] > t[i - 1] + dur[i - 1] + 1e-6:
            out.append((a, i))
            a = i
    if len(t):
        out.append((a, len(t)))
    return out


# ── norms ────────────────────────────────────────────────────────────────


def fit_norms(clips: list[Clip]) -> dict:
    """Per channel x length bin x second bin: mean, sd, n over train clips; thin cells fall back to all lengths."""
    C = clips[0].raw.shape[0]
    L, S = len(LEN_EDGES) - 1, MAX_SEC_BIN + 1
    s1 = np.zeros((C, L, S)); s2 = np.zeros((C, L, S)); n = np.zeros((L, S))
    for c in clips:
        lb, sb = len_bin(c.duration), sec_bin(c.t)
        np.add.at(n[lb], sb, 1)
        for k in range(C):
            np.add.at(s1[k, lb], sb, c.raw[k])
            np.add.at(s2[k, lb], sb, c.raw[k] ** 2)
    pooled_n = n.sum(0)
    mean = np.full((C, L, S), np.nan); sd = np.full((C, L, S), np.nan)
    for k in range(C):
        pm = s1[k].sum(0) / np.maximum(pooled_n, 1)
        pv = s2[k].sum(0) / np.maximum(pooled_n, 1) - pm ** 2
        with np.errstate(invalid="ignore", divide="ignore"):
            m = s1[k] / n
            v = s2[k] / n - m ** 2
        thin = n < MIN_NORM_N
        m[thin] = np.broadcast_to(pm, m.shape)[thin]
        v[thin] = np.broadcast_to(pv, v.shape)[thin]
        mean[k], sd[k] = m, np.sqrt(np.maximum(v, 0))
    # a cell with no data at all (e.g. second 59 of a 20 s bin) is never read; floor sd for safety
    sd = np.where(np.isfinite(sd) & (sd > 1e-9), sd, np.nan)
    return {"mean": mean, "sd": sd, "n": n, "pooled_n": pooled_n}


def normed(clip: Clip, norms: dict) -> np.ndarray:
    lb, sb = len_bin(clip.duration), sec_bin(clip.t)
    return (clip.raw - norms["mean"][:, lb, sb]) / norms["sd"][:, lb, sb]


# ── events (M2) ──────────────────────────────────────────────────────────


def event_runs(t: np.ndarray, dur: np.ndarray, y: np.ndarray, thr: float, sign: int) -> list[tuple[float, float]]:
    """(start, end) s of contiguous runs with sign*y >= thr lasting >= MIN_EVENT_S; gaps and NaN end a run."""
    out = []
    for a, b in runs_of_contiguous(t, dur):
        on = np.nan_to_num(sign * y[a:b], nan=-np.inf) >= thr
        i = 0
        while i < len(on):
            if on[i]:
                j = i
                while j + 1 < len(on) and on[j + 1]:
                    j += 1
                s, e = float(t[a + i]), float(t[a + j] + dur[a + j])
                if e - s >= MIN_EVENT_S - 1e-6:
                    out.append((s, e))
                i = j + 1
            else:
                i += 1
    return out


def phase_surrogate(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Same amplitude spectrum (and mean), random phases (Theiler et al. 1992)."""
    n = len(y)
    if n < 3 or not np.isfinite(y).all():
        return y.copy()
    f = np.fft.rfft(y)
    ph = rng.uniform(0, 2 * np.pi, len(f))
    ph[0] = 0.0
    if n % 2 == 0:
        ph[-1] = 0.0
    return np.fft.irfft(np.abs(f) * np.exp(1j * ph), n)


def count_events(y: np.ndarray, thr_grid: np.ndarray = THR_GRID) -> np.ndarray:
    """Events (rise + drop) per threshold in one contiguous run of uniform segments: [len(thr_grid)].

    Same rule as ``event_runs`` when every segment lasts TR_S: a run counts once if it has at least
    ceil(MIN_EVENT_S / TR_S) consecutive samples at or beyond the threshold."""
    L = int(np.ceil(MIN_EVENT_S / TR_S - 1e-9))
    out = np.zeros(len(thr_grid))
    if len(y) < L:
        return out
    for sign in (+1, -1):
        B = (sign * np.nan_to_num(y, nan=-np.inf))[None, :] >= thr_grid[:, None]      # [G, n]
        cs = np.concatenate([np.zeros((len(thr_grid), 1)), np.cumsum(B, 1)], 1)
        full = (cs[:, L:] - cs[:, :-L]) == L                                          # window [i, i+L) all on
        start = B.copy()
        start[:, 1:] &= ~B[:, :-1]                                                    # a run begins at i
        out += (start[:, : full.shape[1]] & full).sum(1)
    return out


def surrogate_counts(clips_u: list[tuple[np.ndarray, np.ndarray, np.ndarray]], n_surr: int = N_SURROGATES,
                     seed: int = SEED, thr_grid: np.ndarray = THR_GRID) -> np.ndarray:
    """Mean surrogate events per clip, [C, len(thr_grid)] (rise + drop), each contiguous run randomised alone."""
    rng = np.random.default_rng(seed)
    C = clips_u[0][2].shape[0]
    tot = np.zeros((C, len(thr_grid)))
    for t, dur, u in clips_u:
        runs = runs_of_contiguous(t, dur)
        for _ in range(n_surr):
            for k in range(C):
                for a, b in runs:
                    tot[k] += count_events(phase_surrogate(u[k, a:b], rng), thr_grid)
    return tot / (len(clips_u) * n_surr)


def calibrate(clips_u, n_surr: int = N_SURROGATES, seed: int = SEED) -> dict:
    """Smallest threshold per channel whose surrogate rate is <= FALSE_EVENTS_PER_CLIP / C."""
    rates = surrogate_counts(clips_u, n_surr, seed)
    C = rates.shape[0]
    budget = FALSE_EVENTS_PER_CLIP / C
    thr = []
    for k in range(C):
        ok = np.nonzero(rates[k] <= budget)[0]
        thr.append(float(THR_GRID[ok[0]]) if len(ok) else float("inf"))
    at = [float(rates[k][np.searchsorted(THR_GRID, x)]) if np.isfinite(x) else 0.0 for k, x in enumerate(thr)]
    return {"thr": thr, "surrogate_rate_at_thr": at, "budget_per_channel": budget,
            "n_surrogates": n_surr, "seed": seed}


def event_features(t, dur, u, duration: float, keys: list[str], thr: list[float], prefix: str) -> dict:
    """Per channel: events/min (rise, drop), first sustained drop as a fraction of the clip (1.0 = none)."""
    row = {}
    minutes = max(duration, 1e-6) / 60.0
    n_all = 0
    for k, key in enumerate(keys):
        rises = event_runs(t, dur, u[k], thr[k], +1)
        drops = event_runs(t, dur, u[k], thr[k], -1)
        n_all += len(rises) + len(drops)
        row[f"{prefix}rise_per_min_{key}"] = len(rises) / minutes
        row[f"{prefix}drop_per_min_{key}"] = len(drops) / minutes
        row[f"{prefix}first_drop_frac_{key}"] = min(1.0, drops[0][0] / duration) if drops else 1.0
    row[f"{prefix}events_per_min_all"] = n_all / minutes
    return row


# ── M3: event-locked kernels ─────────────────────────────────────────────


def design(clip: Clip) -> np.ndarray:
    """[T, 2 * len(FIR_LAGS)] FIR design, demeaned within the clip: shot cuts, then speech onsets.

    An event at e counts for the row whose segment contains e + lag. Demeaning (a clip fixed effect)
    means the kernels explain only event-locked transients; the clip's level stays in the residual."""
    cols = []
    for ev in (clip.shots_s or [], clip.speech_onsets_s or []):
        for lag in FIR_LAGS:
            x = np.zeros(len(clip.t))
            for e in ev:
                if e < 0.5:          # the clip start is not a cut
                    continue
                x[(clip.t <= e + lag) & (e + lag < clip.t + clip.dur)] += 1.0
            cols.append(x)
    X = np.stack(cols, 1)
    return X - X.mean(0, keepdims=True)


def _demeaned(u: np.ndarray) -> np.ndarray:
    return (u - np.nanmean(u, 1, keepdims=True)).T


def _solve_kernels(Xs: list[np.ndarray], Ys: list[np.ndarray]) -> np.ndarray:
    X, Y = np.concatenate(Xs), np.concatenate(Ys)
    ok = np.isfinite(Y).all(1)
    X, Y = X[ok], Y[ok]
    return np.linalg.solve(X.T @ X + FIR_RIDGE * np.eye(X.shape[1]), X.T @ Y).T


def fit_kernels(clips: list[Clip], us: list[np.ndarray]) -> np.ndarray:
    """Pooled ridge FIR per channel on population-normed curves: [C, 2 * len(FIR_LAGS)]."""
    return _solve_kernels([design(c) for c in clips], [_demeaned(u) for u in us])


def residual(clip: Clip, u: np.ndarray, kernels: np.ndarray) -> np.ndarray:
    return u - (design(clip) @ kernels.T).T


def positive_control(clips: list[Clip], us: list[np.ndarray], keys: list[str], seed: int = SEED,
                     n_boot: int = CONTROL_BOOT) -> dict:
    """Sensory cut kernel: peak lag within CONTROL_LAG_S and clip-bootstrap 2.5 % bound of its peak > 0."""
    k = keys.index(CONTROL_CHANNEL)
    nl = len(FIR_LAGS)
    n_cut = sum(1 for c in clips if c.shots_s and any(s >= 0.5 for s in c.shots_s))
    if n_cut < 10:
        return {"passed": False, "reason": f"only {n_cut} clips with cuts"}
    Xs, Ys = [design(c) for c in clips], [_demeaned(u) for u in us]
    ker = _solve_kernels(Xs, Ys)[k, :nl]
    j = int(np.argmax(ker))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.choice(len(clips), len(clips))
        boots.append(_solve_kernels([Xs[i] for i in idx], [Ys[i] for i in idx])[k, j])
    lo = float(np.percentile(boots, 2.5))
    lag = FIR_LAGS[j]
    passed = CONTROL_LAG_S[0] <= lag <= CONTROL_LAG_S[1] and lo > 0
    return {"passed": bool(passed), "peak_lag_s": lag, "peak": float(ker[j]), "peak_boot_lo": lo,
            "kernel": [round(float(x), 4) for x in ker], "lags_s": list(FIR_LAGS),
            "expected_lag_s": list(CONTROL_LAG_S), "n_clips_with_cuts": n_cut, "n_boot": n_boot}


def m2_power(clips: list[Clip], norms: dict, thr: list[float], keys: list[str], seed: int = SEED) -> dict:
    """Detection rate of a planted sustained drop (POWER_AMPS_SD x norm sd, POWER_LEN_S s) in real clips.

    The surrogate null keeps each clip's own spectrum, so it is conservative on short series; a null M2 only
    means "no moments" where this rate is high. Reported per amplitude, averaged over channels."""
    rng = np.random.default_rng(seed + 2)
    out = {}
    for amp in POWER_AMPS_SD:
        hits, n = 0, 0
        for c in clips:
            runs = [(a, b) for a, b in runs_of_contiguous(c.t, c.dur) if b - a >= POWER_LEN_S + 2]
            if not runs:
                continue
            a, b = runs[int(rng.integers(len(runs)))]
            i0 = int(rng.integers(a + 1, b - POWER_LEN_S))
            u = normed(c, norms)
            for k in range(len(keys)):
                v = u[k].copy()
                v[i0:i0 + POWER_LEN_S] -= amp
                ev = event_runs(c.t, c.dur, v, thr[k], -1)
                s0, s1 = c.t[i0], c.t[i0 + POWER_LEN_S - 1] + c.dur[i0 + POWER_LEN_S - 1]
                hits += any(e > s0 and s < s1 for s, e in ev)
                n += 1
        out[str(amp)] = round(hits / n, 3) if n else None
    return {"detect_rate_by_amp_sd": out, "len_s": POWER_LEN_S}


# ── the frozen artifact ──────────────────────────────────────────────────


def train_ids_hash(ids) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def fit(clips: list[Clip], keys: list[str], n_surr: int = N_SURROGATES, seed: int = SEED,
        n_boot: int = CONTROL_BOOT) -> dict:
    """Norms, M2 thresholds, M3 kernels + thresholds + positive control, from train clips only."""
    norms = fit_norms(clips)
    us = [normed(c, norms) for c in clips]
    m2 = calibrate([(c.t, c.dur, u) for c, u in zip(clips, us)], n_surr, seed)
    m2["power"] = m2_power(clips, norms, m2["thr"], keys, seed)
    kernels = fit_kernels(clips, us)
    control = positive_control(clips, us, keys, seed, n_boot)
    rs = [residual(c, u, kernels) for c, u in zip(clips, us)]
    m3 = calibrate([(c.t, c.dur, r) for c, r in zip(clips, rs)], n_surr, seed + 1)
    return {"version": VERSION, "channels": keys, "n_train_clips": len(clips),
            "train_ids_sha256": train_ids_hash(c.video_id for c in clips),
            "norms": norms, "m2": m2, "m3": {**m3, "kernels": kernels, "control": control},
            "params": params()}


def params() -> dict:
    return {"tr_s": TR_S, "len_edges": [x if np.isfinite(x) else "inf" for x in LEN_EDGES],
            "max_sec_bin": MAX_SEC_BIN, "min_norm_n": MIN_NORM_N, "hook_s": HOOK_S,
            "min_event_s": MIN_EVENT_S, "false_events_per_clip": FALSE_EVENTS_PER_CLIP,
            "n_surrogates": N_SURROGATES, "seed": SEED,
            "thr_grid": [float(THR_GRID[0]), float(THR_GRID[-1]), 0.05],
            "fir_lags_s": list(FIR_LAGS), "fir_ridge": FIR_RIDGE, "control_channel": CONTROL_CHANNEL,
            "control_lag_s": list(CONTROL_LAG_S), "control_boot": CONTROL_BOOT,
            "power_amps_sd": list(POWER_AMPS_SD), "power_len_s": POWER_LEN_S}


def save(state: dict, stem: str | Path) -> tuple[Path, Path]:
    """<stem>.npz (arrays) + <stem>.json (everything else)."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    npz, js = stem.with_suffix(".npz"), stem.with_suffix(".json")
    np.savez_compressed(npz, mean=state["norms"]["mean"], sd=state["norms"]["sd"], n=state["norms"]["n"],
                        pooled_n=state["norms"]["pooled_n"], kernels=state["m3"]["kernels"])
    meta = {k: v for k, v in state.items() if k != "norms"}
    meta["m3"] = {k: v for k, v in state["m3"].items() if k != "kernels"}
    js.write_text(json.dumps(meta, indent=1, default=_jsonable) + "\n")
    return npz, js


def load(stem: str | Path) -> dict:
    stem = Path(stem)
    meta = json.loads(stem.with_suffix(".json").read_text())
    with np.load(stem.with_suffix(".npz")) as z:
        meta["norms"] = {k: z[k] for k in ("mean", "sd", "n", "pooled_n")}
        meta["m3"]["kernels"] = z["kernels"]
    meta["m2"]["thr"] = [float(x) for x in meta["m2"]["thr"]]
    meta["m3"]["thr"] = [float(x) for x in meta["m3"]["thr"]]
    return meta


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if o == float("inf"):
        return "inf"
    raise TypeError(type(o))


# ── per-clip features ────────────────────────────────────────────────────


def clip_features(clip: Clip, state: dict) -> dict:
    """One row of ``mpop_`` columns. M3 columns only when the positive control passed."""
    keys = state["channels"]
    u = normed(clip, state["norms"])
    row: dict = {}
    hook = clip.t < HOOK_S
    for k, key in enumerate(keys):
        row[f"mpop_hook_{key}"] = float(np.nanmean(u[k, hook])) if hook.any() else float("nan")
    row.update(event_features(clip.t, clip.dur, u, clip.duration, keys, state["m2"]["thr"], "mpop_m2_"))
    if state["m3"]["control"].get("passed"):
        r = residual(clip, u, np.asarray(state["m3"]["kernels"]))
        row.update(event_features(clip.t, clip.dur, r, clip.duration, keys, state["m3"]["thr"], "mpop_m3_"))
    return row


def edit_features(clip: Clip) -> dict:
    """Interpretable, non-brain covariates for the good-vs-bad table (``edit_`` prefix, never a model block)."""
    d = max(clip.duration, 1e-6)
    cuts = [s for s in (clip.shots_s or []) if s >= 0.5]
    on = clip.speech_onsets_s or []
    return {
        "edit_cuts_per_min": len(cuts) / (d / 60) if clip.shots_s is not None else float("nan"),
        "edit_first_cut_s": (cuts[0] if cuts else d) if clip.shots_s is not None else float("nan"),
        "edit_speech_onset_s": on[0] if on else d,
        "edit_speech_onsets_per_min": len(on) / (d / 60),
    }


# ── good vs bad (outcome-facing; needs its prereg row before use on real labels) ──


def grid_curves(clip: Clip, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[C, HORIZON] by seconds since onset and [C, FRAC_BINS] by fraction of clip; NaN where not covered."""
    C = u.shape[0]
    a = np.full((C, CONTRAST_HORIZON_S), np.nan)
    sb = np.floor(clip.t).astype(int)
    m = sb < CONTRAST_HORIZON_S
    a[:, sb[m]] = u[:, m]
    f = np.full((C, CONTRAST_FRAC_BINS), np.nan)
    fb = np.minimum((clip.t / max(clip.duration, 1e-6) * CONTRAST_FRAC_BINS).astype(int), CONTRAST_FRAC_BINS - 1)
    for b in range(CONTRAST_FRAC_BINS):
        sel = fb == b
        if sel.any():
            f[:, b] = np.nanmean(u[:, sel], 1)
    return a, f


def tertile_labels(y: np.ndarray, strata: np.ndarray) -> np.ndarray:
    """+1 top third, -1 bottom third, 0 middle, within each stratum (ranks, ties by order)."""
    lab = np.zeros(len(y), int)
    for s in np.unique(strata):
        ix = np.nonzero(strata == s)[0]
        if len(ix) < 3:
            continue
        order = ix[np.argsort(y[ix], kind="stable")]
        k = len(ix) // 3
        lab[order[:k]], lab[order[-k:]] = -1, +1
    return lab


def _diff(X: np.ndarray, lab: np.ndarray, w: np.ndarray, strata: np.ndarray) -> np.ndarray:
    """Stratum-balanced weighted mean(top) - mean(bottom) per column; strata weighted by their pair count."""
    num = np.zeros(X.shape[1]); den = np.zeros(X.shape[1])
    fin = np.isfinite(X)
    Xz = np.where(fin, X, 0.0)
    for s in np.unique(strata):
        ix = strata == s
        top, bot = ix & (lab == 1), ix & (lab == -1)
        wt = w[top, None] * fin[top]; wb = w[bot, None] * fin[bot]
        st, sb_ = wt.sum(0), wb.sum(0)
        ok = (st > 0) & (sb_ > 0)
        d = np.where(ok, (Xz[top] * wt).sum(0) / np.maximum(st, 1e-12) - (Xz[bot] * wb).sum(0) / np.maximum(sb_, 1e-12), 0)
        n = min(top.sum(), bot.sum()) * ok
        num += d * n; den += n
    with np.errstate(invalid="ignore"):
        return np.where(den > 0, num / den, np.nan)


def _clusters(z: np.ndarray) -> list[tuple[int, int, float]]:
    """Contiguous runs of same-sign |z| >= CONTRAST_Z: (a, b, signed mass)."""
    out, i = [], 0
    while i < len(z):
        if np.isfinite(z[i]) and abs(z[i]) >= CONTRAST_Z:
            sg, j = np.sign(z[i]), i
            while j + 1 < len(z) and np.isfinite(z[j + 1]) and abs(z[j + 1]) >= CONTRAST_Z and np.sign(z[j + 1]) == sg:
                j += 1
            out.append((i, j + 1, float(z[i:j + 1].sum())))
            i = j + 1
        else:
            i += 1
    return out


def contrast(X: np.ndarray, y: np.ndarray, strata: np.ndarray, w: np.ndarray | None = None,
             n_perm: int = N_PERM, seed: int = SEED) -> dict:
    """Where do top-third clips differ from bottom-third clips? X [N, B] one curve per CONTENT (NaN = uncovered).

    Rows must be independent units: a content posted on two platforms is one row (its labels averaged
    first), otherwise identical curves in two strata make the within-stratum permutation null too narrow.

    Labels shuffle within stratum (so deal x platform level never enters); per-bin z uses the permutation
    sd; cluster mass is compared with the permutation max-mass distribution (Maris & Oostenveld 2007),
    which controls family-wise error over bins.
    """
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    lab = tertile_labels(np.asarray(y, float), np.asarray(strata))
    obs = _diff(X, lab, w, strata)
    rng = np.random.default_rng(seed)
    perm = np.empty((n_perm, X.shape[1]))
    for p in range(n_perm):
        lp = lab.copy()
        for s in np.unique(strata):
            ix = np.nonzero(strata == s)[0]
            lp[ix] = lp[rng.permutation(ix)]
        perm[p] = _diff(X, lp, w, strata)
    sd = np.nanstd(perm, 0)
    sd = np.where(sd > 1e-12, sd, np.nan)
    z = obs / sd
    null_max = np.array([max((abs(m) for *_, m in _clusters(pz / sd)), default=0.0) for pz in perm])
    cl = [{"start_bin": a, "end_bin": b, "mass": m,
           "p_fwe": float((1 + (null_max >= abs(m)).sum()) / (1 + n_perm)),
           "direction": "top_higher" if m > 0 else "top_lower"} for a, b, m in _clusters(z)]
    return {"diff": obs, "z": z, "clusters": cl, "n_top": int((lab == 1).sum()), "n_bottom": int((lab == -1).sum()),
            "n_perm": n_perm}
