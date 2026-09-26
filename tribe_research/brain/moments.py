"""Rule-based candidate moments from neural proxy curves + event lanes.

Every moment is *relative to this clip* (curves are within-clip z-scores, which
always have peaks, even for noise). Hysteresis and a minimum duration keep
jitter out, but nothing here is calibrated against outcomes: moments are
observations or untested edit hypotheses, never predicted behavior deltas.
"""

from __future__ import annotations

import numpy as np

from tribe_research.brain.proxies import ONSET_S, ProxyDef

EXPLAIN_VERSION = "neural-moments-0.1.0"
ENTER_Z, EXIT_Z = 1.0, 0.5
MIN_RUN_S = 1.5
DROP_LOOKBACK_S = 3.0
DROP_MIN_Z = 1.5
BROAD_MIN_CHANNELS = 3
MAX_MOMENTS = 12
TEST_METRIC = "retention over this span (observed, or predicted once the behavior model exists)"


def _runs(t, dur, z, sign):
    """Hysteresis runs where sign*z enters >= ENTER_Z and stays >= EXIT_Z; gaps end a run."""
    out, cur = [], None
    for i in range(len(t)):
        v = sign * z[i]
        contiguous = cur is not None and t[i] <= t[i - 1] + dur[i - 1] + 1e-6
        if cur is not None and (v < EXIT_Z or not contiguous):
            out.append(cur)
            cur = None
        if cur is None and v >= ENTER_Z:
            cur = [i, i]
        elif cur is not None:
            cur[1] = i
    if cur is not None:
        out.append(cur)
    runs = []
    for a, b in out:
        start, end = float(t[a]), float(t[b] + dur[b])
        if end - start >= MIN_RUN_S - 1e-6:
            k = a + int(np.argmax(sign * z[a:b + 1]))
            runs.append({"start": start, "end": end, "peak_z": float(z[k])})
    return runs


def _ms(s: float) -> int:
    return int(round(s * 1000))


def _overlap(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def detect_moments(
    t: np.ndarray, dur: np.ndarray, z: np.ndarray, channels: list[ProxyDef],
    shots_ms: list[int] | None, speech_ms: list[list[int]] | None,
) -> list[dict]:
    """z [C, T] within-clip curves on native segments -> ranked, time-sorted moments."""
    keys = [c.key for c in channels]
    by_key = {c.key: c for c in channels}
    rises = {k: _runs(t, dur, z[i], +1) for i, k in enumerate(keys)}
    falls = {k: _runs(t, dur, z[i], -1) for i, k in enumerate(keys)}
    shots_s = None if shots_ms is None else [s / 1000 for s in shots_ms]
    speech_s = [(a / 1000, b / 1000) for a, b in speech_ms or []]
    moments: list[dict] = []

    def add(kind, start, end, chans, peak_z, severity, title, description, evidence, hypothesis):
        moments.append({
            "kind": kind, "start_ms": _ms(start), "end_ms": _ms(end), "channels": chans,
            "peak_z": round(peak_z, 2), "severity": round(min(1.0, severity), 2),
            "title": title, "description": description, "evidence": evidence,
            "hypothesis": hypothesis, "test_metric": TEST_METRIC if hypothesis else None,
            "status": "edit_hypothesis_untested" if hypothesis else "observation",
            "relative_to": "this_clip", "confidence": "uncalibrated",
            "in_onset_window": start < ONSET_S,
        })

    # Attention drops: a fall run preceded (within the lookback) by a clearly higher level.
    used_falls = set()
    if "attention" in keys:
        ia = keys.index("attention")
        il = keys.index("language") if "language" in keys else None
        for n, r in enumerate(falls["attention"]):
            look = (t >= r["start"] - DROP_LOOKBACK_S) & (t < r["start"])
            if not look.any():
                continue
            drop = float(z[ia][look].max() - r["peak_z"])
            if drop < DROP_MIN_Z:
                continue
            used_falls.add(n)
            span = r["end"] - r["start"]
            evidence = ["attention_drop"]
            if shots_s is not None and not any(r["start"] - 1.0 <= s <= r["end"] for s in shots_s):
                evidence.append("static_visual")
            if speech_s and sum(_overlap(r["start"], r["end"], a, b) for a, b in speech_s) >= 0.5 * span:
                evidence.append("speech_continues")
            if il is not None:
                inside = (t >= r["start"]) & (t < r["end"])
                if inside.any() and float(z[il][inside].mean()) > 0.5:
                    evidence.append("language_load_high")
            if "static_visual" in evidence:
                hyp = "Test a visual state change here (cut, B-roll, zoom or caption change)."
                desc = "Attention proxy falls while the shot stays unchanged."
            elif "language_load_high" in evidence:
                hyp = "Test tightening the line or bringing the payoff earlier."
                desc = "Attention proxy falls while predicted language load stays high."
            else:
                hyp = "Test trimming or reordering this span."
                desc = "Attention proxy falls relative to the preceding seconds."
            add("attention_drop", r["start"], r["end"], ["attention"], r["peak_z"], drop / 3,
                "Attention proxy drop", f"{desc} Drop of {drop:.1f} z, relative to this clip.", evidence, hyp)

    # Broad response: >= BROAD_MIN_CHANNELS channels rising at the same time.
    edges = sorted({e for runs in rises.values() for r in runs for e in (r["start"], r["end"])})
    span = None
    for a, b in zip(edges, edges[1:]):
        active = [k for k in keys if any(r["start"] <= a and b <= r["end"] for r in rises[k])]
        if len(active) >= BROAD_MIN_CHANNELS:
            span = [a, b, set(active)] if span is None or span[1] < a - 1e-6 else [span[0], b, span[2] | set(active)]
        elif span is not None:
            _add_broad(add, span, z, keys, t)
            span = None
    if span is not None:
        _add_broad(add, span, z, keys, t)

    for k in keys:
        ch = by_key[k]
        for r in rises[k]:
            add("proxy_rise", r["start"], r["end"], [k], r["peak_z"], abs(r["peak_z"]) / 3,
                ch.copy["rise"], f"{ch.label} {r['peak_z']:+.1f} z at its peak, relative to this clip.",
                [f"{k}_rise"], None)
        for n, r in enumerate(falls[k]):
            if k == "attention" and n in used_falls:
                continue
            add("proxy_fall", r["start"], r["end"], [k], r["peak_z"], abs(r["peak_z"]) / 3,
                ch.copy["fall"], f"{ch.label} {r['peak_z']:+.1f} z at its lowest, relative to this clip.",
                [f"{k}_fall"], None)

    moments.sort(key=lambda m: (-m["severity"], m["start_ms"]))
    moments = sorted(moments[:MAX_MOMENTS], key=lambda m: (m["start_ms"], -m["severity"]))
    for i, m in enumerate(moments):
        m["id"] = f"m{i + 1}"
    return moments


def _add_broad(add, span, z, keys, t):
    a, b, active = span
    if b - a < MIN_RUN_S - 1e-6:
        return
    chans = [k for k in keys if k in active]
    inside = (t >= a) & (t < b)
    peak = float(max(z[keys.index(k)][inside].max() for k in chans)) if inside.any() else 0.0
    add("broad_response", a, b, chans, peak, (len(chans) / len(keys)) + peak / 6,
        "Strong predicted response across networks",
        f"{len(chans)} proxy channels rise together ({', '.join(chans)}), relative to this clip.",
        [f"{k}_rise" for k in chans], None)
