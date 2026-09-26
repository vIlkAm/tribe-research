"""Build the frontend analysis object (Frontend spec §07, contract ``docs/analysis.schema.json``).

Neural-only for now: ``predictions`` is explicitly ``not_available`` until a
behavior model exists; no placeholder numbers are ever emitted. Raw
20,484-vertex values never go in here (research-mode assets only).
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from tribe_research.brain.events import UNAVAILABLE_LANES, speech_spans
from tribe_research.brain.features import FEATURE_VERSION, RoiMap
from tribe_research.brain.moments import EXPLAIN_VERSION, detect_moments
from tribe_research.brain.proxies import (
    DISPLAY_HZ, ONSET_S, ProxySpec, channel_masks, channel_raw, gaps, to_display_grid,
    zscore_excluding_onset,
)

SCHEMA_VERSION = "nvi.analysis.v0.2"


def _stability(z: np.ndarray, t: np.ndarray) -> float | None:
    """Sustained-attention style summary: 1 / (1 + sd of first differences), post-onset."""
    post = z[t >= ONSET_S]
    if len(post) < 3:
        return None
    return round(float(1.0 / (1.0 + np.diff(post).std())), 3)


def build_analysis(
    *, meta: dict, preds: np.ndarray, seg_start: np.ndarray, seg_duration: np.ndarray,
    roi: RoiMap, spec: ProxySpec, words: list[dict], shots_ms: list[int] | None,
    assets: dict | None = None, synthetic: bool = False,
) -> dict:
    t = seg_start.astype(np.float64)
    dur = seg_duration.astype(np.float64)
    duration_s = float(meta["duration_s"])
    tr = float(meta["tr_s"])

    raw = channel_raw(preds, channel_masks(spec, roi))
    z, onset_excluded = zscore_excluding_onset(t, raw)
    _, grid_values = to_display_grid(t, dur, z, duration_s)
    speech = speech_spans(words)
    moments = detect_moments(t, dur, z, spec.channels, shots_ms, speech)

    channels = []
    for i, ch in enumerate(spec.channels):
        zi = z[i]
        channels.append({
            "key": ch.key, "label": ch.label, "kind": "neural_proxy", "unit": "z_within_clip",
            "direction": ch.direction, "direction_note": ch.direction_note,
            "confidence": "uncalibrated", "source_model": "tribev2", "research_status": "research_proxy",
            "hz": DISPLAY_HZ, "native_tr_s": tr, "resampling": "sample_and_hold",
            "default_visible": ch.default_visible, "color": ch.color,
            "basis": {"roi_groups": ch.roi_groups, "regions_text": ch.regions_text, "atlas": "HCP-MMP1"},
            "copy": dict(ch.copy),
            "values": grid_values[i],
            "summary": {
                "mean_0_3s": _wmean(t, zi, 0, 3), "mean_0_5s": _wmean(t, zi, 0, 5),
                "peak_z": round(float(zi.max()), 3), "peak_time_ms": int(round(t[int(zi.argmax())] * 1000)),
                "trough_z": round(float(zi.min()), 3), "trough_time_ms": int(round(t[int(zi.argmin())] * 1000)),
                "stability": _stability(zi, t),
            },
        })

    versions = {
        "neural": f"tribe-roi-{spec.version}",
        "behavior": None,
        "explain": EXPLAIN_VERSION,
    }
    prov = {
        "tribe_commit": meta.get("tribe_commit"),
        "roi_map": roi.provenance,
        "proxies_version": spec.version,
        "feature_version": FEATURE_VERSION,
        "normalization": "within_clip_z_excluding_onset" if onset_excluded else "within_clip_z",
    }
    key = json.dumps([meta["video_id"], versions, prov["tribe_commit"], prov["roi_map"].get("groups_sha256")],
                     sort_keys=True, default=str)
    warnings = []
    if synthetic:
        warnings.append("SYNTHETIC: dry-run predictions, not TRIBE output.")
    if len(t) < 6:
        warnings.append("Short clip: fewer than 6 predicted samples; moments are unreliable.")
    if not onset_excluded:
        warnings.append("Clip too short to exclude the onset window from the z-score scale.")
    if not words:
        warnings.append("No transcribed words: language-driven signals rely on audio/video only.")
    for code in meta.get("quality_warnings") or []:
        # worker codes (pod/worker.py transcript_quality); transcript_empty is the no-words line above
        if code.startswith("transcript_non_english:"):
            warnings.append(f"Speech looks non-English ({code.split(':', 1)[1]}); it was transcribed as "
                            "English, so word timing and language-driven signals may be wrong.")
        elif code == "transcript_non_ascii_words":
            warnings.append("Many transcribed words have non-English letters; language-driven signals "
                            "may be unreliable.")
        elif code != "transcript_empty":
            warnings.append(f"Quality flag: {code}.")
    roi_synthetic = bool(roi.provenance.get("synthetic") or roi.provenance.get("groups_version") == "SYNTHETIC")
    if roi_synthetic:
        warnings.append("SYNTHETIC ROI map: region assignment is not anatomical.")

    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_id": "a_" + hashlib.sha256(key.encode()).hexdigest()[:12],
        "video_id": meta["video_id"],
        "source_name": meta.get("source_name"),
        "duration_ms": int(round(duration_s * 1000)),
        "status": "complete",
        "research_only": True,
        "synthetic": bool(synthetic or roi_synthetic),
        "model_versions": versions,
        "provenance": prov,
        "timing": {
            "time_base": "stimulus",
            "note": "TRIBE predictions are pre-shifted 5 s for hemodynamic lag; t=0 is the first video frame.",
            "native_tr_s": tr, "display_hz": DISPLAY_HZ, "n_display_samples": len(grid_values[0]) if grid_values else 0,
            "onset_window_ms": [0, int(ONSET_S * 1000)], "gaps_ms": gaps(t, dur, duration_s),
        },
        "predictions": {"status": "not_available", "reason": "Behavior model not trained yet.", "metrics": {}},
        "channels": channels,
        "unavailable_channels": [
            *({"key": k, "kind": "neural_proxy", "reason": v} for k, v in spec.unavailable.items()),
            {"key": "hold", "kind": "behavior", "reason": "Behavior model not trained yet."},
            {"key": "observed_retention", "kind": "observed", "reason": "Outcome import not built yet."},
        ],
        "events": {
            "shots_ms": shots_ms,
            "words": words,
            "speech_spans_ms": speech,
            "unavailable_lanes": [{"key": k, "reason": v} for k, v in UNAVAILABLE_LANES.items()]
            + ([] if shots_ms is not None else [{"key": "shots", "reason": "Source clip not available for scene detection."}]),
        },
        "moments": moments,
        "assets": assets or {},
        "quality": {"n_segments": int(len(t)), "has_words": bool(words), "warnings": warnings},
    }


def _wmean(t, z, lo, hi):
    m = (t >= lo) & (t < hi)
    return round(float(z[m].mean()), 3) if m.any() else None
