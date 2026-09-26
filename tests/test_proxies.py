"""Neural proxy layer: channels, display grid, moments, analysis contract, copy rules."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tribe_research.brain.analysis import build_analysis  # noqa: E402
from tribe_research.brain.events import speech_spans, words_lane  # noqa: E402
from tribe_research.brain.features import RoiMap  # noqa: E402
from tribe_research.brain.moments import detect_moments  # noqa: E402
from tribe_research.brain.proxies import (  # noqa: E402
    ProxySpec, channel_masks, gaps, to_display_grid, zscore_excluding_onset,
)

SPEC = ProxySpec.load()
V = 20484
FORBIDDEN = re.compile(r"\bfir(e|es|ed|ing)\b|\bcaus(e|es|ed|ing)\b|reads?\b.*\bmind|\bviral", re.I)


def synth_roi() -> RoiMap:
    groups = sorted({g for c in SPEC.channels for g in c.roi_groups})
    mask = np.zeros((len(groups), V), bool)
    for i in range(len(groups)):
        mask[i, i * 100:(i + 1) * 100] = True
    return RoiMap(groups, mask, {"groups_version": "SYNTHETIC", "synthetic": True})


def preds_with(channel_curves: dict[str, np.ndarray], T: int, seed=0) -> np.ndarray:
    """Vertex preds whose channel means follow the given curves (others ~flat noise)."""
    rng = np.random.default_rng(seed)
    roi = synth_roi()
    p = rng.normal(0, 0.01, (T, V)).astype(np.float32)
    masks = channel_masks(SPEC, roi)
    for c, ch in enumerate(SPEC.channels):
        if ch.key in channel_curves:
            p[:, masks[c]] += channel_curves[ch.key][:, None]
    return p


def test_spec_directions_and_control_never_positive():
    keys = [c.key for c in SPEC.channels]
    assert {"attention", "social", "value", "control"} <= set(keys)
    assert sum(c.default_visible for c in SPEC.channels) <= 4  # spec: 3-4 lines by default
    ctrl = next(c for c in SPEC.channels if c.key == "control")
    assert ctrl.direction == "no_monotonic"
    assert set(SPEC.unavailable) >= {"memorability", "arousal"}


def test_channels_are_disjoint_on_real_group_names():
    groups = yaml.safe_load((ROOT / "tribe_research/brain/roi_groups_v0.yaml").read_text())["groups"]
    used = [g for c in SPEC.channels for g in c.roi_groups]
    assert len(used) == len(set(used)), "a ROI group is in two channels"
    assert set(used) <= set(groups)
    parcels = [p for g in used for p in groups[g]["parcels"]]
    assert len(parcels) == len(set(parcels)), "a parcel is in two channels"


def test_copy_rules_scan_every_string():
    texts = []
    for c in SPEC.channels:
        texts += [c.label, c.regions_text, c.direction_note, *c.copy.values()]
    texts += list(SPEC.unavailable.values())
    from tribe_research.brain import moments, render
    for mod in (moments, render):
        texts += re.findall(r'"([^"\n]{8,})"', Path(mod.__file__).read_text())
    bad = [t for t in texts if FORBIDDEN.search(t)]
    assert not bad, bad


def test_display_grid_keeps_gaps_null():
    t = np.array([0.0, 1.0, 3.0])
    d = np.ones(3)
    grid, rows = to_display_grid(t, d, np.array([[1.0, 2.0, 3.0]]), duration_s=4.0)
    assert list(grid) == [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5]
    assert rows[0] == [1.0, 1.0, 2.0, 2.0, None, None, 3.0, 3.0]
    assert gaps(t, d, 4.0) == [[2000, 3000]]
    assert gaps(t, d, 5.0) == [[2000, 3000], [4000, 5000]]


def test_onset_excluded_from_scale():
    t = np.arange(10.0)
    y = np.array([[50, 50, 0, 1, 0, 1, 0, 1, 0, 1]], float)
    z, excluded = zscore_excluding_onset(t, y)
    assert excluded
    assert z[0, 0] > 50  # onset spike stays visible but does not squash the rest
    assert abs(z[0, 2:].mean()) < 1e-9
    _, ex_short = zscore_excluding_onset(np.arange(4.0), y[:, :4])
    assert not ex_short


def test_moments_hysteresis_min_duration_and_drop():
    t = np.arange(20.0)
    d = np.ones(20)
    att = np.zeros(20)
    att[4:7] = 2.0   # 3 s high
    att[7:10] = -2.0  # then falls -> attention_drop
    att[14] = 3.0    # 1 s spike: below MIN_RUN_S, ignored
    keys = [c.key for c in SPEC.channels]
    z = np.zeros((len(keys), 20))
    z[keys.index("attention")] = att
    m = detect_moments(t, d, z, SPEC.channels, shots_ms=[1000, 15000], speech_ms=[[7000, 10000]])
    kinds = {(x["kind"], x["start_ms"]) for x in m}
    assert ("proxy_rise", 4000) in kinds
    drop = next(x for x in m if x["kind"] == "attention_drop")
    assert drop["start_ms"] == 7000 and drop["end_ms"] == 10000
    assert set(drop["evidence"]) >= {"attention_drop", "static_visual", "speech_continues"}
    assert drop["status"] == "edit_hypothesis_untested" and drop["hypothesis"]
    assert not any(x["start_ms"] == 14000 for x in m)
    assert not any(x["kind"] == "proxy_fall" and x["channels"] == ["attention"] for x in m)
    assert all(x["relative_to"] == "this_clip" and x["confidence"] == "uncalibrated" for x in m)
    rise = next(x for x in m if x["kind"] == "proxy_rise")
    assert rise["hypothesis"] is None and rise["status"] == "observation"


def test_shot_inside_span_removes_static_visual():
    t = np.arange(12.0)
    keys = [c.key for c in SPEC.channels]
    z = np.zeros((len(keys), 12))
    z[keys.index("attention"), 3:5] = 2.0
    z[keys.index("attention"), 5:8] = -2.0
    m = detect_moments(t, np.ones(12), z, SPEC.channels, shots_ms=[6000], speech_ms=[])
    drop = next(x for x in m if x["kind"] == "attention_drop")
    assert "static_visual" not in drop["evidence"]
    m2 = detect_moments(t, np.ones(12), z, SPEC.channels, shots_ms=None, speech_ms=[])
    assert "static_visual" not in next(x for x in m2 if x["kind"] == "attention_drop")["evidence"]


def test_broad_response():
    t = np.arange(12.0)
    keys = [c.key for c in SPEC.channels]
    z = np.zeros((len(keys), 12))
    for k in ("attention", "social", "value"):
        z[keys.index(k), 5:8] = 2.0
    m = detect_moments(t, np.ones(12), z, SPEC.channels, None, None)
    b = [x for x in m if x["kind"] == "broad_response"]
    assert len(b) == 1 and set(b[0]["channels"]) == {"attention", "social", "value"}
    assert b[0]["start_ms"] == 5000 and b[0]["end_ms"] == 8000


def test_words_and_speech_spans():
    w = words_lane([{"start": 1.2, "duration": 0.3, "text": "hi"}, {"start": 0.1, "duration": 0.4, "text": "oh"},
                    {"start": 3.0, "duration": 0.5, "text": "later"}])
    assert [x["text"] for x in w] == ["oh", "hi", "later"]
    assert speech_spans(w) == [[100, 500], [1200, 1500], [3000, 3500]]
    assert speech_spans(w, gap_s=1.0) == [[100, 1500], [3000, 3500]]


def test_analysis_contract_validates_against_schema():
    jsonschema = pytest.importorskip("jsonschema")
    T = 16
    curve = np.sin(np.linspace(0, 3 * np.pi, T)).astype(np.float32)
    preds = preds_with({"attention": curve, "social": -curve}, T)
    t = np.arange(T, dtype=float)
    t = np.delete(t, 9)  # a TRIBE-dropped empty segment -> gap
    preds = np.delete(preds, 9, axis=0)
    meta = {"video_id": "abc", "source_name": "vp-1", "duration_s": 16.0, "tr_s": 1.0, "tribe_commit": "af58661"}
    a = build_analysis(meta=meta, preds=preds, seg_start=t, seg_duration=np.ones(len(t)), roi=synth_roi(),
                       spec=SPEC, words=words_lane([{"start": 2, "duration": 1, "text": "x"}]), shots_ms=[4000],
                       synthetic=True)
    schema = json.loads((ROOT / "docs/analysis.schema.json").read_text())
    jsonschema.validate(a, schema)
    assert a["predictions"] == {"status": "not_available", "reason": "Behavior model not trained yet.", "metrics": {}}
    assert a["synthetic"] is True and a["research_only"] is True
    assert a["timing"]["gaps_ms"] == [[9000, 10000]]
    att = next(c for c in a["channels"] if c["key"] == "attention")
    assert len(att["values"]) == 32 and att["values"][18] is None and att["values"][17] is not None
    assert "values" not in json.dumps(a["assets"])  # no raw vertex data in the analysis object
    assert len(json.dumps(a)) < 200_000
    s = json.dumps(a)
    assert not FORBIDDEN.search(" ".join(m["title"] + " " + m["description"] + " " + (m["hypothesis"] or "")
                                         for m in a["moments"])), s


def test_sprite_sheet_geometry():
    from tribe_research.brain.render import sprite_sheet

    frames = np.arange(13 * 2 * 3 * 3, dtype=np.uint8).reshape(13, 2, 3, 3)
    sheet, cols, rows = sprite_sheet(frames, max_cols=5)
    assert (cols, rows) == (5, 3) and sheet.shape == (6, 15, 3)
    assert (sheet[2:4, 3:6] == frames[6]).all()  # tile 6 -> row 1, col 1
    assert (sheet[4:6, 9:] == 0).all()  # unused cells stay black


def test_paint_channels():
    from tribe_research.brain.render import paint_channels

    masks = np.zeros((2, 6), bool)
    masks[0, :2] = masks[1, 3:5] = True
    out = paint_channels(np.array([[1.0, 2.0], [-1.0, 0.5]]), masks)
    assert out.tolist() == [[1, 1, 0, -1, -1, 0], [2, 2, 0, 0.5, 0.5, 0]]


def test_bundle_end_to_end(tmp_path):
    jsonschema = pytest.importorskip("jsonschema")
    import imageio.v2 as iio

    from tribe_research.brain import bundle

    T = 7
    preds = preds_with({"attention": np.linspace(-1, 1, T), "value": np.linspace(1, -1, T)}, T)
    t = np.array([0, 1, 2, 3, 5, 6, 7], float)  # gap at 4 s
    npz = tmp_path / "v.npz"
    np.savez(npz, preds=preds.astype(np.float16), seg_start=t, seg_duration=np.ones(T))
    meta = {"video_id": "v1", "source_name": "vp-9", "duration_s": 8.0, "tr_s": 1.0, "tribe_commit": "x",
            "words": [{"start": 0.5, "duration": 0.4, "text": "hey"}]}
    roi, ana = synth_roi(), tmp_path / "analyses"
    region = bundle.write_static(ana, roi, SPEC)
    a = bundle.write_bundle(ana, meta, npz, roi, SPEC, region, synthetic=True, research_vertex=True)
    jsonschema.validate(a, json.loads((ROOT / "docs/analysis.schema.json").read_text()))
    assert json.loads((ana / "v1/analysis.json").read_text()) == a
    assert a["events"]["words"] == [{"start_ms": 500, "end_ms": 900, "text": "hey"}]
    assert a["events"]["shots_ms"] is None and a["timing"]["gaps_ms"] == [[4000, 5000]]
    for key in ("brain_map", "research_vertex_map"):
        s = a["assets"][key]
        img = iio.imread(ana / "v1" / s["src"])
        assert img.shape[:2] == (s["rows"] * s["tile_h"], s["cols"] * s["tile_w"])
        assert len(s["frame_start_ms"]) == len(s["frame_end_ms"]) == T
        assert s["frame_start_ms"][4] == 5000 and s["frame_end_ms"][3] == 4000
    assert a["assets"]["brain_map"]["mode"] == "proxy"
    idmap = iio.imread(ana / "v1" / region["idmap_src"])[..., :3]
    assert idmap.shape[:2] == (a["assets"]["brain_map"]["tile_h"], a["assets"]["brain_map"]["tile_w"])
    seen = {"#%02x%02x%02x" % tuple(c) for c in np.unique(idmap.reshape(-1, 3), axis=0)}
    assert seen <= set(region["ids"]) | {"#000000"}
    assert set(region["ids"].values()) == {c.key for c in SPEC.channels}
    assert (ana / "v1" / region["legend_src"]).exists()
