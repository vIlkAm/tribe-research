"""Brain layer: summary features on known signals, ROI groups file, renderer smoke."""

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tribe_research.brain.features import RoiMap, count_transitions, roi_curves, summarize  # noqa: E402

GROUPS = ROOT / "tribe_research/brain/roi_groups_v0.yaml"


def test_summary_known_ramp():
    t = np.arange(0, 10, 1.0)
    s = summarize("x", t, t.copy(), tr=1.0)  # linear ramp: z rises steadily
    assert s.peak_time_ms == 9000 and s.trough_time_ms == 0
    assert s.slope_0_3s > 0
    assert s.mean_0_3s < 0 < s.peak_value
    assert s.n_transitions == 1  # low -> high once
    assert abs(s.auc_normalized) < 1e-9  # z integrates to 0 on a regular grid
    assert not s.short_clip


def test_summary_uses_time_not_index():
    # Gap: segments 3..5 dropped by TRIBE. Peak must report real time, and the
    # 0-3 s window must only see t < 3.
    t = np.array([0, 1, 2, 6, 7, 8, 9], dtype=float)
    y = np.array([0, 0, 0, 0, 5, 0, 0], dtype=float)
    s = summarize("x", t, y, tr=1.0)
    assert s.peak_time_ms == 7000
    assert s.mean_0_3s < 0


def test_short_clip_flag_and_nan_slope():
    s = summarize("x", np.array([0.0, 4.0]), np.array([1.0, 2.0]), tr=1.0)
    assert s.short_clip
    assert np.isnan(s.slope_0_3s)  # only one point inside 0-3 s


def test_transitions_ignore_jitter():
    z = np.array([0.1, -0.1, 0.2, -0.2, 1.0, 0.3, -1.0, 1.2])
    assert count_transitions(z, threshold=0.5) == 2


def test_roi_curves_mean_over_mask():
    mask = np.zeros((2, 20484), bool)
    mask[0, :10] = True
    mask[1, 100:110] = True
    preds = np.zeros((3, 20484), np.float32)
    preds[:, :10] = 2.0
    preds[1, 100:110] = 4.0
    c = roi_curves(preds, RoiMap(["a", "b"], mask, {}))
    assert c.shape == (2, 3)
    assert np.allclose(c[0], 2.0) and np.allclose(c[1], [0, 4, 0])


def test_roi_groups_file_is_well_formed():
    doc = yaml.safe_load(GROUPS.read_text())
    assert doc["atlas"] == "HCPMMP1" and doc["version"].startswith("roi_groups_")
    seen = {}
    for g, spec in doc["groups"].items():
        assert spec["parcels"], g
        for p in map(str, spec["parcels"]):
            assert p not in seen, f"{p} in both {seen.get(p)} and {g}"
            seen[p] = g
    # The public TRIBE checkpoint is cortical-only.
    banned = ("accumbens", "nacc", "striatum", "amygdala", "hippocamp", "caudate", "putamen")
    text = GROUPS.read_text().lower().split("groups:")[1]
    assert not any(b in text for b in banned)


def test_roi_groups_match_real_atlas_if_built():
    art = ROOT / "tribe_research/assets/roi_map_roi_groups_v0.npz"
    if not art.exists():
        pytest.skip("ROI map not built yet: run tools/build_roi_map.py on the pod")
    roi = RoiMap.load(art)
    assert roi.group_mask.shape[1] == 20484
    assert roi.group_mask.any(axis=1).all()


def test_render_brain_frames_smoke():
    pytest.importorskip("nilearn")
    from tribe_research.brain.render import render_brain_frames

    frames = render_brain_frames(np.random.default_rng(0).standard_normal((2, 20484)), px=(320, 240))
    assert frames.shape == (2, 240, 320, 3) and frames.dtype == np.uint8
    assert frames[0].std() > 0
