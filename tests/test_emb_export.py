"""pod/emb_export.py: pooled fusion-model inputs (<vid>.emb.npz, emb_pool_v1)."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pod"))
import emb_export  # noqa: E402


def seg(start, dur=100.0):
    return SimpleNamespace(start=start, duration=dur)


def test_layer_groups_match_neuralset_group_mean():
    assert emb_export.layer_groups([0.5, 0.75, 1.0], 20) == [[9, 14], [14, 20]]
    assert emb_export.layer_groups(0.5, 20) == [[9, 10]]


def test_pool_uses_only_stimulus_steps_and_quarters():
    freq, G, D, T = 2.0, 2, 3, 200
    x = np.zeros((1, G, D, T), dtype=np.float32)
    n = 20  # a 10 s clip: steps 0..19 are stimulus, the rest is loader padding
    x[0, :, :, :n] = np.arange(n)[None, None, :] + 1
    x[0, :, :, n:] = 999  # padding must not leak in
    out = emb_export.pool([({"video": x}, [(0.0, 100.0)])], 0.0, 10.0, freq)
    v = np.arange(n) + 1.0
    assert out["video_n"] == n and out["video_mean"].dtype == np.float16
    np.testing.assert_allclose(out["video_mean"], np.full((G, D), v.mean()), rtol=1e-3)
    np.testing.assert_allclose(out["video_sd"], np.full((G, D), v.std()), rtol=1e-3)
    assert out["video_bins"].shape == (4, G, D)
    np.testing.assert_allclose(out["video_bins"][:, 0, 0], [v[:5].mean(), v[5:10].mean(), v[10:15].mean(),
                                                            v[15:].mean()], rtol=1e-3)


def test_pool_across_batches_orders_by_time_and_nan_for_empty_quarter():
    a = np.ones((1, 1, 2, 4), dtype=np.float32)
    b = np.full((1, 1, 2, 4), 3.0, dtype=np.float32)
    # batches arrive out of order; stimulus 0..4 s, but only steps at 0,0.5,1,1.5 and 3,3.5 exist
    out = emb_export.pool([({"audio": b}, [(3.0, 2.0)]), ({"audio": a}, [(0.0, 2.0)])], 0.0, 4.0, 2.0)
    assert out["audio_n"] == 6  # b's steps at 4.0 and 4.5 are outside
    bins = out["audio_bins"][:, 0, 0].astype(float)
    assert bins[0] == 1 and bins[1] == 1 and np.isnan(bins[2]) and bins[3] == 3


def test_missing_modality_is_omitted_not_zero():
    z = np.zeros((1, 2, 4, 10), dtype=np.float32)
    t = np.zeros((1, 2, 4, 10), dtype=np.float32)
    t[0, :, :, 3] = 1.0  # one word: zeros elsewhere are real inputs and stay in the mean
    out = emb_export.pool([({"text": t, "audio": z}, [(0.0, 5.0)])], 0.0, 5.0, 2.0)
    assert "audio_mean" not in out and "text_mean" in out
    assert out["text_n"] == 10 and float(out["text_mean"][0, 0]) == pytest.approx(0.1, rel=1e-3)


def test_capture_tees_and_restores():
    class Brain:
        def aggregate_features(self, batch):
            return "orig"

    torch = pytest.importorskip("torch")
    brain = Brain()
    batch = SimpleNamespace(data={"video": torch.ones(1, 2, 3, 4), "subject_id": torch.zeros(1)},
                            segments=[seg(0.0)])
    with emb_export.Capture(brain) as cap:
        assert brain.aggregate_features(batch) == "orig"
    assert "aggregate_features" not in brain.__dict__
    assert list(cap.batches[0][0]) == ["video"] and cap.batches[0][1] == [(0.0, 100.0)]


def test_build_meta_and_window_from_video_event():
    x = np.random.default_rng(0).standard_normal((1, 2, 5, 40)).astype(np.float32)
    cap = emb_export.Capture(SimpleNamespace())
    cap.batches = [({"video": x}, [(0.0, 20.0)])]
    events = pd.DataFrame([{"type": "Video", "start": 0.0, "duration": 12.0},
                           {"type": "Word", "start": 1.0, "duration": 0.2}])
    image = SimpleNamespace(model_name="facebook/vjepa2-vitg-fpc64-256", layers=[0.5, 0.75, 1.0],
                            cache_n_layers=20, layer_aggregation="group_mean")
    cfg = SimpleNamespace(frequency=2.0, features_to_use=["text", "audio", "video"],
                          video_feature=SimpleNamespace(image=image, frequency=2.0))
    arrays, meta = emb_export.build(cap, events, cfg, duration_s=12.0)
    assert arrays["video_n"] == 24 and meta["stimulus_window_s"] == [0.0, 12.0]
    v = meta["extractors"]["video"]
    assert v["model"].endswith("vjepa2-vitg-fpc64-256") and v["layer_groups"] == [[9, 14], [14, 20]]
    assert (v["groups"], v["dim"], v["freq_hz"]) == (2, 5, 2.0)
    assert meta["image_feature_used"] is False and meta["fp16_overflow_keys"] == []
    cap.error = "boom"
    with pytest.raises(RuntimeError):
        emb_export.build(cap, events, cfg, duration_s=12.0)
