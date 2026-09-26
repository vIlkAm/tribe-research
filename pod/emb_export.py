"""Per-clip pooled extractor embeddings (`<vid>.emb.npz`, format emb_pool_v1).

Source: exactly what TRIBE's fusion model consumes, i.e. ``batch.data[m]`` as passed to
``FmriEncoder.aggregate_features`` (after neuralset's layer group_mean and token mean,
before the per-modality projector): [B, G, D, T] at the data frequency (2 Hz), where
step j of a batch item is stimulus time ``segment.start + j / freq``.

Only steps inside the clip's stimulus window are pooled (the loader pads the last
100 s segment). For m in video/audio/text:

    {m}_mean  float16 [G, D]     mean over steps
    {m}_sd    float16 [G, D]     population sd over steps
    {m}_bins  float16 [4, G, D]  mean per equal quarter of the stimulus window, NaN if empty
    {m}_n     int32 []           steps used

A modality absent from the batch, or all-zero over the clip (allow_missing fills a
missing extractor with zeros), is omitted rather than written as zeros. Zero steps
inside a present modality (e.g. text between words) are kept: the fusion model sees them.
"""

from __future__ import annotations

import logging
import typing as tp

import numpy as np

log = logging.getLogger("emb-export")

VERSION = "emb_pool_v1"
MODALITIES = ("video", "audio", "text")
N_BINS = 4


class Capture:
    """Tee on a TRIBE FmriEncoder's aggregate_features for the duration of a `with` block."""

    def __init__(self, brain_model, modalities: tp.Iterable[str] = MODALITIES):
        self.brain, self.modalities = brain_model, tuple(modalities)
        self.batches: list[tuple[dict[str, np.ndarray], list[tuple[float, float]]]] = []
        self.error: str | None = None

    def __enter__(self):
        orig = self.brain.aggregate_features

        def tee(batch):
            if self.error is None:
                try:
                    data = {m: batch.data[m].detach().float().cpu().numpy()
                            for m in self.modalities if m in batch.data}
                    segs = [(float(s.start), float(s.duration)) for s in batch.segments]
                    self.batches.append((data, segs))
                except Exception as exc:  # noqa: BLE001  never break the prediction
                    self.error = repr(exc)
            return orig(batch)

        self.brain.aggregate_features = tee  # instance attribute shadows the method
        return self

    def __exit__(self, *exc):
        self.brain.__dict__.pop("aggregate_features", None)
        return False


def pool(batches, stim_start: float, stim_end: float, freq: float) -> dict[str, np.ndarray]:
    """Pooled arrays for the npz (see module doc). `batches` as recorded by Capture."""
    steps: dict[str, list[tuple[float, np.ndarray]]] = {}
    for data, segs in batches:
        for m, x in data.items():
            if x.ndim == 3:  # [B, D, T] single layer group
                x = x[:, None]
            for b, (start, _dur) in enumerate(segs):
                t = start + np.arange(x.shape[-1]) / freq
                for j in np.flatnonzero((t >= stim_start - 1e-6) & (t < stim_end - 1e-6)):
                    steps.setdefault(m, []).append((float(t[j]), x[b, :, :, j]))
    out: dict[str, np.ndarray] = {}
    width = (stim_end - stim_start) / N_BINS
    for m, items in steps.items():
        items.sort(key=lambda it: it[0])
        times = np.array([t for t, _ in items])
        arr = np.stack([a for _, a in items]).astype(np.float64)  # [n, G, D]
        if not np.any(arr):
            continue  # missing modality: zeros from allow_missing
        q = np.clip(((times - stim_start) // width).astype(int), 0, N_BINS - 1)
        bins = np.full((N_BINS,) + arr.shape[1:], np.nan)
        for k in range(N_BINS):
            if np.any(q == k):
                bins[k] = arr[q == k].mean(0)
        out[f"{m}_mean"] = arr.mean(0).astype(np.float16)
        out[f"{m}_sd"] = arr.std(0).astype(np.float16)
        out[f"{m}_bins"] = bins.astype(np.float16)
        out[f"{m}_n"] = np.int32(len(items))
    return out


def layer_groups(layers, n_cached: int) -> list[list[int]]:
    """neuralset group_mean groups (cached-layer indices, end exclusive) for these fractions."""
    layers = layers if isinstance(layers, list) else [layers]
    idx = np.unique([int(f * (n_cached - 1)) for f in layers]).tolist()
    if len(idx) == 1:
        return [[idx[0], idx[0] + 1]]
    idx[-1] += 1
    return [[a, b] for a, b in zip(idx[:-1], idx[1:])]


def extractor_meta(data_cfg, pooled: dict[str, np.ndarray]) -> dict:
    """What each pooled modality is, from TRIBE's data config (tribev2 TribeModel.data)."""
    out = {}
    for m in MODALITIES:
        if f"{m}_mean" not in pooled:
            continue
        ext = getattr(data_cfg, f"{m}_feature", None)
        inner = getattr(ext, "image", ext)  # video: HuggingFaceVideo(image=HuggingFaceImage)
        n_cached = getattr(inner, "cache_n_layers", None)
        layers = getattr(inner, "layers", None)
        g, d = pooled[f"{m}_mean"].shape
        out[m] = {
            "model": getattr(inner, "model_name", None),
            "layers": layers,
            "cache_n_layers": n_cached,
            "layer_aggregation": getattr(inner, "layer_aggregation", None),
            "layer_groups": layer_groups(layers, n_cached) if n_cached and layers not in (None, "all") else None,
            "groups": g,
            "dim": d,
            "freq_hz": float(getattr(ext, "frequency", 0.0) or 0.0),
            "fusion_layout": "fusion model rearranges [G, D] to G*D (layer_aggregation cat)",
        }
    return out


def build(capture: Capture, events, data_cfg, duration_s: float) -> tuple[dict[str, np.ndarray], dict]:
    """(arrays, meta) for one clip; raises if nothing usable was captured."""
    if capture.error:
        raise RuntimeError(f"capture failed: {capture.error}")
    if not capture.batches:
        raise RuntimeError("no batches captured")
    video = events[events["type"] == "Video"] if "type" in events else events.iloc[:0]
    if len(video):
        s0 = float(video["start"].min())
        s1 = float((video["start"] + video["duration"]).max())
    else:
        s0, s1 = 0.0, float(duration_s)
    freq = float(getattr(data_cfg, "frequency", 2.0))
    arrays = pool(capture.batches, s0, s1, freq)
    if not arrays:
        raise RuntimeError("no modality had data inside the stimulus window")
    overflow = [k for k, v in arrays.items() if v.dtype == np.float16 and np.isinf(v).any()]
    features = list(getattr(data_cfg, "features_to_use", []) or [])
    meta = {
        "version": VERSION,
        "extractors": extractor_meta(data_cfg, arrays),
        "features_to_use": features,
        "image_feature_used": "image" in features,  # dinov2 image_feature is configured but unused unless listed
        "stimulus_window_s": [round(s0, 3), round(s1, 3)],
        "bins": N_BINS,
        "fp16_overflow_keys": overflow,
    }
    return arrays, meta
