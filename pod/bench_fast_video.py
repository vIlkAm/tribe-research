#!/usr/bin/env python3
"""Pod check for pod/fast_video.py: speed per 0.5 s step, and feature drift per precision.

    python pod/bench_fast_video.py CLIP.mp4 [CLIP2.mp4 ...] [--stock] [--precisions fp32,bf16,fp16]

Runs the V-JEPA2 extractor exactly as TRIBE configures it (no feature cache), once
per precision per clip, and with --stock also the stock neuralset loop on the first
clip. Prints one JSON line per run and a summary: seconds per step, clips/GPU-hour
at the measured rate, peak VRAM, and each precision's features vs fast fp32
(relative L2 and min per-layer cosine). Stock vs fast fp32 should agree to GPU
noise (inputs are bitwise identical; tests/test_fast_video.py proves that on CPU).
wait_s_per_step > ~0.02 means the CPU feeder, not the GPU, sets the pace: then a
second worker per GPU (or more --threads) helps; ~0 means the GPU is saturated.
V-JEPA2 runs alone here: peak VRAM with Llama + w2v-BERT resident comes from the
worker's per-clip peak_vram_gb.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fast_video  # noqa: E402

VJEPA = {"name": "HuggingFaceImage", "model_name": "facebook/vjepa2-vitg-fpc64-256",
         "infra": {"keep_in_ram": False}, "layers": [0.5, 0.75, 1.0], "cache_n_layers": 20,
         "layer_aggregation": "group_mean", "token_aggregation": "mean", "device": "cuda"}


def compare(ref: np.ndarray, x: np.ndarray) -> dict:
    """ref/x: [layers, dim, steps] extractor output."""
    rel = float(np.linalg.norm(x - ref) / max(np.linalg.norm(ref), 1e-12))
    a, b = ref.transpose(0, 2, 1), x.transpose(0, 2, 1)  # layer, step, dim
    cos = (a * b).sum(-1) / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), 1e-12)
    return {"rel_l2": round(rel, 6), "min_cos": round(float(cos.min()), 6), "mean_cos": round(float(cos.mean()), 6)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+", type=Path)
    ap.add_argument("--precisions", default="fp32,tf32,bf16,fp16")
    ap.add_argument("--stock", action="store_true", help="also time the stock loop on the first clip")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--decode", choices=fast_video.DECODES, default="exact")
    args = ap.parse_args()

    import torch
    from neuralset.events import etypes
    from neuralset.extractors.video import HuggingFaceVideo

    ext = HuggingFaceVideo(image=VJEPA, frequency=2.0, clip_duration=4.0, aggregation="sum",
                           event_types="Video", allow_missing=True)
    stock = HuggingFaceVideo.__dict__["_get_data"].fget.method
    precisions = args.precisions.split(",")
    rows, ref = [], {}

    def timed(fn, clip, label):
        ev = etypes.Video(start=0, timeline="t", filepath=str(clip.resolve()))
        before = dict(fast_video.STATS)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        data = next(iter(fn(ext, [ev]))).data
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        steps = data.shape[-1]
        row = {"clip": clip.name, "mode": label, "video_s": round(ev.duration, 2), "steps": steps,
               "wall_s": round(dt, 2), "s_per_step": round(dt / steps, 3),
               "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
        if fast_video.STATS["steps"] > before["steps"]:  # fast path: where the time went
            n = fast_video.STATS["steps"] - before["steps"]
            row["gpu_s_per_step"] = round((fast_video.STATS["gpu_s"] - before["gpu_s"]) / n, 3)
            row["wait_s_per_step"] = round((fast_video.STATS["feeder_wait_s"] - before["feeder_wait_s"]) / n, 3)
            row["frames_processed"] = int(fast_video.STATS["frames_processed"] - before["frames_processed"])
        return row, data

    for i, clip in enumerate(args.clips):
        # warm-up (model load + cudnn autotune) not counted: one throwaway fp32 pass on clip 0
        if i == 0:
            fast_video.install("fp32", threads=args.threads, decode=args.decode)
            timed(fast_video.fast_get_data, clip, "warmup")
        for p in precisions:
            fast_video.install(p, threads=args.threads, decode=args.decode)
            row, data = timed(fast_video.fast_get_data, clip, f"fast-{p}")
            if p == "fp32":
                ref[clip] = data
            elif clip in ref:
                row.update(compare(ref[clip], data))
            rows.append(row)
            print(json.dumps(row), flush=True)
        if args.stock and i == 0:
            fast_video.uninstall()
            row, data = timed(stock, clip, "stock-fp32")
            if clip in ref:
                row.update(compare(ref[clip], data))
            rows.append(row)
            print(json.dumps(row), flush=True)

    print("\nsummary (V-JEPA2 only; other extractors + TRIBE add ~0.4 GPU-s per video-s on the L40S pilot):")
    for mode in dict.fromkeys(r["mode"] for r in rows):
        rs = [r for r in rows if r["mode"] == mode]
        sps = sum(r["wall_s"] for r in rs) / sum(r["steps"] for r in rs)
        vid_per_video_s = 2 * sps  # 2 steps per video second
        drift = [r for r in rs if "rel_l2" in r]
        print(f"  {mode:11s} {sps:.3f} s/step  {vid_per_video_s:.2f} GPU-s per video-s"
              f"  ~{3600 / ((vid_per_video_s + 0.4) * 26):.0f} clips/GPU-h (26 s avg clip)"
              + (f"  worst rel_l2 {max(r['rel_l2'] for r in drift):.4f}"
                 f" min_cos {min(r['min_cos'] for r in drift):.4f}" if drift else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
