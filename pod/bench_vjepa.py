"""GPU floor for TRIBE's video extractor: one V-JEPA2 ViT-G forward, fp16, 64x256x256.

    HF_HUB_OFFLINE=1 python pod/bench_vjepa.py [--steps 10] [--batch 1]

Mirrors neuralset's HuggingFaceVideo load (fp16, hidden states). The worker runs
one such forward per 0.5 s of video, so GPU-s per video-s = 2 x (s/step / batch).
"""

from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoModel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/vjepa2-vitg-fpc64-256")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()

    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.float16).cuda().eval()
    x = torch.randn(args.batch, 64, 3, 256, 256, device="cuda", dtype=torch.float16)
    with torch.inference_mode():
        for _ in range(2):  # warm-up
            model(pixel_values_videos=x, output_hidden_states=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        for _ in range(args.steps):
            model(pixel_values_videos=x, output_hidden_states=True)
        torch.cuda.synchronize()
    s = (time.perf_counter() - t) / args.steps
    print(f"batch {args.batch}: {s:.3f} s/forward, {s / args.batch:.3f} s/step, "
          f"{2 * s / args.batch:.2f} GPU-s per video-s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB "
          f"on {torch.cuda.get_device_name()}")


if __name__ == "__main__":
    main()
