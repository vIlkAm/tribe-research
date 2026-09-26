#!/usr/bin/env python3
"""Sanity gate for the pod's <vid>.emb.npz files (emb_pool_v1) in a worker out-root.

    tools/check_emb.py OUT_ROOT [--min-bins-s 20]

Per clip and modality (video/audio/text present in the file):
  n           steps pooled; must be within 2 of 2 x duration_s (2 Hz; checks step timing)
  nan_bins    empty quarters; must be 0 on clips >= --min-bins-s seconds
  q_min_cos   min cosine between the 4 quarter means (flattened G*D); ~1.0 on a static
              clip, lower where the content changes. Reported, not gated: judge it on a
              clip with a visible scene change.
  q_identical all 4 quarters bitwise equal (fails: bins carry no time information)
A clip also fails if meta.emb is missing, has an error, the npz is absent, or it lacks
video or audio (text may be omitted when a clip has no speech).
Exit 0 if at least one clip was checked and all pass, else 1.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

MODALITIES = ("video", "audio", "text")
REQUIRED = ("video", "audio")
FREQ_HZ = 2.0


def check_clip(meta: dict, npz: Path, min_bins_s: float) -> dict:
    row: dict = {"video_id": meta.get("video_id"), "duration_s": meta.get("duration_s")}
    emb = meta.get("emb")
    if not emb or emb.get("error") or not npz.exists():
        row.update(ok=False, problem=(emb or {}).get("error") or "no emb export")
        return row
    dur = float(meta["duration_s"])
    problems = []
    with np.load(npz) as z:
        # text may be omitted on a clip without speech; video and audio are always extracted
        problems += [f"{m} missing" for m in REQUIRED if f"{m}_mean" not in z]
        for m in MODALITIES:
            if f"{m}_mean" not in z:
                continue
            n = int(z[f"{m}_n"])
            bins = z[f"{m}_bins"].astype(np.float64).reshape(z[f"{m}_bins"].shape[0], -1)
            nan_bins = int(np.isnan(bins).any(1).sum())
            full = bins[~np.isnan(bins).any(1)]
            unit = full / np.maximum(np.linalg.norm(full, axis=1, keepdims=True), 1e-12)
            cos = unit @ unit.T
            row[m] = {"shape": list(z[f"{m}_mean"].shape), "n": n, "n_expected": round(FREQ_HZ * dur, 1),
                      "nan_bins": nan_bins,
                      "q_min_cos": round(float(cos.min()), 6) if len(full) > 1 else None,
                      "q_identical": bool(len(full) > 1 and all(np.array_equal(full[0], b) for b in full[1:]))}
            if abs(n - FREQ_HZ * dur) > 2:
                problems.append(f"{m}_n {n} vs {FREQ_HZ * dur:.1f}")
            if dur >= min_bins_s and nan_bins:
                problems.append(f"{m} {nan_bins} empty quarter(s)")
            if row[m]["q_identical"]:
                problems.append(f"{m} quarters identical")
    row.update(ok=not problems, problem="; ".join(problems) or None)
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_root", type=Path)
    ap.add_argument("--min-bins-s", type=float, default=20.0)
    args = ap.parse_args(argv)
    rows = []
    for mp in sorted(args.out_root.glob("worker-*/*.json")):
        if mp.name.endswith(".error.json"):
            continue
        row = check_clip(json.loads(mp.read_text()), mp.with_name(mp.stem + ".emb.npz"), args.min_bins_s)
        rows.append(row)
        print(json.dumps(row))
    n_ok = sum(r["ok"] for r in rows)
    print(f"emb check: {n_ok}/{len(rows)} clips pass")
    return 0 if rows and n_ok == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
