#!/usr/bin/env python3
"""Compare TRIBE preds between two worker out-roots (same clips, different settings).

    tools/compare_preds.py REF_OUT_ROOT TEST_OUT_ROOT [--gate bf16|fp32]

Per clip, over the shared segments (matched by seg_start):
  r_all        Pearson r over every (segment, vertex) value
  r_vertex_med median over vertices of the temporal Pearson r (vertices with any variance)
  r_space_med  median over segments of the spatial Pearson r (across vertices, per TR)
  rel_rms      RMS(test - ref) / SD(ref)

Gates, fixed before the fast-video GPU test (2026-09-26):
  fp32  fast fp32 vs stock fp32 (identical inputs; GPU nondeterminism only):
        every clip r_all >= 0.9999 and rel_rms <= 0.01
  bf16  bf16/fp16/tf32 fast vs fast fp32 (acceptable for a whole-dataset precision switch):
        every clip r_all >= 0.998, r_vertex_med >= 0.995 and rel_rms <= 0.06.
        Calibration: the 384 px pre-scale we already accepted moves preds by r_all
        0.9989-0.9991, r_vertex_med 0.9984-0.9990, rel_rms 0.044-0.052 (L40S pilot,
        2 clips, originals vs s384). Lower precision may not drift more than that.
Exit 0 if every shared clip passes the gate, there is at least one, and no REF clip is
missing from TEST; else 1.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

GATES = {
    "fp32": {"r_all": 0.9999, "r_vertex_med": None, "rel_rms": 0.01},
    "bf16": {"r_all": 0.998, "r_vertex_med": 0.995, "rel_rms": 0.06},
}


def load(root: Path) -> dict[str, Path]:
    out = {}
    for meta in sorted(root.glob("worker-*/*.json")):
        if meta.name.endswith(".error.json"):
            continue
        npz = meta.with_suffix(".npz")
        if npz.exists():
            out.setdefault(meta.stem, npz)
    return out


def compare(ref_npz: Path, test_npz: Path) -> dict:
    with np.load(ref_npz) as a, np.load(test_npz) as b:
        ra, sa = a["preds"].astype(np.float64), np.round(a["seg_start"], 3)
        rb, sb = b["preds"].astype(np.float64), np.round(b["seg_start"], 3)
    ia = {s: i for i, s in enumerate(sa.tolist())}
    ib = {s: i for i, s in enumerate(sb.tolist())}
    common = sorted(set(ia) & set(ib))
    x, y = ra[[ia[c] for c in common]], rb[[ib[c] for c in common]]
    r_all = float(np.corrcoef(x.ravel(), y.ravel())[0, 1])
    xc, yc = x - x.mean(0), y - y.mean(0)
    den = np.sqrt((xc**2).sum(0) * (yc**2).sum(0))
    ok = den > 0
    r_vertex = (xc * yc).sum(0)[ok] / den[ok]
    xs, ys = x - x.mean(1, keepdims=True), y - y.mean(1, keepdims=True)
    den_s = np.sqrt((xs**2).sum(1) * (ys**2).sum(1))
    r_space = (xs * ys).sum(1)[den_s > 0] / den_s[den_s > 0]
    return {
        "segments": int(len(common)),
        "segments_ref": int(len(sa)),
        "segments_test": int(len(sb)),
        "r_all": round(r_all, 6),
        "r_vertex_med": round(float(np.median(r_vertex)), 6) if r_vertex.size else None,
        "r_vertex_p05": round(float(np.percentile(r_vertex, 5)), 6) if r_vertex.size else None,
        "r_space_med": round(float(np.median(r_space)), 6) if r_space.size else None,
        "rel_rms": round(float(np.sqrt(((y - x) ** 2).mean()) / max(x.std(), 1e-12)), 6),
    }


def passes(row: dict, gate: dict) -> bool:
    if row["segments"] != row["segments_ref"] or row["segments"] != row["segments_test"]:
        return False
    if row["r_all"] < gate["r_all"] or row["rel_rms"] > gate["rel_rms"]:
        return False
    return gate["r_vertex_med"] is None or (row["r_vertex_med"] or 0) >= gate["r_vertex_med"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref", type=Path)
    ap.add_argument("test", type=Path)
    ap.add_argument("--gate", choices=sorted(GATES), default="bf16")
    args = ap.parse_args(argv)
    ref, test = load(args.ref), load(args.test)
    shared = sorted(set(ref) & set(test))
    gate = GATES[args.gate]
    rows = []
    for vid in shared:
        row = {"video_id": vid, **compare(ref[vid], test[vid])}
        row["pass"] = passes(row, gate)
        rows.append(row)
        print(json.dumps(row))
    n_pass = sum(r["pass"] for r in rows)
    missing = sorted(set(ref) - set(test))  # a clip the test run lost must not pass by omission
    print(f"{args.gate} gate {gate}: {n_pass}/{len(rows)} clips pass "
          f"(ref {len(ref)}, test {len(test)}, shared {len(shared)})"
          + (f"; FAIL: {len(missing)} ref clips missing from test: {', '.join(missing)}" if missing else ""))
    return 0 if rows and n_pass == len(rows) and not missing else 1


if __name__ == "__main__":
    sys.exit(main())
