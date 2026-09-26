"""tools/compare_preds.py: preds agreement gates between two worker out-roots."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import compare_preds  # noqa: E402


def write(root: Path, vid: str, preds: np.ndarray) -> None:
    d = root / "worker-0"
    d.mkdir(parents=True, exist_ok=True)
    n = preds.shape[0]
    np.savez(d / f"{vid}.npz", preds=preds.astype(np.float16), seg_start=np.arange(n, dtype=float),
             seg_duration=np.ones(n))
    (d / f"{vid}.json").write_text(json.dumps({"video_id": vid}))
    np.savez(d / f"{vid}.emb.npz", video_mean=np.zeros((2, 3), np.float16))  # must be ignored


def test_gates(tmp_path):
    rng = np.random.default_rng(0)
    base = rng.standard_normal((30, 200))
    write(tmp_path / "ref", "a", base)
    write(tmp_path / "same", "a", base)
    write(tmp_path / "close", "a", base + 0.02 * rng.standard_normal(base.shape))
    write(tmp_path / "far", "a", base + 0.5 * rng.standard_normal(base.shape))
    write(tmp_path / "short", "a", base[:-1])
    r = str(tmp_path / "ref")
    assert compare_preds.main([r, str(tmp_path / "same"), "--gate", "fp32"]) == 0
    assert compare_preds.main([r, str(tmp_path / "close"), "--gate", "fp32"]) == 1
    assert compare_preds.main([r, str(tmp_path / "close"), "--gate", "bf16"]) == 0
    assert compare_preds.main([r, str(tmp_path / "far"), "--gate", "bf16"]) == 1
    assert compare_preds.main([r, str(tmp_path / "short"), "--gate", "bf16"]) == 1  # lost a segment
    assert compare_preds.main([r, str(tmp_path / "missing"), "--gate", "bf16"]) == 1  # nothing shared
