"""tools/check_emb.py: emb.npz sanity gate (step count, empty quarters, time information)."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "pod"))
import check_emb  # noqa: E402
import emb_export  # noqa: E402


def write(root: Path, vid: str, dur: float, x: np.ndarray | None, emb_meta: dict | None = None,
          modalities=("video", "audio")) -> None:
    d = root / "worker-0"
    d.mkdir(parents=True, exist_ok=True)
    meta = {"video_id": vid, "duration_s": dur, "emb": emb_meta or {"version": emb_export.VERSION}}
    (d / f"{vid}.json").write_text(json.dumps(meta))
    if x is not None:  # x: [1, G, D, T] fusion input for one 100 s segment starting at 0
        data = {m: x for m in modalities}
        np.savez(d / f"{vid}.emb.npz", **emb_export.pool([(data, [(0.0, 100.0)])], 0.0, dur, 2.0))


def test_good_and_bad_clips(tmp_path):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 2, 8, 200)).astype(np.float32)
    write(tmp_path / "good", "a", 24.0, x)  # no text: a clip without speech still passes
    assert check_emb.main([str(tmp_path / "good")]) == 0
    write(tmp_path / "silent", "a", 24.0, x, modalities=("video", "text"))
    assert check_emb.main([str(tmp_path / "silent")]) == 1  # audio is required

    static = np.ones((1, 2, 8, 200), np.float32)  # same vector every step: bins carry no time info
    write(tmp_path / "static", "a", 24.0, static)
    assert check_emb.main([str(tmp_path / "static")]) == 1

    write(tmp_path / "err", "a", 24.0, None, {"version": emb_export.VERSION, "error": "boom"})
    assert check_emb.main([str(tmp_path / "err")]) == 1

    wrong_n = tmp_path / "n"
    write(wrong_n, "a", 24.0, x)
    meta = wrong_n / "worker-0" / "a.json"  # claims a longer clip than was pooled
    meta.write_text(json.dumps({**json.loads(meta.read_text()), "duration_s": 30.0}))
    assert check_emb.main([str(wrong_n)]) == 1
    assert check_emb.main([str(tmp_path / "empty")]) == 1
