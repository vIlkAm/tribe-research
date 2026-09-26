"""Offline tests: sharding balance, manifest build, and worker dry-run/resume."""

import json
import random
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from make_manifest import assign_workers, build, mvhd_duration  # noqa: E402

sys.path.insert(0, str(ROOT / "pod"))
import worker  # noqa: E402


def fake_mp4(path: Path, seconds: float, salt: bytes = b"") -> None:
    """Minimal ftyp + moov/mvhd (v0) file that mvhd_duration can read."""
    timescale = 1000
    mvhd_body = bytes(4) + bytes(8) + struct.pack(">II", timescale, int(seconds * timescale)) + bytes(80)
    mvhd = struct.pack(">I4s", 8 + len(mvhd_body), b"mvhd") + mvhd_body
    moov = struct.pack(">I4s", 8 + len(mvhd), b"moov") + mvhd
    ftyp = struct.pack(">I4s", 16, b"ftyp") + b"isom" + bytes(4)
    free = struct.pack(">I4s", 8 + len(salt), b"free") + salt
    path.write_bytes(ftyp + free + moov)


def test_assign_workers_balances_by_duration():
    rng = random.Random(0)
    durations = [rng.uniform(5, 180) for _ in range(200)]
    assignment = assign_workers(durations, 4)
    loads = [0.0] * 4
    for d, w in zip(durations, assignment):
        loads[w] += d
    # LPT is within max-item of optimal; demand tighter than 1 max clip spread.
    assert max(loads) - min(loads) <= max(durations)
    assert max(loads) / (sum(loads) / 4) < 1.02


def test_assign_workers_one_long_clip():
    assert assign_workers([600, 10, 10, 10], 2) == [0, 1, 1, 1]


def test_mvhd_and_build(tmp_path):
    (tmp_path / "viral").mkdir()
    fake_mp4(tmp_path / "viral" / "a.mp4", 42.5, b"a")
    fake_mp4(tmp_path / "b.mov", 10.0, b"b")
    fake_mp4(tmp_path / "dup.mp4", 42.5, b"a")  # same bytes as a.mp4
    (tmp_path / "notes.txt").write_text("ignore me")
    assert abs(mvhd_duration(tmp_path / "b.mov") - 10.0) < 1e-9

    rows, problems = build(tmp_path, 2)
    # Files are walked in sorted order, so dup.mp4 is kept and viral/a.mp4 flagged.
    assert sorted(r["path"] for r in rows) == ["b.mov", "dup.mp4"]
    assert any("duplicate" in p for p in problems)
    assert {r["worker"] for r in rows} == {0, 1}


def _run(*args, check=True):
    return subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True, check=check)


def test_worker_dry_run_resume_and_merge(tmp_path):
    videos = tmp_path / "videos"
    videos.mkdir()
    for i, secs in enumerate([30, 12, 55, 8, 20]):
        fake_mp4(videos / f"clip{i}.mp4", secs, bytes([i]))
    manifest, out = tmp_path / "manifest.jsonl", tmp_path / "outputs"
    _run(ROOT / "tools/make_manifest.py", "--videos-root", videos, "--workers", 2, "--out", manifest)

    worker = [ROOT / "pod/worker.py", "--manifest", manifest, "--videos-root", videos,
              "--out-root", out, "--num-workers", 2, "--dry-run"]
    # Worker 0 does one video, then resumes; worker 1 runs fully.
    _run(*worker, "--worker-id", 0, "--limit", 1)
    assert len(list((out / "worker-0").glob("*.json"))) == 1
    merged = _run(ROOT / "tools/merge.py", "--manifest", manifest, "--out-root", out, check=False)
    assert merged.returncode == 1  # incomplete

    _run(*worker, "--worker-id", 0)
    second = _run(*worker, "--worker-id", 0)
    assert "0 to process" in second.stderr
    _run(*worker, "--worker-id", 1)

    merged = _run(ROOT / "tools/merge.py", "--manifest", manifest, "--out-root", out)
    index = [json.loads(l) for l in (out / "index.jsonl").read_text().splitlines()]
    assert len(index) == 5
    row = index[0]
    data = np.load(out / row["preds_file"])
    assert data["preds"].dtype == np.float16
    assert data["preds"].shape == (row["n_segments"], 20484)
    assert data["seg_start"].shape == (row["n_segments"],)
    assert json.loads((out / "benchmark.json").read_text())["videos_done"] == 5


def test_worker_rejects_mismatched_worker_count(tmp_path):
    videos = tmp_path / "v"
    videos.mkdir()
    fake_mp4(videos / "a.mp4", 5)
    manifest = tmp_path / "m.jsonl"
    _run(ROOT / "tools/make_manifest.py", "--videos-root", videos, "--workers", 4, "--out", manifest)
    r = _run(ROOT / "pod/worker.py", "--manifest", manifest, "--videos-root", videos,
             "--out-root", tmp_path / "o", "--num-workers", 2, "--dry-run", check=False)
    assert r.returncode != 0 and "rebuild the manifest" in r.stderr


def test_order_segments_sorts_and_rejects_bad_timing():
    preds = np.arange(6, dtype=np.float32).reshape(3, 2)
    p, st, _ = worker.order_segments(preds, np.array([2.0, 0.0, 1.0]), np.ones(3))
    assert st.tolist() == [0.0, 1.0, 2.0] and p[:, 0].tolist() == [2.0, 4.0, 0.0]
    for bad in ([0.0, 0.0, 1.0], [0.0, float("nan"), 1.0]):
        try:
            worker.order_segments(preds, np.array(bad), np.ones(3))
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")


def test_bad_timing_writes_error_json(tmp_path):
    class DupModel(worker.StubModel):
        def predict(self, events):
            preds, starts, durs = super().predict(events)
            return preds, np.zeros_like(starts), durs

    row = {"video_id": "v1", "path": "a.mp4", "source_name": "a", "duration_s": 5.0}
    model = DupModel({"a.mp4": 5.0})
    try:
        worker.process(model, row, Path("a.mp4"), tmp_path, {})
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate starts accepted")
    assert not list(tmp_path.glob("*.npz")) and not list(tmp_path.glob("*.json"))
