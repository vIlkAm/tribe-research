"""tools/prep_cpu.py: same transform as pod/downscale.sh, verified, resumable (CPU only, ~5 s)."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import prep_cpu  # noqa: E402

imageio_ffmpeg = pytest.importorskip("imageio_ffmpeg")
FF = imageio_ffmpeg.get_ffmpeg_exe()


def test_x264_args_match_pod_downscale_sh():
    sh = (ROOT / "pod" / "downscale.sh").read_text()
    for flag, val in zip(prep_cpu.X264_ARGS[::2], prep_cpu.X264_ARGS[1::2]):
        assert re.search(rf"{re.escape(flag)} {re.escape(val)}\b", sh), (flag, val)
    assert "min(iw,$short)" in sh and "flags=lanczos" in sh
    assert "min(iw,384)" in prep_cpu.scale_filter(384) and "flags=lanczos" in prep_cpu.scale_filter(384)


def make_clip(path: Path, w: int, h: int, audio: bool = True, fps: int = 60) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [FF, "-nostdin", "-loglevel", "error", "-y", "-f", "lavfi",
           "-i", f"testsrc2=size={w}x{h}:rate={fps}:duration=2"]
    if audio:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-c:a", "aac"]
    cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-shortest", str(path)]
    subprocess.run(cmd, check=True)


def test_batch_downscaled_verified_and_resumable(tmp_path, capsys):
    batch = tmp_path / "batches" / "b00"
    rows = [("acct/tall.mp4", 1080, 1920, True), ("acct/wide.mp4", 1280, 720, True),
            ("acct/small.mp4", 320, 240, False)]
    for rel, w, h, audio in rows:
        make_clip(batch / "videos" / rel, w, h, audio)
    manifest = "".join(json.dumps({"video_id": f"id{i}", "path": rel, "duration_s": 2.0}) + "\n"
                       for i, (rel, *_rest) in enumerate(rows))
    (batch / "manifest.jsonl").write_text(manifest)
    out = tmp_path / "s384"

    assert prep_cpu.main([str(batch), "--out-root", str(out), "--jobs", "2", "--threads", "1",
                          "--max-load", "0"]) == 0
    dst = out / "b00"
    assert (dst / "manifest.jsonl").read_text() == manifest
    got = {rel: prep_cpu.probe(FF, dst / "videos" / rel) for rel, *_ in rows}
    assert (got["acct/tall.mp4"]["w"], got["acct/tall.mp4"]["h"]) == (384, 682)
    assert (got["acct/wide.mp4"]["w"], got["acct/wide.mp4"]["h"]) == (682, 384)
    assert (got["acct/small.mp4"]["w"], got["acct/small.mp4"]["h"]) == (320, 240)  # never upscaled
    assert got["acct/tall.mp4"]["audio"] and not got["acct/small.mp4"]["audio"]
    assert all(g["fps"] == "60" for g in got.values())
    assert not list(dst.rglob("*.part.mp4"))
    rec = json.loads((dst / "prep.json").read_text())
    assert rec["encoded"] == 3 and rec["failed"] == {} and rec["ffmpeg"].startswith("ffmpeg version")

    capsys.readouterr()
    assert prep_cpu.main([str(batch), "--out-root", str(out), "--max-load", "0"]) == 0
    assert "0 to encode, 3 already done" in capsys.readouterr().out


def test_shards_partition_the_batch():
    rels = [f"a/{i}.mp4" for i in range(200)]
    parts = [{r for r in rels if prep_cpu.in_shard(r, (i, 3))} for i in range(3)]
    assert sum(map(len, parts)) == 200 and set().union(*parts) == set(rels)


def test_verify_rejects_bad_outputs():
    src = {"duration": 10.0, "w": 1080, "h": 1920, "fps": "30", "audio": True}
    ok = {"duration": 10.02, "w": 384, "h": 682, "fps": "30", "audio": True}
    assert prep_cpu.verify(src, ok, 384) is None
    assert "audio" in prep_cpu.verify(src, {**ok, "audio": False}, 384)
    assert "fps" in prep_cpu.verify(src, {**ok, "fps": "60"}, 384)
    assert "duration" in prep_cpu.verify(src, {**ok, "duration": 9.0}, 384)
    assert "short side" in prep_cpu.verify(src, {**ok, "w": 360}, 384)
