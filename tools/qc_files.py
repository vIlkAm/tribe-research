#!/usr/bin/env python3
"""File QC for planned clips: streams, resolution, audio, and a short decode test.

    .venv/bin/python tools/qc_files.py --manifest results/run_full/manifest.jsonl \
        --staging results/run_full/staging --out results/run_full/file_qc.csv

One row per video_id: has_video, has_audio, width, height, fps, vcodec, acodec,
audio_mean_db (loudness over the decoded window, -91 = silent), decode_ok, error.
Runs one ffmpeg call per file (header + 3 s decode) in parallel, CPU only;
uses imageio-ffmpeg's binary when ffmpeg isn't on PATH.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

FIELDS = ["video_id", "has_video", "has_audio", "width", "height", "fps", "vcodec", "acodec",
          "audio_mean_db", "decode_ok", "error"]


FFMPEG = None


def _ffmpeg() -> str:
    global FFMPEG
    if FFMPEG is None:
        import shutil
        FFMPEG = shutil.which("ffmpeg")
        if not FFMPEG:
            import imageio_ffmpeg
            FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
    return FFMPEG


def qc(video_id: str, path: Path) -> dict:
    """One ffmpeg call: stream info from the header, 3 s decode, loudness if audio exists."""
    row = {k: "" for k in FIELDS}
    row["video_id"] = video_id
    try:
        cmd = [_ffmpeg(), "-hide_banner", "-nostdin", "-ss", "1", "-t", "3", "-i", str(path),
               "-af", "volumedetect", "-f", "null", "-"]
        d = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        err = d.stderr
        v = re.search(r"Stream #0:\d+.*?: Video: (\w+).*?, (\d{2,5})x(\d{2,5})", err)
        a = re.search(r"Stream #0:\d+.*?: Audio: (\w+)", err)
        fps = re.search(r"([\d.]+) fps", err)
        row["has_video"], row["has_audio"] = v is not None, a is not None
        if v:
            row["vcodec"], row["width"], row["height"] = v.group(1), int(v.group(2)), int(v.group(3))
        if fps:
            row["fps"] = float(fps.group(1))
        if a:
            row["acodec"] = a.group(1)
        m = re.search(r"mean_volume: (-?[\d.]+) dB", err)
        row["audio_mean_db"] = float(m.group(1)) if m else ""
        bad = [ln for ln in err.splitlines() if re.search(r"(?i)error|invalid|corrupt|no such file", ln)]
        row["decode_ok"] = d.returncode == 0 and not bad and v is not None
        if bad or d.returncode:
            row["error"] = (bad[0] if bad else f"exit {d.returncode}")[:200]
    except Exception as exc:  # noqa: BLE001 - QC must not stop on one file
        row["decode_ok"], row["error"] = False, repr(exc)[:200]
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--staging", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 8)
    args = ap.parse_args()

    rows = [json.loads(line) for line in args.manifest.open()]
    done = set()
    if args.out.exists():
        done = {r["video_id"] for r in csv.DictReader(args.out.open())}
    todo = []
    for r in rows:
        if r["video_id"] in done:
            continue
        p = args.staging / (f"chunk{r['chunk']}" if "chunk" in r else "") / r["path"]
        todo.append((r["video_id"], p))
    new = not args.out.exists()
    with args.out.open("a", newline="") as f, ThreadPoolExecutor(args.jobs) as ex:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for i, row in enumerate(ex.map(lambda t: qc(*t), todo), 1):
            w.writerow(row)
            if i % 1000 == 0:
                f.flush()
                print(f"{i}/{len(todo)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
