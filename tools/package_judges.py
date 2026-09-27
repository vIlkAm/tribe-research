#!/usr/bin/env python3
"""Hackathon judges' package: the built demo (frontend/dist) + H.264 copies of HEVC clips + README + serve.py → zip.

    (cd frontend && VITE_REAL_ANALYSIS_INDEX=/demo-stage1/index.json npm run build)
    .venv/bin/python tools/package_judges.py --out /tmp/viralbrain-judges

HEVC (hvc1/hev1) clips do not play in Chrome on Linux or in Firefox, so those are re-encoded to H.264 for the
package only; ``clips/TRANSCODED.json`` records each original's sha256 (= the analysed file, video_id prefix).
The package is never committed to git (it contains footage).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[1]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dist", type=Path, default=ROOT / "frontend/dist")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    site = args.out / "viralbrain-demo"
    if args.out.exists():
        shutil.rmtree(args.out)
    shutil.copytree(args.dist, site)
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    log = {}
    for f in sorted((site / "clips").glob("*.mp4")):
        b = f.read_bytes()
        if b"hvc1" not in b and b"hev1" not in b:
            continue
        tmp = f.with_suffix(".h264.mp4")
        subprocess.run([ff, "-v", "error", "-y", "-i", str(f), "-map", "0:v:0", "-map", "0:a:0?", "-c:v", "libx264",
                        "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
                        "-movflags", "+faststart", str(tmp)], check=True)
        original = hashlib.sha256(b).hexdigest()
        assert original.startswith(f.stem), f"{f.name}: not the analysed file"
        tmp.replace(f)
        log[f.stem] = {"original_sha256": original, "served_sha256": sha(f), "reason": "HEVC -> H.264 for browser playback"}
    (site / "clips/TRANSCODED.json").write_text(json.dumps(log, indent=1) + "\n")
    shutil.copyfile(ROOT / "judges/serve.py", site / "serve.py")
    shutil.copyfile(ROOT / "judges/README.md", site / "README.md")
    zip_path = shutil.make_archive(str(args.out / "viralbrain-demo"), "zip", args.out, "viralbrain-demo")
    print(json.dumps({"site": str(site), "zip": zip_path, "transcoded": len(log),
                      "zip_mb": round(Path(zip_path).stat().st_size / 1e6, 1)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
