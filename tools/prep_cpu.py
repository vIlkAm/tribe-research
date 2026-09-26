#!/usr/bin/env python3
"""CPU prep off the pod: downscale batch clips here, so the billed GPU pod only runs models.

    .venv/bin/python tools/prep_cpu.py results/batches/b00_pilot            # -> results/batches_s384/b00_pilot
    .venv/bin/python tools/prep_cpu.py results/batches/b0* --jobs 3 --threads 4
    tools/pod.sh push-batch results/batches_s384/b00_pilot                  # unchanged push path

Why: TRIBE's V-JEPA2 extractor decodes 64 full-size frames per 0.5 s step on one pod
CPU thread, then its processor shrinks them to a 292 px short side (crop 256). On the
pilot that left the GPU mostly idle (x0.06-0.14 realtime). Scaling once up front
removes most of that per-step CPU work and shrinks the upload.

Same transform as pod/downscale.sh (short side SHORT px, lanczos, never upscales,
libx264 crf 14 veryfast -g 30 yuv420p, audio copied, fps untouched). Relative paths
and manifest.jsonl are kept byte-identical, so video_ids and push-batch still work.

Encoder parity: defaults to imageio-ffmpeg's pinned static binary (ffmpeg 7.0.2 with
imageio-ffmpeg 0.6.0 on Linux) rather than whatever ffmpeg is on PATH, so this server
and a Mac produce the same encode. Use ONE machine per batch, and prefer one machine
for the whole study; <out>/prep.json records the encoder and host.

Splitting across machines (e.g. a Mac helping): --shard 0/2 here and --shard 1/2 on the
Mac; each clip lands in exactly one shard. Copy the Mac's output tree back before pushing.

Production-host safety: runs at nice 19 (+ ionice idle on Linux), --jobs x --threads
caps the cores used, and new clips wait while the 1-min load average is above
--max-load. Outputs are verified (duration, fps, audio, short side) before an atomic
rename; existing outputs are skipped, so re-running resumes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "batches_s384"

# Kept identical to pod/downscale.sh (tests/test_prep_cpu.py checks this).
X264_ARGS = ["-c:v", "libx264", "-crf", "14", "-preset", "veryfast", "-g", "30",
             "-pix_fmt", "yuv420p", "-c:a", "copy"]


def scale_filter(short: int) -> str:
    return (f"scale=w='if(lt(iw,ih),min(iw,{short}),-2)':"
            f"h='if(lt(iw,ih),-2,min(ih,{short}))':flags=lanczos")


def ffmpeg_exe(explicit: str | None) -> str:
    if explicit:
        return explicit
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        exe = shutil.which("ffmpeg")
        if not exe:
            sys.exit("no ffmpeg: pip install imageio-ffmpeg==0.6.0 (preferred, pinned) or pass --ffmpeg")
        print(f"warning: imageio-ffmpeg missing, using {exe}; encoder may differ from other machines",
              file=sys.stderr)
        return exe


def ffmpeg_version(exe: str) -> str:
    out = subprocess.run([exe, "-version"], capture_output=True, text=True).stdout
    return out.splitlines()[0] if out else "?"


_DUR = re.compile(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)")
_VID = re.compile(r"Stream #\S+.*?Video: .*?, (\d{2,5})x(\d{2,5})[, ]")
_FPS = re.compile(r"Video: .*?, ([\d.]+) fps")


def probe(exe: str, path: Path) -> dict:
    """Header info from `ffmpeg -i` (the imageio binary ships without ffprobe)."""
    err = subprocess.run([exe, "-hide_banner", "-nostdin", "-i", str(path)],
                         capture_output=True, text=True, timeout=120).stderr
    info: dict = {"audio": " Audio: " in err}
    if m := _DUR.search(err):
        info["duration"] = int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3])
    if m := _VID.search(err):
        info["w"], info["h"] = int(m[1]), int(m[2])
    if m := _FPS.search(err):
        info["fps"] = m[1]
    return info


def verify(src: dict, out: dict, short: int) -> str | None:
    """None if `out` is a faithful downscale of `src`, else the reason."""
    for k in ("duration", "w", "h", "fps"):
        if k not in out:
            return f"output has no {k}"
    if abs(out["duration"] - src.get("duration", out["duration"])) > 0.15:
        return f"duration {out['duration']:.2f}s vs source {src['duration']:.2f}s"
    if src.get("fps") and out["fps"] != src["fps"]:
        return f"fps {out['fps']} vs source {src['fps']}"
    if src["audio"] and not out["audio"]:
        return "audio stream lost"
    # min() is rotation-invariant; ffmpeg applies display rotation when it re-encodes
    want = min(short, min(src.get("w", short), src.get("h", short)))
    if min(out["w"], out["h"]) != want:
        return f"short side {min(out['w'], out['h'])} != {want}"
    return None


def in_shard(rel: str, shard: tuple[int, int]) -> bool:
    i, n = shard
    return n == 1 or int(hashlib.sha1(rel.encode()).hexdigest(), 16) % n == i


def wait_for_load(max_load: float, lock: threading.Lock) -> None:
    if max_load <= 0 or not hasattr(os, "getloadavg"):
        return
    with lock:  # one waiter polls; the others queue behind it
        warned = False
        while os.getloadavg()[0] > max_load:
            if not warned:
                print(f"load {os.getloadavg()[0]:.1f} > {max_load}; pausing new clips", file=sys.stderr)
                warned = True
            time.sleep(15)


def encode(exe: str, src: Path, dst: Path, short: int, threads: int, ionice: list[str]) -> str | None:
    """Downscale src -> dst atomically. Returns None on success, else the error."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.stem}.{os.getpid()}.{threading.get_ident()}.part.mp4")
    cmd = [*ionice, exe, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
           "-threads", str(threads), "-i", str(src), "-vf", scale_filter(short),
           *X264_ARGS, "-threads", str(threads), str(tmp)]
    try:
        run = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if run.returncode != 0:
            return f"ffmpeg rc={run.returncode}: {run.stderr.strip()[-300:]}"
        reason = verify(probe(exe, src), probe(exe, tmp), short)
        if reason:
            return reason
        os.replace(tmp, dst)
        return None
    except subprocess.TimeoutExpired:
        return "ffmpeg timed out"
    finally:
        tmp.unlink(missing_ok=True)


def prep_batch(batch: Path, out_root: Path, args, exe: str, ionice: list[str], lock) -> dict:
    manifest = batch / "manifest.jsonl"
    if not manifest.is_file():
        sys.exit(f"{batch}: no manifest.jsonl (expected tools/make_batches.py output)")
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    dst_batch = out_root / batch.name
    todo, skipped, missing = [], 0, []
    for r in rows:
        rel = r["path"]
        if not in_shard(rel, args.shard):
            continue
        src, dst = batch / "videos" / rel, dst_batch / "videos" / rel
        if not src.is_file():
            missing.append(rel)
        elif dst.is_file() and dst.stat().st_size > 0:
            skipped += 1
        else:
            todo.append((rel, src, dst))
    print(f"{batch.name}: {len(todo)} to encode, {skipped} already done, {len(missing)} missing"
          + (f" (shard {args.shard[0]}/{args.shard[1]})" if args.shard[1] > 1 else ""))
    if args.dry_run:
        return {"batch": batch.name, "todo": len(todo), "skipped": skipped, "missing": missing}

    failures: dict[str, str] = {}
    t0 = time.time()

    def one(item):
        rel, src, dst = item
        wait_for_load(args.max_load, lock)
        err = encode(exe, src, dst, args.short, args.threads, ionice)
        if err:
            failures[rel] = err
            print(f"  FAILED {rel}: {err}", file=sys.stderr)
        return rel

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for k, _ in enumerate(pool.map(one, todo), 1):
            if k % 25 == 0 or k == len(todo):
                print(f"  {batch.name}: {k}/{len(todo)} ({time.time() - t0:.0f}s)")

    dst_batch.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(manifest, dst_batch / "manifest.jsonl")  # byte-identical: same video_ids
    in_bytes = sum((batch / "videos" / r["path"]).stat().st_size for r in rows
                   if (batch / "videos" / r["path"]).is_file() and in_shard(r["path"], args.shard))
    out_bytes = sum((dst_batch / "videos" / r["path"]).stat().st_size for r in rows
                    if (dst_batch / "videos" / r["path"]).is_file() and in_shard(r["path"], args.shard))
    summary = {
        "batch": batch.name, "encoded": len(todo) - len(failures), "skipped": skipped,
        "failed": failures, "missing": missing, "wall_s": round(time.time() - t0, 1),
        "in_bytes": in_bytes, "out_bytes": out_bytes,
    }
    record = {
        "short": args.short, "x264": X264_ARGS, "filter": scale_filter(args.short),
        "ffmpeg": ffmpeg_version(exe), "host": platform.node(), "machine": platform.machine(),
        "shard": list(args.shard), **summary,
    }
    (dst_batch / f"prep{'' if args.shard[1] == 1 else f'.shard{args.shard[0]}'}.json").write_text(
        json.dumps(record, indent=2) + "\n")
    return summary


def parse_shard(s: str) -> tuple[int, int]:
    i, n = (int(x) for x in s.split("/"))
    if not 0 <= i < n:
        raise argparse.ArgumentTypeError("--shard i/n needs 0 <= i < n")
    return i, n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("batches", nargs="+", type=Path, help="results/batches/<name> dirs")
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--short", type=int, default=384, help="target short side in px (never upscales)")
    ap.add_argument("--jobs", type=int, default=2, help="clips encoded at once")
    ap.add_argument("--threads", type=int, default=4, help="ffmpeg threads per clip")
    ap.add_argument("--max-load", type=float, default=None,
                    help="pause new clips while 1-min load is above this (default: 0.8 x cores; 0 = off)")
    ap.add_argument("--shard", type=parse_shard, default=(0, 1), help="i/n: this machine's share")
    ap.add_argument("--ffmpeg", help="ffmpeg binary (default: imageio-ffmpeg's pinned build)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if args.max_load is None:
        args.max_load = 0.8 * (os.cpu_count() or 4)

    try:
        os.nice(19)  # inherited by every ffmpeg child
    except OSError:
        pass
    ionice = ["ionice", "-c3"] if sys.platform.startswith("linux") and shutil.which("ionice") else []
    exe = ffmpeg_exe(args.ffmpeg)
    print(f"{ffmpeg_version(exe)} | {args.jobs} jobs x {args.threads} threads | "
          f"max load {args.max_load:g} | out {args.out_root}")

    lock = threading.Lock()
    results = [prep_batch(b.resolve(), args.out_root, args, exe, ionice, lock) for b in args.batches]
    if args.dry_run:
        return 0
    failed = sum(len(r["failed"]) + len(r["missing"]) for r in results)
    tin, tout = sum(r["in_bytes"] for r in results), sum(r["out_bytes"] for r in results)
    print(f"done: {sum(r['encoded'] for r in results)} encoded, {sum(r['skipped'] for r in results)} "
          f"skipped, {failed} failed/missing; {tin / 1e9:.2f} GB -> {tout / 1e9:.2f} GB")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
