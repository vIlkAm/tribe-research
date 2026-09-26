#!/usr/bin/env python3
"""TRIBE v2 batch worker: load the model once, process this worker's shard.

Reads manifest.jsonl (see tools/make_manifest.py) and processes every row whose
``worker`` equals ``--worker-id``. For each video it writes, into
``<out-root>/worker-<id>/``:

    <video_id>.npz   preds  float16 [n_segments, n_vertices]   raw cortical predictions
                     seg_start  float64 [n_segments]           segment start (s), may have gaps
                     seg_duration float64 [n_segments]         = TR
    <video_id>.json  metadata + timings; written LAST, so it is the completion marker

A video with an existing .json is skipped, so a crashed/pre-empted worker can
simply be restarted. Failures are recorded as <video_id>.error.json and retried
on the next run.

Extracted V-JEPA2 / Wav2Vec-BERT / Llama features are cached by TRIBE itself
under ``--cache-folder``; keep it on the shared volume so re-scoring never has
to re-run the expensive video encoder.

``--dry-run`` swaps in a stub model (random preds, no torch/GPU) so the
sharding, resume and output format can be exercised anywhere.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import platform
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

log = logging.getLogger("tribe-worker")

N_VERTICES_FSAVERAGE5 = 20484
DEFAULT_TR = 1.0  # overwritten by model.data.TR on the real model


# ── model adapters ────────────────────────────────────────────────────────


class StubModel:
    """Mimics TribeModel's get_events_dataframe/predict surface for dry runs."""

    tr = DEFAULT_TR

    def __init__(self, durations: dict[str, float]):
        self._durations = durations

    def events(self, video_path: str):
        return {"video_path": video_path}

    @staticmethod
    def modalities(events) -> dict:
        return {"event_counts": {"Video": 1, "Audio": 1}, "has_words": False}

    @staticmethod
    def words(events) -> list[dict]:
        return []

    def predict(self, events):
        duration = self._durations[events["video_path"]]
        n = max(1, math.ceil(duration / self.tr))
        rng = np.random.default_rng(abs(hash(events["video_path"])) % (2**32))
        preds = rng.standard_normal((n, N_VERTICES_FSAVERAGE5), dtype=np.float32)
        starts = np.arange(n, dtype=np.float64) * self.tr
        return preds, starts, np.full(n, self.tr)


class TribeAdapter:
    def __init__(self, checkpoint: str, cache_folder: str):
        from tribev2 import TribeModel  # heavy import; only on the pod

        self.model = TribeModel.from_pretrained(checkpoint, cache_folder=cache_folder)
        self.tr = float(self.model.data.TR)

    def events(self, video_path: str):
        return self.model.get_events_dataframe(video_path=video_path)

    @staticmethod
    def modalities(events) -> dict:
        counts = events["type"].value_counts().to_dict() if "type" in events else {}
        return {"event_counts": {str(k): int(v) for k, v in counts.items()},
                "has_words": bool(counts.get("Word", 0))}

    @staticmethod
    def words(events) -> list[dict]:
        """whisperx words TRIBE already transcribed, in stimulus seconds (timeline speech lane)."""
        if "type" not in events:
            return []
        w = events[events["type"] == "Word"].sort_values("start")
        return [{"start": round(float(r.start), 3), "duration": round(float(r.duration), 3),
                 "text": str(getattr(r, "text", ""))} for r in w.itertuples()]

    def predict(self, events):
        preds, segments = self.model.predict(events=events, verbose=False)
        starts = np.array([_seg_start(s) for s in segments], dtype=np.float64)
        durs = np.array(
            [float(getattr(s, "duration", self.tr)) for s in segments], dtype=np.float64
        )
        return np.asarray(preds), starts, durs


def _seg_start(segment) -> float:
    # neuralset 0.0.2 Segment.copy(offset=t) sets start = parent.start + t, so
    # `start` is absolute seconds on the video timeline.
    v = getattr(segment, "start", None)
    return float(v) if v is not None else float("nan")


def order_segments(preds, starts, durs):
    """Sort rows by start time; refuse NaN or duplicate starts rather than save bad timing."""
    if preds.ndim != 2 or preds.shape[0] != len(starts) or len(starts) != len(durs):
        raise ValueError(f"unexpected preds shape {preds.shape} for {len(starts)} segments")
    if np.isnan(starts).any():
        raise ValueError("segment start times missing (NaN); neuralset Segment API changed?")
    order = np.argsort(starts, kind="stable")
    starts = starts[order]
    if len(starts) > 1 and not (np.diff(starts) > 0).all():
        raise ValueError("duplicate segment start times; refusing to write ambiguous timing")
    return preds[order], starts, durs[order]


# ── helpers ───────────────────────────────────────────────────────────────


def load_shard(manifest: Path, worker_id: int, num_workers: int | None) -> list[dict]:
    rows = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
    if not rows:
        raise SystemExit(f"empty manifest: {manifest}")
    m_workers = rows[0].get("num_workers")
    if num_workers is not None and m_workers is not None and m_workers != num_workers:
        raise SystemExit(
            f"manifest was built for {m_workers} workers but NUM_WORKERS={num_workers}; "
            "rebuild the manifest"
        )
    return [r for r in rows if r["worker"] == worker_id]


def atomic_write_bytes(path: Path, write) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        write(f)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def atomic_write_json(path: Path, obj: dict) -> None:
    atomic_write_bytes(path, lambda f: f.write((json.dumps(obj, indent=2) + "\n").encode()))


def failure_category(exc: BaseException) -> str:
    """Coarse bucket so Phase 1 can count how often each failure mode happens."""
    msg = f"{type(exc).__name__}: {exc}"
    if "whisperx failed" in msg:
        return "transcription_failed"
    if "Language" in msg and "not supported" in msg:
        return "unsupported_language"
    if "OutOfMemory" in msg or "out of memory" in msg:
        return "gpu_oom"
    if "segment" in msg and ("duplicate" in msg or "NaN" in msg):
        return "bad_segment_timing"
    if isinstance(exc, FileNotFoundError):
        return "missing_file"
    return "other"


def git_sha(repo: str | None) -> str | None:
    if not repo:
        return None
    try:
        return subprocess.run(
            ["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return None


def runtime_info(dry_run: bool) -> dict:
    info = {"host": socket.gethostname(), "python": platform.python_version(), "dry_run": dry_run}
    if not dry_run:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    return info


# ── main loop ─────────────────────────────────────────────────────────────


def process(model, row: dict, video_path: Path, out_dir: Path, common: dict) -> dict:
    vid = row["video_id"]
    t0 = time.perf_counter()
    events = model.events(str(video_path))
    t1 = time.perf_counter()
    modalities = model.modalities(events)
    words = model.words(events)
    preds, starts, durs = order_segments(*model.predict(events))
    t2 = time.perf_counter()

    atomic_write_bytes(
        out_dir / f"{vid}.npz",
        lambda f: np.savez_compressed(
            f, preds=preds.astype(np.float16), seg_start=starts, seg_duration=durs
        ),
    )
    meta = {
        **{k: row[k] for k in ("video_id", "path", "source_name", "duration_s")},
        "n_segments": int(preds.shape[0]),
        "n_vertices": int(preds.shape[1]),
        "tr_s": float(model.tr),
        "modalities": modalities,
        "words": words,
        "preds_dtype_saved": "float16",
        "hemodynamic_offset_note": "TRIBE preds are shifted 5 s into the past to cancel hemodynamic lag (seg_start is stimulus time)",
        "timing_s": {
            "events_dataframe": round(t1 - t0, 3),  # audio extract + whisperx + text
            "predict": round(t2 - t1, 3),  # feature extraction (V-JEPA2 etc.) + TRIBE
            "total": round(t2 - t0, 3),
        },
        "realtime_factor": round(row["duration_s"] / max(t2 - t0, 1e-9), 4),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **common,
    }
    atomic_write_json(out_dir / f"{vid}.json", meta)
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description="TRIBE v2 batch worker")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--videos-root", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--worker-id", type=int, default=int(os.environ.get("WORKER_ID", 0)))
    ap.add_argument(
        "--num-workers",
        type=int,
        default=int(os.environ["NUM_WORKERS"]) if "NUM_WORKERS" in os.environ else None,
    )
    ap.add_argument("--cache-folder", default=os.environ.get("TRIBE_CACHE", "./feature-cache"))
    ap.add_argument("--checkpoint", default="facebook/tribev2")
    ap.add_argument("--tribe-repo", default=os.environ.get("TRIBE_REPO"))
    ap.add_argument("--limit", type=int, default=None, help="process at most N videos")
    ap.add_argument("--dry-run", action="store_true", help="stub model; no GPU/torch")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [w{args.worker_id}] %(levelname)s %(message)s",
    )

    shard = load_shard(args.manifest, args.worker_id, args.num_workers)
    out_dir = args.out_root / f"worker-{args.worker_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pending = [r for r in shard if not (out_dir / f"{r['video_id']}.json").exists()]
    todo = pending[: args.limit] if args.limit is not None else pending
    log.info(
        "shard: %d videos (%.1f min); %d already done; %d to process",
        len(shard), sum(r["duration_s"] for r in shard) / 60, len(shard) - len(pending), len(todo),
    )
    if not todo:
        return 0

    t_load = time.perf_counter()
    if args.dry_run:
        model = StubModel({str(args.videos_root / r["path"]): r["duration_s"] for r in shard})
    else:
        model = TribeAdapter(args.checkpoint, args.cache_folder)
    load_s = round(time.perf_counter() - t_load, 3)
    log.info("model loaded in %.1fs (TR=%.3fs)", load_s, model.tr)

    common = {
        "worker_id": args.worker_id,
        "checkpoint": args.checkpoint,
        "tribe_commit": git_sha(args.tribe_repo),
        "model_load_s_this_run": load_s,
        "runtime": runtime_info(args.dry_run),
    }

    ok = failed = 0
    source_s = compute_s = 0.0
    for i, row in enumerate(todo, 1):
        vid = row["video_id"]
        err_path = out_dir / f"{vid}.error.json"
        try:
            meta = process(model, row, args.videos_root / row["path"], out_dir, common)
            err_path.unlink(missing_ok=True)
            ok += 1
            source_s += row["duration_s"]
            compute_s += meta["timing_s"]["total"]
            log.info(
                "[%d/%d] %s %.1fs video in %.1fs (x%.2f realtime)",
                i, len(todo), row["path"], row["duration_s"], meta["timing_s"]["total"],
                meta["realtime_factor"],
            )
        except Exception as exc:
            failed += 1
            atomic_write_json(err_path, {
                "video_id": vid, "path": row["path"], "error": repr(exc),
                "category": failure_category(exc),
                "traceback": traceback.format_exc(),
                "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })
            log.error("[%d/%d] %s FAILED: %r", i, len(todo), row["path"], exc)

    rate = source_s / compute_s if compute_s else 0.0
    log.info(
        "done: %d ok, %d failed; %.1f min source in %.1f min compute (%.3f s source / s compute)",
        ok, failed, source_s / 60, compute_s / 60, rate,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
