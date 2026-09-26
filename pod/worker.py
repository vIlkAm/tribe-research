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

Transcription: TRIBE shells out to ``uvx whisperx`` once per clip, reloading
large-v3 + the align model each time. Real runs instead route
``ExtractWordsFromAudio._get_transcript_from_audio`` to one long-lived
``pod/whisper_server.py`` per worker (same flags, same whisperx code path, same
JSON), falling back to the stock function whenever the server is unavailable.
``--no-whisper-server`` (or ``TRIBE_WHISPER_SERVER=0``) restores the stock path.
The whisperx version is pinned for both paths via ``UV_CONSTRAINT``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import platform
import queue
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import whisper_server  # noqa: E402  (constants only; no whisperx import)

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
        out = []
        for r in w.itertuples():
            start, dur, text = float(r.start), float(getattr(r, "duration", 0.0)), getattr(r, "text", "")
            if not math.isfinite(start):
                continue
            out.append({"start": round(start, 3), "duration": round(dur if math.isfinite(dur) else 0.0, 3),
                        "text": "" if not isinstance(text, str) else text})
        return out

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


# ── transcription: one whisperx server per worker ─────────────────────────

WHISPER_SERVER_SCRIPT = Path(__file__).resolve().with_name("whisper_server.py")
# Quality-flag thresholds (cheap heuristics; TRIBE's forced English is never changed).
NON_ENGLISH_MIN_PROB = 0.5
NON_ASCII_WORD_RATIO = 0.2


def upstream_whisperx_flags(language: str, device: str) -> list[str]:
    """TRIBE's `uvx whisperx` flags (tribev2/eventstransforms.py @ af58661), minus wav and --output_dir."""
    language_codes = dict(english="en", french="fr", spanish="es", dutch="nl", chinese="zh")
    cmd = [
        "--model", "large-v3",
        "--language", language_codes[language],
        "--device", device,
        "--compute_type", "float16",
        "--batch_size", "16",
        "--align_model", "WAV2VEC2_ASR_LARGE_LV60K_960H" if language == "english" else "",
        "--output_format", "json",
    ]
    return [c for c in cmd if c]  # remove empty args, as upstream does


def transcript_to_dataframe(transcript: dict):
    """whisperx JSON -> TRIBE's word DataFrame. Verbatim copy of the loop in
    ExtractWordsFromAudio._get_transcript_from_audio (tribev2 @ af58661)."""
    import pandas as pd

    words = []
    for i, segment in enumerate(transcript["segments"]):
        sentence = segment["text"]
        sentence = sentence.replace('"', "")
        for word in segment["words"]:
            if "start" not in word:
                continue
            word_dict = {
                "text": word["word"].replace('"', ""),
                "start": word["start"],
                "duration": word["end"] - word["start"],
                "sequence_id": i,
                "sentence": sentence,
            }
            words.append(word_dict)

    transcript = pd.DataFrame(words)
    return transcript


def ensure_whisperx_constraints() -> str | None:
    """Pin whisperx for every `uvx` this process spawns (server *and* TRIBE's stock call)."""
    if os.environ.get("UV_CONSTRAINT"):
        log.info("UV_CONSTRAINT already set (%s); not overriding the whisperx pins", os.environ["UV_CONSTRAINT"])
        return os.environ["UV_CONSTRAINT"]
    path = Path(tempfile.gettempdir()) / f"tribe-whisperx-constraints-{whisper_server.WHISPERX_VERSION}.txt"
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(whisper_server.constraints_text())
    tmp.replace(path)
    os.environ["UV_CONSTRAINT"] = str(path)
    return str(path)


def default_server_cmd(language: str = "english", device: str | None = None) -> list[str]:
    if device is None:
        import torch  # same device rule as upstream

        device = "cuda" if torch.cuda.is_available() else "cpu"
    return ["uvx", "--from", f"whisperx=={whisper_server.WHISPERX_VERSION}", "python",
            str(WHISPER_SERVER_SCRIPT), "--", *upstream_whisperx_flags(language, device)]


class WhisperServerError(RuntimeError):
    pass


class WhisperServerClient:
    """Lazily started, restartable whisper_server.py subprocess.

    stderr is inherited (lands in the worker log); stdout carries the protocol and
    is drained by a thread so timeouts work. The server exits on stdin EOF, and it
    runs in its own session so a kill takes the uvx child (which holds the GPU) too.
    """

    def __init__(self, cmd: list[str], start_timeout: float = 900.0, request_timeout: float = 900.0,
                 max_restarts: int = 3, max_consecutive_errors: int = 3, max_error_restarts: int = 20):
        self.cmd = cmd
        self.start_timeout = start_timeout
        self.request_timeout = request_timeout
        self.max_restarts = max_restarts
        self.max_consecutive_errors = max_consecutive_errors
        self.max_error_restarts = max_error_restarts
        self.error_restarts = 0  # restarts after error-reply streaks; own budget, not crashes'
        self.proc: subprocess.Popen | None = None
        self.ready_info: dict | None = None
        self.starts = 0
        self.ever_ready = False
        self.disabled = False
        self._lines: queue.Queue = queue.Queue()
        self._next_id = 0
        self._consecutive_errors = 0

    # lifecycle
    def _start(self) -> float:
        env = {k: v for k, v in os.environ.items() if k != "MPLBACKEND"}  # as upstream
        self.starts += 1
        t0 = time.perf_counter()
        self._lines = queue.Queue()
        self.proc = subprocess.Popen(
            self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None, env=env,
            text=True, encoding="utf-8", bufsize=1, start_new_session=True,
        )
        threading.Thread(target=self._drain, args=(self.proc, self._lines), daemon=True).start()
        msg = self._read(self.start_timeout, want=lambda m: "ready" in m)
        if not msg.get("ready"):
            self.close()
            raise WhisperServerError(f"whisper server failed to start: {msg.get('error')}")
        self.ready_info = msg
        self.ever_ready = True
        load_s = time.perf_counter() - t0
        log.info("whisper server ready in %.1fs (whisperx %s, pid %s)", load_s, msg.get("whisperx"), msg.get("pid"))
        if msg.get("whisperx") != whisper_server.WHISPERX_VERSION:
            log.warning("whisper server runs whisperx %s, pinned %s", msg.get("whisperx"),
                        whisper_server.WHISPERX_VERSION)
        return load_s

    @staticmethod
    def _drain(proc, lines: queue.Queue) -> None:
        for line in proc.stdout:
            lines.put(line)
        lines.put(None)  # EOF

    def _read(self, timeout: float, want) -> dict:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.close(kill=True)
                raise WhisperServerError(f"whisper server timed out after {timeout:.0f}s")
            try:
                line = self._lines.get(timeout=remaining)
            except queue.Empty:
                continue
            if line is None:
                rc = self.proc.poll() if self.proc else None
                self.close()
                raise WhisperServerError(f"whisper server exited (rc={rc})")
            try:
                msg = json.loads(line)
            except ValueError:
                log.debug("whisper server non-protocol stdout: %s", line.rstrip())
                continue
            if isinstance(msg, dict) and want(msg):
                return msg

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def close(self, kill: bool = False) -> None:
        """EOF on stdin makes the server exit; a hung one (or its uvx child) is SIGKILLed as a group."""
        proc, self.proc = self.proc, None
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()
        except OSError:
            pass
        if not kill:
            try:
                proc.wait(timeout=30)
                return
            except subprocess.TimeoutExpired:
                pass
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            log.warning("whisper server pid %s did not exit after SIGKILL", proc.pid)

    # requests
    def transcribe(self, wav: Path, out: Path) -> dict:
        """Returns the server reply (``ok`` true). Raises WhisperServerError otherwise."""
        if self.disabled:
            raise WhisperServerError("whisper server disabled")
        start_s = None
        if not self.alive():
            if self.starts - self.error_restarts > self.max_restarts:
                self.disabled = True
                raise WhisperServerError(f"whisper server disabled after {self.starts} starts")
            try:
                start_s = self._start()
            except (OSError, WhisperServerError) as exc:
                # Never came up: systemic (env/driver), don't pay the start timeout again.
                if not self.ever_ready or self.starts - self.error_restarts > self.max_restarts:
                    self.disabled = True
                raise WhisperServerError(str(exc)) from exc
        self._next_id += 1
        rid = self._next_id
        try:
            self.proc.stdin.write(json.dumps({"id": rid, "wav": str(wav), "out": str(out)}) + "\n")
            self.proc.stdin.flush()
        except (OSError, ValueError) as exc:
            self.close()
            raise WhisperServerError(f"whisper server pipe broken: {exc!r}") from exc
        msg = self._read(self.request_timeout, want=lambda m: m.get("id") == rid)
        if start_s is not None:
            msg["server_start_s"] = round(start_s, 3)
        if not msg.get("ok"):
            self._consecutive_errors += 1
            if self._consecutive_errors >= self.max_consecutive_errors:
                self._consecutive_errors = 0
                self.close()
                if self.error_restarts >= self.max_error_restarts:
                    self.disabled = True
                    log.warning("whisper server: error streaks persist after %d restarts; disabling it",
                                self.error_restarts)
                else:  # e.g. a poisoned CUDA context; the next clip starts a fresh server
                    self.error_restarts += 1
                    log.warning("whisper server: %d consecutive failures; restarting it",
                                self.max_consecutive_errors)
            raise WhisperServerError(f"whisper server error: {msg.get('error')}")
        self._consecutive_errors = 0
        return msg


class Transcription:
    """Replacement for ExtractWordsFromAudio._get_transcript_from_audio.

    Tries the server; on *any* server-side problem calls the original upstream
    function for that clip, so a clip is never lost to the optimisation. Records
    backend/timing/language per wav for the worker's metadata.
    """

    def __init__(self, original, client: WhisperServerClient | None):
        self.original = original
        self.client = client
        self.records: dict[str, dict] = {}
        self._warned: set[str] = set()

    def _warn_once(self, key: str, msg: str, *a) -> None:
        if key not in self._warned:
            self._warned.add(key)
            log.warning(msg, *a)
        else:
            log.info(msg, *a)

    def __call__(self, wav_filename, language: str):
        import json as _json

        wav_filename = Path(wav_filename)
        t0 = time.perf_counter()
        rec: dict = {"backend": "stock_cli"}
        transcript = None
        if self.client is not None and language == "english" and not self.client.disabled:
            try:
                with tempfile.TemporaryDirectory() as output_dir:
                    json_path = Path(output_dir) / f"{wav_filename.stem}.json"
                    reply = self.client.transcribe(wav_filename, json_path)
                    transcript = _json.loads(json_path.read_text())
                rec = {"backend": "server", **{k: reply[k] for k in (
                    "detected_language", "language_probability", "language_detection_error",
                    "whisperx_s", "server_start_s") if k in reply}}
            except Exception as exc:  # noqa: BLE001  never lose a clip to the server
                transcript = None
                state = "disabled" if self.client.disabled else "failed"
                self._warn_once(state, "whisper server %s for %s (%s); using stock uvx whisperx",
                                state, wav_filename.name, exc)
                rec = {"backend": "stock_cli", "server_error": str(exc)[:500]}
        if transcript is None:
            df = self.original(wav_filename, language)  # may raise exactly like upstream
        else:
            df = transcript_to_dataframe(transcript)
        rec["seconds"] = round(time.perf_counter() - t0, 3)
        self.records[str(wav_filename)] = rec
        return df

    def pop_record(self, wav: Path) -> dict | None:
        rec = self.records.pop(str(wav), None)
        if rec is None:
            real = os.path.realpath(wav)
            for k in list(self.records):
                if os.path.realpath(k) == real:
                    rec = self.records.pop(k)
        self.records.clear()  # never leak a record into the next clip
        return rec

    def close(self) -> None:
        if self.client is not None:
            self.client.close()


def install_transcription(use_server: bool, server_cmd: list[str] | None = None, **client_kw) -> Transcription | None:
    """Monkeypatch TRIBE's per-clip transcription. Returns None (stock path) if patching fails."""
    try:
        from tribev2.eventstransforms import ExtractWordsFromAudio
    except Exception as exc:  # noqa: BLE001
        log.warning("cannot import ExtractWordsFromAudio (%r); stock transcription", exc)
        return None
    original = ExtractWordsFromAudio.__dict__.get("_get_transcript_from_audio") \
        or getattr(ExtractWordsFromAudio, "_get_transcript_from_audio")
    original_fn = original.__func__ if isinstance(original, staticmethod) else original
    client = None
    if use_server:
        client = WhisperServerClient(server_cmd or default_server_cmd(), **client_kw)
    tr = Transcription(original_fn, client)
    ExtractWordsFromAudio._get_transcript_from_audio = staticmethod(tr)
    try:
        ok = ExtractWordsFromAudio._get_transcript_from_audio is tr
        try:  # also through an instance (pydantic model), which is how _run calls it
            ok = ok and ExtractWordsFromAudio()._get_transcript_from_audio is tr
        except Exception as exc:  # noqa: BLE001
            log.info("could not instantiate ExtractWordsFromAudio to check the patch (%r)", exc)
    except Exception:  # noqa: BLE001
        ok = False
    if not ok:
        ExtractWordsFromAudio._get_transcript_from_audio = original
        log.warning("transcription monkeypatch did not take; stock uvx whisperx per clip")
        return None
    log.info("transcription: %s (whisperx pinned %s)",
             "persistent whisper server" if use_server else "stock uvx whisperx per clip",
             whisper_server.WHISPERX_VERSION)
    return tr


def transcript_quality(words: list[dict], rec: dict | None) -> tuple[dict, list[str]]:
    """Cheap flags so analysis can filter clips whose (forced-English) transcript is doubtful."""
    tokens = [w.get("text", "") for w in words if isinstance(w.get("text"), str) and w["text"].strip()]
    non_ascii = sum(1 for t in tokens if any(c.isalpha() and ord(c) > 127 for c in t))
    ratio = round(non_ascii / len(tokens), 4) if tokens else None
    rec = rec or {}
    lang, prob = rec.get("detected_language"), rec.get("language_probability")
    info = {
        "n_words": len(tokens),
        "detected_language": lang,
        "language_probability": prob,
        "non_ascii_word_ratio": ratio,
    }
    warnings = []
    if not tokens:
        warnings.append("transcript_empty")
    if lang and lang != "en" and (prob is None or prob >= NON_ENGLISH_MIN_PROB):
        warnings.append(f"transcript_non_english:{lang}")
    if ratio is not None and ratio > NON_ASCII_WORD_RATIO:
        warnings.append("transcript_non_ascii_words")
    return info, warnings


def compare_transcription(wavs: list[Path], repeats: int = 1) -> int:
    """Pod check: server vs stock `uvx whisperx` on the same wavs (bypasses TRIBE's .tsv cache)."""
    import pandas as pd

    ensure_whisperx_constraints()
    tr = install_transcription(True, start_timeout=1800, request_timeout=900, max_restarts=0)
    if tr is None or tr.client is None:
        print("monkeypatch failed", file=sys.stderr)
        return 2
    all_same = True
    try:
        for wav in wavs:
            t = time.perf_counter()
            server_df = tr(wav, "english")
            rec, server_s = tr.pop_record(wav), time.perf_counter() - t
            if rec["backend"] != "server":
                print(json.dumps({"wav": str(wav), "error": "server path not used", "record": rec}))
                all_same = False
                continue
            for k in range(repeats):
                t = time.perf_counter()
                stock_df = tr.original(wav, "english")
                stock_s = time.perf_counter() - t
                try:  # exact: whisperx rounds times to 3 decimals, so "close" is not "identical"
                    pd.testing.assert_frame_equal(server_df, stock_df, check_exact=True)
                    same, diff = True, None
                except AssertionError as exc:
                    same, diff = False, str(exc)[:2000]
                # the .tsv TRIBE persists (and re-reads on resume) must match byte for byte too
                same_tsv = (server_df.to_csv(sep="\t", index=False) == stock_df.to_csv(sep="\t", index=False))
                same = same and same_tsv
                sw = list(server_df.get("text", [])) if len(server_df) else []
                cw = list(stock_df.get("text", [])) if len(stock_df) else []
                print(json.dumps({
                    "wav": str(wav), "repeat": k, "identical_frame": same, "identical_tsv": same_tsv,
                    "identical_words": sw == cw,
                    "n_words_server": len(sw), "n_words_stock": len(cw),
                    "max_abs_start_diff": (float((server_df["start"] - stock_df["start"]).abs().max())
                                           if len(sw) == len(cw) and sw else None),
                    "server_s": round(server_s, 2), "stock_s": round(stock_s, 2),
                    "detected_language": rec.get("detected_language"), "diff": diff,
                }))
                all_same &= same
    finally:
        tr.close()
    return 0 if all_same else 1


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


def process(model, row: dict, video_path: Path, out_dir: Path, common: dict,
            transcription: Transcription | None = None) -> dict:
    vid = row["video_id"]
    t0 = time.perf_counter()
    events = model.events(str(video_path))
    t1 = time.perf_counter()
    # neuralset's ExtractAudioFromVideo writes <video>.wav next to the video; TRIBE caches <video>.tsv there.
    wav = video_path.with_suffix(".wav")
    rec = transcription.pop_record(wav) if transcription is not None else None
    if rec is None:
        source = ("dry_run" if isinstance(model, StubModel)
                  else "tsv_cache" if wav.with_suffix(".tsv").exists() else "none")
    else:
        source = rec["backend"]
    modalities = model.modalities(events)
    try:  # the speech lane is optional; never lose a clip's (billed) predictions over it
        words = model.words(events)
    except Exception as exc:  # noqa: BLE001
        log.warning("%s: word export failed (%r); continuing without words", row["path"], exc)
        words = []
    transcript, quality_warnings = transcript_quality(words, rec)
    transcript["source"] = source
    if rec and rec.get("server_error"):
        transcript["server_error"] = rec["server_error"]
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
        "transcript": transcript,
        "quality_warnings": quality_warnings,
        "preds_dtype_saved": "float16",
        "hemodynamic_offset_note": "TRIBE preds are shifted 5 s into the past to cancel hemodynamic lag (seg_start is stimulus time)",
        "timing_s": {
            "events_dataframe": round(t1 - t0, 3),  # audio extract + whisperx + text
            "transcription": rec.get("seconds") if rec else None,  # part of events_dataframe
            "transcription_server_start": rec.get("server_start_s") if rec else None,
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
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--videos-root", type=Path)
    ap.add_argument("--out-root", type=Path)
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
    ap.add_argument(
        "--whisper-server", action=argparse.BooleanOptionalAction,
        default=os.environ.get("TRIBE_WHISPER_SERVER", "1") not in ("0", "false", "no"),
        help="one persistent whisperx process per worker instead of `uvx whisperx` per clip "
             "(falls back to the stock call on any server problem; env TRIBE_WHISPER_SERVER)",
    )
    ap.add_argument("--whisper-server-cmd", default=os.environ.get("TRIBE_WHISPER_SERVER_CMD"),
                    help="override the server command (shell-split); default: uvx --from whisperx==<pin> ...")
    ap.add_argument("--whisper-timeout", type=float, default=float(os.environ.get("TRIBE_WHISPER_TIMEOUT_S", 900)),
                    help="seconds per clip before the server is killed and the stock path used")
    ap.add_argument("--compare-transcription", type=Path, nargs="+", metavar="WAV",
                    help="pod check: transcribe these wavs via server and stock uvx, diff, exit")
    ap.add_argument("--compare-repeats", type=int, default=2,
                    help="stock runs per wav in --compare-transcription (>=2 shows the GPU noise floor)")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [w{args.worker_id}] %(levelname)s %(message)s",
    )

    if args.compare_transcription:
        return compare_transcription(args.compare_transcription, repeats=args.compare_repeats)
    missing = [f"--{n.replace('_', '-')}" for n in ("manifest", "videos_root", "out_root") if getattr(args, n) is None]
    if missing:
        ap.error(f"required: {', '.join(missing)}")

    shard = load_shard(args.manifest, args.worker_id, args.num_workers)
    out_dir = args.out_root / f"worker-{args.worker_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    # done in any worker dir counts: batches may be re-sliced for a different GPU count
    done = {p.name[:-5] for p in args.out_root.glob("worker-*/*.json") if not p.name.endswith(".error.json")}
    pending = [r for r in shard if r["video_id"] not in done]
    todo = pending[: args.limit] if args.limit is not None else pending
    log.info(
        "shard: %d videos (%.1f min); %d already done; %d to process",
        len(shard), sum(r["duration_s"] for r in shard) / 60, len(shard) - len(pending), len(todo),
    )
    if not todo:
        return 0

    t_load = time.perf_counter()
    transcription = None
    if args.dry_run:
        model = StubModel({str(args.videos_root / r["path"]): r["duration_s"] for r in shard})
    else:
        model = TribeAdapter(args.checkpoint, args.cache_folder)
        ensure_whisperx_constraints()
        transcription = install_transcription(
            args.whisper_server,
            shlex.split(args.whisper_server_cmd) if args.whisper_server_cmd else None,
            **({"request_timeout": args.whisper_timeout} if args.whisper_server else {}),
        )
    load_s = round(time.perf_counter() - t_load, 3)
    log.info("model loaded in %.1fs (TR=%.3fs)", load_s, model.tr)

    common = {
        "worker_id": args.worker_id,
        "checkpoint": args.checkpoint,
        "tribe_commit": git_sha(args.tribe_repo),
        "model_load_s_this_run": load_s,
        "runtime": runtime_info(args.dry_run),
        "transcription_config": None if args.dry_run else {
            "whisperx_pins": list(whisper_server.WHISPERX_PINS),
            "uv_constraint": os.environ.get("UV_CONSTRAINT"),
            "server": bool(transcription is not None and transcription.client is not None),
            "patched": transcription is not None,
        },
    }

    ok = failed = 0
    source_s = compute_s = 0.0
    try:
        ok, failed, source_s, compute_s = _run_shard(model, todo, args, out_dir, common, transcription)
    finally:
        if transcription is not None:
            transcription.close()
    rate = source_s / compute_s if compute_s else 0.0
    log.info(
        "done: %d ok, %d failed; %.1f min source in %.1f min compute (%.3f s source / s compute)",
        ok, failed, source_s / 60, compute_s / 60, rate,
    )
    return 1 if failed else 0


def _run_shard(model, todo, args, out_dir, common, transcription):
    ok = failed = 0
    source_s = compute_s = 0.0
    for i, row in enumerate(todo, 1):
        vid = row["video_id"]
        err_path = out_dir / f"{vid}.error.json"
        try:
            meta = process(model, row, args.videos_root / row["path"], out_dir, common, transcription)
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
    return ok, failed, source_s, compute_s


if __name__ == "__main__":
    sys.exit(main())
