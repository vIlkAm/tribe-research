#!/usr/bin/env python3
"""Long-lived whisperx process: load large-v3 + the align model once, transcribe many wavs.

TRIBE's ``ExtractWordsFromAudio._get_transcript_from_audio`` runs a fresh
``uvx whisperx <wav> ...`` per clip, reloading both models every time. This
server runs *whisperx's own CLI code path* (``whisperx.transcribe.transcribe_task``
with the argument dict produced by whisperx's own argparse parser for TRIBE's
exact flags), with only the two model loaders memoised. Defaults, VAD options,
alignment and the JSON writer are therefore the CLI's by construction.

Run it inside whisperx's isolated uv env (never the TRIBE venv: torch conflict)::

    uvx --from whisperx==3.8.6 python pod/whisper_server.py -- \
        --model large-v3 --language en --device cuda --compute_type float16 \
        --batch_size 16 --align_model WAV2VEC2_ASR_LARGE_LV60K_960H --output_format json

Protocol (stdin/stdout, one JSON object per line):

    -> {"ready": true, "whisperx": "3.8.6", ...}      once, after both models loaded
    <- {"id": 1, "wav": "/x/clip.wav", "out": "/tmp/d/clip.json"}
    -> {"id": 1, "ok": true, "detected_language": "en", ...}   or {"id": 1, "ok": false, "error": ...}

``out`` must be ``<dir>/<wav stem>.json`` (what the CLI writes into --output_dir).
One bad clip only produces ``ok: false``; the server exits on stdin EOF.

Also runnable with plain python (no whisperx import) for ``--print-constraints``.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import tempfile
import time
import traceback
import wave
from pathlib import Path

# Single source of truth for the whisperx env, shared by the server launch
# (`uvx --from whisperx==...`), the stock TRIBE path (`uvx whisperx`, pinned via
# UV_CONSTRAINT by pod/worker.py) and pod/setup.sh. These are what uv resolved
# for `whisperx` on Python 3.11 / linux x86_64 on 2026-09-26.
WHISPERX_VERSION = "3.8.6"
WHISPERX_PINS = (
    f"whisperx=={WHISPERX_VERSION}",
    "faster-whisper==1.2.1",
    "ctranslate2==4.8.2",
    "pyannote-audio==4.0.7",
    "torch==2.8.0",
    "torchaudio==2.8.0",
)
PROTOCOL_VERSION = 1
_PLACEHOLDER_WAV = "__tribe_placeholder__.wav"


def constraints_text() -> str:
    return "\n".join(WHISPERX_PINS) + "\n"


# ── request loop (no whisperx dependency; unit-tested offline) ─────────────


def serve(handle, lines, reply) -> int:
    """Answer every request line; a failing request never stops the loop.

    ``handle(req) -> dict`` does the work; ``reply(dict)`` writes one line.
    Catches BaseException (whisperx can call ``parser.error``/``sys.exit``),
    except KeyboardInterrupt.
    """
    for line in lines:
        if not line.strip():
            continue
        req: dict = {}
        try:
            req = json.loads(line)
            if not isinstance(req, dict) or not req.get("wav") or not req.get("out"):
                raise ValueError("request must be a JSON object with 'wav' and 'out'")
            resp = {"ok": True, **(handle(req) or {})}
        except KeyboardInterrupt:
            raise
        except BaseException as exc:  # noqa: BLE001
            resp = {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=8)}
        if isinstance(req, dict) and "id" in req:
            resp["id"] = req["id"]
        reply(resp)
    return 0


# ── whisperx-backed transcriber (only importable inside the uvx env) ────────


def _first_call_cache(fn):
    """Memoise a model loader on its exact arguments; keep at most one model resident."""
    cache: dict[str, object] = {}

    def wrapper(*args, **kwargs):
        key = repr((args, sorted(kwargs.items())))
        if key not in cache:
            cache.clear()
            cache[key] = fn(*args, **kwargs)
        wrapper.last = cache[key]
        return cache[key]

    wrapper.last = None
    return wrapper


class CliTranscriber:
    """Runs whisperx's own ``transcribe_task`` per clip with memoised loaders."""

    def __init__(self, whisperx_flags: list[str], detect_language: bool = True):
        import importlib.metadata

        import whisperx.transcribe as wt
        from whisperx.__main__ import cli

        self.version = importlib.metadata.version("whisperx")
        if "--output_dir" in whisperx_flags or "-o" in whisperx_flags:
            raise ValueError("pass whisperx flags without --output_dir; the server sets it per request")

        # Let whisperx's own parser build the args dict (all version defaults included).
        captured: dict = {}
        real_task = wt.transcribe_task
        real_argv = sys.argv
        wt.transcribe_task = lambda args, parser: captured.update(args=args, parser=parser)
        try:
            sys.argv = ["whisperx", _PLACEHOLDER_WAV, *whisperx_flags, "--output_dir", tempfile.gettempdir()]
            cli()
        finally:
            sys.argv = real_argv
            wt.transcribe_task = real_task
        if "args" not in captured:
            raise RuntimeError("could not capture whisperx CLI arguments (whisperx.__main__.cli changed?)")
        self.args: dict = captured["args"]
        self.parser = captured["parser"]
        self._task = real_task

        # cli() configured whisperx logging at INFO; per-clip chatter is not wanted.
        try:
            from whisperx.log_utils import setup_logging

            setup_logging(level="warning")
        except Exception:  # noqa: BLE001
            pass

        wt.load_model = self._load_model = _first_call_cache(wt.load_model)
        wt.load_align_model = self._load_align = _first_call_cache(wt.load_align_model)
        self.detect_language = detect_language and self.args.get("language") is not None

    def warm_up(self) -> float:
        """Load both models (and nltk punkt_tab) with a short tone clip."""
        t0 = time.perf_counter()
        _ensure_punkt()
        with tempfile.TemporaryDirectory() as d:
            wav = Path(d) / "warmup.wav"
            _write_tone(wav)
            self.transcribe(str(wav), str(Path(d) / "warmup.json"))
        return time.perf_counter() - t0

    def transcribe(self, wav: str, out: str) -> dict:
        wav_p, out_p = Path(wav), Path(out)
        if not wav_p.is_file():
            raise FileNotFoundError(f"wav not found: {wav}")
        out_p.parent.mkdir(parents=True, exist_ok=True)
        args = copy.deepcopy(self.args)
        args["audio"] = [str(wav_p)]
        args["output_dir"] = str(out_p.parent)
        t0 = time.perf_counter()
        self._task(args, self.parser)  # == `whisperx <wav> <flags> --output_dir <dir>`
        whisperx_s = time.perf_counter() - t0
        written = out_p.parent / f"{wav_p.stem}.json"
        if written != out_p:
            os.replace(written, out_p)
        if not out_p.is_file():
            raise RuntimeError(f"whisperx wrote no json for {wav}")
        resp = {"whisperx_s": round(whisperx_s, 3)}
        if self.detect_language:
            resp.update(self._language(str(wav_p)))
        return resp

    def _language(self, wav: str) -> dict:
        """Side channel only: whisper's language guess on the first 30 s (not written to the json).

        Mirrors FasterWhisperPipeline.detect_language without its <30 s warning.
        Transcription itself stays forced to --language.
        """
        try:
            from whisperx.audio import N_SAMPLES, load_audio, log_mel_spectrogram

            pipe = self._load_model.last
            if pipe is None:
                return {}
            audio = load_audio(wav)
            n_mels = pipe.model.feat_kwargs.get("feature_size")
            segment = log_mel_spectrogram(
                audio[:N_SAMPLES],
                n_mels=n_mels if n_mels is not None else 80,
                padding=0 if audio.shape[0] >= N_SAMPLES else N_SAMPLES - audio.shape[0],
            )
            encoder_output = pipe.model.encode(segment)
            token, prob = pipe.model.model.detect_language(encoder_output)[0][0]
            return {"detected_language": token[2:-2], "language_probability": round(float(prob), 4)}
        except Exception as exc:  # noqa: BLE001
            return {"language_detection_error": f"{type(exc).__name__}: {exc}"}


def _ensure_punkt() -> None:
    """whisperx.align loads nltk punkt_tab lazily (and downloads it); do it up front."""
    try:
        import nltk
        from nltk.data import load as nltk_load

        try:
            nltk_load("tokenizers/punkt_tab/english.pickle")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
            nltk_load("tokenizers/punkt_tab/english.pickle")
    except Exception as exc:  # noqa: BLE001  (align retries lazily)
        print(f"whisper_server: punkt_tab not available yet: {exc!r}", file=sys.stderr)


def _write_tone(path: Path, seconds: float = 2.0, rate: int = 16000) -> None:
    import math
    import struct

    n = int(seconds * rate)
    frames = b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / rate))) for i in range(n))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(frames)


# ── entry point ────────────────────────────────────────────────────────────


def _split_argv(argv: list[str]) -> tuple[list[str], list[str]]:
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1:]
    return argv, []


def main(argv: list[str] | None = None) -> int:
    own, whisperx_flags = _split_argv(list(sys.argv[1:] if argv is None else argv))
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--print-constraints", action="store_true", help="print uv constraints and exit (plain python)")
    ap.add_argument("--no-detect-language", action="store_true", help="skip the side-channel language guess")
    ap.add_argument("--noise-to-stderr", action="store_true",
                    help="send whisperx's stdout chatter to stderr instead of /dev/null")
    ap.add_argument("--self-test", metavar="WAV",
                    help="warm up, transcribe WAV once, print the reply to stderr, exit (pod/setup.sh)")
    args = ap.parse_args(own)

    if args.print_constraints:
        sys.stdout.write(constraints_text())
        return 0
    if not whisperx_flags:
        ap.error("whisperx flags required after '--'")

    # Keep a private handle on the real stdout for the protocol; everything else
    # (whisperx logging binds to sys.stdout, verbose prints, C-level writes) goes elsewhere.
    proto = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
    noise = sys.stderr if args.noise_to_stderr else open(os.devnull, "w")
    sys.stdout.flush()
    os.dup2(noise.fileno(), 1)
    sys.stdout = noise

    def reply(obj: dict) -> None:
        proto.write(json.dumps(obj) + "\n")
        proto.flush()

    t0 = time.perf_counter()
    try:
        tr = CliTranscriber(whisperx_flags, detect_language=not args.no_detect_language)
        tr.warm_up()
    except BaseException as exc:  # noqa: BLE001
        reply({"ready": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(limit=8)})
        return 1
    ready = {"ready": True, "protocol": PROTOCOL_VERSION, "whisperx": tr.version, "pid": os.getpid(),
             "load_s": round(time.perf_counter() - t0, 3), "args": {k: v for k, v in tr.args.items()
                                                                       if k not in ("audio", "output_dir", "hf_token")}}

    if args.self_test:
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / f"{Path(args.self_test).stem}.json"
            resp = {"ok": True, **tr.transcribe(args.self_test, str(out))}
            resp["n_segments"] = len(json.loads(out.read_text())["segments"])
        print(json.dumps({**ready, "self_test": resp}, default=str), file=sys.stderr)
        return 0

    reply(json.loads(json.dumps(ready, default=str)))
    return serve(lambda req: tr.transcribe(req["wav"], req["out"]), sys.stdin, reply)


if __name__ == "__main__":
    sys.exit(main())
