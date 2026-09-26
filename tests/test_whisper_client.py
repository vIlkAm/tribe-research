"""Offline tests for the persistent-whisperx transcription path in pod/worker.py.

No GPU, no whisperx, no tribev2: a fake ``tribev2.eventstransforms`` carries a
verbatim copy of upstream's ``_get_transcript_from_audio`` (its `uvx whisperx`
subprocess faked to write a fixture JSON), and a fake server speaks the real
protocol through ``whisper_server.serve``.
"""

import inspect
import json
import logging
import subprocess
import sys
import textwrap
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pod"))
import whisper_server  # noqa: E402
import worker  # noqa: E402

UPSTREAM_FILE = Path("/tmp/tribev2-ref/tribev2/eventstransforms.py")

# Verbatim from tribev2/eventstransforms.py @ af58661 (ExtractWordsFromAudio).
UPSTREAM_SRC = '''
    @staticmethod
    def _get_transcript_from_audio(wav_filename: Path, language: str) -> pd.DataFrame:
        import json
        import os
        import subprocess
        import tempfile

        language_codes = dict(
            english="en", french="fr", spanish="es", dutch="nl", chinese="zh"
        )
        if language not in language_codes:
            raise ValueError(f"Language {language} not supported")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16"

        with tempfile.TemporaryDirectory() as output_dir:
            logger.info("Running whisperx via uvx...")
            cmd = [
                "uvx",
                "whisperx",
                str(wav_filename),
                "--model",
                "large-v3",
                "--language",
                language_codes[language],
                "--device",
                device,
                "--compute_type",
                compute_type,
                "--batch_size",
                "16",
                "--align_model",
                "WAV2VEC2_ASR_LARGE_LV60K_960H" if language == "english" else "",
                "--output_dir",
                output_dir,
                "--output_format",
                "json",
            ]
            cmd = [c for c in cmd if c]  # remove empty args
            env = {k: v for k, v in os.environ.items() if k != "MPLBACKEND"}
            result = subprocess.run(cmd, capture_output=True, text=True, env=env)
            if result.returncode != 0:
                raise RuntimeError(f"whisperx failed:\\n{result.stderr}")

            json_path = Path(output_dir) / f"{wav_filename.stem}.json"
            transcript = json.loads(json_path.read_text())

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
'''

# Shaped like whisperx 3.8 aligned output: several segments, an unaligned numeral
# (no "start"), embedded double quotes, non-ASCII text, extra keys (score/word_segments).
FIXTURE = {
    "segments": [
        {"start": 0.03, "end": 2.1, "text": ' He said "stop" right now.',
         "words": [{"word": "He", "start": 0.03, "end": 0.2, "score": 0.9},
                   {"word": "said", "start": 0.25, "end": 0.5, "score": 0.8},
                   {"word": '"stop"', "start": 0.6, "end": 0.9, "score": 0.7},
                   {"word": "right", "start": 1.0, "end": 1.3, "score": 0.95},
                   {"word": "now.", "start": 1.35, "end": 2.1, "score": 0.9}]},
        {"start": 2.5, "end": 4.0, "text": " It costs 25 dollars, café style.",
         "words": [{"word": "It", "start": 2.5, "end": 2.6, "score": 0.9},
                   {"word": "costs", "start": 2.7, "end": 3.0, "score": 0.9},
                   {"word": "25"},
                   {"word": "dollars,", "start": 3.2, "end": 3.5, "score": 0.8},
                   {"word": "café", "start": 3.55, "end": 3.8, "score": 0.6},
                   {"word": "style.", "start": 3.8, "end": 4.0, "score": 0.7}]},
        {"start": 5.0, "end": 5.5, "text": " Okay.",
         "words": [{"word": "Okay.", "start": 5.0, "end": 5.5, "score": 0.99}]},
    ],
    "word_segments": [],
    "language": "en",
}

FAKE_SERVER = r'''
import json, os, shutil, sys, time
sys.path.insert(0, sys.argv[1])
import whisper_server
fixture = sys.argv[2]
print("uv: some non-protocol chatter on stdout", flush=True)
print(json.dumps({"ready": True, "whisperx": whisper_server.WHISPERX_VERSION, "pid": os.getpid()}), flush=True)

def handle(req):
    name = os.path.basename(req["wav"])
    if name.startswith("bad"):
        raise RuntimeError("Failed to load audio")
    if name.startswith("crash"):
        os._exit(3)
    if name.startswith("hang"):
        time.sleep(60)
    shutil.copy(fixture, req["out"])
    if name.startswith("es"):
        return {"detected_language": "es", "language_probability": 0.93, "whisperx_s": 0.01}
    return {"detected_language": "en", "language_probability": 0.99, "whisperx_s": 0.01}

def reply(obj):
    sys.stdout.write(json.dumps(obj) + "\n"); sys.stdout.flush()

sys.exit(whisper_server.serve(handle, sys.stdin, reply))
'''


@pytest.fixture
def fake_tribe(tmp_path, monkeypatch):
    """Fake tribev2.eventstransforms with the verbatim upstream method; `uvx` faked."""
    fixture = tmp_path / "fixture.json"
    fixture.write_text(json.dumps(FIXTURE, ensure_ascii=False), encoding="utf-8")
    calls = []

    def fake_run(cmd, capture_output, text, env):
        calls.append(cmd)
        if Path(cmd[2]).name.startswith("bad"):
            return subprocess.CompletedProcess(cmd, 1, "", "ffmpeg: invalid data")
        out_dir = Path(cmd[cmd.index("--output_dir") + 1])
        (out_dir / f"{Path(cmd[2]).stem}.json").write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    ns = {"Path": Path, "pd": pd, "logger": logging.getLogger("fake-tribe"),
          "torch": types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))}
    exec("class ExtractWordsFromAudio:\n    language = 'english'\n" + textwrap.dedent(UPSTREAM_SRC).replace(
        "\n", "\n    "), ns)
    mod = types.ModuleType("tribev2.eventstransforms")
    mod.ExtractWordsFromAudio = ns["ExtractWordsFromAudio"]
    pkg = types.ModuleType("tribev2")
    pkg.eventstransforms = mod
    monkeypatch.setitem(sys.modules, "tribev2", pkg)
    monkeypatch.setitem(sys.modules, "tribev2.eventstransforms", mod)
    monkeypatch.setattr(subprocess, "run", fake_run)
    server = tmp_path / "fake_server.py"
    server.write_text(FAKE_SERVER)
    cmd = [sys.executable, str(server), str(ROOT / "pod"), str(fixture)]
    return types.SimpleNamespace(cls=mod.ExtractWordsFromAudio, calls=calls, server_cmd=cmd, tmp=tmp_path,
                                 original=ns["ExtractWordsFromAudio"].__dict__["_get_transcript_from_audio"])


def _wav(tmp_path, name):
    p = tmp_path / "videos" / name
    p.parent.mkdir(exist_ok=True)
    p.write_bytes(b"RIFF")
    return p


def _upstream(fake, wav):
    return fake.original.__func__(wav, "english")


def test_upstream_copy_is_verbatim():
    if not UPSTREAM_FILE.exists():
        pytest.skip("tribev2 reference checkout not present")
    src = UPSTREAM_FILE.read_text()
    assert UPSTREAM_SRC.strip("\n") in src
    # worker's builder loop is a verbatim slice of upstream
    body = inspect.getsource(worker.transcript_to_dataframe)
    loop = textwrap.dedent(body[body.index("    words = []"):body.index("    return transcript")])
    assert textwrap.indent(loop, "        ").strip("\n") in src


def test_flags_match_upstream_cmd(fake_tribe):
    wav = _wav(fake_tribe.tmp, "a.wav")
    _upstream(fake_tribe, wav)
    cmd = fake_tribe.calls[-1]
    i = cmd.index("--output_dir")
    assert cmd[:3] == ["uvx", "whisperx", str(wav)]
    assert cmd[3:i] + cmd[i + 2:] == worker.upstream_whisperx_flags("english", "cpu")
    assert worker.default_server_cmd(device="cuda")[:4] == [
        "uvx", "--from", f"whisperx=={whisper_server.WHISPERX_VERSION}", "python"]


def test_patched_frame_identical_to_upstream(fake_tribe):
    tr = worker.install_transcription(True, fake_tribe.server_cmd, start_timeout=30, request_timeout=30)
    try:
        assert tr is not None and fake_tribe.cls._get_transcript_from_audio is tr
        wav = _wav(fake_tribe.tmp, "clip1.wav")
        n_calls = len(fake_tribe.calls)
        got = fake_tribe.cls()._get_transcript_from_audio(wav, "english")
        assert len(fake_tribe.calls) == n_calls  # stock uvx not used
        pd.testing.assert_frame_equal(got, _upstream(fake_tribe, wav))
        assert list(got.columns) == ["text", "start", "duration", "sequence_id", "sentence"]
        assert len(got) == 11 and "25" not in list(got.text) and '"' not in "".join(got.text)
        # the TSV TRIBE writes from it is byte-identical too
        a, b = fake_tribe.tmp / "a.tsv", fake_tribe.tmp / "b.tsv"
        got.to_csv(a, sep="\t", index=False)
        _upstream(fake_tribe, wav).to_csv(b, sep="\t", index=False)
        assert a.read_bytes() == b.read_bytes()
        rec = tr.pop_record(wav)
        assert rec["backend"] == "server" and rec["detected_language"] == "en" and "server_start_s" in rec
    finally:
        tr.close()


def test_fallback_when_server_cannot_start(fake_tribe, caplog):
    tr = worker.install_transcription(True, [str(fake_tribe.tmp / "no-such-uvx")], start_timeout=5)
    wav = _wav(fake_tribe.tmp, "clip.wav")
    with caplog.at_level(logging.WARNING, logger="tribe-worker"):
        for _ in range(3):
            got = tr(wav, "english")
            pd.testing.assert_frame_equal(got, _upstream(fake_tribe, wav))
            assert tr.pop_record(wav)["backend"] == "stock_cli"
    assert tr.client.disabled and tr.client.starts == 1  # never retried after a failed first start
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1  # warned once


def test_fallback_when_server_reports_not_ready(fake_tribe):
    cmd = [sys.executable, "-c", "import json; print(json.dumps({'ready': False, 'error': 'no CUDA'}))"]
    tr = worker.install_transcription(True, cmd, start_timeout=10)
    wav = _wav(fake_tribe.tmp, "clip.wav")
    pd.testing.assert_frame_equal(tr(wav, "english"), _upstream(fake_tribe, wav))
    assert tr.client.disabled


def test_one_bad_clip_does_not_kill_the_loop(fake_tribe):
    tr = worker.install_transcription(True, fake_tribe.server_cmd, start_timeout=30, request_timeout=5)
    try:
        backends = []
        for name in ("good1.wav", "bad.wav", "good2.wav", "crash.wav", "good3.wav", "hang.wav", "good4.wav"):
            wav = _wav(fake_tribe.tmp, name)
            try:
                got = tr(wav, "english")
                pd.testing.assert_frame_equal(got, _upstream(fake_tribe, wav))
            except RuntimeError as exc:  # the stock path fails on bad.wav exactly like upstream would
                assert name == "bad.wav" and "whisperx failed" in str(exc)
                assert worker.failure_category(exc) == "transcription_failed"
                tr.records.clear()
                backends.append("stock_failed")
                continue
            backends.append(tr.pop_record(wav)["backend"])
        assert backends == ["server", "stock_failed", "server", "stock_cli", "server", "stock_cli", "server"]
        assert tr.client.starts == 3  # restarted after the crash and after the timeout kill
    finally:
        tr.close()
    assert tr.client.proc is None


def test_other_languages_keep_upstream_behaviour(fake_tribe):
    tr = worker.install_transcription(True, fake_tribe.server_cmd)
    with pytest.raises(ValueError, match="not supported"):
        tr(_wav(fake_tribe.tmp, "x.wav"), "klingon")
    assert tr.client.proc is None  # server never started for it


def test_transcript_quality_flags():
    info, w = worker.transcript_quality([], None)
    assert w == ["transcript_empty"] and info["n_words"] == 0
    words = [{"text": t} for t in ("hola", "cómo", "estás", "amigo")]
    info, w = worker.transcript_quality(words, {"detected_language": "es", "language_probability": 0.93})
    assert "transcript_non_english:es" in w and "transcript_non_ascii_words" in w
    assert info["non_ascii_word_ratio"] == 0.5
    _, w = worker.transcript_quality([{"text": "don’t"}, {"text": "stop"}],
                                     {"detected_language": "en", "language_probability": 0.99})
    assert w == []  # curly apostrophe is not a letter
    _, w = worker.transcript_quality([{"text": "hi"}], {"detected_language": "de", "language_probability": 0.3})
    assert w == []  # low-confidence guess


def test_process_records_transcript_source_and_warnings(fake_tribe):
    tr = worker.install_transcription(True, fake_tribe.server_cmd, start_timeout=30, request_timeout=30)

    class FakeTribe(worker.TribeAdapter):
        def __init__(self):
            self.tr = 1.0

        def events(self, video_path):
            wav = Path(video_path).with_suffix(".wav")
            words = fake_tribe.cls()._get_transcript_from_audio(wav, "english")  # what TRIBE's _run does
            words["type"] = "Word"
            return pd.concat([pd.DataFrame([{"type": "Video", "start": 0.0, "duration": 6.0}]), words],
                             ignore_index=True)

        def predict(self, events):
            return np.zeros((6, 4), np.float32), np.arange(6.0), np.ones(6)

    out = fake_tribe.tmp / "out"
    out.mkdir()
    try:
        for name in ("clip.mp4", "es_clip.mp4"):
            video = _wav(fake_tribe.tmp, name)
            row = {"video_id": name[:-4], "path": name, "source_name": name, "duration_s": 6.0}
            meta = worker.process(FakeTribe(), row, video, out, {}, tr)
            assert meta["transcript"]["source"] == "server"
            assert meta["timing_s"]["transcription"] is not None
            assert meta["transcript"]["n_words"] == 11 and len(meta["words"]) == 11
        assert meta["quality_warnings"] == ["transcript_non_english:es"]
        assert sorted(p.name for p in out.glob("*.json")) == ["clip.json", "es_clip.json"]  # merge.py globs these
    finally:
        tr.close()


def test_dry_run_never_touches_whisper(tmp_path, monkeypatch):
    monkeypatch.delenv("UV_CONSTRAINT", raising=False)
    monkeypatch.setattr(worker, "install_transcription", lambda *a, **k: pytest.fail("installed in dry run"))
    row = {"video_id": "v", "path": "a.mp4", "source_name": "a", "duration_s": 3.0, "worker": 0, "num_workers": 1}
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(json.dumps(row) + "\n")
    monkeypatch.setattr(sys, "argv", ["worker.py", "--manifest", str(manifest), "--videos-root", str(tmp_path),
                                      "--out-root", str(tmp_path / "o"), "--dry-run"])
    assert worker.main() == 0
    meta = json.loads((tmp_path / "o/worker-0/v.json").read_text())
    assert meta["transcript"]["source"] == "dry_run" and meta["transcription_config"] is None
    assert "UV_CONSTRAINT" not in __import__("os").environ


def test_constraints_file_pins_whisperx(tmp_path, monkeypatch):
    monkeypatch.delenv("UV_CONSTRAINT", raising=False)
    monkeypatch.setattr(worker.tempfile, "gettempdir", lambda: str(tmp_path))
    path = worker.ensure_whisperx_constraints()
    assert f"whisperx=={whisper_server.WHISPERX_VERSION}" in Path(path).read_text().splitlines()
    assert __import__("os").environ["UV_CONSTRAINT"] == path


def test_client_timeout_kills_server_process_group(tmp_path):
    script = tmp_path / "slow.py"
    script.write_text("import json,sys,time\nprint(json.dumps({'ready': True}), flush=True)\n"
                      "for line in sys.stdin: time.sleep(60)\n")
    client = worker.WhisperServerClient([sys.executable, str(script)], start_timeout=10, request_timeout=1)
    t = time.monotonic()
    with pytest.raises(worker.WhisperServerError, match="timed out"):
        client.transcribe(tmp_path / "a.wav", tmp_path / "a.json")
    assert time.monotonic() - t < 20 and client.proc is None


def test_error_streak_restarts_have_their_own_budget(fake_tribe):
    client = worker.WhisperServerClient(fake_tribe.server_cmd, start_timeout=30, request_timeout=10,
                                        max_restarts=0, max_consecutive_errors=2, max_error_restarts=1)
    bad = _wav(fake_tribe.tmp, "bad.wav")
    try:
        for _ in range(2):  # streak 1 -> restart (budget 1 of 1); crash budget of 0 is untouched
            for _ in range(2):
                with pytest.raises(worker.WhisperServerError):
                    client.transcribe(bad, fake_tribe.tmp / "bad.json")
        assert client.disabled and client.starts == 2 and client.error_restarts == 1
    finally:
        client.close()
