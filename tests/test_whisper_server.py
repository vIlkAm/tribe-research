"""Offline tests for pod/whisper_server.py (no whisperx: a fake package stands in)."""

import argparse
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pod"))
import whisper_server  # noqa: E402


def _run_serve(lines, handle):
    replies = []
    assert whisper_server.serve(handle, iter(lines), replies.append) == 0
    return replies


def test_serve_survives_bad_requests():
    def handle(req):
        if req["wav"] == "boom.wav":
            raise RuntimeError("cuda error")
        if req["wav"] == "exit.wav":
            raise SystemExit(2)  # whisperx's parser.error / sys.exit
        return {"whisperx_s": 1.0}

    lines = [
        json.dumps({"id": 1, "wav": "a.wav", "out": "/t/a.json"}),
        json.dumps({"id": 2, "wav": "boom.wav", "out": "/t/boom.json"}),
        "not json",
        "",
        json.dumps({"id": 3, "wav": "exit.wav", "out": "/t/exit.json"}),
        json.dumps({"id": 4, "wav": "b.wav"}),  # no out
        json.dumps([1, 2]),
        json.dumps({"id": 5, "wav": "c.wav", "out": "/t/c.json"}),
    ]
    r = _run_serve(lines, handle)
    assert [x["ok"] for x in r] == [True, False, False, False, False, False, True]
    assert [x.get("id") for x in r] == [1, 2, None, 3, 4, None, 5]
    assert "cuda error" in r[1]["error"] and "SystemExit" in r[3]["error"]
    assert r[0]["whisperx_s"] == 1.0


def test_print_constraints_needs_no_whisperx():
    out = subprocess.run([sys.executable, str(ROOT / "pod/whisper_server.py"), "--print-constraints"],
                         capture_output=True, text=True, check=True).stdout
    assert out.splitlines()[0] == f"whisperx=={whisper_server.WHISPERX_VERSION}"
    assert out == whisper_server.constraints_text()


def test_first_call_cache():
    loads = []

    def load(name, device, **kw):
        loads.append((name, device, kw))
        return object()

    cached = whisper_server._first_call_cache(load)
    a = cached("large-v3", "cuda", asr_options={"temperatures": (0.0, 0.2)})
    assert cached("large-v3", "cuda", asr_options={"temperatures": (0.0, 0.2)}) is a and len(loads) == 1
    b = cached("small", "cuda")
    assert b is not a and len(loads) == 2 and cached.last is b


@pytest.fixture
def fake_whisperx(monkeypatch):
    """Minimal whisperx: argparse CLI + transcribe_task that uses the module-level loaders."""
    loads = {"model": 0, "align": 0}
    pkg = types.ModuleType("whisperx")
    transcribe = types.ModuleType("whisperx.transcribe")
    main = types.ModuleType("whisperx.__main__")
    log_utils = types.ModuleType("whisperx.log_utils")
    log_utils.setup_logging = lambda level=None: None

    def load_model(name, device, **kw):
        loads["model"] += 1
        return types.SimpleNamespace(name=name)

    def load_align_model(lang, device, model_name=None, **kw):
        loads["align"] += 1
        return object(), {"language": lang}

    def transcribe_task(args, parser):
        import os

        model = transcribe.load_model(args.pop("model"), args.pop("device"), vad_options={"chunk_size": 30})
        transcribe.load_align_model(args["language"], "cuda", model_name=args.pop("align_model"))
        del model
        for audio in args.pop("audio"):
            if "bad" in audio:
                parser.error("bad audio")
            stem = os.path.splitext(os.path.basename(audio))[0]
            with open(os.path.join(args["output_dir"], stem + ".json"), "w") as f:
                json.dump({"segments": [], "word_segments": [], "language": args["language"],
                           "batch_size": args["batch_size"]}, f)

    def cli():
        p = argparse.ArgumentParser()
        p.add_argument("audio", nargs="+")
        for flag, default in (("--model", "small"), ("--language", None), ("--device", "cpu"),
                              ("--compute_type", "default"), ("--align_model", None),
                              ("--output_dir", "."), ("--output_format", "all")):
            p.add_argument(flag, default=default)
        p.add_argument("--batch_size", type=int, default=8)
        p.add_argument("--chunk_size", type=int, default=30)
        from whisperx.transcribe import transcribe_task as task

        task(p.parse_args().__dict__, p)

    transcribe.transcribe_task, transcribe.load_model, transcribe.load_align_model = (
        transcribe_task, load_model, load_align_model)
    main.cli = cli
    for name, mod in (("whisperx", pkg), ("whisperx.transcribe", transcribe), ("whisperx.__main__", main),
                      ("whisperx.log_utils", log_utils)):
        monkeypatch.setitem(sys.modules, name, mod)
    import importlib.metadata

    real_version = importlib.metadata.version
    monkeypatch.setattr(importlib.metadata, "version",
                        lambda n: whisper_server.WHISPERX_VERSION if n == "whisperx" else real_version(n))
    return loads


FLAGS = ["--model", "large-v3", "--language", "en", "--device", "cuda", "--compute_type", "float16",
         "--batch_size", "16", "--align_model", "WAV2VEC2_ASR_LARGE_LV60K_960H", "--output_format", "json"]


def test_cli_transcriber_uses_cli_args_and_loads_models_once(fake_whisperx, tmp_path):
    tr = whisper_server.CliTranscriber(FLAGS, detect_language=False)
    assert tr.args["batch_size"] == 16 and tr.args["chunk_size"] == 30  # parser defaults kept
    assert tr.args["model"] == "large-v3" and tr.version == whisper_server.WHISPERX_VERSION
    for name in ("a", "b", "c"):
        wav = tmp_path / f"{name}.wav"
        wav.write_bytes(b"RIFF")
        out = tmp_path / "o" / f"{name}.json"
        assert tr.transcribe(str(wav), str(out)) == {"whisperx_s": pytest.approx(0, abs=5)}
        assert json.loads(out.read_text())["batch_size"] == 16
    assert fake_whisperx == {"model": 1, "align": 1}
    # a bad clip raises (serve() turns that into ok:false) and the next one still works
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"x")
    r = _run_serve([json.dumps({"id": 1, "wav": str(bad), "out": str(tmp_path / "bad.json")}),
                    json.dumps({"id": 2, "wav": str(tmp_path / "a.wav"), "out": str(tmp_path / "z/a.json")})],
                   lambda req: tr.transcribe(req["wav"], req["out"]))
    assert [x["ok"] for x in r] == [False, True] and fake_whisperx == {"model": 1, "align": 1}


def test_cli_transcriber_rejects_output_dir(fake_whisperx):
    with pytest.raises(ValueError):
        whisper_server.CliTranscriber(FLAGS + ["--output_dir", "/tmp"])


def test_missing_wav_is_an_error(fake_whisperx, tmp_path):
    tr = whisper_server.CliTranscriber(FLAGS, detect_language=False)
    with pytest.raises(FileNotFoundError):
        tr.transcribe(str(tmp_path / "nope.wav"), str(tmp_path / "nope.json"))
