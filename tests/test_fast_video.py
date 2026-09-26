"""pod/fast_video.py: frame selection must match moviepy's get_frame exactly.

The pure tests run everywhere. The moviepy/torch/neuralset parity test runs where
those are installed (a pod, or a CPU replica venv): it compares stock
HuggingFaceVideo._get_data with the fast path on a synthetic clip, bitwise, per step.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pod"))
import fast_video  # noqa: E402
import worker  # noqa: E402


class MoviepyReaderModel:
    """moviepy 2.2.1 FFMPEG_VideoReader get_frame/initialize/skip_frames/read_frame,
    over a list of frame ids instead of an ffmpeg pipe (a seek to frame p yields p, p+1, ...)."""

    def __init__(self, fps, n_frames):
        self.fps, self.n = fps, n_frames
        self.initialize(0)

    def initialize(self, t=0):
        self.pos = int(self.fps * t + 0.00001)
        self.stream = self.pos
        self.last_read = self.read_frame_raw()

    def read_frame_raw(self):
        if self.stream < self.n:
            r = self.stream
            self.stream += 1
            self.last_read = r
        else:
            r = self.last_read
        self.pos += 1
        return r

    def get_frame(self, t):
        pos = int(self.fps * t + 0.00001) + 1
        if pos == self.pos:
            return self.last_read
        if pos < self.pos or pos > self.pos + 100:
            self.initialize(t)
            return self.last_read
        n = pos - self.pos - 1
        self.stream += n  # skip_frames reads and drops n frames (none left past EOF)
        self.pos += n
        return self.read_frame_raw()


def stock_indices(reader, duration, fps_unused=None, freq=2.0, T=4.0, nf=64):
    expect = int(round(duration * freq))
    return [[reader.get_frame(ts) for ts in sts]
            for _, sts in fast_video.plan_steps(duration, freq, T, nf, expect)]


def replay_indices(fps, n_frames, duration, freq=2.0, T=4.0, nf=64):
    r = fast_video.MoviepyReplay(fps, lambda i: i < n_frames)
    expect = int(round(duration * freq))
    return [[r.get(ts) for ts in sts] for _, sts in fast_video.plan_steps(duration, freq, T, nf, expect)]


@pytest.mark.parametrize("fps,n_frames", [(30, 350), (60, 705), (29.89, 522), (24, 585), (50, 751),
                                          (59.94, 1871), (30, 55), (60, 181), (25, 12)])
@pytest.mark.parametrize("short_by", [0, 1, 3])  # stream ends before the container duration
def test_replay_matches_moviepy_state_machine(fps, n_frames, short_by):
    duration = n_frames / fps
    got = replay_indices(fps, n_frames - short_by, duration)
    want = stock_indices(MoviepyReaderModel(fps, n_frames - short_by), duration)
    assert got == want


def test_plan_steps_is_the_stock_schedule():
    steps = fast_video.plan_steps(11.75, 2.0, 4.0, 64, 24)
    times = np.linspace(0, 11.75, 25)[1:]
    sub = [k / 64 * 4.0 for k in reversed(range(64))]
    assert [t for t, _ in steps] == list(times)
    assert steps[3][1] == [max(0, times[3] - s) for s in sub]


def test_worker_refuses_untagged_low_precision_cache(tmp_path):
    base = [sys.executable, str(ROOT / "pod" / "worker.py"), "--manifest", str(tmp_path / "m.jsonl"),
            "--videos-root", str(tmp_path), "--out-root", str(tmp_path / "out"), "--dry-run"]
    (tmp_path / "m.jsonl").write_text("")
    r = subprocess.run(base + ["--fast-video", "--video-precision", "bf16",
                               "--cache-folder", str(tmp_path / "feature-cache")], capture_output=True, text=True)
    assert r.returncode == 2 and "cache folder named for it" in r.stderr
    r = subprocess.run(base + ["--video-precision", "fp16", "--cache-folder", str(tmp_path / "cache-fp16")],
                       capture_output=True, text=True)
    assert r.returncode == 2 and "needs --fast-video" in r.stderr
    r = subprocess.run(base + ["--fast-video", "--video-precision", "bf16",
                               "--cache-folder", str(tmp_path / "feature-cache-bf16")], capture_output=True, text=True)
    assert r.returncode != 2 and "empty manifest" in r.stderr  # passed the precision gate


def test_bad_precision_or_decode_rejected():
    with pytest.raises(ValueError):
        fast_video.install("int8")
    with pytest.raises(ValueError):
        fast_video.install("fp32", decode="guess")


# ── real moviepy + neuralset (pod / replica venv only) ──────────────────────


def _synthetic_clip(path: Path, fps: float, seconds: float) -> None:
    import imageio_ffmpeg

    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y", "-f", "lavfi",
                    "-i", f"testsrc2=size=320x568:rate={fps}:duration={seconds}",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "30", str(path)], check=True)


@pytest.mark.parametrize("decode", ["exact", "replay"])
@pytest.mark.parametrize("fps,seconds", [(30, 9.4), (60, 5.3), (25, 2.2)])
def test_stock_vs_fast_inputs_bitwise(tmp_path, fps, seconds, decode):
    pytest.importorskip("moviepy")
    torch = pytest.importorskip("torch")
    pytest.importorskip("neuralset")
    import hashlib

    import neuralset.extractors.video as nsv
    from neuralset.events import etypes
    from transformers import AutoVideoProcessor

    try:
        proc = AutoVideoProcessor.from_pretrained("facebook/vjepa2-vitg-fpc64-256", do_rescale=True)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"V-JEPA2 processor unavailable: {exc!r}")
    seen: list = []

    class Net(torch.nn.Module):
        device = torch.device("cpu")

        def forward(self, **kw):
            x = kw.pop("pixel_values_videos")
            assert not kw
            seen.append(hashlib.sha1(x.contiguous().numpy().tobytes()).hexdigest())
            f = x.mean(dim=(2, 3, 4))[0]
            return type("O", (), {"hidden_states": tuple(f[None, :, None].expand(1, 64, 4) * (i + 1)
                                                         for i in range(41))})()

    class Stub(nsv._HFVideoModel):
        def __init__(self, model_name, pretrained=True, layer_type="", num_frames=None):
            self.model_name, self.layer_type, self.num_frames = model_name, layer_type, 64
            self.processor, self.model = proc, Net()

    clip = tmp_path / "c.mp4"
    _synthetic_clip(clip, fps, seconds)
    orig_cls = nsv._HFVideoModel
    nsv._HFVideoModel = Stub
    try:
        ext = nsv.HuggingFaceVideo(
            image={"name": "HuggingFaceImage", "model_name": "facebook/vjepa2-vitg-fpc64-256",
                   "infra": {"keep_in_ram": False}, "layers": [0.5, 0.75, 1.0], "cache_n_layers": 20,
                   "layer_aggregation": "group_mean", "token_aggregation": "mean", "device": "cpu"},
            frequency=2.0, clip_duration=4.0, aggregation="sum", event_types="Video", allow_missing=True)
        ev = etypes.Video(start=0, timeline="t", filepath=str(clip))
        stock = nsv.HuggingFaceVideo.__dict__["_get_data"].fget.method
        fast_video.install("fp32", threads=2, decode=decode)
        a = list(stock(ext, [ev]))
        stock_hashes, seen[:] = list(seen), []
        b = list(fast_video.fast_get_data(ext, [ev]))
    finally:
        nsv._HFVideoModel = orig_cls
        fast_video.uninstall()
    assert seen and seen == stock_hashes
    assert np.array_equal(a[0].data, b[0].data) and a[0].duration == b[0].duration


def test_install_keeps_exca_cache_identity(tmp_path):
    """The worker calls the method through exca; its uid needs the stock name/qualname/module."""
    pytest.importorskip("neuralset")
    import neuralset.extractors.video as nsv

    def make():
        return nsv.HuggingFaceVideo(
            image={"name": "HuggingFaceImage", "model_name": "facebook/vjepa2-vitg-fpc64-256",
                   "infra": {"keep_in_ram": False}, "layers": [0.5, 0.75, 1.0], "cache_n_layers": 20,
                   "layer_aggregation": "group_mean", "token_aggregation": "mean", "device": "cpu"},
            frequency=2.0, clip_duration=4.0, aggregation="sum", event_types="Video", allow_missing=True,
            infra={"folder": str(tmp_path), "keep_in_ram": False})

    stock_uid = make().infra.uid_folder(create=False)
    fast_video.install("fp32", threads=2)
    try:
        assert make().infra.uid_folder(create=False) == stock_uid
    finally:
        fast_video.uninstall()
