"""Fast V-JEPA2 frame loop for neuralset's HuggingFaceVideo (opt-in, same model inputs).

Stock loop (neuralset 0.0.2, extractors/video.py): for every 0.5 s step it asks moviepy
for 64 frames spread over the previous clip_duration (4 s). Each step starts earlier than
the last one ended, so moviepy restarts ffmpeg with a seek, re-decodes and pipes the
window, wraps every frame in PIL, then the VJEPA2 processor resizes/crops/normalises all
64 frames again, although ~7/8 of them were already processed by the previous step. It
also reloads the 1B-param model on every _get_data call. All of it runs on one CPU
thread between GPU forwards (pilot: GPU ~54% busy, worker pinned at 100% of one core).

This replacement keeps the model inputs identical in fp32 mode:
  * frames: decode="exact" (default) reads them through the clip's own moviepy reader
    with the same video.get_frame(t) calls, so they are stock's frames by construction,
    VFR files included; decode="replay" decodes the file once and replays moviepy's
    seek/skip/EOF logic on frame indices (faster on large originals, exact only for
    constant-frame-rate files: a 28.72 fps VFR clip failed parity in replay mode);
  * runs the VJEPA2 processor once per unique frame (per-frame processing is bitwise
    identical to the batched call), multi-threaded, instead of 64 frames every step;
  * prepares step k+1 on a background thread while the GPU runs step k;
  * keeps the loaded model across calls.
tests/test_fast_video.py checks stock vs fast model inputs bitwise.

Precision (TRIBE_VIDEO_PRECISION): fp32 (default, stock numerics), tf32, bf16, fp16.
Anything but fp32 changes the features: use a separate --cache-folder (the feature cache
key does not include precision; worker.py enforces this) and one mode per dataset.

Other video models and max_imsize fall back to the stock implementation. keep_model
(default on) keeps V-JEPA2 loaded between clips instead of reloading ~4 GB per call;
turn it off if the other extractors run out of VRAM.
"""

from __future__ import annotations

import contextlib
import copy
import logging
import os
import queue
import threading
import time
import typing as tp
import warnings

import numpy as np

log = logging.getLogger("fast_video")

PRECISIONS = ("fp32", "tf32", "bf16", "fp16")
DECODES = ("exact", "replay")
# cumulative over the process; worker.py diffs it per clip
STATS: dict[str, float] = {"clips": 0, "steps": 0, "feeder_wait_s": 0.0, "gpu_s": 0.0, "frames_processed": 0}
_STATE: dict[str, tp.Any] = {"installed": False, "precision": "fp32", "threads": 1, "keep_model": True,
                             "decode": "exact", "orig": None}
_MODELS: dict[tuple, tp.Any] = {}
PROCESS_CHUNK = 32  # frames per processor call (result does not depend on it)


def frame_number(fps: float, t: float) -> int:
    """moviepy FFMPEG_VideoReader.get_frame_number (incl. its +0.00001)."""
    return int(fps * t + 0.00001)


class MoviepyReplay:
    """moviepy 2.2.1 FFMPEG_VideoReader.get_frame, on frame indices instead of pixels.

    `valid(i)` says whether the stream has a frame i (reads past EOF keep the last frame
    that was successfully read, like moviepy's "Using the last valid frame instead").
    `get(t)` returns the index of the frame moviepy would return for time t.
    """

    def __init__(self, fps: float, valid: tp.Callable[[int], bool]):
        self.fps, self.valid = fps, valid
        self.pos = 0
        self.last: int | None = None
        self._initialize(0.0)  # VideoFileClip() opens the reader at t=0 and reads frame 0

    def _read(self) -> None:
        if self.valid(self.pos):
            self.last = self.pos
        elif self.last is None:
            raise OSError("failed to read the first frame")
        self.pos += 1

    def _initialize(self, t: float) -> None:
        # ffmpeg seek to pos/fps - 1e-5 lands on frame `pos`, which is then read
        self.pos = frame_number(self.fps, t)
        self._read()

    def get(self, t: float) -> int:
        pos = frame_number(self.fps, t) + 1
        if pos == self.pos:
            return self.last  # type: ignore[return-value]
        if pos < self.pos or pos > self.pos + 100:
            self._initialize(t)
            return self.last  # type: ignore[return-value]
        self.pos += pos - self.pos - 1  # skip_frames
        self._read()
        return self.last  # type: ignore[return-value]


class FrameStream:
    """Sequential decode with moviepy's own ffmpeg command (same pixels as get_frame)."""

    def __init__(self, reader):
        self.reader = copy.copy(reader)  # same filename/size/pix_fmt/sws flags, own process
        self.reader.proc = None
        self.reader.initialize(0)  # starts ffmpeg at t=0 and reads frame 0 into last_read
        w, h = self.reader.size
        self.shape = (h, w)
        self.nbytes = self.reader.depth * w * h
        self.decoded = 1  # frames read so far (frame 0)
        self.eof = False
        self._first = self.reader.last_read

    def next(self) -> np.ndarray | None:
        """The next frame (frame 0 first), None at EOF."""
        if self._first is not None:
            f, self._first = self._first, None
            return f
        if self.eof:
            return None
        s = self.reader.proc.stdout.read(self.nbytes)
        if len(s) != self.nbytes:
            self.eof = True
            return None
        self.decoded += 1
        h, w = self.shape
        return np.frombuffer(s, dtype="uint8").reshape(h, w, len(s) // (w * h))

    def close(self) -> None:
        self.reader.close(delete_lastread=True)


def plan_steps(duration: float, freq: float, T: float, num_frames: int, expect_frames: int):
    """Stock loop's (time at end of step, frame times) per step."""
    subtimes = [k / num_frames * T for k in reversed(range(num_frames))]
    times = np.linspace(0, duration, expect_frames + 1)[1:]
    return [(float(t), [max(0, t - t2) for t2 in subtimes]) for t in times]


class StepFeeder:
    """Background producer of (1, num_frames, 3, H, W) processor tensors, one per step.

    decode="exact": frames come from the clip's own moviepy reader through
    video.get_frame(t), the call stock makes, so they are stock's frames by construction
    (any codec, VFR, seeks, EOF). A frame is preprocessed once and reused only when
    byte-identical. Costs stock's decode (~0.1 s/step at 384 px), overlapped with the GPU.
    decode="replay": decode the file once and replay moviepy's seek/skip logic on frame
    indices. Faster on large originals, but only exact for constant-frame-rate files.
    """

    def __init__(self, video, processor, steps, offset: float, pin: bool, decode: str = "exact",
                 depth: int = 2):
        import torch

        self.torch = torch
        self.video, self.processor, self.pin, self.decode = video, processor, pin, decode
        self.steps = steps
        self.offset = offset  # the clip is subclipped(offset, ...): moviepy reads t + offset
        self.fps = float(video.reader.fps)
        self.q: queue.Queue = queue.Queue(maxsize=depth)
        self.stop = threading.Event()
        self.error: BaseException | None = None
        self.stats = {"steps": 0, "frames_processed": 0}
        self.thread = threading.Thread(target=self._run, name="fast-video-feeder", daemon=True)
        self.thread.start()

    def _put(self, item) -> bool:
        while not self.stop.is_set():
            try:
                self.q.put(item, timeout=0.2)
                return True
            except queue.Full:
                continue
        return False

    def _exact_source(self):
        """Per step: (keys, frames by key, keep(key) after the step).

        A frame is reused only if it is byte-identical to one already processed for the
        same nominal index (the usual case: ffmpeg decodes deterministically), so a VFR
        file where a seek lands elsewhere just gets its actual frame processed.
        """
        seen: dict[int, list[tuple[np.ndarray, tuple[int, int]]]] = {}
        for _, sample_times in self.steps:
            keys, frames = [], {}
            for ts in sample_times:
                f = self.video.get_frame(ts)
                idx = frame_number(self.fps, ts + self.offset)
                variants = seen.setdefault(idx, [])
                key = next((k for raw, k in variants if raw is f or np.array_equal(raw, f)), None)
                if key is None:
                    key = (idx, len(variants))
                    variants.append((f, key))
                    frames[key] = f
                keys.append(key)
            current = set(keys)
            yield keys, frames, current.__contains__
            for idx in [i for i in seen if not any(k in current for _, k in seen[i])]:
                del seen[idx]

    def _replay_source(self):
        stream = FrameStream(self.video.reader)
        try:
            steps = [(t, [ts + self.offset for ts in sts]) for t, sts in self.steps]
            candidates = {frame_number(self.fps, ts) for _, sts in steps for ts in sts}
            raw: dict[int, np.ndarray] = {}
            next_idx = 0

            def valid(i: int) -> bool:
                nonlocal next_idx
                while next_idx <= i and not stream.eof:
                    f = stream.next()
                    if f is None:
                        break
                    if next_idx in candidates:
                        raw[next_idx] = f
                    next_idx += 1
                return i < next_idx

            replay = MoviepyReplay(self.fps, valid)
            for k, (_, sample_times) in enumerate(steps):
                idx = [replay.get(ts) for ts in sample_times]
                # windows only move forward: frames before the next window are never read again
                lo = frame_number(self.fps, min(steps[k + 1][1])) - 1 if k + 1 < len(steps) else 1 << 62
                last = replay.last
                yield idx, raw, (lambda i, lo=lo, last=last: i >= lo or i == last)
                for i in [i for i in raw if i < lo and i != last]:
                    del raw[i]
        finally:
            stream.close()

    def _run(self) -> None:
        try:
            # OpenMP thread counts are per thread: set it here, not only in the caller
            self.torch.set_num_threads(_STATE["threads"])
            source = self._exact_source() if self.decode == "exact" else self._replay_source()
            done: dict[tp.Any, tp.Any] = {}  # processed frames, (3, H, W) float32
            for k, (keys, frames, keep) in enumerate(source):
                if self.stop.is_set():
                    return
                new = [key for key in dict.fromkeys(keys) if key not in done]
                for c in range(0, len(new), PROCESS_CHUNK):
                    chunk = new[c:c + PROCESS_CHUNK]
                    arr = np.stack([np.asarray(frames[key], dtype=np.uint8) for key in chunk])
                    pv = self.processor(videos=[arr], return_tensors="pt")["pixel_values_videos"][0]
                    for j, key in enumerate(chunk):
                        done[key] = pv[j]
                self.stats["frames_processed"] += len(new)
                batch = self.torch.stack([done[key] for key in keys]).unsqueeze(0)
                if self.pin:
                    batch = batch.pin_memory()
                if not self._put((k, batch)):
                    return
                self.stats["steps"] += 1
                for key in [key for key in done if not keep(key)]:
                    del done[key]
        except BaseException as exc:  # surfaced in the consumer
            self.error = exc
        finally:
            self._put(None)

    def __iter__(self):
        while True:
            item = self.q.get()
            if item is None:
                if self.error is not None:
                    raise self.error
                return
            yield item

    def close(self) -> None:
        self.stop.set()
        with contextlib.suppress(queue.Empty):
            while True:
                self.q.get_nowait()
        self.thread.join(timeout=30)


@contextlib.contextmanager
def precision_context(precision: str, device_type: str):
    import torch

    if precision == "fp32":
        yield
    elif precision == "tf32":
        prev = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
        try:
            yield
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = prev
    else:
        dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        with torch.autocast(device_type=device_type, dtype=dtype):
            yield


def _supported(ext, events) -> bool:
    return "vjepa2" in ext.image.model_name and ext.max_imsize is None


def _get_model(ext):
    from neuralset.extractors.video import _HFVideoModel

    key = (ext.image.model_name, ext.image.pretrained, ext.layer_type, ext.num_frames, str(ext.image.device))
    if key not in _MODELS:
        model = _HFVideoModel(model_name=ext.image.model_name, pretrained=ext.image.pretrained,
                              layer_type=ext.layer_type, num_frames=ext.num_frames)
        if model.model.device.type == "cpu":
            model.model.to(ext.image.device)
        _MODELS.clear()  # one video model resident at a time
        _MODELS[key] = model
    return _MODELS[key]


def fast_get_data(self, events):
    """Drop-in for HuggingFaceVideo._get_data (the function under @infra.apply)."""
    if not _supported(self, events):
        yield from _STATE["orig"](self, events)
        return
    import torch

    logging.getLogger("neuralset").setLevel(logging.DEBUG)  # as stock
    prev_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(_STATE["threads"])
        yield from _fast_events(self, events)
    finally:
        torch.set_num_threads(prev_threads)
        if not _STATE["keep_model"]:
            _MODELS.clear()


def _fast_events(self, events):
    import torch
    from neuralset import base as nsbase
    from neuralset.extractors.image import _fix_pixel_values

    model = _get_model(self)
    device = model.model.device
    precision = _STATE["precision"]
    for event in events:
        video = event.read()
        freq = self.frequency if self.frequency != "native" else event.frequency
        T = 1 / freq if self.clip_duration is None else self.clip_duration
        expect_frames = nsbase.Frequency(freq).to_ind(event.duration)
        steps = plan_steps(video.duration, freq, T, model.num_frames, expect_frames)
        log.debug("fast video: %s, %d steps, %s fps, precision %s", event.filepath, len(steps),
                  video.reader.fps, precision)
        feeder = StepFeeder(video, model.processor, steps, event.offset, pin=device.type == "cuda",
                            decode=_STATE["decode"])
        output = np.array([])
        steps_done, wait_s, gpu_s = 0, 0.0, 0.0
        try:
            it = iter(feeder)
            while True:
                t0 = time.perf_counter()
                item = next(it, None)
                t1 = time.perf_counter()
                wait_s += t1 - t0  # GPU idle waiting for frames: >0 per step means CPU-bound
                if item is None:
                    break
                k, pixels = item
                inputs = {"pixel_values_videos": pixels}
                _fix_pixel_values(inputs)
                inputs = {n: v.to(device, non_blocking=True) for n, v in inputs.items()}
                with torch.inference_mode(), precision_context(precision, device.type):
                    pred = model.model(**inputs)
                states = pred.hidden_states
                t_embd = torch.cat([x.unsqueeze(1) for x in states], axis=1)[0]
                if t_embd.dtype != torch.float32:
                    t_embd = t_embd.float()
                embd = self.image._aggregate_tokens(t_embd).cpu().numpy()
                if not self.image.cache_all_layers and self.image.cache_n_layers is None:
                    embd = self.image._aggregate_layers(embd)
                if not output.size:
                    output = np.zeros((len(steps),) + embd.shape)
                output[k] = embd
                gpu_s += time.perf_counter() - t1  # H2D + forward + aggregation (.cpu() syncs)
                steps_done += 1
        finally:
            feeder.close()
            video.close()
            for key, v in (("clips", 1), ("steps", steps_done), ("feeder_wait_s", wait_s), ("gpu_s", gpu_s),
                           ("frames_processed", feeder.stats["frames_processed"])):
                STATS[key] += v
        log.info("fast video: %s %d steps, %.3f s/step GPU, %.3f s/step waiting for frames",
                 event.filepath, steps_done, gpu_s / max(steps_done, 1), wait_s / max(steps_done, 1))
        output = output.transpose(list(range(1, output.ndim)) + [0])
        yield nsbase.TimedArray(data=output.astype(np.float32), frequency=freq,
                                start=nsbase._UNSET_START, duration=event.duration)


def install(precision: str = "fp32", threads: int | None = None, keep_model: bool = True,
            decode: str = "exact") -> dict:
    """Swap the function under HuggingFaceVideo._get_data's @infra.apply (cache keys unchanged)."""
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
    if decode not in DECODES:
        raise ValueError(f"decode must be one of {DECODES}, got {decode!r}")
    # stock silences these via neuralset's ignore_all() around each get_frame
    warnings.filterwarnings("ignore", message=".*Using the last valid frame instead.*")
    from neuralset.extractors.video import HuggingFaceVideo

    imethod = HuggingFaceVideo.__dict__["_get_data"].fget
    if not _STATE["installed"]:
        _STATE["orig"] = imethod.method
        imethod.method = fast_get_data
        _STATE["installed"] = True
    _STATE["precision"] = precision
    _STATE["threads"] = threads or min(8, os.cpu_count() or 1)
    _STATE["keep_model"] = keep_model
    _STATE["decode"] = decode
    info = {"fast_video": True, "video_precision": precision, "video_cpu_threads": _STATE["threads"],
            "video_keep_model": keep_model, "video_decode": decode}
    log.info("fast video loop installed: %s", info)
    return info


def uninstall() -> None:
    if _STATE["installed"]:
        from neuralset.extractors.video import HuggingFaceVideo

        HuggingFaceVideo.__dict__["_get_data"].fget.method = _STATE["orig"]
        _STATE["installed"] = False
    _MODELS.clear()
