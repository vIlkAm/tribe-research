"""Timeline event lanes: shot boundaries (ffmpeg scene score) and speech (TRIBE's words).

Both are cheap CPU work done where the source clip lives. Text-overlay changes,
speaker changes and audio onsets are not implemented and are reported as
unavailable lanes rather than guessed.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

SCENE_THRESHOLD = 0.3
SPEECH_GAP_S = 0.6  # words closer than this merge into one speech span
UNAVAILABLE_LANES = {
    "text_changes": "No OCR pass yet.",
    "speaker_changes": "No diarization pass yet.",
    "audio_onsets": "No audio-onset detector yet.",
}

_PTS = re.compile(r"pts_time:([0-9.]+)")


def detect_shots(video: str | Path, threshold: float = SCENE_THRESHOLD) -> list[int]:
    """Shot-boundary times (ms) from ffmpeg's scene-change score."""
    import imageio_ffmpeg

    ff = imageio_ffmpeg.get_ffmpeg_exe()
    r = subprocess.run(
        [ff, "-hide_banner", "-nostats", "-i", str(video), "-an",
         "-vf", f"select='gt(scene,{threshold})',showinfo", "-f", "null", "-"],
        capture_output=True, text=True, check=True,
    )
    return sorted({int(round(float(m) * 1000)) for m in _PTS.findall(r.stderr)})


def words_lane(words: list[dict] | None) -> list[dict]:
    """Worker ``words`` ({start, duration, text} in seconds) -> ms, sorted."""
    out = []
    for w in words or []:
        s = float(w["start"])
        out.append({"start_ms": int(round(s * 1000)),
                    "end_ms": int(round((s + float(w.get("duration") or 0)) * 1000)),
                    "text": str(w.get("text", ""))})
    return sorted(out, key=lambda w: w["start_ms"])


def speech_spans(words: list[dict], gap_s: float = SPEECH_GAP_S) -> list[list[int]]:
    spans: list[list[int]] = []
    for w in words:
        if spans and w["start_ms"] - spans[-1][1] <= gap_s * 1000:
            spans[-1][1] = max(spans[-1][1], w["end_ms"])
        else:
            spans.append([w["start_ms"], w["end_ms"]])
    return spans
