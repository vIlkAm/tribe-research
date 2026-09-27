#!/usr/bin/env python3
"""Pick internal demo showcase clips by response profile: "big spike" and "flat stretch" (top 3 each).

    .venv/bin/python tools/select_showcase.py --out results/demo/showcase.json

INTERNAL ONLY (names source files, quotes transcript snippets). Chosen from the predicted response only, never an
outcome: these clips show what the display reads moment by moment, not why a video did well or badly.

Rule (RULE below): eligible as in ``tools/select_demo_clips.py`` (train outside both lockboxes, English, 20-60 s,
>= 1.5 words/s, >= -30 dB, worker output ok), a clean transcript (``tools/language_screen.py``: no strict or mild
word), a full-resolution source file on this server, clip length <= MAX_LEN_S. Per-second percentile of the
response index vs the library (``tools/build_library_profile.py``, self-excluded).

- spike: a run of >= 2 s at >= 85 starting at 2-15 s, whose previous 3 s average <= 50. Ranked by the run's mean
  minus the previous 3 s mean (then a shot cut or speech onset within 1 s of the rise, then hash order).
- flat: a run of >= 5 s at <= 15, inside [2 s, end - 2 s]. Ranked by fewest shot cuts in the run, then the
  longest run, then the lowest mean, then hash order.

The predicted response is pre-shifted for hemodynamic lag and the cut-lag positive control peaks at 0 s
(``state_v1`` M3), so a moment window needs no extra offset; the demo window starts 2 s before a spike.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_library_profile as blp  # noqa: E402
import build_moments_pop as bmp  # noqa: E402
import language_screen  # noqa: E402
import select_demo_clips as sdc  # noqa: E402
import select_example_pair as sep  # noqa: E402
from tribe_research.brain import moments_pop as mp  # noqa: E402

MAX_LEN_S = 40.0
SPIKE = {"min_pct": 85.0, "min_run_s": 2, "start_s": (2, 15), "pre_s": 3, "max_pre_pct": 50.0, "lead_s": 2,
         "window_s": 6}
FLAT = {"max_pct": 15.0, "min_run_s": 5, "edge_s": 2, "max_window_s": 8}
TOP_N = 3
RULE = (
    "Eligible as in tools/select_demo_clips.py; clean transcript (tools/language_screen.py, no strict or mild "
    f"word); source file present; length <= {MAX_LEN_S:g} s. Per-second library percentile of the response index "
    "(self-excluded). spike: run >= 2 s at >= 85 starting at 2-15 s with the previous 3 s mean <= 50; rank by run "
    "mean - previous mean, then a cut/speech onset within 1 s of the rise, then sha256('20260926:<id>'). flat: "
    "run >= 5 s at <= 15 inside [2 s, end - 2 s]; rank by fewest cuts, longest, lowest mean, then hash. "
    "No outcome is read.")


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end] (inclusive) of each run of True."""
    out, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j + 1 < len(mask) and mask[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def snippet(words: list[dict], a_s: float, b_s: float) -> str:
    return " ".join(str(w["text"]) for w in words if a_s <= float(w["start"]) < b_s)


def spike_candidates(p: np.ndarray, dur: float, cuts: list[float], words: list[dict]) -> list[dict]:
    out = []
    for i, j in runs(np.nan_to_num(p, nan=-1) >= SPIKE["min_pct"]):
        if j - i + 1 < SPIKE["min_run_s"] or not (SPIKE["start_s"][0] <= i <= SPIKE["start_s"][1]):
            continue
        pre = p[i - SPIKE["pre_s"]:i]
        if not np.isfinite(pre).any() or np.nanmean(pre) > SPIKE["max_pre_pct"]:
            continue
        onsets = list(cuts) + [float(w["start"]) for w in words]
        cause = any(abs(t - i) <= 1.0 for t in onsets)
        a = max(0, i - SPIKE["lead_s"])
        b = min(dur, a + SPIKE["window_s"])
        out.append({"run_s": [i, j + 1], "run_mean_pct": float(np.nanmean(p[i:j + 1])),
                    "pre_mean_pct": float(np.nanmean(pre)), "rise": float(np.nanmean(p[i:j + 1]) - np.nanmean(pre)),
                    "cut_or_speech_at_rise": cause, "window_ms": [int(a * 1000), int(round(b * 1000))],
                    "label": f"Spike at {blp.mmss(i * 1000)}"})
    return out


def flat_candidates(p: np.ndarray, dur: float, cuts: list[float]) -> list[dict]:
    out = []
    last = int(np.floor(dur - FLAT["edge_s"]))
    for i, j in runs(np.nan_to_num(p, nan=101) <= FLAT["max_pct"]):
        i, j = max(i, FLAT["edge_s"]), min(j, last - 1)
        if j - i + 1 < FLAT["min_run_s"]:
            continue
        b = min(j + 1, i + FLAT["max_window_s"])
        out.append({"run_s": [i, j + 1], "run_mean_pct": float(np.nanmean(p[i:j + 1])), "length_s": j - i + 1,
                    "n_cuts": sum(i <= t < j + 1 for t in cuts), "window_ms": [i * 1000, b * 1000],
                    "label": f"Flat {blp.mmss(i * 1000)}–{blp.mmss(b * 1000)}"})
    return out


def rank(cands: list[dict]) -> dict[str, list[dict]]:
    spikes = sorted((c for c in cands if c["role"] == "spike"),
                    key=lambda c: (-round(c["rise"], 3), not c["cut_or_speech_at_rise"], c["rank_key"]))
    flats = sorted((c for c in cands if c["role"] == "flat"),
                   key=lambda c: (c["n_cuts"], -c["length_s"], round(c["run_mean_pct"], 3), c["rank_key"]))

    def top(group):
        seen, out = set(), []
        for c in group:
            if c["video_id"] not in seen:
                seen.add(c["video_id"])
                out.append(c)
            if len(out) == TOP_N:
                break
        return out
    return {"spike": top(spikes), "flat": top(flats)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--demo-selection", type=Path, default=ROOT / "results/demo/selection.json")
    ap.add_argument("--state", type=Path, default=blp.DEFAULT_STATE)
    ap.add_argument("--roi-map", type=Path, default=bmp.DEFAULT_ROI_MAP)
    ap.add_argument("--cache", type=Path, default=blp.DEFAULT_OUT / ".library_cache_v0.pkl")
    ap.add_argument("--shots", type=Path, default=ROOT / "results/moments_pop/shots")
    ap.add_argument("--staging", type=Path, default=ROOT / "results/run_full/staging")
    ap.add_argument("--selection", type=Path, default=ROOT / "results/study/selection.csv")
    ap.add_argument("--lockbox-ext", type=Path, default=ROOT / "results/study/lockbox_ext.csv")
    ap.add_argument("--members", type=Path, default=ROOT / "results/run_full/members.csv")
    ap.add_argument("--batches-dir", type=Path, default=ROOT / "results/batches_s384")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    demo = json.loads(args.demo_selection.read_text())
    roots = [Path(b["out_root"]) for b in demo["batches_used"]]
    logs = [ROOT / "results/runs/study-bf16/study2-logs"]
    elig, _, _, _ = sdc.eligible(roots, args.selection, args.lockbox_ext, args.members, args.batches_dir, logs)
    found = bmp.discover(roots)

    state = mp.load(args.state)
    _, keys = bmp.masks_and_keys(args.roi_map)
    clips = blp.load_library(state, args.state, args.roi_map, 8, args.cache)
    lib = blp.Library([blp.entry_from_clip(c, state["norms"]) for c in clips], keys)

    counts, cands = {"eligible": len(elig)}, []
    for e in elig:
        vid = e["video_id"]
        why = None
        if e["duration_s"] > MAX_LEN_S:
            why = "too_long"
        elif vid not in lib.index:
            why = "not_in_library"
        words = json.loads(found[vid][0].read_text()).get("words") or [] if vid in found else []
        scr = language_screen.screen(words)
        if why is None and not scr["clean"]:
            why = "language"
        src = sep.source_file(args.staging, e["source_path"]) if why is None else None
        if why is None and src is None:
            why = "no_source_file"
        if why is not None:
            counts[f"excluded_{why}"] = counts.get(f"excluded_{why}", 0) + 1
            continue
        i = lib.index[vid]
        ent = lib.entries[i]
        p = blp.to_grid(ent, lib.pct[i])
        cuts = bmp.load_shots(args.shots, vid) or []
        base = {"video_id": vid, "deal_id": e["deal_id"], "platform": e["platform"], "duration_s": e["duration_s"],
                "source_path": e["source_path"], "source_file": str(src), "batch": e["batch"],
                "rank_key": e["rank_key"], "language": scr}
        for c in spike_candidates(p, e["duration_s"], cuts, words):
            cands.append({"role": "spike", **base, **c,
                          "snippet": snippet(words, c["window_ms"][0] / 1000, c["window_ms"][1] / 1000)})
        for c in flat_candidates(p, e["duration_s"], cuts):
            cands.append({"role": "flat", **base, **c,
                          "snippet": snippet(words, c["window_ms"][0] / 1000, c["window_ms"][1] / 1000)})
    counts["screened_clean"] = counts["eligible"] - sum(v for k, v in counts.items() if k.startswith("excluded_"))
    counts["spike_candidates"] = sum(c["role"] == "spike" for c in cands)
    counts["flat_candidates"] = sum(c["role"] == "flat" for c in cands)
    res = {"rule": RULE, "internal_only": True, "params": {"max_len_s": MAX_LEN_S, "spike": SPIKE, "flat": FLAT,
                                                           "top_n": TOP_N},
           "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "counts": counts, **rank(cands)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps({"counts": counts, **{r: [{k: c[k] for k in ("video_id", "duration_s", "label", "run_s")}
                                                for c in res[r]] for r in ("spike", "flat")}}, indent=1))
    return 0 if res["spike"] and res["flat"] else 1


if __name__ == "__main__":
    sys.exit(main())
