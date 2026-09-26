# tribe-research

Run **TRIBE v2** over a library of short-form videos, save the full predicted
cortical time series, and later compare it with real performance (retention,
views, virality).

```text
video → V-JEPA2 ViT-G ─┐
audio → Wav2Vec-BERT ──┼─► TRIBE v2 ─► predicted cortical activity [time × ~20k vertices]
words → Llama-3.2-3B ──┘   (whisperx transcribes the audio first)
```

The expensive part is V-JEPA2 feature extraction; the TRIBE fusion model is small.
Isolation rules and layout: [`AGENTS.md`](AGENTS.md).
Where the project stands right now: [`docs/STATUS.md`](docs/STATUS.md).

## Quickstart for collaborators

```bash
git clone https://github.com/vIlkAm/tribe-research && cd tribe-research
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q            # all offline, no GPU, no credentials
```

- **Frontend work:** build against `docs/analysis.schema.json`, use
  `docs/sample_analysis/` (synthetic) as fixture data, and read the spec in
  `docs/frontend_spec/`. The HTML there is the source of the PDF.
- **ROI map:** `tribe_research/assets/roi_map_roi_groups_v0.npz` is gitignored
  (like all `.npz`). Build it from the HCP-MMP1 annotation mirror:
  ```bash
  mkdir -p /tmp/hcpannot && for h in lh rh; do curl -fsSL -o /tmp/hcpannot/$h.HCPMMP1.annot \
    https://raw.githubusercontent.com/tannerjared/HCP-MMP1/master/$h.HCP-MMP1.annot; done
  .venv/bin/python tools/build_roi_map.py --annot-dir /tmp/hcpannot
  ```
- **Real clips and TRIBE outputs are never committed.** Clips live only on the
  owner's server; RunPod runs are launched by the owner. Share results as
  pulled `analysis.json` bundles, not raw videos.
- **Working together:** [`docs/TEAM.md`](docs/TEAM.md) covers who owns what, how the
  contract changes, and shared rules. Frontend: copy `docs/analysis.types.ts`.
- Agents: read [`AGENTS.md`](AGENTS.md) first (`CLAUDE.md` points there).

## Upstream facts (verified against tribev2 @ `af58661`, 2026-06-23)

- Python `>=3.11`, **torch `>=2.5.1,<2.7`**, torchvision `>=0.20,<0.22`. Pick a
  RunPod image with torch 2.5/2.6 + CUDA 12.4, not the generic 2.8 one;
  `setup.sh` refuses to continue outside the pin.
- API: `TribeModel.from_pretrained("facebook/tribev2", cache_folder=...)`,
  `get_events_dataframe(video_path=...)`, `predict(events)` → `(preds, segments)`.
- `cache_folder` is where TRIBE caches **extracted text/audio/video features**,
  so "cache the V-JEPA embeddings" = keep `feature-cache/` on the network volume.
- `predict` drops segments with no events by default, so timestamps can have
  gaps. The worker saves each segment's start time, not just an index.
- Preds are for an "average subject" on fsaverage5 and are **shifted 5 s into the
  past** to cancel hemodynamic lag. Account for this when aligning to retention.
- Transcription shells out to `uvx whisperx … large-v3 --device cuda`. `setup.sh`
  installs `uv` so whisperx gets its own env. The March prototype hit a torch
  conflict installing whisperx into the main env and forced it to CPU. Only fall
  back to that if `uvx` fails.
- Llama-3.2-3B is gated: the HF account behind `HF_TOKEN` must have accepted
  Meta's licence on huggingface.co.
- Accepted containers: `.mp4 .avi .mkv .mov .webm`.
- Licence: **CC-BY-NC-4.0**.

## Transcription (one whisperx process per worker)

Stock TRIBE runs a fresh `uvx whisperx <wav> --model large-v3 …` for every clip,
reloading large-v3 and the align model each time (~20–40 s per clip). Real
worker runs instead patch `ExtractWordsFromAudio._get_transcript_from_audio` to
send each wav to one long-lived `pod/whisper_server.py` per worker. The server:

- runs in whisperx's own uv env (`uvx --from whisperx==3.8.6 python …`), never TRIBE's venv;
- builds its arguments with whisperx's own CLI parser from TRIBE's exact flags;
- calls whisperx's own `transcribe_task` per clip, with only the two model loaders memoised.

The JSON, and therefore the word DataFrame and `.tsv`, should be the stock CLI's.

- **Paths.** neuralset writes `<video>.wav` next to the video, and TRIBE caches the
  words as `<video>.tsv` beside it. A clip with a `.tsv` is never re-transcribed,
  by either path.
- **Pins.** `whisper_server.WHISPERX_PINS` is the single pin list. The worker
  exports it as `UV_CONSTRAINT`, so TRIBE's unmodified `uvx whisperx` fallback
  resolves the same versions. `setup.sh` writes `$JOB/logs/whisperx-constraints.txt`,
  pre-warms both envs and the models, and checks both resolve with `uvx --offline`.
- **Fallback.** Any server problem (start failure, crash, timeout, error reply)
  sends that clip through the stock per-clip call, with one warning, so no clip is
  lost. The server is restarted after a crash (3 restarts max) or after 3 consecutive
  error replies (separate budget of 20), but disabled for the run if it never came up. `--no-whisper-server` / `TRIBE_WHISPER_SERVER=0`
  uses the stock path throughout.
- **Per-clip `.json`.**
  - `transcript.source`: `server`, `stock_cli`, `tsv_cache` (TRIBE reused `<video>.tsv`), `none` (no audio track) or `dry_run`.
  - `timing_s.transcription`: part of `events_dataframe`; plus `transcription_server_start` on the clip that started the server.
  - `quality_warnings`:
    - `transcript_empty`
    - `transcript_non_english:<lang>`: whisper's language guess on the first 30 s, p ≥ 0.5; server only
    - `transcript_non_ascii_words`: more than 20 % of words contain non-ASCII letters

  TRIBE still transcribes as English, so filter on these flags in analysis.
- **VRAM.** The server keeps large-v3 and wav2vec2 resident, about 5 GB, next
  to TRIBE's extractors. Upstream freed them before feature extraction. Check
  peak memory in `gpu-worker-N.csv`.

**Verify on the first pod before a full run.** Use three real clips with speech,
after a TRIBE run has extracted their `.wav`.

`--compare-transcription` calls both paths directly, so the `.tsv` cache is bypassed:

```bash
source pod/env.sh && export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
python pod/worker.py --compare-transcription $JOB/videos/a.wav $JOB/videos/b.wav $JOB/videos/c.wav \
  --compare-repeats 2 | tee $JOB/logs/whisper-compare.jsonl
```

It prints one JSON line per wav per stock run:

- `identical_frame`: exact `assert_frame_equal` of the DataFrames, and the `.tsv` bytes match (`identical_tsv`)
- `identical_words`
- `max_abs_start_diff`
- `server_s` / `stock_s`
- the detected language

It exits 0 only if every frame is identical. Before judging a server diff, check
whether the two stock repeats agree with each other; that is the GPU noise floor.
Expect `server_s` of a few seconds against 20–40 s for `stock_s`.

## Pod volume layout

```text
/workspace/tribe-job/
├── code/            this repo's pod/ tools/ tribe_research/ (tools/pod.sh push-code)
├── repo/            tribev2 pinned commit
├── venv/  hf-cache/  uv-cache/  torch-cache/  feature-cache/
├── videos/          input clips (any sub-folders)
├── manifest.jsonl   one row per video, with assigned worker
├── outputs/worker-N/<video_id>.npz|.json
└── logs/            worker-N.log, gpu-worker-N.csv
```

Per video, the worker writes:

- `<video_id>.npz`:
  - `preds` float16 `[segments, 20484]`
  - `seg_start`
  - `seg_duration`
- `<video_id>.json`:
  - duration and TR
  - TRIBE commit, torch/CUDA/GPU versions
  - timings: `events_dataframe` (audio + whisperx + text) vs `predict` (feature extraction + TRIBE)
  - realtime factor

The `.json` is written last and is the completion marker, so re-running a
worker resumes where it stopped.

## Phase 1: one cheap GPU, get it working, measure

Nothing below has been run yet. Each pod launch is a billed step. The full
spin-up, connect and teardown procedure is in [`docs/RUNPOD.md`](docs/RUNPOD.md).

1. With the watchdog running (`--max-usd 10`), launch one on-demand
   **Community Cloud** pod with `tools/runpod.py pod-create`. Pick the GPU from
   `tools/runpod.py gpus`: ≥ 24 GB VRAM, host RAM ≥ 48 GB, e.g. 4090 or L40S.
   It gets a 120 GB pod volume at `/workspace`, TCP 22 on a public IP, and
   `HF_TOKEN` from `.env`. `pod-wait` prints the `POD`/`POD_PORT` below. The
   pod volume is **deleted on terminate**, so pull results first.
2. From this server, push code and 3–5 representative clips (varied length,
   some with speech):
   ```bash
   export POD=root@<ip> POD_PORT=<port>
   tools/pod.sh push-code && tools/pod.sh push-videos ./videos
   ```
3. On the pod: `setup.sh`, build the manifest, run `--limit 1`, run the rest,
   merge (commands in `docs/RUNPOD.md`).
4. Record the measured numbers:
   - source seconds per compute second
   - events-vs-predict split
   - peak VRAM and GPU utilisation from `gpu-worker-0.csv`
   - model load time
   - `failures_by_category` from `benchmark.json`

   Run one clip twice. The second pass hits `feature-cache/`, which isolates
   TRIBE-only time from V-JEPA time.
5. `tools/pod.sh pull run1`, then terminate the pod.

## Phase 2: fan out across GPUs

Use **one pod with N GPUs**, not N pods sharing a volume.

1. Rebuild the manifest with `--workers N`. Assignment is greedy
   longest-first by duration, so shards finish together.
2. `pod/launch_all.sh` starts one worker per GPU under `nohup`. It refuses to
   start if the manifest was built for a different N.
3. Merge. `merge.py` exits non-zero and lists missing or failed videos, with a
   failure category. Re-run the launcher; finished videos are skipped.

Reuse the same RunPod template for every pod on this volume: the venv links to
the image's Python.

## Brain layer (ROI features, neural proxies, frontend bundle)

`tribe_research/brain/` turns the raw `[segments × 20484]` predictions into
something that can be read and shown:

- `roi_groups_v0.yaml`: named groups of HCP-MMP1 parcels, e.g. visual,
  auditory, language.
- `tools/build_roi_map.py`: builds the versioned fsaverage5 ROI map. The local
  copy was built from the [tannerjared/HCP-MMP1](https://github.com/tannerjared/HCP-MMP1)
  mirror of the [figshare original](https://figshare.com/articles/HCP-MMP1_0_projected_on_fsaverage/3498446),
  MD5-recorded in its provenance. `push-code` sends it to the pod, where
  `setup.sh` cross-checks it against tribev2's own labels.
- `features.py`: per-ROI curves on the real segment timeline, plus within-video
  z-scored summaries. Don't compare absolute activation across videos.
- `proxies_v0.yaml` / `proxies.py`: seven disjoint proxy channels (attention,
  social, value, control, self, language, sensory). Each has an explicit
  `direction` (cognitive control is `no_monotonic`: neither way is better) and
  its UI copy. Values are within-clip z-scores whose scale excludes the 2 s
  onset window, on a 2 Hz sample-and-hold display grid. Segments TRIBE dropped
  stay `null`.
- `events.py`: shot changes (ffmpeg scene score) and TRIBE's transcribed words
  / speech spans. Other lanes are reported as unavailable.
- `moments.py`: rule-based candidate moments (hysteresis, min 1.5 s). They are
  observations or *untested* edit hypotheses, always relative to this clip.
- `analysis.py` / `bundle.py`: the frontend object `nvi.analysis.v0.2`
  (`docs/analysis.schema.json`). `predictions` stay `not_available` until a
  behavior model exists. Scrubbable brain sprites (proxy + research per-vertex),
  a region hover map and a legend are included.
- `tools/brain_report.py --analysis [--png] [--video]`: writes
  `<report>/analyses/<video_id>/analysis.json` plus assets, and a demo MP4
  (source | proxy brain | readout over a timeline of channels, moments and
  shots). It runs on CPU on this server after `pod.sh pull`.

`docs/sample_analysis/` is a **synthetic** example bundle (dry-run predictions,
real ROI map) for frontend work.

Every visual is captioned as a model prediction for an average subject, not
measured brain activity. UI wording is "strong predicted response", never
"brain firing". Label anything built from dry-run data or a synthetic ROI map
as synthetic.

## Planning estimates (from the planning discussion; not measured)

Throughput unit: TRIBE's released config uses 4 s video windows at 2 Hz.
90 min of source is about 10,800 V-JEPA windows.

| Setup | Rate | 90 min of source | Rough cost |
|---|---|---|---|
| 1× L40S | $1.09/h | ~60–90 min | ~$1–2 |
| 4× L40S | $4.36/h | ~15–25 min | ~$1.5 |
| 1× H100 SXM | $3.49/h | ~20–35 min | ~$1–2 |
| 1× 4090 | $0.74/h | ? (24 GB VRAM, 31 GB host RAM risk) | ? |

Replace every row with measured numbers after phase 1. Choose hardware on
measured $/1000 videos, not paper specs. Benchmark L40S / 4090 / H100 on the
**same** 5–10 clips.

## Later: analysis (not started)

- Import performance metrics as an exported file keyed by the same video
  (`source_name` / platform post ID), into `results/`. Don't query the Cartel DB
  live (see AGENTS.md).
- Keep raw time series. Candidate questions:
  - What activity precedes retention drop-off?
  - Are there response signatures during hooks, cuts, and surprising lines?
  - Is the trajectory more predictive than the mean level?
- Region-level summaries come from the brain layer above, on the HCP-MMP1
  atlas; the March prototype's hand-made region map is superseded.

## Local development

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q       # offline: sharding, manifest, dry-run/resume/merge, brain layer
.venv/bin/python pod/worker.py --dry-run ...   # stub model, no GPU
```
