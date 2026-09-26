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

## Pod volume layout

```text
/workspace/tribe-job/
├── code/            this repo's pod/ + tools/ (rsync from the server)
├── repo/            tribev2 pinned commit
├── venv/  hf-cache/  uv-cache/  feature-cache/
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

## Phase 1: one L40S, get it working, measure

Nothing below has been run yet. Each pod launch is a billed step.

1. Create a RunPod **network volume** and a **1× L40S** pod (torch 2.5/2.6
   image) with the volume at `/workspace` and `HF_TOKEN` set as a secret.
2. From this server, push code and a handful of representative clips (3–5,
   varied length, some with speech):
   ```bash
   rsync -av pod tools root@<pod>:/workspace/tribe-job/code/ -e "ssh -p <port>"
   rsync -av ./videos/ root@<pod>:/workspace/tribe-job/videos/ -e "ssh -p <port>"
   ```
3. On the pod:
   ```bash
   bash /workspace/tribe-job/code/pod/setup.sh
   source /workspace/tribe-job/code/pod/env.sh
   python $JOB/code/tools/make_manifest.py --videos-root $JOB/videos --workers 1 --out $JOB/manifest.jsonl
   WORKER_ID=0 NUM_WORKERS=1 $JOB/code/pod/run_worker.sh --limit 1   # first real video
   WORKER_ID=0 NUM_WORKERS=1 $JOB/code/pod/run_worker.sh             # rest of the sample
   python $JOB/code/tools/merge.py --manifest $JOB/manifest.jsonl --out-root $JOB/outputs
   ```
4. Record the measured numbers:
   - source seconds per compute second
   - events-vs-predict split
   - peak VRAM and GPU utilisation from `gpu-worker-0.csv`
   - model load time

   Run one clip twice. The second pass hits `feature-cache/`, which isolates
   TRIBE-only time from V-JEPA time.
5. Pull results back: `rsync -av root@<pod>:/workspace/tribe-job/outputs/ ./results/<run>/`.
   Stop the pod.

## Phase 2: fan out to N workers

1. Rebuild the manifest with `--workers 4`. Assignment is greedy
   longest-first by duration, so shards finish together.
2. Start 3 more pods on the **same** network volume **and the same RunPod
   template/image** (the venv symlinks to the image's Python, so a different
   image breaks it). They share the venv and
   weights, so there is no reinstall. On each pod, run `apt-get install -y ffmpeg git`,
   then `WORKER_ID=k NUM_WORKERS=4 run_worker.sh`.
3. Merge. `merge.py` exits non-zero and lists any missing or failed videos. Re-run
   those workers; finished videos are skipped.

Network volumes are tied to a datacenter, so all pods must be in the volume's region.

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
- The March prototype's fsaverage5 → region mapping
  (`video_brain_analysis/src/tribe/brain_regions.py`) is a starting point for
  region-level summaries. Verify it against a real atlas before trusting it.

## Local development

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q       # offline: sharding, manifest, dry-run/resume/merge
.venv/bin/python pod/worker.py --dry-run ...   # stub model, no GPU
```
