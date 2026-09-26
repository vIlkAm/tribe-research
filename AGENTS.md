# AGENTS.md — tribe-research

Standalone research project: run Meta's TRIBE v2 over short-form videos to get
predicted cortical time series, then study them against performance metrics.

Collaborating (frontend teammate or their agent)? Read [`docs/TEAM.md`](docs/TEAM.md)
first: ownership, the analysis contract, and how it may change. Project
status (what's done, running, next) is in [`docs/STATUS.md`](docs/STATUS.md);
keep it current when you change the state of something it lists.

## Isolation contract (read first)

This project is deliberately **separate from Clipping Cartel production**. Until
the owner explicitly decides to connect it:

- No reads/writes to the self-host Supabase/Postgres, read mirror, Qdrant, or any
  `services/clipping-cartel` container, network, volume, or compose project.
  **One exception:** read-only file copies out of the backfill archive
  (`services/clipping-cartel/media_archive_worker/archive/backfill-20260926/`),
  via `tools/select_sample.py` (owner decision below).
- No Cartel credentials in `.env`; this project has its own RunPod/HF tokens.
- Nothing here binds a port on this host. Videos move to pods by rsync/`runpodctl`
  **from** this server, never by exposing an HTTP listener (the March prototype
  served videos unauthenticated on the public IP — don't repeat that).
- Performance metrics are imported as an exported file (CSV/JSONL) into
  `results/`, not queried live.
- **Owner decision, 2026-09-26 (later, supersedes the file-only rule below
  for these tables):** "I approve full access to the whole database of videos
  and their snapshots, that is public data". `tools/export_metrics.sh` may READ
  (read-only transaction, CSV into `results/metrics/`) deals, social_accounts,
  social_account_stat_snapshots, video_performances, video_snapshots, the
  cross-platform links and the media archive/backfill tables. Every clip file
  on this server (backfill **and** the regular archive under
  `media_archive_worker/archive/`) may go to RunPod; `tools/build_run_manifest.py`
  plans that run. Still no writes, no other tables, no live queries from pods.
  The owner also cleared a full-scale run ("hammer it with all the clips");
  pods still run only under `tools/runpod.py watchdog` with an explicit cap.
- **Owner decision, 2026-09-26:** the backfilled clips are company-owned
  ("those clips are in fact ours, we are owners of it … we are not
  commercialising a product"). They may be copied to RunPod pods for this
  non-commercial research. Still no DB queries: clip selection is file-based and
  metrics arrive later as an exported file keyed by the `source_name` (the
  `video_performances` id in the filename). Revisit the TRIBE CC-BY-NC licence
  before any of this feeds a product or client deliverable.
- Demo visuals are model predictions for an average subject. Keep the caption,
  and label anything built from dry-run data or a synthetic ROI map as such.
- RunPod pods are billed. Don't launch, resume, or submit to one without the
  owner's go-ahead for that run.
- TRIBE v2 is **CC-BY-NC-4.0**. Research use only; wiring it into a paid product
  or client deliverable needs a licence decision first.

## Layout

| Path | Purpose |
|---|---|
| `pod/setup.sh` | Idempotent env setup on the first pod (venv, pinned TRIBE, weights → network volume) |
| `pod/env.sh` | Cache/env vars; source on every pod shell |
| `pod/worker.py` | Load model once, process this worker's shard, resumable, atomic outputs |
| `pod/run_worker.sh` | Worker + `nvidia-smi` sampler, logs to `$JOB/logs/` |
| `pod/run_queue.sh` | One GPU, batches in order: N concurrent bf16 fast-video workers per batch (thread caps, expandable segments), one retry pass, `done-<b>` markers |
| `pod/launch_all.sh` | Phase 2: one worker per GPU on a single N-GPU pod, under `nohup` |
| `pod/downscale.sh` | Pod-side pre-scale to a 384 px short side (prefer `tools/prep_cpu.py` on the server) |
| `pod/bench_vjepa.py` | V-JEPA2 GPU floor: one timed 64-frame forward |
| `pod/fast_video.py` | Opt-in V-JEPA2 frame loop (`TRIBE_FAST_VIDEO=1`): stock's exact frames, each preprocessed once, CPU overlapped with GPU; fp32 inputs bitwise = stock |
| `pod/bench_fast_video.py` | Pod check: s/step, feeder wait vs GPU, feature drift per precision (fp32/tf32/bf16/fp16) |
| `pod/emb_export.py` | `<vid>.emb.npz`: pooled fusion-model inputs per extractor (emb_pool_v1, control arm E) |
| `pod/whisper_server.py` | Persistent whisperx per worker (pinned 3.8.6, stock TRIBE output), stock fallback |
| `tools/runpod.py` | RunPod API: prices, pod create/wait/stop, cost watchdog (`--max-usd`) |
| `tools/export_metrics.sh` | Read-only DB export of the approved tables → `results/metrics/` |
| `tools/build_outcomes.py` | Fair performance labels → `results/outcomes.parquet` (`docs/OUTCOMES.md`) |
| `tools/build_run_manifest.py` | Unique contents across posts/platforms + full-run plan |
| `tools/qc_files.py` | Decode/audio/resolution check per planned file |
| `tools/select_study_set.py` | Curated ~1,500 study set, lockbox, pilot, account deep-dive (`docs/STUDY_SET.md`) |
| `tools/make_batches.py` | Study set → ordered disjoint pod batches (pilot, frontend wiring set, 200s, deep-dive) |
| `tools/select_rest.py` | Stage 2: pinned lockbox extension + the remaining eligible contents as batches `rNN` |
| `tools/prep_cpu.py` | Pre-scale batch clips on this server (niced, load-gated, pinned ffmpeg) before `push-batch` |
| `tools/build_features.py` | Per-clip brain, extractor-embedding (control) and baseline feature table from pulled outputs |
| `tools/build_moments_pop.py` | Prereg family 6 (M1–M3): shot cuts (`shots`), frozen train-only norms + surrogate thresholds + FIR kernels with the cut-lag positive control (`fit`), `mpop_`/`edit_` clip table (`features`), good-vs-bad cluster-permutation contrast (`contrast`, needs its prereg row) |
| `tools/fit_models.py` | Baseline vs brain vs extractor-embedding control (A/B/E/BE; block-stacked ridge primary): grouped CV, leave-one-deal-out, niche tuning, sealed lockbox |
| `tools/predict.py` | One clip's worker output + deal/platform context → the `performance` block (featurizer state + `fit_models.py --save-model` dir; status gates, out-of-scope, lockbox-free references) |
| `tools/make_performance_samples.py` | Frontend fixtures for the proposed `performance` block: runs `predict.py` on `tests/test_predict.py`'s synthetic study → `docs/sample_analysis/performance/` (draft schema `docs/performance.schema.draft.json`) |
| `tools/power_check.py` | CPU power simulation (no TRIBE outputs, never reads lockbox labels) → `results/power/` |
| `tools/profile_account.py` | One deal's or account's brain profile vs the general model |
| `tools/handoff.py` | Validate bundles against the contract, index, tarball for the frontend (no footage) |
| `tools/drive_pod.sh` | Drive one pod through study batches shared via `claim-<b>` files (noclobber): optional setup, push, `run_queue.sh`, pull; 1-worker manifest rewrite for 24 GB cards |
| `tools/pod.sh` | Full-SSH push/pull/shell between this server and a pod (`push-batch`) |
| `tools/make_manifest.py` | Content-hash IDs, durations, duration-balanced worker assignment |
| `tools/compare_preds.py` | Preds agreement between two out-roots, with pre-set fp32/bf16 gates (fast-video checks) |
| `tools/check_emb.py` | `.emb.npz` gate: steps ≈ 2 × duration, no empty quarter on clips ≥ 20 s, quarters not identical |
| `tools/merge.py` | `index.jsonl` + `benchmark.json`, reports missing/failed by category |
| `tools/build_roi_map.py` | HCP-MMP1 → fsaverage5 ROI map artifact (build on the pod; figshare blocks this host) |
| `tools/brain_report.py` | ROI features, summary PNG, demo MP4 from pulled outputs (CPU) |
| `tribe_research/brain/` | ROI groups, feature extraction, headless surface renderer |
| `tribe_research/outcomes/` | Reach at fixed ages, baselines, shrunk engagement, rankings, cross-platform |
| `docs/PREREGISTRATION.md` | Primary endpoint (BE − E), stages, decision rules, frozen pipeline. Read before fitting real outputs |
| `docs/RUNPOD.md` | Spin up, connect, run, tear down |
| `tests/` | Offline tests (no GPU): sharding, manifest, dry-run + resume + merge, brain layer |

## Validate

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
```

## Prior art

`/home/tyler/projects/video_brain_analysis` (March 2026) was a FastAPI + React +
single-H200 prototype that stored region summaries only. Treat it as read-only
reference; its lessons are folded into `README.md`.
