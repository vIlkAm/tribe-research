# Project status

**Living file.** It is the first place to look for "how is the project going?",
from a person or an agent. The newest update is at the top. Details live in the
linked docs; this page only says where things stand. Agents on either side: when
you change the state of something below, update this file in the same commit.

_Last updated: 2026-09-26 21:50 UTC (research/backend side). All times UTC._

## One-paragraph summary

Paste a link or upload a clip. The backend runs Meta's TRIBE v2, which predicts
how an average viewer's cortex responds, second by second. The frontend shows
that as a timeline, a brain map and "moments". Next to it, a small model trained
on our own clips' real performance gives an in-context outlook: how this clip
likely ranks against similar clips in the same deal and platform. TRIBE is fixed
(Meta's weights). What we train is small: ridge models over brain features,
extractor embeddings and metadata. The brain-analysis half works on real clips
today. The performance half is being trained tonight, once enough clips have
gone through the GPU.

End product, gaps to a business and the demo plan: [`VISION_AND_DEMO.md`](VISION_AND_DEMO.md).

## Pipeline and where each part stands

| Step | Runs on | State |
|---|---|---|
| 1. Pick clips, pre-scale to 384 px | CPU (this server) | ✅ Study set (1,500 + 373 deep-dive) ready. The other ~6,000 are being pre-scaled now (`r00–r15`) |
| 2. TRIBE + extractors → brain response + embeddings per clip | GPU (RunPod) | ✅ Fast bf16 path passed every accuracy gate (L40S and A100). ⏳ Four pods share study batches b02–b09 through claim files: L40S (b02→), A100 (b09→b06), two RTX 4090s (b05, b06→). All of b02–b09 are expected by about 01:00 UTC |
| 3. Features per clip (`tools/build_features.py`) | CPU | ✅ Code done and tested |
| 4. Training + honest evaluation (`tools/fit_models.py`) | CPU, minutes | ✅ Code done, rules pre-registered ([`PREREGISTRATION.md`](PREREGISTRATION.md)). ✅ Preliminary fit on the 100 real clips of the complete batches (71 training contents, not validated, not stage 1). ⏳ Stage 1 waits for the study set |
| 5. Save the model, score one new clip (`tools/predict.py`) | CPU, milliseconds | ✅ `model-prelim-v0` saved and released privately (weights, featurizer, sha256 manifest); served only with `--allow-preliminary` ([`MODEL.md`](MODEL.md)) |
| 6. Bundles for the frontend (`tools/handoff.py`) | CPU | ✅ [`data-frontend40-v2`](https://github.com/vIlkAm/tribe-research/releases/tag/data-frontend40-v2): the same 40 bundles plus `performance.json` (preliminary) and per-clip `platform`/`video_link`/`is_lockbox` in `index.json`. ✅ 40 real bf16 bundles (the wiring set: shortest and longest clips, a >60 s clip, a near-silent one) in the private release [`data-frontend40`](https://github.com/vIlkAm/tribe-research/releases/tag/data-frontend40), 68 MB, no footage. Pilot bundles too |
| 7. Frontend | Browser | See "For the frontend" below |

## Numbers that matter

- **Throughput:** about 120 clips/h per GPU worker in bf16. That's roughly 5× the
  stock pipeline. An L40S runs 2 workers; a 24 GB card runs 1.
- **Data:** 7,942 eligible contents, 56.7 h of footage. The study set is 1,500
  clips (225 sealed lockbox). The stage-2 extension adds a second sealed lockbox
  of 909 clips.
- **Budget:** watchdog cap $17 (owner top-up), about $14 projected for the
  study set, keeping ≥ $1 for the demo. The full ~7.9k set needs about $50 more
  (owner decision).

## What "trained" will mean

- **Stage 1** is 5-fold cross-validation on the ~1,275 study training clips.
  - If brain + embeddings beat metadata alone by the pre-registered margin, the
    model ships as `research_preview`.
  - If not, it stays `not_trained`, and the UI shows the brain analysis without a
    performance number.
- **The neural claim** ("the brain response, not just the video embeddings,
  predicts performance") is tested separately. The UI may only say it when that
  test is `supported`.
- **Stage 2** means all ~7.9k clips, then opening the sealed lockboxes once. That
  step is what could make the model `validated`.

## For the frontend (what to build against)

1. **Brain analysis (real now):** contract `nvi.analysis.v0.2`
   ([`analysis.schema.json`](analysis.schema.json),
   [`analysis.types.ts`](analysis.types.ts)); fixture
   [`sample_analysis/`](sample_analysis/). Rules: [`TEAM.md`](TEAM.md).
2. **Performance card (proposed v0.3):** real bundles with `performance.json`,
   the model behind them, what each status shows and how updates arrive:
   [`MODEL.md`](MODEL.md). The `performance` block, its states
   (`not_trained` / `preliminary` / `research_preview` / `validated` / `out_of_scope`), the
   wording rules and the job states are in
   [`PRODUCT_PIPELINE.md`](PRODUCT_PIPELINE.md) §3–4.
   - Build the `not_trained` empty state and the `preliminary` card (grey badge,
     caption verbatim, "Not enough reference clips yet" when the percentile is
     null) first; `data-frontend40-v2` uses `preliminary` and `out_of_scope`.
   - Fixtures made by the real `predict.py` (synthetic test data, one per state) are in
     [`sample_analysis/performance/`](sample_analysis/performance/); its README maps each file
     to a UI state and lists the required labels. The draft schema is
     [`performance.schema.draft.json`](performance.schema.draft.json).
3. **Connecting for the hackathon demo:** use static, pre-computed bundles.
   - The backend produces `index.json` plus `<video_id>/analysis.json` and images
     via `tools/handoff.py`. The frontend loads them from its own static folder.
   - The paste-link / upload screen animates the job states
     (queued → fetching → extracting → predicting → done), then opens the
     matching pre-computed bundle.
   - There's no live API tonight. A real upload needs a hosted API plus a warm
     GPU (RunPod serverless or an always-on pod). That is an owner decision in
     `PRODUCT_PIPELINE.md` §6.
4. **Labels that must stay on screen:**
   - "TRIBE v2 prediction · average subject" on every brain visual;
   - "Illustrative" on any number not produced by a real model;
   - "Research preview · non-commercial (TRIBE CC-BY-NC)".

## Open decisions (owner)

- The ~$50 RunPod top-up for the full set.
- Live-serving host and precision.
- Link fetching from TikTok, Instagram or YouTube (ToS). Uploads of our own clips
  avoid the question.
- Licence before anything commercial.

## Log

- **2026-09-26 22:25 UTC:** b02 (200) and b09 (50) pulled, all clean. The A100 runs b08 at ~170 clips/h on 3 workers; b03, b05 and b06 are on the other pods, and b04/b07 are unclaimed. Pre-scaling of the remaining ~6,000 clips (r00–r15) is done. 4 clips are excluded by TRIBE's own transcript-alignment check (too little speech; 3 are adam-yu), which is 0.7 % and under the 2 % flag.

- **2026-09-26 21:50 UTC:** Preliminary performance model fit on the 100 real bf16 clips of the complete batches (frontend40, b00_pilot, b09; 71 training contents, lockbox unscored, not validated, not stage 1). Private releases [`model-prelim-v0`](https://github.com/vIlkAm/tribe-research/releases/tag/model-prelim-v0) (weights, sha256 manifest) and [`data-frontend40-v2`](https://github.com/vIlkAm/tribe-research/releases/tag/data-frontend40-v2) (the 40 bundles + `performance.json`, `platform`/`video_link`/`is_lockbox` in `index.json`). No percentile yet: fewer than 30 reference clips per deal × platform. Frontend guide: [`MODEL.md`](MODEL.md).
- **2026-09-26 21:30 UTC:** Four pods on the study set (about 470 clips/h combined). Prereg adds exploratory "good vs bad clip" contrast (M4) and an editing-covariates arm (X); moment detection parameters frozen before any outcome is read (900001b, 7aa63ee, d164f3c). The `preliminary` model status is in `predict.py` (6f02b13).

- **2026-09-26 21:05 UTC:** Performance-card fixtures from the real `predict.py` (`sample_analysis/performance/`, `tools/make_performance_samples.py`), draft schema `performance.schema.draft.json`, PRODUCT_PIPELINE §3 aligned to the tool output.
- **2026-09-26 21:00 UTC:** Real bundles for 40 clips published (`data-frontend40` release). The second pod (A100, `tribe-study2`) runs the back half of the study set (b09→b06); the L40S runs b02→b05. The two share b06–b08 through claim files. `tools/predict.py` and `fit_models --save-model` added (981ebca). `tools/fleet.py` added for the rest batches after a top-up (1bc99e3).

- **2026-09-26 20:55 UTC:** Study set queued on the L40S (`pod/run_queue.sh`, d8b921b), about 6.5 h, capped at $8 total. Outputs are pulled per batch to `results/runs/study-bf16/`. b09, and maybe b08, need the top-up.
- **2026-09-26 20:45 UTC:** bf16 fast frame loop passed all gates (pod code
  7163f18, prereg 9f3a4e0). Faster pod setup with parallel weight download and
  done markers (9c923be). Product handoff doc published (5c5857b). Stage-2
  lockbox extension drawn and pinned (b2f5dfb).
