# Performance model: weights, bundles and updates

For the frontend teammate and their agent. The short version: **you never run the
model.** We compute every performance number on our side and ship it per clip as
`performance.json` in a data release. The model weights are a separate release
you don't need to download.

Research only: TRIBE v2 is CC-BY-NC-4.0. No footage or audio is in any release.
Both releases are in the private repo `vIlkAm/tribe-research`.

## What exists now

| Release | What | You need it? |
|---|---|---|
| [`data-frontend40-v2`](https://github.com/vIlkAm/tribe-research/releases/tag/data-frontend40-v2) | The 40 real wiring-set bundles. `analysis.json` is byte-identical to `data-frontend40`, plus `performance.json` per clip and a richer `index.json` | **Yes**: build against this |
| [`model-prelim-v0`](https://github.com/vIlkAm/tribe-research/releases/tag/model-prelim-v0) | Model version `perf-stack-BE_v1+20260926.5b7e4b6`: ridge models (the pre-registered block stack, BE: brain + extractor embeddings + metadata), PCA/featurizer state, reference tables, ROI map, `RELEASE_MANIFEST.json` (sha256 of every file) | No. Backend only. Its reference tables hold training clips' observed percentiles and account ids, so keep it internal |
| `data-frontend40` | The v0.2 bundles without performance | Still valid; unchanged |

`model-prelim-v0` is **preliminary**. It is a first fit on the real clips that had
landed (see "Training summary"). It is not validated, it is not the pre-registered
stage-1 analysis, and it says nothing about that analysis. Its numbers show the
format; they are not forecasts.

## Getting the bundles

```bash
gh release download data-frontend40-v2 -R vIlkAm/tribe-research -p 'frontend40_v2.tar.gz'
tar xzf frontend40_v2.tar.gz   # -> frontend40_v2/{index.json, <video_id>/..., _static/}
```

Per clip: `analysis.json` (contract `nvi.analysis.v0.2`), `performance.json`
(draft schema [`performance.schema.draft.json`](performance.schema.draft.json),
`$id` `nvi.analysis.v0.3-draft.performance`), `brain_proxy.jpg`,
`brain_vertex.jpg`, `summary.png`. `performance.json` holds the block that
v0.3 will put at `analysis.json` → `performance`; until v0.3 it sits next to it.

`index.json` gains, at the top: `performance_schema`, `model_version`,
`model_release` (`model-prelim-v0`), `release` (`data-frontend40-v2`) and
`performance_status_counts`. Per clip it gains:

| Field | Meaning |
|---|---|
| `platform` | The clip's own post: `youtube` (29) or `instagram` (11) |
| `video_link` | Public URL of that post (fine to link, also for lockbox clips) |
| `is_lockbox` | `true` for 3 clips: sealed test clips. **Never show an observed outcome for them.** None is shipped anywhere |
| `performance_path` | `<video_id>/performance.json` |
| `performance` | `{model_status, validated, clip_in_training}` for list views. Never a number |

"Own post": a content can be posted on several platforms. Each bundle uses the post
its analysis was made from (`analysis.json` → `source_name`, which is the content's
representative post in our members table). Its deal, platform and link come from that post.

## Which bundles belong to which model

One data release = one model. `index.json` → `model_version` equals every clip's
`performance.json` → `model_version` (the handoff refuses a mix), and
`model_release` names the weights release. `provenance` in each block also pins the
featurizer and ROI-map sha256. If you cache bundles, key the cache by `model_version`.

## Status → what the UI shows

| `model_status` | In v2 | Show |
|---|---|---|
| `preliminary` | 35 clips | A grey **"Preliminary"** badge and the `caption` verbatim, next to any number: "Preliminary model — trained on 71 clips, not validated. Illustrative of the format, not a forecast." No validation ρ, no brain claim. Never the words "forecast" or "prediction of success", and never imply it passed a test |
| `out_of_scope` | 5 clips | A card with `reason` (here: no training clips for that deal on that platform, or a training clip without an out-of-fold prediction). No numbers |
| `not_trained` | 0 | "Performance model not trained yet". No numbers |
| `research_preview` | 0 | Orange **"Research preview"** badge, numbers in a card |
| `validated` | 0 | Blue **"Validated"** badge |

What you will actually see in v2:

- **No percentile yet.** A percentile needs at least 30 reference clips in the
  same deal × platform; the most any has today is 15. So `percentile_deal_platform`
  and `likely_range` are `null` on every clip, with the warning "Fewer than 30
  reference clips…". Show **"Not enough reference clips yet"** plus `reference_n`,
  never a placeholder number. This fills in as more batches are trained; the
  synthetic fixtures in [`sample_analysis/performance/`](sample_analysis/performance/)
  show the populated card.
- **Training clips** (`clip_in_training: "train_oof"`, 33 scored): ranked from
  their out-of-fold prediction, `retrospective: true`, `drivers: []`. Hide the
  driver bars and show the "drivers omitted" warning.
- **Lockbox clips** (`clip_in_training: "lockbox"`, `is_lockbox: true`): 2 scored
  with drivers, 1 out of scope. Show the prediction; there is no observed outcome to show.
- `reach` is `null` under `preliminary`. `confidence` is always `"low"`: use the
  muted card.

All wording rules (context, range, badges, banned words) are in
[`sample_analysis/performance/README.md`](sample_analysis/performance/README.md)
and [`PRODUCT_PIPELINE.md`](PRODUCT_PIPELINE.md) §3–4.

## How a model update arrives

1. We fit a new model and publish its weights as a new tag (`model-prelim-v1`,
   later `model-stage1-v1` …) with `tools/publish_model.sh`.
2. We recompute every bundle's `performance.json` with it and publish a new data
   release (`data-frontend40-v3`, or a bigger set) whose `index.json` names the new
   `model_version` and `model_release`.
3. You download the new data release and swap the folder. Old releases stay up,
   so you can pin one. You never download weights, and nothing changes in
   `analysis.json`.

Only the status, the numbers and the badges change between models; the block's shape
is fixed by the draft schema until v0.3 lands in the contract (announced per
[`TEAM.md`](TEAM.md)).

## What changes when stage 1 lands

Stage 1 is the pre-registered analysis on the full study set
([`PREREGISTRATION.md`](PREREGISTRATION.md)).

- If its GO rule passes: bundles move to `research_preview` (orange badge). The
  preliminary caption goes away, `validation` gets `scheme: "content_cv"` with a
  ρ and CI, `reach` appears, and `brain_claim` may become `directional`,
  `supported` or `not_supported`.
- If it does not: bundles are `not_trained` (no numbers), or still `preliminary`
  with the caption when we choose to ship the format.
- `validated` (blue) only comes after stage 2 opens the lockbox.

Build all three badges now; the fixtures cover each state.

## Training summary (model-prelim-v0)

**n = 71 contents, preliminary, not a result.** Don't quote these numbers.

- Real bf16 fast-loop TRIBE outputs only (tribe `af58661`, `emb_pool_v1`): 100
  clips from `results/runs/frontend40-bf16/outputs` (40),
  `results/runs/study-bf16/outputs-b00_pilot` (10) and `outputs-b09` (50). Only
  complete batches (`done-<b>` marker). No stock fp32 outputs are mixed in.
- 13 of the 100 are lockbox clips: projected but not fit by PCA, not trained on,
  not in any reference table, not scored. `--score-lockbox` was not used.
- Pre-registered defaults, nothing tuned: primary target `log_interactions_rate`
  (71 contents, 156 posts, 12 deals, 32 deal × platform strata), secondary
  `reach_rel_local` (86 contents).
- Content-grouped CV, within-stratum Spearman of the served model: 0.56
  (95% CI 0.06 to 0.78), n = 71. With this few clips per stratum, that interval
  is too wide to mean anything. The GO gate that `predict.py` applies is not met at
  this size, which is why the model is served only with `--allow-preliminary`.
  That is not the stage-1 decision.
- Everything else (every contrast, per scheme) is in the release's
  `model/manifest.json` → `metrics`, labelled preliminary in `RELEASE_MANIFEST.json`.

## Publishing (backend)

```bash
# 1. features over the complete batches, lockbox kept out of the PCA fit
.venv/bin/python tools/build_features.py --out-root results/runs/frontend40-bf16/outputs \
    --out-root results/runs/study-bf16/outputs-<b> ... --exclude-from-pca-fit results/study/selection.csv \
    --out results/features/<name>/features.parquet
# 2. fit with the pre-registered defaults on a clean, committed tree (no --score-lockbox)
.venv/bin/python tools/fit_models.py --features results/features/<name>/features.parquet \
    --members results/run_full/members.csv --outcomes results/outcomes.parquet \
    --selection results/study/selection.csv --out-dir results/models/<name>-eval --save-model results/models/<name>
# 3. weights release (checks gh auth, private repo, new tag, no media, lockbox-free references)
tools/publish_model.sh model-<name> results/models/<name> results/features/<name>/features.featurizer.json
# 4. performance.json per bundle, server-side (--allow-preliminary only while the GO gate isn't met)
.venv/bin/python tools/bundle_performance.py --analyses results/handoff/frontend40/analyses \
    --out-root <each root> --featurizer results/features/<name>/features.featurizer.json \
    --model-dir results/models/<name> --dest results/handoff/<data>/analyses --allow-preliminary
# 5. validated data tarball, then its private release
.venv/bin/python tools/handoff.py --analyses results/handoff/<data>/analyses --expect-real \
    --clip-meta results/handoff/<data>/clip_meta.json --require-performance \
    --release data-<data> --model-release model-<name> --out results/handoff/<data>.tar.gz
gh release create data-<data> results/handoff/<data>.tar.gz --title ... --notes ...
```

`publish_model.sh … --dry-run` stops before `gh`. Then update this file and `STATUS.md`.
