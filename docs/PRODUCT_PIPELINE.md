# Product pipeline: from a link to a prediction

For the frontend teammate and their agent (2026-09-26): what is fixed and what
is learned, how a clip becomes an analysis, what the UI may show about
performance, and what the owner still has to decide.

**Status:** the brain-analysis half runs today (batch, owner-launched pods). The
performance half has one **preliminary** model (`model-prelim-v0`, 71 training
contents, not validated, not the stage-1 analysis); its numbers are computed on
our side and ship per clip as `performance.json` in `data-frontend40-v2`
([`MODEL.md`](MODEL.md)). There is no inference service. "Inference" below is a design.
Contract: `nvi.analysis.v0.2` (`analysis.schema.json` wins); change rules
[`TEAM.md`](TEAM.md); labels [`OUTCOMES.md`](OUTCOMES.md); decision rules
[`PREREGISTRATION.md`](PREREGISTRATION.md).

## 1. What we're building, in one picture

```text
TRAINING (offline, batch, owner-launched pods; ~1,500 study clips, then all ~7.9k eligible)
  clips ─► extractors (V-JEPA2, w2v-BERT, Llama-3.2) ─► TRIBE v2 (FIXED) ─► cortex [T × 20,484]
        ─► tools/build_features.py: base_* metadata, brain_*, emb_* (extractor PCA, control arm)
  + outcomes.parquet (engagement, reach labels)
        ─► tools/fit_models.py: stack model, grouped CV ─► MODEL ARTEFACT (to be built, see below)

INFERENCE (one new clip)
  link / upload ─► fetch + validate ─► same prep ─► same TRIBE ─► same features
      ─► saved model artefact ─► performance block ─┐
      ─► brain_report (proxies, moments, sprites) ──┴─► analysis.json ─► frontend
```

### Fixed vs learned

| Part | Learned by us? | What it is |
|---|---|---|
| V-JEPA2, w2v-BERT, Llama-3.2-3B, whisperx | No | Pretrained extractors TRIBE reads |
| TRIBE v2 | No | Meta's fixed mapping to average-subject cortex (fsaverage5, ~1 step/s) |
| Proxy channels, moments, ROI map | No (hand-defined) | `proxies_v0.yaml`, `moments.py`, HCP-MMP1 map. Rules, not fits |
| Brain PCA, extractor-embedding PCAs | Yes (unsupervised) | Fit once on non-lockbox clips, no labels |
| Performance predictor | Yes (small) | `stack`: one ridge score per wide block (`brain_*`, `emb_*`), then a ridge over metadata + those scores |
| Reference tables | Yes (derived) | Per deal × platform and per account: the account-level target encoding, and out-of-fold predictions and labels to rank a new clip against |

"The learned weights" means the last three rows: a few hundred ridge
coefficients, the PCA components and the reference tables. Megabytes, CPU-only,
milliseconds per clip. The GPU cost is entirely in the fixed extractors and TRIBE.

### Missing before any inference can run

1. **A saved model artefact.** `fit_models.py` writes metrics and
   `oof_predictions.csv` but persists no fitted model. `build_features.py`
   saves the brain PCA (`.pca.npz`) but not the emb PCAs, imputation medians or
   column order. A new step must save, versioned and hashed: the stack model
   and its block ridges, all PCA components, medians and column list, the
   account/deal encoders and the reference tables.
2. **A predict tool** that takes one clip's worker output plus a context and
   returns the `performance` block (section 3).
3. **Train/serve parity (hard requirement).** Inference must reproduce the
   training run exactly, or the skew is silent: TRIBE commit and config, the
   384 px pre-scale (`tools/prep_cpu.py`), pod code commit and **precision**
   (fp32 vs the bf16 fast frame loop, whichever the training features used; the
   throughput below assumes bf16 passes the pre-set `compare_preds` gate),
   `features_version`, `proxies_version`, ROI
   map and emb export version. All of them go into the block's provenance.

**Implemented (offline, synthetic-tested; not yet run on real outputs):**
items 1 and 2, and the parity checks the worker output records.
`build_features.py` now writes `<stem>.featurizer.npz/.json` (every PCA,
column order, versions, sha256s of the table, ROI map and proxies, and the
training runtime mix) and exposes `featurize_one`. Imputation medians and
encoders live in the fitted pipeline, so they are saved with the model.
`fit_models.py --save-model DIR`, run after the evaluation, refits the served
model (BE `stack` by default) on all non-lockbox train rows. It writes the
joblib file, the manifest (metrics, hashes, git) and the lockbox-free
reference tables. `tools/predict.py` returns the `performance` block. Pre-scale
and pod code aren't in the worker json yet, so they are provenance from the
request context only and are not checked. Build the features with
`--exclude-from-pca-fit`; otherwise the PCAs saw the lockbox and predict
warns.

### Inference needs context

The metadata block (set A) includes deal, platform, log duration, aspect, audio
level, follower bucket, posting weekday and the account's history term. A
pasted link or a bare upload carries none of the business context. The request
must name at least **deal and platform**, and optionally the **posting
account**. Without an account, the account term falls back to the deal ×
platform level. Every number is therefore "within this deal on this platform",
never a universal score.

## 2. Inference request flow

```text
browser ──POST /v1/jobs {link | upload, deal_id, platform, account_id?}──► API
API ─► validate ─► queue ─► GPU worker (warm, model loaded)
     ─► TRIBE + emb export ─► CPU: features, predict, brain_report ─► store bundle
browser ◄── GET /v1/jobs/{id} (poll) ◄── state; on done ─► GET /v1/analyses/{analysis_id}
```

**Link fetching happens on the backend, never in the browser.** It needs
platform downloaders, H.264 transcoding and rate limits, and raw media URLs must
not reach clients. Fetching third-party platform content has platform ToS and
copyright implications. The owner must decide on that before anything public
(section 6). Uploads avoid the question for the uploader's own clips. TRIBE v2
is CC-BY-NC-4.0: the whole flow is research/demo only until a licence decision.

### Validation (before a GPU second is spent)

| Check | Rule | On failure |
|---|---|---|
| Container | `.mp4 .avi .mkv .mov .webm` | `unsupported_format` |
| Decode | ffmpeg decodes video; TikTok's BVC2 codec does not decode, so fetch or transcode to H.264 | `undecodable` |
| Duration | 5–90 s (training scope) | `duration_out_of_range` |
| Audio | audio track present | brain analysis runs; performance becomes `out_of_scope` (the model never saw silent clips) |
| Language | whisperx guess on the first 30 s; TRIBE transcribes everything as English | warning; performance is `low` confidence or `out_of_scope` |
| Context | deal and platform known to the model | brain analysis runs; performance `out_of_scope` |

### Job states

A separate job resource, not part of `analysis.json`; `analysis.status` is unchanged.

| Job state | Meaning | `analysis.status` |
|---|---|---|
| `queued` | accepted, waiting for a worker | `queued` |
| `fetching` | downloading/transcoding the link, or receiving the upload, then validating | `processing` |
| `extracting` | GPU: transcription, extractors, TRIBE | `processing` |
| `predicting` | CPU: features, model, brain_report sprites | `processing` |
| `done` | bundle stored | `complete` |
| `failed` | with `error_code` from the validation table, or `worker_error` / `timeout` | `failed` |

### Latency (per clip)

| Step | Warm worker | Cold (no pod running) |
|---|---|---|
| Pod start + ~20 GB weights + envs | none | ~25 min today (being cut) |
| Fetch + validate + 384 px prep | seconds (to measure) | same |
| TRIBE, fast bf16 loop (valid only if bf16 passes the `compare_preds` gate) | ~25–30 s for a ~26 s clip (≈ 1× realtime; a 90 s clip scales to ~1.5 min, not measured) | same once warm |
| TRIBE, stock | ~0.2× realtime (≈ 2 min for a 26 s clip) | same |
| Features + model | < 1 s (CPU) | same |
| `brain_report` sprites/PNG | **to measure** (CPU) | same |

Only a warm worker makes a "paste a link and wait" UX plausible. Cold start
needs a "we'll notify you" UX or a pre-warmed pool.

### Cost

| Item | 4090 community ($0.34/h) | L40S ($0.79–1.09/h) |
|---|---|---|
| Throughput (fast bf16 loop, estimate being measured) | ~125–140 clips/h | ~125–140 clips/h |
| GPU per clip, fully utilised | ≈ $0.003 | ≈ $0.006–0.009 |
| One cold start (~25 min) | ≈ $0.14 | ≈ $0.33–0.45 |
| **Keeping one worker warm and idle** | **≈ $8/day** | **≈ $19–26/day** |

Per-clip cost is negligible; idle time dominates, so the real choice is an
always-warm pod vs serverless workers with cold starts (section 6).

## 3. Proposed contract extension: `performance` (for `nvi.analysis.v0.3`)

**Proposal only.** Nothing here is in the schema. Under TEAM.md it lands as one
PR updating schema, types, sample bundle, spec §07, tests and changelog. It is
additive, so it's a minor bump to `nvi.analysis.v0.3`.

Build against the real tool output. Fixtures made by running `tools/predict.py`
on synthetic test data are in
[`sample_analysis/performance/`](sample_analysis/performance/), one per UI state.
[`performance.schema.draft.json`](performance.schema.draft.json) is the draft
schema for exactly the block `predict.py` emits.
`tests/test_performance_samples.py` validates the fixtures against it. The
example and TypeScript below follow the tool's output.

**Why a new block and not `predictions`.** `predictions` is reserved for
*behaviour* metrics: 3 s/5 s hold, completion, drop zones (spec §05). Those need
watch-time/retention data, which doesn't exist, so `predictions` stays
`not_available` for the foreseeable future and TEAM.md's rule still applies to
it. `performance` is a different thing: a **relative ranking of expected
engagement within a named deal × platform**. It has its own status gate. Naming
it `prediction` next to `predictions` would invite bugs.

### Rules

- Optional top-level key. Absent means "not computed". Old v0.2 frontends
  ignore it.
- Headline is engagement (`log_interactions_rate`, the shrunk interactions per
  view). Reach (`reach_rel_local`) is optional and carries a stronger caveat,
  because on one platform only ~29 % of reach is clip-driven against ~59 % of
  engagement.
- `model_status` follows the pre-registered stages:

  | `model_status` | Meaning | Numbers? |
  |---|---|---|
  | `not_trained` | no model exists, or none passed the stage-1 GO rule | **none**: `engagement`/`reach` null, `drivers` empty (schema `if/then`, like `predictions`) |
  | `preliminary` | **opt-in only** (`predict.py --allow-preliminary`): a saved model that did *not* pass the GO rule, scored so the frontend has real-format numbers now. Says nothing about stage 1 | yes, with `validated: false`, `n_train`, a mandatory `caption`, a grey "Preliminary" badge; `brain_claim` is `not_tested`, `validation.scheme` is `none` (no ρ), `confidence` `low`, `reach` null |
  | `research_preview` | stage-1 CV passed the pre-registered BE − A GO rule (content features beat metadata) | yes, with a visible "Research preview" badge |
  | `validated` | the served model's own within-stratum ρ on the sealed lockbox has a CI lower bound > 0 (independent of `brain_claim`) | yes; still relative, still not a guarantee |
  | `out_of_scope` | model exists, clip/context outside it (no audio, unknown deal, >90 s) | none; show `reason` |

- `validated` (boolean) mirrors the status; `n_train` is the engagement model's
  training contents on scored blocks (null otherwise); `caption` is set only on
  `preliminary` and must be shown verbatim next to its numbers. Without the
  opt-in, the same model yields `not_trained`; a model that passes GO is never
  `preliminary`. See [`MODEL.md`](MODEL.md) for how the shipped weights and
  bundles relate.
- `brain_claim` is separate. The model can be useful while the brain mapping
  adds nothing beyond its own inputs (BE − E). Never say "the brain response
  predicts" unless it is `supported`.
- **Percentile:** the clip's predicted score ranked among the reference clips'
  **out-of-fold predictions** in the same deal × platform, which is what
  within-stratum Spearman validates. Not against observed labels: predictions
  are shrunk, so that would pile everything near P50. The reference set is the
  `scheme == content` rows of `oof_predictions.csv`, **deduplicated by content**
  (it is post-level) and **weighted by 1 / `incl_prob`**: the study set
  oversamples the tails, so an unweighted rank is distorted. Null below 30
  reference contents; account level only for accounts with ≥ 40 training posts
  (`fit_models.py` defaults).
- **Likely range:** weighted P10–P90 of the *observed* engagement percentiles of
  reference contents predicted close to this one. At a realistic within-deal ρ
  of 0.1–0.3 it is wide, often most of the scale. Design for that; don't hide it.
- **Drivers:** the clip's final-ridge terms grouped into `metadata` (platform,
  duration, aspect, audio, posting time), `account_history`,
  `content_embedding` (extractor block score) and `brain_response` (brain block
  score). Signed, relative to the reference average, for bar lengths only;
  never print them. Model attributions, not causes. Optional `brain_detail`
  splits the brain term by channel (the block ridge is linear over
  `brain_<channel>_*`; PCA and cross-channel columns go to `"other"`), so the UI
  can link a driver to a timeline lane.
- **Training membership:** a training clip gets its out-of-fold prediction
  (`train_oof`), never an in-sample one. `lockbox` means either sealed lockbox:
  `split == lockbox` in `results/study/selection.csv` (225) or the stage-2
  extension `results/study/lockbox_ext.csv` (909); those never show observed
  outcomes. A link to an already-posted clip is `retrospective`; say so.

### JSON example (illustrative numbers, `research_preview`)

```json
"performance": {
  "model_status": "research_preview", "validated": false, "reason": null, "brain_claim": "directional",
  "model_version": "perf-stack-BE_v1+20260930.ab12cd3", "n_train": 1275, "caption": null,
  "context": { "deal_id": "d_123", "deal_label": "Deal A", "platform": "tiktok",
               "account_id": null, "account_level": false },
  "clip_in_training": "no", "retrospective": false,
  "engagement": {
    "target": "log_interactions_rate",
    "percentile_deal_platform": 0.72, "likely_range": [0.31, 0.93], "reference_n": 164,
    "percentile_account": null, "account_reference_n": null, "confidence": "low",
    "validation": { "scheme": "content_cv", "within_stratum_spearman": 0.21, "ci95": [0.14, 0.28] }
  },
  "reach": null,
  "drivers": [
    { "family": "content_embedding", "label": "Video, audio and text content", "contribution": 0.08 },
    { "family": "brain_response", "label": "Predicted brain response", "contribution": 0.03,
      "brain_detail": [ { "channel": "social", "contribution": 0.02 }, { "channel": "other", "contribution": 0.01 } ] },
    { "family": "metadata", "label": "Duration, format, posting time", "contribution": -0.02 },
    { "family": "account_history", "label": "Deal/account typical level", "contribution": 0.0 }
  ],
  "warnings": ["Low confidence: this deal has few reference clips on this platform."],
  "provenance": { "tribe_commit": "af58661", "pod_code": "ba2d55e", "precision": "bf16",
                  "fast_video": true, "emb_export": "emb_pool_v1", "prescale": "s384",
                  "features_version": "clip_features_v1", "emb_version": "emb_features_v1",
                  "video_id": "cd20b16879d630c4", "model_version": "perf-stack-BE_v1+20260930.ab12cd3",
                  "model_git": { "commit": "ab12cd3…", "dirty": false },
                  "training_features_sha256": "…", "proxies_version": "proxies_v0",
                  "roi_map_sha256": "…", "featurizer_npz_sha256": "…", "artefact_sha256": "…" }
}
```

### TypeScript (proposal, to be merged into `analysis.types.ts` by the v0.3 PR)

```ts
export type ModelStatus = "not_trained" | "preliminary" | "research_preview" | "validated" | "out_of_scope";
export type BrainClaim = "not_tested" | "directional" | "supported" | "not_supported";
export type DriverFamily = "metadata" | "account_history" | "content_embedding" | "brain_response";

export type Platform = "tiktok" | "instagram" | "youtube";

/** Always present. deal_id/deal_label/platform can be null (platform even an unsupported string) only
 *  when there are no numbers; scored blocks always have a deal and one of the three platforms. */
export interface PerformanceContext {
  deal_id: string | null; deal_label: string | null; platform: Platform | string | null;
  account_id: string | null;
  /** true iff engagement.percentile_account is set. */
  account_level: boolean;
}

export interface TargetPrediction {
  target: "log_interactions_rate" | "reach_rel_local";
  /** 0..1 weighted (1/incl_prob) mid-rank among reference contents' out-of-fold predictions; null if reference_n < 30. */
  percentile_deal_platform: number | null;
  /** [p10, p90] of observed percentiles for clips predicted like this one. Expect it wide. */
  likely_range: [number, number] | null;
  reference_n: number;
  /** Only for accounts with >= 40 training posts. */
  percentile_account: number | null; account_reference_n: number | null;
  /** "medium" only when validated with reference_n >= 100. Not the analysis Confidence enum. */
  confidence: "low" | "medium";
  /** scheme is "lockbox" for engagement when validated, "none" (null ρ and CI) when preliminary, else
   *  "content_cv". Null when not estimated. */
  validation: { scheme: "content_cv" | "lockbox" | "none"; within_stratum_spearman: number | null;
                ci95: [number | null, number | null] };
}

/** contribution: signed, relative to the reference average. Bar length only; never print it. */
export interface Driver {
  family: DriverFamily; label: string; contribution: number;
  /** brain_response only: channel = channels[].key or "other" (PCA, cross-channel). */
  brain_detail?: { channel: string; contribution: number }[];
}

/** For "See evidence". Model keys appear once a model is loaded, featurizer keys once the clip is
 *  featurised, artefact_sha256 only when scored. */
export interface PerformanceProvenance {
  tribe_commit: string | null; pod_code: string | null; precision: string | null;
  fast_video: boolean | null; emb_export: string | null; prescale: string | null;
  features_version: string; emb_version: string; video_id: string;
  model_version?: string; model_git?: { commit: string | null; dirty: boolean | null };
  training_features_sha256?: string | null; proxies_version?: string; roi_map_sha256?: string | null;
  featurizer_npz_sha256?: string; artefact_sha256?: string;
}

interface PerformanceBase {
  brain_claim: BrainClaim;
  context: PerformanceContext;
  /** train_oof and lockbox are always retrospective; train_oof always has drivers: []. */
  clip_in_training: "no" | "train_oof" | "lockbox"; retrospective: boolean;
  warnings: string[]; provenance: PerformanceProvenance;
}

export type Performance =
  | (PerformanceBase & { model_status: "not_trained"; validated: false; reason: string;
      model_version: string | null; n_train: null; caption: null;
      engagement: null; reach: null; drivers: [] })
  | (PerformanceBase & { model_status: "out_of_scope"; validated: false; reason: string; model_version: string;
      n_train: null; caption: null; engagement: null; reach: null; drivers: [] })
  /** Opt-in, GO not passed: show `caption` verbatim + a "Preliminary" badge; brain_claim is "not_tested". */
  | (PerformanceBase & { model_status: "preliminary"; validated: false; reason: null; model_version: string;
      n_train: number; caption: string; engagement: TargetPrediction; reach: null; drivers: Driver[] })
  | (PerformanceBase & { model_status: "research_preview" | "validated"; validated: boolean; reason: null;
      model_version: string; n_train: number; caption: null;
      engagement: TargetPrediction; reach: TargetPrediction | null; drivers: Driver[] });

// Analysis (v0.3): schema_version "nvi.analysis.v0.3"; performance?: Performance;
```

## 4. Presentation guidance

### Wording

| Say | Don't say |
|---|---|
| "Predicted engagement: ranks around P72 among Deal A's TikTok clips (164 clips)" | "Virality score 72", "72 % chance to go viral" |
| "Similar clips landed between P31 and P93" | a single number without its range |
| "Research preview · correlational · not a guarantee" | "will perform", "guaranteed", "optimised" |
| `preliminary`: the `caption` verbatim ("Preliminary model — trained on N clips, not validated. Illustrative of the format, not a forecast.") + "Preliminary" badge | "forecast", "prediction of success", "Research preview", any hint that the model passed the pre-registered test |
| "Model prediction for an average viewer" (brain) | "your audience's brain", "brain firing" |
| "Content features and predicted brain response contributed" | "the brain response caused this" |
| "Untested edit hypothesis" (moments) | "fix this drop-off" (no retention data exists) |

- Every performance number carries three things: the context (deal ×
  platform, n), the range, and the status badge.
- Spec §05's rule stands: no single headline score until the model is
  calibrated on held-out posts (`validated`). Until then the percentile sits in
  a card, not a hero number. Reach, if shown, adds "mostly driven by account,
  platform and timing".

### States

| `performance` | UI |
|---|---|
| absent / `not_trained` | Brain analysis only. Card: "Performance model not trained yet". No numbers, no skeleton digits |
| `out_of_scope` | Card with `reason` (e.g. "No audio track: outside the model's training scope") |
| `preliminary` | Percentile + range + context + drivers in a muted card, a grey **"Preliminary"** badge and the `caption` next to the numbers, always. No validation ρ, no brain claim. Null percentile (under 30 reference clips) → "Not enough reference clips yet" |
| `research_preview` / `validated` | Percentile + range + context + drivers; orange "Research preview" / blue "Validated" badge. Always relative, always with a range |
| `confidence: low` or `warnings` | Muted card with warning icon (spec §09 low-confidence/OOD state) |
| `clip_in_training: lockbox` | No observed outcome shown, ever |

### v0.2 fields → UI panels (spec §02)

| Panel | Fields |
|---|---|
| Top bar | `model_versions`, `research_only`, `synthetic` (SYNTHETIC banner), `quality.warnings` |
| Video player | `duration_ms`, `timing` (stimulus time, `display_hz`) |
| Outcome strip | `predictions` (empty state in v0.2); proposed `performance` card |
| Timeline | `channels[]` (`default_visible`), `unavailable_channels`, `events` (shots, words, speech), `timing.gaps_ms`, `timing.onset_window_ms` |
| Brain map | `assets.brain_map` sprite, `assets.region_map` hover, `research_vertex_map` (Research mode only) |
| Moment insight | `moments[]`: observations and untested hypotheses, `relative_to: this_clip` |
| "See evidence" | `provenance`, `channels[].basis`, `moments[].evidence` |

## 5. Demo plan

The only real model is preliminary, so a demo filmed soon shows the brain
analysis with either `not_trained` or the `preliminary` card (grey badge plus its
caption, and "Not enough reference clips yet" where the percentile is null;
[`MODEL.md`](MODEL.md)). Any performance card on screen is a mock
labelled **"Illustrative — not model output"**. No real clip gets an invented
number.

### Reliable path: pre-computed bundles

| Source | What | Notes |
|---|---|---|
| `docs/sample_analysis/` | 1 synthetic bundle | Wiring only; SYNTHETIC banner must show |
| `results/handoff/pilot9_real.tar.gz` | 9 real bundles (pilot) | All train split |
| `b01_frontend` batch | 40 real clips (running tonight) | Wiring set: shortest, longest, >60 s, near-silent. Good for edge cases, less so as hero clips. **3 of the 40 are lockbox clips**: never show their observed outcomes |
| Study set / full run | ~1,500 then ~7.9k | Pick hero clips here once available: English, clear speech, in neither `selection.csv` lockbox nor `lockbox_ext.csv` |

Once a `research_preview` model exists, a "predicted vs actual" slide is allowed
only on train clips with their **out-of-fold** prediction, never on either lockbox.
Bundles ship via `tools/handoff.py` without footage. The player needs the clip
from the owner's server; the clips are company-owned.

### Live path, only if a warm worker exists

Needs an owner-approved, billed pod already warm and a pre-validated clip (5–90
s, English, audio); ~30 s for a 26 s clip, cut in the edit, pre-computed bundle
as fallback. No API/queue exists, so "live" means an operator-run worker.

### Shot list (pre-filmed)

1. Paste/upload a company-owned clip; job states tick `queued → fetching →
   extracting → predicting → done` (pre-recorded or mocked).
2. Player + timeline: scrub; channels and brain tile follow the playhead. Hold
   on the caption "TRIBE v2 prediction · average subject".
3. Brain map hover: region tooltip with its within-clip z.
4. Moments: one observation, one dashed "untested edit hypothesis".
5. Outcome strip: `not_trained` empty state, or a mock card stamped "Illustrative".
6. A second clip from another deal: show the context switch; never compare z
   values across clips.
7. Closing card: "Research preview · non-commercial (TRIBE CC-BY-NC) ·
   predictions, not measurements".

**Label on screen:** SYNTHETIC (dry-run/sample data) · "model prediction for an
average subject" (every brain visual) · "Illustrative" (any number not from a
real model) · "Research preview" (any real performance number) · "Untested
hypothesis" (moments). Filming `demo.mp4` or showing clips publicly also touches
the licence and platform-content questions below.

## 6. Open decisions for the owner

| Decision | Why it blocks |
|---|---|
| **Licence.** TRIBE v2 is CC-BY-NC-4.0; the extractors have their own terms (Llama 3.2 is gated under Meta's licence) | Research/demo only until decided; blocks any paid product or client deliverable |
| **Link fetching** of TikTok/Instagram/YouTube content | Platform ToS and copyright; uploads of own clips avoid it |
| **GPU hosting.** RunPod always-warm pod (~$8–26/day idle) vs serverless (cheap idle, minutes of cold start) vs on-demand batch | Decides the UX (instant vs notify-later) and needs an API/queue host outside this server, since the isolation contract forbids listeners here |
| **Serving precision.** fp32 vs bf16 fast loop, fixed for training *and* serving | Parity; bf16 is the throughput assumption |
| **Retention data** from company-owned accounts (own credentials, not via Cartel/CAAR) | The only way to validate moments ("attention drop") and ever fill `predictions` (hold/completion) |
| **Cartel data in the product.** The live deal/account context and baselines | The isolation contract keeps this repo separate; a product needs an explicit decision on where context comes from |
