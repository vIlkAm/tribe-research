# Pre-registration: does the brain mapping predict clip performance?

Written 2026-09-26, **before any model has been fit on real TRIBE outputs**. It
fixes the question, the endpoint, the pipeline and the decision rules, so the
answer can't be chosen after the result is seen. Change it only through the
deviations log at the end, dated, before the affected run.

Everything under "Frozen pipeline" is what `tools/build_features.py` and
`tools/fit_models.py` do at the commit that adds this file. Numbers come from
`tools/power_check.py` (`results/power/`).

## Question

TRIBE v2 turns a clip into predicted average-subject cortical activity by
reading three pretrained extractors (V-JEPA2 video, w2v-BERT audio, Llama-3.2
text) and mapping them to fsaverage5. A gain over metadata alone (B − A) can't
tell the brain mapping apart from the generic extractor features it consumes.
So the study asks two separate questions:

1. **Does the brain mapping add anything beyond its own inputs?**
   Answered by **BE − E**, the primary endpoint and the only "neural" claim.
2. **Do content features help at all?** Answered by **BE − A**. This decides
   whether scaling the run is worth it. The GPU cost sits in the extractors, so
   a full run is equally expensive whether the value is in E or in B.

| set | features |
|---|---|
| A | platform, deal, log duration, aspect, audio level, follower bucket, posting weekday, out-of-fold account target encoding (refit in every fit) |
| B | A + `brain_*` (ROI features + brain PCA) |
| E | A + `emb_*`: per extractor, PCA of the clip's time mean, sd over time and quarter shape (quarter means minus the time mean) of the exact layer-group features TRIBE's mapping reads |
| BE | A + `brain_*` + `emb_*` |

E sees the same coarse temporal structure as the brain features (sd and
quarter shape, not only a time mean). A positive BE − E therefore can't come
only from time-resolved pooling.

## Primary endpoint

- **Target:** `log_interactions_rate`, the empirical-Bayes shrunk engagement
  rate from `docs/OUTCOMES.md`. It's the most clip-driven label: same-platform
  ICC ≈ 0.59, against ≈ 0.29 for reach.
- **Metric:** within deal × platform Spearman ρ, averaged across strata and
  weighted by 1 / `incl_prob`.
- **Contrast:** BE − E, model `stack` (`BlockStackRegressor` in
  `tools/fit_models.py`):
  - Each wide block (`brain_*`, `emb_*`) is compressed to one score by its
    own `GroupedRidgeCV` (alphas 10⁻²…10⁵, inner content-grouped CV) fitted
    on y.
  - Training rows get their score from inner content-grouped folds that
    never saw them.
  - The final ridge sees A plus those scores, so BE is E plus exactly one
    column.
  - Why: one ridge with a shared penalty over 100–280 clip columns costs
    −0.08 to −0.14 within-stratum ρ on pure noise, and it loses most of a
    real signal (see Sample size).
- **Uncertainty:** paired cluster bootstrap over contents, 1,000 draws, 95 %
  percentile CI.
- **Claim rule:** "the brain mapping adds predictive signal beyond its inputs"
  holds only if the CI lower bound is > 0 on the scoring set named in the stage
  below. A point estimate alone never supports the claim.

## Stages and decision rules

### Stage 1: the ~1,500-clip study set (lockbox stays sealed)

- **Scoring:** 5-fold content-grouped CV over all train contents (≈ 1,275),
  from the frozen pipeline: `fit_models.py --schemes content`, **without**
  `--score-lockbox`.
- **Why CV and not the lockbox:** the 225-clip lockbox can't resolve a
  realistic gain (table below). Opening it here would burn it for nothing.
Stage 1 answers two questions, and each has its own rule.

- **Scaling (BE − A decides).** Use `stack`, the primary target and
  within-stratum ρ.
  - If the point is ≥ +0.02 **and** the CI lower bound is > −0.03, recommend
    funding the full eligible set. Otherwise, recommend not scaling.
  - Guard: the `account`-scheme BE − A point must also be ≥ 0. In that
    scheme, account fingerprinting can't help.
  - The owner makes the spending call either way.
  - The −0.03 floor matches the real CV precision. With the 1 / `incl_prob`
    weights (n_eff 528), a point of +0.02 carries a CI of about ±0.05.
- **Neural claim (BE − E decides).** At stage 1, BE − E is **directional
  only**.
  - Report it with its CI. Don't claim it confirmed even if the CI excludes
    0; stage 2 is the confirmatory test.
  - If its CI upper bound is < 0, stop describing the result as a brain
    effect. That drops the claim; it doesn't veto scaling.
  - If stage 2 goes ahead anyway, BE − E stays its confirmatory primary, and
    E − A becomes a declared secondary. The GPU cost is the extractors, so the
    full run is worth the same money whether the signal is in E or in B.

### Stage 2: the full eligible set (~7.9k contents), only after stage 1 GO and an owner budget decision

- **Lockbox = the existing sealed 225 + an extension.** Before any stage-2
  clip is sent to a pod, draw 15 % of each deal's new eligible contents with
  seed `20260926`, remove anything linked to them by `content_group` from
  train, and record the list in `results/study/`. The existing lockbox is
  never redrawn. Eligibility depends on the inputs, so the draw is pinned to
  the inputs as of this document (sha256):
  - `results/outcomes.parquet` `bc679801467603ea5b3bb9608ff057e6a9702685b72be65ccc9337b4e1d5a02c`
  - `results/run_full/manifest.jsonl` `9f778476862eea2a95584fb212492da74052a6ab82ca8ceccf9d73df86a91397`
  - `results/run_full/members.csv` `85c1d1eb8752e682d07667252ac1608bdc0aebd952261efda4f7f6ad66a7cae1`
  - `results/study/selection.csv` `dc47ba561d64e48a725d252afd489a65e0d26493c4cf21a4ddf3753ceed599cb`

  If any of these is re-exported first, log it as a deviation and draw from
  the pinned files.
- **Confirmatory primary:** BE − E, `stack`, `log_interactions_rate`,
  within-stratum ρ on that lockbox, scored **once** with `--score-lockbox`
  after the whole pipeline is frozen. The claim rule above applies.
- **Supporting evidence:** 5-fold content CV on the stage-2 train must have
  the same sign. If it doesn't, report the disagreement rather than choosing
  one.

## Secondary and exploratory

These are reported, never promoted to the headline. Benjamini-Hochberg at
q = 0.10 applies within each family.

1. **Arms, primary target:** B − A, E − A, B − E.
2. **Other target:** `reach_rel_local`, all contrasts. Its ceiling is lower,
   because reach is mostly account, platform and luck.
3. **Robustness:** `ridge` and `hgb` for every contrast; `account` and `lodo` (leave one
   deal out) schemes; `--fit-weighted`.
4. **Subgroups:** per platform and per deal with ≥ 100 scored clips; niche
   (per-deal and per-account) tuning. Accounts contribute 6–12 clips each, so
   per-account layers are shrinkage-dominated and descriptive only.
5. **Interpretation:** ROI/permutation importance and "moments"
   (`attention_drop` etc.). These are **unvalidated hypotheses**: no retention
   data exists to check them (see Known limits).
6. **Time-resolved (M), exploratory, clip level only.** The per-second UI
   moments are not tested against outcomes. All definitions (windows, norms,
   surrogate count, false-event rate) come from train contents without labels,
   and are committed before they meet an outcome.
   - **M1 hook:** proxy means over 0–4 s, normed by seconds-since-onset across
     the study clips. Within-stratum ρ with `log_interactions_rate`, and the
     increment over A and over E.
   - **M2 population-normed events:**
     - Each channel is z-scored against the study-set distribution at the same
       seconds-since-onset and clip-length bin.
     - An event lasts ≥ 5 s (one hemodynamic response) and beats
       phase-randomised surrogates at ≤ 0.5 false events per clip **per
       channel** (primary). The stricter reading, 0.5 summed over the 7
       channels, is a sensitivity run. Zero events is a valid result.
     - M2 is reported only next to its measured power (a planted 7 s drop of
       1.0 / 1.5 / 2.0 sd in real train clips). A null means "can't resolve",
       not "no moments".
     - Reported: events per minute and the time of the first sustained drop.
   - **M3 event-locked residuals:**
     - Each channel is first regressed on cut, speech-onset, loudness and motion
       regressors, then M2 runs on the residual.
     - Positive control: the cut-locked visual response must peak at the
       expected lag. If it doesn't, M3 is not reported.
   - **M4 good vs bad clip (between-clip contrast):**
     - Label: `log_interactions_rate` minus the arm-A `stack` out-of-fold
       prediction (content scheme, the primary folds), at post level. Taken from
       `oof_predictions.csv` (`scheme == content`: `y − pred_A_stack`,
       `stratum`, `w` = 1/incl_prob). Train rows only.
     - Unit: one row per content. Post residuals are averaged per `video_id`
       (weighted by `w`), and the stratum is the deal; a content spanning deals
       is refused. Cross-posted copies would otherwise narrow the permutation
       null.
     - Contrast: within deal, top vs bottom tertile of that residual, on three views of each channel's population-normed curve:
       seconds since onset 0–29, 20 fraction-of-clip bins, and the M3 residual.
     - Test: cluster permutation (Maris & Oostenveld 2007), labels shuffled
       within stratum, 2,000 permutations, cluster-forming |z| ≥ 2. That gives
       FWE per channel × view, then BH q = 0.10 across the 21 channel × view
       tests.
     - Replication: split-half by deal inside train. A cluster counts only if it
       has the same sign in both halves.
   - **Arm X, editing covariates:** A + `edit_*` (cuts per minute, first cut,
     speech onset, speech onsets per minute; CPU only), same folds and model as
     the primary. Reported as X − A, and B − X / E − X, to show whether plain
     editing structure already carries what the brain or embeddings carry.

## Frozen pipeline

| item | value |
|---|---|
| TRIBE | v2, `configs/tribev2-config.yaml` as used in `results/logs/pilot-l40s` |
| input scaling | 384 px short side (`tools/prep_cpu.py`). Parity with stock: spatial r ≥ 0.998 per timepoint, median per-vertex temporal r 0.998–0.999, mean relative abs difference ≈ 5 % |
| brain features | `clip_features_v1`, `n_pca` 20, PCA fit without lockbox clips (projected only); the rule-based `brain_moments_*` columns are **not** in the brain block (`--with-moment-cols` is a sensitivity run only) |
| emb features | `emb_features_v1`, read from the pod's `emb_pool_v1` files (time mean, sd, quarter shape per extractor), `n_pca` 20 per block, same lockbox rule |
| model | `stack` (primary); `ridge`, `hgb` secondary |
| pod code | the fast frame loop (`pod/fast_video.py`) and the emb export only count once the patched path reproduces the stock-s384 predictions on e87b4ed8ee19cbde and f1231478a06f2b30 at the parity thresholds above, **and** the first real export shows sensible shapes, `_n` ≈ 2 × duration and all four quarters non-empty on clips ≥ 20 s. Record the pod code commit in the deviations log |
| exclusions | failed/`status != ok` clips; outcome flags per `fit_models` defaults; **confident non-English** (whisperx language ≠ en with p ≥ 0.5; TRIBE transcribes as English) |
| precision | V-JEPA2 frame loop in **bf16** (`pod/fast_video.py`, `--fast-video --video-precision bf16`, pod code 7163f18) for **every** scored clip, pilot included; its feature cache is tagged separately. fp32 and bf16 outputs are never mixed in one fit |
| missing emb export | kept, median-imputed inside each fit, so every arm scores the same rows. If > 5 % of scored clips lack an export, re-export before scoring |
| weights | bootstrap and metrics weighted by 1 / `incl_prob`; model fitting unweighted |
| seeds | `--seed 0`, `--n-splits 5`, `--n-boot 1000` |

## Sample size

Simulated with `tools/power_check.py`. It uses the real A features and labels
of non-lockbox contents, plus one synthetic feature correlated with the
content's out-of-fold A residual. The table shows the achieved Δ ± 95 % CI
half-width, then power (the share of 40 designs whose CI excludes 0), for
`log_interactions_rate`:

| design | ρ 0.1 | ρ 0.2 | ρ 0.3 | ρ 0.5 |
|---|---|---|---|---|
| 1,275 train / 225 lockbox | +0.007 ±0.064, 0.03 | +0.028 ±0.094, 0.07 | +0.062 ±0.114, 0.12 | +0.152 ±0.134, 0.65 |
| **1,275, 5-fold CV** | +0.012 ±0.021, 0.15 | +0.042 ±0.032, 0.75 | +0.084 ±0.039, 1.00 | +0.187 ±0.046, 1.00 |
| 2,550 / 450 lockbox | +0.002 ±0.036, 0.00 | not run | +0.053 ±0.064, 0.33 | +0.136 ±0.078, 0.95 |
| full eligible, 15 % lockbox | +0.005 ±0.020, 0.03 | +0.024 ±0.030, 0.40 | +0.054 ±0.037, 0.78 | +0.137 ±0.045, 1.00 |

For `reach_rel_local`, CV at 1,275 gives ±0.041–0.059; the 225 lockbox gives
±0.117–0.162; the full-set lockbox gives ±0.047–0.061.

How to read it:

- A plausible real gain is +0.03–0.06.
- The 225-clip lockbox can't see that. CV over the 1,275 train contents is
  about as precise as the full-set lockbox, so CV is the stage-1 scorer.
- A confirmatory claim needs stage 2's larger lockbox.

The table above is **optimistic**:
- It uses one clean feature.
- The feature carries label noise of the posts it's scored on.
- The sampling is random, not the curated study design.

Three corrections were measured before this was frozen.

**Block width.** The real blocks are wide: an estimated ≈ 100 brain columns
(the exact count is logged at the first real build; no real ROI map is on this
server yet) and 180 emb columns (9 PCA blocks × 20). The simulation re-ran CV at 1,275 with
`--n-noise` / `--n-noise-both` for 20 designs each, on `log_interactions_rate`.
"Factor" means a correlated 5-factor block with the signal on one factor,
which is the realistic shape of ROI/PCA features. "Sparse" means one signal
column plus pure noise. Results are `results/power/power_check_cv_*.json`.

| contrast simulated | brain block | model | ρ 0 | ρ 0.2 | ρ 0.3 |
|---|---|---|---|---|---|
| B − A | 1 column | ridge | −0.001 ±0.007 | +0.042 ±0.032, 0.75 | +0.084 ±0.039, 1.00 |
| B − A | 1 column | stack | −0.001 ±0.006 | +0.042 ±0.030, 0.85 | +0.082 ±0.038, 1.00 |
| B − A | 100, factor | ridge | **−0.078** ±0.040 | −0.026 ±0.044, 0.00 | +0.022 ±0.047, 0.20 |
| B − A | 100, factor | stack | −0.002 ±0.008 | +0.030 ±0.027, 0.45 | +0.069 ±0.036, 1.00 |
| BE − E (E = 180 noise) | 100, factor | stack | −0.000 ±0.007 | +0.030 ±0.028, 0.45 | +0.070 ±0.037, 1.00 |
| B − A | 100, sparse | ridge | **−0.087** ±0.041 | −0.037 ±0.045, 0.00 | +0.010 ±0.047, 0.10 |
| B − A | 100, sparse | stack | −0.003 ±0.007 | +0.003 ±0.013, 0.00 | +0.022 ±0.023, 0.45 |
| BE − A | 280, sparse | ridge | **−0.135** ±0.052 | −0.094 ±0.052, 0.00 | −0.053 ±0.053, 0.00 |
| BE − A | 280, sparse | stack | −0.001 ±0.008 | +0.002 ±0.010, 0.00 | +0.012 ±0.016, 0.20 |

**Account fingerprinting.** Block models fit y directly, not A's residuals,
so a block could in principle learn an account's style as a proxy for its
level. The simulation checked this with `--structure account`: a 100-column
block carrying a per-account style vector and no content signal, at ρ 0.
`stack` gives −0.004 ±0.015 and `ridge` −0.058 ±0.039. No leak was detected;
the account-scheme guard in stage 1 backs this up on real data.

Reading the table:
- A shared-penalty ridge can't be the primary model: it overfits pure noise
  by −0.08 to −0.14.
- `stack` holds the null near 0 in every case.
- `stack` keeps most of a signal that lives in a few shared directions of the
  block.
- A signal hidden in one column among 100, or spread evenly over 100
  independent columns, is not learnable at about 1,000 contents by any
  estimator. The dense/isotropic `stack` run gives ρ 0.3 → +0.015 ±0.021.

**Weighting.** The study set's 1 / `incl_prob` weights give Kish
n_eff = 528 of 1,275 train contents. Real CV half-widths should be about
√(1275/528) ≈ **1.55×** the simulated, unweighted ones: about ±0.05 at
Δ ≈ +0.04, not ±0.03. The stage-1 GO floor of −0.03 accounts for this.

**Calibration.** Compare the across-design SD of Δ with the bootstrap SE
(half-width / 1.96):
- For engagement they agree (for example 0.014 against 0.016 at ρ 0.2), so
  the content bootstrap is calibrated.
- For `reach_rel_local` CV, the bootstrap is about **1.4× anti-conservative**
  (0.030 against 0.021 at ρ 0.1). Read reach CIs as too narrow.

Net result: at 1,500, stage 1 can detect a brain gain of about +0.05 or more
that is carried by the block's shared structure. A smaller or sparse gain
needs stage 2.

## Already seen before this was written

- The outcomes table and its ICCs, and the study-set selection. Selection
  used reach/engagement strata; that is handled by the `incl_prob` weights.
- A-only out-of-fold performance on non-lockbox contents, from the power
  check. That includes the labels of the future stage-2 extension contents
  (A-only, no brain or emb features).
- TRIBE outputs for 9 pilot clips (all train, none linked to the lockbox),
  used for parity and speed only. Nothing was fit on them against labels.
  They have no emb export; that's harmless at 9 of 1,500 (median-imputed), or
  they can be rerun.
- No lockbox label has been read by any analysis.

## Known limits (decided now, not after the result)

- **Moments and "attention drop" are not validated.** Public data has no
  watch time or retention. Checking them needs retention exports from
  company-owned accounts. That is a new owner decision under the isolation
  contract (its own credentials, nothing through Cartel/CAAR) and is not part
  of this study.
- **Per-second claims have a published null against them.** Reported by our
  literature pass:
  - TRIBE on 48 YouTube videos against "most replayed" heatmaps (Sahu & Pandey
    2026, arXiv 2607.01400): position-controlled partial r ≈ 0.06, CI spanning 0.
  - Real-fMRI forecasting of engagement came only from the first ~4 s (Tong,
    Knutson et al. 2020, PNAS).
  - Pooled fMRI studies explain ≈ 0.1–2.4 % of message-level variance (Scholz
    et al. 2025, PNAS Nexus).

  So expect small effects, and put the clip-level hook (M1) ahead of
  per-second moments. The release is cortical-only (no NAcc), and prefrontal
  predictions are mostly text-driven, so the `value` proxy is the least-trusted
  channel.
- **Scope:** short clips of 5–90 s, mostly English. Long-form clip picking (choosing
  segments of a long video) is a later, separate study.
- **Licence:** TRIBE is CC-BY-NC. A positive result is a research finding, not
  a product.

## Deviations log

| date | change | why | made before seeing |
|---|---|---|---|
| 2026-09-26 | Brain and emb PCA are fit once on all non-lockbox clips, so they span CV folds | unsupervised (no labels), cheap; refitting PCA per fold changes little | any real fit |
| 2026-09-26 | Primary model `ridge` → `stack` (`BlockStackRegressor`); `ridge` becomes secondary | power_check block-width runs: shared-penalty ridge loses 0.08–0.14 ρ on pure noise and most of a factor-structured signal; `stack` holds the null at ≈ 0 and keeps it (Sample size) | any real fit (committed text: 0539df5) |
| 2026-09-26 | Stage-1 rules split: BE − A decides scaling (floor −0.01 → −0.03); BE − E decides only the neural claim, and a null BE − E no longer vetoes scaling | the committed "GO" and "stop early" rules could both fire; n_eff 528 widens real CIs ≈ 1.55× | any real fit |
| 2026-09-26 | Added: weighting (n_eff), bootstrap calibration and block-width results; stage-2 input hashes; pod-code parity/emb-timing gate; `emb_pool_v2` renamed `emb_features_v1` (reads pod files `emb_pool_v1`) | review before any fit; the name clashed with the pod file format | any real fit |
| 2026-09-26 | Stage-1 GO gains the guard that the `account`-scheme BE − A point is ≥ 0; recorded that `stack` block models fit y, not A residuals, plus the `--structure account` check (no leak); brain width marked as an estimate | block scores fit y without the account term; checked, and guarded on real data | any real fit (committed text: 1207848) |
| 2026-09-26 | Stage-2 extension drawn now (`tools/select_rest.py`: 909 contents, pinned-input hashes verified, `results/study/lockbox_ext.csv`, sha256 `6329f505b6d9893ae4b2ab458674c9259f2abd3e2dbafb670b6452b5c11ba535`); candidates already linked by `content_group` to a study-set content are not drawable (28), and 8 rest-train twins of extension contents are dropped. The owner ordered TRIBE extraction on all ~7.9k eligible contents tonight, before the stage-1 result | the prereg rule is "draw before any stage-2 clip reaches a pod"; extraction reads no label, so it doesn't touch the sealed lockbox. Stage 1 is still fit and reported on the 1,500 first, and the full-set fit only follows it | any real fit |
| 2026-09-26 | Pod code 7163f18 (fast frame loop) with V-JEPA2 in bf16 adopted for all clips, pilot re-run in bf16 | gate on the L40S (reference: stock-s384 pilot): fp32 r_all ≥ 0.99997, rel_rms ≤ 0.009 (2 clips); bf16 9/9 pass, r_all 0.9998–0.99999, r_vertex_med ≥ 0.9997, rel_rms 0.004–0.020, named clips e87b4ed8 r_space 0.99998 / r_vertex 0.99988 and f1231478 0.9998 / 0.9997; `check_emb` 9/9. ~5× the fp32 throughput, so the full set fits the budget. Second GPU type, A100-SXM4 (sm_80), same gate on the same reference: 9/9 pass, r_all ≥ 0.9997, rel_rms ≤ 0.025 | any real fit and any study-set clip on a pod |
| 2026-09-26 | `brain_moments_*` columns dropped from the brain block (kept only in a `--with-moment-cols` sensitivity run); added the exploratory time-resolved family M1–M3 (secondary 6); literature limits under Known limits | CPU check on the pilot outputs (tyler-3f): rule-based moments are reproducible (89/90 stock-s384 vs fast-bf16), but every clip ≥ 11 s hits `MAX_MOMENTS` = 12 and the spans cover 89–376 % of the clip. Within-clip z always finds peaks, so the rates track length and the cap, not content | any real fit (no model had been fit on real outputs) |
| 2026-09-26 | Clarified, no rule change: TRIBE's own text-alignment guard (`AddSentenceToWords`, unmatched-word ratio > 0.05, e.g. 1 of 13 words) is a deterministic stock failure on short transcripts. Such clips are `failed` and fall under the existing exclusion (Frozen pipeline, exclusions), in train and lockbox alike. They are not re-run with the text extractor off, because that would change the feature definition inside the study. The stage-1 report lists failures by category and deal. If more than 2 % of study clips fail this way, that's flagged as a coverage limit (short-speech clips are out of scope) and not patched after the fact | first study-set failure, 9b3f6a8bc7374fc6 in b06 (tyler-29) | any real fit and any outcome read for a failed clip |
| 2026-09-26 | M1–M3 parameters frozen in code (7aa63ee, `tribe_research/brain/moments_pop.py`): clip-length bins, 20 surrogates, seed 20260926, threshold at ≤ 0.5 false events per clip per channel (summed over channels as sensitivity; d164f3c measures power), 5 s minimum, FIR lags −2..8 s, control window 0–3 s in stimulus time. The norm state is fit on bf16 train clips only (`build_moments_pop.py fit` refuses lockbox and non-bf16); its sha256s and train-ID hash are recorded here when it is fit. Added M4 (good vs bad contrast) and arm X (A + `edit_*`) to secondary 6; `oof_predictions.csv` gains a `stratum` column for the M4 labels | owner asked "what makes a good vs bad clip". Label-free development check on 46 local train clips (in-sample norms): the cut-lag positive control passes (peak lag 0, bootstrap lower bound 0.28). Planted-drop power at the summed budget is 10/19/35 % for 1.0/1.5/2.0 sd, and 35/54/71 % per channel, so the per-channel reading is primary (the earlier text was ambiguous). The control peaks at the edge of the frozen 0–3 s window; the window is not widened, so M3 is withheld if the full fit peaks at −1 s. M4 runs one row per content with stratum = deal, so cross-posts don't narrow the null | any real fit and any outcome joined to a time course |
| 2026-09-26 | Not a stage-1 run, no rule change: a **preliminary product model** (`model-prelim-v0`, `perf-stack-BE_v1+20260926.5b7e4b6`) was fit so the frontend can wire the performance card. Inputs: real bf16 fast-loop outputs of the complete batches only (frontend40, b00_pilot, b09; 100 clips, 13 of them lockbox: PCA-projected, not trained on, not referenced, not scored). Pre-registered defaults, nothing tuned, no `--score-lockbox`. `fit_models` gained a general fix (HGB with a column that has no finite training value, f8f6456). It is served only through `predict.py --allow-preliminary` with a mandatory "not validated" caption; its CV metrics are labelled preliminary in the release manifest and decide nothing | owner asked for model weights now, before the study set lands (`docs/MODEL.md`) | the stage-1 fit. From here, "before any real fit" no longer holds for new rows: say "before the stage-1 fit" |
