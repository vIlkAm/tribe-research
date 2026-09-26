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
- **Contrast:** BE − E, `ridge` (`GroupedRidgeCV`, alphas 10⁻²…10⁵, inner
  content-grouped CV).
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
- **Go/no-go on scaling:** BE − A (ridge, primary target, within-stratum ρ)
  point ≥ +0.02 **and** CI lower bound > −0.01 → recommend funding the full
  eligible set. Otherwise recommend not scaling. The owner makes the spending
  call either way.
- **BE − E at stage 1 is directional only.** Report it with its CI. Don't
  claim it confirmed even if the CI excludes 0, because stage 2 is the
  confirmatory test.
- **Stop early without scaling** if BE − E has a CI upper bound < +0.02 **and**
  E − A > 0. The value is then in generic extractor features. Report that
  plainly; the brain story doesn't hold.

### Stage 2: the full eligible set (~7.9k contents), only after stage 1 GO and an owner budget decision

- **Lockbox = the existing sealed 225 + an extension.** Before any stage-2
  clip is sent to a pod, draw 15 % of each deal's new eligible contents with
  seed `20260926`, remove anything linked to them by `content_group` from
  train, and record the list in `results/study/`. The existing lockbox is
  never redrawn.
- **Confirmatory primary:** BE − E, ridge, `log_interactions_rate`,
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
3. **Robustness:** `hgb` for every contrast; `account` and `lodo` (leave one
   deal out) schemes; `--fit-weighted`.
4. **Subgroups:** per platform and per deal with ≥ 100 scored clips; niche
   (per-deal and per-account) tuning. Accounts contribute 6–12 clips each, so
   per-account layers are shrinkage-dominated and descriptive only.
5. **Interpretation:** ROI/permutation importance and "moments"
   (`attention_drop` etc.). These are **unvalidated hypotheses**: no retention
   data exists to check them (see Known limits).

## Frozen pipeline

| item | value |
|---|---|
| TRIBE | v2, `configs/tribev2-config.yaml` as used in `results/logs/pilot-l40s` |
| input scaling | 384 px short side (`tools/prep_cpu.py`). Parity with stock: spatial r ≥ 0.998 per timepoint, median per-vertex temporal r 0.998–0.999, mean relative abs difference ≈ 5 % |
| brain features | `clip_features_v1`, `n_pca` 20, PCA fit without lockbox clips (projected only) |
| emb features | `emb_pool_v2` (mean, sd, quarter shape per extractor), `n_pca` 20 per block, same lockbox rule |
| exclusions | failed/`status != ok` clips; outcome flags per `fit_models` defaults; **confident non-English** (whisperx language ≠ en with p ≥ 0.5; TRIBE transcribes as English) |
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

The simulation is **optimistic**. It uses one clean feature, whereas the real
B and BE have 100+ features that will overfit. The feature carries label noise
of the posts it's scored on. The CV bootstrap treats the fitted models as
fixed. The sampling is random, not the curated study design. Expect real
power to be lower.

## Already seen before this was written

- The outcomes table and its ICCs, and the study-set selection. Selection
  used reach/engagement strata; that is handled by the `incl_prob` weights.
- A-only out-of-fold performance on non-lockbox contents, from the power
  check.
- TRIBE outputs for 9 pilot clips (all train, none linked to the lockbox),
  used for parity and speed only. Nothing was fit on them against labels.
- No lockbox label has been read by any analysis.

## Known limits (decided now, not after the result)

- **Moments and "attention drop" are not validated.** Public data has no
  watch time or retention. Checking them needs retention exports from
  company-owned accounts. That is a new owner decision under the isolation
  contract (its own credentials, nothing through Cartel/CAAR) and is not part
  of this study.
- **Scope:** short clips of 5–90 s, mostly English. Long-form clip picking (choosing
  segments of a long video) is a later, separate study.
- **Licence:** TRIBE is CC-BY-NC. A positive result is a research finding, not
  a product.

## Deviations log

| date | change | why | made before seeing |
|---|---|---|---|
| 2026-09-26 | Brain and emb PCA are fit once on all non-lockbox clips, so they span CV folds | unsupervised (no labels), cheap; refitting PCA per fold changes little | any real fit |
