# Stage-1 result (2026-09-27)

**Result: no-GO.** On the study set, brain features plus extractor embeddings do not beat
metadata alone by the pre-registered margin. The performance card therefore stays
`not_trained`: the demo shows the brain analysis with no performance number, percentile,
drivers or reach. Rules: [`PREREGISTRATION.md`](PREREGISTRATION.md) "Stage 1". The fit ran
once (00:09–00:28 UTC, 1,170 s) with the pre-registered defaults. No lockbox clip was
scored or opened.

## Inputs

| Item | Value |
|---|---|
| Study clips extracted | 1,500 expected, 1,486 ok, 14 excluded, 0 missing, 222 lockbox ok (sealed) |
| Exclusions | TRIBE text-alignment guard (failed main pass and the retry): adam-yu 4, gymshark 1, polymarket 1. Variable-frame-rate prep skip: connor 7, enhanced-games 1. 14 / 1,500 = 0.9 %, under the 2 % flag |
| Runtime | all bf16 fast frame loop, pod code 7163f18, TRIBE af58661; L40S 490, A100 454, RTX 4090 542 clips; 0 NaN or shape problems |
| Features | `results/features/stage1/features.parquet` sha256 93a61595cf751d4e34db36a0deed4fc6ebdf5c226a5032df5422bc7c34c5f226 |
| Moments state | frozen before the fit (prereg deviations log, 2026-09-27 moments row) |

**Why the counts differ.** 1,486 ok minus 222 lockbox leaves 1,264 train clips. The primary
target `log_interactions_rate` needs a usable outcome; that leaves 1,157 contents
(bootstrap units) posted as 1,865 posts. With the 1 / `incl_prob` weights the Kish effective
sample size is **470 contents** (the prereg planned on 528).

## Primary: `log_interactions_rate`, `stack`, within-stratum Spearman

Content-grouped 5-fold CV, bootstrap over contents (95 % CI).

| Arm | ρ |
|---|---|
| A (metadata) | 0.335 |
| B (A + brain) | 0.339 |
| E (A + extractor embeddings) | 0.343 |
| BE (A + brain + embeddings) | 0.344 |

| Contrast | Scheme | Point [95 % CI] | Rule | Outcome |
|---|---|---|---|---|
| BE − A | content | +0.009 [−0.009, +0.024] | point ≥ +0.02 and CI lower > −0.03 | **fails** (point below +0.02) |
| BE − A | account (96 accounts) | −0.012 [−0.030, +0.015] | point ≥ 0 | **fails** |
| BE − A | leave-one-deal-out (12 deals) | −0.001 [−0.014, +0.016] | reported only | — |
| BE − E | content | +0.0004 [−0.010, +0.010] | directional only at stage 1 | see below |

The GO rule fails on **two independent criteria**: the content point is below +0.02, and the
account-scheme guard is negative. This is not a near miss that more tuning would fix.

**Neural claim (BE − E).** The point is +0.0004, so `predict.brain_claim_from` returns
`directional` mechanically (point > 0, CI upper ≥ 0). Read plainly, it is a null: the
interval is centred on zero, and B − E is −0.004 [−0.024, +0.020]. With no-GO the product
shows no brain claim at all, and "directional" never permits the neural claim in the UI.

## Secondary: `reach_rel_local` (declared secondary, not a product claim)

| Contrast | Content CV | Account | LODO |
|---|---|---|---|
| E − A | +0.073 [+0.025, +0.117] | +0.046 [+0.029, +0.089] | +0.041 [+0.022, +0.091] |
| BE − A | +0.065 [+0.018, +0.106] | +0.050 [+0.025, +0.100] | −0.012 [−0.067, +0.095] |
| BE − E | −0.008 [−0.026, +0.009] | +0.004 [−0.015, +0.024] | −0.053 [−0.103, +0.013] |

The extractor embeddings (arm E) carry a reach signal over metadata that holds across
schemes. The brain features add nothing on top of them. This is an exploratory reading of a
secondary endpoint. It does not change the stage-1 decision, and under no-GO the demo shows
no reach.

## What follows

- **Product state:** `not_trained` (`predict.model_status_from`). No stage-1 model release;
  the saved diagnostic model (`results/models/stage1/`) stays internal.
- **Demo release:** `data-demo-stage1-v1` ships brain-analysis bundles with a `not_trained`
  performance block and no numbers.
- **Stage 2** (all ~7.9k clips, lockbox opening) was conditional on GO. The recommendation is
  not to fund it for the brain hypothesis. The owner makes the spending call. The E − A reach
  signal is the thing worth testing if anything is scaled.
- **Proof-of-concept holdout** (`results/study/poc_holdout.csv`, prereg 2026-09-27) is
  unaffected and stays exploratory.
- **Still to run (secondary family 6, exploratory):** arm X (A + editing covariates).

## Secondary: M4 "good vs bad clip" (family 6, exploratory)

Within each deal, the top and bottom thirds of clips by `log_interactions_rate` residual (after the
arm-A out-of-fold prediction) were compared on each channel's population-normed brain curve. There
are three views: seconds since onset, fraction of the clip, and the M3 residual. The test is a
cluster permutation (2,000, labels shuffled within deal) with BH q = 0.10 over 21 tests (prereg
900001b, moments state `state_v1`).

- **Sample:** 1,155 contents in 12 deals. 2 contents posted in two deals were refused, as the
  prereg requires.
- **Result: no cluster survives.** The smallest FWE p is 0.13, and every q is 0.77.
- **Reading:** the predicted brain time course does not separate clips that did better than
  expected from those that did worse. This agrees with the null BE − E above.
- **Report:** `results/moments_pop/m4_v1.json` (local).

Full tables: `results/models/stage1-eval/report.md` and `metrics.json` (local, not in git).
