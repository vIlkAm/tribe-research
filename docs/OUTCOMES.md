# Outcomes table

`tools/build_outcomes.py` turns the read-only metrics export
(`results/metrics/*.csv`) into one row per `video_performances` id:
`results/outcomes.parquet` plus `results/outcomes.parquet.meta.json`
(parameters, fitted engagement priors, cleaning totals, flag counts, column list).
Code: `tribe_research/outcomes/`. It reads files only and never queries a database.

```bash
.venv/bin/python tools/build_outcomes.py --metrics-dir results/metrics --out results/outcomes.parquet
```

**These are descriptive outcomes, not causes and not a score.** Reach and
engagement stay separate. Percentiles and quadrants compare a clip only with
other clips of the same deal (or deal × platform), and the group size `n` is
always next to them.

## Inputs and joins

| Table | Use | If missing |
|---|---|---|
| `video_performances` | one output row per `id`; final counts | required |
| `video_snapshots` | view history for reach at age | required (may be empty) |
| `social_accounts` | platform, `deal_id` | required |
| `deals` | `deal_name` | required |
| `social_account_stat_snapshots` | follower count near upload | optional → null |
| `cross_platform_members` | `content_group` | optional → every video is its own group |

- Ids are read as strings. Booleans accept `t/f/true/false/1/0`. Empty strings are null.
- `deal_id` comes from the account; if the account has none, `deal_campaign_id`
  is used only when it is a known `deals.id` (`deal_id_source` records which).
- `platform` is lower-cased.

## Definitions

### Ages and snapshot cleaning

- `upload_date` is a date, so the upload instant is taken as **12:00 UTC** of that
  day (`--upload-anchor-hours`). Ages therefore carry up to ±12 h of error;
  `views_1d` is the most affected. A snapshot with no `snapshot_at` uses
  `snapshot_date` at 12:00 UTC.
- Cleaning per video, in order:
  1. Drop snapshots of unknown videos, and those with no time or null/negative views.
  2. Dedupe on (video, time), keeping the largest views.
  3. Drop "broken zeros": 0 views after an earlier non-zero value.
  4. Take the running maximum, so views never decrease.

  Counts of each step are output per video.
- `age_days_at_last_obs` is measured at the later of the last snapshot and the
  time `views_final` was observed (`provider_observed_at` if exported, else
  `last_updated`).
- `views_final` is `video_performances.views`, or the last cleaned snapshot if that is null.

### Reach at fixed age: `views_{1,3,7,30}d`

`log1p(views)` is interpolated linearly in age between the last snapshot at or
before A and the first at or after A:

```
log1p(v_A) = log1p(v_lo) + (log1p(v_hi) − log1p(v_lo)) · (A − a_lo) / (a_hi − a_lo)
```

Without snapshots on both sides of A, there is one exception. A one-sided
log-linear extrapolation uses the slope between the two nearest snapshots, and
is allowed only if the nearest snapshot is within **20 % of A**
(`--max-extrapolation-frac`; 0 = strict brackets only). A backward
extrapolation is clamped to [0, first observed value]. Everything else is null.

`views_{A}d_basis` records how each value was obtained:

| Basis | Meaning |
|---|---|
| `exact` | a snapshot at exactly A |
| `interpolated` | bracketed by snapshots on both sides |
| `extrapolated` | one-sided, within 20 % of A |
| `too_young` | the video's last observation is before A |
| `no_bracket` | old enough, but snapshots do not reach A |
| `no_snapshots` | no snapshots at all |
| `no_upload_date` | no upload date, so no age |

`views_{A}d_span_days` is the bracket width, so wide brackets can be filtered out.

### Baseline and relative reach

The **primary age** is 7 days by default (`--primary-age`, e.g. 14 or 30 when
7-day coverage is thin). It is added to the reported ages if missing.

- `reach_log = log1p(views_{primary}d)`. Fallback: `log1p(views_final)` if the
  video is at least the primary age at its last observation.
  - `reach_basis` records which was used.
  - `views_age_used` is the age in days behind `reach_log`: the primary age, or
    `age_days_at_last_obs` for the fallback.
  - `views_age_basis` repeats the primary age's `_basis` column.
- Short-form views mostly accrue in the first week, but the fallback still
  inflates reach for old videos.
- `baseline` is the median `reach_log` of the account's **other** videos
  (leave-one-out). It needs ≥ 5 others. The fallbacks are:
  1. the same leave-one-out median over the deal × platform, also needing ≥ 5 others;
  2. otherwise null.

  `baseline_level` (`account|deal_platform|none`) and `baseline_n` record which
  was used. A video without its own reach value still gets a baseline over all
  the others.
- `reach_rel = reach_log − baseline`, a log ratio: +0.69 ≈ 2× the account's typical 7-day views.
- The baseline uses all of the account's videos, including later ones. This is
  an outcome description, not a forecast.
- **Time-local baseline.** Account growth confounds the whole-history baseline:
  early clips of a growing account look bad and late ones look good.
  - `local_baseline` is the median `reach_log` of the **10 other videos of the same
    account nearest in upload time with the same `reach_basis`** (leave-one-out;
    equal distances take the earlier video). A 7-day value is only compared with
    7-day values and a final-views fallback only with final-views values:
    lifetime views run higher than 7-day views, so mixing them would bias the ratio.
  - It needs ≥ 5 such videos, and otherwise falls back to the deal × platform
    leave-one-out median, then to null (`local_baseline_level` =
    `account_local|deal_platform|none`).
  - `local_baseline_n` is the number of videos used.
  - `local_baseline_max_gap_days` is the furthest neighbour's distance in upload time.
  - `reach_rel_local = reach_log − local_baseline`. `reach_rel` is kept alongside it.
  - Percentiles and quadrants use `reach_rel`.
- `follower_count` comes from the account's stat snapshot nearest to the upload
  anchor, within 30 days (`--follower-tolerance-days`). Snapshots reporting 0
  followers are ignored as provider misses. `follower_snapshot_gap_days` is
  snapshot minus upload. Followers exist only for some accounts.

### Engagement (empirical Bayes)

For `like_rate`, `comment_rate`, `share_rate`, `save_rate` and
`interactions_rate`, counts are over `views_final` (trials).

`interactions` is the sum of the components the row actually has;
`interactions_components` lists them. It is comparable within deal × platform,
not across platforms.

Counts above views are clipped to views and set `flag_rate_gt1`.

The prior is `Beta(μc, (1−μ)c)`, fitted by the method of moments per deal ×
platform on videos with ≥ 100 views:

```
μ    = mean(p_i)
τ²   = var(p_i) − mean(μ(1−μ)/n_i)     # between-video spread minus binomial noise
c    = μ(1−μ)/τ² − 1, clipped to [1, 10⁴]   (τ² ≤ 0 → 10⁴)
```

A group with fewer than 30 such videos falls back to the platform prior, then
to the global prior (`{rate}_prior_level`).

The posterior mean is `(μc + k)/(c + n)`, reported with its 90 % equal-tailed
interval (`_eb`, `_eb_lo90`, `_eb_hi90`). With 3 views it stays near
`_prior_mean`; with millions of views it equals the raw rate. `_raw` is the
unshrunk rate. `meta.json` lists every fitted prior (α, β, n).

### Per-deal ranking

- `pct_reach_rel_{deal,deal_platform}` and `pct_interactions_{deal,deal_platform}`
  are mid-rank percentiles `(rank − 0.5)/n` in (0, 1). They are computed among
  videos of the group that have the value, with `n_*` alongside, and are null when n < 5.
- `quadrant` compares `reach_rel` and `interactions_rate_eb` with the deal's
  medians, giving `high|low_reach_high|low_engagement`.
  `quadrant_deal_platform` does the same within deal × platform; prefer it,
  because engagement levels differ by platform.
- These are two separate axes. There is no combined score.

### Cross-platform

- `content_group` is `link:<link_id>` or `video:<id>`, with `content_group_size`
  and `content_group_n_platforms`.
- Spreads are computed only for groups with ≥ 2 members that have the value:
  - `reach_rel_vs_group`: `reach_rel` minus the group mean;
  - `reach_rel_group_range`;
  - `pct_interactions_deal_platform_vs_group`.
- `group_reach_rel_{tiktok,instagram,youtube}` puts the group's per-platform
  values on every row of the group.
- Raw rates are not compared across platforms.

### Data-quality flags

`flag_*` booleans plus `dq_flags` (`;`-joined):

| Flag | Condition |
|---|---|
| `missing_snapshots` | no snapshots after cleaning |
| `missing_upload_date` | no upload date |
| `upload_date_inconsistent` | a snapshot more than 1 day before the upload anchor |
| `too_young` | younger than 7 days at last observation (an `extrapolated` `views_7d` can carry this flag too; filter on the basis column, not the flag, to keep those) |
| `views_zero` | `views_final` is 0 |
| `views_missing` | `views_final` is null |
| `broken_history` | any of: a broken zero; a raw drop of more than 50 % below the running max (e.g. a spike); final views below half of the snapshot max |
| `rate_gt1` | any count above views |
| `unknown_account` | account not found |
| `no_deal` | no deal could be assigned |
| `history_truncated` | the export marks the video's snapshot history as truncated |
| `video_gone` | `caar_video_gone_since` is set (the post disappeared) |

## Known gaps

- The export has no watch time, retention, completion rate, impressions,
  traffic source, or unique viewers. Views are plays, not people. A viewer can
  like and comment, so the binomial model is only an approximation.
- The upload time of day is unknown. In the first real export, about 7 % of
  videos (`upload_date_inconsistent`) have snapshots weeks before their
  `upload_date`. Their ages, and so their `views_{A}d`, are unreliable. Those
  values are kept (flagged), but such videos get `reach_basis =
  upload_date_inconsistent`, no `reach_rel`, and are excluded from every
  baseline pool.
- Many backfilled videos were first snapshotted months after upload. They have
  no early-age values (`no_bracket`) and fall back to final views for reach.
- Snapshot cadence varies; check `_span_days` and `_basis` before comparing early ages.
- `last_updated` is taken to be when `views` was last refreshed. If the export
  updates it for other reasons, `age_days_at_last_obs` is overstated.
- The running maximum locks in a single spike. Such videos are flagged, not repaired.
- The deal is a proxy for content type. Posting time, account age and platform
  algorithm changes are not controlled for.
