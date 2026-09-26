# The study set (~1,500 clips)

The first real TRIBE run uses a curated library, not a random draw. It is
built by `tools/select_study_set.py` from files only (the read-only export in
`results/metrics/`, the outcomes table, the full-run plan). Rebuild:

```bash
tools/export_metrics.sh                                  # read-only DB export
.venv/bin/python tools/build_outcomes.py                 # -> results/outcomes.parquet
.venv/bin/python tools/build_run_manifest.py             # unique contents + staging links
.venv/bin/python tools/qc_files.py --manifest results/run_full/manifest.jsonl \
    --staging results/run_full/staging --out results/run_full/file_qc.csv
.venv/bin/python tools/select_study_set.py               # -> results/study/
```

`results/study/selection_report.md` has the numbers for the current draw.

## Goal

Learn what separates clips that do well from clips that don't **within the
same kind of content**. That needs trustworthy labels, contrast (good and bad
from the same place), coverage of every deal, and an honest way to check the
result.

## Unique content

The same clip is often posted by several accounts and on several platforms.
TRIBE runs once per content; every post's outcome joins back through
`results/run_full/members.csv`. Contents are merged when files are identical,
when the backfill linked them across platforms, or when their frame
fingerprints are near-identical inside one deal (any date, any platform; strict
thresholds, groups capped at 10 so shared intros can't chain unrelated clips).
21,234 files → 12,189 unique contents.

## Labels

A post's label is usable only when its view history is clean (not truncated,
no upload-date conflict, not too young, not gone, no broken counts) and its
reach is measured against **the same account's clips posted around the same
time**, measured the same way (`reach_rel_local`: a 7-day value against other
7-day values, a lifetime-views fallback against other lifetime values). That
removes account size and account growth. A
content's reach is the mean over its labelled posts. Engagement is the shrunk
interactions rate (likes + comments + shares + saves per view), ranked within
deal × platform; posts with too few views to know are "unknown", not "middle".
Definitions: `docs/OUTCOMES.md`.

## Eligibility

At least one labelled post; the file decodes with video and audio; 5-90 s;
title not mostly non-Latin (TRIBE transcribes English only). 636 TikTok files
use ByteDance's BVC2 codec, which ffmpeg can't decode; they are excluded until
an H.264 copy is fetched.

## Sampling

1. **Deal quotas:** about √(eligible) per deal, at least 60 (or all), at most 250.
2. **Lockbox:** 15 % of each deal, uniform random. Final results are reported
   on it once, so the oversampling below can't flatter any model.
3. **Accounts first:** accounts with ≥ 8 eligible clips get 6-12 each (≤ 15 %
   of a deal unless it has very few accounts), spread over the clip's reach
   quintile within its own account, with the tails oversampled. Same account,
   same audience, different content is the cleanest contrast available.
4. **Engagement balance** inside each reach stratum; posted-more-than-once
   contents win ties (their labels are measured several times).
5. **Weights:** each row records `stratum` and `incl_prob`; analyses weight by
   1 / `incl_prob` to get back to the natural distribution.

## What to expect (measured on this data)

How much of performance is about the clip itself? Among contents posted more
than once:

| comparison | reach | engagement |
|---|---|---|
| same clip, different platforms | ~16 % | ~15 % |
| same clip, same platform, different accounts | ~29 % | ~59 % |

On one platform, engagement is mostly the clip; reach is mostly platform,
account, timing and luck. These are rough ceilings for any content model,
brain features or not, so models are evaluated per platform and engagement is
the most learnable target.

## Known gaps

- No watch time / retention (not in public data).
- Deals with fewer than ~100 clips in the set only feed the cross-deal model.
- Clips from before snapshot history was kept mostly fail the label rules.
