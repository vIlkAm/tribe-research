# Library theory: what separates a good clip from a bad one (rework)

Written 2026-09-27 **before this analysis was run**. It is meant to replace the great-vs-bad tier tables
(`results/library/patterns_{relative,absolute,both,views}.json`, kept for provenance) as the source of the
Library page's statistics, pending the owner's OK and a frontend change (the page still reads the tier table). Exploratory: this is not the pre-registered test (`PREREGISTRATION.md`, stage 1
no-GO), and the lockbox stays sealed.

## Why the tier tables were not good enough

1. **Great vs bad throws most of the data away.** Typical clips were unused, and the owner's 300k threshold leaves
   38 great clips. That gives almost no power, so "no difference" was mostly "not enough clips".
2. **The bootstrap resampled clips, not accounts.** One account contributes many clips, so the intervals were too
   narrow.
3. **Absolute views are mostly account size.** Which account posted a clip explains about a third of its views
   rank. The brain response cannot know account size, so an absolute-views pattern is mostly "big accounts post
   clips like this".
4. **Display labels and statistics were the same thing.** The 300k / 5-15k / <700 labels are fine for showing a
   clip. They are a poor basis for statistics.
5. **Several tier rules were tried after seeing results.** Any rule picked now would be chosen after looking.
   This rework changes how things are measured, fixed here before the run, and reports whatever comes out.

## The theory being tested

A clip's views are **account size x how this video did for its account**. On the log scale this is exact in the
outcomes table: `reach_log = local_baseline + reach_rel_local`. This is the owner's "bad account vs bad video"
distinction.

## Outcomes (one per clip: its own post, the manifest `source_name`)

Posts with empty `dq_flags` and a `reach_rel_local`.

| id | outcome | column | compared within |
|---|---|---|---|
| total | views (7-day where available) | `reach_log` | deal x platform |
| account | account size (the account's recent usual) | `local_baseline` | deal x platform |
| video | this video vs its account's usual | `reach_rel_local` | account (accounts with >= 2 library clips) |
| engagement | interactions per view, small-count adjusted | `log(interactions_rate_eb)` | account (>= 2 clips) |

"Compared within" means that both the feature and the outcome are centred on the group's weighted mean rank
before correlating. So "video" and "engagement" compare clips of the **same account** only.

## Features

The same 12 as `tools/library_tiers.py` `FEATURES`: length, cuts per minute, first cut, first word, words per
second, and seven brain-line summaries (overall, share of seconds above the typical line, opening, middle, peak,
share of low seconds, ending). No features are added.

## Statistic

- Weighted Spearman correlation. Rank the feature and the outcome over the analysed clips, centre both within the
  comparison group, then take the weighted Pearson correlation. Weights are `1 / incl_prob`, which undoes the study
  set's oversampling.
- Features are winsorised at 1-99% before ranking.
- 95% intervals come from an **account-clustered** bootstrap (resample accounts; 2,000 draws; seed 20260926).

## Discovery and confirmation

- Accounts are split 50/50 by `sha256("20260926:<social_account_id>")` order, alternating within each account's
  deal x platform. No account is in both halves.
- **Discovery half:** all 48 feature x outcome pairs, with a bootstrap p-value and Benjamini-Hochberg q.
  A **candidate** has q <= 0.10.
- **Confirmation half:** each candidate is tested once. It **holds up** if the sign matches and the 95% interval
  excludes 0.
- Only pairs that hold up are described as patterns (plain label "holds on new accounts, small" and similar).
  Everything else is "no pattern". Full-library numbers are shown for context only.
- Size words: |r| < 0.10 "very small", 0.10-0.30 "small", 0.30-0.50 "moderate", >= 0.50 "large".

## Also reported

- **How much of views is account vs video:** the share of the within-deal x platform variance of `reach_log` that
  is `local_baseline` versus `reach_rel_local` (and their covariance), over library clips.
- The owner's display tiers (300k+ / 5-15k / <700 below usual) stay as clip labels only.

## What will not change after the run

The outcomes, features, statistic, split, thresholds and wording rules above. If a result looks wrong, the fix
is a bug fix, dated here with its reason, not a new rule.

## Changes after the run

- **2026-09-27, wording bug fix:** the interpreter sentence gave account size 43% and video 69%, which adds to more
  than 100% without the negative overlap (-11%). It now states which part is larger and shows all three shares. No
  number, rule or verdict changed.

- **2026-09-27, Learned chart by views (decided before it was computed):** the owner asked for the Learned page's
  "good vs bad" curves to group clips by views. Fixed before the run: raw `reach_rel_local` (views against the
  account's usual; the "video" outcome above and the stage-1 secondary endpoint), content-scheme train rows of
  `results/models/stage1-eval/oof_predictions.csv`, the same M4 handling (multi-deal contents refused, one row per
  content, w-weighted mean) and the same within-deal thirds (`tertile_labels`). Curves are unweighted means of the
  response index for seconds 0-29 with 1,000-draw bootstrap bands, like the engagement chart. Two summaries, both
  reported whatever they show: top minus bottom third in the clip-mean index over its seconds 0-29, and in the
  opening (first 4 s), each with a bootstrap 95% interval, plus the count of seconds whose interval excludes 0 (out
  of 30; some expected by chance). Descriptive and exploratory, not a pre-registered test; the engagement (M4) chart
  stays on the page below it.

## Result (2026-09-27, run once: `tools/library_theory.py`, `results/library/theory.json`)

- 1,143 library clips from 182 accounts. Discovery half: 699 clips, 103 accounts. Confirmation half: 444 clips,
  81 accounts.
- **Views:** within a deal x platform, a clip's views swing more around its account's usual (variance share 69%)
  than accounts differ from each other (43%; overlap -11%, largely mechanical because the usual is estimated from
  nearby posts). On 7-day posts only (890 clips, same age for every post) it is 78% vs 48% (overlap -26%). The
  swing includes luck, timing and the algorithm, not only the video itself.
- **Held up on new accounts: one pair.** Longer clips have higher engagement per view within the same account:
  r = +0.28 on discovery, +0.27 [+0.15, +0.36] on confirmation, +0.28 on the full library (small). This is not a
  cause: longer clips also tend to get fewer views (-0.11 within account), and rates per view rise as views fall.
- **Candidates that did not hold:** length vs total views (-0.15 on discovery, -0.03 on confirmation) and the
  strongest 3 s vs engagement (+0.14, then +0.01).
- **No brain-line summary held up** for account size, video vs account or engagement. On the full library (for
  context only, not corrected for 48 tests) the share of seconds above the typical line goes with views
  (r = +0.14) and with video vs account (+0.11), and the share of low seconds goes the other way (-0.14, -0.12).
  These are very small to small, not confirmed on held-out accounts, and in line with stage 1's no-GO.
- **2026-09-27, wording fix 2:** the interpreter no longer says views differ "because of how the video did"; the
  swing around the usual includes luck, timing and the algorithm. The "big accounts see smaller jumps" reading of
  the overlap was dropped (the overlap is largely mechanical). A 7-day-only decomposition was added as a
  descriptive check. No verdict changed.
- **2026-09-27, Learned chart by views: result** (`tools/build_library_profile.py --learned`, run once with the
  design above): 1,254 contents, 12 deals, 415 per third. Clips above their account's usual had a higher predicted
  response than clips below it: whole 0-29 s +0.091 library sd [+0.028, +0.155], opening 4 s +0.095 [+0.018,
  +0.175]; 12 of 30 seconds (0-2, 4-12) have intervals excluding 0, and the gap closes after about 25 s. Added check,
  after seeing the result: resampling accounts instead of clips (192 accounts, 2,000 draws) gives +0.091 [+0.029,
  +0.153] and +0.095 [+0.015, +0.172]. This fits the full-library direction above (share above typical vs video
  +0.11); the theory table's within-account rank test of the opening alone stays "no pattern" (+0.04). Exploratory,
  small, and it does not change stage 1's no-GO.

- **2026-09-27, per-signal numbers for the views chart (post hoc; decided before computing, after seeing the
  per-second channel curves):** the owner wants to quote the seven small charts on camera, so each gets fixed
  summaries. For each channel (attention, social, value, control, self, language, sensory), top minus bottom third
  (same thirds as above) in three per-clip windows the product already uses: opening (first 4 s, `mp.HOOK_S`),
  whole (clip mean over seconds 0-29, the area under the curve per second) and ending (the clip's own last 3 s,
  `FINISH_S`, over the full clip, so it is not limited to the 30 s grid). The response index gets the ending window
  too, for context. Intervals: account-clustered bootstrap (a content belongs to the account of most of its training
  posts, ties by sorted id; 2,000 draws; seed 20260926; a separate generator, so the existing numbers do not move);
  two-sided bootstrap p, Benjamini-Hochberg across the 21 channel tests. Wording: "holds" only if q < 0.05 and the
  interval excludes 0; "leans the same way" if the point has the sign but does not hold; otherwise "no difference".
  Everything is reported, whatever it shows. Exploratory and post hoc: the windows are fixed, but the channels were
  looked at first. The 25-29 s part of the chart has about a third of the clips (only clips that long), which is why
  the ending uses each clip's own last seconds.
