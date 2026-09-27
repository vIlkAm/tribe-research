# Demo script (judges, about 4 minutes)

Speaker script for the hackathon demo, in the owner's order: open, Results page, honesty, three clips, future.
Charts and clip videos from `tools/make_demo_media.py` are written outside git (`/tmp/viralbrain-media/`), and
the footage stays out of git.

Wording rules for anything said or shown:

- The brain visuals are TRIBE v2 predictions for an average viewer. No one was scanned for this project.
- Say "had" or "went with". Never "predicts views", "forecast", "edge" or "extreme".
- A clip's swing around its account's usual includes luck, timing and the algorithm, not only the video.
- The views grouping is exploratory (fixed before it was computed, not the pre-registered test).
- One clip is one example. Never state "bad clips rise at the end" as a general rule: each clip's own last
  3 seconds show no difference between the groups.
- TRIBE v2 was trained on 25 people and 452 h. The 720 people and 1,117 h are the paper's full dataset, which
  also tests on new people. Never say "trained on 720 people".
- Cite the paper's numbers and link its repo, weights and demo. Don't show the paper's figures.

## 1. Open (15 s)

"This is ViralBrain. We ran Meta's TRIBE v2 brain model over our agency's clips, second by second, and put the
predicted response next to each clip's real numbers."

## 2. Results page (60 s)

Open **Results**.

- "We grouped clips by views against their own account's usual. The top third had a higher predicted response
  than the bottom third: +0.09 standard deviations, and it holds when we resample whole accounts."
- "By signal, social and people, and personal relevance, hold, and they're strongest in the opening."
- "Visual and sound intensity is higher in the first seconds, the loud-opening trick, but it only leans."
- "Attention, reward and mental effort lean the same way."

Skip speech and meaning.

## 3. Honesty (15 s)

"Our strict test, written down before we looked, was on likes and comments per view. It was a no-GO, so the app
shows no score. That result is on the same page."

## 4. Clips (75 s)

- **Connor, great** (331,774 views, 34× the account's usual): "The opening is at the 95th percentile of our
  library, the middle the 19th, the ending the 7th. The hook carried it."
- **Greg, typical** (5,968 views, 2.6× usual): "A spike at 0:02, opening at the 76th percentile, a dip at
  0:06 to 0:09, then on the typical line."
- **Diagofit, bad** (677 views, 0.44× usual): "Below from the start, bottom 10% in the middle, and it rises only
  in the last 2 seconds. That's this clip, not a general rule."

## 5. Future (20 s or less)

"Next: the full library of about 7,900 clips, sentiment and niche models, and grading a clip before it's posted.
TRIBE v2 is Meta FAIR's model under a non-commercial licence. This is a non-commercial project, not affiliated
with Meta. Thanks."

## If a judge asks

- **"So does it predict virality?"** No. The pre-registered test on likes and comments per view was a no-GO. What
  we see on views is a small, pre-specified difference in the direction you'd expect.
- **"Does it hold across the whole library?"** Across all clips, a higher share of seconds above the typical line
  went with more views (r +0.14, 95% range +0.06 to +0.22) and with beating the account's usual (r +0.11). Our
  strict split-half check hasn't confirmed it yet. It's on the Library page.
- **"Results says it holds, the Library says not yet confirmed. Which is it?"** Results checks the views gap by
  resampling whole accounts, and it holds. The Library adds a stricter test, found on half the accounts and
  re-checked on the other half, and an effect this small needs more clips to pass it.
- **"Why is the bad clip's line high at the end?"** One clip is one draw. The whole-library groups are what count,
  and they're on the Results page.
- **"Whose brain is that?"** Nobody's. It's the model's prediction for an average viewer.
