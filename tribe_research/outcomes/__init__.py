"""Per-video outcome table from the read-only metrics export.

One row per ``video_performances`` id with reach at fixed ages, reach relative
to the posting account's own baseline, sample-size-adjusted engagement rates,
within-deal percentiles, cross-platform content groups and data-quality flags.
Definitions, formulas and fallbacks are in ``docs/OUTCOMES.md``.

Assumptions and honesty rules (read before using the numbers):

* **Descriptive only.** Nothing here estimates causal effects. A clip's reach
  depends on account, timing, platform algorithm state and many things the
  export does not contain.
* **No universal score.** Reach and engagement stay separate. Percentiles and
  the ``quadrant`` label are *within one deal* (or deal x platform), with the
  group size ``n`` alongside; compare clips only inside those groups.
* **Upload time is a date.** The upload instant is assumed to be
  ``upload_date`` at 12:00 UTC (``upload_anchor_hours``), so ages carry up to
  +-12 h of error; ``views_1d`` is the most affected.
* **Reach at age A** interpolates ``log1p(views)`` linearly in age between the
  bracketing snapshots. A one-sided extrapolation is allowed only within 20 %
  of A (``max_extrapolation_frac``; 0 = strict brackets) and is labelled
  ``extrapolated``. Snapshots are deduped, broken zeros dropped and views made
  monotone with a running maximum; a large drop flags the history.
* **Baselines** are leave-one-out medians over the same account's other videos
  (>= 5), falling back to the deal x platform median, else none. They use all
  of the account's videos, including later ones (outcome description, not
  prediction). ``reach_rel_local`` uses the 10 same-account videos nearest in
  upload time instead, to reduce confounding by account growth. Reach falls
  back to final views for videos older than the primary age (default 7 d,
  selectable) that lack a value at that age; that inflates it for old videos.
* **Engagement** is beta-binomial empirical Bayes with views as trials. Views
  are not unique viewers and one viewer can like and comment, so the binomial
  model is an approximation; the beta prior absorbs over-dispersion.
  ``interactions_rate`` sums only the components the platform reports, so it
  is comparable within deal x platform, not across platforms.
* **Missing measures.** No watch time, retention, completion rate,
  impressions/reach-vs-views distinction or traffic source is in the export.
"""

from .build import FLAG_COLUMNS, OutcomeParams, build_outcomes
from .io import coerce_tables, load_tables

__all__ = ["FLAG_COLUMNS", "OutcomeParams", "build_outcomes", "coerce_tables", "load_tables"]
