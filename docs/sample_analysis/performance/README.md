# `performance` block fixtures (SYNTHETIC)

**Every file here is synthetic.** The worker outputs are dry-run stubs, the deals
(`deal-a`, `deal-b`), accounts and outcomes are made up, and the models were fit
on a planted signal. The numbers are meaningless: they show shape and states only.
Each file says so in its top-level `_fixture` field, and every block's
`warnings[0]` starts with `SYNTHETIC`. Show a SYNTHETIC banner when you render one.

The `performance` block is a **proposal** for `nvi.analysis.v0.3`. It is not in
`analysis.schema.json` yet. Its draft schema is
[`../../performance.schema.draft.json`](../../performance.schema.draft.json), and
the rules and wording are in [`../../PRODUCT_PIPELINE.md`](../../PRODUCT_PIPELINE.md) §3–4.

## Format and how they were made

Each file is `{"_fixture": …, "_scenario": …, "performance": {…}}`.
`performance` is exactly what `tools/predict.py` wrote, unchanged. In the v0.3
bundle it will sit at `analysis.json` → `performance`.

The files come from running the real `tools/predict.py` CLI on the synthetic
study that `tests/test_predict.py` builds: 200 clips, 2 deals × 2 platforms, and
`fit_models.py --save-model`. To regenerate:

```bash
.venv/bin/python tools/make_performance_samples.py --out docs/sample_analysis/performance
```

It takes about half a minute on CPU and needs no network, pods or DB. Some fields
change between runs:

- `model_version` (`perf-stack-BE_v1+<date>.<commit>`);
- `provenance.model_git`;
- the sha256 values.

`tests/test_performance_samples.py` validates every file against the draft schema.

## Files → UI state

| File | `model_status` | Drives | Notes |
|---|---|---|---|
| `not_trained.json` | `not_trained` | **Build this first.** Brain analysis only, plus a card "Performance model not trained yet". No numbers and no skeleton digits | No model is saved: tonight's honest state. `model_version` is null |
| `not_trained_go_failed.json` | `not_trained` | Same card as above | A model was fit but failed the pre-registered stage-1 GO rule. To reach this state, the fixture model's manifest GO metrics were **edited** to a failing value, as in `test_not_trained_has_no_numbers`; `predict.py` is unchanged. `reason` is technical: put it behind "See evidence" and keep the card text plain |
| `research_preview.json` | `research_preview` | Percentile, range, context, drivers, and an orange **"Research preview"** badge. Also the account line (`percentile_account`, `account_level: true`) | Stage-1 model; the lockbox is not scored. The account percentile appears only because the fixture fits with `--min-account-n 10`; production uses 40, so expect it rarely |
| `research_preview_no_account.json` | `research_preview` | Same, deal × platform only (`account_level: false`) | No account given |
| `validated.json` | `validated` | Same as above with a blue **"Validated"** badge; `validation.scheme: "lockbox"` | The fixture's lockbox-scored model really clears the rule (the served model's lockbox ρ CI lower bound > 0). Still relative, still a card, still with its range |
| `out_of_scope_too_long.json` | `out_of_scope` | Card showing `reason` ("Duration 120.0 s is outside the model's 5-90 s training scope"). No numbers | |
| `out_of_scope_no_audio.json` | `out_of_scope` | Card showing `reason` ("No audio track: outside the model's training scope") | |
| `train_oof.json` | `research_preview` | Scored card; `clip_in_training: "train_oof"`, `retrospective: true`, `drivers: []` | A training clip gets its out-of-fold prediction and is ranked without itself. With no drivers, hide the driver bars and show the "drivers omitted" warning |
| `retrospective.json` | `research_preview` | Scored card with an "Already posted: retrospective prediction" note; drivers shown | A new clip (not a training clip) with `posted_at` set |
| `lockbox.json` | `research_preview` | Scored card; `clip_in_training: "lockbox"` | **Never show an observed outcome for it**; the block contains none. `percentile_deal_platform` is `0.0`, a real edge case (P0): render it, don't treat it as missing |

Every scored sample has `confidence: "low"` and the warning "Low confidence: this deal has few reference
clips on this platform", because the fixture's references are 65-67 clips (fewer than 100). Show those as
the muted card with a warning icon. `confidence: "medium"` happens only when the status is `validated` and
there are at least 100 references; no fixture reaches it. Every scored sample also has a `reach` block and
the warning "Reach is mostly driven by account, platform and timing."

`brain_claim` is `not_tested` (no model) or `directional` in these files; the fixture cannot reach
`supported` or `not_supported`. Only `supported` allows "the brain response predicts". For `directional`,
at most say "content features and predicted brain response contributed".

## Labels the UI must show (PRODUCT_PIPELINE.md §4, STATUS.md)

- Every performance number carries three things together:
  - the context: deal label × platform and `reference_n`, e.g. "Predicted engagement: ranks around
    P{round(100 × percentile_deal_platform)} among {deal_label}'s TikTok clips ({reference_n} clips)";
  - the range, from `likely_range`: "Similar clips landed between P{lo} and P{hi}";
  - the status badge: "Research preview" or "Validated".
- "Research preview · correlational · not a guarantee". Never "virality score", "% chance to go viral",
  "will perform", "guaranteed" or "optimised".
- Until the status is `validated`, the percentile stays inside a card. Never make it a hero number.
- Driver `contribution` values set bar length only. Never print them. They are model attributions, not causes:
  "Content features and predicted brain response contributed", never "the brain response caused this".
  `brain_detail[].channel` matches the analysis `channels[].key`, or is `"other"`.
- Reach, when shown, adds "mostly driven by account, platform and timing".
- Show `warnings`; any warning or `confidence: "low"` makes the card muted, with a warning icon.
- For `not_trained` and `out_of_scope`: no performance numbers. That means no percentile, range, score,
  driver bars or skeleton digits. The schema forbids them, and `tests/test_performance_samples.py`
  checks that no JSON number appears anywhere in those blocks. The `reason` text may still mention a
  duration.
- Show `brain_claim` only next to a scored card. The GO-failed and out-of-scope samples still carry
  `directional`; showing it on a "not trained" card would mislead.
- Keep the global labels too:
  - "TRIBE v2 prediction · average subject" on brain visuals;
  - "Illustrative" on any number not produced by a real model (all of these files);
  - "Research preview · non-commercial (TRIBE CC-BY-NC)".
