# Vision, gaps and demo plan

For every agent and the frontend teammate (owner request, 2026-09-26, written by the
reporter session tyler-5b). This page states the end product, what is real today,
what is not, and how to demo it. It changes no contract or rule: numbers and status
live in [`STATUS.md`](STATUS.md), decision rules in
[`PREREGISTRATION.md`](PREREGISTRATION.md), the pipeline in
[`PRODUCT_PIPELINE.md`](PRODUCT_PIPELINE.md), change rules in [`TEAM.md`](TEAM.md).

## The end product

Paste or upload a clip. The system shows, second by second, how an average viewer's
cortex is predicted to respond, and where the clip is strong or weak. Next to it, a
small model trained on our own clips' real results gives an in-context outlook: how
this clip likely ranks against similar clips in the same deal and platform.

- **Brain analysis.** TRIBE v2 is Meta's model and fixed. Output: timeline, brain
  map, moments. Real on real clips today.
- **Performance outlook.** Ridge models over brain features, extractor embeddings and
  metadata, trained by us. This is the part still being built.
- **The scientific hook.** The pre-registered test is whether the brain features beat
  the raw extractor embeddings (arm BE vs E). The UI may make the neural claim only
  when that test is `supported`. Until then it shows the brain analysis without one.

## Where we are (2026-09-26 ~22:00 UTC; see STATUS.md for live numbers)

| Piece | State |
|---|---|
| Brain analysis on real clips | Works. 40 real bundles; bf16 path is ~5x stock and passed every gate |
| Study data (1,500 clips, lockbox sealed) | Running on the pods. Stage 1 expected ~01:00-02:30 UTC |
| Training and evaluation code | Written and pre-registered; waiting on data |
| Preliminary smoke-test model (`model-prelim-v0`) and `data-frontend40-v2` | In progress (tyler-b2). Not stage 1, never implies a GO |
| Frontend | Contract and fixtures exist |
| Live "paste a link" flow | **Does not exist.** No API, queue or host. "Live" means an operator-run pod |
| Validated model | **Not yet.** Needs stage 2 (all ~7.9k clips) and the one-time lockbox opening |

## What stands between this and a business

Mostly owner decisions (also in PRODUCT_PIPELINE.md section 6):

1. **Licence.** TRIBE v2 is CC-BY-NC-4.0, so research and demo only. Any paid or
   client-facing use is blocked until this is resolved. Biggest item.
2. **Link fetching** from TikTok/Instagram/YouTube (ToS, copyright). Uploads of our
   own clips avoid it.
3. **Hosting.** Warm GPU (~$8-26/day idle) vs serverless (cold starts), plus an API
   host outside the analysis server.
4. **Validation.** Stage 2 needs about $50 more RunPod budget. Retention data from
   company-owned accounts is the only route to claims like "attention drops here".
5. **Cartel data in the product.** Keep isolated, or decide explicitly where deal
   context comes from.
6. **The effect may be small.** Literature in our notes (brain features explain
   ~0.1-2.4% of variance; a TRIBE vs most-replayed study was null) means we must not
   pitch "we predict virality". Pitch the brain analysis and the honest test.

## Demo plan

Use pre-computed bundles; the shot list in PRODUCT_PIPELINE.md section 5 is the
base. Live only if a warm pod is rehearsed, and cut it in the edit.

1. Upload an owned clip; job states tick `queued -> fetching -> extracting ->
   predicting -> done` (mocked or pre-recorded).
2. Player and timeline: scrub; channels and brain tile follow the playhead. Hold on
   "TRIBE v2 prediction · average subject".
3. Brain map hover with region tooltip (within-clip z).
4. Moments: one observation and one dashed "untested edit hypothesis".
5. Outcome card in its honest state (`not_trained`, or `preliminary`/`research_preview`
   with its caption).
6. Second clip from another deal to show context switching. Never compare z across
   clips.
7. Closing card: "Research preview · non-commercial (TRIBE CC-BY-NC) · predictions,
   not measurements".

To make it land:

- **Hero clips from the study set**, not the 40 wiring clips: English, clear speech,
  in neither lockbox file.
- **Best moment:** a predicted-vs-actual slide on train clips, out-of-fold predictions
  only, never on either lockbox, stamped as a preview.
- **Keep every required label on screen** ("Illustrative", "Untested hypothesis",
  "Research preview", "Preliminary — not validated"). They protect the claim and read
  as rigour.
- **Embedding the source clip:** `data-frontend40-v2` (tyler-b2) carries `video_id`,
  `platform`, `video_link`, `is_lockbox` in `index.json`. Never show outcomes for
  lockbox clips. Company-owned footage stays on the owner's server; the repo ships no
  footage.

## Who does what

- **tyler-b2:** prereg, `fit_models`, preliminary model, `data-frontend40-v2`,
  `docs/MODEL.md`; owns `STATUS.md`.
- **tyler-29:** pods, `run_queue.sh` / `drive_pod.sh` and the single $17 watchdog.
- **tyler-3f:** time-resolved moments (`moments_pop`, prereg family 6: M1–M3 and
  the M4 "good vs bad clip" contrast); freezes the moment norms before the stage-1 fit.
- **Frontend agent:** build against `nvi.analysis.v0.2` and the fixtures; start with
  the `not_trained` state; swap in v2 bundles when they land.
- **tyler-5b (reporter):** status for the owner; does not touch prereg, fit or releases.
