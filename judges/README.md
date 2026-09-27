# ViralBrain: predicted brain response for short-form clips (hackathon demo)

ViralBrain runs Meta's **TRIBE v2** brain-encoding model over short-form videos (TikTok, Instagram Reels,
YouTube Shorts). It shows, second by second, how strongly an average viewer's brain is predicted to respond,
and compares that with a library of about 1,264 real clips and their real platform numbers.

## Key results

- **1,486 real clips** from 12 clients on TikTok, Instagram Reels and YouTube Shorts were run through TRIBE v2.
  Every second of a clip is compared with the same second of 1,264 library clips of similar length.
- **Clips that beat their account's usual views had a higher predicted brain response** than clips below it:
  +0.09 library standard deviations over the first 30 seconds (95% range +0.03 to +0.15), and +0.09 in the
  opening 4 seconds (+0.02 to +0.17). That is 415 vs 415 clips within 12 clients, with the grouping fixed before
  it was computed. It still holds when whole accounts are resampled. Small, and in the direction you would expect
  (Results page).
- Across all 1,143 library clips, those with more seconds above the typical line got more views within the same
  client and platform (rank correlation +0.14, 95% range +0.06 to +0.22). Exploratory: our stricter split-half
  check on the Library page does not confirm it on its own.
- The pre-registered test was a **no-GO**, so the app shows no performance score (see Honest results). The brain
  line is a perception lens for editing, not a views forecast.

## How to open it (no GPU or internet needed)

Everything is precomputed. TRIBE v2 needs a large GPU (we used rented cloud GPUs), so the analyses were run in
advance and are included here as static files.

1. Unzip this folder.
2. In a terminal inside the folder, run **`python3 serve.py`** (Python 3.8+, nothing to install).
   Alternative: `npx serve .` (Node.js).
3. Open **http://127.0.0.1:8000/** in Chrome, Edge or Firefox.

Opening `index.html` directly from the file system does not work (browsers block the data files); use the
small server above. It only listens on your own computer.

## What to look at

- **Main view:** pick a clip from the clip menu (grouped *Did great*, *Typical*, *Did badly*). The video plays in
  sync with the predicted brain response and the "response strip": each second of the clip compared with the
  same second of the library clips.
- **Compare:** two clips side by side (starts with a clip that did great and one that did badly).
- **Library:** all demo clips with their real views, and what holds across the whole library (see below).
- **Results:** start here for the evidence: the predicted response of clips above vs below their account's usual
  views, and the pre-registered test and its result.

The **Analyze video** / upload flow needs the GPU backend, so it does not run in this offline copy.

## The clips (28)

Real clips from our clipping agency's library, with each post's real views, likes and comments from the
platform. Groups use views the post got:

| Group | Rule | Clips shown |
|---|---|---|
| Did great | 300,000+ views | 8 (every clip of 45 s or less with a clean transcript that qualified) |
| Typical | 5,000-15,000 views | 10 |
| Did badly | under 700 views and below the account's usual (at least 100 views) | 10 |

Two were hand-picked by us (a Connor clip that did great, a Polymarket TikTok that did badly); the rest follow a
fixed rule (hash order, at most two per client per group). Transcripts were screened for explicit language.
Some TikTok clips were re-encoded from HEVC to H.264 only so that every browser can play them
(`clips/TRANSCODED.json`).

## Honest results

- **The pre-registered test was a no-GO.** Adding the predicted brain response to basic information about the clip
  and its account did not rank clips any better on the pre-registered target (likes and comments per view). The
  brain line is **not a views forecast**.
- **Library statistics** (Library page): each pattern was looked for on half of the accounts and checked once on
  the other half. Most of the differences in views within a client and platform are a clip's swing around its
  own account's usual (luck, timing and the algorithm included), not account size. Across all clips, a higher
  share of seconds above the typical brain line went with more views (r +0.14, 95% range +0.06 to +0.22) and with
  beating the account's usual (r +0.11); the strict split-half check has not confirmed it yet. One pattern was
  confirmed: longer clips get more interactions per view within the same account (small, and not a cause: longer
  clips also get fewer views).
- **Grouped by views against the account's usual** (Results page; exploratory, grouping fixed before it was
  computed): within each client, the third of clips above their account's usual had a higher predicted brain
  response than the third below it, over the first 30 seconds (+0.09 library standard deviations, 95% range +0.03
  to +0.15) and in the opening 4 seconds (+0.09, +0.02 to +0.17); both hold when whole accounts are resampled.
  Small, but in the direction you would expect. The stricter Library test (rank correlation within the same
  account, confirmed on held-out accounts) does not confirm the opening on its own, so this is a lead, not a
  finding.
- So the tool shows *where* attention is predicted to rise and fall inside a clip, which is useful for editing
  review. It does not tell you whether a clip will go viral.

## The brain model (TRIBE v2)

From the paper (d'Ascoli et al., "A foundation model of vision, audition, and language for in-silico neuroscience",
FAIR at Meta, 2026; code https://github.com/facebookresearch/tribev2):

- Evaluated on 1,117 hours of fMRI from 720 people across 8 datasets (trained on 25 people and 452 hours).
- Reads video (V-JEPA2), audio (w2v-BERT) and text (Llama 3.2) features of a clip.
- On the 7T HCP set, its prediction of the group-average brain response reached R ≈ 0.4, about twice the median
  single person's scan as a predictor of the group average. That is why we use its average-viewer prediction.
- The brain responds about 5 s after what caused it; our timelines shift the prediction back to line up with the video.
- The paper's own limits: fMRI time scale (seconds), and the brain is modelled as a passive observer, not an agent
  producing behaviour. That fits our result: it describes perception, not whether people keep watching or share.

## Important notes

- **Predictions, not measurements.** Every brain visual is TRIBE v2's prediction for an *average* viewer; no one's
  brain was scanned. The 3D brain is a presentation layer; the 2D atlas and the signals are the data.
- **Training data.** Meta's TRIBE v2 was not trained on these clips. Our own library statistics and the
  pre-registered test did use them (the test kept a separate sealed set of clips, none shown here).
- **Licence.** TRIBE v2 (Meta FAIR) is licensed **CC BY-NC 4.0**. This is a
  non-commercial hackathon project, not a product. The clips belong to the agency and are included only for
  judging.
