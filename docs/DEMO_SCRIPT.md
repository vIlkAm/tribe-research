# Demo script (judges, about 4 minutes)

Speaker script for the hackathon demo. The slides are generated outside git
(`/tmp/viralbrain-media/deck/`), and so are the chart and clip videos from
`tools/make_demo_media.py` (`/tmp/viralbrain-media/`). The footage stays out of git.

Wording rules for anything said or shown:

- The brain visuals are TRIBE v2 predictions for an average viewer. No one was
  scanned for this project.
- The brain response does not predict views or virality. The pre-registered
  test was a no-GO.
- A clip's swing around its account's usual includes luck, timing and the
  algorithm, not only the video.
- The views-grouped chart is exploratory.
- TRIBE v2 was trained on 25 people and 452 h. The 720 people and 1,117 h are
  the paper's full dataset, which also tests on new people. Never say "trained
  on 720 people".
- Cite the paper's numbers and link its repo, weights and demo. Don't show the
  paper's figures.

## 1. Title (15 s)

"This is ViralBrain. We run Meta's TRIBE v2 brain model over short-form clips
from our clipping agency. It shows, second by second, how an average viewer's
brain is predicted to respond, and we check that against each clip's real
numbers."

## 2. The model (30 s)

"TRIBE v2 is from Meta FAIR, published this year. It reads a clip's video,
audio and text and predicts fMRI. The paper uses over a thousand hours of scans
from 720 people. On the best dataset, its prediction is about twice as good as
a typical single person's scan at predicting the group-average brain. It models
perception, not behaviour: it doesn't know who keeps watching or shares."

## 3. What we built (20 s)

"Each clip goes through TRIBE v2 on a GPU. We reduce the prediction to seven
brain signals per second. Then we compare every second with the same second of
1,264 of our library clips of similar length."

## 4. Live: the app (60 s)

Open the Connor "Diamond Gym" clip (331.8K views, 34× the account's usual).

- Press play. "The brain and the strip follow the video."
- Point at the strip. "Orange is above a typical clip at that second, blue is
  below. This opening sits high, then the middle drops."
- Open **Compare**. "Here is a clip that did great next to one that did badly.
  They're hand-picked to show the reading, and single clips go either way. So
  we checked the whole library."

## 5. What drives views (20 s), chart `02_views_split`

"Within the same client and platform, most of the difference between clips is
a clip's swing around its own account's usual, not the account's size. That
swing includes luck, timing and the algorithm, as well as the video."

## 6. The honest result (30 s), chart `03_stage1_result`

"Before running anything, we wrote down a test. Does adding the brain response
rank clips better than basic clip and account information? It added under a
hundredth, and the bar was two hundredths. So it's a no-GO, and the app shows
no performance score. The features TRIBE reads did help with views against the
account's usual (+0.07)."

## 7. It points the right way (30 s), chart `01_views_vs_usual`

"Group clips by whether they beat their own account's usual views. The top
third has a higher predicted response than the bottom third, over the first 30
seconds and in the opening. It's small, about a tenth of a standard deviation,
and exploratory. It still holds when we resample whole accounts."

## 8. Where it stands (20 s)

"Today ViralBrain is a perception lens for editors. Forecasting needs more
data: we used about 1,500 of nearly 8,000 eligible clips. Next, we'll scale
up, tune per account and test the signal in TRIBE's input features that held up."

## 9. Close (10 s)

"TRIBE v2 is Meta FAIR's model under a non-commercial licence. This is a
non-commercial project, not affiliated with Meta. Thanks."

## If a judge asks

- **"So does it predict virality?"** No. The pre-registered test said no-GO.
  What we see is a small exploratory difference in the direction you'd expect.
- **"Why is the bad clip's line high?"** One clip is one draw. The whole-library
  groups are what count, and they're on the Results page.
- **"Whose brain is that?"** Nobody's. It's the model's prediction for an
  average viewer.
