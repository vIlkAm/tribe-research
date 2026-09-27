#!/usr/bin/env python3
"""Performance tiers for the INTERNAL tailnet demo: a tiered demo set and "what great clips have in common".

    .venv/bin/python tools/library_tiers.py --patterns results/library/patterns.json \\
        --demo results/demo/library_demo.json

INTERNAL ONLY (observed outcomes of training clips, deal names, source files). Never in a release; never a lockbox
clip. Rules are fixed here, before any result is looked at (TIER_RULE, DEMO_RULE, FEATURES).

Tier (per post): the post's ``reach_rel_local`` (views at a fixed age vs the same account's clips posted around
the same time; displayed as "N× the account's recent usual") as a percentile among every post of the same deal
and platform with a ``reach_rel_local`` and empty ``dq_flags`` (reference >= MIN_REF posts). great: >= 80 and
above the account's usual (> 1×); typical: 40-60; bad: <= 20 and below usual (< 1×). A clip's tier is its own
post's (the manifest ``source_name``, the file TRIBE ran on).

Patterns (exploratory, not pre-registered): over the 1,264 library (train) clips with a tier, weighted by
1 / ``incl_prob`` (the study set oversampled reach tails), features winsorised at the 1st/99th percentile and
centred within deal|platform; great - bad difference of weighted means, 95 % bootstrap CI over contents (2,000,
seed 20260926), two-sided bootstrap p, BH q over the 12 features. "Reliable" = q <= 0.10 and the CI excludes 0.

Demo: eligible as in ``tools/select_demo_clips.py`` (English, 20-60 s, >= 1.5 words/s, >= -30 dB), clean
transcript (``tools/language_screen.py``), <= 45 s, empty ``dq_flags``, a tier, a source file on this server.
Per tier, sha256("20260926:<id>") order, the first clip of each new deal, 3 per tier. Never re-rolled for how the
brain line looks; a content veto after viewing takes the next clip and is recorded (``--veto``).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_library_profile as blp  # noqa: E402
import build_moments_pop as bmp  # noqa: E402
import language_screen  # noqa: E402
import select_demo_clips as sdc  # noqa: E402
import select_example_pair as sep  # noqa: E402
from tribe_research.brain import moments_pop as mp  # noqa: E402

SEED = 20260926
MIN_REF = 100
TIERS = {"great": (80.0, 100.0), "typical": (40.0, 60.0), "bad": (0.0, 20.0)}
TIER_LABEL = {"great": "Did great", "typical": "Typical", "bad": "Did badly"}
DEMO_MAX_S, DEMO_PER_TIER = 45.0, 3
N_BOOT, WINSOR, Q_MAX = 2000, (1.0, 99.0), 0.10
TIER_RULE = (
    "Post percentile of reach_rel_local (views vs the account's recent usual) among every post of the same deal and "
    f"platform with reach_rel_local and empty dq_flags (>= {MIN_REF} reference posts). great >= 80 and > 1x usual; "
    "typical 40-60; bad <= 20 and < 1x usual. A clip's tier is its own post's (manifest source_name).")
# Stated 2026-09-27 before any pattern was computed on them (owner: "look at absolute numbers as well"). "both" is
# the proposed display rule; "relative" and "absolute" are reported alongside it as sensitivity checks, whatever
# they show. Absolute = 7-day views (reach_basis views_7d only, so every post is compared at the same age) as a
# percentile among the same deal x platform's 7-day posts.
TIER_BASES = ("relative", "absolute", "both", "views")
# Owner rule, 2026-09-27 (hackathon demo): the views the post actually got (views_final, the number the metrics row
# shows), with "below its account's usual" on bad so a bad video is not just a small account.
VIEWS_GREAT, VIEWS_TYPICAL, VIEWS_BAD = 300_000, (5_000, 15_000), 700
TIER_RULES = {
    "relative": TIER_RULE,
    "absolute": ("Post percentile of 7-day views among the same deal and platform's posts with 7-day views and empty "
                 f"dq_flags (>= {MIN_REF}). great >= 80; typical 40-60; bad <= 20."),
    "both": ("great: top 20% on views vs the account's usual (and above usual) AND top half on 7-day views; bad: "
             "bottom 20% vs usual (and below usual) AND bottom half on 7-day views; typical: 40-60 vs usual AND "
             "25-75 on 7-day views. Percentiles within the same deal and platform; 7-day views only."),
    "views": (f"Owner rule (2026-09-27), on the views the post got (last snapshot, empty dq_flags): great >= "
              f"{VIEWS_GREAT:,}; typical {VIEWS_TYPICAL[0]:,}-{VIEWS_TYPICAL[1]:,}; bad < {VIEWS_BAD:,} and below "
              "the account's recent usual."),
}
TIER_PLAIN = {
    "relative": {"great": "Top 20% of this deal's posts on this platform, above the account's usual",
                 "typical": "Middle (40th-60th percentile)", "bad": "Bottom 20%, below the account's usual"},
    "absolute": {"great": "Top 20% of this deal's posts on this platform by 7-day views",
                 "typical": "Middle (40th-60th percentile) by 7-day views",
                 "bad": "Bottom 20% by 7-day views"},
    "both": {"great": "Well above the account's usual (top 20%) and in the top half of the deal's posts by 7-day views",
             "typical": "Around the account's usual and in the middle of the deal's posts by 7-day views",
             "bad": "Well below the account's usual (bottom 20%) and in the bottom half by 7-day views"},
    "views": {"great": f"{VIEWS_GREAT:,}+ views", "typical": f"{VIEWS_TYPICAL[0]:,}-{VIEWS_TYPICAL[1]:,} views",
              "bad": f"Under {VIEWS_BAD:,} views and below the account's usual"},
}
CANDIDATES_RULE = ("Every demo-eligible clip in a tier (eligible as in tools/select_demo_clips.py, clean transcript, "
                   "<= 45 s, source file present), most views first: a browse set for the owner to choose from, "
                   "not a sample.")
OWNER_RULE = ("Hand-picked by the owner for the demo (2026-09-27), not by a rule. Each clip still passes the demo "
              "eligibility, transcript language, lockbox and tier checks, and its tier is its own computed tier. The "
              "library statistics are computed over every library clip, not these.")
DEMO_RULE = (
    "Eligible as in tools/select_demo_clips.py; clean transcript (tools/language_screen.py); <= 45 s; empty "
    "dq_flags; a tier; source file present. Per tier, sha256('20260926:<id>') order, first clip of each new deal, "
    "3 per tier. Not re-rolled on the brain line; vetoes are recorded and take the next clip.")
# key, plain label, unit, group
FEATURES = [
    ("length_s", "Clip length", "s", "editing"),
    ("cuts_per_min", "Cuts per minute", "per min", "editing"),
    ("first_cut_s", "Time to first cut", "s", "editing"),
    ("first_word_s", "Time to first word", "s", "editing"),
    ("words_per_s", "Words per second", "per s", "editing"),
    ("brain_overall", "Predicted response, whole clip", "percentile", "brain"),
    ("brain_above_typical", "Share of seconds above the typical line", "% of seconds", "brain"),
    ("brain_opening", "Predicted response, opening (first 4 s)", "percentile", "brain"),
    ("brain_middle", "Predicted response, middle", "percentile", "brain"),
    ("brain_peak", "Predicted response, strongest 3 s", "percentile", "brain"),
    ("brain_low_share", "Share of low-response seconds (bottom 20%)", "% of seconds", "brain"),
    ("brain_ending", "Predicted response, ending (last 3 s)", "percentile", "brain"),
]


# ── tiers ────────────────────────────────────────────────────────────────


def post_tiers(o: pd.DataFrame, basis: str = "relative") -> pd.DataFrame:
    """Per post: percentile (0-100, mid-rank) of reach_rel_local within deal|platform (views_pct, n_ref), of 7-day
    views among the stratum's 7-day posts (abs_pct, n_abs), and the tier under ``basis`` (or None)."""
    if basis not in TIER_BASES:
        raise SystemExit(f"tier basis {basis!r} not in {TIER_BASES}")
    ref = o[o["reach_rel_local"].notna() & o["dq_flags"].isna() & o["deal_id"].notna()].copy()
    ref["stratum"] = ref["deal_id"].astype(str) + "|" + ref["platform"].astype(str)
    g = ref.groupby("stratum")["reach_rel_local"]
    ref["n_ref"] = g.transform("size")
    ref["views_pct"] = 100.0 * (g.rank(method="average") - 0.5) / ref["n_ref"]
    d7 = ref["reach_basis"].eq("views_7d") if "reach_basis" in ref else pd.Series(False, index=ref.index)
    g7 = ref[d7].groupby("stratum")["reach_log"]
    ref["n_abs"] = g7.transform("size").reindex(ref.index)
    ref["abs_pct"] = (100.0 * (g7.rank(method="average") - 0.5) / ref.loc[d7, "n_abs"]).reindex(ref.index)
    ref["views_7d_abs"] = np.where(d7, np.expm1(ref["reach_log"]) if "reach_log" in ref else np.nan, np.nan)
    x, r, a = ref["reach_rel_local"], ref["views_pct"], ref["abs_pct"]
    ok_r = ref["n_ref"] >= MIN_REF
    ok_a = (ref["n_abs"] >= MIN_REF) & a.notna()
    rel = {"great": ok_r & (r >= TIERS["great"][0]) & (x > 0), "typical": ok_r & r.between(*TIERS["typical"]),
           "bad": ok_r & (r <= TIERS["bad"][1]) & (x < 0)}
    ab = {"great": ok_a & (a >= TIERS["great"][0]), "typical": ok_a & a.between(*TIERS["typical"]),
          "bad": ok_a & (a <= TIERS["bad"][1])}
    both = {"great": rel["great"] & ok_a & (a >= 50), "typical": rel["typical"] & ok_a & a.between(25, 75),
            "bad": rel["bad"] & ok_a & (a <= 50)}
    v = ref["views_final"] if "views_final" in ref else pd.Series(np.nan, index=ref.index)
    views = {"great": v >= VIEWS_GREAT, "typical": v.between(*VIEWS_TYPICAL), "bad": (v < VIEWS_BAD) & (x < 0)}
    tier = pd.Series(None, index=ref.index, dtype=object)
    for t, m in {"relative": rel, "absolute": ab, "both": both, "views": views}[basis].items():
        tier[m] = t
    ref["tier"] = tier
    return ref.set_index("id")[["stratum", "n_ref", "views_pct", "n_abs", "abs_pct", "views_7d_abs", "tier",
                                "reach_rel_local"]]


# ── patterns ─────────────────────────────────────────────────────────────


def clip_features(e: blp.Entry, pct: np.ndarray, edit: pd.Series | None, words: list[dict], dur: float) -> dict:
    p = blp.to_grid(e, pct)
    fin = np.isfinite(p)
    n = len(p)
    roll = [np.mean(p[i:i + blp.PEAK_S]) for i in range(0, n - blp.PEAK_S + 1) if fin[i:i + blp.PEAK_S].all()]
    mid = p[int(mp.HOOK_S):n - blp.FINISH_S]
    return {
        "length_s": dur,
        "cuts_per_min": float(edit["edit_cuts_per_min"]) if edit is not None else np.nan,
        "first_cut_s": float(edit["edit_first_cut_s"]) if edit is not None else np.nan,
        "first_word_s": float(edit["edit_speech_onset_s"]) if edit is not None else np.nan,
        "words_per_s": len(words) / dur if dur > 0 else np.nan,
        "brain_overall": blp._nanmean(p),
        "brain_above_typical": 100.0 * float((p[fin] > 50).mean()) if fin.any() else np.nan,
        "brain_opening": blp._nanmean(p[:int(mp.HOOK_S)]),
        "brain_middle": blp._nanmean(mid) if len(mid) else np.nan,
        "brain_peak": float(max(roll)) if roll else np.nan,
        "brain_low_share": 100.0 * float((p[fin] < blp.DEAD_PCT).mean()) if fin.any() else np.nan,
        "brain_ending": blp._nanmean(p[max(n - blp.FINISH_S, 0):]),
    }


def wmean(x: np.ndarray, w: np.ndarray) -> float:
    ok = np.isfinite(x) & (w > 0)
    return float(np.sum(x[ok] * w[ok]) / np.sum(w[ok])) if ok.any() else float("nan")


def wauc(xg: np.ndarray, wg: np.ndarray, xb: np.ndarray, wb: np.ndarray) -> float:
    """Weighted P(great clip's value > bad clip's value), ties 1/2: 0.5 = a coin flip."""
    g, b = np.isfinite(xg), np.isfinite(xb)
    xg, wg, xb, wb = xg[g], wg[g], xb[b], wb[b]
    if not len(xg) or not len(xb):
        return float("nan")
    gt = (xg[:, None] > xb[None, :]) + 0.5 * (xg[:, None] == xb[None, :])
    ww = wg[:, None] * wb[None, :]
    return float((gt * ww).sum() / ww.sum())


def bh(p: np.ndarray) -> np.ndarray:
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    q[order] = np.minimum.accumulate((p[order] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1.0)


def patterns(tab: pd.DataFrame, rng: np.random.Generator, n_boot: int = N_BOOT) -> list[dict]:
    """tab: one row per content with tier, stratum, w and the FEATURES columns."""
    tab = tab[tab["tier"].notna()].reset_index(drop=True)
    w = tab["w"].to_numpy(float)
    tier = tab["tier"].to_numpy()
    strat = tab["stratum"].to_numpy()
    rows, pvals = [], []
    idx = np.arange(len(tab))
    boots = [rng.choice(idx, len(idx)) for _ in range(n_boot)]
    for key, label, unit, group in FEATURES:
        x = tab[key].to_numpy(float).copy()
        lo, hi = np.nanpercentile(x, WINSOR)
        x = np.clip(x, lo, hi)
        xc = x.copy()
        for s in np.unique(strat):
            m = strat == s
            xc[m] = x[m] - wmean(x[m], w[m])
        means = {t: wmean(x[tier == t], w[tier == t]) for t in TIERS}

        def diff(ix):
            g, b = ix[tier[ix] == "great"], ix[tier[ix] == "bad"]
            return wmean(xc[g], w[g]) - wmean(xc[b], w[b])
        def auc(ix):
            g, b = ix[tier[ix] == "great"], ix[tier[ix] == "bad"]
            return wauc(xc[g], w[g], xc[b], w[b])
        point = diff(idx)
        d = np.array([diff(b) for b in boots])
        d = d[np.isfinite(d)]
        ci = np.percentile(d, [2.5, 97.5])
        a = np.array([auc(b) for b in boots])
        a_ci = np.nanpercentile(a, [2.5, 97.5])
        p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
        pvals.append(p)
        rows.append({"key": key, "label_plain": label, "unit": unit, "group": group, "mean": means,
                     "diff_great_minus_bad": {"point": point, "lo": float(ci[0]), "hi": float(ci[1])},
                     "p_boot": p, "coin_flip": {"point": auc(idx), "lo": float(a_ci[0]), "hi": float(a_ci[1])}})
    q = bh(np.array(pvals))
    for r, qq in zip(rows, q):
        d = r["diff_great_minus_bad"]
        reliable = qq <= Q_MAX and (d["lo"] > 0 or d["hi"] < 0)
        r["q"], r["verdict"] = float(qq), "weak_tendency" if reliable else "no_reliable_difference"
        u = r["unit"]
        if reliable:
            r["plain"] = (f"Weak tendency (exploratory): clips that did great are {'higher' if d['point'] > 0 else 'lower'} "
                          f"on this than clips that did badly by about {abs(d['point']):.2g} {u} (same deal and "
                          "platform), but it does not sort single clips.")
        else:
            r["plain"] = "No reliable difference between clips that did great and clips that did badly."
    return rows


def account_check(tab: pd.DataFrame, rng: np.random.Generator, n_boot: int = 1000) -> dict:
    """Is a great-vs-bad difference one account's style? Per tier: clips, accounts, the top account's share; then
    the pattern table again without the account with the most tiered clips, and centred within account (accounts
    with >= 2 tiered clips) instead of within deal x platform."""
    t = tab[tab["tier"].notna()]
    by_tier = {k: {"n_clips": int(len(g)), "n_accounts": int(g["account"].nunique()),
                   "top_account_share": round(float(g["account"].value_counts(normalize=True).iloc[0]), 3)}
               for k, g in t.groupby("tier")}
    top = t["account"].value_counts().index[0]

    def slim(rows):
        return [{"key": r["key"], "diff": round(r["diff_great_minus_bad"]["point"], 3),
                 "lo": round(r["diff_great_minus_bad"]["lo"], 3), "hi": round(r["diff_great_minus_bad"]["hi"], 3),
                 "q": round(r["q"], 3), "verdict": r["verdict"]} for r in rows]
    drop = t[t["account"] != top]
    multi = t[t["account"].map(t["account"].value_counts()) >= 2].assign(stratum=lambda d: d["account"])
    return {"by_tier": by_tier,
            "drop_top_account": {"top_account_share_of_tiered": round(float((t["account"] == top).mean()), 3),
                                 "features": slim(patterns(drop, rng, n_boot))},
            "within_account": {"n_clips": int(len(multi)), "n_accounts": int(multi["account"].nunique()),
                               "features": slim(patterns(multi, rng, n_boot))}}


def interpreter_line(f: dict) -> str:
    """Stated from the numbers, whatever they are: where each tier sits against the typical line."""
    m, c = f["mean"], f["coin_flip"]
    head = (f"The typical line is the middle of the library at each second, so an average clip is above it about half "
            f"the time. Clips that did great are above it in {m['great']:.0f}% of seconds, typical clips in "
            f"{m['typical']:.0f}% and clips that did badly in {m['bad']:.0f}%. ")
    if f["verdict"] == "weak_tendency":
        return head + (f"Pick one great and one bad clip at random and the great one has more seconds above the line "
                       f"{100 * c['point']:.0f}% of the time (50% would be a coin flip). So a line well below the "
                       "middle leans slightly towards a weaker clip, but being above it does not mean a clip will do "
                       "well: it is a weak tendency, not a verdict on any single clip.")
    return head + ("That is no reliable difference between clips that did great and clips that did badly, so the line "
                   "shows where attention is predicted to rise and fall in this clip, not whether it will do well.")


# ── demo picks ───────────────────────────────────────────────────────────


def abs_block(t: pd.Series) -> dict:
    ok = pd.notna(t["abs_pct"])
    return {"abs_pct": float(t["abs_pct"]) if ok else None, "n_abs": int(t["n_abs"]) if ok else None,
            "views_7d": float(t["views_7d_abs"]) if ok else None}


def demo_picks(elig: list[dict], tiers: pd.DataFrame, language: dict[str, dict], staging: Path, lockbox: set[str],
               veto: set[str]) -> tuple[dict[str, list[dict]], dict]:
    counts: dict[str, int] = {}
    by_tier: dict[str, list[dict]] = {t: [] for t in TIERS}

    def skip(why):
        counts[why] = counts.get(why, 0) + 1
    for e in sorted(elig, key=lambda e: e["rank_key"]):
        vid, vp = e["video_id"], e["source_name"]
        if vid in lockbox:
            raise SystemExit(f"{vid}: lockbox clip among the eligible clips")
        if e["duration_s"] > DEMO_MAX_S:
            skip("too_long"); continue  # noqa: E702
        if not language.get(vid, {}).get("clean"):
            skip("language"); continue  # noqa: E702
        if vp not in tiers.index or tiers.loc[vp, "tier"] is None or pd.isna(tiers.loc[vp, "tier"]):
            skip("no_tier_or_dq"); continue  # noqa: E702
        t = tiers.loc[vp, "tier"]
        if vid in veto:
            skip("vetoed"); continue  # noqa: E702
        if len(by_tier[t]) >= DEMO_PER_TIER or any(p["deal_id"] == e["deal_id"] for p in by_tier[t]):
            continue
        src = sep.source_file(staging, e["source_path"])
        if src is None:
            skip("no_source_file"); continue  # noqa: E702
        by_tier[t].append({**e, "tier": t, "source_file": str(src), "views_pct": float(tiers.loc[vp, "views_pct"]),
                           "n_ref": int(tiers.loc[vp, "n_ref"]), "reach_rel_local": float(tiers.loc[vp, "reach_rel_local"]), **abs_block(tiers.loc[vp])})
    for t, ps in by_tier.items():
        for p in ps:
            x = np.exp(p["reach_rel_local"])
            if (t == "great" and x <= 1) or (t == "bad" and x >= 1):
                raise SystemExit(f"{p['video_id']}: tier {t} but {x:.2f}x usual")
    return by_tier, counts


def owner_picks(spec: list[str], elig: list[dict], tiers: pd.DataFrame, language: dict[str, dict], staging: Path,
                lockbox: set[str]) -> tuple[dict[str, list[dict]], dict]:
    """Hand-picked clips ("tier=video_id"). Chosen by the owner, not by a rule; every other check still applies and
    the tier must be the clip's own computed tier."""
    by_id = {e["video_id"]: e for e in elig}
    by_tier: dict[str, list[dict]] = {t: [] for t in TIERS}
    for item in spec:
        t, _, vid = item.partition("=")
        if t not in TIERS:
            raise SystemExit(f"{item}: tier must be one of {sorted(TIERS)}")
        if vid in lockbox:
            raise SystemExit(f"{vid}: lockbox clip")
        e = by_id.get(vid)
        if e is None:
            raise SystemExit(f"{vid}: not an eligible demo clip")
        if not language.get(vid, {}).get("clean"):
            raise SystemExit(f"{vid}: transcript language screen")
        vp = e["source_name"]
        got = tiers.loc[vp, "tier"] if vp in tiers.index else None
        if got != t:
            raise SystemExit(f"{vid}: computed tier is {got}, not {t}")
        src = sep.source_file(staging, e["source_path"])
        if src is None:
            raise SystemExit(f"{vid}: no source file")
        by_tier[t].append({**e, "tier": t, "source_file": str(src), "selection": "owner",
                           "views_pct": float(tiers.loc[vp, "views_pct"]), "n_ref": int(tiers.loc[vp, "n_ref"]),
                           "reach_rel_local": float(tiers.loc[vp, "reach_rel_local"]), **abs_block(tiers.loc[vp])})
    return by_tier, {"owner_picks": len(spec)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--patterns", type=Path, required=True)
    ap.add_argument("--demo", type=Path, default=None, help="also write demo picks here")
    ap.add_argument("--tier-basis", choices=TIER_BASES, default="relative")
    ap.add_argument("--owner-picks", nargs="*", default=[], metavar="TIER=VIDEO_ID",
                    help="hand-picked demo clips (owner selection) instead of the hash-order rule; same eligibility, "
                         "language, lockbox and tier checks")
    ap.add_argument("--candidates", type=Path, default=None,
                    help="write every demo-eligible tiered clip (clean transcript, <= DEMO_MAX_S, source file) as CSV")
    ap.add_argument("--all-candidates", action="store_true",
                    help="demo picks = every demo-eligible tiered clip (a browse set for the owner), most views first")
    ap.add_argument("--table", type=Path, default=None, help="also write the per-clip feature table (parquet)")
    ap.add_argument("--veto", nargs="*", default=[], help="video IDs vetoed after viewing (recorded)")
    ap.add_argument("--demo-selection", type=Path, default=ROOT / "results/demo/selection.json")
    ap.add_argument("--outcomes", type=Path, default=ROOT / "results/outcomes.parquet")
    ap.add_argument("--mpop", type=Path, default=ROOT / "results/moments_pop/mpop_v1.parquet")
    ap.add_argument("--state", type=Path, default=blp.DEFAULT_STATE)
    ap.add_argument("--roi-map", type=Path, default=bmp.DEFAULT_ROI_MAP)
    ap.add_argument("--cache", type=Path, default=blp.DEFAULT_OUT / ".library_cache_v0.pkl")
    ap.add_argument("--staging", type=Path, default=ROOT / "results/run_full/staging")
    ap.add_argument("--selection", type=Path, default=ROOT / "results/study/selection.csv")
    ap.add_argument("--lockbox-ext", type=Path, default=ROOT / "results/study/lockbox_ext.csv")
    ap.add_argument("--members", type=Path, default=ROOT / "results/run_full/members.csv")
    ap.add_argument("--batches-dir", type=Path, default=ROOT / "results/batches_s384")
    args = ap.parse_args(argv)

    cols = sep.OBSERVED_COLS + ["deal_id", "deal_name", "reach_log", "reach_basis", "social_account_id"]
    o = pd.read_parquet(args.outcomes, columns=cols).astype({"id": str})
    tiers = post_tiers(o, args.tier_basis)
    obs = o.set_index("id")

    sel = pd.read_csv(args.selection, dtype={"video_id": str})
    lockbox = set(sel.loc[sel["split"] == "lockbox", "video_id"])
    if args.lockbox_ext.exists():
        lockbox |= set(pd.read_csv(args.lockbox_ext, usecols=["video_id"], dtype=str)["video_id"])
    incl = sel.set_index("video_id")["incl_prob"].astype(float)

    # library table
    state = mp.load(args.state)
    _, keys = bmp.masks_and_keys(args.roi_map)
    clips = blp.load_library(state, args.state, args.roi_map, 8, args.cache)
    lib = blp.Library([blp.entry_from_clip(c, state["norms"]) for c in clips], keys)
    batches = sorted(p.name for p in args.batches_dir.iterdir() if (p / "manifest.jsonl").exists())
    man = sdc.load_manifests(args.batches_dir, batches)
    edit = pd.read_parquet(args.mpop).set_index("video_id")
    found = bmp.discover([ROOT / p if not Path(p).is_absolute() else Path(p) for p in state["inputs"]["out_roots"]])
    rows = []
    for i, e in enumerate(lib.entries):
        vid = e.video_id
        if vid in lockbox:
            raise SystemExit(f"{vid}: lockbox clip in the library")
        vp = man[vid]["source_name"]
        words = json.loads(found[vid][0].read_text()).get("words") or []
        t = tiers.loc[vp] if vp in tiers.index else None
        rows.append({"video_id": vid, "vp_id": vp, "w": 1.0 / incl[vid],
                     "account": str(obs.loc[vp, "social_account_id"]) if vp in obs.index else None,
                     "tier": None if t is None or pd.isna(t["tier"]) else t["tier"],
                     "stratum": None if t is None else t["stratum"],
                     **clip_features(e, lib.pct[i], edit.loc[vid] if vid in edit.index else None, words, e.duration)})
    tab = pd.DataFrame(rows)
    if args.table:
        tab.to_parquet(args.table, index=False)
    feats = patterns(tab, np.random.default_rng(SEED))
    tiered = tab[tab["tier"].notna()]
    n_by = tiered["tier"].value_counts().to_dict()
    above_f = {f["key"]: f for f in feats}["brain_above_typical"]
    deals = obs.loc[tab["vp_id"], "deal_id"].nunique()
    pat = {
        "schema": "nvi.patterns.v0", "internal_only": True, "exploratory": True,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "definition": ("Library clips (training split) grouped by how their own post did on views compared with the "
                       "same account's recent posts, ranked within the same deal and platform. Means are weighted to "
                       "undo the study set's oversampling; differences compare clips within the same deal and "
                       "platform; 95% bootstrap intervals over clips, Benjamini-Hochberg across the 12 features."),
        "outcome_plain": "Grouped by views compared with the same account's recent posts.",
        "tier_basis": args.tier_basis, "tier_rule": TIER_RULES[args.tier_basis], "n_library": int(len(tab)), "n_tiered": int(len(tiered)), "n_deals": int(deals),
        "tiers": {t: {"label": TIER_LABEL[t], "n_contents": int(n_by.get(t, 0)),
                      "rule_plain": TIER_PLAIN[args.tier_basis][t]} for t in TIERS},
        "features": feats,
        "account_check": account_check(tab, np.random.default_rng(SEED)),
        "interpreter_line": interpreter_line(above_f),
        "caveats": [
            "Exploratory: these comparisons were not pre-registered. In the pre-registered stage-1 test, adding the "
            "predicted brain response to basic information (account, length, timing) did not improve predictions of "
            "views or engagement (no-GO), so a pattern here mostly overlaps with what that basic information "
            "already tells you.",
            "The brain lines are predictions for an average viewer (TRIBE v2), not measured brain activity.",
            "A difference here is an association within your library, not a cause.",
        ],
        "params": {"seed": SEED, "n_boot": N_BOOT, "winsor_pct": WINSOR, "q_max": Q_MAX, "min_ref": MIN_REF,
                   "tiers": TIERS},
    }
    args.patterns.parent.mkdir(parents=True, exist_ok=True)
    args.patterns.write_text(json.dumps(blp.clean(pat), indent=1) + "\n")

    if args.demo is None:
        print(json.dumps({"tier_basis": args.tier_basis, "tiers_library": n_by, "interpreter_line": pat["interpreter_line"],
                          "features": [(f["key"], {k: round(v, 1) for k, v in f["mean"].items()},
                                        round(f["diff_great_minus_bad"]["point"], 2), round(f["q"], 3), f["verdict"])
                                       for f in feats], "account_check": pat["account_check"]}, indent=1, default=str))
        return 0
    # demo picks
    demo = json.loads(args.demo_selection.read_text())
    roots = [Path(b["out_root"]) for b in demo["batches_used"]]
    elig, _, _, _ = sdc.eligible(roots, args.selection, args.lockbox_ext, args.members, args.batches_dir,
                                 [ROOT / "results/runs/study-bf16/study2-logs"])
    lang = {e["video_id"]: language_screen.screen(json.loads(found[e["video_id"]][0].read_text()).get("words") or [])
            for e in elig if e["video_id"] in found}
    cand, cand_e = [], []
    if args.candidates or args.all_candidates:
        feat = tab.set_index("video_id")
        for e in elig:
            vid, vp = e["video_id"], e["source_name"]
            if vid in lockbox or e["duration_s"] > DEMO_MAX_S or not lang.get(vid, {}).get("clean"):
                continue
            if vp not in tiers.index or not isinstance(tiers.loc[vp, "tier"], str) or vid not in feat.index:
                continue
            if sep.source_file(args.staging, e["source_path"]) is None:
                continue
            row = obs.loc[vp]
            f = feat.loc[vid]
            cand_e.append(e)
            cand.append({"tier": tiers.loc[vp, "tier"], "source_file": str(sep.source_file(args.staging, e["source_path"])), "video_id": vid, "rank_key": e["rank_key"],
                         "deal": str(row["deal_name"]).replace(" X Clipping Cartel", ""), "platform": e["platform"],
                         "duration_s": round(e["duration_s"], 1), "views": row["views_final"],
                         "x_usual": round(float(np.exp(row["reach_rel_local"])), 2),
                         "engagement_pct": sep.engagement_pct(row), "video_link": row["video_link"],
                         **{k: round(float(f[k]), 1) for k in ("brain_opening", "brain_above_typical",
                                                               "brain_low_share", "brain_ending", "brain_peak",
                                                               "first_cut_s", "cuts_per_min")}})
        if args.candidates:
            pd.DataFrame(cand).sort_values(["tier", "rank_key"]).to_csv(args.candidates, index=False)
    if args.all_candidates:
        by_tier = {t: [] for t in TIERS}
        for e, c in sorted(zip(cand_e, cand), key=lambda ec: -float(ec[1]["views"])):
            vp = e["source_name"]
            by_tier[c["tier"]].append({**e, "tier": c["tier"], "source_file": c["source_file"], "selection": "candidates",
                                       "views_pct": float(tiers.loc[vp, "views_pct"]), "n_ref": int(tiers.loc[vp, "n_ref"]),
                                       "reach_rel_local": float(tiers.loc[vp, "reach_rel_local"]),
                                       **abs_block(tiers.loc[vp])})
        counts = {t: len(v) for t, v in by_tier.items()}
    elif args.owner_picks:
        by_tier, counts = owner_picks(args.owner_picks, elig, tiers, lang, args.staging, lockbox)
    else:
        by_tier, counts = demo_picks(elig, tiers, lang, args.staging, lockbox, set(args.veto))
    for ps in by_tier.values():
        for p in ps:
            row = obs.loc[p["source_name"]]
            p["deal_label"] = str(row["deal_name"]).replace(" X Clipping Cartel", "").replace(" x Clipping Cartel", "")
            p["observed"] = sep.observed_block(row)
    owner = bool(args.owner_picks)
    selection = "candidates" if args.all_candidates else "owner" if owner else "rule"
    res = {"rule": {"candidates": CANDIDATES_RULE, "owner": OWNER_RULE, "rule": DEMO_RULE}[selection],
           "selection": selection,
           "tier_basis": args.tier_basis, "tier_rule": TIER_RULES[args.tier_basis], "internal_only": True, "veto": sorted(args.veto),
           "created_utc": pat["created_utc"], "counts": counts, "picks": by_tier,
           "complete": all(len(v) >= 1 for v in by_tier.values()) if selection != "rule"
           else all(len(v) == DEMO_PER_TIER for v in by_tier.values())}
    args.demo.parent.mkdir(parents=True, exist_ok=True)
    args.demo.write_text(json.dumps(res, indent=1, default=str) + "\n")

    print(json.dumps({"tiers_library": n_by, "interpreter_line": pat["interpreter_line"],
                      "features": [(f["key"], {k: round(v, 1) for k, v in f["mean"].items()},
                                    round(f["diff_great_minus_bad"]["point"], 2),
                                    [round(f["diff_great_minus_bad"]["lo"], 2), round(f["diff_great_minus_bad"]["hi"], 2)],
                                    round(f["q"], 3), f["verdict"]) for f in feats],
                      "demo": {t: [(p["video_id"], p["deal_label"], p["platform"], p["duration_s"],
                                    round(np.exp(p["reach_rel_local"]), 2), round(p["views_pct"])) for p in ps]
                               for t, ps in by_tier.items()}, "counts": counts}, indent=1, default=str))
    return 0 if res["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
