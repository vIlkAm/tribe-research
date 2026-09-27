#!/usr/bin/env python3
"""Library theory (docs/LIBRARY_THEORY.md): which clip features go with account size, with how a video did for its
account, and with engagement; found on half the accounts, checked on the other half.

    .venv/bin/python tools/library_theory.py --table results/library/clip_table.parquet --out results/library/theory.json

INTERNAL and exploratory (reads observed outcomes of training clips; the lockbox stays sealed). The design is
fixed in docs/LIBRARY_THEORY.md; change it there, dated, not here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import library_tiers as lt  # noqa: E402

SEED = 20260926
N_BOOT = 2000
Q_MAX = 0.10
WINSOR = (1.0, 99.0)
OUTCOMES = {  # id: (column, comparison group, plain)
    "total": ("reach_log", "stratum", "views"),
    "account": ("local_baseline", "stratum", "account size (the account's usual views)"),
    "video": ("reach_rel_local", "account", "how the video did against its account's usual"),
    "engagement": ("log_engagement", "account", "engagement (interactions per view)"),
}
SIZE_WORDS = ((0.10, "very small"), (0.30, "small"), (0.50, "moderate"), (np.inf, "large"))


def size_word(r: float) -> str:
    return next(w for lim, w in SIZE_WORDS if abs(r) < lim)


def split_accounts(acc_stratum: pd.Series, seed: int = SEED) -> dict[str, int]:
    """account -> 0 (discovery) / 1 (confirmation): hash order, alternating within each deal x platform."""
    out = {}
    for _, accs in acc_stratum.groupby(acc_stratum):
        order = sorted(accs.index, key=lambda a: hashlib.sha256(f"{seed}:{a}".encode()).hexdigest())
        for i, a in enumerate(order):
            out[a] = i % 2
    return out


def ranks(v: np.ndarray) -> np.ndarray:
    return pd.Series(v).rank(method="average").to_numpy()


def centred_corr(x: np.ndarray, y: np.ndarray, w: np.ndarray, g: np.ndarray) -> float:
    """Weighted Pearson of x and y after removing each group's weighted mean (g = integer group codes)."""
    n = g.max() + 1
    sw = np.bincount(g, w, n)
    sw[sw == 0] = 1.0  # groups absent from a bootstrap draw
    xc = x - (np.bincount(g, w * x, n) / sw)[g]
    yc = y - (np.bincount(g, w * y, n) / sw)[g]
    vx, vy = np.sum(w * xc * xc), np.sum(w * yc * yc)
    if vx <= 0 or vy <= 0:
        return float("nan")
    return float(np.sum(w * xc * yc) / np.sqrt(vx * vy))


def prepare(d: pd.DataFrame, feat: str, out: str, group: str) -> tuple | None:
    """Rows with both values (and a group of >= 2 clips): ranks, weights, group codes, account codes."""
    s = d[list(dict.fromkeys([feat, out, "w", group, "account"]))].dropna()
    s = s[s[group].map(s[group].value_counts()) >= 2]
    if len(s) < 10:
        return None
    x = s[feat].to_numpy(float)
    lo, hi = np.nanpercentile(x, WINSOR)
    x = ranks(np.clip(x, lo, hi))
    y = ranks(s[out].to_numpy(float))
    return (x, y, s["w"].to_numpy(float), pd.factorize(s[group])[0], pd.factorize(s["account"])[0])


def corr_ci(prep: tuple, rng: np.random.Generator, n_boot: int = N_BOOT) -> dict:
    """Point estimate, account-clustered bootstrap 95% interval and two-sided bootstrap p."""
    x, y, w, g, a = prep
    r = centred_corr(x, y, w, g)
    n_acc = a.max() + 1
    rows_of = [np.flatnonzero(a == k) for k in range(n_acc)]
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, n_acc, n_acc)
        idx = np.concatenate([rows_of[k] for k in pick])
        boots[b] = centred_corr(x[idx], y[idx], w[idx], g[idx])
    boots = boots[np.isfinite(boots)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean()))
    return {"r": r, "lo": float(lo), "hi": float(hi), "p_boot": float(p), "n_clips": int(len(x)),
            "n_accounts": int(n_acc)}


def decompose(d: pd.DataFrame) -> dict:
    """Within deal x platform: share of the (weighted) variance of log views that is account size vs video."""
    s = d[["reach_log", "local_baseline", "reach_rel_local", "w", "stratum"]].dropna()
    g = pd.factorize(s["stratum"])[0]
    w = s["w"].to_numpy(float)

    def c(v):
        v = s[v].to_numpy(float)
        n = g.max() + 1
        return v - (np.bincount(g, w * v, n) / np.bincount(g, w, n))[g]
    t, a, v = c("reach_log"), c("local_baseline"), c("reach_rel_local")
    var = lambda z: float(np.sum(w * z * z) / w.sum())  # noqa: E731
    cov = float(np.sum(w * a * v) / w.sum())
    return {"n_clips": int(len(s)), "var_total": var(t), "share_account": var(a) / var(t),
            "share_video": var(v) / var(t), "share_covariance": 2 * cov / var(t)}


def plain_line(f: str, o: str, r: float) -> str:
    feat = {k: lab for k, lab, _, _ in lt.FEATURES}[f]
    return (f"{feat[:1].upper() + feat[1:]}: {'higher' if r > 0 else 'lower'} goes with more "
            f"{OUTCOMES[o][2]} ({size_word(r)}, r = {r:+.2f}; holds on accounts it was not found on).")


def run(d: pd.DataFrame, rng_seed: int = SEED, n_boot: int = N_BOOT) -> dict:
    half = d["half"].to_numpy()
    pairs = []
    for fkey, flabel, _, fgroup in lt.FEATURES:
        for okey, (ocol, group, _) in OUTCOMES.items():
            pairs.append({"feature": fkey, "feature_plain": flabel, "feature_group": fgroup, "outcome": okey,
                          "col": ocol, "group": group})
    rng = np.random.default_rng(rng_seed)
    for p in pairs:
        for name, sub in (("discovery", d[half == 0]), ("confirmation", d[half == 1]), ("full", d)):
            prep = prepare(sub, p["feature"], p["col"], p["group"])
            p[name] = None if prep is None else corr_ci(prep, rng, n_boot if name != "confirmation" else n_boot)
    disc_p = np.array([p["discovery"]["p_boot"] if p["discovery"] else 1.0 for p in pairs])
    q = lt.bh(disc_p)
    for p, qq in zip(pairs, q):
        p["q_discovery"] = float(qq)
        p["candidate"] = bool(qq <= Q_MAX)
        c = p["confirmation"]
        p["holds"] = bool(p["candidate"] and c is not None and np.sign(c["r"]) == np.sign(p["discovery"]["r"])
                          and (c["lo"] > 0 or c["hi"] < 0))
        p["verdict"] = "holds" if p["holds"] else "did_not_hold" if p["candidate"] else "no_pattern"
        p["plain"] = plain_line(p["feature"], p["outcome"], p["full"]["r"]) if p["holds"] else (
            "Found on half the accounts but not on the other half: no pattern." if p["candidate"] else "No pattern.")
        p.pop("col")
    return {"pairs": pairs}


def excludes_zero(c: dict | None) -> bool:
    return c is not None and (c["lo"] > 0 or c["hi"] < 0)


def full_sample_brain_line(res: dict) -> str:
    """When no brain pair passes the split-half rule: the brain summary whose all-clips 95% range excludes zero
    for both views and video-vs-account (largest |r| on views), stated as a tendency that is not yet confirmed."""
    by = {(p["feature"], p["outcome"]): p for p in res["pairs"]}
    both = [p for p in res["pairs"] if p["feature_group"] == "brain" and p["outcome"] == "total"
            and excludes_zero(p["full"]) and excludes_zero((by.get((p["feature"], "video")) or {}).get("full"))
            and np.sign(p["full"]["r"]) == np.sign(by[(p["feature"], "video")]["full"]["r"])]
    if not both:
        return "No brain-line summary has passed the strict split-half check. "
    best = max(both, key=lambda p: abs(p["full"]["r"]))
    t, v = best["full"], by[(best["feature"], "video")]["full"]
    feat = best["feature_plain"][:1].lower() + best["feature_plain"][1:]
    word = "more" if t["r"] > 0 else "fewer"
    return (f"Across all clips, a higher {feat} went with {word} views (r {t['r']:+.2f}, 95% range {t['lo']:+.2f} "
            f"to {t['hi']:+.2f}) and with {'beating' if v['r'] > 0 else 'falling short of'} the account's own usual "
            f"(r {v['r']:+.2f}): a small tendency, not yet confirmed by the strict split-half check. ")


def interpreter(res: dict, dec: dict) -> str:
    held = [p for p in res["pairs"] if p["holds"]]
    lead = ("a clip's views swing more around its account's usual than accounts differ from each other"
            if dec["share_video"] > dec["share_account"] else
            "accounts differ from each other more than a clip's views swing around its account's usual")
    head = (f"Within the same client and platform, {lead} (variance shares: around the usual "
            f"{100 * dec['share_video']:.0f}%, account size {100 * dec['share_account']:.0f}%, overlap "
            f"{100 * dec['share_covariance']:+.0f}%). The swing includes luck, timing and the algorithm, not only the "
            "video itself. ")
    brain_video = [p for p in held if p["feature_group"] == "brain" and p["outcome"] in ("video", "engagement")]
    if not brain_video:
        return head + full_sample_brain_line(res) + ("So the line shows where attention is predicted to rise and "
                                                     "fall in this clip, not whether it will do well.")
    best = max(brain_video, key=lambda p: abs(p["full"]["r"]))
    return head + (f"Within the same account, {best['feature_plain']} goes with {OUTCOMES[best['outcome']][2]} "
                   f"({size_word(best['full']['r'])}, r = {best['full']['r']:+.2f}), found on half the accounts and "
                   "holding on the other half. That is a tendency across many clips, not a verdict on one.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--table", type=Path, default=ROOT / "results/library/clip_table.parquet")
    ap.add_argument("--outcomes", type=Path, default=ROOT / "results/outcomes.parquet")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args(argv)

    tab = pd.read_parquet(args.table)
    o = pd.read_parquet(args.outcomes, columns=["id", "deal_id", "platform", "social_account_id", "reach_log",
                                                "local_baseline", "reach_rel_local", "interactions_rate_eb",
                                                "dq_flags"]).astype({"id": str}).set_index("id")
    d = tab.drop(columns=[c for c in ("stratum", "tier", "account") if c in tab]).join(o, on="vp_id")
    d = d[d["dq_flags"].isna() & d["reach_rel_local"].notna() & d["deal_id"].notna()].copy()
    d["stratum"] = d["deal_id"].astype(str) + "|" + d["platform"].astype(str)
    d["account"] = d["social_account_id"].astype(str)
    d["log_engagement"] = np.log(d["interactions_rate_eb"].where(d["interactions_rate_eb"] > 0))
    acc_stratum = d.groupby("account")["stratum"].agg(lambda s: s.value_counts().index[0])
    half = split_accounts(acc_stratum)
    d["half"] = d["account"].map(half)

    res = run(d, n_boot=args.n_boot)
    dec = decompose(d)
    out = {"schema": "nvi.theory.v0", "internal_only": True, "exploratory": True, "design": "docs/LIBRARY_THEORY.md",
           "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "n_clips": int(len(d)), "n_accounts": int(d["account"].nunique()),
           "halves": {h: {"n_clips": int((d["half"] == h).sum()), "n_accounts": int(d.loc[d["half"] == h, "account"].nunique())}
                      for h in (0, 1)},
           "outcomes": {k: {"column": c, "compared_within": g, "plain": p} for k, (c, g, p) in OUTCOMES.items()},
           "decomposition": dec, **res, "interpreter_line": interpreter(res, dec),
           "params": {"seed": SEED, "n_boot": args.n_boot, "q_max": Q_MAX, "winsor_pct": WINSOR}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(json.dumps({"n_clips": out["n_clips"], "halves": out["halves"], "decomposition": dec,
                      "interpreter_line": out["interpreter_line"],
                      "candidates": [(p["feature"], p["outcome"], round(p["discovery"]["r"], 3),
                                      round(p["q_discovery"], 3), round(p["confirmation"]["r"], 3),
                                      [round(p["confirmation"]["lo"], 3), round(p["confirmation"]["hi"], 3)],
                                      p["verdict"]) for p in res["pairs"] if p["candidate"]]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
