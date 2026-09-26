#!/usr/bin/env python3
"""Profile one deal or account: which predicted-response features go with better outcomes there,
how that differs from the general model, and whether a niche-tuned model predicts it better.

    python tools/profile_account.py --features results/features/features.parquet \
        --members results/run_full/members.csv --outcomes results/outcomes.parquet \
        --selection results/study/selection.csv --deal <deal_id> --target reach_rel_local \
        --out-dir results/profiles

Writes ``<out-dir>/<kind>-<id>-<target>.md`` and ``.json``. Uses the train split only (the lockbox
stays untouched). The general model is fit without the niche; the tuned model adds a partially pooled
(ridge-shrunk, penalty by content-grouped CV) adjustment fit on the niche's own clips. Quality is
measured on the niche's held-out contents. Everything is an association in observational data for
model-predicted (average-subject) responses.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit_models as fm  # noqa: E402

DIFFERS_Q = 0.05  # BH q over the profile features for "differs from the general model"
CAVEATS = [
    "Brain features are TRIBE v2 predictions for an average subject, not measurements of this audience.",
    "Slopes and correlations are associations in this niche's past posts; they do not show that editing a clip "
    "toward a feature value would change its performance.",
    "Profile slopes are partial (the other profile features and platform/deal held fixed) and correlated features "
    "share credit; a small or unstable niche shrinks toward the general model by design.",
    "Intervals are conditional on the chosen deviation penalty (it is not re-selected per bootstrap resample) and "
    "unadjusted for the number of features; 'differs' uses Benjamini–Hochberg q over the profile features.",
    "Train-split rows over-sample reach tails within accounts; metrics are weighted by 1/incl_prob.",
    "TRIBE v2 is CC-BY-NC-4.0: research use only.",
]


def niche_cv(df: pd.DataFrame, niche: str, ycol: str, *, base: str, n_splits: int, n_boot: int, seed: int,
             fit_weighted: bool = False) -> dict:
    """Content-grouped folds over the niche: general (all other rows, niche's train folds included),
    general without the niche, and tuned (without niche + shrunk adjustment on the niche's train folds)."""
    from sklearn.model_selection import GroupKFold

    y, w = df[ycol].to_numpy(float), df["_w"].to_numpy(float)
    m = fm.niche_mask(df, niche)
    idx = np.flatnonzero(m)
    content = df["_content"].to_numpy()
    k = min(n_splits, len(np.unique(content[idx])))
    if k < 2:
        return {}
    fsets = fm.feature_sets(df)
    cols = fm.profile_columns(fsets["B"][1])
    gen, gen_wo, tuned = (np.full(len(df), np.nan) for _ in range(3))
    alphas = []
    other = np.flatnonzero(~m & ~np.isin(content, np.unique(content[idx])))
    for tr_n, te_n in GroupKFold(n_splits=k, shuffle=True, random_state=seed).split(idx, groups=content[idx]):
        tr_d, te_d = idx[tr_n], idx[te_n]
        tr_all = np.r_[other, tr_d]
        gen[te_d] = fm.fit_predict(base, fsets["B"], df.iloc[tr_all], y[tr_all], w[tr_all], df.iloc[te_d], seed,
                                   fit_weighted)[1]
        _, p = fm.fit_predict(base, fsets["B"], df.iloc[other], y[other], w[other], df.iloc[np.r_[tr_d, te_d]],
                              seed, fit_weighted)
        adj = fm.fit_adjustment(df.iloc[tr_d], y[tr_d] - p[: len(tr_d)], content[tr_d], w[tr_d], cols, seed=seed)
        alphas.append(adj.alpha)
        gen_wo[te_d] = p[len(tr_d):]
        tuned[te_d] = p[len(tr_d):] + adj.predict(df.iloc[te_d])
    ok = idx[np.isfinite(tuned[idx])]
    res = fm.bootstrap(y[ok], {"general": gen[ok], "general_without_niche": gen_wo[ok], "tuned": tuned[ok]},
                       content[ok], df["_stratum"].to_numpy()[ok], w[ok], n_boot, seed,
                       [("tuned", "general"), ("tuned", "general_without_niche")])
    res["n_folds"] = k
    res["alphas"] = alphas
    return res


def _fmt(s):
    return fm._fmt(s)


def render(p: dict) -> str:
    L = [f"# Niche profile: {p['niche']} · `{p['target']}`\n"]
    if p.get("synthetic_clips"):
        L.append(f"> **SYNTHETIC**: {p['synthetic_clips']} clips come from dry-run or synthetic data; this profile "
                 "tests the pipeline, not TRIBE.\n")
    L.append("Model predictions for an average subject; associations only.\n")
    L.append("## Size\n")
    L.append(f"- posts (train split, labelled): {p['n_posts']}; contents: {p['n_contents']}; "
             f"accounts: {p['n_accounts']}; platforms: {', '.join(p['platforms'])}")
    L.append(f"- general model rows outside the niche: {p['n_general_rows']}\n")
    q = p.get("quality") or {}
    L.append("## Model quality on this niche's held-out contents\n")
    if q:
        mo, dl = q["models"], q["deltas"]
        L.append("| model | within-stratum ρ | pooled ρ | R² |\n|---|---|---|---|")
        for k, lab in (("general", "general (all data, niche's other folds included)"),
                       ("general_without_niche", "general without this niche (unseen niche)"),
                       ("tuned", "niche-tuned (partial pooling)")):
            L.append(f"| {lab} | {_fmt(mo[k]['within_stratum_spearman'])} | {_fmt(mo[k]['spearman'])} | "
                     f"{_fmt(mo[k]['r2'])} |")
        d = dl["tuned-general"]
        L.append(f"\nTuned − general: Δ R² {_fmt(d['r2'])}, Δ within-stratum ρ {_fmt(d['within_stratum_spearman'])} "
                 f"→ {fm._verdict(d['r2'], 'tuned', 'general')}. Adjustment penalties by fold: "
                 f"{', '.join(f'{a:.3g}' for a in q['alphas'])} (the largest grid value means no reliable niche "
                 "deviation).\n")
    else:
        L.append("Too few contents for held-out evaluation.\n")
    prof = p["profile"]["features"]
    L.append("## Top associated features in this niche\n")
    L.append("Partial slope in target units per 1 SD (niche = general + shrunk niche deviation), 95% bootstrap CI "
             f"over contents, conditional on the deviation penalty α = {p['profile']['alpha']:.3g} (selected on the "
             "profile's own standardised features) and unadjusted for the number of features; q = Benjamini–"
             "Hochberg q of the deviation over all profile features; ρ = weighted Spearman with the target inside "
             "the niche vs overall.\n")
    L.append("| feature | niche slope | general slope | deviation | q | ρ niche | ρ overall |\n"
             "|---|---|---|---|---|---|---|")
    for r in prof[:10]:
        L.append(f"| {r['feature']} | {_fmt({'point': r['niche_slope'], 'ci': r['niche_ci']})} | "
                 f"{_fmt({'point': r['global_slope'], 'ci': r['global_ci']})} | "
                 f"{_fmt({'point': r['deviation'], 'ci': r['deviation_ci']})} | {r['deviation_q']:.3g} | "
                 f"{r['niche_spearman']:.2f} | {r['overall_spearman']:.2f} |")
    L.append("\n## How this niche differs from the general model\n")
    diff = p["differs"]
    if diff:
        for r in diff:
            sign = "stronger positive" if r["deviation"] > 0 else "weaker / more negative"
            L.append(f"- `{r['feature']}`: {sign} association here than overall "
                     f"(deviation {_fmt({'point': r['deviation'], 'ci': r['deviation_ci']})}, "
                     f"BH q {r['deviation_q']:.3g}).")
    else:
        L.append(f"- No profile feature's niche deviation has a Benjamini–Hochberg q < {DIFFERS_Q} over the "
                 f"{len(prof)} profile features; the general associations apply.")
    L.append("\n## Caveats\n")
    L.extend(f"- {c}" for c in CAVEATS)
    return "\n".join(L) + "\n"


def profile(df_all: pd.DataFrame, info: dict, niche: str, target: str, *, base: str = "hgb", n_splits: int = 5,
            n_boot: int = 300, seed: int = 0, fit_weighted: bool = False) -> dict:
    ycol = f"y_{target}"
    df = df_all[(df_all["split"] == "train") & np.isfinite(df_all[ycol].to_numpy(float))].reset_index(drop=True)
    m = fm.niche_mask(df, niche)
    if m.sum() < 5:
        raise SystemExit(f"{niche}: only {int(m.sum())} labelled train posts")
    t0 = time.perf_counter()
    nm = fm.fit_niche(df, niche, target, base=base, seed=seed, n_boot=n_boot, fit_weighted=fit_weighted)
    q = niche_cv(df, niche, ycol, base=base, n_splits=n_splits, n_boot=n_boot, seed=seed, fit_weighted=fit_weighted)
    feats = nm.profile["features"]
    # one test per profile feature: BH over the features (niche_profile's deviation_q), not a per-feature CI
    differs = [r for r in feats if np.isfinite(r.get("deviation_q", np.nan)) and r["deviation_q"] < DIFFERS_Q]
    return {
        "niche": niche, "target": target, "n_posts": int(m.sum()),
        "n_contents": int(df.loc[m, "video_id"].nunique()),
        "n_accounts": int(df.loc[m, "social_account_id"].nunique()),
        "platforms": sorted(df.loc[m, "platform"].unique().tolist()),
        "n_general_rows": int((~m).sum()), "general_base": base,
        "synthetic_clips": info.get("synthetic_clips", 0),
        "adjustment": {"alpha": nm.adjustment.alpha, "intercept": nm.adjustment.intercept,
                       "coef": dict(zip(nm.adjustment.cols, nm.adjustment.coef.tolist()))},
        "quality": q, "profile": nm.profile, "differs": differs, "caveats": CAVEATS,
        "runtime_s": round(time.perf_counter() - t0, 1),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--members", type=Path, required=True)
    ap.add_argument("--outcomes", type=Path, required=True)
    ap.add_argument("--selection", type=Path, default=None)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--deal")
    g.add_argument("--account", help="social_account_id or anchor account")
    ap.add_argument("--target", default="reach_rel_local")
    ap.add_argument("--out-dir", type=Path, default=Path("results/profiles"))
    ap.add_argument("--base", default="hgb", choices=fm.MODELS)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--fit-weighted", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    from threadpoolctl import threadpool_limits

    niche = f"deal:{args.deal}" if args.deal else f"account:{args.account}"
    with threadpool_limits(args.threads):
        df, info, _ = fm.build_dataset(
            fm.read_table(args.features), fm.read_table(args.members), fm.read_table(args.outcomes),
            [args.target], fm.read_table(args.selection) if args.selection else None)
        p = profile(df, info, niche, args.target, base=args.base, n_splits=args.n_splits, n_boot=args.n_boot,
                    seed=args.seed, fit_weighted=args.fit_weighted)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", niche.replace(":", "-")) + f"-{args.target}"
    (args.out_dir / f"{stem}.json").write_text(json.dumps(p, indent=2, default=fm._json_default) + "\n")
    (args.out_dir / f"{stem}.md").write_text(render(p))
    print(f"{args.out_dir / stem}.md ({p['runtime_s']}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
