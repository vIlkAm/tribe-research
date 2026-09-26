#!/usr/bin/env python3
"""Can a lockbox of a given size resolve a brain increment? Power check with NO TRIBE outputs (CPU, no pods).

Builds the A features that exist without TRIBE (platform, deal, log duration, aspect, audio level, follower
bucket, posting weekday, account target encoding) for every eligible unique content, then adds ONE synthetic
``brain_synth`` feature with a chosen strength ``rho`` and measures, over repeated random designs:

    train n_train contents  ->  fit A and B (= A + brain_synth)  ->  score a pseudo-lockbox of n_lockbox contents

reporting the achieved within-stratum Spearman gain B − A, its paired cluster-bootstrap CI (the one
fit_models uses on the real lockbox) and the power = share of designs whose CI excludes 0.

``brain_synth`` is built from the content's out-of-fold A residual, so it carries label noise of the very posts
it is scored on and flatters itself. Read the output as "power at an ACHIEVED gain of X", not "power at rho".

The real lockbox (``selection.csv`` split == lockbox) and every content linked to it are removed before
anything else, so this never looks at a lockbox label.

    python tools/power_check.py --out results/power/power_check.json [--reps 12] [--threads 8]
    # overfitting penalty of a realistic block size, e.g. BE − E: 180 shared columns, 100 brain columns
    python tools/power_check.py --designs study_cv_1275 --n-noise-both 180 --n-noise 99 --rhos 0 0.2 0.3 [--model stack]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit_models as fm  # noqa: E402

DESIGNS = {  # name -> (train contents, lockbox contents); lockbox "cv" = 5-fold content CV over the train contents
    "study_1500": (1275, 225),
    "study_cv_1275": (1275, "cv"),
    "double_3000": (2550, 450),
    "full_eligible": (None, 0.15),  # all remaining eligible contents, 15 % held out
}
RHOS = (0.0, 0.1, 0.2, 0.3, 0.5)  # --rhos overrides


def eligible_features(run_dir: Path, selection: pd.DataFrame) -> pd.DataFrame:
    man = pd.read_json(run_dir / "manifest.jsonl", lines=True)[["video_id", "duration_s"]]
    qc = pd.read_csv(run_dir / "file_qc.csv")
    qc = qc[qc["decode_ok"].astype(str).str.lower().eq("true") & qc["has_audio"].astype(str).str.lower().eq("true")]
    f = man.merge(qc[["video_id", "width", "height", "audio_mean_db"]], on="video_id")
    f = f[(f["duration_s"] >= 5) & (f["duration_s"] <= 90)]
    lock = set(selection.loc[selection["split"] == "lockbox", "video_id"])
    f = f[~f["video_id"].isin(lock)]
    return pd.DataFrame({
        "video_id": f["video_id"], "status": "ok",
        "base_log_duration": np.log(f["duration_s"].clip(lower=1.0)),
        "base_aspect": f["height"] / f["width"].replace(0, np.nan),
        "base_audio_mean_db": f["audio_mean_db"],
    })


def design_split(df: pd.DataFrame, n_train, n_lb, rng) -> tuple[np.ndarray, np.ndarray]:
    """Random contents per deal (lockbox proportional to the deal's share), whole content components."""
    comp = df.drop_duplicates("_content")[["_content", "deal_id"]]
    comps = comp["_content"].to_numpy()
    if n_train is None:  # full: 15 % per deal held out, the rest trains
        lb = np.concatenate([rng.permutation(g["_content"].to_numpy())[: max(1, round(len(g) * n_lb))]
                             for _, g in comp.groupby("deal_id")])
        tr = np.setdiff1d(comps, lb)
    else:
        share = comp["deal_id"].value_counts(normalize=True)
        lb = np.concatenate([rng.permutation(g["_content"].to_numpy())[: max(1, round(n_lb * share[d]))]
                             for d, g in comp.groupby("deal_id")])
        rest = rng.permutation(np.setdiff1d(comps, lb))
        tr = rest[:n_train]
    c = df["_content"].to_numpy()
    return np.flatnonzero(np.isin(c, tr)), np.flatnonzero(np.isin(c, lb))


def run(args) -> dict:
    outcomes = fm.read_table(args.outcomes)
    members = fm.read_table(args.members)
    selection = pd.read_csv(args.selection)
    feats = eligible_features(args.run_dir, selection)
    lock = set(selection.loc[selection["split"] == "lockbox", "video_id"])
    res = {"config": {"designs": DESIGNS, "rhos": list(args.rhos), "reps": args.reps, "n_boot": args.n_boot,
                      "model": args.model, "n_noise_both": args.n_noise_both, "n_noise_b_only": args.n_noise, "structure": args.structure, "seed": args.seed}, "targets": {}}
    for t in args.targets:
        df, info, _ = fm.build_dataset(feats, members, outcomes, [t])
        ycol = f"y_{t}"
        df = df[np.isfinite(df[ycol].to_numpy(float))].reset_index(drop=True)
        # drop anything linked (content_group) to a real lockbox content: those labels stay sealed
        idc = fm.resolve(outcomes.columns, fm.OUTCOME_COLUMNS["vp_id"])
        lock_vp = set(members.loc[members["video_id"].isin(lock), "vp_id"].astype(str))
        lg = set(outcomes.loc[outcomes[idc].astype(str).isin(lock_vp), "content_group"].dropna()) \
            if "content_group" in outcomes else set()
        if lg and "content_group" in df:
            df = df[~df["content_group"].isin(lg)].reset_index(drop=True)
        df["_content"] = fm.content_components(df)
        df["_stratum"] = df["deal_id"] + "|" + df["platform"]
        y = df[ycol].to_numpy(float)
        fsA = fm.feature_sets(df)["A"]
        # out-of-fold A residual per content -> the synthetic brain signal
        splits = fm.make_splits(df, "content", 5, args.seed)
        oof, _, _ = fm.cross_validate(df, y, np.ones(len(df)), splits, {"A": fsA}, ["ridge"], args.seed)
        resid = pd.Series(y - oof["A_ridge"]).groupby(df["_content"]).transform("mean").to_numpy()
        z = (resid - np.nanmean(resid)) / np.nanstd(resid)
        rng0 = np.random.default_rng(args.seed)
        noise = pd.Series(rng0.standard_normal(df["_content"].max() + 1))[df["_content"]].to_numpy()
        # pure-noise columns, constant within a content like real clip features, to measure overfitting:
        # --n-noise-both go into A and B (emb_noise_*, stand-ins for E's columns), --n-noise into B only
        # (brain_noise_*, the rest of a brain block). The prefixes put them in the stack model's blocks.
        n_c = df["_content"].max() + 1
        nz = pd.DataFrame(np.random.default_rng(args.seed + 99).standard_normal((n_c, args.n_noise_both + args.n_noise))
                          [df["_content"].to_numpy()],
                          columns=[f"emb_noise_{j}" for j in range(args.n_noise_both)]
                          + [f"brain_noise_{j}" for j in range(args.n_noise)])
        df = pd.concat([df, nz], axis=1)
        fsA = (fsA[0], fsA[1] + [c for c in nz.columns if c.startswith("emb_noise_")])
        extra_b = [c for c in nz.columns if c.startswith("brain_noise_")]
        rngf = np.random.default_rng(args.seed + 5)
        fz = rngf.standard_normal((n_c, 4))[df["_content"].to_numpy()]
        load = rngf.standard_normal((5, len(extra_b)))
        load /= np.linalg.norm(load, axis=0, keepdims=True)
        acc_codes = df["social_account_id"].astype("category").cat.codes.to_numpy()
        acc_vec = rngf.standard_normal((acc_codes.max() + 1, len(extra_b)))[acc_codes]
        tres = {"n_posts": int(len(df)), "n_contents": int(df["_content"].nunique()),
                "n_noise_both": args.n_noise_both, "n_noise_b_only": args.n_noise, "designs": {}}
        print(f"[{t}] posts={len(df)} contents={tres['n_contents']} A={fsA[1][:8]}… ({len(fsA[1])} cols) "
              f"B adds {1 + len(extra_b)}", flush=True)
        for dname, (n_tr, n_lb) in DESIGNS.items():
            if args.designs and dname not in args.designs:
                continue
            if n_tr is not None and n_tr + (0 if n_lb == "cv" else n_lb) > tres["n_contents"]:
                continue
            for rho in args.rhos:
                df["brain_synth"] = rho * z + math.sqrt(1 - rho ** 2) * noise
                if args.structure == "dense" and extra_b:  # rho spread evenly: the block mean correlates rho with z
                    k = 1 + len(extra_b)
                    a = rho / math.sqrt(k * (1 - rho ** 2))
                    df["brain_synth"] = a * z + noise
                    df[extra_b] = a * z[:, None] + nz[extra_b].to_numpy()
                elif args.structure == "account" and extra_b:  # no content signal, only who posted it:
                    # an account "style" vector carries half of every column's variance (fingerprinting check)
                    df[extra_b] = math.sqrt(0.5) * acc_vec + math.sqrt(0.5) * nz[extra_b].to_numpy()
                elif args.structure == "factor" and extra_b:  # correlated low-rank block (like ROI/PCA features):
                    # 5 shared content factors carry half of every column's variance; factor 1 is brain_synth
                    fac = np.column_stack([df["brain_synth"].to_numpy(), fz])
                    df[extra_b] = math.sqrt(0.5) * (fac @ load) + math.sqrt(0.5) * nz[extra_b].to_numpy()
                fsB = (fsA[0], fsA[1] + ["brain_synth"] + extra_b)
                deltas, lows, highs, ns = [], [], [], []
                ts = time.perf_counter()
                for r in range(args.reps):
                    rng = np.random.default_rng(args.seed + 1000 * r + 7)
                    if n_lb == "cv":  # frozen pipeline, out-of-fold predictions for every sampled content
                        keep = rng.permutation(df["_content"].unique())[:n_tr]
                        lb = np.flatnonzero(df["_content"].isin(keep).to_numpy())
                        dlb = df.iloc[lb].reset_index(drop=True)
                        oofs, _, _ = fm.cross_validate(dlb, y[lb], np.ones(len(lb)),
                                                       fm.make_splits(dlb, "content", 5, args.seed + r),
                                                       {"A": fsA, "B": fsB}, [args.model], args.seed)
                        preds = {k: v for k, v in oofs.items()}
                    else:
                        tr, lb = design_split(df, n_tr, n_lb, rng)
                        dtr, dlb = df.iloc[tr], df.iloc[lb]
                        preds = {f"{k}_{args.model}": fm.fit_predict(args.model, cols, dtr, y[tr], None, dlb, args.seed)[1]
                                 for k, cols in (("A", fsA), ("B", fsB))}
                    b = fm.bootstrap(y[lb], preds, dlb["_content"].to_numpy(), dlb["_stratum"].to_numpy(), None,
                                     args.n_boot, args.seed + r)
                    d = b["deltas"][f"B_{args.model}-A_{args.model}"]["within_stratum_spearman"]
                    deltas.append(d["point"])
                    lows.append(d["ci"][0])
                    highs.append(d["ci"][1])
                    ns.append(int(dlb["_content"].nunique()))
                deltas, lows, highs = map(lambda a: np.array(a, float), (deltas, lows, highs))
                row = {"rho": rho, "lockbox_contents": int(np.median(ns)),
                       "train_contents": n_tr if n_tr is not None else int(tres["n_contents"] - np.median(ns)),
                       "scored_by": "5-fold content CV (out-of-fold)" if n_lb == "cv" else "held-out lockbox",
                       "achieved_delta_mean": float(np.nanmean(deltas)),
                       "achieved_delta_sd_across_designs": float(np.nanstd(deltas)),
                       "ci_halfwidth_mean": float(np.nanmean((highs - lows) / 2)),
                       "power_ci_excludes_0": float(np.nanmean(lows > 0))}
                tres["designs"].setdefault(dname, []).append(row)
                print(f"  {dname} rho={rho}: Δ={row['achieved_delta_mean']:+.3f} ±{row['ci_halfwidth_mean']:.3f} "
                      f"power={row['power_ci_excludes_0']:.2f} ({time.perf_counter() - ts:.0f}s)", flush=True)
        res["targets"][t] = tres
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outcomes", type=Path, default=Path("results/outcomes.parquet"))
    ap.add_argument("--members", type=Path, default=Path("results/run_full/members.csv"))
    ap.add_argument("--selection", type=Path, default=Path("results/study/selection.csv"))
    ap.add_argument("--run-dir", type=Path, default=Path("results/run_full"))
    ap.add_argument("--targets", nargs="+", default=["log_interactions_rate", "reach_rel_local"])
    ap.add_argument("--designs", nargs="*", default=None, choices=list(DESIGNS), help="default: all")
    ap.add_argument("--rhos", nargs="+", type=float, default=list(RHOS))
    ap.add_argument("--n-noise", type=int, default=0, help="pure-noise clip columns added to B only")
    ap.add_argument("--n-noise-both", type=int, default=0, help="pure-noise clip columns added to A and B")
    ap.add_argument("--model", default="ridge", choices=fm.MODELS)
    ap.add_argument("--structure", default="sparse", choices=("sparse", "dense", "factor", "account"),
                    help="brain block with --n-noise: sparse = one signal column + pure noise; dense = signal "
                         "spread evenly over independent columns; factor = correlated 5-factor block, signal on one; "
                         "account = per-account style vector, no content signal (fingerprinting check)")
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("results/power/power_check.json"))
    args = ap.parse_args()
    from threadpoolctl import threadpool_limits

    with threadpool_limits(args.threads):
        res = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
