#!/usr/bin/env python3
"""Build the per-video outcomes table from the exported metrics CSVs.

Reads ``<metrics-dir>/{video_performances,video_snapshots,social_accounts,deals}.csv``
(required) and ``social_account_stat_snapshots.csv`` / ``cross_platform_members.csv``
(optional), and writes one row per video_performance to ``--out`` (parquet, or CSV
when the suffix is .csv or pyarrow is missing) plus ``<out>.meta.json`` with the
parameters, fitted engagement priors, cleaning totals and flag counts.
Definitions: docs/OUTCOMES.md. Files only; this never touches a database.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tribe_research.outcomes import OutcomeParams, build_outcomes, load_tables  # noqa: E402


def _write(df, out: Path) -> Path:
    if out.suffix != ".csv":
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            out = out.with_suffix(".csv")
            print("pyarrow not installed; writing CSV instead", file=sys.stderr)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    if out.suffix == ".csv":
        df.to_csv(tmp, index=False)
    else:
        df.to_parquet(tmp, index=False)
    tmp.replace(out)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--metrics-dir", type=Path, default=ROOT / "results" / "metrics")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "outcomes.parquet")
    ap.add_argument("--primary-age", type=float, default=OutcomeParams.primary_age,
                    help="age (days) behind reach_rel / baselines; added to the reported ages")
    ap.add_argument("--max-extrapolation-frac", type=float, default=OutcomeParams.max_extrapolation_frac)
    ap.add_argument("--follower-tolerance-days", type=float, default=OutcomeParams.follower_tolerance_days)
    ap.add_argument("--upload-anchor-hours", type=float, default=OutcomeParams.upload_anchor_hours)
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    tables, load_report = load_tables(args.metrics_dir)
    params = OutcomeParams(primary_age=args.primary_age, max_extrapolation_frac=args.max_extrapolation_frac,
                           upload_anchor_hours=args.upload_anchor_hours,
                           follower_tolerance_days=args.follower_tolerance_days)
    df, meta = build_outcomes(tables, params)
    out = _write(df, args.out)
    meta["load"] = load_report
    meta["columns"] = list(df.columns)
    meta["seconds"] = round(time.perf_counter() - t0, 2)
    meta_path = out.with_name(out.name + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, default=str) + "\n")

    print(f"{len(df)} videos -> {out} ({meta['seconds']} s); meta: {meta_path}")
    for age, counts in meta["views_age_basis_counts"].items():
        print(f"  views_{age} basis: {counts}")
    print(f"  baseline_level: {df['baseline_level'].value_counts().to_dict()}")
    print(f"  local_baseline_level: {df['local_baseline_level'].value_counts().to_dict()}")
    print(f"  flags: {meta['flag_counts']}")
    if load_report["missing_columns"] or load_report["missing_optional_tables"]:
        print(f"  load warnings: {load_report}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
