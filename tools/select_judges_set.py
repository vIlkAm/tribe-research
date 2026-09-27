#!/usr/bin/env python3
"""INTERNAL: the hackathon judges' browse set, about 10 clips per views group, from the candidates file.

    .venv/bin/python tools/select_judges_set.py --candidates results/demo/candidates_demo.json \
        --out results/demo/judges_demo.json

Rule (fixed before looking at brain data; never reads it): the owner's two hand-picked clips first (Connor great,
Polymarket TikTok bad), then the remaining candidates in hash order (``rank_key``), at most 2 clips per deal per
group, up to 10 per group. Bad clips need at least 100 views (clips with a handful of views were probably never
distributed). Candidates already passed the transcript screen, the 45 s cap and the lockbox exclusion
(``tools/library_tiers.py --all-candidates``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PER_TIER = 10
PER_DEAL = 2
MIN_BAD_VIEWS = 100
OWNER_FIRST = {"great": ["ef3335a5c1b282e3"], "bad": ["b17bdcaf52dc2195"]}


def select(picks: dict) -> dict:
    out = {}
    for tier, ps in picks.items():
        ok = [p for p in ps if tier != "bad" or p["observed"]["views"] >= MIN_BAD_VIEWS]
        first = [p for v in OWNER_FIRST.get(tier, []) for p in ok if p["video_id"] == v]
        rest = sorted((p for p in ok if p not in first), key=lambda p: p["rank_key"])
        chosen, per_deal = [], {}
        for p in first + rest:
            if len(chosen) == PER_TIER:
                break
            if per_deal.get(p["deal_id"], 0) >= PER_DEAL and p not in first:
                continue
            per_deal[p["deal_id"]] = per_deal.get(p["deal_id"], 0) + 1
            chosen.append(p)
        out[tier] = chosen
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    c = json.loads(args.candidates.read_text())
    picks = select(c["picks"])
    out = {k: v for k, v in c.items() if k != "picks"} | {
        "selection": "candidates", "picks": picks,
        "judges_rule": (f"Owner's hand-picked great and bad clip first, then hash order, at most {PER_DEAL} per deal, "
                        f"up to {PER_TIER} per group; bad clips need >= {MIN_BAD_VIEWS} views."),
        "counts": {t: len(ps) for t, ps in picks.items()}}
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    for t, ps in picks.items():
        print(t, len(ps), [(p["deal_label"], p["platform"], int(p["observed"]["views"])) for p in ps])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
