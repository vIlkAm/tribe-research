#!/usr/bin/env python3
"""Select the study library: ~1,500 unique clips chosen for learning, not at random.

Inputs (all files, no DB): the full-run plan (``results/run_full``: manifest,
members, file_qc), the outcomes table (``tools/build_outcomes.py``) and the
metrics export (titles).

Design (docs/STUDY_SET.md explains the why):

1. **Labels first.** A member (one post of a content) has a usable label when
   its history is clean (no missing/broken/truncated history, not gone, not
   too young, upload date consistent) and its reach is measured against the
   same account's clips posted around the same time (``reach_rel_local`` with
   an ``account_local`` baseline). That controls for account size and growth.
   A content's reach is the mean over its labelled members, so clips posted on
   several platforms/accounts get a better-measured label.
2. **Eligibility.** At least one labelled member; the representative file
   decodes with video and audio; 5-90 s; title not mostly non-Latin (TRIBE
   transcribes English only).
3. **Deal quotas.** ~sqrt(eligible) per deal, floor 60 (or everything when a
   deal has fewer), cap 250, scaled to the target size.
4. **Lockbox.** 15 % of each deal's quota is drawn uniformly at random from
   the eligible pool (natural distribution, no oversampling). Final numbers
   are reported on it, so tail-oversampling below can't inflate results.
5. **Accounts first, then clips.** For the rest, accounts with >= 8 eligible
   contents get 6-12 clips each (<= 15 % of the deal unless the deal has too
   few accounts), stratified by the clip's reach quintile *within its own
   account*, tails oversampled (weights .25/.15/.20/.15/.25). Same account
   and audience with different content is the cleanest contrast we have.
   Engagement is balanced inside each reach stratum (terciles of the
   shrunk interactions rate within deal x platform; wide/unknown last), and
   multi-post contents win ties. Leftover quota is filled the same way from
   the remaining pool.
6. **Weights.** Every row records its stratum and inclusion probability so
   analyses can reweight to the natural distribution.

Output (``--out``, default results/study):
    manifest.jsonl   worker.py rows (+ split, deal, stratum, incl_prob, ...)
    selection.csv    one row per selected content with labels and design fields
    pilot.jsonl      10 pilot clips (manifest rows, 1 worker) ; first 3 = transcript check
    selection_report.md
    staging/<deal_slug>/<video_id>.mp4   hard links for tools/pod.sh push-videos
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from make_manifest import assign_workers  # noqa: E402

csv.field_size_limit(1 << 30)
BAD_FLAGS = ("flag_missing_snapshots", "flag_missing_upload_date", "flag_upload_date_inconsistent",
             "flag_too_young", "flag_views_missing", "flag_broken_history", "flag_unknown_account",
             "flag_history_truncated", "flag_video_gone")
QUINTILE_WEIGHTS = np.array([0.25, 0.15, 0.20, 0.15, 0.25])


def nonlatin_ratio(t: str) -> float:
    t = t or ""
    letters = [c for c in t if c.isalpha()]
    if not letters:
        return 0.0
    return sum(ord(c) > 0x24F for c in letters) / len(letters)


def truthy(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin(("true", "1", "t"))


def load(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    man = pd.read_json(args.run / "manifest.jsonl", lines=True)
    mem = pd.read_csv(args.run / "members.csv")
    qc = pd.read_csv(args.run / "file_qc.csv")
    oc = pd.read_parquet(args.outcomes) if args.outcomes.suffix == ".parquet" else pd.read_csv(args.outcomes)
    vp = pd.read_csv(args.metrics / "video_performances.csv", usecols=["id", "social_account_id", "video_title"],
                     engine="python")

    m = mem.merge(oc, left_on="vp_id", right_on="id", how="left", suffixes=("", "_oc"))
    m = m.merge(vp.rename(columns={"id": "vp_id"}), on="vp_id", how="left", suffixes=("", "_vp"))
    if "social_account_id" not in m and "social_account_id_vp" in m:
        m["social_account_id"] = m["social_account_id_vp"]
    bad = np.zeros(len(m), bool)
    for f in BAD_FLAGS:
        if f in m:
            bad |= truthy(m[f]).to_numpy()
    m["labelled"] = (~bad & m["reach_rel_local"].notna()
                     & (m["local_baseline_level"] == "account_local"))
    eng_w = (m["interactions_rate_eb_hi90"] - m["interactions_rate_eb_lo90"]) / m["interactions_rate_eb"]
    m["eng_known"] = m["labelled"] & m["pct_interactions_deal_platform"].notna() & (eng_w < 1.0)

    lab = m[m["labelled"]].copy()
    # anchor member = the labelled post with the most views (most reliable label)
    lab["_v"] = lab.get("views_final", pd.Series(0, index=lab.index)).fillna(0)
    anchor = lab.sort_values("_v", ascending=False).drop_duplicates("video_id")
    content = lab.groupby("video_id").agg(
        reach=("reach_rel_local", "mean"), n_labelled=("vp_id", "size"),
        platforms=("platform", lambda s: "+".join(sorted(set(s)))))
    eng = m[m["eng_known"]].groupby("video_id")["pct_interactions_deal_platform"].mean().rename("eng_pct")
    content = content.join(eng)
    content = content.join(anchor.set_index("video_id")[["vp_id", "social_account_id", "video_title", "views_final"]]
                           .rename(columns={"vp_id": "anchor_vp", "social_account_id": "anchor_account",
                                            "views_final": "anchor_views"}))
    c = man.merge(content, left_on="video_id", right_index=True, how="left")
    c = c.merge(qc, on="video_id", how="left")
    return c, m


def eligibility(c: pd.DataFrame, args) -> pd.DataFrame:
    reasons = pd.Series("", index=c.index)
    def mark(mask, why):
        reasons[(reasons == "") & mask] = why
    mark(c["reach"].isna(), "no_usable_label")
    mark(~truthy(c["decode_ok"]), "file_undecodable")
    mark(~truthy(c["has_audio"]), "no_audio")
    mark((c["duration_s"] < args.min_duration) | (c["duration_s"] > args.max_duration), "duration")
    mark(c["video_title"].fillna("").map(nonlatin_ratio) > 0.3, "non_latin_title")
    c = c.copy()
    c["excluded"] = reasons
    return c


def deal_quotas(sizes: pd.Series, target: int, floor: int, cap: int) -> pd.Series:
    lo, hi = 0.1, 100.0
    for _ in range(60):
        k = (lo + hi) / 2
        q = np.minimum(sizes, np.clip(np.round(k * np.sqrt(sizes)), floor, cap)).astype(int)
        if q.sum() > target:
            hi = k
        else:
            lo = k
    return pd.Series(np.minimum(sizes, np.clip(np.round(lo * np.sqrt(sizes)), floor, cap)).astype(int),
                     index=sizes.index)


def pick_stratified(pool: pd.DataFrame, n: int, rng: np.random.Generator, rank_col: str) -> list:
    """Pick n rows over reach quintiles (tails oversampled), engagement-balanced within each."""
    if n <= 0 or pool.empty:
        return []
    if n >= len(pool):
        return list(pool.index)
    q = pd.qcut(pool[rank_col].rank(method="first"), min(5, len(pool)), labels=False)
    nq = int(q.max()) + 1
    w = QUINTILE_WEIGHTS if nq == 5 else np.full(nq, 1 / nq)
    want = np.floor(w * n).astype(int)
    for i in np.argsort(-(w * n - want))[: n - want.sum()]:
        want[i] += 1
    chosen = []
    for qi in range(nq):
        cell = pool[q == qi]
        chosen += balanced_take(cell, want[qi], rng)
    short = n - len(chosen)
    if short > 0:  # a stratum ran dry: fill from the rest, nearest to the tails first
        rest = pool.drop(chosen)
        chosen += balanced_take(rest, short, rng)
    return chosen


def balanced_take(cell: pd.DataFrame, k: int, rng: np.random.Generator) -> list:
    if k <= 0 or cell.empty:
        return []
    cell = cell.assign(_r=rng.random(len(cell)))
    et = pd.Series("unknown", index=cell.index)
    known = cell["eng_pct"].notna()
    et[known] = pd.cut(cell.loc[known, "eng_pct"], [-1, 1 / 3, 2 / 3, 2], labels=["lo", "mid", "hi"]).astype(str)
    queues = []
    levels = ["lo", "mid", "hi"]
    shift = int(rng.integers(3))  # rotate the start so small cells don't always favour "lo"
    for lvl in levels[shift:] + levels[:shift] + ["unknown"]:
        sub = cell[et == lvl].sort_values(["n_members", "_r"], ascending=[False, True])
        queues.append(list(sub.index))
    out = []
    while len(out) < k and any(queues):
        for qu in queues[:3] if any(queues[:3]) else queues:
            if qu and len(out) < k:
                out.append(qu.pop(0))
    return out


def select(c: pd.DataFrame, args) -> pd.DataFrame:
    rng = np.random.default_rng(args.seed)
    el = c[c["excluded"] == ""].copy()
    el["acct_pct"] = el.groupby("anchor_account")["reach"].rank(pct=True)
    sizes = el.groupby("deal_id").size()
    quotas = deal_quotas(sizes, args.n, args.floor, args.cap)
    el["split"], el["stratum"], el["incl_prob"] = "", "", np.nan
    for deal, quota in quotas.items():
        pool = el[el["deal_id"] == deal]
        n_lock = int(round(args.lockbox * quota))
        lock = rng.choice(pool.index, size=n_lock, replace=False) if n_lock else []
        el.loc[lock, ["split", "stratum"]] = ["lockbox", "uniform"]
        el.loc[lock, "incl_prob"] = n_lock / len(pool)
        rest = pool.drop(lock)
        n_train = quota - n_lock
        chosen: list = []
        accts = rest.groupby("anchor_account").size()
        accts = accts[accts >= args.min_account_clips]
        if len(accts):
            cap = max(math.ceil(0.15 * n_train), math.ceil(n_train / len(accts)))
            per = np.clip(np.round(np.sqrt(accts) * n_train / np.sqrt(accts).sum()), 6, 12)
            per = np.minimum(np.minimum(per, cap), accts).astype(int)
            order = rng.permutation(len(per))
            budget = n_train
            for i in order:
                a, k = per.index[i], min(int(per.iloc[i]), budget)
                if k <= 0:
                    break
                ap = rest[(rest["anchor_account"] == a)]
                got = pick_stratified(ap, k, rng, "acct_pct")
                chosen += got
                budget -= len(got)
                for idx in got:
                    el.loc[idx, "stratum"] = f"acct:{int(min(4, el.loc[idx, 'acct_pct'] * 5 // 1))}"
            # top-up in account order if quota left and accounts have more clips (up to the cap)
            for i in order:
                if budget <= 0:
                    break
                a = per.index[i]
                have = sum(1 for x in chosen if el.loc[x, "anchor_account"] == a)
                room = min(cap - have, budget)
                if room <= 0:
                    continue
                ap = rest[(rest["anchor_account"] == a) & ~rest.index.isin(chosen)]
                got = pick_stratified(ap, room, rng, "acct_pct")
                chosen += got
                budget -= len(got)
                for idx in got:
                    el.loc[idx, "stratum"] = f"acct:{int(min(4, el.loc[idx, 'acct_pct'] * 5 // 1))}"
        left = n_train - len(chosen)
        if left > 0:
            fill_pool = rest.drop(chosen)
            fill_pool = fill_pool.assign(deal_pct=fill_pool["reach"].rank(pct=True))
            got = pick_stratified(fill_pool, left, rng, "deal_pct")
            chosen += got
            for idx in got:
                el.loc[idx, "stratum"] = f"deal:{int(min(4, fill_pool.loc[idx, 'deal_pct'] * 5 // 1))}"
        el.loc[chosen, "split"] = "train"
        # inclusion probability = selected / available in the same (pool, reach-quintile) cell
        ch = el.loc[chosen]
        acct_rows = ch[ch["stratum"].str.startswith("acct:")]
        for a_id, grp in acct_rows.groupby("anchor_account"):
            pool_a = rest[rest["anchor_account"] == a_id]
            pool_bins = (pool_a["acct_pct"] * 5 // 1).clip(upper=4).astype(int)
            for b_, g2 in grp.groupby(grp["stratum"].str[-1].astype(int)):
                el.loc[g2.index, "incl_prob"] = len(g2) / max(len(g2), int((pool_bins == b_).sum()))
        fill_rows = ch[ch["stratum"].str.startswith("deal:")]
        if len(fill_rows):
            fp = rest.drop([x for x in chosen if x not in fill_rows.index])
            fbins = (fp["reach"].rank(pct=True) * 5 // 1).clip(upper=4).astype(int)
            for b_, g2 in fill_rows.groupby(fill_rows["stratum"].str[-1].astype(int)):
                el.loc[g2.index, "incl_prob"] = len(g2) / max(len(g2), int((fbins == b_).sum()))
    return el[el["split"] != ""].copy(), quotas


def deep_dive(c: pd.DataFrame, sel: pd.DataFrame, args) -> pd.DataFrame:
    """Per-account add-on: the biggest eligible account of the largest deals, up to N clips each.

    The main set has 6-12 clips per account: enough to tune per deal, not per
    account. These accounts get enough clips (natural distribution, uniform
    random, no oversampling) to fit and test an account-tuned model and to draw
    a learning curve of how many clips a new account must submit.
    """
    if args.deep_dive_accounts <= 0:
        return pd.DataFrame()
    el = c[(c["excluded"] == "") & ~c["video_id"].isin(sel["video_id"])]
    counts = el.groupby(["deal_id", "anchor_account"]).size().rename("n").reset_index()
    counts = counts[counts["n"] >= args.deep_dive_n // 2]
    best = counts.sort_values("n", ascending=False).drop_duplicates("deal_id").head(args.deep_dive_accounts)
    rng = np.random.default_rng(args.seed + 1)
    rows = []
    for _, b in best.iterrows():
        pool = el[el["anchor_account"] == b["anchor_account"]]
        k = min(args.deep_dive_n, len(pool))
        pick = pool.loc[rng.choice(pool.index, size=k, replace=False)].copy()
        pick["split"], pick["stratum"], pick["incl_prob"] = "deep_dive", "account_uniform", k / len(pool)
        rows.append(pick)
    return pd.concat(rows) if rows else pd.DataFrame()


def icc(m: pd.DataFrame) -> dict:
    lab = m[m["labelled"]]
    g = lab.groupby("video_id")["reach_rel_local"]
    multi = lab[lab["video_id"].isin(g.size()[g.size() > 1].index)]
    if multi.empty:
        return {"n_groups": 0}
    tot = multi["reach_rel_local"].var()
    within = multi.groupby("video_id")["reach_rel_local"].var().mean()
    return {"n_groups": int(multi["video_id"].nunique()), "n_posts": int(len(multi)),
            "total_var": round(float(tot), 3), "within_content_var": round(float(within), 3),
            "icc_content": round(float(1 - within / tot), 3)}


def pick_pilot(sel: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    tr = sel[sel["split"] == "train"]
    targets = np.quantile(tr["duration_s"], [0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9])
    chosen = []
    plats_needed = ["tiktok", "instagram", "youtube"]
    for t in targets:
        cand = tr.drop(chosen).assign(d=(tr["duration_s"] - t).abs()).sort_values("d").head(40)
        need = [p for p in plats_needed if not any(p in str(x) for x in sel.loc[chosen, "platforms"])]
        if need:
            pref = cand[cand["platforms"].astype(str).str.contains(need[0])]
            cand = pref if len(pref) else cand
        chosen.append(cand.index[0])
    long = tr.drop(chosen)
    long = long[long["duration_s"].between(70, 90)]
    if len(long):
        chosen.append(long.sample(1, random_state=1).index[0])
    quiet = tr.drop(chosen).sort_values("audio_mean_db").head(1)  # likely little/no speech
    chosen += list(quiet.index)
    for deal_kw in ("diagofit", "polymarket"):
        if not any(deal_kw in p for p in sel.loc[chosen, "path"]):
            alt = tr.drop(chosen)
            alt = alt[alt["path"].str.startswith(deal_kw)]
            if len(alt):
                chosen.append(alt.assign(d=(alt["duration_s"] - np.median(tr["duration_s"])).abs())
                              .sort_values("d").index[0])
    while len(chosen) < 10:  # fill from deals not yet in the pilot
        have = {p.split("/")[0] for p in sel.loc[chosen, "path"]}
        alt = tr.drop(chosen)
        alt2 = alt[~alt["path"].str.split("/").str[0].isin(have)]
        alt = alt2 if len(alt2) else alt
        chosen.append(alt.sample(1, random_state=len(chosen)).index[0])
    return sel.loc[chosen[:10]]


def write_manifest(rows: pd.DataFrame, path: Path, workers: int, extra: list[str]) -> None:
    rows = rows.copy()
    rows["worker"] = assign_workers(list(rows["duration_s"]), workers)
    rows["num_workers"] = workers
    base = ["video_id", "path", "source_name", "duration_s", "size_bytes", "worker", "num_workers"]
    with path.open("w") as f:
        for _, r in rows.iterrows():
            f.write(json.dumps({k: (None if (isinstance(r[k], float) and math.isnan(r[k])) else
                                    (r[k].item() if hasattr(r[k], "item") else r[k]))
                                for k in base + extra}) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", type=Path, default=ROOT / "results/run_full")
    ap.add_argument("--outcomes", type=Path, default=ROOT / "results/outcomes.parquet")
    ap.add_argument("--metrics", type=Path, default=ROOT / "results/metrics")
    ap.add_argument("--out", type=Path, default=ROOT / "results/study")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--floor", type=int, default=60)
    ap.add_argument("--cap", type=int, default=250)
    ap.add_argument("--lockbox", type=float, default=0.15)
    ap.add_argument("--min-account-clips", type=int, default=8)
    ap.add_argument("--min-duration", type=float, default=5.0)
    ap.add_argument("--max-duration", type=float, default=90.0)
    ap.add_argument("--workers", type=int, default=8, help="reassign on the pod once the GPU count is known")
    ap.add_argument("--seed", type=int, default=20260926)
    ap.add_argument("--deep-dive-accounts", type=int, default=5,
                    help="extra per-account set for niche/account tuning (0 = none)")
    ap.add_argument("--deep-dive-n", type=int, default=80, help="clips per deep-dive account")
    ap.add_argument("--no-stage", action="store_true")
    args = ap.parse_args()

    c, m = load(args)
    c = eligibility(c, args)
    sel, quotas = select(c, args)
    args.out.mkdir(parents=True, exist_ok=True)
    extra = ["split", "stratum", "incl_prob", "deal_id", "n_members", "platforms"]
    write_manifest(sel, args.out / "manifest.jsonl", args.workers, extra)
    pilot = pick_pilot(sel, np.random.default_rng(args.seed))
    write_manifest(pilot, args.out / "pilot.jsonl", 1, extra)
    cols = ["video_id", "path", "deal_id", "split", "stratum", "incl_prob", "duration_s", "reach", "eng_pct",
            "n_members", "n_labelled", "platforms", "anchor_vp", "anchor_account", "anchor_views",
            "width", "height", "vcodec", "audio_mean_db"]
    sel[cols].to_csv(args.out / "selection.csv", index=False)

    deep = deep_dive(c, sel, args)
    if len(deep):
        write_manifest(deep, args.out / "deep_dive.jsonl", args.workers, extra)
        deep[cols].to_csv(args.out / "deep_dive.csv", index=False)

    if not args.no_stage:
        # drop links left by an earlier draw so push-videos sends exactly this set
        # (they are hard links/copies; the run staging and the archive keep the files)
        keep = {args.out / "staging" / p for p in pd.concat([sel, deep])["path"]}
        for old in (args.out / "staging").rglob("*.mp4"):
            if old not in keep:
                old.unlink()
        for _, r in pd.concat([sel, deep]).iterrows():
            src = args.run / "staging" / f"chunk{int(r['chunk'])}" / r["path"]
            dst = args.out / "staging" / r["path"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                try:
                    os.link(src, dst)
                except OSError:
                    shutil.copyfile(src, dst)

    report(c, m, sel, quotas, pilot, args, deep)
    print(f"selected {len(sel)} ({(sel['split'] == 'lockbox').sum()} lockbox), "
          f"{sel['duration_s'].sum() / 3600:.1f} h of source -> {args.out}", file=sys.stderr)
    return 0


def report(c, m, sel, quotas, pilot, args, deep=None) -> None:
    deal_name = pd.read_csv(args.metrics / "deals.csv").set_index("id")["name"]
    L = ["# Study set selection report", "",
         f"Seed {args.seed}. Target {args.n}; selected **{len(sel)}** unique contents "
         f"({sel['duration_s'].sum() / 3600:.1f} h of source, median {sel['duration_s'].median():.1f} s). "
         f"Lockbox {int((sel['split'] == 'lockbox').sum())} (uniform random, 15 % per deal).", "",
         "## Exclusions (full plan of %d contents)" % len(c), "",
         "| reason | contents |", "|---|---|"]
    for k, v in c["excluded"].replace("", "eligible").value_counts().items():
        L.append(f"| {k} | {v} |")
    L += ["", "## Per deal", "", "| deal | eligible | quota | lockbox | train | accounts | multi-post | "
          "reach sd | engagement known |", "|---|---|---|---|---|---|---|---|---|"]
    el = c[c["excluded"] == ""]
    for d, q in quotas.sort_values(ascending=False).items():
        s = sel[sel["deal_id"] == d]
        L.append(f"| {deal_name.get(d, d)} | {int((el['deal_id'] == d).sum())} | {q} | "
                 f"{int((s['split'] == 'lockbox').sum())} | {int((s['split'] == 'train').sum())} | "
                 f"{s['anchor_account'].nunique()} | {int((s['n_members'] > 1).sum())} | "
                 f"{s['reach'].std():.2f} | {int(s['eng_pct'].notna().sum())} |")
    L += ["", "Per-deal findings are only meaningful for deals with ~100+ clips here; smaller deals "
          "feed the cross-deal model.", "", "## Strata (train)", ""]
    st = sel[sel["split"] == "train"]["stratum"].value_counts().sort_index()
    L += ["| stratum | clips |", "|---|---|"] + [f"| {k} | {v} |" for k, v in st.items()]
    q = pd.crosstab(pd.cut(sel["reach"].rank(pct=True), [0, 1 / 3, 2 / 3, 1], labels=["low", "mid", "high"]),
                    pd.cut(sel["eng_pct"], [-1, 1 / 3, 2 / 3, 2], labels=["low", "mid", "high"]).astype(str),
                    rownames=["reach"], colnames=["engagement"])
    L += ["", "## Reach x engagement coverage (reach terciles of the set; engagement terciles within deal x platform; "
          "nan = too few views to know)", "", "```", q.to_string(), "```", "",
          "## Platform mix (posts behind the selected contents)", ""]
    posts = m[m["video_id"].isin(sel["video_id"])]
    L += ["| platform | posts | labelled |", "|---|---|---|"]
    for p, g in posts.groupby("platform"):
        L.append(f"| {p} | {len(g)} | {int(g['labelled'].sum())} |")
    ic = icc(m)
    L += ["", "## How much of reach is about the content at all?", "",
          f"Across {ic.get('n_groups', 0)} contents posted more than once with labels "
          f"({ic.get('n_posts', 0)} posts), the share of reach variance explained by *which content it is* "
          f"(intraclass correlation) is **{ic.get('icc_content', 'n/a')}**. The rest is platform, account, "
          "timing and luck. Treat it as a rough ceiling for any content model, brain features or not.", "",
          "## Pilot (10 clips; first 3 also run the stock transcription path for comparison)", "",
          "| video_id | deal | duration s | platforms | audio dB |", "|---|---|---|---|---|"]
    for _, r in pilot.iterrows():
        L.append(f"| {r['video_id']} | {r['path'].split('/')[0]} | {r['duration_s']:.1f} | {r['platforms']} | "
                 f"{r['audio_mean_db']} |")
    if deep is not None and len(deep):
        deal_name2 = pd.read_csv(args.metrics / "deals.csv").set_index("id")["name"]
        L += ["", "## Account deep-dive add-on (optional second batch)", "",
              f"{len(deep)} more clips ({deep['duration_s'].sum() / 3600:.1f} h), uniform random within each "
              "account, for account-tuned models and the 'how many clips must a new account submit' curve.", "",
              "| deal | account | clips | of eligible |", "|---|---|---|---|"]
        for (d, a), g in deep.groupby(["deal_id", "anchor_account"]):
            L.append(f"| {deal_name2.get(d, d)} | {str(a)[:8]} | {len(g)} | {round(len(g) / g['incl_prob'].iloc[0])} |")
    (args.out / "selection_report.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
