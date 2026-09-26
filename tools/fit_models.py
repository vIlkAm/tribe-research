#!/usr/bin/env python3
"""Does predicted brain response add out-of-sample predictive value beyond the obvious features?

Joins per-clip features (tools/build_features.py, keyed by ``video_id``) to
posts via ``members.csv`` (``video_id`` -> ``vp_id``) and to per-post outcomes
(one row per ``vp_id``), restricted to the study selection
(``results/study/selection.csv``: ``split`` train/lockbox, ``incl_prob``), and
compares, at POST level:

    A  baseline: platform, deal, log duration, follower bucket, posting weekday/hour
       (when available), speech/shot features (``base_*``), selection audio level/aspect,
       and an out-of-fold account-mean target encoding (``te_account_mean``, refit in every fit)
    B  A + every ``brain_*`` feature
    E  A + every ``emb_*`` feature: the extractor features TRIBE's brain mapping consumes, without the mapping
       (control arm; only when the features table has them). B > A alone shows that SOME content
       representation beats metadata; the brain mapping earns credit only through BE − E (BE = A + brain + emb)
       and B − E. docs/PREREGISTRATION.md names the primary comparison.

Models: ``stack`` (the pre-registered primary: each ``brain_*``/``emb_*`` block compressed to one inner-OOF
ridge score, then ridge on A + scores; see BlockStackRegressor), ``ridge`` (one shared penalty over every
column; overfits wide blocks) and ``hgb``.
    C  niche-tuned: a general B model fit without the niche, plus a partially pooled
       (ridge-shrunk, penalty by grouped inner CV) residual adjustment fit on the
       niche's own training clips (deal; accounts with enough posts on top)

Evaluation:

    train split, weighted by 1/incl_prob in every metric:
      content  GroupKFold by content component (video_id + outcomes content_group)
      account  GroupKFold by account component (posting + anchor accounts, merged with content components)
      lodo     leave one deal out (general model on an unseen niche; bootstrap over deals)
      niche    per deal: general (B) vs tuned (C) on content folds, with a learning
               curve of C vs the number of the niche's clips used
    lockbox split, unweighted, computed ONCE with models fit on all train rows, and only with
      ``--score-lockbox`` (otherwise no lockbox model is fit or reported):
      the final A vs B (and B vs C) numbers with paired cluster-bootstrap CIs.

Headline metric: within-stratum Spearman (weighted mean rank correlation inside
each deal×platform), next to pooled Spearman and R², per platform, with the
content-identity ICC ceilings computed from reposts.

Everything is an association on observational data; nothing here shows that
changing a clip toward a feature value would change its performance.

    python tools/fit_models.py --features results/features/features.parquet \
        --members results/run_full/members.csv --outcomes results/outcomes.parquet \
        --selection results/study/selection.csv --out-dir results/models [--score-lockbox] \
        [--save-model results/models/served [--save-feature-set BE] [--save-model-kind stack] \
         [--lockbox-ext results/study/lockbox_ext.csv]]

``--save-model DIR`` is opt-in and runs only after the evaluation above has written its outputs (which it does
not change): it refits the chosen feature set/model exactly as the lockbox fit does (all non-lockbox train rows,
same seed and groups), and writes the model, its manifest (stage-1 metrics, hashes, versions) and the reference
tables tools/predict.py ranks a new clip against (content-scheme out-of-fold predictions; never a lockbox row).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin

sys.path.insert(0, str(Path(__file__).resolve().parent))
from select_study_set import BAD_FLAGS  # noqa: E402  (the study set's label rules; one source of truth)

# ── adjust here if the outcomes/selection tables use other names ────────────
# canonical -> candidate columns in the outcomes table (first present wins). Only these,
# the target source columns, their row requirements and the exclusion flags are read:
# views, percentiles, quadrants and other rates are derived from the targets and would leak.
OUTCOME_COLUMNS = {
    "vp_id": ["vp_id", "video_performance_id", "id"],
    "deal_id": ["deal_id"],
    "platform": ["platform"],
    "social_account_id": ["social_account_id", "account_id"],
    "content_group": ["content_group", "content_group_id"],
    "follower_count": ["followers_at_post", "follower_count_at_post", "follower_count", "followers"],
    "posted_at": ["posted_at", "upload_date", "published_at"],
}
# target key -> source column candidates, transform (None | "log"), row requirements {column: allowed values},
# target-only exclusion flags, and whether the shrunk-rate "unknown" rule applies (see UNKNOWN_REL_WIDTH)
TARGETS = {
    "reach_rel_local": {"column": ["reach_rel_local"], "transform": None,
                        "require": {"local_baseline_level": ["account_local"]}},
    "log_interactions_rate": {"column": ["interactions_rate_eb", "interactions_rate_post"], "transform": "log",
                              "require": {}, "exclude_if_true": ["flag_views_zero", "flag_rate_gt1"],
                              "unknown_if_wide": True},
}
DEFAULT_TARGETS = ["reach_rel_local", "log_interactions_rate"]
# post rows where any of these is truthy are dropped (for every target): the study set's label flags
EXCLUDE_IF_TRUE = list(BAD_FLAGS) + ["exclude", "quality_exclude"]
# select_study_set.load (eng_known): a shrunk rate whose 90% interval is at least as wide as the rate itself,
# (<col>_hi90 - <col>_lo90) / <col> >= 1, is "unknown" (too few views to know), not "middle"
UNKNOWN_REL_WIDTH = 1.0
# selection table: content-level study design
SELECTION = {"video_id": "video_id", "split": "split", "incl_prob": "incl_prob",
             "anchor_account": "anchor_account", "stratum": "sel_stratum"}
SELECTION_BASE = {"audio_mean_db": "base_audio_mean_db"}  # + base_aspect from height/width
# ──────────────────────────────────────────────────────────────────────────

SCHEMES = ("content", "account", "lodo")
# (new, reference) feature-set pairs whose paired delta every bootstrap reports, when both sets were fit
PAIRS = (("B", "A"), ("E", "A"), ("BE", "E"), ("B", "E"), ("BE", "A"))  # docs/PREREGISTRATION.md: BE − E primary, BE − A go/no-go
MODELS = ("stack", "ridge", "hgb")  # stack: pre-registered primary (docs/PREREGISTRATION.md)
BLOCK_PREFIXES = (("brain", "brain_"), ("emb", "emb_"))  # wide clip-feature blocks the stack compresses to one score
MIN_STRATUM_N = 10
MIN_ICC_GROUPS = 20
MIN_CLUSTERS = 5  # accounts needed for an account-clustered p-value
LEARNING_CURVE_K = (10, 20, 40, 80)
ALPHAS_ADJ = np.logspace(-1, 5, 13)
ACCOUNT_TE = "te_account_mean"  # A's account term: out-of-fold account-mean target encoding (AccountTargetEncoder)
ACCOUNT_TE_SMOOTH = 10.0  # pseudo-posts pulling an account's mean toward its deal×platform mean
TE_INPUTS = ["social_account_id", "deal_id", "platform", "_content"]
HGB_PARAMS = {"learning_rate": 0.05, "max_iter": 300, "max_leaf_nodes": 15, "min_samples_leaf": 20,
              "l2_regularization": 1.0}
FORBIDDEN_REPORT_WORDS = ("cause", "caused", "causes", "causing", "causal", "drives", "impact", "viral", "firing")


# ── io ────────────────────────────────────────────────────────────────────


def read_table(path: Path) -> pd.DataFrame:
    s = path.suffix.lower()
    if s == ".parquet":
        return pd.read_parquet(path)
    if s in (".jsonl", ".ndjson"):
        return pd.read_json(path, lines=True)
    return pd.read_csv(path, low_memory=False)


def resolve(columns, candidates: list[str]) -> str | None:
    return next((c for c in candidates if c in columns), None)


def _truthy(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    return s.astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes", "y", "1.0"})


# ── dataset ───────────────────────────────────────────────────────────────


def load_posts(members: pd.DataFrame, outcomes: pd.DataFrame, targets: list[str],
               exclude_cols: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """members ⨝ outcomes at post level, quality-filtered, with transformed targets (no features yet)."""
    info: dict = {"n_outcomes": int(len(outcomes))}
    ren, keep = {}, []
    for canon, cands in OUTCOME_COLUMNS.items():
        c = resolve(outcomes.columns, cands)
        if c is not None:
            ren[c] = canon
            keep.append(c)
    if "vp_id" not in ren.values():
        raise SystemExit(f"outcomes has no vp id column (tried {OUTCOME_COLUMNS['vp_id']})")
    tspec = {}
    for t in targets:
        spec = TARGETS.get(t, {"column": [t], "transform": None, "require": {}})
        c = resolve(outcomes.columns, spec["column"])
        if c is None:
            print(f"warning: target {t!r} not found (tried {spec['column']})", file=sys.stderr)
            continue
        req = {k: v for k, v in spec.get("require", {}).items() if k in outcomes.columns}
        texcl = [f for f in spec.get("exclude_if_true", []) if f in outcomes.columns]
        width = [f"{c}_lo90", f"{c}_hi90"] if spec.get("unknown_if_wide") else []
        width = width if set(width) <= set(outcomes.columns) else []  # no interval (e.g. a raw rate): rule skipped
        tspec[t] = {"column": c, "transform": spec.get("transform"), "require": req, "exclude_if_true": texcl,
                    "unknown_rule": bool(width)}
        keep += [c, *req, *texcl, *width]
    if not tspec:
        raise SystemExit("no target column found in outcomes")
    flags = [c for c in EXCLUDE_IF_TRUE + list(exclude_cols or []) if c in outcomes.columns]
    o = outcomes[list(dict.fromkeys(keep + flags))].copy()
    bad = pd.Series(False, index=o.index)
    info["excluded_by_flag"] = {}
    for c in flags:
        b = _truthy(o[c])
        info["excluded_by_flag"][c] = int(b.sum())
        bad |= b
    info["excluded_by_quality_flags"] = int(bad.sum())
    o = o[~bad]
    for t, s in tspec.items():
        v = pd.to_numeric(o[s["column"]], errors="coerce").astype(float)
        if s["transform"] == "log":
            v = np.log(v.where(v > 0))
        ok = pd.Series(True, index=o.index)
        for col, allowed in s["require"].items():
            ok &= o[col].isin(allowed)
        info.setdefault("target_rows_failing_requirements", {})[t] = int((~ok & v.notna()).sum())
        flagged = pd.Series(False, index=o.index)
        for col in s["exclude_if_true"]:
            flagged |= _truthy(o[col])
        info.setdefault("target_rows_excluded_by_flag", {})[t] = int((ok & flagged & v.notna()).sum())
        ok &= ~flagged
        if s["unknown_rule"]:
            c = s["column"]
            eb = pd.to_numeric(o[c], errors="coerce")
            wid = (pd.to_numeric(o[f"{c}_hi90"], errors="coerce") - pd.to_numeric(o[f"{c}_lo90"], errors="coerce")) / eb
            known = wid < UNKNOWN_REL_WIDTH  # NaN width (no interval) is unknown too, as in the selector
            info.setdefault("target_rows_unknown", {})[t] = int((ok & ~known & v.notna()).sum())
            ok &= known
        o[f"y_{t}"] = v.where(ok)
    o = o.rename(columns=ren)
    o = o[[c for c in o.columns if c in OUTCOME_COLUMNS or c.startswith("y_")]]
    o["vp_id"] = o["vp_id"].astype(str)
    o = o.drop_duplicates("vp_id")
    m = members[["video_id", "vp_id"] + [c for c in ("platform", "deal_id") if c in members.columns]].copy()
    m["vp_id"] = m["vp_id"].astype(str)
    df = m.merge(o, on="vp_id", how="inner", suffixes=("_members", ""))
    for c in ("platform", "deal_id"):
        if f"{c}_members" in df:
            df[c] = df[c].fillna(df[f"{c}_members"]) if c in df else df[f"{c}_members"]
            df = df.drop(columns=f"{c}_members")
    for c in ("platform", "deal_id", "social_account_id"):
        if c not in df:
            df[c] = "missing"
        df[c] = df[c].astype("string").fillna("missing").astype(str)
    info["n_posts_with_outcomes"] = int(len(df))
    return df.reset_index(drop=True), {"targets": tspec, **info}


def follower_bucket(follower_count) -> pd.Series:
    """A's follower term: floor(log10(1 + followers)); unknown stays NaN."""
    fc = pd.to_numeric(pd.Series(follower_count), errors="coerce")
    return np.floor(np.log10(1.0 + fc.clip(lower=0)))


def post_time(posted_at) -> tuple[pd.Series, pd.Series]:
    """(weekday, hour + minute/60) in UTC; unparseable stays NaN."""
    ts = pd.to_datetime(pd.Series(posted_at), utc=True, errors="coerce", format="mixed")
    return ts.dt.weekday.astype("float"), ts.dt.hour + ts.dt.minute / 60.0


def aspect(height, width) -> pd.Series:
    """base_aspect: height / width (0 width -> NaN)."""
    return pd.to_numeric(pd.Series(height), errors="coerce") / pd.to_numeric(
        pd.Series(width), errors="coerce").replace(0, np.nan)


def build_dataset(features: pd.DataFrame, members: pd.DataFrame, outcomes: pd.DataFrame,
                  targets: list[str], selection: pd.DataFrame | None = None,
                  exclude_cols: list[str] | None = None,
                  exclude_non_english: bool = True) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """-> (post rows with features + design columns, info, all quality-filtered posts for ICCs)."""
    posts, info = load_posts(members, outcomes, targets, exclude_cols)
    info["n_features_rows"] = int(len(features))
    df = posts
    if selection is not None:
        sel = selection.rename(columns={k: v for k, v in SELECTION.items() if k in selection.columns})
        cols = [c for c in SELECTION.values() if c in sel.columns]
        extra = sel[["video_id"]].copy()
        for src, dst in SELECTION_BASE.items():
            if src in sel:
                extra[dst] = pd.to_numeric(sel[src], errors="coerce")
        if {"width", "height"} <= set(sel.columns):
            extra["base_aspect"] = aspect(sel["height"], sel["width"])
        sel = sel[cols].merge(extra, on="video_id")
        df = df.merge(sel, on="video_id", how="inner")
        info["n_posts_in_selection"] = int(len(df))
    if "split" not in df:
        df["split"] = "train"
    if "incl_prob" not in df:
        df["incl_prob"] = 1.0
    df["split"] = df["split"].astype(str)
    df["_w"] = 1.0 / pd.to_numeric(df["incl_prob"], errors="coerce").clip(lower=1e-6).fillna(1.0)

    f = features[features["status"] == "ok"] if "status" in features else features
    if exclude_non_english and "qc_non_english" in f:
        # TRIBE transcribes as English: a confident non-English detection makes the text features garbage
        non_en = pd.to_numeric(f["qc_non_english"], errors="coerce").fillna(0) > 0
        info["clips_excluded_non_english"] = int(non_en.sum())
        f = f[~non_en]
    fcols = ["video_id"] + [c for c in f.columns if c.startswith(("base_", "brain_", "emb_"))]
    info["synthetic_clips"] = int(f["synthetic"].fillna(False).astype(bool).sum()) if "synthetic" in f else 0
    df = df.merge(f[fcols], on="video_id", how="inner").copy()
    info["n_posts_with_features"] = int(len(df))

    if "follower_count" in df:
        df["base_follower_bucket"] = follower_bucket(df["follower_count"])
        fc = pd.to_numeric(df["follower_count"], errors="coerce")
        info["follower_known_share"] = round(float(fc.notna().mean()), 4) if len(df) else None
    if "posted_at" in df:
        df["base_post_weekday"], hours = post_time(df["posted_at"])
        if hours.dropna().nunique() > 1:  # date-only timestamps carry no hour
            df["base_post_hour"] = hours.astype("float")
    df = df.reset_index(drop=True)
    df["_content"] = content_components(df)
    df["_account"] = account_components(df)
    df["_stratum"] = df["deal_id"] + "|" + df["platform"]
    # lockbox content linked (via content_group) to train content is not a clean holdout
    comp_splits = df.groupby("_content")["split"].nunique()
    leaked = df["_content"].isin(comp_splits[comp_splits > 1].index) & (df["split"] == "lockbox")
    info["lockbox_posts_linked_to_train_dropped"] = int(leaked.sum())
    df = df[~leaked].reset_index(drop=True)
    for s in ("train", "lockbox"):
        d = df[df["split"] == s]
        info[s] = {"posts": int(len(d)), "contents": int(d["video_id"].nunique()),
                   "accounts": int(d["social_account_id"].nunique()), "deals": int(d["deal_id"].nunique())}
    tr = df[df["split"] == "train"]
    info["account_groups_train"] = int(tr["_account"].nunique())
    info["largest_account_group_share"] = (round(float(tr["_account"].value_counts().iloc[0] / len(tr)), 4)
                                           if len(tr) else None)
    return df, info, posts


def _components(keys: list[pd.Series]) -> np.ndarray:
    """Row -> connected-component id, linking every non-null key in the same row."""
    n = len(keys[0])
    codes, offset = [], 0
    for k in keys:
        ks = k.astype("string")
        valid = ks.notna() & (ks != "") & (ks != "missing") & (ks != "nan")
        c, uniq = pd.factorize(ks.where(valid))
        codes.append(np.where(c >= 0, c + offset, -1))
        offset += len(uniq)
    row_nodes = np.arange(n) + offset
    src = np.concatenate([row_nodes[c >= 0] for c in codes])
    dst = np.concatenate([c[c >= 0] for c in codes])
    g = coo_matrix((np.ones(len(src)), (src, dst)), shape=(offset + n, offset + n))
    _, lab = connected_components(g, directed=False)
    return pd.factorize(lab[row_nodes])[0]


def content_components(df: pd.DataFrame) -> np.ndarray:
    keys = [df["video_id"]]
    if "content_group" in df:
        keys.append(df["content_group"])
    return _components(keys)


def account_components(df: pd.DataFrame) -> np.ndarray:
    """Content components merged by every posting account and, when the selection provides it, the anchor
    account. About half of the train posts come from non-anchor accounts, so grouping by the anchor alone
    would let a posting account appear on both sides of a fold."""
    keys = [df["video_id"], df["social_account_id"]]
    if "anchor_account" in df:
        keys.append(df["anchor_account"])
    if "content_group" in df:
        keys.append(df["content_group"])
    return _components(keys)


# ── splits ────────────────────────────────────────────────────────────────


def make_splits(df: pd.DataFrame, scheme: str, n_splits: int = 5, seed: int = 0,
                min_deal_n: int = 30) -> list[tuple[str, np.ndarray, np.ndarray]]:
    from sklearn.model_selection import GroupKFold

    if scheme in ("content", "account"):
        groups = df["_content" if scheme == "content" else "_account"].to_numpy()
        k = min(n_splits, len(np.unique(groups)))
        if k < 2:
            return []
        gkf = GroupKFold(n_splits=k, shuffle=True, random_state=seed)
        return [(f"fold{i}", tr, te) for i, (tr, te) in enumerate(gkf.split(df, groups=groups))]
    if scheme == "lodo":
        out = []
        deal, content = df["deal_id"].to_numpy(), df["_content"].to_numpy()
        for d, n in df["deal_id"].value_counts().items():
            if n < min_deal_n or d == "missing":
                continue
            te = np.flatnonzero(deal == d)
            tr = np.flatnonzero((deal != d) & ~np.isin(content, np.unique(content[te])))
            if len(tr) >= min_deal_n and len(np.unique(deal[tr])) >= 2:
                out.append((str(d), tr, te))
        return out
    raise ValueError(scheme)


# ── models ────────────────────────────────────────────────────────────────


class GroupedRidgeCV(RegressorMixin, BaseEstimator):
    """Ridge whose penalty is chosen by GroupKFold over ``groups`` (content), not leave-one-row-out.

    Reposts share identical clip features and correlated outcomes, so row-level LOO (RidgeCV's default)
    rewards memorising a content and picks too small a penalty. All penalties are solved per fold via one SVD.
    """

    def __init__(self, alphas=tuple(np.logspace(-2, 5, 29)), n_splits: int = 5, random_state: int = 0):
        self.alphas = alphas
        self.n_splits = n_splits
        self.random_state = random_state

    @staticmethod
    def _solve(X, y, w, alphas):
        sw = np.sqrt(w / w.sum())
        xm, ym = (w / w.sum()) @ X, (w / w.sum()) @ y
        U, sv, Vt = np.linalg.svd((X - xm) * sw[:, None], full_matrices=False)
        uy = U.T @ ((y - ym) * sw)
        # the rows are scaled by sqrt(w / sum w); dividing alpha by sum(w) keeps sklearn's sum-of-squares scale
        d = sv[None, :] / (sv[None, :] ** 2 + np.asarray(alphas)[:, None] / w.sum())
        coefs = (d * uy[None, :]) @ Vt  # (n_alpha, p)
        return coefs, ym - coefs @ xm

    def fit(self, X, y, sample_weight=None, groups=None):
        from sklearn.model_selection import GroupKFold

        X = np.asarray(X.toarray() if hasattr(X, "toarray") else X, float)
        y = np.asarray(y, float)
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
        groups = np.arange(len(y)) if groups is None else np.asarray(groups)
        alphas = np.asarray(self.alphas, float)
        k = min(self.n_splits, len(np.unique(groups)))
        if k >= 2:
            sse = np.zeros(len(alphas))
            gkf = GroupKFold(n_splits=k, shuffle=True, random_state=self.random_state)
            for tr, te in gkf.split(X, groups=groups):
                c, b = self._solve(X[tr], y[tr], w[tr], alphas)
                sse += (w[te][:, None] * (y[te][:, None] - (X[te] @ c.T + b)) ** 2).sum(0)
            self.alpha_ = float(alphas[int(np.argmin(sse))])
        else:
            self.alpha_ = float(alphas[len(alphas) // 2])
        c, b = self._solve(X, y, w, [self.alpha_])
        self.coef_, self.intercept_ = c[0], float(b[0])
        return self

    def predict(self, X):
        X = np.asarray(X.toarray() if hasattr(X, "toarray") else X, float)
        return X @ self.coef_ + self.intercept_


class BlockStackRegressor(RegressorMixin, BaseEstimator):
    """Ridge on A plus one score per wide clip-feature block (``brain_*``, ``emb_*``), each with its own penalty.

    One shared ridge penalty lets 100-280 weak clip columns ride on the small penalty A's strong columns need:
    in ``tools/power_check.py`` 100 pure-noise columns cost about −0.09 within-stratum ρ. Here every block is
    first compressed to a single score by its own ``GroupedRidgeCV`` on y. The score a training row gets comes
    from inner content-grouped folds that never saw that row, so the final ridge can't over-trust it; at predict
    time the block models fit on all training rows score the new rows. BE − E is then exactly one extra column.
    """

    def __init__(self, cat=(), num=(), n_splits: int = 5, seed: int = 0):
        self.cat = cat
        self.num = num
        self.n_splits = n_splits
        self.seed = seed

    def _block_model(self):
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                         ("model", GroupedRidgeCV(random_state=self.seed))])

    def fit(self, X, y, sample_weight=None, groups=None):
        from sklearn.model_selection import GroupKFold

        X = X.reset_index(drop=True)
        y = np.asarray(y, float)
        if groups is None:
            groups = X["_content"].to_numpy() if "_content" in X else np.arange(len(y))
        groups = np.asarray(groups)
        w = None if sample_weight is None else np.asarray(sample_weight, float)
        self.blocks_ = {b: [c for c in self.num if c.startswith(pre)] for b, pre in BLOCK_PREFIXES}
        self.blocks_ = {b: c for b, c in self.blocks_.items() if c}
        in_block = {c for cols in self.blocks_.values() for c in cols}
        self.a_num_ = [c for c in self.num if c not in in_block]

        def kw(ix):
            out = {"model__groups": groups[ix]}
            if w is not None:
                out["model__sample_weight"] = w[ix]
            return out

        scores = {}
        self.block_models_ = {}
        k = min(self.n_splits, len(np.unique(groups)))
        every = np.arange(len(y))
        for b, cols in self.blocks_.items():
            oof = np.full(len(y), np.nan)
            if k >= 2:
                for tr, te in GroupKFold(n_splits=k, shuffle=True, random_state=self.seed).split(X, groups=groups):
                    oof[te] = self._block_model().fit(X.iloc[tr][cols], y[tr], **kw(tr)).predict(X.iloc[te][cols])
            self.block_models_[b] = self._block_model().fit(X[cols], y, **kw(every))
            scores[f"stack_{b}"] = np.where(np.isfinite(oof), oof, self.block_models_[b].predict(X[cols]))
        self.score_cols_ = list(scores)
        self.final_cols_ = (list(self.cat), self.a_num_ + self.score_cols_)
        self.final_ = make_model("ridge", *self.final_cols_, seed=self.seed)
        fkw = {"model__groups": groups}
        if w is not None:
            fkw["model__sample_weight"] = w
        Xf = pd.concat([X, pd.DataFrame(scores, index=X.index)], axis=1)
        self.final_.fit(Xf[input_columns(self.final_cols_)], y, **fkw)
        return self

    def predict(self, X):
        X = X.reset_index(drop=True)
        S = pd.DataFrame({f"stack_{b}": m.predict(X[self.blocks_[b]]) for b, m in self.block_models_.items()},
                         index=X.index)
        return self.final_.predict(pd.concat([X, S], axis=1)[input_columns(self.final_cols_)])


class AccountTargetEncoder(TransformerMixin, BaseEstimator):
    """Adds ``te_account_mean``: the posting account's mean target, smoothed toward its deal×platform mean.

    Without an account term in A, B could gain by fingerprinting accounts through brain features. ``transform``
    uses only the rows the encoder was fit on (a fold's training rows); unseen accounts get their deal×platform
    mean, unseen deal×platforms the platform mean, then the global mean. ``fit_transform`` (what a Pipeline calls
    on the training rows) cross-fits instead: each training row is encoded from the other inner folds (grouped by
    ``_content``), so a row's own outcome never enters its own feature and the model cannot over-trust it.
    """

    def __init__(self, smooth: float = ACCOUNT_TE_SMOOTH, n_splits: int = 5, random_state: int = 0):
        self.smooth = smooth
        self.n_splits = n_splits
        self.random_state = random_state

    @staticmethod
    def _keys(X) -> list[np.ndarray]:
        """Account, deal×platform, platform keys (most to least specific); an unknown account is never pooled."""
        acct = X["social_account_id"].astype("string").fillna("missing")
        dp = X["deal_id"].astype(str) + "|" + X["platform"].astype(str)
        return [acct.where(acct != "missing").to_numpy(), dp.to_numpy(), X["platform"].astype(str).to_numpy()]

    def _encode(self, sums, counts, codes, g) -> np.ndarray:
        """Smoothed account mean from per-key (sum, count); codes < 0 or empty keys fall to the next level."""
        prior = np.full(len(codes[0]), g)
        for lvl in (2, 1):  # platform, then deal×platform overrides where it has rows
            c = codes[lvl]
            n = np.where(c >= 0, counts[lvl][np.maximum(c, 0)], 0)
            prior = np.where(n > 0, sums[lvl][np.maximum(c, 0)] / np.maximum(n, 1), prior)
        c = codes[0]
        n = np.where(c >= 0, counts[0][np.maximum(c, 0)], 0.0)
        sa = np.where(c >= 0, sums[0][np.maximum(c, 0)], 0.0)
        return (sa + self.smooth * prior) / (n + self.smooth)

    @staticmethod
    def _sums(codes, y, mask, sizes):
        ms = [mask & (c >= 0) for c in codes]
        return ([np.bincount(c[m], y[m], k) for c, m, k in zip(codes, ms, sizes)],
                [np.bincount(c[m], minlength=k).astype(float) for c, m, k in zip(codes, ms, sizes)])

    def _out(self, X, enc):
        X = X.copy()
        X[ACCOUNT_TE] = enc
        return X

    def _fit(self, X, y) -> list[np.ndarray]:
        fac = [pd.factorize(k) for k in self._keys(X)]  # missing account -> code -1
        codes, self.uniques_ = [c for c, _ in fac], [u for _, u in fac]
        self.sums_, self.counts_ = self._sums(codes, y, np.ones(len(y), bool), [len(u) for u in self.uniques_])
        self.global_ = float(y.mean()) if len(y) else 0.0
        return codes

    def fit(self, X, y=None):
        self._fit(X, np.asarray(y, float))
        return self

    def transform(self, X):
        codes = [pd.Index(u).get_indexer(k) for u, k in zip(self.uniques_, self._keys(X))]
        return self._out(X, self._encode(self.sums_, self.counts_, codes, self.global_))

    def fit_transform(self, X, y=None, **_):
        y = np.asarray(y, float)
        codes = self._fit(X, y)
        sizes = [len(u) for u in self.uniques_]
        groups = pd.factorize(X["_content"])[0] if "_content" in X else np.arange(len(X))
        G = int(groups.max()) + 1 if len(groups) else 0
        k = min(self.n_splits, G)
        if k < 2:
            return self._out(X, np.full(len(X), self.global_))
        f = (np.random.default_rng(self.random_state).permutation(G) % k)[groups]
        enc = np.empty(len(X))
        for j in range(k):
            te = f == j
            sums, counts = self._sums(codes, y, ~te, sizes)
            enc[te] = self._encode(sums, counts, [c[te] for c in codes], float(y[~te].mean()))
        return self._out(X, enc)


def input_columns(cols: tuple[list[str], list[str]]) -> list[str]:
    """Frame columns an estimator from make_model(cat, num) reads (the encoder's keys when A has its account term)."""
    cat, num = cols
    extra = TE_INPUTS if ACCOUNT_TE in num else []
    return list(dict.fromkeys(cat + [c for c in num if c != ACCOUNT_TE] + extra))


def make_model(kind: str, cat: list[str], num: list[str], seed: int = 0):
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline, make_pipeline
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

    if kind == "stack":
        return BlockStackRegressor(cat=tuple(cat), num=tuple(num), seed=seed)
    te = [("te", AccountTargetEncoder(random_state=seed))] if ACCOUNT_TE in num else []
    if kind == "ridge":
        pre = ColumnTransformer([
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
        ])
        return Pipeline(te + [("pre", pre), ("model", GroupedRidgeCV(random_state=seed))])
    if kind == "hgb":
        pre = ColumnTransformer([
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan), cat),
            ("num", "passthrough", num),
        ])
        hgb = HistGradientBoostingRegressor(
            **HGB_PARAMS, early_stopping=False, random_state=seed,
            categorical_features=list(range(len(cat))) or None,
        )
        return Pipeline(te + [("pre", pre), ("model", hgb)])
    raise ValueError(kind)


def fit_predict(kind, cols, train: pd.DataFrame, y, w, test: pd.DataFrame, seed=0, fit_weighted=False):
    cat, num = cols
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est = make_model(kind, cat, num, seed)
        pre = "" if kind == "stack" else "model__"  # the stack routes groups/weights to its own sub-models
        kw = {f"{pre}sample_weight": w} if fit_weighted and w is not None else {}
        if kind in ("ridge", "stack") and "_content" in train:
            kw[f"{pre}groups"] = train["_content"].to_numpy()
        use = input_columns(cols)
        est.fit(train[use], y, **kw)  # the account encoding is refit here, on this fit's training rows only
        return est, est.predict(test[use])


def feature_sets(df: pd.DataFrame) -> dict[str, tuple[list[str], list[str]]]:
    def usable(c):
        s = df[c]
        return s.notna().any() and s.nunique(dropna=True) > 1

    cat = [c for c in ("platform", "deal_id") if df[c].nunique() > 1]
    base = [c for c in df.columns if c.startswith("base_") and usable(c)]
    if "social_account_id" in df and df["social_account_id"].nunique() > 1:
        base = base + [ACCOUNT_TE]  # computed inside each fit from its training outcomes, never from df
    brain = [c for c in df.columns if c.startswith("brain_") and usable(c)]
    emb = [c for c in df.columns if c.startswith("emb_") and usable(c)]
    sets = {"A": (cat, base), "B": (cat, base + brain)}
    if emb:  # the control arm: the brain mapping's inputs without the mapping
        sets.update({"E": (cat, base + emb), "BE": (cat, base + brain + emb)})
    return sets


def profile_columns(cols: list[str]) -> list[str]:
    """Compact, interpretable brain summary used for niche adjustments and profiles."""
    keep = ("_mean_0_3s", "_slope_half", "_raw_mean")
    out = [c for c in cols if c.startswith("brain_") and not c.startswith(("brain_pca_", "brain_moments_", "brain_xch_"))
           and c.endswith(keep)]
    out += [c for c in cols if c.startswith("brain_moments_") and c.endswith("_per_min")
            and "total" not in c]
    out += [c for c in cols if c in ("brain_xch_attention_drops_per_min", "brain_xch_synchrony",
                                     "brain_xch_broad_frac")]
    return out


# ── weighted metrics & bootstrap ──────────────────────────────────────────


def _wranks(v: np.ndarray, codes: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted mid-ranks within each code; ties share their block's mid-rank."""
    order = np.lexsort((v, codes))
    vs, cs, ws = v[order], codes[order], w[order]
    new_group = np.r_[True, cs[1:] != cs[:-1]]
    new_block = new_group | np.r_[True, vs[1:] != vs[:-1]]
    cw = np.cumsum(ws)
    bid = np.cumsum(new_block) - 1
    gid = np.cumsum(new_group) - 1
    bw = np.bincount(bid, ws)
    bend = cw[np.r_[np.flatnonzero(new_block)[1:] - 1, len(vs) - 1]]
    gstart = (cw - ws)[new_group]
    mid = bend[bid] - bw[bid] / 2.0 - gstart[gid]
    out = np.empty_like(mid)
    out[order] = mid
    return out


def strat_spearman(y, p, codes, w=None, min_n: int = MIN_STRATUM_N) -> float:
    """Weighted Spearman inside each stratum, averaged with stratum weight sums (strata with >= min_n rows)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    if len(y) < 3:
        return float("nan")
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    codes = np.asarray(codes)
    if codes.dtype.kind not in "iu":
        codes = pd.factorize(codes)[0]
    ry, rp = _wranks(y, codes, w), _wranks(p, codes, w)
    G = int(codes.max()) + 1
    W = np.bincount(codes, w, G)
    n = np.bincount(codes, minlength=G)
    Wd = np.where(W > 0, W, 1)
    my, mp = np.bincount(codes, w * ry, G) / Wd, np.bincount(codes, w * rp, G) / Wd
    dy, dp = ry - my[codes], rp - mp[codes]
    cov = np.bincount(codes, w * dy * dp, G)
    vy, vp = np.bincount(codes, w * dy * dy, G), np.bincount(codes, w * dp * dp, G)
    ok = (n >= min_n) & (vy > 1e-12) & (vp > 1e-12)
    if not ok.any():
        return float("nan")
    rho = cov[ok] / np.sqrt(vy[ok] * vp[ok])
    return float((rho * W[ok]).sum() / W[ok].sum())


def spearman(y, p, w=None) -> float:
    return strat_spearman(y, p, np.zeros(len(y), int), w, min_n=3)


def r2(y, p, w=None) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    if len(y) < 2:
        return float("nan")
    mu = np.average(y, weights=w)
    sst = float((w * (y - mu) ** 2).sum())
    return float(1.0 - (w * (y - p) ** 2).sum() / sst) if sst > 0 else float("nan")


def r2_centred(y, p, groups, w=None) -> float:
    """R² against each group's own mean (groups = platform): the share of platform-centred variance explained,
    the same basis as the platform-centred ICC ceilings. Within one platform it equals the ordinary R²."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    g = pd.factorize(np.asarray(groups))[0]
    if len(y) < 2:
        return float("nan")
    mu = np.bincount(g, w * y) / np.bincount(g, w)
    sst = float((w * (y - mu[g]) ** 2).sum())
    return float(1.0 - (w * (y - p) ** 2).sum() / sst) if sst > 0 else float("nan")


def deal_spread(per_deal: dict, pairs: list[str], metrics=("within_stratum_spearman", "r2")) -> dict:
    """Spread of per-deal point deltas across deals (how consistent a pooled gain is from deal to deal)."""
    out = {}
    for pr in pairs:
        for m in metrics:
            v = np.array([r["deltas"][pr][m]["point"] for r in per_deal.values()
                          if pr in r["deltas"] and r["deltas"][pr][m]["point"] is not None], float)
            if len(v):
                out.setdefault(pr, {})[m] = {"n_deals": int(len(v)), "median": float(np.median(v)),
                                             "min": float(v.min()), "max": float(v.max()),
                                             "sd": float(v.std(ddof=1)) if len(v) > 1 else None,
                                             "n_positive": int((v > 0).sum())}
    return out


def all_metrics(y, p, strata, w=None) -> dict:
    return {"within_stratum_spearman": strat_spearman(y, p, strata, w), "spearman": spearman(y, p, w),
            "r2": r2(y, p, w)}


def boot_p(point, arr) -> float:
    """Two-sided p for ``point`` = 0 from the bootstrap SE (normal approximation). Unlike a count of resamples
    on the far side of zero it is not floored at ~1/n_boot, so it can feed a Benjamini–Hochberg correction."""
    arr = np.asarray(arr, float)
    arr = arr[np.isfinite(arr)]
    if point is None or not math.isfinite(point) or len(arr) <= 10:
        return float("nan")
    se = float(arr.std(ddof=1))
    return float(2 * stats.norm.sf(abs(point) / se)) if se > 0 else float(point == 0)


def bh(p) -> np.ndarray:
    """Benjamini–Hochberg q-values; NaN p stays NaN and is not counted as a test."""
    p = np.asarray(p, float)
    q = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if ok.any():
        q[ok] = stats.false_discovery_control(p[ok], method="bh")
    return q


def _summ(pt, arr) -> dict:
    arr = np.asarray(arr, float)
    arr = arr[np.isfinite(arr)]
    enough = len(arr) > 10
    return {"point": None if pt is None or not math.isfinite(pt) else float(pt),
            "ci": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))] if enough else [None, None],
            "p_le_0": float((arr <= 0).mean()) if enough else None,
            "p_boot": boot_p(pt, arr) if enough else None}


def bootstrap(y: np.ndarray, preds: dict[str, np.ndarray], units: np.ndarray, strata: np.ndarray,
              w: np.ndarray | None, n_boot: int, seed: int, pairs: list[tuple[str, str]] | None = None) -> dict:
    """Cluster bootstrap (resample ``units``) of every metric per model, and paired deltas new−ref."""
    rng = np.random.default_rng(seed)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    ucode = pd.factorize(units)[0]
    scode = pd.factorize(strata)[0]
    U = int(ucode.max()) + 1 if len(ucode) else 0
    keys = list(preds)
    if pairs is None:
        pairs = [(f"{a}_{m}", f"{b}_{m}") for a, b in PAIRS for m in MODELS
                 if f"{a}_{m}" in preds and f"{b}_{m}" in preds]
    point = {k: all_metrics(y, preds[k], scode, w) for k in keys}
    boots = {k: {m: [] for m in point[k]} for k in keys}
    for _ in range(n_boot if U > 1 else 0):
        cnt = np.bincount(rng.integers(0, U, U), minlength=U)
        idx = np.repeat(np.arange(len(y)), cnt[ucode])
        for k in keys:
            for m, v in all_metrics(y[idx], preds[k][idx], scode[idx], w[idx]).items():
                boots[k][m].append(v)
    return {
        "n": int(len(y)), "n_units": U,
        "models": {k: {m: _summ(point[k][m], boots[k][m]) for m in point[k]} for k in keys},
        "deltas": {f"{a}-{b}": {m: _summ(point[a][m] - point[b][m], np.subtract(boots[a][m], boots[b][m]))
                                for m in point[a]} for a, b in pairs if a in point and b in point},
    }


def breakdown(y, preds, units, strata, w, by: np.ndarray, min_n: int, n_boot: int, seed: int, pairs=None) -> dict:
    out = {}
    for g in pd.unique(by):
        m = by == g
        if m.sum() >= min_n:
            out[str(g)] = bootstrap(y[m], {k: v[m] for k, v in preds.items()}, units[m], strata[m], w[m],
                                    n_boot, seed, pairs)
    return out


# ── ICC ceilings ──────────────────────────────────────────────────────────


def icc1(values: np.ndarray, groups: np.ndarray) -> tuple[float, int, int]:
    """One-way random-effects ICC(1) with unequal group sizes (groups with >= 2 rows)."""
    d = pd.DataFrame({"v": values, "g": groups}).dropna()
    d = d[d.groupby("g")["v"].transform("size") >= 2]
    G, N = d["g"].nunique(), len(d)
    if G < 3:
        return float("nan"), int(G), int(N)
    gm = d.groupby("g")["v"].transform("mean")
    n_i = d.groupby("g").size().to_numpy()
    ssb = float(((gm - d["v"].mean()) ** 2).sum())
    ssw = float(((d["v"] - gm) ** 2).sum())
    msb, msw = ssb / (G - 1), ssw / max(N - G, 1)
    k0 = (N - (n_i ** 2).sum() / N) / (G - 1)
    return float((msb - msw) / (msb + (k0 - 1) * msw)), int(G), int(N)


def icc_ceilings(posts: pd.DataFrame, ycol: str, seed: int = 0) -> dict:
    """Share of post-level variance explained by content identity: same-platform reposts and across platforms.

    The target is centred by platform first (platform is a model feature, so its main effect is not a
    content ceiling; without this, engagement-rate level differences between platforms swamp the
    across-platform ICC).
    """
    d = posts.dropna(subset=[ycol]).copy()
    d[ycol] = d[ycol] - d.groupby("platform")[ycol].transform("mean")
    d["_cp"] = d["video_id"] + "|" + d["platform"]
    # same platform, different accounts: one post per account within a content×platform
    same = d.drop_duplicates(["_cp", "social_account_id"])
    out = {"same_platform": dict(zip(("icc", "groups", "posts"), icc1(same[ycol].to_numpy(), same["_cp"].to_numpy())))}
    out["same_platform_by_platform"] = {
        str(p): dict(zip(("icc", "groups", "posts"), icc1(g[ycol].to_numpy(), g["_cp"].to_numpy())))
        for p, g in same.groupby("platform")}
    one = d.sample(frac=1.0, random_state=seed).drop_duplicates("_cp")  # one post per content×platform
    out["across_platforms"] = dict(zip(("icc", "groups", "posts"), icc1(one[ycol].to_numpy(), one["video_id"].to_numpy())))
    return out


# ── cross-validation ──────────────────────────────────────────────────────


def cross_validate(df, y, w, splits, fsets, models, seed, fit_weighted=False, perm=None):
    n = len(df)
    oof = {f"{fs}_{m}": np.full(n, np.nan) for fs in fsets for m in models}
    tested = np.zeros(n, bool)
    folds = []
    strata = df["_stratum"].to_numpy()
    for name, tr, te in splits:
        tested[te] = True
        row = {"fold": name, "n_train": int(len(tr)), "n_test": int(len(te))}
        for fs, cols in fsets.items():
            for m in models:
                est, p = fit_predict(m, cols, df.iloc[tr], y[tr], w[tr], df.iloc[te], seed, fit_weighted)
                oof[f"{fs}_{m}"][te] = p
                row[f"{fs}_{m}"] = all_metrics(y[te], p, strata[te], w[te])
                if perm is not None and fs == "B" and m == perm["model"]:
                    _perm_importance(est, df.iloc[te][input_columns(cols)], y[te], w[te], p, perm, seed)
        folds.append(row)
    return oof, tested, folds


def _perm_importance(est, X, y, w, p0, perm, seed):
    rng = np.random.default_rng(seed)
    base = {"spearman": spearman(y, p0, w), "r2": r2(y, p0, w)}
    groups = dict(perm["families"])
    if perm["per_feature"]:
        groups.update({c: [c] for c in perm["features"]})
    for name, cols in groups.items():
        for _ in range(perm["repeats"]):
            Xp = X.copy()
            order = rng.permutation(len(Xp))
            for c in cols:  # a family is permuted jointly
                Xp[c] = Xp[c].to_numpy()[order]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pp = est.predict(Xp)
            acc = perm["acc"].setdefault(name, {"spearman": [], "r2": [],
                                                "kind": "family" if name in perm["families"] else "feature"})
            acc["spearman"].append(base["spearman"] - spearman(y, pp, w))
            acc["r2"].append(base["r2"] - r2(y, pp, w))


def brain_families(cols: list[str]) -> dict[str, list[str]]:
    fam: dict[str, list[str]] = {}
    for c in cols:
        if c.startswith("brain_"):
            fam.setdefault("family:" + c.split("_")[1], []).append(c)
    fam["family:all_brain"] = [c for c in cols if c.startswith("brain_")]
    return fam


# ── niche models (partial pooling) ────────────────────────────────────────


@dataclass
class Adjustment:
    """Shrunk ridge on residuals: r ≈ intercept + z(x)·coef, z = standardized profile features."""
    cols: list[str]
    mean: np.ndarray
    scale: np.ndarray
    fill: np.ndarray
    intercept: float
    coef: np.ndarray
    alpha: float
    n: int

    def z(self, X: pd.DataFrame) -> np.ndarray:
        a = X[self.cols].to_numpy(float)
        a = np.where(np.isfinite(a), a, self.fill)
        return (a - self.mean) / self.scale

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.intercept + self.z(X) @ self.coef


def _ridge_w(Z, r, w, alpha):
    """Weighted ridge with unpenalized intercept on already-standardized Z."""
    W = w / w.sum()
    zm, rm = W @ Z, W @ r
    Zc, rc = Z - zm, r - rm
    A = (Zc * w[:, None]).T @ Zc + alpha * np.eye(Z.shape[1])
    coef = np.linalg.solve(A, (Zc * w[:, None]).T @ rc)
    return rm - zm @ coef, coef


def fit_adjustment(X: pd.DataFrame, r: np.ndarray, groups: np.ndarray, w: np.ndarray | None,
                   cols: list[str], alphas=ALPHAS_ADJ, alpha: float | None = None, seed: int = 0) -> Adjustment:
    """Partially pooled niche deviation from the general model: ridge on the general model's residuals.

    The intercept (the niche's mean offset) is unpenalised; the slopes are shrunk toward zero, i.e. toward the
    general model. The penalty is chosen by content-grouped inner CV with the one-standard-error rule (the
    largest penalty whose CV error is within one SE of the best; select_alpha), so a small or noisy niche falls
    back to the general model's slopes rather than fitting noise.
    """
    w = np.ones(len(r)) if w is None else np.asarray(w, float)
    a = X[cols].to_numpy(float)
    fill = np.nanmedian(np.where(np.isfinite(a), a, np.nan), axis=0)
    fill = np.where(np.isfinite(fill), fill, 0.0)
    a = np.where(np.isfinite(a), a, fill)
    mean, scale = a.mean(0), a.std(0)
    scale = np.where(scale > 1e-9, scale, 1.0)
    Z = (a - mean) / scale
    if alpha is None:
        alpha = select_alpha(Z, r, groups, w, alphas, seed)
    b0, b = _ridge_w(Z, r, w, alpha)
    return Adjustment(cols, mean, scale, fill, float(b0), b, alpha, len(r))


def select_alpha(Z: np.ndarray, r: np.ndarray, groups: np.ndarray, w: np.ndarray, alphas=ALPHAS_ADJ,
                 seed: int = 0) -> float:
    """Ridge penalty for ``_ridge_w(Z, r, w, ·)`` by content-grouped CV with the one-standard-error rule.

    A penalty only means something for the Z it was chosen on: callers must pass exactly the standardised
    matrix and residuals they then solve with.
    """
    alphas = np.sort(np.asarray(alphas, float))
    ug = np.unique(groups)
    k = min(5, len(ug))
    if k < 3:
        return float(alphas[-1])
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(ug, rng.permutation(len(ug)) % k))
    f = np.array([fold_of[g] for g in groups])
    mse = np.zeros((len(alphas), k))
    for j in range(k):
        tr, te = f != j, f == j
        for i, al in enumerate(alphas):
            b0, b = _ridge_w(Z[tr], r[tr], w[tr], al)
            mse[i, j] = float((w[te] * (r[te] - (b0 + Z[te] @ b)) ** 2).sum() / w[te].sum())
    mean_err, se = mse.mean(1), mse.std(1, ddof=1) / np.sqrt(k)
    best = int(np.argmin(mean_err))
    return float(alphas[np.flatnonzero(mean_err <= mean_err[best] + se[best]).max()])


@dataclass
class NicheModel:
    """General model fit without the niche + partially pooled niche adjustment."""
    niche: str
    general: object
    general_cols: tuple
    adjustment: Adjustment
    profile: dict = field(default_factory=dict)
    n_general: int = 0  # rows the general model was fit on (outside the niche and its contents' reposts)

    def predict_general(self, X: pd.DataFrame) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.general.predict(X[input_columns(self.general_cols)])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_general(X) + self.adjustment.predict(X)


def niche_mask(df: pd.DataFrame, niche: str) -> np.ndarray:
    """``deal:<id>``, ``account:<id>`` or a bare id matched against deal_id then social_account_id."""
    kind, _, val = niche.partition(":")
    if kind == "deal":
        return (df["deal_id"] == val).to_numpy()
    if kind == "account":  # the account's own posts; the selection's anchor account only as a fallback
        m = (df["social_account_id"] == val).to_numpy()
        if not m.any() and "anchor_account" in df:
            m = (df["anchor_account"].astype(str) == val).to_numpy()
        return m
    m = (df["deal_id"] == niche).to_numpy()
    return m if m.any() else (df["social_account_id"] == niche).to_numpy()


def fit_niche(df: pd.DataFrame, niche: str, target: str, *, base: str = "hgb", fsets=None, weights=None,
              seed: int = 0, fit_weighted: bool = False, with_profile: bool = True, n_boot: int = 200,
              alpha: float | None = None) -> NicheModel:
    """Fit the general B model on rows outside the niche, then the niche's shrunk residual adjustment.

    ``df`` is a build_dataset() frame (training rows only); ``target`` is a target key (column ``y_<key>``).
    """
    ycol = f"y_{target}" if f"y_{target}" in df else target
    d = df[np.isfinite(df[ycol].to_numpy(float))].reset_index(drop=True)
    m = niche_mask(d, niche)
    if m.sum() < 5:
        raise ValueError(f"niche {niche!r}: only {int(m.sum())} labelled posts")
    fsets = fsets or feature_sets(d)
    y = d[ycol].to_numpy(float)
    w = d["_w"].to_numpy(float) if weights is None else np.asarray(weights, float)
    # the general model must not see the niche's clips through reposts on other accounts/deals, or the
    # adjustment is fit on residuals the general model has partly memorised
    content = d["_content"].to_numpy()
    other = ~m & ~np.isin(content, content[m])
    general, p_in = fit_predict(base, fsets["B"], d[other], y[other], w[other], d[m], seed, fit_weighted)
    cols = profile_columns(fsets["B"][1])
    adj = fit_adjustment(d[m], y[m] - p_in, content[m], w[m], cols, alpha=alpha, seed=seed)
    nm = NicheModel(niche, general, fsets["B"], adj, n_general=int(other.sum()))
    if with_profile:  # the profile re-selects its own penalty on its own (globally standardised) scale
        nm.profile = niche_profile(d, m, y, w, cols, n_boot=n_boot, seed=seed)
        nm.profile.update({"niche": niche, "target": target, "n_posts": int(m.sum()),
                           "n_contents": int(d.loc[m, "video_id"].nunique()),
                           "adjustment_alpha": adj.alpha, "general_base": base})
    return nm


def _linear_global(d, y, w, cols):
    """Global linear association model: standardized profile features + platform/deal indicators."""
    from sklearn.linear_model import Ridge

    a = d[cols].to_numpy(float)
    fill = np.nanmedian(np.where(np.isfinite(a), a, np.nan), axis=0)
    fill = np.where(np.isfinite(fill), fill, 0.0)
    a = np.where(np.isfinite(a), a, fill)
    mu, sd = a.mean(0), a.std(0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    dummies = pd.get_dummies(d[["platform", "deal_id"]].astype(str), drop_first=True).to_numpy(float)
    X = np.hstack([(a - mu) / sd, dummies])
    return Ridge(alpha=1.0).fit(X, y, sample_weight=w).coef_[: len(cols)], X


def niche_profile(d: pd.DataFrame, m: np.ndarray, y: np.ndarray, w: np.ndarray, cols: list[str],
                  alpha: float | None = None, n_boot: int = 200, seed: int = 0) -> dict:
    """Per standardized profile feature: global slope, niche slope (= global + shrunk deviation), CIs.

    Slopes are partial associations (other profile features and platform/deal held fixed), in target
    units per 1 SD of the feature. They describe this data, not what an edit would do. The deviation penalty
    is selected (content-grouped CV, 1-SE rule) on the same globally standardised niche rows and residuals it
    is applied to, unless ``alpha`` is given. CIs hold that penalty fixed (conditional on it) and are not
    adjusted for the number of features; ``deviation_q`` is the Benjamini–Hochberg q over the features.
    """
    from sklearn.linear_model import Ridge

    rng = np.random.default_rng(seed)
    beta_g, X = _linear_global(d, y, w, cols)
    k = len(cols)
    # deviation of the niche from the global slopes: shrunk ridge on the niche's residuals
    gfit = Ridge(alpha=1.0).fit(X, y, sample_weight=w)
    res = y - gfit.predict(X)
    Zn = X[m][:, :k]
    content = d["_content"].to_numpy()
    if alpha is None:
        alpha = select_alpha(Zn, res[m], content[m], w[m], seed=seed)
    _, delta = _ridge_w(Zn, res[m], w[m], alpha)
    uc_all, uc_n = np.unique(content), np.unique(content[m])
    bg, bd = [], []
    idx_n = np.flatnonzero(m)
    cn = content[idx_n]
    for _ in range(n_boot):
        s = rng.choice(uc_all, len(uc_all))
        cnt = pd.Series(s).value_counts()
        rep = cnt.reindex(content).fillna(0).to_numpy(int)
        ii = np.repeat(np.arange(len(y)), rep)
        if len(ii) < k + 5:
            continue
        bg.append(Ridge(alpha=1.0).fit(X[ii], y[ii], sample_weight=w[ii]).coef_[:k])
        sn = rng.choice(uc_n, len(uc_n))
        cnt_n = pd.Series(sn).value_counts()
        jj = np.repeat(idx_n, cnt_n.reindex(cn).fillna(0).to_numpy(int))
        bd.append(_ridge_w(X[jj][:, :k], res[jj], w[jj], alpha)[1])
    bg, bd = np.array(bg), np.array(bd)
    rows = []
    for j, c in enumerate(cols):
        def ci(a):
            return [float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))] if len(a) > 10 else [None, None]

        rows.append({
            "feature": c,
            "global_slope": float(beta_g[j]), "global_ci": ci(bg[:, j]) if len(bg) else [None, None],
            "niche_slope": float(beta_g[j] + delta[j]),
            "niche_ci": ci(bg[:, j] + bd[:, j]) if len(bd) else [None, None],
            "deviation": float(delta[j]), "deviation_ci": ci(bd[:, j]) if len(bd) else [None, None],
            "deviation_p": boot_p(float(delta[j]), bd[:, j]) if len(bd) else float("nan"),
            "niche_spearman": spearman(y[m], d.loc[m, c].fillna(d[c].median()).to_numpy(float), w[m]),
            "overall_spearman": spearman(y, d[c].fillna(d[c].median()).to_numpy(float), w),
        })
    for r, q in zip(rows, bh([r["deviation_p"] for r in rows])):
        r["deviation_q"] = float(q)
    rows.sort(key=lambda r: -abs(r["niche_slope"]))
    return {"features": rows, "units": "target units per 1 SD of the feature (partial, standardized)",
            "n_boot": int(len(bd)), "alpha": float(alpha),
            "ci_note": "95% content-bootstrap CIs conditional on the chosen penalty alpha; unadjusted for the "
                       "number of features (deviation_q is the BH q over them; p from the bootstrap SE)"}


def evaluate_niches(df, y, w, splits, fsets, base, seed, min_niche_n, n_boot, fit_weighted=False,
                    general_oof=None, ks=LEARNING_CURVE_K, lc_repeats=3, min_account_n=40):
    """Per deal on content-grouped folds: general B (all deals) vs general without the deal (unseen niche)
    vs tuned C; learning curve of C vs k niche clips; account-level tuning on top of the deal model."""
    n = len(df)
    deal = df["deal_id"].to_numpy()
    acct = df["social_account_id"].to_numpy()
    content = df["_content"].to_numpy()
    cols = profile_columns(fsets["B"][1])
    lodo_g, tuned, tuned_acct, offset = (np.full(n, np.nan) for _ in range(4))
    lc = {k: np.full((lc_repeats, n), np.nan) for k in ks}
    deals = [d for d, c in df["deal_id"].value_counts().items() if c >= min_niche_n and d != "missing"]
    alphas = {}
    acct_max: dict = {}  # (deal, account) -> most training posts it had in any fold
    rng = np.random.default_rng(seed)
    deal_contents = {d: np.unique(content[deal == d]) for d in deals}
    for _, tr, te in splits:
        for d in deals:
            te_d = te[deal[te] == d]
            tr_d = tr[deal[tr] == d]
            tr_o = tr[(deal[tr] != d) & ~np.isin(content[tr], deal_contents[d])]  # no reposts of the deal's clips
            if len(te_d) == 0 or len(tr_d) < 5:
                continue
            X_all = df.iloc[np.r_[tr_d, te_d]]
            _, p = fit_predict(base, fsets["B"], df.iloc[tr_o], y[tr_o], w[tr_o], X_all, seed, fit_weighted)
            p_tr, p_te = p[: len(tr_d)], p[len(tr_d):]
            lodo_g[te_d] = p_te
            adj = fit_adjustment(df.iloc[tr_d], y[tr_d] - p_tr, content[tr_d], w[tr_d], cols, seed=seed)
            alphas.setdefault(d, []).append(adj.alpha)
            tuned[te_d] = p_te + adj.predict(df.iloc[te_d])
            # offset-only control: the niche's mean residual, no brain slopes (isolates what the slopes add)
            offset[te_d] = p_te + float(np.average(y[tr_d] - p_tr, weights=w[tr_d]))
            tuned_acct[te_d] = tuned[te_d]
            # accounts with enough training posts: a second shrunk layer on the deal model's residuals
            r_deal = y[tr_d] - (p_tr + adj.predict(df.iloc[tr_d]))
            for a in pd.unique(acct[te_d]):
                ma = acct[tr_d] == a
                acct_max[(d, a)] = max(acct_max.get((d, a), 0), int(ma.sum()))
                if ma.sum() < min_account_n:
                    continue
                adj_a = fit_adjustment(df.iloc[tr_d[ma]], r_deal[ma], content[tr_d[ma]], w[tr_d[ma]], cols,
                                       seed=seed)
                ta = te_d[acct[te_d] == a]
                tuned_acct[ta] = tuned[ta] + adj_a.predict(df.iloc[ta])
            # learning curve: tune on k of the niche's training clips
            uc = np.unique(content[tr_d])
            for k in ks:
                if k > len(uc):
                    continue
                for rep in range(lc_repeats):
                    pick = rng.choice(uc, k, replace=False)
                    sel = np.isin(content[tr_d], pick)
                    adj_k = fit_adjustment(df.iloc[tr_d[sel]], (y[tr_d] - p_tr)[sel], content[tr_d[sel]],
                                           w[tr_d[sel]], cols, seed=seed)
                    lc[k][rep, te_d] = p_te + adj_k.predict(df.iloc[te_d])
    strata = df["_stratum"].to_numpy()
    out = {"per_deal": {}, "learning_curve": {}, "base": base, "profile_columns": cols,
           "account_layer": {"min_account_n": int(min_account_n), "accounts_considered": len(acct_max),
                             "accounts_qualified": sum(v >= min_account_n for v in acct_max.values()),
                             "max_train_posts": max(acct_max.values(), default=0)}}
    for d in deals:
        md = (deal == d) & np.isfinite(tuned)
        if md.sum() < 3:
            continue
        preds = {"general_B": general_oof[md], "general_without_deal": lodo_g[md], "offset_only": offset[md],
                 "tuned_C": tuned[md], "tuned_C_accounts": tuned_acct[md]}
        pairs = [("tuned_C", "general_B"), ("tuned_C", "general_without_deal"), ("tuned_C", "offset_only"),
                 ("tuned_C_accounts", "tuned_C")]
        r = bootstrap(y[md], preds, content[md], strata[md], w[md], n_boot, seed, pairs)
        r["alpha_median"] = float(np.median(alphas.get(d, [np.nan])))
        r["n_contents"] = int(pd.Series(content[md]).nunique())
        out["per_deal"][str(d)] = r
        curve = {"0": {"within_stratum_spearman": strat_spearman(y[md], lodo_g[md], strata[md], w[md]),
                       "r2": r2(y[md], lodo_g[md], w[md])}}
        for k in ks:  # a point is reported only when every fold had k of the niche's training contents
            reps = [lc[k][rep][md] for rep in range(lc_repeats) if np.isfinite(lc[k][rep][md]).all()]
            if reps:
                curve[str(k)] = {"within_stratum_spearman": float(np.mean([strat_spearman(y[md], q, strata[md], w[md])
                                                                          for q in reps])),
                                 "r2": float(np.mean([r2(y[md], q, w[md]) for q in reps]))}
        curve["all"] = {"within_stratum_spearman": strat_spearman(y[md], tuned[md], strata[md], w[md]),
                        "r2": r2(y[md], tuned[md], w[md]), "contents_per_fold": int(round(
                            r["n_contents"] * (len(splits) - 1) / max(len(splits), 1)))}
        out["learning_curve"][str(d)] = curve
    # one C-vs-B verdict per deal: Benjamini–Hochberg over the deals (p from the bootstrap SE of Δ R²)
    per = out["per_deal"]
    pv = [per[d]["deltas"]["tuned_C-general_B"]["r2"]["p_boot"] for d in per]
    for d, q in zip(per, bh([np.nan if v is None else v for v in pv])):
        per[d]["q_bh_tuned_vs_general_r2"] = float(q)
    ok = np.isfinite(tuned) & np.isfinite(general_oof)
    if ok.sum() >= 3:
        out["pooled"] = bootstrap(y[ok], {"general_B": general_oof[ok], "general_without_deal": lodo_g[ok],
                                          "offset_only": offset[ok], "tuned_C": tuned[ok],
                                          "tuned_C_accounts": tuned_acct[ok]},
                                  content[ok], strata[ok], w[ok], n_boot, seed,
                                  [("tuned_C", "general_B"), ("tuned_C", "general_without_deal"),
                                   ("tuned_C", "offset_only"), ("tuned_C_accounts", "tuned_C")])
    return out


# ── per-deal top-quartile association table ───────────────────────────────


def cluster_rank_p(x, y, clusters, min_clusters: int = MIN_CLUSTERS) -> float:
    """Two-sided p for Spearman(x, y) with cluster-robust SE: OLS of rank(y) on rank(x), CR1 sandwich over
    ``clusters``, t with G−1 df. NaN with fewer than ``min_clusters`` clusters (too few to estimate the SE)."""
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    g = pd.factorize(np.asarray(clusters))[0]
    G, n, sxx = int(g.max()) + 1 if len(g) else 0, len(rx), float((rx ** 2).sum())
    if G < min_clusters or n < 3 or sxx <= 0:
        return float("nan")
    b = float((rx * ry).sum() / sxx)
    score = np.bincount(g, rx * (ry - b * rx), G)
    var = G / (G - 1) * (n - 1) / (n - 2) * float((score ** 2).sum()) / sxx ** 2
    return float(2 * stats.t.sf(abs(b) / math.sqrt(var), G - 1)) if var > 0 else float("nan")


def quartile_table(df: pd.DataFrame, ycol: str, feature_cols: list[str], min_n: int) -> pd.DataFrame:
    """Per deal, content level: Spearman(feature, top-quartile-of-target indicator), BH-FDR within deal.

    Contents of one account (6-12 per account in the study set) are not independent, so ``p_account`` clusters
    by the content's anchor account (posting account without a selection) and the q-values use it; ``p`` is
    the naive independent-contents value, kept for reference.
    """
    rows = []
    d0 = df.dropna(subset=[ycol]).copy()
    d0["_cluster"] = (d0["anchor_account"].astype("string").fillna(d0["social_account_id"].astype("string"))
                      if "anchor_account" in d0 else d0["social_account_id"])
    for d, g in d0.groupby("deal_id"):
        c = g.groupby("_content").agg({ycol: "mean", "_cluster": "first", **{f: "first" for f in feature_cols}})
        if len(c) < min_n:
            continue
        top = (c[ycol] >= c[ycol].quantile(0.75)).astype(float)
        res = []
        for f in feature_cols:
            x = c[f].astype(float)
            ok = x.notna()
            if ok.sum() < min_n or x[ok].nunique() < 2 or top[ok].nunique() < 2:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rho, pv = stats.spearmanr(x[ok], top[ok])
            if np.isfinite(rho):
                res.append({"deal_id": d, "feature": f, "n_contents": int(ok.sum()),
                            "n_accounts": int(c.loc[ok, "_cluster"].nunique()), "rho": float(rho), "p": float(pv),
                            "p_account": cluster_rank_p(x[ok].to_numpy(), top[ok].to_numpy(),
                                                        c.loc[ok, "_cluster"].to_numpy())})
        if res:
            for r, q in zip(res, bh([r["p_account"] for r in res])):
                r["q_bh_within_deal"] = float(q)
            rows.extend(res)
    t = pd.DataFrame(rows, columns=["deal_id", "feature", "n_contents", "n_accounts", "rho", "p", "p_account",
                                    "q_bh_within_deal"])
    if len(t):
        t["q_bh_global"] = bh(t["p_account"].to_numpy())
    return t


# ── report ────────────────────────────────────────────────────────────────


def _fmt(s: dict | None, digits=3) -> str:
    if not s or s.get("point") is None:
        return "n/a"
    lo, hi = s["ci"]
    ci = f" [{lo:.{digits}f}, {hi:.{digits}f}]" if lo is not None else ""
    return f"{s['point']:.{digits}f}{ci}"


def _verdict(delta: dict | None, a="B", b="A") -> str:
    if not delta or delta.get("point") is None or delta["ci"][0] is None:
        return "not estimable"
    lo, hi = delta["ci"]
    if lo > 0:
        return f"{a} better than {b} (CI above zero)"
    if hi < 0:
        return f"{a} worse than {b} (CI below zero)"
    return "no reliable difference (CI includes zero)"


def _verdict_q(delta: dict | None, q: float | None, a="tuned", b="general", level: float = 0.05) -> str:
    if not delta or delta.get("point") is None or q is None or not np.isfinite(q):
        return "not estimable"
    if q < level:
        return f"{a} {'better' if delta['point'] > 0 else 'worse'} than {b} (BH q={q:.3g})"
    return f"no reliable difference (BH q={q:.3g})"


def _ab_table(L, pooled, models, title_cols=True):
    L.append("| model | A within-stratum ρ | B within-stratum ρ | Δ within-stratum ρ | A pooled ρ | B pooled ρ | "
             "Δ pooled ρ | A R² | B R² | Δ R² | verdict (within-stratum ρ) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for m in models:
        a, b = pooled["models"].get(f"A_{m}"), pooled["models"].get(f"B_{m}")
        dl = pooled["deltas"].get(f"B_{m}-A_{m}", {})
        if a and b:
            L.append(f"| {m} | {_fmt(a['within_stratum_spearman'])} | {_fmt(b['within_stratum_spearman'])} | "
                     f"{_fmt(dl.get('within_stratum_spearman'))} | {_fmt(a['spearman'])} | {_fmt(b['spearman'])} | "
                     f"{_fmt(dl.get('spearman'))} | {_fmt(a['r2'])} | {_fmt(b['r2'])} | {_fmt(dl.get('r2'))} | "
                     f"{_verdict(dl.get('within_stratum_spearman'))} |")
    L.append("")


def _control_table(L, pooled, models):
    """The brain mapping against its own inputs (only when the E arm was fit)."""
    rows = [(m, a, b) for m in models for a, b in PAIRS[1:] if f"{a}_{m}-{b}_{m}" in pooled["deltas"]]
    if not rows:
        return
    L.append("Control arm: E = A + the extractor features TRIBE's brain mapping reads; BE = A + brain + those "
             "features. **BE − E** is what the brain mapping adds beyond its own inputs (the pre-registered primary); "
             "BE − A is what all content features add (the scale-up go/no-go).\n")
    L.append("| model | comparison | Δ within-stratum ρ | Δ pooled ρ | Δ R² | verdict (within-stratum ρ) |")
    L.append("|---|---|---|---|---|---|")
    for m, a, b in rows:
        dl = pooled["deltas"][f"{a}_{m}-{b}_{m}"]
        L.append(f"| {m} | {a} − {b} | {_fmt(dl.get('within_stratum_spearman'))} | {_fmt(dl.get('spearman'))} | "
                 f"{_fmt(dl.get('r2'))} | {_verdict(dl.get('within_stratum_spearman'), a, b)} |")
    L.append("")


def _ceiling(icc: dict | None, platform: str | None = None) -> str:
    if not icc:
        return "n/a"
    s = icc["same_platform_by_platform"].get(platform) if platform else icc["same_platform"]
    if not s or not np.isfinite(s["icc"]):
        return "n/a"
    flag = " — too few groups to trust" if s["groups"] < MIN_ICC_GROUPS else ""
    return f"{s['icc']:.2f} ({s['groups']} groups{flag})"


def _centred(rc: dict | None, models) -> str:
    if not rc:
        return "n/a"
    return ", ".join(f"{fs} {m} {rc[f'{fs}_{m}']:.3f}" for m in models for fs in ("A", "B") if f"{fs}_{m}" in rc)


def write_report(path: Path, res: dict) -> str:
    L: list[str] = []
    info = res["data"]
    models = res["models"]
    L.append("# Brain features vs baseline: out-of-sample model comparison\n")
    if info.get("synthetic_clips"):
        L.append(f"> **SYNTHETIC**: {info['synthetic_clips']} clips come from dry-run (stub) predictions or "
                 "synthetic fixtures; the numbers test the pipeline, not TRIBE.\n")
    L.append("Brain features are TRIBE v2 predictions for an average subject, not measured brain activity. "
             "Everything below is an association on observational data: a feature that predicts an outcome "
             "is *associated with* it; this analysis cannot show that changing the feature would change the "
             "outcome.\n")
    if not res["config"].get("score_lockbox"):
        L.append("## Final result: lockbox not scored\n")
        L.append("The lockbox was **not scored** in this run: no model was fit for it and no lockbox number is "
                 "reported (lockbox posts per target: "
                 + ", ".join(f"`{t}` {tr['n_lockbox']}" for t, tr in res["targets"].items())
                 + "). Everything below is cross-validation on the train split. Score it once, when the analysis "
                   "is final, with `--score-lockbox`.\n")
    else:
        L.append("## Final result: lockbox (held out once, uniform random within each deal, unweighted)\n")
        L.append("Within a deal the lockbox follows that deal's eligible clips; pooled over deals it mixes them in "
                 "proportion to the study set's deal quotas (≈ √size, floor/cap), not their natural shares. "
                 "Per-deal rows are in metrics.json.\n")
    for t, tr in res["targets"].items() if res["config"].get("score_lockbox") else ():
        lb = tr.get("lockbox")
        L.append(f"### `{t}` (n posts = {lb['pooled']['n'] if lb else 0})\n")
        if not lb:
            L.append("No lockbox rows.\n")
            continue
        icc = tr.get("icc")
        L.append(f"Content-identity ceiling (ICC of the platform-centred target, same-platform reposts): "
                 f"{_ceiling(icc)}; across platforms: {icc['across_platforms']['icc']:.2f}. Compare it with the "
                 f"platform-centred R² ({_centred(lb.get('r2_platform_centred'), models)}), not the pooled R² in "
                 "the table, which also counts platform differences. Clip features are not expected to exceed "
                 "the same-platform ICC.\n" if icc else "")
        _ab_table(L, lb["pooled"], models)
        _control_table(L, lb["pooled"], models)
        if lb.get("niche"):
            nd = lb["niche"]["deltas"]
            L.append(f"General vs niche-tuned on the lockbox: tuned C − general B "
                     f"Δ within-stratum ρ = {_fmt(nd.get('tuned_C-general_B', {}).get('within_stratum_spearman'))}, "
                     f"Δ R² = {_fmt(nd.get('tuned_C-general_B', {}).get('r2'))}.\n")
        L.append("Per platform (lockbox, " + models[-1] + "; within one platform R² is already platform-centred, "
                 "so it compares directly with that platform's ICC):\n")
        L.append("| platform | n | A R² | B R² | Δ R² | B ρ | ICC ceiling (same platform) |\n|---|---|---|---|---|---|---|")
        for p, pr in sorted(lb.get("per_platform", {}).items()):
            a, b = pr["models"].get(f"A_{models[-1]}"), pr["models"].get(f"B_{models[-1]}")
            dl = pr["deltas"].get(f"B_{models[-1]}-A_{models[-1]}", {})
            L.append(f"| {p} | {pr['n']} | {_fmt(a['r2'])} | {_fmt(b['r2'])} | {_fmt(dl.get('r2'))} | "
                     f"{_fmt(b['spearman'])} | {_ceiling(icc, p)} |")
        L.append("")

    L.append("## Data\n")
    L.append("| item | value |\n|---|---|")
    for k in ("n_outcomes", "excluded_by_quality_flags", "n_posts_with_outcomes", "n_posts_in_selection",
              "n_posts_with_features", "lockbox_posts_linked_to_train_dropped", "account_groups_train",
              "largest_account_group_share", "follower_known_share", "synthetic_clips",
              "clips_excluded_non_english"):
        if k in info:
            L.append(f"| {k} | {info[k]} |")
    for s in ("train", "lockbox"):
        if s in info:
            L.append(f"| {s} | {info[s]} |")
    L.append(f"| excluded by flag | {info.get('excluded_by_flag')} |")
    L.append(f"| target rows failing requirements | {info.get('target_rows_failing_requirements')} |")
    L.append(f"| target rows excluded by target flags (views zero, counts above views) | "
             f"{info.get('target_rows_excluded_by_flag')} |")
    L.append(f"| target rows with an unknown shrunk rate (90% interval ≥ rate) | {info.get('target_rows_unknown')} |")
    L.append("")
    for t, fs in res["features"].items():
        L.append(f"`{t}`: feature set A ({len(fs['A'])} columns): `{', '.join(fs['A'])}`. Feature set B = A + "
                 f"{len(fs['B']) - len(fs['A'])} brain columns"
                 + (f"; E = A + {len(fs['E']) - len(fs['A'])} extractor-embedding columns; BE = A + both"
                    if "E" in fs else "; no extractor embeddings in the features table, so the E control arm "
                    "was not fit and B − A cannot separate the brain mapping from its inputs")
                 + f". Niche adjustments and profiles use {len(res['profile_columns'].get(t, []))} compact brain "
                 "columns.\n")
    L.append(f"`{ACCOUNT_TE}` is the account term in A (and so in B and the niche models' general part): the "
             "posting account's mean target, smoothed toward its "
             "deal×platform mean, recomputed inside every fit (each CV fold, leave-one-deal-out split, niche model and "
             "the lockbox fit) from that fit's training rows only, and cross-fitted across content-grouped inner "
             "folds for the training rows themselves. B gains therefore cannot come from recognising accounts "
             "that A already knows.\n")

    for t, tr in res["targets"].items():
        L.append(f"## Target `{t}`: cross-validation on the train split (weighted by 1/incl_prob)\n")
        icc = tr.get("icc")
        if icc:
            ia = tr.get("icc_all_posts") or icc
            L.append("ICC ceilings from reposts (study-set contents, all their quality-filtered posts; "
                     f"platform-centred): same platform {_ceiling(icc)}; by platform "
                     + ", ".join(f"{p} {_ceiling(icc, p)}" for p in sorted(icc["same_platform_by_platform"]))
                     + f"; across platforms {icc['across_platforms']['icc']:.2f} "
                       f"({icc['across_platforms']['groups']} groups). Over every post in members ⨝ outcomes: "
                       f"same platform {_ceiling(ia)}, across platforms {ia['across_platforms']['icc']:.2f} "
                       f"({ia['across_platforms']['groups']} groups).\n")
        for sch, sres in tr["schemes"].items():
            L.append(f"### Scheme `{sch}` ({sres['n_splits']} splits, bootstrap unit: "
                     f"{sres.get('bootstrap_unit', sch)})\n")
            _ab_table(L, sres["pooled"], models)
            _control_table(L, sres["pooled"], models)
            if sres.get("r2_platform_centred"):
                L.append(f"Platform-centred R² (same basis as the ICC ceilings): "
                         f"{_centred(sres['r2_platform_centred'], models)}.\n")
            for pr_, ms in sres.get("deal_spread", {}).items():
                L.append(f"Deal-to-deal spread of {pr_} (per-deal points over {next(iter(ms.values()))['n_deals']} held-out "
                         "deals): " + "; ".join(
                             f"{lab} median {v['median']:+.3f}, range [{v['min']:+.3f}, {v['max']:+.3f}], "
                             f"{v['n_positive']}/{v['n_deals']} above zero"
                             for lab, v in (("within ρ", ms.get("within_stratum_spearman")), ("R²", ms["r2"]))
                             if v) + ".\n")
            pp = sres.get("per_platform", {})
            if pp:
                m = models[-1]
                L.append(f"Per platform ({m}): " + "; ".join(
                    f"{p}: n={v['n']}, B R² {_fmt(v['models'][f'B_{m}']['r2'], 2)}, "
                    f"ΔR² {_fmt(v['deltas'].get(f'B_{m}-A_{m}', {}).get('r2'), 2)}, ceiling {_ceiling(icc, p)}"
                    for p, v in sorted(pp.items())) + "\n")
            per = sres.get("per_deal", {})
            if per:
                m = models[-1]
                L.append(f"| deal | n | A ρ (within) | B ρ (within) | Δρ | A R² | B R² | ΔR² |\n|---|---|---|---|---|---|---|---|")
                for d, pr in sorted(per.items(), key=lambda kv: -kv[1]["n"]):
                    a, b = pr["models"][f"A_{m}"], pr["models"][f"B_{m}"]
                    dl = pr["deltas"].get(f"B_{m}-A_{m}", {})
                    L.append(f"| {d} | {pr['n']} | {_fmt(a['within_stratum_spearman'])} | "
                             f"{_fmt(b['within_stratum_spearman'])} | {_fmt(dl.get('within_stratum_spearman'))} | "
                             f"{_fmt(a['r2'])} | {_fmt(b['r2'])} | {_fmt(dl.get('r2'))} |")
                L.append("")
        ni = tr.get("niche")
        if ni:
            L.append(f"### Niche tuning: general vs tuned per deal (content folds, general base = {ni['base']})\n")
            L.append("`general B` is fit on all deals (deal as a feature); `without deal` is fit on the other "
                     "deals only (an unseen niche); `tuned C` adds the deal's partially pooled adjustment to "
                     "`without deal`; `C+accounts` adds a second shrunk layer for accounts with enough posts. "
                     "`offset-only` adds just the niche's mean residual: part of any C gain over B can be a level "
                     "offset, and only `C − offset-only` measures what the niche's brain-feature slopes add.\n")
            if ni.get("pooled"):
                pd_ = ni["pooled"]["deltas"]
                L.append(f"All deals pooled: C − general B Δ within-stratum ρ = "
                         f"{_fmt(pd_['tuned_C-general_B']['within_stratum_spearman'])}, Δ R² = "
                         f"{_fmt(pd_['tuned_C-general_B']['r2'])}; C − without deal Δ R² = "
                         f"{_fmt(pd_['tuned_C-general_without_deal']['r2'])}; C − offset-only Δ R² = "
                         f"{_fmt(pd_['tuned_C-offset_only']['r2'])}.\n")
            L.append("| deal | n posts | contents | general B R² | without deal R² | tuned C R² | Δ C−B R² | "
                     "Δ C−B within ρ | Δ C − offset-only R² | C+accounts − C R² | verdict (R²) |")
            L.append("|---|---|---|---|---|---|---|---|---|---|---|")
            for d, r in sorted(ni["per_deal"].items(), key=lambda kv: -kv[1]["n"]):
                mo, dl = r["models"], r["deltas"]
                L.append(f"| {d} | {r['n']} | {r['n_contents']} | {_fmt(mo['general_B']['r2'])} | "
                         f"{_fmt(mo['general_without_deal']['r2'])} | {_fmt(mo['tuned_C']['r2'])} | "
                         f"{_fmt(dl['tuned_C-general_B']['r2'])} | "
                         f"{_fmt(dl['tuned_C-general_B']['within_stratum_spearman'])} | "
                         f"{_fmt(dl['tuned_C-offset_only']['r2'])} | "
                         f"{_fmt(dl['tuned_C_accounts-tuned_C']['r2'])} | "
                         f"{_verdict_q(dl['tuned_C-general_B']['r2'], r.get('q_bh_tuned_vs_general_r2'))} |")
            L.append("")
            L.append("The verdict is Benjamini–Hochberg adjusted over the deals in this table (q < 0.05; p from the "
                     "bootstrap SE of Δ R²). The CIs are per-deal 95% intervals, not adjusted for the number of "
                     "deals.\n")
            al = ni.get("account_layer") or {}
            if al and not al.get("accounts_qualified"):
                L.append(f"Account-level layers were skipped: no account had ≥ {al['min_account_n']} training posts "
                         f"in any fold ({al['accounts_considered']} accounts considered, the most had "
                         f"{al['max_train_posts']}; the study set has 6-12 contents per account), so "
                         "`C+accounts` equals `C`.\n")
            elif al:
                L.append(f"Account-level layers: {al['accounts_qualified']} of {al['accounts_considered']} accounts "
                         f"had ≥ {al['min_account_n']} training posts in at least one fold.\n")
            L.append("Learning curve: held-out R² (within-stratum ρ) of the tuned model vs the number of the "
                     "niche's clips used for tuning (0 = general model without the deal; all = every training "
                     "content of the niche in the fold). A point is shown only when every fold had that many "
                     "contents. Indicates how many clips a new account/deal needs before tuning is worth it.\n")
            ks = ["0"] + [str(k) for k in LEARNING_CURVE_K] + ["all"]
            L.append("| deal | " + " | ".join(f"k={k}" for k in ks) + " |")
            L.append("|---|" + "---|" * len(ks))
            for d, c in ni["learning_curve"].items():
                L.append(f"| {d} | " + " | ".join(
                    f"{c[k]['r2']:.3f} ({c[k]['within_stratum_spearman']:.2f})" if k in c else "–" for k in ks) + " |")
            L.append("")
        prof = tr.get("niche_profiles")
        if prof:
            L.append("### What is associated with higher values here vs overall (per deal)\n")
            L.append("Partial slopes of the compact brain features (target units per 1 SD, other profile features "
                     "and platform/deal held fixed). `niche` = global slope + partially pooled deal deviation; "
                     "95% cluster-bootstrap CIs, conditional on the deviation penalty α chosen for the profile (not "
                     "re-selected per resample) and not adjusted for the number of features or deals "
                     "(`deviation_q` in metrics.json is the BH q over the features). Top 6 per deal by |niche "
                     "slope|; full tables in metrics.json. Associations in this data only.\n")
            for d, p in prof.items():
                L.append(f"**deal `{d}`** (n posts {p['n_posts']}, contents {p['n_contents']}, "
                         f"profile penalty α {p['alpha']:.3g})\n")
                L.append("| feature | global slope | niche slope | deviation | ρ niche | ρ overall |\n|---|---|---|---|---|---|")
                for r in p["features"][:6]:
                    L.append(f"| {r['feature']} | {_fmt({'point': r['global_slope'], 'ci': r['global_ci']})} | "
                             f"{_fmt({'point': r['niche_slope'], 'ci': r['niche_ci']})} | "
                             f"{_fmt({'point': r['deviation'], 'ci': r['deviation_ci']})} | "
                             f"{r['niche_spearman']:.2f} | {r['overall_spearman']:.2f} |")
                L.append("")
        imp = tr.get("permutation_importance")
        if imp:
            L.append(f"### Permutation importance (feature set B, {imp['model']}, content folds, weighted)\n")
            L.append("Mean drop in held-out R² and Spearman when the column(s) are shuffled; families jointly. "
                     "Correlated features share credit.\n")
            L.append("| family | ΔR² | Δρ |\n|---|---|---|")
            L.extend(f"| {r['name']} | {r['r2']:.4f} | {r['spearman']:.4f} |" for r in imp["families"])
            if imp["features"]:
                L.append("\n| top brain feature | ΔR² | Δρ |\n|---|---|---|")
                L.extend(f"| {r['name']} | {r['r2']:.4f} | {r['spearman']:.4f} |" for r in imp["features"][:20])
            L.append("")
        qt = tr.get("quartile_table")
        if qt and qt["top"]:
            L.append(f"### Features associated with top-quartile `{t}` within each deal (train split)\n")
            L.append("Content level (target averaged over posts); Spearman with a top-quartile indicator; "
                     "Benjamini–Hochberg q within deal and across all tests. Contents of one account are not "
                     f"independent (6-12 per account), so p-values are account-clustered (CR1 sandwich on the rank "
                     f"regression, t with accounts−1 df; none with fewer than {MIN_CLUSTERS} accounts); the naive "
                     "p is in the CSV. The train split over-samples reach tails, so ρ is not a population estimate. "
                     f"Top 6 per deal; full table in `quartile_table_{t}.csv`.\n")
            for d, rows in qt["top"].items():
                L.append(f"**deal `{d}`** (q<0.10 within deal: {qt['n_sig'].get(d, 0)})\n")
                L.append("| feature | n | accounts | ρ | q (deal) | q (global) |\n|---|---|---|---|---|---|")
                L.extend(f"| {r['feature']} | {r['n_contents']} | {r['n_accounts']} | {r['rho']:+.3f} | "
                         f"{r['q_bh_within_deal']:.3g} | {r['q_bh_global']:.3g} |" for r in rows)
                L.append("")
    L.append("## Method notes and caveats\n")
    L.extend(f"- {n}" for n in res["notes"])
    text = "\n".join(L) + "\n"
    path.write_text(text)
    return text


# ── main ──────────────────────────────────────────────────────────────────


NOTES = [
    "Grouping: a content component joins every post of the same clip (video_id) and any outcomes content_group "
    "link, so reposts are never split between train and test. The account scheme merges content components with "
    "every posting account and the selection's anchor account, so no posting account is on both sides of a fold. "
    "Lockbox posts linked to train content are dropped.",
    "Train-split metrics are weighted by 1/incl_prob (the selection over-samples reach tails within accounts). The "
    "lockbox is a uniform random 15% per deal, scored unweighted and only with --score-lockbox, with models fit on "
    "all train rows. Within a deal it follows that deal's eligible clips; pooled lockbox metrics mix deals by the "
    "study set's deal quotas, so they describe the study set's deal mix, not the natural one.",
    "Feature sets A and B (and the niche general models) include an account term (te_account_mean): the posting "
    "account's mean target smoothed toward "
    "its deal×platform mean, refit inside every fit from that fit's training rows (cross-fitted over content-"
    "grouped inner folds for the training rows). Without it, B could beat A by recognising accounts.",
    "Engagement target: the same label flags as the study set (select_study_set.BAD_FLAGS), plus views zero and "
    "counts above views; a shrunk rate whose 90% interval is at least as wide as the rate is unknown and excluded, "
    "as in the selection. The selection's per-deal percentile requirement is not applied.",
    "Headline metric: within-stratum Spearman, the weighted mean rank correlation inside each deal×platform, so "
    "the brain features must rank clips inside a context the baseline already knows. Pooled ρ and R² also shown.",
    "Bootstrap: 95% percentile intervals from resampling the split unit (content, account group, or deal for "
    "leave-one-deal-out, whose per-deal spread is shown too); deltas are paired on the same resamples. p_le_0 in "
    "metrics.json is the share of resampled deltas <= 0; p_boot is a two-sided normal-approximation p from the "
    "bootstrap SE, used for Benjamini–Hochberg corrections. Intervals are unadjusted for multiplicity unless a q "
    "is shown.",
    "Feature set A is a whitelist; views, percentiles, quadrants and other rates in the outcomes table are never "
    "used as features. Brain counts are per-minute rates and A includes log duration, so B cannot gain simply by "
    "encoding clip length.",
    "ICC ceilings: one-way ICC(1) of the platform-centred post-level target with content as the group (same "
    "platform: content×platform groups, one post per account; across platforms: one post per platform). Clip "
    "features are identical across reposts, so beyond platform/deal/timing effects they can explain at most about "
    "this share of post-level variance; the across-platform value bounds a platform-agnostic clip score. Compare "
    "it with the platform-centred R² (or a per-platform R²), not the pooled R², which also counts platform "
    "differences.",
    "Niche tuning (C): the general model is fit without the niche, then a ridge-shrunk residual adjustment on the "
    "compact brain profile is fit on the niche's own training clips; its penalty is chosen by content-grouped "
    "inner CV, so a niche with no reliable deviation falls back to the general model.",
    "Leave-one-deal-out and the 'without deal' model show performance on an unseen niche; the learning curve "
    "shows how many of the niche's clips the adjustment needs. Every general model under a niche adjustment "
    "excludes reposts of the niche's own contents.",
    "Per-deal C vs B verdicts are Benjamini–Hochberg adjusted over deals. Niche-profile penalties are selected on "
    "the profile's own globally standardised features; profile CIs are conditional on that penalty.",
    "PCA of the time-mean cortex map is label-free; it is fit on all clips unless build_features.py ran with "
    "--exclude-from-pca-fit (the lockbox is then only projected; see the features meta.json).",
    "Onset transient: brain_*_mean_all, peak/trough, raw_mean/raw_sd, xch_broad_frac/mean_abs_z and the PCA "
    "time-mean include the first 2 s, where predicted responses carry a stimulus-onset transient (the within-clip "
    "z scale excludes it, the values do not; xch_synchrony excludes it). mean_0_3s/0_5s and early_minus_rest "
    "cover the onset by design.",
    "Follower counts are sparse; when present they may post-date the post.",
    "All results are associations. A positive Δ means brain features carry predictive information beyond A on "
    "held-out content; it does not show that editing a clip toward a feature value would change performance.",
]


def resolve_niche_base(niche_base: str | None, models: list[str]) -> str:
    """The niche's general model must be the B model it is compared with: default = the last chosen model
    (hgb when present); an explicit base outside ``models`` would compare C and B across model kinds."""
    if niche_base is None:
        return models[-1]
    if niche_base not in models:
        raise ValueError(f"niche base {niche_base!r} is not among the fitted models {models}; "
                         "pass it in --models or choose --niche-base from them")
    return niche_base


def run(features, members, outcomes, *, targets=DEFAULT_TARGETS, out_dir: Path, selection=None,
        schemes=SCHEMES, models=MODELS, n_splits=5, n_boot=1000, seed=0, min_deal_n=30, perm_repeats=3,
        perm_model="hgb", perm_per_feature=True, exclude_cols=None, fit_weighted=False, niche_base=None,
        niche=True, min_account_n=40, score_lockbox=False, threads=8, exclude_non_english=True, log=print) -> dict:
    from threadpoolctl import threadpool_limits

    t0 = time.perf_counter()
    models = [m for m in MODELS if m in models]
    niche_base = resolve_niche_base(niche_base, models)
    out_dir.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(threads):
        res = _run(features, members, outcomes, targets, out_dir, selection, schemes, models,
                   n_splits, n_boot, seed, min_deal_n, perm_repeats, perm_model, perm_per_feature, exclude_cols,
                   fit_weighted, niche_base, niche, min_account_n, score_lockbox, exclude_non_english, log)
    res["runtime_s"] = round(time.perf_counter() - t0, 1)
    (out_dir / "metrics.json").write_text(json.dumps(res, indent=2, default=_json_default) + "\n")
    text = write_report(out_dir / "report.md", res)
    low = f" {text.lower()} "
    bad = [w for w in FORBIDDEN_REPORT_WORDS if any(f" {w}{e}" in low for e in (" ", ".", ",", ")"))]
    if bad:
        log(f"warning: report contains forbidden wording {bad}")
    log(f"done in {res['runtime_s']}s -> {out_dir}/report.md")
    return res


def _run(features, members, outcomes, targets, out_dir, selection, schemes, models, n_splits, n_boot, seed,
         min_deal_n, perm_repeats, perm_model, perm_per_feature, exclude_cols, fit_weighted, niche_base, niche,
         min_account_n, score_lockbox, exclude_non_english, log):
    df_all, info, posts = build_dataset(features, members, outcomes, targets, selection, exclude_cols,
                                        exclude_non_english=exclude_non_english)
    res: dict = {"data": info, "models": models, "targets": {}, "features": {}, "profile_columns": {},
                 "config": {"n_splits": n_splits, "n_boot": n_boot, "seed": seed, "min_deal_n": min_deal_n,
                            "schemes": list(schemes), "perm_repeats": perm_repeats, "perm_model": perm_model,
                            "fit_weighted": fit_weighted, "niche_base": niche_base, "min_account_n": min_account_n,
                            "score_lockbox": score_lockbox},
                 "notes": NOTES}
    sub_boot = min(n_boot, max(200, n_boot // 5))
    oof_frames = []
    for t in info["targets"]:
        ycol = f"y_{t}"
        dfl = df_all[np.isfinite(df_all[ycol].to_numpy(float))]
        df = dfl[dfl["split"] == "train"].reset_index(drop=True)
        lb = dfl[dfl["split"] == "lockbox"].reset_index(drop=True)
        y, w = df[ycol].to_numpy(float), df["_w"].to_numpy(float)
        fsets = feature_sets(df)
        res["features"][t] = {k: v[0] + v[1] for k, v in fsets.items()}  # usable columns differ by target rows
        res["profile_columns"][t] = profile_columns(fsets["B"][1])
        tr: dict = {"column": info["targets"][t]["column"], "transform": info["targets"][t]["transform"],
                    "n_train": int(len(df)), "n_lockbox": int(len(lb)), "schemes": {}}
        res["targets"][t] = tr
        tr["icc_all_posts"] = icc_ceilings(posts, ycol, seed)
        tr["icc"] = (icc_ceilings(posts[posts["video_id"].isin(df_all["video_id"])], ycol, seed)
                     if selection is not None else tr["icc_all_posts"])
        log(f"[{t}] train n={len(df)} contents={df['video_id'].nunique()} lockbox n={len(lb)} "
            f"A={sum(map(len, fsets['A']))} B={sum(map(len, fsets['B']))} cols")
        strata = df["_stratum"].to_numpy()
        content_oof = None
        for sch in schemes:
            splits = make_splits(df, sch, n_splits, seed, min_deal_n)
            if not splits:
                log(f"  {sch}: not enough groups/deals, skipped")
                continue
            perm = None
            if sch == "content" and perm_repeats > 0 and perm_model in models:
                brain_cols = [c for c in fsets["B"][1] if c.startswith("brain_")]
                perm = {"model": perm_model, "repeats": perm_repeats, "families": brain_families(brain_cols),
                        "features": brain_cols, "per_feature": perm_per_feature, "acc": {}}
            ts = time.perf_counter()
            oof, tested, folds = cross_validate(df, y, w, splits, fsets, models, seed, fit_weighted, perm)
            units = df["_account" if sch == "account" else "_content"].to_numpy()
            # leave-one-deal-out: deals are the independent units of the pooled result (resampling contents
            # would treat one deal's shared held-out model error as many independent draws)
            pooled_units = df["deal_id"].to_numpy() if sch == "lodo" else units
            T = tested
            sres = {"folds": folds, "n_splits": len(splits), "bootstrap_unit": "deal" if sch == "lodo" else sch,
                    "pooled": bootstrap(y[T], {k: v[T] for k, v in oof.items()}, pooled_units[T], strata[T], w[T],
                                        n_boot, seed),
                    "r2_platform_centred": {k: r2_centred(y[T], v[T], df["platform"].to_numpy()[T], w[T])
                                            for k, v in oof.items()}}
            sres["per_platform"] = breakdown(y[T], {k: v[T] for k, v in oof.items()}, units[T], strata[T], w[T],
                                             df["platform"].to_numpy()[T], 10, sub_boot, seed)
            sres["per_deal"] = breakdown(y[T], {k: v[T] for k, v in oof.items()}, units[T], strata[T], w[T],
                                         df["deal_id"].to_numpy()[T], min_deal_n, sub_boot, seed)
            if sch == "lodo":
                sres["deal_spread"] = deal_spread(sres["per_deal"], [f"B_{m}-A_{m}" for m in models])
            tr["schemes"][sch] = sres
            f = pd.DataFrame({"target": t, "scheme": sch, "vp_id": df["vp_id"].to_numpy()[T],
                              "video_id": df["video_id"].to_numpy()[T], "y": y[T], "w": w[T]})
            for k, v in oof.items():
                f[f"pred_{k}"] = v[T]
            oof_frames.append(f)
            log(f"  {sch}: {len(splits)} splits, {time.perf_counter() - ts:.1f}s")
            if sch == "content":
                content_oof = (oof, splits)
            if perm is not None and perm["acc"]:
                recs = [{"name": k, "kind": v["kind"], "r2": float(np.mean(v["r2"])),
                         "spearman": float(np.mean(v["spearman"]))} for k, v in perm["acc"].items()]
                tr["permutation_importance"] = {
                    "model": perm_model, "repeats": perm_repeats,
                    "families": sorted([r for r in recs if r["kind"] == "family"], key=lambda r: -r["r2"]),
                    "features": sorted([r for r in recs if r["kind"] == "feature"], key=lambda r: -r["r2"])}
        best = niche_base  # general B shares the niche model's base (resolve_niche_base)
        if niche and content_oof is not None:
            ts = time.perf_counter()
            oof, splits = content_oof
            tr["niche"] = evaluate_niches(df, y, w, splits, fsets, niche_base, seed, min_deal_n, sub_boot,
                                          fit_weighted, general_oof=oof[f"B_{best}"], min_account_n=min_account_n)
            log(f"  niche: {len(tr['niche']['per_deal'])} deals, {time.perf_counter() - ts:.1f}s")
            profiles = {}
            for d, c in df["deal_id"].value_counts().items():
                if c < min_deal_n or d == "missing":
                    continue
                cols = profile_columns(fsets["B"][1])
                m = (df["deal_id"] == d).to_numpy()
                # the penalty is re-selected inside niche_profile on its own (globally standardised) Z; the C
                # adjustment's penalty was chosen on niche-standardised features and is not comparable
                prof = niche_profile(df, m, y, w, cols, n_boot=min(200, sub_boot), seed=seed)
                prof.update({"n_posts": int(m.sum()), "n_contents": int(df.loc[m, "video_id"].nunique()),
                             "adjustment_alpha": tr["niche"]["per_deal"].get(str(d), {}).get("alpha_median")})
                profiles[str(d)] = prof
            tr["niche_profiles"] = profiles
        # final: lockbox, fit once on all train rows, scored unweighted -- only when asked (--score-lockbox), so
        # development runs cannot peek at it
        if score_lockbox and len(lb) >= 3:
            ts = time.perf_counter()
            preds = {}
            for fs, cols in fsets.items():
                for m in models:
                    preds[f"{fs}_{m}"] = fit_predict(m, cols, df, y, w, lb, seed, fit_weighted)[1]
            ylb, ulb, slb = lb[ycol].to_numpy(float), lb["_content"].to_numpy(), lb["_stratum"].to_numpy()
            lbres = {"pooled": bootstrap(ylb, preds, ulb, slb, None, n_boot, seed),
                     "r2_platform_centred": {k: r2_centred(ylb, v, lb["platform"].to_numpy()) for k, v in preds.items()},
                     "per_platform": breakdown(ylb, preds, ulb, slb, np.ones(len(lb)), lb["platform"].to_numpy(),
                                               5, sub_boot, seed),
                     "per_deal": breakdown(ylb, preds, ulb, slb, np.ones(len(lb)), lb["deal_id"].to_numpy(),
                                           10, sub_boot, seed)}
            if niche:
                tuned = preds[f"B_{best}"].copy()
                have = np.zeros(len(lb), bool)
                for d in pd.unique(lb["deal_id"]):
                    md_tr = (df["deal_id"] == d).to_numpy()
                    md_lb = (lb["deal_id"] == d).to_numpy()
                    if md_tr.sum() < min_deal_n:
                        continue
                    o_tr = ~md_tr & ~df["_content"].isin(df.loc[md_tr, "_content"]).to_numpy()
                    _, p = fit_predict(niche_base, fsets["B"], df[o_tr], y[o_tr], w[o_tr],
                                       pd.concat([df[md_tr], lb[md_lb]]), seed, fit_weighted)
                    adj = fit_adjustment(df[md_tr], y[md_tr] - p[: md_tr.sum()], df.loc[md_tr, "_content"].to_numpy(),
                                         w[md_tr], profile_columns(fsets["B"][1]), seed=seed)
                    tuned[md_lb] = p[md_tr.sum():] + adj.predict(lb[md_lb])
                    have[md_lb] = True
                if have.sum() >= 3:
                    lbres["niche"] = bootstrap(ylb[have], {"general_B": preds[f"B_{best}"][have], "tuned_C": tuned[have]},
                                               ulb[have], slb[have], None, n_boot, seed, [("tuned_C", "general_B")])
                preds["C_tuned"] = tuned
            tr["lockbox"] = lbres
            f = pd.DataFrame({"target": t, "scheme": "lockbox", "vp_id": lb["vp_id"].to_numpy(),
                              "video_id": lb["video_id"].to_numpy(), "y": ylb, "w": 1.0})
            for k, v in preds.items():
                f[f"pred_{k}"] = v
            oof_frames.append(f)
            log(f"  lockbox: {len(lb)} posts, {time.perf_counter() - ts:.1f}s")
        qt = quartile_table(df, ycol, [c for c in fsets["B"][1] if c in df], min_n=max(20, min_deal_n // 2))
        qt.to_csv(out_dir / f"quartile_table_{t}.csv", index=False)
        tr["quartile_table"] = {
            "n_deals": int(qt["deal_id"].nunique()) if len(qt) else 0,
            "top": {str(d): g.reindex(g["rho"].abs().sort_values(ascending=False).index).head(6).to_dict("records")
                    for d, g in qt.groupby("deal_id")},
            "n_sig": {str(d): int((g["q_bh_within_deal"] < 0.10).sum()) for d, g in qt.groupby("deal_id")}}
    if oof_frames:
        pd.concat(oof_frames).to_csv(out_dir / "oof_predictions.csv", index=False)
    return res


# ── served model artefact (--save-model) ──────────────────────────────────

ARTEFACT_VERSION = "perf_artefact_v1"
DEFAULT_LOCKBOX_EXT = Path(__file__).resolve().parents[1] / "results/study/lockbox_ext.csv"


def _sha256(path) -> str | None:
    import hashlib

    if path is None or not Path(path).exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git() -> dict:
    import subprocess

    root = Path(__file__).resolve().parents[1]
    try:
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True,
                                    text=True, check=True).stdout.strip())
        return {"commit": head, "dirty": dirty}
    except Exception:  # noqa: BLE001
        return {"commit": None, "dirty": None}


def _pipeline_terms(pipe, X: pd.DataFrame) -> tuple[np.ndarray, list[str], np.ndarray]:
    """(transformed design, output names, ridge coefficients) of a make_model("ridge") pipeline."""
    T = np.asarray(pipe[:-1].transform(X), float)
    return T, list(pipe.named_steps["pre"].get_feature_names_out()), np.asarray(pipe[-1].coef_, float)


def linear_terms(est, X: pd.DataFrame) -> pd.DataFrame:
    """Per-row additive terms of a fitted ``stack`` or ``ridge`` model: prediction = row sum + one constant.

    Names: ``num__<col>`` / ``cat__<col>_<level>`` for the final ridge's own columns; for ``stack`` each block
    score is expanded into its block ridge's columns, ``<block>:<col>`` (brain:…, emb:…), which is exact because
    the block model is linear in its standardised inputs. Differences between rows (or to a reference average)
    are therefore exact decompositions of prediction differences. Model attributions, not causes."""
    X = X.reset_index(drop=True)
    if isinstance(est, BlockStackRegressor):
        scores, expanded = {}, {}
        for b, m in est.block_models_.items():
            cols = est.blocks_[b]
            scores[f"stack_{b}"] = m.predict(X[cols])
            names = list(m[0].get_feature_names_out())  # the imputer drops all-NaN training columns
            expanded[b] = (names, np.asarray(m[:-1].transform(X[cols]), float) * m[-1].coef_)
        Xf = pd.concat([X, pd.DataFrame(scores, index=X.index)], axis=1)
        T, names, coef = _pipeline_terms(est.final_, Xf[input_columns(est.final_cols_)])
        scale = est.final_.named_steps["pre"].named_transformers_["num"][-1].scale_
        num = [n for n in names if n.startswith("num__")]
        out = {}
        for j, n in enumerate(names):
            if n.startswith("num__stack_"):
                b = n[len("num__stack_"):]
                f = coef[j] / scale[num.index(n)]
                bn, bt = expanded[b]
                for i, c in enumerate(bn):
                    out[f"{b}:{c}"] = f * bt[:, i]
            else:
                out[n] = T[:, j] * coef[j]
        return pd.DataFrame(out)
    if hasattr(est, "named_steps") and isinstance(est[-1], GroupedRidgeCV):
        T, names, coef = _pipeline_terms(est, X)
        return pd.DataFrame(T * coef, columns=names)
    raise ValueError(f"no linear decomposition for {type(est).__name__}")


def imputation_medians(est) -> dict:
    """The medians the fitted pipeline actually imputes with (fit on its training rows)."""
    def med(imp):
        return {str(c): float(v) for c, v in zip(imp.get_feature_names_out(), imp.statistics_)
                if np.isfinite(v)}

    if isinstance(est, BlockStackRegressor):
        return {"blocks": {b: med(m[0]) for b, m in est.block_models_.items()},
                "final": med(est.final_.named_steps["pre"].named_transformers_["num"][0])}
    return {"final": med(est.named_steps["pre"].named_transformers_["num"][0])}


def _wsum(x) -> dict | None:
    if not x:
        return None
    return {"point": x.get("point"), "ci": x.get("ci")}


def served_metrics(res: dict, target: str, feature_set: str, model: str) -> dict:
    """The stage-1 (and, when scored, lockbox) numbers tools/predict.py derives model_status/brain_claim from."""
    tr = res["targets"][target]
    sch = tr.get("schemes", {})

    def delta(scheme, a, b):
        return _wsum(sch.get(scheme, {}).get("pooled", {}).get("deltas", {}).get(f"{a}-{b}", {})
                     .get("within_stratum_spearman"))

    lb = tr.get("lockbox", {}).get("pooled") if res["config"].get("score_lockbox") else None
    served = f"{feature_set}_{model}"
    return {
        "metric": "within_stratum_spearman (weighted 1/incl_prob in CV; lockbox unweighted)",
        "n_boot": res["config"]["n_boot"], "schemes": list(sch), "score_lockbox": bool(lb),
        "go": {"contrast": "BE_stack-A_stack", "content": delta("content", "BE_stack", "A_stack"),
               "account": delta("account", "BE_stack", "A_stack")},
        "served_go": {"contrast": f"{served}-A_{model}", "content": delta("content", served, f"A_{model}"),
                      "account": delta("account", served, f"A_{model}")},
        "served_cv": _wsum(sch.get("content", {}).get("pooled", {}).get("models", {}).get(served, {})
                           .get("within_stratum_spearman")),
        "brain": {"contrast": "BE_stack-E_stack", "content": delta("content", "BE_stack", "E_stack"),
                  "lockbox": _wsum(lb["deltas"].get("BE_stack-E_stack", {}).get("within_stratum_spearman"))
                  if lb else None},
        "served_lockbox": _wsum(lb["models"].get(served, {}).get("within_stratum_spearman")) if lb else None,
    }


def read_lockbox_ext(path) -> set[str]:
    if path is None or not Path(path).exists():
        return set()
    t = read_table(Path(path))
    if "video_id" not in t:
        raise SystemExit(f"{path}: needs a video_id column")
    return set(t["video_id"].astype(str))


def _stratum_pct(y: np.ndarray, strata: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted mid-rank of y within its stratum / the stratum's weight (0..1), the metric's rank basis."""
    codes = pd.factorize(strata)[0]
    W = np.bincount(codes, w)
    return _wranks(y, codes, w) / W[codes]


def save_model(features, members, outcomes, *, res: dict, eval_dir: Path, model_dir: Path, selection=None,
               feature_set: str = "BE", model: str = "stack", lockbox_ext: set[str] | None = None,
               exclude_cols=None, exclude_non_english: bool = True, paths: dict | None = None,
               log=print) -> dict:
    """Refit the served model on all non-lockbox train rows and write it with its reference tables.

    Runs after ``run`` and reads only its outputs (``res`` and ``eval_dir/oof_predictions.csv``); the evaluation
    is untouched. The fit is ``fit_predict`` with the evaluation's feature sets, seed and ``fit_weighted``, i.e.
    exactly the model the lockbox would be scored with. Refuses (SystemExit) when a stage-2 lockbox-extension
    content is a train row: the evaluation itself would then have trained on the lockbox."""
    import joblib
    import sklearn

    cfg = res["config"]
    if model not in ("stack", "ridge"):
        raise SystemExit(f"--save-model-kind {model!r}: only stack/ridge (linear, so drivers are exact)")
    if model not in res["models"] or "content" not in cfg["schemes"]:
        raise SystemExit(f"--save-model needs the content scheme and {model!r} in --models (its out-of-fold "
                         "predictions are the reference tables and its metrics set model_status)")
    df_all, _, _ = build_dataset(features, members, outcomes, list(res["targets"]), selection, exclude_cols,
                                 exclude_non_english=exclude_non_english)
    lockbox_ext = set(lockbox_ext or ())
    train_ids = set(df_all.loc[df_all["split"] == "train", "video_id"].astype(str))
    clash = sorted(train_ids & lockbox_ext)
    if clash:
        raise SystemExit(f"{len(clash)} lockbox-extension contents are train rows in this selection (e.g. "
                         f"{clash[:3]}); the evaluation used them. Mark them lockbox before fitting.")
    lock_ids = set(df_all.loc[df_all["split"] == "lockbox", "video_id"].astype(str)) | lockbox_ext
    if selection is not None and {"video_id", "split"} <= set(selection.columns):
        lock_ids |= set(selection.loc[selection["split"].astype(str) == "lockbox", "video_id"].astype(str))
    oof_all = pd.read_csv(eval_dir / "oof_predictions.csv", dtype={"vp_id": str, "video_id": str})
    model_dir.mkdir(parents=True, exist_ok=True)
    pred_col = f"pred_{feature_set}_{model}"
    git = _git()
    stamp = time.strftime("%Y%m%d")
    manifest = {
        "artefact_version": ARTEFACT_VERSION,
        "model_version": f"perf-{model}-{feature_set}_v1+{stamp}.{(git['commit'] or 'nogit')[:7]}",
        "feature_set": feature_set, "model": model, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": git, "versions": {"sklearn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__,
                                 "joblib": joblib.__version__, "python": sys.version.split()[0]},
        "config": {k: cfg.get(k) for k in ("seed", "n_splits", "fit_weighted", "min_account_n", "schemes",
                                            "n_boot", "score_lockbox")},
        "exclude_non_english": exclude_non_english, "inputs": {}, "lockbox_ids": "lockbox_ids.json",
        "n_lockbox_ids": len(lock_ids), "targets": {},
        "reference_rules": {"percentile": "weighted (1/incl_prob) mid-rank among the deal×platform reference "
                            "contents' content-CV out-of-fold predictions; null below 30 contents",
                            "min_reference_n": 30, "min_account_posts": cfg.get("min_account_n", 40)},
    }
    for key, path in (paths or {}).items():
        manifest["inputs"][key] = {"path": str(path) if path else None, "sha256": _sha256(path)}
    feat = (paths or {}).get("features")
    fz = Path(f"{Path(feat).with_suffix('')}.featurizer.json") if feat else None
    manifest["inputs"]["featurizer"] = {"path": str(fz) if fz and fz.exists() else None,
                                        "sha256": _sha256(fz) if fz else None}
    (model_dir / "lockbox_ids.json").write_text(json.dumps(sorted(lock_ids)) + "\n")
    for t in res["targets"]:
        ycol = f"y_{t}"
        dfl = df_all[np.isfinite(df_all[ycol].to_numpy(float))]
        df = dfl[dfl["split"] == "train"].reset_index(drop=True)
        y, w = df[ycol].to_numpy(float), df["_w"].to_numpy(float)
        fsets = feature_sets(df)
        if feature_set not in fsets:
            raise SystemExit(f"[{t}] feature set {feature_set!r} not available (have {sorted(fsets)}); "
                             "choose one with --save-feature-set")
        cols = fsets[feature_set]
        use = input_columns(cols)
        est, _ = fit_predict(model, cols, df, y, w, df.iloc[:1], cfg["seed"], cfg["fit_weighted"])
        mfile = model_dir / f"model_{t}.joblib"
        joblib.dump(est, mfile)
        # reference: content-scheme out-of-fold predictions of this model, joined to the train rows
        o = oof_all[(oof_all["target"] == t) & (oof_all["scheme"] == "content")]
        if pred_col not in o:
            raise SystemExit(f"[{t}] {pred_col} missing from oof_predictions.csv")
        ref = o[["vp_id", "video_id", "y", "w", pred_col]].rename(columns={pred_col: "pred"}).merge(
            df[["vp_id", "deal_id", "platform", "social_account_id", "incl_prob", "_content"]].astype(
                {"vp_id": str}), on="vp_id", how="inner", validate="one_to_one")
        if len(ref) != len(o) or not np.allclose(ref["y"], df.set_index("vp_id").loc[ref["vp_id"], ycol]):
            raise SystemExit(f"[{t}] oof_predictions.csv does not match this dataset (rerun the evaluation)")
        ref = ref[~ref["video_id"].isin(lock_ids)]  # none by construction (train rows only); belt and braces
        content = (ref.groupby(["deal_id", "platform", "video_id"], sort=True)
                   .agg(pred=("pred", "mean"), y=("y", "mean"), w=("w", "first"), incl_prob=("incl_prob", "first"),
                        n_posts=("vp_id", "size")).reset_index())
        content["obs_pct"] = _stratum_pct(content["y"].to_numpy(float),
                                          (content["deal_id"] + "|" + content["platform"]).to_numpy(),
                                          content["w"].to_numpy(float))
        n_acct = df.groupby(["deal_id", "platform", "social_account_id"]).size()
        big = n_acct[n_acct >= cfg.get("min_account_n", 40)].reset_index()[["deal_id", "platform",
                                                                            "social_account_id"]]
        big = big[big["social_account_id"] != "missing"]
        acct = (ref.merge(big, on=["deal_id", "platform", "social_account_id"])
                .groupby(["deal_id", "platform", "social_account_id", "video_id"], sort=True)
                .agg(pred=("pred", "mean"), y=("y", "mean"), w=("w", "first")).reset_index())
        acct = acct.merge(n_acct.rename("n_train_posts").reset_index(), on=["deal_id", "platform",
                                                                           "social_account_id"])
        # reference-average linear terms per deal×platform (drivers are relative to these)
        terms = linear_terms(est, df[use])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred_in = est.predict(df[use])
        key = df[["deal_id", "platform"]].reset_index(drop=True)
        tw = terms.mul(w, axis=0)
        g = pd.concat([key, tw, pd.DataFrame({"_w": w, "_p": pred_in * w, "_n": 1})], axis=1).groupby(
            ["deal_id", "platform"], sort=True).sum()
        rterms = g[terms.columns].div(g["_w"], axis=0)
        rterms.insert(0, "pred_mean", g["_p"] / g["_w"])
        rterms.insert(0, "n_posts", g["_n"].astype(int))
        files = {"model": mfile.name, "reference": f"reference_{t}.csv",
                 "reference_accounts": f"reference_accounts_{t}.csv", "reference_terms": f"reference_terms_{t}.csv",
                 "imputation": f"imputation_{t}.json"}
        content.to_csv(model_dir / files["reference"], index=False)
        acct.to_csv(model_dir / files["reference_accounts"], index=False)
        rterms.reset_index().to_csv(model_dir / files["reference_terms"], index=False)
        (model_dir / files["imputation"]).write_text(json.dumps(imputation_medians(est), indent=1) + "\n")
        metrics = served_metrics(res, t, feature_set, model)
        # prereg: a confirmatory brain claim needs the stage-2 lockbox (the sealed 225 + the extension)
        scored = set(oof_all.loc[(oof_all["target"] == t) & (oof_all["scheme"] == "lockbox"), "video_id"])
        metrics["lockbox_contents"] = {"n": len(scored), "n_extension": len(scored & lockbox_ext),
                                       "includes_extension": bool(scored & lockbox_ext)}
        manifest["targets"][t] = {
            "files": files, "sha256": {k: _sha256(model_dir / v) for k, v in files.items()},
            "columns": {"cat": cols[0], "num": cols[1]}, "input_columns": use,
            "n_train_posts": int(len(df)), "n_train_contents": int(df["video_id"].nunique()),
            "train_video_ids_sha256": _sha256_text("\n".join(sorted(set(df["video_id"].astype(str))))),
            "n_reference_contents": int(len(content)), "deals": sorted(df["deal_id"].unique().tolist()),
            "strata": sorted((df["deal_id"] + "|" + df["platform"]).unique().tolist()),
            "metrics": metrics,
        }
        log(f"  saved {t}: {mfile} ({len(df)} train posts, {len(content)} reference contents)")
    (model_dir / "train_video_ids.json").write_text(json.dumps(sorted(train_ids)) + "\n")
    (model_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=_json_default) + "\n")
    return manifest


def _sha256_text(s: str) -> str:
    import hashlib

    return hashlib.sha256(s.encode()).hexdigest()


def _json_default(o):
    if isinstance(o, np.floating):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--members", type=Path, required=True)
    ap.add_argument("--outcomes", type=Path, required=True)
    ap.add_argument("--selection", type=Path, default=None, help="study selection (split, incl_prob, anchor_account)")
    ap.add_argument("--out-dir", type=Path, default=Path("results/models"))
    ap.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    ap.add_argument("--schemes", nargs="+", default=list(SCHEMES), choices=SCHEMES)
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--min-deal-n", type=int, default=30)
    ap.add_argument("--min-account-n", type=int, default=40, help="training posts before an account gets its own layer")
    ap.add_argument("--perm-repeats", type=int, default=3, help="0 disables permutation importance")
    ap.add_argument("--perm-model", default="hgb", choices=MODELS)
    ap.add_argument("--no-perm-per-feature", action="store_true", help="families only (faster)")
    ap.add_argument("--fit-weighted", action="store_true", help="also weight model fitting by 1/incl_prob")
    ap.add_argument("--niche-base", default=None, choices=MODELS,
                    help="general model under the niche adjustment (default: the last of --models; must be one)")
    ap.add_argument("--no-niche", action="store_true")
    ap.add_argument("--score-lockbox", action="store_true",
                    help="fit on all train rows and score the lockbox (final numbers; do this once, at the end)")
    ap.add_argument("--exclude-if-true", nargs="*", default=[], help="extra outcomes flag columns to drop on")
    ap.add_argument("--threads", type=int, default=8, help="BLAS/OpenMP threads (keep modest on a shared box)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-non-english", action="store_true",
                    help="keep clips whisperx confidently detected as non-English (their text features are garbage)")
    ap.add_argument("--save-model", type=Path, default=None, metavar="DIR",
                    help="after the evaluation: refit the served model on all non-lockbox train rows and write it "
                         "with its manifest and reference tables (tools/predict.py)")
    ap.add_argument("--save-feature-set", default="BE", choices=("A", "B", "E", "BE"))
    ap.add_argument("--save-model-kind", default="stack", choices=("stack", "ridge"))
    ap.add_argument("--lockbox-ext", type=Path, default=DEFAULT_LOCKBOX_EXT,
                    help="stage-2 lockbox extension (video_id); with --save-model only: never trained on or used "
                         "as a reference (default: %(default)s when it exists)")
    args = ap.parse_args()
    features, members, outcomes = read_table(args.features), read_table(args.members), read_table(args.outcomes)
    selection = read_table(args.selection) if args.selection else None
    ext = read_lockbox_ext(args.lockbox_ext) if args.save_model else set()
    if args.save_model and ext and selection is not None and "video_id" in selection:  # fail before a long run
        tr_ids = set(selection.loc[selection.get("split", pd.Series("train", index=selection.index)).astype(str)
                                   == "train", "video_id"].astype(str))
        if tr_ids & ext:
            raise SystemExit(f"{len(tr_ids & ext)} lockbox-extension contents are train rows in {args.selection}")
    res = run(features, members, outcomes, targets=args.targets,
        out_dir=args.out_dir, selection=selection,
        schemes=args.schemes, models=args.models, n_splits=args.n_splits, n_boot=args.n_boot, seed=args.seed,
        min_deal_n=args.min_deal_n, perm_repeats=args.perm_repeats, perm_model=args.perm_model,
        perm_per_feature=not args.no_perm_per_feature, exclude_cols=args.exclude_if_true,
        fit_weighted=args.fit_weighted, niche_base=args.niche_base, niche=not args.no_niche,
        min_account_n=args.min_account_n, score_lockbox=args.score_lockbox, threads=args.threads,
        exclude_non_english=not args.keep_non_english)
    if args.save_model:
        from threadpoolctl import threadpool_limits

        with threadpool_limits(args.threads):
            _save_from_cli(args, features, members, outcomes, selection, res, ext)
    return 0


def _save_from_cli(args, features, members, outcomes, selection, res, ext):
    save_model(features, members, outcomes, res=res, eval_dir=args.out_dir, model_dir=args.save_model,
               selection=selection, feature_set=args.save_feature_set, model=args.save_model_kind,
               lockbox_ext=ext, exclude_cols=args.exclude_if_true, exclude_non_english=not args.keep_non_english,
               paths={"features": args.features, "members": args.members, "outcomes": args.outcomes,
                      "selection": args.selection, "lockbox_ext": args.lockbox_ext if ext else None})


if __name__ == "__main__":
    # run through the importable module, so a saved model pickles as fit_models.<Class>, not __main__.<Class>
    import fit_models

    sys.exit(fit_models.main())
