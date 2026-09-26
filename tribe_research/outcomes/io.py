"""Load and type-coerce the read-only metrics export (CSV files with header rows).

Every table is read with all columns as strings and then coerced explicitly, so
that an id column with a null never turns into ``"123.0"`` and a malformed
number becomes null instead of failing the whole load. Missing *columns* are
added as all-null (and reported); missing *optional files*
(``social_account_stat_snapshots``, ``cross_platform_members``) become empty
tables. Nothing here talks to a database.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# column name -> kind: id | num | bool | ts (timestamptz) | date | text
SCHEMA: dict[str, dict[str, str]] = {
    "video_performances": {
        "id": "id", "social_account_id": "id", "video_link": "text",
        "views": "num", "likes": "num", "comments": "num", "shares": "num", "saves": "num",
        "engagement_rate": "num", "upload_date": "date", "last_updated": "ts",
        "duration_seconds": "num", "is_campaign_video": "bool", "payout_eligible": "bool",
        "deal_campaign_id": "id",
        # optional extras (used when present, never reported as missing)
        "provider_observed_at": "ts", "history_truncated": "bool", "caar_video_gone_since": "ts",
    },
    "video_snapshots": {
        "video_performance_id": "id", "snapshot_at": "ts", "snapshot_date": "date",
        "views": "num", "likes": "num", "comments": "num", "shares": "num", "saves": "num",
        "source": "text",
    },
    "social_accounts": {
        "id": "id", "deal_id": "id", "platform": "text", "handle": "text", "user_id": "id",
    },
    "social_account_stat_snapshots": {
        "social_account_id": "id", "follower_count": "num", "snapshot_at": "ts",
    },
    "deals": {"id": "id", "name": "text"},
    "cross_platform_members": {"link_id": "id", "video_performance_id": "id"},
}
OPTIONAL_COLUMNS = {"provider_observed_at", "history_truncated", "caar_video_gone_since"}
REQUIRED_TABLES = ("video_performances", "video_snapshots", "social_accounts", "deals")
OPTIONAL_TABLES = ("social_account_stat_snapshots", "cross_platform_members")
NA_VALUES = ["", "null", "NULL", "None", "\\N", "NaN", "nan"]
_TRUE = {"t", "true", "1", "yes", "y"}
_FALSE = {"f", "false", "0", "no", "n"}
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def _as_id(s: pd.Series) -> pd.Series:
    """Ids as strings; integral floats (from a nullable int column) lose the '.0'."""
    if pd.api.types.is_float_dtype(s):
        s = s.astype("Int64")
    st = s.astype("string").str.strip()
    keep = (st.notna() & st.ne("").fillna(False)).to_numpy(dtype=bool)
    return st.astype(object).where(keep, None)


def _as_bool(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.astype("boolean")
    low = s.astype(object).map(lambda v: None if pd.isna(v) else str(v).strip().lower())
    out = pd.Series(pd.NA, index=s.index, dtype="boolean")
    out[low.isin(_TRUE)] = True
    out[low.isin(_FALSE)] = False
    return out


def _as_ts(s: pd.Series) -> pd.Series:
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        return s.dt.tz_convert("UTC")
    if pd.api.types.is_datetime64_dtype(s):
        return s.dt.tz_localize("UTC")
    return pd.to_datetime(s, utc=True, format="ISO8601", errors="coerce")


def coerce_table(name: str, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Coerce one table to the expected kinds; returns (frame, missing_columns)."""
    spec = SCHEMA[name]
    df = df.copy()
    absent = [c for c in spec if c not in df.columns]
    for col in absent:
        df[col] = None
    missing = [c for c in absent if c not in OPTIONAL_COLUMNS]
    for col, kind in spec.items():
        s = df[col]
        if kind == "id":
            df[col] = _as_id(s)
        elif kind == "num":
            df[col] = pd.to_numeric(s, errors="coerce").astype("float64")
        elif kind == "bool":
            df[col] = _as_bool(s)
        elif kind == "ts":
            df[col] = _as_ts(s)
        elif kind == "date":
            df[col] = _as_ts(s).dt.normalize()
        else:
            df[col] = s.astype(object).where(pd.notna(s), None)
    return df, missing


def empty_table(name: str) -> pd.DataFrame:
    return coerce_table(name, pd.DataFrame({c: pd.Series([], dtype=object) for c in SCHEMA[name]}))[0]


def coerce_tables(raw: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], dict]:
    """Coerce in-memory frames (tests / notebooks) exactly like the CSV loader."""
    tables, report = {}, {"missing_columns": {}, "missing_optional_tables": []}
    for name in REQUIRED_TABLES:
        if name not in raw:
            raise KeyError(f"required table {name!r} missing")
    for name in SCHEMA:
        if name not in raw or raw[name] is None:
            tables[name] = empty_table(name)
            report["missing_optional_tables"].append(name)
            continue
        tables[name], missing = coerce_table(name, raw[name])
        if missing:
            report["missing_columns"][name] = missing
    return tables, report


def load_tables(metrics_dir: Path) -> tuple[dict[str, pd.DataFrame], dict]:
    """Read ``<metrics_dir>/<table>.csv`` for every known table."""
    metrics_dir = Path(metrics_dir)
    raw: dict[str, pd.DataFrame | None] = {}
    for name in SCHEMA:
        path = metrics_dir / f"{name}.csv"
        if not path.exists():
            if name in REQUIRED_TABLES:
                raise FileNotFoundError(f"required export missing: {path}")
            raw[name] = None
            continue
        raw[name] = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=NA_VALUES)
    return coerce_tables(raw)


def to_days(s: pd.Series) -> np.ndarray:
    """UTC timestamps -> float days since epoch (NaN for NaT)."""
    return ((s - EPOCH).dt.total_seconds() / 86400.0).to_numpy(dtype="float64", na_value=np.nan)
