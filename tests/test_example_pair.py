"""select_example_pair: thirds per stratum, views must agree with the label, lockbox refused, hash order."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import select_example_pair as sep  # noqa: E402


def _posts(tmp_path, n=30):
    rows = [{"target": sep.TARGET, "scheme": "content", "vp_id": f"p{i}", "video_id": f"v{i}", "y": float(i),
             "w": 1.0, "stratum": "d1|youtube", "pred_A_stack": 0.0} for i in range(n)]
    rows.append({**rows[0], "stratum": "d2|youtube", "vp_id": "other"})
    f = tmp_path / "oof.csv"
    pd.DataFrame(rows).to_csv(f, index=False)
    return sep.label_posts(f, "d1|youtube")


def _outcomes(reach):
    return pd.DataFrame([{"id": k, "platform": "youtube", "video_link": f"https://x/{k}", "upload_date": "2026-01-01",
                          "views_final": 100.0, "likes": 1.0, "comments": 0.0, "shares": 0.0, "saves": np.nan,
                          "engagement_rate_reported": 1.0, "reach_rel_local": r, "local_baseline_n": 10,
                          "age_days_at_last_obs": 50.0, "dq_flags": np.nan} for k, r in reach.items()]).set_index("id")


def _elig(vids, rank):
    return [{"video_id": v, "deal_id": "d1", "platform": "youtube", "source_name": f"p{v[1:]}",
             "source_path": f"deal/{v}.mp4", "rank_key": rank[v]} for v in vids]


def _staging(tmp_path, vids):
    for v in vids:
        p = tmp_path / "staging/chunk0/deal" / f"{v}.mp4"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    return tmp_path / "staging"


def test_thirds_and_views_gate(tmp_path):
    posts = _posts(tmp_path)
    assert len(posts) == 30 and (posts["third"] == -1).sum() == 10 and (posts["third"] == 1).sum() == 10
    vids = ["v1", "v2", "v28", "v29", "v15"]
    rank = {"v1": "b", "v2": "a", "v28": "a", "v29": "b", "v15": "0"}
    # v2 is bottom third but got more views than usual: views contradict the label, so v1 is taken
    out = _outcomes({"p1": -0.5, "p2": 0.3, "p28": 1.0, "p29": 2.0, "p15": -1.0})
    res = sep.pick(_elig(vids, rank), {"deal_id": "d1", "platform": "youtube"}, posts, out, set(),
                   _staging(tmp_path, vids))
    got = {p["role"]: p["video_id"] for p in res["picks"]}
    assert got == {"fell_short": "v1", "beat_expectations": "v28"}
    assert res["counts"]["not_in_a_group"] == 2  # v2 (views up) and v15 (middle third)
    assert "account_id" not in res["picks"][0]["observed"]


def test_dq_flags_and_missing_source_excluded(tmp_path):
    posts = _posts(tmp_path)
    out = _outcomes({"p1": -0.5, "p2": -0.5})
    out["dq_flags"] = pd.Series({"p1": "history_truncated"}, dtype=object).reindex(out.index)
    res = sep.pick(_elig(["v1", "v2"], {"v1": "a", "v2": "b"}), {"deal_id": "d1", "platform": "youtube"}, posts,
                   out, set(), _staging(tmp_path, []))
    assert res["picks"] == [] and res["counts"]["dq_flags"] == 1 and res["counts"]["no_source_file"] == 1


def test_lockbox_refused(tmp_path):
    posts = _posts(tmp_path)
    with pytest.raises(SystemExit, match="lockbox"):
        sep.pick(_elig(["v1"], {"v1": "a"}), {"deal_id": "d1", "platform": "youtube"}, posts,
                 _outcomes({"p1": -0.5}), {"v1"}, _staging(tmp_path, ["v1"]))


def test_small_stratum_refused(tmp_path):
    f = tmp_path / "oof.csv"
    pd.DataFrame([{"target": sep.TARGET, "scheme": "content", "vp_id": "p", "video_id": "v", "y": 0.0, "w": 1.0,
                   "stratum": "d1|youtube", "pred_A_stack": 0.0}]).to_csv(f, index=False)
    with pytest.raises(SystemExit, match="need >= 30"):
        sep.label_posts(f, "d1|youtube")
