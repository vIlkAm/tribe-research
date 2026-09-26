"""tools/handoff.py with the optional per-bundle performance.json and the per-clip index metadata.

Bundles are the synthetic docs/sample_analysis bundle under two ids; performance.json comes from the predict.py
fixture docs/sample_analysis/performance/preliminary.json (re-targeted to each bundle). Offline, no ROI map.
"""

from __future__ import annotations

import copy
import json
import shutil
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import handoff  # noqa: E402

pytest.importorskip("jsonschema")
SAMPLE = ROOT / "docs/sample_analysis"
VIDS = ("aaaa000000000001", "bbbb000000000002")


def _perf(vid: str, **kw) -> dict:
    blk = json.loads((SAMPLE / "performance/preliminary.json").read_text())["performance"]
    blk = copy.deepcopy(blk)
    blk["provenance"]["video_id"] = vid
    blk.update(kw)
    return blk


def _bundles(tmp: Path, perf: dict | None = None) -> Path:
    src = next(p for p in SAMPLE.iterdir() if (p / "analysis.json").exists())
    root = tmp / "analyses"
    shutil.copytree(SAMPLE / "_static", root / "_static")
    for i, vid in enumerate(VIDS):
        d = root / vid
        shutil.copytree(src, d)
        a = json.loads((d / "analysis.json").read_text())
        a.update(video_id=vid, analysis_id=f"a_{i:012x}")
        (d / "analysis.json").write_text(json.dumps(a))
        (d / "demo.mp4").write_bytes(b"footage")  # never ships
        if perf and vid in perf:
            (d / "performance.json").write_text(json.dumps(perf[vid]))
    return root


def _meta(tmp: Path, lock=(False, True), **override) -> Path:
    rows = [{"video_id": v, "platform": "tiktok", "video_link": f"https://www.tiktok.com/@x/video/{i}",
             "is_lockbox": lock[i], "vp_id": f"vp-{i}", "deal_id": "deal-a", "deal_label": "Deal A"}
            for i, v in enumerate(VIDS)]
    for vid, upd in override.items():
        next(r for r in rows if r["video_id"] == vid).update(upd)
    p = tmp / "clip_meta.json"
    p.write_text(json.dumps(rows))
    return p


def _ok_perf():
    return {VIDS[0]: _perf(VIDS[0]), VIDS[1]: _perf(VIDS[1], clip_in_training="lockbox", retrospective=True)}


def test_without_performance_the_old_handoff_is_unchanged(tmp_path):
    root = _bundles(tmp_path)
    out = tmp_path / "h.tar.gz"
    assert handoff.main(["--analyses", str(root), "--out", str(out)]) == 0
    tar = tarfile.open(out)
    names = tar.getnames()
    assert not any(n.endswith((".mp4", "performance.json")) for n in names)
    idx = json.load(tar.extractfile("h/index.json"))
    assert "model_version" not in idx and idx["count"] == 2
    assert all(r["performance"] is None for r in idx["bundles"])


def test_performance_and_clip_meta_land_in_the_index(tmp_path):
    root = _bundles(tmp_path, _ok_perf())
    out = tmp_path / "v2.tar.gz"
    rc = handoff.main(["--analyses", str(root), "--out", str(out), "--clip-meta", str(_meta(tmp_path)),
                       "--require-performance", "--release", "data-x-v2", "--model-release", "model-x"])
    assert rc == 0
    tar = tarfile.open(out)
    names = tar.getnames()
    assert all(f"v2/{v}/performance.json" in names for v in VIDS)
    assert not any(n.endswith((".mp4", ".wav", ".mp3", ".m4a")) for n in names)
    idx = json.load(tar.extractfile("v2/index.json"))
    assert idx["model_version"] == _perf(VIDS[0])["model_version"] and idx["model_release"] == "model-x"
    assert idx["release"] == "data-x-v2" and idx["performance_status_counts"] == {"preliminary": 2}
    assert idx["performance_schema"] == json.loads(handoff.PERF_SCHEMA.read_text())["$id"]
    rows = {r["video_id"]: r for r in idx["bundles"]}
    for i, v in enumerate(VIDS):
        r = rows[v]
        assert r["platform"] == "tiktok" and r["video_link"].startswith("https://") and r["is_lockbox"] is (i == 1)
        assert r["performance_path"] == f"{v}/performance.json"
        assert r["performance"] == {"model_status": "preliminary", "validated": False,
                                    "clip_in_training": "lockbox" if i else "no"}
    # the index carries statuses, never a performance number
    assert not any(isinstance(x, (int, float)) and not isinstance(x, bool) for r in idx["bundles"]
                   for x in r["performance"].values())


@pytest.mark.parametrize("case", ["wrong_vid", "two_models", "lock_mismatch", "outcome_key", "no_caption",
                                  "no_meta_row", "missing_perf", "platform"])
def test_bad_performance_stops_the_handoff(tmp_path, case):
    perf = _ok_perf()
    meta = {}
    if case == "wrong_vid":
        perf[VIDS[0]]["provenance"]["video_id"] = VIDS[1]
    elif case == "two_models":
        perf[VIDS[1]]["model_version"] = "perf-stack-BE_v1+20990101.0000000"
    elif case == "lock_mismatch":
        meta = {VIDS[1]: {"is_lockbox": False}}
    elif case == "outcome_key":  # an observed outcome smuggled into a (lockbox) clip's block
        perf[VIDS[1]]["engagement"]["observed_percentile"] = 0.9
    elif case == "no_caption":  # schema: preliminary needs its caption
        perf[VIDS[0]]["caption"] = None
    elif case == "no_meta_row":
        meta = {VIDS[0]: {"video_id": "someone-else"}}
    elif case == "missing_perf":
        del perf[VIDS[1]]
    elif case == "platform":
        meta = {VIDS[0]: {"platform": "youtube"}}
    root = _bundles(tmp_path, perf)
    out = tmp_path / "bad.tar.gz"
    args = ["--analyses", str(root), "--out", str(out), "--clip-meta", str(_meta(tmp_path, **meta)),
            "--require-performance"]
    if case == "outcome_key":
        # the schema already refuses an unknown key; the outcome-key guard stands behind it
        assert handoff.outcome_keys(perf[VIDS[1]]) == ["/engagement/observed_percentile"]
    assert handoff.main(args) == 1 and not out.exists()


def test_clip_meta_needs_the_owner_fields(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps([{"video_id": VIDS[0], "platform": "tiktok", "video_link": "https://x"}]))
    with pytest.raises(SystemExit, match="is_lockbox"):
        handoff.load_clip_meta(p)
    p.write_text(json.dumps([{"video_id": VIDS[0], "platform": "tiktok", "video_link": "", "is_lockbox": False}]))
    with pytest.raises(SystemExit, match="video_link"):
        handoff.load_clip_meta(p)


def test_bundle_performance_uses_the_clips_own_post_and_no_label():
    import pandas as pd

    import bundle_performance as bp

    members = pd.DataFrame({"video_id": ["v1", "v1"], "vp_id": ["p-ig", "p-yt"], "platform": ["instagram", "youtube"],
                            "deal_id": ["d-1", "d-1"], "representative": ["False", "True"]})
    posts = pd.DataFrame({"id": ["p-ig", "p-yt"], "deal_id": ["0123abcd-ef", "0123abcd-ef"],
                          "platform": ["instagram", "youtube"], "social_account_id": ["acc-1", None],
                          "follower_count": [None, 1200.0],
                          "upload_date": [None, pd.Timestamp("2026-01-02", tz="UTC")],
                          "video_link": ["https://ig/p", "https://yt/s"]}).set_index("id")
    assert list(posts.reset_index().columns) == bp.CONTEXT_COLS  # the only outcomes columns it reads
    sel = pd.DataFrame({"video_id": ["v1"], "split": ["lockbox"], "width": [1080], "height": [1920],
                        "audio_mean_db": [-18.0]}).set_index("video_id")
    ctx, link = bp.post_context("v1", "p-yt", posts, members, sel)
    assert link == "https://yt/s" and ctx["platform"] == "youtube" and ctx["account_id"] is None
    assert ctx["deal_label"] == "Deal 0123abcd" and ctx["posted_at"].startswith("2026-01-02")
    assert ctx["follower_count"] == 1200.0 and ctx["width"] == 1080.0
    with pytest.raises(SystemExit, match="not one of its posts"):
        bp.post_context("v1", "p-other", posts, members, sel)
    assert bp.lockbox_set(sel.reset_index(), None) == {"v1"}
