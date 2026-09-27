"""tools/select_demo_clips.py (B6) and tools/build_demo_release.py (B7), on synthetic files only.

The selector must follow its rule literally and never depend on an outcome column; the builder must let through
only ``research_preview`` / ``not_trained`` blocks (never ``preliminary``), keep the hero first and write a sha256
manifest into the tarball.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import build_demo_release as bdr  # noqa: E402
import select_demo_clips as sdc  # noqa: E402

SAMPLE = ROOT / "docs/sample_analysis"
DEALS = ["deal-a", "deal-b", "deal-c", "deal-d"]


def _clip(i: int, deal: str, **kw) -> dict:
    c = {"video_id": f"v{i:03d}", "deal_id": deal, "split": "train", "duration_s": 30.0, "audio_mean_db": -15.0,
         "lang": "en", "prob": 0.97, "n_words": 75, "platform": "tiktok", "status": "ok"}
    c.update(kw)
    return c


def make_world(tmp: Path, clips: list[dict], ext=(), incomplete=False, extra_members=()) -> dict:
    out = tmp / "runs" / "outputs-b02"
    (out / "worker-0").mkdir(parents=True)
    bdir = tmp / "batches" / "b02"
    bdir.mkdir(parents=True)
    man, sel, mem = [], [], []
    for c in clips:
        vid = c["video_id"]
        man.append({"video_id": vid, "path": f"{c['deal_id']}-folder/{vid}.mp4", "source_name": f"vp-{vid}",
                    "duration_s": c["duration_s"]})
        sel.append({"video_id": vid, "path": f"{c['deal_id']}-folder/{vid}.mp4", "deal_id": c["deal_id"],
                    "split": c["split"], "stratum": "acct:0", "incl_prob": 1.0, "duration_s": c["duration_s"],
                    "reach": 0.5, "eng_pct": 0.5, "anchor_views": 1000.0, "audio_mean_db": c["audio_mean_db"]})
        mem.append({"video_id": vid, "vp_id": f"vp-{vid}", "platform": c["platform"], "deal_id": c["deal_id"],
                    "representative": "True"})
        if c["status"] == "error":
            (out / "worker-0" / f"{vid}.error.json").write_text(json.dumps({"video_id": vid, "error": "x"}))
            continue
        if incomplete and c is clips[-1]:
            continue
        meta = {"video_id": vid, "duration_s": c["duration_s"], "source_name": f"vp-{vid}",
                "transcript": {"n_words": c["n_words"], "detected_language": c["lang"],
                               "language_probability": c["prob"]},
                "tribe_commit": "t", "video_config": {"fast_video": True, "video_precision": "bf16"},
                "emb": {"version": "emb_pool_v1"}, "runtime": {"dry_run": False}}
        (out / "worker-0" / f"{vid}.json").write_text(json.dumps(meta))
        np.savez(out / "worker-0" / f"{vid}.npz", preds=np.zeros((2, 4), np.float16))
        np.savez(out / "worker-0" / f"{vid}.emb.npz", x=np.zeros(2))
    mem += list(extra_members)
    (bdir / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in man))
    p = {"selection": tmp / "selection.csv", "members": tmp / "members.csv", "ext": tmp / "lockbox_ext.csv"}
    pd.DataFrame(sel).to_csv(p["selection"], index=False)
    pd.DataFrame(mem).to_csv(p["members"], index=False)
    pd.DataFrame({"video_id": list(ext) or ["none"]}).to_csv(p["ext"], index=False)
    return {**p, "out": out, "batches": tmp / "batches", "logs": tmp / "logs"}


def _select(w, **kw):
    return sdc.select([w["out"]], w["selection"], w["ext"], w["members"], w["batches"], [w["logs"]], **kw)


def _stratum_fill(deal: str, platform: str, n: int, start: int) -> list[dict]:
    """Extra train contents (members only, no outputs) so a deal x platform reaches n contents."""
    return [{"video_id": f"x{deal}{start + k}", "vp_id": f"vp-x{deal}{start + k}", "platform": platform,
             "deal_id": deal, "representative": "True"} for k in range(n)]


def _fill_selection(w, members_extra):
    sel = pd.read_csv(w["selection"])
    add = pd.DataFrame([{"video_id": m["video_id"], "path": "p", "deal_id": m["deal_id"], "split": "train",
                         "stratum": "acct:0", "incl_prob": 1.0, "duration_s": 30.0, "reach": 0.1, "eng_pct": 0.1,
                         "anchor_views": 1.0, "audio_mean_db": -15.0} for m in members_extra])
    pd.concat([sel, add]).to_csv(w["selection"], index=False)


def test_selection_rule_filters_draws_three_deals_and_ignores_outcomes(tmp_path):
    clips = [
        _clip(0, "deal-a"), _clip(1, "deal-a"), _clip(2, "deal-b"), _clip(3, "deal-c"), _clip(4, "deal-d"),
        _clip(5, "deal-b", lang="es"), _clip(6, "deal-c", prob=0.8), _clip(7, "deal-d", duration_s=15.0),
        _clip(8, "deal-d", duration_s=65.0), _clip(9, "deal-a", n_words=30),  # 1.0 word/s
        _clip(10, "deal-b", audio_mean_db=-45.0), _clip(11, "deal-c", split="lockbox"), _clip(12, "deal-a"),
        _clip(13, "deal-b", status="error"),
    ]
    fill = _stratum_fill("deal-b", "tiktok", 30, 0) + _stratum_fill("deal-c", "tiktok", 30, 0) + \
        _stratum_fill("deal-d", "tiktok", 30, 0)
    w = make_world(tmp_path, clips, ext=["v012"], extra_members=fill)
    _fill_selection(w, fill)
    res = _select(w)
    c = res["counts"]
    assert c["excluded_language"] == 2 and c["excluded_duration"] == 2 and c["excluded_words_per_s"] == 1
    assert c["excluded_audio_level"] == 1 and c["excluded_not a study train clip outside both lockboxes"] == 2
    assert c["eligible"] == 5 and res["complete"]
    picks = res["picks"]
    assert [p["role"] for p in picks] == ["hero", "fallback_a", "fallback_b"]
    assert len({p["deal_id"] for p in picks}) == 3
    # deal-a's stratum has < 30 contents: its clips come only after the preferred strata
    assert all(p["preferred_stratum"] for p in picks) and "deal-a" not in {p["deal_id"] for p in picks}
    assert [p["video_id"] for p in picks] == [e["video_id"] for e in sorted(
        (p for p in picks), key=lambda e: sdc.hash_rank(e["video_id"]))]
    assert all(p["source_path"].endswith(f"{p['video_id']}.mp4") and p["source_name"] == f"vp-{p['video_id']}"
               for p in picks)
    assert res["inputs"]["outcomes_read"] is False and "reach" not in res["inputs"]["selection"]["columns"]
    # outcome columns scrambled or removed: same picks (they are never read)
    sel = pd.read_csv(w["selection"])
    sel[["reach", "eng_pct", "anchor_views"]] = sel[["reach", "eng_pct", "anchor_views"]].sample(
        frac=1, random_state=1).to_numpy() * -7
    sel.to_csv(w["selection"], index=False)
    assert _select(w)["ids"] == res["ids"]
    sel.drop(columns=["reach", "eng_pct", "anchor_views"]).to_csv(w["selection"], index=False)
    assert _select(w)["ids"] == res["ids"]


def test_selection_is_deterministic_and_falls_back_to_small_strata(tmp_path):
    clips = [_clip(i, DEALS[i % 3]) for i in range(9)]
    w = make_world(tmp_path, clips)
    a, b = _select(w), _select(w)
    assert a["ids"] == b["ids"] and a["complete"]
    assert not any(p["preferred_stratum"] for p in a["picks"])  # no stratum reaches 30: still three deals
    assert _select(w, params={**sdc.PARAMS, "seed": 1})["ids"] != a["ids"] or len(clips) < 4


def test_selection_skips_incomplete_batches_and_accounts_for_errors(tmp_path):
    clips = [_clip(i, DEALS[i % 3]) for i in range(6)] + [_clip(6, "deal-a", status="error")]
    w = make_world(tmp_path, clips)
    ok, why = sdc.batch_complete(w["out"], w["batches"], [w["logs"]])
    assert ok and "1 worker errors" in why
    shutil.rmtree(tmp_path)
    w = make_world(tmp_path, [_clip(i, DEALS[i % 3]) for i in range(6)], incomplete=True)
    res = _select(w)
    assert res["batches_used"] == [] and not res["complete"] and res["picks"] == []
    w["logs"].mkdir()
    (w["logs"] / "done-b02").write_text("")  # the A100 queue's marker counts as complete
    assert _select(w)["batches_used"][0]["evidence"] == "done-b02"


# ── build_demo_release ───────────────────────────────────────────────────


def _perf(name: str, **kw) -> dict:
    blk = copy.deepcopy(json.loads((SAMPLE / "performance" / f"{name}.json").read_text())["performance"])
    blk.update(kw)
    return blk


@pytest.mark.parametrize("name,ok", [("not_trained", True), ("research_preview", True), ("train_oof", True),
                                     ("preliminary", False), ("validated", False), ("out_of_scope_too_long", False),
                                     ("lockbox", False)])
def test_only_research_preview_or_not_trained_ship(tmp_path, name, ok):
    blk = _perf(name)
    (tmp_path / "v1").mkdir()
    (tmp_path / "v1" / "performance.json").write_text(json.dumps(blk))
    if ok:
        assert bdr.check_blocks(tmp_path, ["v1"], not_trained=False)["v1"]["model_status"] == blk["model_status"]
    else:
        with pytest.raises(SystemExit):
            bdr.check_blocks(tmp_path, ["v1"], not_trained=False)
    if name == "research_preview":
        with pytest.raises(SystemExit, match="--not-trained"):
            bdr.check_blocks(tmp_path, ["v1"], not_trained=True)


def test_not_trained_with_numbers_is_refused(tmp_path):
    blk = _perf("not_trained", drivers=[{"family": "metadata", "label": "x", "contribution": 0.1}])
    (tmp_path / "v1").mkdir()
    (tmp_path / "v1" / "performance.json").write_text(json.dumps(blk))
    with pytest.raises(SystemExit, match="drivers"):
        bdr.check_blocks(tmp_path, ["v1"], not_trained=True)


def test_build_not_trained_release_end_to_end(tmp_path, monkeypatch):
    clips = [_clip(i, DEALS[i % 3]) for i in range(6)]
    w = make_world(tmp_path, clips)
    sel_json = tmp_path / "demo_selection.json"
    res = _select(w)
    sel_json.write_text(json.dumps(res))
    posts = pd.DataFrame([{"id": f"vp-{c['video_id']}", "deal_id": c["deal_id"] + "0000", "platform": "tiktok",
                           "social_account_id": None, "follower_count": None, "upload_date": None,
                           "video_link": f"https://www.tiktok.com/@x/video/{i}", "views": 123}
                          for i, c in enumerate(clips)])
    posts.to_parquet(tmp_path / "outcomes.parquet")
    src = next(p for p in SAMPLE.iterdir() if (p / "analysis.json").exists())

    def fake_render(vid, out_root, roi_map, work, videos_root, synthetic=False):  # brain_report needs a real ROI map
        d = work / f"report-{vid}" / "analyses"
        shutil.copytree(src, d / vid)
        shutil.copytree(SAMPLE / "_static", d / "_static", dirs_exist_ok=True)
        (d / vid / "demo.mp4").write_bytes(b"footage")  # rendered locally, never shipped
        a = json.loads((d / vid / "analysis.json").read_text())
        a.update(video_id=vid, analysis_id="a_" + hashlib.sha256(vid.encode()).hexdigest()[:12], synthetic=False, source_name=f"vp-{vid}")
        (d / vid / "analysis.json").write_text(json.dumps(a))
        return d

    monkeypatch.setattr(bdr, "render_bundle", fake_render)
    out = tmp_path / "rel"
    rc = bdr.main(["--selection", str(sel_json), "--not-trained", "--out", str(out), "--study-selection",
                   str(w["selection"]), "--lockbox-ext", str(w["ext"]), "--members", str(w["members"]),
                   "--outcomes", str(tmp_path / "outcomes.parquet"), "--batches-dir", str(w["batches"])])
    assert rc == 0
    tgz = out / "data-demo-stage1-v1.tar.gz"
    tar = tarfile.open(tgz)
    names = tar.getnames()
    assert not any(n.endswith((".mp4", ".npz", ".joblib")) for n in names)
    idx = json.load(tar.extractfile("data-demo-stage1-v1/index.json"))
    assert [b["video_id"] for b in idx["bundles"]] == res["ids"]  # hero first
    assert idx["performance_status_counts"] == {"not_trained": 3} and idx["release"] == "data-demo-stage1-v1"
    for b in idx["bundles"]:
        assert b["is_lockbox"] is False and b["platform"] == "tiktok" and b["video_link"].startswith("https://")
    man = json.load(tar.extractfile("data-demo-stage1-v1/RELEASE_MANIFEST.json"))
    assert man["status"] == "not_trained" and man["validated"] is False
    assert [c["role"] for c in man["clips"]] == ["hero", "fallback_a", "fallback_b"]
    assert man["selection"]["ids"] == res["ids"] and "folder" not in json.dumps(man)  # no source filenames
    for rel, digest in man["files_sha256"].items():
        assert hashlib.sha256(tar.extractfile(f"data-demo-stage1-v1/{rel}").read()).hexdigest() == digest
    for b in idx["bundles"]:
        blk = json.load(tar.extractfile(f"data-demo-stage1-v1/{b['performance_path']}"))
        assert blk["engagement"] is None and blk["reach"] is None and blk["drivers"] == []
        assert "views" not in json.dumps(blk) and blk["context"]["account_id"] is None
        assert json.load(tar.extractfile(f"data-demo-stage1-v1/{b['path']}"))["source_name"] is None
    assert (out / "data-demo-stage1-v1.RELEASE_MANIFEST.json").exists() and not (out / "work").exists()
    # exactly one of --model-dir / --not-trained
    with pytest.raises(SystemExit):
        bdr.main(["--selection", str(sel_json), "--out", str(tmp_path / "rel2")])


def test_scrub_drops_account_id_and_refuses_numbers_in_the_reason(tmp_path):
    blk = _perf("not_trained")
    blk["context"]["account_id"] = "acct-1"
    blk["reason"] = "BE - A = +0.009 fails the GO rule"
    (tmp_path / "v1").mkdir()
    (tmp_path / "v1" / "performance.json").write_text(json.dumps(blk))
    with pytest.raises(SystemExit, match="numbers"):
        bdr.scrub_blocks(tmp_path, ["v1"], None)  # the stock reason quotes CV numbers
    with pytest.raises(SystemExit, match="numbers"):
        bdr.scrub_blocks(tmp_path, ["v1"], "failed at 0.02")
    bdr.scrub_blocks(tmp_path, ["v1"], "The stage-one test did not pass.")
    out = json.loads((tmp_path / "v1" / "performance.json").read_text())
    assert out["context"]["account_id"] is None and out["reason"] == "The stage-one test did not pass."
