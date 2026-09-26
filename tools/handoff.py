#!/usr/bin/env python3
"""Package analysis bundles for the frontend: validate, index, tar.

    .venv/bin/python tools/handoff.py --analyses results/run1/report/analyses \
        --out results/handoff/run1.tar.gz [--expect-real] [--clip-meta clip_meta.json] [--require-performance]

Every ``<video_id>/analysis.json`` must validate against
``docs/analysis.schema.json``; any failure stops the handoff (exit 1, nothing
written). The tarball holds ``index.json``, each bundle's JSON and brain images,
and ``_static/``. It never holds clip footage: ``*.mp4`` (the demo video shows
the source clip) and anything not on the allow-list are left out, per
docs/TEAM.md. ``--expect-real`` also refuses bundles marked synthetic.

Optional per bundle: ``performance.json`` (tools/predict.py, written next to ``analysis.json`` by
tools/bundle_performance.py so the v0.2 analysis stays untouched). It must validate against
``docs/performance.schema.draft.json``, name its own bundle in ``provenance.video_id`` and carry no observed-outcome
field; all bundles of one handoff must come from one ``model_version``. ``index.json`` then carries that
``model_version`` and, per clip, the status (``model_status``, ``validated``, ``clip_in_training``), never a
number. ``--clip-meta`` (a JSON list from tools/bundle_performance.py) adds ``platform``, ``video_link`` and
``is_lockbox`` per clip; ``is_lockbox`` must agree with ``clip_in_training == "lockbox"``.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHIP = {"analysis.json", "performance.json", "brain_proxy.jpg", "brain_vertex.jpg", "summary.png"}
PERF_SCHEMA = ROOT / "docs/performance.schema.draft.json"
CLIP_META_KEYS = ("platform", "video_link", "is_lockbox")
# keys that would carry an observed outcome (a label) into a bundle; predict.py never writes them
# (predict.py's `engagement`/`reach` are predicted blocks and stay allowed)
OUTCOME_KEYS = {"y", "obs_pct", "observed", "observed_percentile", "views", "views_final", "likes", "comments",
                "shares", "saves", "interactions", "reach_rel", "reach_rel_local", "log_interactions_rate",
                "eng_pct", "anchor_views", "engagement_rate_reported", "pct_interactions_deal_platform"}


def _validator(path: Path):
    import jsonschema

    schema = json.loads(path.read_text())
    return jsonschema.Draft202012Validator(schema) if "2020-12" in schema.get("$schema", "") \
        else jsonschema.validators.validator_for(schema)(schema)


def _first_error(validator, doc) -> str | None:
    errs = sorted(validator.iter_errors(doc), key=lambda e: list(e.path))
    if not errs:
        return None
    e = errs[0]
    return (f"{'/'.join(map(str, e.path)) or '<root>'}: {e.message[:200]}"
            + (f" (+{len(errs) - 1} more)" if len(errs) > 1 else ""))


def outcome_keys(x, path="") -> list[str]:
    """Paths of keys anywhere in ``x`` that name an observed outcome."""
    out = []
    if isinstance(x, dict):
        for k, v in x.items():
            if k in OUTCOME_KEYS:
                out.append(f"{path}/{k}")
            out += outcome_keys(v, f"{path}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            out += outcome_keys(v, f"{path}[{i}]")
    return out


def load_clip_meta(path: Path | None) -> dict[str, dict] | None:
    if path is None:
        return None
    rows = json.loads(path.read_text())
    meta = {}
    for r in rows:
        missing = [k for k in ("video_id", *CLIP_META_KEYS) if k not in r]
        if missing:
            raise SystemExit(f"{path}: {r.get('video_id')}: missing {missing}")
        if not isinstance(r["is_lockbox"], bool) or not r.get("platform") or not r.get("video_link"):
            raise SystemExit(f"{path}: {r['video_id']}: platform/video_link empty or is_lockbox not a boolean")
        meta[str(r["video_id"])] = r
    return meta


def collect(analyses: Path, expect_real: bool, clip_meta: dict[str, dict] | None = None,
            require_performance: bool = False) -> tuple[list[dict], list[str], str | None]:
    """(index rows, errors, model_version of the performance blocks or None)."""
    validator = _validator(ROOT / "docs/analysis.schema.json")
    perf_validator = None
    index, errors, versions = [], [], set()
    for p in sorted(analyses.glob("*/analysis.json")):
        a = json.loads(p.read_text())
        err = _first_error(validator, a)
        if err:
            errors.append(f"{p.parent.name}: {err}")
            continue
        if expect_real and a["synthetic"]:
            errors.append(f"{p.parent.name}: synthetic bundle in a real handoff")
            continue
        row = {
            "video_id": a["video_id"], "analysis_id": a["analysis_id"], "path": f"{p.parent.name}/analysis.json",
            "duration_ms": a["duration_ms"], "status": a["status"], "synthetic": a["synthetic"],
            "n_channels": len(a["channels"]), "n_moments": len(a["moments"]),
            "has_words": a["quality"]["has_words"], "n_warnings": len(a["quality"]["warnings"]),
        }
        pp = p.parent / "performance.json"
        if pp.exists():
            perf_validator = perf_validator or _validator(PERF_SCHEMA)
            blk = json.loads(pp.read_text())
            err = _first_error(perf_validator, blk)
            if err:
                errors.append(f"{p.parent.name}: performance.json: {err}")
                continue
            if blk["provenance"].get("video_id") != a["video_id"] or p.parent.name != a["video_id"]:
                errors.append(f"{p.parent.name}: performance.json is for {blk['provenance'].get('video_id')}")
                continue
            bad = outcome_keys(blk)
            if bad:
                errors.append(f"{p.parent.name}: performance.json has observed-outcome keys {bad[:3]}")
                continue
            versions.add(blk["model_version"])
            row["performance_path"] = f"{p.parent.name}/performance.json"
            row["performance"] = {"model_status": blk["model_status"], "validated": blk["validated"],
                                  "clip_in_training": blk["clip_in_training"]}
        elif require_performance:
            errors.append(f"{p.parent.name}: no performance.json")
            continue
        else:
            row["performance_path"], row["performance"] = None, None
        if clip_meta is not None:
            m = clip_meta.get(a["video_id"])
            if m is None:
                errors.append(f"{p.parent.name}: not in --clip-meta")
                continue
            if row["performance"] and m["is_lockbox"] != (row["performance"]["clip_in_training"] == "lockbox"):
                errors.append(f"{p.parent.name}: is_lockbox {m['is_lockbox']} disagrees with clip_in_training "
                              f"{row['performance']['clip_in_training']!r}")
                continue
            if row["performance"] and m["platform"] != blk["context"]["platform"]:
                errors.append(f"{p.parent.name}: clip-meta platform {m['platform']!r} vs performance context "
                              f"{blk['context']['platform']!r}")
                continue
            row.update({k: m[k] for k in CLIP_META_KEYS})
        index.append(row)
    if len(versions) > 1:
        errors.append(f"performance.json from {len(versions)} model versions {sorted(map(str, versions))}: one "
                      "handoff, one model")
    return index, errors, (next(iter(versions)) if len(versions) == 1 else None)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--analyses", type=Path, required=True, help="brain_report's <report-dir>/analyses")
    ap.add_argument("--out", type=Path, required=True, help="output .tar.gz")
    ap.add_argument("--expect-real", action="store_true", help="refuse synthetic bundles")
    ap.add_argument("--clip-meta", type=Path, default=None,
                    help="JSON list (tools/bundle_performance.py): video_id, platform, video_link, is_lockbox")
    ap.add_argument("--require-performance", action="store_true", help="every bundle must have performance.json")
    ap.add_argument("--release", default=None, help="release tag recorded in index.json (e.g. data-frontend40-v2)")
    ap.add_argument("--model-release", default=None, help="model release tag recorded in index.json")
    args = ap.parse_args(argv)

    clip_meta = load_clip_meta(args.clip_meta)
    index, errors, model_version = collect(args.analyses, args.expect_real, clip_meta, args.require_performance)
    if errors or not index:
        for e in errors[:50]:
            print(f"INVALID {e}", file=sys.stderr)
        print(f"handoff stopped: {len(errors)} invalid, {len(index)} valid", file=sys.stderr)
        return 1
    schema_version = json.loads((args.analyses / index[0]["path"]).read_text())["schema_version"]
    doc = {"schema_version": schema_version, "count": len(index),
           "synthetic": any(r["synthetic"] for r in index), "bundles": index}
    if any(r.get("performance") for r in index):
        statuses: dict[str, int] = {}
        for r in index:
            s = (r["performance"] or {}).get("model_status", "absent")
            statuses[s] = statuses.get(s, 0) + 1
        doc = {"schema_version": schema_version,
               "performance_schema": json.loads(PERF_SCHEMA.read_text())["$id"],
               "model_version": model_version, "model_release": args.model_release, "release": args.release,
               "performance_status_counts": statuses, **{k: v for k, v in doc.items() if k != "schema_version"}}
    elif args.release:
        doc["release"] = args.release

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".part")
    top = args.out.name.removesuffix(".tar.gz")
    with tarfile.open(tmp, "w:gz") as tar:
        data = json.dumps(doc, indent=1).encode()
        info = tarfile.TarInfo(f"{top}/index.json")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
        for r in index:
            d = args.analyses / r["video_id"]
            for f in sorted(d.iterdir()):
                if f.name in SHIP:
                    tar.add(f, arcname=f"{top}/{r['video_id']}/{f.name}")
        static = args.analyses / "_static"
        if static.exists():
            for f in sorted(static.iterdir()):
                if f.suffix in {".png", ".jpg", ".json"}:
                    tar.add(f, arcname=f"{top}/_static/{f.name}")
    tmp.replace(args.out)
    print(f"{len(index)} bundles ({'SYNTHETIC' if doc['synthetic'] else 'real'}), "
          f"{args.out.stat().st_size / 1e6:.1f} MB -> {args.out}"
          + (f", model {model_version}" if model_version else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
