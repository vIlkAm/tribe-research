#!/usr/bin/env python3
"""Package analysis bundles for the frontend: validate, index, tar.

    .venv/bin/python tools/handoff.py --analyses results/run1/report/analyses \
        --out results/handoff/run1.tar.gz [--expect-real]

Every ``<video_id>/analysis.json`` must validate against
``docs/analysis.schema.json``; any failure stops the handoff (exit 1, nothing
written). The tarball holds ``index.json``, each bundle's JSON and brain images,
and ``_static/``. It never holds clip footage: ``*.mp4`` (the demo video shows
the source clip) and anything not on the allow-list are left out, per
docs/TEAM.md. ``--expect-real`` also refuses bundles marked synthetic.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHIP = {"analysis.json", "brain_proxy.jpg", "brain_vertex.jpg", "summary.png"}


def collect(analyses: Path, expect_real: bool) -> tuple[list[dict], list[str]]:
    import jsonschema

    schema = json.loads((ROOT / "docs/analysis.schema.json").read_text())
    validator = jsonschema.Draft202012Validator(schema) if "2020-12" in schema.get("$schema", "") \
        else jsonschema.validators.validator_for(schema)(schema)
    index, errors = [], []
    for p in sorted(analyses.glob("*/analysis.json")):
        a = json.loads(p.read_text())
        errs = sorted(validator.iter_errors(a), key=lambda e: list(e.path))
        if errs:
            e = errs[0]
            errors.append(f"{p.parent.name}: {'/'.join(map(str, e.path)) or '<root>'}: {e.message[:200]}"
                          + (f" (+{len(errs) - 1} more)" if len(errs) > 1 else ""))
            continue
        if expect_real and a["synthetic"]:
            errors.append(f"{p.parent.name}: synthetic bundle in a real handoff")
            continue
        index.append({
            "video_id": a["video_id"], "analysis_id": a["analysis_id"], "path": f"{p.parent.name}/analysis.json",
            "duration_ms": a["duration_ms"], "status": a["status"], "synthetic": a["synthetic"],
            "n_channels": len(a["channels"]), "n_moments": len(a["moments"]),
            "has_words": a["quality"]["has_words"], "n_warnings": len(a["quality"]["warnings"]),
        })
    return index, errors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--analyses", type=Path, required=True, help="brain_report's <report-dir>/analyses")
    ap.add_argument("--out", type=Path, required=True, help="output .tar.gz")
    ap.add_argument("--expect-real", action="store_true", help="refuse synthetic bundles")
    args = ap.parse_args()

    index, errors = collect(args.analyses, args.expect_real)
    if errors or not index:
        for e in errors[:50]:
            print(f"INVALID {e}", file=sys.stderr)
        print(f"handoff stopped: {len(errors)} invalid, {len(index)} valid", file=sys.stderr)
        return 1
    schema_version = json.loads((args.analyses / index[0]["path"]).read_text())["schema_version"]
    doc = {"schema_version": schema_version, "count": len(index),
           "synthetic": any(r["synthetic"] for r in index), "bundles": index}

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
          f"{args.out.stat().st_size / 1e6:.1f} MB -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
