"""docs/analysis.types.ts must match docs/analysis.schema.json field for field."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / "docs/analysis.schema.json").read_text())
TS = (ROOT / "docs/analysis.types.ts").read_text()


def ts_fields(name: str) -> dict[str, bool]:
    """Top-level fields of `export interface name` -> optional?"""
    body = re.search(rf"export interface {name} \{{\n(.*?)\n\}}", TS, re.S).group(1)
    return {m[0]: bool(m[1]) for m in re.findall(r"^  (\w+)(\?)?:", body, re.M)}


def check(name: str, schema: dict):
    got = ts_fields(name)
    props, req = set(schema.get("properties", {})), set(schema.get("required", []))
    assert not (set(got) - props), f"{name}: fields not in schema {set(got) - props}"
    missing = req - {k for k, opt in got.items() if not opt}
    assert not missing, f"{name}: schema-required fields missing or optional in TS {missing}"
    if schema.get("additionalProperties") is False:
        assert props <= set(got), f"{name}: schema fields missing in TS {props - set(got)}"


def test_types_match_schema():
    d = SCHEMA["$defs"]
    check("Analysis", SCHEMA)
    check("Channel", d["channel"])
    check("Moment", d["moment"])
    check("Sprite", d["sprite"])
    check("RegionMap", SCHEMA["properties"]["assets"]["properties"]["region_map"])
    for enum_name, values in [("Direction", d["channel"]["properties"]["direction"]["enum"]),
                              ("MomentKind", d["moment"]["properties"]["kind"]["enum"]),
                              ("SignalKind", d["kind"]["enum"])]:
        block = re.search(rf"export type {enum_name} =(.*?);", TS, re.S).group(1)
        assert set(re.findall(r'"(\w+)"', block)) == set(values), enum_name
    assert f'schema_version: "{SCHEMA["properties"]["schema_version"]["const"]}"' in TS
