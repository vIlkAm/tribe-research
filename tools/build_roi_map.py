#!/usr/bin/env python3
"""Build the versioned fsaverage5 ROI map artifact from the HCP-MMP1 annotation.

Output (``tribe_research/assets/roi_map_<groups-version>.npz``):

    vertex_parcel  int16 [20484]  parcel index per fsaverage5 vertex (-1 = medial wall)
    parcel_names   str   [P]      bilateral HCP-MMP1 names (e.g. "V1", "TPOJ1")
    group_names    str   [G]
    group_mask     bool  [G, 20484]
    provenance     JSON string: annot md5s, source URLs, groups file sha256

Vertex order matches TRIBE predictions: left hemisphere 0..10241, then right
10242..20483. fsaverage5 is the first 10242 vertices of each fsaverage (ico7)
hemisphere, so the fsaverage annotation is subset, not resampled. This mirrors
``tribev2.utils.get_hcp_labels``; run ``--check-against-tribe`` on the pod to
prove the indices agree before trusting any ROI curve.

The annotation is fetched from figshare (the same source MNE and TRIBE use),
verified against MNE's pinned MD5s. Some hosts are blocked by figshare (this
server is); pass ``--annot-dir`` with lh/rh.HCPMMP1.annot copied from anywhere.
The HCP-MMP1 data is subject to the HCP data-use terms.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GROUPS = ROOT / "tribe_research/brain/roi_groups_v0.yaml"
ASSETS = ROOT / "tribe_research/assets"

FS5 = 10242  # vertices per hemisphere
ANNOT = {
    "lh": ("https://ndownloader.figshare.com/files/5528816", "46a102b59b2fb1bb4bd62d51bf02e975"),
    "rh": ("https://ndownloader.figshare.com/files/5528819", "75e96b331940227bbcb07c1c791c2463"),
}


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def fetch_annots(annot_dir: Path | None) -> dict[str, Path]:
    out = {}
    tmp = Path(tempfile.mkdtemp(prefix="hcpmmp1-"))
    for hemi, (url, want) in ANNOT.items():
        path = (annot_dir / f"{hemi}.HCPMMP1.annot") if annot_dir else tmp / f"{hemi}.HCPMMP1.annot"
        if not path.exists():
            req = urllib.request.Request(url, headers={"User-Agent": "tribe-research/roi-map"})
            with urllib.request.urlopen(req, timeout=60) as r:
                path.write_bytes(r.read())
        got = md5(path)
        if got != want:
            raise SystemExit(
                f"{path}: md5 {got} != expected {want}. "
                "figshare may be blocking this host (a 403 page is 118 bytes); use --annot-dir."
            )
        out[hemi] = path
    return out


def clean_name(raw: bytes | str) -> str:
    # "L_V1_ROI" -> "V1", same normalisation as tribev2.utils.get_hcp_labels
    name = raw.decode() if isinstance(raw, bytes) else raw
    return name[2:].replace("_ROI", "")


def build(annots: dict[str, Path], groups_path: Path) -> dict:
    import nibabel as nib

    parcel_index: dict[str, int] = {}
    vertex_parcel = np.full(2 * FS5, -1, dtype=np.int16)
    for h, hemi in enumerate(("lh", "rh")):
        labels, _ctab, names = nib.freesurfer.read_annot(str(annots[hemi]))
        if labels.shape[0] != 163842:
            raise SystemExit(f"{hemi}: expected fsaverage (163842 vertices), got {labels.shape[0]}")
        for v, lab in enumerate(labels[:FS5]):
            if lab < 0 or names[lab] in (b"???", "???"):
                continue
            name = clean_name(names[lab])
            idx = parcel_index.setdefault(name, len(parcel_index))
            vertex_parcel[h * FS5 + v] = idx

    parcel_names = list(parcel_index)
    groups_doc = yaml.safe_load(groups_path.read_text())
    unknown = {
        g: [p for p in spec["parcels"] if str(p) not in parcel_index]
        for g, spec in groups_doc["groups"].items()
    }
    unknown = {g: ps for g, ps in unknown.items() if ps}
    if unknown:
        raise SystemExit(f"parcels not in HCP-MMP1: {unknown}")

    group_names = list(groups_doc["groups"])
    group_mask = np.zeros((len(group_names), 2 * FS5), dtype=bool)
    for gi, g in enumerate(group_names):
        ids = [parcel_index[str(p)] for p in groups_doc["groups"][g]["parcels"]]
        group_mask[gi] = np.isin(vertex_parcel, ids)

    provenance = {
        "atlas": "HCPMMP1 (Glasser et al. 2016) on fsaverage, subset to fsaverage5",
        "annot": {h: {"url": ANNOT[h][0], "md5": md5(p)} for h, p in annots.items()},
        "groups_file": groups_path.name,
        "groups_version": groups_doc["version"],
        "groups_sha256": hashlib.sha256(groups_path.read_bytes()).hexdigest(),
        "vertex_order": "lh 0..10241, rh 10242..20483 (TRIBE fsaverage5 order)",
        "n_parcels": len(parcel_names),
        "group_vertex_counts": {g: int(group_mask[i].sum()) for i, g in enumerate(group_names)},
    }
    return {
        "vertex_parcel": vertex_parcel,
        "parcel_names": np.array(parcel_names),
        "group_names": np.array(group_names),
        "group_mask": group_mask,
        "provenance": json.dumps(provenance),
    }


def check_against_tribe(art: dict) -> None:
    from tribev2.utils import get_hcp_labels  # pod only

    tribe = get_hcp_labels(mesh="fsaverage5", combine=False, hemi="both")
    names = list(art["parcel_names"])
    bad = []
    for name, verts in tribe.items():
        if "?" in name:  # medial wall ("???"), which we store as -1
            continue
        if name not in names:
            bad.append(f"{name}: missing")
            continue
        ours = np.flatnonzero(art["vertex_parcel"] == names.index(name))
        if not np.array_equal(np.sort(ours), np.sort(np.asarray(verts))):
            bad.append(f"{name}: vertex sets differ")
    if bad:
        raise SystemExit("ROI map disagrees with tribev2.utils.get_hcp_labels:\n  " + "\n  ".join(bad[:20]))
    print(f"matches tribev2.get_hcp_labels for all {len(tribe)} parcels")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--groups", type=Path, default=DEFAULT_GROUPS)
    ap.add_argument("--annot-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--check-against-tribe", action="store_true")
    args = ap.parse_args()

    art = build(fetch_annots(args.annot_dir), args.groups)
    version = json.loads(art["provenance"])["groups_version"]
    out = args.out or ASSETS / f"roi_map_{version}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **art)
    print(f"wrote {out}")
    print(json.dumps(json.loads(art["provenance"])["group_vertex_counts"], indent=2))
    if args.check_against_tribe:
        check_against_tribe(art)
    return 0


if __name__ == "__main__":
    sys.exit(main())
