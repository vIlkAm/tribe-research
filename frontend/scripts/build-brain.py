"""Build the anatomical web mesh. Requires numpy, nibabel and pyyaml.

python scripts/build-brain.py --research-repo /path/to/tribe-research
The HCP annotation hashes and fsaverage5 subset follow that repo's build_roi_map.py.
Only anatomy and region membership are exported, never model vertex predictions.
"""
import argparse
import concurrent.futures
import hashlib
import json
import struct
import tempfile
import urllib.request
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'https://raw.githubusercontent.com/nilearn/nilearn/0.12.1/nilearn/datasets/data/fsaverage5'
ANNOT = 'https://raw.githubusercontent.com/tannerjared/HCP-MMP1/main'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--research-repo', type=Path, required=True)
    args = parser.parse_args()
    cache = Path(tempfile.gettempdir()) / 'viralbrain-anatomy'
    cache.mkdir(exist_ok=True)
    files = {f'{kind}_{hemi}.gii.gz': f'{SOURCE}/{kind}_{hemi}.gii.gz'
             for kind in ['pial', 'sulc'] for hemi in ['left', 'right']}
    files.update({f'{h}.HCPMMP1.annot': f'{ANNOT}/{h}.HCP-MMP1.annot' for h in ['lh', 'rh']})
    def fetch(item):
        name, url = item
        path = cache / name
        if not path.exists():
            req = urllib.request.Request(url, headers={'User-Agent': 'ViralBrain-research-demo'})
            with urllib.request.urlopen(req, timeout=60) as response:
                path.write_bytes(response.read())
        return name
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for name in pool.map(fetch, files.items()):
            print('Ready:', name, flush=True)
    expected = {'lh': '46a102b59b2fb1bb4bd62d51bf02e975', 'rh': '75e96b331940227bbcb07c1c791c2463'}
    for hemi, digest in expected.items():
        assert hashlib.md5((cache / f'{hemi}.HCPMMP1.annot').read_bytes()).hexdigest() == digest

    sample = json.loads((ROOT / 'public/data/cd20b16879d630c4/analysis.json').read_text())
    # Prototype only: refuse changed internal data until contract issue #1 is agreed.
    group_bytes = (args.research_repo / 'tribe_research/brain/roi_groups_v0.yaml').read_bytes()
    assert hashlib.sha256(group_bytes).hexdigest() == sample['provenance']['roi_map']['groups_sha256']
    groups = yaml.safe_load(group_bytes)['groups']
    channels = sample['channels']
    parcel_channels = {}
    for i, channel in enumerate(channels):
        for group in channel['basis']['roi_groups']:
            for parcel in groups[group]['parcels']:
                name = str(parcel)
                assert name not in parcel_channels or parcel_channels[name] == i
                parcel_channels[name] = i

    positions, faces, folds, regions = [], [], [], []
    for h, hemi in enumerate(['left', 'right']):
        mesh = nib.load(cache / f'pial_{hemi}.gii.gz')
        coords, tris = mesh.darrays[0].data, mesh.darrays[1].data
        assert len(coords) == 10242
        # FreeSurfer RAS -> WebGL: x right, y superior, z anterior.
        pos = np.column_stack([coords[:, 0], coords[:, 2], -coords[:, 1]])
        positions.append(pos)
        faces.append(tris + h * 10242)
        folds.append(nib.load(cache / f'sulc_{hemi}.gii.gz').darrays[0].data)
        labels, _, names = nib.freesurfer.read_annot(str(cache / f'{["lh", "rh"][h]}.HCPMMP1.annot'))
        region = np.full(10242, -1, dtype=np.float32)
        for i, label in enumerate(labels[:10242]):
            if label >= 0:
                name = names[label].decode()[2:].replace('_ROI', '')
                region[i] = parcel_channels.get(name, -1)
        regions.append(region)
    pos = np.vstack(positions).astype('<f4')
    pos -= (pos.max(axis=0) + pos.min(axis=0)) / 2
    tris = np.vstack(faces).astype('<u4')
    sulc = np.concatenate(folds).astype('<f4')
    ids = np.concatenate(regions).astype('<f4')
    out = ROOT / 'public/brain'
    out.mkdir(exist_ok=True)
    binary = struct.pack('<II', len(pos), len(tris)) + pos.tobytes() + sulc.tobytes() + ids.tobytes() + tris.tobytes()
    (out / 'brain.bin').write_bytes(binary)
    metadata = {
        'format': 'header u32[2], positions f32[N*3], sulc f32[N], channel f32[N], faces u32[F*3]',
        'vertices': len(pos), 'triangles': len(tris), 'channel_keys': [c['key'] for c in channels],
        'channel_vertex_counts': {c['key']: int((ids == i).sum()) for i, c in enumerate(channels)},
        'anatomy': 'FreeSurfer fsaverage5, pial surface, supplied by Nilearn 0.12.1',
        'atlas': 'HCP-MMP1, exact annotation hashes from tribe-research',
        'mapping': 'Same region unions as tribe-research proxies_v0; left then right, first 10242 fsaverage vertices per hemisphere.',
        'display': 'A region gets its channel z-score. This is not per-vertex TRIBE output. Glow shimmer is an illustrative display effect.',
        'sha256': hashlib.sha256(binary).hexdigest(),
        'sources': files,
    }
    (out / 'brain.meta.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({k: metadata[k] for k in ['vertices', 'triangles', 'channel_vertex_counts']}, indent=2))

if __name__ == '__main__':
    main()
