#!/usr/bin/env python3
"""Independent audit of the served INTERNAL demo set (frontend/dist): clip identity and length, per-second
percentiles recomputed by brute force, observed numbers vs the raw DB export, lockbox, selection independence.

    .venv/bin/python tools/audit_demo_local.py
"""
import sys, json, hashlib, pickle
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(ROOT/'tools'), str(ROOT)]
import make_manifest as mm
import build_library_profile as blp
from tribe_research.brain import moments_pop as mp

PUB = ROOT/'frontend/dist'
idx = json.load(open(PUB/'demo-stage1/index.json'))
ids = [b['video_id'] for b in idx['bundles']]
ok = True
def check(cond, msg):
    global ok
    print(('PASS ' if cond else 'FAIL ') + msg); ok &= bool(cond)

# A. identity + duration
for b in idx['bundles']:
    v = b['video_id']; f = PUB/'clips'/f'{v}.mp4'
    check(mm.content_id(f) == v, f'{v}: served mp4 sha256[:16] == video_id (same file the brain analysis ran on)')
    d = mm.mvhd_duration(f); a = json.load(open(PUB/f'demo-stage1/{v}/analysis.json'))
    check(d is not None and abs(d*1000 - a['duration_ms']) < 150, f'{v}: mp4 {d:.2f}s vs analysis {a["duration_ms"]/1000:.2f}s')

# B. percentiles by brute force from the raw library (not the Library class)
state = mp.load(blp.DEFAULT_STATE)
clips = pickle.load(open(ROOT/'results/library/.library_cache_v0.pkl','rb'))['clips']
ents = [blp.entry_from_clip(c, state['norms']) for c in clips]
pos = {e.video_id: i for i, e in enumerate(ents)}
rows = []
for i, e in enumerate(ents):
    for t in range(len(e.r)):
        if np.isfinite(e.r[t]): rows.append((i, e.lb, int(e.sb[t]), e.r[t]))
R = pd.DataFrame(rows, columns=['owner','lb','sb','r'])
for v in ids:
    lj = json.load(open(PUB/f'demo-stage1/{v}/library.json'))
    e = ents[pos[v]]; me = pos[v]; mine = np.full(len(e.r), np.nan)
    for t in range(len(e.r)):
        if not np.isfinite(e.r[t]): continue
        ref = R[(R.lb == e.lb) & (R.sb == e.sb[t]) & (R.owner != me)]
        if ref.owner.nunique() < mp.MIN_NORM_N: ref = R[(R.sb == e.sb[t]) & (R.owner != me)]
        x = ref.r.to_numpy(); mine[t] = 100*((x < e.r[t]).sum() + 0.5*(x == e.r[t]).sum())/len(x)
    grid = np.full(len(lj['index']['percentile']), np.nan); grid[e.sec] = mine
    served = np.array([np.nan if p is None else p for p in lj['index']['percentile']], float)
    diff = np.nanmax(np.abs(grid - served))
    check(diff < 0.01, f'{v}: per-second percentile recomputed independently, max diff {diff:.4f}')
    check(lj['reference']['self_excluded'], f'{v}: clip excluded from its own reference')
    r_served = np.array([np.nan if x is None else x for x in lj['index']['values']], float)
    rg = np.full(len(r_served), np.nan); rg[e.sec] = np.nanmean(e.u, 0)
    check(np.nanmax(np.abs(rg - r_served)) < 0.002, f'{v}: response index = plain mean of the 7 channels')

# C/D. observed numbers vs the raw DB export, and post <-> clip mapping
vp = pd.read_csv(ROOT/'results/metrics/video_performances.csv', dtype=str).set_index('id')
mem = pd.read_csv(ROOT/'results/run_full/members.csv', dtype=str)
pair = json.load(open(ROOT/'results/demo/example_pair.json'))
for p in pair['picks']:
    v, sn = p['video_id'], p['source_name']
    o = json.load(open(PUB/f'demo-stage1/{v}/observed.json')); raw = vp.loc[sn]
    check(sn in set(mem[mem.video_id == v].vp_id), f'{v}: shown post is one of this clip\'s own posts')
    check(o['video_link'] == raw['video_link'], f'{v}: link matches raw export')
    for k, rk in (('views','views'), ('likes','likes'), ('comments','comments')):
        check(float(o[k]) == float(raw[rk]), f'{v}: {k} {o[k]:.0f} == raw export {float(raw[rk]):.0f}')

# E. lockbox + selection independence
sel = pd.read_csv(ROOT/'results/study/selection.csv', dtype=str); ext = pd.read_csv(ROOT/'results/study/lockbox_ext.csv', dtype=str)
lock = set(sel[sel.split == 'lockbox'].video_id) | set(ext.video_id)
check(not (set(ids) & lock), 'no demo clip is a sealed lockbox clip')
src = (ROOT/'tools/select_example_pair.py').read_text()
check('library' not in src and 'npz' not in src, 'pair rule never reads brain data')
src = (ROOT/'tools/select_showcase.py').read_text()
check('outcomes' not in src.replace('No outcome is read', '').replace('never reads outcomes', '').replace('never an\noutcome', ''), 'showcase rule never reads outcomes')

# F. learned.json headline numbers vs the stage-1 metrics file
L = json.load(open(PUB/'learned.json')); s1 = L['stage1']
check((s1['rho_A_metadata'], s1['rho_BE_brain_video']) == (0.335, 0.344), 'learned page stage-1 numbers match STAGE1_RESULT.md')
print('\nALL PASS' if ok else '\nSOME CHECKS FAILED')
sys.exit(0 if ok else 1)
