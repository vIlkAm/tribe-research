#!/usr/bin/env python3
"""Independent audit of the served INTERNAL demo set (frontend/dist): clip identity and length, per-second
percentiles recomputed by brute force, observed numbers vs the raw DB export, lockbox, selection independence.

    .venv/bin/python tools/audit_demo_local.py [dist|public]
"""
import sys, json, hashlib, pickle, inspect
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(ROOT/'tools'), str(ROOT)]
import make_manifest as mm
import build_library_profile as blp
from tribe_research.brain import moments_pop as mp

PUB = ROOT/'frontend'/(sys.argv[1] if len(sys.argv) > 1 else 'dist')
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

# C/D. observed numbers vs the raw DB export, post <-> clip mapping, tier recomputed from the outcomes table
vp = pd.read_csv(ROOT/'results/metrics/video_performances.csv', dtype=str).set_index('id')
mem = pd.read_csv(ROOT/'results/run_full/members.csv', dtype=str)
man = {r['video_id']: r for r in map(json.loads, open(ROOT/'results/run_full/manifest.jsonl'))} \
    if (ROOT/'results/run_full/manifest.jsonl').exists() else {}
demo = json.load(open(ROOT/idx.get('demo_source', 'results/demo/library_demo.json')))
owner = demo.get('selection') in ('owner', 'candidates')
check(demo.get('selection', 'rule') == idx.get('selection', 'rule'), f"index says selection={idx.get('selection')}, pick file agrees")
picks = {p['video_id']: p for ps in demo['picks'].values() for p in ps}
check(set(picks) == set(ids), 'served clips == the tier picks (nothing extra, nothing missing)')
oc = pd.read_parquet(ROOT/'results/outcomes.parquet', columns=['id', 'deal_id', 'platform', 'reach_rel_local', 'dq_flags'])
oc = oc[oc.reach_rel_local.notna() & oc.dq_flags.isna() & oc.deal_id.notna()]
for b in idx['bundles']:
    v = b['video_id']; p = picks[v]; sn = p['source_name']
    o = json.load(open(PUB/f'demo-stage1/{v}/observed.json')); raw = vp.loc[sn]
    check(sn in set(mem[mem.video_id == v].vp_id), f'{v}: shown post is one of this clip\'s own posts')
    if man:
        check(man[v].get('source_name') == sn, f'{v}: shown post is the manifest representative post')
    check(o['video_link'] == raw['video_link'] and b['video_link'] == raw['video_link'], f'{v}: link matches raw export')
    for k in ('views', 'likes', 'comments'):
        check(float(o[k]) == float(raw[k]), f'{v}: {k} {o[k]:.0f} == raw export {float(raw[k]):.0f}')
    k = sum(float(raw[c]) for c in ('likes', 'comments', 'shares', 'saves') if isinstance(raw.get(c), str) and raw[c] != '')
    check(o['engagement_rate_pct'] is not None and abs(o['engagement_rate_pct'] - 100 * min(k, float(raw['views'])) / float(raw['views'])) < 1e-6,
          f"{v}: engagement {o['engagement_rate_pct']:.2f}% == (likes+comments+shares+saves)/views from the raw export")
    # tier from scratch: plain percentile of this post's reach_rel_local among its deal x platform posts
    me = oc[oc.id.astype(str) == sn].iloc[0]
    ref = oc[(oc.deal_id == me.deal_id) & (oc.platform == me.platform)].reach_rel_local.to_numpy()
    x = me.reach_rel_local; pct = 100 * ((ref < x).sum() + 0.5 * (ref == x).sum()) / len(ref)
    basis = demo.get('tier_basis', 'relative')
    if basis == 'views':  # owner rule: views the post got; bad also below its account's usual
        vf = float(raw['views'])
        want = 'great' if vf >= 300_000 else 'typical' if 5_000 <= vf <= 15_000 else 'bad' if vf < 700 and x < 0 else None
        check(o['tier'] == b['demo_role'] == want, f'{v}: tier {o["tier"]} ok ({vf:,.0f} views, {np.exp(x):.2f}x usual)')
    else:
        lo, hi = {'great': (80, 100), 'typical': (40, 60), 'bad': (0, 20)}[o['tier']]
        check(o['tier'] == b['demo_role'] and lo <= pct <= hi and len(ref) >= 100 and len(ref) == o['n_ref_posts']
              and abs(pct - o['views_pct_in_deal_platform']) < 0.1,
              f'{v}: tier {o["tier"]} ok (recomputed {pct:.1f}th pct of {len(ref)} posts)')
    xx = np.exp(x)
    check(abs(o['views_vs_account_usual_x'] - xx) < 1e-6 and (o['tier'] != 'great' or xx > 1 or basis == 'views')
          and (o['tier'] != 'bad' or xx < 1), f'{v}: {xx:.2f}x usual agrees with tier {o["tier"]}')

# E. lockbox + selection independence
sel = pd.read_csv(ROOT/'results/study/selection.csv', dtype=str); ext = pd.read_csv(ROOT/'results/study/lockbox_ext.csv', dtype=str)
lock = set(sel[sel.split == 'lockbox'].video_id) | set(ext.video_id)
check(not (set(ids) & lock), 'no demo clip is a sealed lockbox clip')
import library_tiers as lt
src = inspect.getsource(lt.demo_picks)
check(not any(w in src for w in ('brain', 'library', 'pct[', 'npz', 'FEATURES')), 'tier picks never read brain data')
for t, ps in ([] if owner else demo['picks'].items()):  # owner picks: hand-picked, no order rule to check
    keys = [p['rank_key'] for p in ps]
    check(keys == sorted(keys) and len({p['deal_id'] for p in ps}) == len(ps), f'{t}: picks in hash order, one per deal')
pat = json.load(open(PUB/'library_patterns.json'))
f = next(r for r in pat['features'] if r['key'] == 'brain_above_typical')
check(f"{f['mean']['great']:.0f}%" in pat['interpreter_line'] and f"{100*f['coin_flip']['point']:.0f}%" in pat['interpreter_line'],
      'interpreter line quotes the computed numbers')

# F. learned.json headline numbers vs the stage-1 metrics file
L = json.load(open(PUB/'learned.json')); s1 = L['stage1']
check((s1['rho_A_metadata'], s1['rho_BE_brain_video']) == (0.335, 0.344), 'learned page stage-1 numbers match STAGE1_RESULT.md')
print('\nALL PASS' if ok else '\nSOME CHECKS FAILED')
sys.exit(0 if ok else 1)
