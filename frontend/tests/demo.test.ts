import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { chooseDemoClip, clipHeading, clipParam, demoClips, groupDemoClips, lengthLabel, parseDemoMoment, parseTier, TIER_LABELS, momentStopMs } from '../src/lib/demo.ts';
import { parseRealBundleIndex } from '../src/lib/real-analysis-index.ts';
import { DemoClipPicker, DemoShell, TOPBAR_PX } from '../src/components/demo-layout.ts';

// Synthetic fixture only: made-up IDs, deals and timings.
const index = parseRealBundleIndex(JSON.parse(readFileSync(new URL('./fixtures/demo-index.synthetic.json', import.meta.url), 'utf8')));
const css = readFileSync(new URL('../src/components/demo-layout.css', import.meta.url), 'utf8');

test('tiers parse strictly; legacy roles fall into no tier', () => {
  assert.deepEqual(TIER_LABELS, { great: 'Did great', typical: 'Typical', bad: 'Did badly' });
  assert.equal(parseTier('great'), 'great');
  assert.equal(parseTier('spike'), null);
  assert.equal(parseTier('toString'), null);
  assert.equal(parseTier(3), null);
});

test('clip labels read "<deal> · <platform> · <length>" in index order', () => {
  const clips = demoClips(index);
  assert.deepEqual(clips.map(clip => [clip.video_id, clip.tier, clip.label]), [
    ['synthetic-great-1', 'great', 'Deal Alpha · YouTube · 21 s'],
    ['synthetic-great-2', 'great', 'Deal Beta · TikTok · 15 s'],
    ['synthetic-typical-1', 'typical', 'Deal Gamma · Instagram · 1:05'],
    ['synthetic-bad-1', 'bad', 'Deal Alpha · TikTok · 18 s'],
    ['synthetic-bad-2', 'bad', 'Deal Delta · YouTube · 18 s'],
    ['synthetic-legacy', null, 'synthetic-legacy · 10 s'],
    ['synthetic-typical-2', 'typical', 'synthetic-typical-2 · YouTube · 9 s'],
  ], 'duplicate IDs dropped; a missing or non-string deal falls back to the video ID');
  assert.equal(clipHeading(clips[0]), 'Did great · Deal Alpha');
  assert.equal(clipHeading(clips[5]), 'synthetic-legacy');
  assert.equal(lengthLabel(400), '1 s');
  assert.deepEqual(demoClips(null), []);
});

test('clips group by tier in fixed order: great, typical, bad, then others', () => {
  const groups = groupDemoClips(demoClips(index));
  assert.deepEqual(groups.map(group => [group.label, group.clips.map(clip => clip.video_id)]), [
    ['Did great', ['synthetic-great-1', 'synthetic-great-2']],
    ['Typical', ['synthetic-typical-1', 'synthetic-typical-2']],
    ['Did badly', ['synthetic-bad-1', 'synthetic-bad-2']],
    ['Other clips', ['synthetic-legacy']],
  ]);
  assert.deepEqual(groupDemoClips(demoClips(index).filter(clip => clip.tier === 'bad')).map(group => group.label), ['Did badly'], 'empty groups dropped');
});

test('demo moments stay supported and parse defensively', () => {
  const [great1, great2, typical1, bad1, , legacy] = demoClips(index);
  assert.deepEqual(great1.moment, { start_ms: 3000, end_ms: 7000, label: 'SYNTHETIC moment at 0:03' });
  assert.deepEqual(great2.moment, { start_ms: 12000, end_ms: 15000, label: 'Moment at 0:12' }, 'end clamped to the clip, default label');
  assert.equal(typical1.moment, null, 'absent');
  assert.equal(bad1.moment, null, 'end before start');
  assert.equal(legacy.moment, null, 'not an object');
  assert.equal(parseDemoMoment({ start_ms: -1, end_ms: 2000 }), null);
  assert.equal(parseDemoMoment({ start_ms: 9000, end_ms: 12000 }, 8000), null, 'starts after the clip');
  assert.equal(parseDemoMoment({ start_ms: '1', end_ms: 2000 }), null);
  assert.equal(parseDemoMoment({ start_ms: 0, end_ms: Infinity }), null);
});

test('deep link picks the requested clip, otherwise the first', () => {
  const clips = demoClips(index);
  assert.equal(clipParam('?clip=synthetic-bad-1'), 'synthetic-bad-1');
  assert.equal(clipParam('?a=synthetic-bad-1', 'a'), 'synthetic-bad-1');
  assert.equal(clipParam('?clip=../etc'), null);
  assert.equal(clipParam('?view=compare'), null);
  assert.equal(chooseDemoClip(clips, 'synthetic-bad-1')?.video_id, 'synthetic-bad-1');
  assert.equal(chooseDemoClip(clips, 'missing')?.video_id, 'synthetic-great-1');
  assert.equal(chooseDemoClip(clips, null)?.video_id, 'synthetic-great-1');
  assert.equal(chooseDemoClip([], null), null);
});

test('play this moment stops at the end, inside the playable clip', () => {
  const moment = { start_ms: 3000, end_ms: 7000, label: 'x' };
  assert.equal(momentStopMs(moment, 20000), 7000);
  assert.equal(momentStopMs({ ...moment, end_ms: 20000 }, 20000), 19940);
  assert.equal(momentStopMs(moment, 2000), 3000);
});

test('picker groups deep links by tier with the current clip marked', () => {
  const html = renderToStaticMarkup(createElement(DemoClipPicker, { clips: demoClips(index), current: 'synthetic-bad-1', busy: null, onChoose: () => undefined }));
  assert.match(html, /^<details class="demo-picker"><summary[^>]*><span class="tier-chip is-bad">Did badly<\/span><span class="demo-picker-current">Deal Alpha · TikTok · 18 s<\/span><\/summary>/);
  const groups = [...html.matchAll(/<span class="demo-picker-group-label">([^<]+)<\/span>/g)].map(match => match[1]);
  assert.deepEqual(groups, ['Did great', 'Typical', 'Did badly', 'Other clips']);
  const labels = [...html.matchAll(/<a [^>]*>([^<]+)<\/a>/g)].map(match => match[1]);
  assert.deepEqual(labels, [
    'Deal Alpha · YouTube · 21 s', 'Deal Beta · TikTok · 15 s',
    'Deal Gamma · Instagram · 1:05', 'synthetic-typical-2 · YouTube · 9 s',
    'Deal Alpha · TikTok · 18 s', 'Deal Delta · YouTube · 18 s', 'synthetic-legacy · 10 s',
  ]);
  assert.match(html, /href="\?clip=synthetic-great-1"/);
  assert.match(html, /href="\?clip=synthetic-bad-1"[^>]*aria-current="page"/);
  assert.equal((html.match(/aria-current/g) ?? []).length, 1);
  assert.doesNotMatch(html, /spike|flat|Beat expectations|Fell short/i);
  const none = renderToStaticMarkup(createElement(DemoClipPicker, { clips: demoClips(index), current: null, busy: 'synthetic-great-2', onChoose: () => undefined }));
  assert.match(none, /Choose a clip/);
  assert.match(none, /Opening…/);
  assert.match(none, /class="demo-picker-item is-busy"/);
  assert.equal(renderToStaticMarkup(createElement(DemoClipPicker, { clips: [], current: null, busy: null, onChoose: () => undefined })), '');
});

test('stage holds only the clip, metrics, brain and strip; everything else is below the fold', () => {
  const slot = (name: string) => createElement('div', { 'data-slot': name });
  const html = renderToStaticMarkup(createElement(DemoShell, {
    topbar: slot('topbar'), video: slot('video'), metrics: slot('metrics'), brain: slot('brain'), strip: slot('strip'),
    below: [slot('scorecard'), slot('observed'), slot('timeline'), slot('moments'), slot('performance')].map((el, i) => createElement('div', { key: i }, el)),
    footer: slot('footer'),
  }));
  const stageStart = html.indexOf('<section class="demo-stage has-strip has-metrics"');
  const stageEnd = html.indexOf('</section>', stageStart);
  assert.ok(stageStart > html.indexOf('data-slot="topbar"'), 'top bar comes first');
  const stage = html.slice(stageStart, stageEnd);
  assert.deepEqual([...stage.matchAll(/data-slot="(\w+)"/g)].map(match => match[1]), ['video', 'metrics', 'brain', 'strip']);
  assert.match(stage, /class="demo-stage-video"><div data-slot="video">/);
  assert.match(stage, /class="demo-stage-side"><div class="demo-stage-metrics"><div data-slot="metrics"><\/div><\/div><div class="demo-stage-brain"><div data-slot="brain"><\/div><\/div><div class="demo-stage-strip">/);
  for (const name of ['scorecard', 'observed', 'timeline', 'moments', 'performance', 'footer']) {
    assert.ok(html.indexOf(`data-slot="${name}"`) > stageEnd, `${name} is below the stage`);
  }
  const noStrip = renderToStaticMarkup(createElement(DemoShell, { topbar: null, video: slot('video'), brain: slot('brain'), below: null, footer: null }));
  assert.match(noStrip, /class="demo-stage no-strip no-metrics"/);
  assert.doesNotMatch(noStrip, /demo-stage-strip|demo-stage-metrics/);
});

test('stage CSS enforces one viewport: fixed heights, min-height 0, overflow hidden', () => {
  const rule = (selector: string) => {
    const match = css.match(new RegExp(`(?:^|\\n|\\})${selector.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\{([^}]*)\\}`));
    assert.ok(match, `missing rule ${selector}`);
    return match[1];
  };
  assert.match(css, new RegExp(`--demo-topbar:${TOPBAR_PX}px`));
  assert.match(rule('.demo-topbar'), /height:var\(--demo-topbar\)/);
  // The top bar stays one line; it is not clipped so the clip menu can drop below it.
  assert.match(rule('.demo-topbar'), /white-space:nowrap/);
  assert.match(rule('.demo-picker-menu'), /position:absolute/);
  assert.match(rule('.demo-picker-menu'), /max-height:calc\(100vh - var\(--demo-topbar\) - 24px\);overflow:auto/);
  const stage = rule('.demo-stage');
  assert.match(stage, /height:calc\(100vh - var\(--demo-topbar\)\)/);
  assert.match(stage, /min-height:0/);
  assert.match(stage, /overflow:hidden/);
  assert.match(stage, /box-sizing:border-box/);
  // Clip column is height-constrained at 9:16; everything in the grid may shrink.
  assert.match(stage, /grid-template-columns:minmax\(0,calc\(\(100vh - var\(--demo-topbar\) - 2 \* var\(--stage-pad\)\) \* 9 \/ 16\)\) minmax\(0,1fr\)/);
  assert.match(stage, /grid-template-rows:minmax\(0,1fr\)/);
  assert.match(rule('.demo-stage>*,.demo-stage-side>*'), /min-height:0;min-width:0;overflow:hidden/);
  assert.match(rule('.demo-stage-side'), /grid-template-rows:minmax\(0,3fr\) minmax\(0,2fr\)/);
  assert.match(rule('.demo-stage.has-metrics .demo-stage-side'), /grid-template-rows:var\(--metrics-h\) minmax\(0,3fr\) minmax\(0,2fr\)/);
  assert.match(css, /--metrics-h:42px/);
  const row = rule('.observed-row');
  assert.match(row, /height:var\(--metrics-h\)/);
  assert.match(row, /white-space:nowrap;overflow:hidden/);
  assert.match(rule('.observed-row-plain'), /overflow:hidden;text-overflow:ellipsis/);
  // The strip guide opens inside the strip, never adding page height.
  assert.match(rule('.response-strip'), /position:relative/);
  assert.match(rule('.strip-guide'), /position:absolute/);
  assert.match(rule('.strip-guide'), /max-height:calc\(100% - 44px\);overflow:auto/);
  assert.match(rule('.demo-stage .brain-stage'), /height:100%;min-height:0/);
  assert.match(rule('.stage-video-frame video'), /object-fit:contain/);
  assert.match(rule('.stage-video-frame'), /min-height:0/);
  assert.match(rule('.response-strip.is-compact .strip-chart'), /min-height:0/);
  assert.match(rule('.compare-screen'), /height:100vh;display:flex;flex-direction:column;overflow:hidden/);
  assert.match(rule('.compare-stage'), /min-height:0;overflow:hidden/);
});

test('the stage fits both presentation screens', () => {
  // Same arithmetic as the CSS: top bar + stage = viewport, clip column at 9:16 of the stage height.
  for (const [width, height] of [[1536, 864], [1920, 1080]]) {
    const pad = 12, gap = 12;
    const stageHeight = height - TOPBAR_PX;
    const clipWidth = (stageHeight - 2 * pad) * 9 / 16;
    const sideWidth = width - 2 * pad - gap - clipWidth;
    const metrics = 42;
    const rest = stageHeight - 2 * pad - metrics - 2 * gap;
    const brainHeight = rest * 3 / 5;
    const stripHeight = rest * 2 / 5;
    assert.equal(TOPBAR_PX + stageHeight, height);
    assert.equal(metrics + 2 * gap + brainHeight + stripHeight, stageHeight - 2 * pad, 'metrics row takes its height from brain and strip');
    assert.ok(sideWidth > clipWidth, `${width}×${height}: brain column wider than the clip`);
    assert.ok(sideWidth >= 1000, `${width}×${height}: metrics row has ${sideWidth}px for one line`);
    assert.ok(brainHeight >= 430 && stripHeight >= 285, `${width}×${height}: brain ${brainHeight}px, strip ${stripHeight}px`);
  }
});
