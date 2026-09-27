import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { chooseDemoClip, clipParam, DEMO_ROLE_LABELS, demoClips, momentStopMs, parseDemoMoment, parseDemoRole } from '../src/lib/demo.ts';
import { parseRealBundleIndex } from '../src/lib/real-analysis-index.ts';
import { DemoClipPicker, DemoShell, TOPBAR_PX } from '../src/components/demo-layout.ts';

// Synthetic fixture only: made-up IDs and timings.
const index = parseRealBundleIndex(JSON.parse(readFileSync(new URL('./fixtures/demo-index.synthetic.json', import.meta.url), 'utf8')));
const css = readFileSync(new URL('../src/components/demo-layout.css', import.meta.url), 'utf8');

test('demo roles map to presenter labels; anything else falls back to the video ID', () => {
  assert.deepEqual(DEMO_ROLE_LABELS, { spike: 'Big spike', flat: 'Flat stretch', fell_short: 'Fell short', beat_expectations: 'Beat expectations' });
  assert.equal(parseDemoRole('spike'), 'spike');
  assert.equal(parseDemoRole('toString'), null);
  assert.equal(parseDemoRole(3), null);
  const clips = demoClips(index);
  assert.deepEqual(clips.map(clip => [clip.video_id, clip.label]), [
    ['synthetic-spike', 'Big spike'], ['synthetic-flat', 'Flat stretch'], ['synthetic-beat', 'Beat expectations'],
    ['synthetic-fell', 'Fell short'], ['synthetic-plain', 'synthetic-plain'],
  ], 'index order, duplicate IDs dropped');
  assert.deepEqual(demoClips(null), []);
});

test('demo moments parse defensively', () => {
  const [spike, flat, beat, fell, plain] = demoClips(index);
  assert.deepEqual(spike.moment, { start_ms: 3000, end_ms: 7000, label: 'SYNTHETIC spike at 0:03' });
  assert.deepEqual(flat.moment, { start_ms: 12000, end_ms: 15000, label: 'Moment at 0:12' }, 'end clamped to the clip, default label');
  assert.equal(beat.moment, null, 'absent');
  assert.equal(fell.moment, null, 'end before start');
  assert.equal(plain.moment, null, 'not an object');
  assert.equal(parseDemoMoment({ start_ms: -1, end_ms: 2000 }), null);
  assert.equal(parseDemoMoment({ start_ms: 9000, end_ms: 12000 }, 8000), null, 'starts after the clip');
  assert.equal(parseDemoMoment({ start_ms: '1', end_ms: 2000 }), null);
  assert.equal(parseDemoMoment({ start_ms: 0, end_ms: Infinity }), null);
});

test('deep link picks the requested clip, otherwise the first', () => {
  const clips = demoClips(index);
  assert.equal(clipParam('?clip=synthetic-flat'), 'synthetic-flat');
  assert.equal(clipParam('?clip=../etc'), null);
  assert.equal(clipParam('?view=compare'), null);
  assert.equal(chooseDemoClip(clips, 'synthetic-fell')?.video_id, 'synthetic-fell');
  assert.equal(chooseDemoClip(clips, 'missing')?.video_id, 'synthetic-spike');
  assert.equal(chooseDemoClip(clips, null)?.video_id, 'synthetic-spike');
  assert.equal(chooseDemoClip([], null), null);
});

test('play this moment stops at the end, inside the playable clip', () => {
  const moment = { start_ms: 3000, end_ms: 7000, label: 'x' };
  assert.equal(momentStopMs(moment, 20000), 7000);
  assert.equal(momentStopMs({ ...moment, end_ms: 20000 }, 20000), 19940);
  assert.equal(momentStopMs(moment, 2000), 3000);
});

test('picker renders deep links in index order with the current clip marked', () => {
  const html = renderToStaticMarkup(createElement(DemoClipPicker, { clips: demoClips(index), current: 'synthetic-flat', busy: null, onChoose: () => undefined }));
  const labels = [...html.matchAll(/<a [^>]*>([^<]+)<\/a>/g)].map(match => match[1]);
  assert.deepEqual(labels, ['Big spike', 'Flat stretch', 'Beat expectations', 'Fell short', 'synthetic-plain']);
  assert.match(html, /href="\?clip=synthetic-spike"/);
  assert.match(html, /href="\?clip=synthetic-flat"[^>]*aria-current="page"/);
  assert.equal((html.match(/aria-current/g) ?? []).length, 1);
  assert.equal(renderToStaticMarkup(createElement(DemoClipPicker, { clips: [], current: null, busy: null, onChoose: () => undefined })), '');
});

test('stage holds only the clip, brain and strip; everything else is below the fold', () => {
  const slot = (name: string) => createElement('div', { 'data-slot': name });
  const html = renderToStaticMarkup(createElement(DemoShell, {
    topbar: slot('topbar'), video: slot('video'), brain: slot('brain'), strip: slot('strip'),
    below: [slot('scorecard'), slot('observed'), slot('timeline'), slot('moments'), slot('performance')].map((el, i) => createElement('div', { key: i }, el)),
    footer: slot('footer'),
  }));
  const stageStart = html.indexOf('<section class="demo-stage has-strip"');
  const stageEnd = html.indexOf('</section>', stageStart);
  assert.ok(stageStart > html.indexOf('data-slot="topbar"'), 'top bar comes first');
  const stage = html.slice(stageStart, stageEnd);
  assert.deepEqual([...stage.matchAll(/data-slot="(\w+)"/g)].map(match => match[1]), ['video', 'brain', 'strip']);
  assert.match(stage, /class="demo-stage-video"><div data-slot="video">/);
  assert.match(stage, /class="demo-stage-side"><div class="demo-stage-brain"><div data-slot="brain"><\/div><\/div><div class="demo-stage-strip">/);
  for (const name of ['scorecard', 'observed', 'timeline', 'moments', 'performance', 'footer']) {
    assert.ok(html.indexOf(`data-slot="${name}"`) > stageEnd, `${name} is below the stage`);
  }
  const noStrip = renderToStaticMarkup(createElement(DemoShell, { topbar: null, video: slot('video'), brain: slot('brain'), below: null, footer: null }));
  assert.match(noStrip, /class="demo-stage no-strip"/);
  assert.doesNotMatch(noStrip, /demo-stage-strip/);
});

test('stage CSS enforces one viewport: fixed heights, min-height 0, overflow hidden', () => {
  const rule = (selector: string) => {
    const match = css.match(new RegExp(`(?:^|\\n|\\})${selector.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\{([^}]*)\\}`));
    assert.ok(match, `missing rule ${selector}`);
    return match[1];
  };
  assert.match(css, new RegExp(`--demo-topbar:${TOPBAR_PX}px`));
  assert.match(rule('.demo-topbar'), /height:var\(--demo-topbar\)/);
  assert.match(rule('.demo-topbar'), /overflow:hidden/);
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
    const brainHeight = (stageHeight - 2 * pad - gap) * 3 / 5;
    const stripHeight = (stageHeight - 2 * pad - gap) * 2 / 5;
    assert.equal(TOPBAR_PX + stageHeight, height);
    assert.ok(sideWidth > clipWidth, `${width}×${height}: brain column wider than the clip`);
    assert.ok(brainHeight >= 450 && stripHeight >= 300, `${width}×${height}: brain ${brainHeight}px, strip ${stripHeight}px`);
  }
});
