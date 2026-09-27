import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import {
  compareModel, formatCount, formatDate, formatMultiplier, formatPercent, isLockboxBundle, parseExamples, parseObserved, siblingUrl,
} from '../src/lib/observed.ts';
import { parseRealBundleIndex } from '../src/lib/real-analysis-index.ts';
import CompareFrame from '../src/components/compare-frame.ts';
import { ObservedRow } from '../src/components/observed-row.ts';
import { demoClips, groupDemoClips } from '../src/lib/demo.ts';

// All fixtures here are synthetic ("synthetic": true): no real clip IDs or metrics.
const fixture = (name: string) => JSON.parse(readFileSync(new URL(`./fixtures/${name}`, import.meta.url), 'utf8'));
const observed = fixture('observed.synthetic.json');
const examples = fixture('examples.synthetic.json');
const entry = (video_id: string, extra: Record<string, unknown> = {}) => ({ video_id, analysis_id: `a-${video_id}`, path: `${video_id}/analysis.json`, duration_ms: 10000, status: 'complete', synthetic: false, n_channels: 7, n_moments: 3, has_words: true, n_warnings: 0, ...extra });
const index = parseRealBundleIndex(fixture('demo-index.synthetic.json'));

test('fixtures are synthetic', () => {
  assert.equal(observed.synthetic, true);
  assert.equal(examples.synthetic, true);
});

test('observed metrics parse from a valid file', () => {
  const parsed = parseObserved(observed, { lockbox: false });
  assert.ok(parsed);
  assert.equal(parsed.views, 12345);
  assert.equal(parsed.saves, null);
  assert.equal(parsed.shares, null, 'null stays null (shown as —), never 0');
  assert.equal(parsed.views_vs_account_usual_x, 0.5);
  assert.equal(parsed.caption, 'SYNTHETIC caption shown verbatim.');
  assert.equal(parsed.caption_null, 'SYNTHETIC null-result caption shown verbatim.');
  assert.equal(parsed.tier, 'bad');
  assert.equal(parsed.tier_label, 'Did badly');
  assert.equal(parsed.tier_plain, 'SYNTHETIC tier sentence shown verbatim.');
  assert.equal(parsed.views_pct_in_deal_platform, 12.5);
  assert.equal(parsed.n_ref_posts, 40);
  assert.equal('vs_expectation' in parsed, false);
  assert.equal(parsed.video_link, 'https://example.invalid/shorts/synthetic');
  assert.equal(parsed.upload_date, '2020-01-02');
});

test('a missing, wrong or HTML observed file renders nothing', () => {
  assert.equal(parseObserved(null, { lockbox: false }), null, 'a 404 is fetched as null');
  assert.equal(parseObserved('<!doctype html>', { lockbox: false }), null);
  assert.equal(parseObserved({ ...observed, schema: 'nvi.observed.v9' }, { lockbox: false }), null);
  assert.equal(parseObserved({ schema: 'nvi.observed.v0' }, { lockbox: false }), null, 'no numbers, nothing to show');
  assert.equal(parseObserved({ ...observed, video_link: 'javascript:alert(1)' }, { lockbox: false })?.video_link, null);
  const odd = parseObserved({ ...observed, tier: 'viral', tier_label: '', views_pct_in_deal_platform: 140, n_ref_posts: 'x' }, { lockbox: false })!;
  assert.deepEqual([odd.tier, odd.tier_label, odd.views_pct_in_deal_platform, odd.n_ref_posts], [null, '', null, null]);
  assert.equal(parseObserved({ ...observed, tier: 'great', tier_label: undefined }, { lockbox: false })?.tier_label, 'Did great', 'label falls back to the tier');
});

test('lockbox clips never show observed numbers, whatever the file says', () => {
  assert.equal(parseObserved(observed, { lockbox: true }), null);
  assert.equal(isLockboxBundle({ is_lockbox: true }), true);
  assert.equal(isLockboxBundle({ is_lockbox: false, performance: { model_status: 'not_trained', validated: false, clip_in_training: 'lockbox' } }), true);
  assert.equal(isLockboxBundle({ is_lockbox: false }, { performance: { clip_in_training: 'lockbox' } as never }), true);
  assert.equal(isLockboxBundle({ is_lockbox: false }, { performance: undefined }), false);
  assert.equal(isLockboxBundle(null, null), false);
  // Combined: a lockbox index entry blocks a perfectly valid file.
  assert.equal(parseObserved(observed, { lockbox: isLockboxBundle({ is_lockbox: true }) }), null);
});

test('observed formatting', () => {
  assert.equal(formatCount(12345), '12.3K');
  assert.equal(formatCount(321), '321');
  assert.equal(formatCount(null), '—');
  assert.equal(formatMultiplier(0.7076), '0.71×');
  assert.equal(formatMultiplier(34.4), '34×');
  assert.equal(formatPercent(1.1703), '1.17%');
  assert.equal(formatDate('2020-01-02'), '2 Jan 2020');
  assert.equal(siblingUrl('/demo/x/analysis.json', 'observed.json'), 'http://localhost/demo/x/observed.json');
  assert.equal(siblingUrl('/demo/index.json', 'examples.json'), 'http://localhost/demo/examples.json');
});

test('examples parse v1 default pairs (and v0 pairs), and require the caveat', () => {
  const parsed = parseExamples(examples);
  assert.ok(parsed);
  assert.deepEqual(parsed.defaultPair, { a: 'synthetic-great-1', b: 'synthetic-bad-1' });
  assert.equal(parsed.caveat, 'SYNTHETIC caveat: two clips, not evidence.');
  assert.equal(parseExamples({ ...examples, caveat: '' }), null);
  assert.equal(parseExamples({ ...examples, schema: 'other' }), null);
  assert.equal(parseExamples({ ...examples, default_pair: { a: 'bad id/..', b: 'x' } })?.defaultPair, null);
  assert.equal(parseExamples({ ...examples, default_pair: { a: 'same', b: 'same' } })?.defaultPair, null);
  assert.deepEqual(parseExamples({ schema: 'nvi.examples.v0', caveat: 'c', pairs: [{ fell_short: 'synthetic-bad-1', beat_expectations: 'synthetic-great-1' }] })?.defaultPair, { a: 'synthetic-great-1', b: 'synthetic-bad-1' });
  assert.equal(parseExamples(null), null);
});

test('compare model defaults to the example pair, honours requests and never repeats a clip', () => {
  const ex = parseExamples(examples);
  const ids = (model: ReturnType<typeof compareModel>) => model!.columns.map(c => [c.slot, c.title, c.entry.video_id]);
  assert.deepEqual(ids(compareModel(ex, index)), [['a', 'Did great · Deal Alpha', 'synthetic-great-1'], ['b', 'Did badly · Deal Alpha', 'synthetic-bad-1']]);
  assert.deepEqual(ids(compareModel(ex, index, { a: 'synthetic-typical-1', b: 'synthetic-great-2' })).map(c => c[2]), ['synthetic-typical-1', 'synthetic-great-2']);
  assert.deepEqual(ids(compareModel(ex, index, { a: 'synthetic-bad-1' })).map(c => c[2]), ['synthetic-bad-1', 'synthetic-bad-2'], 'default b taken by a: first other bad clip');
  assert.deepEqual(ids(compareModel(ex, index, { a: 'synthetic-great-1', b: 'synthetic-great-1' })).map(c => c[2]), ['synthetic-great-1', 'synthetic-bad-1'], 'same clip twice falls back');
  assert.deepEqual(ids(compareModel(ex, index, { a: 'missing', b: 'missing' })).map(c => c[2]), ['synthetic-great-1', 'synthetic-bad-1']);
  const noDefault = { ...ex!, defaultPair: null };
  assert.deepEqual(ids(compareModel(noDefault, index)).map(c => c[2]), ['synthetic-great-1', 'synthetic-bad-1'], 'first great vs first bad');
  const one = parseRealBundleIndex({ schema_version: 'nvi.analysis.v0.2', count: 1, synthetic: false, bundles: [entry('synthetic-great-1')] });
  assert.equal(compareModel(ex, one), null);
  assert.equal(compareModel(null, index), null);
});

test('compare view: caveat verbatim, tier column heads, dropdowns grouped by tier', () => {
  const model = compareModel(parseExamples(examples), index)!;
  const html = renderToStaticMarkup(createElement(CompareFrame, {
    model, groups: groupDemoClips(demoClips(index)), onChoose: () => undefined, onBack: () => undefined, onLearned: () => undefined,
    renderStage: column => createElement('p', { className: 'column-body' }, `body:${column.entry.video_id}`),
    renderBelow: column => createElement('p', { className: 'column-below' }, `below:${column.entry.video_id}`),
  }));
  assert.match(html, /<div class="compare-caveat" role="note"><p>SYNTHETIC caveat: two clips, not evidence\.<\/p>/, 'caveat verbatim, alone in its note');
  assert.ok(html.indexOf('SYNTHETIC caveat') < html.indexOf('class="compare-grid"'), 'caveat on top');
  assert.match(html, /Internal research view/);
  assert.match(html, /href="\?view=learned"[^>]*>See the results/);
  const a = html.indexOf('>Did great · Deal Alpha</h2>'), b = html.indexOf('>Did badly · Deal Alpha</h2>');
  assert.ok(a > 0 && b > a, 'great column first');
  assert.match(html, /<section class="compare-column is-great"/);
  assert.match(html, /<section class="compare-column is-bad"/);
  assert.equal((html.match(/<select class="compare-select"/g) ?? []).length, 2);
  assert.deepEqual([...html.slice(a, b).matchAll(/<optgroup label="([^"]+)"/g)].map(m => m[1]), ['Did great', 'Typical', 'Did badly', 'Other clips']);
  // Each dropdown disables the clip already in the other column.
  assert.match(html.slice(a, b), /<option value="synthetic-bad-1" disabled="">/);
  assert.match(html.slice(b), /<option value="synthetic-great-1" disabled="">/);
  assert.doesNotMatch(html, /Fell short|Beat expectations/);
  assert.ok(html.indexOf('body:synthetic-great-1') > a && html.indexOf('body:synthetic-bad-1') > b);
  const screenEnd = html.indexOf('class="compare-below"');
  assert.ok(html.indexOf('body:synthetic-bad-1') < screenEnd);
  assert.ok(html.indexOf('below:synthetic-great-1') > screenEnd && html.indexOf('below:synthetic-bad-1') > screenEnd);
  assert.match(html, /Research preview · non-commercial \(TRIBE CC-BY-NC\)/);
});

test('metrics row: tier chip, numbers, × usual and the tier sentence verbatim', () => {
  const parsed = parseObserved(observed, { lockbox: false })!;
  const html = renderToStaticMarkup(createElement(ObservedRow, { observed: parsed }));
  assert.match(html, /^<div class="observed-row" aria-label="Observed on platform"><div class="observed-row-line"><span class="tier-chip is-bad">Did badly<\/span>/);
  for (const text of ['12.3K</strong> views', '321</strong> likes', '12</strong> comments', '2.75%</strong> engagement', '0.50×</strong> account’s usual', 'observed on platform']) assert.ok(html.includes(text), text);
  assert.match(html, /<p class="observed-row-plain" title="SYNTHETIC tier sentence shown verbatim\.">SYNTHETIC tier sentence shown verbatim\.<\/p>/);
  const bare = renderToStaticMarkup(createElement(ObservedRow, { observed: { ...parsed, tier: null, tier_label: '', tier_plain: '', likes: null } }));
  assert.doesNotMatch(bare, /tier-chip|likes|observed-row-plain/);
});
