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

// All fixtures here are synthetic ("synthetic": true): no real clip IDs or metrics.
const fixture = (name: string) => JSON.parse(readFileSync(new URL(`./fixtures/${name}`, import.meta.url), 'utf8'));
const observed = fixture('observed.synthetic.json');
const examples = fixture('examples.synthetic.json');
const entry = (video_id: string, extra: Record<string, unknown> = {}) => ({ video_id, analysis_id: `a-${video_id}`, path: `${video_id}/analysis.json`, duration_ms: 10000, status: 'complete', synthetic: false, n_channels: 7, n_moments: 3, has_words: true, n_warnings: 0, ...extra });
const index = parseRealBundleIndex({ schema_version: 'nvi.analysis.v0.2', count: 3, synthetic: false, bundles: [entry('synthetic-beat', { internal_example: 'beat_expectations' }), entry('synthetic-other'), entry('synthetic-fell', { internal_example: 'fell_short' })] });

test('fixtures are synthetic', () => {
  assert.equal(observed.synthetic, true);
  assert.equal(examples.synthetic, true);
});

test('observed metrics parse from a valid file', () => {
  const parsed = parseObserved(observed, { lockbox: false });
  assert.ok(parsed);
  assert.equal(parsed.views, 12345);
  assert.equal(parsed.saves, null);
  assert.equal(parsed.views_vs_account_usual_x, 0.5);
  assert.equal(parsed.caption, 'SYNTHETIC caption shown verbatim.');
  assert.equal(parsed.caption_null, 'SYNTHETIC null-result caption shown verbatim.');
  assert.deepEqual(parsed.vs_expectation, { role: 'fell_short', plain: 'SYNTHETIC: fell short of expectation.' });
  assert.equal(parsed.video_link, 'https://example.invalid/shorts/synthetic');
  assert.equal(parsed.upload_date, '2020-01-02');
});

test('a missing, wrong or HTML observed file renders nothing', () => {
  assert.equal(parseObserved(null, { lockbox: false }), null, 'a 404 is fetched as null');
  assert.equal(parseObserved('<!doctype html>', { lockbox: false }), null);
  assert.equal(parseObserved({ ...observed, schema: 'nvi.observed.v9' }, { lockbox: false }), null);
  assert.equal(parseObserved({ schema: 'nvi.observed.v0' }, { lockbox: false }), null, 'no numbers, nothing to show');
  assert.equal(parseObserved({ ...observed, video_link: 'javascript:alert(1)' }, { lockbox: false })?.video_link, null);
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

test('examples parse, drop invalid pairs and require the caveat', () => {
  const parsed = parseExamples(examples);
  assert.ok(parsed);
  assert.deepEqual(parsed.pairs, [{ fell_short: 'synthetic-fell', beat_expectations: 'synthetic-beat', same: 'same synthetic group' }]);
  assert.equal(parsed.caveat, 'SYNTHETIC caveat: one illustrative pair, not evidence.');
  assert.equal(parseExamples({ ...examples, caveat: '' }), null);
  assert.equal(parseExamples({ ...examples, schema: 'other' }), null);
  assert.equal(parseExamples({ ...examples, pairs: [] }), null);
  assert.equal(parseExamples(null), null);
});

test('compare model puts fell short left and needs both clips in the index', () => {
  const model = compareModel(parseExamples(examples), index);
  assert.ok(model);
  assert.deepEqual(model.columns.map(c => [c.title, c.entry.video_id]), [['Fell short', 'synthetic-fell'], ['Beat expectations', 'synthetic-beat']]);
  const partial = parseRealBundleIndex({ schema_version: 'nvi.analysis.v0.2', count: 1, synthetic: false, bundles: [entry('synthetic-fell')] });
  assert.equal(compareModel(parseExamples(examples), partial), null);
  assert.equal(compareModel(null, index), null);
});

test('compare view renders both columns and the caveat', () => {
  const model = compareModel(parseExamples(examples), index)!;
  const html = renderToStaticMarkup(createElement(CompareFrame, {
    model, onBack: () => undefined, onLearned: () => undefined,
    renderColumn: column => createElement('p', { className: 'column-body' }, `body:${column.entry.video_id}`),
  }));
  assert.match(html, /SYNTHETIC caveat: one illustrative pair, not evidence\./);
  assert.match(html, /Internal research view/);
  assert.match(html, /href="\?view=learned"[^>]*>What the model learned/);
  const fell = html.indexOf('>Fell short</h2>'), beat = html.indexOf('>Beat expectations</h2>');
  assert.ok(fell > 0 && beat > fell, 'fell short column comes first');
  assert.ok(html.indexOf('body:synthetic-fell') > fell && html.indexOf('body:synthetic-beat') > beat);
  assert.equal((html.match(/class="compare-column /g) ?? []).length, 2);
  assert.match(html, /Research preview · non-commercial \(TRIBE CC-BY-NC\)/);
});
