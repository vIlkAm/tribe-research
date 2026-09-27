import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { ciBar, coinFlipText, featureGroups, formatFeatureValue, parsePatterns, verdictText } from '../src/lib/patterns.ts';
import { PatternsSection } from '../src/components/patterns-view.ts';
import { STRIP_LEGEND, STRIP_NOT_QUALITY, STRIP_PREDICTED, StripGuide } from '../src/components/strip-guide.ts';
import LibraryFrame, { LIBRARY_BADGE, loadLibraryRow, type LibraryRow } from '../src/components/library-frame.ts';
import { demoClips } from '../src/lib/demo.ts';
import { parseLibrary } from '../src/lib/library.ts';
import { parseObserved } from '../src/lib/observed.ts';
import { parseRealBundleIndex } from '../src/lib/real-analysis-index.ts';

// Synthetic fixtures only: made-up features, numbers, IDs and deals.
const fixture = (name: string) => JSON.parse(readFileSync(new URL(`./fixtures/${name}`, import.meta.url), 'utf8'));
const raw = fixture('patterns.synthetic.json');
const patterns = parsePatterns(raw)!;
const index = parseRealBundleIndex(fixture('demo-index.synthetic.json'));
const clips = demoClips(index);

test('patterns fixture is synthetic and parses defensively', () => {
  assert.equal(raw.synthetic, true);
  assert.ok(patterns);
  assert.equal(patterns.exploratory, true);
  assert.deepEqual([patterns.n_library, patterns.n_tiered, patterns.n_deals], [1000, 500, 7]);
  assert.deepEqual(Object.keys(patterns.tiers), ['great', 'typical', 'bad'], 'unknown tier dropped');
  assert.deepEqual(patterns.features.map(f => f.key), ['syn_len', 'syn_cuts', 'syn_above', 'SYNTHETIC named only'], 'invalid features dropped; name accepted');
  const named = patterns.features[3];
  assert.deepEqual([named.label_plain, named.diff, named.coin_flip, named.verdict, named.mean.great], ['SYNTHETIC named only', null, null, null, null]);
  assert.equal(patterns.features[1].verdict, 'weak_tendency');
  assert.equal(patterns.features[2].verdict, 'no_reliable_difference');
  assert.equal(parsePatterns({ ...raw, features: [{ ...raw.features[0], verdict: 'reliable' }] })?.features[0].verdict, 'weak_tendency', 'legacy value shown as a weak tendency');
  assert.equal(patterns.features[2].coin_flip, null);
  assert.deepEqual(patterns.features[1].coin_flip, { point: 0.62, lo: 0.55, hi: 0.69 });
  assert.deepEqual(patterns.caveats, ['SYNTHETIC caveat one.', 'SYNTHETIC caveat two.']);
  assert.equal(parsePatterns({ ...raw, exploratory: undefined })?.exploratory, true, 'exploratory unless the file says otherwise');
  assert.equal(parsePatterns({ ...raw, schema: 'nvi.patterns.v9' }), null);
  assert.equal(parsePatterns({ schema: 'nvi.patterns.v0' }), null, 'nothing to show');
  assert.equal(parsePatterns('<!doctype html>'), null);
  assert.equal(parsePatterns(null), null);
});

test('weak tendencies come first: within a group, and groups holding one first', () => {
  assert.deepEqual(featureGroups(patterns).map(g => [g.title, g.features.map(f => f.key)]), [
    ['Editing', ['syn_cuts', 'syn_len']],
    ['Predicted brain response', ['syn_above', 'SYNTHETIC named only']],
  ]);
  // Only the brain group has a tendency (as in the staged data): it moves above editing.
  const brainOnly = parsePatterns({ ...raw, features: raw.features.map((f: { key?: string }) => f.key === 'syn_cuts' ? { ...f, verdict: 'no_reliable_difference' } : f.key === 'syn_above' ? { ...f, verdict: 'weak_tendency' } : f) })!;
  assert.deepEqual(featureGroups(brainOnly).map(g => [g.title, g.features[0].key]), [['Predicted brain response', 'syn_above'], ['Editing', 'syn_len']]);
});

test('interval bar is symmetric around zero; values format with units', () => {
  const bar = ciBar({ point: 4, lo: 1, hi: 7 });
  assert.equal(bar.zero, 50);
  assert.ok(bar.lo > 50 && bar.point > bar.lo && bar.hi > bar.point && bar.hi <= 100);
  const straddle = ciBar({ point: -4, lo: -8, hi: 1 });
  assert.ok(straddle.lo < 50 && straddle.hi > 50, 'interval crossing 0 spans the zero mark');
  assert.deepEqual(ciBar({ point: 0, lo: 0, hi: 0 }), { lo: 50, hi: 50, point: 50, zero: 50 });
  assert.equal(formatFeatureValue(20, 's'), '20.0 s');
  assert.equal(formatFeatureValue(51, '% of seconds'), '51.0 % of seconds');
  assert.equal(formatFeatureValue(4.5, '%'), '4.50%');
  assert.equal(formatFeatureValue(null, 's'), '—');
  assert.equal(coinFlipText({ point: 0.62, lo: 0.55, hi: 0.69 }), '62% (55–69%)');
  assert.equal(verdictText('weak_tendency'), 'Weak tendency (exploratory)');
  assert.equal(verdictText('no_reliable_difference'), 'No clear difference');
});

test('patterns section: neutral chip for no difference, caveats verbatim', () => {
  const html = renderToStaticMarkup(createElement(PatternsSection, { patterns }));
  assert.match(html, /What great clips have in common \(and what they don’t\)/);
  assert.match(html, /<caption>Editing<\/caption>/);
  assert.ok(html.indexOf('SYNTHETIC cuts') < html.indexOf('SYNTHETIC length'), 'tendency first');
  assert.match(html, /<span class="pattern-verdict is-tendency">Weak tendency \(exploratory\)<\/span>/);
  assert.equal((html.match(/<span class="pattern-verdict is-neutral">No clear difference<\/span>/g) ?? []).length, 2);
  assert.doesNotMatch(html, /reliable|significant/i, 'no "reliable" or "significant" wording of our own');
  assert.match(html, /class="ci-zero" style="left:50%"/);
  assert.match(html, /62% \(55–69%\)/);
  assert.match(html, /SYNTHETIC weak tendency\./);
  assert.match(html, /<div class="library-caveats"><h3>Caveats<\/h3><ul><li>SYNTHETIC caveat one\.<\/li><li>SYNTHETIC caveat two\.<\/li><\/ul><\/div>/);
});

test('strip guide: fixed sentences plus the interpreter line verbatim', () => {
  assert.equal(STRIP_LEGEND, '50 = a typical clip of this length at this second · shaded = middle half of your clips · above = stronger predicted brain response');
  const html = renderToStaticMarkup(createElement(StripGuide, { interpreterLine: patterns.interpreter_line, onClose: () => undefined }));
  assert.match(html, /role="dialog"/);
  assert.match(html, /How to read this/);
  for (const text of [STRIP_NOT_QUALITY, STRIP_PREDICTED, 'SYNTHETIC interpreter line shown verbatim.']) assert.ok(html.includes(text), text);
  assert.equal(STRIP_NOT_QUALITY, 'It is not a quality score or a views forecast.');
  assert.equal(STRIP_PREDICTED, 'Predicted response of an average viewer’s brain (TRIBE v2), not measured.');
  const without = renderToStaticMarkup(createElement(StripGuide, {}));
  assert.doesNotMatch(without, /SYNTHETIC interpreter/);
  assert.ok(without.includes(STRIP_NOT_QUALITY));
});

test('library rows: a lockbox clip never requests observed.json', async () => {
  const requested: string[] = [];
  const observed = fixture('observed.synthetic.json');
  const library = { ...fixture('library.synthetic.json') };
  const fetchJson = async (url: string) => {
    requested.push(new URL(url).pathname);
    if (url.endsWith('/observed.json')) return observed;
    if (url.endsWith('/library.json')) return { ...library, video_id: url.split('/').at(-2) };
    return null;
  };
  const open = clips.find(c => c.video_id === 'synthetic-great-1')!;
  const sealed = clips.find(c => c.video_id === 'synthetic-bad-2')!;
  const row = await loadLibraryRow('/demo/index.json', open, fetchJson);
  assert.equal(row.observed?.views, 12345);
  assert.equal(row.library?.video_id, 'synthetic-great-1');
  const lockboxRow = await loadLibraryRow('/demo/index.json', sealed, fetchJson);
  assert.equal(lockboxRow.observed, null);
  assert.ok(lockboxRow.library);
  assert.deepEqual(requested.sort(), ['/demo/synthetic-bad-2/library.json', '/demo/synthetic-great-1/library.json', '/demo/synthetic-great-1/observed.json']);
});

test('library page: size, badge, clip table and patterns', () => {
  const observed = parseObserved(fixture('observed.synthetic.json'), { lockbox: false });
  const library = parseLibrary(fixture('library.synthetic.json'), 'synthetic0000clip');
  const rows: LibraryRow[] = clips.map((clip, i) => ({ clip, observed: i === 0 ? observed : null, library: i === 0 ? library : null }));
  const html = renderToStaticMarkup(createElement(LibraryFrame, { patterns, rows, onBack: () => undefined, onOpenClip: () => undefined }));
  assert.equal(LIBRARY_BADGE, 'Internal research view · exploratory');
  assert.match(html, /<span class="internal-badge">Internal research view · exploratory<\/span>/);
  assert.match(html, /<strong>1,000 clips from 7 deals<\/strong>/);
  assert.match(html, /SYNTHETIC outcome sentence\./);
  assert.match(html, /SYNTHETIC great rule/);
  assert.equal((html.match(/<tr class="library-clip-row"/g) ?? []).length, clips.length);
  assert.match(html, /href="\?clip=synthetic-great-1"[^>]*>Deal Alpha<\/a>/);
  const first = html.slice(html.indexOf('<tr class="library-clip-row"'), html.indexOf('</tr>', html.indexOf('<tr class="library-clip-row"')));
  for (const text of ['tier-chip is-great', 'YouTube', '21 s', '12.3K', '0.50×', '2.75%', 'verdict-chip strong']) assert.ok(first.includes(text), text);
  assert.ok(html.indexOf('The demo clips') < html.indexOf('What great clips have in common'));
  assert.match(html, /SYNTHETIC caveat two\./);
  assert.match(html, /non-commercial \(TRIBE CC-BY-NC\)/);
});
