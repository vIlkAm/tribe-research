import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
  bandPath, clockTime, momentText, hasLibraryContent, libraryUrlFor, parseLearned, parseLibrary, percentileAt,
  percentileReadout, referenceLine, seriesPaths, verdictLabel,
} from '../src/lib/library.ts';

// Both fixtures are synthetic (marked "synthetic": true); they contain no real clip data.
const fixture = (name: string) => JSON.parse(readFileSync(new URL(`./fixtures/${name}`, import.meta.url), 'utf8'));
const library = fixture('library.synthetic.json');
const learned = fixture('learned.synthetic.json');

test('fixtures are marked synthetic', () => {
  assert.equal(library.synthetic, true);
  assert.equal(learned.synthetic, true);
});

test('library parses, orders scores and caps moments and summary', () => {
  const parsed = parseLibrary(library, 'synthetic0000clip');
  assert.ok(parsed);
  assert.deepEqual(parsed.scores.map(s => s.key), ['hook', 'hold', 'peak', 'dead_zones', 'finish']);
  assert.equal(parsed.scores[0].verdict, 'strong');
  assert.deepEqual(parsed.scores[0].window_ms, [0, 3000]);
  assert.equal(parsed.channels.length, 2, 'channel without label_plain is dropped');
  assert.equal(parsed.moments.length, 3);
  assert.equal(parsed.summary.length, 3);
  assert.equal(parsed.index?.percentile[12], null);
  assert.match(parsed.caveat, /SYNTHETIC/);
});

test('library is ignored for another clip, wrong schema, HTML fallback or empty content', () => {
  assert.equal(parseLibrary(library, 'another-clip'), null);
  assert.equal(parseLibrary({ ...library, schema_version: 'nvi.library.v9' }), null);
  assert.equal(parseLibrary('<!doctype html>'), null);
  assert.equal(parseLibrary(null), null);
  assert.equal(parseLibrary([]), null);
  assert.equal(parseLibrary({ schema_version: 'nvi.library.v0', video_id: 'x' }), null);
  assert.equal(hasLibraryContent(null), false);
});

test('library tolerates missing or malformed optional parts', () => {
  const partial = parseLibrary({ schema_version: 'nvi.library.v0', video_id: 'x', summary: ['Only a summary.', 3, ''], scores: 'nope', index: { hz: 1, percentile: [null, null] } });
  assert.ok(partial);
  assert.deepEqual(partial.summary, ['Only a summary.']);
  assert.equal(partial.index, null, 'an index without any percentile is dropped');
  assert.deepEqual(partial.scores, []);
  assert.equal(partial.caveat, '');
  const clamped = parseLibrary({ schema_version: 'nvi.library.v0', video_id: 'x', index: { hz: 1, percentile: [150, -3, 'a'] } });
  assert.deepEqual(clamped?.index?.percentile, [100, 0, null]);
});

test('playhead readout uses the second under the playhead', () => {
  const parsed = parseLibrary(library)!;
  const index = parsed.index!;
  assert.equal(percentileAt(index, 0), index.percentile[0]);
  assert.equal(percentileAt(index, 7999), index.percentile[7]);
  assert.equal(percentileAt(index, 12500), null);
  assert.equal(percentileAt(index, 999_999), index.percentile[index.percentile.length - 1]);
  assert.equal(percentileAt(index, -1), null);
  assert.equal(percentileReadout(78.4, 7200), 'Stronger than 78% of similar clips at 0:07');
  assert.equal(percentileReadout(null, 65000), 'No library comparison at 1:05');
  assert.equal(clockTime(0), '0:00');
  assert.equal(momentText('0:25–0:28: predicted response dips.'), 'Predicted response dips.');
  assert.equal(momentText('Plain copy.'), 'Plain copy.');
});

test('labels and reference copy', () => {
  assert.equal(verdictLabel('strong'), 'Strong');
  assert.equal(verdictLabel('typical'), 'Typical');
  assert.equal(verdictLabel('weak'), 'Weak');
  assert.equal(verdictLabel(null), '');
  assert.equal(referenceLine(parseLibrary(library)!.reference), 'Compared with 120 synthetic clips of similar length (this clip left out).');
  assert.equal(referenceLine(null), '');
});

test('library URL sits next to analysis.json on the same origin', () => {
  assert.equal(libraryUrlFor('https://h.test/demo-stage1/abc/analysis.json'), 'https://h.test/demo-stage1/abc/library.json');
  assert.equal(libraryUrlFor('/demo-stage1/abc/analysis.json'), 'http://localhost/demo-stage1/abc/library.json');
});

test('series paths break at gaps and close areas to the midline', () => {
  const segments = seriesPaths([50, 100, null, 0], 300, 100);
  assert.equal(segments.length, 2);
  assert.equal(segments[0].line, 'M0.0 50.0L100.0 0.0');
  assert.match(segments[0].area, /L100\.0 50\.0L0\.0 50\.0Z$/);
  assert.equal(seriesPaths([], 10, 10).length, 0);
  assert.equal(bandPath([0, 0, null], [1, 1, 1], 10, 10, [0, 1]).endsWith('Z'), true);
  assert.equal(bandPath([0], [1], 10, 10, [0, 1]), '');
});

test('learned summary parses statements and the good-vs-bad comparison', () => {
  const parsed = parseLearned(learned);
  assert.ok(parsed);
  assert.equal(parsed.statements.length, 3);
  assert.deepEqual(parsed.statements[0], { text: 'Synthetic: metadata drives ranking.', value: '0.34' });
  assert.equal(parsed.statements[2].value, '');
  assert.equal(parsed.decision, 'no-GO (synthetic)');
  const gvb = parsed.good_vs_bad!;
  assert.equal(gvb.seconds.length, 30);
  assert.equal(gvb.top.mean.length, 30);
  assert.deepEqual(gvb.channels.map(c => c.key), ['attention'], 'a channel with no bottom values is dropped');
  assert.equal(gvb.result_plain, 'Synthetic: the two groups overlap.');
});

test('learned summary tolerates other shapes and rejects junk', () => {
  assert.equal(parseLearned(null), null);
  assert.equal(parseLearned('<!doctype html>'), null);
  assert.equal(parseLearned({ schema_version: 'nvi.learned.v1', stage1: ['x'] }), null);
  assert.equal(parseLearned({}), null);
  assert.deepEqual(parseLearned({ stage1: ['Line one.', ''] })?.statements, [{ text: 'Line one.', value: '' }]);
  assert.deepEqual(parseLearned({ stage1: { ranking: 'Metadata ranks clips.', rho: 0.3 } })?.statements, [{ text: 'Metadata ranks clips.', value: '' }]);
  assert.equal(parseLearned({ good_vs_bad: { seconds: [0, 1], top: { mean: [1, 2] } } }), null);
});

test('learned summary accepts the producer layout: curves under index, channel curves as {mean, lo, hi}', () => {
  // Synthetic values in the shape tools/build_library_profile.py writes.
  const curve = (offset: number) => ({ mean: [offset, offset + 1, offset], lo: [offset - 1, offset, offset - 1], hi: [offset + 1, offset + 2, offset + 1] });
  const parsed = parseLearned({
    schema_version: 'nvi.learned.v0', caveat: 'Synthetic caveat.',
    stage1: { result: 'no-GO', lines: ['Line A.', 'Line B.'], primary_target: 'log_interactions_rate', go_rule: 'BE - A >= 0.02', rho_A_metadata: 0.335, BE_minus_E_content: { point: 0.0004, ci95: [-0.01, 0.01] }, reach_E_minus_A_content: { point: 0.073 } },
    good_vs_bad: { seconds: [0, 1, 2], n_top: 3, n_bottom: 3, index: { top: curve(0), bottom: curve(0.1), diff_top_minus_bottom: curve(0) }, channels: { attention: { label_plain: 'Grabs attention', top: curve(0), bottom: curve(1) } }, result_plain: 'Same.' },
  });
  assert.ok(parsed);
  assert.deepEqual(parsed.statements.map(s => s.text), ['Line A.', 'Line B.']);
  assert.equal(parsed.decision, 'no-GO');
  assert.equal(parsed.caveat, 'Synthetic caveat.');
  assert.deepEqual(parsed.key_numbers.map(k => k.value), ['≈ 0.34', '≈ 0.00', '+0.07']);
  const gvb = parsed.good_vs_bad!;
  assert.deepEqual(gvb.top.mean, [0, 1, 0]);
  assert.deepEqual(gvb.bottom.hi, [1.1, 2.1, 1.1]);
  assert.deepEqual(gvb.channels, [{ key: 'attention', label: 'Grabs attention', top: [0, 1, 0], bottom: [1, 2, 1] }]);
});

test('learned object fallback ignores non-prose fields', () => {
  assert.deepEqual(parseLearned({ stage1: { result: 'no-GO', primary_target: 'log_interactions_rate', go_rule: 'a rule with spaces', note: 'A prose line.' } })?.statements, [{ text: 'A prose line.', value: '' }]);
});
