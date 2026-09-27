import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { assertAnalysis, formatTime, frameAt, isGap, sampleAt, stepPath } from '../src/lib/analysis.ts';
import type { Analysis, Channel } from '../src/data/analysis.types.ts';
import { BRAIN_CHANNEL_KEYS, BRAIN_DISPLAY_NOTE, supportsBrain3d } from '../src/lib/demo-compat.ts';

const sample = JSON.parse(readFileSync(new URL('../public/data/cd20b16879d630c4/analysis.json', import.meta.url), 'utf8')) as Analysis;

test('sample boundaries use sample-and-hold without filling null values', () => {
  const channel = { hz: 2, values: [1, null, -2, 3] } as Channel;
  assert.equal(sampleAt(channel, 499, 2000), 1);
  assert.equal(sampleAt(channel, 500, 2000), null);
  assert.equal(sampleAt(channel, 999, 2000), null);
  assert.equal(sampleAt(channel, 1000, 2000), -2);
  assert.equal(sampleAt(channel, 2000, 2000), null);
  assert.equal(sampleAt(channel, -1, 2000), null);
});

test('chart breaks the path at a missing sample, preserving steps', () => {
  const path = stepPath([1, null, -1], 2, 1500, 300, 100);
  assert.equal((path.match(/M/g) || []).length, 2);
  assert.match(path, /H100\.00M200\.00/);
  assert.doesNotMatch(path, /[CQ]/);
});

test('atlas lookup respects end-exclusive timestamps and gaps', () => {
  const sprite = { ...sample.assets.brain_map!, frame_start_ms: [0, 2000], frame_end_ms: [1000, 3000] };
  assert.equal(frameAt(sprite, 999), 0);
  assert.equal(frameAt(sprite, 1000), -1);
  assert.equal(frameAt(sprite, 2000), 1);
  assert.equal(frameAt(sprite, 3000), -1);
});

test('gaps are end-exclusive and clip end has no prediction', () => {
  const analysis = { ...sample, timing: { ...sample.timing, gaps_ms: [[1000, 2000] as [number, number]] } };
  assert.equal(isGap(analysis, 1000), true);
  assert.equal(isGap(analysis, 2000), false);
  assert.equal(isGap(analysis, sample.duration_ms), true);
});

test('time display preserves precision without extra zeroes', () => {
  assert.equal(formatTime(4200), '00:04.2');
  assert.equal(formatTime(26100), '00:26.1');
  assert.equal(formatTime(61400), '01:01.4');
});

test('fixture preserves the agreed schema and unavailable behavior metrics', () => {
  assertAnalysis(sample);
  assert.equal(sample.synthetic, true);
  assert.equal(sample.predictions.status, 'not_available');
  assert.deepEqual(sample.predictions.metrics, {});
  assert.throws(() => assertAnalysis({ ...sample, schema_version: 'nvi.analysis.v1' }));
  assert.ok(sample.channels.every(c => c.values.length === sample.timing.n_display_samples));
});

test('rejects malformed nested data and fake behavior placeholders while tolerating extra fields', () => {
  assertAnalysis({ ...sample, future_optional_field: true });
  assert.throws(() => assertAnalysis({ ...sample, events: {} }));
  assert.throws(() => assertAnalysis({ ...sample, predictions: { status: 'not_available', metrics: { hold: { value: 0.8 } } } }));
  assert.throws(() => assertAnalysis({ ...sample, channels: [{ ...sample.channels[0], hz: 0 }] }));
  assert.throws(() => assertAnalysis({ ...sample, assets: { brain_map: { ...sample.assets.brain_map, tile_w: 0 } } }));
  assertAnalysis({ ...sample, status: 'processing', duration_ms: 0, channels: [] });
});

test('validates optional performance data and forbids numbers in unavailable states', () => {
  const emptyPerformance = { model_status: 'not_trained', validated: false, reason: 'not ready', brain_claim: 'not_tested', model_version: null, n_train: null, caption: null, context: { deal_id: 'deal-a', deal_label: 'Deal A', platform: 'tiktok', account_id: null, account_level: false }, clip_in_training: 'no', retrospective: false, engagement: null, reach: null, drivers: [], warnings: [], provenance: {} };
  assertAnalysis({ ...sample, performance: emptyPerformance });
  assert.throws(() => assertAnalysis({ ...sample, performance: { ...emptyPerformance, engagement: { percentile_deal_platform: 0.5 } } }));
  assert.throws(() => assertAnalysis({ ...sample, performance: { ...emptyPerformance, drivers: [{ family: 'metadata', label: 'Metadata', contribution: 0.1 }] } }));
  const preliminary = { ...emptyPerformance, model_status: 'preliminary', model_version: 'perf-test', n_train: 171, caption: 'Preliminary model — trained on 171 clips, not validated. Illustrative of the format, not a forecast.', reason: null, engagement: { target: 'log_interactions_rate', percentile_deal_platform: null, likely_range: null, reference_n: 20, percentile_account: null, account_reference_n: null, confidence: 'low', validation: { scheme: 'none', within_stratum_spearman: null, ci95: [null, null] } } };
  assertAnalysis({ ...sample, performance: preliminary });
  assert.throws(() => assertAnalysis({ ...sample, performance: { ...preliminary, caption: null } }));
});

test('3D is enabled structurally: every mesh channel present with the same key and z-score unit', () => {
  const meta = JSON.parse(readFileSync(new URL('../public/brain/brain.meta.json', import.meta.url), 'utf8'));
  assert.deepEqual([...BRAIN_CHANNEL_KEYS], meta.channel_keys);
  assert.deepEqual([...BRAIN_CHANNEL_KEYS], ['attention', 'social', 'value', 'control', 'self', 'language', 'sensory']);
  assert.match(BRAIN_DISPLAY_NOTE, /A region gets its channel z-score\. This is not per-vertex TRIBE output/);
  assert.equal(supportsBrain3d(sample), true);
  // A real bundle is a different clip with its own provenance; only the channel structure matters.
  const real: Analysis = { ...sample, synthetic: false, analysis_id: 'a_000000000000', video_id: 'ffffffffffffffff', duration_ms: 25000, provenance: { ...sample.provenance, normalization: 'other' }, channels: [...sample.channels].reverse().map(c => ({ ...c, values: c.values.map(v => v === null ? null : -v) })) };
  assert.equal(supportsBrain3d(real), true);
});

test('3D stays off when a mesh channel is missing or not a within-clip z-score', () => {
  assert.equal(supportsBrain3d({ ...sample, channels: sample.channels.slice(1) }), false);
  assert.equal(supportsBrain3d({ ...sample, channels: sample.channels.map(c => c.key === 'value' ? { ...c, key: 'reward' } : c) }), false);
  assert.equal(supportsBrain3d({ ...sample, channels: sample.channels.map(c => c.key === 'self' ? { ...c, unit: 'percent' as unknown as Channel['unit'] } : c) }), false);
  assert.equal(supportsBrain3d({ ...sample, channels: [] }), false);
});

test('anatomical mesh matches channel identities and stays within index bounds', () => {
  const meta = JSON.parse(readFileSync(new URL('../public/brain/brain.meta.json', import.meta.url), 'utf8'));
  const file = readFileSync(new URL('../public/brain/brain.bin', import.meta.url));
  const n = file.readUInt32LE(0), faces = file.readUInt32LE(4);
  assert.equal(n, 20484);
  assert.equal(faces, 40960);
  assert.equal(file.length, 8 + n * 20 + faces * 12);
  assert.deepEqual(meta.channel_keys, sample.channels.map(c => c.key));
  for (let i = 0; i < n; i++) {
    const id = file.readFloatLE(8 + n * 16 + i * 4);
    assert.ok(Number.isInteger(id) && id >= -1 && id < sample.channels.length);
  }
  for (let i = 0; i < faces * 3; i++) assert.ok(file.readUInt32LE(8 + n * 20 + i * 4) < n);
});
