import test from 'node:test';
import assert from 'node:assert/strict';
import { parseRealBundleIndex, resolveRealBundleUrls } from '../src/lib/real-analysis-index.ts';

const entry = { video_id: 'clip-1', analysis_id: 'analysis-1', path: 'clip-1/analysis.json', duration_ms: 12000, status: 'complete', synthetic: false, n_channels: 7, n_moments: 3, has_words: true, n_warnings: 0 };

test('accepts the current handoff and the v2 performance handoff', () => {
  const current = parseRealBundleIndex({ schema_version: 'nvi.analysis.v0.2', count: 1, synthetic: false, bundles: [entry] });
  assert.equal(current.bundles[0].performance, undefined);
  const performance = { model_status: 'preliminary' as const, validated: false, clip_in_training: 'no' as const };
  const next = parseRealBundleIndex({ schema_version: 'nvi.analysis.v0.2', count: 1, synthetic: false, model_version: 'prelim-v0', bundles: [{ ...entry, performance_path: 'clip-1/performance.json', performance, platform: 'youtube', video_link: 'https://youtube.test/watch?v=1', is_lockbox: false }] });
  assert.equal(next.bundles[0].performance?.model_status, 'preliminary');
  assert.deepEqual(resolveRealBundleUrls('https://data.example.test/v2/index.json', next.bundles[0]), {
    analysisUrl: 'https://data.example.test/v2/clip-1/analysis.json',
    performanceUrl: 'https://data.example.test/v2/clip-1/performance.json',
  });
});

test('rejects incomplete performance and inconsistent lockbox metadata', () => {
  assert.throws(() => parseRealBundleIndex({ schema_version: 'nvi.analysis.v0.2', count: 1, synthetic: false, bundles: [{ ...entry, performance_path: 'clip-1/performance.json' }] }), /incomplete performance metadata/);
  assert.throws(() => parseRealBundleIndex({ schema_version: 'nvi.analysis.v0.2', count: 1, synthetic: false, bundles: [{ ...entry, performance_path: 'clip-1/performance.json', performance: { model_status: 'preliminary', validated: false, clip_in_training: 'lockbox' }, is_lockbox: false }] }), /inconsistent lockbox/);
  assert.throws(() => resolveRealBundleUrls('https://data.example.test/v2/index.json', { ...entry, path: 'https://other.example.test/analysis.json' }), /analysis path is invalid/);
});
