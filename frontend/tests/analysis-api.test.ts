import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { apiBaseUrl, openJobAnalysis, openRemoteAnalysis } from '../src/lib/analysis-api.ts';

const sample = JSON.parse(readFileSync(new URL('../public/data/cd20b16879d630c4/analysis.json', import.meta.url), 'utf8'));

test('analysis API is opt-in and requires HTTPS outside local development', () => {
  assert.equal(apiBaseUrl(undefined), null);
  assert.equal(apiBaseUrl('https://analysis.example.test/'), 'https://analysis.example.test');
  assert.equal(apiBaseUrl('http://127.0.0.1:8787/'), 'http://127.0.0.1:8787');
  assert.throws(() => apiBaseUrl('http://analysis.example.test'), /HTTPS/);
});

test('completed inline jobs still pass the authoritative v0.2 validator', async () => {
  const bundle = await openJobAnalysis({ id: 'job-1', state: 'done', analysis: structuredClone(sample) });
  assert.equal(bundle.analysis.analysis_id, sample.analysis_id);
  assert.equal(bundle.origin, 'remote');
  assert.equal(bundle.resolveAsset('brain_proxy.jpg'), null);
  await assert.rejects(openJobAnalysis({ id: 'job-2', state: 'done', analysis: { ...sample, schema_version: 'nvi.analysis.v9' } }), /supports nvi.analysis.v0.2/);
});

test('remote handoff loads and verifies a separate preliminary performance result', async () => {
  const preliminary = {
    model_status: 'preliminary', validated: false, reason: null, brain_claim: 'not_tested', model_version: 'prelim-v0', n_train: 171,
    caption: 'Preliminary model — trained on 171 clips, not validated. Illustrative of the format, not a forecast.',
    context: { deal_id: 'deal-a', deal_label: 'Deal A', platform: 'youtube', account_id: null, account_level: false },
    clip_in_training: 'no', retrospective: false,
    engagement: { target: 'log_interactions_rate', percentile_deal_platform: null, likely_range: null, reference_n: 20, percentile_account: null, account_reference_n: null, confidence: 'low', validation: { scheme: 'none', within_stratum_spearman: null, ci95: [null, null] } },
    reach: null, drivers: [], warnings: [], provenance: { video_id: sample.video_id },
  };
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async input => new Response(JSON.stringify(String(input).endsWith('performance.json') ? preliminary : sample), { status: 200, headers: { 'Content-Type': 'application/json' } });
  try {
    const bundle = await openRemoteAnalysis('https://data.example.test/clip/analysis.json', {
      performanceUrl: 'https://data.example.test/clip/performance.json',
      expectedPerformance: { model_status: 'preliminary', validated: false, clip_in_training: 'no' },
    });
    assert.equal(bundle.analysis.performance?.model_version, 'prelim-v0');
    assert.equal(bundle.analysis.performance?.caption, preliminary.caption);
    await assert.rejects(openRemoteAnalysis('https://data.example.test/clip/analysis.json', {
      performanceUrl: 'https://data.example.test/clip/performance.json',
      expectedPerformance: { model_status: 'validated', validated: true, clip_in_training: 'no' },
    }), /does not match the analysis index/);
    await assert.rejects(openRemoteAnalysis('https://data.example.test/clip/analysis.json', {
      performanceUrl: 'https://other.example.test/clip/performance.json',
    }), /must use the analysis origin/);
  } finally {
    globalThis.fetch = originalFetch;
  }
});
