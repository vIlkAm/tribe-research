import test from 'node:test';
import assert from 'node:assert/strict';
import type { Performance } from '../src/data/analysis.types.ts';
import { percentile, performanceHasNumbers, performanceStatusLabel, platformLabel } from '../src/lib/performance.ts';

const scored = { model_status: 'research_preview', engagement: { percentile_deal_platform: 0 } } as Performance;

test('performance numbers are shown only for scored model states', () => {
  assert.equal(performanceHasNumbers(scored), true);
  assert.equal(performanceHasNumbers({ ...scored, model_status: 'preliminary' }), true);
  assert.equal(performanceHasNumbers({ ...scored, model_status: 'validated' }), true);
  assert.equal(performanceHasNumbers({ ...scored, model_status: 'not_trained' }), false);
  assert.equal(performanceHasNumbers({ ...scored, model_status: 'out_of_scope' }), false);
  assert.equal(performanceHasNumbers({ ...scored, engagement: null }), false);
});

test('percentiles preserve P0 and clamp malformed values', () => {
  assert.equal(percentile(0), 'P0');
  assert.equal(percentile(0.5969), 'P60');
  assert.equal(percentile(1.2), 'P100');
  assert.equal(percentile(-0.2), 'P0');
});

test('contract labels remain user-facing and explicit', () => {
  assert.equal(performanceStatusLabel('research_preview'), 'Research preview');
  assert.equal(performanceStatusLabel('preliminary'), 'Preliminary');
  assert.equal(performanceStatusLabel('out_of_scope'), 'Outside model scope');
  assert.equal(platformLabel('youtube'), 'YouTube');
  assert.equal(platformLabel('tiktok'), 'TikTok');
});
