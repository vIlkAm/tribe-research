import test from 'node:test';
import assert from 'node:assert/strict';
import { REFERENCE_CLIPS, REFERENCE_DURATION, referencePosition, referenceTimeline, referenceTimestamp, referenceDuration, referenceVideoRequest } from '../src/demo/reference-clips.ts';
import { referenceBrainState } from '../src/demo/reference-brain.ts';

test('calm excerpt stays fixed and the final reference runs to its natural end', () => {
  assert.deepEqual(REFERENCE_CLIPS.map(({ id, start, end }) => ({ id, start, end })), [
    { id: 'J5Mv-BehKY4', start: 8, end: 16 },
    { id: '8FKGDKEwGf0', start: 5, end: 20 },
  ]);
  assert.equal(REFERENCE_CLIPS.reduce((sum, clip) => sum + clip.end - clip.start, 0), REFERENCE_DURATION);
  assert.equal(REFERENCE_DURATION, 23);
  assert.equal(referencePosition(0).sourceTime, 8);
  assert.equal(referencePosition(7.9).index, 0);
  assert.equal(referencePosition(8).index, 1);
  assert.equal(referencePosition(8).sourceTime, 5);
  assert.equal(referencePosition(15).sourceTime, 12);
  assert.equal(referencePosition(18).sourceTime, 15);
  assert.equal(referenceTimestamp(REFERENCE_CLIPS[1].start), '0:05');
  assert.equal(REFERENCE_CLIPS[1].naturalEnd, true);
  assert.equal(referencePosition(23).sourceTime, 20);
});

test('media time maps back to the same timeline position, and out-of-range seeks clamp', () => {
  for (const time of [0, 0.5, 7.99, 8, 8.01, 12.5, 15, 17.999, 18, 22.99, 23]) {
    const position = referencePosition(time);
    assert.ok(Math.abs(referenceTimeline(position.index, position.sourceTime) - time) < 1e-9);
  }
  assert.equal(referencePosition(-10).sourceTime, 8);
  assert.equal(referencePosition(99).sourceTime, 20);
  assert.equal(referenceTimeline(0, 0), 0);
  assert.equal(referenceTimeline(0, 80), 8);
  assert.equal(referenceTimeline(1, 0), 8);
  assert.equal(referenceTimeline(1, 5), 8);
  assert.equal(referenceTimeline(1, 80), 23);
});

test('brain stays entirely gray until one second of exciting playback has elapsed', () => {
  for (const time of [0, 4, 7.99, 8, 8.5, 8.99, 9]) {
    const brain = referenceBrainState(time);
    assert.ok(Object.values(brain.values).every(value => value === 0));
    assert.equal(brain.activity, 0);
    assert.ok(brain.rotationSpeed > 0.6, 'calm rotation is faster than the workspace default');
  }
  assert.ok(referenceBrainState(9.5).activity > 0);
  assert.ok(referenceBrainState(9.5).activity < 0.1, 'response starts gently');
  assert.ok(referenceBrainState(12).activity > 0.8);
});

test('regional pulses shift across all seven channels while preserving neutral cortex', () => {
  const brightRegions = new Set<string>();
  for (let time = 12; time < 23.6; time += 0.1) {
    const entries = Object.entries(referenceBrainState(time).values);
    for (const [region, value] of entries) if (value > 1) brightRegions.add(region);
    assert.ok(entries.some(([, value]) => Math.abs(value) < 0.18));
    assert.ok(entries.some(([, value]) => value > 0.4));
    assert.ok(entries.some(([, value]) => value < -0.4));
  }
  assert.equal(brightRegions.size, 7);
  assert.notDeepEqual(referenceBrainState(12).values, referenceBrainState(14).values);
});

test('seeking back restores the same brain state and the quiet delay', () => {
  const active = referenceBrainState(13.5);
  referenceBrainState(23.6);
  assert.deepEqual(referenceBrainState(13.5), active);
  assert.equal(referenceBrainState(8.5).activity, 0);
});

test('YouTube receives a cutoff only for the calm clip, including after seeking', () => {
  assert.deepEqual(referenceVideoRequest(0), { videoId: 'J5Mv-BehKY4', startSeconds: 8, endSeconds: 16 });
  assert.deepEqual(referenceVideoRequest(1), { videoId: '8FKGDKEwGf0', startSeconds: 5 });
  assert.deepEqual(referenceVideoRequest(1, 17.5), { videoId: '8FKGDKEwGf0', startSeconds: 17.5 });
});

test('precise provider duration updates timeline and seeks', () => {
  const end = 20.234;
  assert.equal(referenceDuration(end), 23.234);
  assert.equal(referenceDuration(0), REFERENCE_DURATION);
  assert.equal(referenceDuration(NaN), REFERENCE_DURATION);
  for (const time of [8, 18, 22, 23.2, referenceDuration(end)]) {
    const position = referencePosition(time, end);
    assert.ok(Math.abs(referenceTimeline(position.index, position.sourceTime, end) - time) < 1e-9);
  }
  assert.equal(referencePosition(100, end).sourceTime, end);
  assert.equal(referenceTimeline(1, 100, end), referenceDuration(end));
});
