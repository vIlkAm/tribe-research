import test from 'node:test';
import assert from 'node:assert/strict';
import { BEAT, FILM_DURATION, SHOTS, beatPulse, illustrativeSignals, phaseAt, shotAt } from '../src/demo/sequence.ts';

test('the comparison plays original first, then transition, re-edit, and conclusion', () => {
  assert.equal(FILM_DURATION, 30);
  assert.equal(phaseAt(0), 'intro');
  assert.equal(phaseAt(2), 'original');
  assert.equal(phaseAt(9.999), 'original');
  assert.equal(phaseAt(10), 'transition');
  assert.equal(phaseAt(12), 'optimized');
  assert.equal(phaseAt(26), 'outro');
  assert.equal(phaseAt(30), 'outro');
});

test('shots fill the re-edit without gaps, accelerate on the beat, and end with a hold', () => {
  assert.equal(SHOTS[0].start, 12);
  assert.equal(SHOTS.at(-1)!.end, 26);
  SHOTS.forEach((shot, i) => {
    if (i > 0) assert.equal(shot.start, SHOTS[i - 1].end);
    assert.ok(shot.end - shot.start >= BEAT - 1e-8);
    assert.equal(shotAt(shot.start), shot);
    assert.equal(shotAt(shot.end - 0.0001), shot);
  });
  assert.equal(shotAt(10), undefined);
  assert.equal(shotAt(26), undefined);
  assert.ok(SHOTS[8].end - SHOTS[8].start < SHOTS[0].end - SHOTS[0].start);
});

test('scripted brain pulse and shot cuts share the musical clock', () => {
  for (const shot of SHOTS) assert.ok(beatPulse(shot.start) > 0.9999);
  assert.equal(beatPulse(2), 0);
  assert.equal(beatPulse(26), 0);
  assert.ok(beatPulse(12.2) < beatPulse(12));
});

test('illustration controls are deterministic and build above a quiet baseline', () => {
  const quiet = Object.values(illustrativeSignals(4));
  const peak = Object.values(illustrativeSignals(22));
  assert.ok(quiet.every(value => value >= 0 && value < 0.4));
  assert.ok(peak.every(value => value > 0.8 && value < 4));
  assert.deepEqual(illustrativeSignals(22), illustrativeSignals(22));
  assert.equal(Object.keys(illustrativeSignals(0)).length, 7);
  assert.ok(Object.values(illustrativeSignals(30)).every(value => value < 0.4));
});
