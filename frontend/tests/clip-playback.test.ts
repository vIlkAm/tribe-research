import test from 'node:test';
import assert from 'node:assert/strict';
import { clipEndMs, clipSeekMs, clipFileError } from '../src/lib/clip-playback.ts';

test('matching media stays in stimulus time without stretching or an extra lag', () => {
  const end = clipEndMs(26100, 26.1);
  assert.equal(end, 26100);
  assert.equal(clipSeekMs(4200, end), 4200);
  assert.equal(clipSeekMs(0, end), 0);
});

test('short clips stop at their real end; long clips stop at the analysis end', () => {
  assert.equal(clipEndMs(26100, 8), 8000);
  assert.equal(clipSeekMs(10000, clipEndMs(26100, 8)), 7999);
  assert.equal(clipEndMs(26100, 60), 26100);
  assert.equal(clipSeekMs(60000, clipEndMs(26100, 60)), 26099);
});

test('unloaded metadata and invalid seeks cannot move outside the analysis', () => {
  for (const seconds of [undefined, NaN, Infinity, 0, -1]) assert.equal(clipEndMs(26100, seconds), 26100);
  for (const ms of [NaN, Infinity, -500]) assert.equal(clipSeekMs(ms, 26100), 0);
});

test('local selection supports common containers and rejects non-video and empty files', () => {
  assert.equal(clipFileError({ name: 'my-clip.MP4', type: '', size: 12345 }), null);
  assert.equal(clipFileError({ name: 'clip', type: 'video/webm', size: 12345 }), null);
  assert.match(clipFileError({ name: 'analysis.json', type: 'application/json', size: 12345 })!, /Choose an MP4/);
  assert.match(clipFileError({ name: 'empty.mp4', type: 'video/mp4', size: 0 })!, /empty/);
});
