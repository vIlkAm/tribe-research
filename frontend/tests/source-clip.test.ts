import test from 'node:test';
import assert from 'node:assert/strict';
import { probeSourceClip, resolveSourceClipUrl, SOURCE_CLIP_LABEL } from '../src/lib/source-clip.ts';

test('source clip resolves as ../clips/<video_id>.mp4 against the index URL', () => {
  assert.equal(resolveSourceClipUrl('/demo-stage1/index.json', '39a56688d3b5b35e'), 'http://localhost/clips/39a56688d3b5b35e.mp4');
  assert.equal(resolveSourceClipUrl('https://h.test/a/b/index.json', 'clip_1-x'), 'https://h.test/a/clips/clip_1-x.mp4');
  assert.equal(resolveSourceClipUrl('https://h.test/index.json', 'abc'), 'https://h.test/clips/abc.mp4');
  for (const bad of ['', '../x', 'a/b', 'a.mp4', 'a?b', 'a#b', '-lead', 'x y', '%2e%2e']) {
    assert.throws(() => resolveSourceClipUrl('/demo-stage1/index.json', bad), /video ID is invalid/, bad);
  }
  assert.equal(SOURCE_CLIP_LABEL, 'Source clip (local server)');
});

type Call = { url: string; init?: RequestInit };
function fakeFetch(responses: (Response | Error)[]) {
  const calls: Call[] = [];
  const impl = async (url: string, init?: RequestInit) => {
    calls.push({ url, init });
    const next = responses.shift();
    if (!next) throw new Error('unexpected request');
    if (next instanceof Error) throw next;
    return next;
  };
  return { calls, impl };
}
const video = (status: number, type = 'video/mp4') => new Response(status === 200 || status === 206 ? 'x' : null, { status, headers: { 'content-type': type } });

test('probe accepts only a 200 video response to HEAD', async () => {
  const ok = fakeFetch([video(200)]);
  assert.equal(await probeSourceClip('http://h/clips/a.mp4', ok.impl), true);
  assert.equal(ok.calls.length, 1);
  assert.equal(ok.calls[0].init?.method, 'HEAD');
  assert.equal(await probeSourceClip('u', fakeFetch([video(200, 'Video/MP4; codecs="avc1"')]).impl), true);
});

test('probe rejects the SPA fallback (200 text/html), 404 and network errors', async () => {
  assert.equal(await probeSourceClip('u', fakeFetch([video(200, 'text/html')]).impl), false);
  assert.equal(await probeSourceClip('u', fakeFetch([new Response(null, { status: 200 })]).impl), false);
  assert.equal(await probeSourceClip('u', fakeFetch([video(404, 'text/plain')]).impl), false);
  assert.equal(await probeSourceClip('u', fakeFetch([new TypeError('network')]).impl), false);
});

test('probe falls back to a one-byte range GET only when HEAD is not allowed', async () => {
  const fallback = fakeFetch([video(405, 'text/plain'), video(206)]);
  assert.equal(await probeSourceClip('u', fallback.impl), true);
  assert.equal(fallback.calls[1].init?.method, 'GET');
  assert.deepEqual(fallback.calls[1].init?.headers, { Range: 'bytes=0-0' });
  assert.equal(await probeSourceClip('u', fakeFetch([video(501, 'text/plain'), video(200)]).impl), true);
  assert.equal(await probeSourceClip('u', fakeFetch([video(405, 'text/plain'), video(200, 'text/html')]).impl), false);
  assert.equal(await probeSourceClip('u', fakeFetch([video(405, 'text/plain'), video(404, 'text/plain')]).impl), false);
});
