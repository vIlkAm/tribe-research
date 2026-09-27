import test from 'node:test';
import assert from 'node:assert/strict';
import { ReferencePlayback } from '../src/demo/reference-playback.ts';
import { REFERENCE_CLIPS } from '../src/demo/reference-clips.ts';

function setup() {
  const calls: Array<[string, unknown?]> = [];
  const player = {
    id: REFERENCE_CLIPS[0].id as string, time: 8, duration: 78, code: -1,
    loadVideoById: (request: unknown) => { calls.push(['load', request]); },
    cueVideoById: (request: unknown) => { calls.push(['cue', request]); },
    playVideo: () => { calls.push(['play']); },
    pauseVideo: () => { calls.push(['pause']); },
    seekTo: (seconds: number) => { calls.push(['seek', seconds]); },
    mute() {}, unMute() {}, destroy() {},
    getCurrentTime: () => player.time,
    getDuration: () => player.duration,
    getPlayerState: () => player.code,
    getVideoUrl: () => `https://www.youtube.com/watch?v=${player.id}`,
  };
  const transport = new ReferencePlayback(() => {});
  transport.attach(player);
  const event = (code: number) => { player.code = code; transport.onState(code); };
  return { calls, player, transport, event };
}

test('first play and pause/resume do not reload or re-cue the video', () => {
  const { calls, player, transport, event } = setup();
  assert.deepEqual(calls, []);
  transport.toggle(); event(1); player.time = 10.5; transport.tick();
  transport.toggle(); event(2);
  assert.equal(transport.state.time, 2.5);
  transport.toggle(); event(1);
  assert.deepEqual(calls.map(([command]) => command), ['play', 'pause', 'play']);
  assert.equal(transport.state.playing, true);
});

test('quick chapter selection then Play survives a late cued event', () => {
  const { calls, player, transport, event } = setup();
  transport.go(8, false);
  transport.toggle();
  player.id = REFERENCE_CLIPS[1].id; player.time = 5; player.duration = 20.6;
  event(5);
  assert.equal(calls.at(-1)?.[0], 'play');
  event(1); transport.tick();
  assert.equal(transport.state.time, 8);
  assert.equal(transport.state.videoDuration, 20.6);
  assert.equal(transport.state.playing, true);
});

test('automatic transition ignores the previous clip clock, duration and ended event', () => {
  const { player, transport, event, calls } = setup();
  transport.toggle(); event(1); player.time = 16; transport.tick();
  assert.deepEqual(calls.at(-1), ['load', { videoId: REFERENCE_CLIPS[1].id, startSeconds: 5 }]);
  event(0); transport.tick();
  assert.equal(transport.state.time, 8);
  assert.equal(transport.state.videoDuration, 20);
  player.id = REFERENCE_CLIPS[1].id; player.time = 5; player.duration = 20.6;
  event(0);
  assert.notEqual(transport.state.status, 'Comparison complete');
  event(1); player.time = 5.8; transport.tick();
  assert.equal(transport.state.time, 8.8);
});

test('buffering freezes the clock and missing playing events recover without seek flooding', () => {
  const { calls, player, transport, event } = setup();
  transport.toggle(); event(1); player.time = 7;
  for (let i = 0; i < 10; i++) transport.tick();
  assert.equal(calls.filter(([command]) => command === 'seek').length, 1);
  player.time = 9; transport.tick(); event(3); player.time = 11; transport.tick();
  assert.equal(transport.state.time, 1);
  assert.equal(transport.state.playing, false);
  player.code = 1; transport.tick();
  assert.equal(transport.state.time, 3);
  assert.equal(transport.state.playing, true);
});

test('pause during loading prevents a late playing event from restarting the demo', () => {
  const { calls, player, transport, event } = setup();
  transport.go(8, true); transport.pause();
  player.id = REFERENCE_CLIPS[1].id; player.time = 5;
  event(1);
  assert.equal(calls.at(-1)?.[0], 'pause');
  assert.equal(transport.state.playing, false);
  event(2); event(1); // Later Play from YouTube's own controls is respected.
  assert.equal(transport.state.playing, true);
});

test('the final clip finishes naturally and replay returns to the calm excerpt', () => {
  const { calls, player, transport, event } = setup();
  transport.go(8, true);
  player.id = REFERENCE_CLIPS[1].id; player.time = 5; player.duration = 20.6;
  event(1); player.time = 18.2; transport.tick();
  assert.equal(transport.state.playing, true);
  player.time = 20.6; event(0);
  assert.equal(transport.state.status, 'Comparison complete');
  assert.equal(transport.state.time, 23.6);
  transport.toggle();
  assert.deepEqual(calls.at(-1), ['load', { videoId: REFERENCE_CLIPS[0].id, startSeconds: 8, endSeconds: 16 }]);
  assert.equal(transport.state.time, 0);
});

test('seeking while already paused still allows native YouTube Play on the first click', () => {
  const { player, transport, event } = setup();
  event(1); event(2);
  transport.go(3, false);
  player.time = 11; transport.tick(); // No new PAUSED event for an already paused player.
  event(1);
  assert.equal(transport.state.playing, true);
  assert.equal(transport.state.time, 3);
});

test('scrubbing before first playback cues a start position without seeking and immediately pausing an unstarted video', () => {
  const { calls, player, transport, event } = setup();
  transport.go(2.5, false);
  assert.deepEqual(calls, [['cue', { videoId: REFERENCE_CLIPS[0].id, startSeconds: 10.5, endSeconds: 16 }]]);
  event(5); transport.toggle();
  assert.equal(calls.at(-1)?.[0], 'play');
  player.time = 10.5; event(1); transport.tick();
  assert.equal(transport.state.time, 2.5);
  assert.equal(transport.state.playing, true);
});
