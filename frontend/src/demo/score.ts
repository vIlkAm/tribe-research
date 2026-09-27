import { BEAT, FILM_DURATION } from './sequence';

/** Original synthesized score; no samples, licensed tracks, or network requests. */
async function makeScore(): Promise<AudioBuffer> {
  const context = new OfflineAudioContext(2, 24000 * FILM_DURATION, 24000);
  const master = context.createGain();
  master.gain.value = 0.35;
  master.connect(context.destination);
  function note(time: number, duration: number, frequency: number, volume: number, type: OscillatorType = 'sine', pan = 0) {
    const oscillator = context.createOscillator(), gain = context.createGain(), stereo = context.createStereoPanner();
    oscillator.type = type;
    oscillator.frequency.value = frequency;
    gain.gain.setValueAtTime(0, time);
    gain.gain.linearRampToValueAtTime(volume, time + Math.min(0.025, duration / 4));
    gain.gain.exponentialRampToValueAtTime(0.001, time + duration);
    stereo.pan.value = pan;
    oscillator.connect(gain).connect(stereo).connect(master);
    oscillator.start(time); oscillator.stop(time + duration + 0.01);
  }
  function kick(time: number) {
    const osc = context.createOscillator(), gain = context.createGain();
    osc.frequency.setValueAtTime(135, time); osc.frequency.exponentialRampToValueAtTime(43, time + 0.16);
    gain.gain.setValueAtTime(0.8, time); gain.gain.exponentialRampToValueAtTime(0.001, time + 0.32);
    osc.connect(gain).connect(master); osc.start(time); osc.stop(time + 0.34);
  }
  const noise = context.createBuffer(1, 24000, 24000);
  const samples = noise.getChannelData(0);
  let seed = 9817;
  for (let i = 0; i < samples.length; i++) { seed = (seed * 16807) % 2147483647; samples[i] = seed / 1073741823.5 - 1; }
  function percussion(time: number, duration: number, volume: number, frequency: number) {
    const source = context.createBufferSource(), filter = context.createBiquadFilter(), gain = context.createGain();
    source.buffer = noise; filter.type = 'highpass'; filter.frequency.value = frequency;
    gain.gain.setValueAtTime(volume, time); gain.gain.exponentialRampToValueAtTime(0.001, time + duration);
    source.connect(filter).connect(gain).connect(master); source.start(time); source.stop(time + duration);
  }
  // Sparse opening: the same tonal palette as the re-edit, without the rhythm section.
  for (let time = 0; time < 10; time += 2) {
    note(time, 1.8, 146.83, 0.07);
    note(time + 0.15, 1.6, 220, 0.035, 'sine', -0.25);
  }
  for (let i = 0; i < 8; i++) note(10 + i * 0.23, 0.25, 146.83 * 2 ** (i / 12), 0.07, 'triangle', i % 2 ? 0.4 : -0.4);
  const bass = [73.416, 73.416, 98, 65.406, 87.307, 73.416, 65.406, 110];
  const melody = [293.665, 440, 349.228, 523.251, 440, 587.33, 349.228, 391.995];
  for (let i = 0, time = 12; time < 26; i++, time = 12 + i * BEAT) {
    kick(time);
    note(time + 0.02, BEAT * 0.75, bass[Math.floor(i / 2) % bass.length], 0.28, 'triangle');
    percussion(time, 0.045, 0.10, 7000);
    if (i % 2) { percussion(time, 0.16, 0.19, 1800); note(time, 0.11, 180, 0.07); }
    if (time > 16) percussion(time + BEAT / 2, 0.04, 0.07, 8500);
    note(time, 0.32, melody[i % melody.length], 0.08, 'triangle', i % 2 ? -0.35 : 0.35);
    if (time > 19) note(time + BEAT / 2, 0.19, melody[(i + 3) % melody.length] * 2, 0.04, 'sine', i % 2 ? 0.5 : -0.5);
  }
  [146.83, 220, 293.665, 349.228].forEach((frequency, i) => note(26 + i * 0.04, 3.5, frequency, 0.13, 'sine', (i - 1.5) * 0.2));
  const buffer = await context.startRendering();
  return buffer;
}

export class ComparisonAudio {
  private context: AudioContext | null = null;
  private buffer: Promise<AudioBuffer> | null = null;
  private source: AudioBufferSourceNode | null = null;
  private operation = 0;
  private closed = false;

  async play(currentTime: () => number) {
    this.pause();
    const operation = this.operation;
    if (this.closed) return;
    this.context ??= new AudioContext();
    await this.context.resume();
    this.buffer ??= makeScore();
    const buffer = await this.buffer;
    if (this.closed || operation !== this.operation) return;
    const offset = currentTime();
    if (offset >= FILM_DURATION) return;
    const source = this.context.createBufferSource();
    source.buffer = buffer;
    source.connect(this.context.destination);
    source.onended = () => source.disconnect();
    source.start(0, Math.max(0, offset));
    this.source = source;
  }

  pause() {
    this.operation++;
    this.source?.stop();
    this.source?.disconnect();
    this.source = null;
  }

  close() { this.closed = true; this.pause(); void this.context?.close(); }
}
