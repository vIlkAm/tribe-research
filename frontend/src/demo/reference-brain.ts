import { REFERENCE_SPLIT } from './reference-clips.ts';

/** Choreography for the labeled reference demo only; never analysis data. */
export const RESPONSE_DELAY = 1;
export const CALM_ROTATION_SPEED = 0.9;
export const ACTIVE_ROTATION_SPEED = 1.7;
const KEYS = ['attention', 'social', 'value', 'control', 'self', 'language', 'sensory'];
const ORDER = [6, 0, 5, 1, 3, 2, 4];
const SPACING = 0.68;
const PULSE_LENGTH = 1.85;

function envelope(age: number) {
  if (age <= 0 || age >= PULSE_LENGTH) return 0;
  return Math.sin(Math.PI * age / PULSE_LENGTH) ** 2;
}

export function referenceBrainState(time: number) {
  const elapsed = Math.max(0, time - REFERENCE_SPLIT - RESPONSE_DELAY);
  const ramp = Math.min(1, elapsed / 2.4);
  const build = ramp * ramp * (3 - 2 * ramp);
  const signals = new Array<number>(KEYS.length).fill(0);
  // Overlapping pulses move between channels. A softer blue echo follows in a
  // different channel, leaving neutral regions visible instead of a solid glow.
  for (let beat = Math.max(0, Math.floor((elapsed - PULSE_LENGTH) / SPACING)); beat <= Math.floor(elapsed / SPACING); beat++) {
    const pulse = envelope(elapsed - beat * SPACING) * build;
    signals[ORDER[beat % ORDER.length]] += pulse * 2.15;
    signals[ORDER[(beat + 3) % ORDER.length]] -= pulse * 1.15;
  }
  const activity = Math.max(...signals.map(Math.abs)) / 2.15;
  return {
    values: Object.fromEntries(KEYS.map((key, i) => [key, signals[i]])),
    elapsed,
    activity,
    rotationSpeed: CALM_ROTATION_SPEED + (ACTIVE_ROTATION_SPEED - CALM_ROTATION_SPEED) * build,
  };
}
