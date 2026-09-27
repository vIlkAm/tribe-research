/** A scripted concept film. These values are animation controls, never research data. */
export const FILM_DURATION = 30;
export const BEAT = 60 / 144;
export type SceneName = 'city' | 'wheel' | 'tunnel' | 'eye' | 'wave' | 'mountain' | 'orbit';
export interface Shot { start: number; end: number; scene: SceneName; caption: string; }
const scenes: SceneName[] = ['city', 'wheel', 'tunnel', 'city', 'eye', 'wave', 'mountain', 'tunnel', 'wheel', 'orbit', 'city', 'mountain', 'eye', 'wave', 'tunnel', 'orbit', 'wheel', 'city', 'wave', 'eye', 'mountain', 'orbit'];
const captions = ['FIND THE SPARK', 'FOLLOW THE MOTION', 'CHANGE THE PERSPECTIVE', 'MAKE THE MOMENT'];
let cursor = 12;
export const SHOTS: Shot[] = scenes.map((scene, i) => {
  const duration = i < 8 ? BEAT * 2 : BEAT;
  const start = cursor;
  cursor += duration;
  return { start, end: cursor, scene, caption: captions[Math.min(3, Math.floor(i / 6))] };
});
SHOTS.push({ start: cursor, end: 26, scene: 'mountain', caption: 'MAKE THE MOMENT' });

export function phaseAt(time: number): 'intro' | 'original' | 'transition' | 'optimized' | 'outro' {
  return time < 2 ? 'intro' : time < 10 ? 'original' : time < 12 ? 'transition' : time < 26 ? 'optimized' : 'outro';
}

export function shotAt(time: number): Shot | undefined {
  return SHOTS.find(shot => time >= shot.start && time < shot.end);
}

export function beatPulse(time: number): number {
  if (time < 12 || time >= 26) return 0;
  const beat = (time - 12) / BEAT;
  const sinceBeat = Math.max(0, beat - Math.floor(beat + 1e-8)) * BEAT;
  return Math.exp(-sinceBeat * 11);
}

const regionKeys = ['attention', 'social', 'value', 'control', 'self', 'language', 'sensory'];
export function illustrativeSignals(time: number): Record<string, number> {
  const phase = phaseAt(time);
  const build = Math.min(1, Math.max(0, (time - 12) / 12));
  return Object.fromEntries(regionKeys.map((key, i) => {
    if (phase === 'optimized') {
      const rolling = 0.5 + 0.5 * Math.sin(time * 2.8 - i * 1.25);
      return [key, 0.45 + build * 0.85 + rolling * 0.9 + beatPulse(time) * (0.75 + build * 0.7)];
    }
    const fade = phase === 'outro' ? Math.max(0, 1 - (time - 26) / 2) : 0;
    return [key, 0.19 + 0.14 * (0.5 + Math.sin(time * 0.65 + i) / 2) + fade * (1.2 + i * 0.08)];
  }));
}

export function editDescription(time: number): string {
  switch (phaseAt(time)) {
    case 'intro': return 'One original story. Two ways to tell it.';
    case 'original': return 'One long shot. Gentle movement. A quiet soundtrack.';
    case 'transition': return 'Now change the rhythm.';
    case 'optimized': return time < 19 ? 'Build the pace. Match movement to the beat.' : 'Faster cuts. New perspectives. A deliberate payoff.';
    case 'outro': return 'Timing. Movement. Sound. Edit with intent.';
  }
}
