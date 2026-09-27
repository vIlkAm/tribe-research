/**
 * Demo mode: the approved local index doubles as the presenter's clip list.
 *
 * Index entries may carry `demo_role` and `demo_moment` (both optional). Parsing is
 * defensive: an unknown role falls back to the video ID, and an unusable moment is
 * dropped, so a stale or partial index still opens.
 */
import type { RealBundleIndex, RealBundleIndexEntry } from './real-analysis-index.ts';
import { clockTime } from './library.ts';

export type DemoRole = 'spike' | 'flat' | 'fell_short' | 'beat_expectations';
export const DEMO_ROLE_LABELS: Record<DemoRole, string> = {
  spike: 'Big spike', flat: 'Flat stretch', fell_short: 'Fell short', beat_expectations: 'Beat expectations',
};

export interface DemoMoment { start_ms: number; end_ms: number; label: string }
export interface DemoClip { video_id: string; label: string; role: DemoRole | null; moment: DemoMoment | null; entry: RealBundleIndexEntry }

const VIDEO_ID = /^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$/;
type Obj = Record<string, unknown>;
const isObj = (value: unknown): value is Obj => !!value && typeof value === 'object' && !Array.isArray(value);
const finite = (value: unknown): number | null => typeof value === 'number' && Number.isFinite(value) ? value : null;

export function parseDemoRole(value: unknown): DemoRole | null {
  return typeof value === 'string' && Object.hasOwn(DEMO_ROLE_LABELS, value) ? value as DemoRole : null;
}

/** A playable window inside the clip; clamped to `durationMs` when given. */
export function parseDemoMoment(value: unknown, durationMs?: number): DemoMoment | null {
  if (!isObj(value)) return null;
  const start = finite(value.start_ms), end = finite(value.end_ms);
  if (start === null || end === null || start < 0) return null;
  const limit = durationMs !== undefined && durationMs > 0 ? durationMs : Infinity;
  const clampedEnd = Math.min(end, limit);
  if (start >= limit || clampedEnd - start < 100) return null;
  const label = typeof value.label === 'string' && value.label.trim() ? value.label.trim().slice(0, 80) : `Moment at ${clockTime(start)}`;
  return { start_ms: start, end_ms: clampedEnd, label };
}

/** Picker entries, in index order. */
export function demoClips(index: RealBundleIndex | null): DemoClip[] {
  if (!index) return [];
  const seen = new Set<string>();
  return index.bundles.flatMap(entry => {
    if (!VIDEO_ID.test(entry.video_id) || seen.has(entry.video_id)) return [];
    seen.add(entry.video_id);
    const role = parseDemoRole(entry.demo_role);
    return [{ video_id: entry.video_id, label: role ? DEMO_ROLE_LABELS[role] : entry.video_id, role, moment: parseDemoMoment(entry.demo_moment, entry.duration_ms), entry }];
  });
}

/** `?clip=<video_id>`, validated; anything else is ignored. */
export function clipParam(search: string): string | null {
  const value = new URLSearchParams(search).get('clip');
  return value && VIDEO_ID.test(value) ? value : null;
}

/** The deep-linked clip when it is in the list, otherwise the first one. */
export function chooseDemoClip(clips: DemoClip[], requested: string | null): DemoClip | null {
  return clips.find(clip => clip.video_id === requested) ?? clips[0] ?? null;
}

/** Where "Play this moment" pauses: the moment's end, kept just inside the playable clip. */
export function momentStopMs(moment: DemoMoment, playableEndMs: number): number {
  return Math.max(moment.start_ms, Math.min(moment.end_ms, playableEndMs - 60));
}
