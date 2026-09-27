/**
 * Demo mode: the approved local index doubles as the presenter's clip list.
 *
 * Index entries may carry `demo_role` (performance tier: great / typical / bad),
 * `deal_label`, `platform` and `demo_moment` (all optional). Parsing is defensive:
 * an unknown tier lands in "Other clips", a missing deal label falls back to the
 * video ID, and an unusable moment is dropped, so a stale or partial index still opens.
 */
import type { RealBundleIndex, RealBundleIndexEntry } from './real-analysis-index.ts';
import { clockTime } from './library.ts';
import { platformLabel } from './performance.ts';

export type DemoTier = 'great' | 'typical' | 'bad';
export const TIER_ORDER: readonly DemoTier[] = ['great', 'typical', 'bad'];
export const TIER_LABELS: Record<DemoTier, string> = { great: 'Did great', typical: 'Typical', bad: 'Did badly' };

export interface DemoMoment { start_ms: number; end_ms: number; label: string }
export interface DemoClip {
  video_id: string;
  /** "<deal_label> · <platform> · <length>" */
  label: string;
  tier: DemoTier | null;
  deal_label: string | null;
  platform: string | null;
  duration_ms: number;
  moment: DemoMoment | null;
  entry: RealBundleIndexEntry;
}
export interface DemoGroup { tier: DemoTier | null; label: string; clips: DemoClip[] }

export const VIDEO_ID = /^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$/;
type Obj = Record<string, unknown>;
const isObj = (value: unknown): value is Obj => !!value && typeof value === 'object' && !Array.isArray(value);
const finite = (value: unknown): number | null => typeof value === 'number' && Number.isFinite(value) ? value : null;
const text = (value: unknown, max = 60): string | null => typeof value === 'string' && value.trim() ? value.trim().slice(0, max) : null;

export function parseTier(value: unknown): DemoTier | null {
  return value === 'great' || value === 'typical' || value === 'bad' ? value : null;
}

/** A playable window inside the clip; clamped to `durationMs` when given. */
export function parseDemoMoment(value: unknown, durationMs?: number): DemoMoment | null {
  if (!isObj(value)) return null;
  const start = finite(value.start_ms), end = finite(value.end_ms);
  if (start === null || end === null || start < 0) return null;
  const limit = durationMs !== undefined && durationMs > 0 ? durationMs : Infinity;
  const clampedEnd = Math.min(end, limit);
  if (start >= limit || clampedEnd - start < 100) return null;
  return { start_ms: start, end_ms: clampedEnd, label: text(value.label, 80) ?? `Moment at ${clockTime(start)}` };
}

/** Clip length for labels: "32 s", or "1:05" from a minute. */
export function lengthLabel(ms: number): string {
  return ms >= 60000 ? clockTime(ms) : `${Math.max(1, Math.round(ms / 1000))} s`;
}

/** Picker entries, in index order. */
export function demoClips(index: RealBundleIndex | null): DemoClip[] {
  if (!index) return [];
  const seen = new Set<string>();
  return index.bundles.flatMap(entry => {
    if (!VIDEO_ID.test(entry.video_id) || seen.has(entry.video_id)) return [];
    seen.add(entry.video_id);
    const deal = text(entry.deal_label);
    const platform = text(entry.platform, 30);
    return [{
      video_id: entry.video_id,
      label: [deal ?? entry.video_id, platform ? platformLabel(platform) : null, lengthLabel(entry.duration_ms)].filter(Boolean).join(' · '),
      tier: parseTier(entry.demo_role), deal_label: deal, platform, duration_ms: entry.duration_ms,
      moment: parseDemoMoment(entry.demo_moment, entry.duration_ms), entry,
    }];
  });
}

/** Tier groups in fixed order (great, typical, bad, then untiered); empty groups are dropped. */
export function groupDemoClips(clips: DemoClip[]): DemoGroup[] {
  const groups: DemoGroup[] = TIER_ORDER.map(tier => ({ tier, label: TIER_LABELS[tier], clips: clips.filter(clip => clip.tier === tier) }));
  groups.push({ tier: null, label: 'Other clips', clips: clips.filter(clip => !clip.tier) });
  return groups.filter(group => group.clips.length);
}

/** "Did great · <deal_label>" heading for one clip. */
export function clipHeading(clip: DemoClip): string {
  return [clip.tier ? TIER_LABELS[clip.tier] : null, clip.deal_label ?? clip.video_id].filter(Boolean).join(' · ');
}

/** `?<name>=<video_id>`, validated; anything else is ignored. */
export function clipParam(search: string, name = 'clip'): string | null {
  const value = new URLSearchParams(search).get(name);
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
