/** User-selected references. These are two different videos, not a before/after edit. */
export const REFERENCE_CLIPS = [
  { id: 'J5Mv-BehKY4', label: 'Calm reference', title: 'Rolltreppe KONE TravelMaster 110 im Westfield Mega Mall Hamburg', start: 8, end: 16, offset: 0, naturalEnd: false },
  // The 20s duration is a display fallback until YouTube supplies precise metadata.
  { id: '8FKGDKEwGf0', label: 'High-energy reference', title: 'The Big Bang Theory - Intro (Deutsch) 1080p HD', start: 5, end: 20, offset: 8, naturalEnd: true },
] as const;
export const REFERENCE_SPLIT = REFERENCE_CLIPS[1].offset;
export function referenceDuration(finalVideoDuration: number = REFERENCE_CLIPS[1].end) {
  const end = Number.isFinite(finalVideoDuration) && finalVideoDuration > REFERENCE_CLIPS[1].start ? finalVideoDuration : REFERENCE_CLIPS[1].end;
  return REFERENCE_SPLIT + end - REFERENCE_CLIPS[1].start;
}
export const REFERENCE_DURATION = referenceDuration();

export function referencePosition(time: number, finalVideoDuration: number = REFERENCE_CLIPS[1].end) {
  const bounded = Math.max(0, Math.min(referenceDuration(finalVideoDuration), time));
  const index = bounded < REFERENCE_SPLIT ? 0 : 1;
  const configured = REFERENCE_CLIPS[index];
  const clip = { ...configured, end: configured.naturalEnd ? referenceDuration(finalVideoDuration) - REFERENCE_SPLIT + configured.start : configured.end };
  return { index, clip, sourceTime: clip.start + bounded - clip.offset };
}

export function referenceTimeline(index: number, sourceTime: number, finalVideoDuration: number = REFERENCE_CLIPS[1].end) {
  const clip = REFERENCE_CLIPS[index];
  const length = clip.naturalEnd ? referenceDuration(finalVideoDuration) - clip.offset : clip.end - clip.start;
  return clip.offset + Math.max(0, Math.min(length, sourceTime - clip.start));
}

export function referenceVideoRequest(index: number, sourceTime: number = REFERENCE_CLIPS[index].start) {
  const clip = REFERENCE_CLIPS[index];
  return { videoId: clip.id, startSeconds: sourceTime, ...(!clip.naturalEnd ? { endSeconds: clip.end } : {}) };
}

export function referenceTimestamp(seconds: number) {
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`;
}
