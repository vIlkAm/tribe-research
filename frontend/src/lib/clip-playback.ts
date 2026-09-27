/** Both displays use stimulus time. Never add the already-applied hemodynamic lag. */
export function clipEndMs(analysisDurationMs: number, mediaDurationSeconds?: number): number {
  return mediaDurationSeconds && Number.isFinite(mediaDurationSeconds) && mediaDurationSeconds > 0
    ? Math.min(analysisDurationMs, mediaDurationSeconds * 1000)
    : analysisDurationMs;
}

export function clipSeekMs(ms: number, endMs: number): number {
  return Math.max(0, Math.min(Number.isFinite(ms) ? ms : 0, Math.max(0, endMs - 1)));
}

export function clipFileError(file: { name: string; type: string; size: number }): string | null {
  if (!/\.(mp4|webm|mov|m4v)$/i.test(file.name) && !['video/mp4', 'video/webm', 'video/quicktime'].includes(file.type)) {
    return 'Choose an MP4, WebM or MOV video.';
  }
  if (!file.size) return 'This video file is empty. Choose another file.';
  return null;
}
