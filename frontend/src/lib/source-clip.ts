/**
 * Optional source clip served next to an approved real analysis index.
 *
 * The local research server serves `/clips/<video_id>.mp4` beside
 * `/demo-stage1/index.json`, so the clip is `../clips/<video_id>.mp4` relative
 * to the index. Clips are never bundled or committed; when the file is absent the
 * normal browser-only attach flow stays in place.
 */
export const SOURCE_CLIP_LABEL = 'Source clip (local server)';

const VIDEO_ID = /^[A-Za-z0-9][A-Za-z0-9_-]*$/;

export function resolveSourceClipUrl(indexUrl: string, videoId: string): string {
  if (!VIDEO_ID.test(videoId)) throw new Error('The clip video ID is invalid.');
  const base = new URL(indexUrl, typeof window === 'undefined' ? 'http://localhost/' : window.location.href);
  const resolved = new URL(`../clips/${videoId}.mp4`, base);
  if (resolved.origin !== base.origin) throw new Error('The source clip must use the analysis index origin.');
  return resolved.href;
}

type Fetch = (url: string, init?: RequestInit) => Promise<Response>;

function isVideo(response: Response): boolean {
  return /^video\//i.test((response.headers.get('content-type') ?? '').trim());
}

/**
 * True only for a real video response. A static host with SPA fallback answers a
 * missing clip with `200 text/html`, so the content type is the deciding check.
 */
export async function probeSourceClip(url: string, fetchImpl: Fetch = fetch, signal?: AbortSignal): Promise<boolean> {
  try {
    const head = await fetchImpl(url, { method: 'HEAD', signal, cache: 'no-store' });
    if (head.status === 200) return isVideo(head);
    if (head.status !== 405 && head.status !== 501) return false;
    const partial = await fetchImpl(url, { method: 'GET', signal, cache: 'no-store', headers: { Range: 'bytes=0-0' } });
    const ok = (partial.status === 200 || partial.status === 206) && isVideo(partial);
    await partial.body?.cancel().catch(() => undefined);
    return ok;
  } catch {
    return false;
  }
}
