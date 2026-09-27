import type { LocalBundle } from './local-bundle.ts';
import { openRemoteAnalysis } from './analysis-api.ts';
import { resolveRealBundleUrls, type RealBundleIndexEntry } from './real-analysis-index.ts';
import { resolveSourceClipUrl } from './source-clip.ts';
import { libraryUrlFor } from './library.ts';
import { isLockboxBundle, siblingUrl } from './observed.ts';

/** Open one approved real bundle with its optional sibling files (clip, library, observed). */
export async function openRealBundle(indexUrl: string, entry: RealBundleIndexEntry, signal?: AbortSignal): Promise<LocalBundle> {
  const urls = resolveRealBundleUrls(indexUrl, entry);
  const bundle = await openRemoteAnalysis(urls.analysisUrl, { performanceUrl: urls.performanceUrl, expectedPerformance: entry.performance, signal });
  let sourceClipUrl: string | undefined;
  try { sourceClipUrl = resolveSourceClipUrl(indexUrl, entry.video_id); } catch { sourceClipUrl = undefined; }
  const lockbox = isLockboxBundle(entry, bundle.analysis);
  return {
    ...bundle, sourceClipUrl, lockbox,
    libraryUrl: libraryUrlFor(urls.analysisUrl),
    // Sealed lockbox clips never request observed outcomes.
    observedUrl: lockbox ? undefined : siblingUrl(urls.analysisUrl, 'observed.json'),
  };
}

/** Optional JSON: a 404, network error or HTML fallback page all resolve to null. */
export async function fetchOptionalJson(url: string, signal?: AbortSignal): Promise<unknown> {
  try {
    const response = await fetch(url, { signal, headers: { Accept: 'application/json' } });
    if (!response.ok) return null;
    return await response.json();
  } catch { return null; }
}
