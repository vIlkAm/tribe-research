import type { Analysis } from '../data/analysis.types.ts';
import meta from '../../public/brain/brain.meta.json' with { type: 'json' };

/** Channel keys the fsaverage5 mesh maps its HCP-MMP1 region unions to (proxies_v0). */
export const BRAIN_CHANNEL_KEYS: readonly string[] = meta.channel_keys;
/** Display caveat shipped with the mesh; shown under the 3D view. */
export const BRAIN_DISPLAY_NOTE: string = meta.display;
const BRAIN_UNIT = 'z_within_clip';

/**
 * The 3D surface colours each region with its channel's within-clip z-score, so
 * it is enabled only when every channel the mesh knows is present with the same
 * key and unit. Anything else uses the contracted 2D atlas.
 */
export function supportsBrain3d(analysis: Analysis): boolean {
  return BRAIN_CHANNEL_KEYS.length > 0 && BRAIN_CHANNEL_KEYS.every(key => analysis.channels.some(channel =>
    channel.key === key && channel.unit === BRAIN_UNIT && Array.isArray(channel.values)));
}
