import type { Analysis } from '../data/analysis.types.ts';
import demo from '../../public/data/cd20b16879d630c4/analysis.json' with { type: 'json' };

/** The extension is deliberately limited to this fixture until contract issue #1 is agreed. */
export function compatibleDemo(analysis: Analysis): boolean {
  return analysis.synthetic && analysis.analysis_id === demo.analysis_id
    && analysis.video_id === demo.video_id
    && analysis.duration_ms === demo.duration_ms
    && JSON.stringify(analysis.provenance) === JSON.stringify(demo.provenance)
    && JSON.stringify(analysis.channels) === JSON.stringify(demo.channels);
}
