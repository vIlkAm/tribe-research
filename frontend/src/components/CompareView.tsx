import { useCallback, useEffect, useState } from 'react';
import { LoaderCircle } from 'lucide-react';
import type { LocalBundle } from '../lib/local-bundle';
import type { CompareColumn, CompareModel, Observed } from '../lib/observed';
import { formatMultiplier, parseObserved } from '../lib/observed';
import { parseLibrary, type Library } from '../lib/library';
import { fetchOptionalJson, openRealBundle } from '../lib/real-bundle';
import { probeSourceClip, SOURCE_CLIP_LABEL } from '../lib/source-clip';
import { useClipPlayer } from '../lib/use-clip-player';
import CompareFrame from './compare-frame';
import ClipScorecard from './ClipScorecard';
import ResponseStrip from './ResponseStrip';
import ObservedPanel from './ObservedPanel';
import StageVideo from './StageVideo';
import './library.css';
import './demo-layout.css';

interface Loaded { bundle: LocalBundle; library: Library | null; observed: Observed | null }

/** One column's data plus its own clip clock. Called once per column (always two). */
function useCompareColumn(indexUrl: string, column: CompareColumn) {
  const [state, setState] = useState<Loaded | null>(null);
  const [error, setError] = useState<string | null>(null);
  const durationMs = state?.bundle.analysis.duration_ms ?? column.entry.duration_ms;
  const [timeMs, setTimeMs] = useState(0);
  const [playing, setPlaying] = useState(false);
  const clip = useClipPlayer({ durationMs, playing, speed: 1, loop: false, onTime: setTimeMs, onPlaying: setPlaying });
  useEffect(() => {
    const controller = new AbortController();
    (async () => {
      const bundle = await openRealBundle(indexUrl, column.entry, controller.signal);
      const [libraryValue, observedValue] = await Promise.all([
        bundle.libraryUrl ? fetchOptionalJson(bundle.libraryUrl, controller.signal) : null,
        // Sealed lockbox clips never request observed outcomes.
        bundle.observedUrl && !bundle.lockbox ? fetchOptionalJson(bundle.observedUrl, controller.signal) : null,
      ]);
      if (controller.signal.aborted) return;
      setState({ bundle, library: parseLibrary(libraryValue, bundle.analysis.video_id), observed: parseObserved(observedValue, { lockbox: !!bundle.lockbox }) });
    })().catch(err => { if (!controller.signal.aborted) setError(err instanceof Error ? err.message : 'This clip could not be loaded.'); });
    return () => controller.abort();
  }, [indexUrl, column.entry]);
  const { loadServer } = clip;
  const sourceClipUrl = state?.bundle.sourceClipUrl;
  useEffect(() => {
    if (!sourceClipUrl) return;
    const controller = new AbortController();
    void probeSourceClip(sourceClipUrl, (url, init) => fetch(url, init), controller.signal)
      .then(found => { if (found && !controller.signal.aborted) loadServer(sourceClipUrl, SOURCE_CLIP_LABEL); });
    return () => controller.abort();
  }, [sourceClipUrl, loadServer]);
  const seek = useCallback((ms: number) => setTimeMs(clip.seek(ms)), [clip.seek]);
  const togglePlay = useCallback(() => {
    if (!clip.active || !clip.ready) return;
    if (timeMs >= clip.endMs - 1) setTimeMs(clip.seek(0));
    setPlaying(value => !value);
  }, [timeMs, clip.active, clip.ready, clip.endMs, clip.seek]);
  return { state, error, clip, timeMs, playing, seek, togglePlay };
}
type ColumnState = ReturnType<typeof useCompareColumn>;

function Stage({ column }: { column: ColumnState }) {
  const { state, error, clip, timeMs, playing, seek, togglePlay } = column;
  if (error) return <p className="compare-error" role="alert">{error}</p>;
  if (!state) return <div className="compare-loading"><LoaderCircle size={20} className="spin" /><span>Loading clip</span></div>;
  const analysis = state.bundle.analysis;
  return <>
    <StageVideo analysis={analysis} player={clip} timeMs={timeMs} playing={playing} onTogglePlay={togglePlay} onSeek={seek} />
    {state.library?.index ? <ResponseStrip compact library={state.library} durationMs={analysis.duration_ms} timeMs={timeMs} onSeek={seek} /> : <p className="compare-loading">No library comparison for this clip.</p>}
  </>;
}

/** Internal side-by-side of one example pair (fell short vs beat expectations). */
export default function CompareView({ indexUrl, model, onBack, onLearned }: { indexUrl: string; model: CompareModel; onBack: () => void; onLearned: () => void }) {
  const first = useCompareColumn(indexUrl, model.columns[0]);
  const second = useCompareColumn(indexUrl, model.columns[1]);
  const byRole = (column: CompareColumn) => column.role === model.columns[0].role ? first : second;
  return <CompareFrame model={model} onBack={onBack} onLearned={onLearned}
    renderHeader={column => {
      const observed = byRole(column).state?.observed;
      return observed?.views_vs_account_usual_x != null ? <span className="compare-observed-chip">Views vs this account’s recent usual: <strong>{formatMultiplier(observed.views_vs_account_usual_x)}</strong></span> : null;
    }}
    renderStage={column => <Stage column={byRole(column)} />}
    renderBelow={column => {
      const { state, seek } = byRole(column);
      if (!state) return null;
      return <>
        {state.library && <ClipScorecard library={state.library} onSeek={seek} showCaveat={false} />}
        {state.observed && <ObservedPanel observed={state.observed} />}
      </>;
    }} />;
}
