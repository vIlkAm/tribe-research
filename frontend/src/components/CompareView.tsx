import { useCallback, useEffect, useMemo, useState } from 'react';
import { LoaderCircle } from 'lucide-react';
import type { LocalBundle } from '../lib/local-bundle';
import type { CompareColumn, CompareModel, CompareSlot, Examples, Observed } from '../lib/observed';
import { compareModel, parseObserved } from '../lib/observed';
import { clipParam, demoClips, groupDemoClips, type DemoGroup } from '../lib/demo';
import type { RealBundleIndex } from '../lib/real-analysis-index';
import { ObservedRow } from './observed-row';
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

function Stage({ column, interpreterLine }: { column: ColumnState; interpreterLine?: string }) {
  const { state, error, clip, timeMs, playing, seek, togglePlay } = column;
  if (error) return <p className="compare-error" role="alert">{error}</p>;
  if (!state) return <div className="compare-loading"><LoaderCircle size={20} className="spin" /><span>Loading clip</span></div>;
  const analysis = state.bundle.analysis;
  return <>
    <StageVideo analysis={analysis} player={clip} timeMs={timeMs} playing={playing} onTogglePlay={togglePlay} onSeek={seek} />
    {state.library?.index ? <ResponseStrip compact library={state.library} durationMs={analysis.duration_ms} timeMs={timeMs} onSeek={seek} interpreterLine={interpreterLine} /> : <p className="compare-loading">No library comparison for this clip.</p>}
  </>;
}

function ComparePair({ indexUrl, model, groups, onChoose, onBack, onLearned, interpreterLine }: { indexUrl: string; model: CompareModel; groups: DemoGroup[]; onChoose: (slot: CompareSlot, videoId: string) => void; onBack: () => void; onLearned: () => void; interpreterLine?: string }) {
  const first = useCompareColumn(indexUrl, model.columns[0]);
  const second = useCompareColumn(indexUrl, model.columns[1]);
  const bySlot = (column: CompareColumn) => column.slot === 'a' ? first : second;
  return <CompareFrame model={model} groups={groups} onChoose={onChoose} onBack={onBack} onLearned={onLearned}
    renderHeader={column => {
      const observed = bySlot(column).state?.observed;
      return observed ? <ObservedRow observed={observed} /> : null;
    }}
    renderStage={column => <Stage column={bySlot(column)} interpreterLine={interpreterLine} />}
    renderBelow={column => {
      const { state, seek } = bySlot(column);
      if (!state) return null;
      return <>
        {state.library && <ClipScorecard library={state.library} onSeek={seek} showCaveat={false} />}
        {state.observed && <ObservedPanel observed={state.observed} />}
      </>;
    }} />;
}

/** Internal side-by-side of two clips the viewer picks (default: the example pair). */
export default function CompareView({ indexUrl, examples, index, onBack, onLearned, interpreterLine }: { indexUrl: string; examples: Examples; index: RealBundleIndex; onBack: () => void; onLearned: () => void; interpreterLine?: string }) {
  const [requested, setRequested] = useState(() => ({ a: clipParam(window.location.search, 'a'), b: clipParam(window.location.search, 'b') }));
  const model = useMemo(() => compareModel(examples, index, requested), [examples, index, requested]);
  const groups = useMemo(() => groupDemoClips(demoClips(index)), [index]);
  if (!model) return null;
  function choose(slot: CompareSlot, videoId: string) {
    const next = { a: model!.columns[0].clip.video_id, b: model!.columns[1].clip.video_id, [slot]: videoId };
    const url = new URL(window.location.href);
    url.searchParams.set('a', next.a); url.searchParams.set('b', next.b);
    history.replaceState(null, '', url);
    setRequested(next);
  }
  // Remount per pair so each column starts with a fresh clip player.
  return <ComparePair key={`${model.columns[0].clip.video_id}|${model.columns[1].clip.video_id}`} indexUrl={indexUrl} model={model} groups={groups} onChoose={choose} onBack={onBack} onLearned={onLearned} interpreterLine={interpreterLine} />;
}
