import { useCallback, useEffect, useState } from 'react';
import { LoaderCircle } from 'lucide-react';
import type { LocalBundle } from '../lib/local-bundle';
import type { CompareColumn, CompareModel, Observed } from '../lib/observed';
import { parseObserved } from '../lib/observed';
import { parseLibrary, type Library } from '../lib/library';
import { fetchOptionalJson, openRealBundle } from '../lib/real-bundle';
import { probeSourceClip, SOURCE_CLIP_LABEL } from '../lib/source-clip';
import { useClipPlayer } from '../lib/use-clip-player';
import CompareFrame from './compare-frame';
import ClipPreview from './ClipPreview';
import ClipScorecard from './ClipScorecard';
import ResponseStrip from './ResponseStrip';
import ObservedPanel from './ObservedPanel';
import './library.css';

function ColumnBody({ bundle, library, observed }: { bundle: LocalBundle; library: Library | null; observed: Observed | null }) {
  const analysis = bundle.analysis;
  const [timeMs, setTimeMs] = useState(0);
  const [playing, setPlaying] = useState(false);
  const clip = useClipPlayer({ durationMs: analysis.duration_ms, playing, speed: 1, loop: false, onTime: setTimeMs, onPlaying: setPlaying });
  const { loadServer } = clip;
  const sourceClipUrl = bundle.sourceClipUrl;
  useEffect(() => {
    if (!sourceClipUrl) return;
    const controller = new AbortController();
    void probeSourceClip(sourceClipUrl, (url, init) => fetch(url, init), controller.signal)
      .then(found => { if (found && !controller.signal.aborted) loadServer(sourceClipUrl, SOURCE_CLIP_LABEL); });
    return () => controller.abort();
  }, [sourceClipUrl, loadServer]);
  const seek = (ms: number) => setTimeMs(clip.seek(ms));
  const togglePlay = useCallback(() => {
    if (!clip.active || !clip.ready) return;
    if (timeMs >= clip.endMs - 1) setTimeMs(clip.seek(0));
    setPlaying(value => !value);
  }, [timeMs, clip.active, clip.ready, clip.endMs, clip.seek]);
  const visibleKeys = analysis.channels.filter(channel => channel.default_visible).map(channel => channel.key);
  return <>
    {clip.source && <ClipPreview layout="side" analysis={analysis} player={clip} timeMs={timeMs} playing={playing} selected={null} visibleKeys={visibleKeys} onTogglePlay={togglePlay} onSeek={seek} />}
    {library && <ClipScorecard library={library} onSeek={seek} showCaveat={false} />}
    {library && <ResponseStrip library={library} durationMs={analysis.duration_ms} timeMs={timeMs} onSeek={seek} />}
    {observed && <ObservedPanel observed={observed} />}
  </>;
}

function ColumnLoader({ indexUrl, column }: { indexUrl: string; column: CompareColumn }) {
  const [state, setState] = useState<{ bundle: LocalBundle; library: Library | null; observed: Observed | null } | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    (async () => {
      const bundle = await openRealBundle(indexUrl, column.entry, controller.signal);
      const [libraryValue, observedValue] = await Promise.all([
        bundle.libraryUrl ? fetchOptionalJson(bundle.libraryUrl, controller.signal) : null,
        bundle.observedUrl && !bundle.lockbox ? fetchOptionalJson(bundle.observedUrl, controller.signal) : null,
      ]);
      if (controller.signal.aborted) return;
      setState({ bundle, library: parseLibrary(libraryValue, bundle.analysis.video_id), observed: parseObserved(observedValue, { lockbox: !!bundle.lockbox }) });
    })().catch(err => { if (!controller.signal.aborted) setError(err instanceof Error ? err.message : 'This clip could not be loaded.'); });
    return () => controller.abort();
  }, [indexUrl, column.entry]);
  if (error) return <p className="compare-error" role="alert">{error}</p>;
  if (!state) return <div className="compare-loading"><LoaderCircle size={20} className="spin" /><span>Loading clip</span></div>;
  return <ColumnBody bundle={state.bundle} library={state.library} observed={state.observed} />;
}

/** Internal side-by-side of one example pair (fell short vs beat expectations). */
export default function CompareView({ indexUrl, model, onBack, onLearned }: { indexUrl: string; model: CompareModel; onBack: () => void; onLearned: () => void }) {
  return <CompareFrame model={model} onBack={onBack} onLearned={onLearned} renderColumn={column => <ColumnLoader indexUrl={indexUrl} column={column} />} />;
}
