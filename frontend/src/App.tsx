import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ArrowUpRight, AudioLines, BarChart3, Box, Brain, Columns2, ChevronDown, CircleHelp, Crosshair, Expand, ExternalLink, Focus, FolderOpen, Info, Layers3, LoaderCircle, Pause, Play, Repeat2, RotateCcw, Rotate3D, Scan, SkipBack, Sparkles, Upload, X } from 'lucide-react';
import type { Analysis, Channel, Moment } from './data/analysis.types';
import { assertAnalysis, BUNDLE_PATH, formatTime, isGap, sampleAt, signalText, stepPath } from './lib/analysis';
import BrainAtlas from './components/BrainAtlas';
import Timeline from './components/Timeline';
import { assetUrl } from './lib/assets';
import { BRAIN_DISPLAY_NOTE, supportsBrain3d } from './lib/demo-compat';
import { probeSourceClip, SOURCE_CLIP_LABEL } from './lib/source-clip';
import AnalysisPicker from './components/AnalysisPicker';
import AnalysisIntake from './components/AnalysisIntake';
import PerformanceCard from './components/PerformanceCard';
import { useClipPlayer } from './lib/use-clip-player';
import type { LocalBundle } from './lib/local-bundle';
import ClipScorecard from './components/ClipScorecard';
import ResponseStrip from './components/ResponseStrip';
import StandoutMoments from './components/StandoutMoments';
import LearnedView from './components/LearnedView';
import CompareView from './components/CompareView';
import ObservedPanel from './components/ObservedPanel';
import { parseExamples, parseObserved, siblingUrl, type Examples, type Observed } from './lib/observed';
import { parsePatterns, type Patterns } from './lib/patterns';
import { parseTheory, type Theory } from './lib/theory';
import LibraryView from './components/LibraryView';
import { ObservedRow } from './components/observed-row';
import { fetchOptionalJson } from './lib/real-bundle';
import { parseRealBundleIndex, type RealBundleIndex } from './lib/real-analysis-index';
import { parseLearned, parseLibrary, verdictLabel, type Learned, type Library, type LibraryMoment } from './lib/library';
import { chooseDemoClip, clipParam, demoClips, momentStopMs, parseDemoMoment, type DemoClip, type DemoMoment } from './lib/demo';
import { openRealBundle } from './lib/real-bundle';
import { DemoClipPicker, DemoShell } from './components/demo-layout';
import StageVideo from './components/StageVideo';
import './components/comparison-demo.css';
import './components/library.css';
import './components/demo-layout.css';

const BrainScene = lazy(() => import('./components/BrainScene'));
const ComparisonDemo = lazy(() => import('./components/ComparisonDemo'));
const REPO = 'https://github.com/vIlkAm/tribe-research';

function Brand() {
  return <a className="brand" href="#" aria-label="ViralBrain Si home"><img src={assetUrl('favicon.svg')} alt="" /><span>Viral<span className="brand-light">Brain</span><span className="brand-dot">.</span><span className="brand-si">Si</span></span></a>;
}

function MiniSignal({ channel, durationMs }: { channel: Channel; durationMs: number }) {
  return <svg viewBox="0 0 74 27" width="74" height="27" aria-hidden="true"><path d={stepPath(channel.values, channel.hz, durationMs, 74, 27)} fill="none" stroke={channel.color} strokeWidth="1.15" /></svg>;
}

function MethodDialog({ open, onClose, analysis }: { open: boolean; onClose: () => void; analysis: Analysis }) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => { if (open) ref.current?.showModal(); else ref.current?.close(); }, [open]);
  return <dialog ref={ref} className="method-dialog" onCancel={onClose} onClick={e => { if (e.target === e.currentTarget) onClose(); }} aria-labelledby="method-title">
    <div className="dialog-head"><span className="eyebrow">BEHIND THE VISUALIZATION</span><button className="icon-button" onClick={onClose} aria-label="Close methodology"><X size={20} /></button></div>
    <h2 id="method-title">A window into predicted response.</h2>
    <p>This is an interactive research demo inspired by Meta’s TRIBE v2. It is independently built and is not affiliated with Meta.</p>
    {analysis.synthetic && <div className="method-callout"><Info size={19} /><div><strong>The included sample is synthetic.</strong><p>Synthetic data is used for interface testing, not measured brain activity. Consult the bundle provenance for how the sample was produced.</p></div></div>}
    <h3>Real anatomy. Region-level signals.</h3>
    <p>The 3D surface is FreeSurfer fsaverage5: 20,484 cortical vertices. HCP-MMP1 annotations map the seven proxy channels onto the same region groups the research proxies use. The 3D view is available when a bundle carries all seven channels as within-clip z-scores; other bundles use the contracted 2D atlas. Each region receives its channel’s value; this is not a raw per-vertex response map.</p>
    <h3>How to read the light</h3>
    <p>Amber is above a region’s average for this clip. Blue is below it. Brightness represents magnitude, not quality. Values are within-clip z-scores and cannot be compared across clips. Shimmer is an illustrative effect, not neuronal firing.</p>
    <h3>Timing and uncertainty</h3>
    <p>This bundle’s native resolution is {analysis.timing.native_tr_s} seconds, displayed on a {analysis.timing.display_hz} Hz sample-and-hold grid. Missing samples remain gaps. The onset window is {formatTime(analysis.timing.onset_window_ms[0])}–{formatTime(analysis.timing.onset_window_ms[1])}. {analysis.timing.note}</p>
    <p>Signal confidence and direction come from the bundle. Values are relative to this clip; they do not establish behavioral outcomes. The bundle contains no source video. You can attach a matching clip locally; its playback then drives the analysis timeline. When the local research server holds the analyzed clip, it is loaded automatically and labelled as the source clip. Time zero is the first video frame, with no additional lag applied. Attaching a clip does not run an analysis.</p>
    <div className="source-links"><a href={`${REPO}/tree/65ab40704ca484b453e2e2675d777423f325f97a/docs/sample_analysis`} target="_blank" rel="noreferrer">Sample & contract <ExternalLink size={14} /></a><a href="https://github.com/nilearn/nilearn/tree/0.12.1/nilearn/datasets/data/fsaverage5" target="_blank" rel="noreferrer">Brain anatomy <ExternalLink size={14} /></a><a href="https://aidemos.atmeta.com/tribev2" target="_blank" rel="noreferrer">Meta TRIBE v2 <ExternalLink size={14} /></a></div>
  </dialog>;
}

interface WorkspaceProps {
  analysis: Analysis; localBundle: LocalBundle | null; onOpen: () => void; onReset: () => void; onComparison: () => void; onAnalyze: () => void;
  learnedAvailable: boolean; onLearned: () => void; compareAvailable: boolean; onCompare: () => void; libraryAvailable: boolean; onLibrary: () => void; interpreterLine?: string;
  clips: DemoClip[]; currentClip: string | null; clipBusy: string | null; onChooseClip: (clip: DemoClip) => void; demoMoment: DemoMoment | null;
}

function Workspace({ analysis, localBundle, onOpen, onReset, onComparison, onAnalyze, learnedAvailable, onLearned, compareAvailable, onCompare, libraryAvailable, onLibrary, interpreterLine, clips, currentClip, clipBusy, onChooseClip, demoMoment }: WorkspaceProps) {
  const demoCompatible = supportsBrain3d(analysis);
  const [timeMs, setTimeMs] = useState(localBundle ? 0 : Math.min(4200, analysis.duration_ms - 1));
  const [playing, setPlaying] = useState(() => !matchMedia('(prefers-reduced-motion: reduce)').matches);
  const [speed, setSpeed] = useState(1);
  const [loop, setLoop] = useState(true);
  const [selected, setSelected] = useState<string | null>(null);
  const [visibleKeys, setVisibleKeys] = useState(() => analysis.channels.filter(channel => channel.default_visible).map(channel => channel.key));
  const [hovered, setHovered] = useState<string | null>(null);
  const [autoRotate, setAutoRotate] = useState(true);
  const [glow, setGlow] = useState(true);
  const [view, setView] = useState<'perspective' | 'front' | 'side' | 'top'>('perspective');
  const [resetKey, setResetKey] = useState(0);
  const [mode, setMode] = useState<'3d' | 'atlas'>(demoCompatible ? '3d' : 'atlas');
  const [brainReady, setBrainReady] = useState(false);
  const [brainFailed, setBrainFailed] = useState(!demoCompatible);
  const [methodOpen, setMethodOpen] = useState(false);
  const [allMoments, setAllMoments] = useState(false);
  const [activeMoment, setActiveMoment] = useState<string | null>(null);
  const [reducedMotion, setReducedMotion] = useState(() => matchMedia('(prefers-reduced-motion: reduce)').matches);
  const clip = useClipPlayer({ durationMs: analysis.duration_ms, playing, speed, loop, onTime: setTimeMs, onPlaying: setPlaying });
  const stageRef = useRef<HTMLElement>(null);
  const { loadServer } = clip;
  const sourceClipUrl = localBundle?.sourceClipUrl;
  useEffect(() => {
    if (!sourceClipUrl) return;
    const controller = new AbortController();
    void probeSourceClip(sourceClipUrl, (url, init) => fetch(url, init), controller.signal)
      .then(found => { if (found && !controller.signal.aborted) loadServer(sourceClipUrl, SOURCE_CLIP_LABEL); });
    return () => controller.abort();
  }, [sourceClipUrl, loadServer]);
  const [library, setLibrary] = useState<Library | null>(null);
  const libraryUrl = localBundle?.libraryUrl;
  useEffect(() => {
    setLibrary(null);
    if (!libraryUrl) return;
    const controller = new AbortController();
    // Optional file: a missing one (404, or HTML from SPA fallback) simply shows nothing.
    fetch(libraryUrl, { signal: controller.signal, headers: { Accept: 'application/json' } })
      .then(response => response.ok ? response.json() : null)
      .then(value => { if (!controller.signal.aborted) setLibrary(parseLibrary(value, analysis.video_id)); })
      .catch(() => undefined);
    return () => controller.abort();
  }, [libraryUrl, analysis.video_id]);
  const [observed, setObserved] = useState<Observed | null>(null);
  const observedUrl = localBundle?.lockbox ? undefined : localBundle?.observedUrl;
  const lockbox = !!localBundle?.lockbox;
  useEffect(() => {
    setObserved(null);
    if (!observedUrl || lockbox) return;
    const controller = new AbortController();
    void fetchOptionalJson(observedUrl, controller.signal).then(value => { if (!controller.signal.aborted) setObserved(parseObserved(value, { lockbox })); });
    return () => controller.abort();
  }, [observedUrl, lockbox]);
  const plainChannels = new Map((library?.channels ?? []).map(channel => [channel.key, channel]));
  const playbackRef = useRef({ timeMs, speed, loop, duration: analysis.duration_ms });
  playbackRef.current = { timeMs, speed, loop, duration: analysis.duration_ms };

  useEffect(() => {
    const media = matchMedia('(prefers-reduced-motion: reduce)');
    const update = () => setReducedMotion(media.matches);
    media.addEventListener('change', update);
    return () => media.removeEventListener('change', update);
  }, []);

  useEffect(() => {
    if (!playing || clip.active) return;
    let frame = 0, previous = performance.now(), accumulated = 0;
    const tick = (now: number) => {
      const delta = document.hidden ? 0 : Math.min(now - previous, 100);
      previous = now;
      accumulated += delta;
      if (accumulated > 30) {
        const state = playbackRef.current;
        let next = state.timeMs + accumulated * state.speed;
        accumulated = 0;
        if (next >= state.duration) {
          if (state.loop) next %= state.duration;
          else { next = state.duration - 1; setPlaying(false); }
        }
        playbackRef.current.timeMs = next;
        setTimeMs(next);
      }
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [playing, clip.active]);

  const togglePlay = useCallback(() => {
    if (clip.active && !clip.ready) return;
    if (timeMs >= clip.endMs - 1) setTimeMs(clip.seek(0));
    setActiveMoment(null);
    setMomentStop(null);
    setPlaying(v => !v);
  }, [timeMs, clip.active, clip.ready, clip.endMs, clip.seek]);

  // "Play this moment": seek to the window, play, pause at its end. Any pause or seek cancels it.
  const [momentStop, setMomentStop] = useState<number | null>(null);
  function playMoment() {
    if (!demoMoment || (clip.active && !clip.ready)) return;
    setTimeMs(clip.seek(demoMoment.start_ms));
    setActiveMoment(null);
    setMomentStop(momentStopMs(demoMoment, clip.endMs));
    setPlaying(true);
  }
  useEffect(() => {
    if (momentStop === null) return;
    if (!playing) { setMomentStop(null); return; }
    if (timeMs >= momentStop) { setPlaying(false); setMomentStop(null); }
  }, [timeMs, playing, momentStop]);

  useEffect(() => {
    const keyboard = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement;
      if (document.querySelector('dialog[open]') || target.closest('button, input, select, textarea, a, [role="slider"]')) return;
      if (event.code === 'Space') { event.preventDefault(); togglePlay(); }
    };
    document.addEventListener('keydown', keyboard);
    return () => document.removeEventListener('keydown', keyboard);
  }, [togglePlay, methodOpen]);

  const seek = (ms: number) => { setTimeMs(clip.seek(ms)); setActiveMoment(null); setMomentStop(null); };
  const onReady = useCallback(() => { setBrainReady(true); }, []);
  const onError = useCallback(() => { setBrainFailed(true); setBrainReady(true); setMode('atlas'); }, []);
  const values = Object.fromEntries(analysis.channels.map(c => [c.key, isGap(analysis, timeMs) ? null : sampleAt(c, timeMs, analysis.duration_ms)]));
  const inspected = analysis.channels.find(c => c.key === (hovered || selected));
  const focused = analysis.channels.find(c => c.key === selected);
  const focusedValue = focused ? values[focused.key] : null;
  const currentMoment = analysis.moments.find(m => m.id === activeMoment) ?? analysis.moments.find(m => !m.in_onset_window && m.start_ms <= timeMs && timeMs < m.end_ms);
  const sortedMoments = [...analysis.moments].filter(m => !m.in_onset_window).sort((a, b) => b.severity - a.severity);
  const visibleMoments = allMoments ? sortedMoments : sortedMoments.slice(0, 3);
  const onset = timeMs >= analysis.timing.onset_window_ms[0] && timeMs < analysis.timing.onset_window_ms[1];
  const gap = isGap(analysis, timeMs);

  function chooseMoment(moment: Moment) {
    setPlaying(false);
    setTimeMs(clip.seek(moment.start_ms));
    setActiveMoment(moment.id);
    setSelected(moment.channels[0] ?? null);
  }
  function chooseStandout(moment: LibraryMoment) {
    setPlaying(false);
    setTimeMs(clip.seek(moment.start_ms));
    setActiveMoment(null);
    setSelected(moment.channels.find(key => analysis.channels.some(c => c.key === key)) ?? null);
  }
  const plainLabel = (key: string) => plainChannels.get(key)?.label_plain ?? analysis.channels.find(c => c.key === key)?.label ?? '';
  function setPreset(preset: typeof view) { setView(preset); setResetKey(v => v + 1); setAutoRotate(false); }

  const analysisDetails = <details className="analysis-details"><summary>Analysis details · {analysis.analysis_id} · research only{analysis.quality.warnings.length > 0 ? ` · ${analysis.quality.warnings.length} quality notes` : ''}</summary><div><p>Source: {analysis.source_name || analysis.video_id}. Native resolution: {analysis.timing.native_tr_s}s · display: {analysis.timing.display_hz} Hz · {analysis.timing.n_display_samples} samples.</p><p>Proxy version: {analysis.provenance.proxies_version}. Normalization: {analysis.provenance.normalization}. Explanation model: {analysis.model_versions.explain}.</p><p>{analysis.timing.note}</p>{analysis.quality.warnings.map((warning, i) => <p key={i}>{warning}</p>)}</div></details>;
  const withinClipMoments = <><section className="moments-section" aria-label="Moments within this clip"><div className="moments-heading"><div><h2>Moments within this clip <span>{sortedMoments.length}</span></h2></div><button className="text-button" onClick={() => setAllMoments(v => !v)}>{allMoments ? 'Show highlights' : 'View all moments'}<ChevronDown size={15} className={allMoments ? 'flipped' : ''} /></button></div><div className="moment-grid">{visibleMoments.map((moment, i) => <button className={`moment-card ${moment.status} ${currentMoment?.id === moment.id ? 'current' : ''}`} key={moment.id} onClick={() => chooseMoment(moment)}><div className="moment-meta"><span className="moment-index">{String(i + 1).padStart(2, '0')}</span><span className="mono">{formatTime(moment.start_ms)} — {formatTime(moment.end_ms)}</span><span className="moment-arrow"><ArrowUpRight size={17} /></span></div><h3>{moment.title}</h3><p>{moment.hypothesis || moment.description}</p><div className="moment-bottom"><span>{moment.status === 'edit_hypothesis_untested' ? 'Untested hypothesis' : 'Observation'}</span><span>{moment.channels.map(key => analysis.channels.find(c => c.key === key)?.label.replace(' proxy', '')).join(' · ')}</span></div></button>)}</div></section>
    {currentMoment && <div className="moment-detail"><Info size={16} /><div><p><strong>{currentMoment.title}.</strong> {currentMoment.description}</p>{currentMoment.hypothesis && <p>Untested hypothesis: {currentMoment.hypothesis}</p>}<ul>{currentMoment.evidence.map((evidence, i) => <li key={i}>{evidence.replaceAll('_', ' ')}</li>)}</ul>{currentMoment.test_metric && <p>Test against: {currentMoment.test_metric}</p>}<small>Confidence: {currentMoment.confidence} · relative to this clip</small></div></div>}</>;
  const performanceCard = <PerformanceCard performance={analysis.performance} synthetic={analysis.synthetic} />;

  const momentButton = demoMoment && <button type="button" className="moment-play" disabled={clip.active && !clip.ready} onClick={playMoment}><Play size={13} fill="currentColor" /> Play this moment <span>{demoMoment.label}</span></button>;
  const brainPanel = <section ref={stageRef} className={`brain-stage ${gap ? 'has-gap' : ''}`} aria-label="Brain visualization">
    <div className="stage-top"><div className="stage-title"><span className="stage-subtitle">Predicted brain response</span></div><div className="view-tabs" role="group" aria-label="Brain display mode"><button className={mode === '3d' ? 'active' : ''} onClick={() => { if (!brainFailed) setMode('3d'); }} disabled={brainFailed} aria-pressed={mode === '3d'}><Box size={15} /> 3D</button><button className={mode === 'atlas' ? 'active' : ''} onClick={() => setMode('atlas')} aria-pressed={mode === 'atlas'}><Layers3 size={15} /> Atlas</button></div></div>
    <div className="brain-view" style={{ display: mode === '3d' ? undefined : 'none' }}>
      {!brainFailed && <Suspense fallback={null}><BrainScene values={values} selected={selected} autoRotate={autoRotate} glow={glow} playing={playing} visible={mode === '3d'} view={view} resetKey={resetKey} reducedMotion={reducedMotion} onHover={setHovered} onSelect={setSelected} onReady={onReady} onError={onError} /></Suspense>}
      {!brainReady && <div className="brain-loading"><LoaderCircle size={22} className="spin" /><span>Loading cortical surface</span></div>}
    </div>
    {mode === 'atlas' && <BrainAtlas analysis={analysis} timeMs={timeMs} onHover={setHovered} resolveAsset={localBundle?.resolveAsset} />}
    <div className="stage-context"><span className={`status-dot ${playing ? 'pulsing' : ''}`} />{gap ? 'No prediction' : onset ? 'Onset response' : playing ? 'Playing analysis' : 'Exploring response'}<span className="mono">{formatTime(timeMs)}</span></div>
    <div className={`brain-tooltip ${inspected && mode === '3d' ? 'visible' : ''}`} aria-live="polite">{inspected && <><span style={{ color: inspected.color }}>{inspected.label}</span><strong>{signalText(values[inspected.key])}<small> z</small></strong><p>{inspected.basis.regions_text}</p></>}</div>
    <div className="brain-tools"><div className="camera-presets" role="group" aria-label="Camera views"><button aria-label="Perspective view" title="Perspective view" disabled={mode !== '3d'} aria-pressed={view === 'perspective'} onClick={() => setPreset('perspective')}><Box size={16} /></button><button aria-label="Front view" title="Front view" disabled={mode !== '3d'} aria-pressed={view === 'front'} onClick={() => setPreset('front')}><Focus size={16} /></button><button aria-label="Side view" title="Side view" disabled={mode !== '3d'} aria-pressed={view === 'side'} onClick={() => setPreset('side')}><Scan size={16} /></button><button aria-label="Top view" title="Top view" disabled={mode !== '3d'} aria-pressed={view === 'top'} onClick={() => setPreset('top')}><Layers3 size={16} /></button></div><button className="icon-button" title="Expand brain view" aria-label="Expand brain view" onClick={() => { if (!document.fullscreenElement) void stageRef.current?.requestFullscreen?.(); else void document.exitFullscreen?.(); }}><Expand size={17} /></button></div>
    <div className="stage-bottom"><div className="brain-options"><button className={autoRotate && !reducedMotion ? 'enabled' : ''} onClick={() => setAutoRotate(v => !v)} aria-pressed={autoRotate && !reducedMotion} disabled={reducedMotion || mode !== '3d'}><Rotate3D size={16} /> Auto-rotate<span className="toggle-track"><i /></span></button><button className={glow ? 'enabled' : ''} onClick={() => setGlow(v => !v)} aria-pressed={glow} disabled={mode !== '3d'}><Sparkles size={15} /> Glow</button></div></div>
    {mode === '3d' && <div className="brain-scale"><span>Below average</span><i /><span>Above average</span></div>}
    <p className="brain-caption"><span>{analysis.synthetic ? 'Synthetic sample · not measured brain activity' : 'TRIBE v2 prediction · average subject · research proxy, not measured brain activity'}</span>{mode === '3d' && <span>{BRAIN_DISPLAY_NOTE}</span>}</p>
  </section>;
  const topbar = <header className="demo-topbar">
    <Brand />
    <DemoClipPicker clips={clips} current={currentClip} busy={clipBusy} onChoose={onChooseClip} />
    {localBundle && !currentClip && <span className="demo-current" title={localBundle.filename}><FolderOpen size={13} /><span>{localBundle.filename}</span><button type="button" onClick={onReset} aria-label="Close this analysis"><X size={12} /></button></span>}
    {analysis.synthetic && <span className="synthetic-tag">SYNTHETIC SAMPLE</span>}
    <div className="demo-topbar-actions">
      {learnedAvailable && <a className="topbar-button" href="?view=learned" onClick={event => { event.preventDefault(); onLearned(); }}><Brain size={14} /> Learned</a>}
      {compareAvailable && <a className="topbar-button" href="?view=compare" onClick={event => { event.preventDefault(); onCompare(); }}><Columns2 size={14} /> Compare</a>}
      {libraryAvailable && <a className="topbar-button" href="?view=library" onClick={event => { event.preventDefault(); onLibrary(); }}><BarChart3 size={14} /> Library</a>}
      <span className="research-chip" title="TRIBE v2 is CC-BY-NC: research use only">Research preview</span>
      <button type="button" className="topbar-icon" onClick={onOpen} aria-label="Open analysis" title="Open analysis"><FolderOpen size={15} /></button>
      <button type="button" className="topbar-button is-secondary" onClick={onAnalyze}><Upload size={14} /> Upload</button>
      <button type="button" className="topbar-icon" aria-label="About the demo" title="About the demo" onClick={() => setMethodOpen(true)}><CircleHelp size={16} /></button>
    </div>
  </header>;
  const below = <>
    {localBundle && localBundle.warnings.length > 0 && <div className="local-asset-notes" role="status">{localBundle.warnings.map(note => <p key={note}>{note}</p>)}</div>}
    {library && <ClipScorecard library={library} onSeek={seek} />}
    {observed && <ObservedPanel observed={observed} />}
    <div className="below-signals">
      <section className="timeline-panel" aria-label="Response timeline"><div className="timeline-header"><div><h2>Signals over time</h2></div><div className="timeline-meta"><span><i className="zone-key" /> Moments</span><span>Time (seconds) · within-clip z-score (0 = this clip’s average)</span></div></div><Timeline analysis={analysis} timeMs={timeMs} selected={selected} onSeek={seek} visibleKeys={visibleKeys} onToggleChannel={key => setVisibleKeys(keys => keys.includes(key) ? keys.filter(value => value !== key) : [...keys, key])} /><div className="transport"><div className="playback-controls"><button className="icon-button" aria-label="Restart sample" title="Restart sample" onClick={() => { seek(0); }}><SkipBack size={17} /></button><button className="play-button" disabled={clip.active && !clip.ready} onClick={togglePlay} aria-label={playing ? 'Pause sample' : 'Play sample'}>{playing ? <Pause size={18} fill="currentColor" /> : <Play size={18} fill="currentColor" />}</button><span className="playback-time mono">{formatTime(timeMs)}<span> / {formatTime(analysis.duration_ms)}</span></span><button className={`icon-button loop-button ${loop ? 'active' : ''}`} onClick={() => setLoop(v => !v)} aria-label="Loop playback" title="Loop playback" aria-pressed={loop}><Repeat2 size={17} /></button><button className="speed-button mono" onClick={() => setSpeed(v => v === 0.5 ? 1 : v === 1 ? 2 : 0.5)} aria-label={`Playback speed ${speed} times. Click to change.`}>{speed}×</button></div><span className="sample-duration">{clip.active ? 'Clip + analysis playback' : 'Analysis playback'}<span>{clip.active ? 'Video drives the shared timeline' : 'Attach a clip to watch in sync'}</span></span></div></section>
      <aside className="signal-panel" aria-label="Brain signals"><div className="panel-heading"><div><h2>Brain signals</h2></div><AudioLines size={20} /></div><div className="signal-context"><span>{library && plainChannels.size ? 'Whole clip vs your library' : 'Relative to this clip'}</span><span className="mono">{formatTime(timeMs)} <span>/ {formatTime(analysis.duration_ms)}</span></span></div><div className="signal-list">
        {analysis.channels.map(channel => {
          const plain = plainChannels.get(channel.key);
          return <button key={channel.key} className={`signal-row ${selected === channel.key ? 'selected' : ''}`} onClick={() => setSelected(selected === channel.key ? null : channel.key)} aria-pressed={selected === channel.key} style={{ '--signal': channel.color } as React.CSSProperties} title={plain ? `${channel.label}: select to light up its brain regions` : `Isolate ${channel.label}`}><span className="signal-color" /><span className="signal-name">{plain ? plain.label_plain : channel.label}</span><MiniSignal channel={channel} durationMs={analysis.duration_ms} />{plain ? plain.verdict && <span className={`verdict-chip ${plain.verdict}`}>{verdictLabel(plain.verdict)}</span> : <span className="signal-number">{signalText(values[channel.key])}<small>z</small></span>}</button>;
        })}
      </div>{plainChannels.size > 0 && <details className="technical-detail"><summary>Technical detail</summary><table className="technical-table"><tbody>{analysis.channels.map(channel => <tr key={channel.key}><td>{channel.label}</td><td>{signalText(values[channel.key])} z</td></tr>)}</tbody></table><p className="technical-note">Within-clip z-scores at {formatTime(timeMs)}: 0 is this clip’s own average, so they cannot be compared across clips. Strong / Typical / Weak compare the whole clip with your library.</p></details>}<div className="signal-selection"><span><Crosshair size={14} />{selected ? 'One signal isolated' : 'Select a signal to isolate its regions'}</span>{selected && <button onClick={() => setSelected(null)}>Clear <X size={12} /></button>}</div><div className="signal-insight"><span className="eyebrow">{selected ? 'SELECTED SIGNAL' : 'READING THE RESPONSE'}</span><p>{focused ? focused.copy.tooltip : 'Each signal is a group of brain regions. Select one to light it up on the brain and the timeline.'}</p>{focused && <div className="signal-evidence"><p>{focusedValue == null ? 'No prediction at this time.' : focusedValue > 0 ? focused.copy.rise : focusedValue < 0 ? focused.copy.fall : 'At this clip’s average.'}</p><p>{focused.direction_note}</p><span>Confidence: {focused.confidence} · source: {focused.source_model}</span></div>}<span className="research-label"><Info size={13} /> {focused ? focused.research_status.replaceAll('_', ' ') : 'Research proxies · relative to this clip'}</span></div><div className="behavior-state"><span>Behavior predictions</span><span>{analysis.predictions.status.replaceAll('_', ' ')} <Info size={13} /></span>{analysis.predictions.status === 'not_available' ? <p>No behavior predictions yet. {analysis.predictions.reason || 'A trained behavior model is required.'}</p> : <p>Behavior metrics are supplied; this research demo does not yet interpret their units or calibration.</p>}</div></aside>
    </div>
    {library && <StandoutMoments library={library} timeMs={timeMs} labelFor={plainLabel} onChoose={chooseStandout} />}
    {library?.moments.length ? <details className="within-clip-moments"><summary>All within-clip moments (technical)</summary>{withinClipMoments}</details> : withinClipMoments}
    {performanceCard}
    {analysisDetails}
  </>;
  const footer = <footer><span><img src={assetUrl('favicon.svg')} alt="" /> Built for the curious.</span><div><span className="footer-license">Research preview · non-commercial (TRIBE CC-BY-NC)</span><span className="footer-dot">·</span><span>{analysis.schema_version}</span><span className="footer-dot">·</span><a href="?demo=comparison" onClick={event => { event.preventDefault(); onComparison(); }}>Video preview</a><span className="footer-dot">·</span><a href={REPO} target="_blank" rel="noreferrer">Research source <ExternalLink size={11} /></a><span className="footer-dot">·</span><button onClick={() => setMethodOpen(true)}>Data & methodology <ArrowUpRight size={13} /></button></div></footer>;

  return <>
    <DemoShell topbar={topbar}
      video={<StageVideo analysis={analysis} player={clip} timeMs={timeMs} playing={playing} onTogglePlay={togglePlay} onSeek={seek} footer={!library?.index ? momentButton : undefined} />}
      brain={brainPanel}
      metrics={observed ? <ObservedRow observed={observed} /> : undefined}
      strip={library?.index ? <ResponseStrip compact library={library} durationMs={analysis.duration_ms} timeMs={timeMs} onSeek={seek} highlight={demoMoment} action={momentButton} interpreterLine={interpreterLine} /> : undefined}
      below={below} footer={footer} />
    <MethodDialog analysis={analysis} open={methodOpen} onClose={() => setMethodOpen(false)} />
  </>;
}

export default function App() {
  const [comparison, setComparison] = useState(() => new URLSearchParams(window.location.search).get('demo') === 'comparison');
  const [view, setView] = useState<string | null>(() => new URLSearchParams(window.location.search).get('view'));
  const learnedView = view === 'learned';
  const realIndexUrl = import.meta.env.VITE_REAL_ANALYSIS_INDEX as string | undefined;
  const [realIndex, setRealIndex] = useState<RealBundleIndex | null>(null);
  const [examples, setExamples] = useState<Examples | null>(null);
  const [examplesChecked, setExamplesChecked] = useState(!realIndexUrl);
  const clips = useMemo(() => demoClips(realIndex), [realIndex]);
  const compareAvailable = !!examples && !!realIndex && clips.length >= 2;
  const [patterns, setPatterns] = useState<Patterns | null>(null);
  const [patternsChecked, setPatternsChecked] = useState(false);
  const [theory, setTheory] = useState<Theory | null>(null);
  const [clipBusy, setClipBusy] = useState<string | null>(null);
  const [demoReady, setDemoReady] = useState(!realIndexUrl);
  const clipOperation = useRef(0);
  const landed = useRef(false);
  const [learned, setLearned] = useState<Learned | null>(null);
  const [learnedChecked, setLearnedChecked] = useState(false);
  const [sample, setSample] = useState<Analysis | null>(null);
  const [localBundle, setLocalBundle] = useState<LocalBundle | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);
  const [pickerOpen, setPickerOpen] = useState(false);
  const [intakeOpen, setIntakeOpen] = useState(false);
  const [revision, setRevision] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    setError(null);
    fetch(assetUrl(BUNDLE_PATH), { signal: controller.signal })
      .then(response => { if (!response.ok) throw new Error('The sample analysis could not be loaded.'); return response.json(); })
      .then(data => { assertAnalysis(data); setSample(data); })
      .catch(error => { if (error.name !== 'AbortError') setError(error.message); });
    return () => controller.abort();
  }, [attempt]);
  useEffect(() => () => localBundle?.dispose(), [localBundle]);
  useEffect(() => {
    // Optional internal research summary served next to index.html; absent → no button, no page content.
    const controller = new AbortController();
    fetch(assetUrl('learned.json'), { signal: controller.signal, headers: { Accept: 'application/json' } })
      .then(response => response.ok ? response.json() : null)
      .then(value => { if (!controller.signal.aborted) setLearned(parseLearned(value)); })
      .catch(() => undefined)
      .finally(() => { if (!controller.signal.aborted) setLearnedChecked(true); });
    return () => controller.abort();
  }, []);
  useEffect(() => {
    // Optional internal library-wide patterns served next to index.html; absent → no button.
    const controller = new AbortController();
    void fetchOptionalJson(assetUrl('library_patterns.json'), controller.signal)
      .then(value => { if (!controller.signal.aborted) setPatterns(parsePatterns(value)); })
      .finally(() => { if (!controller.signal.aborted) setPatternsChecked(true); });
    // Optional library statistics (nvi.theory.v0); supersede the tier table on the Library page.
    void fetchOptionalJson(assetUrl('library_theory.json'), controller.signal)
      .then(value => { if (!controller.signal.aborted) setTheory(parseTheory(value)); });
    return () => controller.abort();
  }, []);
  useEffect(() => {
    // Optional internal example pair next to the approved index; absent → no button.
    if (!realIndexUrl) return;
    const controller = new AbortController();
    const examplesUrl = siblingUrl(realIndexUrl, 'examples.json');
    void Promise.all([fetchOptionalJson(realIndexUrl, controller.signal), examplesUrl ? fetchOptionalJson(examplesUrl, controller.signal) : null])
      .then(([indexValue, examplesValue]) => {
        if (controller.signal.aborted) return;
        try { setRealIndex(parseRealBundleIndex(indexValue)); } catch { setRealIndex(null); }
        setExamples(parseExamples(examplesValue));
      })
      .finally(() => { if (!controller.signal.aborted) setExamplesChecked(true); });
    return () => controller.abort();
  }, [realIndexUrl]);
  function showView(next: 'learned' | 'compare' | 'library' | null) {
    const url = new URL(window.location.href);
    if (next) url.searchParams.set('view', next); else url.searchParams.delete('view');
    history.replaceState(null, '', url);
    setView(next);
    window.scrollTo({ top: 0 });
  }
  const showLearned = (show: boolean) => showView(show ? 'learned' : null);
  function setClipUrl(videoId: string | null) {
    const url = new URL(window.location.href);
    if (videoId) url.searchParams.set('clip', videoId); else url.searchParams.delete('clip');
    history.replaceState(null, '', url);
  }
  function openBundle(bundle: LocalBundle) {
    setLocalBundle(bundle);
    setRevision(value => value + 1);
    const demo = bundle.origin === 'remote' && clips.some(clip => clip.video_id === bundle.analysis.video_id);
    setClipUrl(demo ? bundle.analysis.video_id : null);
  }
  async function openDemoClip(clip: DemoClip) {
    const id = ++clipOperation.current;
    setClipBusy(clip.video_id);
    try {
      const bundle = await openRealBundle(realIndexUrl!, clip.entry);
      if (id !== clipOperation.current) { bundle.dispose(); return; }
      setLocalBundle(bundle);
      setRevision(value => value + 1);
      setClipUrl(clip.video_id);
      window.scrollTo({ top: 0 });
    } catch {
      // Keep the current page; a clip that fails to open simply stays unselected.
    } finally {
      if (id === clipOperation.current) setClipBusy(null);
    }
  }
  // Land in demo mode: open the deep-linked (?clip=) or first index clip instead of the synthetic sample.
  useEffect(() => {
    if (!realIndexUrl || !examplesChecked || landed.current) return;
    landed.current = true;
    const first = chooseDemoClip(clips, clipParam(window.location.search));
    if (!first) { setDemoReady(true); return; }
    void openDemoClip(first).finally(() => setDemoReady(true));
  }, [realIndexUrl, examplesChecked, clips]);
  const currentClip = localBundle?.origin === 'remote' ? clips.find(clip => clip.video_id === localBundle.analysis.video_id) ?? null : null;
  const demoMoment = currentClip ? parseDemoMoment(currentClip.entry.demo_moment, localBundle!.analysis.duration_ms) : null;
  function showComparison(show: boolean) {
    const url = new URL(window.location.href);
    if (show) url.searchParams.set('demo', 'comparison'); else url.searchParams.delete('demo');
    history.replaceState(null, '', url);
    setComparison(show);
    window.scrollTo({ top: 0 });
  }
  const analysis = localBundle?.analysis ?? sample;
  const openPicker = () => setPickerOpen(true);
  const reset = () => {
    const first = chooseDemoClip(clips, null);
    if (first) { void openDemoClip(first); return; }
    setLocalBundle(null); setRevision(value => value + 1); setClipUrl(null);
  };
  const picker = <AnalysisPicker open={pickerOpen} onClose={() => setPickerOpen(false)} onOpen={openBundle} />;
  const intake = <AnalysisIntake open={intakeOpen} onClose={() => setIntakeOpen(false)} onOpen={openBundle} onOpenExport={() => setPickerOpen(true)} />;
  if (learnedView) return learned || learnedChecked ? <LearnedView learned={learned} onBack={() => showLearned(false)} /> : <div className="loading-screen"><LoaderCircle className="spin" /><p>Loading the research summary</p></div>;
  if (view === 'compare') {
    if (compareAvailable && realIndexUrl) return <CompareView indexUrl={realIndexUrl} examples={examples!} index={realIndex!} onBack={() => showView(null)} onLearned={() => showView('learned')} interpreterLine={theory?.interpreter_line || patterns?.interpreter_line || undefined} />;
    if (!examplesChecked) return <div className="loading-screen"><LoaderCircle className="spin" /><p>Loading the example pair</p></div>;
    return <div className="learned-page"><header className="learned-top"><button type="button" className="learned-back" onClick={() => showView(null)}>← Back to the analysis</button><span className="internal-badge">Internal research view</span></header><main className="learned-main"><p className="learned-empty">The example pair is not available on this server.</p></main></div>;
  }
  if (view === 'library') {
    if (patterns) return <LibraryView indexUrl={realIndexUrl} clips={clips} patterns={patterns} theory={theory} onBack={() => showView(null)} onOpenClip={clip => { showView(null); void openDemoClip(clip); }} />;
    if (!patternsChecked || !examplesChecked) return <div className="loading-screen"><LoaderCircle className="spin" /><p>Loading the library view</p></div>;
    return <div className="learned-page"><header className="learned-top"><button type="button" className="learned-back" onClick={() => showView(null)}>← Back to the analysis</button><span className="internal-badge">Internal research view · exploratory</span></header><main className="learned-main"><p className="learned-empty">The library view is not available on this server.</p></main></div>;
  }
  if (comparison) return <Suspense fallback={<div className="loading-screen"><LoaderCircle className="spin" /><p>Preparing the comparison</p></div>}><ComparisonDemo onBack={() => showComparison(false)} /></Suspense>;
  if (!demoReady && !localBundle) return <div className="loading-screen"><Brand /><LoaderCircle size={24} className="spin" /><p>Opening the demo clip</p></div>;
  if (!analysis) return <><div className="loading-screen"><Brand />{error ? <><h1>Couldn’t load the analysis.</h1><p>{error}</p><button className="retry-button" onClick={() => setAttempt(v => v + 1)}><RotateCcw size={17} /> Try again</button><button className="open-analysis-button" onClick={openPicker}><FolderOpen size={15} /> Open analysis</button></> : <><LoaderCircle size={24} className="spin" /><p>Preparing your research workspace</p></>}</div>{picker}{intake}</>;
  if (analysis.status !== 'complete') return <><div className="loading-screen"><Brand /><h1>{analysis.status === 'failed' ? 'Analysis failed.' : analysis.status === 'queued' ? 'Analysis queued.' : 'Analysis is processing.'}</h1><p>{analysis.analysis_id} · {localBundle ? 'Open a completed export when it is available.' : 'Results appear when a completed bundle is available.'}</p>{analysis.synthetic && <span className="synthetic-tag">SYNTHETIC SAMPLE</span>}{analysis.quality.warnings.map((warning, i) => <p key={i}>{warning}</p>)}<div className="analysis-state-actions"><button className="open-analysis-button" onClick={openPicker}><FolderOpen size={15} /> Open analysis</button>{localBundle ? <button className="retry-button" onClick={reset}>Back to demo</button> : <button className="retry-button" onClick={() => { setSample(null); setAttempt(v => v + 1); }}>Check again</button>}</div></div>{picker}{intake}</>;
  return <><Workspace key={revision} analysis={analysis} localBundle={localBundle} onOpen={openPicker} onReset={reset} onComparison={() => showComparison(true)} onAnalyze={() => setIntakeOpen(true)} learnedAvailable={!!learned} onLearned={() => showLearned(true)} compareAvailable={compareAvailable} onCompare={() => showView('compare')} libraryAvailable={!!patterns} onLibrary={() => showView('library')} interpreterLine={theory?.interpreter_line || patterns?.interpreter_line || undefined} clips={clips} currentClip={currentClip?.video_id ?? null} clipBusy={clipBusy} onChooseClip={clip => void openDemoClip(clip)} demoMoment={demoMoment} />{picker}{intake}</>;
}
