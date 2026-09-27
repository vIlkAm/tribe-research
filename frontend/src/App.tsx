import { lazy, Suspense, useCallback, useEffect, useRef, useState } from 'react';
import { ArrowUpRight, AudioLines, Box, ChevronDown, ChevronRight, CircleHelp, Crosshair, Expand, ExternalLink, Focus, FolderOpen, GitBranch, Info, Layers3, LoaderCircle, MousePointer2, Pause, Play, Repeat2, RotateCcw, Rotate3D, Scan, SkipBack, Sparkles, Upload, X } from 'lucide-react';
import type { Analysis, Channel, Moment } from './data/analysis.types';
import { assertAnalysis, BUNDLE_PATH, formatTime, isGap, sampleAt, signalText, stepPath } from './lib/analysis';
import BrainAtlas from './components/BrainAtlas';
import Timeline from './components/Timeline';
import { assetUrl } from './lib/assets';
import { compatibleDemo } from './lib/demo-compat';
import AnalysisPicker from './components/AnalysisPicker';
import AnalysisIntake from './components/AnalysisIntake';
import PerformanceCard from './components/PerformanceCard';
import ClipPreview from './components/ClipPreview';
import { useClipPlayer } from './lib/use-clip-player';
import type { LocalBundle } from './lib/local-bundle';
import './components/comparison-demo.css';

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
    <p>The 3D surface is FreeSurfer fsaverage5: 20,484 cortical vertices. The 3D demonstration is pinned to one synthetic bundle; HCP-MMP1 annotations map its supplied proxy channels onto the sample’s region groups. Other bundles use the contracted 2D atlas until a 3D asset contract is agreed. Each region receives its channel’s value; this is not a raw per-vertex response map.</p>
    <h3>How to read the light</h3>
    <p>Amber is above a region’s average for this clip. Blue is below it. Brightness represents magnitude, not quality. Values are within-clip z-scores and cannot be compared across clips. Shimmer is an illustrative effect, not neuronal firing.</p>
    <h3>Timing and uncertainty</h3>
    <p>This bundle’s native resolution is {analysis.timing.native_tr_s} seconds, displayed on a {analysis.timing.display_hz} Hz sample-and-hold grid. Missing samples remain gaps. The onset window is {formatTime(analysis.timing.onset_window_ms[0])}–{formatTime(analysis.timing.onset_window_ms[1])}. {analysis.timing.note}</p>
    <p>Signal confidence and direction come from the bundle. Values are relative to this clip; they do not establish behavioral outcomes. The bundle contains no source video. You can attach a matching clip locally; its playback then drives the analysis timeline. Time zero is the first video frame, with no additional lag applied. Attaching a clip does not run an analysis.</p>
    <div className="source-links"><a href={`${REPO}/tree/65ab40704ca484b453e2e2675d777423f325f97a/docs/sample_analysis`} target="_blank" rel="noreferrer">Sample & contract <ExternalLink size={14} /></a><a href="https://github.com/nilearn/nilearn/tree/0.12.1/nilearn/datasets/data/fsaverage5" target="_blank" rel="noreferrer">Brain anatomy <ExternalLink size={14} /></a><a href="https://aidemos.atmeta.com/tribev2" target="_blank" rel="noreferrer">Meta TRIBE v2 <ExternalLink size={14} /></a></div>
  </dialog>;
}

function Workspace({ analysis, localBundle, onOpen, onReset, onComparison, onAnalyze }: { analysis: Analysis; localBundle: LocalBundle | null; onOpen: () => void; onReset: () => void; onComparison: () => void; onAnalyze: () => void }) {
  const demoCompatible = compatibleDemo(analysis);
  const [timeMs, setTimeMs] = useState(Math.min(4200, analysis.duration_ms - 1));
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
    setPlaying(v => !v);
  }, [timeMs, clip.active, clip.ready, clip.endMs, clip.seek]);

  useEffect(() => {
    const keyboard = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement;
      if (document.querySelector('dialog[open]') || target.closest('button, input, select, textarea, a, [role="slider"]')) return;
      if (event.code === 'Space') { event.preventDefault(); togglePlay(); }
    };
    document.addEventListener('keydown', keyboard);
    return () => document.removeEventListener('keydown', keyboard);
  }, [togglePlay, methodOpen]);

  const seek = (ms: number) => { setTimeMs(clip.seek(ms)); setActiveMoment(null); };
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
  function setPreset(preset: typeof view) { setView(preset); setResetKey(v => v + 1); setAutoRotate(false); }

  return <>
    <header className="topbar"><div className="header-main"><Brand /><span className="header-divider" /><span className="product-name">Neural video intelligence</span></div><div className="header-actions"><button className="analyze-video-button" onClick={onAnalyze}><Upload size={14} /><span>Analyze video</span></button><a className="watch-comparison-button" href="?demo=comparison" onClick={event => { event.preventDefault(); onComparison(); }}><Play size={14} /> Watch video preview</a><button className="open-analysis-button" onClick={onOpen}><FolderOpen size={15} /> Open analysis</button><a className="repo-link" href={REPO} target="_blank" rel="noreferrer"><GitBranch size={15} /> Research source <ExternalLink size={12} /></a><button className="help-button" aria-label="About the demo" onClick={() => setMethodOpen(true)}><CircleHelp size={17} /><span>About the demo</span></button><div className="edition">LAB<span>01</span></div></div></header>
    <main>
      <div className="page-heading"><div><div className="breadcrumb">RESEARCH WORKSPACE <ChevronRight size={12} /> {analysis.synthetic ? 'SYNTHETIC DEMO' : 'ANALYSIS'}</div><h1>See the response.<span> Find the moment.</span></h1><p>Explore how predicted cortical responses change over time.</p></div><div className="analysis-badge"><span className="tiny-orbit"><Scan size={17} /></span><div><strong>{analysis.model_versions.neural}</strong><span>Research use · {analysis.status}</span></div><span className="version">{analysis.schema_version.replace('nvi.analysis.', '')}</span></div></div>
      {localBundle && <div className="local-analysis-strip"><FolderOpen size={19} /><div><span className="eyebrow">{localBundle.origin === 'remote' ? 'COMPLETED RESEARCH ANALYSIS' : 'LOCAL ANALYSIS · IN THIS TAB ONLY'}</span><strong>{localBundle.filename}</strong></div><button onClick={onReset}>Back to demo</button></div>}
      {localBundle && localBundle.warnings.length > 0 && <div className="local-asset-notes" role="status">{localBundle.warnings.map(note => <p key={note}>{note}</p>)}</div>}
      {analysis.synthetic && <div className="sample-notice"><span className="synthetic-tag">SYNTHETIC SAMPLE</span><button onClick={() => setMethodOpen(true)}>How to read this <ArrowUpRight size={14} /></button></div>}
      <details className="analysis-details"><summary>Analysis details · {analysis.analysis_id} · research only{analysis.quality.warnings.length > 0 ? ` · ${analysis.quality.warnings.length} quality notes` : ''}</summary><div><p>Source: {analysis.source_name || analysis.video_id}. Native resolution: {analysis.timing.native_tr_s}s · display: {analysis.timing.display_hz} Hz · {analysis.timing.n_display_samples} samples.</p><p>Proxy version: {analysis.provenance.proxies_version}. Normalization: {analysis.provenance.normalization}. Explanation model: {analysis.model_versions.explain}.</p><p>{analysis.timing.note}</p>{analysis.quality.warnings.map((warning, i) => <p key={i}>{warning}</p>)}</div></details>
      <PerformanceCard performance={analysis.performance} synthetic={analysis.synthetic} />
      <div className="workspace-grid">
        <section ref={stageRef} className={`brain-stage ${gap ? 'has-gap' : ''}`} aria-label="Brain visualization">
          <div className="stage-top"><div className="stage-title"><span className="eyebrow">CORTICAL RESPONSE</span><span className="stage-subtitle">The brain, in motion.</span></div><div className="view-tabs" role="group" aria-label="Brain display mode"><button className={mode === '3d' ? 'active' : ''} onClick={() => { if (!brainFailed) setMode('3d'); }} disabled={brainFailed} aria-pressed={mode === '3d'}><Box size={15} /> 3D</button><button className={mode === 'atlas' ? 'active' : ''} onClick={() => setMode('atlas')} aria-pressed={mode === 'atlas'}><Layers3 size={15} /> Atlas</button></div></div>
          <div className="brain-view" style={{ display: mode === '3d' ? undefined : 'none' }}>
            {!brainFailed && <Suspense fallback={null}><BrainScene values={values} selected={selected} autoRotate={autoRotate} glow={glow} playing={playing} visible={mode === '3d'} view={view} resetKey={resetKey} reducedMotion={reducedMotion} onHover={setHovered} onSelect={setSelected} onReady={onReady} onError={onError} /></Suspense>}
            {!brainReady && <div className="brain-loading"><LoaderCircle size={22} className="spin" /><span>Loading cortical surface</span></div>}
          </div>
          {mode === 'atlas' && <BrainAtlas analysis={analysis} timeMs={timeMs} onHover={setHovered} resolveAsset={localBundle?.resolveAsset} />}
          <div className="stage-context"><span className={`status-dot ${playing ? 'pulsing' : ''}`} />{gap ? 'No prediction' : onset ? 'Onset response' : playing ? 'Playing analysis' : 'Exploring response'}<span className="mono">{formatTime(timeMs)}</span></div>
          {mode === '3d' && <div className="brain-axis" aria-hidden="true"><span className="axis-y">S</span><span className="axis-x">R</span><span className="axis-z">A</span><i /><b /><em /></div>}
          <div className={`brain-tooltip ${inspected && mode === '3d' ? 'visible' : ''}`} aria-live="polite">{inspected && <><span style={{ color: inspected.color }}>{inspected.label}</span><strong>{signalText(values[inspected.key])}<small> z</small></strong><p>{inspected.basis.regions_text}</p></>}</div>
          <div className="brain-tools"><div className="camera-presets" role="group" aria-label="Camera views"><button aria-label="Perspective view" title="Perspective view" disabled={mode !== '3d'} aria-pressed={view === 'perspective'} onClick={() => setPreset('perspective')}><Box size={16} /></button><button aria-label="Front view" title="Front view" disabled={mode !== '3d'} aria-pressed={view === 'front'} onClick={() => setPreset('front')}><Focus size={16} /></button><button aria-label="Side view" title="Side view" disabled={mode !== '3d'} aria-pressed={view === 'side'} onClick={() => setPreset('side')}><Scan size={16} /></button><button aria-label="Top view" title="Top view" disabled={mode !== '3d'} aria-pressed={view === 'top'} onClick={() => setPreset('top')}><Layers3 size={16} /></button></div><button className="icon-button" title="Expand brain view" aria-label="Expand brain view" onClick={() => { if (!document.fullscreenElement) void stageRef.current?.requestFullscreen?.(); else void document.exitFullscreen?.(); }}><Expand size={17} /></button></div>
          <div className="stage-bottom"><div className="brain-options"><button className={autoRotate && !reducedMotion ? 'enabled' : ''} onClick={() => setAutoRotate(v => !v)} aria-pressed={autoRotate && !reducedMotion} disabled={reducedMotion || mode !== '3d'}><Rotate3D size={16} /> Auto-rotate<span className="toggle-track"><i /></span></button><button className={glow ? 'enabled' : ''} onClick={() => setGlow(v => !v)} aria-pressed={glow} disabled={mode !== '3d'}><Sparkles size={15} /> Glow</button></div><span className="drag-hint"><MousePointer2 size={13} /> Drag to explore</span></div>
          {mode === '3d' && <div className="brain-scale"><span>Below average</span><i /><span>Above average</span></div>}
        </section>

        <aside className="signal-panel" aria-label="Signal explorer"><div className="panel-heading"><div><span className="eyebrow">SIGNAL EXPLORER</span><h2>What changes here?</h2></div><AudioLines size={20} /></div><div className="signal-context"><span>Relative to this clip</span><span className="mono">{formatTime(timeMs)} <span>/ {formatTime(analysis.duration_ms)}</span></span></div><div className="signal-list">
          {analysis.channels.map(channel => <button key={channel.key} className={`signal-row ${selected === channel.key ? 'selected' : ''}`} onClick={() => setSelected(selected === channel.key ? null : channel.key)} aria-pressed={selected === channel.key} style={{ '--signal': channel.color } as React.CSSProperties} title={`Isolate ${channel.label}`}><span className="signal-color" /><span className="signal-name">{channel.label}</span><MiniSignal channel={channel} durationMs={analysis.duration_ms} /><span className="signal-number">{signalText(values[channel.key])}<small>z</small></span></button>)}
        </div><div className="signal-selection"><span><Crosshair size={14} />{selected ? 'One signal isolated' : 'Select a signal to isolate its regions'}</span>{selected && <button onClick={() => setSelected(null)}>Clear <X size={12} /></button>}</div><div className="signal-insight"><span className="eyebrow">{selected ? 'SELECTED SIGNAL' : 'READING THE RESPONSE'}</span><p>{focused ? focused.copy.tooltip : 'Every region tells a part of the story. Explore the patterns together, or focus on a single signal.'}</p>{focused && <div className="signal-evidence"><p>{focusedValue == null ? 'No prediction at this time.' : focusedValue > 0 ? focused.copy.rise : focusedValue < 0 ? focused.copy.fall : 'At this clip’s average.'}</p><p>{focused.direction_note}</p><span>Confidence: {focused.confidence} · source: {focused.source_model}</span></div>}<span className="research-label"><Info size={13} /> {focused ? focused.research_status.replaceAll('_', ' ') : 'Research proxies · relative to this clip'}</span></div><div className="behavior-state"><span>Behavior predictions</span><span>{analysis.predictions.status.replaceAll('_', ' ')} <Info size={13} /></span>{analysis.predictions.status === 'not_available' ? <p>No behavior predictions yet. {analysis.predictions.reason || 'A trained behavior model is required.'}</p> : <p>Behavior metrics are supplied; this research demo does not yet interpret their units or calibration.</p>}</div></aside>
      </div>

      <section className="timeline-panel" aria-label="Response timeline"><div className="timeline-header"><div><span className="eyebrow">RESPONSE OVER TIME</span><h2>Every moment has a pattern.</h2></div><div className="timeline-meta"><span><i className="zone-key" /> Candidate moments</span><span>Time (seconds) · within-clip z-score</span></div></div><Timeline analysis={analysis} timeMs={timeMs} selected={selected} onSeek={seek} visibleKeys={visibleKeys} onToggleChannel={key => setVisibleKeys(keys => keys.includes(key) ? keys.filter(value => value !== key) : [...keys, key])} /><ClipPreview analysis={analysis} player={clip} timeMs={timeMs} playing={playing} selected={selected} visibleKeys={visibleKeys} onTogglePlay={togglePlay} onSeek={seek} /><div className="transport"><div className="playback-controls"><button className="icon-button" aria-label="Restart sample" title="Restart sample" onClick={() => { seek(0); }}><SkipBack size={17} /></button><button className="play-button" disabled={clip.active && !clip.ready} onClick={togglePlay} aria-label={playing ? 'Pause sample' : 'Play sample'}>{playing ? <Pause size={18} fill="currentColor" /> : <Play size={18} fill="currentColor" />}</button><span className="playback-time mono">{formatTime(timeMs)}<span> / {formatTime(analysis.duration_ms)}</span></span><button className={`icon-button loop-button ${loop ? 'active' : ''}`} onClick={() => setLoop(v => !v)} aria-label="Loop playback" title="Loop playback" aria-pressed={loop}><Repeat2 size={17} /></button><button className="speed-button mono" onClick={() => setSpeed(v => v === 0.5 ? 1 : v === 1 ? 2 : 0.5)} aria-label={`Playback speed ${speed} times. Click to change.`}>{speed}×</button></div><span className="sample-duration">{clip.active ? 'Clip + analysis playback' : 'Analysis playback'}<span>{clip.active ? 'Video drives the shared timeline' : 'Attach a clip to watch in sync'}</span></span></div></section>

      <section className="moments-section" aria-label="Candidate moments"><div className="moments-heading"><div><span className="eyebrow">LOOK A LITTLE CLOSER</span><h2>Moments worth exploring <span>{sortedMoments.length}</span></h2></div><button className="text-button" onClick={() => setAllMoments(v => !v)}>{allMoments ? 'Show highlights' : 'View all moments'}<ChevronDown size={15} className={allMoments ? 'flipped' : ''} /></button></div><div className="moment-grid">{visibleMoments.map((moment, i) => <button className={`moment-card ${moment.status} ${currentMoment?.id === moment.id ? 'current' : ''}`} key={moment.id} onClick={() => chooseMoment(moment)}><div className="moment-meta"><span className="moment-index">{String(i + 1).padStart(2, '0')}</span><span className="mono">{formatTime(moment.start_ms)} — {formatTime(moment.end_ms)}</span><span className="moment-arrow"><ArrowUpRight size={17} /></span></div><h3>{moment.title}</h3><p>{moment.hypothesis || moment.description}</p><div className="moment-bottom"><span>{moment.status === 'edit_hypothesis_untested' ? 'Untested edit hypothesis' : 'Observation'}</span><span>{moment.channels.map(key => analysis.channels.find(c => c.key === key)?.label.replace(' proxy', '')).join(' · ')}</span></div></button>)}</div></section>
      {currentMoment && <div className="moment-detail"><Info size={16} /><div><p><strong>{currentMoment.title}.</strong> {currentMoment.description}</p>{currentMoment.hypothesis && <p>Untested hypothesis: {currentMoment.hypothesis}</p>}<ul>{currentMoment.evidence.map((evidence, i) => <li key={i}>{evidence.replaceAll('_', ' ')}</li>)}</ul>{currentMoment.test_metric && <p>Test against: {currentMoment.test_metric}</p>}<small>Confidence: {currentMoment.confidence} · relative to this clip</small></div></div>}
      <footer><span><img src={assetUrl('favicon.svg')} alt="" /> Built for the curious.</span><div><span>{analysis.schema_version}</span><span className="footer-dot">·</span><button onClick={() => setMethodOpen(true)}>Data & methodology <ArrowUpRight size={13} /></button></div></footer>
    </main>
    <MethodDialog analysis={analysis} open={methodOpen} onClose={() => setMethodOpen(false)} />
  </>;
}

export default function App() {
  const [comparison, setComparison] = useState(() => new URLSearchParams(window.location.search).get('demo') === 'comparison');
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
  function showComparison(show: boolean) {
    const url = new URL(window.location.href);
    if (show) url.searchParams.set('demo', 'comparison'); else url.searchParams.delete('demo');
    history.replaceState(null, '', url);
    setComparison(show);
    window.scrollTo({ top: 0 });
  }
  const analysis = localBundle?.analysis ?? sample;
  const openPicker = () => setPickerOpen(true);
  const reset = () => { setLocalBundle(null); setRevision(value => value + 1); };
  const picker = <AnalysisPicker open={pickerOpen} onClose={() => setPickerOpen(false)} onOpen={bundle => { setLocalBundle(bundle); setRevision(value => value + 1); }} />;
  const intake = <AnalysisIntake open={intakeOpen} onClose={() => setIntakeOpen(false)} onOpen={bundle => { setLocalBundle(bundle); setRevision(value => value + 1); }} onOpenExport={() => setPickerOpen(true)} />;
  if (comparison) return <Suspense fallback={<div className="loading-screen"><LoaderCircle className="spin" /><p>Preparing the comparison</p></div>}><ComparisonDemo onBack={() => showComparison(false)} /></Suspense>;
  if (!analysis) return <><div className="loading-screen"><Brand />{error ? <><h1>Couldn’t load the analysis.</h1><p>{error}</p><button className="retry-button" onClick={() => setAttempt(v => v + 1)}><RotateCcw size={17} /> Try again</button><button className="open-analysis-button" onClick={openPicker}><FolderOpen size={15} /> Open analysis</button></> : <><LoaderCircle size={24} className="spin" /><p>Preparing your research workspace</p></>}</div>{picker}{intake}</>;
  if (analysis.status !== 'complete') return <><div className="loading-screen"><Brand /><h1>{analysis.status === 'failed' ? 'Analysis failed.' : analysis.status === 'queued' ? 'Analysis queued.' : 'Analysis is processing.'}</h1><p>{analysis.analysis_id} · {localBundle ? 'Open a completed export when it is available.' : 'Results appear when a completed bundle is available.'}</p>{analysis.synthetic && <span className="synthetic-tag">SYNTHETIC SAMPLE</span>}{analysis.quality.warnings.map((warning, i) => <p key={i}>{warning}</p>)}<div className="analysis-state-actions"><button className="open-analysis-button" onClick={openPicker}><FolderOpen size={15} /> Open analysis</button>{localBundle ? <button className="retry-button" onClick={reset}>Back to demo</button> : <button className="retry-button" onClick={() => { setSample(null); setAttempt(v => v + 1); }}>Check again</button>}</div></div>{picker}{intake}</>;
  return <><Workspace key={revision} analysis={analysis} localBundle={localBundle} onOpen={openPicker} onReset={reset} onComparison={() => showComparison(true)} onAnalyze={() => setIntakeOpen(true)} />{picker}{intake}</>;
}
