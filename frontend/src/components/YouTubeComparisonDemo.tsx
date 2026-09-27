import { lazy, Suspense, useCallback, useEffect, useRef, useState } from 'react';
import { ArrowLeft, ArrowUpRight, BrainCircuit, Clapperboard, Expand, Pause, Play, RotateCcw, Volume2, VolumeX } from 'lucide-react';
import { assetUrl } from '../lib/assets';
import { referenceBrainState } from '../demo/reference-brain';
import { REFERENCE_CLIPS, REFERENCE_SPLIT, referenceDuration, referencePosition, referenceTimestamp } from '../demo/reference-clips';
import { initialPlayback, ReferencePlayback } from '../demo/reference-playback';
import { loadYouTube, youtubeError, type YouTubePlayer } from '../demo/youtube';
import './comparison-demo.css';

const BrainScene = lazy(() => import('./BrainScene'));
const noop = () => {};

export default function YouTubeComparisonDemo({ onBack }: { onBack: () => void }) {
  const [playback, setPlayback] = useState(initialPlayback);
  const [transport] = useState(() => new ReferencePlayback(setPlayback));
  const { time, playing, requested, status, videoDuration } = playback;
  const [sound, setSound] = useState(true);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [session, setSession] = useState<{ iframe: HTMLIFrameElement; time: number; loaded: Promise<void> } | null>(null);
  const [attempt, setAttempt] = useState(0);
  const [failedBrain, setFailedBrain] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const [reducedMotion, setReducedMotion] = useState(() => matchMedia('(prefers-reduced-motion: reduce)').matches);
  const host = useRef<HTMLIFrameElement>(null);
  const player = useRef<YouTubePlayer | null>(null);
  const duration = referenceDuration(videoDuration);
  const { clip, index: visibleIndex, sourceTime } = referencePosition(time, videoDuration);
  const intense = visibleIndex === 1;
  const brain = referenceBrainState(time);
  const responding = brain.activity > 0.02;

  const pause = useCallback(() => transport.pause(), [transport]);
  const go = useCallback((next: number, run: boolean) => {
    const position = Math.max(0, Math.min(referenceDuration(transport.state.videoDuration), next));
    if (!player.current) { transport.prepare(position); return; }
    setError(null);
    transport.go(next, run);
  }, [transport]);
  const startVideo = useCallback(() => {
    if (session) {
      transport.prepare(transport.state.time);
      setSession(null); setAttempt(value => value + 1); setReady(false); setError(null);
      return;
    }
    const startTime = transport.state.time >= referenceDuration(transport.state.videoDuration) ? 0 : transport.state.time;
    const initial = referencePosition(startTime, transport.state.videoDuration);
    const iframe = host.current;
    if (!iframe) return;
    const params = new URLSearchParams({ enablejsapi: '1', origin: window.location.origin,
      start: String(initial.sourceTime), playsinline: '1', controls: '1', rel: '0', autoplay: '0', mute: '1' });
    if (!initial.clip.naturalEnd) params.set('end', String(initial.clip.end));
    iframe.title = 'YouTube reference clip';
    iframe.allow = 'autoplay; encrypted-media; picture-in-picture; fullscreen';
    iframe.allowFullscreen = true;
    iframe.referrerPolicy = 'strict-origin-when-cross-origin';
    // Navigate the mounted frame from Play. Keep the launch control mounted
    // until the frame and API are ready so startup survives embedded browsers.
    const loaded = new Promise<void>(resolve => iframe.addEventListener('load', () => resolve(), { once: true }));
    iframe.src = `https://www.youtube-nocookie.com/embed/${initial.clip.id}?${params}`;
    setReady(false); setError(null);
    setSession({ iframe, time: startTime, loaded });
  }, [session, transport]);
  const toggle = useCallback(() => {
    if (!session) { startVideo(); return; }
    setError(null); transport.toggle();
  }, [session, startVideo, transport]);

  useEffect(() => {
    if (!session) return;
    let disposed = false;
    let failed = false;
    let instance: YouTubePlayer | null = null;
    const { iframe } = session;
    transport.prepare(session.time);
    const fail = (message: string) => {
      if (disposed) return;
      failed = true;
      transport.pause('Video unavailable'); setError(message);
    };
    const readyTimeout = setTimeout(() => fail('YouTube is taking too long to respond. Please retry or open the source video below.'), 30000);
    void session.loaded.then(() => disposed ? null : loadYouTube()).then(api => {
      if (disposed || failed || !api) return;
      instance = new api.Player(iframe, { events: {
        onReady: event => {
          if (disposed || failed) return;
          clearTimeout(readyTimeout); player.current = event.target;
          // Keep iframe initialization quiet; enable audio before the first Play.
          event.target.unMute(); setSound(true);
          transport.attach(event.target);
          setError(null); setReady(true);
          if (!document.hidden) transport.go(session.time, true);
        },
        onStateChange: event => {
          if (disposed || failed || !player.current) return;
          if (event.data === 1 && document.hidden) { pause(); return; }
          transport.onState(event.data);
        },
        onError: event => { clearTimeout(readyTimeout); fail(youtubeError(event.data)); },
        onAutoplayBlocked: () => { if (!disposed) transport.blocked(); },
      } });
    }).catch(cause => { clearTimeout(readyTimeout); fail(cause instanceof Error ? cause.message : 'YouTube could not load.'); });
    return () => {
      disposed = true; clearTimeout(readyTimeout);
      instance?.destroy(); player.current = null;
      iframe.remove();
    };
  }, [session, pause, transport]);

  useEffect(() => {
    let frame = 0;
    let last = 0;
    const tick = (now: number) => {
      if (now - last >= 50 && !document.hidden) { last = now; transport.tick(); }
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    const visibility = () => { if (document.hidden && transport.state.requested) pause(); };
    document.addEventListener('visibilitychange', visibility);
    return () => { cancelAnimationFrame(frame); document.removeEventListener('visibilitychange', visibility); };
  }, [pause, transport]);

  useEffect(() => {
    const media = matchMedia('(prefers-reduced-motion: reduce)');
    const change = () => setReducedMotion(media.matches);
    media.addEventListener('change', change);
    return () => media.removeEventListener('change', change);
  }, []);
  useEffect(() => {
    if (!expanded) return;
    const overflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => { document.body.style.overflow = overflow; };
  }, [expanded]);
  useEffect(() => {
    const key = (event: KeyboardEvent) => {
      if (event.key === 'Escape') { setExpanded(false); return; }
      if ((session && !ready) || (event.target as HTMLElement).closest('button, input, select, textarea, a, iframe')) return;
      if (event.code === 'Space') { event.preventDefault(); toggle(); }
    };
    document.addEventListener('keydown', key);
    return () => document.removeEventListener('keydown', key);
  }, [ready, session, toggle]);
  const brainError = useCallback(() => setFailedBrain(true), []);
  function toggleSound() {
    setSound(!sound);
    if (!sound) player.current?.unMute(); else player.current?.mute();
  }

  return <div className={`comparison-page reference-comparison ${expanded ? 'is-presenting' : ''}`}>
    <header className="comparison-topbar"><a className="brand" href="?demo=comparison" aria-label="ViralBrain Si comparison"><img src={assetUrl('favicon.svg')} alt="" /><span>Viral<span className="brand-light">Brain</span><span className="brand-dot">.</span><span className="brand-si">Si</span></span></a><span className="comparison-edition">THE EDITING EXPERIMENT / 002</span><button onClick={onBack}><ArrowLeft size={15} /> Research workspace</button></header>
    <main className="comparison-main">
      <div className="comparison-heading"><div><span className="eyebrow">A TWO-CLIP REFERENCE COMPARISON</span><h1>Quiet moment.<br /><span>Full momentum.</span></h1></div><div className="reference-intro"><p>Two clips. Two very different rhythms.</p></div></div>
      <section className={`comparison-stage phase-${intense ? 'optimized' : 'original'} ${expanded ? 'is-expanded' : ''}`} aria-label="Interactive video reference comparison" style={{ '--reference-split': `${REFERENCE_SPLIT / duration * 100}%` } as React.CSSProperties}>
        <div className="comparison-stage-head"><span><i />{intense ? '02 / High-energy reference' : '01 / Calm reference'}</span><span className="concept-label">ILLUSTRATIVE DEMO</span></div>
        <div className="comparison-grid">
          <div className="comparison-film reference-film"><div className="youtube-host"><iframe key={attempt} ref={host} title="YouTube reference clip" allow="autoplay; encrypted-media; picture-in-picture; fullscreen" allowFullScreen referrerPolicy="strict-origin-when-cross-origin" /></div>
            {(!session || (!ready && !error)) && <div className="reference-start" style={{ '--reference-poster': `url("https://i.ytimg.com/vi/${clip.id}/hqdefault.jpg")` } as React.CSSProperties}><span className="eyebrow">CURRENT VIDEO PREVIEW</span><button className="reference-launch" disabled={Boolean(session)} onClick={startVideo}><span className="reference-launch-icon"><Play size={28} fill="currentColor" /></span><strong>{session ? 'Loading video…' : 'Play video demo'}</strong></button><p>{clip.label} · excerpt {referenceTimestamp(clip.start)}–{clip.naturalEnd ? 'end' : referenceTimestamp(clip.end)}</p><small>Starts with sound on · mute below</small></div>}
            {error && <div className="reference-error" role="alert"><Clapperboard size={26} /><p>{error}</p><a href={`https://www.youtube.com/watch?v=${clip.id}&t=${clip.start}s`} target="_blank" rel="noreferrer">Open this excerpt on YouTube <ArrowUpRight size={12} /></a>{ready && <button onClick={() => go(0, true)}>Play calm excerpt</button>}<button onClick={startVideo}>Retry YouTube</button></div>}
          </div>
          <div className="comparison-brain">
            {!failedBrain && <Suspense fallback={null}><BrainScene values={brain.values} selected={null} autoRotate={playing} glow playing={playing} visible view="perspective" resetKey={0} reducedMotion={reducedMotion} illustrationTime={brain.elapsed} rotationSpeed={brain.rotationSpeed} ariaLabel="Rotating anatomical brain with staggered regional light effects, starting one second into the exciting clip. Scripted illustration, not analysis of these videos." onHover={noop} onSelect={noop} onReady={noop} onError={brainError} /></Suspense>}
            {failedBrain && <div className="comparison-brain-fallback" style={{ color: responding ? '#e2b675' : '#556374' }}><BrainCircuit size={180} strokeWidth={0.65} /><span>2D illustration · 3D unavailable</span></div>}
            <div className={`brain-rhythm ${responding ? '' : 'is-quiet'}`}><span>{responding ? 'A shifting regional response' : 'A quiet visual response'}</span><div aria-hidden="true">{Array.from({ length: 32 }, (_, i) => <i key={i} style={{ height: `${3 + (Math.sin(i * 1.7 + brain.elapsed * 7) + 1) * brain.activity * 18}px` }} />)}</div><small>Scripted animation · not model output</small></div>
          </div>
        </div>
        <div className="comparison-caption"><span className="comparison-caption-icon"><Clapperboard size={18} /></span><p>{intense ? 'Rapid imagery. Music. A stronger illustrative response.' : 'An everyday moment. A restrained illustrative response.'}</p><button aria-label={expanded ? 'Exit expanded comparison' : 'Expand comparison'} aria-pressed={expanded} onClick={() => setExpanded(value => !value)}><Expand size={17} /></button></div>
        <div className="comparison-transport"><button className="comparison-play" disabled={Boolean(session) && !ready} onClick={toggle} aria-label={requested ? 'Pause comparison' : time >= duration ? 'Replay comparison' : 'Play comparison'}>{requested ? <Pause size={17} fill="currentColor" /> : <Play size={17} fill="currentColor" />}</button><button className="comparison-restart" disabled={!ready} aria-label="Restart comparison" onClick={() => go(0, false)}><RotateCcw size={17} /></button><span className="comparison-time">{time.toFixed(1).padStart(4, '0')}<small> / {duration.toFixed(1)}</small></span><button className={`comparison-sound ${sound ? 'sound-enabled' : ''}`} disabled={!ready} onClick={toggleSound} aria-pressed={sound} aria-label={sound ? 'Mute video sound' : 'Enable video sound'}>{sound ? <Volume2 size={17} /> : <VolumeX size={17} />}<span>{sound ? 'Sound on' : 'Sound off'}</span></button><span className="reference-status" role="status">{session ? status : 'Ready · press Play to load the videos'}</span></div>
        <div className="comparison-scrubber"><input aria-label="Comparison timeline" type="range" min="0" max={duration} step="0.1" disabled={!ready} value={time} onChange={event => go(Number(event.target.value), false)} aria-valuetext={`${time.toFixed(1)} seconds. ${clip.label}`} style={{ '--progress': `${time / duration * 100}%` } as React.CSSProperties} /><div className="comparison-chapters"><button aria-pressed={!intense} onClick={() => go(0, false)}>01<span>Calm · {REFERENCE_CLIPS[0].end - REFERENCE_CLIPS[0].start} sec</span></button><button aria-pressed={intense} onClick={() => go(REFERENCE_SPLIT, false)}>02<span>High energy · to end</span><ArrowUpRight size={13} /></button><span>{Math.round(duration)} SEC</span></div></div>
        <div className="reference-source"><span>Source {Math.min(sourceTime, clip.end).toFixed(1)}s · excerpt {referenceTimestamp(clip.start)}–{clip.naturalEnd ? 'end' : referenceTimestamp(clip.end)}</span><a href={`https://www.youtube.com/watch?v=${clip.id}&t=${clip.start}s`} target="_blank" rel="noreferrer">{clip.title}<ArrowUpRight size={12} /></a></div>
      </section>
      <div className="comparison-context"><p><strong>Two reference clips, one creative demonstration.</strong> These are different videos, not a ViralBrain re-edit. The light effects follow playback time; they are scripted, not audio analysis, measured neural activity, TRIBE predictions or evidence of higher engagement.</p><span>{reducedMotion ? 'Reduced motion enabled' : 'Press Space to play / pause'}<br />YouTube playback requires an internet connection.</span></div>
    </main>
  </div>;
}
