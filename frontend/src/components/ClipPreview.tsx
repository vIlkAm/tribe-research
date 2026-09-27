import { useId, useRef, useState } from 'react';
import { Check, FileVideo, Layers3, Link2, LoaderCircle, Pause, Play, Upload, Volume2, VolumeX, X } from 'lucide-react';
import type { Analysis } from '../data/analysis.types';
import type { ClipPlayer } from '../lib/use-clip-player';
import { formatTime, isGap, sampleAt, signalText, stepPath } from '../lib/analysis';
import './clip-preview.css';

interface Props {
  analysis: Analysis; player: ClipPlayer; timeMs: number; playing: boolean;
  selected: string | null; visibleKeys: string[]; onTogglePlay: () => void; onSeek: (ms: number) => void;
  /** `side`: compact column next to the brain (only while a clip is loaded). */
  layout?: 'inline' | 'side';
}

export default function ClipPreview({ analysis, player, timeMs, playing, selected, visibleKeys, onTogglePlay, onSeek, layout = 'inline' }: Props) {
  const input = useRef<HTMLInputElement>(null);
  const [overlay, setOverlay] = useState(false);
  const id = useId().replaceAll(':', '');
  const channels = analysis.channels.filter(channel => selected ? channel.key === selected : visibleKeys.includes(channel.key));
  const gap = isGap(analysis, timeMs);
  const onset = timeMs >= analysis.timing.onset_window_ms[0] && timeMs < analysis.timing.onset_window_ms[1];
  const progress = timeMs / analysis.duration_ms * 100;
  const durationDiffers = player.duration !== undefined && Math.abs(player.duration * 1000 - analysis.duration_ms) > 500;
  const attach = () => input.current?.click();
  const side = layout === 'side';
  const server = player.source?.kind === 'server';
  const overlayToggle = <button type="button" className={`clip-overlay-toggle ${overlay ? 'active' : ''}`} aria-pressed={overlay} onClick={() => setOverlay(value => !value)} disabled={!player.ready}><Layers3 size={16} /> Overlay pattern {overlay && <Check size={13} />}</button>;

  return <section className={`clip-preview ${player.source ? 'has-clip' : ''} ${side ? 'is-side' : ''}`} aria-label="Synchronized clip preview">
    <div className="clip-preview-heading"><div><span className="eyebrow">THE MOMENT, IN CONTEXT</span>{!side && <h3>See the clip. Follow the pattern.</h3>}</div><span className="clip-local"><Link2 size={13} /> One shared timeline</span></div>
    <input ref={input} type="file" accept="video/mp4,video/webm,video/quicktime,.mp4,.webm,.mov,.m4v" aria-label="Choose matching source clip" hidden onChange={event => { const file = event.target.files?.[0]; if (file) player.choose(file); event.target.value = ''; }} />
    {player.source ? <>
      <div className="clip-preview-layout">
        <div className="clip-screen">
          <video key={player.source.url} ref={player.video} src={player.source.url} playsInline preload="metadata" muted={player.muted}
            aria-label={server ? `${player.source.name}: ${analysis.video_id}` : `Local clip: ${player.source.name}`} onLoadedMetadata={player.loaded} onError={player.failed} onEnded={player.finish}
            onWaiting={player.onWaiting} onPlaying={player.onPlaying} onSeeked={player.onSeeked} />
          <div className="clip-screen-top"><span className="clip-status"><i /> {player.error ? 'Preview unavailable' : !player.ready ? 'Loading clip' : player.waiting && playing ? 'Buffering' : playing ? 'In sync' : 'Paused'}</span><span className="clip-timestamp mono">{formatTime(timeMs)}</span></div>
          {!player.error && !player.ready && <div className="clip-loading"><LoaderCircle size={24} className="spin" /></div>}
          {player.ready && <button type="button" className={`clip-play ${playing ? 'is-playing' : ''}`} onClick={onTogglePlay} aria-label={playing ? 'Pause clip and pattern' : 'Play clip and pattern'}>{playing ? <Pause size={25} fill="currentColor" /> : <Play size={25} fill="currentColor" />}</button>}
          {overlay && player.ready && <div className="clip-pattern-overlay" aria-label="Response pattern overlay">
            <div><span>{analysis.synthetic ? 'SYNTHETIC PATTERN' : 'PREDICTED RESPONSE'}</span><span>{gap ? 'No prediction' : onset ? 'Onset response' : 'Within-clip z-score'}</span></div>
            <svg viewBox="0 0 1000 100" preserveAspectRatio="none" aria-label="Stepped response pattern, with current clip position">
              <defs><mask id={`${id}-clip-gaps`}><rect width="1000" height="100" fill="white" />{analysis.timing.gaps_ms.map(([start, end], i) => <rect key={i} x={start / analysis.duration_ms * 1000} y="0" width={(end - start) / analysis.duration_ms * 1000} height="100" fill="black" />)}</mask></defs>
              <line x1="0" x2="1000" y1="50" y2="50" stroke="#ffffff35" />
              {channels.map(channel => <path key={channel.key} d={stepPath(channel.values, channel.hz, analysis.duration_ms, 1000, 100)} mask={`url(#${id}-clip-gaps)`} stroke={channel.color} strokeWidth="1.7" fill="none" vectorEffect="non-scaling-stroke" />)}
              <line x1={progress * 10} x2={progress * 10} y1="0" y2="100" stroke="#f4d35e" strokeWidth="2" vectorEffect="non-scaling-stroke" />
            </svg>
          </div>}
          <div className="clip-screen-bottom"><span>{analysis.synthetic ? 'Synthetic signals · not an analysis of this clip' : server ? `${player.source.name} · matched by video ID` : 'User-selected clip · identity not verified'}</span>{player.ready && <button type="button" onClick={player.toggleMute} aria-label={player.muted ? 'Unmute clip' : 'Mute clip'}>{player.muted ? <VolumeX size={17} /> : <Volume2 size={17} />}</button>}</div>
        </div>
        {!side && <aside className="clip-context" aria-label="Response at the current frame"><span className="eyebrow">AT THIS MOMENT</span><strong className="clip-context-time mono">{formatTime(timeMs)}</strong><p>{gap ? 'No prediction at this time.' : onset ? 'Inside the onset response window.' : 'The clip and response share the same position.'}</p>
          <div className="clip-signal-values">{channels.map(channel => <div key={channel.key}><i style={{ background: channel.color }} /><span>{channel.label.replace(/ proxy$/, '')}</span><strong className="mono">{signalText(gap ? null : sampleAt(channel, timeMs, analysis.duration_ms))}<small> z</small></strong></div>)}</div>
          {overlayToggle}
        </aside>}
      </div>
      <input className="clip-scrubber" type="range" min="0" max={Math.max(0, player.endMs - 1)} step="1" value={Math.min(timeMs, player.endMs - 1)} disabled={!player.ready} onChange={event => onSeek(Number(event.target.value))} aria-label="Clip and pattern position" aria-valuetext={`${formatTime(timeMs)} of ${formatTime(player.endMs)}`} />
      <div className="clip-file-row"><span><FileVideo size={15} /><span title={player.source.name}>{player.source.name}</span><small>{server ? 'Served locally · not bundled' : 'Local preview only'}</small></span><div>{side && overlayToggle}<button type="button" onClick={attach}>Replace clip</button><button type="button" onClick={player.remove} aria-label="Remove clip"><X size={16} /></button></div></div>
      {durationDiffers && <p className="clip-note">Clip length ({formatTime(player.duration! * 1000)}) differs from the analysis ({formatTime(analysis.duration_ms)}). Playback stops or loops at {formatTime(player.endMs)}. Use the same edit and first frame for a matching preview.</p>}
    </> : <div className="clip-empty"><div className="clip-empty-frame" aria-hidden="true"><FileVideo size={31} /><span>CLIP PREVIEW</span></div><div><h4>Your clip, alongside its response.</h4><p>This bundle has no source video. Attach the matching clip to watch, pause and scrub both together.</p><span>{analysis.synthetic ? 'The sample signals are synthetic. Attaching a video does not analyze it.' : 'Use the same edit and first frame as the analyzed clip.'} The file stays in this browser tab.</span></div><button type="button" className="clip-attach" onClick={attach}><Upload size={16} /> Attach matching clip</button></div>}
    {(player.error || player.notice) && <p className="clip-error" role="alert">{player.error || player.notice}</p>}
  </section>;
}
