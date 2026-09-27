import { useRef, type ReactNode } from 'react';
import { FileVideo, LoaderCircle, Pause, Play, Upload, Volume2, VolumeX } from 'lucide-react';
import type { Analysis } from '../data/analysis.types';
import type { ClipPlayer } from '../lib/use-clip-player';
import { formatTime } from '../lib/analysis';

/**
 * Height-constrained vertical clip for the one-screen stage: the video fills its
 * column (object-fit: contain) and drives the shared timeline.
 */
export default function StageVideo({ analysis, player, timeMs, playing, onTogglePlay, onSeek, footer }: {
  analysis: Analysis; player: ClipPlayer; timeMs: number; playing: boolean;
  onTogglePlay: () => void; onSeek: (ms: number) => void; footer?: ReactNode;
}) {
  const input = useRef<HTMLInputElement>(null);
  const server = player.source?.kind === 'server';
  const picker = <input ref={input} type="file" accept="video/mp4,video/webm,video/quicktime,.mp4,.webm,.mov,.m4v" aria-label="Choose matching source clip" hidden onChange={event => { const file = event.target.files?.[0]; if (file) player.choose(file); event.target.value = ''; }} />;
  if (!player.source) return <div className="stage-video is-empty">
    {picker}
    <div className="stage-video-empty"><FileVideo size={28} /><p>{analysis.synthetic ? 'Synthetic sample · no source clip.' : 'No source clip on this server.'}</p><button type="button" onClick={() => input.current?.click()}><Upload size={14} /> Attach matching clip</button></div>
    {footer}
  </div>;
  return <div className="stage-video">
    {picker}
    <div className="stage-video-frame">
      <video key={player.source.url} ref={player.video} src={player.source.url} playsInline preload="auto" muted={player.muted}
        aria-label={server ? `${player.source.name}: ${analysis.video_id}` : `Local clip: ${player.source.name}`} onLoadedMetadata={player.loaded} onError={player.failed} onEnded={player.finish}
        onWaiting={player.onWaiting} onPlaying={player.onPlaying} onSeeked={player.onSeeked} onClick={onTogglePlay} />
      <div className="stage-video-top"><span className="mono">{formatTime(timeMs)}</span>{player.ready && <button type="button" onClick={player.toggleMute} aria-label={player.muted ? 'Unmute clip' : 'Mute clip'}>{player.muted ? <VolumeX size={15} /> : <Volume2 size={15} />}</button>}</div>
      {!player.error && !player.ready && <div className="stage-video-loading"><LoaderCircle size={22} className="spin" /></div>}
      {player.ready && !playing && <button type="button" className="stage-video-play" onClick={onTogglePlay} aria-label="Play clip and response"><Play size={26} fill="currentColor" /></button>}
      {player.error && <p className="stage-video-error" role="alert">{player.error}</p>}
    </div>
    <div className="stage-video-controls">
      <button type="button" className="stage-video-toggle" disabled={!player.ready} onClick={onTogglePlay} aria-label={playing ? 'Pause clip and response' : 'Play clip and response'}>{playing ? <Pause size={15} fill="currentColor" /> : <Play size={15} fill="currentColor" />}</button>
      <input type="range" min="0" max={Math.max(0, player.endMs - 1)} step="1" value={Math.min(timeMs, player.endMs - 1)} disabled={!player.ready} onChange={event => onSeek(Number(event.target.value))} aria-label="Clip and response position" aria-valuetext={`${formatTime(timeMs)} of ${formatTime(player.endMs)}`} />
    </div>
    {player.notice && <p className="stage-video-error" role="status">{player.notice}</p>}
    {footer}
  </div>;
}
