import { useId, useMemo, useRef, useState, type PointerEvent, type KeyboardEvent } from 'react';
import type { Analysis } from '../data/analysis.types';
import { formatTime, stepPath } from '../lib/analysis';
import './contract-widgets.css';

interface Props { analysis: Analysis; timeMs: number; selected: string | null; onSeek: (ms: number) => void; visibleKeys: string[]; onToggleChannel: (key: string) => void; }

function timeTicks(duration: number): number[] {
  const target = duration / 5;
  const magnitude = 10 ** Math.floor(Math.log10(target));
  const step = ([1, 2, 5, 10].find(n => n * magnitude >= target) ?? 10) * magnitude;
  const ticks = Array.from({ length: Math.floor(duration / step) + 1 }, (_, i) => i * step);
  if (duration - ticks[ticks.length - 1] < step * .4 && ticks.length > 1) ticks.pop();
  if (ticks[ticks.length - 1] !== duration) ticks.push(duration);
  return ticks;
}

const readableKey = (key: string) => key.replaceAll('_', ' ');

export default function Timeline({ analysis, timeMs, selected, onSeek, visibleKeys, onToggleChannel }: Props) {
  const wrap = useRef<HTMLDivElement>(null);
  const [hover, setHover] = useState<number | null>(null);
  const id = useId().replaceAll(':', '');
  const width = 1000, height = 130;
  const duration = analysis.duration_ms;
  const ticks = useMemo(() => timeTicks(duration), [duration]);
  const channels = analysis.channels.filter(c => selected ? c.key === selected : visibleKeys.includes(c.key));
  const selectedChannel = analysis.channels.find(c => c.key === selected);
  const shotsReason = analysis.events.unavailable_lanes.find(lane => lane.key === 'shots')?.reason;
  const speech = analysis.events.speech_spans_ms.length ? analysis.events.speech_spans_ms : analysis.events.words.map(word => [word.start_ms, word.end_ms] as [number, number]);
  const clampTime = (ms: number) => Math.max(0, Math.min(duration - 1, ms));
  const percent = (ms: number) => Math.max(0, Math.min(100, ms / duration * 100));
  function getTime(event: PointerEvent) {
    const rect = wrap.current!.getBoundingClientRect();
    return clampTime((event.clientX - rect.left) / rect.width * duration);
  }
  function keydown(event: KeyboardEvent) {
    const moves: Record<string, number> = { ArrowLeft: -500, ArrowRight: 500, PageDown: -5000, PageUp: 5000 };
    if (event.key in moves) { event.preventDefault(); onSeek(clampTime(timeMs + moves[event.key])); }
    if (event.key === 'Home') { event.preventDefault(); onSeek(0); }
    if (event.key === 'End') { event.preventDefault(); onSeek(duration - 1); }
  }
  const crosshairs = <><i className="contract-lane-playhead" style={{ left: `${percent(timeMs)}%` }} />{hover !== null && <i className="contract-lane-hover" style={{ left: `${percent(hover)}%` }} />}</>;

  return <div className="contract-timeline">
    <div className="contract-channel-toggles" role="group" aria-label="Visible timeline signals">
      {analysis.channels.map(channel => <button key={channel.key} type="button" disabled={!!selected} aria-pressed={selected ? selected === channel.key : visibleKeys.includes(channel.key)} onClick={() => onToggleChannel(channel.key)}><i style={{ backgroundColor: channel.color }} />{channel.label}</button>)}
    </div>
    {selectedChannel && <p className="contract-timeline-note">{selectedChannel.label} is isolated. Clear the selection in the signal explorer to choose multiple signals.</p>}
    {!channels.length && <p className="contract-timeline-note">Choose a signal to show its response over time.</p>}
    <p className="contract-timeline-note">Native resolution: {analysis.timing.native_tr_s} s · sample-and-hold steps · shaded area: onset response · hatched area: no prediction</p>
    <div className="timeline-chart">
      <div className="y-axis"><span>+3 z</span><span>0</span><span>−3 z</span></div>
      <div className="chart-interaction" ref={wrap} role="slider" tabIndex={0} aria-label="Analysis timeline" aria-valuemin={0} aria-valuemax={duration / 1000} aria-valuenow={Number((timeMs / 1000).toFixed(1))} aria-valuetext={`${formatTime(timeMs)} of ${formatTime(duration)}`} onKeyDown={keydown}
        onPointerDown={e => { e.currentTarget.setPointerCapture(e.pointerId); onSeek(getTime(e)); }}
        onPointerMove={e => { const t = getTime(e); setHover(t); if (e.buttons === 1) onSeek(t); }}
        onPointerLeave={() => setHover(null)}>
        <svg className="signal-plot" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" aria-label="Research proxy responses over time in within-clip z-scores">
          <defs>
            <pattern id={`${id}-gap`} width="8" height="8" patternUnits="userSpaceOnUse"><path d="M0 8L8 0" stroke="#8e9ba9" strokeWidth=".7" /></pattern>
            <mask id={`${id}-available`}><rect width={width} height={height} fill="white" />{analysis.timing.gaps_ms.map(([a, b], i) => <rect key={i} x={percent(a) / 100 * width} width={(percent(b) - percent(a)) / 100 * width} height={height} fill="black" />)}</mask>
          </defs>
          <rect x={percent(analysis.timing.onset_window_ms[0]) / 100 * width} width={(percent(analysis.timing.onset_window_ms[1]) - percent(analysis.timing.onset_window_ms[0])) / 100 * width} height={height} fill="#becbd7" opacity=".09" />
          {[4, 35, 65, 95, 126].map(y => <line key={y} x1="0" y1={y} x2={width} y2={y} className={y === 65 ? 'chart-baseline' : 'chart-grid'} />)}
          {ticks.map(t => <line key={t} x1={t / duration * width} y1="0" x2={t / duration * width} y2={height} className="chart-grid" />)}
          {analysis.timing.gaps_ms.map(([a, b], i) => <rect key={i} x={percent(a) / 100 * width} width={(percent(b) - percent(a)) / 100 * width} y="0" height={height} fill={`url(#${id}-gap)`}><title>No prediction: {formatTime(a)}–{formatTime(b)}</title></rect>)}
          {channels.map(c => <path key={c.key} d={stepPath(c.values, c.hz, duration, width, height)} mask={`url(#${id}-available)`} stroke={c.color} fill="none" strokeWidth="1.8" vectorEffect="non-scaling-stroke" />)}
        </svg>
        <div className="timeline-playhead" style={{ left: `${percent(timeMs)}%` }}><span /></div>
        {hover !== null && <div className="timeline-hover" style={{ left: `${percent(hover)}%` }}><span>{formatTime(hover)}</span></div>}
        <div className="x-axis">{ticks.map((t, i) => <span key={t} className={i === 0 ? 'contract-first-tick' : i === ticks.length - 1 ? 'contract-last-tick' : undefined} style={{ left: `${percent(t)}%` }}>{Number((t / 1000).toFixed(2))}s</span>)}</div>
      </div>
    </div>
    <div className="contract-event-lanes" aria-label="Timeline events">
      <div className="contract-event-lane"><span>Moments</span><div className="contract-event-track contract-moment-track">{analysis.moments.filter(m => !m.in_onset_window).map(moment => <button key={moment.id} type="button" className={`contract-moment ${moment.status === 'edit_hypothesis_untested' ? 'is-hypothesis' : 'is-observation'}`} aria-label={`${moment.title}, ${formatTime(moment.start_ms)} to ${formatTime(moment.end_ms)}, ${readableKey(moment.status)}`} title={`${moment.title} · ${readableKey(moment.status)} · relative to this clip`} onClick={() => onSeek(clampTime(moment.start_ms))} style={{ left: `${percent(moment.start_ms)}%`, width: `${percent(moment.end_ms) - percent(moment.start_ms)}%` }} />)}{crosshairs}</div></div>
      <div className="contract-event-lane"><span>Shots</span><div className="contract-event-track">{analysis.events.shots_ms === null ? <span className="contract-empty-lane">{shotsReason ?? 'Shot boundaries unavailable.'}</span> : analysis.events.shots_ms.length === 0 ? <span className="contract-empty-lane">No shot boundaries supplied</span> : analysis.events.shots_ms.map((ms, i) => <button key={i} type="button" className="contract-shot" style={{ left: `${percent(ms)}%` }} aria-label={`Shot boundary at ${formatTime(ms)}`} title={`Shot boundary · ${formatTime(ms)}`} onClick={() => onSeek(clampTime(ms))} />)}{crosshairs}</div></div>
      <div className="contract-event-lane"><span>Speech</span><div className="contract-event-track">{speech.length ? speech.map(([start, end], i) => {
        const words = analysis.events.words.filter(word => word.start_ms < end && word.end_ms > start).map(word => word.text).join(' ');
        return <button key={i} type="button" className="contract-speech" style={{ left: `${percent(start)}%`, width: `${percent(end) - percent(start)}%` }} aria-label={`Speech at ${formatTime(start)}${words ? `: ${words}` : ''}`} title={`${formatTime(start)}–${formatTime(end)}${words ? ` · ${words}` : ''}`} onClick={() => onSeek(clampTime(start))} />;
      }) : <span className="contract-empty-lane">{analysis.quality.has_words ? 'No speech spans detected' : 'No transcribed words available'}</span>}{crosshairs}</div></div>
    </div>
    <div className="contract-moment-key"><span><i /> Observation</span><span><i className="is-hypothesis" /> Untested hypothesis</span><span>Relative to this clip</span></div>
    {(analysis.unavailable_channels.length > 0 || analysis.events.unavailable_lanes.some(lane => lane.key !== 'shots')) && <details className="contract-unavailable"><summary>Unavailable signals and event lanes</summary><ul>
      {analysis.unavailable_channels.map(channel => <li key={`channel-${channel.key}`}><button type="button" disabled>{readableKey(channel.key)}</button><span>{channel.reason}</span></li>)}
      {analysis.events.unavailable_lanes.filter(lane => lane.key !== 'shots').map(lane => <li key={`lane-${lane.key}`}><button type="button" disabled>{readableKey(lane.key)}</button><span>{lane.reason}</span></li>)}
    </ul></details>}
  </div>;
}
