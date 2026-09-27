import { useId, useRef, useState, type KeyboardEvent, type PointerEvent, type ReactNode } from 'react';
import type { Library } from '../lib/library';
import { clockTime, percentileAt, percentileReadout, referenceLine, seriesPaths } from '../lib/library';
import type { DemoMoment } from '../lib/demo';
import './library.css';

const W = 1000, H = 120;

/** "Engagement at any moment": this clip's predicted response as a percentile of similar clips, per second. */
export default function ResponseStrip({ library, durationMs, timeMs, onSeek, compact = false, highlight = null, action }: {
  library: Library; durationMs: number; timeMs: number; onSeek: (ms: number) => void;
  /** Stage variant: one-line header and a chart that fills the available height. */
  compact?: boolean;
  /** Demo moment window, shaded on the chart. */
  highlight?: DemoMoment | null;
  action?: ReactNode;
}) {
  const index = library.index;
  const wrap = useRef<HTMLDivElement>(null);
  const [hover, setHover] = useState<number | null>(null);
  const id = useId().replaceAll(':', '');
  if (!index) return null;
  const secondMs = 1000 / index.hz;
  const xAt = (i: number) => Math.min(durationMs, (i + 0.5) * secondMs) / durationMs * W;
  const segments = seriesPaths(index.percentile, W, H, 0, 100, xAt);
  const y = (p: number) => H - p / 100 * H;
  const clamp = (ms: number) => Math.max(0, Math.min(durationMs - 1, ms));
  const percent = (ms: number) => Math.max(0, Math.min(100, ms / durationMs * 100));
  const current = percentileAt(index, timeMs);
  const hoverValue = hover === null ? null : percentileAt(index, hover);
  const reference = referenceLine(library.reference);
  function getTime(event: PointerEvent) {
    const rect = wrap.current!.getBoundingClientRect();
    return clamp((event.clientX - rect.left) / rect.width * durationMs);
  }
  function keydown(event: KeyboardEvent) {
    const moves: Record<string, number> = { ArrowLeft: -1000, ArrowRight: 1000, PageDown: -5000, PageUp: 5000 };
    if (event.key in moves) { event.preventDefault(); onSeek(clamp(timeMs + moves[event.key])); }
    if (event.key === 'Home') { event.preventDefault(); onSeek(0); }
    if (event.key === 'End') { event.preventDefault(); onSeek(durationMs - 1); }
  }
  const tone = current === null ? '' : current >= 50 ? 'is-above' : 'is-below';

  const readout = <div className={`strip-readout ${tone}`} aria-live="polite"><strong>{current === null ? '—' : `${Math.round(current)}%`}</strong><span>{percentileReadout(current, timeMs)}</span></div>;
  const shade = highlight && highlight.start_ms < durationMs ? { left: percent(highlight.start_ms), width: percent(Math.min(highlight.end_ms, durationMs)) - percent(highlight.start_ms) } : null;

  return <section className={`lib-panel response-strip ${compact ? 'is-compact' : ''}`} aria-labelledby={`${id}-title`}>
    {compact ? <div className="strip-compact-head">
      <h2 id={`${id}-title`}>Response vs your library</h2>
      {readout}
      {action}
    </div> : <div className="lib-head">
      <div><h2 id={`${id}-title`}>Response vs your library</h2><p>How strongly an average viewer’s brain is predicted to respond at each second, compared with your clips of similar length. Not a views forecast.</p></div>
      {readout}
    </div>}
    <div className="strip-chart">
      <div className="strip-y" aria-hidden="true"><span>Stronger</span><span>Typical</span><span>Weaker</span></div>
      <div className="strip-plot" ref={wrap} role="slider" tabIndex={0} aria-label="Response compared with similar clips" aria-valuemin={0} aria-valuemax={durationMs / 1000} aria-valuenow={Math.round(timeMs / 1000)} aria-valuetext={percentileReadout(current, timeMs)} onKeyDown={keydown}
        onPointerDown={e => { e.currentTarget.setPointerCapture(e.pointerId); onSeek(getTime(e)); }}
        onPointerMove={e => { const t = getTime(e); setHover(t); if (e.buttons === 1) onSeek(t); }}
        onPointerLeave={() => setHover(null)}>
        <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" aria-hidden="true">
          <defs>
            <clipPath id={`${id}-above`}><rect x="0" y="0" width={W} height={y(50)} /></clipPath>
            <clipPath id={`${id}-below`}><rect x="0" y={y(50)} width={W} height={H - y(50)} /></clipPath>
          </defs>
          <rect className="strip-band" x="0" y={y(75)} width={W} height={y(25) - y(75)} />
          <line className="strip-mid" x1="0" x2={W} y1={y(50)} y2={y(50)} vectorEffect="non-scaling-stroke" />
          {segments.map((s, i) => <g key={i}>
            <path className="strip-area-above" d={s.area} clipPath={`url(#${id}-above)`} />
            <path className="strip-area-below" d={s.area} clipPath={`url(#${id}-below)`} />
            <path className="strip-line-above" d={s.line} clipPath={`url(#${id}-above)`} vectorEffect="non-scaling-stroke" />
            <path className="strip-line-below" d={s.line} clipPath={`url(#${id}-below)`} vectorEffect="non-scaling-stroke" />
          </g>)}
        </svg>
        {shade && <div className="strip-moment" style={{ left: `${shade.left}%`, width: `${shade.width}%` }}><span>{highlight!.label}</span></div>}
        <i className="strip-playhead" style={{ left: `${percent(timeMs)}%` }} />
        {hover !== null && <div className="strip-hover" style={{ left: `${percent(hover)}%` }}><span>{clockTime(hover)} · {hoverValue === null ? 'no data' : `${Math.round(hoverValue)}%`}</span></div>}
      </div>
    </div>
    <div className="strip-foot">
      <span><i className="key-band" /> Middle half of similar clips</span>
      <span><i className="key-above" /> Above typical</span>
      <span><i className="key-below" /> Below typical</span>
      <span className="strip-hint">{compact ? "Not a views forecast · click to jump" : "Click to jump"}</span>
    </div>
    <p className="lib-caption">TRIBE v2 prediction · average subject{reference ? ` · ${reference}` : ''}</p>
  </section>;
}
