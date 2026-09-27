import type { Library } from '../lib/library';
import { clockTime, referenceLine, verdictLabel } from '../lib/library';
import './library.css';

/** Plain-language summary of how this clip compares with the library. Renders nothing without content. */
export default function ClipScorecard({ library, onSeek, showCaveat = true }: { library: Library; onSeek: (ms: number) => void; showCaveat?: boolean }) {
  if (!library.summary.length && !library.scores.length) return null;
  // A summary sentence that repeats a tile verbatim (typically opening/ending) is shown only on its tile.
  const summary = library.summary.filter(line => !library.scores.some(score => score.plain === line));
  const reference = referenceLine(library.reference);
  return <section className="lib-panel clip-scorecard" aria-labelledby="scorecard-title">
    <div className="lib-head"><div><h2 id="scorecard-title">How this clip compares</h2>{reference && <p>{reference}</p>}</div></div>
    {summary.length > 0 && <ul className="scorecard-summary">{summary.map((line, i) => <li key={i}>{line}</li>)}</ul>}
    {library.scores.length > 0 && <div className="scorecard-tiles">{library.scores.map(score => {
      const body = <>
        <span className="score-top"><span className="score-label">{score.label}</span>{score.verdict && <span className={`verdict-chip ${score.verdict}`}>{verdictLabel(score.verdict)}</span>}</span>
        {score.plain && <p>{score.plain}</p>}
        {score.window_ms && <small className="mono">{clockTime(score.window_ms[0])}–{clockTime(score.window_ms[1])}</small>}
      </>;
      return score.window_ms
        ? <button key={score.key} type="button" className="score-tile" onClick={() => onSeek(score.window_ms![0])} title={`Jump to ${clockTime(score.window_ms[0])}`}>{body}</button>
        : <div key={score.key} className="score-tile">{body}</div>;
    })}</div>}
    {showCaveat && <p className="lib-caveat">{library.caveat || 'Predicted brain response for an average viewer, compared with similar clips. Not a views forecast.'}</p>}
  </section>;
}
