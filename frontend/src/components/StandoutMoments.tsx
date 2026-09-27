import { ArrowUpRight } from 'lucide-react';
import type { Library, LibraryMoment } from '../lib/library';
import { clockTime, momentText } from '../lib/library';
import './library.css';

/** Up to three moments where this clip differs most from similar clips. */
export default function StandoutMoments({ library, timeMs, labelFor, onChoose }: { library: Library; timeMs: number; labelFor: (key: string) => string; onChoose: (moment: LibraryMoment) => void }) {
  if (!library.moments.length) return null;
  return <section className="lib-moments" aria-labelledby="standout-title">
    <div className="lib-head"><div><h2 id="standout-title">Standout moments</h2><p>Where this clip’s predicted response differs most from similar clips.</p></div></div>
    <div className="lib-moment-grid">{library.moments.map((moment, i) => {
      const high = moment.kind === 'standout_high';
      const current = moment.start_ms <= timeMs && timeMs < moment.end_ms;
      const channels = moment.channels.map(labelFor).filter(label => label && !moment.plain.includes(label));
      return <button key={i} type="button" className={`lib-moment ${high ? 'is-above' : 'is-below'} ${current ? 'current' : ''}`} onClick={() => onChoose(moment)}>
        <span className="lib-moment-meta"><span className="mono">{clockTime(moment.start_ms)}–{clockTime(moment.end_ms)}</span><span className={`verdict-chip ${high ? 'strong' : 'weak'}`}>{high ? 'Above similar clips' : 'Below similar clips'}</span><ArrowUpRight size={16} /></span>
        <p>{momentText(moment.plain)}</p>
        {(channels.length > 0 || moment.percentile !== null) && <small>{[channels.join(' · '), moment.percentile !== null ? `Higher than ${Math.round(moment.percentile)}% of similar clips` : ''].filter(Boolean).join(' — ')}</small>}
      </button>;
    })}</div>
  </section>;
}
