import { ArrowLeft } from 'lucide-react';
import type { Learned, LearnedGoodVsBad } from '../lib/library';
import { bandPath, linePath, seriesRange } from '../lib/library';
import './library.css';

const W = 1000, H = 260;

function GoodVsBadChart({ data }: { data: LearnedGoodVsBad }) {
  const range = seriesRange(data.top.mean, data.top.lo, data.top.hi, data.bottom.mean, data.bottom.lo, data.bottom.hi);
  const first = data.seconds[0], last = data.seconds[data.seconds.length - 1];
  const ticks = data.seconds.filter(s => s % 5 === 0);
  const x = (s: number) => (s - first) / (last - first || 1) * 100;
  const n = (value: number | null) => value === null ? '' : ` (${Math.round(value).toLocaleString('en-US')} clips)`;
  return <figure className="learned-chart">
    <div className="learned-plot">
      <div className="learned-y" aria-hidden="true"><span>Higher response</span><span>Lower response</span></div>
      <div className="learned-svg">
        <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" role="img" aria-label="Average predicted response over the first seconds for clips with more and with fewer likes and comments per view than expected">
          {data.top.lo.length > 0 && <path className="band-above" d={bandPath(data.top.lo, data.top.hi, W, H, range)} />}
          {data.bottom.lo.length > 0 && <path className="band-below" d={bandPath(data.bottom.lo, data.bottom.hi, W, H, range)} />}
          <path className="line-above" d={linePath(data.top.mean, W, H, range)} vectorEffect="non-scaling-stroke" />
          <path className="line-below" d={linePath(data.bottom.mean, W, H, range)} vectorEffect="non-scaling-stroke" />
        </svg>
        <div className="learned-x" aria-hidden="true">{ticks.map(s => <span key={s} style={{ left: `${x(s)}%` }}>{s}s</span>)}</div>
      </div>
    </div>
    <figcaption>
      <span><i className="key-above" /> More likes & comments per view than expected{n(data.n_top)}</span>
      <span><i className="key-below" /> Fewer than expected{n(data.n_bottom)}</span>
      <span>Shaded: likely range of the group average</span>
    </figcaption>
  </figure>;
}

function ChannelMultiples({ data }: { data: LearnedGoodVsBad }) {
  if (!data.channels.length) return null;
  const w = 300, h = 90;
  return <div className="learned-multiples">{data.channels.map(channel => {
    const range = seriesRange(channel.top, channel.bottom);
    return <figure key={channel.key}>
      <figcaption>{channel.label}</figcaption>
      <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" role="img" aria-label={`${channel.label}: clips with more versus fewer likes and comments per view than expected`}>
        <path className="line-above" d={linePath(channel.top, w, h, range)} vectorEffect="non-scaling-stroke" />
        <path className="line-below" d={linePath(channel.bottom, w, h, range)} vectorEffect="non-scaling-stroke" />
      </svg>
    </figure>;
  })}</div>;
}

/** Internal research summary from `learned.json`. Everything shown comes from the file. */
export default function LearnedView({ learned, onBack }: { learned: Learned | null; onBack: () => void }) {
  return <div className="learned-page">
    <header className="learned-top">
      <button type="button" className="learned-back" onClick={onBack}><ArrowLeft size={15} /> Back to the analysis</button>
      <span className="internal-badge">Internal research view</span>
    </header>
    <main className="learned-main">
      <h1>What the model learned</h1>
      <p className="learned-lede">What the first full study found when it tested whether predicted brain response helps rank clips. Internal research view.</p>
      {!learned && <p className="learned-empty">The research summary is not available on this server.</p>}
      {learned && learned.statements.length > 0 && <section className="learned-section" aria-labelledby="learned-findings">
        <h2 id="learned-findings">What the study found</h2>
        {(learned.key_numbers.length > 0 || learned.decision) && <div className="learned-keys">
          {learned.key_numbers.map((k, i) => <div key={i} className="learned-key"><strong>{k.value}</strong><span>{k.text}</span></div>)}
          {learned.decision && <div className="learned-key is-decision"><strong>{learned.decision}</strong><span>Pre-registered test result</span></div>}
        </div>}
        <div className="learned-statements">{learned.statements.map((s, i) => <div key={i} className="learned-statement">{s.value && <strong>{s.value}</strong>}<p>{s.text}</p></div>)}</div>
        {learned.key_numbers.length > 0 && <p className="technical-note">Numbers are rank correlations (Spearman) between predicted and actual performance, from 0 (no ranking) to 1 (perfect); “adds” is the change in that correlation.</p>}
      </section>}
      {learned?.good_vs_bad && <section className="learned-section" aria-labelledby="learned-gvb">
        <h2 id="learned-gvb">More vs fewer likes and comments per view than expected</h2>
        <p className="learned-sub">Average predicted response over the first seconds, for the top third and bottom third of clips by likes and comments per view against what basic clip and account information predicted. This is engagement per view, not views: clips with fewer views tend to have higher rates. If brain response explained it, the two lines would separate; gaps inside the shaded ranges are noise.</p>
        <GoodVsBadChart data={learned.good_vs_bad} />
        {learned.good_vs_bad.result_plain && <p className="learned-result">{learned.good_vs_bad.result_plain}</p>}
        {learned.good_vs_bad.channels.length > 0 && <><h3>By brain signal <span className="learned-sub">(each chart has its own scale, so small gaps look bigger)</span></h3><ChannelMultiples data={learned.good_vs_bad} /></>}
        <p className="lib-caption">TRIBE v2 prediction · average subject · Internal research view</p>
      </section>}
      {learned?.caveat && <p className="learned-caveat">{learned.caveat}</p>}
      <p className="learned-license">Research preview · non-commercial (TRIBE CC-BY-NC)</p>
    </main>
  </div>;
}
