import { ArrowLeft } from 'lucide-react';
import type { Learned, LearnedGoodVsBad, LearnedInterval } from '../lib/library';
import { bandPath, linePath, seriesRange, SIGNAL_WINDOW_LABEL, SIGNAL_WINDOW_ORDER } from '../lib/library';
import { TRIBE_PAPER } from '../lib/tribe-paper';
import './library.css';

const W = 1000, H = 260;

interface GroupLabels { top: string; bottom: string; aria: string }
const ENGAGEMENT: GroupLabels = { top: 'More likes & comments per view than expected', bottom: 'Fewer than expected', aria: 'clips with more and with fewer likes and comments per view than expected' };
const VIEWS: GroupLabels = { top: "Above their account's usual views", bottom: "Below their account's usual views", aria: "clips above and below their account's usual views" };

function GoodVsBadChart({ data, labels }: { data: LearnedGoodVsBad; labels: GroupLabels }) {
  const range = seriesRange(data.top.mean, data.top.lo, data.top.hi, data.bottom.mean, data.bottom.lo, data.bottom.hi);
  const first = data.seconds[0], last = data.seconds[data.seconds.length - 1];
  const ticks = data.seconds.filter(s => s % 5 === 0);
  const x = (s: number) => (s - first) / (last - first || 1) * 100;
  const n = (value: number | null) => value === null ? '' : ` (${Math.round(value).toLocaleString('en-US')} clips)`;
  return <figure className="learned-chart">
    <div className="learned-plot">
      <div className="learned-y" aria-hidden="true"><span>Higher response</span><span>Lower response</span></div>
      <div className="learned-svg">
        <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" role="img" aria-label={`Average predicted response over the first seconds for ${labels.aria}`}>
          {data.top.lo.length > 0 && <path className="band-above" d={bandPath(data.top.lo, data.top.hi, W, H, range)} />}
          {data.bottom.lo.length > 0 && <path className="band-below" d={bandPath(data.bottom.lo, data.bottom.hi, W, H, range)} />}
          <path className="line-above" d={linePath(data.top.mean, W, H, range)} vectorEffect="non-scaling-stroke" />
          <path className="line-below" d={linePath(data.bottom.mean, W, H, range)} vectorEffect="non-scaling-stroke" />
        </svg>
        <div className="learned-x" aria-hidden="true">{ticks.map(s => <span key={s} style={{ left: `${x(s)}%` }}>{s}s</span>)}</div>
      </div>
    </div>
    <figcaption>
      <span><i className="key-above" /> {labels.top}{n(data.n_top)}</span>
      <span><i className="key-below" /> {labels.bottom}{n(data.n_bottom)}</span>
      <span>Shaded: likely range of the group average</span>
    </figcaption>
  </figure>;
}

const VERDICT_TEXT = { holds: 'holds', leans: 'leans higher', none: 'no difference' } as const;

function ChannelMultiples({ data, labels }: { data: LearnedGoodVsBad; labels: GroupLabels }) {
  if (!data.channels.length) return null;
  const w = 300, h = 90;
  return <div className="learned-multiples">{data.channels.map(channel => {
    const range = seriesRange(channel.top, channel.bottom);
    const windows = data.signals?.channels[channel.key];
    return <figure key={channel.key}>
      <figcaption>{channel.label}</figcaption>
      <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" role="img" aria-label={`${channel.label}: ${labels.aria}`}>
        <path className="line-above" d={linePath(channel.top, w, h, range)} vectorEffect="non-scaling-stroke" />
        <path className="line-below" d={linePath(channel.bottom, w, h, range)} vectorEffect="non-scaling-stroke" />
      </svg>
      {windows && <dl className="signal-stats">{SIGNAL_WINDOW_ORDER.map(key => {
        const x = windows[key];
        return x && <div key={key} className={`sig-${x.verdict}`} title={`95% range ${x.text.lo} to ${x.text.hi}`}>
          <dt>{SIGNAL_WINDOW_LABEL[key]}</dt><dd>{x.text.diff}</dd><dd className="sig-verdict">{VERDICT_TEXT[x.verdict]}</dd>
        </div>;
      })}</dl>}
    </figure>;
  })}</div>;
}

const range = (x: LearnedInterval) => `95% range ${x.text.lo} to ${x.text.hi}`;
const count = (n: number | null) => n === null ? '' : Math.round(n).toLocaleString('en-US');

/** Headline tiles for the views chart, read from its summary; nothing is shown for a missing number. */
function ViewsEvidence({ data }: { data: LearnedGoodVsBad }) {
  const { whole, opening } = data.summary;
  const acc = data.signals?.index_accounts;
  const tiles: { value: string; text: string }[] = [];
  if (whole) tiles.push({ value: `${whole.text.diff} sd`, text: `Higher predicted response over the first 30 seconds (${range(whole)})` });
  if (opening) tiles.push({ value: `${opening.text.diff} sd`, text: `Higher in the opening 4 seconds (${range(opening)})` });
  if (data.n_top !== null && data.n_bottom !== null) tiles.push({ value: `${count(data.n_top)} vs ${count(data.n_bottom)}`,
    text: `Clips compared${data.n_deals !== null ? `, across ${count(data.n_deals)} clients` : ''}` });
  if (acc?.whole && acc.whole.lo > 0 && data.signals?.n_accounts) tiles.push({ value: 'Holds',
    text: `When whole accounts are resampled (${count(data.signals.n_accounts)} accounts): ${acc.whole.text.diff} sd, ${range(acc.whole)}` });
  return tiles.length ? <div className="learned-keys">{tiles.map(t => <div key={t.text} className="learned-key"><strong>{t.value}</strong><span>{t.text}</span></div>)}</div> : null;
}

/** Internal research summary from `learned.json`. Everything shown comes from the file. */
export default function LearnedView({ learned, onBack }: { learned: Learned | null; onBack: () => void }) {
  return <div className="learned-page">
    <header className="learned-top">
      <button type="button" className="learned-back" onClick={onBack}><ArrowLeft size={15} /> Back to the analysis</button>
      <span className="internal-badge">Internal research view</span>
    </header>
    <main className="learned-main">
      <h1>Results</h1>
      <p className="learned-lede">What our study of 1,486 real clips from our agency found. Every brain curve here is a TRIBE v2 prediction for an average viewer; nobody was scanned.</p>
      {!learned && <p className="learned-empty">The research summary is not available on this server.</p>}
      {learned?.views_vs_usual && <section className="learned-section" aria-labelledby="learned-views-title" id="learned-views">
        <h2 id="learned-views-title">Clips that beat their account's usual views had a higher predicted brain response</h2>
        <ViewsEvidence data={learned.views_vs_usual} />
        <p className="learned-sub">Within each client, clips were split into thirds by views against their own account's usual, so account size is taken out; the top and bottom thirds are compared second by second. “sd” is library standard deviations. Exploratory: the grouping was fixed before it was computed, but it is not the strict test further down.</p>
        <GoodVsBadChart data={learned.views_vs_usual} labels={VIEWS} />
        {learned.views_vs_usual.result_plain && <p className="learned-result">{learned.views_vs_usual.result_plain}</p>}
        {learned.views_vs_usual.channels.length > 0 && <><h3>By brain signal <span className="learned-sub">(each chart has its own scale, so small gaps look bigger)</span></h3><ChannelMultiples data={learned.views_vs_usual} labels={VIEWS} /></>}
        {learned.views_vs_usual.signals && <p className="technical-note">Numbers are top minus bottom third in library sd: opening (first 4 s), average (seconds 0–29) and ending (each clip's own last 3 s). “Holds” survives a correction for {count(learned.views_vs_usual.signals.n_tests) || 'all the'} tests with whole accounts resampled; “leans higher” points the same way but is not reliable on its own. The last seconds of the chart only include the longer clips, which is why the ending uses each clip's own last seconds. Checked after the curves were seen, so exploratory.</p>}
        <p className="lib-caption">TRIBE v2 prediction · average subject · Internal research view</p>
      </section>}
      <section className="learned-section" aria-labelledby="learned-model">
        <h2 id="learned-model">The brain model: TRIBE v2 (Meta FAIR, {TRIBE_PAPER.date})</h2>
        <p className="learned-sub">Every brain visual here is a prediction from Meta’s TRIBE v2 for an average viewer. From the paper, “{TRIBE_PAPER.title}” ({TRIBE_PAPER.authors}):</p>
        <div className="learned-keys">{TRIBE_PAPER.facts.map(f => <div key={f.value} className="learned-key"><strong>{f.value}</strong><span>{f.text}</span></div>)}</div>
        <p className="learned-result">{TRIBE_PAPER.limits}</p>
        <p className="lib-caption">Numbers from the TRIBE v2 paper · {TRIBE_PAPER.links.map((l, i) => <span key={l.href}>{i ? ' · ' : ''}<a href={l.href} target="_blank" rel="noreferrer">{l.label}</a></span>)} · TRIBE v2 is CC BY-NC 4.0; this demo is independent and not affiliated with Meta</p>
      </section>

      {learned && learned.statements.length > 0 && <section className="learned-section" aria-labelledby="learned-findings">
        <h2 id="learned-findings">The strict test we wrote down first: likes and comments per view</h2>
        {(learned.key_numbers.length > 0 || learned.decision) && <div className="learned-keys">
          {learned.key_numbers.map((k, i) => <div key={i} className="learned-key"><strong>{k.value}</strong><span>{k.text}</span></div>)}
          {learned.decision && <div className="learned-key is-decision"><strong>{learned.decision}</strong><span>Strict test result (pre-registered)</span></div>}
        </div>}
        <div className="learned-statements">{learned.statements.map((s, i) => <div key={i} className="learned-statement">{s.value && <strong>{s.value}</strong>}<p>{s.text}</p></div>)}</div>
        {learned.key_numbers.length > 0 && <p className="technical-note">Numbers are rank correlations (Spearman) between predicted and actual performance within each client and platform, from 0 (no ranking) to 1 (perfect); “adds” is the change in that correlation. This test measures likes and comments per view, not views.</p>}
      </section>}
      {learned?.good_vs_bad && <section className="learned-section" aria-labelledby="learned-gvb">
        <h2 id="learned-gvb">Strict test, by second: more vs fewer likes and comments per view than expected</h2>
        <p className="learned-sub">Average predicted response over the first seconds, for the top third and bottom third of clips by likes and comments per view against what basic clip and account information predicted. This is engagement per view, not views: clips with fewer views tend to have higher rates. If brain response explained it, the two lines would separate; gaps inside the shaded ranges are noise.</p>
        <GoodVsBadChart data={learned.good_vs_bad} labels={ENGAGEMENT} />
        {learned.good_vs_bad.result_plain && <p className="learned-result">{learned.good_vs_bad.result_plain}</p>}
        {learned.good_vs_bad.channels.length > 0 && <><h3>By brain signal <span className="learned-sub">(each chart has its own scale, so small gaps look bigger)</span></h3><ChannelMultiples data={learned.good_vs_bad} labels={ENGAGEMENT} /></>}
        <p className="lib-caption">TRIBE v2 prediction · average subject · Internal research view</p>
      </section>}
      {learned?.caveat && <p className="learned-caveat">{learned.caveat}</p>}
      <p className="learned-license">Research preview · non-commercial (TRIBE CC-BY-NC)</p>
    </main>
  </div>;
}
