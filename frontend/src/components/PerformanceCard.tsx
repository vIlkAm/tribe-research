import { AlertTriangle, BarChart3, Brain, CircleHelp, Info, ShieldCheck } from 'lucide-react';
import type { Performance } from '../data/analysis.types';
import { percentile, performanceHasNumbers, performanceStatusLabel, platformLabel } from '../lib/performance';
import './performance-card.css';

function EmptyPerformance({ performance }: { performance?: Performance }) {
  const outOfScope = performance?.model_status === 'out_of_scope';
  const title = outOfScope ? 'This clip is outside the model’s scope' : performance ? 'Performance model not trained yet' : 'Performance outlook unavailable';
  const detail = performance?.reason || (performance ? 'The preregistered training gate has not produced a model that can show performance numbers.' : 'This analysis bundle does not contain a performance result. Brain-response exploration is still available.');
  return <section className="performance-card performance-empty" aria-labelledby="performance-title">
    <div className="performance-icon">{outOfScope ? <AlertTriangle size={19} /> : <CircleHelp size={19} />}</div>
    <div className="performance-empty-copy"><span className="eyebrow">PERFORMANCE OUTLOOK</span><h2 id="performance-title">{title}</h2><p>{detail}</p></div>
    <span className={`performance-badge ${outOfScope ? 'scope' : 'waiting'}`}>{outOfScope ? 'Out of scope' : 'No numbers shown'}</span>
  </section>;
}

export default function PerformanceCard({ performance, synthetic = false }: { performance?: Performance; synthetic?: boolean }) {
  if (!performanceHasNumbers(performance)) return <EmptyPerformance performance={performance} />;
  const result = performance!;
  const engagement = result.engagement!;
  const maxDriver = Math.max(...result.drivers.map(driver => Math.abs(driver.contribution)), 0.000001);
  const muted = engagement.confidence === 'low' || result.warnings.length > 0;
  const context = `${result.context.deal_label || result.context.deal_id || 'Unknown deal'} · ${platformLabel(result.context.platform || '')}`;
  const hasRank = engagement.percentile_deal_platform !== null && engagement.likely_range !== null;
  return <section className={`performance-card performance-scored ${muted ? 'muted' : ''}`} aria-labelledby="performance-title">
    <div className="performance-card-head">
      <div><span className="eyebrow">PERFORMANCE OUTLOOK</span><h2 id="performance-title">Predicted engagement</h2></div>
      <span className={`performance-badge ${result.model_status}`}>{result.model_status === 'validated' ? <ShieldCheck size={13} /> : <Info size={13} />}{performanceStatusLabel(result.model_status)}</span>
    </div>
    {synthetic && <div className="performance-synthetic">SYNTHETIC · illustrative numbers only</div>}
    {result.model_status === 'preliminary' && <p className="performance-preliminary-caption">{result.caption}</p>}
    <div className="performance-summary">
      <div className="performance-rank"><strong>{hasRank ? percentile(engagement.percentile_deal_platform!) : 'Not enough data'}</strong><span>{hasRank ? `among ${context} clips` : 'Not enough reference clips yet'}</span><small>{engagement.reference_n} reference clips</small></div>
      {hasRank && <div className="performance-range"><span>Similar clips landed between</span><strong>{percentile(engagement.likely_range![0])}–{percentile(engagement.likely_range![1])}</strong><small>Likely range, not a guarantee</small></div>}
      {result.context.account_level && engagement.percentile_account !== null && <div className="performance-account"><span>Within this account</span><strong>{percentile(engagement.percentile_account)}</strong><small>{engagement.account_reference_n ?? 0} reference posts</small></div>}
    </div>
    {result.retrospective && <p className="performance-note"><Info size={14} /> Already posted · retrospective prediction</p>}
    {result.clip_in_training === 'train_oof' && <p className="performance-note"><Info size={14} /> Training clip · honest out-of-fold prediction</p>}
    {result.clip_in_training === 'lockbox' && <p className="performance-note"><ShieldCheck size={14} /> Sealed lockbox clip · observed outcome remains hidden</p>}
    {result.drivers.length > 0 && <div className="performance-drivers"><div className="performance-subhead"><BarChart3 size={15} /><span>Model contributors</span><small>Attributions, not causes</small></div>{result.drivers.map(driver => <div className="driver-row" key={driver.family}><span>{driver.label}</span><div className="driver-track"><i className={driver.contribution >= 0 ? 'positive' : 'negative'} style={{ width: `${Math.max(6, Math.abs(driver.contribution) / maxDriver * 100)}%` }} /></div></div>)}</div>}
    {result.model_status !== 'preliminary' && result.brain_claim !== 'not_tested' && <p className="performance-claim"><Brain size={15} />{result.brain_claim === 'supported' ? 'The preregistered test supports added predictive value from brain-response features.' : result.brain_claim === 'not_supported' ? 'Brain-response features did not add predictive value in the preregistered test.' : 'Content features and predicted brain response contributed; the neural claim is not yet validated.'}</p>}
    {result.reach && <p className="performance-reach">Reach outlook: {result.reach.percentile_deal_platform === null ? 'not enough reference clips' : percentile(result.reach.percentile_deal_platform)} · mostly driven by account, platform and timing.</p>}
    {result.warnings.length > 0 && <div className="performance-warnings" role="status"><AlertTriangle size={15} /><div>{result.warnings.map((warning, index) => <p key={index}>{warning}</p>)}</div></div>}
    <p className="performance-disclaimer">{result.model_status === 'preliminary' ? 'Preliminary · not validated · illustrative, not a forecast' : `${performanceStatusLabel(result.model_status)} · correlational · not a guarantee`} · research preview · non-commercial (TRIBE CC-BY-NC)</p>
  </section>;
}
