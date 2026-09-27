import { ExternalLink } from 'lucide-react';
import type { Observed } from '../lib/observed';
import { formatCount, formatDate, formatMultiplier, formatPercent } from '../lib/observed';
import { platformLabel } from '../lib/performance';
import './library.css';

/** Internal-only: the post's own observed numbers. Callers never pass a lockbox clip (parseObserved refuses it). */
export default function ObservedPanel({ observed }: { observed: Observed }) {
  const metrics: [string, string][] = [
    ['Views', formatCount(observed.views)],
    ['Likes', formatCount(observed.likes)],
    ['Comments', formatCount(observed.comments)],
    ['Shares', formatCount(observed.shares)],
    ...(observed.saves !== null ? [['Saves', formatCount(observed.saves)] as [string, string]] : []),
    ['Engagement rate', formatPercent(observed.engagement_rate_pct)],
  ];
  return <section className="lib-panel observed-panel" aria-labelledby="observed-title">
    <div className="lib-head"><div><h2 id="observed-title">Observed on platform</h2>{observed.caption && <p>{observed.caption}</p>}</div><span className="internal-badge">Internal research view</span></div>
    <div className="observed-metrics">{metrics.map(([label, value]) => <div key={label}><strong className="mono">{value}</strong><span>{label}</span></div>)}</div>
    {observed.views_vs_account_usual_x !== null && <p className="observed-usual">Views vs this account’s recent usual: <strong className="mono">{formatMultiplier(observed.views_vs_account_usual_x)}</strong>{observed.account_usual_n_posts !== null && <small> (usual from {Math.round(observed.account_usual_n_posts)} posts)</small>}</p>}
    {(observed.tier_label || observed.tier_plain) && <p className={`observed-expectation ${observed.tier ? `is-${observed.tier}` : ''}`}>{observed.tier_label && <span className={`tier-chip ${observed.tier ? `is-${observed.tier}` : ''}`}>{observed.tier_label}</span>} {observed.tier_plain}</p>}
    <p className="observed-meta">{[observed.platform ? platformLabel(observed.platform) : '', observed.upload_date ? `Uploaded ${formatDate(observed.upload_date)}` : ''].filter(Boolean).join(' · ')}{observed.video_link && <> · <a href={observed.video_link} target="_blank" rel="noreferrer">Open the post <ExternalLink size={12} /></a></>}</p>
    {observed.caption_null && <p className="lib-caveat">{observed.caption_null}</p>}
  </section>;
}
