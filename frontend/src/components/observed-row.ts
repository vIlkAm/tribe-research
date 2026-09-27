import { createElement as h } from 'react';
import type { Observed } from '../lib/observed.ts';
import { formatCount, formatMultiplier, formatPercent } from '../lib/observed.ts';

/**
 * Compact observed numbers for the first screen: tier chip, views, likes, comments,
 * engagement and views vs the account's usual on one line, with the tier sentence
 * (tier_plain, which puts a large "× usual" in context) as a small second line.
 * Fixed height; the full panel sits below the fold. Never given a lockbox clip.
 */
export function ObservedRow({ observed }: { observed: Observed }) {
  const item = (value: string, label: string) => h('span', { className: 'observed-row-item', key: label }, h('strong', null, value), ` ${label}`);
  return h('div', { className: 'observed-row', 'aria-label': 'Observed on platform' },
    h('div', { className: 'observed-row-line' },
      observed.tier_label ? h('span', { className: `tier-chip${observed.tier ? ` is-${observed.tier}` : ''}` }, observed.tier_label) : null,
      observed.views !== null ? item(formatCount(observed.views), 'views') : null,
      observed.likes !== null ? item(formatCount(observed.likes), 'likes') : null,
      observed.comments !== null ? item(formatCount(observed.comments), 'comments') : null,
      observed.engagement_rate_pct !== null ? item(formatPercent(observed.engagement_rate_pct), 'engagement') : null,
      observed.views_vs_account_usual_x !== null ? item(formatMultiplier(observed.views_vs_account_usual_x), 'account’s usual') : null,
      h('small', { className: 'observed-row-note' }, 'observed on platform')),
    observed.tier_plain ? h('p', { className: 'observed-row-plain', title: observed.tier_plain }, observed.tier_plain) : null);
}
