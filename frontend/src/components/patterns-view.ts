import { createElement as h } from 'react';
import { TIER_LABELS, TIER_ORDER } from '../lib/demo.ts';
import { ciBar, coinFlipText, featureGroups, formatFeatureValue, verdictText, type PatternFeature, type Patterns } from '../lib/patterns.ts';

function FeatureRow({ feature }: { feature: PatternFeature }) {
  const bar = feature.diff ? ciBar(feature.diff) : null;
  const tendency = feature.verdict === 'weak_tendency';
  return h('tr', { className: `pattern-row ${tendency ? 'is-tendency' : 'is-neutral'}` },
    h('th', { scope: 'row' }, h('strong', null, feature.label_plain), feature.plain ? h('p', null, feature.plain) : null),
    ...TIER_ORDER.map(tier => h('td', { key: tier, className: `pattern-mean is-${tier}` }, formatFeatureValue(feature.mean[tier], feature.unit))),
    h('td', { className: 'pattern-ci' },
      bar ? h('div', { className: 'ci-bar', role: 'img', 'aria-label': `Great minus bad: ${formatFeatureValue(feature.diff!.point, feature.unit)} (${formatFeatureValue(feature.diff!.lo, feature.unit)} to ${formatFeatureValue(feature.diff!.hi, feature.unit)})` },
        h('i', { className: 'ci-zero', style: { left: `${bar.zero}%` } }),
        h('i', { className: 'ci-range', style: { left: `${bar.lo}%`, width: `${Math.max(0.5, bar.hi - bar.lo)}%` } }),
        h('i', { className: 'ci-point', style: { left: `${bar.point}%` } })) : '—'),
    h('td', { className: 'pattern-coin' }, feature.coin_flip ? coinFlipText(feature.coin_flip) : '—'),
    h('td', null, feature.verdict ? h('span', { className: `pattern-verdict ${tendency ? 'is-tendency' : 'is-neutral'}` }, verdictText(feature.verdict)) : null));
}

/** "What great clips have in common (and what they don't)" plus the caveats, verbatim. */
export function PatternsSection({ patterns }: { patterns: Patterns }) {
  return h('section', { className: 'library-section patterns-section', 'aria-labelledby': 'patterns-title' },
    h('h2', { id: 'patterns-title' }, 'What great clips have in common (and what they don’t)'),
    patterns.definition ? h('p', { className: 'library-note' }, patterns.definition) : null,
    featureGroups(patterns).map(group => h('table', { key: group.title, className: 'pattern-table' },
      h('caption', null, group.title),
      h('thead', null, h('tr', null,
        h('th', { scope: 'col' }, 'Feature'),
        ...TIER_ORDER.map(tier => h('th', { key: tier, scope: 'col', className: `is-${tier}` }, patterns.tiers[tier]?.label || TIER_LABELS[tier])),
        h('th', { scope: 'col' }, 'Great − bad (interval, 0 marked)'),
        h('th', { scope: 'col', title: 'How often a random clip that did great is higher than a random clip that did badly; 50% is a coin flip' }, 'Great higher (50% = coin flip)'),
        h('th', { scope: 'col' }, 'Verdict'))),
      h('tbody', null, group.features.map(feature => h(FeatureRow, { key: feature.key, feature }))))),
    patterns.caveats.length ? h('div', { className: 'library-caveats' },
      h('h3', null, 'Caveats'),
      h('ul', null, patterns.caveats.map((caveat, i) => h('li', { key: i }, caveat)))) : null);
}
