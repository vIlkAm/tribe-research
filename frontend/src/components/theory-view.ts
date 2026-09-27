import { createElement as h } from 'react';
import { formatR, THEORY_OUTCOME_TITLES, THEORY_OUTCOMES, THEORY_VERDICT_TEXT, theoryRows, type Theory, type TheoryPair } from '../lib/theory.ts';

const pct = (v: number) => `${Math.round(v * 100)}%`;

function Cell({ pair }: { pair: TheoryPair | undefined }) {
  if (!pair) return h('td', null, '—');
  const shown = pair.verdict === 'holds' ? pair.confirmation : pair.full;
  const title = [
    pair.plain,
    pair.discovery ? `Found-on half r = ${formatR(pair.discovery.r)}` : '',
    pair.confirmation ? `Checked-on half r = ${formatR(pair.confirmation.r)} (${formatR(pair.confirmation.lo)} to ${formatR(pair.confirmation.hi)})` : '',
    pair.full ? `All clips r = ${formatR(pair.full.r)}` : '',
  ].filter(Boolean).join('\n');
  return h('td', { className: `theory-cell is-${pair.verdict}`, title },
    h('span', { className: 'theory-verdict' }, THEORY_VERDICT_TEXT[pair.verdict]),
    shown ? h('small', null, `r ${formatR(shown.r)}`) : null);
}

/** "What holds across the library": views split into account size vs the video's swing, and the 12 × 4 table. */
export function TheorySection({ theory }: { theory: Theory }) {
  const d = theory.decomposition;
  const held = theory.pairs.filter(p => p.verdict === 'holds');
  return h('section', { className: 'library-section theory-section', 'aria-labelledby': 'theory-title' },
    h('h2', { id: 'theory-title' }, 'What holds across the library'),
    h('p', { className: 'library-note' },
      `${theory.n_clips?.toLocaleString('en-US') ?? '?'} library clips from ${theory.n_accounts ?? '?'} accounts. `
      + 'Each pattern is looked for on half of the accounts and then checked once on the other half; only patterns that hold on the second half count. '
      + 'Views, likes and comments are the platforms’ own numbers; the brain line is TRIBE v2’s prediction for an average viewer.'),
    d ? h('div', { className: 'theory-split', role: 'img', 'aria-label': `Variance of views within a client and platform: around the account's usual ${pct(d.share_video)}, account size ${pct(d.share_account)}, overlap ${pct(d.share_covariance)}` },
      h('div', { className: 'theory-bar' },
        h('i', { className: 'is-video', style: { flexGrow: d.share_video } }, h('span', null, `Swing around the account’s usual ${pct(d.share_video)}`)),
        h('i', { className: 'is-account', style: { flexGrow: d.share_account } }, h('span', null, `Account size ${pct(d.share_account)}`))),
      h('p', { className: 'lib-caption' }, `Overlap ${pct(d.share_covariance)}, so the two shares add to about 100%. The swing includes luck, timing and the algorithm, not only the video itself.`)) : null,
    theory.interpreter_line ? h('p', { className: 'theory-interpreter' }, theory.interpreter_line) : null,
    held.length ? h('ul', { className: 'theory-held' }, held.map(p => h('li', { key: `${p.feature}-${p.outcome}` }, p.plain,
      p.feature === 'length_s' && p.outcome === 'engagement' ? ' Not a cause: longer clips also tend to get fewer views, and rates per view rise as views fall.' : null))) : null,
    h('div', { className: 'theory-table-wrap' },
      h('table', { className: 'pattern-table theory-table' },
        h('caption', null, '12 clip features × 4 outcomes (hover a cell for the numbers)'),
        h('thead', null, h('tr', null, h('th', { scope: 'col' }, 'Feature'),
          ...THEORY_OUTCOMES.map(o => h('th', { key: o, scope: 'col', title: theory.outcomes[o] || undefined }, THEORY_OUTCOME_TITLES[o])))),
        h('tbody', null, theoryRows(theory).map(row => h('tr', { key: row.feature },
          h('th', { scope: 'row' }, row.label, row.group === 'brain' ? h('small', { className: 'theory-group' }, ' brain') : null),
          ...THEORY_OUTCOMES.map(o => h(Cell, { key: o, pair: row.cells[o] }))))))),
    h('p', { className: 'library-note' }, 'r is a rank correlation within the same client and platform (views, account size) or within the same account (video vs its account, engagement). Exploratory: not the pre-registered test, which was a no-GO for the brain response as a views predictor.'));
}
