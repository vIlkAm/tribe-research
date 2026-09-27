import { createElement as h } from 'react';
import { formatEnd, formatR, shownVerdict, THEORY_OUTCOME_TITLES, THEORY_OUTCOMES, THEORY_SHOWN_TEXT, theoryRows, type Theory, type TheoryPair } from '../lib/theory.ts';

const pct = (v: number) => `${Math.round(v * 100)}%`;

function Cell({ pair }: { pair: TheoryPair | undefined }) {
  if (!pair) return h('td', null, '—');
  const shown = pair.verdict === 'holds' ? pair.confirmation : pair.full;
  const verdict = shownVerdict(pair);
  const title = [
    pair.plain,
    pair.discovery ? `Found-on half r = ${formatR(pair.discovery.r)}` : '',
    pair.confirmation ? `Checked-on half r = ${formatR(pair.confirmation.r)} (${formatEnd(pair.confirmation.lo)} to ${formatEnd(pair.confirmation.hi)})` : '',
    pair.full ? `All clips r = ${formatR(pair.full.r)} (${formatEnd(pair.full.lo)} to ${formatEnd(pair.full.hi)})` : '',
  ].filter(Boolean).join('\n');
  return h('td', { className: `theory-cell is-${verdict}`, title },
    h('span', { className: 'theory-verdict' }, THEORY_SHOWN_TEXT[verdict]),
    shown ? h('small', null, `r ${formatR(shown.r)} (${formatEnd(shown.lo)} to ${formatEnd(shown.hi)})`) : null);
}

/** "What the library shows": views split into account size vs the video's swing, and the 12 × 4 table. */
export function TheorySection({ theory, onResults }: { theory: Theory; onResults?: () => void }) {
  const d = theory.decomposition;
  const held = theory.pairs.filter(p => p.verdict === 'holds');
  return h('section', { className: 'library-section theory-section', 'aria-labelledby': 'theory-title' },
    h('h2', { id: 'theory-title' }, 'What the library shows'),
    h('p', { className: 'library-note' },
      `${theory.n_clips?.toLocaleString('en-US') ?? '?'} library clips from ${theory.n_accounts ?? '?'} accounts. `
      + 'Each pattern is looked for on half of the accounts and then checked once on the other half; patterns that pass are marked confirmed. '
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
        h('caption', null, '12 clip features × 4 outcomes: r and its 95% range (hover a cell for both halves)'),
        h('thead', null, h('tr', null, h('th', { scope: 'col' }, 'Feature'),
          ...THEORY_OUTCOMES.map(o => h('th', { key: o, scope: 'col', title: theory.outcomes[o] || undefined }, THEORY_OUTCOME_TITLES[o])))),
        h('tbody', null, theoryRows(theory).map(row => h('tr', { key: row.feature },
          h('th', { scope: 'row' }, row.label, row.group === 'brain' ? h('small', { className: 'theory-group' }, ' brain') : null),
          ...THEORY_OUTCOMES.map(o => h(Cell, { key: o, pair: row.cells[o] }))))))),
    h('p', { className: 'library-note theory-legend' },
      h('b', null, 'Confirmed on new accounts'), ': found on half of the accounts and held on the other half (r from that second half). ',
      h('b', null, 'Not yet confirmed'), ': across all clips the 95% range excludes zero, but the split-half check did not confirm it. ',
      h('b', null, 'Did not hold'), ': found on the first half, not on the second. ',
      h('b', null, 'No clear link'), ': the 95% range across all clips includes zero.'),
    h('p', { className: 'library-note' }, 'r is a rank correlation within the same client and platform (views, account size) or within the same account (video vs its account, engagement). '
      + 'Ranges resample whole accounts and are not corrected across the 48 cells. The seven brain rows summarise the same predicted line, so read them as one finding, not seven. '
      + 'Exploratory: not the pre-registered test, which measured likes and comments per view and was a no-GO.'),
    onResults ? h('p', { className: 'theory-results-link' }, 'Grouped by views against the account’s usual instead, the top third of clips had a higher predicted response than the bottom third, a small gap fixed before it was computed. ',
      h('a', { href: '?view=learned', onClick: (event: { preventDefault: () => void }) => { event.preventDefault(); onResults(); } }, 'See the results →')) : null);
}
