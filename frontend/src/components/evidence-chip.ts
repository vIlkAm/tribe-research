import { createElement as h } from 'react';
import type { LearnedInterval } from '../lib/library.ts';

/**
 * Topbar link to the Results page carrying its headline: the views-vs-usual gap with its 95% range from
 * resampling whole accounts (`views_vs_usual.signal_windows.index_accounts_resampled.whole`). Exploratory, and it
 * says "had", never "predicts": the pre-registered engagement test was a no-GO, one click away on the same page.
 */
export function EvidenceChip({ interval, onOpen }: { interval: LearnedInterval; onOpen: () => void }) {
  const { diff, lo, hi } = interval.text;
  const title = `Within each client, the third of clips above their account's usual views had a higher predicted brain response `
    + `than the third below it: ${diff} library standard deviations over the first 30 s (95% range ${lo} to ${hi}, whole accounts resampled). `
    + 'Exploratory: the grouping was fixed before it was computed. The pre-registered test on likes and comments per view was a no-GO. Opens the results.';
  return h('a', { className: 'evidence-chip', href: '?view=learned', title, onClick: (event: { preventDefault: () => void }) => { event.preventDefault(); onOpen(); } },
    h('span', null, 'Clips above their account’s usual views had'),
    h('strong', null, `${diff} sd higher predicted response `, h('small', null, `(95% ${lo} to ${hi})`)));
}
