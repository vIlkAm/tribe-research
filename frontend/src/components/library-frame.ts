import { createElement as h } from 'react';
import { clipHeading, lengthLabel, TIER_LABELS, TIER_ORDER, type DemoClip } from '../lib/demo.ts';
import type { Library, ScoreKey } from '../lib/library.ts';
import { libraryUrlFor, parseLibrary, verdictLabel } from '../lib/library.ts';
import type { Observed } from '../lib/observed.ts';
import { formatCount, formatMultiplier, formatPercent, isLockboxBundle, parseObserved, siblingUrl } from '../lib/observed.ts';
import { resolveRealBundleUrls } from '../lib/real-analysis-index.ts';
import type { Patterns } from '../lib/patterns.ts';
import { platformLabel } from '../lib/performance.ts';
import { PatternsSection } from './patterns-view.ts';
import type { Theory } from '../lib/theory.ts';
import { TheorySection } from './theory-view.ts';

export const LIBRARY_BADGE = 'Internal research view · exploratory';
export const LIBRARY_SCORE_KEYS: readonly ScoreKey[] = ['hook', 'hold', 'peak', 'finish'];
const SCORE_TITLES: Record<string, string> = { hook: 'Opening', hold: 'Middle', peak: 'Best 3 s', finish: 'Ending' };

/** One demo clip's row: observed numbers are null for lockbox clips or when absent. */
export interface LibraryRow { clip: DemoClip; observed: Observed | null; library: Library | null }

type Click = { preventDefault: () => void };

/**
 * Load one row's optional files. A sealed lockbox clip never requests observed.json;
 * a missing or unusable file just leaves that part of the row empty.
 */
export async function loadLibraryRow(indexUrl: string, clip: DemoClip, fetchJson: (url: string, signal?: AbortSignal) => Promise<unknown>, signal?: AbortSignal): Promise<LibraryRow> {
  const lockbox = isLockboxBundle(clip.entry);
  let analysisUrl: string;
  try { analysisUrl = resolveRealBundleUrls(indexUrl, clip.entry).analysisUrl; } catch { return { clip, observed: null, library: null }; }
  const libraryUrl = libraryUrlFor(analysisUrl);
  const observedUrl = lockbox ? undefined : siblingUrl(analysisUrl, 'observed.json');
  const [libraryValue, observedValue] = await Promise.all([
    libraryUrl ? fetchJson(libraryUrl, signal) : null,
    observedUrl ? fetchJson(observedUrl, signal) : null,
  ]);
  return { clip, library: parseLibrary(libraryValue, clip.video_id), observed: parseObserved(observedValue, { lockbox }) };
}

function ClipRow({ row, onOpenClip }: { row: LibraryRow; onOpenClip: (clip: DemoClip) => void }) {
  const { clip, observed, library } = row;
  const tier = clip.tier ?? observed?.tier ?? null;
  const open = (event?: Click) => { event?.preventDefault(); onOpenClip(clip); };
  const score = (key: ScoreKey) => library?.scores.find(s => s.key === key) ?? null;
  return h('tr', { className: 'library-clip-row', onClick: () => open() },
    h('td', null, tier ? h('span', { className: `tier-chip is-${tier}` }, (observed?.tier === tier && observed.tier_label) || TIER_LABELS[tier]) : '—'),
    h('th', { scope: 'row' }, h('a', { href: `?clip=${encodeURIComponent(clip.video_id)}`, title: clipHeading(clip), onClick: (event: Click & { stopPropagation?: () => void }) => { event.stopPropagation?.(); open(event); } }, clip.deal_label ?? clip.video_id)),
    h('td', null, clip.platform ? platformLabel(clip.platform) : '—'),
    h('td', { className: 'num' }, lengthLabel(clip.duration_ms)),
    h('td', { className: 'num' }, formatCount(observed?.views ?? null)),
    h('td', { className: 'num' }, formatMultiplier(observed?.views_vs_account_usual_x ?? null)),
    h('td', { className: 'num' }, formatPercent(observed?.engagement_rate_pct ?? null)),
    ...LIBRARY_SCORE_KEYS.map(key => {
      const s = score(key);
      return h('td', { key }, s?.verdict ? h('span', { className: `verdict-chip ${s.verdict}` }, verdictLabel(s.verdict)) : '—');
    }));
}

/** Internal library page: the demo clips as a table plus what great clips have in common. Node-renderable. */
export default function LibraryFrame({ patterns, theory = null, rows, onBack, onOpenClip, onResults }: {
  patterns: Patterns; theory?: Theory | null; rows: LibraryRow[]; onBack: () => void; onOpenClip: (clip: DemoClip) => void; onResults?: () => void;
}) {
  const size = [
    patterns.n_library !== null ? `${patterns.n_library.toLocaleString('en-US')} clips` : null,
    patterns.n_deals !== null ? `from ${patterns.n_deals} deals` : null,
  ].filter(Boolean).join(' ');
  const tiers = TIER_ORDER.flatMap(tier => {
    const t = patterns.tiers[tier];
    return t && (t.rule_plain || t.n_contents !== null) ? [{ tier, ...t }] : [];
  });
  return h('div', { className: 'learned-page library-page' },
    h('header', { className: 'demo-topbar library-top' },
      h('button', { type: 'button', className: 'topbar-button is-secondary', onClick: onBack }, '← Back'),
      h('h1', null, 'Library'),
      h('div', { className: 'demo-topbar-actions' }, h('span', { className: 'internal-badge' }, LIBRARY_BADGE))),
    h('main', { className: 'library-main' },
      h('section', { className: 'library-section library-summary', 'aria-label': 'Library' },
        size ? h('p', { className: 'library-size' }, h('strong', null, size),
          patterns.n_tiered !== null ? h('span', null, ` · ${patterns.n_tiered.toLocaleString('en-US')} of them in a performance group`) : null) : null,
        patterns.outcome_plain ? h('p', null, patterns.outcome_plain) : null,
        tiers.length ? h('ul', { className: 'library-tiers' }, tiers.map(t => h('li', { key: t.tier },
          h('span', { className: `tier-chip is-${t.tier}` }, t.label || TIER_LABELS[t.tier]),
          t.n_contents !== null ? ` ${t.n_contents.toLocaleString('en-US')} clips` : null,
          t.rule_plain ? ` · ${t.rule_plain}` : null))) : null),
      theory ? h(TheorySection, { theory, onResults }) : null,
      rows.length ? h('section', { className: 'library-section', 'aria-labelledby': 'library-clips-title' },
        h('h2', { id: 'library-clips-title' }, `The clips (${rows.length})`),
        h('table', { className: 'library-clip-table' },
          h('thead', null, h('tr', null,
            ...['Tier', 'Deal', 'Platform', 'Length', 'Views', '× usual', 'Engagement'].map(label => h('th', { key: label, scope: 'col' }, label)),
            ...LIBRARY_SCORE_KEYS.map(key => h('th', { key, scope: 'col' }, SCORE_TITLES[key])))),
          h('tbody', null, rows.map(row => h(ClipRow, { key: row.clip.video_id, row, onOpenClip })))),
        h('p', { className: 'library-note' }, 'Views and engagement are observed on the platform. Opening, middle, best 3 s and ending compare the predicted brain response with the same part of other library clips (above or below the library, not strong or weak); they are not quality scores and did not go with views across the library.')) : null,
      theory ? null : h(PatternsSection, { patterns }),
      h('p', { className: 'learned-license' }, 'Research preview · non-commercial (TRIBE CC-BY-NC)')));
}
