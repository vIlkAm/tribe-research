import { createElement as h, type ReactNode } from 'react';
import type { CompareColumn, CompareModel } from '../lib/observed.ts';

/**
 * Page shell of the internal comparison view. The first screen holds a compact
 * header, the caveat (verbatim) and both columns' stages (clip + response strip);
 * per-column details (scorecard, observed numbers) follow below the fold.
 * Written without JSX so it can be rendered in node tests.
 */
export default function CompareFrame({ model, onBack, onLearned, renderStage, renderBelow, renderHeader }: {
  model: CompareModel; onBack: () => void; onLearned: () => void;
  renderStage: (column: CompareColumn) => ReactNode;
  renderBelow?: (column: CompareColumn) => ReactNode;
  renderHeader?: (column: CompareColumn) => ReactNode;
}) {
  return h('div', { className: 'learned-page compare-page' },
    h('div', { className: 'compare-screen' },
      h('header', { className: 'demo-topbar compare-top' },
        h('button', { type: 'button', className: 'topbar-button is-secondary', onClick: onBack }, '← Back'),
        h('h1', null, model.title),
        h('div', { className: 'demo-topbar-actions' },
          h('span', { className: 'internal-badge' }, 'Internal research view'))),
      h('div', { className: 'compare-caveat', role: 'note' },
        h('p', null, model.caveat, model.same ? ` Both clips: ${model.same}.` : ''),
        h('a', { href: '?view=learned', onClick: (event: { preventDefault: () => void }) => { event.preventDefault(); onLearned(); } }, 'What the model learned →')),
      h('div', { className: 'compare-grid' }, model.columns.map(column =>
        h('section', { key: column.role, className: `compare-column is-${column.role}`, 'aria-label': column.title },
          h('div', { className: 'compare-column-head' },
            h('h2', { className: 'compare-column-title' }, column.title),
            renderHeader?.(column)),
          h('div', { className: 'compare-stage' }, renderStage(column)))))),
    h('div', { className: 'compare-below' },
      renderBelow && h('div', { className: 'compare-below-grid' }, model.columns.map(column =>
        h('section', { key: column.role, 'aria-label': `${column.title}: details` },
          h('h2', null, column.title),
          renderBelow(column)))),
      model.rule && h('details', { className: 'compare-rule' }, h('summary', null, 'How this pair was picked'), h('p', null, model.rule)),
      h('p', { className: 'learned-license' }, 'Research preview · non-commercial (TRIBE CC-BY-NC)')));
}
