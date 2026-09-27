import { createElement as h, type ReactNode } from 'react';
import type { CompareColumn, CompareModel } from '../lib/observed.ts';

/**
 * Page shell of the internal comparison view: the caveat, the learned link and two
 * labelled columns. Written without JSX so it can be rendered in node tests.
 */
export default function CompareFrame({ model, onBack, onLearned, renderColumn }: {
  model: CompareModel; onBack: () => void; onLearned: () => void; renderColumn: (column: CompareColumn) => ReactNode;
}) {
  return h('div', { className: 'learned-page compare-page' },
    h('header', { className: 'learned-top' },
      h('button', { type: 'button', className: 'learned-back', onClick: onBack }, '← Back to the analysis'),
      h('span', { className: 'internal-badge' }, 'Internal research view')),
    h('main', { className: 'compare-main' },
      h('h1', null, model.title),
      h('div', { className: 'compare-caveat', role: 'note' },
        h('p', null, model.caveat),
        h('a', { href: '?view=learned', onClick: (event: { preventDefault: () => void }) => { event.preventDefault(); onLearned(); } }, 'What the model learned →')),
      model.same && h('p', { className: 'compare-same' }, `Both clips: ${model.same}.`),
      h('div', { className: 'compare-grid' }, model.columns.map(column =>
        h('section', { key: column.role, className: `compare-column is-${column.role}`, 'aria-label': column.title },
          h('h2', { className: 'compare-column-title' }, column.title),
          renderColumn(column)))),
      model.rule && h('details', { className: 'compare-rule' }, h('summary', null, 'How this pair was picked'), h('p', null, model.rule)),
      h('p', { className: 'learned-license' }, 'Research preview · non-commercial (TRIBE CC-BY-NC)')));
}
