import { createElement as h, type ReactNode } from 'react';
import type { CompareColumn, CompareModel, CompareSlot } from '../lib/observed.ts';
import type { DemoGroup } from '../lib/demo.ts';

const other = (model: CompareModel, column: CompareColumn) => model.columns[column.slot === 'a' ? 1 : 0];

/**
 * Page shell of the internal comparison view. The first screen holds a compact
 * header, the caveat (verbatim) and both columns' stages (clip + response strip);
 * per-column details (scorecard, observed numbers) follow below the fold.
 * Written without JSX so it can be rendered in node tests.
 */
export default function CompareFrame({ model, groups, onChoose, onBack, onLearned, renderStage, renderBelow, renderHeader }: {
  model: CompareModel; onBack: () => void; onLearned: () => void;
  /** Clips to choose from, grouped by tier; each column gets its own dropdown. */
  groups?: DemoGroup[]; onChoose?: (slot: CompareSlot, videoId: string) => void;
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
        h('p', null, model.caveat),
        h('a', { href: '?view=learned', onClick: (event: { preventDefault: () => void }) => { event.preventDefault(); onLearned(); } }, 'See the results →')),
      h('div', { className: 'compare-grid' }, model.columns.map(column =>
        h('section', { key: column.slot, className: `compare-column is-${column.tier ?? 'untiered'}`, 'aria-label': column.title },
          h('div', { className: 'compare-column-head' },
            h('h2', { className: 'compare-column-title' }, column.title),
            groups?.length ? h('select', {
              className: 'compare-select', value: column.clip.video_id, 'aria-label': `Clip ${column.slot.toUpperCase()}`,
              onChange: (event: { target: { value: string } }) => onChoose?.(column.slot, event.target.value),
            }, groups.map(group => h('optgroup', { key: group.label, label: group.label },
              group.clips.map(clip => h('option', { key: clip.video_id, value: clip.video_id, disabled: clip.video_id === other(model, column).clip.video_id }, clip.label))))) : null,
            renderHeader?.(column)),
          h('div', { className: 'compare-stage' }, renderStage(column)))))),
    h('div', { className: 'compare-below' },
      renderBelow && h('div', { className: 'compare-below-grid' }, model.columns.map(column =>
        h('section', { key: column.slot, 'aria-label': `${column.title}: details` },
          h('h2', null, column.title),
          renderBelow(column)))),
      model.rule && h('details', { className: 'compare-rule' }, h('summary', null, 'How this pair was picked'), h('p', null, model.rule)),
      h('p', { className: 'learned-license' }, 'Research preview · non-commercial (TRIBE CC-BY-NC)')));
}
