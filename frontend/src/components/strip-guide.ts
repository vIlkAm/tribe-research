import { createElement as h } from 'react';

export const STRIP_LEGEND = '50 = a typical clip of this length at this second · shaded = middle half of your clips · above = stronger predicted brain response';
export const STRIP_NOT_QUALITY = 'It is not a quality score or a views forecast.';
export const STRIP_PREDICTED = 'Predicted response of an average viewer’s brain (TRIBE v2), not measured.';

/** Contents of the strip's "How to read this" popover; the library-wide line is shown verbatim when loaded. */
export function StripGuide({ interpreterLine, onClose }: { interpreterLine?: string | null; onClose?: () => void }) {
  return h('div', { className: 'strip-guide', role: 'dialog', 'aria-label': 'How to read this' },
    h('div', { className: 'strip-guide-head' },
      h('strong', null, 'How to read this'),
      onClose ? h('button', { type: 'button', onClick: onClose, 'aria-label': 'Close' }, '×') : null),
    h('p', null, STRIP_LEGEND + '.'),
    h('p', { className: 'strip-guide-key' }, STRIP_NOT_QUALITY),
    interpreterLine ? h('p', { className: 'strip-guide-line' }, interpreterLine) : null,
    h('p', { className: 'strip-guide-note' }, STRIP_PREDICTED));
}
