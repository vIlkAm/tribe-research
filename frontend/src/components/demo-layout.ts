import { createElement as h, type ReactNode } from 'react';
import type { DemoClip } from '../lib/demo.ts';

/** Top bar height in px; `demo-layout.css` uses the same value for the stage height. */
export const TOPBAR_PX = 48;

/**
 * One-screen demo page: a 48 px top bar, then a stage exactly one viewport tall
 * (clip left, brain over response strip right), then everything else below the fold.
 * Written without JSX so the structure can be checked in node tests.
 */
export function DemoShell({ topbar, video, brain, strip, below, footer }: {
  topbar: ReactNode; video: ReactNode; brain: ReactNode; strip?: ReactNode; below: ReactNode; footer: ReactNode;
}) {
  return h('div', { className: 'demo-app' },
    topbar,
    h('main', { className: 'demo-main' },
      h('section', { className: `demo-stage ${strip ? 'has-strip' : 'no-strip'}`, 'aria-label': 'Clip, predicted brain response and response strip' },
        h('div', { className: 'demo-stage-video' }, video),
        h('div', { className: 'demo-stage-side' },
          h('div', { className: 'demo-stage-brain' }, brain),
          strip ? h('div', { className: 'demo-stage-strip' }, strip) : null)),
      h('div', { className: 'demo-below' }, below),
      footer));
}

type ClickEvent = { preventDefault: () => void; button: number; metaKey: boolean; ctrlKey: boolean; shiftKey: boolean; altKey: boolean };

/**
 * Segmented demo clip list for the top bar, in index order. Each item is a real
 * `?clip=<video_id>` link, so clips can be opened in preloaded tabs.
 */
export function DemoClipPicker({ clips, current, busy, onChoose }: {
  clips: DemoClip[]; current: string | null; busy: string | null; onChoose: (clip: DemoClip) => void;
}) {
  if (!clips.length) return null;
  return h('nav', { className: 'demo-picker', 'aria-label': 'Demo clips' },
    clips.map(clip => h('a', {
      key: clip.video_id,
      href: `?clip=${encodeURIComponent(clip.video_id)}`,
      className: `demo-picker-item${clip.role ? ` is-${clip.role}` : ''}${busy === clip.video_id ? ' is-busy' : ''}`,
      'aria-current': current === clip.video_id ? 'page' : undefined,
      'aria-busy': busy === clip.video_id ? true : undefined,
      title: clip.role ? `${clip.label} · ${clip.video_id}` : clip.video_id,
      onClick: (event: ClickEvent) => {
        if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
        event.preventDefault();
        if (!busy && current !== clip.video_id) onChoose(clip);
      },
    }, clip.label)));
}
