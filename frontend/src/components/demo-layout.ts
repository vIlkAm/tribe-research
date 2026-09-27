import { createElement as h, useEffect, useRef, type ReactNode } from 'react';
import { clipHeading, groupDemoClips, TIER_LABELS, type DemoClip } from '../lib/demo.ts';

/** Top bar height in px; `demo-layout.css` uses the same value for the stage height. */
export const TOPBAR_PX = 48;

/**
 * One-screen demo page: a 48 px top bar, then a stage exactly one viewport tall
 * (clip left; observed row, brain and response strip right), then everything else
 * below the fold. Written without JSX so the structure can be checked in node tests.
 */
export function DemoShell({ topbar, video, metrics, brain, strip, below, footer }: {
  topbar: ReactNode; video: ReactNode; metrics?: ReactNode; brain: ReactNode; strip?: ReactNode; below: ReactNode; footer: ReactNode;
}) {
  return h('div', { className: 'demo-app' },
    topbar,
    h('main', { className: 'demo-main' },
      h('section', { className: `demo-stage ${strip ? 'has-strip' : 'no-strip'} ${metrics ? 'has-metrics' : 'no-metrics'}`, 'aria-label': 'Clip, predicted brain response and response strip' },
        h('div', { className: 'demo-stage-video' }, video),
        h('div', { className: 'demo-stage-side' },
          metrics ? h('div', { className: 'demo-stage-metrics' }, metrics) : null,
          h('div', { className: 'demo-stage-brain' }, brain),
          strip ? h('div', { className: 'demo-stage-strip' }, strip) : null)),
      h('div', { className: 'demo-below' }, below),
      footer));
}

type ClickEvent = { preventDefault: () => void; button: number; metaKey: boolean; ctrlKey: boolean; shiftKey: boolean; altKey: boolean; currentTarget: unknown };

/**
 * Top-bar clip menu grouped by tier ("Did great", "Typical", "Did badly"). Each item
 * is a real `?clip=<video_id>` link, so clips can be opened in preloaded tabs.
 */
export function DemoClipPicker({ clips, current, busy, onChoose }: {
  clips: DemoClip[]; current: string | null; busy: string | null; onChoose: (clip: DemoClip) => void;
}) {
  const menu = useRef<HTMLDetailsElement>(null);
  useEffect(() => {
    // Close the open menu on an outside click or Escape so it never lingers over the stage.
    const close = (event: Event) => {
      const el = menu.current;
      if (!el?.open) return;
      if (event.type === 'keydown' ? (event as KeyboardEvent).key === 'Escape' : !el.contains(event.target as Node)) el.open = false;
    };
    document.addEventListener('pointerdown', close);
    document.addEventListener('keydown', close);
    return () => { document.removeEventListener('pointerdown', close); document.removeEventListener('keydown', close); };
  }, []);
  if (!clips.length) return null;
  const selected = clips.find(clip => clip.video_id === current) ?? null;
  return h('details', { className: 'demo-picker', ref: menu },
    h('summary', { 'aria-label': 'Choose a demo clip' },
      selected?.tier ? h('span', { className: `tier-chip is-${selected.tier}` }, TIER_LABELS[selected.tier]) : null,
      h('span', { className: 'demo-picker-current' }, selected ? selected.label : 'Choose a clip'),
      busy ? h('span', { className: 'demo-picker-busy' }, 'Opening…') : null),
    h('nav', { className: 'demo-picker-menu', 'aria-label': 'Demo clips' },
      groupDemoClips(clips).map(group => h('div', { key: group.label, className: `demo-picker-group${group.tier ? ` is-${group.tier}` : ''}` },
        h('span', { className: 'demo-picker-group-label' }, group.label),
        group.clips.map(clip => h('a', {
          key: clip.video_id,
          href: `?clip=${encodeURIComponent(clip.video_id)}`,
          className: `demo-picker-item${busy === clip.video_id ? ' is-busy' : ''}`,
          'aria-current': current === clip.video_id ? 'page' : undefined,
          title: `${clipHeading(clip)} · ${clip.video_id}`,
          onClick: (event: ClickEvent) => {
            if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
            event.preventDefault();
            const menu = (event.currentTarget as HTMLElement | null)?.closest?.('details');
            if (menu) menu.open = false;
            if (!busy && current !== clip.video_id) onChoose(clip);
          },
        }, clip.label))))));
}
