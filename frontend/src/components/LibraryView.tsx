import { useEffect, useState } from 'react';
import type { DemoClip } from '../lib/demo';
import type { Patterns } from '../lib/patterns';
import type { Theory } from '../lib/theory';
import { fetchOptionalJson } from '../lib/real-bundle';
import LibraryFrame, { loadLibraryRow, type LibraryRow } from './library-frame';
import './library.css';
import './demo-layout.css';

const emptyRows = (clips: DemoClip[]): LibraryRow[] => clips.map(clip => ({ clip, observed: null, library: null }));

/** Internal, exploratory library view (`?view=library`). */
export default function LibraryView({ indexUrl, clips, patterns, theory = null, onBack, onOpenClip, onResults }: { indexUrl?: string; clips: DemoClip[]; patterns: Patterns; theory?: Theory | null; onBack: () => void; onOpenClip: (clip: DemoClip) => void; onResults?: () => void }) {
  const [rows, setRows] = useState<LibraryRow[]>(() => emptyRows(clips));
  useEffect(() => {
    setRows(emptyRows(clips));
    if (!indexUrl || !clips.length) return;
    const controller = new AbortController();
    void Promise.all(clips.map(clip => loadLibraryRow(indexUrl, clip, fetchOptionalJson, controller.signal)))
      .then(loaded => { if (!controller.signal.aborted) setRows(loaded); });
    return () => controller.abort();
  }, [indexUrl, clips]);
  return <LibraryFrame patterns={patterns} theory={theory} rows={rows} onBack={onBack} onOpenClip={onOpenClip} onResults={onResults} />;
}
