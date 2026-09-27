import { useEffect, useRef, useState, type ChangeEvent } from 'react';
import { ArrowUpRight, FileJson, FolderOpen, LoaderCircle, X } from 'lucide-react';
import { analysisFiles, filePath, openLocalBundle, type LocalBundle } from '../lib/local-bundle';
import { openRemoteAnalysis } from '../lib/analysis-api';
import { parseRealBundleIndex, resolveRealBundleUrls, type RealBundleIndex, type RealBundleIndexEntry } from '../lib/real-analysis-index';
import './analysis-picker.css';

export default function AnalysisPicker({ open, onClose, onOpen }: {
  open: boolean; onClose: () => void; onOpen: (bundle: LocalBundle) => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const folder = useRef<HTMLInputElement>(null);
  const filesInput = useRef<HTMLInputElement>(null);
  const operation = useRef(0);
  const [files, setFiles] = useState<File[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [realIndex, setRealIndex] = useState<RealBundleIndex | null>(null);
  const [realError, setRealError] = useState<string | null>(null);
  const realIndexUrl = import.meta.env.VITE_REAL_ANALYSIS_INDEX as string | undefined;
  const candidates = analysisFiles(files);

  useEffect(() => {
    if (open) {
      dialog.current?.showModal();
      if (realIndexUrl && !realIndex && !realError) fetch(realIndexUrl)
        .then(response => { if (!response.ok) throw new Error('The real analysis set could not be loaded.'); return response.json(); })
        .then(value => {
          setRealIndex(parseRealBundleIndex(value));
        })
        .catch(error => setRealError(error instanceof Error ? error.message : 'The real analysis set could not be loaded.'));
    }
    else {
      dialog.current?.close();
      operation.current++;
      setBusy(null);
      setError(null);
      setFiles([]);
    }
    return () => { operation.current++; };
  }, [open]);

  function choose(event: ChangeEvent<HTMLInputElement>) {
    const selection = Array.from(event.target.files ?? []);
    event.target.value = '';
    if (!selection.length) return;
    operation.current++;
    setBusy(null);
    setFiles(selection);
    setError(analysisFiles(selection).length ? null : 'No analysis JSON found. Choose the folder containing analysis.json and its image assets.');
  }

  async function load(file: File) {
    const id = ++operation.current;
    setBusy(filePath(file));
    setError(null);
    try {
      const bundle = await openLocalBundle(file, files, {
        create: file => URL.createObjectURL(file), revoke: url => URL.revokeObjectURL(url),
      });
      if (id !== operation.current) { bundle.dispose(); return; }
      onOpen(bundle);
      onClose();
    } catch (error) {
      if (id === operation.current) setError(error instanceof Error ? error.message : 'The analysis could not be opened.');
    } finally {
      if (id === operation.current) setBusy(null);
    }
  }

  async function loadReal(entry: RealBundleIndexEntry) {
    const id = ++operation.current;
    setBusy(entry.path); setError(null);
    try {
      const urls = resolveRealBundleUrls(realIndexUrl!, entry);
      const bundle = await openRemoteAnalysis(urls.analysisUrl, { performanceUrl: urls.performanceUrl, expectedPerformance: entry.performance });
      if (id !== operation.current) return;
      onOpen(bundle); onClose();
    } catch (error) {
      if (id === operation.current) setError(error instanceof Error ? error.message : 'The real analysis could not be opened.');
    } finally { if (id === operation.current) setBusy(null); }
  }

  return <dialog ref={dialog} className="method-dialog analysis-picker" aria-labelledby="analysis-picker-title" onCancel={onClose} onClick={event => { if (event.target === event.currentTarget) onClose(); }}>
    <div className="dialog-head"><span className="eyebrow">FROM YOUR RESEARCH TEAM</span><button className="icon-button" aria-label="Close analysis picker" onClick={onClose}><X size={20} /></button></div>
    <h2 id="analysis-picker-title">Open an analysis.</h2>
    <p>Choose a research export to explore its signals, moments and cortical atlas. Files stay in this browser tab and are cleared when you reload.</p>
    {realIndexUrl && <section className="real-analysis-set" aria-labelledby="real-analysis-heading"><div className="real-analysis-head"><div><strong id="real-analysis-heading">Approved real analysis set</strong><span>TRIBE analysis{realIndex?.model_version ? ' + preliminary performance' : ''} · average subject · no source footage</span></div><span>{realIndex ? `${realIndex.count} clips` : realError ? 'Unavailable' : 'Loading…'}</span></div>
      {realIndex && <div className="real-analysis-list">{realIndex.bundles.map((bundle, index) => <button key={bundle.video_id} disabled={!!busy} onClick={() => void loadReal(bundle)}><span className="real-analysis-number">{String(index + 1).padStart(2, '0')}</span><span><strong>{bundle.video_id}</strong><small>{(bundle.duration_ms / 1000).toFixed(1)} s · {bundle.n_moments} moments · {bundle.has_words ? 'speech' : 'no speech'}{bundle.platform ? ` · ${bundle.platform}` : ''}{bundle.performance?.model_status === 'preliminary' ? ' · preliminary model' : ''}{bundle.n_warnings ? ` · ${bundle.n_warnings} warning${bundle.n_warnings === 1 ? '' : 's'}` : ''}</small></span>{busy === bundle.path ? <LoaderCircle size={17} className="spin" /> : <ArrowUpRight size={17} />}</button>)}</div>}
      {realError && <p className="real-analysis-error">{realError}</p>}
    </section>}
    {realIndexUrl && <div className="analysis-source-divider"><span>Or open your own export</span></div>}
    <div className="analysis-file-actions">
      <button className="analysis-folder-button" disabled={!!busy} onClick={() => folder.current?.click()}><FolderOpen size={19} /><span>Choose folder<small>Analysis files + shared atlas images</small></span><ArrowUpRight size={17} /></button>
      <button className="analysis-json-button" disabled={!!busy} onClick={() => filesInput.current?.click()}><FileJson size={18} /> Choose files</button>
    </div>
    <input ref={folder} type="file" {...{ webkitdirectory: '' }} multiple hidden aria-label="Analysis export folder" onChange={choose} />
    <input ref={filesInput} type="file" accept=".json,.png,.jpg,.jpeg,.webp" multiple hidden aria-label="Analysis JSON and images" onChange={choose} />
    <div className="analysis-picker-help"><strong>Include the parent folder.</strong><span>Select the folder containing the clip folders and shared <code>_static</code> images. You can also open a JSON file alone to explore its timeline.</span></div>
    {error && <div className="analysis-import-error" role="alert">{error}</div>}
    {candidates.length > 0 && <div className="analysis-file-list"><div className="analysis-list-heading">{candidates.length} {candidates.length === 1 ? 'analysis file' : 'analysis files'}<span>Choose one to open</span></div>{candidates.map(file => <button key={filePath(file)} disabled={!!busy} onClick={() => void load(file)} aria-label={`Open ${filePath(file)}`}><FileJson size={18} /><span>{filePath(file)}</span>{busy === filePath(file) ? <LoaderCircle size={17} className="spin" /> : <ArrowUpRight size={17} />}</button>)}</div>}
    <p className="analysis-picker-note">Supports nvi.analysis.v0.2. Real analyses use their supplied contracted atlas; the 3D surface remains restricted to the synthetic demo until issue #1 is agreed.</p>
  </dialog>;
}
