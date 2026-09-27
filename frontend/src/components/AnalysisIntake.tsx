import { useEffect, useMemo, useRef, useState, type ChangeEvent, type FormEvent } from 'react';
import { ArrowUpRight, CircleAlert, FileVideo, Link2, LoaderCircle, Upload, X } from 'lucide-react';
import { apiBaseUrl, getJob, openJobAnalysis, submitLink, submitUpload, type AnalysisContext, type JobSnapshot, type Platform } from '../lib/analysis-api';
import type { LocalBundle } from '../lib/local-bundle';
import './analysis-intake.css';

const stateCopy: Record<string, string> = {
  queued: 'Waiting for a research worker', fetching: 'Receiving and validating the video',
  extracting: 'Running extractors and TRIBE v2', predicting: 'Building signals, moments and brain assets', done: 'Analysis complete', failed: 'Analysis failed',
};

export default function AnalysisIntake({ open, onClose, onOpen, onOpenExport }: {
  open: boolean; onClose: () => void; onOpen: (bundle: LocalBundle) => void; onOpenExport: () => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const controller = useRef<AbortController | null>(null);
  const [mode, setMode] = useState<'link' | 'upload'>('link');
  const [link, setLink] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [deal, setDeal] = useState('');
  const [platform, setPlatform] = useState<Platform>('tiktok');
  const [account, setAccount] = useState('');
  const [job, setJob] = useState<JobSnapshot | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const api = useMemo(() => {
    try { return apiBaseUrl(import.meta.env.VITE_ANALYSIS_API_URL); }
    catch { return null; }
  }, []);

  useEffect(() => {
    if (open) dialog.current?.showModal();
    else dialog.current?.close();
    if (!open) { controller.current?.abort(); controller.current = null; setBusy(false); setJob(null); setError(null); }
  }, [open]);

  useEffect(() => () => controller.current?.abort(), []);

  function chooseFile(event: ChangeEvent<HTMLInputElement>) {
    setFile(event.target.files?.[0] ?? null);
    event.target.value = '';
  }

  async function waitForResult(initial: JobSnapshot, signal: AbortSignal) {
    let current = initial;
    setJob(current);
    while (!signal.aborted && current.state !== 'done' && current.state !== 'failed') {
      await new Promise<void>((resolve, reject) => {
        const timer = window.setTimeout(resolve, 2000);
        signal.addEventListener('abort', () => { window.clearTimeout(timer); reject(new DOMException('Aborted', 'AbortError')); }, { once: true });
      });
      current = await getJob(api!, current.id, signal);
      setJob(current);
    }
    if (current.state === 'failed') throw new Error(current.message || current.error_code?.replaceAll('_', ' ') || 'The research worker could not analyze this video.');
    if (current.state === 'done') {
      const bundle = await openJobAnalysis(current, signal);
      onOpen(bundle);
      onClose();
    }
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!api || !deal.trim() || (mode === 'link' ? !link.trim() : !file)) return;
    controller.current?.abort();
    const request = new AbortController();
    controller.current = request;
    setBusy(true); setError(null); setJob(null);
    const context: AnalysisContext = { deal_id: deal.trim(), platform, ...(account.trim() ? { account_id: account.trim() } : {}) };
    try {
      const initial = mode === 'link' ? await submitLink(api, link.trim(), context, request.signal) : await submitUpload(api, file!, context, request.signal);
      await waitForResult(initial, request.signal);
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') return;
      setError(error instanceof Error ? error.message : 'The video could not be submitted.');
    } finally { if (!request.signal.aborted) setBusy(false); }
  }

  return <dialog ref={dialog} className="method-dialog analysis-intake" aria-labelledby="analysis-intake-title" onCancel={onClose} onClick={event => { if (event.target === event.currentTarget && !busy) onClose(); }}>
    <div className="dialog-head"><span className="eyebrow">REAL VIDEO ANALYSIS</span><button className="icon-button" aria-label="Close video analysis" onClick={onClose} disabled={busy}><X size={20} /></button></div>
    <h2 id="analysis-intake-title">Analyze a video.</h2>
    <p>Submit a company-owned clip to the research pipeline. Model weights stay on the backend; this browser receives only the completed analysis bundle.</p>
    {!api ? <div className="intake-unavailable"><CircleAlert size={21} /><div><strong>Live analysis is not connected yet.</strong><p>The research team is running the first real GPU batch and defining the service contract. You can already open a completed v0.2 export.</p><button onClick={() => { onClose(); onOpenExport(); }}>Open research export <ArrowUpRight size={15} /></button></div></div> : <form onSubmit={submit}>
      <div className="intake-tabs" role="tablist" aria-label="Video source"><button type="button" role="tab" aria-selected={mode === 'link'} className={mode === 'link' ? 'active' : ''} onClick={() => setMode('link')} disabled={busy}><Link2 size={16} /> Paste link</button><button type="button" role="tab" aria-selected={mode === 'upload'} className={mode === 'upload' ? 'active' : ''} onClick={() => setMode('upload')} disabled={busy}><Upload size={16} /> Upload video</button></div>
      {mode === 'link' ? <label>Video link<input type="url" value={link} onChange={event => setLink(event.target.value)} placeholder="https://…" required disabled={busy} /></label> : <div className="intake-file"><button type="button" onClick={() => fileInput.current?.click()} disabled={busy}><FileVideo size={19} /><span>{file ? file.name : 'Choose a video'}<small>MP4, AVI, MKV, MOV or WebM · 5–90 seconds</small></span></button><input ref={fileInput} type="file" accept="video/mp4,video/x-msvideo,video/x-matroska,video/quicktime,video/webm,.mp4,.avi,.mkv,.mov,.webm" hidden onChange={chooseFile} /></div>}
      <div className="intake-context"><label>Deal ID<input value={deal} onChange={event => setDeal(event.target.value)} required disabled={busy} /></label><label>Platform<select value={platform} onChange={event => setPlatform(event.target.value as Platform)} disabled={busy}><option value="tiktok">TikTok</option><option value="instagram">Instagram</option><option value="youtube">YouTube</option></select></label></div>
      <label>Account ID <span>optional</span><input value={account} onChange={event => setAccount(event.target.value)} disabled={busy} /></label>
      <p className="intake-scope">Research preview · non-commercial · links require platform approval. Uploaded clips must be yours or cleared for analysis.</p>
      {job && <div className={`intake-job ${job.state}`} role="status"><LoaderCircle size={18} className={job.state !== 'done' && job.state !== 'failed' ? 'spin' : ''} /><div><strong>{stateCopy[job.state]}</strong>{job.message && <span>{job.message}</span>}</div></div>}
      {error && <div className="analysis-import-error" role="alert">{error}</div>}
      <button className="intake-submit" type="submit" disabled={busy || !deal.trim() || (mode === 'link' ? !link.trim() : !file)}>{busy ? <LoaderCircle size={17} className="spin" /> : <ArrowUpRight size={17} />} {busy ? 'Analyzing…' : 'Start analysis'}</button>
    </form>}
  </dialog>;
}
