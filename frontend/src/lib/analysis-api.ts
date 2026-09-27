import type { Analysis } from '../data/analysis.types.ts';
import { assertAnalysis } from './analysis.ts';
import type { LocalBundle } from './local-bundle.ts';
import type { PerformanceIndexSummary } from './real-analysis-index.ts';

export type JobState = 'queued' | 'fetching' | 'extracting' | 'predicting' | 'done' | 'failed';
export type Platform = 'tiktok' | 'instagram' | 'youtube';

export interface JobSnapshot {
  id: string;
  state: JobState;
  message?: string;
  error_code?: string;
  analysis_url?: string;
  analysis?: unknown;
}

export interface AnalysisContext {
  deal_id: string;
  platform: Platform;
  account_id?: string;
}

const states = new Set<JobState>(['queued', 'fetching', 'extracting', 'predicting', 'done', 'failed']);

export function apiBaseUrl(value: string | undefined): string | null {
  if (!value?.trim()) return null;
  const url = new URL(value);
  if (url.protocol !== 'https:' && !(url.protocol === 'http:' && ['localhost', '127.0.0.1'].includes(url.hostname))) {
    throw new Error('The analysis API must use HTTPS outside local development.');
  }
  return url.href.replace(/\/$/, '');
}

function parseJob(value: unknown): JobSnapshot {
  const job = value as Partial<JobSnapshot> | null;
  if (!job || typeof job.id !== 'string' || !job.id || typeof job.state !== 'string' || !states.has(job.state as JobState)) {
    throw new Error('The analysis service returned an invalid job response.');
  }
  if (job.message !== undefined && typeof job.message !== 'string') throw new Error('The analysis service returned an invalid job message.');
  if (job.error_code !== undefined && typeof job.error_code !== 'string') throw new Error('The analysis service returned an invalid error code.');
  if (job.analysis_url !== undefined && typeof job.analysis_url !== 'string') throw new Error('The analysis service returned an invalid analysis URL.');
  return job as JobSnapshot;
}

async function responseJson(response: Response): Promise<unknown> {
  let value: unknown;
  try { value = await response.json(); }
  catch { throw new Error(`The analysis service returned ${response.status || 'an unreadable response'}.`); }
  if (!response.ok) {
    const detail = value as { message?: unknown; error_code?: unknown } | null;
    const message = typeof detail?.message === 'string' ? detail.message : typeof detail?.error_code === 'string' ? detail.error_code.replaceAll('_', ' ') : `Request failed (${response.status}).`;
    throw new Error(message);
  }
  return value;
}

export async function submitLink(base: string, link: string, context: AnalysisContext, signal?: AbortSignal): Promise<JobSnapshot> {
  const response = await fetch(`${base}/v1/jobs`, {
    method: 'POST', signal, headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
    body: JSON.stringify({ link, ...context, account_id: context.account_id || null }),
  });
  return parseJob(await responseJson(response));
}

export async function submitUpload(base: string, file: File, context: AnalysisContext, signal?: AbortSignal): Promise<JobSnapshot> {
  const body = new FormData();
  body.set('upload', file);
  body.set('deal_id', context.deal_id);
  body.set('platform', context.platform);
  if (context.account_id) body.set('account_id', context.account_id);
  const response = await fetch(`${base}/v1/jobs`, { method: 'POST', signal, headers: { Accept: 'application/json' }, body });
  return parseJob(await responseJson(response));
}

export async function getJob(base: string, id: string, signal?: AbortSignal): Promise<JobSnapshot> {
  const response = await fetch(`${base}/v1/jobs/${encodeURIComponent(id)}`, { signal, headers: { Accept: 'application/json' } });
  return parseJob(await responseJson(response));
}

function remoteAssetResolver(analysisUrl: string) {
  const base = new URL(analysisUrl);
  return (relative: string): string | null => {
    if (!relative || /^[a-z][a-z\d+.-]*:/i.test(relative) || /^[\\/]/.test(relative) || /[\\?#\u0000-\u001f]/.test(relative)) return null;
    const resolved = new URL(relative, base);
    return resolved.origin === base.origin ? resolved.href : null;
  };
}

export interface RemoteAnalysisOptions {
  signal?: AbortSignal;
  performanceUrl?: string;
  expectedPerformance?: PerformanceIndexSummary | null;
}

export async function openRemoteAnalysis(analysisUrl: string, options: RemoteAnalysisOptions = {}): Promise<LocalBundle> {
  const base = typeof window === 'undefined' ? 'http://localhost/' : window.location.href;
  const url = new URL(analysisUrl, base).href;
  const performanceUrl = options.performanceUrl ? new URL(options.performanceUrl, base).href : null;
  if (performanceUrl && new URL(performanceUrl).origin !== new URL(url).origin) throw new Error('The performance result must use the analysis origin.');
  const [analysisResponse, performanceResponse] = await Promise.all([
    fetch(url, { signal: options.signal, headers: { Accept: 'application/json' } }),
    performanceUrl ? fetch(performanceUrl, { signal: options.signal, headers: { Accept: 'application/json' } }) : Promise.resolve(null),
  ]);
  const analysisValue = await responseJson(analysisResponse);
  const performanceValue = performanceResponse ? await responseJson(performanceResponse) : undefined;
  if (!analysisValue || typeof analysisValue !== 'object' || Array.isArray(analysisValue)) throw new Error('The analysis service returned an invalid analysis bundle.');
  const value = performanceValue === undefined ? analysisValue : { ...analysisValue, performance: performanceValue };
  assertAnalysis(value);
  const analysis = value as Analysis;
  if (performanceValue !== undefined) {
    const provenanceVideoId = analysis.performance?.provenance.video_id;
    if (provenanceVideoId !== analysis.video_id) throw new Error('The performance result belongs to a different clip.');
    const expected = options.expectedPerformance;
    if (expected && (analysis.performance?.model_status !== expected.model_status || analysis.performance.validated !== expected.validated || analysis.performance.clip_in_training !== expected.clip_in_training)) {
      throw new Error('The performance result does not match the analysis index.');
    }
  }
  const source = new URL(url).pathname.split('/').filter(Boolean).slice(-2).join('/') || 'Remote analysis';
  return { analysis, filename: source, origin: 'remote', warnings: [], resolveAsset: remoteAssetResolver(url), dispose: () => undefined };
}

export async function openJobAnalysis(job: JobSnapshot, signal?: AbortSignal): Promise<LocalBundle> {
  if (job.analysis === undefined) {
    if (!job.analysis_url) throw new Error('The completed job did not include an analysis bundle.');
    return openRemoteAnalysis(job.analysis_url, { signal });
  }
  assertAnalysis(job.analysis);
  return { analysis: job.analysis as Analysis, filename: 'Completed analysis', origin: 'remote', warnings: [], resolveAsset: () => null, dispose: () => undefined };
}
