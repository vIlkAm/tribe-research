import type { PerformanceModelStatus } from '../data/analysis.types.ts';

export interface PerformanceIndexSummary {
  model_status: PerformanceModelStatus;
  validated: boolean;
  clip_in_training: 'no' | 'train_oof' | 'lockbox';
}

export interface RealBundleIndexEntry {
  video_id: string;
  analysis_id: string;
  path: string;
  duration_ms: number;
  status: string;
  synthetic: boolean;
  n_channels: number;
  n_moments: number;
  has_words: boolean;
  n_warnings: number;
  performance_path?: string | null;
  performance?: PerformanceIndexSummary | null;
  platform?: string;
  video_link?: string | null;
  is_lockbox?: boolean;
}

export interface RealBundleIndex {
  schema_version: string;
  count: number;
  synthetic: boolean;
  release?: string | null;
  model_version?: string | null;
  model_release?: string | null;
  performance_schema?: string;
  performance_status_counts?: Record<string, number>;
  bundles: RealBundleIndexEntry[];
}

const performanceStates = new Set<PerformanceModelStatus>(['not_trained', 'preliminary', 'research_preview', 'validated', 'out_of_scope']);
const trainingStates = new Set(['no', 'train_oof', 'lockbox']);

export function parseRealBundleIndex(value: unknown): RealBundleIndex {
  const index = value as RealBundleIndex | null;
  if (!index || index.schema_version !== 'nvi.analysis.v0.2' || index.synthetic !== false || !Number.isInteger(index.count) || !Array.isArray(index.bundles) || index.count !== index.bundles.length) {
    throw new Error('The real analysis index is invalid.');
  }
  for (const bundle of index.bundles) {
    if (!bundle || typeof bundle.video_id !== 'string' || !bundle.video_id || typeof bundle.analysis_id !== 'string' || !bundle.analysis_id || typeof bundle.path !== 'string' || !bundle.path || !(bundle.duration_ms > 0) || bundle.synthetic !== false || !Number.isInteger(bundle.n_channels) || !Number.isInteger(bundle.n_moments) || !Number.isInteger(bundle.n_warnings) || typeof bundle.has_words !== 'boolean') {
      throw new Error('The real analysis index contains an invalid bundle.');
    }
    const hasPerformancePath = typeof bundle.performance_path === 'string' && bundle.performance_path.length > 0;
    const hasPerformance = bundle.performance !== undefined && bundle.performance !== null;
    if (hasPerformancePath !== hasPerformance) throw new Error('The real analysis index has incomplete performance metadata.');
    if (hasPerformance) {
      const summary = bundle.performance!;
      if (!performanceStates.has(summary.model_status) || typeof summary.validated !== 'boolean' || !trainingStates.has(summary.clip_in_training) || summary.validated !== (summary.model_status === 'validated')) {
        throw new Error('The real analysis index contains invalid performance metadata.');
      }
      if (bundle.is_lockbox !== undefined && bundle.is_lockbox !== (summary.clip_in_training === 'lockbox')) {
        throw new Error('The real analysis index contains inconsistent lockbox metadata.');
      }
    }
  }
  return index;
}

function resolveRelative(indexUrl: string, path: string, label: string): string {
  if (/^[a-z][a-z\d+.-]*:/i.test(path) || /^[\\/]/.test(path) || /[\\?#\u0000-\u001f]/.test(path)) throw new Error(`The ${label} path is invalid.`);
  const base = new URL(indexUrl, typeof window === 'undefined' ? 'http://localhost/' : window.location.href);
  const resolved = new URL(path, base);
  if (resolved.origin !== base.origin) throw new Error(`The ${label} must use the analysis index origin.`);
  return resolved.href;
}

export function resolveRealBundleUrls(indexUrl: string, bundle: RealBundleIndexEntry): { analysisUrl: string; performanceUrl?: string } {
  return {
    analysisUrl: resolveRelative(indexUrl, bundle.path, 'analysis'),
    ...(bundle.performance_path ? { performanceUrl: resolveRelative(indexUrl, bundle.performance_path, 'performance') } : {}),
  };
}
