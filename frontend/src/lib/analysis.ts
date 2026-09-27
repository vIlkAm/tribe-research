import Ajv2020 from 'ajv/dist/2020.js';
import schema from '../data/analysis.schema.json' with { type: 'json' };
import type { Analysis, Channel, Performance, PerformanceMetric, Sprite } from '../data/analysis.types.ts';

export const BUNDLE_PATH = 'data/cd20b16879d630c4/analysis.json';

export function sampleAt(channel: Channel, timeMs: number, durationMs: number): number | null {
  if (timeMs < 0 || timeMs >= durationMs) return null;
  return channel.values[Math.floor(timeMs / 1000 * channel.hz)] ?? null;
}

export function frameAt(sprite: Sprite, timeMs: number): number {
  return sprite.frame_start_ms.findIndex((start, i) => start <= timeMs && timeMs < sprite.frame_end_ms[i]);
}

export function formatTime(ms: number, decimals = 1): string {
  const seconds = Math.max(0, ms) / 1000;
  return `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${(seconds % 60).toFixed(decimals).padStart(decimals ? 3 + decimals : 2, '0')}`;
}

export function signalText(value: number | null): string {
  return value === null ? '—' : `${value >= 0 ? '+' : ''}${value.toFixed(2)}`;
}

export function isGap(analysis: Analysis, ms: number): boolean {
  return ms < 0 || ms >= analysis.duration_ms || analysis.timing.gaps_ms.some(([start, end]) => start <= ms && ms < end);
}

export function stepPath(values: (number | null)[], hz: number, durationMs: number, width: number, height: number, range = 3): string {
  let path = '';
  let connected = false;
  const y = (value: number) => height / 2 - Math.max(-range, Math.min(range, value)) / range * (height / 2 - 4);
  values.forEach((value, i) => {
    const start = i / hz * 1000;
    if (start >= durationMs) return;
    if (value === null) { connected = false; return; }
    const x = start / durationMs * width;
    const end = Math.min((i + 1) / hz * 1000, durationMs) / durationMs * width;
    path += `${connected ? 'L' : 'M'}${x.toFixed(2)},${y(value).toFixed(2)}H${end.toFixed(2)}`;
    connected = true;
  });
  return path;
}

// Keep the upstream schema unchanged on disk. TEAM.md permits additive unknown fields.
const runtimeSchema = structuredClone(schema);
function allowExtraFields(node: unknown) {
  if (!node || typeof node !== 'object') return;
  const object = node as Record<string, unknown>;
  if (object.additionalProperties === false) object.additionalProperties = true;
  Object.values(object).forEach(allowExtraFields);
}
allowExtraFields(runtimeSchema);
const validate = new Ajv2020({ strict: false, allErrors: false }).compile(runtimeSchema);

export function assertAnalysis(value: unknown): asserts value is Analysis {
  const a = value as Analysis | null;
  if (!a || a.schema_version !== 'nvi.analysis.v0.2') throw new Error('This demo supports nvi.analysis.v0.2 analysis bundles.');
  if (!validate(value)) throw new Error(`Invalid analysis bundle: ${validate.errors?.[0]?.instancePath || 'root'} ${validate.errors?.[0]?.message || ''}.`);
  if (a.status !== 'complete') return;
  if (!['proxies_version', 'feature_version', 'normalization'].every(key => typeof a.provenance[key] === 'string')) throw new Error('The analysis provenance is invalid.');
  const span = (s: number[]) => s.length === 2 && s[0] >= 0 && s[1] >= s[0];
  if (!(a.duration_ms > 0) || !(a.timing.native_tr_s > 0) || !span(a.timing.onset_window_ms) || a.timing.gaps_ms.some(s => !span(s))) throw new Error('The analysis timing is invalid.');
  if (new Set(a.channels.map(c => c.key)).size !== a.channels.length) throw new Error('The analysis has duplicate channel identities.');
  if (a.channels.some(c => !(c.hz > 0) || !c.values.every(v => v === null || Number.isFinite(v)) || !['tooltip', 'rise', 'fall'].every(k => typeof c.copy[k as keyof typeof c.copy] === 'string') || !Array.isArray(c.basis.roi_groups) || typeof c.basis.regions_text !== 'string')) throw new Error('The analysis contains an invalid signal.');
  if (a.events.unavailable_lanes.some(l => typeof l.key !== 'string' || typeof l.reason !== 'string')) throw new Error('The event lane metadata is invalid.');
  if (a.performance) assertPerformance(a.performance);
  for (const sprite of [a.assets.brain_map, a.assets.research_vertex_map]) {
    if (sprite && (!(sprite.tile_w > 0 && sprite.tile_h > 0 && sprite.cols > 0 && sprite.rows > 0) || sprite.frame_start_ms.length !== sprite.frame_end_ms.length || sprite.frame_start_ms.length > sprite.rows * sprite.cols || sprite.frame_start_ms.some((t, i) => t < 0 || sprite.frame_end_ms[i] <= t))) throw new Error('The cortical image timing is invalid.');
  }
}

function assertPerformanceMetric(metric: PerformanceMetric) {
  const percentile = (value: number) => Number.isFinite(value) && value >= 0 && value <= 1;
  const noRank = metric.percentile_deal_platform === null && metric.likely_range === null && metric.reference_n < 30;
  const ranked = metric.percentile_deal_platform !== null && percentile(metric.percentile_deal_platform) && metric.likely_range !== null && metric.likely_range.length === 2 && metric.likely_range.every(percentile) && metric.likely_range[0] <= metric.likely_range[1];
  if ((!noRank && !ranked) || !Number.isInteger(metric.reference_n) || metric.reference_n < 0 || (metric.percentile_account !== null && !percentile(metric.percentile_account))) throw new Error('The performance metric is invalid.');
}

function assertPerformance(performance: Performance) {
  const scored = ['preliminary', 'research_preview', 'validated'].includes(performance.model_status);
  if (!['not_trained', 'preliminary', 'research_preview', 'validated', 'out_of_scope'].includes(performance.model_status) || !performance.context || !Array.isArray(performance.drivers) || !Array.isArray(performance.warnings) || performance.validated !== (performance.model_status === 'validated')) throw new Error('The performance block is invalid.');
  if (!scored && (performance.engagement !== null || performance.reach !== null || performance.drivers.length > 0)) throw new Error('Unavailable performance states must not contain numbers.');
  if (scored && performance.engagement === null) throw new Error('Scored performance requires an engagement result.');
  if (performance.model_status === 'preliminary' && (typeof performance.caption !== 'string' || !performance.caption.startsWith('Preliminary model') || performance.brain_claim !== 'not_tested' || performance.engagement?.validation.scheme !== 'none')) throw new Error('Preliminary performance requires its disclosure caption.');
  if (performance.model_status !== 'preliminary' && performance.caption !== null) throw new Error('Only preliminary performance may carry a caption.');
  if (performance.engagement) assertPerformanceMetric(performance.engagement);
  if (performance.reach) assertPerformanceMetric(performance.reach);
  if (performance.drivers.some(driver => !Number.isFinite(driver.contribution))) throw new Error('The performance drivers are invalid.');
}
