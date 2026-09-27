/**
 * Internal-only observed platform metrics (`observed.json`, nvi.observed.v0) and the
 * example pair list (`examples.json`, nvi.examples.v0).
 *
 * Both are optional and served only by the local research server. Parsers are
 * defensive and return `null` for anything unusable, so absent files render nothing.
 * Sealed lockbox clips never show observed numbers, even when a file exists.
 */
import type { Analysis } from '../data/analysis.types.ts';
import type { RealBundleIndex, RealBundleIndexEntry } from './real-analysis-index.ts';

export const OBSERVED_SCHEMA = 'nvi.observed.v0';
export const EXAMPLES_SCHEMA = 'nvi.examples.v0';

type Obj = Record<string, unknown>;
const isObj = (value: unknown): value is Obj => !!value && typeof value === 'object' && !Array.isArray(value);
const str = (value: unknown): string => typeof value === 'string' ? value.trim() : '';
const num = (value: unknown): number | null => typeof value === 'number' && Number.isFinite(value) ? value : null;
const count = (value: unknown): number | null => { const n = num(value); return n === null || n < 0 ? null : n; };
const schemaOf = (value: Obj): unknown => value.schema_version ?? value.schema;
const VIDEO_ID = /^[A-Za-z0-9][A-Za-z0-9_-]*$/;

export type ExampleRole = 'fell_short' | 'beat_expectations';
export const ROLE_TITLES: Record<ExampleRole, string> = { fell_short: 'Fell short', beat_expectations: 'Beat expectations' };

export interface Observed {
  platform: string;
  video_link: string | null;
  upload_date: string;
  views: number | null;
  likes: number | null;
  comments: number | null;
  shares: number | null;
  saves: number | null;
  engagement_rate_pct: number | null;
  views_vs_account_usual_x: number | null;
  account_usual_n_posts: number | null;
  caption: string;
  caption_null: string;
  vs_expectation: { role: ExampleRole | null; plain: string } | null;
}

/** Lockbox status from every place it can be recorded; any positive signal wins. */
export function isLockboxBundle(entry?: Pick<RealBundleIndexEntry, 'is_lockbox' | 'performance'> | null, analysis?: Pick<Analysis, 'performance'> | null): boolean {
  return entry?.is_lockbox === true || entry?.performance?.clip_in_training === 'lockbox' || analysis?.performance?.clip_in_training === 'lockbox';
}

function safeLink(value: unknown): string | null {
  const text = str(value);
  if (!text) return null;
  try { const url = new URL(text); return url.protocol === 'https:' ? url.href : null; } catch { return null; }
}

/**
 * Parse `observed.json`. `lockbox: true` always returns null: sealed test clips
 * never show observed outcomes, whatever the file says.
 */
export function parseObserved(value: unknown, options: { lockbox: boolean }): Observed | null {
  if (options.lockbox) return null;
  if (!isObj(value) || schemaOf(value) !== OBSERVED_SCHEMA) return null;
  const vs = isObj(value.vs_expectation) ? value.vs_expectation : null;
  const role = vs?.role === 'fell_short' || vs?.role === 'beat_expectations' ? vs.role : null;
  const observed: Observed = {
    platform: str(value.platform),
    video_link: safeLink(value.video_link),
    upload_date: /^\d{4}-\d{2}-\d{2}/.test(str(value.upload_date)) ? str(value.upload_date).slice(0, 10) : '',
    views: count(value.views), likes: count(value.likes), comments: count(value.comments), shares: count(value.shares), saves: count(value.saves),
    engagement_rate_pct: count(value.engagement_rate_pct),
    views_vs_account_usual_x: count(value.views_vs_account_usual_x),
    account_usual_n_posts: count(value.account_usual_n_posts),
    caption: str(value.caption),
    caption_null: str(value.caption_null),
    vs_expectation: vs && (role || str(vs.plain)) ? { role, plain: str(vs.plain) } : null,
  };
  return observed.views !== null || observed.likes !== null || observed.engagement_rate_pct !== null ? observed : null;
}

/** Sibling file next to `analysis.json`, same origin only. */
export function siblingUrl(analysisUrl: string, name: string): string | undefined {
  try {
    const base = new URL(analysisUrl, typeof window === 'undefined' ? 'http://localhost/' : window.location.href);
    const resolved = new URL(name, base);
    return resolved.origin === base.origin ? resolved.href : undefined;
  } catch { return undefined; }
}

export function formatCount(value: number | null): string {
  if (value === null) return '—';
  return new Intl.NumberFormat('en-US', { notation: value >= 10_000 ? 'compact' : 'standard', maximumFractionDigits: 1 }).format(value);
}

export function formatMultiplier(value: number | null): string {
  if (value === null) return '—';
  return value >= 10 ? `${Math.round(value)}×` : `${value.toFixed(2)}×`;
}

export function formatPercent(value: number | null): string {
  return value === null ? '—' : `${value.toFixed(value >= 10 ? 1 : 2)}%`;
}

export function formatDate(value: string): string {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value);
  if (!match) return value;
  const date = new Date(Date.UTC(Number(match[1]), Number(match[2]) - 1, Number(match[3])));
  return new Intl.DateTimeFormat('en-GB', { day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC' }).format(date);
}

// ---------------------------------------------------------------------------
// examples.json

export interface ExamplePair { fell_short: string; beat_expectations: string; same: string }
export interface Examples { title: string; pairs: ExamplePair[]; rule: string; caveat: string; caption_null: string }

export function parseExamples(value: unknown): Examples | null {
  if (!isObj(value) || schemaOf(value) !== EXAMPLES_SCHEMA) return null;
  const pairs = (Array.isArray(value.pairs) ? value.pairs : []).flatMap(pair => {
    if (!isObj(pair)) return [];
    const fell = str(pair.fell_short), beat = str(pair.beat_expectations);
    if (!VIDEO_ID.test(fell) || !VIDEO_ID.test(beat) || fell === beat) return [];
    return [{ fell_short: fell, beat_expectations: beat, same: str(pair.same) }];
  });
  const caveat = str(value.caveat);
  // The caveat is mandatory: the pair must never be shown without it.
  if (!pairs.length || !caveat) return null;
  return { title: str(value.title), pairs, rule: str(value.rule), caveat, caption_null: str(value.caption_null) };
}

export interface CompareColumn { role: ExampleRole; title: string; entry: RealBundleIndexEntry }
export interface CompareModel { title: string; caveat: string; same: string; rule: string; caption_null: string; columns: [CompareColumn, CompareColumn] }

/** First pair whose two clips are both in the index; fell short on the left. */
export function compareModel(examples: Examples | null, index: RealBundleIndex | null): CompareModel | null {
  if (!examples || !index) return null;
  for (const pair of examples.pairs) {
    const fell = index.bundles.find(bundle => bundle.video_id === pair.fell_short);
    const beat = index.bundles.find(bundle => bundle.video_id === pair.beat_expectations);
    if (!fell || !beat) continue;
    return {
      title: examples.title || 'A clip that fell short vs one that beat expectations',
      caveat: examples.caveat, same: pair.same, rule: examples.rule, caption_null: examples.caption_null,
      columns: [
        { role: 'fell_short', title: ROLE_TITLES.fell_short, entry: fell },
        { role: 'beat_expectations', title: ROLE_TITLES.beat_expectations, entry: beat },
      ],
    };
  }
  return null;
}
