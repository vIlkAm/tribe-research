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
import { chooseDemoClip, clipHeading, demoClips, parseTier, TIER_LABELS, VIDEO_ID, type DemoClip, type DemoTier } from './demo.ts';

export const OBSERVED_SCHEMA = 'nvi.observed.v0';
export const EXAMPLES_SCHEMAS = ['nvi.examples.v1', 'nvi.examples.v0'];

type Obj = Record<string, unknown>;
const isObj = (value: unknown): value is Obj => !!value && typeof value === 'object' && !Array.isArray(value);
const str = (value: unknown): string => typeof value === 'string' ? value.trim() : '';
const num = (value: unknown): number | null => typeof value === 'number' && Number.isFinite(value) ? value : null;
const count = (value: unknown): number | null => { const n = num(value); return n === null || n < 0 ? null : n; };
const schemaOf = (value: Obj): unknown => value.schema_version ?? value.schema;

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
  /** Performance tier within this deal × platform (observed views). */
  tier: DemoTier | null;
  tier_label: string;
  tier_plain: string;
  views_pct_in_deal_platform: number | null;
  n_ref_posts: number | null;
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
  const tier = parseTier(value.tier);
  const pct = num(value.views_pct_in_deal_platform);
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
    tier,
    tier_label: str(value.tier_label) || (tier ? TIER_LABELS[tier] : ''),
    tier_plain: str(value.tier_plain),
    views_pct_in_deal_platform: pct !== null && pct >= 0 && pct <= 100 ? pct : null,
    n_ref_posts: count(value.n_ref_posts),
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

export interface Examples { title: string; rule: string; caveat: string; caption_null: string; defaultPair: { a: string; b: string } | null }

/**
 * `examples.json`: v1 `{default_pair: {a, b}, caveat, caption_null}`; v0 (`pairs`) is
 * read as a = beat expectations, b = fell short. The caveat is mandatory: the
 * comparison is never shown without it.
 */
export function parseExamples(value: unknown): Examples | null {
  if (!isObj(value) || !EXAMPLES_SCHEMAS.includes(schemaOf(value) as string)) return null;
  const caveat = str(value.caveat);
  if (!caveat) return null;
  let a = '', b = '';
  if (isObj(value.default_pair)) { a = str(value.default_pair.a); b = str(value.default_pair.b); }
  else if (Array.isArray(value.pairs)) {
    const pair = value.pairs.find(isObj);
    if (pair) { a = str(pair.beat_expectations); b = str(pair.fell_short); }
  }
  const defaultPair = VIDEO_ID.test(a) && VIDEO_ID.test(b) && a !== b ? { a, b } : null;
  return { title: str(value.title), rule: str(value.rule), caveat, caption_null: str(value.caption_null), defaultPair };
}

export type CompareSlot = 'a' | 'b';
export interface CompareColumn { slot: CompareSlot; tier: DemoTier | null; title: string; clip: DemoClip; entry: RealBundleIndexEntry }
export interface CompareModel { title: string; caveat: string; rule: string; caption_null: string; columns: [CompareColumn, CompareColumn] }

/**
 * The two clips to compare: the requested ones when they are in the index, else the
 * default pair, else the first great and first bad clip. Never the same clip twice.
 */
export function compareModel(examples: Examples | null, index: RealBundleIndex | null, requested: { a?: string | null; b?: string | null } = {}): CompareModel | null {
  if (!examples || !index) return null;
  const clips = demoClips(index);
  if (clips.length < 2) return null;
  const find = (id?: string | null) => id ? clips.find(clip => clip.video_id === id) ?? null : null;
  const a = find(requested.a) ?? find(examples.defaultPair?.a) ?? clips.find(clip => clip.tier === 'great') ?? chooseDemoClip(clips, null)!;
  const other = clips.filter(clip => clip !== a);
  const pick = (clip: DemoClip | null) => clip && clip !== a ? clip : null;
  const b = pick(find(requested.b)) ?? pick(find(examples.defaultPair?.b)) ?? other.find(clip => clip.tier === 'bad') ?? other[0];
  const column = (slot: CompareSlot, clip: DemoClip): CompareColumn => ({ slot, tier: clip.tier, title: clipHeading(clip), clip, entry: clip.entry });
  return {
    title: examples.title || 'Compare two clips',
    caveat: examples.caveat, rule: examples.rule, caption_null: examples.caption_null,
    columns: [column('a', a), column('b', b)],
  };
}
