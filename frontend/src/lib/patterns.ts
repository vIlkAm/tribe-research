/**
 * Internal, exploratory library-wide summary (`library_patterns.json`, nvi.patterns.v0):
 * what great, typical and bad clips (by observed views within a deal × platform) have
 * in common. Optional and defensive: anything malformed is dropped, and a file
 * without content parses to null so no button or page appears.
 */
import { parseTier, type DemoTier } from './demo.ts';

export const PATTERNS_SCHEMA = 'nvi.patterns.v0';

export type FeatureGroup = 'editing' | 'brain';
/** Exploratory verdicts only: a weak tendency at most, never "reliable" or "significant". */
export type FeatureVerdict = 'weak_tendency' | 'no_reliable_difference';
export interface Interval { point: number; lo: number; hi: number }
export interface PatternFeature {
  key: string; label_plain: string; unit: string; group: FeatureGroup | null;
  mean: Record<DemoTier, number | null>;
  diff: Interval | null;
  /** P(a random great clip beats a random bad one on this feature); 0.5 = coin flip. */
  coin_flip: Interval | null;
  q: number | null; verdict: FeatureVerdict | null; plain: string;
}
export interface PatternTier { label: string; n_contents: number | null; rule_plain: string }
export interface Patterns {
  definition: string; exploratory: boolean; outcome_plain: string;
  tiers: Partial<Record<DemoTier, PatternTier>>;
  n_library: number | null; n_tiered: number | null; n_deals: number | null;
  features: PatternFeature[]; interpreter_line: string; caveats: string[];
}

type Obj = Record<string, unknown>;
const isObj = (value: unknown): value is Obj => !!value && typeof value === 'object' && !Array.isArray(value);
const str = (value: unknown): string => typeof value === 'string' ? value.trim() : '';
const num = (value: unknown): number | null => typeof value === 'number' && Number.isFinite(value) ? value : null;
const count = (value: unknown): number | null => { const n = num(value); return n !== null && n >= 0 ? Math.round(n) : null; };

function interval(value: unknown): Interval | null {
  if (!isObj(value)) return null;
  const point = num(value.point), lo = num(value.lo), hi = num(value.hi);
  return point !== null && lo !== null && hi !== null && lo <= hi ? { point, lo, hi } : null;
}

function feature(value: unknown): PatternFeature | null {
  if (!isObj(value)) return null;
  const label = str(value.label_plain) || str(value.name);
  const key = str(value.key) || str(value.name) || label;
  if (!key || !label) return null;
  const means = isObj(value.mean) ? value.mean : {};
  const group = value.group === 'editing' || value.group === 'brain' ? value.group : null;
  // Older files said "reliable"; it is shown as a weak tendency like the current label.
  const verdict: FeatureVerdict | null = value.verdict === 'weak_tendency' || value.verdict === 'reliable' ? 'weak_tendency'
    : value.verdict === 'no_reliable_difference' || value.verdict === 'not reliable' || value.verdict === 'not_reliable' ? 'no_reliable_difference' : null;
  const coin = interval(value.coin_flip);
  return {
    key, label_plain: label, unit: str(value.unit), group,
    mean: { great: num(means.great), typical: num(means.typical), bad: num(means.bad) },
    diff: interval(value.diff_great_minus_bad ?? value.diff),
    coin_flip: coin && coin.lo >= 0 && coin.hi <= 1 ? coin : null,
    q: num(value.q), verdict, plain: str(value.plain),
  };
}

export function parsePatterns(value: unknown): Patterns | null {
  if (!isObj(value) || (value.schema ?? value.schema_version) !== PATTERNS_SCHEMA) return null;
  const tiers: Partial<Record<DemoTier, PatternTier>> = {};
  if (isObj(value.tiers)) for (const [key, tier] of Object.entries(value.tiers)) {
    const name = parseTier(key);
    if (name && isObj(tier)) tiers[name] = { label: str(tier.label), n_contents: count(tier.n_contents), rule_plain: str(tier.rule_plain) };
  }
  const features = (Array.isArray(value.features) ? value.features : []).map(feature).filter((f): f is PatternFeature => !!f);
  const patterns: Patterns = {
    definition: str(value.definition),
    // Treated as exploratory unless the file explicitly says otherwise.
    exploratory: value.exploratory !== false,
    outcome_plain: str(value.outcome_plain),
    tiers, n_library: count(value.n_library), n_tiered: count(value.n_tiered), n_deals: count(value.n_deals), features,
    interpreter_line: str(value.interpreter_line),
    caveats: (Array.isArray(value.caveats) ? value.caveats : []).map(str).filter(Boolean),
  };
  return patterns.features.length || patterns.interpreter_line ? patterns : null;
}

/**
 * Features by group. Groups holding a weak tendency come first (editing, brain,
 * other as the tiebreak); within a group, tendencies first, file order otherwise.
 */
export function featureGroups(patterns: Patterns): { group: FeatureGroup | null; title: string; features: PatternFeature[] }[] {
  const groups: { group: FeatureGroup | null; title: string }[] = [
    { group: 'editing', title: 'Editing' }, { group: 'brain', title: 'Predicted brain response' }, { group: null, title: 'Other' },
  ];
  const rank = (f: PatternFeature) => f.verdict === 'weak_tendency' ? 0 : 1;
  const filled = groups.map(g => ({ ...g, features: patterns.features.filter(f => f.group === g.group).sort((x, y) => rank(x) - rank(y)) })).filter(g => g.features.length);
  const groupRank = (g: { features: PatternFeature[] }) => g.features.some(f => rank(f) === 0) ? 0 : 1;
  return filled.sort((x, y) => groupRank(x) - groupRank(y));
}

/**
 * Positions (0–100 %) for a small great − bad interval bar on a scale symmetric
 * around zero, so 0 is always the centre line.
 */
export function ciBar(diff: Interval): { lo: number; hi: number; point: number; zero: number } {
  const span = Math.max(Math.abs(diff.lo), Math.abs(diff.hi), Math.abs(diff.point)) * 1.15 || 1;
  const x = (v: number) => Math.max(0, Math.min(100, 50 + v / span * 50));
  return { lo: x(diff.lo), hi: x(diff.hi), point: x(diff.point), zero: 50 };
}

export function formatFeatureValue(value: number | null, unit: string): string {
  if (value === null) return '—';
  const abs = Math.abs(value);
  const text = abs >= 100 ? value.toFixed(0) : abs >= 10 ? value.toFixed(1) : value.toFixed(2);
  if (!unit) return text;
  return unit === '%' ? `${text}%` : `${text} ${unit}`;
}

/** "60% (51–68%)": how often a random great clip beats a random bad one. */
export function coinFlipText(coin: Interval): string {
  const pct = (v: number) => Math.round(v * 100);
  return `${pct(coin.point)}% (${pct(coin.lo)}–${pct(coin.hi)}%)`;
}

export function verdictText(verdict: FeatureVerdict | null): string {
  return verdict === 'weak_tendency' ? 'Weak tendency (exploratory)' : verdict === 'no_reliable_difference' ? 'No clear difference' : '';
}

