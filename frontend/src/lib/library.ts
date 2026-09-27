/**
 * Optional "compared with your library" data (`library.json`, nvi.library.v0) and
 * the global research summary (`learned.json`, nvi.learned.v0).
 *
 * Both files are optional and produced outside the frontend. Every parser here is
 * defensive: anything malformed is dropped, and a file without usable content
 * parses to `null` so the page renders exactly as it does without it.
 */

export const LIBRARY_SCHEMA = 'nvi.library.v0';
export const LEARNED_SCHEMA = 'nvi.learned.v0';

export type Verdict = 'strong' | 'typical' | 'weak';
export type ScoreKey = 'hook' | 'hold' | 'peak' | 'dead_zones' | 'finish';
export const SCORE_ORDER: readonly ScoreKey[] = ['hook', 'hold', 'peak', 'dead_zones', 'finish'];
const SCORE_FALLBACK_LABEL: Record<ScoreKey, string> = { hook: 'Opening', hold: 'Middle', peak: 'Best 3 s', dead_zones: 'Low seconds', finish: 'Ending' };

export interface LibraryScore {
  key: ScoreKey;
  label: string;
  value: number | null;
  percentile: number | null;
  verdict: Verdict | null;
  plain: string;
  window_ms: [number, number] | null;
}
export interface LibraryChannel { key: string; label_plain: string; percentile: number | null; verdict: Verdict | null }
export interface LibraryMoment { start_ms: number; end_ms: number; kind: 'standout_high' | 'standout_low'; channels: string[]; plain: string; percentile: number | null }
export interface LibraryIndex {
  definition: string;
  hz: number;
  values: (number | null)[];
  percentile: (number | null)[];
  band: { p25: (number | null)[]; p50: (number | null)[]; p75: (number | null)[] } | null;
}
export interface LibraryReference { n_clips: number | null; description: string; state: string; self_excluded: boolean | null }
export interface Library {
  video_id: string;
  duration_ms: number | null;
  reference: LibraryReference | null;
  caveat: string;
  index: LibraryIndex | null;
  scores: LibraryScore[];
  channels: LibraryChannel[];
  moments: LibraryMoment[];
  summary: string[];
}

type Obj = Record<string, unknown>;
const isObj = (value: unknown): value is Obj => !!value && typeof value === 'object' && !Array.isArray(value);
const str = (value: unknown): string => typeof value === 'string' ? value.trim() : '';
const num = (value: unknown): number | null => typeof value === 'number' && Number.isFinite(value) ? value : null;
const pct = (value: unknown): number | null => { const n = num(value); return n === null ? null : Math.max(0, Math.min(100, n)); };
const numArray = (value: unknown): (number | null)[] => Array.isArray(value) ? value.map(num) : [];
const verdict = (value: unknown): Verdict | null => value === 'strong' || value === 'typical' || value === 'weak' ? value : null;
const hasNumber = (values: (number | null)[]) => values.some(value => value !== null);

function parseIndex(value: unknown): LibraryIndex | null {
  if (!isObj(value)) return null;
  const hz = num(value.hz);
  const percentile = numArray(value.percentile).map(v => v === null ? null : Math.max(0, Math.min(100, v)));
  if (!hz || hz <= 0 || !hasNumber(percentile)) return null;
  const band = isObj(value.band) ? { p25: numArray(value.band.p25), p50: numArray(value.band.p50), p75: numArray(value.band.p75) } : null;
  return { definition: str(value.definition), hz, values: numArray(value.values), percentile, band };
}

function parseScore(value: unknown): LibraryScore | null {
  if (!isObj(value) || !SCORE_ORDER.includes(value.key as ScoreKey)) return null;
  const key = value.key as ScoreKey;
  const plain = str(value.plain);
  const v = verdict(value.verdict);
  if (!plain && !v) return null;
  const w = Array.isArray(value.window_ms) && value.window_ms.length === 2 ? value.window_ms.map(num) : null;
  const window_ms = w && w[0] !== null && w[1] !== null && w[1] >= w[0] ? [w[0], w[1]] as [number, number] : null;
  return { key, label: str(value.label) || SCORE_FALLBACK_LABEL[key], value: num(value.value), percentile: pct(value.percentile), verdict: v, plain, window_ms };
}

function parseChannel(value: unknown): LibraryChannel | null {
  if (!isObj(value)) return null;
  const key = str(value.key), label_plain = str(value.label_plain);
  if (!key || !label_plain) return null;
  return { key, label_plain, percentile: pct(value.percentile), verdict: verdict(value.verdict) };
}

function parseMoment(value: unknown): LibraryMoment | null {
  if (!isObj(value)) return null;
  const start_ms = num(value.start_ms), end_ms = num(value.end_ms), plain = str(value.plain);
  if (start_ms === null || end_ms === null || end_ms < start_ms || start_ms < 0 || !plain) return null;
  if (value.kind !== 'standout_high' && value.kind !== 'standout_low') return null;
  const channels = Array.isArray(value.channels) ? value.channels.filter((key): key is string => typeof key === 'string' && !!key) : [];
  return { start_ms, end_ms, kind: value.kind, channels, plain, percentile: pct(value.percentile) };
}

/**
 * Parse `library.json`. Returns `null` when the value is not a library for this
 * clip or carries nothing to show. A static host with SPA fallback answers a
 * missing file with HTML, so callers pass whatever they fetched straight in.
 */
export function parseLibrary(value: unknown, expectedVideoId?: string): Library | null {
  if (!isObj(value) || value.schema_version !== LIBRARY_SCHEMA) return null;
  const video_id = str(value.video_id);
  if (!video_id || (expectedVideoId !== undefined && video_id !== expectedVideoId)) return null;
  const ref = isObj(value.reference) ? value.reference : null;
  const reference: LibraryReference | null = ref ? {
    n_clips: num(ref.n_clips), description: str(ref.description), state: str(ref.state),
    self_excluded: typeof ref.self_excluded === 'boolean' ? ref.self_excluded : null,
  } : null;
  const scoresByKey = new Map<ScoreKey, LibraryScore>();
  for (const score of Array.isArray(value.scores) ? value.scores.map(parseScore) : []) if (score && !scoresByKey.has(score.key)) scoresByKey.set(score.key, score);
  const library: Library = {
    video_id,
    duration_ms: num(value.duration_ms),
    reference,
    caveat: str(value.caveat),
    index: parseIndex(value.index),
    scores: SCORE_ORDER.map(key => scoresByKey.get(key)).filter((score): score is LibraryScore => !!score),
    channels: Array.isArray(value.channels) ? value.channels.map(parseChannel).filter((c): c is LibraryChannel => !!c) : [],
    moments: (Array.isArray(value.moments) ? value.moments.map(parseMoment).filter((m): m is LibraryMoment => !!m) : []).slice(0, 3),
    summary: (Array.isArray(value.summary) ? value.summary.map(str).filter(Boolean) : []).slice(0, 3),
  };
  return hasLibraryContent(library) ? library : null;
}

export function hasLibraryContent(library: Library | null): library is Library {
  return !!library && (!!library.index || library.scores.length > 0 || library.channels.length > 0 || library.moments.length > 0 || library.summary.length > 0);
}

/** `library.json` sits in the same directory as `analysis.json`, on the same origin. */
export function libraryUrlFor(analysisUrl: string): string | undefined {
  try {
    const base = new URL(analysisUrl, typeof window === 'undefined' ? 'http://localhost/' : window.location.href);
    const resolved = new URL('library.json', base);
    return resolved.origin === base.origin ? resolved.href : undefined;
  } catch { return undefined; }
}

/** Percentile (0–100) of the library index at a playhead time, or null. */
export function percentileAt(index: LibraryIndex, ms: number): number | null {
  if (!(ms >= 0)) return null;
  const i = Math.floor(ms / 1000 * index.hz);
  return index.percentile[Math.min(i, index.percentile.length - 1)] ?? null;
}

/** `m:ss` for plain-language copy. */
export function clockTime(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000));
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, '0')}`;
}

export function percentileReadout(value: number | null, ms: number): string {
  if (value === null) return `No library comparison at ${clockTime(ms)}`;
  return `Predicted response stronger than ${Math.round(value)}% of similar clips at ${clockTime(ms)}`;
}

/** Moment copy without a leading `m:ss–m:ss:` range (the card shows the range itself). */
export function momentText(plain: string): string {
  const text = plain.replace(/^\s*\d+:\d{2}\s*[–-]\s*\d+:\d{2}\s*:\s*/, '');
  return text ? text[0].toUpperCase() + text.slice(1) : plain;
}

export function verdictLabel(value: Verdict | null): string {
  return value === 'strong' ? 'Above library' : value === 'weak' ? 'Below library' : value === 'typical' ? 'Typical' : '';
}

export function referenceLine(reference: LibraryReference | null): string {
  if (!reference) return '';
  // The producer's description already carries its own count, so it is used verbatim.
  const n = reference.n_clips !== null && reference.n_clips > 0 ? `${Math.round(reference.n_clips).toLocaleString('en-US')} ` : '';
  const what = reference.description.replace(/[.\s]+$/, '') || `${n}clips in your library`;
  return `Compared with ${what}${reference.self_excluded ? ' (this clip left out)' : ''}.`;
}

/** Chart geometry for a 0–100 series, broken at nulls. Returns SVG path segments. */
export function seriesPaths(values: (number | null)[], width: number, height: number, lo = 0, hi = 100, xAt?: (i: number) => number): { line: string; area: string }[] {
  const n = values.length;
  if (!n) return [];
  const x = xAt ?? ((i: number) => n === 1 ? width / 2 : i / (n - 1) * width);
  const y = (v: number) => height - (Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo) * height;
  const mid = y((lo + hi) / 2);
  const segments: { line: string; area: string }[] = [];
  let run: [number, number][] = [];
  const flush = () => {
    if (run.length) {
      const pts = run.length === 1 ? [[run[0][0] - 2, run[0][1]], [run[0][0] + 2, run[0][1]]] : run;
      const line = pts.map(([px, py], i) => `${i ? 'L' : 'M'}${px.toFixed(1)} ${py.toFixed(1)}`).join('');
      segments.push({ line, area: `${line}L${pts[pts.length - 1][0].toFixed(1)} ${mid.toFixed(1)}L${pts[0][0].toFixed(1)} ${mid.toFixed(1)}Z` });
    }
    run = [];
  };
  values.forEach((v, i) => { if (v === null) flush(); else run.push([x(i), y(v)]); });
  flush();
  return segments;
}

// ---------------------------------------------------------------------------
// learned.json

export interface LearnedStatement { text: string; value: string }
export interface LearnedSeries { mean: (number | null)[]; lo: (number | null)[]; hi: (number | null)[] }
export interface LearnedGoodVsBad {
  seconds: number[];
  top: LearnedSeries;
  bottom: LearnedSeries;
  n_top: number | null;
  n_bottom: number | null;
  channels: { key: string; label: string; top: (number | null)[]; bottom: (number | null)[] }[];
  result_plain: string;
}
export interface Learned {
  statements: LearnedStatement[];
  /** Headline numbers read from known stage-1 fields, when present. */
  key_numbers: LearnedStatement[];
  decision: string;
  caveat: string;
  good_vs_bad: LearnedGoodVsBad | null;
}

function formatValue(value: unknown): string {
  if (typeof value === 'string') return value.trim();
  const n = num(value);
  if (n === null) return '';
  return Math.abs(n) >= 10 ? n.toFixed(0) : n.toFixed(2);
}

function statement(value: unknown): LearnedStatement | null {
  if (typeof value === 'string') return value.trim() ? { text: value.trim(), value: '' } : null;
  if (!isObj(value)) return null;
  const text = str(value.plain) || str(value.text) || str(value.statement) || str(value.line) || str(value.label);
  if (!text) return null;
  return { text, value: formatValue(value.value ?? value.number ?? value.headline) };
}

/**
 * `stage1` has no fixed shape yet. Accept, in order: an array of lines, an array
 * of `{plain|text, value}`, or an object carrying one of those under
 * `statements` / `lines` / `plain`, or an object whose values are plain strings.
 */
function parseStatements(stage1: unknown): LearnedStatement[] {
  let list: unknown[] | null = null;
  if (Array.isArray(stage1)) list = stage1;
  else if (isObj(stage1)) {
    for (const key of ['statements', 'lines', 'plain', 'plain_lines', 'items']) if (Array.isArray(stage1[key])) { list = stage1[key] as unknown[]; break; }
    if (!list) {
      const nested = Object.entries(stage1).filter(([key, v]) => !['decision', 'verdict', 'result'].includes(key) && !/rule|source|target|version|sha|state/i.test(key) && ((typeof v === 'string' && /\s/.test(v.trim())) || (isObj(v) && (typeof v.plain === 'string' || typeof v.text === 'string'))));
      list = nested.map(([, v]) => v);
    }
  }
  return (list ?? []).map(statement).filter((s): s is LearnedStatement => !!s).slice(0, 6);
}

const signed = (n: number) => Math.abs(n) < 0.005 ? '0.00' : `${n > 0 ? '+' : '−'}${Math.abs(n).toFixed(2)}`;
const point = (value: unknown): number | null => isObj(value) ? num(value.point) : num(value);

/** Known stage-1 fields → short headline numbers. Unknown or missing fields are skipped. */
function parseKeyNumbers(stage1: unknown): LearnedStatement[] {
  if (!isObj(stage1)) return [];
  const out: LearnedStatement[] = [];
  const rhoA = num(stage1.rho_A_metadata);
  if (rhoA !== null) out.push({ value: `≈ ${rhoA.toFixed(2)}`, text: 'How well metadata alone ranks clips' });
  const brain = point(stage1.BE_minus_E_content);
  if (brain !== null) out.push({ value: `≈ ${signed(brain)}`, text: 'What predicted brain response adds' });
  const reach = point(stage1.reach_E_minus_A_content);
  if (reach !== null) out.push({ value: signed(reach), text: 'What video features add for reach' });
  return out;
}

function parseDecision(stage1: unknown): string {
  if (!isObj(stage1)) return '';
  return str(stage1.decision) || str(stage1.verdict) || str(stage1.result);
}

/** A curve is either a plain array or `{mean, lo?, hi?}`. */
const curveMean = (value: unknown): (number | null)[] => Array.isArray(value) ? numArray(value) : isObj(value) ? numArray(value.mean) : [];

function parseSeries(value: unknown, n: number): LearnedSeries | null {
  if (!isObj(value)) return null;
  const mean = numArray(value.mean).slice(0, n);
  if (!hasNumber(mean)) return null;
  const lo = numArray(value.lo).slice(0, n), hi = numArray(value.hi).slice(0, n);
  return { mean, lo: lo.length === mean.length ? lo : [], hi: hi.length === mean.length ? hi : [] };
}

function parseGoodVsBad(value: unknown): LearnedGoodVsBad | null {
  if (!isObj(value)) return null;
  const seconds = Array.isArray(value.seconds) ? value.seconds.map(num).filter((s): s is number => s !== null) : [];
  if (seconds.length < 2) return null;
  // The index curves sit either at the top level or under `index`.
  const holder = isObj(value.index) && value.top === undefined ? value.index : value;
  const top = parseSeries(holder.top, seconds.length), bottom = parseSeries(holder.bottom, seconds.length);
  if (!top || !bottom) return null;
  const channels = isObj(value.channels) ? Object.entries(value.channels).flatMap(([key, entry]) => {
    if (!isObj(entry)) return [];
    const t = curveMean(entry.top).slice(0, seconds.length), b = curveMean(entry.bottom).slice(0, seconds.length);
    if (!hasNumber(t) || !hasNumber(b)) return [];
    return [{ key, label: str(entry.label_plain) || str(entry.label) || key.replaceAll('_', ' '), top: t, bottom: b }];
  }) : [];
  return { seconds, top, bottom, n_top: num(value.n_top), n_bottom: num(value.n_bottom), channels, result_plain: str(value.result_plain) };
}

export function parseLearned(value: unknown): Learned | null {
  if (!isObj(value)) return null;
  if (value.schema_version !== undefined && value.schema_version !== LEARNED_SCHEMA) return null;
  const learned: Learned = {
    statements: parseStatements(value.stage1), key_numbers: parseKeyNumbers(value.stage1), decision: parseDecision(value.stage1),
    caveat: str(value.caveat), good_vs_bad: parseGoodVsBad(value.good_vs_bad),
  };
  return learned.statements.length || learned.good_vs_bad ? learned : null;
}

/** Shared y-range for curves with optional bands. */
export function seriesRange(...series: (number | null)[][]): [number, number] {
  const all = series.flat().filter((v): v is number => v !== null);
  if (!all.length) return [0, 1];
  let lo = Math.min(...all), hi = Math.max(...all);
  if (hi - lo < 1e-9) { lo -= 1; hi += 1; }
  const pad = (hi - lo) * 0.08;
  return [lo - pad, hi + pad];
}

/** Path helper for arbitrary ranges: points at x = i/(n-1). */
export function linePath(values: (number | null)[], width: number, height: number, range: [number, number]): string {
  return seriesPaths(values, width, height, range[0], range[1]).map(s => s.line).join('');
}

/** Closed band polygon between two series (skips any index where either is null). */
export function bandPath(lo: (number | null)[], hi: (number | null)[], width: number, height: number, range: [number, number]): string {
  const n = Math.min(lo.length, hi.length);
  if (n < 2) return '';
  const x = (i: number) => i / (n - 1) * width;
  const y = (v: number) => height - (Math.max(range[0], Math.min(range[1], v)) - range[0]) / (range[1] - range[0]) * height;
  const parts: string[] = [];
  let run: number[] = [];
  const flush = () => {
    if (run.length >= 2) {
      const top = run.map((i, k) => `${k ? 'L' : 'M'}${x(i).toFixed(1)} ${y(hi[i]!).toFixed(1)}`).join('');
      const bottom = [...run].reverse().map(i => `L${x(i).toFixed(1)} ${y(lo[i]!).toFixed(1)}`).join('');
      parts.push(`${top}${bottom}Z`);
    }
    run = [];
  };
  for (let i = 0; i < n; i++) { if (lo[i] === null || hi[i] === null) flush(); else run.push(i); }
  flush();
  return parts.join('');
}
