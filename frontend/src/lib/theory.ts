/**
 * Internal, exploratory library statistics (`library_theory.json`, nvi.theory.v0; docs/LIBRARY_THEORY.md):
 * which clip features go with account size, with how a video did against its account's usual, and with
 * engagement. Found on half the accounts, checked on the other half. Defensive: malformed parts are dropped.
 */
export const THEORY_SCHEMA = 'nvi.theory.v0';
export const THEORY_OUTCOMES = ['total', 'account', 'video', 'engagement'] as const;
export type TheoryOutcome = typeof THEORY_OUTCOMES[number];
export type TheoryVerdict = 'holds' | 'did_not_hold' | 'no_pattern';
export interface TheoryCorr { r: number; lo: number; hi: number }
export interface TheoryPair {
  feature: string; feature_plain: string; feature_group: string; outcome: TheoryOutcome;
  discovery: TheoryCorr | null; confirmation: TheoryCorr | null; full: TheoryCorr | null;
  verdict: TheoryVerdict; plain: string;
}
export interface Theory {
  n_clips: number | null; n_accounts: number | null;
  outcomes: Partial<Record<TheoryOutcome, string>>;
  decomposition: { share_account: number; share_video: number; share_covariance: number } | null;
  pairs: TheoryPair[]; interpreter_line: string;
}

type Obj = Record<string, unknown>;
const isObj = (v: unknown): v is Obj => !!v && typeof v === 'object' && !Array.isArray(v);
const str = (v: unknown): string => typeof v === 'string' ? v.trim() : '';
const num = (v: unknown): number | null => typeof v === 'number' && Number.isFinite(v) ? v : null;
const isOutcome = (v: unknown): v is TheoryOutcome => (THEORY_OUTCOMES as readonly unknown[]).includes(v);

function corr(v: unknown): TheoryCorr | null {
  if (!isObj(v)) return null;
  const r = num(v.r), lo = num(v.lo), hi = num(v.hi);
  return r !== null && lo !== null && hi !== null ? { r, lo, hi } : null;
}

function pair(v: unknown): TheoryPair | null {
  if (!isObj(v) || !isOutcome(v.outcome) || !str(v.feature)) return null;
  const verdict: TheoryVerdict = v.verdict === 'holds' ? 'holds' : v.verdict === 'did_not_hold' ? 'did_not_hold' : 'no_pattern';
  return {
    feature: str(v.feature), feature_plain: str(v.feature_plain) || str(v.feature), feature_group: str(v.feature_group),
    outcome: v.outcome, discovery: corr(v.discovery), confirmation: corr(v.confirmation), full: corr(v.full),
    verdict, plain: str(v.plain),
  };
}

export function parseTheory(value: unknown): Theory | null {
  if (!isObj(value) || value.schema !== THEORY_SCHEMA) return null;
  const outcomes: Partial<Record<TheoryOutcome, string>> = {};
  if (isObj(value.outcomes)) for (const [k, o] of Object.entries(value.outcomes)) if (isOutcome(k) && isObj(o)) outcomes[k] = str(o.plain);
  const d = isObj(value.decomposition) ? value.decomposition : null;
  const sa = num(d?.share_account), sv = num(d?.share_video), sc = num(d?.share_covariance);
  const pairs = (Array.isArray(value.pairs) ? value.pairs : []).map(pair).filter((p): p is TheoryPair => !!p);
  const theory: Theory = {
    n_clips: num(value.n_clips), n_accounts: num(value.n_accounts), outcomes,
    decomposition: sa !== null && sv !== null && sc !== null ? { share_account: sa, share_video: sv, share_covariance: sc } : null,
    pairs, interpreter_line: str(value.interpreter_line),
  };
  return pairs.length || theory.interpreter_line ? theory : null;
}

export const THEORY_OUTCOME_TITLES: Record<TheoryOutcome, string> = {
  total: 'Views', account: 'Account size', video: 'Video vs its account', engagement: 'Engagement per view',
};
export const THEORY_VERDICT_TEXT: Record<TheoryVerdict, string> = {
  holds: 'Confirmed on new accounts', did_not_hold: 'Did not hold', no_pattern: 'No clear link',
};

/**
 * What a cell shows. The verdict itself is fixed by the split-half rule; a pair that was never a discovery
 * candidate is shown as "Not yet confirmed" when its all-clips 95% range excludes zero, else "No clear link".
 */
export type TheoryShown = 'holds' | 'did_not_hold' | 'unconfirmed' | 'no_link';
export const THEORY_SHOWN_TEXT: Record<TheoryShown, string> = {
  holds: THEORY_VERDICT_TEXT.holds, did_not_hold: THEORY_VERDICT_TEXT.did_not_hold, unconfirmed: 'Not yet confirmed', no_link: THEORY_VERDICT_TEXT.no_pattern,
};
export const excludesZero = (c: TheoryCorr | null): boolean => !!c && (c.lo > 0 || c.hi < 0);
export function shownVerdict(p: TheoryPair): TheoryShown {
  if (p.verdict !== 'no_pattern') return p.verdict;
  return excludesZero(p.full) ? 'unconfirmed' : 'no_link';
}

/** Features with one pair per outcome: brain rows first, otherwise in file order. */
export function theoryRows(theory: Theory): { feature: string; label: string; group: string; cells: Partial<Record<TheoryOutcome, TheoryPair>> }[] {
  const rows = new Map<string, { feature: string; label: string; group: string; cells: Partial<Record<TheoryOutcome, TheoryPair>> }>();
  for (const p of theory.pairs) {
    const row = rows.get(p.feature) ?? { feature: p.feature, label: p.feature_plain, group: p.feature_group, cells: {} };
    row.cells[p.outcome] = p;
    rows.set(p.feature, row);
  }
  // Brain rows first (stable within each group).
  return [...rows.values()].sort((a, b) => Number(a.group !== 'brain') - Number(b.group !== 'brain'));
}

export const formatR = (r: number): string => `${r >= 0 ? '+' : '−'}${Math.abs(r).toFixed(2)}`;
/** A range end: three decimals when two would round to 0.00, so the side of zero stays visible. */
export const formatEnd = (x: number): string => x !== 0 && Math.abs(x) < 0.005 ? `${x > 0 ? '+' : '−'}${Math.abs(x).toFixed(3)}` : formatR(x);
