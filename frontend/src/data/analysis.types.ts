/**
 * TypeScript types for the analysis object `nvi.analysis.v0.2`.
 *
 * Mirrors docs/analysis.schema.json (the authority). Copy this file into a
 * frontend as-is; tests/test_types.py type-checks docs/sample_analysis against it.
 * All neural values are z-scores within ONE clip: never compare them across clips.
 */

export type Ms = number; // integer milliseconds from the first video frame (stimulus time)
export type Span = [Ms, Ms]; // [start_ms, end_ms)

export type SignalKind = "neural_proxy" | "behavior" | "observed";

/** Never infer from the key. `no_monotonic`: neither direction is better; never colour one as good. */
export type Direction =
  | "learned_positive"
  | "learned_negative"
  | "hypothesis_positive"
  | "context_dependent"
  | "no_monotonic";

export type Confidence = "uncalibrated" | "low" | "medium" | "high";

export interface Interval {
  value: number;
  lo?: number;
  hi?: number;
}

export interface ChannelSummary {
  mean_0_3s: number | null;
  mean_0_5s: number | null;
  peak_z: number;
  peak_time_ms: Ms;
  trough_z: number;
  trough_time_ms: Ms;
  /** Sustained-attention style summary, 0..1; null for very short clips. */
  stability: number | null;
}

export interface Channel {
  /** v0.2 keys: attention | social | value | control | self | language | sensory */
  key: string;
  label: string;
  kind: SignalKind;
  unit: "z_within_clip";
  direction: Direction;
  direction_note: string;
  confidence: Confidence;
  source_model: string;
  research_status: "research_proxy" | "validated";
  /** Display grid rate (2). The real model resolution is native_tr_s; draw steps, don't smooth. */
  hz: number;
  native_tr_s: number;
  resampling: "sample_and_hold";
  default_visible: boolean;
  color: string; // "#RRGGBB"
  basis: { roi_groups: string[]; regions_text: string; atlas: string };
  copy: { tooltip: string; rise: string; fall: string };
  /** values[i] is at t = i / hz seconds; null = no prediction (gap) — never interpolate. */
  values: (number | null)[];
  summary: ChannelSummary;
}

export type MomentKind = "attention_drop" | "broad_response" | "proxy_rise" | "proxy_fall";

export interface Moment {
  id: string;
  kind: MomentKind;
  start_ms: Ms;
  end_ms: Ms;
  channels: string[];
  peak_z: number;
  /** 0..1, ranks moments within this clip only. */
  severity: number;
  title: string;
  description: string;
  evidence: string[];
  /** Edit suggestion; null for plain observations. Always untested in v0.2. */
  hypothesis: string | null;
  test_metric: string | null;
  status: "observation" | "edit_hypothesis_untested" | "validated";
  relative_to: "this_clip";
  confidence: Confidence;
  /** Inside the first 2 s onset response: don't headline it. */
  in_onset_window: boolean;
}

export interface Sprite {
  /** research_vertex: per-vertex map, Research mode only. */
  mode: "proxy" | "research_vertex";
  /** Relative to the folder holding analysis.json. */
  src: string;
  tile_w: number;
  tile_h: number;
  cols: number;
  rows: number;
  /** Show tile i (row-major) while frame_start_ms[i] <= t < frame_end_ms[i]; otherwise it's a gap. */
  frame_start_ms: Ms[];
  frame_end_ms: Ms[];
  views: string[];
  colormap: { name: string; vmin: number; vmax: number; threshold: number; unit: string };
}

export interface RegionMap {
  /** Same tile geometry as brain_map; read the pixel under the cursor (no smoothing). */
  idmap_src: string;
  legend_src: string;
  /** "#rrggbb" (lowercase) -> channel key. "#000000" = no channel. */
  ids: Record<string, string>;
}

export type PerformanceModelStatus = "not_trained" | "preliminary" | "research_preview" | "validated" | "out_of_scope";
export type PerformanceBrainClaim = "not_tested" | "directional" | "supported" | "not_supported";

export interface PerformanceMetric {
  target: string;
  /** 0..1 rank among reference clips in the same deal and platform. */
  percentile_deal_platform: number | null;
  /** Observed P10–P90 outcome range for similarly predicted reference clips. */
  likely_range: [number, number] | null;
  reference_n: number;
  percentile_account: number | null;
  account_reference_n: number | null;
  confidence: "low" | "medium";
  validation: {
    scheme: "content_cv" | "lockbox" | "none";
    within_stratum_spearman: number | null;
    ci95: [number | null, number | null];
  };
}

export interface PerformanceDriver {
  family: "metadata" | "account_history" | "content_embedding" | "brain_response";
  label: string;
  /** Signed model attribution. Use for bar length only; never display the raw number. */
  contribution: number;
  brain_detail?: { channel: string; contribution: number }[];
}

export interface Performance {
  model_status: PerformanceModelStatus;
  validated: boolean;
  reason: string | null;
  brain_claim: PerformanceBrainClaim;
  model_version: string | null;
  n_train: number | null;
  /** Mandatory, verbatim UI copy for preliminary results; null in every other state. */
  caption: string | null;
  context: {
    deal_id: string | null;
    deal_label: string | null;
    platform: string | null;
    account_id: string | null;
    account_level: boolean;
  };
  clip_in_training: "no" | "train_oof" | "lockbox";
  retrospective: boolean;
  engagement: PerformanceMetric | null;
  reach: PerformanceMetric | null;
  drivers: PerformanceDriver[];
  warnings: string[];
  provenance: Record<string, unknown>;
}

export interface Analysis {
  schema_version: "nvi.analysis.v0.2";
  analysis_id: string;
  video_id: string;
  /** Upstream id for the outcome join. */
  source_name?: string | null;
  duration_ms: Ms;
  status: "queued" | "processing" | "complete" | "failed";
  research_only: true;
  /** true: show a SYNTHETIC banner (dry-run predictions or synthetic ROI map). */
  synthetic: boolean;
  model_versions: { neural: string; behavior: string | null; explain: string };
  provenance: {
    tribe_commit: string | null;
    roi_map: Record<string, unknown>;
    proxies_version: string;
    feature_version: string;
    normalization: string;
    [k: string]: unknown;
  };
  timing: {
    time_base: "stimulus";
    note?: string;
    native_tr_s: number;
    display_hz: 2;
    n_display_samples: number;
    onset_window_ms: Span;
    gaps_ms: Span[];
  };
  /** v0.2: always not_available with empty metrics. Render an empty state, never placeholder numbers. */
  predictions:
    | { status: "not_available"; reason?: string; metrics: Record<string, never> }
    | { status: "available"; reason?: string; metrics: Record<string, Interval> };
  /** Proposed additive v0.3 block. Absent means this bundle did not compute performance. */
  performance?: Performance;
  channels: Channel[];
  /** Show as disabled toggles with the reason. */
  unavailable_channels: { key: string; kind: SignalKind; reason: string }[];
  events: {
    /** null = not computed (see unavailable_lanes). */
    shots_ms: Ms[] | null;
    words: { start_ms: Ms; end_ms: Ms; text: string }[];
    speech_spans_ms: Span[];
    unavailable_lanes: { key: string; reason: string }[];
  };
  moments: Moment[];
  assets: {
    brain_map?: Sprite;
    research_vertex_map?: Sprite;
    region_map?: RegionMap;
    summary_png?: string;
    demo_mp4?: string;
  };
  quality: { n_segments: number; has_words: boolean; warnings: string[] };
}
