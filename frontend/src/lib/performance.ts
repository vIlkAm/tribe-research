import type { Performance } from '../data/analysis.types.ts';

export function percentile(value: number): string {
  return `P${Math.round(Math.max(0, Math.min(1, value)) * 100)}`;
}

export function platformLabel(value: string): string {
  if (!value) return 'Unknown platform';
  return value === 'youtube' ? 'YouTube' : value === 'tiktok' ? 'TikTok' : value === 'instagram' ? 'Instagram' : value.charAt(0).toUpperCase() + value.slice(1);
}

export function performanceHasNumbers(performance?: Performance): boolean {
  return !!performance && ['preliminary', 'research_preview', 'validated'].includes(performance.model_status) && performance.engagement !== null;
}

export function performanceStatusLabel(status: Performance['model_status']): string {
  if (status === 'preliminary') return 'Preliminary';
  if (status === 'research_preview') return 'Research preview';
  if (status === 'validated') return 'Validated';
  if (status === 'out_of_scope') return 'Outside model scope';
  return 'Not trained';
}
