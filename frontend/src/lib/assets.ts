import { BUNDLE_PATH } from './analysis';

declare global {
  interface Window { __VIRALBRAIN_ASSETS?: Record<string, string>; }
}

/** The single-file ClickFunnels build provides the same files as local blob URLs. */
export function assetUrl(path: string): string {
  const key = new URL(path, 'https://viralbrain.invalid/').pathname.slice(1);
  return window.__VIRALBRAIN_ASSETS?.[key] ?? `${import.meta.env.BASE_URL}${key}`;
}

export function analysisAssetUrl(relative: string): string {
  const path = new URL(relative, `https://viralbrain.invalid/${BUNDLE_PATH}`);
  if (path.origin !== 'https://viralbrain.invalid') throw new Error('Analysis assets must be bundled local files.');
  return assetUrl(path.pathname);
}
