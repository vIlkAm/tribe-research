import type { Analysis } from '../data/analysis.types.ts';
import { assertAnalysis } from './analysis.ts';

export interface BundleFile {
  name: string;
  size: number;
  webkitRelativePath?: string;
  text(): Promise<string>;
}
export interface LocalBundle {
  analysis: Analysis;
  filename: string;
  origin?: 'local' | 'remote';
  /** Candidate source clip next to an approved real index; loaded only if the server has it. */
  sourceClipUrl?: string;
  warnings: string[];
  resolveAsset: (relative: string) => string | null;
  dispose: () => void;
}

export function filePath(file: BundleFile): string {
  return file.webkitRelativePath || file.name;
}

/** Resolve within the selected files, without network URLs or escaping their root. */
export function localAssetPath(analysisPath: string, relative: string): string {
  if (!relative || /^[a-z][a-z\d+.-]*:/i.test(relative) || /^[\\/]/.test(relative) || /[\\?#\u0000-\u001f]/.test(relative)) {
    throw new Error(`Asset must be a relative local file: ${relative}`);
  }
  const parts = analysisPath.split('/').slice(0, -1);
  for (const raw of relative.split('/')) {
    let part: string;
    try { part = decodeURIComponent(raw); } catch { throw new Error(`Invalid asset path: ${relative}`); }
    if (/[\\/\u0000-\u001f]/.test(part)) throw new Error(`Invalid asset path: ${relative}`);
    if (part === '..') {
      if (!parts.length) throw new Error(`Asset is outside the selected files: ${relative}`);
      parts.pop();
    } else if (part && part !== '.') parts.push(part);
  }
  return parts.join('/');
}

export function analysisFiles<T extends BundleFile>(files: readonly T[]): T[] {
  const exact = files.filter(file => file.name === 'analysis.json');
  return (exact.length ? exact : files.filter(file => /\.json$/i.test(file.name)))
    .sort((a, b) => filePath(a).localeCompare(filePath(b)));
}

/** Only the three proxy-view images are opened. Research maps and videos stay unused. */
export async function openLocalBundle<T extends BundleFile>(
  selected: T,
  files: readonly T[],
  urls: { create: (file: T) => string; revoke: (url: string) => void },
): Promise<LocalBundle> {
  if (selected.size > 10_000_000) throw new Error('This analysis JSON exceeds the 10 MB limit.');
  let value: unknown;
  try { value = JSON.parse(await selected.text()); }
  catch { throw new Error('This file is not valid JSON. Choose an analysis.json from the research export.'); }
  assertAnalysis(value);
  const analysis = value;
  const filename = filePath(selected);
  const index = new Map<string, T>();
  for (const file of files) {
    const path = filePath(file);
    if (index.has(path)) throw new Error(`Two selected files have the same path: ${path}. Choose their parent folder instead.`);
    index.set(path, file);
  }
  const references = analysis.assets.brain_map?.mode === 'proxy'
    ? [analysis.assets.brain_map.src, analysis.assets.region_map?.idmap_src, analysis.assets.region_map?.legend_src].filter((path): path is string => !!path)
    : [];
  const warnings: string[] = [];
  const resolved = new Map<string, string | null>();
  const opened: string[] = [];
  let imageBytes = 0;
  try {
    for (const reference of new Set(references)) {
      // A JSON-only selection cannot include sibling assets. Keep its signals usable.
      let path: string;
      try { path = localAssetPath(filename, reference); }
      catch (error) {
        if (!filename.includes('/') && /^\.\.\//.test(reference)) {
          warnings.push(`Image not selected: ${reference}. Choose the export’s parent folder to include shared assets.`);
          resolved.set(reference, null);
          continue;
        }
        throw error;
      }
      const file = index.get(path);
      if (!file) {
        warnings.push(`Image not selected: ${reference}. The signal timeline is still available.`);
        resolved.set(reference, null);
        continue;
      }
      if (!/\.(png|jpe?g|webp)$/i.test(file.name)) throw new Error(`Unsupported image: ${reference}. Use PNG, JPEG or WebP.`);
      imageBytes += file.size;
      if (imageBytes > 100_000_000) throw new Error('The selected analysis images exceed the 100 MB limit.');
      const url = urls.create(file);
      opened.push(url);
      resolved.set(reference, url);
    }
  } catch (error) {
    opened.forEach(urls.revoke);
    throw error;
  }
  return {
    analysis, filename, origin: 'local', warnings,
    resolveAsset: relative => resolved.get(relative) ?? null,
    dispose: () => { opened.splice(0).forEach(urls.revoke); },
  };
}
