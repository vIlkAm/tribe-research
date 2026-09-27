import { build } from 'esbuild';
import { readFile, readdir, writeFile, mkdir } from 'node:fs/promises';
import { gzipSync, gunzipSync } from 'node:zlib';
import { extname, join, relative } from 'node:path';
import { createHash } from 'node:crypto';

// A complete ClickFunnels Custom HTML Page, below its 2 MB document limit.
// Core assets are embedded. The optional reference comparison loads YouTube.
const result = await build({
  entryPoints: ['src/main.tsx'], bundle: true, minify: true, write: false,
  outdir: 'dist', format: 'esm', target: 'es2022', legalComments: 'inline',
  define: { 'import.meta.env.BASE_URL': '"/"', 'process.env.NODE_ENV': '"production"' },
});
const mime = { '.json': 'application/json', '.bin': 'application/octet-stream', '.jpg': 'image/jpeg', '.png': 'image/png', '.svg': 'image/svg+xml' };
const assets = {};
async function collect(directory) {
  for (const item of await readdir(directory, { withFileTypes: true })) {
    const path = join(directory, item.name);
    if (item.isDirectory()) await collect(path);
    else {
      const type = mime[extname(path)];
      if (!type) continue;
      assets[relative('public', path)] = { type, data: (await readFile(path)).toString('base64') };
    }
  }
}
await collect('public');
const js = result.outputFiles.find(f => f.path.endsWith('.js')).text;
// The portable build uses local font fallbacks to eliminate external requests.
const css = result.outputFiles.find(f => f.path.endsWith('.css')).text.replace(/@import\s*(?:"[^"]*"|'[^']*'|url\([^)]*\))\s*;/g, '');
const notices = await readFile('THIRD_PARTY_NOTICES.md', 'utf8');
const payload = JSON.stringify({ js, css, assets, notices });
const zipped = gzipSync(payload, { level: 9 });
if (gunzipSync(zipped).toString() !== payload) throw new Error('Package round-trip failed.');
const html = `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="ViralBrain — an interactive neural video research demo with synthetic sample data."><title>ViralBrain · Neural video intelligence</title><style>body{margin:0;background:#090c11;color:#eceef2;font:16px system-ui}.boot{padding:8vh 6vw;max-width:700px;line-height:1.6}</style></head><body><div id="root"><div class="boot"><h1>ViralBrain.</h1><p>Preparing your research workspace…</p><noscript>Enable JavaScript to explore the interactive brain.</noscript></div></div><script type="module">
try {
  if (!('DecompressionStream' in window)) throw new Error('Please use a current version of Safari, Chrome, Edge, or Firefox.');
  const bytes = Uint8Array.from(atob('${zipped.toString('base64')}'), c => c.charCodeAt(0));
  const packed = JSON.parse(await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).text());
  window.__VIRALBRAIN_ASSETS = Object.fromEntries(Object.entries(packed.assets).map(([path, asset]) => [path, URL.createObjectURL(new Blob([Uint8Array.from(atob(asset.data), c => c.charCodeAt(0))], {type: asset.type}))]));
  const style = document.createElement('style'); style.textContent = packed.css; document.head.append(style);
  const icon = document.createElement('link'); icon.rel = 'icon'; icon.href = window.__VIRALBRAIN_ASSETS['favicon.svg']; document.head.append(icon);
  const credits = document.createElement('script'); credits.type = 'text/plain'; credits.id = 'third-party-notices'; credits.textContent = packed.notices; document.body.append(credits);
  await import(URL.createObjectURL(new Blob([packed.js], {type:'text/javascript'})));
} catch (error) {
  const target = document.querySelector('#root'); target.textContent = ''; const box = document.createElement('div'); box.className = 'boot'; const title = document.createElement('h1'); title.textContent = 'ViralBrain could not start.'; const message = document.createElement('p'); message.textContent = error.message; box.append(title, message); target.append(box);
}
</script></body></html>`;
const bytes = Buffer.byteLength(html);
if (bytes >= 2_000_000) throw new Error(`Custom HTML is ${bytes} bytes; exceeds conservative 2 MB limit.`);
await mkdir('deploy', { recursive: true });
await mkdir('dist', { recursive: true });
await writeFile('deploy/viralbrain-clickfunnels.html', html);
await writeFile('dist/clickfunnels.html', html);
await writeFile('deploy/package-manifest.json', JSON.stringify({ format: 'ClickFunnels Custom HTML Page', bytes, max_bytes: 2_000_000, sha256: createHash('sha256').update(html).digest('hex'), fixture_commit: '65ab40704ca484b453e2e2675d777423f325f97a', assets: Object.keys(assets), network_dependencies: [], optional_network_features: { reference_comparison: ['YouTube IFrame Player API', 'YouTube embedded video playback'] }, generated_at: new Date().toISOString() }, null, 2) + '\n');
console.log(`ClickFunnels page: ${(bytes / 1_000_000).toFixed(3)} MB (${bytes} bytes), ${Object.keys(assets).length} embedded assets.`);
