import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { analysisFiles, localAssetPath, openLocalBundle, type BundleFile } from '../src/lib/local-bundle.ts';

const sample = JSON.parse(readFileSync(new URL('../public/data/cd20b16879d630c4/analysis.json', import.meta.url), 'utf8'));
const file = (path: string, data = '', size = data.length): BundleFile => ({
  name: path.split('/').at(-1)!, webkitRelativePath: path.includes('/') ? path : '', size, text: async () => data,
});
const json = (value: unknown = sample, path = 'export/clip/analysis.json') => file(path, JSON.stringify(value));
const images = () => [file('export/clip/brain_proxy.jpg'), file('export/_static/region_idmap.png'), file('export/_static/region_legend.png')];
function objectUrls() {
  const created: string[] = [], revoked: string[] = [];
  return { created, revoked, create: (file: BundleFile) => { const url = `blob:test/${file.webkitRelativePath}`; created.push(url); return url; }, revoke: (url: string) => { revoked.push(url); } };
}

test('folder import resolves sibling shared assets and releases each URL once', async () => {
  const analysis = json(), urls = objectUrls();
  const bundle = await openLocalBundle(analysis, [analysis, ...images()], urls);
  assert.equal(bundle.analysis.analysis_id, sample.analysis_id);
  assert.deepEqual(bundle.warnings, []);
  assert.equal(bundle.resolveAsset('../_static/region_idmap.png'), 'blob:test/export/_static/region_idmap.png');
  assert.equal(bundle.resolveAsset('brain_proxy.jpg'), 'blob:test/export/clip/brain_proxy.jpg');
  assert.equal(bundle.resolveAsset('brain_vertex.jpg'), null);
  bundle.dispose(); bundle.dispose();
  assert.deepEqual(urls.revoked, urls.created);
});

test('JSON-only import keeps signals and never substitutes fixture images', async () => {
  const analysis = json(sample, 'analysis.json'), urls = objectUrls();
  const bundle = await openLocalBundle(analysis, [analysis], urls);
  assert.equal(bundle.warnings.length, 3);
  assert.equal(bundle.resolveAsset('brain_proxy.jpg'), null);
  assert.equal(bundle.analysis.channels.length, sample.channels.length);
  assert.deepEqual(urls.created, []);
});

test('relative paths preserve clip identity and reject remote or escaping references', () => {
  assert.equal(localAssetPath('export/clip/analysis.json', '../_static/region_idmap.png'), 'export/_static/region_idmap.png');
  for (const path of ['https://example.com/map.png', '//example.com/map.png', 'data:image/png,abc', '/map.png', '../../../secret.png', '..\\secret.png', '%2fsecret.png', '%2e%2e/%2e%2e/%2e%2e/secret.png', 'map.png?token=1']) {
    assert.throws(() => localAssetPath('export/clip/analysis.json', path), path);
  }
});

test('invalid JSON, unsupported versions and duplicate paths fail without opening assets', async () => {
  const urls = objectUrls();
  const broken = file('analysis.json', '{');
  await assert.rejects(openLocalBundle(broken, [broken], urls), /not valid JSON/);
  const future = json({ ...sample, schema_version: 'nvi.analysis.v9' });
  await assert.rejects(openLocalBundle(future, [future], urls), /supports nvi.analysis.v0.2/);
  const valid = json();
  await assert.rejects(openLocalBundle(valid, [valid, valid], urls), /same path/);
  assert.deepEqual(urls.created, []);
});

test('later asset errors release previously created URLs and leave the export untouched', async () => {
  const data = structuredClone(sample);
  data.assets.region_map.idmap_src = 'https://example.com/map.png';
  const analysis = json(data), urls = objectUrls();
  await assert.rejects(openLocalBundle(analysis, [analysis, ...images()], urls), /relative local file/);
  assert.equal(urls.created.length, 1);
  assert.deepEqual(urls.revoked, urls.created);
  assert.equal(data.assets.region_map.idmap_src, 'https://example.com/map.png');
});

test('large files are rejected and research vertex maps are never opened', async () => {
  const large = file('analysis.json', '', 10_000_001), urls = objectUrls();
  await assert.rejects(openLocalBundle(large, [large], urls), /10 MB/);
  const data = structuredClone(sample);
  data.assets.brain_map.mode = 'research_vertex';
  data.assets.research_vertex_map = data.assets.brain_map;
  const analysis = json(data);
  const bundle = await openLocalBundle(analysis, [analysis, ...images()], urls);
  assert.deepEqual(urls.created, []);
  assert.equal(bundle.resolveAsset('brain_proxy.jpg'), null);
});

test('folder candidates prefer analysis.json and retain separate clip paths', () => {
  const files = [file('export/b/analysis.json'), file('export/metadata.json'), file('export/a/analysis.json')];
  assert.deepEqual(analysisFiles(files).map(f => f.webkitRelativePath), ['export/a/analysis.json', 'export/b/analysis.json']);
});
