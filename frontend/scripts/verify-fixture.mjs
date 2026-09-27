import { readFile, readdir } from 'node:fs/promises';
import { join, relative } from 'node:path';
import assert from 'node:assert/strict';
const upstream = process.argv[2] ?? '../tribe-research';
const pairs = [
 ['src/data/analysis.types.ts', join(upstream, 'docs/analysis.types.ts')],
 ['src/data/analysis.schema.json', join(upstream, 'docs/analysis.schema.json')],
];
async function fixtures(path) {
 for (const entry of await readdir(path, { withFileTypes: true })) {
  const file = join(path, entry.name);
  if (entry.isDirectory()) await fixtures(file);
  else pairs.push([join('public/data', relative(join(upstream, 'docs/sample_analysis'), file)), file]);
 }
}
await fixtures(join(upstream, 'docs/sample_analysis'));
for (const [local, source] of pairs) assert.deepEqual(await readFile(local), await readFile(source), `Fixture differs: ${local}`);
console.log(`${pairs.length} fixture/contract files match the sibling research repository byte for byte.`);
