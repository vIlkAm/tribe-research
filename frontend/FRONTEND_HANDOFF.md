# ViralBrain frontend handoff

This directory is the complete, self-hostable ViralBrain frontend. It is safe to
develop and verify without the Sites deployment service.

## Quick start

Requirements: Node.js 20 or newer and npm.

```bash
npm ci
npm test
npm run build
npm run dev
```

The development server listens on `127.0.0.1:5175`. Open
`http://127.0.0.1:5175/`. The comparison/video-preview route is
`http://127.0.0.1:5175/?demo=comparison`.

For a production-style local check:

```bash
npm run build
npm run preview -- --host 127.0.0.1
```

The build output is the static `dist/` directory. It can be served by any static
host; no OpenAI Sites account is required. Keep SPA fallback enabled so `/`
continues to return `index.html`.

## Current product surface

- interactive 3D brain plus the contracted 2D atlas view;
- seven within-clip neural proxy signals, synchronized timeline and moments;
- browser-memory-only MP4/WebM/MOV attachment with shared playback controls;
- schema-validated local-folder and remote static-bundle loading;
- separate performance states with explicit unavailable/preliminary/research
  preview handling;
- a scripted comparison/video-preview route using the current YouTube reference;
- responsive public landing page and research workspace.

The default checked-in bundle is synthetic. The comparison and 3D presentation
are demo visuals, not measured subject activity. No source footage, private model
weights, credentials, lockbox outcomes or private backend releases are included.

## Backend integration seam

The frontend consumes `nvi.analysis.v0.2` bundles and an optional separate
`performance.json`. It already accepts the `data-frontend40-v2` layout and the
planned `data-demo-stage1-v1` layout described in tribe-research issue #2.

To start from a hosted approved demo index, set this at build time:

```bash
VITE_REAL_ANALYSIS_INDEX=https://example.invalid/data-demo-stage1-v1/index.json npm run build
```

The index and every referenced asset must be reachable from the browser, either
same-origin or with suitable CORS headers. Do not point a public build at a
private handoff unless the owner has explicitly approved exposing it.

An optional live job client remains isolated behind:

```bash
VITE_ANALYSIS_API_URL=https://api.example.invalid npm run build
```

It expects the provisional issue #2 job contract. Live ingestion is not required
for the static P0 demo and should not be presented as connected until the backend
confirms host, auth, CORS, expiry and upload limits.

Model weights, encoders, PCA state and reference tables must remain backend-only.
The browser receives only versioned JSON and display assets.

## Verification baseline

At frontend source commit `6a69665`:

- `npm test`: 48/48 passing;
- `npm run build`: passing TypeScript and Vite production build;
- live Sites publication: version 12 at
  `https://viralbrain-si.edelweiss-pro.com`.

The handoff documentation commit follows that verified code-only commit and does
not change runtime behavior. Re-run both commands after any change.
The exported handoff tree includes the later formatting-only source cleanup
`ff6e29d`.

## Files that must stay out of Git

The nested `.gitignore` excludes dependencies, builds, release archives,
TypeScript build state and `.env*` files. Also do not add:

- owned or downloaded source videos unless the owner explicitly clears them;
- `data-frontend40*`, `data-demo-stage1-v1` or other private release payloads;
- model artifacts, raw vertex arrays or lockbox material;
- GitHub, Sites, RunPod or API credentials.

## Backend takeover checklist

1. Work from the GitHub handoff branch/PR and keep the app under `frontend/`.
2. Run `npm ci`, `npm test` and `npm run build` inside `frontend/`.
3. Serve `frontend/dist/` locally and verify `/` and `/?demo=comparison`.
4. When `data-demo-stage1-v1` arrives, validate all three index entries without
   committing the private payload.
5. Attach the exact owner-cleared video locally and verify duration, first frame,
   play/pause/seek and the analysis playhead.
6. Preserve the scientific wording and status gates in issue #2.
7. Publish later from the same code only after the owner asks; local verification
   does not require a live deployment.

Coordination and backend delivery status remain in
`https://github.com/vIlkAm/tribe-research/issues/2`.
