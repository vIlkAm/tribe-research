# ViralBrain SI — self-hostable frontend

This checkout is the source of truth for the ViralBrain app. It can run locally,
build to a portable static `dist/` directory, or be published through Sites.
See [`FRONTEND_HANDOFF.md`](FRONTEND_HANDOFF.md) for the backend takeover guide.

- Main app: `/`
- Comparison: `/?demo=comparison`
- Target custom domain: `viralbrain-si.edelweiss-pro.com`
- `npm run dev`: development at port 5175
- `npm test`: contract, import and playback checks
- `npm run build`: static production output in `dist/`

`.openai/hosting.json` identifies the Sites project and static output. Publish
through the Sites workflow; keep source credentials out of files. Assets, styles
and scripts are served normally instead of being compressed into a ClickFunnels
HTML field. YouTube remains the only optional external playback dependency.

The included default sample is synthetic. The rich workspace includes the
interactive 3D presentation, the contracted atlas and all seven supplied proxy
signals. The backend team's private
`data-frontend40` handoff (40 real bf16 TRIBE v2 analyses) has been validated
locally but is not committed or published with this public Site. An approved hosted handoff
can be exposed at build time through `VITE_REAL_ANALYSIS_INDEX`; otherwise users
can open an extracted v0.2 export locally. Real bundles use the contracted 2D
atlas; the 3D view remains a presentation layer and must not be described as
measured subject activity.

The **Analyze video** flow is ready for the proposed job API and supports a link
or whole-video upload, deal/platform context, polling through
`queued → fetching → extracting → predicting → done`, errors, and remote bundle
loading. Configure it at build time with an HTTPS `VITE_ANALYSIS_API_URL` (HTTP
is accepted only for localhost). Without that setting, the public Site honestly
shows that live analysis is not connected and directs users to the real static
handoff. GitHub issue `tribe-research#2` tracks the API/auth/CORS agreement.

Model weights never enter the browser. Fixed extractor/TRIBE weights stay on the
GPU worker; the learned performance artefact stays on the backend; the frontend
receives versioned analysis JSON and normal-mode assets only. Behavior and
performance predictions remain unavailable until the pre-registered training
gate passes and the proposed v0.3 contract is agreed.

The UI accepts the proposed additive `performance` block. It renders honest
`not_trained` and `out_of_scope` empty states and renders percentile, range,
context, drivers, caveats, and warnings for scored `preliminary`,
`research_preview`, or `validated` results. Preliminary output is explicitly
labelled unvalidated and shows the backend caption verbatim. Missing blocks never
produce placeholder performance numbers.

The comparison brain is a scripted illustration, not a measurement or TRIBE
analysis. See THIRD_PARTY_NOTICES.md for asset sources and attribution.

The main timeline includes a local source-clip preview with an optional response
overlay. Attach the matching MP4/WebM/MOV through the preview row. The file stays
in browser memory, is never uploaded, and is released when replaced, removed or
the workspace closes. The video clock drives the pattern, brain and transport;
play/pause, seek, speed and loop are shared. No extra hemodynamic lag is added.
Different durations stop/loop at the shorter endpoint and show a mismatch note.
Synthetic labels remain visible; attaching a clip does not analyze it or verify
that it matches the bundle. No source video is shipped with the synthetic sample.
Hosted source-preview authorization/delivery remains part of `tribe-research#2`.

The parent directory retains the former ClickFunnels deployment for rollback.
