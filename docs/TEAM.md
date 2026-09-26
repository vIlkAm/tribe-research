# Working together

Two people, two agents, one product. This file is how the research/backend side
(this repo) and the frontend side (the teammate's app) stay in step. Humans and
agents on both sides should read it before changing anything that crosses the
seam.

## Who owns what

| Area | Owner | Lives in |
|---|---|---|
| TRIBE runs, RunPod, clip selection | Research/backend (repo owner) | this repo: `pod/`, `tools/` |
| Neural proxies, moments, analysis JSON, brain assets | Research/backend | `tribe_research/brain/` |
| The UI: layout, interaction, visual design | Frontend (teammate) | the frontend repo |
| **The contract between them** | **Both, by agreement** | `docs/analysis.schema.json`, `docs/analysis.types.ts` |
| Frontend spec | UI sections: frontend. Data sections (§07 API contract, §08 brain assets): both | `docs/frontend_spec/spec_v0.2.html` (PDF is rendered from it) |

The contract is the only thing either side depends on. The frontend never
needs to know how TRIBE works; the backend never assumes a UI layout.

## Sources of truth (read in this order)

1. `docs/analysis.schema.json`: the contract (JSON Schema). Wins any disagreement.
2. `docs/analysis.types.ts`: the same contract as TypeScript. Copy it into the
   frontend as-is; `tests/test_types.py` keeps it identical to the schema.
3. `docs/sample_analysis/`: a synthetic bundle with assets. Use it as the fixture.
4. `docs/frontend_spec/`: why the UI shows what it shows, the states and the copy rules.
5. This repo's `README.md` / `AGENTS.md`: how the data is produced.

## Where we are (2026-09-26)

- **Real now:** 7 neural proxy channels, candidate moments, shots/speech lanes,
  brain sprites and hover map. The format is final for v0.2; only the numbers
  in the sample are fake (dry-run).
- **Next:** the first real run on a curated 1,500-clip study set, in batches.
  The first 50 real bundles (pilot + a wiring set with the shortest and longest
  clips, a >60 s clip and a near-silent one) come as a tarball from
  `tools/handoff.py`: `index.json` plus `<video_id>/analysis.json`,
  `brain_proxy.jpg`, `brain_vertex.jpg` (Research mode) and `_static/`. Every bundle is schema-validated before it
  ships; clips themselves are never included. Same format as the sample.
  `quality.warnings` can now also say the speech looks non-English.
- **Later:** import real outcomes (retention etc.), then a behavior model. Only
  then does `predictions` become `available`, and the hold/completion cards get
  real numbers. Until then the frontend shows the empty state.
- **Product path:** link/upload → GPU worker → analysis + a proposed `performance`
  block (v0.3, not yet in the contract): [`PRODUCT_PIPELINE.md`](PRODUCT_PIPELINE.md).

## Changing the contract

Nobody changes it alone, and nobody works around it silently.

- **Frontend needs something new or different:** open an issue here with the
  `contract` label saying what the UI needs and why (not how to compute it).
- **Backend changes it:** one PR that updates schema, types, sample bundle,
  spec §07/§08, tests and the spec changelog together. Additive changes (new
  optional field, new channel, new moment kind) bump the minor version
  (`nvi.analysis.v0.3`). Renames/removals are breaking and need the frontend's
  OK on the PR first.
- **The frontend should:** check `schema_version` and show a clear error for an
  unknown major version; ignore unknown extra fields; never hard-code channel
  keys or colours (read `channels[]`); never infer good/bad from a key (use
  `direction`).

## Shared rules that both sides enforce

- Say "strong predicted response", never "brain firing". No "caused", no
  mind-reading, no universal score.
- Every neural value is a z-score **within one clip**. Never compare across
  clips; `severity` ranks moments inside one clip only.
- `null` in `values` = no prediction. Show a gap; never interpolate.
- `synthetic: true` → visible SYNTHETIC banner.
- No placeholder behavior numbers while `predictions.status` is `not_available`.
- Raw per-vertex maps (`research_vertex_map`) only in Research mode.

## How we talk

- Use GitHub issues and PRs on this repo for anything touching the contract or
  the spec, so both agents can read the history. Put decisions in the PR/issue,
  not only in chat.
- Agents: summarise what you changed and why in the PR description; link the
  issue. Ask the human owner before anything outward-facing.
- Off-limits for collaborators and their agents: launching RunPod, the owner's
  server, any company database, and committing clips or TRIBE outputs.
