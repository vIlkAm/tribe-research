# Sample analysis bundle (SYNTHETIC)

`cd20b16879d630c4/analysis.json` follows `docs/analysis.schema.json`
(`nvi.analysis.v0.2`). It was built from **dry-run** predictions (random, not
TRIBE output) and the real HCP-MMP1 ROI map, so `synthetic: true`. Use it for
layout and wiring only; none of its numbers mean anything.

Asset paths in `assets` are relative to the folder holding `analysis.json`:

- `brain_proxy.jpg`: `brain_map` sprite. One 480×360 tile per native TRIBE
  step, row-major, `cols`×`rows`. Show tile *i* while
  `frame_start_ms[i] <= t < frame_end_ms[i]`. Each channel's region is painted
  with its within-clip z (`colormap`: RdBu_r, ±2.5, faded below 0.5).
- `../_static/region_idmap.png`: same tile geometry, flat colour per channel.
  Read the pixel under the cursor and look it up in `region_map.ids`
  (`#000000` = no channel).
- `../_static/region_legend.png`: channel colours, directions and the colormap.

Real bundles also have `brain_vertex.jpg` (`research_vertex_map`, Research mode
only), `summary.png` and `demo.mp4`. Regenerate this sample with:

```bash
.venv/bin/python tools/brain_report.py --out-root <dry-run outputs> --report-dir /tmp/sample \
    --roi-map tribe_research/assets/roi_map_roi_groups_v0.npz --videos-root <videos> \
    --analysis --no-research-vertex --synthetic
```
