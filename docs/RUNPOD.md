# RunPod: spin up, connect, run, tear down

Every pod launch is billed. Nothing here has been run yet; replace the
estimates in the README with measured numbers after the first run.

## Setup on this server (once)

Put the project's own tokens in `.env` in the repo root (gitignored, never a
Cartel credential):

```
RUNPOD_API_KEY=...
HF_TOKEN=...
```

Keys with the least access that works:

- **Runpod:** console → Settings → API Keys → **Restricted**, with read/write
  for Pods and read for everything else this tool uses (GPU catalog, billing
  balance); no Serverless, registries or templates. The watchdog uses this key
  to stop pods at the cap, so it must live here: a Runpod MCP sign-in is
  session-scoped and can't back an unattended cap. Pods created through the
  Runpod MCP also skip `pod-create`'s watchdog check, so create them here.
- **Hugging Face:** a fine-grained **read** token.
  The HF account must have accepted Meta's licence for `meta-llama/Llama-3.2-3B`.
`tools/runpod.py` reads both, never prints them, and masks the `env` block in
any pod JSON it shows. The pod gets this server's `~/.ssh/id_ed25519.pub` as
`PUBLIC_KEY`, so no console SSH-key setup is needed.

## Launch with `tools/runpod.py` (cheap path: one community pod)

The budget is about **$10 in total**. The default path is the cheapest one:

- one on-demand **Community Cloud** pod;
- a **pod volume** (120 GB) mounted at `/workspace`, so `pod/env.sh` and
  `setup.sh` paths work unchanged;
- no network volume.

```bash
cd ~/projects/tribe-research
R=tools/runpod.py

# 1. Pick a GPU. Cheapest first, community vs secure $/h (on-demand and spot),
#    stock, VRAM and host RAM. Hides types under 24 GB VRAM (TRIBE needs >= 24 GB)
#    and filters stock to hosts with >= 48 GB RAM per GPU (--min-ram-gb).
#    RTX 4090, 3090, A40, A6000, L40S and L4 are always listed.
$R gpus
$R gpus --min-ram-gb 64 --dc EU-RO-1      # stricter RAM, one datacenter

# 2. Watchdog FIRST (see below). pod-create refuses to run without a fresh heartbeat.
mkdir -p results
nohup $R watchdog --max-usd 10 --max-hours 6 >> results/watchdog.log 2>&1 &

# 3. Pod: community, on-demand, 120 GB pod volume at /workspace, 30 GB container disk,
#    TCP 22 on a public IP (mapped port), host RAM >= 48 GB per GPU.
$R pod-create --name tribe-p1 --gpu-type 4090 --max-hours 3 --dry-run   # print request, send nothing
$R pod-create --name tribe-p1 --gpu-type 4090 --max-hours 3

# 4. Wait for SSH; this sets POD / POD_PORT for tools/pod.sh.
eval "$($R pod-wait <POD_ID>)"
```

> **WARNING: `pod-terminate` DELETES THE POD VOLUME.** Everything under
> `/workspace` (venv, weights, feature cache, clips, **outputs**) is gone the
> moment the pod is terminated. There is no undo. **Pull results first**
> (`tools/pod.sh pull <run>`, see "Pull results and tear down"). For that reason
> `pod-terminate` on a pod-volume pod refuses to act without `--yes`, and the
> watchdog **stops** such pods rather than terminating them.

`pod-create` defaults and options:

- `--cloud COMMUNITY` (default) or `SECURE`. Community hosts expose TCP 22
  through a **mapped** public port (e.g. `213.173.109.39:13007 -> :22`);
  `pod-wait` reads the mapping. It falls back to GraphQL runtime ports when
  REST shows none. A community pod's IP can change after a stop/start, so run
  `pod-wait` again after each start.
- On-demand by default. `--interruptible` asks for a **spot** pod: cheaper, but
  RunPod may stop it at any time. A stopped spot pod keeps its pod volume, but
  a worker mid-clip loses that clip (workers resume on restart). Only use spot
  with an explicit decision.
- `--volume-gb 120` (pod volume size) and `--container-disk-gb 30`. `setup.sh`
  puts the venv, weights, feature cache and clips on `/workspace`; size the
  volume up for a large clip set (sizes not yet measured).
- `--min-ram-gb 48` is sent as `minRAMPerGPU`. Our README flags 4090 hosts with
  31 GB RAM as a risk for loading V-JEPA2 ViT-G, Llama-3.2-3B, Wav2Vec-BERT and
  whisper large-v3 together.
- Image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`, on hosts
  with a driver for CUDA ≥ 12.8 by default (`--min-cuda`): the whisperx uvx env
  pins torch 2.8 (cu128 wheels). `--min-cuda 12.4` widens the pool but may break
  transcription. `setup.sh` installs torch 2.6.0+cu124 into its own venv on
  `/workspace`. The venv links to the image's Python, so **keep the same image**
  for any pod that reuses that disk.
- The name must start with `tribe-`, the only pods the watchdog touches.
  `--max-hours` is required and recorded in `results/runpod_state.json`.
- Blackwell GPUs (B200, RTX 50xx, RTX PRO Blackwell) are refused: torch 2.6
  doesn't support them.
- "No instances currently available" (HTTP 500) is common. Retry another GPU
  or `--cloud SECURE`; loosening one filter at a time rarely helps.
- Check a fresh pod before setup: `nvidia-smi` should show 0 % utilisation and
  ~0 MiB used. A 2026-09-26 community 4090 showed 100 % / 5 GiB busy at start
  and refused the pod's own key; terminate such a host rather than debug it.
- Stock images lack rsync; `pod.sh` installs it on the pod before the first
  push. Pod env vars (HF_TOKEN) reach only PID 1, so `pod/env.sh` reads them
  from `/proc/1/environ` for SSH shells.
- GPU names can be short (`4090`, `3090`, `A40`, `A6000`, `L40S`, `L4`).

Phase 1 is one pod with one GPU. For Phase 2 use **one pod with N GPUs**
(`--gpu-count N`, and `gpus --gpu-count N` to find hosts with that many free);
`launch_all.sh` runs one worker per GPU in that pod.

### Optional: network volume (Secure Cloud only)

A network volume outlives every pod, so weights and outputs survive a terminate
and a later pod skips the ~20–40 min `setup.sh`. It costs more:

- Secure Cloud prices;
- the volume bills monthly (about $0.07/GB/month) until deleted, pod or no pod;
- the pod must run in the volume's datacenter.

Only worth it if you'll run several separate sessions.

```bash
$R gpus --dc EU-RO-1                       # "network volumes here: yes/no" in the header
$R volume-create --name tribe-vol --size-gb 150 --dc EU-RO-1
$R volume-list
$R pod-create --name tribe-p1 --gpu-type L40S --cloud SECURE --volume-id <VOLUME_ID> --max-hours 3
```

`--volume-id` without `--cloud SECURE` is refused. The datacenter is taken
from the volume. The watchdog **terminates** network-volume pods, since
`/workspace` survives. Delete the volume in the console when the project is
done.

## Watchdog (required during any pod run)

Run the watchdog on this server, under `nohup`, with explicit caps, for as
long as any pod exists. Each interval (60 s) it lists `tribe-*` pods and logs
one line: estimated spend, burn rate, per-pod age, $/h and GPU utilisation.

It acts on a pod when:

- the pod is older than its own `--max-hours` or the watchdog's `--max-hours`,
  whichever is lower;
- estimated total spend reaches `--max-usd`. Every `tribe-` pod is acted on.
  The total includes pods that ended while this watchdog ran, so any new pod is
  also acted on;
- the pod has been at 0% GPU for `--idle-minutes` (default 20; 0 turns it off).
  This applies only once the pod has used the GPU, so setup isn't killed.
  Utilisation comes from RunPod's GraphQL runtime data; when that's
  unavailable, the idle rule waits and the caps still apply.

The action depends on the volume:

- **Pod-volume pods (the default) are STOPPED, not terminated.** GPU billing
  ends and `/workspace` is kept. A stopped pod's volume disk is still billed at
  a small per-GB rate, and it counts no further toward `--max-usd`. To pull
  after a stop:

  ```bash
  $R pod-start <POD_ID> --grace-min 30     # watchdog leaves it alone for 30 min
  eval "$($R pod-wait <POD_ID>)"
  tools/pod.sh pull run1
  $R pod-terminate <POD_ID> --yes          # deletes the pod volume
  ```

  A stopped community pod can only restart on **its own host**, and only if a
  GPU there is free. If none is, the data may be unreachable for a while. So
  pull during the run, not only at the end (see "Pull results and tear down").
- **Network-volume pods are terminated**; `/workspace` survives on the volume.

Idle stops also hit hands-on pauses. After the `--limit 1` test clip the pod
counts as used, so checking that output for more than 20 minutes stops it. At
the end of a run the pod is stopped 20 minutes after the workers finish. Use a
larger `--idle-minutes` for hands-on phases.

The spend estimate is $/h × running time since creation. A failed pod list
takes no action and is logged as an error. A failed stop or terminate is
retried on the next tick.

```bash
tail -f results/watchdog.log
$R balance                                # credit, account-wide $/h, runway
$R pod-list
$R watchdog --max-usd 10 --max-hours 6 --once   # one check, then exit
```

## Connect

`pod-wait` prints `export POD=root@IP POD_PORT=PORT`, and the `eval` above
applies it. Then, from this server:

```bash
tools/pod.sh ssh                # shell on the pod
tools/pod.sh push-code          # pod/ tools/ tribe_research/ -> /workspace/tribe-job/code/
tools/pod.sh push-videos ./videos
```

The IP and port change with every new pod and may change after a stop/start.
Run `pod-wait` again each time.

## Manual fallback (console)

If the script can't be used:

1. Add `~/.ssh/id_ed25519.pub` under Settings → SSH public keys, and add
   `HF_TOKEN` as a RunPod Secret.
2. Deploy an on-demand **Community Cloud** pod with:
   - the image above;
   - a 120 GB **volume disk** mounted at `/workspace`;
   - a 30 GB container disk;
   - TCP 22 exposed with a public IP;
   - env `HF_TOKEN={{ RUNPOD_SECRET_HF_TOKEN }}`.
3. Name it `tribe-…` so the watchdog still covers it.
4. Use the "SSH over exposed TCP" address from the Connect tab as
   `POD`/`POD_PORT`.

The same warning applies: terminating deletes the volume disk.

## Run (on the pod)

```bash
bash /workspace/tribe-job/code/pod/setup.sh          # first time ~20–40 min; later runs are quick
source /workspace/tribe-job/code/pod/env.sh
python $JOB/code/tools/make_manifest.py --videos-root $JOB/videos --workers 1 --out $JOB/manifest.jsonl
WORKER_ID=0 NUM_WORKERS=1 $JOB/code/pod/run_worker.sh --limit 1     # one real clip first
WORKER_ID=0 NUM_WORKERS=1 $JOB/code/pod/run_worker.sh               # rest of the sample
python $JOB/code/tools/merge.py --manifest $JOB/manifest.jsonl --out-root $JOB/outputs
```

Phase 2 on an N-GPU pod: rebuild the manifest with `--workers N`, then run
`$JOB/code/pod/launch_all.sh`. It starts workers under `nohup`, so the SSH
session can close. Follow progress with `tail -f $JOB/logs/worker-*.log`.

`setup.sh` re-installs the container-disk apt packages (ffmpeg, git, rsync),
so run it again after any pod restart. It is idempotent.

## ROI map (for the brain visuals)

`tribe_research/assets/roi_map_roi_groups_v0.npz` is built locally from the
GitHub mirror of the HCP-MMP1 annotation (MD5s in its provenance) and pushed by
`pod.sh push-code`. `setup.sh` then rebuilds it and cross-checks the vertex
order against tribev2's own labels. If figshare blocks the pod, the rebuild
fails without touching the pushed file. To run the cross-check anyway, copy
`lh.HCPMMP1.annot` / `rh.HCPMMP1.annot` to the pod and run:

```bash
python $JOB/code/tools/build_roi_map.py --annot-dir <dir> --check-against-tribe
```

Inference does not depend on it; only `brain_report.py` does.

## Pull results and tear down

> **Pull before you terminate.** On the default path, the outputs exist only
> on the pod volume. `pod-terminate` deletes that volume.

Pull repeatedly during a long run as well. `pod.sh pull` is an incremental
rsync, so a loop on this server is cheap. It also covers a watchdog stop on a
host that later has no free GPU:

```bash
while sleep 900; do tools/pod.sh pull run1; done    # Ctrl-C when the run is done
```

At the end:

```bash
tools/pod.sh pull run1          # -> results/run1/{outputs,logs,manifest.jsonl}, plus the ROI map
ls results/run1/outputs | head  # check the pull landed before terminating
.venv/bin/python tools/brain_report.py --out-root results/run1/outputs \
    --report-dir results/run1/report --roi-map tribe_research/assets/roi_map_roi_groups_v0.npz \
    --analysis --png --video --videos-root videos
```

Then terminate:

- pod volume: `tools/runpod.py pod-terminate <POD_ID> --yes`. It refuses
  without `--yes` and prints the warning. **This deletes `/workspace`.**
- network volume: `tools/runpod.py pod-terminate <POD_ID>`; `/workspace`
  survives on the volume.

`pod-stop` ends GPU billing but keeps billing the disk. A stopped community
pod may find no free GPU on its host when resumed, so pull, then terminate.
Stop the watchdog once `pod-list` shows no `tribe-` pods.

## Quick batches, frontend handoff, overnight

The curated study set (`docs/STUDY_SET.md`) is sliced into disjoint batches
that share one `$JOB/outputs`. Keep **one pod** for the day: setup is paid per
pod. The watchdog stops a pod after 20 min at 0 % GPU, so run batches back to
back or pass `--idle-minutes 60` for an interactive session.

```bash
# on this server, once the GPU count N of the pod is known
.venv/bin/python tools/make_batches.py --gpus N      # results/batches/{b00_pilot,b01_frontend,b02..,d00_deep_dive..}
tools/pod.sh push-code
tools/pod.sh push-batch results/batches/b00_pilot
# on the pod
bash $JOB/code/pod/setup.sh && source $JOB/code/pod/env.sh
$JOB/code/pod/launch_all.sh                           # N workers, nohup
# next batch: push-batch results/batches/b01_frontend, launch_all.sh again
```

Back here after each batch: pull, build bundles, validate and package.

```bash
tools/pod.sh pull study_run
.venv/bin/python tools/brain_report.py --out-root results/study_run/outputs \
    --report-dir results/study_run/report --roi-map tribe_research/assets/roi_map_roi_groups_v0.npz \
    --videos-root results/study/staging --analysis --jobs 8 --skip-existing
.venv/bin/python tools/handoff.py --analyses results/study_run/report/analyses \
    --out results/handoff/study_run.tar.gz --expect-real
```

`handoff.py` validates every bundle against the contract and stops on the
first invalid one; the tarball never contains clip footage (`*.mp4`). Send
b00 + b01 (50 clips) to the frontend first. Each later batch is a small
representative sample, so a finished prefix is usable on its own.

Overnight: push every remaining batch's clips, then run them one after the
other in a single `nohup` loop on the pod, and keep a pull loop here. The
watchdog's idle stop ends billing after the last batch; its `--max-usd` must
cover the whole run (set it from the measured $/clip, not the $10 default).

```bash
for b in results/batches/b0[2-9] results/batches/d*; do tools/pod.sh push-batch "$b"; done
# on the pod
nohup bash -c 'for m in $JOB/batches/b0[2-9].jsonl $JOB/batches/d*.jsonl; do
  cp "$m" $JOB/manifest.jsonl; $JOB/code/pod/launch_all.sh; sleep 30
  while pgrep -f pod/worker.py >/dev/null; do sleep 60; done; done' > $JOB/logs/overnight.log 2>&1 &
# here
while sleep 900; do tools/pod.sh pull study_run; done
```

## Which videos to send

The backfilled clips are company-owned and cleared by the owner (2026-09-26)
for this non-commercial research, including copying them to RunPod. Pick a
stratified sample with `tools/select_sample.py`. It copies the files (never
symlinks, which rsync would push as broken links), checks each sha256, and
keeps the `video_performances` id as `source_name` for the later metrics join.
Performance metrics still arrive as an exported file, never as a live
database query. TRIBE v2 is CC-BY-NC-4.0: revisit the licence before any of
this feeds a product or client deliverable.

```bash
.venv/bin/python tools/select_sample.py --n 60 --out videos --dry-run   # plan only
.venv/bin/python tools/select_sample.py --n 60 --out videos             # copy + sha256 check
tools/pod.sh push-videos videos
```

It reads only the backfill's `*.results.jsonl` logs and only files under
`archive/backfill-20260926/`. Strata are deal × platform × duration bucket
(<15 s, 15–30 s, 30–60 s, >60 s), clips over `--max-duration` (90 s) are
skipped, and identical content is sent once. `videos/_sample.jsonl` records
vp_id, deal_id, platform, duration and sha256 for each clip. A 60-clip draw
(seed 0) is about 28 min of source across 14 deals.

## Pilot findings (L40S, 2026-09-26)

- TRIBE's V-JEPA2 extractor dominates: ~92 % of clip time. On stock 1080p clips
  it ran at 0.06-0.14x realtime with the GPU ~30 % busy, because each 0.5 s
  step decodes and resizes 64 full-size frames on one CPU thread.
- Pre-scaling to a 384 px short side (`tools/prep_cpu.py`) gave 0.18-0.20x
  realtime (~5 GPU-s per video-s) with predictions r >= 0.998 vs stock. At that
  rate the 13.6 h set is ~68 GPU-h (~$74 on an L40S): too slow to scale.
- GPU floor (`pod/bench_vjepa.py`): 0.256 s per 64-frame forward, 0.51 GPU-s
  per video-s; batching does not help. A frame-loop patch is the lever.
- Peak VRAM per worker including the whisper server: 18.2 GB (fits 24 GB).
- Pilot artifacts (outputs, logs, pip freeze, configs): `results/logs/pilot-l40s/`.

## Fleet run

`tools/fleet.py` runs many pods in parallel from one queue of batch dirs. It
calls `tools/runpod.py` for every API call. It does its own SSH/rsync (the same
`-rlpt` flags as `pod.sh`) with a known_hosts file per pod, because community
hosts reuse IP:port. Nothing about it has been tried on a live pod yet.

```bash
# plan only (no API call, nothing written): queue, clips, rate, wall time, cost
tools/fleet.py run --pods 3 --gpu-type L40S --max-usd 60 --usd-per-hour 0.90
# skip batches a manually driven pod already runs
tools/fleet.py run ... --exclude b02,b03,b04
# the same with --go creates pods (billed); log in results/fleet/fleet.log
nohup tools/fleet.py run ... --go >> results/fleet/driver.out 2>&1 &
tools/fleet.py status              # per pod: phase, batch, next, clips/h, $; spend, ETA, halt, waiting
tools/fleet.py add r16 'results/batches_s384/x*'   # join the running queue (merged at the next tick)
tools/fleet.py exclude b05,b06                     # leave the running queue
tools/fleet.py run --go            # resume after a crash/restart; the stored config is reused
```

- **Defaults (measured on the L40S, 2026-09-26).**
  - `--fast-video --video-precision bf16` with `$JOB/feature-cache-bf16`.
    Turn it off with `--no-fast-video`.
  - Workers per GPU = floor((VRAM − 3 GB) / 19 GB), because each worker uses
    about 13.5 GB and its whisper server about 5.5 GB. That gives 2 on 48 GB
    cards, 1 on 24 GB cards and 4 on 80 GB cards. Override with
    `--workers-per-gpu`.
  - The launch script reads the pod's cgroup CPU quota (`cpu.max`, or v1
    `cpu.cfs_quota_us`/`cpu.cfs_period_us`) because nproc reports the host. It
    sets `OMP_NUM_THREADS` and `MKL_NUM_THREADS` to max(4, floor(quota /
    workers)), and `TRIBE_VIDEO_THREADS` to min(8, that). It also sets
    `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Override the threads
    with `--omp-threads`.
  - The plan assumes 1.1 s of wall time per source-second per worker
    (`--wall-per-source-s`). That is ~120 clips/h per worker at a 26 s mean.
    Other GPUs are assumed to match.
- **Queue order without `--batches`:** `results/batches_s384/` b02..b09, then
  d00_deep_dive and d01_deep_dive, then r00..r15.
  - Only a batch with `prep.json` can be claimed. The rest wait, and a
    missing dir waits too.
  - Globs given to `--batches` are rescanned while the driver runs.
  - `add` and `exclude` go through `results/fleet/inbox.jsonl`, because the
    driver owns `state.json`.
  - An excluded batch this fleet is already running finishes, but it is not
    requeued.
- **Resume:** `run` starts from the config stored in `state.json`, and only
  flags given on the command line override it.
  - Pod-shaping flags are refused while a fleet pod is live: GPU, disk,
    image, job dir, workers and threads.
  - Output-shaping flags are refused for the whole run: fast-video, precision
    and worker args. Use `--fleet-dir` for a separate run.
  - A restart clears a halt.
- **Job dir on the container disk.** `JOB=/root/tribe-job` by default
  (`--container-disk-gb 100`, `--volume-gb 20` for the unused pod volume).
  - JOB is exported at the top of every remote script and for every local
    subprocess.
  - A stop, a spot pre-emption or a watchdog action **erases** the container
    disk. So a running batch is pulled every `--pull-every-min` (15), and
    `--go` refuses a watchdog whose `--max-usd` or `--max-hours` is lower
    than the fleet's.
- **Per pod:**
  - create → SSH → push code (rsync first, so it never races setup's apt) →
    `setup.sh` under `setsid nohup` (`logs/setup.out`) → push the first batch
    while setup runs → wait for `$JOB/.setup_done`.
  - On `$JOB/.setup_failed` or after `--setup-timeout-min` (30), the driver
    pulls `setup-timings.txt`, `prefetch.log` and `setup.out`, then
    terminates the pod.
  - A `.setup_failed` also **halts** the fleet: no new pods until a restart.
    A timeout gets a replacement pod.
  - A pod that finds nothing left to claim during setup is terminated at once.
  - Workers are launched through `run_worker.sh` with `GPU = k % gpu_count`,
    one pid file each.
- **Prefetch:**
  - Batch k+1 is pushed as soon as batch k is launched.
  - When k finishes, k+1 is launched *before* k is pulled.
  - A pull is verified per clip: `<vid>.json` + `.npz` for every clip the
    pod reports done. Only then is the batch done and its videos deleted from
    the pod.
  - A batch with no outputs dir on the pod is not rsynced.
  - A batch of ≥ 3 clips that finishes with **0 clips done** (after worker
    retries) means the pod or the code is broken. The driver pulls logs,
    terminates the pod, requeues the batch and halts the fleet.
- **Queue:** `results/fleet/state.json` is file-locked, and only one driver
  runs at a time.
  - A batch that loses its pod goes back to the queue. The next pod's
    manifest drops the clips already pulled into
    `results/fleet/outputs/<batch>/`, and `worker`/`num_workers` are
    rewritten for that pod's worker count.
  - A sharded prep (`prep.shard*.json` only) does not count as ready.
  - After 3 failed attempts a batch is marked failed.
  - Dead workers are relaunched once (`--worker-retries`).
  - A pod unreachable for `--dead-after-min` (10), or stopped/missing in
    `pod-list`, is terminated and replaced. The limit is `--max-creates` pods
    (default 2 × `--pods`). The same applies while draining.
  - If a finished batch's pull never verifies, the pod is ended after
    `--drain-timeout-min` (30) and the batch is requeued.
- **Money:**
  - The estimate is $/h × uptime for every pod the fleet created.
    `--usd-per-hour` is used until the API reports the real rate.
  - Within 15 min of burn of `--max-usd`, the driver stops creating pods and
    claiming batches. Idle pods and pods still in setup are terminated.
  - At `--max-usd`, it pulls partial outputs and terminates every fleet pod.
  - An idle pod with nothing ready is pulled and terminated after
    `--idle-grace-min` (10, below the watchdog's 20).
  - A failed terminate keeps the pod in the spend. It is retried every
    `pod-list` poll until the API shows the pod gone. `status` shows it as
    STILL BILLED, and the driver does not exit before then.
  - It only ever stops or terminates pod ids recorded in its own state.
- Merge a batch afterwards with `tools/merge.py --manifest
  results/batches_s384/<b>/manifest.jsonl --out-root results/fleet/outputs/<b>`.
