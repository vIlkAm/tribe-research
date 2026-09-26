# RunPod: spin up, connect, run, tear down

Every pod launch is billed. Nothing here has been run yet; replace the
estimates in the README with measured numbers after the first run.

## One-time account setup

1. **SSH key.** Add this server's public key (`~/.ssh/id_ed25519.pub`, the
   whole `ssh-ed25519 …` line, not the fingerprint) in the RunPod
   console's SSH public keys settings (Settings / Credentials). RunPod injects account keys into new pods.
2. **HF token.** On huggingface.co, accept Meta's licence for
   `meta-llama/Llama-3.2-3B` with the account that owns the token. Then add the
   token in RunPod **Secrets** with the name `HF_TOKEN`.
3. **Network volume.** Create one (~150 GB is comfortable: venv, weights,
   feature cache, clips) in a datacenter that has L40S stock. The pod must be
   in the same datacenter. It bills monthly ($0.07/GB/month under 1 TB) until
   you delete it, whether or not a pod is running.

## Launch a pod

- GPU: **1× L40S** for Phase 1. For Phase 2 use **one pod with N GPUs**, not N
  pods on one volume. `launch_all.sh` runs one worker per GPU in that pod.
- Template: an official `runpod/pytorch` image with Python 3.11 and CUDA 12.4.
  `setup.sh` installs torch 2.6.0+cu124 into its own venv on the volume, so the
  image's torch version doesn't matter. The venv links to the image's Python,
  though: reuse the **same template** every time you launch against this volume.
  Blackwell GPUs (B200, RTX 50xx) are not supported by torch 2.6; `setup.sh`
  stops if it sees one.
- Attach the network volume at `/workspace`.
- Expose **TCP port 22** and enable the public IP. This is "full SSH". The
  default proxied `ssh.runpod.io` login cannot carry rsync or scp.
- Environment variable: `HF_TOKEN` = `{{ RUNPOD_SECRET_HF_TOKEN }}`.

## Connect

The pod's **Connect** tab shows "SSH over exposed TCP", for example
`ssh root@213.173.108.12 -p 17445`. From this server:

```bash
cd ~/projects/tribe-research
export POD=root@213.173.108.12 POD_PORT=17445
tools/pod.sh ssh                # shell on the pod
tools/pod.sh push-code          # pod/ tools/ tribe_research/ -> /workspace/tribe-job/code/
tools/pod.sh push-videos ./videos
```

The IP and port change every time a pod is created, so re-export them after
each launch.

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

`setup.sh` builds `tribe_research/assets/roi_map_roi_groups_v0.npz` from the
HCP-MMP1 annotation and cross-checks the vertex order against tribev2's own
labels. Figshare blocks this server, which is why the build runs on the pod.
If figshare also blocks the pod, copy `lh.HCPMMP1.annot` and `rh.HCPMMP1.annot`
from anywhere and run:

```bash
python $JOB/code/tools/build_roi_map.py --annot-dir <dir> --check-against-tribe
```

Inference does not depend on it; only `brain_report.py` does.

## Pull results and tear down

```bash
tools/pod.sh pull run1          # -> results/run1/{outputs,logs,manifest.jsonl}, plus the ROI map
.venv/bin/python tools/brain_report.py --out-root results/run1/outputs \
    --report-dir results/run1/report --roi-map tribe_research/assets/roi_map_roi_groups_v0.npz \
    --png --video --videos-root videos
```

Then **terminate** the pod in the console. Compute billing stops when the pod
stops. Everything under `/workspace` is on the network volume and survives
termination. Delete the volume only when you're finished with the project.

## Which videos to send

Start with 3–5 representative clips you own or have cleared for research use.
Pushing clips to a pod sends them to a third party, RunPod, and TRIBE v2 is
CC-BY-NC-4.0. The Clipping Cartel backfill archive
(`/archive/backfill-20260926/…`) is client media. It is **not** a default input.
Using it needs an explicit owner decision covering both the transfer and the
licence (see AGENTS.md). Performance metrics come in as an exported file, never
as a live database query.
