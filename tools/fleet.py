#!/usr/bin/env python3
"""Fleet driver: N RunPod pods pull clip batches from one persistent queue, GPUs kept busy.

    tools/fleet.py run --pods 3 --gpu-type L40S --max-usd 60 --usd-per-hour 0.90   # plan only
    tools/fleet.py run ... --exclude b02,b03 --go        # creates pods (billed)
    tools/fleet.py run --go                              # resume: stored config, flags override
    tools/fleet.py status                                # per pod, per batch, spend, ETA
    tools/fleet.py add r16 | exclude b04,b05             # change a running queue (inbox.jsonl)

Defaults are the L40S measurements of 2026-09-26: bf16 fast-video, workers per GPU =
floor((VRAM - 3 GB) / 19 GB), OMP/MKL threads = max(4, cgroup CPU quota / workers) read on
the pod, 1.1 s wall per source-second per worker for the plan. Queue without --batches:
results/batches_s384/ b02..b09, d00/d01_deep_dive, r00..r15 (claimable once prep.json exists).

Per pod slot: pod-create -> pod-wait -> push code -> setup.sh in the background (nohup,
$JOB/.setup_done or .setup_failed, rc file) -> WHILE setup runs, push the first batch -> launch
workers when setup is done -> poll every --poll-s -> prefetch (push) the next batch as
soon as the current one is launched -> when the current batch finishes, launch the
prefetched one FIRST, then pull the finished outputs (verified per clip) into
results/fleet/outputs/<batch>/ -> when nothing is left, pull logs and terminate.

Everything on the pod lives under --job-dir (default /root/tribe-job, the container
disk), exported as JOB at the top of every remote script: pod env vars do not reach
non-interactive SSH shells. Container disk is ERASED by a stop, a spot pre-emption or a
watchdog action, so running batches are also pulled every --pull-every-min.

Queue: results/fleet/state.json (file-locked, atomic). A batch is claimed by one pod at a
time; a dead pod's batches go back to the queue and, because rows already pulled are
dropped from the pod manifest, finished clips are never re-run.

Safety: refuses to create pods without a fresh tools/runpod.py watchdog heartbeat whose
caps are at least this run's; --max-usd guard (no new pods/claims near the cap, every
pod pulled+terminated at it); only pod ids recorded in this state file are ever
stopped/terminated; output is scrubbed of .env values (runpod.scrub). All RunPod API
calls go through tools/runpod.py as a subprocess; SSH/rsync are thin and mirror
tools/pod.sh (-rlpt), adding a per-pod known_hosts file, BatchMode and timeouts.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import fcntl
import glob
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import runpod  # noqa: E402  (secret scrubbing, watchdog heartbeat, GPU names)
from make_manifest import assign_workers  # noqa: E402

FLEET_DIR = ROOT / "results" / "fleet"
RUNPOD_PY = ROOT / "tools" / "runpod.py"
SSH_KEY = Path(os.environ.get("POD_KEY", str(Path.home() / ".ssh" / "id_ed25519")))

# Measured on the L40S, 2026-09-26: ~13.5 GB per bf16 fast-video worker + ~5.5 GB for its
# persistent whisper server. Workers per GPU default to floor(usable VRAM / 19 GB), usable =
# nominal minus a CUDA-context/fragmentation reserve: 48 GB -> 2, 24 GB -> 1, 80 GB -> 4.
WORKER_VRAM_GB = 19
VRAM_RESERVE_GB = 3
# Measured on the same L40S (2 concurrent bf16 workers): ~1.1 s wall per source-second per
# worker (~120 clips/h per worker at a 26 s mean). Other GPUs are ASSUMED to match.
WALL_PER_SOURCE_S = 1.1
# Queue order when --batches is not given: study batches, deep dive, then the rest. Only a
# batch with prep.json is claimable; the others wait (tools/prep_cpu.py may still be running).
DEFAULT_BATCH_ROOT = "results/batches_s384"
DEFAULT_BATCHES = ([f"b{i:02d}" for i in range(2, 10)] + ["d00_deep_dive", "d01_deep_dive"]
                   + [f"r{i:02d}" for i in range(16)])
# A finished batch with no clip done at all (and at least this many clips) means the pod or the
# code is broken: pull logs, terminate, requeue, and create no more pods until a restart.
ZERO_OUTPUT_MIN_CLIPS = 3
VRAM_GB = {
    "NVIDIA GeForce RTX 3090": 24, "NVIDIA GeForce RTX 3090 Ti": 24, "NVIDIA GeForce RTX 4090": 24,
    "NVIDIA L4": 24, "NVIDIA RTX A5000": 24, "NVIDIA RTX 5000 Ada Generation": 32,
    "NVIDIA A40": 48, "NVIDIA RTX A6000": 48, "NVIDIA L40": 48, "NVIDIA L40S": 48,
    "NVIDIA RTX 6000 Ada Generation": 48, "NVIDIA A100-SXM4-40GB": 40,
    "NVIDIA A100 80GB PCIe": 80, "NVIDIA A100-SXM4-80GB": 80, "NVIDIA H100 80GB HBM3": 80,
    "NVIDIA H100 PCIe": 80, "NVIDIA H100 NVL": 94, "NVIDIA H200": 141, "NVIDIA H200 NVL": 141,
}
ACTIVE = ("new", "creating", "wait_ssh", "push_code", "setup_start", "setup", "run", "drain")


class SshError(Exception):
    """Connection-level failure (ssh rc 255 or timeout): the pod may be dead."""


class RemoteError(Exception):
    """The command reached the pod and failed."""


class ApiFailure(Exception):
    pass


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def scrub(text: str) -> str:
    return runpod.scrub(str(text))


# --------------------------------------------------------------------------- config

@dataclass
class Config:
    pods: int = 1
    gpu_type: str = ""
    gpu_count: int = 1
    cloud: str = "COMMUNITY"
    interruptible: bool = False
    dc: str | None = None
    image: str | None = None
    container_disk_gb: int = 100
    volume_gb: int = 20
    min_ram_gb: int | None = None
    pod_max_hours: float = 8.0
    max_usd: float = 0.0
    usd_per_hour: float | None = None
    job_dir: str = "/root/tribe-job"
    workers_per_gpu: int | None = None  # None: floor((VRAM - reserve) / WORKER_VRAM_GB)
    fast_video: bool = True
    video_precision: str = "bf16"
    video_threads: int | None = None
    omp_threads: int | None = None  # None: max(4, floor(pod cgroup CPU quota / workers)), read on the pod
    worker_args: list = field(default_factory=list)
    on_done: str = "terminate"
    poll_s: float = 60
    api_poll_s: float = 120
    pull_every_min: float = 15
    dead_after_min: float = 10
    idle_grace_min: float = 10
    create_retries: int = 5
    create_backoff_s: float = 300
    max_creates: int | None = None
    worker_retries: int = 1
    max_batch_attempts: int = 3
    min_free_gb: float = 15
    reserve_hours: float = 0.25
    wall_per_source_s: float = WALL_PER_SOURCE_S
    setup_min: float = 25.0
    keep_remote_videos: bool = False
    wait_ssh_min: float = 30
    setup_timeout_min: float = 30
    drain_timeout_min: float = 30

    def gpu_vram_gb(self) -> int | None:
        try:
            return VRAM_GB.get(runpod.resolve_gpu(self.gpu_type)) if self.gpu_type else None
        except runpod.UsageError:
            return None

    @property
    def wpg(self) -> int:
        """Workers per GPU: explicit, else from the measured ~19 GB per worker."""
        if self.workers_per_gpu:
            return self.workers_per_gpu
        vram = self.gpu_vram_gb()
        return max(1, (vram - VRAM_RESERVE_GB) // WORKER_VRAM_GB) if vram else 1

    @property
    def num_workers(self) -> int:
        return self.gpu_count * self.wpg

    @property
    def cache_folder(self) -> str:
        # worker.py refuses a non-fp32 precision unless the cache folder name carries it
        return f"{self.job_dir}/feature-cache" + ("" if self.video_precision == "fp32" else f"-{self.video_precision}")

    def validate(self) -> list[str]:
        problems = []
        if self.pods < 1:
            problems.append("--pods must be >= 1")
        if (self.workers_per_gpu is not None and self.workers_per_gpu < 1) or self.gpu_count < 1:
            problems.append("--workers-per-gpu and --gpu-count must be >= 1")
        if self.video_precision != "fp32" and not self.fast_video:
            problems.append("--video-precision needs --fast-video (worker.py refuses otherwise)")
        if not self.job_dir.startswith("/") or any(c in self.job_dir for c in " '\"$`\\"):
            problems.append("--job-dir must be an absolute path without spaces/quotes")
        if self.omp_threads is not None and self.omp_threads < 1:
            problems.append("--omp-threads must be >= 1")
        if self.on_done not in ("terminate", "stop", "keep"):
            problems.append("--on-done must be terminate|stop|keep")
        if self.gpu_type:
            try:
                gid = runpod.resolve_gpu(self.gpu_type)
            except runpod.UsageError as e:
                problems.append(str(e))
            else:
                if runpod.is_blackwell(gid):
                    problems.append(f"{gid} is Blackwell; torch 2.6 in setup.sh does not support it")
                vram = VRAM_GB.get(gid)
                if vram and self.wpg * WORKER_VRAM_GB > vram:
                    problems.append(f"{self.wpg} workers x ~{WORKER_VRAM_GB} GB peak VRAM do not fit "
                                    f"{gid} ({vram} GB); use --workers-per-gpu {max(1, vram // WORKER_VRAM_GB)}")
        return problems

    def worker_flags(self) -> list[str]:
        flags = []
        if self.fast_video:
            flags.append("--fast-video")
            if self.video_precision != "fp32":
                flags += ["--video-precision", self.video_precision]
        if self.video_threads:
            flags += ["--video-threads", str(self.video_threads)]
        return flags + [str(a) for a in self.worker_args]

    def create_args(self, name: str) -> list[str]:
        a = ["pod-create", "--name", name, "--gpu-type", self.gpu_type, "--gpu-count", str(self.gpu_count),
             "--cloud", self.cloud, "--container-disk-gb", str(self.container_disk_gb),
             "--volume-gb", str(self.volume_gb), "--max-hours", f"{self.pod_max_hours:g}"]
        if self.interruptible:
            a.append("--interruptible")
        if self.dc:
            a += ["--dc", self.dc]
        if self.image:
            a += ["--image", self.image]
        if self.min_ram_gb:
            a += ["--min-ram-gb", str(self.min_ram_gb)]
        return a


# --------------------------------------------------------------------------- batches (local)

def read_rows(manifest: Path) -> list[dict]:
    return [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]


def batch_ready(batch_dir: Path) -> bool:
    """tools/prep_cpu.py writes prep.json when a whole (unsharded) batch is pre-scaled."""
    return (batch_dir / "manifest.jsonl").is_file() and (batch_dir / "prep.json").is_file()


def local_done_ids(out_dir: Path) -> set[str]:
    """Clips whose completion marker (<vid>.json, written last by worker.py) and preds are here."""
    ids = set()
    for p in out_dir.glob("worker-*/*.json"):
        if p.name.endswith(".error.json"):
            continue
        vid = p.name[:-5]
        if (p.parent / f"{vid}.npz").is_file():
            ids.add(vid)
    return ids


def build_pod_manifest(batch_dir: Path, done: set[str], num_workers: int) -> tuple[list[dict], dict]:
    """Rows still to run (video present locally, not already pulled), re-sliced for this pod.

    Rewrites both `worker` and `num_workers` (worker.py's load_shard checks the latter
    against NUM_WORKERS) with make_manifest.assign_workers (duration-balanced)."""
    rows = read_rows(batch_dir / "manifest.jsonl")
    todo, skipped_done, missing = [], 0, []
    for r in rows:
        if r["video_id"] in done:
            skipped_done += 1
        elif not (batch_dir / "videos" / r["path"]).is_file():
            missing.append(r["path"])
        else:
            todo.append(dict(r))
    for r, w in zip(todo, assign_workers([float(r["duration_s"]) for r in todo], num_workers)):
        r["worker"] = w
        r["num_workers"] = num_workers
    return todo, {"total": len(rows), "done": skipped_done, "missing": missing}


# --------------------------------------------------------------------------- remote scripts

REMOTE_STATUS = r'''
import glob, json, os, shutil
A = json.loads(%s)
job = A["job"]; fl = os.path.join(job, "fleet")
def pid_alive(pidfile, needle):
    try:
        pid = int(open(pidfile).read().split()[0])
        os.kill(pid, 0)
        cmd = open("/proc/%%d/cmdline" %% pid, "rb").read().replace(b"\0", b" ").decode(errors="replace")
    except Exception:
        return False
    return needle in cmd
procs = []
for p in os.listdir("/proc"):
    if p.isdigit():
        try:
            procs.append([x.decode(errors="replace") for x in open("/proc/%%s/cmdline" %% p, "rb").read().split(b"\0")])
        except Exception:
            pass
def worker_running(man, k):
    for a in procs:
        if man in a and any(x.endswith("worker.py") for x in a):
            if any(a[i] == "--worker-id" and a[i + 1] == str(k) for i in range(len(a) - 1)):
                return True
    return False
setup_failed = None
if os.path.exists(os.path.join(job, ".setup_done")):
    setup = "complete"
elif os.path.exists(os.path.join(job, ".setup_failed")):
    setup = "failed"
    setup_failed = open(os.path.join(job, ".setup_failed")).read().strip()[:200]
else:
    try:
        rc = open(os.path.join(fl, "setup.rc")).read().strip()
    except Exception:
        rc = ""
    if rc:
        setup = "failed"
        setup_failed = "rc=" + rc + " (no .setup_failed marker)"
    elif pid_alive(os.path.join(fl, "setup.pid"), "setup.sh"):
        setup = "running"
    else:
        setup = "died" if os.path.exists(os.path.join(fl, "setup.pid")) else "absent"
try:
    free = shutil.disk_usage(job).free / 1e9
except Exception:
    free = None
out = {"setup": setup, "setup_failed": setup_failed, "free_gb": free, "batches": {}}
for b in A["batches"]:
    man = os.path.join(fl, "batches", b + ".jsonl")
    try:
        rows = [json.loads(l) for l in open(man) if l.strip()]
    except Exception:
        rows = None
    done, errs, npz, emb = set(), set(), set(), set()
    for wd in glob.glob(os.path.join(job, "outputs", b, "worker-*")):
        for f in os.listdir(wd):
            if f.endswith(".error.json"): errs.add(f[:-11])
            elif f.endswith(".json"): done.add(f[:-5])
            elif f.endswith(".emb.npz"): emb.add(f[:-8])
            elif f.endswith(".npz"): npz.add(f[:-4])
    done &= npz
    errs -= done
    ids = set(r["video_id"] for r in rows or [])
    workers = {}
    for r in rows or []:
        w = workers.setdefault(str(r["worker"]), {"shard": 0, "done": 0, "errors": 0})
        w["shard"] += 1
        if r["video_id"] in done: w["done"] += 1
        elif r["video_id"] in errs: w["errors"] += 1
    for k, w in workers.items():
        w["alive"] = pid_alive(os.path.join(fl, "run", "%%s.w%%s.pid" %% (b, k)), man) or worker_running(man, int(k))
    info = {"manifest": rows is not None, "outputs": os.path.isdir(os.path.join(job, "outputs", b)),
            "total": len(ids), "done": len(done & ids),
            "errors": len(errs & ids), "npz": len(npz), "emb": len(emb), "workers": workers}
    if b in A.get("ids_for", []):
        info["done_ids"] = sorted(done & ids)
    out["batches"][b] = info
print("FLEETSTATUS " + json.dumps(out))
'''


def job_header(job: str) -> str:
    """Every remote script starts here: JOB must be explicit (pod env does not reach SSH shells)."""
    return f"set -eu\nexport JOB={shlex.quote(job)}\n"


def setup_script(job: str) -> str:
    return job_header(job) + r'''mkdir -p "$JOB/logs" "$JOB/fleet"
if [ -f "$JOB/.setup_done" ]; then echo "FLEET setup complete"; exit 0; fi
if [ -f "$JOB/fleet/setup.pid" ] && kill -0 "$(cat "$JOB/fleet/setup.pid")" 2>/dev/null; then
  echo "FLEET setup running"; exit 0
fi
rm -f "$JOB/fleet/setup.rc"
# own session + nohup + no inherited stdio: survives the SSH disconnect. setup.sh writes
# $JOB/.setup_done or $JOB/.setup_failed; the rc file also catches a death before its trap.
setsid nohup env JOB="$JOB" bash -c 'echo $$ > "$JOB/fleet/setup.pid"; bash "$JOB/code/pod/setup.sh"; echo $? > "$JOB/fleet/setup.rc"' \
  > "$JOB/logs/setup.out" 2>&1 < /dev/null &
echo "FLEET setup started"
'''


def threads_script(num_workers: int, override: int | None = None) -> str:
    """Per-worker OMP/MKL threads from the pod's cgroup CPU quota: nproc reports the host
    (256 on the L40S) while the quota was 27.2 CPUs. T = max(4, floor(quota / workers))."""
    if override:
        return f"CPUS=override\nT={int(override)}\n"
    return f"""W={int(num_workers)}
CG="${{FLEET_CGROUP_ROOT:-/sys/fs/cgroup}}"
CPUS=0
if [ -r "$CG/cpu.max" ]; then
  Q=max; P=0
  read -r Q P < "$CG/cpu.max" || true
  if [ "$Q" != max ] && [ "${{P:-0}}" -gt 0 ]; then CPUS=$((Q / P)); fi
elif [ -r "$CG/cpu/cpu.cfs_quota_us" ] && [ -r "$CG/cpu/cpu.cfs_period_us" ]; then
  Q=$(cat "$CG/cpu/cpu.cfs_quota_us"); P=$(cat "$CG/cpu/cpu.cfs_period_us")
  if [ "$Q" -gt 0 ] && [ "$P" -gt 0 ]; then CPUS=$((Q / P)); fi
fi
if [ "$CPUS" -lt 1 ]; then CPUS=$(nproc); fi
T=$((CPUS / W))
if [ "$T" -lt 4 ]; then T=4; fi
"""


def launch_script(job: str, batch: str, workers: list[int], num_workers: int, gpu_count: int,
                  cache_folder: str, flags: list[str], omp_threads: int | None = None) -> str:
    man = f"{job}/fleet/batches/{batch}.jsonl"
    lines = [job_header(job), 'mkdir -p "$JOB/fleet/run" "$JOB/logs"',
             threads_script(num_workers, omp_threads).rstrip("\n"),
             'export OMP_NUM_THREADS="$T" MKL_NUM_THREADS="$T"',
             'export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"',
             # worker.py's frame-loop default is min(8, os.cpu_count()), and cpu_count lies too
             'export TRIBE_VIDEO_THREADS="${TRIBE_VIDEO_THREADS:-$(( T < 8 ? T : 8 ))}"']
    for k in workers:
        pidfile = f"{job}/fleet/run/{batch}.w{k}.pid"
        args = ["--manifest", man, "--videos-root", f"{job}/videos/{batch}",
                "--out-root", f"{job}/outputs/{batch}", "--cache-folder", cache_folder, *flags]
        # the child records its own pid, then execs run_worker.sh (same pid) -> pid file is exact
        lines.append(
            f"WORKER_ID={k} NUM_WORKERS={num_workers} GPU={k % gpu_count} setsid nohup bash -c "
            + shlex.quote('echo $$ > "$0"; exec bash "$JOB/code/pod/run_worker.sh" "$@"')
            + " " + " ".join(shlex.quote(x) for x in [pidfile, *args])
            + f" > {shlex.quote(f'{job}/logs/fleet-{batch}-w{k}.out')} 2>&1 < /dev/null &")
    lines.append(f"echo FLEET launched {batch} workers {' '.join(map(str, workers))} cpus $CPUS threads $T")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- real ops

def run_cmd(argv: list[str], input: str | None = None, timeout: float | None = None,
            env: dict | None = None) -> tuple[int, str, str]:
    try:
        p = subprocess.run(argv, input=input, capture_output=True, text=True, timeout=timeout,
                           env={**os.environ, **env} if env else None)
    except subprocess.TimeoutExpired:
        return 124, "", f"timeout after {timeout}s"
    return p.returncode, p.stdout, p.stderr


class RealOps:
    """RunPod API via tools/runpod.py; SSH/rsync mirroring tools/pod.sh. Tests replace the whole object."""

    def __init__(self, fleet_dir: Path, job: str, runner=run_cmd):
        self.fleet_dir = fleet_dir
        self.job = job
        self._runner = runner

    def run(self, argv, input=None, timeout=None):
        # JOB is exported to every local subprocess as well (tools/pod.sh would default to
        # /workspace/tribe-job); remote scripts set it themselves (job_header).
        return self._runner(argv, input=input, timeout=timeout, env={"JOB": self.job})

    # ---- API (tools/runpod.py subprocess; its output is already scrubbed)
    def _rp(self, args: list[str], timeout: float = 120) -> str:
        rc, out, err = self.run([sys.executable, str(RUNPOD_PY), *args], timeout=timeout)
        if rc != 0:
            raise ApiFailure(scrub(f"runpod.py {args[0]} rc={rc}: {(err or out).strip()[-600:]}"))
        return out

    def watchdog(self, now: float) -> tuple[bool, str, dict]:
        ok, why = runpod.watchdog_fresh(now)
        try:
            hb = json.loads(runpod.HEARTBEAT_FILE.read_text())
        except (OSError, ValueError):
            hb = {}
        return ok, why, hb

    def create_pod(self, cfg: Config, name: str) -> tuple[str, float | None]:
        out = self._rp(cfg.create_args(name), timeout=180)
        m = re.search(r"created pod (\S+) ", out)
        if not m:
            raise ApiFailure("pod-create: no pod id in output")
        rate = re.search(r"\$([0-9.]+)/h", out)
        return m.group(1), (float(rate.group(1)) if rate else None)

    def list_pods(self) -> dict[str, dict]:
        pods = json.loads(self._rp(["pod-list", "--json"]) or "[]")
        return {p.get("id"): {"name": p.get("name"), "status": p.get("desiredStatus"),
                              "rate": runpod.to_float(p.get("costPerHr")) or None,
                              "terminated": runpod.is_terminated(p)} for p in pods}

    def recorded_rate(self, pod_id: str) -> float | None:
        rec = runpod.load_state()["pods"].get(pod_id) or {}  # read-only
        return rec.get("cost_per_hr")

    def wait_ssh(self, pod_id: str, timeout_min: float) -> dict:
        out = self._rp(["pod-wait", pod_id, "--timeout-min", f"{timeout_min:g}"], timeout=timeout_min * 60 + 120)
        m = re.search(r"export POD=(\S+) POD_PORT=(\d+)", out)
        if not m:
            raise ApiFailure("pod-wait: no endpoint in output")
        return {"pod_id": pod_id, "host": m.group(1), "port": int(m.group(2))}

    def terminate(self, pod_id: str) -> None:
        self._rp(["pod-terminate", pod_id, "--yes"])

    def stop(self, pod_id: str) -> None:
        self._rp(["pod-stop", pod_id])

    # ---- SSH / rsync
    def _ssh_opts(self, ep: dict) -> list[str]:
        kh = self.fleet_dir / "known_hosts" / ep["pod_id"]
        kh.parent.mkdir(parents=True, exist_ok=True)
        return ["-p", str(ep["port"]), "-i", str(SSH_KEY), "-o", "StrictHostKeyChecking=accept-new",
                "-o", f"UserKnownHostsFile={kh}", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
                "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=4", "-o", "LogLevel=ERROR"]

    def _check(self, what: str, rc: int, out: str, err: str) -> str:
        if rc in (124, 255):
            raise SshError(scrub(f"{what}: rc={rc} {err.strip()[-300:]}"))
        if rc != 0:
            raise RemoteError(scrub(f"{what}: rc={rc} {(err or out).strip()[-600:]}"))
        return out

    def ssh_script(self, ep: dict, script: str, timeout: float = 180) -> str:
        rc, out, err = self.run(["ssh", *self._ssh_opts(ep), ep["host"], "bash -s"], input=script, timeout=timeout)
        return self._check("ssh", rc, out, err)

    def rsync(self, ep: dict, args: list[str], timeout: float | None = None) -> None:
        e = " ".join(["ssh", *self._ssh_opts(ep)])
        # -rlpt like pod.sh: RunPod MFS rejects chown. rsync exit 12/30/35 = connection trouble.
        rc, out, err = self.run(["rsync", "-rlpt", "--timeout=300", "--exclude", "*.tmp", "-e", e, *args],
                                timeout=timeout)
        if rc in (12, 30, 35):
            rc = 255
        self._check("rsync", rc, out, err)

    def remote(self, ep: dict, path: str) -> str:
        return f"{ep['host']}:{path}"

    def push_code(self, ep: dict, job: str) -> None:
        self.ssh_script(ep, job_header(job) + 'command -v rsync >/dev/null || '
                        '{ apt-get update -qq && apt-get install -y -qq rsync >/dev/null; }\nmkdir -p "$JOB/code"\n',
                        timeout=600)
        self.rsync(ep, ["--delete", "--exclude", "__pycache__", "--exclude", "*.pyc",
                        str(ROOT / "pod"), str(ROOT / "tools"), str(ROOT / "tribe_research"),
                        self.remote(ep, f"{job}/code/")], timeout=1800)

    def start_setup(self, ep: dict, job: str) -> str:
        return self.ssh_script(ep, setup_script(job))

    def push_batch(self, ep: dict, job: str, batch: str, videos_dir: Path, files_list: Path, manifest: Path) -> None:
        self.ssh_script(ep, job_header(job) + f'mkdir -p "$JOB/videos/{batch}" "$JOB/fleet/batches"\n')
        self.rsync(ep, [f"--files-from={files_list}", f"{videos_dir}/", self.remote(ep, f"{job}/videos/{batch}/")])
        # the manifest goes last: its presence on the pod means the clips are there
        self.rsync(ep, [str(manifest), self.remote(ep, f"{job}/fleet/batches/{batch}.jsonl")], timeout=600)

    def launch(self, ep: dict, job: str, batch: str, workers: list[int], cfg: Config) -> str:
        return self.ssh_script(ep, launch_script(job, batch, workers, cfg.num_workers, cfg.gpu_count,
                                                 cfg.cache_folder, cfg.worker_flags(), cfg.omp_threads))

    def status(self, ep: dict, job: str, batches: list[str], ids_for: list[str] = ()) -> dict:
        arg = json.dumps({"job": job, "batches": batches, "ids_for": list(ids_for)})
        script = REMOTE_STATUS % repr(arg)
        rc, out, err = self.run(["ssh", *self._ssh_opts(ep), ep["host"], "python3 -"], input=script, timeout=120)
        out = self._check("status", rc, out, err)
        for line in out.splitlines():
            if line.startswith("FLEETSTATUS "):
                return json.loads(line[len("FLEETSTATUS "):])
        raise RemoteError("status: no FLEETSTATUS line")

    def pull_batch(self, ep: dict, job: str, batch: str, dest: Path) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        self.rsync(ep, [self.remote(ep, f"{job}/outputs/{batch}/"), f"{dest}/"], timeout=3600)

    def pull_logs(self, ep: dict, job: str, dest: Path, setup_only: bool = False) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        if setup_only:  # diagnosis before terminating a failed/slow setup
            for f in ("setup-timings.txt", "prefetch.log", "setup.out"):
                with contextlib.suppress(RemoteError):
                    self.rsync(ep, [self.remote(ep, f"{job}/logs/{f}"), f"{dest}/"], timeout=300)
            with contextlib.suppress(RemoteError):
                self.rsync(ep, [self.remote(ep, f"{job}/.setup_failed"), f"{dest}/setup_failed.txt"], timeout=120)
            return
        self.rsync(ep, [self.remote(ep, f"{job}/logs/"), f"{dest}/"], timeout=1800)

    def cleanup_batch(self, ep: dict, job: str, batch: str) -> None:
        self.ssh_script(ep, job_header(job) + f'rm -rf "$JOB/videos/{batch}"\n')


# --------------------------------------------------------------------------- state

class Store:
    """results/fleet/state.json: flock on state.json.lock, atomic replace."""

    def __init__(self, path: Path):
        self.path = path
        self.lock_path = path.with_name(path.name + ".lock")

    @contextlib.contextmanager
    def _flock(self, mode):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.lock_path, "a") as fh:
            fcntl.flock(fh, mode)
            try:
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    def load(self) -> dict | None:
        if not self.path.exists():
            return None
        with self._flock(fcntl.LOCK_SH):
            return json.loads(self.path.read_text())

    def save(self, state: dict) -> None:
        with self._flock(fcntl.LOCK_EX):
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(state, indent=1, sort_keys=True) + "\n")
            os.replace(tmp, self.path)


def new_state(now: float) -> dict:
    return {"version": 1, "run_id": datetime.fromtimestamp(now, timezone.utc).strftime("%m%d%H%M"),
            "created_at": now, "order": [], "batches": {}, "slots": {}, "pods": {}, "creates": 0,
            "sources": [], "excluded": [], "halt": None}


class Inbox:
    """results/fleet/inbox.jsonl: `add`/`exclude` requests from other processes. The running
    driver owns state.json (it rewrites it from memory), so they queue here and the driver
    merges them every tick; without a driver they are merged at the next start."""

    def __init__(self, fleet_dir: Path):
        self.path = fleet_dir / "inbox.jsonl"
        self.lock_path = fleet_dir / "inbox.lock"

    @contextlib.contextmanager
    def _flock(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.lock_path, "a") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    def put(self, op: str, items: list[str]) -> None:
        with self._flock(), open(self.path, "a") as f:
            f.write(json.dumps({"op": op, "items": items}) + "\n")

    def peek(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self._flock():
            return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def take(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self._flock():
            reqs = [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]
            self.path.unlink()
        return reqs


# --------------------------------------------------------------------------- fleet

class Fleet:
    def __init__(self, cfg: Config, ops, fleet_dir: Path = FLEET_DIR, now=time.time, sleep=None,
                 echo: bool = True):
        self.cfg = cfg
        self.ops = ops
        self.dir = fleet_dir
        self.now = now
        self.stop_event = threading.Event()
        self.sleep = sleep or (lambda s: self.stop_event.wait(s))
        self.echo = echo
        self.lock = threading.RLock()
        self.store = Store(fleet_dir / "state.json")
        self.state = self.store.load() or new_state(now())
        self.api: dict[str, dict] = {}
        self.api_at: float | None = None

    # ---- bookkeeping
    def log(self, event: str, **kv) -> None:
        line = scrub(f"{iso(self.now())} {event} " + " ".join(f"{k}={v}" for k, v in kv.items())).rstrip()
        with self.lock:
            self.dir.mkdir(parents=True, exist_ok=True)
            with open(self.dir / "fleet.log", "a") as f:
                f.write(line + "\n")
        if self.echo:
            print(line, flush=True)

    def save(self) -> None:
        with self.lock:
            self.state["saved_at"] = self.now()
            self.store.save(self.state)

    def out_dir(self, batch: str) -> Path:
        return self.dir / "outputs" / batch

    # ---- queue
    def add_batches(self, dirs: list[Path]) -> list[str]:
        added = []
        with self.lock:
            for d in dirs:
                name = d.name
                if name in self.state["batches"] or name in self.state.setdefault("excluded", []):
                    continue
                self.state["batches"][name] = {"dir": str(d), "status": "waiting_prep", "attempts": 0,
                                               "total": None, "done": 0, "failed": 0, "slot": None, "pod_id": None}
                self.state["order"].append(name)
                added.append(name)
            self.refresh_ready()
            self.save()
        return added

    def exclude(self, names: list[str]) -> None:
        """Skip batches (e.g. b02-b09 already run by a manually driven pod). A batch this fleet
        is already running finishes; a later requeue of it is skipped."""
        with self.lock:
            for name in names:
                if name not in self.state.setdefault("excluded", []):
                    self.state["excluded"].append(name)
                b = self.state["batches"].get(name)
                if b and b["status"] in ("queued", "waiting_prep", "failed"):
                    b["prev_status"] = b["status"]
                    b["status"] = "excluded"
                    self.log("batch_excluded", batch=name)
                elif b and b["status"] == "claimed":
                    self.log("batch_exclude_deferred", batch=name, pod=b.get("pod_id"),
                             note="already running on a fleet pod; it finishes, a requeue is skipped")
                elif not b:
                    self.log("batch_excluded", batch=name, note="not in the queue")
            self.save()

    def include(self, items: list[str]) -> None:
        """Undo an exclusion (by name), or add batch dirs/globs to the end of the queue."""
        with self.lock:
            dirs = []
            for item in items:
                if item in self.state.setdefault("excluded", []):
                    self.state["excluded"].remove(item)
                    b = self.state["batches"].get(item)
                    if b and b["status"] == "excluded":
                        b["status"] = "queued" if b.get("prev_status") == "queued" else "waiting_prep"
                        self.log("batch_included", batch=item)
                    if b:
                        continue
                if "/" in item or any(c in item for c in "*?["):
                    dirs += resolve_batches([item])
                else:  # a bare name: the default batch root
                    dirs.append(ROOT / DEFAULT_BATCH_ROOT / item)
            added = self.add_batches(dirs)
            if added:
                self.log("batches_added", n=len(added), names=",".join(added))
            self.save()

    def merge_inbox(self) -> None:
        for req in Inbox(self.dir).take():
            if req.get("op") == "exclude":
                self.exclude(req["items"])
            elif req.get("op") == "add":
                self.include(req["items"])
                with self.lock:
                    for it in req["items"]:
                        if any(c in it for c in "*?[") and it not in self.state.setdefault("sources", []):
                            self.state["sources"].append(it)  # rescanned every tick

    def rescan(self) -> None:
        """Batch dirs that appeared since the start under a --batches glob join the queue."""
        srcs = self.state.get("sources") or []
        if srcs:
            added = self.add_batches(resolve_batches(srcs))
            if added:
                self.log("batches_added", n=len(added), names=",".join(added), note="rescan")

    def refresh_ready(self) -> None:
        with self.lock:
            for name in self.state["order"]:
                b = self.state["batches"][name]
                if b["status"] == "waiting_prep" and batch_ready(Path(b["dir"])):
                    b["status"] = "queued"
                    b["total"] = len(read_rows(Path(b["dir"]) / "manifest.jsonl"))
                    self.log("batch_ready", batch=name, clips=b["total"])

    def count(self, status: str) -> int:
        return sum(1 for b in self.state["batches"].values() if b["status"] == status)

    def claim(self, slot: str) -> tuple[str, list[dict], dict] | None:
        """First queued batch -> this slot, with the rows the pod still has to run."""
        with self.lock:
            self.refresh_ready()
            pod_id = self.state["slots"][slot].get("pod_id")
            for name in self.state["order"]:
                b = self.state["batches"][name]
                if b["status"] != "queued":
                    continue
                rows, info = build_pod_manifest(Path(b["dir"]), local_done_ids(self.out_dir(name)),
                                                self.cfg.num_workers)
                b["done"] = info["done"]
                if info["missing"]:
                    self.log("batch_missing_videos", batch=name, n=len(info["missing"]))
                if not rows:
                    b["status"] = "done"
                    b["done_at"] = self.now()
                    self.log("batch_done", batch=name, note="nothing left to run", done=info["done"],
                             missing=len(info["missing"]))
                    continue
                b.update(status="claimed", slot=slot, pod_id=pod_id, claimed_at=self.now())
                self.save()
                self.log("batch_claimed", batch=name, slot=slot, pod=pod_id, clips=len(rows),
                         already_done=info["done"])
                return name, rows, info
            self.save()
        return None

    def release(self, name: str, reason: str) -> None:
        with self.lock:
            b = self.state["batches"][name]
            if b["status"] != "claimed":
                return
            b["attempts"] += 1
            b.update(slot=None, pod_id=None)
            b["status"] = "failed" if b["attempts"] >= self.cfg.max_batch_attempts else "queued"
            if name in (self.state.get("excluded") or []):
                b["prev_status"], b["status"] = b["status"], "excluded"
            self.save()
        self.log({"queued": "batch_requeued", "failed": "batch_failed"}.get(b["status"], "batch_released"),
                 batch=name, reason=reason, attempts=b["attempts"], status=b["status"])

    def complete(self, name: str) -> None:
        """After a verified final pull: done, or back to the queue if clips are neither done nor failed."""
        with self.lock:
            b = self.state["batches"][name]
            bdir = Path(b["dir"])
            ids = {r["video_id"] for r in read_rows(bdir / "manifest.jsonl") if (bdir / "videos" / r["path"]).is_file()}
            out = self.out_dir(name)
            done = local_done_ids(out) & ids
            errs = {p.name[:-11] for p in out.glob("worker-*/*.error.json")} & ids - done
            missing = len(ids) - len(done) - len(errs)
            b["done"] = len(done)
            if missing > 0:
                self.release(name, f"{missing} clips neither done nor failed (workers died)")
                return
            b.update(status="done", failed=len(errs), done_at=self.now(), slot=None)
            self.save()
        self.log("batch_done", batch=name, done=len(done), failed=len(errs), total=len(ids))

    # ---- money
    def pod_rate(self, rec: dict) -> float:
        return rec.get("rate") or self.cfg.usd_per_hour or 0.0

    def spend(self) -> tuple[float, float]:
        """(estimated USD so far, current burn $/h) over every pod this fleet created."""
        now = self.now()
        usd = burn = 0.0
        with self.lock:
            for rec in self.state["pods"].values():
                end = rec.get("ended_at") or now
                usd += self.pod_rate(rec) * max(0.0, end - rec["created_at"]) / 3600
                if not rec.get("ended_at"):
                    burn += self.pod_rate(rec)
        return usd, burn

    def budget(self) -> str:
        """ok | soft (no new pods/claims; idle and setup pods end) | hard (pull and end every pod)."""
        usd, burn = self.spend()
        if self.cfg.max_usd and usd >= self.cfg.max_usd:
            return "hard"
        projected = usd + max(burn, self.cfg.usd_per_hour or 0.0) * self.cfg.reserve_hours
        if self.cfg.max_usd and projected >= self.cfg.max_usd:
            return "soft"
        return "ok"

    # ---- pods
    def owned(self, pod_id: str | None) -> bool:
        return bool(pod_id) and pod_id in self.state["pods"]

    def halt(self, reason: str) -> None:
        """Deterministic failure (setup.sh failed, a batch produced nothing): no new pods until
        the driver is restarted. Pods already working carry on."""
        with self.lock:
            if not self.state.get("halt"):
                self.state["halt"] = {"reason": reason, "at": self.now()}
                self.save()
        self.log("fleet_halt", reason=reason, note="no new pods; fix, then restart the driver")

    def end_pending(self) -> list[str]:
        return [pid for pid, r in self.state["pods"].items() if r.get("end_pending") and not r.get("ended_at")]

    def end_pod(self, slot: str, action: str, reason: str) -> None:
        s = self.state["slots"][slot]
        pod_id = s.get("pod_id")
        ended = True
        if not self.owned(pod_id):
            self.log("refuse_end_unowned_pod", slot=slot, pod=pod_id)
        elif action in ("terminate", "stop"):
            try:
                (self.ops.terminate if action == "terminate" else self.ops.stop)(pod_id)
                self.log(f"pod_{action}", slot=slot, pod=pod_id, reason=reason)
            except Exception as e:
                # still billed: keep it in the spend, retry from api_tick until pod-list confirms
                ended = False
                self.log(f"pod_{action}_failed", slot=slot, pod=pod_id, error=e,
                         note="retried every api poll until pod-list shows it gone")
        else:
            self.log("pod_kept", slot=slot, pod=pod_id, reason=reason, note="still billed")
        with self.lock:
            if self.owned(pod_id):
                rec = self.state["pods"][pod_id]
                rec["end_reason"] = reason
                if ended:
                    rec["ended_at"] = self.now()
                    rec.pop("end_pending", None)
                else:
                    rec["end_pending"] = action
            for key in ("current", "next"):
                if s.get(key):
                    self.release(s[key]["batch"], f"pod ended: {reason}")
                    s[key] = None
            for name in s.get("pending_pulls") or []:  # finished on the pod but never pulled+verified
                self.release(name, f"pod ended before its pull was verified: {reason}")
            s.update(pod_id=None, ep=None, pending_pulls=[], ssh_fail_since=None, idle_since=None)
            self.save()

    def slot_clips_per_hour(self, s: dict) -> float | None:
        if not s.get("first_launch_at") or not s.get("clips_done"):
            return None
        hours = (self.now() - s["first_launch_at"]) / 3600
        return s["clips_done"] / hours if hours > 0 else None

    # ---- the per-slot state machine: one transition per call, returns seconds to wait
    def step(self, slot: str) -> float:
        s = self.state["slots"][slot]
        phase = s["phase"]
        try:
            return getattr(self, f"_p_{phase}")(slot, s)
        except SshError as e:
            return self._ssh_trouble(slot, s, e)
        except (RemoteError, ApiFailure, OSError, ValueError) as e:
            self.log("step_error", slot=slot, phase=phase, error=e)
            return self.cfg.poll_s

    def _ssh_trouble(self, slot: str, s: dict, e: Exception) -> float:
        now = self.now()
        with self.lock:
            s["ssh_fail_since"] = s.get("ssh_fail_since") or now
            self.save()
        down_min = (now - s["ssh_fail_since"]) / 60
        self.log("ssh_fail", slot=slot, pod=s.get("pod_id"), down_min=f"{down_min:.1f}", error=e)
        if down_min >= self.cfg.dead_after_min:
            self._dead(slot, s, f"unreachable for {down_min:.0f} min")
        return self.cfg.poll_s

    def _ssh_ok(self, s: dict) -> None:
        if s.get("ssh_fail_since"):
            with self.lock:
                s["ssh_fail_since"] = None

    def _dead(self, slot: str, s: dict, reason: str) -> None:
        """Batches back to the queue; a still-listed pod is terminated (its container disk is useless now)."""
        self.log("pod_dead", slot=slot, pod=s.get("pod_id"), reason=reason)
        api = self.api.get(s.get("pod_id") or "")
        gone = api is None and self.api_at is not None or (api or {}).get("terminated")
        self.end_pod(slot, "keep" if gone else "terminate", f"dead: {reason}")
        with self.lock:
            s["phase"] = "new"
            self.save()

    def _api_verdict(self, s: dict) -> str | None:
        """Pod gone or stopped per the last pod-list (e.g. watchdog action, spot pre-emption)."""
        pod_id = s.get("pod_id")
        if not pod_id or self.api_at is None:
            return None
        rec = self.state["pods"].get(pod_id) or {}
        info = self.api.get(pod_id)
        if info is None:
            return "not in pod-list" if self.api_at - rec.get("created_at", 0) > 300 else None
        if info.get("terminated"):
            return "terminated"
        if info.get("status") == "EXITED":
            return "stopped (container disk erased)"
        return None

    def _age_h(self, s: dict) -> float:
        rec = self.state["pods"].get(s.get("pod_id") or "") or {}
        return (self.now() - rec.get("created_at", self.now())) / 3600

    def _can_claim(self, s: dict, st: dict | None = None) -> bool:
        if self.budget() != "ok":
            return False
        if self._age_h(s) >= self.cfg.pod_max_hours - 0.75:
            return False  # a new batch would not finish before the watchdog's per-pod cap
        free = (st or {}).get("free_gb")
        return not (free is not None and free < self.cfg.min_free_gb)

    def _claim_and_push(self, slot: str, s: dict, key: str) -> bool:
        got = self.claim(slot)
        if not got:
            return False
        name, rows, _ = got
        b = self.state["batches"][name]
        mdir = self.dir / "manifests"
        mdir.mkdir(parents=True, exist_ok=True)
        manifest = mdir / f"{name}.{s['pod_id']}.jsonl"
        files = mdir / f"{name}.{s['pod_id']}.files"
        manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
        files.write_text("".join(r["path"] + "\n" for r in rows))
        rec = {"batch": name, "n": len(rows), "pushed": False, "launched": False, "launches": {},
               "workers": sorted({r["worker"] for r in rows}), "remote_done": 0, "remote_errors": 0,
               "claimed_at": self.now(), "last_pull": None}
        with self.lock:
            s[key] = rec
            self.save()
        t0 = self.now()
        self.log("push_start", slot=slot, pod=s["pod_id"], batch=name, clips=len(rows), role=key)
        self.ops.push_batch(s["ep"], self.cfg.job_dir, name, Path(b["dir"]) / "videos", files, manifest)
        with self.lock:
            rec["pushed"] = True
            self.save()
        self.log("push_done", slot=slot, pod=s["pod_id"], batch=name, s=f"{self.now() - t0:.0f}")
        return True

    def _ensure_pushed(self, slot: str, s: dict, key: str) -> None:
        rec = s.get(key)
        if rec and not rec["pushed"]:  # driver restarted mid-push: rsync is resumable
            b = self.state["batches"][rec["batch"]]
            mdir = self.dir / "manifests"
            self.ops.push_batch(s["ep"], self.cfg.job_dir, rec["batch"], Path(b["dir"]) / "videos",
                                mdir / f"{rec['batch']}.{s['pod_id']}.files",
                                mdir / f"{rec['batch']}.{s['pod_id']}.jsonl")
            with self.lock:
                rec["pushed"] = True
                self.save()

    def _launch(self, slot: str, s: dict, rec: dict, workers: list[int]) -> None:
        out = self.ops.launch(s["ep"], self.cfg.job_dir, rec["batch"], workers, self.cfg) or ""
        m = re.search(r"cpus (\S+) threads (\d+)", out)
        now = self.now()
        with self.lock:
            for k in workers:
                rec["launches"][str(k)] = rec["launches"].get(str(k), 0) + 1
            rec["last_launch_at"] = now
            if not rec["launched"]:
                rec["launched"] = True
                rec["launched_at"] = now
            s["first_launch_at"] = s.get("first_launch_at") or now
            s["idle_since"] = None
            self.save()
        self.log("launch", slot=slot, pod=s["pod_id"], batch=rec["batch"], workers=",".join(map(str, workers)),
                 num_workers=self.cfg.num_workers, **({"cpus": m.group(1), "threads": m.group(2)} if m else {}))

    def _pull(self, slot: str, s: dict, name: str, final: bool) -> bool:
        """Pull a batch's outputs; when final, verify every clip done on the pod is here."""
        dest = self.out_dir(name)
        if not final:
            self.ops.pull_batch(s["ep"], self.cfg.job_dir, name, dest)
            self.log("pull_partial", slot=slot, batch=name, local_done=len(local_done_ids(dest)))
            return True
        st = self.ops.status(s["ep"], self.cfg.job_dir, [name], ids_for=[name])
        info = st["batches"].get(name) or {}
        if info.get("outputs") is False:  # no output dir at all: rsync would fail (rc 23) forever
            self.log("pull_verified", slot=slot, batch=name, clips=0, note="no outputs on the pod")
            return True
        self.ops.pull_batch(s["ep"], self.cfg.job_dir, name, dest)
        remote_ids = set(info.get("done_ids") or [])
        missing = remote_ids - local_done_ids(dest)
        if missing:
            self.log("pull_verify_failed", slot=slot, batch=name, remote_done=len(remote_ids), missing=len(missing))
            return False
        self.log("pull_verified", slot=slot, batch=name, clips=len(remote_ids))
        return True

    # phases ---------------------------------------------------------------------------
    def _p_ended(self, slot, s):
        return 0

    def _p_new(self, slot, s):
        cfg = self.cfg
        with self.lock:
            self.refresh_ready()
            queued, waiting = self.count("queued"), self.count("waiting_prep")
        max_creates = cfg.max_creates if cfg.max_creates is not None else 2 * cfg.pods
        why_end = None
        if self.state.get("halt"):
            why_end = f"fleet halted: {self.state['halt']['reason']}"
        elif int(slot[1:]) >= cfg.pods:
            why_end = f"--pods {cfg.pods}"
        elif self.budget() != "ok":
            why_end = f"budget {self.budget()}"
        elif not queued and not waiting:
            why_end = "no work left"
        elif self.state["creates"] >= max_creates:
            why_end = f"max creates {max_creates} reached"
        elif s.get("create_failures", 0) >= cfg.create_retries:
            why_end = f"{s['create_failures']} create failures"
        if why_end:
            with self.lock:
                s["phase"] = "ended"
                self.save()
            self.log("slot_ended", slot=slot, reason=why_end)
            return 0
        if not queued:
            return cfg.poll_s  # prep still running: do not pay for a pod that would wait
        if s.get("retry_at") and self.now() < s["retry_at"]:
            return min(cfg.poll_s, s["retry_at"] - self.now())
        ok, why, _ = self.ops.watchdog(self.now())
        if not ok:
            self.log("create_blocked", slot=slot, reason=why)
            return cfg.poll_s
        with self.lock:
            s["attempt"] = s.get("attempt", 0) + 1
            s["pod_name"] = f"tribe-fleet-{self.state['run_id']}-{slot}-{s['attempt']}"
            s["phase"] = "creating"
            s["create_started"] = self.now()
            self.save()
        created_at = self.now()
        try:
            pod_id, rate = self.ops.create_pod(cfg, s["pod_name"])
        except ApiFailure as e:
            # never blind-retry a create: a timeout may still have made a billed pod
            self.log("create_failed", slot=slot, name=s["pod_name"], error=e)
            return self._p_creating(slot, s, failed=True)
        self._adopt(slot, s, pod_id, rate, created_at)
        return 0

    def _adopt(self, slot, s, pod_id, rate, created_at):
        with self.lock:
            rate = rate or self.ops.recorded_rate(pod_id)
            self.state["pods"][pod_id] = {"name": s["pod_name"], "slot": slot, "created_at": created_at,
                                          "rate": rate, "ended_at": None}
            self.state["creates"] += 1
            # every per-pod field starts fresh: a previous pod's drain/ssh/idle timers must not leak
            s.update(pod_id=pod_id, phase="wait_ssh", create_failures=0, retry_at=None, clips_done=0, clips_prev=0,
                     first_launch_at=None, current=None, next=None, pending_pulls=[], drain_since=None,
                     ssh_fail_since=None, idle_since=None, setup_started=None, setup_done=None)
            self.save()
        self.log("pod_created", slot=slot, pod=pod_id, name=s["pod_name"], rate=rate or "?")

    def _p_creating(self, slot, s, failed=False):
        """Outcome of a create unknown (failure or driver crash): adopt an exact-name pod, else retry later."""
        try:
            pods = self.ops.list_pods()
        except ApiFailure as e:
            self.log("pod_list_failed", slot=slot, error=e)
            return self.cfg.poll_s
        match = [pid for pid, p in pods.items() if p.get("name") == s["pod_name"] and not p.get("terminated")]
        if match:
            self._adopt(slot, s, match[0], pods[match[0]].get("rate"), s.get("create_started") or self.now())
            return 0
        with self.lock:
            s["phase"] = "new"
            s["create_failures"] = s.get("create_failures", 0) + 1
            s["retry_at"] = self.now() + self.cfg.create_backoff_s
            self.save()
        return self.cfg.poll_s

    def _p_wait_ssh(self, slot, s):
        try:
            ep = self.ops.wait_ssh(s["pod_id"], self.cfg.wait_ssh_min)
        except ApiFailure as e:
            self._dead(slot, s, f"no SSH: {e}")
            return 0
        with self.lock:
            s.update(ep=ep, phase="push_code")
            self.save()
        self.log("pod_ssh", slot=slot, pod=s["pod_id"], host=ep["host"], port=ep["port"])
        return 0

    def _p_push_code(self, slot, s):
        # installs rsync BEFORE setup.sh starts, so the two never race for the apt lock
        self.ops.push_code(s["ep"], self.cfg.job_dir)
        self._ssh_ok(s)
        rev = git_rev()
        with self.lock:
            s["phase"] = "setup_start"
            s["code_rev"] = rev
            self.save()
        self.log("code_pushed", slot=slot, pod=s["pod_id"], rev=s["code_rev"])
        return 0

    def _p_setup_start(self, slot, s):
        out = self.ops.start_setup(s["ep"], self.cfg.job_dir)
        with self.lock:
            s["phase"] = "setup"
            s["setup_started"] = self.now()
            self.save()
        self.log("setup_started", slot=slot, pod=s["pod_id"], remote=out.strip().splitlines()[-1:] or "")
        return 0

    def _p_setup(self, slot, s):
        if self._check_pod(slot, s):
            return 0
        if not s.get("current") and self._can_claim(s):
            if self._claim_and_push(slot, s, "current"):  # overlaps setup.sh on the pod
                return 0
            with self.lock:
                self.refresh_ready()
                left = self.count("queued") + self.count("waiting_prep")
            if not left:  # other pods took the rest: do not pay for a setup with nothing to run
                self.end_pod(slot, "terminate", "no work left for this pod")
                with self.lock:
                    s["phase"] = "new"
                    self.save()
                return 0
        self._ensure_pushed(slot, s, "current")
        st = self.ops.status(s["ep"], self.cfg.job_dir, [])
        self._ssh_ok(s)
        if st["setup"] == "complete":
            with self.lock:
                s["phase"] = "run"
                s["setup_done"] = self.now()
                self.save()
            self.log("setup_done", slot=slot, pod=s["pod_id"],
                     min=f"{(self.now() - s.get('setup_started', self.now())) / 60:.1f}")
            return 0
        took_min = (self.now() - s.get("setup_started", self.now())) / 60
        why = None
        if st["setup"] in ("failed", "died", "absent"):
            why = f"setup {st['setup']}" + (f" ({st.get('setup_failed')})" if st.get("setup_failed") else "")
        elif took_min >= self.cfg.setup_timeout_min:
            why = f"setup timeout after {took_min:.0f} min"
        if why:
            # diagnosis first (setup-timings.txt, prefetch.log, setup.out, .setup_failed), then terminate
            with contextlib.suppress(Exception):
                self.ops.pull_logs(s["ep"], self.cfg.job_dir,
                                   self.dir / "logs" / (s.get("pod_name") or s["pod_id"]), setup_only=True)
            self.log("setup_failed", slot=slot, pod=s["pod_id"], reason=why)
            if st["setup"] == "failed":  # setup.sh's own failure marker: a replacement would fail too
                self.halt(f"setup.sh failed on {s['pod_id']}: {st.get('setup_failed') or ''}".strip())
            self._dead(slot, s, why)
            return 0
        return self.cfg.poll_s

    def _check_pod(self, slot, s) -> bool:
        """Budget / age / API checks shared by setup and run. True if the pod was ended."""
        verdict = self._api_verdict(s)
        if verdict:
            self._dead(slot, s, verdict)
            return True
        budget = self.budget()
        age_cap = self._age_h(s) >= self.cfg.pod_max_hours - 0.2
        running = bool(s.get("current") and s["current"].get("launched"))
        if budget == "hard" or age_cap or (budget == "soft" and not running):
            reason = "max-usd reached" if budget == "hard" else (
                "pod-max-hours near" if age_cap else "budget reserve reached, no running batch")
            for key in ("current", "next"):
                rec = s.get(key)
                if rec and rec.get("launched") and (rec.get("remote_done") or rec.get("remote_errors")):
                    with contextlib.suppress(Exception):
                        self._pull(slot, s, rec["batch"], final=False)
            with contextlib.suppress(Exception):
                self._flush_pulls(slot, s)
            self.end_pod(slot, "terminate", reason)
            with self.lock:
                s["phase"] = "new" if age_cap and budget == "ok" else "ended"
                self.save()
            return True
        return False

    def _flush_pulls(self, slot, s) -> bool:
        """Retry finished-batch pulls that failed verification. True when none are pending."""
        for name in list(s.get("pending_pulls") or []):
            try:
                ok = self._pull(slot, s, name, final=True)
            except RemoteError as e:  # SshError propagates: step() counts it towards "dead"
                self.log("pull_failed", slot=slot, batch=name, error=e)
                ok = False
            if ok:
                self.complete(name)
                with self.lock:
                    s["pending_pulls"].remove(name)
                    self.save()
                if not self.cfg.keep_remote_videos:
                    with contextlib.suppress(Exception):
                        self.ops.cleanup_batch(s["ep"], self.cfg.job_dir, name)
        return not s.get("pending_pulls")

    def _p_run(self, slot, s):
        cfg = self.cfg
        if self._check_pod(slot, s):
            return 0
        cur = s.get("current")
        if cur and not cur["pushed"]:
            self._ensure_pushed(slot, s, "current")
        if cur and not cur["launched"]:
            self._launch(slot, s, cur, cur["workers"])
            return cfg.poll_s  # give the workers time to write their pid files
        if s.get("next") and not s["next"]["pushed"]:
            self._ensure_pushed(slot, s, "next")
        names = [r["batch"] for r in (cur, s.get("next")) if r]
        st = self.ops.status(s["ep"], cfg.job_dir, names)
        self._ssh_ok(s)
        finished = False
        if cur:
            info = st["batches"].get(cur["batch"]) or {}
            if not info.get("manifest"):  # never count a batch the pod does not have as finished
                self.log("batch_missing_on_pod", slot=slot, batch=cur["batch"], note="re-push and relaunch")
                with self.lock:
                    cur["pushed"] = cur["launched"] = False
                    self.save()
                return 0
            with self.lock:
                cur["remote_done"], cur["remote_errors"] = info.get("done", 0), info.get("errors", 0)
                s["clips_done"] = s.get("clips_prev", 0) + cur["remote_done"]
                self.save()
            relaunch, busy = [], False
            fresh = self.now() - cur.get("last_launch_at", 0) < 90  # pid files may not exist yet
            for k, w in (info.get("workers") or {}).items():
                if w["alive"] or (fresh and w["done"] < w["shard"]):
                    busy = True
                elif w["done"] < w["shard"] and cur["launches"].get(k, 0) <= cfg.worker_retries:
                    relaunch.append(int(k))
            if relaunch:
                self.log("worker_dead", slot=slot, batch=cur["batch"], workers=",".join(map(str, relaunch)))
                self._launch(slot, s, cur, relaunch)
                busy = True
            finished = not busy
        if finished and cur["remote_done"] == 0 and cur["n"] >= ZERO_OUTPUT_MIN_CLIPS:
            # every worker exited (after its retries) without a single clip: broken pod or code
            reason = (f"batch {cur['batch']} finished with 0 of {cur['n']} clips done "
                      f"({cur['remote_errors']} errors)")
            self.log("batch_zero_output", slot=slot, pod=s["pod_id"], batch=cur["batch"], errors=cur["remote_errors"])
            with contextlib.suppress(Exception):
                self.ops.pull_logs(s["ep"], cfg.job_dir, self.dir / "logs" / (s.get("pod_name") or s["pod_id"]))
            self.halt(f"{reason} on {s['pod_id']}")
            self.end_pod(slot, "terminate", reason)  # requeues cur and next
            with self.lock:
                s["phase"] = "new"  # _p_new ends the slot while the fleet is halted
                self.save()
            return 0
        if finished:
            # launch the prefetched batch FIRST so the GPU never waits on the pull
            nxt = s.get("next")
            with self.lock:
                s["clips_prev"] = s.get("clips_prev", 0) + cur["remote_done"]
                s.setdefault("pending_pulls", []).append(cur["batch"])
                s["current"], s["next"] = nxt, None
                self.save()
            self.log("batch_finished_on_pod", slot=slot, pod=s["pod_id"], batch=cur["batch"],
                     done=cur["remote_done"], errors=cur["remote_errors"], of=cur["n"])
            if nxt and nxt["pushed"]:
                self._launch(slot, s, nxt, nxt["workers"])
            self._flush_pulls(slot, s)
            return 0
        if cur and cur.get("last_pull") is None:
            with self.lock:
                cur["last_pull"] = self.now()
        if cur and self.now() - cur["last_pull"] >= cfg.pull_every_min * 60 and (cur["remote_done"] or cur["remote_errors"]):
            self._pull(slot, s, cur["batch"], final=False)
            with self.lock:
                cur["last_pull"] = self.now()
                self.save()
        self._flush_pulls(slot, s)
        if cur and not s.get("next") and self._can_claim(s, st):
            self._claim_and_push(slot, s, "next")  # prefetch while the GPU works on cur
            return 0
        if not cur:
            if s.get("next"):
                with self.lock:
                    s["current"], s["next"] = s["next"], None
                    self.save()
                return 0
            if self._can_claim(s, st) and self._claim_and_push(slot, s, "current"):
                return 0
            now = self.now()
            with self.lock:
                s["idle_since"] = s.get("idle_since") or now
                self.save()
            idle_min = (now - s["idle_since"]) / 60
            with self.lock:
                self.refresh_ready()
                waiting = self.count("waiting_prep") + self.count("queued")
            if not waiting or self.budget() != "ok" or idle_min >= cfg.idle_grace_min:
                with self.lock:
                    s["phase"] = "drain"
                    self.save()
                self.log("pod_idle", slot=slot, pod=s["pod_id"], idle_min=f"{idle_min:.1f}", waiting=waiting)
                return 0
        return cfg.poll_s

    def _p_drain(self, slot, s):
        if self._check_pod(slot, s):  # gone/stopped per the API, max-usd, age cap
            return 0
        with self.lock:
            s["drain_since"] = s.get("drain_since") or self.now()
        if not self._flush_pulls(slot, s):
            drain_min = (self.now() - s["drain_since"]) / 60
            if drain_min < self.cfg.drain_timeout_min:
                return self.cfg.poll_s  # never terminate a pod holding unverified outputs...
            # ... unless verification keeps failing: end_pod requeues those batches
            self.log("drain_unpulled", slot=slot, batches=",".join(s["pending_pulls"]),
                     note=f"not verified after {drain_min:.0f} min; ending anyway, batches requeued")
        with contextlib.suppress(Exception):
            self.ops.pull_logs(s["ep"], self.cfg.job_dir, self.dir / "logs" / (s.get("pod_name") or s["pod_id"]))
        self.end_pod(slot, self.cfg.on_done, "queue empty" if self.budget() == "ok" else "budget")
        with self.lock:
            # "new": if batches are still waiting for prep, _p_new creates a pod once one is ready
            s.update(phase="new", drain_since=None)
            self.save()
        return 0

    # ---- driver loop
    def ensure_slots(self) -> None:
        with self.lock:
            for i in range(self.cfg.pods):
                self.state["slots"].setdefault(f"s{i}", {"phase": "new", "pod_id": None, "current": None,
                                                         "next": None, "pending_pulls": []})
            for s in self.state["slots"].values():
                if s["phase"] == "ended" and not s.get("pod_id") and (self.count("queued") or self.count("waiting_prep")):
                    s.update(phase="new", create_failures=0, retry_at=None)
            self.save()

    def api_tick(self) -> None:
        try:
            pods = self.ops.list_pods()
        except ApiFailure as e:
            self.log("pod_list_failed", error=e)
            return
        with self.lock:
            self.api, self.api_at = pods, self.now()
            for pid, rec in self.state["pods"].items():
                if not rec.get("ended_at") and (pods.get(pid) or {}).get("rate"):
                    rec["rate"] = pods[pid]["rate"]
            self.save()
        for pid in self.end_pending():  # a terminate/stop that failed: retry until pod-list agrees
            rec, info = self.state["pods"][pid], pods.get(pid)
            action = rec["end_pending"]
            gone = (info is None and self.now() - rec.get("created_at", 0) > 300) or (info or {}).get("terminated") \
                or (action == "stop" and (info or {}).get("status") == "EXITED")
            if not gone:
                try:
                    (self.ops.terminate if action == "terminate" else self.ops.stop)(pid)
                except Exception as e:
                    self.log(f"pod_{action}_retry_failed", pod=pid, error=e)
                    continue
            with self.lock:
                rec["ended_at"] = self.now()
                rec.pop("end_pending", None)
                self.save()
            self.log(f"pod_{action}", pod=pid, reason=rec.get("end_reason"),
                     note="confirmed by pod-list" if gone else "retry succeeded")

    def slot_loop(self, slot: str) -> None:
        while not self.stop_event.is_set() and self.state["slots"][slot]["phase"] != "ended":
            try:
                wait = self.step(slot)
            except Exception as e:  # a slot thread must not die silently
                self.log("slot_error", slot=slot, error=f"{type(e).__name__}: {e}")
                wait = self.cfg.poll_s
            if wait:
                self.sleep(wait)

    def revive_slots(self) -> list[str]:
        """Ended, pod-less slots restart when work shows up (a runtime `add`, prep finished)."""
        cfg = self.cfg
        max_creates = cfg.max_creates if cfg.max_creates is not None else 2 * cfg.pods
        revived = []
        with self.lock:
            self.refresh_ready()
            if (self.state.get("halt") or not self.count("queued") or self.budget() != "ok"
                    or self.state["creates"] >= max_creates):
                return revived
            for i in range(cfg.pods):
                s = self.state["slots"].get(f"s{i}")
                if s and s["phase"] == "ended" and not s.get("pod_id"):
                    s.update(phase="new", create_failures=0, retry_at=None)
                    revived.append(f"s{i}")
            if revived:
                self.save()
        return revived

    def run(self) -> int:
        self.ensure_slots()
        self.log("driver_start", run=self.state["run_id"], pods=self.cfg.pods, max_usd=self.cfg.max_usd,
                 job=self.cfg.job_dir, workers=self.cfg.num_workers)
        self.api_tick()
        threads: dict[str, threading.Thread] = {}

        def start(slot):
            threads[slot] = threading.Thread(target=self.slot_loop, args=(slot,), name=slot, daemon=True)
            threads[slot].start()
        for slot in sorted(self.state["slots"]):
            start(slot)
        last_api = self.now()
        while (any(t.is_alive() for t in threads.values()) or self.end_pending()) and not self.stop_event.is_set():
            self.sleep(min(self.cfg.poll_s, self.cfg.api_poll_s))
            if self.now() - last_api >= self.cfg.api_poll_s:
                self.api_tick()
                last_api = self.now()
            try:
                self.merge_inbox()
                self.rescan()
            except (OSError, ValueError) as e:
                self.log("queue_update_failed", error=e)
            for slot in self.revive_slots():
                if not threads.get(slot) or not threads[slot].is_alive():
                    self.log("slot_revived", slot=slot)
                    start(slot)
            usd, burn = self.spend()
            with self.lock:
                self.state["driver_at"] = self.now()
                self.save()
            self.log("tick", spend=f"${usd:.2f}", burn=f"${burn:.2f}/h", budget=self.budget(),
                     queued=self.count("queued"), claimed=self.count("claimed"), done=self.count("done"),
                     waiting_prep=self.count("waiting_prep"), **({"halt": 1} if self.state.get("halt") else {}))
        for t in threads.values():
            t.join(timeout=5)
        live = [pid for pid, r in self.state["pods"].items() if not r.get("ended_at")]
        if live:
            self.log("driver_exit", live_pods=",".join(live),
                     note="pods still billed; rerun `tools/fleet.py run --go` to resume or terminate them")
        else:
            self.log("driver_exit", note="no fleet pods left")
        return 0


def git_rev() -> str:
    rc, out, _ = run_cmd(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], timeout=10)
    rc2, dirty, _ = run_cmd(["git", "-C", str(ROOT), "status", "--porcelain", "--", "pod", "tools", "tribe_research"],
                            timeout=10)
    return (out.strip() or "?") + ("+dirty" if dirty.strip() else "")


# --------------------------------------------------------------------------- plan / status

def resolve_batches(patterns: list[str]) -> list[Path]:
    """Globs match existing dirs (rescanned while running); a literal path is kept even if it
    does not exist yet (tools/prep_cpu.py creates it) and simply waits for prep.json."""
    dirs = []
    for pat in patterns:
        full = pat if os.path.isabs(pat) else str(ROOT / pat)
        if any(c in pat for c in "*?["):
            hits = [Path(h) for h in sorted(glob.glob(full)) if Path(h).is_dir()]
        else:
            hits = [Path(full)]
        for p in hits:
            if p not in dirs:
                dirs.append(p)
    return dirs


def default_batches() -> list[Path]:
    return [ROOT / DEFAULT_BATCH_ROOT / n for n in DEFAULT_BATCHES]


def plan(cfg: Config, dirs: list[Path], fleet_dir: Path = FLEET_DIR, excluded: list[str] = (),
         skip: dict | None = None) -> str:
    """Dry run: local files only, no API call, no state written. `skip`: batches a resumed
    state already finished or is running ({name: status})."""
    lines = []
    ready_clips = ready_min = 0.0
    n_ready = 0
    waiting, skipped = [], []
    for d in dirs:
        if d.name in excluded or (skip or {}).get(d.name) in ("done", "failed", "excluded"):
            skipped.append(d.name)
            continue
        if not batch_ready(d):
            waiting.append(d.name)
            continue
        rows, info = build_pod_manifest(d, local_done_ids(fleet_dir / "outputs" / d.name), cfg.num_workers)
        n_ready += 1
        ready_clips += len(rows)
        ready_min += sum(float(r["duration_s"]) for r in rows) / 60
        lines.append(f"  {d.name:<16} {len(rows):>5} clips to run ({info['done']} already pulled, "
                     f"{len(info['missing'])} missing locally), {sum(float(r['duration_s']) for r in rows) / 60:.0f} min source")
    # ~wall_per_source_s seconds of wall time per source-second, per concurrent worker
    mean_s = ready_min * 60 / ready_clips if ready_clips else 0.0
    per_worker = 3600 / (cfg.wall_per_source_s * mean_s) if mean_s else 0.0
    per_pod = cfg.num_workers * per_worker
    fleet_rate = cfg.pods * per_pod
    work_h = ready_min * 60 * cfg.wall_per_source_s / (cfg.num_workers * cfg.pods * 3600) if ready_min else 0.0
    wall_h = cfg.setup_min / 60 + work_h
    vram = cfg.gpu_vram_gb()
    wsrc = ("--workers-per-gpu" if cfg.workers_per_gpu else
            f"auto: ({vram} GB - {VRAM_RESERVE_GB}) / ~{WORKER_VRAM_GB} GB" if vram else "auto: VRAM unknown, 1")
    head = [
        f"fleet plan (dry run: no API call, nothing created)",
        f"pods: {cfg.pods} x {cfg.gpu_count}x {cfg.gpu_type or '?'} {cfg.cloud}{' SPOT' if cfg.interruptible else ''}, "
        f"container disk {cfg.container_disk_gb} GB (JOB={cfg.job_dir}), pod volume {cfg.volume_gb} GB (unused)",
        f"workers: {cfg.wpg}/GPU ({wsrc}) = {cfg.num_workers}/pod; flags: {' '.join(cfg.worker_flags()) or '(stock fp32)'}; "
        f"cache {cfg.cache_folder}",
        f"threads: OMP/MKL = " + (f"{cfg.omp_threads} (--omp-threads)" if cfg.omp_threads else
                                  f"max(4, cgroup CPU quota / {cfg.num_workers}) read on the pod")
        + "; PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        f"batches: {n_ready} ready, {len(waiting)} waiting for prep{(' (' + ' '.join(waiting) + ')') if waiting else ''}"
        + (f", {len(skipped)} skipped ({' '.join(skipped)})" if skipped else ""),
        *lines,
        f"ready work: {ready_clips:.0f} clips, {ready_min / 60:.1f} h source, mean {mean_s:.0f} s",
        f"rate: {cfg.wall_per_source_s:g} s wall per source-second per worker (measured: L40S, 2 bf16 workers; "
        f"other GPUs ASSUMED equal) -> {per_worker:.0f} clips/h per worker, {per_pod:.0f}/pod, {fleet_rate:.0f}/fleet",
    ]
    if not (cfg.fast_video and cfg.video_precision == "bf16"):
        head.append("  note: the measured rate is for --fast-video --video-precision bf16; stock fp32 was ~5x slower")
    head.append(f"wall: ~{cfg.setup_min:g} min setup + {work_h:.1f} h work = {wall_h:.1f} h")
    if wall_h > cfg.pod_max_hours - 0.75:
        head.append(f"  note: > --pod-max-hours {cfg.pod_max_hours:g} minus 45 min: pods stop claiming, are "
                    "replaced near their cap and each replacement pays setup again (raise it or add pods)")
    if cfg.usd_per_hour:
        cost = cfg.pods * cfg.usd_per_hour * wall_h
        head.append(f"cost: ~${cost:.2f} at ${cfg.usd_per_hour:g}/h per pod (estimate); cap --max-usd ${cfg.max_usd:g}")
        if cfg.max_usd and cost > cfg.max_usd:
            afford_h = cfg.max_usd / (cfg.pods * cfg.usd_per_hour) - cfg.setup_min / 60
            head.append(f"  cap binds: ~{max(0.0, afford_h) * fleet_rate:.0f} clips before --max-usd "
                        f"(the rest stays queued for a later run)")
    else:
        head.append("cost: pass --usd-per-hour (see tools/runpod.py gpus) for an estimate")
    head.append("pod-create per slot: tools/runpod.py " + " ".join(shlex.quote(a) for a in cfg.create_args("tribe-fleet-<run>-sN-1")))
    return "\n".join(head)


def status_text(state: dict | None, now: float, cfg_rate: float | None = None, inbox: list[dict] = ()) -> str:
    if not state:
        return "no fleet state (results/fleet/state.json)"
    lines = []
    usd = burn = 0.0
    for rec in state["pods"].values():
        rate = rec.get("rate") or cfg_rate or 0.0
        usd += rate * max(0.0, (rec.get("ended_at") or now) - rec["created_at"]) / 3600
        burn += 0 if rec.get("ended_at") else rate
    counts: dict[str, int] = {}
    for b in state["batches"].values():
        counts[b["status"]] = counts.get(b["status"], 0) + 1
    total_rate = 0.0
    lines.append(f"{'slot':<4} {'pod':<16} {'phase':<11} {'batch':<24} {'next':<14} {'clips/h':>7} {'up h':>5} {'$':>6}")
    for key in sorted(state["slots"]):
        s = state["slots"][key]
        rec = state["pods"].get(s.get("pod_id") or "") or {}
        cur, nxt = s.get("current"), s.get("next")
        prog = f"{cur['batch']} {cur['remote_done']}/{cur['n']}" + ("" if cur["launched"] else " (pushed)" if cur["pushed"] else " (pushing)") if cur else "-"
        rate_h = None
        if s.get("first_launch_at") and s.get("clips_done") and s.get("pod_id"):
            h = ((rec.get("ended_at") or now) - s["first_launch_at"]) / 3600
            rate_h = s["clips_done"] / h if h > 0 else None
            if not rec.get("ended_at") and rate_h:
                total_rate += rate_h
        up = ((rec.get("ended_at") or now) - rec["created_at"]) / 3600 if rec else 0.0
        cost = (rec.get("rate") or cfg_rate or 0.0) * up
        lines.append(f"{key:<4} {s.get('pod_id') or '-':<16} {s['phase']:<11} {prog:<24} "
                     f"{(nxt['batch'] if nxt else '-'):<14} {(f'{rate_h:.0f}' if rate_h else '-'):>7} {up:>5.1f} {cost:>6.2f}")
    remaining = 0
    for b in state["batches"].values():
        if b["status"] in ("queued", "claimed", "waiting_prep"):
            remaining += max(0, (b.get("total") or 0) - (b.get("done") or 0))
    for s in state["slots"].values():
        for key in ("current", "next"):
            if s.get(key):
                remaining -= s[key].get("remote_done", 0)
    eta = f"{remaining / total_rate:.1f} h" if total_rate > 0 else "? (no measured rate yet)"
    head = [f"fleet run {state['run_id']}: spend est ${usd:.2f} (burn ${burn:.2f}/h); "
            f"batches " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())),
            f"remaining ~{max(remaining, 0)} clips (ready/claimed/waiting-prep); fleet {total_rate:.0f} clips/h measured; ETA {eta}"]
    if state.get("driver_at"):
        head.append(f"driver last tick {int((now - state['driver_at']) / 60)} min ago")
    if state.get("halt"):
        head.append(f"HALTED (no new pods until a driver restart): {state['halt']['reason']}")
    pending = [pid for pid, r in state["pods"].items() if r.get("end_pending") and not r.get("ended_at")]
    if pending:
        head.append(f"STILL BILLED, {'/'.join(sorted({state['pods'][p]['end_pending'] for p in pending}))} failed "
                    f"(retrying): {', '.join(pending)}")
    waiting = [n for n in state["order"] if state["batches"][n]["status"] == "waiting_prep"]
    if waiting:
        head.append("waiting for prep.json: " + " ".join(waiting))
    if state.get("excluded"):
        head.append("excluded: " + " ".join(state["excluded"]))
    for req in inbox:
        head.append(f"inbox (merged at the driver's next tick): {req['op']} {' '.join(req['items'])}")
    failed = [n for n, b in state["batches"].items() if b["status"] == "failed" or b.get("failed")]
    if failed:
        head.append("batches with failures: " + ", ".join(
            f"{n}({state['batches'][n].get('failed', 0)} clips)" if state["batches"][n]["status"] != "failed"
            else f"{n}(gave up)" for n in failed))
    return "\n".join(head + lines)


# --------------------------------------------------------------------------- cli

# Fields a live pod was created/launched with: a resume may not change them while one runs.
POD_FIELDS = ("gpu_type", "gpu_count", "cloud", "interruptible", "dc", "image", "container_disk_gb", "volume_gb",
              "min_ram_gb", "pod_max_hours", "job_dir", "workers_per_gpu", "video_threads", "omp_threads")
# Fields that change what lands in results/fleet/outputs: fixed for the whole run.
OUTPUT_FIELDS = ("fast_video", "video_precision", "worker_args")
NON_CONFIG_ARGS = {"cmd", "batches", "go", "dry_run", "exclude", "fleet_dir"}
ARG_FIELD = {"worker_arg": "worker_args"}


def stored_config(state: dict | None) -> Config | None:
    raw = (state or {}).get("config")
    if not raw:
        return None
    names = {f.name for f in fields(Config)}
    return Config(**{k: v for k, v in raw.items() if k in names})


def resolve_config(a, state: dict | None) -> tuple[Config, list[str], list[str]]:
    """(config, changed fields, problems). A resumed run starts from the config stored in
    state.json; only flags given on this command line override it (argparse SUPPRESS)."""
    stored = stored_config(state)
    base = stored or Config()
    given = {ARG_FIELD.get(k, k): v for k, v in vars(a).items() if k not in NON_CONFIG_ARGS}
    if "worker_args" in given:
        given["worker_args"] = [str(x) for x in given["worker_args"]]
    cfg = dataclasses.replace(base, **given)
    if not stored:
        return cfg, [], []
    changed = [k for k, v in given.items() if getattr(base, k) != v]
    problems = []
    live = [pid for pid, r in state["pods"].items() if not r.get("ended_at")]
    flag = lambda k: "--" + {"worker_args": "worker-arg"}.get(k, k.replace("_", "-"))  # noqa: E731
    bad = [k for k in changed if k in OUTPUT_FIELDS] if state.get("creates") else []
    if bad:
        problems.append(f"{', '.join(map(flag, bad))} would mix outputs within fleet run {state['run_id']}; "
                        "use a new --fleet-dir for a different pipeline")
    bad = [k for k in changed if k in POD_FIELDS] if live else []
    if bad:
        problems.append(f"cannot change {', '.join(map(flag, bad))} while fleet pods are live ({', '.join(live)})")
    return cfg, changed, problems


def split_names(items: list[str]) -> list[str]:
    return [n.strip().rstrip("/").rsplit("/", 1)[-1] for it in items for n in it.split(",") if n.strip()]


def preflight(cfg: Config, ops, now: float) -> list[str]:
    problems = cfg.validate()
    if not cfg.gpu_type:
        problems.append("--gpu-type is required")
    if cfg.max_usd <= 0:
        problems.append("--max-usd must be > 0")
    if not cfg.usd_per_hour:
        problems.append("--usd-per-hour is required (fallback rate for the spend guard; see tools/runpod.py gpus)")
    ok, why, hb = ops.watchdog(now)
    if not ok:
        problems.append(f"{why}: start tools/runpod.py watchdog first (docs/RUNPOD.md)")
    else:
        # a watchdog action STOPS a pod-volume pod, which erases the container disk (JOB): its caps must be wider
        if runpod.to_float(hb.get("max_usd")) < cfg.max_usd:
            problems.append(f"watchdog --max-usd {hb.get('max_usd')} < fleet --max-usd {cfg.max_usd:g} "
                            "(it counts every tribe- pod; its stop erases unpulled outputs): restart it with a higher cap")
        if runpod.to_float(hb.get("max_hours")) < cfg.pod_max_hours:
            problems.append(f"watchdog --max-hours {hb.get('max_hours')} < --pod-max-hours {cfg.pod_max_hours:g}")
    return problems


@contextlib.contextmanager
def driver_lock(fleet_dir: Path):
    fleet_dir.mkdir(parents=True, exist_ok=True)
    with open(fleet_dir / "driver.lock", "a") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("another fleet driver is running (results/fleet/driver.lock)") from None
        yield


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--fleet-dir", default=None, help="state/log/outputs dir (default results/fleet)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    # SUPPRESS: only flags given on the command line override a resumed run's stored config
    r = sub.add_parser("run", parents=[common], argument_default=argparse.SUPPRESS,
                       help="plan (default) or, with --go, run/resume the fleet")
    r.add_argument("--batches", nargs="+", help="batch dirs or globs (repo-relative ok), queued in order; globs are "
                   f"rescanned while running. Default (new run): {DEFAULT_BATCH_ROOT}/"
                   "b02..b09, d00_deep_dive, d01_deep_dive, r00..r15")
    r.add_argument("--exclude", action="append", help="batch names to skip, comma-separated (e.g. b02,b03)")
    r.add_argument("--go", action="store_true", help="actually create pods (billed). Without it: print the plan")
    r.add_argument("--dry-run", action="store_true", help="print the plan (the default without --go)")
    r.add_argument("--pods", type=int, help="default 1")
    r.add_argument("--gpu-type")
    r.add_argument("--gpu-count", type=int, help="default 1")
    r.add_argument("--cloud", choices=["COMMUNITY", "SECURE"], help="default COMMUNITY")
    r.add_argument("--interruptible", action=argparse.BooleanOptionalAction,
                   help="spot pods (pre-emption erases the container disk)")
    r.add_argument("--dc")
    r.add_argument("--image")
    r.add_argument("--container-disk-gb", type=int, help="holds JOB: venv, weights, caches, clips (default 100)")
    r.add_argument("--volume-gb", type=int, help="the unused /workspace pod volume (default 20, min size unverified)")
    r.add_argument("--min-ram-gb", type=int, help="host RAM per GPU (runpod.py default 48)")
    r.add_argument("--pod-max-hours", type=float, help="per-pod cap passed to pod-create (default 8)")
    r.add_argument("--max-usd", type=float, help="fleet spend cap (estimate from $/h x uptime)")
    r.add_argument("--usd-per-hour", type=float, help="per-pod $/h used until the API reports the real rate")
    r.add_argument("--job-dir", help="JOB on the pod (default /root/tribe-job, container disk)")
    r.add_argument("--workers-per-gpu", type=int,
                   help=f"default floor((VRAM - {VRAM_RESERVE_GB} GB) / {WORKER_VRAM_GB} GB): 48 GB -> 2, 24 GB -> 1")
    r.add_argument("--fast-video", action=argparse.BooleanOptionalAction, help="default on")
    r.add_argument("--video-precision", choices=["fp32", "tf32", "bf16", "fp16"], help="default bf16")
    r.add_argument("--video-threads", type=int, help="worker.py --video-threads (default min(8, OMP threads))")
    r.add_argument("--omp-threads", type=int, help="OMP/MKL threads per worker (default max(4, cgroup quota / workers))")
    r.add_argument("--worker-arg", action="append", help="extra worker.py arg (repeatable)")
    r.add_argument("--on-done", choices=["terminate", "stop", "keep"], help="default terminate")
    r.add_argument("--poll-s", type=float, help="default 60")
    r.add_argument("--pull-every-min", type=float, help="default 15")
    r.add_argument("--dead-after-min", type=float, help="default 10")
    r.add_argument("--setup-timeout-min", type=float, help="hard cap on setup.sh, then terminate (default 30)")
    r.add_argument("--idle-grace-min", type=float, help="< the watchdog's 20 min idle rule (default 10)")
    r.add_argument("--drain-timeout-min", type=float, help="end a pod whose final pull never verifies (default 30)")
    r.add_argument("--max-creates", type=int, help="pods created in total, replacements included (default 2 x --pods)")
    r.add_argument("--worker-retries", type=int, help="default 1")
    r.add_argument("--wall-per-source-s", type=float, help=f"plan rate (default {WALL_PER_SOURCE_S}, measured L40S)")
    r.add_argument("--keep-remote-videos", action=argparse.BooleanOptionalAction)
    st = sub.add_parser("status", parents=[common], help="per-pod state, batch progress, clips/h, spend, ETA (reads state only)")
    st.add_argument("--usd-per-hour", type=float)
    ad = sub.add_parser("add", parents=[common], help="append batch dirs/globs/names to a running (or the next) fleet run")
    ad.add_argument("items", nargs="+", help=f"dir, glob, or a name under {DEFAULT_BATCH_ROOT}; undoes an exclude")
    ex = sub.add_parser("exclude", parents=[common], help="skip batches (names, comma-separated) in a running fleet")
    ex.add_argument("items", nargs="+")
    return ap


def main(argv: list[str] | None = None, ops=None, fleet_dir: Path = FLEET_DIR) -> int:
    a = build_parser().parse_args(argv)
    runpod.load_env()  # registers .env values for scrubbing; nothing is printed
    fleet_dir = Path(a.fleet_dir).resolve() if a.fleet_dir else fleet_dir
    store = Store(fleet_dir / "state.json")
    if a.cmd == "status":
        print(status_text(store.load(), time.time(), a.usd_per_hour, Inbox(fleet_dir).peek()))
        return 0
    if a.cmd in ("add", "exclude"):
        items = split_names(a.items) if a.cmd == "exclude" else a.items
        Inbox(fleet_dir).put(a.cmd, items)
        print(f"queued: {a.cmd} {' '.join(items)} (the driver merges it at its next tick, else at the next run --go)")
        return 0

    state = store.load()
    cfg, changed, problems = resolve_config(a, state)
    excluded = split_names(getattr(a, "exclude", None) or [])
    patterns = getattr(a, "batches", None)
    if patterns:
        dirs = resolve_batches(patterns)
    elif state and state["order"]:
        dirs = []  # resume: the stored queue
    else:
        dirs = default_batches()
    if getattr(a, "dry_run", False) or not getattr(a, "go", False):
        skip = None
        plan_dirs = dirs
        if state:
            print(f"resuming fleet run {state['run_id']} ({store.path}): config from its state file"
                  + (f"; overridden: {', '.join(changed)}" if changed else ""))
            skip = {n: b["status"] for n, b in state["batches"].items()}
            known = [Path(state["batches"][n]["dir"]) for n in state["order"]]
            plan_dirs = known + [d for d in dirs if d.name not in state["batches"]]
            excluded = excluded + list(state.get("excluded") or [])
        print(plan(cfg, plan_dirs, fleet_dir, excluded, skip))
        problems = cfg.validate() + problems
        for p in problems:
            print(f"problem: {p}")
        if not getattr(a, "go", False):
            print("(plan only; add --go to create pods)")
        return 2 if problems else 0
    ops = ops or RealOps(fleet_dir, cfg.job_dir)
    problems += preflight(cfg, ops, time.time())
    if problems:
        for p in problems:
            print(f"refused: {p}", file=sys.stderr)
        return 2
    with driver_lock(fleet_dir):
        fleet = Fleet(cfg, ops, fleet_dir)
        if changed:
            fleet.log("config_changed", fields=",".join(changed))
        if fleet.state.get("halt"):
            fleet.log("halt_cleared", reason=fleet.state["halt"]["reason"], note="driver restarted")
            fleet.state["halt"] = None
        if excluded:
            fleet.exclude(excluded)
        added = fleet.add_batches(dirs)
        if added:
            fleet.log("batches_added", n=len(added), names=",".join(added))
        with fleet.lock:
            for pat in patterns or []:
                if any(c in pat for c in "*?[") and pat not in fleet.state.setdefault("sources", []):
                    fleet.state["sources"].append(pat)
            fleet.state["config"] = asdict(cfg)
            fleet.save()
        fleet.merge_inbox()

        def stop(signum, _frame):
            fleet.log("driver_signal", signal=signum, note="stopping loop; pods keep running")
            fleet.stop_event.set()
        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
        return fleet.run()


if __name__ == "__main__":
    sys.exit(main())
