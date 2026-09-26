#!/usr/bin/env bash
# Drive one pod through study batches, sharing them with other pods through claim files.
#
#   tools/drive_pod.sh NAME HOST PORT WORKERS THREADS "b05 b04 b06" [--setup] [--one-worker-manifest]
#
# For each batch in order: claim it (results/runs/study-bf16/claim-<b>, noclobber; skip if another
# pod holds it), push videos + manifest, touch pushed-<b>, run pod/run_queue.sh for that batch,
# wait for done-<b>, pull outputs to results/runs/study-bf16/outputs-<b>.
#   --setup                 first check the GPU is idle, push code, run pod/setup.sh, wait for .setup_done
#   --one-worker-manifest   rewrite the batch manifest to num_workers=1 (24 GB cards fit one worker:
#                           each worker + its whisper server needs ~19 GB)
# Never creates or stops pods; the watchdog owns spend.
set -u
cd "$(dirname "$0")/.."
NAME=$1 HOST=$2 PORT=$3 WORKERS=$4 THREADS=$5 ORDER=$6
shift 6
SETUP=0 W1=0
for a in "$@"; do
  case "$a" in
    --setup) SETUP=1 ;;
    --one-worker-manifest) W1=1 ;;
    *) echo "unknown option $a" >&2; exit 2 ;;
  esac
done
export POD=root@$HOST POD_PORT=$PORT JOB=/workspace/tribe-job
SSHO=(-o ConnectTimeout=20 -o ServerAliveInterval=30 -o StrictHostKeyChecking=accept-new -p "$PORT")
R() { timeout 180 ssh -n "${SSHO[@]}" "$POD" "$@"; }
OUT=results/runs/study-bf16
mkdir -p "$OUT"
log() { echo "$(date -u +%FT%TZ) [$NAME] $*"; }

if [ "$SETUP" = 1 ]; then
  until R true 2>/dev/null; do sleep 20; done
  log "gpu at start: $(R 'nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader')"
  log "cpu quota: $(R 'cat /sys/fs/cgroup/cpu.max 2>/dev/null || cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us')"
  until nice tools/pod.sh push-code >/dev/null 2>&1; do sleep 20; done
  R "mkdir -p $JOB/logs && cd $JOB && setsid -f bash -c 'bash code/pod/setup.sh > logs/setup.log 2>&1' </dev/null >/dev/null 2>&1"
  log "setup started"
  until R "test -f $JOB/.setup_done" 2>/dev/null; do
    if R "test -f $JOB/.setup_failed" 2>/dev/null; then log "SETUP FAILED: $(R "cat $JOB/.setup_failed")"; exit 1; fi
    sleep 30
  done
  log "setup done: $(R "tail -1 $JOB/logs/setup-timings.txt")"
fi

for b in $ORDER; do
  if ( set -o noclobber; echo "$NAME $(date -u +%FT%TZ)" >"$OUT/claim-$b" ) 2>/dev/null; then
    log "claimed $b"
  elif [ "$(cut -d' ' -f1 "$OUT/claim-$b")" = "$NAME" ]; then
    log "resuming own claim $b"   # restart of this driver: outputs are resumable
  else
    log "skip $b: claimed by $(cat "$OUT/claim-$b")"
    continue
  fi
  until nice tools/pod.sh push-batch "results/batches_s384/$b" >/dev/null 2>&1; do sleep 30; done
  if [ "$W1" = 1 ]; then
    python3 -c 'import json,sys
for l in open(sys.argv[1]):
    r = json.loads(l); r["num_workers"] = 1; r["worker"] = 0; print(json.dumps(r))' \
      "results/batches_s384/$b/manifest.jsonl" >"/tmp/drive_${NAME}_$b.jsonl"
    until scp -q "${SSHO[@]/-p/-P}" "/tmp/drive_${NAME}_$b.jsonl" "$POD:$JOB/batches/$b.jsonl"; do sleep 30; done
  fi
  R "touch $JOB/logs/pushed-$b; cd $JOB && QUEUE=$b WORKERS=$WORKERS THREADS=$THREADS setsid -f bash -c 'bash code/pod/run_queue.sh >> logs/queue.log 2>&1' </dev/null >/dev/null 2>&1"
  log "started $b"
  until R "test -f $JOB/logs/done-$b" 2>/dev/null; do sleep 60; done
  until nice rsync -rlpt -e "ssh ${SSHO[*]}" "$POD:$JOB/outputs-$b/" "$OUT/outputs-$b/"; do sleep 30; done
  log "done $b: $(R "tail -1 $JOB/logs/queue.log")"
done
log "order exhausted"
