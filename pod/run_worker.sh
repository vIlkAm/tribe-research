#!/usr/bin/env bash
# Run one worker's shard on one GPU, with a utilisation sampler for that GPU.
#
#   WORKER_ID=0 NUM_WORKERS=1 pod/run_worker.sh --limit 1   # first smoke test
#   WORKER_ID=2 NUM_WORKERS=4 GPU=2 pod/run_worker.sh       # one of four (see launch_all.sh)
#
# GPU defaults to WORKER_ID. Extra args pass through to worker.py. Safe to
# re-run: finished videos are skipped.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=env.sh
source "$HERE/env.sh"
: "${WORKER_ID:?set WORKER_ID}"
: "${NUM_WORKERS:?set NUM_WORKERS}"
GPU="${GPU:-$WORKER_ID}"
[ -f "$JOB/logs/setup-complete" ] || { echo "run pod/setup.sh first" >&2; exit 1; }
mkdir -p "$JOB/logs"

# Setup already downloaded every weight; workers must not write to hf-cache.
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES="$GPU"

GPU_LOG="$JOB/logs/gpu-worker-$WORKER_ID.csv"
nvidia-smi -i "$GPU" \
  --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
  --format=csv -l 2 >>"$GPU_LOG" &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT

python "$HERE/worker.py" \
  --manifest "$JOB/manifest.jsonl" \
  --videos-root "$JOB/videos" \
  --out-root "$JOB/outputs" \
  --worker-id "$WORKER_ID" \
  --num-workers "$NUM_WORKERS" \
  --cache-folder "$TRIBE_CACHE" \
  --tribe-repo "$TRIBE_REPO" \
  "$@" 2>&1 | tee -a "$JOB/logs/worker-$WORKER_ID.log"
