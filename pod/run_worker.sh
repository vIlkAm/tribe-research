#!/usr/bin/env bash
# Run one worker's shard with a GPU utilisation sampler alongside it.
#
#   WORKER_ID=0 NUM_WORKERS=4 pod/run_worker.sh            # full shard
#   WORKER_ID=0 NUM_WORKERS=1 pod/run_worker.sh --limit 1  # first smoke test
#
# Extra args pass through to worker.py. Safe to re-run: finished videos are skipped.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=env.sh
source "$HERE/env.sh"
: "${WORKER_ID:?set WORKER_ID}"
: "${NUM_WORKERS:?set NUM_WORKERS}"
mkdir -p "$JOB/logs"

GPU_LOG="$JOB/logs/gpu-worker-$WORKER_ID.csv"
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
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
