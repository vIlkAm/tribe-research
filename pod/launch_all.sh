#!/usr/bin/env bash
# Phase 2: one pod, N GPUs, one worker per GPU, all in the background.
#
#   pod/launch_all.sh            # N = number of visible GPUs
#   NUM_WORKERS=2 pod/launch_all.sh
#
# The manifest must have been built with the same --workers N. Progress:
#   tail -f $JOB/logs/worker-*.log
# Then: python $JOB/code/tools/merge.py --manifest $JOB/manifest.jsonl --out-root $JOB/outputs
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=env.sh
source "$HERE/env.sh"
NUM_WORKERS="${NUM_WORKERS:-$(nvidia-smi -L | wc -l)}"

m_workers="$(head -1 "$JOB/manifest.jsonl" | python -c 'import json,sys; print(json.load(sys.stdin)["num_workers"])')"
if [ "$m_workers" != "$NUM_WORKERS" ]; then
  echo "manifest built for $m_workers workers, launching $NUM_WORKERS; rebuild with --workers $NUM_WORKERS" >&2
  exit 1
fi

for ((k = 0; k < NUM_WORKERS; k++)); do
  WORKER_ID=$k NUM_WORKERS=$NUM_WORKERS GPU=$k nohup "$HERE/run_worker.sh" "$@" \
    >"$JOB/logs/launch-worker-$k.out" 2>&1 &
  echo "worker-$k on GPU $k: pid $!"
done
echo "all launched; 'wait' is not held so the SSH session can close safely"
