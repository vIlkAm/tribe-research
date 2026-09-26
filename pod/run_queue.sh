#!/usr/bin/env bash
# Run pushed batches in order on one GPU: WORKERS concurrent fast-video workers per batch
# (bf16 by default, the pre-registered precision), then one retry pass for failed clips.
#
#   QUEUE="b02 b03 b04" setsid nohup bash $JOB/code/pod/run_queue.sh > $JOB/logs/queue.log 2>&1 &
#
# A batch starts once $JOB/logs/pushed-<b> exists (the server writes it after `pod.sh push-batch`
# has copied the videos and $JOB/batches/<b>.jsonl). WAIT_FOR=<file> holds the first batch until
# an earlier run has finished. Outputs: $JOB/outputs-<b>; marker: $JOB/logs/done-<b>.
# Each worker plus its whisper server needs ~19 GB of GPU memory: 2 workers on a 48 GB card.
# The pod cgroup quota (~27 CPUs on an L40S host) is far below nproc, so thread pools are capped.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=env.sh
source "$HERE/env.sh"
L="$JOB/logs"
WORKERS="${WORKERS:-2}"
PREC="${PREC:-bf16}"
THREADS="${THREADS:-12}"
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

run_batch() {  # batch tag
  local b="$1" tag="$2" w
  for ((w = 0; w < WORKERS; w++)); do
    WORKER_ID=$w NUM_WORKERS=$WORKERS GPU=0 "$HERE/run_worker.sh" --manifest "$JOB/batches/$b.jsonl" \
      --fast-video --video-precision "$PREC" --out-root "$JOB/outputs-$b" \
      --cache-folder "$JOB/feature-cache-$PREC" >"$L/run-$b-$tag-w$w.out" 2>&1 &
  done
  wait
}

[ -n "${WAIT_FOR:-}" ] && until [ -f "$WAIT_FOR" ]; do sleep 15; done
for b in ${QUEUE:?set QUEUE}; do
  until [ -f "$L/pushed-$b" ]; do sleep 20; done
  echo "$(date -u +%FT%TZ) start $b"
  run_batch "$b" main
  if compgen -G "$JOB/outputs-$b/worker-*/*.error.json" >/dev/null; then
    mkdir -p "$L/errors-$b"
    mv "$JOB"/outputs-"$b"/worker-*/*.error.json "$L/errors-$b/"
    echo "$(date -u +%FT%TZ) retry $b: $(ls "$L/errors-$b" | wc -l) failed"
    run_batch "$b" retry
  fi
  date -u +%FT%TZ >"$L/done-$b"
  echo "$(date -u +%FT%TZ) done $b ok=$(ls "$JOB"/outputs-"$b"/worker-*/*.json 2>/dev/null | grep -vc error)"
done
