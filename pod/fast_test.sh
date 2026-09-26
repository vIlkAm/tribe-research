#!/usr/bin/env bash
# Fast-video GPU test, then the frontend wiring batch, unattended on one GPU.
# Waits for setup.sh, then runs (each worker shard in turn, so timings stay clean):
#   1. b00_pilot  --fast-video fp32   -> outputs-fast-fp32   (speed + fp32 gate vs stock)
#   2. b00_pilot  --fast-video bf16   -> outputs-fast-bf16   (speed + bf16 gate vs fp32)
#   3. b01_frontend (or $FRONT_BATCH) bf16 -> outputs-front-bf16
# Per-core CPU goes to logs/mpstat.log; logs/done-<tag> marks each finished stage.
#
#   setsid nohup bash $JOB/code/pod/fast_test.sh > $JOB/logs/fast_test.log 2>&1 &
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=env.sh
source "$HERE/env.sh"
L="$JOB/logs"
STAGES="${STAGES:-fast-fp32 fast-bf16 front-bf16}"

until [ -f "$L/setup-complete" ]; do
  pgrep -f "[s]etup.sh" >/dev/null || { echo "setup.sh stopped without setup-complete" >&2; exit 1; }
  sleep 20
done

command -v mpstat >/dev/null || apt-get install -y -qq sysstat >/dev/null 2>&1
mpstat -P ALL 10 >"$L/mpstat.log" 2>&1 &
MP=$!
trap 'kill $MP 2>/dev/null || true' EXIT

run() {  # tag manifest [worker args]
  local tag="$1" man="$2"
  shift 2
  echo "$(date -u +%FT%TZ) start $tag"
  for w in 0 1; do
    WORKER_ID=$w NUM_WORKERS=2 GPU=0 "$HERE/run_worker.sh" --manifest "$man" --fast-video \
      --out-root "$JOB/outputs-$tag" --cache-folder "$JOB/feature-cache-$tag" "$@" >"$L/run-$tag-w$w.out" 2>&1
  done
  date -u +%FT%TZ >"$L/done-$tag"
  echo "$(date -u +%FT%TZ) done $tag"
}

for s in $STAGES; do
  case "$s" in
    fast-fp32) run fast-fp32 "$JOB/batches/b00_pilot.jsonl" ;;
    fast-bf16) run fast-bf16 "$JOB/batches/b00_pilot.jsonl" --video-precision bf16 ;;
    front-bf16) run front-bf16 "$JOB/batches/${FRONT_BATCH:-b01_frontend}.jsonl" --video-precision bf16 ;;
    front-fp32) run front-fp32 "$JOB/batches/${FRONT_BATCH:-b01_frontend}.jsonl" ;;
    *) echo "unknown stage $s" >&2 ;;
  esac
done
