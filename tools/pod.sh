#!/usr/bin/env bash
# Move code, clips and results between this server and a RunPod pod over full SSH.
# See docs/RUNPOD.md. Needs the pod's "SSH over exposed TCP" address; the proxied
# ssh.runpod.io endpoint does not carry rsync/scp.
#
#   export POD=root@213.173.108.12 POD_PORT=17445     # from the pod's Connect tab
#   tools/pod.sh ssh                                   # interactive shell
#   tools/pod.sh push-code                             # pod/ tools/ tribe_research/ -> $JOB/code/
#   tools/pod.sh push-videos ./videos                  # clip folder -> $JOB/videos/
#   tools/pod.sh push-batch results/batches/b02        # batch clips (added) + its manifest -> $JOB/
#   tools/pod.sh pull run1                             # $JOB/outputs + logs + manifest -> results/run1/
#
# Only pushes what you name. Nothing here listens on a port (AGENTS.md).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
: "${POD:?set POD=root@<ip> from the pod's Connect tab}"
: "${POD_PORT:?set POD_PORT=<port> from the pod's Connect tab}"
KEY="${POD_KEY:-$HOME/.ssh/id_ed25519}"
JOB="${JOB:-/workspace/tribe-job}"
SSH=(ssh -p "$POD_PORT" -i "$KEY" -o StrictHostKeyChecking=accept-new)
RSYNC=(rsync -rlpt --info=progress2 -e "${SSH[*]}")
# stock RunPod images lack rsync, and setup.sh (which installs it) arrives by rsync
need_rsync() {
  "${SSH[@]}" "$POD" 'command -v rsync >/dev/null || { apt-get update -qq && apt-get install -y -qq rsync >/dev/null; }'
}

cmd="${1:-}"; shift || true
case "$cmd" in
  ssh)
    exec "${SSH[@]}" "$POD" "$@" ;;
  push-code)
    need_rsync
    "${SSH[@]}" "$POD" "mkdir -p $JOB/code"
    "${RSYNC[@]}" --delete --exclude __pycache__ --exclude '*.pyc' \
      "$ROOT/pod" "$ROOT/tools" "$ROOT/tribe_research" "$POD:$JOB/code/" ;;
  push-videos)
    src="${1:?usage: pod.sh push-videos <local-dir>}"
    need_rsync
    "${SSH[@]}" "$POD" "mkdir -p $JOB/videos"
    "${RSYNC[@]}" "${src%/}/" "$POD:$JOB/videos/" ;;
  push-batch)
    # tools/make_batches.py output: videos are added to $JOB/videos (earlier batches stay,
    # finished outputs are skipped); the batch manifest becomes $JOB/manifest.jsonl
    b="${1:?usage: pod.sh push-batch <results/batches/NAME>}"
    [ -f "$b/manifest.jsonl" ] || { echo "no $b/manifest.jsonl" >&2; exit 1; }
    need_rsync
    "${SSH[@]}" "$POD" "mkdir -p $JOB/videos $JOB/batches"
    "${RSYNC[@]}" "${b%/}/videos/" "$POD:$JOB/videos/"
    "${RSYNC[@]}" "$b/manifest.jsonl" "$POD:$JOB/batches/$(basename "$b").jsonl"
    "${RSYNC[@]}" "$b/manifest.jsonl" "$POD:$JOB/manifest.jsonl" ;;
  pull)
    run="${1:?usage: pod.sh pull <run-name>}"
    dest="$ROOT/results/$run"
    mkdir -p "$dest"
    need_rsync
    "${RSYNC[@]}" "$POD:$JOB/outputs/" "$dest/outputs/"
    "${RSYNC[@]}" "$POD:$JOB/logs/" "$dest/logs/"
    "${RSYNC[@]}" "$POD:$JOB/manifest.jsonl" "$dest/"
    # ROI map is built on the pod when figshare blocks this server.
    "${RSYNC[@]}" "$POD:$JOB/code/tribe_research/assets/" "$ROOT/tribe_research/assets/" || true
    echo "pulled to $dest" ;;
  *)
    sed -n '2,13p' "$0"; exit 2 ;;
esac
