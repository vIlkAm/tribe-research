# Source on every pod shell: points every cache at the shared network volume.
export JOB="${JOB:-/workspace/tribe-job}"
export HF_HOME="$JOB/hf-cache"
export UV_CACHE_DIR="$JOB/uv-cache"
export TORCH_HOME="$JOB/torch-cache"   # whisperx align model (torchaudio) lands here
export TRIBE_CACHE="$JOB/feature-cache"
export TRIBE_REPO="$JOB/repo"
export PYTHONUNBUFFERED=1
# pod env vars (HF_TOKEN) reach PID 1 but not non-interactive SSH shells
if [ -z "${HF_TOKEN:-}" ] && [ -r /proc/1/environ ]; then
  HF_TOKEN="$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^HF_TOKEN=//p')"
  [ -n "$HF_TOKEN" ] && export HF_TOKEN || unset HF_TOKEN
fi
if [ -f "$JOB/venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$JOB/venv/bin/activate"
fi
