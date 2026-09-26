# Source on every pod shell: points every cache at the shared network volume.
export JOB="${JOB:-/workspace/tribe-job}"
export HF_HOME="$JOB/hf-cache"
export UV_CACHE_DIR="$JOB/uv-cache"
export TRIBE_CACHE="$JOB/feature-cache"
export TRIBE_REPO="$JOB/repo"
export PYTHONUNBUFFERED=1
if [ -f "$JOB/venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$JOB/venv/bin/activate"
fi
