#!/usr/bin/env bash
# Idempotent TRIBE v2 environment setup on a RunPod pod (see docs/RUNPOD.md).
#
# State lives at $JOB (default /workspace/tribe-job, the network volume). For a fleet pod, put it
# on the container disk instead (JOB=/root/tribe-job, --container-disk-gb >= 100): no MFS stale
# handles in uv-cache, local-disk weights, nothing to keep after the pod is terminated.
# Pass JOB to every remote command: pod env vars do not reach non-interactive SSH shells.
#
#   repo/            facebookresearch/tribev2 pinned to $TRIBE_COMMIT
#   venv/            ISOLATED venv: torch 2.6.0+cu124 regardless of the image's torch
#   hf-cache/        HF weights (TRIBE ckpt, V-JEPA2, Wav2Vec-BERT, Llama-3.2-3B, whisper large-v3)
#   uv-cache/        uvx env for whisperx (stock `uvx whisperx` + pod/whisper_server.py, pinned)
#   feature-cache/   TRIBE's per-modality extracted features (the expensive part)
#   code/            this repo (rsync'd by tools/pod.sh push)
#   videos/ outputs/ logs/ manifest.jsonl
#
# Expected image: runpod/pytorch:*-py3.11-cuda12.4* (official template, full SSH).
# Requires HF_TOKEN (RunPod secret) whose account has accepted meta-llama/Llama-3.2-3B.
# Run once per volume; re-running is safe and fast. Container-disk apt packages
# must be reinstalled after every pod restart: `bash setup.sh` does that too.
#
# Weights are prefetched in the background (hf_transfer, one process per repo) while the venv,
# torch and the whisperx env install; PREFETCH_WEIGHTS=0 turns that off. A failed prefetch is not
# fatal: the serial steps below download whatever is missing.
#
# Markers for unattended drivers (tools/fleet.py): $JOB/logs/setup-complete and $JOB/.setup_done
# on success, $JOB/.setup_failed (exit code + step) on failure; per-step seconds in
# $JOB/logs/setup-timings.txt.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
JOB="${JOB:-/workspace/tribe-job}"
TRIBE_COMMIT="${TRIBE_COMMIT:-af58661791a351a448a489042a28f6c37e1c14b7}"
TORCH_VERSION="2.6.0"
TORCHVISION_VERSION="0.21.0"
TORCH_INDEX="https://download.pytorch.org/whl/cu124"

# shellcheck source=env.sh
source "$HERE/env.sh"
mkdir -p "$JOB"/{videos,outputs,logs,hf-cache,uv-cache,feature-cache}
rm -f "$JOB/.setup_done" "$JOB/.setup_failed" "$JOB/logs/setup-complete"
STEP=start
T_START=$(date +%s); T_LAST=$T_START
TIMINGS="$JOB/logs/setup-timings.txt"
: >"$TIMINGS"
step() {  # close the previous step's timing, name the next one
  local now; now=$(date +%s)
  echo "$STEP $((now - T_LAST))s" >>"$TIMINGS"
  T_LAST=$now; STEP="$1"; echo "== $1"
}
trap 'rc=$?; if [ "$rc" -ne 0 ]; then echo "rc=$rc step=$STEP" >"$JOB/.setup_failed"; fi' EXIT

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN not set. Add it as a RunPod secret and reference it in the pod env." >&2
  exit 1
fi

step "gpu"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

PY="${PYTHON:-python3.11}"
command -v "$PY" >/dev/null || PY=python3
"$PY" -c 'import sys; assert sys.version_info >= (3, 11), f"TRIBE needs Python >=3.11, have {sys.version}"'

PREFETCH_PID=""
if [ "${PREFETCH_WEIGHTS:-1}" = "1" ]; then
  step "weight prefetch start (background, log: logs/prefetch.log)"
  # A throwaway venv: hf_transfer never enters venv/, so TRIBE's environment is unchanged.
  [ -x "$JOB/dl-venv/bin/python" ] || "$PY" -m venv "$JOB/dl-venv"
  (
    set -e
    "$JOB/dl-venv/bin/pip" install -q "huggingface_hub>=0.30" hf_transfer
    HF_HUB_ENABLE_HF_TRANSFER=1 "$JOB/dl-venv/bin/python" - <<'PYEOF'
import concurrent.futures as cf, os, re, time
from huggingface_hub import snapshot_download
IGNORE = ["original/*", "*.msgpack", "*.h5", "*.onnx", "*.ot"]  # same filter as the serial step
t0 = time.time()
cfg = open(os.path.join(snapshot_download("facebook/tribev2"), "config.yaml")).read()
print(f"facebook/tribev2 {time.time() - t0:.0f}s", flush=True)
names = sorted(set(re.findall(r"^\s*model_name:\s*([\w.-]+/[\w.-]+)\s*$", cfg, re.M)))
names.append("Systran/faster-whisper-large-v3")  # whisperx large-v3 (faster-whisper's repo)
def get(name):
    t = time.time()
    snapshot_download(name, ignore_patterns=IGNORE, max_workers=8)
    return name, time.time() - t
with cf.ThreadPoolExecutor(len(names)) as ex:
    for f in cf.as_completed([ex.submit(get, n) for n in names]):
        name, dt = f.result()
        print(f"{name} {dt:.0f}s", flush=True)
print(f"prefetch done {time.time() - t0:.0f}s", flush=True)
PYEOF
  ) >"$JOB/logs/prefetch.log" 2>&1 &
  PREFETCH_PID=$!
fi

step "system packages (container disk)"
apt-get update -qq
apt-get install -y -qq git ffmpeg rsync >/dev/null

step "venv + torch $TORCH_VERSION (cu124), pinned for the whole install"
if [ ! -f "$JOB/venv/bin/activate" ]; then
  "$PY" -m venv "$JOB/venv"   # no --system-site-packages: never inherit the image's torch
fi
# shellcheck disable=SC1091
source "$JOB/venv/bin/activate"
pip install -q --upgrade pip

CONSTRAINTS="$JOB/logs/constraints.txt"
printf 'torch==%s\ntorchvision==%s\n' "$TORCH_VERSION" "$TORCHVISION_VERSION" >"$CONSTRAINTS"
pip install -q "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" --index-url "$TORCH_INDEX"

step "tribev2 @ $TRIBE_COMMIT"
if [ ! -d "$JOB/repo/.git" ]; then
  git clone -q https://github.com/facebookresearch/tribev2.git "$JOB/repo"
fi
git -C "$JOB/repo" fetch -q origin
git -C "$JOB/repo" checkout -q "$TRIBE_COMMIT"
pip install -q -c "$CONSTRAINTS" --extra-index-url "$TORCH_INDEX" -e "$JOB/repo"
# uv provides `uvx` (TRIBE runs whisperx through it); the rest is for our brain layer.
pip install -q -c "$CONSTRAINTS" uv nibabel pyyaml mne

python - <<'EOF'
import torch
assert torch.__version__.startswith("2.6.0"), f"torch drifted to {torch.__version__}"
assert torch.cuda.is_available(), "CUDA not available in venv"
cap = torch.cuda.get_device_capability(0)
assert cap < (10, 0), f"GPU sm_{cap[0]}{cap[1]} (Blackwell) is not supported by torch 2.6+cu124"
print(f"torch {torch.__version__} cuda {torch.version.cuda} on {torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]}")
EOF

step "spacy model"
python -m spacy download en_core_web_sm >/dev/null

step "whisperx: pinned uv env, stock CLI smoke + persistent-server pre-warm"
# whisperx stays in its own uv env (installing it into venv/ breaks TRIBE's torch pin).
# One constraints file pins BOTH TRIBE's stock `uvx whisperx` call and the server
# (pod/worker.py sets the same pins via UV_CONSTRAINT); versions live in whisper_server.py.
WX_CONSTRAINTS="$JOB/logs/whisperx-constraints.txt"
python "$HERE/whisper_server.py" --print-constraints >"$WX_CONSTRAINTS"
export UV_CONSTRAINT="$WX_CONSTRAINTS"
WX_VERSION="$(sed -n 's/^whisperx==//p' "$WX_CONSTRAINTS")"
# TRIBE's exact flags (minus wav/--output_dir), from the same function the worker uses.
read -r -a WX_FLAGS <<<"$(cd "$HERE" && python -c 'import worker; print(" ".join(worker.upstream_whisperx_flags("english", "cuda")))')"
WX_TMP="$(mktemp -d)"
WX_LOG="$JOB/logs/whisperx-smoke.log"
: >"$WX_LOG"
# A tone clip exercises CUDA model load without needing speech.
ffmpeg -v error -f lavfi -i "sine=frequency=440:duration=3" -ar 16000 "$WX_TMP/tone.wav"
# 1) the stock per-clip path exactly as TRIBE runs it (the worker's fallback)
if ! uvx whisperx "$WX_TMP/tone.wav" "${WX_FLAGS[@]}" --output_dir "$WX_TMP" >>"$WX_LOG" 2>&1; then
  echo "whisperx on CUDA failed; see $WX_LOG (driver/CUDA mismatch in its env?)" >&2
  exit 1
fi
# 2) the persistent server: loads large-v3 + align model once, fetches nltk punkt_tab
#    (container disk, so re-run after a pod restart), transcribes the tone, exits.
if ! uvx --from "whisperx==$WX_VERSION" python "$HERE/whisper_server.py" --noise-to-stderr \
      --self-test "$WX_TMP/tone.wav" -- "${WX_FLAGS[@]}" >>"$WX_LOG" 2>&1; then
  echo "whisper server self-test failed; see $WX_LOG. Workers will fall back to per-clip uvx whisperx." >&2
fi
# 3) both envs must now resolve from the uv cache alone (no per-worker resolve at launch)
uvx --offline whisperx --version >>"$WX_LOG" 2>&1 \
  || { echo "stock uvx whisperx env not fully cached; see $WX_LOG" >&2; exit 1; }
uvx --offline --from "whisperx==$WX_VERSION" python -c \
  'import importlib.metadata as m; print("whisperx", m.version("whisperx"), "faster-whisper", m.version("faster-whisper"), "ctranslate2", m.version("ctranslate2"), "torch", m.version("torch"))' \
  | tee "$JOB/logs/whisperx-version.txt"
rm -rf "$WX_TMP"

if [ -n "$PREFETCH_PID" ]; then
  step "wait for weight prefetch"
  if wait "$PREFETCH_PID"; then tail -n 1 "$JOB/logs/prefetch.log"
  else echo "weight prefetch failed (see $JOB/logs/prefetch.log); downloading serially" >&2; fi
fi

step "TRIBE weights (cached at \$HF_HOME unless the prefetch got them)"
python - <<'EOF'
import os, time
from tribev2 import TribeModel
t = time.time()
m = TribeModel.from_pretrained("facebook/tribev2", cache_folder=os.environ["TRIBE_CACHE"])
print(f"TRIBE loaded in {time.time() - t:.0f}s, TR={m.data.TR}")
EOF

step "extractor weights named in the TRIBE config (workers run with HF_HUB_OFFLINE=1)"
python - <<'EOF'
import glob, os, re
from huggingface_hub import snapshot_download
cfg = glob.glob(os.path.join(os.environ["HF_HOME"], "hub/models--facebook--tribev2/snapshots/*/config.yaml"))[0]
names = sorted(set(re.findall(r"^\s*model_name:\s*([\w.-]+/[\w.-]+)\s*$", open(cfg).read(), re.M)))
for name in names:
    # original/ is Llama's duplicate consolidated checkpoint; transformers never reads it
    path = snapshot_download(name, ignore_patterns=["original/*", "*.msgpack", "*.h5", "*.onnx", "*.ot"])
    print(f"cached {name} -> {path}")
EOF

step "ROI map (HCP-MMP1 -> fsaverage5), cross-checked against tribev2's own labels"
if [ -f "$JOB/code/tools/build_roi_map.py" ]; then
  python "$JOB/code/tools/build_roi_map.py" --check-against-tribe \
    || echo "ROI map build failed (figshare blocked?). Inference is unaffected; see docs/RUNPOD.md." >&2
fi

step "done"
git -C "$JOB/repo" rev-parse HEAD >"$JOB/logs/tribe-commit.txt"
pip freeze >"$JOB/logs/pip-freeze.txt"   # provenance: fleet pods set up hours apart must match
echo "total $(( $(date +%s) - T_START ))s" >>"$TIMINGS"
date -u +%FT%TZ >"$JOB/logs/setup-complete"
touch "$JOB/.setup_done"
du -sh "$JOB"/{venv,repo,hf-cache,uv-cache,feature-cache} 2>/dev/null || true
echo "== setup complete. Next: NUM_WORKERS=1 WORKER_ID=0 $HERE/run_worker.sh --limit 1"
