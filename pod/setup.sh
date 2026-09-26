#!/usr/bin/env bash
# Idempotent TRIBE v2 environment setup on a RunPod pod (see docs/RUNPOD.md).
#
# Persistent state lives on the network volume at $JOB (default /workspace/tribe-job):
#
#   repo/            facebookresearch/tribev2 pinned to $TRIBE_COMMIT
#   venv/            ISOLATED venv: torch 2.6.0+cu124 regardless of the image's torch
#   hf-cache/        HF weights (TRIBE ckpt, V-JEPA2, Wav2Vec-BERT, Llama-3.2-3B, whisper large-v3)
#   uv-cache/        uvx env for whisperx (TRIBE shells out to `uvx whisperx`)
#   feature-cache/   TRIBE's per-modality extracted features (the expensive part)
#   code/            this repo (rsync'd by tools/pod.sh push)
#   videos/ outputs/ logs/ manifest.jsonl
#
# Expected image: runpod/pytorch:*-py3.11-cuda12.4* (official template, full SSH).
# Requires HF_TOKEN (RunPod secret) whose account has accepted meta-llama/Llama-3.2-3B.
# Run once per volume; re-running is safe and fast. Container-disk apt packages
# must be reinstalled after every pod restart: `bash setup.sh` does that too.
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

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN not set. Add it as a RunPod secret and reference it in the pod env." >&2
  exit 1
fi

echo "== system packages (container disk)"
apt-get update -qq
apt-get install -y -qq git ffmpeg rsync >/dev/null

echo "== gpu"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

PY="${PYTHON:-python3.11}"
command -v "$PY" >/dev/null || PY=python3
"$PY" -c 'import sys; assert sys.version_info >= (3, 11), f"TRIBE needs Python >=3.11, have {sys.version}"'

if [ ! -f "$JOB/venv/bin/activate" ]; then
  "$PY" -m venv "$JOB/venv"   # no --system-site-packages: never inherit the image's torch
fi
# shellcheck disable=SC1091
source "$JOB/venv/bin/activate"
pip install -q --upgrade pip

echo "== torch $TORCH_VERSION (cu124), pinned for the whole install"
CONSTRAINTS="$JOB/logs/constraints.txt"
printf 'torch==%s\ntorchvision==%s\n' "$TORCH_VERSION" "$TORCHVISION_VERSION" >"$CONSTRAINTS"
pip install -q "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" --index-url "$TORCH_INDEX"

echo "== tribev2 @ $TRIBE_COMMIT"
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

echo "== spacy model"
python -m spacy download en_core_web_sm >/dev/null

echo "== whisperx GPU smoke (exactly the flags TRIBE uses; also caches large-v3)"
# TRIBE calls the unpinned `uvx whisperx`; its env resolves its own torch. A tone
# clip exercises CUDA model load without needing speech.
WX_TMP="$(mktemp -d)"
ffmpeg -v error -f lavfi -i "sine=frequency=440:duration=3" -ar 16000 "$WX_TMP/tone.wav"
if ! uvx whisperx "$WX_TMP/tone.wav" --model large-v3 --language en --device cuda \
      --compute_type float16 --batch_size 16 --align_model WAV2VEC2_ASR_LARGE_LV60K_960H \
      --output_dir "$WX_TMP" --output_format json >"$JOB/logs/whisperx-smoke.log" 2>&1; then
  echo "whisperx on CUDA failed; see $JOB/logs/whisperx-smoke.log (driver/CUDA mismatch in its env?)" >&2
  exit 1
fi
uvx --from whisperx python -c 'import importlib.metadata as m; print("whisperx", m.version("whisperx"))' \
  | tee "$JOB/logs/whisperx-version.txt"
rm -rf "$WX_TMP"

echo "== TRIBE weights (first run downloads; cached on the volume)"
python - <<'EOF'
import os, time
from tribev2 import TribeModel
t = time.time()
m = TribeModel.from_pretrained("facebook/tribev2", cache_folder=os.environ["TRIBE_CACHE"])
print(f"TRIBE loaded in {time.time() - t:.0f}s, TR={m.data.TR}")
EOF

echo "== ROI map (HCP-MMP1 -> fsaverage5), cross-checked against tribev2's own labels"
if [ -f "$JOB/code/tools/build_roi_map.py" ]; then
  python "$JOB/code/tools/build_roi_map.py" --check-against-tribe \
    || echo "ROI map build failed (figshare blocked?). Inference is unaffected; see docs/RUNPOD.md." >&2
fi

git -C "$JOB/repo" rev-parse HEAD >"$JOB/logs/tribe-commit.txt"
date -u +%FT%TZ >"$JOB/logs/setup-complete"
du -sh "$JOB"/{venv,repo,hf-cache,uv-cache,feature-cache} 2>/dev/null || true
echo "== setup complete. Next: NUM_WORKERS=1 WORKER_ID=0 $HERE/run_worker.sh --limit 1"
