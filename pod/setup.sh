#!/usr/bin/env bash
# Idempotent TRIBE v2 environment setup on a RunPod pod.
#
# Everything persistent lives on the network volume so that workers 1..N (and
# restarts) reuse the same venv, weights and feature cache:
#
#   $JOB/repo/            facebookresearch/tribev2 pinned to $TRIBE_COMMIT
#   $JOB/venv/            Python venv
#   $JOB/hf-cache/        HF weights (TRIBE ckpt, V-JEPA2, Wav2Vec-BERT, Llama-3.2-3B)
#   $JOB/uv-cache/        uvx env for whisperx (TRIBE shells out to `uvx whisperx`)
#   $JOB/feature-cache/   TRIBE's per-modality extracted features (the expensive part)
#   $JOB/code/            this repo's pod/ + tools/ (rsync'd from the server)
#   $JOB/videos/ outputs/ logs/ manifest.jsonl
#
# Run on the FIRST pod only; later pods just `source $JOB/code/pod/env.sh`.
# Requires HF_TOKEN in the environment (RunPod secret), with access granted to
# meta-llama/Llama-3.2-3B on huggingface.co.
set -euo pipefail

JOB="${JOB:-/workspace/tribe-job}"
TRIBE_COMMIT="${TRIBE_COMMIT:-af58661791a351a448a489042a28f6c37e1c14b7}"

# shellcheck source=env.sh
source "$(dirname "$0")/env.sh"
mkdir -p "$JOB"/{videos,outputs,logs,hf-cache,uv-cache,feature-cache}

echo "== system packages (container disk; reinstall after every pod restart)"
apt-get update -qq
apt-get install -y -qq git ffmpeg rsync >/dev/null

echo "== python"
PY="${PYTHON:-python3}"
"$PY" -c 'import sys; assert sys.version_info >= (3, 11), f"TRIBE needs Python >=3.11, have {sys.version}"'

if [ ! -f "$JOB/venv/bin/activate" ]; then
  # --system-site-packages lets us reuse the image's CUDA torch if it is in range.
  "$PY" -m venv --system-site-packages "$JOB/venv"
fi
# shellcheck disable=SC1091
source "$JOB/venv/bin/activate"
pip install -q --upgrade pip

echo "== tribev2 @ $TRIBE_COMMIT"
if [ ! -d "$JOB/repo/.git" ]; then
  git clone -q https://github.com/facebookresearch/tribev2.git "$JOB/repo"
fi
git -C "$JOB/repo" fetch -q origin
git -C "$JOB/repo" checkout -q "$TRIBE_COMMIT"
pip install -q -e "$JOB/repo"
pip install -q uv   # provides `uvx`, which TRIBE uses to run whisperx in its own env

python - <<'EOF'
import torch, torchvision
v = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
assert (2, 5) <= v < (2, 7), f"torch {torch.__version__} outside TRIBE's >=2.5.1,<2.7 pin"
assert torch.cuda.is_available(), "CUDA not available in this venv"
print(f"torch {torch.__version__} cuda {torch.version.cuda} on {torch.cuda.get_device_name(0)}")
EOF

echo "== spacy model"
python -m spacy download -q en_core_web_sm >/dev/null 2>&1 || python -m spacy download en_core_web_sm

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN not set: Llama-3.2-3B (gated) will fail to download" >&2
  exit 1
fi

echo "== pre-download weights (first run only; cached on the volume)"
python - <<'EOF'
import os, time
from tribev2 import TribeModel
t = time.time()
TribeModel.from_pretrained("facebook/tribev2", cache_folder=os.environ["TRIBE_CACHE"])
print(f"TRIBE loaded in {time.time() - t:.0f}s")
EOF

du -sh "$JOB"/{venv,repo,hf-cache,uv-cache,feature-cache} 2>/dev/null || true
echo "== setup complete. Next: pod/run_worker.sh with --limit 1 on one real clip."
