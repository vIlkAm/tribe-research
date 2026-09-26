#!/usr/bin/env bash
# Publish a saved performance model as a PRIVATE GitHub release of this repo (one command per model version).
#
#   tools/publish_model.sh TAG MODEL_DIR FEATURIZER_JSON [--dry-run]
#   e.g. tools/publish_model.sh model-prelim-v1 results/models/prelim-v1 \
#            results/features/prelim-v1/features.featurizer.json
#
# 1. refuses unless gh is logged in and the repo is PRIVATE, and unless the tag is new;
# 2. tools/package_model.py -> results/releases/TAG/{TAG.tar.gz, RELEASE_MANIFEST.json, NOTES.md}
#    (refuses media files, lockbox clips in references/training, sha mismatches);
# 3. lists the tarball and refuses any video/audio extension;
# 4. gh release create TAG <tarball> <manifest> --notes-file NOTES.md (not a draft, never public).
# --dry-run stops after step 3. Data bundles are a separate release (docs/MODEL.md, "Publishing").
set -euo pipefail
cd "$(dirname "$0")/.."
[ $# -ge 3 ] || { sed -n '2,13p' "$0"; exit 2; }
TAG=$1 MODEL_DIR=$2 FZ=$3 DRY=${4:-}
PY=${PY:-.venv/bin/python}

gh auth status >/dev/null 2>&1 || { echo "gh is not authenticated: stop" >&2; exit 1; }
VIS=$(gh repo view --json visibility -q .visibility)
[ "$VIS" = "PRIVATE" ] || { echo "repo visibility is $VIS, not PRIVATE: stop" >&2; exit 1; }
if gh release view "$TAG" >/dev/null 2>&1; then echo "release $TAG already exists: pick a new tag" >&2; exit 1; fi

OUT=results/releases/$TAG
"$PY" tools/package_model.py --tag "$TAG" --model-dir "$MODEL_DIR" --featurizer "$FZ" --out "$OUT"
TGZ=$OUT/$TAG.tar.gz
echo "--- tarball contents ($TGZ)"
tar tzf "$TGZ" | tee "$OUT/CONTENTS.txt"
if grep -Eiq '\.(mp4|mov|mkv|webm|avi|m4v|wav|mp3|m4a|aac|flac|ogg|opus|wma)$' "$OUT/CONTENTS.txt"; then
  echo "video/audio file in the tarball: stop" >&2; exit 1
fi
[ "$DRY" = "--dry-run" ] && { echo "dry run: not published"; exit 0; }
TITLE="$TAG: $("$PY" -c "import json,sys; m=json.load(open(sys.argv[1])); print(m['model_version'], '-', m['status'])" "$OUT/RELEASE_MANIFEST.json")"
gh release create "$TAG" "$TGZ" "$OUT/RELEASE_MANIFEST.json" --title "$TITLE" --notes-file "$OUT/NOTES.md"
gh release view "$TAG" --json url -q .url
