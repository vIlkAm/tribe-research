#!/usr/bin/env bash
# Pre-scale clips so the short side is SHORT px (default 384), same fps and audio.
# TRIBE's V-JEPA2 extractor decodes 64 full-size frames per 0.5 s step on one CPU
# thread and its processor then shrinks them to a 292 px short side (crop 256);
# scaling once up front removes most of that CPU work. Never upscales.
#
#   pod/downscale.sh SRC_DIR DST_DIR [SHORT] [JOBS]
set -euo pipefail
src="${1:?usage: downscale.sh SRC_DIR DST_DIR [SHORT] [JOBS]}"
dst="${2:?usage: downscale.sh SRC_DIR DST_DIR [SHORT] [JOBS]}"
short="${3:-384}"
jobs="${4:-16}"

one() {
  local rel="$1" src="$2" dst="$3" short="$4"
  local out="$dst/$rel"
  [ -s "$out" ] && return 0
  mkdir -p "$(dirname "$out")"
  local vf="scale=w='if(lt(iw,ih),min(iw,$short),-2)':h='if(lt(iw,ih),-2,min(ih,$short))':flags=lanczos"
  ffmpeg -nostdin -loglevel error -y -i "$src/$rel" -vf "$vf" \
    -c:v libx264 -crf 14 -preset veryfast -g 30 -pix_fmt yuv420p -c:a copy "$out.$$.part.mp4" \
    && mv "$out.$$.part.mp4" "$out" \
    || { echo "downscale failed: $rel" >&2; rm -f "$out.$$.part.mp4"; return 1; }
}
export -f one

(cd "$src" && find . -name '*.mp4' -printf '%P\n') \
  | xargs -P "$jobs" -I{} bash -c 'one "$@"' _ {} "$src" "$dst" "$short"
