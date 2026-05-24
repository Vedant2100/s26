#!/usr/bin/env bash
# =============================================================================
# EE243 Problem 3 — ONE script for crisgc. Run everything with:
#
#   cd ~/problem3 && ./run_problem3.sh
#
# First time only (from your Mac):
#   scp problem3_data.zip run_problem3.sh vbork001@crisgc:~/problem3/
#
# On crisgc (get a GPU node, then run the script):
#   salloc --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=6:00:00
#   cd ~/problem3 && chmod +x run_problem3.sh && ./run_problem3.sh
#
# When done, from your Mac:
#   scp vbork001@crisgc:~/problem3/problem3_results.zip .
#   Upload problem3_results.zip to Google Drive, then finish Problem3.ipynb in Colab.
# =============================================================================

set -euo pipefail

ROOT="${ROOT:-$HOME/problem3}"
FRAMES="$ROOT/frames"
COLMAP="$ROOT/colmap"
SCENE="$ROOT/scene"
GS_REPO="$ROOT/gaussian-splatting"
GS_OUT="$ROOT/gs_output"
RENDERS="$ROOT/renders"
ZIP_OUT="$ROOT/problem3_results.zip"
ITERATIONS="${ITERATIONS:-30000}"
PYTHON="${PYTHON:-python}"

# --- edit if your cluster needs these ---
CONDA_ENV="/data/AmitRoyChowdhury/vedant/envs/ee243-3dgs"

log() { echo "[$(date +%H:%M:%S)] $*"; }

setup_env() {
  if command -v conda >/dev/null; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
    log "conda env: $CONDA_ENV"
  fi
}

ensure_data() {
  cd "$ROOT"
  mkdir -p "$FRAMES" "$COLMAP/sparse" "$SCENE/images" "$SCENE/sparse/0" "$RENDERS"

  if [ -f "$ROOT/problem3_data.zip" ] && [ "$(ls -1 "$FRAMES"/*.jpg 2>/dev/null | wc -l)" -eq 0 ]; then
    log "Unzipping problem3_data.zip ..."
    unzip -o "$ROOT/problem3_data.zip" -d "$ROOT"
  fi

  n_frames=$(ls -1 "$FRAMES"/*.jpg 2>/dev/null | wc -l || true)
  if [ "$n_frames" -eq 0 ]; then
    echo "ERROR: No frames in $FRAMES"
    echo "  Put problem3_data.zip here and re-run, or unzip frames/ manually."
    exit 1
  fi
  log "Found $n_frames frames"
}

run_colmap() {
  if [ -f "$COLMAP/sparse/0/cameras.bin" ]; then
    log "COLMAP already done — skipping"
    return
  fi
  command -v colmap >/dev/null || {
    echo "ERROR: colmap not found. Set MODULE_COLMAP at top of script, or: module load colmap"
    exit 1
  }
  log "COLMAP feature_extractor ..."
  colmap feature_extractor --database_path "$COLMAP/database.db" --image_path "$FRAMES"
  log "COLMAP exhaustive_matcher ..."
  colmap exhaustive_matcher --database_path "$COLMAP/database.db"
  log "COLMAP mapper ..."
  colmap mapper \
    --database_path "$COLMAP/database.db" \
    --image_path "$FRAMES" \
    --output_path "$COLMAP/sparse"
  log "COLMAP done ($(ls "$COLMAP/sparse/0"/*.bin | wc -l) model files)"
}

prepare_scene() {
  cp "$FRAMES"/*.jpg "$SCENE/images/"
  cp "$COLMAP/sparse/0/"*.bin "$SCENE/sparse/0/"
  log "3DGS scene ready at $SCENE"
}

run_train() {
  if [ -d "$GS_OUT/point_cloud" ] && [ -n "$(find "$GS_OUT/point_cloud" -name 'point_cloud.ply' 2>/dev/null | head -1)" ]; then
    log "3DGS training output exists — skipping train (delete gs_output/ to re-train)"
    return
  fi
  command -v nvidia-smi >/dev/null && nvidia-smi -L || {
    echo "ERROR: No GPU visible. Run inside salloc --gres=gpu:1 ..."
    exit 1
  }
  if [ ! -d "$GS_REPO" ]; then
    log "Cloning gaussian-splatting ..."
    git clone https://github.com/graphdeco-inria/gaussian-splatting.git "$GS_REPO"
  fi
  log "3DGS train ($ITERATIONS iterations) — this takes ~1-2 hours ..."
  "$PYTHON" "$GS_REPO/train.py" -s "$SCENE" -m "$GS_OUT" --iterations "$ITERATIONS"
  log "Training done"
}

run_render() {
  if [ "$(ls -1 "$RENDERS"/*.jpg 2>/dev/null | wc -l)" -gt 0 ]; then
    log "Renders already in $RENDERS — skipping render (delete renders/ to re-render)"
    return
  fi
  log "3DGS render ..."
  "$PYTHON" "$GS_REPO/render.py" -m "$GS_OUT"

  RENDER_SRC=$(find "$GS_OUT" -type d -path '*/renders' | head -1)
  [ -n "$RENDER_SRC" ] || { echo "ERROR: no renders/ under $GS_OUT"; exit 1; }

  mkdir -p "$RENDERS"
  "$PYTHON" - <<PY
from pathlib import Path
import cv2
src = Path("$RENDER_SRC")
dst = Path("$RENDERS")
for p in sorted(src.iterdir()):
    if p.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        continue
    img = cv2.imread(str(p))
    if img is None:
        continue
    out = dst / f"{p.stem}.jpg"
    cv2.imwrite(str(out), img)
print(f"Wrote {len(list(dst.glob('*.jpg')))} renders")
PY
  log "Renders saved to $RENDERS"
}

package_results() {
  log "Packaging $ZIP_OUT ..."
  rm -f "$ZIP_OUT"
  cd "$ROOT"
  zip -rq "$ZIP_OUT" renders/ colmap/sparse/0/
  ply=$(find gs_output/point_cloud -name 'point_cloud.ply' 2>/dev/null | sort | tail -1)
  [ -n "$ply" ] && zip -rq "$ZIP_OUT" "$ply"
  log "Created $(du -h "$ZIP_OUT" | cut -f1) zip"
}

print_summary() {
  n_render=$(ls -1 "$RENDERS"/*.jpg 2>/dev/null | wc -l || true)
  ply=$(find "$GS_OUT" -name 'point_cloud.ply' 2>/dev/null | sort | tail -1)
  num_gaussians="?"
  if [ -n "$ply" ]; then
    num_gaussians=$(grep '^element vertex' "$ply" | awk '{print $3}')
  fi

  cat <<EOF

================================================================================
 DONE
================================================================================
  frames:      $(ls -1 "$FRAMES"/*.jpg | wc -l)
  renders:     $n_render
  iterations:  $ITERATIONS
  gaussians:   $num_gaussians
  results zip: $ZIP_OUT

 Next (on your Mac):
   scp vbork001@crisgc:$ZIP_OUT .

 Then in Colab Problem3.ipynb: upload zip to Drive, unzip under /content/Problem3/,
 re-run the last few cells for PSNR/SSIM + side-by-side video + REPORT.
================================================================================
EOF
}

# --- main ---
setup_env
ensure_data
run_colmap
prepare_scene
run_train
run_render
package_results
print_summary
