#!/usr/bin/env bash
set -e
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

set -eo pipefail

ROOT="${ROOT:-$HOME/problem3}"
TMPDIR="$ROOT/tmp"
mkdir -p "$TMPDIR"
export TMPDIR

FRAMES="$ROOT/frames"
COLMAP="$ROOT/colmap"
SCENE="$ROOT/scene"
GS_REPO="$ROOT/gaussian-splatting"
GS_OUT="$ROOT/gs_output"
RENDERS="$ROOT/renders"
ZIP_OUT="$ROOT/problem3_results.zip"
ITERATIONS="${ITERATIONS:-30000}"
PYTHON="${PYTHON:-python}"

CONDA_ENV_NAME="gs_env"

log() { echo "[$(date +%H:%M:%S)] $*"; }

setup_env() {
  if command -v conda >/dev/null; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    
    if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV_NAME"; then
      log "Creating Conda environment and installing all system dependencies (CUDA, GCC, Colmap)..."
      conda create -y -n "$CONDA_ENV_NAME" -c nvidia -c conda-forge \
        python=3.10 \
        cuda-version=11.8 cuda-toolkit=11.8.0 cuda-nvcc=11.8.89 \
        gcc=11 gxx=11 colmap ffmpeg
      conda activate "$CONDA_ENV_NAME"
      export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
      
      log "Installing Python dependencies..."
      pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
      pip install "numpy<2.0.0" "setuptools<70.0.0" plyfile tqdm scipy opencv-python wheel ninja
      
      if [ ! -d "$GS_REPO" ]; then
        log "Cloning gaussian-splatting for submodules..."
        git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting.git "$GS_REPO"
      fi
      
      log "Building 3DGS C++ extensions..."
      # Explicitly set CUDA arch for Ampere (e.g. A6000, RTX 3090, A10G) to avoid PyTorch detection bugs
      export TORCH_CUDA_ARCH_LIST="8.6"
      pip install --no-build-isolation "$GS_REPO/submodules/simple-knn"
      pip install --no-build-isolation "$GS_REPO/submodules/diff-gaussian-rasterization"
      log "Environment setup complete!"
    else
      conda activate "$CONDA_ENV_NAME"
      export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
      log "conda env activated: $CONDA_ENV_NAME"
    fi
  else
    echo "ERROR: conda is required but not found in PATH."
    exit 1
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

  # Prevent COLMAP from crashing on headless servers (missing X11 display)
  export QT_QPA_PLATFORM=offscreen

  log "COLMAP feature_extractor ..."
  colmap feature_extractor --database_path "$COLMAP/database.db" --image_path "$FRAMES" --SiftExtraction.use_gpu 0 --SiftExtraction.num_threads 8
  log "Downloading vocabulary tree for robust matching..."
  wget -qO "$ROOT/vocab_tree.bin" "https://demuc.de/colmap/vocab_tree_flickr100K_words32K.bin" || true
  if [ -f "$ROOT/vocab_tree.bin" ]; then
    log "COLMAP vocab_tree_matcher ..."
    colmap vocab_tree_matcher \
      --database_path "$COLMAP/database.db" \
      --VocabTreeMatching.vocab_tree_path "$ROOT/vocab_tree.bin" \
      --SiftMatching.use_gpu 0 \
      --SiftMatching.num_threads 8
  else
    log "Fallback to COLMAP sequential_matcher with wide-baseline..."
    colmap sequential_matcher \
      --database_path "$COLMAP/database.db" \
      --SequentialMatching.overlap 20 \
      --SequentialMatching.quadratic_overlap 1 \
      --SiftMatching.use_gpu 0 \
      --SiftMatching.num_threads 8
  fi
  
  log "COLMAP mapper ..."
  mkdir -p "$COLMAP/sparse"
  colmap mapper \
    --database_path "$COLMAP/database.db" \
    --image_path "$FRAMES" \
    --output_path "$COLMAP/sparse"
    
  log "COLMAP image_undistorter (Converting to PINHOLE for 3DGS) ..."
  mkdir -p "$COLMAP/undistorted"
  colmap image_undistorter \
    --image_path "$FRAMES" \
    --input_path "$COLMAP/sparse/0" \
    --output_path "$COLMAP/undistorted" \
    --output_type COLMAP

  log "COLMAP done"
}

prepare_scene() {
  log "Copying undistorted PINHOLE scene for 3DGS ..."
  cp -a "$COLMAP/undistorted/images/." "$SCENE/images/"
  mkdir -p "$SCENE/sparse/0"
  cp "$COLMAP/undistorted/sparse/cameras.bin" "$SCENE/sparse/0/"
  cp "$COLMAP/undistorted/sparse/images.bin" "$SCENE/sparse/0/"
  cp "$COLMAP/undistorted/sparse/points3D.bin" "$SCENE/sparse/0/"
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

cleanup() {
  log "Cleaning up intermediate files to save quota..."

  # Aggressively wipe package caches to reclaim temporary storage
  log "Clearing Conda and pip caches..."
  conda clean --all -y >/dev/null 2>&1 || true
  rm -rf ~/.cache/pip

  # Remove temporary build directories but preserve intermediate results
  rm -rf "$ROOT/problem3_data.zip" "$TMPDIR"
  log "Cleaned up caches and tmp dirs. Intermediate results preserved!"
}

print_summary() {
  cat <<EOF

================================================================================
 DONE
================================================================================
  Pipeline finished and cleanup complete!
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
cleanup
print_summary
