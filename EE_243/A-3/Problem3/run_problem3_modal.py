import os
import shutil
import subprocess
from pathlib import Path

import modal

app = modal.App("ee243-problem3")

vol = modal.Volume.from_name("ee243-results", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:11.8.0-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "colmap", "zip", "unzip", "ffmpeg", "libgl1-mesa-glx", "libglib2.0-0", "build-essential", "gcc", "g++", "clang", "ninja-build")
    .pip_install("torch==2.1.2+cu118", "torchvision==0.16.2+cu118", index_url="https://download.pytorch.org/whl/cu118")
    .pip_install("numpy<2", "wheel", "ninja", "setuptools", "plyfile", "tqdm", "scipy", "opencv-python")
    .run_commands(
        "git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting.git /gs",
        "pip install --no-build-isolation /gs/submodules/simple-knn",
        "pip install --no-build-isolation /gs/submodules/diff-gaussian-rasterization"
    )
)

@app.function(image=image, gpu="A10G", timeout=7200, volumes={"/results_vol": vol})
def run_pipeline(zip_bytes: bytes, iterations: int = 30000) -> str:
    ROOT = Path("/workspace")
    ROOT.mkdir(exist_ok=True, parents=True)
    os.chdir(ROOT)
    
    zip_path = ROOT / "problem3_data.zip"
    zip_path.write_bytes(zip_bytes)
    
    print("Unzipping data...")
    subprocess.run(["unzip", "-q", "-o", str(zip_path), "-d", str(ROOT)], check=True)
    
    # Run Colmap
    COLMAP = ROOT / "colmap"
    FRAMES = ROOT / "frames"
    COLMAP.mkdir(exist_ok=True)
    db = COLMAP / "database.db"
    sparse = COLMAP / "sparse"
    
    print("Running COLMAP feature_extractor...")
    subprocess.run(["colmap", "feature_extractor", "--database_path", str(db), "--image_path", str(FRAMES)], check=True)
    print("Running COLMAP exhaustive_matcher...")
    subprocess.run(["colmap", "exhaustive_matcher", "--database_path", str(db)], check=True)
    print("Running COLMAP mapper...")
    sparse.mkdir(exist_ok=True)
    subprocess.run(["colmap", "mapper", "--database_path", str(db), "--image_path", str(FRAMES), "--output_path", str(sparse)], check=True)
    
    # Prepare 3DGS scene
    print("Preparing 3DGS scene...")
    SCENE = ROOT / "scene"
    images = SCENE / "images"
    dst_sparse = SCENE / "sparse" / "0"
    images.mkdir(parents=True, exist_ok=True)
    dst_sparse.mkdir(parents=True, exist_ok=True)
    
    for src in FRAMES.glob("*.jpg"):
        shutil.copy(src, images / src.name)
        
    for f in ["cameras.bin", "images.bin", "points3D.bin"]:
        src = sparse / "0" / f
        if src.exists():
            shutil.copy(src, dst_sparse / f)
    
    # Train 3DGS
    print(f"Training 3DGS for {iterations} iterations...")
    GS_OUT = ROOT / "gs_output"
    subprocess.run(["python", "/gs/train.py", "-s", str(SCENE), "-m", str(GS_OUT), "--iterations", str(iterations)], check=True)
    
    # Render
    print("Rendering 3DGS...")
    subprocess.run(["python", "/gs/render.py", "-m", str(GS_OUT)], check=True)
    
    renders_found = list(GS_OUT.rglob("renders"))
    if not renders_found:
        raise RuntimeError("No renders folder found!")
    RENDER_SRC = renders_found[0]
    
    RENDERS = ROOT / "renders"
    RENDERS.mkdir(exist_ok=True)
    
    import cv2
    for p in sorted(RENDER_SRC.iterdir()):
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        img = cv2.imread(str(p))
        if img is not None:
            out = RENDERS / f"{p.stem}.jpg"
            cv2.imwrite(str(out), img)
            
    # Zip results
    print("Packaging results...")
    ZIP_OUT = Path("/results_vol") / "problem3_results.zip"
    ply_files = sorted(GS_OUT.rglob("point_cloud.ply"))
    ply_arg = str(ply_files[-1]) if ply_files else ""
    
    if ply_arg:
        subprocess.run(["zip", "-rq", str(ZIP_OUT), "renders/", "colmap/sparse/0/", ply_arg], check=True)
    else:
        subprocess.run(["zip", "-rq", str(ZIP_OUT), "renders/", "colmap/sparse/0/"], check=True)
        
    vol.commit()
    return "done"
