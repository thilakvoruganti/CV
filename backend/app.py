from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import os, uuid, time
import cv2
import numpy as np
from stitcher.pano import build_panorama   # you'll implement using your main.py logic
from sift.compare import run_comparison    # your comp.py logic wrapped as a function

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_origin_regex=r"https://thilakvoruganti\.github\.io.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
os.makedirs("uploads", exist_ok=True)
os.makedirs("static", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

def _save_upload(f: UploadFile, subdir="uploads"):
    os.makedirs(subdir, exist_ok=True)
    ext = os.path.splitext(f.filename)[1].lower()
    name = f"{uuid.uuid4().hex}{ext if ext else '.jpg'}"
    path = os.path.join(subdir, name)
    with open(path, "wb") as out:
        out.write(f.file.read())
    return path

@app.post("/api/stitch")
async def api_stitch(
    images: list[UploadFile] = File(...),
    cylindrical: bool = Form(False),
    focal: float | None = Form(None),
    feature: str = Form("sift"),      # "sift" | "orb"
    max_width: int = Form(1200),
    compare: UploadFile | None = File(None)
):
    start = time.time()
    paths = [_save_upload(img) for img in images]
    compare_path = _save_upload(compare) if compare else None

    out_path, compare_path_out = build_panorama(
        image_paths=paths,
        cylindrical=cylindrical,
        focal=focal,
        feature=feature,
        max_width=max_width,
        compare_path=compare_path,
        out_dir="static"
    )
    return JSONResponse({
        "ok": True,
        "panorama_url": f"/static/{os.path.basename(out_path)}",
        "compare_url": f"/static/{os.path.basename(compare_path_out)}" if compare_path_out else None,
        "elapsed_sec": round(time.time()-start, 2)
    })

@app.post("/api/sift_compare")
async def api_sift_compare(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
    target_height: int = Form(1100),
    octaves: int = Form(5),
    layers: int = Form(4),
    contrast: float = Form(0.01),
    edge: float = Form(10.0),
    root_sift: bool = Form(True),
    ratio: float = Form(0.85),
    cross_check: bool = Form(False),
    ransac_iters: int = Form(8000),
    ransac_thresh: float = Form(6.0),
    prewarp_cyl: bool = Form(False),
    focal: float | None = Form(None)
):
    p1 = _save_upload(image1)
    p2 = _save_upload(image2)
    res = run_comparison(
        p1, p2,
        target_height=target_height,
        octaves=octaves, layers=layers, contrast=contrast, edge=edge,
        root_sift=root_sift, ratio=ratio, cross_check=cross_check,
        ransac_iters=ransac_iters, ransac_thresh=ransac_thresh,
        prewarp_cyl=prewarp_cyl, focal=focal, out_dir="static"
    )
    # res includes counts and saved visualizations
    res["custom_inliers_url"] = f"/static/{os.path.basename(res['custom_inliers'])}"
    res["opencv_inliers_url"] = f"/static/{os.path.basename(res['opencv_inliers'])}"
    del res["custom_inliers"]; del res["opencv_inliers"]
    return JSONResponse(res)
