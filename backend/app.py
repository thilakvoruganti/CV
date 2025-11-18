from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import os, uuid, time
import cv2
import numpy as np
from stitcher.pano import build_panorama
from sift.compare import run_comparison
from edge.service import analyze_gradients, analyze_edges_and_corners, analyze_boundaries

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

def _save_batch(files: list[UploadFile], batch: str):
    upload_dir = os.path.join("uploads", batch)
    saved = []
    for file in files:
        path = _save_upload(file, subdir=upload_dir)
        original = os.path.splitext(file.filename or "")[0] or os.path.basename(path)
        saved.append({"path": path, "name": original})
    return saved

def _static_url(path: str):
    rel = os.path.relpath(path, "static")
    return f"/static/{rel.replace(os.sep, '/')}"

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

@app.post("/api/edge/gradients")
async def api_edge_gradients(
    images: list[UploadFile] = File(...),
    sigma: float = Form(1.4),
    ksize: int = Form(3),
):
    batch = uuid.uuid4().hex
    saved = _save_batch(images, batch)
    out_dir = os.path.join("static", "edge", batch, "gradients")
    results = analyze_gradients([s["path"] for s in saved], out_dir=out_dir, sigma=sigma, ksize=ksize)
    for item, meta in zip(results, saved):
        item["image"] = meta["name"]
        item["mag_url"] = _static_url(item.pop("mag_path"))
        item["angle_url"] = _static_url(item.pop("angle_path"))
        item["log_url"] = _static_url(item.pop("log_path"))
        item["grid_url"] = _static_url(item.pop("grid_path"))
    return JSONResponse({"ok": True, "results": results, "batch": batch})

@app.post("/api/edge/features")
async def api_edge_features(
    images: list[UploadFile] = File(...),
    sigma: float = Form(1.0),
    low: float = Form(20.0),
    high: float = Form(60.0),
    harris_k: float = Form(0.04),
    win_sigma: float = Form(1.0),
    corner_thresh: float = Form(0.01),
    nms_radius: int = Form(6),
    edge_stride: int = Form(5),
):
    batch = uuid.uuid4().hex
    saved = _save_batch(images, batch)
    out_dir = os.path.join("static", "edge", batch, "features")
    results = analyze_edges_and_corners(
        [s["path"] for s in saved],
        out_dir=out_dir,
        sigma=sigma,
        low=low,
        high=high,
        harris_k=harris_k,
        win_sigma=win_sigma,
        corner_thresh=corner_thresh,
        nms_radius=nms_radius,
        edge_stride=edge_stride,
    )
    for item, meta in zip(results, saved):
        item["image"] = meta["name"]
        item["edges_binary_url"] = _static_url(item.pop("edges_binary_path"))
        item["edges_overlay_url"] = _static_url(item.pop("edges_overlay_path"))
        item["edge_keypoints_url"] = _static_url(item.pop("edge_keypoints_path"))
        item["harris_response_url"] = _static_url(item.pop("harris_response_path"))
        item["corner_overlay_url"] = _static_url(item.pop("corner_overlay_path"))
    return JSONResponse({"ok": True, "results": results, "batch": batch})

@app.post("/api/edge/boundaries")
async def api_edge_boundaries(
    images: list[UploadFile] = File(...),
):
    batch = uuid.uuid4().hex
    saved = _save_batch(images, batch)
    out_dir = os.path.join("static", "edge", batch, "boundaries")
    results = analyze_boundaries([s["path"] for s in saved], out_dir=out_dir)
    for item, meta in zip(results, saved):
        item["image"] = meta["name"]
        item["edges_url"] = _static_url(item.pop("edges_path"))
        item["edges_closed_url"] = _static_url(item.pop("edges_closed_path"))
        item["bbox_overlay_url"] = _static_url(item.pop("bbox_overlay_path"))
    return JSONResponse({"ok": True, "results": results, "batch": batch})
