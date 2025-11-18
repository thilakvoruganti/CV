from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import os, uuid, time, json
import cv2
import numpy as np
from stitcher.pano import build_panorama
from sift.compare import run_comparison
from edge.service import analyze_gradients, analyze_edges_and_corners, analyze_boundaries
from measure.service import measure_distance
from objectdetection.service import (
    linspace_list,
    match_template,
    match_template_library,
    gaussian_blur_fourier,
)

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


def _to_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}

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


@app.post("/api/measure/distance")
async def api_measure_distance(
    image: UploadFile = File(...),
    fx: float = Form(...),
    fy: float = Form(...),
    distance_z: float = Form(...),
    points: str = Form(...),
    actual_cm: float | None = Form(None),
):
    try:
        parsed = json.loads(points)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid points payload: {exc}")

    if isinstance(parsed, dict) and "point1" in parsed and "point2" in parsed:
        pts = [parsed["point1"], parsed["point2"]]
    elif isinstance(parsed, list):
        pts = parsed
    else:
        raise HTTPException(status_code=400, detail="Points must be a list or an object with point1/point2.")

    if len(pts) != 2:
        raise HTTPException(status_code=400, detail="Exactly two points are required.")

    def _coord(obj):
        if not isinstance(obj, dict) or "x" not in obj or "y" not in obj:
            raise HTTPException(status_code=400, detail="Each point must include x and y.")
        return float(obj["x"]), float(obj["y"])

    pt1 = _coord(pts[0])
    pt2 = _coord(pts[1])

    img_path = _save_upload(image)
    out_dir = os.path.join("static", "measure")
    res = measure_distance(
        img_path,
        pt1,
        pt2,
        fx=fx,
        fy=fy,
        distance_z=distance_z,
        out_dir=out_dir,
    )
    res["annotated_url"] = _static_url(res.pop("annotated_path"))

    if actual_cm is not None:
        actual_m = actual_cm / 100.0
        abs_error_m = abs(res["length_m"] - actual_m)
        rel_error_pct = abs_error_m / actual_m * 100.0 if actual_m > 1e-9 else None
        res.update({
            "actual_length_cm": float(actual_cm),
            "actual_length_m": float(actual_m),
            "absolute_error_cm": float(abs_error_m * 100.0),
            "absolute_error_m": float(abs_error_m),
            "relative_error_pct": float(rel_error_pct) if rel_error_pct is not None else None,
        })

    return JSONResponse({"ok": True, **res})


@app.post("/api/object/match")
async def api_object_match(
    scene: UploadFile = File(...),
    template: UploadFile = File(...),
    score_thresh: float = Form(0.7),
    allow_flip: bool = Form(True),
    scale_min: float = Form(0.6),
    scale_max: float = Form(1.3),
    scale_steps: int = Form(12),
):
    scene_path = _save_upload(scene)
    template_path = _save_upload(template)
    steps = max(1, int(scale_steps))
    if scale_min > scale_max:
        scale_min, scale_max = scale_max, scale_min
    scales = linspace_list(float(scale_min), float(scale_max), steps)
    result = match_template(
        scene_path,
        template_path,
        scales=scales,
        allow_flip=_to_bool(allow_flip),
        threshold=score_thresh,
        out_dir=os.path.join("static", "objectdetection"),
    )
    result["annotated_url"] = _static_url(result.pop("annotated_path"))
    return JSONResponse({"ok": True, **result})


@app.post("/api/object/fourier")
async def api_object_fourier(
    image: UploadFile = File(...),
    sigma: float = Form(4.0),
    display_width: int = Form(1024),
    wiener_k: float | None = Form(None),
):
    image_path = _save_upload(image)
    result = gaussian_blur_fourier(
        image_path,
        sigma=sigma,
        display_width=display_width,
        wiener_k=wiener_k,
        out_dir=os.path.join("static", "objectdetection"),
    )
    result["original_url"] = _static_url(result.pop("original_path"))
    result["blurred_url"] = _static_url(result.pop("blurred_path"))
    result["restored_url"] = _static_url(result.pop("restored_path"))
    return JSONResponse({"ok": True, **result})


@app.post("/api/object/library_detect")
async def api_object_library(
    scene: UploadFile = File(...),
    score_thresh: float = Form(0.7),
    blur_kernel: int = Form(35),
    allow_flip: bool = Form(True),
):
    scene_path = _save_upload(scene)
    result = match_template_library(
        scene_path,
        template_root=os.path.join("objectdetection", "templates"),
        threshold=score_thresh,
        blur_kernel=blur_kernel,
        allow_flip=_to_bool(allow_flip),
        out_dir=os.path.join("static", "objectdetection"),
    )
    if result.get("annotated_path"):
        result["annotated_url"] = _static_url(result.pop("annotated_path"))
    else:
        result["annotated_url"] = None
    if result.get("blurred_path"):
        result["blurred_url"] = _static_url(result.pop("blurred_path"))
    else:
        result["blurred_url"] = None
    return JSONResponse({"ok": True, **result})
