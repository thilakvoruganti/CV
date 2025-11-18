from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


# ----------------------------- #
# Helpers shared across tasks   #
# ----------------------------- #

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def _ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


# ----------------------------- #
# Task 1: Gradients + LoG       #
# ----------------------------- #

def _grad_mag_ang(gray: np.ndarray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    mag8 = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    H = ((ang * (179.0 / 360.0))).astype(np.uint8)
    S = np.full_like(H, 255, dtype=np.uint8)
    V = mag8.copy()
    ang_bgr = cv2.cvtColor(cv2.merge([H, S, V]), cv2.COLOR_HSV2BGR)
    return mag8, ang_bgr


def _log_response(gray: np.ndarray, sigma: float, ksize: int):
    blur = cv2.GaussianBlur(gray, (0, 0), sigma, sigma)
    lap = cv2.Laplacian(blur, cv2.CV_32F, ksize=ksize)
    lap_abs = cv2.convertScaleAbs(lap)
    return cv2.normalize(lap_abs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _make_grid(src, mag8, ang_bgr, log8, scale_max=900):
    h, w = src.shape[:2]
    scale = 1.0 if max(h, w) <= scale_max else scale_max / float(max(h, w))

    def _rsz(im):
        return cv2.resize(im, (int(im.shape[1] * scale), int(im.shape[0] * scale)), interpolation=cv2.INTER_AREA)

    a = _rsz(src)
    b = cv2.cvtColor(_rsz(mag8), cv2.COLOR_GRAY2BGR)
    c = _rsz(ang_bgr)
    d = cv2.cvtColor(_rsz(log8), cv2.COLOR_GRAY2BGR)

    min_w = min(a.shape[1], b.shape[1], c.shape[1], d.shape[1])
    a, b, c, d = a[:, :min_w], b[:, :min_w], c[:, :min_w], d[:, :min_w]
    return np.vstack([np.hstack([a, b]), np.hstack([c, d])])


def analyze_gradients(image_paths: Iterable[str], out_dir: str, sigma=1.4, ksize=3):
    _ensure_dir(out_dir)
    results = []
    for path in image_paths:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        name = Path(path).stem
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        mag8, ang_bgr = _grad_mag_ang(gray)
        log8 = _log_response(gray, sigma=sigma, ksize=ksize)
        grid = _make_grid(bgr, mag8, ang_bgr, log8)

        mag_path = os.path.join(out_dir, f"{name}_grad_mag.png")
        ang_path = os.path.join(out_dir, f"{name}_grad_ang.png")
        log_path = os.path.join(out_dir, f"{name}_log.png")
        grid_path = os.path.join(out_dir, f"{name}_grid.png")

        cv2.imwrite(mag_path, mag8)
        cv2.imwrite(ang_path, ang_bgr)
        cv2.imwrite(log_path, log8)
        cv2.imwrite(grid_path, grid)

        edge_density = float((mag8 > 30).mean())
        log_energy = float(np.mean((log8.astype(np.float32) / 255.0) ** 2))

        results.append({
            "image": name,
            "mag_path": mag_path,
            "angle_path": ang_path,
            "log_path": log_path,
            "grid_path": grid_path,
            "edge_density": edge_density,
            "log_energy": log_energy,
        })
    return results


# ----------------------------- #
# Task 2: Edges & Corners       #
# ----------------------------- #

def _gaussian_blur(gray_f32, sigma):
    return cv2.GaussianBlur(gray_f32, (0, 0), sigma, sigma)


def _sobel_gradients(gray_f32):
    gx = cv2.Sobel(gray_f32, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f32, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    ang = (np.rad2deg(np.arctan2(gy, gx)) + 180.0) % 180.0
    return gx, gy, mag, ang


def _nonmax_suppression(mag, ang):
    H, W = mag.shape
    Z = np.zeros_like(mag, dtype=np.float32)
    ang_q = ((ang + 22.5) // 45).astype(np.int32) % 4
    for y in range(1, H - 1):
        for x in range(1, W - 1):
            m = mag[y, x]
            d = ang_q[y, x]
            if d == 0:
                m1, m2 = mag[y, x - 1], mag[y, x + 1]
            elif d == 1:
                m1, m2 = mag[y - 1, x + 1], mag[y + 1, x - 1]
            elif d == 2:
                m1, m2 = mag[y - 1, x], mag[y + 1, x]
            else:
                m1, m2 = mag[y - 1, x - 1], mag[y + 1, x + 1]
            if m >= m1 and m >= m2:
                Z[y, x] = m
    return Z


def _hysteresis(nms, low, high):
    strong = (nms >= high)
    weak = (nms >= low) & ~strong
    out = np.zeros_like(nms, dtype=np.uint8)
    stack = list(zip(*np.nonzero(strong)))
    for (y, x) in stack:
        out[y, x] = 255
    H, W = nms.shape
    while stack:
        y, x = stack.pop()
        for yy in (y - 1, y, y + 1):
            for xx in (x - 1, x, x + 1):
                if 0 <= yy < H and 0 <= xx < W and out[yy, xx] == 0 and weak[yy, xx]:
                    out[yy, xx] = 255
                    stack.append((yy, xx))
    return out


def _edge_keypoints(edge_bin, stride=5):
    ys, xs = np.nonzero(edge_bin)
    keep = (np.arange(len(xs)) % max(1, stride)) == 0
    return list(zip(xs[keep], ys[keep]))


def _harris(gray_f32, k=0.04, win_sigma=1.0):
    Ix = cv2.Sobel(gray_f32, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray_f32, cv2.CV_32F, 0, 1, ksize=3)
    Ixx, Iyy, Ixy = Ix * Ix, Iy * Iy, Ix * Iy
    Sxx = cv2.GaussianBlur(Ixx, (0, 0), win_sigma, win_sigma)
    Syy = cv2.GaussianBlur(Iyy, (0, 0), win_sigma, win_sigma)
    Sxy = cv2.GaussianBlur(Ixy, (0, 0), win_sigma, win_sigma)
    detM = (Sxx * Syy) - (Sxy * Sxy)
    traceM = (Sxx + Syy)
    return detM - k * (traceM ** 2)


def _nms_peaks(resp, radius=6, thresh=0.01):
    rmin, rmax = float(resp.min()), float(resp.max())
    resp_n = (resp - rmin) / (rmax - rmin + 1e-12)
    mask = (resp_n >= thresh).astype(np.uint8)
    k = 2 * radius + 1
    dil = cv2.dilate(resp_n, np.ones((k, k), np.uint8))
    peaks = (resp_n == dil) & (mask > 0)
    ys, xs = np.nonzero(peaks)
    return list(zip(xs, ys)), resp_n


def _draw_pts(img, pts, color, radius=3):
    out = img.copy()
    for (x, y) in pts:
        cv2.circle(out, (int(x), int(y)), radius, color, -1, lineType=cv2.LINE_AA)
    return out


def analyze_edges_and_corners(
    image_paths: Iterable[str],
    out_dir: str,
    sigma=1.0,
    low=20.0,
    high=60.0,
    harris_k=0.04,
    win_sigma=1.0,
    corner_thresh=0.01,
    nms_radius=6,
    edge_stride=5,
):
    _ensure_dir(out_dir)
    results = []
    for path in image_paths:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        name = Path(path).stem
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray_f32 = gray.astype(np.float32)

        blur = _gaussian_blur(gray_f32, sigma)
        _, _, mag, ang = _sobel_gradients(blur)
        mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        nms = _nonmax_suppression(mag_norm, ang)
        edges_bin = _hysteresis(nms, low, high)
        edge_px = int((edges_bin > 0).sum())
        edge_kps = _edge_keypoints(edges_bin, stride=edge_stride)

        R = _harris(gray_f32, k=harris_k, win_sigma=win_sigma)
        corner_pts, Rn = _nms_peaks(R, radius=nms_radius, thresh=corner_thresh)

        edges_binary_path = os.path.join(out_dir, f"{name}_edges_binary.png")
        edges_overlay_path = os.path.join(out_dir, f"{name}_edges_overlay.png")
        edge_kp_path = os.path.join(out_dir, f"{name}_edge_keypoints.png")
        harris_resp_path = os.path.join(out_dir, f"{name}_harris.png")
        corner_overlay_path = os.path.join(out_dir, f"{name}_corner_overlay.png")

        cv2.imwrite(edges_binary_path, edges_bin)
        overlay = bgr.copy()
        overlay[edges_bin > 0] = (255, 200, 0)
        cv2.imwrite(edges_overlay_path, overlay)
        cv2.imwrite(edge_kp_path, _draw_pts(bgr, edge_kps, (255, 200, 0), radius=2))
        cv2.imwrite(harris_resp_path, (np.clip(Rn, 0, 1) * 255).astype(np.uint8))
        cv2.imwrite(corner_overlay_path, _draw_pts(bgr, corner_pts, (0, 0, 255), radius=3))

        results.append({
            "image": name,
            "edges_binary_path": edges_binary_path,
            "edges_overlay_path": edges_overlay_path,
            "edge_keypoints_path": edge_kp_path,
            "harris_response_path": harris_resp_path,
            "corner_overlay_path": corner_overlay_path,
            "edge_pixel_count": edge_px,
            "edge_keypoint_count": len(edge_kps),
            "corner_count": len(corner_pts),
        })
    return results


# ----------------------------- #
# Task 3: Boundary detection    #
# ----------------------------- #

def _auto_canny_thresholds(gray):
    med = np.median(gray)
    lower = int(max(0, 0.66 * med))
    upper = int(min(255, 1.33 * med))
    return lower, upper


def analyze_boundaries(image_paths: Iterable[str], out_dir: str):
    _ensure_dir(out_dir)
    rows = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        name = Path(path).stem
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 1.2)
        t1, t2 = _auto_canny_thresholds(blur)
        edges = cv2.Canny(blur, t1, t2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        cnts, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_cnt, best_rect, best_score, best_stats = None, None, -1, None
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < 0.01 * (h * w):
                continue
            rect = cv2.minAreaRect(cnt)
            score, rectness, center_score, area_frac = _score_contour(cnt, rect, img.shape)
            if score > best_score:
                best_cnt, best_rect, best_score = cnt, rect, score
                best_stats = (score, rectness, center_score, area_frac, area)

        out_paths = {
            "edges_path": os.path.join(out_dir, f"{name}_edges.png"),
            "edges_closed_path": os.path.join(out_dir, f"{name}_edges_closed.png"),
            "bbox_overlay_path": os.path.join(out_dir, f"{name}_bbox.png"),
        }
        cv2.imwrite(out_paths["edges_path"], edges)
        cv2.imwrite(out_paths["edges_closed_path"], edges_closed)

        row = {
            "image": name,
            "found": best_cnt is not None,
            **out_paths,
        }

        if best_rect is not None:
            box_pts = cv2.boxPoints(best_rect).astype(np.int32)
            overlay = img.copy()
            cv2.drawContours(overlay, [box_pts], -1, (0, 0, 255), 3)
            cv2.circle(overlay, tuple(np.int32(best_rect[0])), 5, (0, 255, 0), -1)
            cv2.imwrite(out_paths["bbox_overlay_path"], overlay)
            score, rectness, center_score, area_frac, area = best_stats
            row.update({
                "score": float(f"{score:.4f}"),
                "rectangularity": float(f"{rectness:.4f}"),
                "center_score": float(f"{center_score:.4f}"),
                "area_fraction": float(f"{area_frac:.4f}"),
                "contour_area": float(f"{area:.1f}"),
                "box_width": float(f"{best_rect[1][0]:.1f}"),
                "box_height": float(f"{best_rect[1][1]:.1f}"),
                "angle": float(f"{best_rect[2]:.1f}"),
            })
        rows.append(row)
    return rows


def _score_contour(cnt, rect, image_shape):
    h, w = image_shape[:2]
    image_area = h * w
    (cx, cy), (rw, rh), _ = rect
    box_area = rw * rh
    c_area = cv2.contourArea(cnt)
    rectangularity = c_area / (box_area + 1e-5)
    dist_center = np.linalg.norm(np.array([cx - w / 2, cy - h / 2]))
    center_score = 1 - dist_center / np.linalg.norm([w / 2, h / 2])
    area_frac = c_area / image_area
    score = 0.5 * rectangularity + 0.3 * center_score + 0.2 * area_frac
    return score, rectangularity, center_score, area_frac
