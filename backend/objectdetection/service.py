from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def linspace_list(start: float, stop: float, steps: int) -> List[float]:
    if steps <= 1:
        return [float(start)]
    return np.linspace(start, stop, steps).tolist()


def rotate_keep_all(img: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 1e-6:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - w / 2
    M[1, 2] += (nH / 2) - h / 2
    return cv2.warpAffine(img, M, (nW, nH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _normalize_scales(scales: Iterable[float] | None) -> List[float]:
    if not scales:
        return linspace_list(0.6, 1.3, 15)
    vals = [float(s) for s in scales if s is not None]
    return [s for s in vals if s > 0]


def _match_best(
    scene_gray: np.ndarray,
    template_gray: np.ndarray,
    scales: Sequence[float],
    angles: Sequence[float],
    method: int,
) -> dict | None:
    best = None
    H, W = scene_gray.shape[:2]
    tpl = cv2.GaussianBlur(template_gray, (3, 3), 0)
    img = cv2.GaussianBlur(scene_gray, (3, 3), 0)
    for angle in angles:
        tpl_rot = rotate_keep_all(tpl, angle) if abs(angle) > 1e-6 else tpl
        for scale in scales:
            tw = max(8, int(round(tpl_rot.shape[1] * scale)))
            th = max(8, int(round(tpl_rot.shape[0] * scale)))
            if tw >= W or th >= H:
                continue
            tpl_scaled = cv2.resize(
                tpl_rot,
                (tw, th),
                interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC,
            )
            result = cv2.matchTemplate(img, tpl_scaled, method)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if (best is None) or (max_val > best["score"]):
                best = {
                    "score": float(max_val),
                    "top_left": (int(max_loc[0]), int(max_loc[1])),
                    "size": (int(tw), int(th)),
                    "scale": float(scale),
                    "angle": float(angle),
                }
    return best


def match_template(
    scene_path: str,
    template_path: str,
    *,
    scales: Iterable[float] | None = None,
    allow_flip: bool = True,
    method: int = cv2.TM_CCOEFF_NORMED,
    out_dir: str = "static/objectdetection",
    threshold: float = 0.7,
) -> dict:
    scene_gray = cv2.imread(scene_path, cv2.IMREAD_GRAYSCALE)
    if scene_gray is None:
        raise RuntimeError(f"Unable to read scene image: {scene_path}")
    template_gray = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template_gray is None:
        raise RuntimeError(f"Unable to read template image: {template_path}")

    angles = [0.0, 180.0] if allow_flip else [0.0]
    scale_list = _normalize_scales(scales)
    best = _match_best(scene_gray, template_gray, scale_list, angles, method)
    if best is None:
        raise RuntimeError("Template size exceeds scene dimensions at all scales.")

    scene_bgr = cv2.imread(scene_path, cv2.IMREAD_COLOR)
    if scene_bgr is None:
        raise RuntimeError(f"Unable to read scene image (color): {scene_path}")
    x, y = best["top_left"]
    w, h = best["size"]
    color = (0, 255, 0) if best["score"] >= threshold else (0, 165, 255)
    cv2.rectangle(scene_bgr, (x, y), (x + w, y + h), color, 2)
    label = f"{Path(template_path).stem} · {best['score']:.2f} @ {best['scale']:.2f}x"
    cv2.putText(scene_bgr, label, (x, max(15, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    ensure_dir(out_dir)
    annotated_path = os.path.join(out_dir, f"match_{Path(scene_path).stem}_{Path(template_path).stem}.jpg")
    cv2.imwrite(annotated_path, scene_bgr)

    return {
        "best_score": best["score"],
        "scale": best["scale"],
        "angle": best["angle"],
        "location": {"x": x, "y": y, "width": w, "height": h},
        "annotated_path": annotated_path,
        "threshold": float(threshold),
        "template_name": Path(template_path).name,
    }


def load_template_library(root: str | Path) -> List[Path]:
    root_path = Path(root)
    if not root_path.exists():
        return []
    files: List[Path] = []
    for path in root_path.rglob("*"):
        if path.suffix.lower() in IMG_EXTS and path.is_file():
            files.append(path)
    return sorted(files)


def blur_region(img: np.ndarray, rect: Tuple[int, int, int, int], kernel: int) -> None:
    x, y, w, h = rect
    k = max(3, kernel | 1)  # ensure odd
    roi = img[y : y + h, x : x + w]
    if roi.size == 0:
        return
    blurred = cv2.GaussianBlur(roi, (k, k), 0)
    img[y : y + h, x : x + w] = blurred


def match_template_library(
    scene_path: str,
    template_root: str,
    *,
    scales: Iterable[float] | None = None,
    allow_flip: bool = True,
    method: int = cv2.TM_CCOEFF_NORMED,
    threshold: float = 0.7,
    blur_kernel: int = 35,
    out_dir: str = "static/objectdetection",
) -> dict:
    scene_gray = cv2.imread(scene_path, cv2.IMREAD_GRAYSCALE)
    scene_bgr = cv2.imread(scene_path, cv2.IMREAD_COLOR)
    if scene_gray is None or scene_bgr is None:
        raise RuntimeError("Unable to read scene image for template library matching.")
    templates = load_template_library(template_root)
    if not templates:
        return {
            "templates": [],
            "annotated_path": None,
            "blurred_path": None,
            "threshold": float(threshold),
            "matches": [],
        }

    angles = [0.0, 180.0] if allow_flip else [0.0]
    scale_list = _normalize_scales(scales)
    annotated = scene_bgr.copy()
    blurred = scene_bgr.copy()
    color_cycle = [
        (0, 255, 0),
        (0, 200, 255),
        (255, 180, 0),
        (255, 0, 125),
        (125, 255, 125),
        (180, 120, 255),
    ]

    template_results = []
    passing_matches = []
    for idx, tpl_path in enumerate(templates):
        template_gray = cv2.imread(str(tpl_path), cv2.IMREAD_GRAYSCALE)
        if template_gray is None:
            template_results.append({"template": tpl_path.name, "error": "unreadable"})
            continue
        best = _match_best(scene_gray, template_gray, scale_list, angles, method)
        if best is None:
            template_results.append({"template": tpl_path.name, "error": "template larger than scene"})
            continue
        x, y = best["top_left"]
        w, h = best["size"]
        color = color_cycle[idx % len(color_cycle)]
        passed = best["score"] >= threshold
        cv2.rectangle(
            annotated,
            (x, y),
            (x + w, y + h),
            color if passed else (0, 165, 255),
            2 if passed else 1,
        )
        cv2.putText(
            annotated,
            f"{tpl_path.stem} {best['score']:.2f}",
            (x, max(15, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
        info = {
            "template": tpl_path.name,
            "score": float(best["score"]),
            "scale": best["scale"],
            "angle": best["angle"],
            "top_left": {"x": x, "y": y},
            "size": {"width": w, "height": h},
            "passed": passed,
        }
        template_results.append(info)
        if passed:
            passing_matches.append(info)
            blur_region(blurred, (x, y, w, h), blur_kernel)

    ensure_dir(out_dir)
    ann_path = os.path.join(out_dir, f"library_{Path(scene_path).stem}.jpg")
    blur_path = os.path.join(out_dir, f"library_{Path(scene_path).stem}_blurred.jpg")
    cv2.imwrite(ann_path, annotated)
    cv2.imwrite(blur_path, blurred)

    return {
        "templates": template_results,
        "annotated_path": ann_path,
        "blurred_path": blur_path,
        "threshold": float(threshold),
        "matches": passing_matches,
    }


def gaussian_psf(ksize: int, sigma: float) -> np.ndarray:
    ax = np.arange(-(ksize // 2), ksize // 2 + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    psf = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    psf /= np.sum(psf)
    return psf


def edgetaper(img: np.ndarray, psf: np.ndarray, strength: float = 0.25, iters: int = 2) -> np.ndarray:
    H, W = img.shape

    def tukey(n: int, alpha: float = 0.6):
        x = np.linspace(0, 1, n, dtype=np.float32)
        w = np.ones_like(x)
        e = alpha / 2.0
        m1 = x < e
        w[m1] = 0.5 * (1 + np.cos(np.pi * (2 * x[m1] / alpha - 1)))
        m2 = x > 1 - e
        w[m2] = 0.5 * (1 + np.cos(np.pi * (2 * (1 - x[m2]) / alpha - 1)))
        return w

    apod = np.outer(tukey(H), tukey(W)).astype(np.float32)
    out = img.astype(np.float32)
    for _ in range(iters):
        blurred = cv2.filter2D(out, -1, psf, borderType=cv2.BORDER_REFLECT101).astype(np.float32)
        out = apod * out + (1 - apod) * (strength * blurred + (1 - strength) * out)
    return out.astype(img.dtype)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    mse = np.mean((a - b) ** 2)
    if mse <= 1e-12:
        return 99.0
    return 20 * np.log10(255.0) - 10 * np.log10(mse)


def suggest_wiener_k(sigma: float) -> float:
    if sigma <= 2:
        return 0.003
    if sigma <= 4:
        return 0.004
    if sigma <= 6:
        return 0.006
    if sigma <= 10:
        return 0.008
    return 0.012


def gaussian_blur_fourier(
    image_path: str,
    *,
    sigma: float = 4.0,
    display_width: int = 1024,
    wiener_k: float | None = None,
    out_dir: str = "static/objectdetection",
) -> dict:
    src = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if src is None:
        raise RuntimeError(f"Unable to read image for Fourier filtering: {image_path}")

    h, w = src.shape
    scale = display_width / float(w)
    img = cv2.resize(src, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA) if scale != 1.0 else src

    ksize = int(max(3, (6 * sigma))) | 1
    blurred = cv2.GaussianBlur(img, (ksize, ksize), sigma)

    psf = gaussian_psf(ksize, sigma)
    padded = np.zeros_like(img, dtype=np.float32)
    ph, pw = psf.shape
    padded[:ph, :pw] = psf
    padded = np.roll(np.roll(padded, -ph // 2, axis=0), -pw // 2, axis=1)
    H = np.fft.fft2(padded)

    if wiener_k is None:
        wiener_k = suggest_wiener_k(sigma)

    tap = edgetaper(blurred, psf, strength=0.25, iters=2)
    F = np.fft.fft2(tap)
    restored = np.fft.ifft2(F * np.conj(H) / (np.abs(H) ** 2 + wiener_k)).real
    restored = np.clip(restored, 0, 255).astype(np.uint8)

    ensure_dir(out_dir)
    src_path = os.path.join(out_dir, f"fourier_{Path(image_path).stem}_orig.jpg")
    blur_path = os.path.join(out_dir, f"fourier_{Path(image_path).stem}_blurred.jpg")
    rec_path = os.path.join(out_dir, f"fourier_{Path(image_path).stem}_restored.jpg")

    cv2.imwrite(src_path, img)
    cv2.imwrite(blur_path, blurred)
    cv2.imwrite(rec_path, restored)

    return {
        "sigma": float(sigma),
        "ksize": int(ksize),
        "wiener_k": float(wiener_k),
        "psnr_blur": psnr(img, blurred),
        "psnr_restore": psnr(img, restored),
        "original_path": src_path,
        "blurred_path": blur_path,
        "restored_path": rec_path,
    }
