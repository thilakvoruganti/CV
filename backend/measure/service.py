import os
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def measure_distance(
    image_path: str,
    point1: Tuple[float, float],
    point2: Tuple[float, float],
    fx: float,
    fy: float,
    distance_z: float,
    out_dir: str = "static/measure",
):
    """
    Measure the real-world distance between two points in the image using the provided
    calibrated focal lengths (fx, fy) and the camera-object distance Z.
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image from {image_path}")

    u1, v1 = point1
    u2, v2 = point2

    fx = float(fx)
    fy = float(fy)
    distance_z = float(distance_z)

    if fx <= 0 or fy <= 0:
        raise ValueError("fx and fy must be positive.")
    if distance_z <= 0:
        raise ValueError("distance_z must be positive.")

    # Projection-based length estimate
    length_m = distance_z * np.sqrt(((u1 - u2) / fx) ** 2 + ((v1 - v2) / fy) ** 2)
    length_cm = length_m * 100.0
    pixel_distance = float(np.hypot(u1 - u2, v1 - v2))

    # Annotate and save
    ensure_dir(out_dir)
    name = Path(image_path).stem
    out_path = os.path.join(out_dir, f"{name}_annotated.jpg")
    annot = img.copy()
    p1 = (int(round(u1)), int(round(v1)))
    p2 = (int(round(u2)), int(round(v2)))
    cv2.line(annot, p1, p2, (0, 255, 0), 3, lineType=cv2.LINE_AA)
    label = f"{length_cm:.2f} cm"
    text_pos = (min(p1[0], p2[0]) + 10, min(p1[1], p2[1]) + 30)
    cv2.putText(annot, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, lineType=cv2.LINE_AA)
    cv2.imwrite(out_path, annot)

    return {
        "length_m": float(length_m),
        "length_cm": float(length_cm),
        "pixel_distance": pixel_distance,
        "point1": {"x": float(u1), "y": float(v1)},
        "point2": {"x": float(u2), "y": float(v2)},
        "annotated_path": out_path,
    }
