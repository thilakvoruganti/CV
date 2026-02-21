from __future__ import annotations

import csv
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import mediapipe as mp  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    mp = None


def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def _read_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img


def _parse_points(payload: str) -> list[tuple[float, float]]:
    arr = json.loads(payload)
    if not isinstance(arr, list):
        raise ValueError("Points must be a list.")
    pts = []
    for item in arr:
        if not isinstance(item, dict) or "x" not in item or "y" not in item:
            raise ValueError("Each point must include x and y.")
        pts.append((float(item["x"]), float(item["y"])))
    return pts


def _triangulate(
    left: list[tuple[float, float]],
    right: list[tuple[float, float]],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    baseline: float,
) -> tuple[list[dict[str, float]], list[float]]:
    points_3d: list[dict[str, float]] = []
    disparities: list[float] = []
    for (xl, yl), (xr, _yr) in zip(left, right):
        d = float(xl - xr)
        disparities.append(d)
        safe_d = abs(d) if abs(d) > 1e-6 else 1e-6
        z = fx * baseline / safe_d
        x = (xl - cx) * z / fx
        y = (yl - cy) * z / fy
        points_3d.append({"x": float(x), "y": float(y), "z": float(z)})
    return points_3d, disparities


def _edge_lengths(points_3d: list[dict[str, float]], closed: bool) -> tuple[list[float], list[float]]:
    if len(points_3d) < 2:
        return [], []
    pairs = [(i, i + 1) for i in range(len(points_3d) - 1)]
    if closed and len(points_3d) > 2:
        pairs.append((len(points_3d) - 1, 0))
    edges_m: list[float] = []
    for i, j in pairs:
        p = points_3d[i]
        q = points_3d[j]
        dist = float(np.sqrt((p["x"] - q["x"]) ** 2 + (p["y"] - q["y"]) ** 2 + (p["z"] - q["z"]) ** 2))
        edges_m.append(dist)
    edges_cm = [e * 100.0 for e in edges_m]
    return edges_m, edges_cm


def _annotate_points(img: np.ndarray, points: list[tuple[float, float]], closed: bool) -> np.ndarray:
    out = img.copy()
    pts_i = np.array([[int(round(x)), int(round(y))] for x, y in points], dtype=np.int32)
    for idx, (x, y) in enumerate(pts_i.tolist()):
        cv2.circle(out, (x, y), 6, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(out, str(idx), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    if len(pts_i) >= 2:
        cv2.polylines(out, [pts_i], closed, (0, 200, 255), 2, cv2.LINE_AA)
    return out


def stereo_measure(
    *,
    left_image_path: str,
    right_image_path: str,
    points_left_payload: str,
    points_right_payload: str,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    baseline: float,
    closed: bool,
    out_dir: str,
) -> dict[str, Any]:
    left_pts = _parse_points(points_left_payload)
    right_pts = _parse_points(points_right_payload)
    if len(left_pts) != len(right_pts):
        raise ValueError("Left/right point counts must match.")
    if len(left_pts) < 2:
        raise ValueError("At least 2 correspondence points are required.")
    if fx <= 0 or fy <= 0 or baseline <= 0:
        raise ValueError("fx, fy, and baseline must be positive.")

    session_id = uuid.uuid4().hex
    out_root = Path(out_dir) / "module7" / session_id
    ensure_dir(out_root)

    points_3d, disparities = _triangulate(left_pts, right_pts, fx, fy, cx, cy, baseline)
    edges_m, edges_cm = _edge_lengths(points_3d, closed=closed)

    depths = [p["z"] for p in points_3d]
    summary = {
        "mean_depth_m": float(np.mean(depths)),
        "min_depth_m": float(np.min(depths)),
        "max_depth_m": float(np.max(depths)),
        "perimeter_cm": float(np.sum(edges_cm)) if edges_cm else 0.0,
    }

    left_img = _read_image(left_image_path)
    right_img = _read_image(right_image_path)
    left_annot = _annotate_points(left_img, left_pts, closed=closed)
    right_annot = _annotate_points(right_img, right_pts, closed=closed)

    left_out = out_root / "left_annotated.jpg"
    right_out = out_root / "right_annotated.jpg"
    cv2.imwrite(str(left_out), left_annot)
    cv2.imwrite(str(right_out), right_annot)

    return {
        "session_id": session_id,
        "points_3d": points_3d,
        "disparities": [float(d) for d in disparities],
        "edges_m": [float(e) for e in edges_m],
        "edges_cm": [float(e) for e in edges_cm],
        "closed": bool(closed),
        "summary": summary,
        "point_count": len(points_3d),
        "annotated_left_path": str(left_out),
        "annotated_right_path": str(right_out),
    }


def _draw_pose_landmarks(
    frame_bgr: np.ndarray,
    result: Any,
    mp_drawing: Any,
    mp_holistic: Any,
):
    if result.pose_landmarks:
        mp_drawing.draw_landmarks(frame_bgr, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if result.left_hand_landmarks:
        mp_drawing.draw_landmarks(frame_bgr, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if result.right_hand_landmarks:
        mp_drawing.draw_landmarks(frame_bgr, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)


def pose_track_video(
    *,
    video_path: str,
    sample_stride: int,
    max_frames: int | None,
    detection_confidence: float,
    tracking_confidence: float,
    out_dir: str,
) -> dict[str, Any]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video.")

    input_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    session_id = uuid.uuid4().hex
    out_root = Path(out_dir) / "module7" / "pose" / session_id
    ensure_dir(out_root)
    out_video_path = out_root / "annotated.mp4"
    out_csv_path = out_root / "landmarks.csv"

    writer = cv2.VideoWriter(
        str(out_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(1.0, input_fps / max(1, sample_stride)),
        (width, height),
    )

    mp_holistic = mp.solutions.holistic if mp is not None else None
    mp_drawing = mp.solutions.drawing_utils if mp is not None else None
    holistic = None
    if mp_holistic is not None:
        holistic = mp_holistic.Holistic(
            min_detection_confidence=float(detection_confidence),
            min_tracking_confidence=float(tracking_confidence),
        )

    processed = 0
    sampled = 0
    rows: list[dict[str, Any]] = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % sample_stride != 0:
            frame_idx += 1
            continue
        sampled += 1
        if max_frames is not None and sampled > max_frames:
            break

        ts = frame_idx / input_fps if input_fps > 0 else 0.0
        annotated = frame.copy()

        if holistic is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = holistic.process(rgb)
            _draw_pose_landmarks(annotated, result, mp_drawing, mp_holistic)
            pose_count = len(result.pose_landmarks.landmark) if result.pose_landmarks else 0
            lhand_count = len(result.left_hand_landmarks.landmark) if result.left_hand_landmarks else 0
            rhand_count = len(result.right_hand_landmarks.landmark) if result.right_hand_landmarks else 0
        else:
            pose_count = 0
            lhand_count = 0
            rhand_count = 0
            cv2.putText(
                annotated,
                "mediapipe not installed",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 165, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            annotated,
            f"frame {frame_idx}",
            (20, max(60, height - 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(annotated)

        rows.append(
            {
                "frame_index": frame_idx,
                "timestamp_sec": float(ts),
                "pose_landmarks": pose_count,
                "left_hand_landmarks": lhand_count,
                "right_hand_landmarks": rhand_count,
            }
        )
        processed += 1
        frame_idx += 1

    cap.release()
    writer.release()
    if holistic is not None:
        holistic.close()

    with open(out_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["frame_index", "timestamp_sec", "pose_landmarks", "left_hand_landmarks", "right_hand_landmarks"]
        writer_obj = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer_obj.writeheader()
        writer_obj.writerows(rows)

    duration_sec = float(processed / max(1e-6, (input_fps / max(1, sample_stride))))
    return {
        "session_id": session_id,
        "annotated_video_path": str(out_video_path),
        "csv_path": str(out_csv_path),
        "processed_frames": int(processed),
        "input_frames": int(total_frames),
        "fps": float(input_fps),
        "width": int(width),
        "height": int(height),
        "duration_sec": float(duration_sec),
        "sample_stride": int(sample_stride),
    }
