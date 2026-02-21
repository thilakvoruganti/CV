from __future__ import annotations

import io
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

MAX_FRAME_BYTES = 4 * 1024 * 1024
MAX_FRAME_DIM = 960
SESSION_MAX_FRAMES = 180
SESSION_TTL_SEC = 10 * 60


def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def read_frame_from_bytes(payload: bytes) -> np.ndarray:
    arr = np.frombuffer(payload, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Unable to decode uploaded frame.")
    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest > MAX_FRAME_DIM:
        scale = MAX_FRAME_DIM / float(longest)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return frame


def save_frame(frame: np.ndarray, out_root: str, session_id: str) -> str:
    out_dir = Path(out_root) / "module5" / session_id
    ensure_dir(out_dir)
    out_path = out_dir / f"frame_{uuid.uuid4().hex}.jpg"
    cv2.imwrite(str(out_path), frame)
    return str(out_path)


def parse_roi(payload: str | None, width: int, height: int) -> tuple[int, int, int, int] | None:
    if not payload:
        return None
    try:
        import json

        obj = json.loads(payload)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    keys = ("x", "y", "width", "height")
    if any(k not in obj for k in keys):
        return None
    x = int(max(0, min(width - 1, round(float(obj["x"])))))
    y = int(max(0, min(height - 1, round(float(obj["y"])))))
    w = int(max(1, min(width - x, round(float(obj["width"])))))
    h = int(max(1, min(height - y, round(float(obj["height"])))))
    return x, y, w, h


@dataclass
class TrackSession:
    session_id: str
    mode: str
    created_at: float
    updated_at: float
    frame_count: int
    state: dict[str, Any]


_SESSIONS: dict[str, TrackSession] = {}


def _now() -> float:
    return time.time()


def _cleanup_sessions():
    now = _now()
    to_delete = []
    for sid, sess in _SESSIONS.items():
        if now - sess.updated_at > SESSION_TTL_SEC or sess.frame_count >= SESSION_MAX_FRAMES:
            to_delete.append(sid)
    for sid in to_delete:
        _SESSIONS.pop(sid, None)


def _get_or_create_session(mode: str, session_id: str | None) -> tuple[TrackSession, bool]:
    _cleanup_sessions()
    if session_id:
        sess = _SESSIONS.get(session_id)
        if not sess:
            raise KeyError("Unknown session_id")
        if sess.mode != mode:
            raise ValueError(f"Session mode mismatch: expected {sess.mode}, received {mode}")
        return sess, False
    sid = uuid.uuid4().hex
    sess = TrackSession(
        session_id=sid,
        mode=mode,
        created_at=_now(),
        updated_at=_now(),
        frame_count=0,
        state={},
    )
    _SESSIONS[sid] = sess
    return sess, True


def _finish_if_needed(sess: TrackSession) -> bool:
    timed_out = (_now() - sess.updated_at) > SESSION_TTL_SEC
    over_frames = sess.frame_count >= SESSION_MAX_FRAMES
    finished = timed_out or over_frames
    if finished:
        _SESSIONS.pop(sess.session_id, None)
    return finished


def _aruco_process(frame: np.ndarray, marker_id: int | None) -> tuple[np.ndarray, dict[str, Any]]:
    annotated = frame.copy()
    metadata: dict[str, Any] = {"markers": [], "count": 0}

    aruco_mod = getattr(cv2, "aruco", None)
    if aruco_mod is None:
        cv2.putText(
            annotated,
            "OpenCV aruco module unavailable",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 165, 255),
            2,
            cv2.LINE_AA,
        )
        metadata["warning"] = "cv2.aruco unavailable (opencv-contrib not installed)."
        return annotated, metadata

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dictionary = aruco_mod.getPredefinedDictionary(aruco_mod.DICT_4X4_100)
    detector = aruco_mod.ArucoDetector(dictionary, aruco_mod.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)

    found = []
    if ids is not None:
        for idx, c in zip(ids.flatten().tolist(), corners):
            if marker_id is not None and idx != marker_id:
                continue
            pts = c.reshape(-1, 2).astype(int)
            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))
            found.append({"id": int(idx), "center": {"x": cx, "y": cy}})
            cv2.polylines(annotated, [pts], True, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(
                annotated,
                f"id:{idx}",
                (cx + 6, cy - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

    metadata["markers"] = found
    metadata["count"] = len(found)
    return annotated, metadata


def _redetect_points(gray: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
    x, y, w, h = bbox
    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[y : y + h, x : x + w] = 255
    return cv2.goodFeaturesToTrack(gray, maxCorners=120, qualityLevel=0.01, minDistance=5, mask=mask, blockSize=7)


def _of_process(
    frame: np.ndarray,
    sess: TrackSession,
    roi_payload: str | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    annotated = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    bbox = sess.state.get("bbox")
    if bbox is None:
        parsed = parse_roi(roi_payload, frame.shape[1], frame.shape[0])
        if parsed is None:
            raise ValueError("ROI is required for optical-flow session initialization.")
        bbox = parsed
        sess.state["bbox"] = bbox

    prev_gray = sess.state.get("prev_gray")
    prev_pts = sess.state.get("prev_pts")

    if prev_gray is None or prev_pts is None:
        prev_pts = _redetect_points(gray, bbox)
        sess.state["prev_gray"] = gray
        sess.state["prev_pts"] = prev_pts
    else:
        next_pts, st, _err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, winSize=(21, 21), maxLevel=3)
        good_new = next_pts[st == 1] if next_pts is not None and st is not None else np.empty((0, 2))
        good_old = prev_pts[st == 1] if st is not None else np.empty((0, 2))
        if len(good_new) >= 4 and len(good_old) >= 4:
            disp = np.median(good_new - good_old, axis=0)
            dx, dy = int(round(float(disp[0]))), int(round(float(disp[1])))
            x, y, w, h = bbox
            nx = int(np.clip(x + dx, 0, max(0, frame.shape[1] - w)))
            ny = int(np.clip(y + dy, 0, max(0, frame.shape[0] - h)))
            bbox = (nx, ny, w, h)
            sess.state["bbox"] = bbox
            for p in good_new[:80]:
                cv2.circle(annotated, (int(p[0]), int(p[1])), 2, (255, 220, 0), -1, cv2.LINE_AA)
            updated_pts = good_new.reshape(-1, 1, 2).astype(np.float32)
        else:
            updated_pts = None

        if updated_pts is None or len(updated_pts) < 8:
            updated_pts = _redetect_points(gray, bbox)
        sess.state["prev_pts"] = updated_pts
        sess.state["prev_gray"] = gray

    x, y, w, h = bbox
    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 200, 255), 2, cv2.LINE_AA)
    cv2.putText(
        annotated,
        "Optical Flow ROI",
        (x, max(20, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 200, 255),
        2,
        cv2.LINE_AA,
    )
    metadata = {
        "bbox": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
        "track_points": int(0 if sess.state.get("prev_pts") is None else len(sess.state["prev_pts"])),
    }
    return annotated, metadata


def _read_npz_first_mask(payload: bytes) -> np.ndarray:
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        if not data.files:
            raise ValueError("SAM2 npz contains no arrays.")
        for key in data.files:
            arr = data[key]
            if arr.ndim >= 2:
                if arr.ndim > 2:
                    arr = arr[0]
                return (arr > 0).astype(np.uint8)
    raise ValueError("SAM2 npz contains no 2D masks.")


def _sam2_process(
    frame: np.ndarray,
    sess: TrackSession,
    sam2_payload: bytes | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    mask = sess.state.get("mask")
    if mask is None:
        if not sam2_payload:
            raise ValueError("SAM2 mask npz is required when starting a SAM2 session.")
        mask = _read_npz_first_mask(sam2_payload)
        sess.state["mask"] = mask

    annotated = frame.copy()
    mh, mw = mask.shape[:2]
    fh, fw = frame.shape[:2]
    if (mh, mw) != (fh, fw):
        mask_resized = cv2.resize(mask, (fw, fh), interpolation=cv2.INTER_NEAREST)
    else:
        mask_resized = mask
    mask_bool = mask_resized.astype(bool)

    overlay = annotated.copy()
    overlay[mask_bool] = (30, 180, 255)
    annotated = cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0)

    contours, _ = cv2.findContours((mask_bool.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(annotated, contours, -1, (0, 200, 255), 2, cv2.LINE_AA)
    cv2.putText(
        annotated,
        "SAM2 mask overlay",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 200, 255),
        2,
        cv2.LINE_AA,
    )
    metadata = {
        "mask_pixels": int(mask_bool.sum()),
        "mask_ratio": float(mask_bool.mean()),
    }
    return annotated, metadata


def process_track_request(
    *,
    mode: str,
    frame_payload: bytes,
    out_root: str,
    session_id: str | None = None,
    marker_id: int | None = None,
    roi_payload: str | None = None,
    sam2_payload: bytes | None = None,
) -> dict[str, Any]:
    if len(frame_payload) > MAX_FRAME_BYTES:
        raise ValueError(f"Frame exceeds {MAX_FRAME_BYTES} bytes.")
    mode = (mode or "").strip().lower()
    if mode not in {"aruco", "of", "sam2"}:
        raise ValueError("Unsupported mode. Expected one of: aruco, of, sam2.")

    sess, _created = _get_or_create_session(mode=mode, session_id=session_id)
    frame = read_frame_from_bytes(frame_payload)

    if mode == "aruco":
        annotated, metadata = _aruco_process(frame, marker_id=marker_id)
    elif mode == "of":
        annotated, metadata = _of_process(frame, sess=sess, roi_payload=roi_payload)
    else:
        annotated, metadata = _sam2_process(frame, sess=sess, sam2_payload=sam2_payload)

    sess.frame_count += 1
    sess.updated_at = _now()
    out_path = save_frame(annotated, out_root=out_root, session_id=sess.session_id)
    finished = _finish_if_needed(sess)

    return {
        "session_id": sess.session_id,
        "frame_path": out_path,
        "metadata": metadata,
        "finished": finished,
    }
