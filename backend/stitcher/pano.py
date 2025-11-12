# backend/stitcher/pano.py
import os, cv2, numpy as np
from typing import List, Tuple

def _read_images(paths: List[str], max_w: int | None = 1600) -> List[np.ndarray]:
    imgs = []
    for p in sorted(paths):
        im = cv2.imread(p)
        if im is None:
            raise RuntimeError(f"Failed to read: {p}")
        if max_w and im.shape[1] > max_w:
            s = max_w / im.shape[1]
            im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        imgs.append(im)
    if not imgs:
        raise RuntimeError("No input images.")
    return imgs

def _cylindrical_warp(img: np.ndarray, f: float | None = None) -> np.ndarray:
    h, w = img.shape[:2]
    if f is None or f <= 0:
        f = 0.5 * w
    ys, xs = np.indices((h, w))
    X = (xs - w/2) / f
    Y = (ys - h/2) / f
    sinX = np.sin(X); cosX = np.cos(X)
    x_c = f * sinX + w/2
    y_c = f * (Y / (cosX + 1e-8)) + h/2
    map_x = x_c.astype(np.float32); map_y = y_c.astype(np.float32)
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

def _get_fd(name: str = "sift"):
    name = (name or "sift").lower()
    if name == "orb":
        return cv2.ORB_create(nfeatures=4000), "orb", cv2.NORM_HAMMING
    # default SIFT
    return cv2.SIFT_create(nfeatures=4000), "sift", cv2.NORM_L2

def _detect_desc(img: np.ndarray, det) -> Tuple[np.ndarray, np.ndarray]:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kps, desc = det.detectAndCompute(g, None)
    if desc is None or len(kps) < 8:
        raise RuntimeError("Not enough features.")
    pts = np.float32([kp.pt for kp in kps])
    return pts, desc

def _match(descA, descB, norm_type, ratio=0.75):
    bf = cv2.BFMatcher(norm_type)
    raw = bf.knnMatch(descA, descB, k=2)
    good = [m for m, n in raw if m.distance < ratio * n.distance]
    return good

def _H(ptsA, ptsB, matches, rth=4.0):
    if len(matches) < 4:
        raise RuntimeError("Not enough matches for homography")
    src = np.float32([ptsA[m.queryIdx] for m in matches])
    dst = np.float32([ptsB[m.trainIdx] for m in matches])
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, rth, maxIters=5000, confidence=0.999)
    if H is None:
        raise RuntimeError("Homography estimation failed")
    return H

def _warp_pair(base: np.ndarray, img: np.ndarray, H: np.ndarray):
    h1, w1 = base.shape[:2]; h2, w2 = img.shape[:2]
    c2 = np.float32([[0,0],[w2,0],[w2,h2],[0,h2]]).reshape(-1,1,2)
    cw = cv2.perspectiveTransform(c2, H)
    allp = np.vstack((cw, np.float32([[0,0],[w1,0],[w1,h1],[0,h1]]).reshape(-1,1,2)))
    [xmin, ymin] = np.floor(allp.min(axis=0).ravel()).astype(int)
    [xmax, ymax] = np.ceil(allp.max(axis=0).ravel()).astype(int)
    tx, ty = -xmin if xmin < 0 else 0, -ymin if ymin < 0 else 0
    T = np.array([[1,0,tx],[0,1,ty],[0,0,1]], dtype=np.float32)
    out_w, out_h = xmax - xmin, ymax - ymin
    base_w = cv2.warpPerspective(base, T, (out_w, out_h))
    img_w  = cv2.warpPerspective(img,  T @ H, (out_w, out_h))
    maskA = (cv2.cvtColor(base_w, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)
    maskB = (cv2.cvtColor(img_w,  cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)
    return (base_w, maskA), (img_w, maskB)

def _feather(A, Am, B, Bm):
    Amf = Am.astype(np.float32); Bmf = Bm.astype(np.float32)
    wA = cv2.distanceTransform((Amf*255).astype(np.uint8), cv2.DIST_L2, 3)
    wB = cv2.distanceTransform((Bmf*255).astype(np.uint8), cv2.DIST_L2, 3)
    s = (wA + wB + 1e-8)
    wA = (wA / s)[...,None]; wB = (1.0 - wA)
    out = A.astype(np.float32)*wA + B.astype(np.float32)*wB
    return np.clip(out,0,255).astype(np.uint8)

def _crop_black(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(g, 1, 255, cv2.THRESH_BINARY)
    cnts,_ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return img
    x,y,w,h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    return img[y:y+h, x:x+w]

def _side_by_side(a, b, pad=10):
    ha, wa = a.shape[:2]; hb, wb = b.shape[:2]
    h = max(ha, hb)
    a2 = cv2.resize(a, (int(wa*h/ha), h))
    b2 = cv2.resize(b, (int(wb*h/hb), h))
    canvas = np.full((h, a2.shape[1]+pad+b2.shape[1], 3), 255, np.uint8)
    canvas[:, :a2.shape[1]] = a2
    canvas[:, a2.shape[1]+pad:] = b2
    return canvas

def build_panorama(
    image_paths: List[str],
    cylindrical: bool = False,
    focal: float | None = None,
    feature: str = "sift",
    max_width: int = 1600,
    compare_path: str | None = None,
    out_dir: str = "static",
) -> tuple[str, str | None]:
    """
    Returns: (panorama_path, compare_path_out_or_None)
    """
    os.makedirs(out_dir, exist_ok=True)
    det, method, norm_type = _get_fd(feature)

    ims = _read_images(image_paths, max_w=max_width)
    if cylindrical:
        ims = [_cylindrical_warp(im, f=focal) for im in ims]

    pano = ims[0]
    for i in range(1, len(ims)):
        A, B = pano, ims[i]
        ptsA, dA = _detect_desc(A, det)
        ptsB, dB = _detect_desc(B, det)
        good = _match(dB, dA, norm_type)          # note: B → A, we’ll warp B onto A
        H = _H(ptsB, ptsA, good, rth=4.0)
        (Aw, Am), (Bw, Bm) = _warp_pair(A, B, H)
        pano = _feather(Aw, Am, Bw, Bm)
        pano = _crop_black(pano)

    pano_name = "pano_result.jpg"
    pano_path = os.path.join(out_dir, pano_name)
    cv2.imwrite(pano_path, pano)

    comp_out = None
    if compare_path and os.path.exists(compare_path):
        mobile = cv2.imread(compare_path)
        if mobile is not None:
            comp = _side_by_side(pano, mobile)
            comp_name = "pano_compare.jpg"
            comp_out = os.path.join(out_dir, comp_name)
            cv2.imwrite(comp_out, comp)

    return pano_path, comp_out
