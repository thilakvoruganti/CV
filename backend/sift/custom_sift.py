#!/usr/bin/env python3
import cv2
import numpy as np
import time
import math
import os
import argparse

# ============================================================
#                 Utility: I/O and scale-matching
# ============================================================
def ensure_dir(p):
    d = os.path.dirname(p)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def cylindrical_warp_gray(img_gray, f=None):
    """Project grayscale image onto a cylinder to align pano projections."""
    h, w = img_gray.shape[:2]
    if f is None or f <= 0:
        f = 0.5 * w
    ys, xs = np.indices((h, w), dtype=np.float32)
    X = (xs - w / 2) / f
    Y = (ys - h / 2) / f
    sinX = np.sin(X)
    cosX = np.cos(X)
    x_c = f * sinX + w / 2
    y_c = f * (Y / (cosX + 1e-8)) + h / 2
    map_x = x_c.astype(np.float32)
    map_y = y_c.astype(np.float32)
    return cv2.remap(img_gray, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT)

def resize_to_same_height_imgs(img1, img2, target_height=None, hard_cap=1400, min_cap=700):
    """Resize already-loaded grayscale images to a common height."""
    h1, _ = img1.shape[:2]
    h2, _ = img2.shape[:2]
    if target_height is None:
        target_height = max(min_cap, min(min(h1, h2), hard_cap))

    def _resize(img, th):
        h, w = img.shape[:2]
        s = th / float(h)
        return cv2.resize(img, (int(round(w * s)), th), interpolation=cv2.INTER_AREA), s

    img1_r, s1 = _resize(img1, target_height)
    img2_r, s2 = _resize(img2, target_height)
    print(f"[Scale match] Target height: {target_height}px | scales: img1 x{s1:.3f}, img2 x{s2:.3f}")
    print(f"[Scale match] New sizes: img1 {img1_r.shape[::-1]}, img2 {img2_r.shape[::-1]}")
    return img1_r, img2_r

def kps_to_image_coords(kps):
    """Convert octave-level keypoints back to base image coordinates."""
    coords = np.empty((len(kps), 2), dtype=np.float64)
    for i, kp in enumerate(kps):
        scale = 2.0 ** kp.octave
        coords[i, 0] = kp.x * scale
        coords[i, 1] = kp.y * scale
    return coords

def load_and_resize_to_same_height(path1, path2, target_height=None, hard_cap=1400, min_cap=700):
    """Load images in grayscale and resize BOTH to the same height.
       If target_height is None: choose min(original heights, hard_cap), but not < min_cap.
    """
    img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        raise FileNotFoundError(f"Could not read one of the images:\n  {path1}\n  {path2}")

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    if target_height is None:
        target_height = max(min_cap, min(min(h1, h2), hard_cap))

    def resize_to_h(img, th):
        h, w = img.shape[:2]
        scale = th / float(h)
        return cv2.resize(img, (int(round(w * scale)), th), interpolation=cv2.INTER_AREA), scale

    img1_r, s1 = resize_to_h(img1, target_height)
    img2_r, s2 = resize_to_h(img2, target_height)

    print(f"[Scale match] Target height: {target_height}px | scales: img1 x{s1:.3f}, img2 x{s2:.3f}")
    print(f"[Scale match] New sizes: img1 {img1_r.shape[::-1]}, img2 {img2_r.shape[::-1]}")
    return img1_r, img2_r, s1, s2

# ============================================================
#                 Custom SIFT (Mini-SIFT)
# ============================================================
class Keypoint:
    def __init__(self, x, y, octave, layer):
        self.x = x
        self.y = y
        self.octave = octave
        self.layer = layer
        self.angle = 0.0
        self.descriptor = None

def gaussian_pyramid(image, num_octaves=4, num_layers=3, initial_sigma=1.6):
    """Build Gaussian/DoG pyramids plus per-level gradients (float32)."""
    gauss_pyr = []
    gradx_pyr = []
    grady_pyr = []
    dog_pyr = []
    k = 2 ** (1.0 / num_layers)

    img_base = image.astype(np.float32) / 255.0
    base = cv2.GaussianBlur(img_base, (0, 0), sigmaX=initial_sigma, sigmaY=initial_sigma)
    base_sigma = initial_sigma

    for _ in range(num_octaves):
        gauss = [base]
        sigmas = [base_sigma]
        for layer in range(1, num_layers + 3):
            sigma_prev = sigmas[-1]
            sigma_total = base_sigma * (k ** layer)
            sigma_inc = math.sqrt(max(1e-8, sigma_total**2 - sigma_prev**2))
            img = cv2.GaussianBlur(gauss[-1], (0, 0), sigmaX=sigma_inc, sigmaY=sigma_inc)
            gauss.append(img)
            sigmas.append(sigma_total)
        gauss_pyr.append(gauss)

        gx_levels = []
        gy_levels = []
        for g in gauss:
            gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
            gx_levels.append(gx)
            gy_levels.append(gy)
        gradx_pyr.append(gx_levels)
        grady_pyr.append(gy_levels)

        dog = [cv2.subtract(gauss[i+1], gauss[i]) for i in range(len(gauss)-1)]
        dog_pyr.append(dog)

        nxt = cv2.resize(gauss[0], (gauss[0].shape[1]//2, gauss[0].shape[0]//2), interpolation=cv2.INTER_NEAREST)
        base = nxt
        base_sigma *= 2

    return gauss_pyr, dog_pyr, gradx_pyr, grady_pyr

def find_keypoints(dog_pyr, num_octaves=4, contrast_threshold=0.02, edge_threshold=10):
    keypoints = []
    for o in range(min(num_octaves, len(dog_pyr))):
        dog = dog_pyr[o]
        for i in range(1, len(dog)-1):
            prev = dog[i-1]
            curr = dog[i]
            nxt  = dog[i+1]
            rows, cols = curr.shape
            # Border-safe range
            for y in range(1, rows-1):
                row_prev = prev[y-1:y+2]
                row_curr = curr[y-1:y+2]
                row_next = nxt[y-1:y+2]
                for x in range(1, cols-1):
                    val = curr[y, x]
                    if abs(val) < contrast_threshold:
                        continue
                    nb = np.array([row_prev[:, x-1:x+2],
                                   row_curr[:, x-1:x+2],
                                   row_next[:, x-1:x+2]])
                    if not (val == nb.max() or val == nb.min()):
                        continue
                    # Hessian edge filter (approx)
                    Dxx = curr[y, x+1] + curr[y, x-1] - 2 * val
                    Dyy = curr[y+1, x] + curr[y-1, x] - 2 * val
                    Dxy = ((curr[y+1, x+1] - curr[y+1, x-1]) - (curr[y-1, x+1] - curr[y-1, x-1])) / 4.0
                    tr = Dxx + Dyy
                    det = Dxx * Dyy - Dxy * Dxy
                    if det <= 0:
                        continue
                    r = edge_threshold
                    if (tr * tr) / det > ((r + 1)**2) / r:
                        continue
                    keypoints.append(Keypoint(float(x), float(y), o, i))
    return keypoints

def assign_orientation(keypoints, gradx_pyr, grady_pyr, num_bins=36):
    oriented = []
    two_pi = 2.0 * math.pi
    for kp in keypoints:
        gx = gradx_pyr[kp.octave][kp.layer]
        gy = grady_pyr[kp.octave][kp.layer]
        h, w = gx.shape[:2]
        x = int(round(kp.x))
        y = int(round(kp.y))
        if y <= 1 or y >= h-2 or x <= 1 or x >= w-2:
            continue
        radius = 8
        sigma_w = 1.5 * radius
        hist = np.zeros(num_bins, dtype=np.float32)
        for dy in range(-radius, radius + 1):
            ny = y + dy
            if ny <= 0 or ny >= h-1:
                continue
            for dx in range(-radius, radius + 1):
                nx = x + dx
                if nx <= 0 or nx >= w-1:
                    continue
                gxv = gx[ny, nx]
                gyv = gy[ny, nx]
                mag = math.hypot(gxv, gyv)
                if mag == 0:
                    continue
                theta = math.atan2(gyv, gxv) % two_pi
                wgt = math.exp(-(dx*dx + dy*dy) / (2 * (sigma_w**2)))
                binf = theta * num_bins / two_pi
                base = math.floor(binf)
                b0 = int(base) % num_bins
                b1 = (b0 + 1) % num_bins
                t = binf - base
                hist[b0] += wgt * mag * (1 - t)
                hist[b1] += wgt * mag * t
        if hist.size == 0:
            continue
        hist = np.convolve(np.r_[hist[-1], hist, hist[0]], np.array([1/3, 1/3, 1/3], dtype=np.float32), mode='valid')
        max_val = hist.max() if hist.size else 0
        if max_val <= 0:
            continue
        for i, v in enumerate(hist):
            if v >= 0.8 * max_val:
                ang = (i + 0.5) * (360.0 / num_bins)
                new_kp = Keypoint(kp.x, kp.y, kp.octave, kp.layer)
                new_kp.angle = ang % 360.0
                oriented.append(new_kp)
    return oriented

def compute_descriptors(keypoints, gradx_pyr, grady_pyr, desc_width=4, desc_bins=8, win_size=16):
    two_pi = 2.0 * math.pi
    cell = win_size / desc_width
    half = win_size / 2.0
    sigma_desc = 0.5 * win_size
    for kp in keypoints:
        gx = gradx_pyr[kp.octave][kp.layer]
        gy = grady_pyr[kp.octave][kp.layer]
        h, w = gx.shape[:2]
        x0 = kp.x
        y0 = kp.y
        if y0 < 1 or y0 > h-2 or x0 < 1 or x0 > w-2:
            kp.descriptor = None
            continue
        angle = math.radians(kp.angle)
        ca, sa = math.cos(angle), math.sin(angle)
        hist = np.zeros((desc_width, desc_width, desc_bins), dtype=np.float32)
        for yy in np.arange(-half + 0.5, half, 1.0):
            for xx in np.arange(-half + 0.5, half, 1.0):
                rx = xx * ca + yy * sa
                ry = -xx * sa + yy * ca
                xr = x0 + rx
                yr = y0 + ry
                ix = int(round(xr))
                iy = int(round(yr))
                if ix <= 0 or ix >= w-1 or iy <= 0 or iy >= h-1:
                    continue
                gxv = gx[iy, ix]
                gyv = gy[iy, ix]
                mag = math.hypot(gxv, gyv)
                if mag == 0:
                    continue
                theta = (math.atan2(gyv, gxv) - angle) % two_pi
                wgt = math.exp(-(rx*rx + ry*ry) / (2 * (sigma_desc**2)))
                bx = (rx + half) / cell
                by = (ry + half) / cell
                if not (0 <= bx < desc_width and 0 <= by < desc_width):
                    continue
                bo = theta * desc_bins / two_pi
                bx0 = int(math.floor(bx))
                by0 = int(math.floor(by))
                bo0 = int(math.floor(bo))
                dbx = bx - bx0
                dby = by - by0
                dbo = bo - bo0
                for dy in (0, 1):
                    wy = (1 - dby) if dy == 0 else dby
                    byi = by0 + dy
                    if byi < 0 or byi >= desc_width:
                        continue
                    for dx in (0, 1):
                        wx = (1 - dbx) if dx == 0 else dbx
                        bxi = bx0 + dx
                        if bxi < 0 or bxi >= desc_width:
                            continue
                        for do in (0, 1):
                            wo = (1 - dbo) if do == 0 else dbo
                            boi = (bo0 + do) % desc_bins
                            hist[byi, bxi, boi] += wgt * mag * wx * wy * wo
        desc = hist.flatten()
        n = np.linalg.norm(desc)
        if n > 1e-8:
            desc /= n
        desc = np.clip(desc, 0, 0.2)
        n = np.linalg.norm(desc)
        if n > 1e-8:
            desc /= n
        kp.descriptor = desc.astype(np.float32)

def rootsift(D):
    """Convert SIFT descriptors to RootSIFT to improve distinctiveness."""
    if D is None or len(D) == 0:
        return D
    eps = 1e-12
    L1 = D.sum(axis=1, keepdims=True) + eps
    D = D / L1
    return np.sqrt(D, dtype=np.float32)

def match_descriptors(desc1, desc2, ratio=0.75, cross_check=True):
    matches = []
    if len(desc1) == 0 or len(desc2) == 0:
        return matches
    # forward
    idx2_best = -np.ones(len(desc1), dtype=int)
    idx2_second = -np.ones(len(desc1), dtype=int)
    d_best = np.full(len(desc1), np.inf, dtype=float)
    d_second = np.full(len(desc1), np.inf, dtype=float)

    for i, d1 in enumerate(desc1):
        dists = np.linalg.norm(desc2 - d1, axis=1)
        order = np.argsort(dists)
        if len(order) < 2:
            continue
        idx2_best[i] = order[0]
        idx2_second[i] = order[1]
        d_best[i] = dists[idx2_best[i]]
        d_second[i] = dists[idx2_second[i]]

    forward = [(i, idx2_best[i]) for i in range(len(desc1)) if d_best[i] < ratio * d_second[i]]

    if not cross_check:
        return forward

    # backward cross-check
    idx1_best = -np.ones(len(desc2), dtype=int)
    d_best_b = np.full(len(desc2), np.inf, dtype=float)
    for j, d2 in enumerate(desc2):
        dists = np.linalg.norm(desc1 - d2, axis=1)
        order = np.argsort(dists)
        if len(order) == 0:
            continue
        idx1_best[j] = order[0]
        d_best_b[j] = dists[idx1_best[j]]

    mutual = []
    for (i, j) in forward:
        if idx1_best[j] == i:
            mutual.append((i, j))
    return mutual

def _normalize_points(pts):
    """
    Hartley normalization: shift to mean 0 and scale so mean distance = sqrt(2).
    """
    pts = np.asarray(pts, dtype=np.float64)
    if pts.size == 0:
        return np.eye(3, dtype=np.float64), pts
    mean = pts.mean(axis=0)
    d = np.sqrt(((pts - mean) ** 2).sum(axis=1))
    s = np.sqrt(2.0) / (d.mean() + 1e-12)
    T = np.array([[s, 0, -s * mean[0]],
                  [0, s, -s * mean[1]],
                  [0, 0, 1.0]], dtype=np.float64)
    pts_h = np.c_[pts, np.ones(len(pts))]
    pts_n_h = (T @ pts_h.T).T
    return T, pts_n_h[:, :2]

def _compute_homography_dlt_norm(pairs, ptsA, ptsB):
    """
    DLT with Hartley normalization and denormalization.
    """
    if len(pairs) < 4:
        return None
    idxA = [i for (i, _) in pairs]
    idxB = [j for (_, j) in pairs]
    A = np.asarray([ptsA[i] for i in idxA], dtype=np.float64)
    B = np.asarray([ptsB[j] for j in idxB], dtype=np.float64)

    TA, A_n = _normalize_points(A)
    TB, B_n = _normalize_points(B)

    M = []
    for (p, q) in zip(A_n, B_n):
        x, y = p
        xp, yp = q
        M.append([-x, -y, -1, 0, 0, 0, x * xp, y * xp, xp])
        M.append([0, 0, 0, -x, -y, -1, x * yp, y * yp, yp])
    M = np.asarray(M, dtype=np.float64)
    _, _, Vt = np.linalg.svd(M)
    Hn = Vt[-1].reshape(3, 3)
    if abs(Hn[2, 2]) > 1e-12:
        Hn /= Hn[2, 2]

    TB_inv = np.linalg.inv(TB)
    H = TB_inv @ Hn @ TA
    if abs(H[2, 2]) > 1e-12:
        H /= H[2, 2]
    return H

def compute_homography_dlt(pairs, ptsA, ptsB):
    if len(pairs) < 4:
        return None
    return _compute_homography_dlt_norm(pairs, ptsA, ptsB)

def _msac_score(errs, tau):
    """MSAC score: sum(min(err_i, tau)), lower is better."""
    return np.minimum(errs, tau).sum()

def _refit_H_from_inliers(pairs, ptsA, ptsB):
    if len(pairs) < 4:
        return None
    return compute_homography_dlt(pairs, ptsA, ptsB)

def _grid_spread_sample(idxA, idxB, ptsA, ptsB, W=4, H=3, rng=None):
    """Pick a spatially spread 4-tuple of pair indices."""
    if rng is None:
        rng = np.random.default_rng()
    n = len(idxA)
    if n < 4:
        return None
    A = ptsA[idxA]
    B = ptsB[idxB]

    def bucketize(P, w, h):
        x = P[:, 0]
        y = P[:, 1]
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
        eps = 1e-6
        gx = np.clip(((x - xmin) / (xmax - xmin + eps) * w).astype(int), 0, w-1)
        gy = np.clip(((y - ymin) / (ymax - ymin + eps) * h).astype(int), 0, h-1)
        return gy * w + gx

    bA = bucketize(A, W, H)
    bB = bucketize(B, W, H)
    order = rng.permutation(n)
    chosen = []
    seen = set()
    for k in order:
        key = (bA[k], bB[k])
        if key not in seen:
            seen.add(key)
            chosen.append(k)
        if len(chosen) >= 8:
            break

    if len(chosen) < 4:
        pool = rng.choice(n, size=min(n, 8), replace=False)
    else:
        pool = np.array(chosen, dtype=int)

    if len(pool) < 4:
        return None

    def quad_area(P):
        if P.shape[0] < 3:
            return 0.0
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(P)
            return hull.volume
        except Exception:
            d = np.linalg.norm(P[None, :] - P[:, None], axis=2)
            return float(d.mean())

    best = None
    best_score = -1.0
    trials = min(32, len(pool) * 2)
    for _ in range(trials):
        cand = rng.choice(pool, size=4, replace=False)
        score = quad_area(A[cand]) + quad_area(B[cand])
        if score > best_score:
            best_score = score
            best = cand
    return best

def _forward_reprojection_error(H, ptsA, ptsB):
    """Forward-only reprojection error: H(A) vs B."""
    N = ptsA.shape[0]
    if N == 0:
        return np.empty((0,), dtype=np.float64)
    A = np.c_[ptsA, np.ones((N, 1))]
    HA = (H @ A.T).T
    HA /= np.clip(HA[:, 2:], 1e-12, None)
    err = np.sum((ptsB - HA[:, :2]) ** 2, axis=1)  # squared Euclidean error
    return err


def ransac_homography(pairs, ptsA, ptsB,
                      max_iters=8000, thresh=8.0, early_stop=True,
                      lo_iters=2, seed=None):
    """
    Improved LO-RANSAC for homography with symmetric transfer error and MSAC scoring.
    Now uses correct thresholding, no aggressive tightening, and consensus count.
    """
    if len(pairs) < 4:
        return None, []

    rng = np.random.default_rng(seed)
    ptsA = np.asarray(ptsA, dtype=np.float64)
    ptsB = np.asarray(ptsB, dtype=np.float64)
    idxA_all = np.array([i for (i, _) in pairs], dtype=int)
    idxB_all = np.array([j for (_, j) in pairs], dtype=int)
    Aall = ptsA[idxA_all]
    Ball = ptsB[idxB_all]
    tau = float(thresh) ** 2  # squared pixel threshold

    best_H = None
    best_inliers_idx = None
    best_num_inliers = 0
    best_score = np.inf

    def needed_iters(p_inlier, sample_size=4, conf=0.99):
        p_inlier = max(min(p_inlier, 0.9999), 1e-6)
        num = np.log(1.0 - conf)
        denom = np.log(1.0 - p_inlier ** sample_size)
        if denom == 0:
            denom = -1e-12
        return max(500, min(int(np.ceil(num / denom)), 30000))

    iters = max_iters
    for k in range(iters):
        subset_idx = _grid_spread_sample(idxA_all, idxB_all, ptsA, ptsB, rng=rng)
        if subset_idx is None or len(subset_idx) < 4:
            subset_idx = rng.choice(len(pairs), size=4, replace=False)
        subset_pairs = [pairs[t] for t in subset_idx]

        H = compute_homography_dlt(subset_pairs, ptsA, ptsB)
        if H is None:
            continue

        errs = _forward_reprojection_error(H, Aall, Ball)

        inl_mask = errs < tau
        score = _msac_score(errs, tau)
        num_inliers = np.count_nonzero(inl_mask)

        if num_inliers > best_num_inliers or (num_inliers == best_num_inliers and score < best_score):
            best_H = H
            best_inliers_idx = np.where(inl_mask)[0]
            best_num_inliers = num_inliers
            best_score = score
            inl_ratio = num_inliers / len(pairs)
            iters = min(iters, needed_iters(inl_ratio))
            if early_stop and num_inliers >= 120 and inl_ratio > 0.4:
                break

    if best_H is None or best_inliers_idx is None or len(best_inliers_idx) < 4:
        return None, []

    # Local optimization (no threshold tightening)
    inliers_idx = best_inliers_idx
    H = best_H
    for _ in range(max(0, lo_iters)):
        inlier_pairs = [pairs[t] for t in inliers_idx]
        H = _refit_H_from_inliers(inlier_pairs, ptsA, ptsB)
        if H is None:
            break
        errs = _forward_reprojection_error(H, Aall, Ball)
        inliers_idx = np.where(errs < tau)[0]
        if len(inliers_idx) < 4:
            break

    final_pairs = [pairs[t] for t in inliers_idx]
    H = _refit_H_from_inliers(final_pairs, ptsA, ptsB)
    return H, final_pairs

def ransac_homography_v2(pairs, ptsA, ptsB,
                         max_iters=10000,
                         thresh=12.0,
                         early_stop=True,
                         lo_iters=3,
                         tighten=0.8,
                         sample_method='grid',
                         seed=None):
    """
    Improved RANSAC with forward reprojection error, local optimization, and spatial sampling.
    """
    if len(pairs) < 4:
        return None, []

    rng = np.random.default_rng(seed)
    idxA_all = np.array([i for i, _ in pairs], dtype=int)
    idxB_all = np.array([j for _, j in pairs], dtype=int)
    Aall = ptsA[idxA_all]
    Ball = ptsB[idxB_all]
    tau = thresh ** 2

    def forward_error(H):
        A = np.c_[Aall, np.ones((len(Aall), 1))]
        HA = (H @ A.T).T
        HA /= np.clip(HA[:, 2:], 1e-12, None)
        return np.sum((Ball - HA[:, :2]) ** 2, axis=1)

    def msac_score(errs):
        return np.minimum(errs, tau).sum()

    best_H = None
    best_inliers = []
    best_score = np.inf
    iters = max_iters

    def sample():
        if sample_method == 'grid':
            return _grid_spread_sample(idxA_all, idxB_all, ptsA, ptsB, rng=rng)
        return rng.choice(len(pairs), size=4, replace=False)

    def needed_iters(p, s=4, conf=0.99):
        p = np.clip(p, 1e-6, 0.9999)
        return int(np.ceil(np.log(1 - conf) / np.log(1 - p**s)))

    for k in range(iters):
        subset_idx = sample()
        if subset_idx is None or len(subset_idx) < 4:
            continue
        H = compute_homography_dlt([pairs[i] for i in subset_idx], ptsA, ptsB)
        if H is None:
            continue
        errs = forward_error(H)
        mask = errs < tau
        score = msac_score(errs)
        if score < best_score:
            best_score = score
            best_H = H
            best_inliers = np.where(mask)[0]
            inlier_ratio = mask.mean()
            iters = max(iters, needed_iters(inlier_ratio))
            if early_stop and inlier_ratio > 0.4 and len(best_inliers) > 100:
                break

    if best_H is None or len(best_inliers) < 4:
        return None, []

    cur_tau = tau
    inliers = best_inliers
    H = best_H
    for _ in range(lo_iters):
        cur_tau = max(1.0, tighten * cur_tau)
        inlier_pairs = [pairs[i] for i in inliers]
        H = compute_homography_dlt(inlier_pairs, ptsA, ptsB)
        if H is None:
            break
        errs = forward_error(H)
        inliers = np.where(errs < cur_tau)[0]
        if len(inliers) < 4:
            break

    final_pairs = [pairs[i] for i in inliers]
    H = compute_homography_dlt(final_pairs, ptsA, ptsB)
    return H, final_pairs




def draw_inlier_matches(img1, img2, kps1, kps2, pairs, out_path):
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    vis[:h1, :w1, 0] = img1
    vis[:h1, :w1, 1] = img1
    vis[:h1, :w1, 2] = img1
    vis[:h2, w1:w1+w2, 0] = img2
    vis[:h2, w1:w1+w2, 1] = img2
    vis[:h2, w1:w1+w2, 2] = img2
    for (i, j) in pairs:
        p1 = (int(round(kps1[i].x)), int(round(kps1[i].y)))
        p2 = (int(round(kps2[j].x)) + w1, int(round(kps2[j].y)))
        color = (0, 255, 0)
        cv2.circle(vis, p1, 3, color, 1)
        cv2.circle(vis, p2, 3, color, 1)
        cv2.line(vis, p1, p2, color, 1)
    ensure_dir(out_path)
    cv2.imwrite(out_path, vis)

# ============================================================
#                          Main
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="SIFT + RANSAC from scratch with OpenCV comparison")
    ap.add_argument("--image1", default="data/mobile_panorama/pano_horizontal.jpeg")
    ap.add_argument("--image2", default="output/final_panorama.jpg")
    ap.add_argument("--target_height", type=int, default=0, help="Resize both images to this height (0=auto)")
    ap.add_argument("--hard_cap", type=int, default=1400)
    ap.add_argument("--min_cap", type=int, default=700)
    ap.add_argument("--octaves", type=int, default=4)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--contrast", type=float, default=0.02)
    ap.add_argument("--edge", type=float, default=10.0)
    ap.add_argument("--cross_check", action="store_true", default=False,
                    help="Require mutual best matches (default: off).")
    ap.add_argument("--root_sift", action="store_true", default=False,
                    help="Apply RootSIFT (L1 norm + sqrt) to descriptors before matching.")
    ap.add_argument("--ratio", type=float, default=0.85,
                    help="Lowe ratio threshold (recommend 0.80–0.90 for pano).")
    ap.add_argument("--prewarp_cyl", action="store_true",
                    help="Apply cylindrical pre-warp before feature extraction.")
    ap.add_argument("--focal", type=float, default=0.0,
                    help="Focal length (pixels) for cylindrical warp; 0 uses heuristic.")
    ap.add_argument("--ransac_iters", type=int, default=3000)
    ap.add_argument("--ransac_thresh", type=float, default=3.0)
    ap.add_argument("--outdir", default="output")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1) Load + scale match
    t0 = time.time()
    th = None if args.target_height == 0 else args.target_height
    img1, img2, _, _ = load_and_resize_to_same_height(
        args.image1, args.image2,
        target_height=th, hard_cap=args.hard_cap, min_cap=args.min_cap
    )

    # 2) Custom SIFT
    t1 = time.time()
    gauss1, dog1, gx1, gy1 = gaussian_pyramid(img1, num_octaves=args.octaves, num_layers=args.layers, initial_sigma=1.6)
    gauss2, dog2, gx2, gy2 = gaussian_pyramid(img2, num_octaves=args.octaves, num_layers=args.layers, initial_sigma=1.6)

    kps1 = find_keypoints(dog1, num_octaves=args.octaves, contrast_threshold=args.contrast, edge_threshold=args.edge)
    kps2 = find_keypoints(dog2, num_octaves=args.octaves, contrast_threshold=args.contrast, edge_threshold=args.edge)

    kps1 = assign_orientation(kps1, gx1, gy1)
    kps2 = assign_orientation(kps2, gx2, gy2)

    compute_descriptors(kps1, gx1, gy1)
    compute_descriptors(kps2, gx2, gy2)

    desc1 = np.array([kp.descriptor for kp in kps1 if kp.descriptor is not None])
    desc2 = np.array([kp.descriptor for kp in kps2 if kp.descriptor is not None])
    kp_idx1 = [i for i, kp in enumerate(kps1) if kp.descriptor is not None]
    kp_idx2 = [i for i, kp in enumerate(kps2) if kp.descriptor is not None]
    kps1_use = [kps1[i] for i in kp_idx1]
    kps2_use = [kps2[i] for i in kp_idx2]

    if args.root_sift:
        desc1 = rootsift(desc1)
        desc2 = rootsift(desc2)

    pairs = match_descriptors(desc1, desc2, ratio=args.ratio, cross_check=args.cross_check)

    # remap pair indices to original keypoint indices
    pairs = [(kp_idx1[i], kp_idx2[j]) for (i, j) in pairs]
    t2 = time.time()

    print(f"Custom SIFT: {len(kps1_use)}+{len(kps2_use)} usable keypoints, {len(pairs)} matches "
          f"(time {(t2 - t1):.2f}s, total {(t2 - t0):.2f}s so far).")

    # 3) Custom RANSAC
    kp_coords1 = kps_to_image_coords(kps1)
    kp_coords2 = kps_to_image_coords(kps2)
    H_c, inliers = ransac_homography(pairs, kp_coords1, kp_coords2,
                                     max_iters=args.ransac_iters, thresh=args.ransac_thresh)

    t3 = time.time()
    print(f"Custom RANSAC: {len(inliers)} inliers (time {(t3 - t2):.2f}s).")

    # Save custom inliers viz
    out_matches_custom = os.path.join(args.outdir, "custom_inliers.png")
    draw_inlier_matches(img1, img2, kps1, kps2, inliers, out_matches_custom)
    print(f"Saved custom inlier matches → {out_matches_custom}")

    # 4) OpenCV SIFT + RANSAC for comparison
    t4 = time.time()
    sift = cv2.SIFT_create()
    kp1_cv, d1_cv = sift.detectAndCompute(img1, None)
    kp2_cv, d2_cv = sift.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw = bf.knnMatch(d1_cv, d2_cv, k=2)
    good = [m for m, n in raw if m.distance < 0.75 * n.distance]
    pts1_cv = np.float32([kp1_cv[m.queryIdx].pt for m in good])
    pts2_cv = np.float32([kp2_cv[m.trainIdx].pt for m in good])
    H_cv, mask = cv2.findHomography(pts1_cv, pts2_cv, cv2.RANSAC, args.ransac_thresh)
    inl_cnt = int(mask.sum()) if mask is not None else 0
    t5 = time.time()
    print(f"OpenCV SIFT: {len(kp1_cv)}+{len(kp2_cv)} keypoints, {len(good)} matches, {inl_cnt} inliers "
          f"(time {(t5 - t4):.2f}s).")

    # Save OpenCV inlier viz
    inlier_matches = []
    if mask is not None:
        for idx, m in enumerate(good):
            if mask[idx]:
                inlier_matches.append(m)
    out_matches_cv = os.path.join(args.outdir, "opencv_inliers.png")
    ensure_dir(out_matches_cv)
    vis2 = cv2.drawMatches(img1, kp1_cv, img2, kp2_cv, inlier_matches, None,
                           matchColor=(0, 255, 0), singlePointColor=(0, 0, 255), flags=2)
    cv2.imwrite(out_matches_cv, vis2)
    print(f"Saved OpenCV inlier matches → {out_matches_cv}")

if __name__ == "__main__":
    main()
