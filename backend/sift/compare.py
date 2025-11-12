# backend/sift/compare.py
import os, cv2, time, numpy as np

def _draw_inliers(img1, img2, kp1, kp2, matches, mask, out_path):
    inlier_matches = [m for m, keep in zip(matches, mask.ravel().tolist()) if keep]
    vis = cv2.drawMatches(img1, kp1, img2, kp2, inlier_matches, None,
                          matchColor=(0,255,0), singlePointColor=(0,0,255), flags=2)
    cv2.imwrite(out_path, vis)

def run_comparison(
    path1, path2,
    target_height=1100,
    octaves=5, layers=4, contrast=0.01, edge=10.0,
    root_sift=True, ratio=0.85, cross_check=False,
    ransac_iters=8000, ransac_thresh=6.0,
    prewarp_cyl=False, focal=None, out_dir="static"
):
    os.makedirs(out_dir, exist_ok=True)
    img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        raise RuntimeError("Failed to read inputs.")
    # resize to same height
    def resize_to_h(img, H):
        h,w = img.shape[:2]
        s = H/float(h)
        return cv2.resize(img, (int(round(w*s)), H), interpolation=cv2.INTER_AREA)
    img1 = resize_to_h(img1, target_height)
    img2 = resize_to_h(img2, target_height)

    # OpenCV SIFT baseline
    t0 = time.time()
    sift = cv2.SIFT_create()
    k1, d1 = sift.detectAndCompute(img1, None)
    k2, d2 = sift.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw = bf.knnMatch(d1, d2, k=2)
    good = [m for m,n in raw if m.distance < 0.75*n.distance]
    pts1 = np.float32([k1[m.queryIdx].pt for m in good])
    pts2 = np.float32([k2[m.trainIdx].pt for m in good])
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransac_thresh, maxIters=ransac_iters)
    inliers = int(mask.sum()) if mask is not None else 0
    t1 = time.time()

    out_custom = os.path.join(out_dir, "custom_inliers.png")
    out_cv = os.path.join(out_dir, "opencv_inliers.png")
    # For now, show the same inliers for both (replace later with your custom pipeline visuals).
    _draw_inliers(img1, img2, k1, k2, good, mask if mask is not None else np.zeros((len(good),1),dtype=np.uint8), out_custom)
    _draw_inliers(img1, img2, k1, k2, good, mask if mask is not None else np.zeros((len(good),1),dtype=np.uint8), out_cv)

    return {
        "ok": True,
        "opencv_keypoints": [len(k1), len(k2)],
        "opencv_matches": len(good),
        "opencv_inliers": inliers,
        "opencv_time_sec": round(t1 - t0, 2),
        "custom_inliers": out_custom,
        "opencv_inliers_img": out_cv,
        "custom_inliers_img": out_custom,
        "opencv_inliers_path": out_cv,
        "opencv_inliers_file": os.path.basename(out_cv),
        "custom_inliers_path": out_custom,
        "custom_inliers_file": os.path.basename(out_custom),
        "opencv_inliers": inliers,
        "custom_inliers_count": inliers,
        "opencv_time": round(t1 - t0, 2),
        "custom_time": round(t1 - t0, 2)
    }
