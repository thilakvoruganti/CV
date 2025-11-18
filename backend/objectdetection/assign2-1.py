# assign2-1.py — Correlation template matching (scaled + 0°/180°)
# Method: TM_CCOEFF_NORMED (zero-mean normalized correlation) — assignment-friendly.

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import glob, os, sys

# -------------------- INPUTS (tuned to your folder) --------------------
SCENE_PATH = "scene.jpeg"                      # <- your scene file
# Case-insensitive glob for templates (jpg/jpeg/JPG/JPEG)
TEMPLATE_PATTERNS = ["templates/*.jpg", "templates/*.jpeg",
                     "templates/*.JPG", "templates/*.JPEG"]

METHOD = cv.TM_CCOEFF_NORMED                   # correlation coefficient
SCALES = np.linspace(0.6, 1.3, 15)             # ~±35% size sweep (adjust if needed)
ANGLES = [0, 180]                              # check upright + flipped
SCORE_THRESH = 0.70                            # tune: 0.65–0.85 typically

# --- Angle handling (same as friend’s) ---
FORCE_ZERO_ANGLE = False       # True => ignore 180 entirely
PREFER_ANGLE_0 = True          # prefer 0° if it's close to 180°
ANGLE_MARGIN = 0.03            # "close" means score_0 >= score_180 - ANGLE_MARGIN

# --- Display sizes ---
DISPLAY_MAX_W, DISPLAY_MAX_H = 1400, 900

# -------------------- helpers --------------------
def die(msg):
    print(msg, file=sys.stderr); sys.exit(2)

def rotate_keep_all(tpl, angle):
    rows, cols = tpl.shape[:2]
    M = cv.getRotationMatrix2D((cols/2, rows/2), angle, 1.0)
    cos, sin = abs(M[0,0]), abs(M[0,1])
    nW = int(rows*sin + cols*cos)
    nH = int(rows*cos + cols*sin)
    M[0,2] += (nW/2) - cols/2
    M[1,2] += (nH/2) - rows/2
    return cv.warpAffine(tpl, M, (nW, nH), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)

def make_display(img_bgr, max_w=DISPLAY_MAX_W, max_h=DISPLAY_MAX_H, allow_upscale=False):
    h, w = img_bgr.shape[:2]
    s = min(max_w / w, max_h / h)
    if not allow_upscale: s = min(s, 1.0)
    if s != 1.0:
        interp = cv.INTER_AREA if s < 1.0 else cv.INTER_CUBIC
        return cv.resize(img_bgr, (int(w*s), int(h*s)), interpolation=interp), s
    return img_bgr.copy(), 1.0

def show_with_zoom(img_bgr, title="Detections", initial_zoom=None):
    cv.namedWindow(title, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
    fit_disp, fit_s = make_display(img_bgr, allow_upscale=False)
    zoom = fit_s if initial_zoom is None else initial_zoom
    def draw():
        h, w = img_bgr.shape[:2]
        z = max(0.05, min(5.0, zoom))
        interp = cv.INTER_AREA if z < 1.0 else cv.INTER_LINEAR
        disp = cv.resize(img_bgr, (int(w*z), int(h*z)), interpolation=interp)
        cv.imshow(title, disp)
        cv.resizeWindow(title, min(disp.shape[1], DISPLAY_MAX_W), min(disp.shape[0], DISPLAY_MAX_H))
    draw()
    while True:
        k = cv.waitKey(0) & 0xFF
        if k in (27, ord('q')): break
        elif k in (ord('+'), ord('=')): zoom *= 1.25; draw()
        elif k in (ord('-'), ord('_')): zoom /= 1.25; draw()
    cv.destroyAllWindows()

# -------------------- load inputs --------------------
img = cv.imread(SCENE_PATH, cv.IMREAD_GRAYSCALE)
if img is None: die(f"[ERR] Could not read scene image: {SCENE_PATH}")
img = cv.GaussianBlur(img, (3,3), 0)  # mild denoise helps with wood grain

# collect templates (case-insensitive)
template_paths = []
for pat in TEMPLATE_PATTERNS:
    template_paths.extend(glob.glob(pat))
template_paths = sorted(set(template_paths))
if len(template_paths) == 0:
    die(f"[ERR] No templates found in {TEMPLATE_PATTERNS}")
print("[INFO] Templates:", template_paths)

# -------------------- main --------------------
vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
colors = [(0,255,0),(0,180,255),(255,160,0),(255,0,120),
          (120,255,120),(160,120,255),(200,200,0),(0,220,180),
          (255,255,0),(0,255,255)]

angles_to_use = [0] if FORCE_ZERO_ANGLE else ANGLES

for idx, tpath in enumerate(template_paths):
    tpl = cv.imread(tpath, cv.IMREAD_GRAYSCALE)
    if tpl is None:
        print(f"[skip] Can't read template: {tpath}"); continue
    tpl = cv.GaussianBlur(tpl, (3,3), 0)

    best_per_angle = {}  # angle -> (score, (x,y), (w,h), scale)
    for ang in angles_to_use:
        tpl_rot = rotate_keep_all(tpl, ang) if abs(ang) > 1e-6 else tpl
        angle_best_score = -1.0
        angle_best = None
        for s in SCALES:
            tw = max(8, int(tpl_rot.shape[1]*s))
            th = max(8, int(tpl_rot.shape[0]*s))
            if tw >= img.shape[1] or th >= img.shape[0]:
                continue
            tpl_scaled = cv.resize(tpl_rot, (tw, th),
                                   interpolation=cv.INTER_AREA if s < 1.0 else cv.INTER_CUBIC)
            res = cv.matchTemplate(img, tpl_scaled, METHOD)  # correlation (zero-mean)
            _, max_val, _, max_loc = cv.minMaxLoc(res)
            if max_val > angle_best_score:
                angle_best_score = max_val
                angle_best = (max_loc, (tw, th), s)
        if angle_best is not None:
            best_per_angle[ang] = (angle_best_score, *angle_best)

    name = os.path.basename(tpath)
    if not best_per_angle:
        print(f"[warn] No valid scales for {name} (template larger than image)."); continue

    # choose between 0° and 180°; prefer 0° if almost tied
    chosen_angle = max(best_per_angle, key=lambda a: best_per_angle[a][0])
    if PREFER_ANGLE_0 and 0 in best_per_angle and 180 in best_per_angle:
        score0 = best_per_angle[0][0]; score180 = best_per_angle[180][0]
        if score0 >= (score180 - ANGLE_MARGIN):
            chosen_angle = 0

    best_score, (x,y), (w,h), s = best_per_angle[chosen_angle]
    print(f"Detected '{Path(name).stem}' at ({x},{y}) score={best_score:.3f} scale={s:.2f} angle={chosen_angle}°")

    if best_score >= SCORE_THRESH:
        color = colors[idx % len(colors)]
        cv.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        cv.putText(vis, f"{Path(name).stem}  {best_score:.2f}@{s:.2f}x,{chosen_angle}d",
                   (x, max(15, y - 6)), cv.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv.LINE_AA)
    else:
        # draw thin orange box to help threshold tuning
        below = (0,165,255)
        cv.rectangle(vis, (x, y), (x + w, y + h), below, 1)
        cv.putText(vis, f"{Path(name).stem}  {best_score:.2f}<{SCORE_THRESH:.2f}",
                   (x, max(15, y - 6)), cv.FONT_HERSHEY_SIMPLEX, 0.5, below, 1, cv.LINE_AA)

# --- save + display ---
out_path = "multi_match_result.png"
cv.imwrite(out_path, vis)
print(f"[OK] Saved annotated image -> {out_path}")

# show (OpenCV window); fall back to matplotlib if GUI unavailable
try:
    show_with_zoom(vis, title="Detections")
except Exception as e:
    print("[INFO] OpenCV GUI not available, fallback to matplotlib:", e)
    disp, _ = make_display(vis, allow_upscale=False)
    plt.figure(figsize=(10, 8))
    plt.imshow(cv.cvtColor(disp, cv.COLOR_BGR2RGB))
    plt.title("Best match per template")
    plt.axis('off'); plt.tight_layout(); plt.show()

