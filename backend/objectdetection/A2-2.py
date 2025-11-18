# a2_q2_fourier_wiener.py (light/medium/strong blur friendly)
import cv2, numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

# ---------- config ----------
IMG_PATH  = "scene.jpeg"        # your image
SIGMA     = 6.0                # <-- set your blur strength here (try 2, 4, 6)
K_SIZE    = int(6*SIGMA) | 1   # kernel width ~ 6σ+1 and force odd
DISPLAY_W = 1024               # optional downscale width

def suggest_wiener_k(sigma: float) -> float:
    # smaller sigma -> smaller K (sharper); larger sigma -> larger K (more smoothing)
    # ranges are conservative; feel free to tune slightly up/down.
    if sigma <= 2:  return 0.003
    if sigma <= 4:  return 0.004
    if sigma <= 6:  return 0.006
    if sigma <= 10: return 0.008
    return 0.012

WIENER_K = suggest_wiener_k(SIGMA)

# ---------- helpers ----------
def gaussian_psf(ksize, sigma):
    ax = np.arange(-(ksize//2), ksize//2 + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    psf = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    psf /= psf.sum()
    return psf

def psnr(a, b):
    a = a.astype(np.float32); b = b.astype(np.float32)
    mse = np.mean((a - b)**2)
    return 20*np.log10(255.0) - 10*np.log10(max(mse, 1e-12))

def edgetaper(img, psf, strength=0.25, iters=2):
    """Blend image with its PSF-blurred version near boundaries to cut FFT ringing."""
    # 2D Tukey apodization mask
    H, W = img.shape
    def tukey(n, alpha=0.6):
        x = np.linspace(0, 1, n, dtype=np.float32)
        w = np.ones_like(x)
        e = alpha/2
        m1 = x < e;  w[m1] = 0.5*(1 + np.cos(np.pi*(2*x[m1]/alpha - 1)))
        m2 = x > 1-e; w[m2] = 0.5*(1 + np.cos(np.pi*(2*(1-x[m2])/alpha - 1)))
        return w
    apod = np.outer(tukey(H), tukey(W)).astype(np.float32)
    out = img.astype(np.float32)
    for _ in range(iters):
        blurred = cv2.filter2D(out, -1, psf, borderType=cv2.BORDER_REFLECT101).astype(np.float32)
        out = apod*out + (1-apod)*(strength*blurred + (1-strength)*out)
    return out.astype(img.dtype)

# ---------- load ----------
L0 = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
if L0 is None:
    raise FileNotFoundError(IMG_PATH)

# optional downscale for consistent blur perception
h, w = L0.shape[:2]
s = DISPLAY_W / float(w)
L = cv2.resize(L0, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA) if s != 1.0 else L0

# ---------- forward blur (spatial domain) ----------
L_b = cv2.GaussianBlur(L, (K_SIZE, K_SIZE), SIGMA)

# ---------- PSF -> OTF (pad+center -> FFT) ----------
psf  = gaussian_psf(K_SIZE, SIGMA).astype(np.float32)
Hpad = np.zeros_like(L, dtype=np.float32)
kh, kw = psf.shape
Hpad[:kh, :kw] = psf
Hpad = np.roll(np.roll(Hpad, -kh//2, axis=0), -kw//2, axis=1)   # center PSF
H    = np.fft.fft2(Hpad)

# ---------- precondition to reduce ringing ----------
L_b_taper = edgetaper(L_b, psf, strength=0.25, iters=2)

# ---------- Wiener deconvolution ----------
F_blur = np.fft.fft2(L_b_taper)
L_hat  = np.fft.ifft2(F_blur * np.conj(H) / (np.abs(H)**2 + WIENER_K)).real
L_hat  = np.clip(L_hat, 0, 255).astype(np.uint8)

# ---------- metrics ----------
print(f"[cfg] sigma={SIGMA}, ksize={K_SIZE}, Wiener K={WIENER_K}")
print(f"PSNR(L, L_b)   : {psnr(L, L_b):.2f} dB  (blur strength)")
print(f"PSNR(L, L_hat) : {psnr(L, L_hat):.2f} dB  (recovery)")

# ---------- save ----------
out_dir = Path(".")
cv2.imwrite(str(out_dir / "A2_Q2_original.jpg"), L)
cv2.imwrite(str(out_dir / "A2_Q2_blurred.jpg"), L_b)
cv2.imwrite(str(out_dir / "A2_Q2_wiener.jpg"), L_hat)
print("Saved: A2_Q2_original.jpg, A2_Q2_blurred.jpg, A2_Q2_wiener.jpg")

# ---------- quick viz ----------
plt.figure(figsize=(13,5))
plt.subplot(1,3,1); plt.title("Original"); plt.imshow(L, cmap='gray'); plt.axis('off')
plt.subplot(1,3,2); plt.title(f"Blurred (σ={SIGMA}, k={K_SIZE})"); plt.imshow(L_b, cmap='gray'); plt.axis('off')
plt.subplot(1,3,3); plt.title(f"Recovered (Wiener K={WIENER_K})"); plt.imshow(L_hat, cmap='gray'); plt.axis('off')
plt.tight_layout(); plt.show()