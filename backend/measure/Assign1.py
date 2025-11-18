import cv2
import numpy as np

# ---------- Your camera calibration values (from calibration_best.npz) ----------
fx = 4172.88879
fy = 4164.42586
cx = 1610.47852
cy = 2928.65123
# -----------------------------------------------------------------------------

# Known distance between camera and object plane (in meters)
Z = 0.264  # 26.4 cm

clicked_points = []  # list to store clicked (u,v)

# ---------- Mouse callback ----------
def get_pixel_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at: x={x}, y={y}")
        clicked_points.append((x, y))

# ---------- Load the image ----------
image_path = "Assign1-2.jpeg"  # replace with your captured photo
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Could not load image from {image_path}")
    exit()

# ---------- Display image and capture clicks ----------
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", get_pixel_coordinates)
cv2.imshow("Image", image)
print("\n[Instruction] Click exactly TWO points on the object edge, then press any key to close.\n")
cv2.waitKey(0)
cv2.destroyAllWindows()

# ---------- Compute real-world length ----------
if len(clicked_points) != 2:
    print(f"You clicked {len(clicked_points)} points; need exactly 2.")
    exit()

(u1, v1), (u2, v2) = clicked_points

# ----- Perspective Projection Equation -----
L = Z * np.sqrt(((u1 - u2) / fx) ** 2 + ((v1 - v2) / fy) ** 2)

print("\n========= RESULT =========")
print(f"Point 1: (u1, v1) = ({u1}, {v1})")
print(f"Point 2: (u2, v2) = ({u2}, {v2})")
print(f"Camera–Object Distance Z = {Z:.3f} m")
print(f"Estimated Real-World Length L = {L:.6f} m  ({L*100:.3f} cm)")
print("==========================")

# Annotate and save result image
annot = image.copy()
cv2.line(annot, (u1, v1), (u2, v2), (0, 255, 0), 2)
cv2.putText(annot, f"L={L*100:.2f} cm", (min(u1,u2)+10, min(v1,v2)+30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
cv2.imwrite("annotated_result.jpg", annot)
print("Saved annotated_result.jpg")
