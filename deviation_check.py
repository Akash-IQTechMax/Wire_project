
# import cv2
# import numpy as np
# import os
# import csv
# import math
# from datetime import datetime

# # ============================================================
# # CONFIGURATION
# # ============================================================
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
# RESULTS_DIR = os.path.join(BASE_DIR, "results")
# REFERENCE_IMAGE_PATH = os.path.join(BASE_DIR, "wire_reference.png")

# os.makedirs(RESULTS_DIR, exist_ok=True)

# REFERENCE_LENGTH_MM = 12.0        # fixed AR wire length (1.2 cm)
# COLOR_LOWER = np.array([5, 80, 50])
# COLOR_UPPER = np.array([25, 255, 255])

# PIXELS_PER_MM = None

# # ---- Fixed Unity overlay angles (degrees) ----
# UNITY_ANGLES = {
#     "X": 44.1,
#     "Y": 45.7,
#     "Z": 89.8
# }


# # ============================================================
# # CALIBRATION
# # ============================================================
# def calibrate_reference():
#     """Calibrate using reference image."""
#     global PIXELS_PER_MM

#     img = cv2.imread(REFERENCE_IMAGE_PATH)
#     if img is None:
#         print("❌ Missing reference image for calibration.")
#         return False

#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     mask = cv2.inRange(hsv, COLOR_LOWER, COLOR_UPPER)
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if not contours:
#         print("❌ No contour found in reference image.")
#         return False

#     c = max(contours, key=cv2.contourArea)
#     _, (w, h), _ = cv2.minAreaRect(c)
#     PIXELS_PER_MM = max(w, h) / REFERENCE_LENGTH_MM
#     print(f"✅ Calibrated: 1 mm = {PIXELS_PER_MM:.3f} px  |  1 px = {1/PIXELS_PER_MM:.6f} mm")
#     return True


# # ============================================================
# # ANALYZE SINGLE IMAGE
# # ============================================================
# def analyze_wire_image(path):
#     img = cv2.imread(path)
#     if img is None:
#         print(f"❌ Cannot read {path}")
#         return None

#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     mask = cv2.inRange(hsv, COLOR_LOWER, COLOR_UPPER)
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if not contours:
#         print(f"⚠️ No wire detected in {os.path.basename(path)}")
#         return None

#     cnt = max(contours, key=cv2.contourArea)

#     # Fit a straight line for more stable angle detection
#     [vx, vy, x0, y0] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
#     vx, vy = float(vx), float(vy)
#     angle_rad = math.atan2(vy, vx)
#     angle_deg = math.degrees(angle_rad)
#     if angle_deg < 0:
#         angle_deg += 180

#     # Bounding box for length estimation
#     x, y, w, h = cv2.boundingRect(cnt)
#     wire_length_px = max(w, h)
#     wire_length_mm = wire_length_px / PIXELS_PER_MM
#     deviation_len = abs(wire_length_mm - REFERENCE_LENGTH_MM)

#     # Orientation / direction
#     if 85 <= angle_deg <= 95:
#         orientation = "Vertical"
#     elif angle_deg < 85:
#         orientation = f"Tilted Right ({angle_deg:.1f}°)"
#     else:
#         orientation = f"Tilted Left ({180 - angle_deg:.1f}°)"

#     # ------------------------------------------------------------
#     # 3D vector calculation
#     # ------------------------------------------------------------
#     overlay_len = REFERENCE_LENGTH_MM
#     a1, a2, a3 = map(math.radians, [UNITY_ANGLES["X"], UNITY_ANGLES["Y"], UNITY_ANGLES["Z"]])
#     x2 = overlay_len * math.cos(a1)
#     y2 = overlay_len * math.cos(a2)
#     z2 = overlay_len * math.cos(a3)

#     # Real wire vector
#     beta1 = angle_deg
#     b1 = math.radians(beta1)
#     x1 = wire_length_mm * math.cos(b1)
#     y1 = wire_length_mm * math.sin(b1)
#     z1 = 0

#     # Axis deviations
#     dx = x2 - x1
#     dy = y2 - y1
#     dz = z2 - z1

#     # 3D deviation distance
#     deviation_3d = math.sqrt(dx**2 + dy**2 + dz**2)

#     # Angular deviation
#     angle_dev = abs(beta1 - UNITY_ANGLES["X"])

#     # ------------------------------------------------------------
#     # Annotate image
#     # ------------------------------------------------------------
#     annotated = img.copy()
#     box = np.intp(cv2.boxPoints(cv2.minAreaRect(cnt)))
#     cv2.drawContours(annotated, [box], 0, (0, 255, 0), 2)

#     text1 = f"Len: {wire_length_mm:.2f}mm | Angle: {angle_deg:.2f}°"
#     text2 = f"3D Dev: {deviation_3d:.3f}mm | ΔAngle: {angle_dev:.2f}°"
#     cv2.putText(annotated, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
#     cv2.putText(annotated, text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
#     cv2.putText(annotated, orientation, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

#     out_path = os.path.join(RESULTS_DIR, f"annotated_{os.path.basename(path)}")
#     cv2.imwrite(out_path, annotated)

#     # ------------------------------------------------------------
#     # Return structured results
#     # ------------------------------------------------------------
#     result = {
#         "file": os.path.basename(path),
#         "length_mm": round(wire_length_mm, 3),
#         "angle_deg": round(angle_deg, 2),
#         "orientation": orientation,
#         "deviation_len_mm": round(deviation_len, 3),
#         "3D_deviation_mm": round(deviation_3d, 3),
#         "angle_deviation_deg": round(angle_dev, 3),
#         "wire_tip_real": {"x1": round(x1, 4), "y1": round(y1, 4), "z1": round(z1, 4)},
#         "overlay_tip": {"x2": round(x2, 4), "y2": round(y2, 4), "z2": round(z2, 4)},
#         "delta": {"dx": round(dx, 4), "dy": round(dy, 4), "dz": round(dz, 4)}
#     }

#     print(f"[INFO] {os.path.basename(path)} | Angle: {angle_deg:.2f}° | 3D Dev: {deviation_3d:.3f} mm")
#     return result


# # ============================================================
# # ANALYZE 6 IMAGES AND AVERAGE
# # ============================================================
# def analyze_all():
#     if not calibrate_reference():
#         print("❌ Calibration failed.")
#         return

#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     folder = os.path.join(RESULTS_DIR, timestamp)
#     os.makedirs(folder, exist_ok=True)

#     files = sorted(
#         [f for f in os.listdir(UPLOAD_FOLDER) if f.lower().endswith((".png", ".jpg", ".jpeg"))],
#         key=lambda x: os.path.getmtime(os.path.join(UPLOAD_FOLDER, x)),
#         reverse=True
#     )[:6]

#     results = []
#     for f in files:
#         fp = os.path.join(UPLOAD_FOLDER, f)
#         res = analyze_wire_image(fp)
#         if res:
#             results.append(res)

#     if not results:
#         print("❌ No valid images processed.")
#         return

#     # ---- Compute averages ----
#     avg_length = np.mean([r["length_mm"] for r in results])
#     avg_angle = np.mean([r["angle_deg"] for r in results])
#     avg_dev_len = np.mean([r["deviation_len_mm"] for r in results])
#     avg_3d_dev = np.mean([r["3D_deviation_mm"] for r in results])
#     avg_angle_dev = np.mean([r["angle_deviation_deg"] for r in results])

#     # Orientation majority
#     orientations = [r["orientation"].split()[0] for r in results]
#     avg_orientation = max(set(orientations), key=orientations.count)

#     # Real and overlay endpoints (averaged)
#     avg_x1 = np.mean([r["wire_tip_real"]["x1"] for r in results])
#     avg_y1 = np.mean([r["wire_tip_real"]["y1"] for r in results])
#     avg_z1 = np.mean([r["wire_tip_real"]["z1"] for r in results])

#     avg_x2 = np.mean([r["overlay_tip"]["x2"] for r in results])
#     avg_y2 = np.mean([r["overlay_tip"]["y2"] for r in results])
#     avg_z2 = np.mean([r["overlay_tip"]["z2"] for r in results])

#     avg_dx = np.mean([r["delta"]["dx"] for r in results])
#     avg_dy = np.mean([r["delta"]["dy"] for r in results])
#     avg_dz = np.mean([r["delta"]["dz"] for r in results])

#     # ---- Ordered output ----
#     averages = {
#         "Avg. Real Wire Length": f"{avg_length:.3f} mm",
#         "β Avg. Real Wire Tilt Angle (From X-axis)": f"{avg_angle:.2f}°",
#         "Avg. Real Wire Orientation": avg_orientation,
#         "Avg. Length Deviation": f"{avg_dev_len:.3f} mm",
#         "Avg. 3D Spatial Deviation": f"{avg_3d_dev/10:.3f} mm",
#         "Avg. Angular Deviation": f"{avg_angle_dev:.3f}°",
#         "Avg. Real Wire Endpoint Deviation": f"({avg_x1:.3f}, {avg_y1:.3f}, {avg_z1:.3f})",
#         "Avg. Overlay Wire Endpoint Deviation": f"({avg_x2:.3f}, {avg_y2:.3f}, {avg_z2:.3f})",
#         "Δ Avg. Per Axis Deviation": f"dx={avg_dx:.3f}, dy={avg_dy:.3f}, dz={avg_dz:.3f}"
#     }

#     # ---- Save all data ----
#     csv_path = os.path.join(folder, "wire_analysis_results.csv")
#     with open(csv_path, "w", newline="") as f:
#         writer = csv.DictWriter(f, fieldnames=results[0].keys())
#         writer.writeheader()
#         writer.writerows(results)

#     summary_path = os.path.join(folder, "average_summary.txt")
#     with open(summary_path, "w", encoding="utf-8") as f:
#         for title, value in averages.items():
#             f.write(f"{title}: {value}\n")

#     print(f"✅ Analysis complete for {len(results)} images.")
#     print(f"📊 CSV saved: {csv_path}")
#     print(f"📄 Summary saved: {summary_path}")


# # ============================================================
# # ENTRY POINT
# # ============================================================
# if __name__ == "__main__":
#     analyze_all()






import cv2
import numpy as np
import os
import csv
import math
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
RESULTS_DIR = os.path.join(BASE_DIR, "results_final")
REFERENCE_IMAGE_PATH = os.path.join(BASE_DIR, "wire_reference.png")
os.makedirs(RESULTS_DIR, exist_ok=True)

REFERENCE_LENGTH_MM = 12.0  # AR wire reference (1.2 cm)
COLOR_LOWER = np.array([5, 80, 50])
COLOR_UPPER = np.array([25, 255, 255])
PIXELS_PER_MM = None

UNITY_ANGLES = {"X": 44.1, "Y": 45.7, "Z": 89.8}

PIXEL_TO_MM_SKELETON = 0.26  # default pixel-to-mm for skeleton mode


# ============================================================
# CALIBRATION
# ============================================================
def calibrate_reference():
    """Calibrate using reference image."""
    global PIXELS_PER_MM
    img = cv2.imread(REFERENCE_IMAGE_PATH)
    if img is None:
        print("❌ Missing reference image for calibration.")
        return False

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, COLOR_LOWER, COLOR_UPPER)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("❌ No contour found in reference image.")
        return False

    c = max(contours, key=cv2.contourArea)
    _, (w, h), _ = cv2.minAreaRect(c)
    PIXELS_PER_MM = max(w, h) / REFERENCE_LENGTH_MM
    print(f"✅ Calibrated: 1 mm = {PIXELS_PER_MM:.3f} px  |  1 px = {1/PIXELS_PER_MM:.6f} mm")
    return True


# ============================================================
# HELPER FUNCTIONS (from Code 2)
# ============================================================
def get_best_blob(blobs):
    best_blob, best_size = None, 0
    for blob in blobs:
        rot_rect = cv2.minAreaRect(blob)
        (cx, cy), (sx, sy), angle = rot_rect
        if sx * sy > best_size:
            best_blob = rot_rect
            best_size = sx * sy
    return best_blob


def draw_blob_rect(frame, blob, color):
    box = cv2.boxPoints(blob)
    box = box.astype(int)
    return cv2.drawContours(frame, [box], 0, color, 2)


def measure_bent_wire_length(img):
    """Estimate bent wire length using skeleton centerline."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_bg = cv2.inRange(hsv, (0, 40, 40), (180, 255, 255))
    h, s, v = cv2.split(hsv)
    h_values, s_values, v_values = h[mask_bg > 0], s[mask_bg > 0], v[mask_bg > 0]
    lower = np.array([np.percentile(h_values, 5),
                      np.percentile(s_values, 5),
                      np.percentile(v_values, 5)], dtype=np.uint8)
    upper = np.array([np.percentile(h_values, 95),
                      np.percentile(s_values, 95),
                      np.percentile(v_values, 95)], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    size = np.size(mask)
    skel = np.zeros(mask.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    done = False
    while not done:
        eroded = cv2.erode(mask, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(mask, temp)
        skel = cv2.bitwise_or(skel, temp)
        mask = eroded.copy()
        zeros = size - cv2.countNonZero(mask)
        if zeros == size:
            done = True

    points = np.column_stack(np.where(skel > 0))
    if len(points) < 2:
        return None, None
    length_px = 0
    for i in range(1, len(points)):
        length_px += np.linalg.norm(points[i] - points[i - 1])
    length_mm = length_px * PIXEL_TO_MM_SKELETON
    return length_px, length_mm


# ============================================================
# ANALYZE SINGLE IMAGE (COMBINED)
# ============================================================
def analyze_wire_image(path):
    img = cv2.imread(path)
    if img is None:
        print(f"❌ Cannot read {path}")
        return None

    # ---- Skeleton-based bend length ----
    wire_length_px, wire_length_mm_skel = measure_bent_wire_length(img)
    if wire_length_px is None:
        print(f"⚠️ Skeleton not detected in {path}")
        return None

    # ---- Normal AR/Angle calculation (Code 1) ----
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, COLOR_LOWER, COLOR_UPPER)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"⚠️ No wire detected in {os.path.basename(path)}")
        return None
    cnt = max(contours, key=cv2.contourArea)
    [vx, vy, x0, y0] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
    angle_rad = math.atan2(vy, vx)
    angle_deg = (math.degrees(angle_rad) + 180) % 180

    x, y, w, h = cv2.boundingRect(cnt)
    wire_length_mm_box = max(w, h) / PIXELS_PER_MM
    deviation_len = abs(wire_length_mm_box - REFERENCE_LENGTH_MM)

    # Orientation
    orientation = "Vertical" if 85 <= angle_deg <= 95 else (
        f"Tilted Right ({angle_deg:.1f}°)" if angle_deg < 85 else f"Tilted Left ({180 - angle_deg:.1f}°)"
    )

    # ---- Tilt Detection (from Code 2) ----
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    dy = cv2.Sobel(thresh, cv2.CV_32F, 0, 1, ksize=21)
    dy = dy * dy
    cv2.normalize(dy, dy, alpha=255, beta=0, norm_type=cv2.NORM_MINMAX)
    dy = dy.astype("uint8")
    _, tilt_mask = cv2.threshold(dy, 0.6 * 255, 255, cv2.THRESH_BINARY)
    contours_tilt, _ = cv2.findContours(tilt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_tilt = get_best_blob(contours_tilt) if contours_tilt else None
    tilt_angle = 90 + best_tilt[2] if best_tilt else 0

    if abs(tilt_angle) < 1:
        tilt_dir = "Aligned (0°)"
    elif tilt_angle > 90:
        tilt_dir = f"Upward (+{abs(tilt_angle - 90):.2f}°)"
    else:
        tilt_dir = f"Downward (-{abs(90 - tilt_angle):.2f}°)"

    # ---- 3D vector + deviation ----
    a1, a2, a3 = map(math.radians, [UNITY_ANGLES["X"], UNITY_ANGLES["Y"], UNITY_ANGLES["Z"]])
    overlay_len = REFERENCE_LENGTH_MM
    x2, y2, z2 = (overlay_len * math.cos(a1),
                  overlay_len * math.cos(a2),
                  overlay_len * math.cos(a3))
    b1 = math.radians(angle_deg)
    x1, y1, z1 = (wire_length_mm_box * math.cos(b1),
                  wire_length_mm_box * math.sin(b1),
                  0)
    dx, dy3d, dz = (x2 - x1, y2 - y1, z2 - z1)
    deviation_3d = math.sqrt(dx**2 + dy3d**2 + dz**2)
    angle_dev = abs(angle_deg - UNITY_ANGLES["X"])

    # ---- Annotate image ----
    annotated = img.copy()
    cv2.drawContours(annotated, [cv2.boxPoints(cv2.minAreaRect(cnt)).astype(int)], 0, (0, 255, 0), 2)
    cv2.putText(annotated, f"Len(Box): {wire_length_mm_box:.2f}mm", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(annotated, f"Len(Skel): {wire_length_mm_skel:.2f}mm", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(annotated, f"Angle: {angle_deg:.2f}° | Tilt: {tilt_dir}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    out_path = os.path.join(RESULTS_DIR, f"annotated_{os.path.basename(path)}")
    cv2.imwrite(out_path, annotated)

    print(f"[INFO] {os.path.basename(path)} | Angle: {angle_deg:.2f}° | Tilt: {tilt_dir} | 3D Dev: {deviation_3d:.3f} mm")

    return {
        "file": os.path.basename(path),
        "length_mm_box": round(wire_length_mm_box, 3),
        "length_mm_skel": round(wire_length_mm_skel, 3),
        "angle_deg": round(angle_deg, 2),
        "orientation": orientation,
        "tilt_direction": tilt_dir,
        "deviation_len_mm": round(deviation_len, 3),
        "3D_deviation_mm": round(deviation_3d, 3),
        "angle_deviation_deg": round(angle_dev, 3),
        "wire_tip_real": {"x1": round(x1, 4), "y1": round(y1, 4), "z1": round(z1, 4)},
        "overlay_tip": {"x2": round(x2, 4), "y2": round(y2, 4), "z2": round(z2, 4)},
        "delta": {"dx": round(dx, 4), "dy": round(dy3d, 4), "dz": round(dz, 4)}
    }


# ============================================================
# MAIN — MULTI IMAGE ANALYSIS
# ============================================================
def analyze_all():
    if not calibrate_reference():
        print("❌ Calibration failed.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = os.path.join(RESULTS_DIR, timestamp)
    os.makedirs(folder, exist_ok=True)

    files = sorted(
        [f for f in os.listdir(UPLOAD_FOLDER) if f.lower().endswith((".png", ".jpg", ".jpeg"))],
        key=lambda x: os.path.getmtime(os.path.join(UPLOAD_FOLDER, x)),
        reverse=True
    )

    results = []
    for f in files:
        fp = os.path.join(UPLOAD_FOLDER, f)
        res = analyze_wire_image(fp)
        if res:
            results.append(res)

    if not results:
        print("❌ No valid images processed.")
        return

    # ---- Save CSV ----
    csv_path = os.path.join(folder, "wire_analysis_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # ---- Compute Averages ----
    avg_box = np.mean([r["length_mm_box"] for r in results])
    avg_skel = np.mean([r["length_mm_skel"] for r in results])
    avg_angle = np.mean([r["angle_deg"] for r in results])
    avg_3d = np.mean([r["3D_deviation_mm"] for r in results])
    avg_angle_dev = np.mean([r["angle_deviation_deg"] for r in results])
    avg_tilt = [r["tilt_direction"] for r in results]

    upward = sum(1 for t in avg_tilt if "Upward" in t)
    downward = sum(1 for t in avg_tilt if "Downward" in t)
    aligned = sum(1 for t in avg_tilt if "Aligned" in t)
    tilt_summary = max([("Upward", upward), ("Downward", downward), ("Aligned", aligned)], key=lambda x: x[1])[0]

    # ---- Save Summary ----
    summary_path = os.path.join(folder, "average_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=== FINAL WIRE ANALYSIS SUMMARY ===\n")
        f.write(f"Average Length (Box): {avg_box:.3f} mm\n")
        f.write(f"Average Length (Skeleton): {avg_skel:.3f} mm\n")
        f.write(f"Average Angle: {avg_angle:.2f}°\n")
        f.write(f"Average 3D Deviation: {avg_3d:.3f} mm\n")
        f.write(f"Average Angular Deviation: {avg_angle_dev:.3f}°\n")
        f.write(f"Most Common Tilt: {tilt_summary}\n")

    print(f"\n✅ Analysis complete for {len(results)} images.")
    print(f"📊 CSV saved: {csv_path}")
    print(f"📄 Summary saved: {summary_path}")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    analyze_all()
