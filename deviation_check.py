
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
RESULTS_DIR = os.path.join(BASE_DIR, "results")
REFERENCE_IMAGE_PATH = os.path.join(BASE_DIR, "wire_reference.png")

os.makedirs(RESULTS_DIR, exist_ok=True)

REFERENCE_LENGTH_MM = 12.0        # fixed AR wire length in mm (12 mm = 1.2 cm)
COLOR_LOWER = np.array([5, 80, 50])
COLOR_UPPER = np.array([25, 255, 255])

PIXELS_PER_MM = None

# ---- Fixed Unity overlay angles (degrees) ----
UNITY_ANGLES = {
    "X": 44.1,
    "Y": 45.7,
    "Z": 89.8
}

# ============================================================
# HELPERS
# ============================================================
def measure_curve_length(contour):
    contour = contour.squeeze()
    if len(contour.shape) != 2 or contour.shape[0] < 2:
        return 0.0
    length = 0.0
    for i in range(1, len(contour)):
        length += np.linalg.norm(contour[i] - contour[i - 1])
    return float(length)

def longest_side_of_minrect(contour):
    # return pixel length of long side of the minAreaRect
    _, (w, h), _ = cv2.minAreaRect(contour)
    return max(w, h)

# ============================================================
# CALIBRATION
# ============================================================
def calibrate_reference(min_area_ratio=0.0005, min_px_fraction=0.01, max_px_fraction=0.9):
    """
    Calibrate using the reference image. Abort if reference detection looks unrealistic.
    min_area_ratio: min contour area fraction of image to accept (small number)
    min_px_fraction/max_px_fraction: long side of rect must be between these fractions of image diagonal
    """
    global PIXELS_PER_MM

    img = cv2.imread(REFERENCE_IMAGE_PATH)
    if img is None:
        print("❌ Missing reference image for calibration.")
        return False

    h_img, w_img = img.shape[:2]
    diag = math.hypot(w_img, h_img)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, COLOR_LOWER, COLOR_UPPER)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # adaptive fallback
        h, s, v = cv2.split(hsv)
        lower = np.array([int(np.percentile(h, 5)), int(np.percentile(s, 5)), int(np.percentile(v, 5))], dtype=np.uint8)
        upper = np.array([int(np.percentile(h, 95)), int(np.percentile(s, 95)), int(np.percentile(v, 95))], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("❌ Still no contour found. Calibration failed.")
            return False

    # pick largest contour by area and ensure it is not tiny noise
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    img_area = w_img * h_img
    if area / img_area < min_area_ratio:
        print(f"⚠️ Detected contour too small for calibration (area ratio {area/img_area:.6f}).")
        return False

    ref_px_len = longest_side_of_minrect(c)
    px_fraction = ref_px_len / diag
    print(f"[CAL] ref_px_len={ref_px_len:.2f}px  | image_diag={diag:.2f}px  | fraction={px_fraction:.4f}")

    if not (min_px_fraction <= px_fraction <= max_px_fraction):
        print("⚠️ Reference object pixel size looks unrealistic (too small/large).")
        return False

    # pixels per mm = pixels representing the known physical length
    PIXELS_PER_MM = ref_px_len / REFERENCE_LENGTH_MM
    print(f"✅ Calibrated: {PIXELS_PER_MM:.4f} px per mm  |  1 px = {1.0/PIXELS_PER_MM:.6f} mm")

    # sanity bounds on PIXELS_PER_MM
    if PIXELS_PER_MM < 0.05 or PIXELS_PER_MM > 500:
        print("❌ Calibration gives unrealistic pixels/mm. Aborting calibration.")
        PIXELS_PER_MM = None
        return False

    return True

# ============================================================
# ANALYZE SINGLE IMAGE
# ============================================================
def analyze_wire_image(path):
    img = cv2.imread(path)
    if img is None:
        print(f"❌ Cannot read {path}")
        return None

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, COLOR_LOWER, COLOR_UPPER)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # adaptive fallback
        h, s, v = cv2.split(hsv)
        lower = np.array([int(np.percentile(h, 5)), int(np.percentile(s, 5)), int(np.percentile(v, 5))], dtype=np.uint8)
        upper = np.array([int(np.percentile(h, 95)), int(np.percentile(s, 95)), int(np.percentile(v, 95))], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print(f"⚠️ No wire detected in {os.path.basename(path)}")
        return None

    cnt = max(contours, key=cv2.contourArea)

    # ---- PIXEL LENGTH ----
    wire_length_px = measure_curve_length(cnt)
    if PIXELS_PER_MM is None or PIXELS_PER_MM <= 0:
        print("❌ PIXELS_PER_MM is not set properly.")
        return None

    wire_length_mm = wire_length_px / PIXELS_PER_MM
    deviation_len = abs(wire_length_mm - REFERENCE_LENGTH_MM)

    # ---- ANGLE (signed) ----
    [vx, vy, x0, y0] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy = float(vx), float(vy)
    angle_rad = math.atan2(vy, vx)            # keeps sign: -pi..+pi
    angle_deg_signed = math.degrees(angle_rad)

    # Relative signed angle to Unity’s X-axis
    angle_relative = angle_deg_signed - UNITY_ANGLES["X"]
    if angle_relative > 0:
        direction = "Tilted Up (+)"
    elif angle_relative < 0:
        direction = "Tilted Down (−)"
    else:
        direction = "Aligned (0°)"

    # Orientation text
    if 85 <= abs(angle_deg_signed) <= 95:
        orientation = "Vertical"
    elif abs(angle_deg_signed) < 85:
        orientation = f"Tilted Right ({angle_deg_signed:.1f}°)"
    else:
        # show signed left/right properly
        orientation = f"Tilted Left ({angle_deg_signed:.1f}°)"

    # ------------------------------------------------------------
    # 3D vector calculation (use signed angle)
    # ------------------------------------------------------------
    overlay_len = REFERENCE_LENGTH_MM
    a1, a2, a3 = map(math.radians, [UNITY_ANGLES["X"], UNITY_ANGLES["Y"], UNITY_ANGLES["Z"]])
    x2 = overlay_len * math.cos(a1)
    y2 = overlay_len * math.cos(a2)
    z2 = overlay_len * math.cos(a3)

    b1 = math.radians(angle_deg_signed)   # use signed angle
    x1 = wire_length_mm * math.cos(b1)
    y1 = wire_length_mm * math.sin(b1)
    z1 = 0.0

    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    deviation_3d = math.sqrt(dx**2 + dy**2 + dz**2)
    angle_dev = abs(angle_deg_signed - UNITY_ANGLES["X"])

    # ------------------------------------------------------------
    # Annotate image
    # ------------------------------------------------------------
    annotated = img.copy()
    box = np.intp(cv2.boxPoints(cv2.minAreaRect(cnt)))
    cv2.drawContours(annotated, [box], 0, (0, 255, 0), 2)

    text1 = f"Bent Len: {wire_length_mm:.2f} mm | Angle: {angle_deg_signed:.2f}°"
    text2 = f"3D Dev: {deviation_3d:.3f} mm | ΔAngle: {angle_dev:.2f}°"
    text3 = direction
    cv2.putText(annotated, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(annotated, text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(annotated, text3, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    out_path = os.path.join(RESULTS_DIR, f"annotated_{os.path.basename(path)}")
    cv2.imwrite(out_path, annotated)

    print(f"[INFO] {os.path.basename(path)} | wire_px={wire_length_px:.2f}px | "
          f"px_per_mm={PIXELS_PER_MM:.4f} | Len={wire_length_mm:.3f}mm | SignedTilt={angle_deg_signed:.2f}° | 3DDev={deviation_3d:.3f}mm")

    result = {
        "file": os.path.basename(path),
        "length_mm": round(wire_length_mm, 3),
        "bent_length_mm": round(wire_length_mm, 3),
        "angle_deg": round(angle_deg_signed, 3),           # keep signed angle here
        "signed_tilt_deg": round(angle_relative, 3),
        "orientation": orientation,
        "direction": direction,
        "deviation_len_mm": round(deviation_len, 3),
        "3D_deviation_mm": round(deviation_3d, 3),
        "angle_deviation_deg": round(angle_dev, 3),
        "wire_tip_real": {"x1": round(x1, 4), "y1": round(y1, 4), "z1": round(z1, 4)},
        "overlay_tip": {"x2": round(x2, 4), "y2": round(y2, 4), "z2": round(z2, 4)},
        "delta": {"dx": round(dx, 4), "dy": round(dy, 4), "dz": round(dz, 4)}
    }

    return result

# ============================================================
# ANALYZE N IMAGES AND AVERAGE
# ============================================================
def analyze_all():
    if not calibrate_reference():
        print("❌ Calibration failed. Fix the reference image or color thresholds.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = os.path.join(RESULTS_DIR, timestamp)
    os.makedirs(folder, exist_ok=True)

    files = sorted(
        [f for f in os.listdir(UPLOAD_FOLDER) if f.lower().endswith((".png", ".jpg", ".jpeg"))],
        key=lambda x: os.path.getmtime(os.path.join(UPLOAD_FOLDER, x)),
        reverse=True
    )[:6]

    results = []
    for f in files:
        fp = os.path.join(UPLOAD_FOLDER, f)
        res = analyze_wire_image(fp)
        if res:
            results.append(res)

    if not results:
        print("❌ No valid images processed.")
        return

    # ---- Compute averages ----
    avg_length = np.mean([r["length_mm"] for r in results])
    avg_bent_length = np.mean([r["bent_length_mm"] for r in results])
    avg_angle_signed = np.mean([r["angle_deg"] for r in results])       # signed average
    avg_signed_tilt = np.mean([r["signed_tilt_deg"] for r in results])
    avg_dev_len = np.mean([r["deviation_len_mm"] for r in results])
    avg_3d_dev = np.mean([r["3D_deviation_mm"] for r in results])
    avg_angle_dev = np.mean([r["angle_deviation_deg"] for r in results])

    orientations = [r["orientation"].split()[0] for r in results]
    avg_orientation = max(set(orientations), key=orientations.count)

    avg_x1 = np.mean([r["wire_tip_real"]["x1"] for r in results])
    avg_y1 = np.mean([r["wire_tip_real"]["y1"] for r in results])
    avg_z1 = np.mean([r["wire_tip_real"]["z1"] for r in results])
    avg_x2 = np.mean([r["overlay_tip"]["x2"] for r in results])
    avg_y2 = np.mean([r["overlay_tip"]["y2"] for r in results])
    avg_z2 = np.mean([r["overlay_tip"]["z2"] for r in results])
    avg_dx = np.mean([r["delta"]["dx"] for r in results])
    avg_dy = np.mean([r["delta"]["dy"] for r in results])
    avg_dz = np.mean([r["delta"]["dz"] for r in results])

    # ---- Ordered summary output ----
    averages = {
        "Avg. Real Wire Length": f"{avg_length:.3f} mm",
        "Avg. Bent Wire Length": f"{avg_bent_length:.3f} mm",
        "β Avg. Real Wire Tilt Angle (From X-axis)": f"{avg_angle_signed:.2f}°",
        "Avg. Signed Tilt Angle": f"{avg_signed_tilt:.2f}°",
        "Avg. Real Wire Orientation": avg_orientation,
        "Avg. Length Deviation": f"{avg_dev_len:.3f} mm",
        "Avg. 3D Spatial Deviation": f"{avg_3d_dev:.3f} mm",
        "Avg. Angular Deviation": f"{avg_angle_dev:.3f}°",
        "Avg. Real Wire Endpoint Deviation": f"({avg_x1:.3f}, {avg_y1:.3f}, {avg_z1:.3f})",
        "Avg. Overlay Wire Endpoint Deviation": f"({avg_x2:.3f}, {avg_y2:.3f}, {avg_z2:.3f})",
        "Δ Avg. Per Axis Deviation": f"dx={avg_dx:.3f}, dy={avg_dy:.3f}, dz={avg_dz:.3f}"
    }

    csv_path = os.path.join(folder, "wire_analysis_results.csv")
    # write results list of dicts to CSV (flatten nested fields simply)
    keys = list(results[0].keys())
    with open(csv_path, "w", newline="") as f:
        # naive writer: write JSON-like string of nested fields if any
        writer = csv.writer(f)
        writer.writerow(keys)
        for r in results:
            writer.writerow([r[k] if not isinstance(r[k], dict) else str(r[k]) for k in keys])

    summary_path = os.path.join(folder, "average_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        for title, value in averages.items():
            f.write(f"{title}: {value}\n")

    print(f"✅ Analysis complete for {len(results)} images.")
    print(f"📊 CSV saved: {csv_path}")
    print(f"📄 Summary saved: {summary_path}")
    for title, value in averages.items():
        print(f"{title}: {value}")

# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    analyze_all()
