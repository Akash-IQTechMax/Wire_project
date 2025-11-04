
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

# Reference physical wire length (mm)
REFERENCE_LENGTH_MM = 12.0  # 1.2 cm

# HSV range for copper-like color (fine-tuned)
COLOR_LOWER = np.array([5, 80, 50])
COLOR_UPPER = np.array([25, 255, 255])

# Manual calibration (fixed scale)
PIXELS_PER_MM = 20.0  # 1 mm = 20 pixels → adjust based on actual setup

# Unity overlay angles (degrees)
UNITY_ANGLES = {
    "X": 44.1,  # ARVR reference wire X-axis angle
    "Y": 45.7,
    "Z": 89.8
}


# ============================================================
# CALIBRATION
# ============================================================
def calibrate_reference():
    """Manual calibration for stable results."""
    global PIXELS_PER_MM
    print(f"✅ Manual calibration: 1 mm = {PIXELS_PER_MM:.2f} px  |  1 px = {1/PIXELS_PER_MM:.4f} mm")
    return True


# ============================================================
# MEASURE CURVED (BENT) LENGTH
# ============================================================
def calculate_bent_length(contour):
    """Compute actual curve length instead of bounding box."""
    length_px = cv2.arcLength(contour, False)
    return length_px


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

    # Morphological cleanup
    mask = cv2.medianBlur(mask, 7)
    mask = cv2.dilate(mask, None, iterations=2)
    mask = cv2.erode(mask, None, iterations=1)

    # Save debug mask
    debug_mask_path = os.path.join(RESULTS_DIR, f"mask_{os.path.basename(path)}")
    cv2.imwrite(debug_mask_path, mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"⚠️ No wire detected in {os.path.basename(path)}")
        return None

    cnt = max(contours, key=cv2.contourArea)
    wire_length_px = calculate_bent_length(cnt)
    wire_length_mm = wire_length_px / PIXELS_PER_MM

    # Fit a line for tilt angle
    [vx, vy, x0, y0] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy = float(vx), float(vy)
    angle_rad = math.atan2(vy, vx)
    angle_deg = math.degrees(angle_rad)

    # Normalize to -90° to +90° range
    if angle_deg > 90:
        angle_deg -= 180
    elif angle_deg < -90:
        angle_deg += 180

    # Tilt direction sign
    if angle_deg >= 0:
        tilt_direction = f"Positive ({angle_deg:.2f}°)"
    else:
        tilt_direction = f"Negative ({angle_deg:.2f}°)"

    # Compare with ARVR wire X-axis reference
    ref_angle = UNITY_ANGLES["X"]
    angle_deviation = angle_deg - ref_angle

    # 3D deviation computation (same logic)
    overlay_len = REFERENCE_LENGTH_MM
    a1, a2, a3 = map(math.radians, [UNITY_ANGLES["X"], UNITY_ANGLES["Y"], UNITY_ANGLES["Z"]])
    x2 = overlay_len * math.cos(a1)
    y2 = overlay_len * math.cos(a2)
    z2 = overlay_len * math.cos(a3)

    b1 = math.radians(angle_deg)
    x1 = wire_length_mm * math.cos(b1)
    y1 = wire_length_mm * math.sin(b1)
    z1 = 0

    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1

    deviation_3d = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    # Annotate image
    annotated = img.copy()
    cv2.drawContours(annotated, [cnt], -1, (0, 255, 0), 2)

    text1 = f"Len: {wire_length_mm:.2f}mm (bent)"
    text2 = f"Angle: {tilt_direction}"
    text3 = f"3D Dev: {deviation_3d:.2f}mm"
    cv2.putText(annotated, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(annotated, text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(annotated, text3, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    out_path = os.path.join(RESULTS_DIR, f"annotated_{os.path.basename(path)}")
    cv2.imwrite(out_path, annotated)

    result = {
        "file": os.path.basename(path),
        "bent_length_mm": round(wire_length_mm, 3),
        "angle_deg": round(angle_deg, 2),
        "tilt_direction": tilt_direction,
        "3D_deviation_mm": round(deviation_3d, 3),
        "angle_deviation_from_ARVR_deg": round(angle_deviation, 3),
        "wire_tip_real": {"x1": round(x1, 3), "y1": round(y1, 3), "z1": round(z1, 3)},
        "overlay_tip": {"x2": round(x2, 3), "y2": round(y2, 3), "z2": round(z2, 3)},
        "delta": {"dx": round(dx, 3), "dy": round(dy, 3), "dz": round(dz, 3)}
    }

    print(f"[INFO] {os.path.basename(path)} | Bent Len: {wire_length_mm:.2f} mm | Tilt: {tilt_direction} | 3D Dev: {deviation_3d:.2f} mm")
    return result


# ============================================================
# ANALYZE MULTIPLE IMAGES
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
    )[:6]

    results = []
    for f in files:
        fp = os.path.join(UPLOAD_FOLDER, f)
        res = analyze_wire_image(fp)
        if res:
            results.append(res)

    if not results:
        print("❌ No valid wires detected.")
        return

    avg_len = np.mean([r["bent_length_mm"] for r in results])
    avg_angle = np.mean([r["angle_deg"] for r in results])
    avg_3d_dev = np.mean([r["3D_deviation_mm"] for r in results])

    summary = {
        "Average Bent Wire Length": f"{avg_len:.3f} mm",
        "Average Tilt Angle": f"{avg_angle:.2f}°",
        "Average 3D Deviation": f"{avg_3d_dev:.3f} mm"
    }

    csv_path = os.path.join(folder, "wire_analysis_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    summary_path = os.path.join(folder, "average_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    print(f"✅ Analysis complete for {len(results)} images.")
    print(f"📊 CSV saved: {csv_path}")
    print(f"📄 Summary saved: {summary_path}")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    analyze_all()


