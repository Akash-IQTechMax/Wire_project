
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

REFERENCE_LENGTH_MM = 12.0        # fixed AR wire length (1.2 cm)
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
        print(f"⚠️ No wire detected in {os.path.basename(path)}")
        return None

    cnt = max(contours, key=cv2.contourArea)

    # Fit a straight line for more stable angle detection
    [vx, vy, x0, y0] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy = float(vx), float(vy)
    angle_rad = math.atan2(vy, vx)
    angle_deg = math.degrees(angle_rad)
    if angle_deg < 0:
        angle_deg += 180

    # Bounding box for length estimation
    x, y, w, h = cv2.boundingRect(cnt)
    wire_length_px = max(w, h)
    wire_length_mm = wire_length_px / PIXELS_PER_MM
    deviation_len = abs(wire_length_mm - REFERENCE_LENGTH_MM)

    # Orientation / direction
    if 85 <= angle_deg <= 95:
        orientation = "Vertical"
    elif angle_deg < 85:
        orientation = f"Tilted Right ({angle_deg:.1f}°)"
    else:
        orientation = f"Tilted Left ({180 - angle_deg:.1f}°)"

    # ------------------------------------------------------------
    # 3D vector calculation
    # ------------------------------------------------------------
    overlay_len = REFERENCE_LENGTH_MM
    a1, a2, a3 = map(math.radians, [UNITY_ANGLES["X"], UNITY_ANGLES["Y"], UNITY_ANGLES["Z"]])
    x2 = overlay_len * math.cos(a1)
    y2 = overlay_len * math.cos(a2)
    z2 = overlay_len * math.cos(a3)

    # Real wire vector
    beta1 = angle_deg
    b1 = math.radians(beta1)
    x1 = wire_length_mm * math.cos(b1)
    y1 = wire_length_mm * math.sin(b1)
    z1 = 0

    # Axis deviations
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1

    # 3D deviation distance
    deviation_3d = math.sqrt(dx**2 + dy**2 + dz**2)

    # Angular deviation
    angle_dev = abs(beta1 - UNITY_ANGLES["X"])

    # ------------------------------------------------------------
    # Annotate image
    # ------------------------------------------------------------
    annotated = img.copy()
    box = np.intp(cv2.boxPoints(cv2.minAreaRect(cnt)))
    cv2.drawContours(annotated, [box], 0, (0, 255, 0), 2)

    text1 = f"Len: {wire_length_mm:.2f}mm | Angle: {angle_deg:.2f}°"
    text2 = f"3D Dev: {deviation_3d:.3f}mm | ΔAngle: {angle_dev:.2f}°"
    cv2.putText(annotated, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(annotated, text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(annotated, orientation, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    out_path = os.path.join(RESULTS_DIR, f"annotated_{os.path.basename(path)}")
    cv2.imwrite(out_path, annotated)

    # ------------------------------------------------------------
    # Return structured results
    # ------------------------------------------------------------
    result = {
        "file": os.path.basename(path),
        "length_mm": round(wire_length_mm, 3),
        "angle_deg": round(angle_deg, 2),
        "orientation": orientation,
        "deviation_len_mm": round(deviation_len, 3),
        "3D_deviation_mm": round(deviation_3d, 3),
        "angle_deviation_deg": round(angle_dev, 3),
        "wire_tip_real": {"x1": round(x1, 4), "y1": round(y1, 4), "z1": round(z1, 4)},
        "overlay_tip": {"x2": round(x2, 4), "y2": round(y2, 4), "z2": round(z2, 4)},
        "delta": {"dx": round(dx, 4), "dy": round(dy, 4), "dz": round(dz, 4)}
    }

    print(f"[INFO] {os.path.basename(path)} | Angle: {angle_deg:.2f}° | 3D Dev: {deviation_3d:.3f} mm")
    return result


# ============================================================
# ANALYZE 6 IMAGES AND AVERAGE
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
        print("❌ No valid images processed.")
        return

    # ---- Compute averages ----
    avg_length = np.mean([r["length_mm"] for r in results])
    avg_angle = np.mean([r["angle_deg"] for r in results])
    avg_dev_len = np.mean([r["deviation_len_mm"] for r in results])
    avg_3d_dev = np.mean([r["3D_deviation_mm"] for r in results])
    avg_angle_dev = np.mean([r["angle_deviation_deg"] for r in results])

    # Orientation majority
    orientations = [r["orientation"].split()[0] for r in results]
    avg_orientation = max(set(orientations), key=orientations.count)

    # Real and overlay endpoints (averaged)
    avg_x1 = np.mean([r["wire_tip_real"]["x1"] for r in results])
    avg_y1 = np.mean([r["wire_tip_real"]["y1"] for r in results])
    avg_z1 = np.mean([r["wire_tip_real"]["z1"] for r in results])

    avg_x2 = np.mean([r["overlay_tip"]["x2"] for r in results])
    avg_y2 = np.mean([r["overlay_tip"]["y2"] for r in results])
    avg_z2 = np.mean([r["overlay_tip"]["z2"] for r in results])

    avg_dx = np.mean([r["delta"]["dx"] for r in results])
    avg_dy = np.mean([r["delta"]["dy"] for r in results])
    avg_dz = np.mean([r["delta"]["dz"] for r in results])

    # ---- Ordered output ----
    averages = {
        "Avg. Real Wire Length": f"{avg_length:.3f} mm",
        "β Avg. Real Wire Tilt Angle (From X-axis)": f"{avg_angle:.2f}°",
        "Avg. Real Wire Orientation": avg_orientation,
        "Avg. Length Deviation": f"{avg_dev_len:.3f} mm",
        "Avg. 3D Spatial Deviation": f"{avg_3d_dev/10:.3f} mm",
        "Avg. Angular Deviation": f"{avg_angle_dev:.3f}°",
        "Avg. Real Wire Endpoint Deviation": f"({avg_x1:.3f}, {avg_y1:.3f}, {avg_z1:.3f})",
        "Avg. Overlay Wire Endpoint Deviation": f"({avg_x2:.3f}, {avg_y2:.3f}, {avg_z2:.3f})",
        "Δ Avg. Per Axis Deviation": f"dx={avg_dx:.3f}, dy={avg_dy:.3f}, dz={avg_dz:.3f}"
    }

    # ---- Save all data ----
    csv_path = os.path.join(folder, "wire_analysis_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    summary_path = os.path.join(folder, "average_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        for title, value in averages.items():
            f.write(f"{title}: {value}\n")

    print(f"✅ Analysis complete for {len(results)} images.")
    print(f"📊 CSV saved: {csv_path}")
    print(f"📄 Summary saved: {summary_path}")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    analyze_all()







# import cv2
# import numpy as np
# import os
# import math
# import csv
# from datetime import datetime

# # ================================================================
# # CONFIGURATION
# # ================================================================
# REFERENCE_LENGTH_MM = 1.20  # Known wire length (mm)
# UPLOAD_FOLDER = "uploads"
# RESULTS_DIR = "results"

# os.makedirs(RESULTS_DIR, exist_ok=True)


# # ================================================================
# # SINGLE IMAGE ANALYSIS
# # ================================================================
# def analyze_single_image(image_path, results_dir, return_json=True):
#     """Analyze a single wire image for length, angle, direction, and deviation."""

#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError(f"❌ Invalid image: {image_path}")

#     original = image.copy()
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)
#     edges = cv2.Canny(blur, 40, 150)

#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if not contours:
#         raise ValueError("⚠️ No wire detected")

#     cnt = max(contours, key=cv2.contourArea)
#     x, y, w, h = cv2.boundingRect(cnt)
#     wire_length_px = max(w, h)

#     # Fit line to find angle
#     [vx, vy, x0, y0] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
#     angle_rad = math.atan2(vy, vx)
#     angle_deg = math.degrees(angle_rad)
#     if angle_deg < 0:
#         angle_deg += 180

#     # Direction logic
#     if 85 <= angle_deg <= 95:
#         direction = "Vertical"
#     elif angle_deg < 85:
#         direction = f"Tilted Right ({round(angle_deg, 1)}°)"
#     else:
#         direction = f"Tilted Left ({round(180 - angle_deg, 1)}°)"

#     # Wire deviation
#     line_pts = cnt.reshape(-1, 2)
#     distances = [abs(vy * px - vx * py + x0 * vy - y0 * vx) for (px, py) in line_pts]
#     deviation_px = np.std(distances)
#     deviation_mm = round(deviation_px / (wire_length_px / REFERENCE_LENGTH_MM), 4)

#     # Convert to mm
#     pixels_per_mm = wire_length_px / REFERENCE_LENGTH_MM
#     wire_length_mm = round(wire_length_px / pixels_per_mm, 3)
#     wire_diameter_mm = round(wire_length_mm * 0.22, 3)  # simple estimation

#     # Annotate
#     annotated = original.copy()
#     cv2.drawContours(annotated, [cnt], -1, (0, 255, 0), 2)
#     cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 0, 0), 2)
#     height, width = annotated.shape[:2]
#     lefty = int((-x0 * vy / vx) + y0)
#     righty = int(((width - x0) * vy / vx) + y0)
#     cv2.line(annotated, (width - 1, righty), (0, lefty), (0, 0, 255), 2)

#     # Overlay text
#     cv2.putText(annotated, f"Angle: {round(angle_deg, 2)}°", (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
#     cv2.putText(annotated, direction, (10, 60),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
#     cv2.putText(annotated, f"Deviation: {deviation_mm} mm", (10, 90),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
#     cv2.putText(annotated, f"Length: {wire_length_mm} mm", (10, 120),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

#     out_name = f"analyzed_{os.path.basename(image_path)}"
#     out_path = os.path.join(results_dir, out_name)
#     cv2.imwrite(out_path, annotated)

#     result = {
#         "filename": os.path.basename(image_path),
#         "length_mm": wire_length_mm,
#         "diameter_mm": wire_diameter_mm,
#         "angle_deg": round(angle_deg, 2),
#         "direction": direction,
#         "deviation_mm": deviation_mm,
#         "annotated_image": f"/results/{out_name}"
#     }

#     print(f"✅ Analysis for {os.path.basename(image_path)}: {result}")
#     return result if return_json else annotated


# # ================================================================
# # MULTI-IMAGE SUMMARY + CSV
# # ================================================================
# def analyze_all_images():
#     image_files = sorted(
#         [f for f in os.listdir(UPLOAD_FOLDER) if f.lower().endswith((".png", ".jpg", ".jpeg"))],
#         key=lambda f: os.path.getmtime(os.path.join(UPLOAD_FOLDER, f)),
#         reverse=True
#     )[:6]

#     if not image_files:
#         raise ValueError("No images found for analysis")

#     all_results = []
#     for img in image_files:
#         path = os.path.join(UPLOAD_FOLDER, img)
#         result = analyze_single_image(path, RESULTS_DIR)
#         all_results.append(result)

#     # Compute mean values
#     avg_length = np.mean([r["length_mm"] for r in all_results])
#     avg_diameter = np.mean([r["diameter_mm"] for r in all_results])
#     avg_dev = np.mean([r["deviation_mm"] for r in all_results])

#     summary_text = (
#         "=== SUMMARY OF RESULTS ===\n"
#         f"Average Wire Length Across All Views: {round(avg_length, 3)} mm\n"
#         f"Average Wire Diameter: {round(avg_diameter, 3)} mm\n"
#         f"Average Length Deviation from AR Model: {round(avg_dev, 3)} mm"
#     )

#     # Save to CSV
#     csv_name = f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
#     csv_path = os.path.join(RESULTS_DIR, csv_name)

#     with open(csv_path, "w", newline="") as csvfile:
#         fieldnames = list(all_results[0].keys())
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         writer.writeheader()
#         writer.writerows(all_results)
#         writer.writerow({})
#         writer.writerow({"filename": "AVERAGES", "length_mm": avg_length, "diameter_mm": avg_diameter, "deviation_mm": avg_dev})

#     print(summary_text)
#     print(f"✅ Detailed results saved to: {csv_path}")

#     return {
#         "summary": summary_text,
#         "csv_path": f"/results/{csv_name}",
#         "results": all_results
#     }