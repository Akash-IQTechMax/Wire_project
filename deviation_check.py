
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

# ================= CONFIG =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
RESULTS_DIR = os.path.join(BASE_DIR, "results_checked")
REFERENCE_IMAGE_PATH = os.path.join(BASE_DIR, "wire_reference.png")
os.makedirs(RESULTS_DIR, exist_ok=True)

REFERENCE_LENGTH_MM = 12.0        # known reference length
COLOR_LOWER = np.array([5, 80, 50])
COLOR_UPPER = np.array([25, 255, 255])

# Sensible pixel/mm bounds (tunable)
MIN_PX_PER_MM = 0.5     # if below, calibration is probably wrong
MAX_PX_PER_MM = 500.0   # if above, calibration is probably wrong

PIXELS_PER_MM = None
PIXEL_TO_MM_SKELETON = 0.26  # fallback, but we'll prefer calibrated scale

UNITY_ANGLES = {"X": 44.1, "Y": 45.7, "Z": 89.8}


# ================ UTILITIES =================
def debug_imwrite(path, img):
    try:
        cv2.imwrite(path, img)
    except Exception:
        pass


def compute_pixels_per_mm_from_contour(contour):
    # use the longer side of minAreaRect as pixel length for reference
    _, (w, h), _ = cv2.minAreaRect(contour)
    px = max(w, h)
    if px <= 0:
        return None
    return px / REFERENCE_LENGTH_MM


# ================ CALIBRATION =================
def calibrate_reference():
    """Calibrate using reference image. Returns True if OK."""
    global PIXELS_PER_MM
    img = cv2.imread(REFERENCE_IMAGE_PATH)
    if img is None:
        print("❌ Missing reference image for calibration.")
        return False

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, COLOR_LOWER, COLOR_UPPER)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("⚠️ No contour found in reference image — trying adaptive range.")
        # adaptive fallback
        h, s, v = cv2.split(hsv)
        if np.count_nonzero(h) == 0:
            print("❌ Reference image empty / invalid.")
            return False
        lower = np.array([np.percentile(h, 5), np.percentile(s, 5), np.percentile(v, 5)], dtype=np.uint8)
        upper = np.array([np.percentile(h, 95), np.percentile(s, 95), np.percentile(v, 95)], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        debug_imwrite(os.path.join(RESULTS_DIR, "ref_mask_adaptive.png"), mask)

    if not contours:
        print("❌ Still no contour found in reference image. Calibration failed.")
        return False

    # pick largest contour
    c = max(contours, key=cv2.contourArea)
    px_per_mm = compute_pixels_per_mm_from_contour(c)
    if px_per_mm is None:
        print("❌ Could not compute pixels per mm from contour.")
        return False

    # sanity check
    if not (MIN_PX_PER_MM <= px_per_mm <= MAX_PX_PER_MM):
        print(f"⚠️ Unusual PIXELS_PER_MM = {px_per_mm:.4f} px/mm (outside [{MIN_PX_PER_MM},{MAX_PX_PER_MM}])")
        # Save debug images for inspection
        annotated = img.copy()
        cv2.drawContours(annotated, [c], -1, (0, 0, 255), 2)
        debug_imwrite(os.path.join(RESULTS_DIR, "ref_annotated_bad.png"), annotated)
        # still accept but warn -- better to abort so user can inspect
        print("❌ Calibration looks wrong. Please check wire_reference.png and color thresholds.")
        return False

    PIXELS_PER_MM = px_per_mm
    print(f"✅ Calibrated: 1 mm = {PIXELS_PER_MM:.3f} px  |  1 px = {1/PIXELS_PER_MM:.6f} mm")
    debug_imwrite(os.path.join(RESULTS_DIR, "ref_mask.png"), mask)
    return True


# ================ SKELETON LENGTH HELPER =================
def measure_bent_wire_length_skeleton(img):
    """Return (length_px, length_mm) or (None, None)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # limit to non-black regions
    mask_bg = cv2.inRange(hsv, (0, 30, 30), (180, 255, 255))
    h, s, v = cv2.split(hsv)
    # avoid empty percentiles
    hvals = h[mask_bg > 0]
    if hvals.size == 0:
        return None, None
    lower = np.array([np.percentile(hvals, 5), np.percentile(s[mask_bg > 0], 5), np.percentile(v[mask_bg > 0], 5)], dtype=np.uint8)
    upper = np.array([np.percentile(hvals, 95), np.percentile(s[mask_bg > 0], 95), np.percentile(v[mask_bg > 0], 95)], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    # skeletonization
    skel = np.zeros(mask.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    size = np.size(mask)
    done = False
    count = 0
    while not done and count < 1000:
        eroded = cv2.erode(mask, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(mask, temp)
        skel = cv2.bitwise_or(skel, temp)
        mask = eroded.copy()
        zeros = size - cv2.countNonZero(mask)
        if zeros == size:
            done = True
        count += 1
    points = np.column_stack(np.where(skel > 0))
    if len(points) < 2:
        return None, None
    # sort points to follow path approximately (sort by row then col)
    points = points[np.lexsort((points[:, 1], points[:, 0]))]
    length_px = sum(np.linalg.norm(points[i] - points[i - 1]) for i in range(1, len(points)))
    # prefer calibrated pixels per mm if available
    if PIXELS_PER_MM and PIXELS_PER_MM > 0:
        length_mm = length_px / PIXELS_PER_MM
    else:
        length_mm = length_px * PIXEL_TO_MM_SKELETON
    return length_px, length_mm


# ================ SINGLE IMAGE ANALYSIS =================
def analyze_wire_image(path):
    img = cv2.imread(path)
    if img is None:
        print(f"❌ Cannot read {path}")
        return None

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, COLOR_LOWER, COLOR_UPPER)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # fallback: adaptive mask
    if not contours:
        h, s, v = cv2.split(hsv)
        if np.count_nonzero(h) == 0:
            print("⚠️ Empty image or invalid color channels.")
            return None
        lower = np.array([np.percentile(h, 5), np.percentile(s, 5), np.percentile(v, 5)], dtype=np.uint8)
        upper = np.array([np.percentile(h, 95), np.percentile(s, 95), np.percentile(v, 95)], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        debug_imwrite(os.path.join(RESULTS_DIR, os.path.basename(path) + "_adaptive_mask.png"), mask)

    if not contours:
        print(f"⚠️ No wire detected in {os.path.basename(path)}")
        return None

    # pick the largest contour by area
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    img_area = img.shape[0] * img.shape[1]
    # if contour is huge (like > 50% of image), it's likely wrong
    if area > 0.5 * img_area:
        print(f"⚠️ Largest contour area is large ({area:.0f} px) — possible mask overreach. Saving debug mask.")
        debug_imwrite(os.path.join(RESULTS_DIR, os.path.basename(path) + "_mask.png"), mask)
        # try to find a smaller candidate (by aspect ratio or thinness)
        # find thin contours by bounding rectangle aspect ratio
        candidates = [c for c in contours if cv2.contourArea(c) > 50]
        thin_candidates = []
        for c in candidates:
            x, y, w, h = cv2.boundingRect(c)
            if (w == 0 or h == 0):
                continue
            aspect = max(w/h, h/w)
            if aspect > 3 or cv2.contourArea(c) < 0.05 * img_area:
                thin_candidates.append(c)
        if thin_candidates:
            cnt = max(thin_candidates, key=cv2.contourArea)
            print("➡️ Selected thin candidate contour instead of giant area.")
        else:
            # continue but will use skeleton fallback later
            print("➡️ No good thin candidate — will attempt skeleton-based length as fallback.")

    # compute fitted line and signed angle
    vx, vy, x0, y0 = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
    angle_rad = math.atan2(float(vy), float(vx))
    angle_deg = math.degrees(angle_rad)
    # normalize to [-180,180]
    if angle_deg > 180:
        angle_deg -= 360

    # contour arc length (true curved length)
    try:
        arc_len_px = cv2.arcLength(cnt, closed=False)
    except Exception:
        arc_len_px = None

    # skeleton fallback length
    skel_px, skel_mm = measure_bent_wire_length_skeleton(img)

    # convert to mm using calibrated scale if present
    arc_len_mm = None
    if arc_len_px is not None and PIXELS_PER_MM and PIXELS_PER_MM > 0:
        arc_len_mm = arc_len_px / PIXELS_PER_MM

    # Decide which length to use:
    chosen_len_mm = None
    reason = ""
    # if both exist, compare
    if arc_len_mm is not None and skel_mm is not None:
        # If arc length is wildly larger than skeleton (noise), prefer skeleton
        if arc_len_mm > 5 * skel_mm and skel_mm < 500:
            chosen_len_mm = skel_mm
            reason = "Used skeleton length (arc >> skel)"
        else:
            # choose average of the two for stability
            chosen_len_mm = (arc_len_mm + skel_mm) / 2.0
            reason = "Averaged arc & skeleton lengths"
    elif arc_len_mm is not None:
        chosen_len_mm = arc_len_mm
        reason = "Used arc length (no skeleton)"
    elif skel_mm is not None:
        chosen_len_mm = skel_mm
        reason = "Used skeleton length (no arc)"
    else:
        print(f"⚠️ Could not compute length for {os.path.basename(path)}")
        return None

    # sanity clamp: if chosen length is absurd (e.g., > 1000 mm), flag it
    if chosen_len_mm > 1000:
        print(f"❌ Measured length {chosen_len_mm:.2f} mm is unrealistic. Check calibration/reference/mask.")
        # Save debug images and return None so it won't be included
        annotated = img.copy()
        cv2.drawContours(annotated, [cnt], -1, (0, 0, 255), 2)
        debug_imwrite(os.path.join(RESULTS_DIR, "debug_" + os.path.basename(path)), annotated)
        return None

    # bounding rectangle based box length (useful for extra check)
    x, y, w, h = cv2.boundingRect(cnt)
    bbox_len_mm = max(w, h) / PIXELS_PER_MM if PIXELS_PER_MM else None

    # orientation label (signed)
    if -5 <= angle_deg <= 5:
        orientation = "Horizontal"
    elif angle_deg > 5:
        orientation = f"Tilted Up (+{angle_deg:.1f}°)"
    else:
        orientation = f"Tilted Down ({angle_deg:.1f}°)"

    # compute 3D deviation using overlay vector
    a1, a2, a3 = map(math.radians, [UNITY_ANGLES["X"], UNITY_ANGLES["Y"], UNITY_ANGLES["Z"]])
    overlay_len = REFERENCE_LENGTH_MM
    x2, y2, z2 = (overlay_len * math.cos(a1), overlay_len * math.cos(a2), overlay_len * math.cos(a3))
    b1 = math.radians(angle_deg)
    x1, y1, z1 = (chosen_len_mm * math.cos(b1), chosen_len_mm * math.sin(b1), 0)
    dx, dy3d, dz = (x2 - x1, y2 - y1, z2 - z1)
    deviation_3d = math.sqrt(dx**2 + dy3d**2 + dz**2)
    angle_dev = abs(angle_deg - UNITY_ANGLES["X"])

    # annotate and save
    annotated = img.copy()
    cv2.drawContours(annotated, [cv2.boxPoints(cv2.minAreaRect(cnt)).astype(int)], 0, (0, 255, 0), 2)
    info1 = f"Len(mm):{chosen_len_mm:.2f} ({reason})"
    info2 = f"Angle:{angle_deg:.2f}° | 3DDev:{deviation_3d:.3f}mm"
    cv2.putText(annotated, info1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(annotated, info2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    debug_imwrite(os.path.join(RESULTS_DIR, "annotated_" + os.path.basename(path)), annotated)

    print(f"[INFO] {os.path.basename(path)} | Len(mm): {chosen_len_mm:.2f} | Angle: {angle_deg:.2f}° | Reason: {reason}")

    return {
        "file": os.path.basename(path),
        "length_mm": round(chosen_len_mm, 3),
        "angle_deg": round(angle_deg, 2),
        "orientation": orientation,
        "deviation_len_mm": round(abs(chosen_len_mm - REFERENCE_LENGTH_MM), 3),
        "3D_deviation_mm": round(deviation_3d, 3),
        "angle_deviation_deg": round(angle_dev, 3),
        "wire_tip_real": {"x1": round(x1, 4), "y1": round(y1, 4), "z1": round(z1, 4)},
        "overlay_tip": {"x2": round(x2, 4), "y2": round(y2, 4), "z2": round(z2, 4)},
        "delta": {"dx": round(dx, 4), "dy": round(dy3d, 4), "dz": round(dz, 4)},
        "arc_len_px": round(arc_len_px, 2) if arc_len_px is not None else None,
        "skel_len_mm": round(skel_mm, 3) if skel_mm is not None else None,
        "bbox_len_mm": round(bbox_len_mm, 3) if bbox_len_mm is not None else None
    }


# ================ MULTI-IMAGE ANALYSIS =================
def analyze_all():
    if not calibrate_reference():
        print("❌ Calibration failed. Fix reference image or HSV thresholds.")
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
        res = analyze_wire_image(os.path.join(UPLOAD_FOLDER, f))
        if res:
            results.append(res)

    if not results:
        print("❌ No valid images processed.")
        return

    # write CSV
    csv_path = os.path.join(folder, "wire_analysis_results_checked.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # compute averages
    avg_length = np.mean([r["length_mm"] for r in results])
    avg_angle = np.mean([r["angle_deg"] for r in results])
    avg_3d = np.mean([r["3D_deviation_mm"] for r in results])
    avg_angle_dev = np.mean([r["angle_deviation_deg"] for r in results])

    summary_path = os.path.join(folder, "average_summary_checked.txt")
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write(f"Avg. Real (bent) Wire Length: {avg_length:.3f} mm\n")
        fh.write(f"β Avg. Real Wire Tilt Angle (From X-axis): {avg_angle:.2f}°\n")
        fh.write(f"Avg. 3D Spatial Deviation: {avg_3d:.3f} mm\n")
        fh.write(f"Avg. Angular Deviation: {avg_angle_dev:.3f}°\n")

    print(f"\n✅ Analysis complete for {len(results)} images.")
    print(f"📊 CSV saved: {csv_path}")
    print(f"📄 Summary saved: {summary_path}")


# ================ ENTRY POINT =================
if __name__ == "__main__":
    analyze_all()
