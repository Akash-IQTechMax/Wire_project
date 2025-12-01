
# # import cv2
# # import numpy as np
# # import os
# # import csv
# # import math
# # from datetime import datetime

# # # ============================================================
# # # CONFIGURATION
# # # ============================================================
# # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# # UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
# # RESULTS_DIR = os.path.join(BASE_DIR, "results")
# # REFERENCE_IMAGE_PATH = os.path.join(BASE_DIR, "wire_reference.png")

# # os.makedirs(RESULTS_DIR, exist_ok=True)

# # REFERENCE_LENGTH_MM = 12.0        # fixed AR wire length (1.2 cm)
# # COLOR_LOWER = np.array([5, 80, 50])
# # COLOR_UPPER = np.array([25, 255, 255])

# # PIXELS_PER_MM = None

# # # ---- Fixed Unity overlay angles (degrees) ----
# # UNITY_ANGLES = {
# #     "X": 44.1,
# #     "Y": 45.7,
# #     "Z": 89.8
# # }


# # # ============================================================
# # # CALIBRATION
# # # ============================================================
# # def calibrate_reference():
# #     """Calibrate using reference image."""
# #     global PIXELS_PER_MM

# #     img = cv2.imread(REFERENCE_IMAGE_PATH)
# #     if img is None:
# #         print("❌ Missing reference image for calibration.")
# #         return False

# #     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# #     mask = cv2.inRange(hsv, COLOR_LOWER, COLOR_UPPER)
# #     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# #     if not contours:
# #         print("❌ No contour found in reference image.")
# #         return False

# #     c = max(contours, key=cv2.contourArea)
# #     _, (w, h), _ = cv2.minAreaRect(c)
# #     PIXELS_PER_MM = max(w, h) / REFERENCE_LENGTH_MM
# #     print(f"✅ Calibrated: 1 mm = {PIXELS_PER_MM:.3f} px  |  1 px = {1/PIXELS_PER_MM:.6f} mm")
# #     return True


# # # ============================================================
# # # ANALYZE SINGLE IMAGE
# # # ============================================================
# # def analyze_wire_image(path):
# #     img = cv2.imread(path)
# #     if img is None:
# #         print(f"❌ Cannot read {path}")
# #         return None

# #     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# #     mask = cv2.inRange(hsv, COLOR_LOWER, COLOR_UPPER)
# #     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# #     if not contours:
# #         print(f"⚠️ No wire detected in {os.path.basename(path)}")
# #         return None

# #     cnt = max(contours, key=cv2.contourArea)

# #     # Fit a straight line for more stable angle detection
# #     [vx, vy, x0, y0] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
# #     vx, vy = float(vx), float(vy)
# #     angle_rad = math.atan2(vy, vx)
# #     angle_deg = math.degrees(angle_rad)
# #     if angle_deg < 0:
# #         angle_deg += 180

# #     # Bounding box for length estimation
# #     x, y, w, h = cv2.boundingRect(cnt)
# #     wire_length_px = max(w, h)
# #     wire_length_mm = wire_length_px / PIXELS_PER_MM
# #     deviation_len = abs(wire_length_mm - REFERENCE_LENGTH_MM)

# #     # Orientation / direction
# #     if 85 <= angle_deg <= 95:
# #         orientation = "Vertical"
# #     elif angle_deg < 85:
# #         orientation = f"Tilted Right ({angle_deg:.1f}°)"
# #     else:
# #         orientation = f"Tilted Left ({180 - angle_deg:.1f}°)"

# #     # ------------------------------------------------------------
# #     # 3D vector calculation
# #     # ------------------------------------------------------------
# #     overlay_len = REFERENCE_LENGTH_MM
# #     a1, a2, a3 = map(math.radians, [UNITY_ANGLES["X"], UNITY_ANGLES["Y"], UNITY_ANGLES["Z"]])
# #     x2 = overlay_len * math.cos(a1)
# #     y2 = overlay_len * math.cos(a2)
# #     z2 = overlay_len * math.cos(a3)

# #     # Real wire vector
# #     beta1 = angle_deg
# #     b1 = math.radians(beta1)
# #     x1 = wire_length_mm * math.cos(b1)
# #     y1 = wire_length_mm * math.sin(b1)
# #     z1 = 0

# #     # Axis deviations
# #     dx = x2 - x1
# #     dy = y2 - y1
# #     dz = z2 - z1

# #     # 3D deviation distance
# #     deviation_3d = math.sqrt(dx**2 + dy**2 + dz**2)

# #     # Angular deviation
# #     angle_dev = abs(beta1 - UNITY_ANGLES["X"])

# #     # ------------------------------------------------------------
# #     # Annotate image
# #     # ------------------------------------------------------------
# #     annotated = img.copy()
# #     box = np.intp(cv2.boxPoints(cv2.minAreaRect(cnt)))
# #     cv2.drawContours(annotated, [box], 0, (0, 255, 0), 2)

# #     text1 = f"Len: {wire_length_mm:.2f}mm | Angle: {angle_deg:.2f}°"
# #     text2 = f"3D Dev: {deviation_3d:.3f}mm | ΔAngle: {angle_dev:.2f}°"
# #     cv2.putText(annotated, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
# #     cv2.putText(annotated, text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
# #     cv2.putText(annotated, orientation, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

# #     out_path = os.path.join(RESULTS_DIR, f"annotated_{os.path.basename(path)}")
# #     cv2.imwrite(out_path, annotated)

# #     # ------------------------------------------------------------
# #     # Return structured results
# #     # ------------------------------------------------------------
# #     result = {
# #         "file": os.path.basename(path),
# #         "length_mm": round(wire_length_mm, 3),
# #         "angle_deg": round(angle_deg, 2),
# #         "orientation": orientation,
# #         "deviation_len_mm": round(deviation_len, 3),
# #         "3D_deviation_mm": round(deviation_3d, 3),
# #         "angle_deviation_deg": round(angle_dev, 3),
# #         "wire_tip_real": {"x1": round(x1, 4), "y1": round(y1, 4), "z1": round(z1, 4)},
# #         "overlay_tip": {"x2": round(x2, 4), "y2": round(y2, 4), "z2": round(z2, 4)},
# #         "delta": {"dx": round(dx, 4), "dy": round(dy, 4), "dz": round(dz, 4)}
# #     }

# #     print(f"[INFO] {os.path.basename(path)} | Angle: {angle_deg:.2f}° | 3D Dev: {deviation_3d:.3f} mm")
# #     return result


# # # ============================================================
# # # ANALYZE 6 IMAGES AND AVERAGE
# # # ============================================================
# # def analyze_all():
# #     if not calibrate_reference():
# #         print("❌ Calibration failed.")
# #         return

# #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# #     folder = os.path.join(RESULTS_DIR, timestamp)
# #     os.makedirs(folder, exist_ok=True)

# #     files = sorted(
# #         [f for f in os.listdir(UPLOAD_FOLDER) if f.lower().endswith((".png", ".jpg", ".jpeg"))],
# #         key=lambda x: os.path.getmtime(os.path.join(UPLOAD_FOLDER, x)),
# #         reverse=True
# #     )[:6]

# #     results = []
# #     for f in files:
# #         fp = os.path.join(UPLOAD_FOLDER, f)
# #         res = analyze_wire_image(fp)
# #         if res:
# #             results.append(res)

# #     if not results:
# #         print("❌ No valid images processed.")
# #         return

# #     # ---- Compute averages ----
# #     avg_length = np.mean([r["length_mm"] for r in results])
# #     avg_angle = np.mean([r["angle_deg"] for r in results])
# #     avg_dev_len = np.mean([r["deviation_len_mm"] for r in results])
# #     avg_3d_dev = np.mean([r["3D_deviation_mm"] for r in results])
# #     avg_angle_dev = np.mean([r["angle_deviation_deg"] for r in results])

# #     # Orientation majority
# #     orientations = [r["orientation"].split()[0] for r in results]
# #     avg_orientation = max(set(orientations), key=orientations.count)

# #     # Real and overlay endpoints (averaged)
# #     avg_x1 = np.mean([r["wire_tip_real"]["x1"] for r in results])
# #     avg_y1 = np.mean([r["wire_tip_real"]["y1"] for r in results])
# #     avg_z1 = np.mean([r["wire_tip_real"]["z1"] for r in results])

# #     avg_x2 = np.mean([r["overlay_tip"]["x2"] for r in results])
# #     avg_y2 = np.mean([r["overlay_tip"]["y2"] for r in results])
# #     avg_z2 = np.mean([r["overlay_tip"]["z2"] for r in results])

# #     avg_dx = np.mean([r["delta"]["dx"] for r in results])
# #     avg_dy = np.mean([r["delta"]["dy"] for r in results])
# #     avg_dz = np.mean([r["delta"]["dz"] for r in results])

# #     # ---- Ordered output ----
# #     averages = {
# #         "Avg. Real Wire Length": f"{avg_length:.3f} mm",
# #         "β Avg. Real Wire Tilt Angle (From X-axis)": f"{avg_angle:.2f}°",
# #         "Avg. Real Wire Orientation": avg_orientation,
# #         "Avg. Length Deviation": f"{avg_dev_len:.3f} mm",
# #         "Avg. 3D Spatial Deviation": f"{avg_3d_dev/10:.3f} mm",
# #         "Avg. Angular Deviation": f"{avg_angle_dev:.3f}°",
# #         "Avg. Real Wire Endpoint Deviation": f"({avg_x1:.3f}, {avg_y1:.3f}, {avg_z1:.3f})",
# #         "Avg. Overlay Wire Endpoint Deviation": f"({avg_x2:.3f}, {avg_y2:.3f}, {avg_z2:.3f})",
# #         "Δ Avg. Per Axis Deviation": f"dx={avg_dx:.3f}, dy={avg_dy:.3f}, dz={avg_dz:.3f}"
# #     }

# #     # ---- Save all data ----
# #     csv_path = os.path.join(folder, "wire_analysis_results.csv")
# #     with open(csv_path, "w", newline="") as f:
# #         writer = csv.DictWriter(f, fieldnames=results[0].keys())
# #         writer.writeheader()
# #         writer.writerows(results)

# #     summary_path = os.path.join(folder, "average_summary.txt")
# #     with open(summary_path, "w", encoding="utf-8") as f:
# #         for title, value in averages.items():
# #             f.write(f"{title}: {value}\n")

# #     print(f"✅ Analysis complete for {len(results)} images.")
# #     print(f"📊 CSV saved: {csv_path}")
# #     print(f"📄 Summary saved: {summary_path}")


# # # ============================================================
# # # ENTRY POINT
# # # ============================================================
# # if __name__ == "__main__":
# #     analyze_all()







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
# os.makedirs(RESULTS_DIR, exist_ok=True)

# # Calibration: 1 mm = 9.727 px
# PIXELS_PER_MM = 9.727
# MM_PER_PIXEL = 1 / PIXELS_PER_MM

# # HSV range for copper-like wire color
# COLOR_LOWER = np.array([5, 80, 50])
# COLOR_UPPER = np.array([25, 255, 255])


# def count_bends(contour, epsilon_factor=0.01):
#     """Approximate contour and count number of bend points."""
#     epsilon = epsilon_factor * cv2.arcLength(contour, True)
#     approx = cv2.approxPolyDP(contour, epsilon, True)

#     # Bends = number of significant direction changes
#     bend_count = len(approx) - 2 if len(approx) > 2 else 0
#     return bend_count


# # ============================================================
# # FUNCTION: Analyze a single wire
# # ============================================================
# def analyze_wire(image_path):
#     image = cv2.imread(image_path)
#     if image is None:
#         print(f"⚠️ Cannot read image: {image_path}")
#         return None

#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     mask = cv2.inRange(hsv, COLOR_LOWER, COLOR_UPPER)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if not contours:
#         print(f"⚠️ No wire detected in {os.path.basename(image_path)}")
#         return None

#     contour = max(contours, key=cv2.contourArea)
#     length_px = cv2.arcLength(contour, False)
#     length_mm = length_px * MM_PER_PIXEL

#     # Fit a line to determine tilt
#     [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
#     angle_deg = math.degrees(math.atan2(vy, vx))

#     # Normalize tilt angle to (-90, 90)
#     if angle_deg < -90:
#         angle_deg += 180
#     elif angle_deg > 90:
#         angle_deg -= 180

#     tilt_direction = "Positive" if angle_deg >= 0 else "Negative"

#     # Count number of bends
#     bend_count = count_bends(contour)

#     return {
#         "image": os.path.basename(image_path),
#         "length_mm": float(length_mm),
#         "angle_deg": float(angle_deg),
#         "tilt_direction": tilt_direction,
#         "bend_count": bend_count
#     }


# # ============================================================
# # FUNCTION: Compute Averages
# # ============================================================
# def compute_averages(results):
#     if not results:
#         return None

#     lengths = [r["length_mm"] for r in results]
#     angles = [r["angle_deg"] for r in results]
#     bends = [r["bend_count"] for r in results]

#     avg_length = np.mean(lengths)
#     avg_angle = np.mean(angles)
#     avg_bends = np.mean(bends)
#     avg_3d_dev = np.std(lengths)


#     pos_tilts = sum(1 for r in results if r["tilt_direction"] == "Positive")
#     neg_tilts = sum(1 for r in results if r["tilt_direction"] == "Negative")
#     tilt_summary = (
#         "Mostly Positive Tilt" if pos_tilts > neg_tilts
#         else "Mostly Negative Tilt" if neg_tilts > pos_tilts
#         else "Balanced Tilt"
#     )

#     # Real wire (for Unity reference)
#     avg_real_length = avg_length * 1.39
#     avg_real_angle = 92.33
#     orientation = "Vertical" if abs(avg_angle) > 45 else "Horizontal"

#     summary = {
#         "Average Bent Wire Length": f"{avg_length:.3f} mm",
#         "Average Tilt Angle": f"{avg_angle:.2f}° ({tilt_summary})",
#         "Average 3D Deviation": f"{avg_3d_dev:.3f} mm",
#         "Average Bend Count": f"{avg_bends:.1f} bends",
#         "---": "----------------------------------------",
#         "Avg. Real Wire Length": f"{avg_real_length:.3f} mm",
#         "β Avg. Real Wire Tilt Angle (From X-axis)": f"{avg_real_angle:.2f}°",
#         "Avg. Real Wire Orientation": orientation,
#         "Avg. Length Deviation": f"{abs(avg_real_length - avg_length):.3f} mm",
#         "Avg. 3D Spatial Deviation": f"{avg_3d_dev * 1.28:.3f} mm",
#         "Avg. Angular Deviation": f"{abs(avg_real_angle - avg_angle):.3f}°",
#         "Avg. Real Wire Endpoint": "(-2.142, 51.395, 0.000)",
#         "Avg. Overlay Wire Endpoint": "(8.617, 8.381, 0.042)",
#         "Δ Avg. Axis Deviation": "dx=10.760, dy=-43.014, dz=0.042"
#     }

#     return summary


# # ============================================================
# # FUNCTION: Analyze all uploads
# # ============================================================
# def analyze_all():
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     result_dir = os.path.join(RESULTS_DIR, timestamp)
#     os.makedirs(result_dir, exist_ok=True)

#     results = []
#     for file in os.listdir(UPLOAD_FOLDER):
#         if file.lower().endswith((".png", ".jpg", ".jpeg")):
#             data = analyze_wire(os.path.join(UPLOAD_FOLDER, file))
#             if data:
#                 results.append(data)

#     if not results:
#         summary_text = "❌ No valid wires detected in input images.\n"
#         summary_path = os.path.join(result_dir, "average_summary.txt")
#         with open(summary_path, "w") as f:
#             f.write(summary_text)
#         return

#     summary = compute_averages(results)

#     # Write summary text
#     summary_path = os.path.join(result_dir, "average_summary.txt")
#     with open(summary_path, "w", encoding="utf-8") as f:
#         for k, v in summary.items():
#             f.write(f"{k}: {v}\n")

#     # Write CSV
#     csv_path = os.path.join(result_dir, "wire_analysis_results.csv")
#     with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(["Metric", "Value"])
#         for k, v in summary.items():
#             writer.writerow([k, v])

#     # Log output
#     print("\n✅ Final Summary:")
#     for k, v in summary.items():
#         print(f"{k}: {v}")
#     print(f"\n📂 Results saved in: {result_dir}")

#     return summary_path







# deviation_check.py
# Enhanced image-based wire detector with ArUco calibration, skeleton markings,
# segment tilt detection, and Flask-compatible batch analysis.

import cv2
import numpy as np
import math
import os
import json
from datetime import datetime
from ultralytics import YOLO
from skimage.morphology import skeletonize

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = r"./yolov8n-seg.pt"
UPLOAD_FOLDER = "uploads"
RESULTS_DIR = "results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

FALLBACK_PIXELS_PER_MM = 9.727
MM_TO_CM = 0.1

ARUCO_SIDE_MM = 35.0
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()
ARUCO_DETECTOR = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

# -----------------------------
# Load YOLO
# -----------------------------
print(f"Loading YOLO model from: {MODEL_PATH}")
try:
    model = YOLO(MODEL_PATH)
    print("YOLO loaded.")
except Exception as e:
    print(f"Error loading YOLO: {e}")
    class DummyYOLO:
        def __call__(self, *args, **kwargs):
            return []
    model = DummyYOLO()

# -----------------------------
# Utility Functions
# -----------------------------
def clamp(v, a, b): return max(a, min(v, b))

def normalize_angle(angle):
    while angle > 180: angle -= 360
    while angle <= -180: angle += 360
    if angle > 90: angle -= 180
    if angle < -90: angle += 180
    return angle

# -----------------------------
# Skeleton util
# -----------------------------
def find_skeleton_endpoints(skel):
    endpoints = []
    h, w = skel.shape
    for r in range(h):
        for c in range(w):
            if not skel[r, c]: continue
            r0 = max(0, r - 1)
            r1 = min(h - 1, r + 1)
            c0 = max(0, c - 1)
            c1 = min(w - 1, c + 1)
            neigh = skel[r0:r1+1, c0:c1+1]
            cnt = np.count_nonzero(neigh) - 1
            if cnt == 1:
                endpoints.append((r, c))
    return endpoints

def trace_skeleton_path(skel):
    sk = (skel > 0).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(sk)
    if num_labels <= 1:
        return [], 0.0

    best_label = 1
    best_count = 0
    for lab in range(1, num_labels):
        cnt = int(np.count_nonzero(labels == lab))
        if cnt > best_count:
            best_count = cnt
            best_label = lab
    comp = (labels == best_label).astype(np.uint8)

    coords_arr = np.column_stack(np.where(comp))
    coords = set((int(r), int(c)) for r, c in coords_arr)
    if not coords: return [], 0.0

    endpoints = []
    for (r, c) in coords:
        cnt = 0
        for dr in (-1,0,1):
            for dc in (-1,0,1):
                if dr==0 and dc==0: continue
                if (r+dr, c+dc) in coords: cnt += 1
        if cnt == 1:
            endpoints.append((r, c))

    start = endpoints[0] if endpoints else next(iter(coords))

    visited = {start}
    path = [start]
    cur = start

    while True:
        r, c = cur
        neighbors = []
        for dr in (-1,0,1):
            for dc in (-1,0,1):
                if dr==0 and dc==0: continue
                n = (r+dr, c+dc)
                if n in coords and n not in visited:
                    neighbors.append(n)

        if not neighbors:
            found = False
            for node in reversed(path):
                rr, cc = node
                for dr in (-1,0,1):
                    for dc in (-1,0,1):
                        if dr==0 and dc==0: continue
                        n = (rr+dr, cc+dc)
                        if n in coords and n not in visited:
                            cur = node
                            found = True
                            break
                if found: break
            if not found: break
            else: continue

        nxt = neighbors[0]
        path.append(nxt)
        visited.add(nxt)
        cur = nxt

    return path, 0.0

# -----------------------------
# ArUco detection
# -----------------------------
def detect_aruco_and_pixels_per_mm(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_enh = clahe.apply(gray)

    corners, ids, rej = ARUCO_DETECTOR.detectMarkers(gray_enh)
    vis = frame.copy()

    if ids is None or len(corners) == 0:
        return None, vis

    best_idx = 0
    best_area = 0
    for i, c in enumerate(corners):
        pts = c.reshape(-1, 2).astype(np.float32)
        area = cv2.contourArea(pts)
        if area > best_area:
            best_area = area
            best_idx = i

    best_c = corners[best_idx].reshape(4,2)
    cv2.polylines(vis, [best_c.astype(np.int32)], True, (0,255,0), 2)

    side_lengths = []
    for k in range(4):
        p1 = best_c[k]; p2 = best_c[(k+1)%4]
        side_lengths.append(math.hypot(p2[0]-p1[0], p2[1]-p1[1]))

    avg_side_px = float(np.mean(side_lengths))
    if avg_side_px <= 0: return None, vis

    pixels_per_mm = avg_side_px / ARUCO_SIDE_MM

    return pixels_per_mm, vis

# -----------------------------
# YOLO bbox helper
# -----------------------------
def get_largest_bbox_from_res(res):
    boxes = getattr(res, "boxes", None)
    if boxes is None or boxes.xyxy.numel() == 0:
        return None

    try:
        xyxy = boxes.xyxy.cpu().numpy()
    except Exception:
        return None

    areas = (xyxy[:,2]-xyxy[:,0])*(xyxy[:,3]-xyxy[:,1])
    idx = int(np.argmax(areas))
    return xyxy[idx].astype(int).tolist()

# -----------------------------
# Segment tilt detection
# -----------------------------
def calculate_segment_tilt(path, x_offset, y_offset, mm_per_pixel, segment_len_px=70):

    max_tilt = 0.0
    best_start = best_end = None

    step = max(1, segment_len_px // 2)

    if len(path) < segment_len_px:
        if len(path) >= 2:
            r1,c1 = path[0]; r2,c2 = path[-1]
            dx, dy = c2-c1, r2-r1
            if math.hypot(dx,dy) >= 5:
                ang = normalize_angle(math.degrees(math.atan2(dy, dx)))
                max_tilt = abs(ang)
                best_start = (x_offset + c1, y_offset + r1)
                best_end   = (x_offset + c2, y_offset + r2)
    else:
        for i in range(0, len(path) - segment_len_px, step):
            (r1,c1) = path[i]
            (r2,c2) = path[i+segment_len_px]
            dx, dy = c2-c1, r2-r1
            if math.hypot(dx,dy) < 5: continue
            ang = normalize_angle(math.degrees(math.atan2(dy, dx)))
            if abs(ang) > max_tilt:
                max_tilt = abs(ang)
                best_start = (x_offset + c1, y_offset + r1)
                best_end   = (x_offset + c2, y_offset + r2)

    if best_start and best_end:
        dx = best_end[0] - best_start[0]
        dy = best_end[1] - best_start[1]
        length_px = math.hypot(dx, dy)
        length_cm = length_px * mm_per_pixel * MM_TO_CM
        return max_tilt, length_cm, best_start, best_end

    return 0.0, 0.0, None, None

# -----------------------------
# Frame analysis
# -----------------------------
def analyze_frame(frame, last_pixels_per_mm=FALLBACK_PIXELS_PER_MM):

    pix_per_mm_frame, aruco_img = detect_aruco_and_pixels_per_mm(frame)
    pixels_per_mm = pix_per_mm_frame if pix_per_mm_frame else last_pixels_per_mm
    mm_per_pixel = 1.0 / pixels_per_mm

    results = model(frame, verbose=False)
    if len(results) == 0: return aruco_img, pixels_per_mm, None

    bbox = get_largest_bbox_from_res(results[0])
    if not bbox: return aruco_img, pixels_per_mm, None

    x1,y1,x2,y2 = bbox
    h, w = frame.shape[:2]
    x1 = clamp(x1, 0, w-1); x2 = clamp(x2, 0, w-1)
    y1 = clamp(y1, 0, h-1); y2 = clamp(y2, 0, h-1)

    roi = frame[y1:y2, x1:x2].copy()
    if roi.size == 0: return aruco_img, pixels_per_mm, None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)
    gray_blur = cv2.GaussianBlur(gray_eq, (5,5), 0)
    _, thr = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    mask = (thr > 0)
    if np.count_nonzero(mask) < 10:
        mask = ~mask

    mask = mask.astype(bool)
    try:
        skel = skeletonize(mask).astype(np.uint8)
    except:
        skel = np.zeros_like(mask, dtype=np.uint8)

    if np.count_nonzero(skel) == 0:
        out = aruco_img.copy()
        cv2.rectangle(out, (x1,y1), (x2,y2), (0,128,255), 2)
        return out, pixels_per_mm, None

    path, _ = trace_skeleton_path(skel)

    length_px = float(np.count_nonzero(skel))
    length_cm = length_px * mm_per_pixel * MM_TO_CM

    ys, xs = np.where(mask)
    angle_pca = 0.0
    if len(xs) >= 3:
        coords = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
        try:
            mean, eig = cv2.PCACompute(coords, mean=None)
            p = eig[0]
            angle_pca = normalize_angle(math.degrees(math.atan2(float(p[1]), float(p[0]))))
        except:
            angle_pca = 0.0

    tilt_deg, tilt_len_cm, tilt_start, tilt_end = calculate_segment_tilt(
        path, x1, y1, mm_per_pixel)

    cx0 = roi.shape[1] / 2
    cy0 = roi.shape[0] / 2
    A_xy = (int(cx0), int(cy0))

    endpoints_xy = [(c,r) for (r,c) in find_skeleton_endpoints(skel)]
    if endpoints_xy:
        B_xy = max(endpoints_xy, key=lambda p: math.hypot(p[0]-A_xy[0], p[1]-A_xy[1]))
    elif len(path) >= 2:
        (r1,c1) = path[0]; (r2,c2) = path[-1]
        p1 = (c1,r1); p2 = (c2,r2)
        B_xy = p1 if math.hypot(p1[0]-A_xy[0], p1[1]-A_xy[1]) > math.hypot(p2[0]-A_xy[0], p2[1]-A_xy[1]) else p2
    else:
        B_xy = (A_xy[0]+40, A_xy[1])

    A_img = (x1 + A_xy[0], y1 + A_xy[1])
    B_img = (x1 + B_xy[0], y1 + B_xy[1])

    a_px = math.hypot(B_xy[0]-A_xy[0], B_xy[1]-A_xy[1])
    a_mm = a_px * mm_per_pixel
    ABx = B_xy[0]-A_xy[0]
    ABy = B_xy[1]-A_xy[1]

    AB_angle = math.degrees(math.atan2(ABy, ABx)) if a_px > 0 else 0.0
    theta_deg = abs(normalize_angle(AB_angle - 90.0))
    theta_rad = math.radians(theta_deg)

    special_cm = math.sqrt(max(0, 2*(a_mm**2)*(1 - math.cos(theta_rad)))) * MM_TO_CM

    vx,vy = ABx, ABy
    t = 2
    dirx = "Right" if vx > t else "Left" if vx < -t else ""
    diry = "Down" if vy > t else "Up" if vy < -t else ""
    direction = (diry + "-" + dirx).strip("-") if (dirx or diry) else "None"

    meas = {
        "length_cm": float(length_cm),
        "overall_tilt_pca_deg": float(angle_pca),
        "theta_dev_deg": float(theta_deg),
        "special_dev_cm": float(special_cm),
        "direction": direction,
        "max_segment_tilt_deg": float(tilt_deg),
        "max_tilt_segment_len_cm": float(tilt_len_cm),
        "pixels_per_mm": float(pixels_per_mm)
    }

    out = aruco_img.copy()

    cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,0), 2)

    roi_out = roi.copy()
    roi_out[skel>0] = (255,255,0)
    out[y1:y2, x1:x2] = roi_out

    if tilt_start and tilt_end:
        cv2.line(out, tilt_start, tilt_end, (255,0,255), 4)

    text = f"Len:{length_cm:.3f}cm | Tilt:{angle_pca:.2f}deg | Dev:{special_cm:.4f}cm | Dir:{direction}"
    cv2.putText(out, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255),2)

    return out, pixels_per_mm, meas

# -----------------------------
# Flask-compatible batch function
# -----------------------------
def analyze_all_images_new():
    """
    Used by Flask: reads all images in uploads/, analyzes them,
    and returns JSON + result folder.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(RESULTS_DIR, timestamp)
    os.makedirs(result_dir, exist_ok=True)

    imgs = [f for f in os.listdir(UPLOAD_FOLDER) if f.lower().endswith(IMAGE_EXTS)]
    if not imgs:
        return {"status":"error","message":"No images in uploads/"}

    pixels_per_mm = FALLBACK_PIXELS_PER_MM
    results = []

    for filename in imgs:
        path = os.path.join(UPLOAD_FOLDER, filename)
        frame = cv2.imread(path)
        if frame is None: continue

        out, ppm, meas = analyze_frame(frame, pixels_per_mm)
        pixels_per_mm = ppm

        save_path = os.path.join(result_dir, filename)
        cv2.imwrite(save_path, out)

        if meas:
            meas["image"] = filename
            results.append(meas)

    summary_path = os.path.join(result_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=4)

    return {
        "status": "success",
        "result_folder": result_dir,
        "summary_file": summary_path,
        "measurements": results
    }

# -----------------------------
# Standalone script run
# -----------------------------
def main():
    print("Running deviation_check.py directly…")
    analyze_all_images_new()
    print("Finished. Check /results.")

if __name__ == "__main__":
    main()
