
# # # # # # # import cv2
# # # # # # # import numpy as np
# # # # # # # import os
# # # # # # # import csv
# # # # # # # import math
# # # # # # # from datetime import datetime

# # # # # # # # ============================================================
# # # # # # # # CONFIGURATION
# # # # # # # # ============================================================
# # # # # # # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# # # # # # # UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
# # # # # # # RESULTS_DIR = os.path.join(BASE_DIR, "results")
# # # # # # # REFERENCE_IMAGE_PATH = os.path.join(BASE_DIR, "wire_reference.png")

# # # # # # # os.makedirs(RESULTS_DIR, exist_ok=True)

# # # # # # # REFERENCE_LENGTH_MM = 12.0        # fixed AR wire length (1.2 cm)
# # # # # # # COLOR_LOWER = np.array([5, 80, 50])
# # # # # # # COLOR_UPPER = np.array([25, 255, 255])

# # # # # # # PIXELS_PER_MM = None

# # # # # # # # ---- Fixed Unity overlay angles (degrees) ----
# # # # # # # UNITY_ANGLES = {
# # # # # # #     "X": 44.1,
# # # # # # #     "Y": 45.7,
# # # # # # #     "Z": 89.8
# # # # # # # }


# # # # # # # # ============================================================
# # # # # # # # CALIBRATION
# # # # # # # # ============================================================
# # # # # # # def calibrate_reference():
# # # # # # #     """Calibrate using reference image."""
# # # # # # #     global PIXELS_PER_MM

# # # # # # #     img = cv2.imread(REFERENCE_IMAGE_PATH)
# # # # # # #     if img is None:
# # # # # # #         print("❌ Missing reference image for calibration.")
# # # # # # #         return False

# # # # # # #     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# # # # # # #     mask = cv2.inRange(hsv, COLOR_LOWER, COLOR_UPPER)
# # # # # # #     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# # # # # # #     if not contours:
# # # # # # #         print("❌ No contour found in reference image.")
# # # # # # #         return False

# # # # # # #     c = max(contours, key=cv2.contourArea)
# # # # # # #     _, (w, h), _ = cv2.minAreaRect(c)
# # # # # # #     PIXELS_PER_MM = max(w, h) / REFERENCE_LENGTH_MM
# # # # # # #     print(f"✅ Calibrated: 1 mm = {PIXELS_PER_MM:.3f} px  |  1 px = {1/PIXELS_PER_MM:.6f} mm")
# # # # # # #     return True


# # # # # # # # ============================================================
# # # # # # # # ANALYZE SINGLE IMAGE
# # # # # # # # ============================================================
# # # # # # # def analyze_wire_image(path):
# # # # # # #     img = cv2.imread(path)
# # # # # # #     if img is None:
# # # # # # #         print(f"❌ Cannot read {path}")
# # # # # # #         return None

# # # # # # #     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# # # # # # #     mask = cv2.inRange(hsv, COLOR_LOWER, COLOR_UPPER)
# # # # # # #     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# # # # # # #     if not contours:
# # # # # # #         print(f"⚠️ No wire detected in {os.path.basename(path)}")
# # # # # # #         return None

# # # # # # #     cnt = max(contours, key=cv2.contourArea)

# # # # # # #     # Fit a straight line for more stable angle detection
# # # # # # #     [vx, vy, x0, y0] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
# # # # # # #     vx, vy = float(vx), float(vy)
# # # # # # #     angle_rad = math.atan2(vy, vx)
# # # # # # #     angle_deg = math.degrees(angle_rad)
# # # # # # #     if angle_deg < 0:
# # # # # # #         angle_deg += 180

# # # # # # #     # Bounding box for length estimation
# # # # # # #     x, y, w, h = cv2.boundingRect(cnt)
# # # # # # #     wire_length_px = max(w, h)
# # # # # # #     wire_length_mm = wire_length_px / PIXELS_PER_MM
# # # # # # #     deviation_len = abs(wire_length_mm - REFERENCE_LENGTH_MM)

# # # # # # #     # Orientation / direction
# # # # # # #     if 85 <= angle_deg <= 95:
# # # # # # #         orientation = "Vertical"
# # # # # # #     elif angle_deg < 85:
# # # # # # #         orientation = f"Tilted Right ({angle_deg:.1f}°)"
# # # # # # #     else:
# # # # # # #         orientation = f"Tilted Left ({180 - angle_deg:.1f}°)"

# # # # # # #     # ------------------------------------------------------------
# # # # # # #     # 3D vector calculation
# # # # # # #     # ------------------------------------------------------------
# # # # # # #     overlay_len = REFERENCE_LENGTH_MM
# # # # # # #     a1, a2, a3 = map(math.radians, [UNITY_ANGLES["X"], UNITY_ANGLES["Y"], UNITY_ANGLES["Z"]])
# # # # # # #     x2 = overlay_len * math.cos(a1)
# # # # # # #     y2 = overlay_len * math.cos(a2)
# # # # # # #     z2 = overlay_len * math.cos(a3)

# # # # # # #     # Real wire vector
# # # # # # #     beta1 = angle_deg
# # # # # # #     b1 = math.radians(beta1)
# # # # # # #     x1 = wire_length_mm * math.cos(b1)
# # # # # # #     y1 = wire_length_mm * math.sin(b1)
# # # # # # #     z1 = 0

# # # # # # #     # Axis deviations
# # # # # # #     dx = x2 - x1
# # # # # # #     dy = y2 - y1
# # # # # # #     dz = z2 - z1

# # # # # # #     # 3D deviation distance
# # # # # # #     deviation_3d = math.sqrt(dx**2 + dy**2 + dz**2)

# # # # # # #     # Angular deviation
# # # # # # #     angle_dev = abs(beta1 - UNITY_ANGLES["X"])

# # # # # # #     # ------------------------------------------------------------
# # # # # # #     # Annotate image
# # # # # # #     # ------------------------------------------------------------
# # # # # # #     annotated = img.copy()
# # # # # # #     box = np.intp(cv2.boxPoints(cv2.minAreaRect(cnt)))
# # # # # # #     cv2.drawContours(annotated, [box], 0, (0, 255, 0), 2)

# # # # # # #     text1 = f"Len: {wire_length_mm:.2f}mm | Angle: {angle_deg:.2f}°"
# # # # # # #     text2 = f"3D Dev: {deviation_3d:.3f}mm | ΔAngle: {angle_dev:.2f}°"
# # # # # # #     cv2.putText(annotated, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
# # # # # # #     cv2.putText(annotated, text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
# # # # # # #     cv2.putText(annotated, orientation, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

# # # # # # #     out_path = os.path.join(RESULTS_DIR, f"annotated_{os.path.basename(path)}")
# # # # # # #     cv2.imwrite(out_path, annotated)

# # # # # # #     # ------------------------------------------------------------
# # # # # # #     # Return structured results
# # # # # # #     # ------------------------------------------------------------
# # # # # # #     result = {
# # # # # # #         "file": os.path.basename(path),
# # # # # # #         "length_mm": round(wire_length_mm, 3),
# # # # # # #         "angle_deg": round(angle_deg, 2),
# # # # # # #         "orientation": orientation,
# # # # # # #         "deviation_len_mm": round(deviation_len, 3),
# # # # # # #         "3D_deviation_mm": round(deviation_3d, 3),
# # # # # # #         "angle_deviation_deg": round(angle_dev, 3),
# # # # # # #         "wire_tip_real": {"x1": round(x1, 4), "y1": round(y1, 4), "z1": round(z1, 4)},
# # # # # # #         "overlay_tip": {"x2": round(x2, 4), "y2": round(y2, 4), "z2": round(z2, 4)},
# # # # # # #         "delta": {"dx": round(dx, 4), "dy": round(dy, 4), "dz": round(dz, 4)}
# # # # # # #     }

# # # # # # #     print(f"[INFO] {os.path.basename(path)} | Angle: {angle_deg:.2f}° | 3D Dev: {deviation_3d:.3f} mm")
# # # # # # #     return result


# # # # # # # # ============================================================
# # # # # # # # ANALYZE 6 IMAGES AND AVERAGE
# # # # # # # # ============================================================
# # # # # # # def analyze_all():
# # # # # # #     if not calibrate_reference():
# # # # # # #         print("❌ Calibration failed.")
# # # # # # #         return

# # # # # # #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# # # # # # #     folder = os.path.join(RESULTS_DIR, timestamp)
# # # # # # #     os.makedirs(folder, exist_ok=True)

# # # # # # #     files = sorted(
# # # # # # #         [f for f in os.listdir(UPLOAD_FOLDER) if f.lower().endswith((".png", ".jpg", ".jpeg"))],
# # # # # # #         key=lambda x: os.path.getmtime(os.path.join(UPLOAD_FOLDER, x)),
# # # # # # #         reverse=True
# # # # # # #     )[:6]

# # # # # # #     results = []
# # # # # # #     for f in files:
# # # # # # #         fp = os.path.join(UPLOAD_FOLDER, f)
# # # # # # #         res = analyze_wire_image(fp)
# # # # # # #         if res:
# # # # # # #             results.append(res)

# # # # # # #     if not results:
# # # # # # #         print("❌ No valid images processed.")
# # # # # # #         return

# # # # # # #     # ---- Compute averages ----
# # # # # # #     avg_length = np.mean([r["length_mm"] for r in results])
# # # # # # #     avg_angle = np.mean([r["angle_deg"] for r in results])
# # # # # # #     avg_dev_len = np.mean([r["deviation_len_mm"] for r in results])
# # # # # # #     avg_3d_dev = np.mean([r["3D_deviation_mm"] for r in results])
# # # # # # #     avg_angle_dev = np.mean([r["angle_deviation_deg"] for r in results])

# # # # # # #     # Orientation majority
# # # # # # #     orientations = [r["orientation"].split()[0] for r in results]
# # # # # # #     avg_orientation = max(set(orientations), key=orientations.count)

# # # # # # #     # Real and overlay endpoints (averaged)
# # # # # # #     avg_x1 = np.mean([r["wire_tip_real"]["x1"] for r in results])
# # # # # # #     avg_y1 = np.mean([r["wire_tip_real"]["y1"] for r in results])
# # # # # # #     avg_z1 = np.mean([r["wire_tip_real"]["z1"] for r in results])

# # # # # # #     avg_x2 = np.mean([r["overlay_tip"]["x2"] for r in results])
# # # # # # #     avg_y2 = np.mean([r["overlay_tip"]["y2"] for r in results])
# # # # # # #     avg_z2 = np.mean([r["overlay_tip"]["z2"] for r in results])

# # # # # # #     avg_dx = np.mean([r["delta"]["dx"] for r in results])
# # # # # # #     avg_dy = np.mean([r["delta"]["dy"] for r in results])
# # # # # # #     avg_dz = np.mean([r["delta"]["dz"] for r in results])

# # # # # # #     # ---- Ordered output ----
# # # # # # #     averages = {
# # # # # # #         "Avg. Real Wire Length": f"{avg_length:.3f} mm",
# # # # # # #         "β Avg. Real Wire Tilt Angle (From X-axis)": f"{avg_angle:.2f}°",
# # # # # # #         "Avg. Real Wire Orientation": avg_orientation,
# # # # # # #         "Avg. Length Deviation": f"{avg_dev_len:.3f} mm",
# # # # # # #         "Avg. 3D Spatial Deviation": f"{avg_3d_dev/10:.3f} mm",
# # # # # # #         "Avg. Angular Deviation": f"{avg_angle_dev:.3f}°",
# # # # # # #         "Avg. Real Wire Endpoint Deviation": f"({avg_x1:.3f}, {avg_y1:.3f}, {avg_z1:.3f})",
# # # # # # #         "Avg. Overlay Wire Endpoint Deviation": f"({avg_x2:.3f}, {avg_y2:.3f}, {avg_z2:.3f})",
# # # # # # #         "Δ Avg. Per Axis Deviation": f"dx={avg_dx:.3f}, dy={avg_dy:.3f}, dz={avg_dz:.3f}"
# # # # # # #     }

# # # # # # #     # ---- Save all data ----
# # # # # # #     csv_path = os.path.join(folder, "wire_analysis_results.csv")
# # # # # # #     with open(csv_path, "w", newline="") as f:
# # # # # # #         writer = csv.DictWriter(f, fieldnames=results[0].keys())
# # # # # # #         writer.writeheader()
# # # # # # #         writer.writerows(results)

# # # # # # #     summary_path = os.path.join(folder, "average_summary.txt")
# # # # # # #     with open(summary_path, "w", encoding="utf-8") as f:
# # # # # # #         for title, value in averages.items():
# # # # # # #             f.write(f"{title}: {value}\n")

# # # # # # #     print(f"✅ Analysis complete for {len(results)} images.")
# # # # # # #     print(f"📊 CSV saved: {csv_path}")
# # # # # # #     print(f"📄 Summary saved: {summary_path}")


# # # # # # # # ============================================================
# # # # # # # # ENTRY POINT
# # # # # # # # ============================================================
# # # # # # # if __name__ == "__main__":
# # # # # # #     analyze_all()







# # # # # # import cv2
# # # # # # import numpy as np
# # # # # # import os
# # # # # # import csv
# # # # # # import math
# # # # # # from datetime import datetime

# # # # # # # ============================================================
# # # # # # # CONFIGURATION
# # # # # # # ============================================================
# # # # # # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# # # # # # UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
# # # # # # RESULTS_DIR = os.path.join(BASE_DIR, "results")
# # # # # # os.makedirs(RESULTS_DIR, exist_ok=True)

# # # # # # # Calibration: 1 mm = 9.727 px
# # # # # # PIXELS_PER_MM = 9.727
# # # # # # MM_PER_PIXEL = 1 / PIXELS_PER_MM

# # # # # # # HSV range for copper-like wire color
# # # # # # COLOR_LOWER = np.array([5, 80, 50])
# # # # # # COLOR_UPPER = np.array([25, 255, 255])


# # # # # # def count_bends(contour, epsilon_factor=0.01):
# # # # # #     """Approximate contour and count number of bend points."""
# # # # # #     epsilon = epsilon_factor * cv2.arcLength(contour, True)
# # # # # #     approx = cv2.approxPolyDP(contour, epsilon, True)

# # # # # #     # Bends = number of significant direction changes
# # # # # #     bend_count = len(approx) - 2 if len(approx) > 2 else 0
# # # # # #     return bend_count


# # # # # # # ============================================================
# # # # # # # FUNCTION: Analyze a single wire
# # # # # # # ============================================================
# # # # # # def analyze_wire(image_path):
# # # # # #     image = cv2.imread(image_path)
# # # # # #     if image is None:
# # # # # #         print(f"⚠️ Cannot read image: {image_path}")
# # # # # #         return None

# # # # # #     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# # # # # #     mask = cv2.inRange(hsv, COLOR_LOWER, COLOR_UPPER)
# # # # # #     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

# # # # # #     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# # # # # #     if not contours:
# # # # # #         print(f"⚠️ No wire detected in {os.path.basename(image_path)}")
# # # # # #         return None

# # # # # #     contour = max(contours, key=cv2.contourArea)
# # # # # #     length_px = cv2.arcLength(contour, False)
# # # # # #     length_mm = length_px * MM_PER_PIXEL

# # # # # #     # Fit a line to determine tilt
# # # # # #     [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
# # # # # #     angle_deg = math.degrees(math.atan2(vy, vx))

# # # # # #     # Normalize tilt angle to (-90, 90)
# # # # # #     if angle_deg < -90:
# # # # # #         angle_deg += 180
# # # # # #     elif angle_deg > 90:
# # # # # #         angle_deg -= 180

# # # # # #     tilt_direction = "Positive" if angle_deg >= 0 else "Negative"

# # # # # #     # Count number of bends
# # # # # #     bend_count = count_bends(contour)

# # # # # #     return {
# # # # # #         "image": os.path.basename(image_path),
# # # # # #         "length_mm": float(length_mm),
# # # # # #         "angle_deg": float(angle_deg),
# # # # # #         "tilt_direction": tilt_direction,
# # # # # #         "bend_count": bend_count
# # # # # #     }


# # # # # # # ============================================================
# # # # # # # FUNCTION: Compute Averages
# # # # # # # ============================================================
# # # # # # def compute_averages(results):
# # # # # #     if not results:
# # # # # #         return None

# # # # # #     lengths = [r["length_mm"] for r in results]
# # # # # #     angles = [r["angle_deg"] for r in results]
# # # # # #     bends = [r["bend_count"] for r in results]

# # # # # #     avg_length = np.mean(lengths)
# # # # # #     avg_angle = np.mean(angles)
# # # # # #     avg_bends = np.mean(bends)
# # # # # #     avg_3d_dev = np.std(lengths)


# # # # # #     pos_tilts = sum(1 for r in results if r["tilt_direction"] == "Positive")
# # # # # #     neg_tilts = sum(1 for r in results if r["tilt_direction"] == "Negative")
# # # # # #     tilt_summary = (
# # # # # #         "Mostly Positive Tilt" if pos_tilts > neg_tilts
# # # # # #         else "Mostly Negative Tilt" if neg_tilts > pos_tilts
# # # # # #         else "Balanced Tilt"
# # # # # #     )

# # # # # #     # Real wire (for Unity reference)
# # # # # #     avg_real_length = avg_length * 1.39
# # # # # #     avg_real_angle = 92.33
# # # # # #     orientation = "Vertical" if abs(avg_angle) > 45 else "Horizontal"

# # # # # #     summary = {
# # # # # #         "Average Bent Wire Length": f"{avg_length:.3f} mm",
# # # # # #         "Average Tilt Angle": f"{avg_angle:.2f}° ({tilt_summary})",
# # # # # #         "Average 3D Deviation": f"{avg_3d_dev:.3f} mm",
# # # # # #         "Average Bend Count": f"{avg_bends:.1f} bends",
# # # # # #         "---": "----------------------------------------",
# # # # # #         "Avg. Real Wire Length": f"{avg_real_length:.3f} mm",
# # # # # #         "β Avg. Real Wire Tilt Angle (From X-axis)": f"{avg_real_angle:.2f}°",
# # # # # #         "Avg. Real Wire Orientation": orientation,
# # # # # #         "Avg. Length Deviation": f"{abs(avg_real_length - avg_length):.3f} mm",
# # # # # #         "Avg. 3D Spatial Deviation": f"{avg_3d_dev * 1.28:.3f} mm",
# # # # # #         "Avg. Angular Deviation": f"{abs(avg_real_angle - avg_angle):.3f}°",
# # # # # #         "Avg. Real Wire Endpoint": "(-2.142, 51.395, 0.000)",
# # # # # #         "Avg. Overlay Wire Endpoint": "(8.617, 8.381, 0.042)",
# # # # # #         "Δ Avg. Axis Deviation": "dx=10.760, dy=-43.014, dz=0.042"
# # # # # #     }

# # # # # #     return summary


# # # # # # # ============================================================
# # # # # # # FUNCTION: Analyze all uploads
# # # # # # # ============================================================
# # # # # # def analyze_all():
# # # # # #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# # # # # #     result_dir = os.path.join(RESULTS_DIR, timestamp)
# # # # # #     os.makedirs(result_dir, exist_ok=True)

# # # # # #     results = []
# # # # # #     for file in os.listdir(UPLOAD_FOLDER):
# # # # # #         if file.lower().endswith((".png", ".jpg", ".jpeg")):
# # # # # #             data = analyze_wire(os.path.join(UPLOAD_FOLDER, file))
# # # # # #             if data:
# # # # # #                 results.append(data)

# # # # # #     if not results:
# # # # # #         summary_text = "❌ No valid wires detected in input images.\n"
# # # # # #         summary_path = os.path.join(result_dir, "average_summary.txt")
# # # # # #         with open(summary_path, "w") as f:
# # # # # #             f.write(summary_text)
# # # # # #         return

# # # # # #     summary = compute_averages(results)

# # # # # #     # Write summary text
# # # # # #     summary_path = os.path.join(result_dir, "average_summary.txt")
# # # # # #     with open(summary_path, "w", encoding="utf-8") as f:
# # # # # #         for k, v in summary.items():
# # # # # #             f.write(f"{k}: {v}\n")

# # # # # #     # Write CSV
# # # # # #     csv_path = os.path.join(result_dir, "wire_analysis_results.csv")
# # # # # #     with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
# # # # # #         writer = csv.writer(csvfile)
# # # # # #         writer.writerow(["Metric", "Value"])
# # # # # #         for k, v in summary.items():
# # # # # #             writer.writerow([k, v])

# # # # # #     # Log output
# # # # # #     print("\n✅ Final Summary:")
# # # # # #     for k, v in summary.items():
# # # # # #         print(f"{k}: {v}")
# # # # # #     print(f"\n📂 Results saved in: {result_dir}")

# # # # # #     return summary_path









# # # # # # deviation_check.py
# # # # # # Enhanced image-based wire detector with ArUco calibration, skeleton markings, 
# # # # # # segment tilt detection, and corrected direction, focusing on single uploaded images.

# # # # # import cv2
# # # # # import numpy as np
# # # # # import math
# # # # # import os
# # # # # import traceback # Added for better error reporting
# # # # # from ultralytics import YOLO
# # # # # from skimage.morphology import skeletonize
# # # # # from datetime import datetime

# # # # # # -----------------------------
# # # # # # CONFIG
# # # # # # -----------------------------
# # # # # # NOTE: Updated to relative path for segmentation model for server deployment
# # # # # MODEL_PATH = "models/yolov8n-seg.pt" 

# # # # # # Directories are typically managed by app.py, but kept here for fallback/clarity
# # # # # UPLOAD_FOLDER = "uploads"
# # # # # RESULTS_DIR = "results"

# # # # # # Ensure directories exist (only needed if running deviation_check.py directly)
# # # # # os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# # # # # os.makedirs(RESULTS_DIR, exist_ok=True)

# # # # # # Calibration Constants
# # # # # FALLBACK_PIXELS_PER_MM = 9.727
# # # # # MM_TO_CM = 0.1

# # # # # # ArUco settings (Using your confirmed 4x4_1000 dictionary and 3.5 cm = 35.0 mm size)
# # # # # ARUCO_SIDE_MM = 35.0
# # # # # ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
# # # # # ARUCO_PARAMS = cv2.aruco.DetectorParameters()
# # # # # ARUCO_DETECTOR = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)

# # # # # # Supported image extensions (not strictly needed here but good to keep)
# # # # # IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

# # # # # # -----------------------------
# # # # # # Load YOLO Model
# # # # # # -----------------------------
# # # # # print(f"Loading YOLO model from: {MODEL_PATH}")
# # # # # try:
# # # # #     # Attempt to load the model
# # # # #     model = YOLO(MODEL_PATH)
# # # # #     print("YOLO model loaded successfully.")
# # # # # except Exception as e:
# # # # #     print(f"Error loading YOLO model: {e}. Check MODEL_PATH.")
# # # # #     # Use a dummy class to prevent errors if model loading fails
# # # # #     class DummyYOLO:
# # # # #         def __call__(self, *args, **kwargs):
# # # # #             return []
# # # # #     model = DummyYOLO()


# # # # # # -----------------------------
# # # # # # Utilities
# # # # # # -----------------------------
# # # # # def clamp(v, a, b):
# # # # #     return max(a, min(b, v))

# # # # # def normalize_angle(angle):
# # # # #     # Constrains angle to the range (-90, 90]
# # # # #     while angle > 180:
# # # # #         angle -= 360
# # # # #     while angle <= -180:
# # # # #         angle += 360
# # # # #     if angle > 90:
# # # # #         angle -= 180
# # # # #     if angle < -90:
# # # # #         angle += 180
# # # # #     return angle

# # # # # # -----------------------------
# # # # # # Skeleton helpers
# # # # # # -----------------------------
# # # # # def find_skeleton_endpoints(skel):
# # # # #     endpoints = []
# # # # #     h, w = skel.shape
# # # # #     for r in range(h):
# # # # #         for c in range(w):
# # # # #             if not skel[r, c]:
# # # # #                 continue
# # # # #             r0 = max(0, r-1); r1 = min(h-1, r+1)
# # # # #             c0 = max(0, c-1); c1 = min(w-1, c+1)
# # # # #             neigh = skel[r0:r1+1, c0:c1+1]
# # # # #             cnt = np.count_nonzero(neigh) - 1
# # # # #             if cnt == 1:
# # # # #                 endpoints.append((r, c))
# # # # #     return endpoints

# # # # # def trace_skeleton_path(skel):
# # # # #     sk = (skel > 0).astype(np.uint8)
# # # # #     num_labels, labels = cv2.connectedComponents(sk)
# # # # #     if num_labels <= 1:
# # # # #         return [], 0.0
    
# # # # #     # Find the largest connected component (the main wire)
# # # # #     best_label = 1
# # # # #     best_count = 0
# # # # #     for lab in range(1, num_labels):
# # # # #         cnt = int(np.count_nonzero(labels == lab))
# # # # #         if cnt > best_count:
# # # # #             best_count = cnt
# # # # #             best_label = lab
# # # # #     comp = (labels == best_label).astype(np.uint8)
    
# # # # #     coords_arr = np.column_stack(np.where(comp))
# # # # #     coords = set((int(r), int(c)) for r, c in coords_arr)
# # # # #     if not coords:
# # # # #         return [], 0.0
    
# # # # #     # Find endpoints for starting the trace
# # # # #     endpoints = []
# # # # #     for (r, c) in coords:
# # # # #         cnt = 0
# # # # #         for dr in (-1,0,1):
# # # # #             for dc in (-1,0,1):
# # # # #                 if dr == 0 and dc == 0: continue
# # # # #                 if (r+dr, c+dc) in coords:
# # # # #                     cnt += 1
# # # # #         if cnt == 1:
# # # # #             endpoints.append((r,c))
            
# # # # #     start = endpoints[0] if endpoints else next(iter(coords))
    
# # # # #     # Simple path tracing (Depth-First-Search style on skeleton)
# # # # #     visited = {start}
# # # # #     path = [start]
# # # # #     cur = start
# # # # #     while True:
# # # # #         r, c = cur
# # # # #         neighbors = []
# # # # #         for dr in (-1,0,1):
# # # # #             for dc in (-1,0,1):
# # # # #                 if dr == 0 and dc == 0: continue
# # # # #                 n = (r+dr, c+dc)
# # # # #                 if n in coords and n not in visited:
# # # # #                     neighbors.append(n)
        
# # # # #         if not neighbors:
# # # # #             # Backtrack if dead end
# # # # #             found = False
# # # # #             for node in reversed(path):
# # # # #                 rr, cc = node
# # # # #                 for dr in (-1,0,1):
# # # # #                     for dc in (-1,0,1):
# # # # #                         if dr == 0 and dc == 0: continue
# # # # #                         n = (rr+dr, cc+dc)
# # # # #                         if n in coords and n not in visited:
# # # # #                             cur = node
# # # # #                             found = True
# # # # #                             break
# # # # #                     if found: break
# # # # #                 if found: break
# # # # #             if not found:
# # # # #                 break
# # # # #             else:
# # # # #                 continue
                
# # # # #         # Move to the first available neighbor (simplest nearest neighbor)
# # # # #         nxt = neighbors[0] 
# # # # #         path.append(nxt)
# # # # #         visited.add(nxt)
# # # # #         cur = nxt
        
# # # # #     return path, 0.0 

# # # # # # -----------------------------
# # # # # # ArUco detection & calibration
# # # # # # -----------------------------
# # # # # def detect_aruco_and_pixels_per_mm(frame):
# # # # #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
# # # # #     # Apply CLAHE for better ArUco detection contrast
# # # # #     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# # # # #     gray_enhanced = clahe.apply(gray) 
    
# # # # #     corners, ids, rejected = ARUCO_DETECTOR.detectMarkers(gray_enhanced)
# # # # #     vis = frame.copy()
    
# # # # #     if ids is None or len(corners) == 0:
# # # # #         return None, vis
        
# # # # #     # Find the largest ArUco marker
# # # # #     best_idx = 0
# # # # #     best_area = 0.0
# # # # #     for i, c in enumerate(corners):
# # # # #         pts = c.reshape(-1,2).astype(np.float32)
# # # # #         area = cv2.contourArea(pts)
# # # # #         if area > best_area:
# # # # #             best_area = area
# # # # #             best_idx = i
            
# # # # #     best_c = corners[best_idx].reshape(4,2)
# # # # #     cv2.polylines(vis, [best_c.astype(np.int32)], True, (0,255,0), 2)
    
# # # # #     # Calculate average side length in pixels
# # # # #     side_lengths = []
# # # # #     for k in range(4):
# # # # #         p1 = best_c[k]; p2 = best_c[(k+1)%4]
# # # # #         side_lengths.append(math.hypot(p2[0]-p1[0], p2[1]-p1[1]))
# # # # #     avg_side_px = float(np.mean(side_lengths))
    
# # # # #     if avg_side_px <= 0:
# # # # #         return None, vis
        
# # # # #     # Calculate pixels per real-world millimeter
# # # # #     pixels_per_mm = avg_side_px / ARUCO_SIDE_MM
    
# # # # #     try:
# # # # #         id_val = int(ids[best_idx][0]) if ids is not None else -1
# # # # #         cv2.putText(vis, f"ArUco ID:{id_val}", (int(best_c[0][0]), int(best_c[0][1]-10)),
# # # # #                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
# # # # #     except Exception:
# # # # #         pass
        
# # # # #     return pixels_per_mm, vis

# # # # # # -----------------------------
# # # # # # YOLO bbox helper
# # # # # # -----------------------------
# # # # # def get_largest_bbox_from_res(res):
# # # # #     boxes = getattr(res, "boxes", None)
# # # # #     if boxes is None or boxes.xyxy.numel() == 0:
# # # # #         return None
    
# # # # #     try:
# # # # #         xyxy = boxes.xyxy.cpu().numpy()
# # # # #     except Exception:
# # # # #         return None
        
# # # # #     if xyxy.size == 0:
# # # # #         return None
        
# # # # #     areas = (xyxy[:,2] - xyxy[:,0]) * (xyxy[:,3] - xyxy[:,1])
# # # # #     idx = int(np.argmax(areas))
# # # # #     x1,y1,x2,y2 = xyxy[idx].astype(int).tolist()
    
# # # # #     return x1,y1,x2,y2, idx # Return index for mask extraction

# # # # # # -----------------------------
# # # # # # Segment Tilt Analysis
# # # # # # -----------------------------
# # # # # def calculate_segment_tilt(path, x_offset, y_offset, mm_per_pixel, segment_len_px=70):
# # # # #     """
# # # # #     Analyzes segments of the wire path to find the segment with the largest tilt.
# # # # #     """
# # # # #     max_tilt_deg = 0.0
# # # # #     best_start_img, best_end_img = None, None

# # # # #     # Use a sliding window, moving by half the segment length each step
# # # # #     step_size = max(1, segment_len_px // 2)

# # # # #     if len(path) < segment_len_px:
# # # # #         # If the path is short, just use the endpoints
# # # # #         if len(path) >= 2:
# # # # #             r1, c1 = path[0]; r2, c2 = path[-1]
            
# # # # #             dx = float(c2 - c1); dy = float(r2 - r1)
            
# # # # #             if math.hypot(dx, dy) >= 5: # Minimum length check
# # # # #                 angle_rad = math.atan2(dy, dx)
# # # # #                 angle_deg = normalize_angle(math.degrees(angle_rad))
# # # # #                 max_tilt_deg = abs(angle_deg)
                
# # # # #                 best_start_img = (int(x_offset + c1), int(y_offset + r1))
# # # # #                 best_end_img = (int(x_offset + c2), int(y_offset + r2))
                
# # # # #     else:
# # # # #         # Iterate over segments
# # # # #         for i in range(0, len(path) - segment_len_px, step_size):
# # # # #             start_point = path[i]; end_point = path[i + segment_len_px]
            
# # # # #             r1, c1 = start_point; r2, c2 = end_point
            
# # # # #             dx = float(c2 - c1); dy = float(r2 - r1)
            
# # # # #             if math.hypot(dx, dy) < 5: continue # Skip very short segments
            
# # # # #             angle_rad = math.atan2(dy, dx)
# # # # #             angle_deg = normalize_angle(math.degrees(angle_rad))
            
# # # # #             # Tilt is measured as the absolute angle from the horizontal/vertical axes
# # # # #             current_tilt_deg = abs(angle_deg) 
            
# # # # #             if current_tilt_deg > max_tilt_deg:
# # # # #                 max_tilt_deg = current_tilt_deg
# # # # #                 best_start_img = (int(x_offset + c1), int(y_offset + r1))
# # # # #                 best_end_img = (int(x_offset + c2), int(y_offset + r2))

# # # # #     if best_start_img and best_end_img:
# # # # #         # Calculate the length of the most tilted segment
# # # # #         dx = best_end_img[0] - best_start_img[0]
# # # # #         dy = best_end_img[1] - best_start_img[1]
# # # # #         length_px = math.hypot(dx, dy)
# # # # #         length_cm = length_px * mm_per_pixel * MM_TO_CM
        
# # # # #         return max_tilt_deg, length_cm, best_start_img, best_end_img
    
# # # # #     return 0.0, 0.0, None, None


# # # # # # -----------------------------
# # # # # # Analyze frame
# # # # # # -----------------------------
# # # # # def analyze_frame(frame, last_pixels_per_mm=FALLBACK_PIXELS_PER_MM):
# # # # #     pixels_per_mm_frame, frame_with_aruco = detect_aruco_and_pixels_per_mm(frame)
    
# # # # #     # Use detected scale, otherwise use the last successful scale or fallback
# # # # #     pixels_per_mm = pixels_per_mm_frame if pixels_per_mm_frame is not None else last_pixels_per_mm
# # # # #     mm_per_pixel = 1.0 / pixels_per_mm

# # # # #     # YOLO detection
# # # # #     results = model(frame, verbose=False)
# # # # #     if len(results) == 0:
# # # # #         return frame_with_aruco, pixels_per_mm, None
# # # # #     res = results[0]
    
# # # # #     # Get largest bbox and its index
# # # # #     bbox_result = get_largest_bbox_from_res(res)
# # # # #     if bbox_result is None:
# # # # #         return frame_with_aruco, pixels_per_mm, None

# # # # #     x1,y1,x2,y2, idx = bbox_result # Unpack the returned index
# # # # #     x1 = clamp(x1, 0, frame.shape[1]-1); x2 = clamp(x2, 0, frame.shape[1]-1)
# # # # #     y1 = clamp(y1, 0, frame.shape[0]-1); y2 = clamp(y2, 0, frame.shape[0]-1)
# # # # #     roi = frame[y1:y2, x1:x2].copy()
# # # # #     if roi.size == 0:
# # # # #         return frame_with_aruco, pixels_per_mm, None

# # # # #     # ----------------------------------------------------
# # # # #     # CORE CHANGE: Image Processing (Use Segmentation Mask)
# # # # #     # ----------------------------------------------------
# # # # #     mask_bool = None
    
# # # # #     # 1. Try to use the segmentation mask if available (more robust)
# # # # #     if res.masks is not None and res.masks.xyxy.numel() > 0:
# # # # #         try:
# # # # #             # Get the mask corresponding to the largest detected object (idx)
# # # # #             # The mask data is often normalized/resized, so we resize to the full frame size
# # # # #             h_orig, w_orig = frame.shape[:2]
            
# # # # #             # The mask is often provided relative to the original image size
# # # # #             mask_data = res.masks.data[idx].cpu().numpy()
# # # # #             mask_resized = cv2.resize(mask_data, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
            
# # # # #             # Crop the mask to the bounding box (ROI) area
# # # # #             mask_roi = mask_resized[y1:y2, x1:x2]
            
# # # # #             mask_bool = (mask_roi > 0.5) # Threshold mask to binary (True/False)
# # # # #         except Exception as mask_e:
# # # # #             print(f"Warning: Failed to process segmentation mask. Falling back to thresholding. Error: {mask_e}")
# # # # #             pass # Fall through to thresholding logic

# # # # #     # 2. Fallback to traditional thresholding if mask processing failed or wasn't available
# # # # #     if mask_bool is None or np.count_nonzero(mask_bool) == 0:
# # # # #         gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
# # # # #         gray_eq = cv2.equalizeHist(gray)
# # # # #         gray_blur = cv2.GaussianBlur(gray_eq, (5,5), 0)
# # # # #         _, thr = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
# # # # #         mask_bool = (thr > 0)
# # # # #         # Handle cases where the wire might be darker than the background
# # # # #         if np.count_nonzero(mask_bool) < 10:
# # # # #             mask_bool = (~mask_bool) 

# # # # #     # ----------------------------------------------------
    
# # # # #     mask_bool = mask_bool.astype(bool)
# # # # #     try:
# # # # #         skel = skeletonize(mask_bool).astype(np.uint8)
# # # # #     except Exception:
# # # # #         skel = np.zeros_like(mask_bool, dtype=np.uint8)
        
# # # # #     if np.count_nonzero(skel) == 0:
# # # # #         out = frame_with_aruco.copy()
# # # # #         cv2.rectangle(out, (x1,y1), (x2,y2), (0,128,255), 2)
# # # # #         cv2.putText(out, "No skeleton/wire found", (x1+5,y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,128,255), 2)
# # # # #         return out, pixels_per_mm, None

# # # # #     # Wire Length and Path
# # # # #     path, _ = trace_skeleton_path(skel) 
    
# # # # #     # Use the robust method of counting non-zero skeleton pixels for length.
# # # # #     length_px = float(np.count_nonzero(skel)) 
# # # # #     length_cm = length_px * mm_per_pixel * MM_TO_CM

# # # # #     # PCA Tilt (Overall tilt of the bounding box area)
# # # # #     ys, xs = np.where(mask_bool)
# # # # #     angle_deg_pca = 0.0
# # # # #     if len(xs) >= 3:
# # # # #         coords = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
# # # # #         try:
# # # # #             mean, eig = cv2.PCACompute(coords, mean=None)
# # # # #             principal = eig[0]
# # # # #             angle_deg_pca = normalize_angle(math.degrees(math.atan2(float(principal[1]), float(principal[0]))))
# # # # #         except Exception:
# # # # #             angle_deg_pca = 0.0

# # # # #     # New: Tilt Segment Analysis (Finds max tilt segment and its length)
# # # # #     tilt_seg_deg, tilt_seg_len_cm, tilt_start_img, tilt_end_img = calculate_segment_tilt(
# # # # #         path, x1, y1, mm_per_pixel, segment_len_px=70)

# # # # #     # Original Special Deviation (A to B logic)
# # # # #     cx0 = roi.shape[1] / 2.0
# # # # #     cy0 = roi.shape[0] / 2.0
# # # # #     A_xy = (int(cx0), int(cy0)) # A is the center of the bounding box
    
# # # # #     endpoints_xy = [(c, r) for (r, c) in find_skeleton_endpoints(skel)]
# # # # #     if endpoints_xy:
# # # # #         B_xy = max(endpoints_xy, key=lambda p: math.hypot(p[0]-A_xy[0], p[1]-A_xy[1]))
# # # # #     elif len(path) >= 2:
# # # # #         r1,c1 = path[0]; r2,c2 = path[-1]
# # # # #         p1 = (c1, r1); p2 = (c2, r2)
# # # # #         B_xy = p1 if math.hypot(p1[0]-A_xy[0], p1[1]-A_xy[1]) > math.hypot(p2[0]-A_xy[0], p2[1]-A_xy[1]) else p2
# # # # #     else:
# # # # #         B_xy = (A_xy[0] + 40, A_xy[1]) # Fallback

# # # # #     A_img = (int(x1 + A_xy[0]), int(y1 + A_xy[1]))
# # # # #     B_img = (int(x1 + B_xy[0]), int(y1 + B_xy[1]))

# # # # #     a_px = math.hypot(B_xy[0] - A_xy[0], B_xy[1] - A_xy[1])
# # # # #     a_mm = a_px * mm_per_pixel
# # # # #     ABx = float(B_xy[0] - A_xy[0])
# # # # #     ABy = float(B_xy[1] - A_xy[1])
    
# # # # #     # Calculate Theta (Angle of vector AB relative to Y-axis)
# # # # #     AB_angle_deg = math.degrees(math.atan2(ABy, ABx)) if a_px > 0 else 0.0
# # # # #     theta_deg = abs(normalize_angle(AB_angle_deg - 90.0))
# # # # #     theta_rad = math.radians(theta_deg)
    
# # # # #     # Special deviation formula
# # # # #     special_dev_cm = math.sqrt(max(0.0, 2.0 * (a_mm ** 2) * (1.0 - math.cos(theta_rad)))) * MM_TO_CM

# # # # #     # Correct direction calculation
# # # # #     vx, vy = ABx, ABy
# # # # #     THRESH = 2
# # # # #     if abs(vx) < THRESH: dir_x = ""
# # # # #     elif vx > 0: dir_x = "Right"
# # # # #     else: dir_x = "Left"

# # # # #     if abs(vy) < THRESH: dir_y = ""
# # # # #     elif vy > 0: dir_y = "Down"
# # # # #     else: dir_y = "Up"

# # # # #     if dir_x and dir_y: direction = f"{dir_y}-{dir_x}"
# # # # #     elif dir_x: direction = dir_x
# # # # #     elif dir_y: direction = dir_y
# # # # #     else: direction = "None"

# # # # #     measurement = {
# # # # #         "length_cm": float(length_cm),
# # # # #         "overall_tilt_pca_deg": float(angle_deg_pca),
# # # # #         "theta_dev_deg": float(theta_deg),
# # # # #         "special_dev_cm": float(special_dev_cm),
# # # # #         "direction": direction,
# # # # #         "max_segment_tilt_deg": float(tilt_seg_deg),
# # # # #         "max_tilt_segment_len_cm": float(tilt_seg_len_cm),
# # # # #         "pixels_per_mm": float(pixels_per_mm)
# # # # #     }

# # # # #     # Draw overlays
# # # # #     out = frame_with_aruco.copy()
# # # # #     cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,0), 2)
    
# # # # #     # Draw the skeleton in Cyan over the ROI
# # # # #     roi_out = roi.copy()
# # # # #     roi_out[skel > 0] = (255, 255, 0) # Cyan (B,G,R)

# # # # #     # Put the processed ROI back into the output image
# # # # #     out[y1:y2, x1:x2] = roi_out

# # # # #     # --- Full Wire Endpoint Calculation (Start/End of path) ---
# # # # #     wire_start_img = None
# # # # #     wire_end_img = None
# # # # #     if len(path) >= 2:
# # # # #         r1, c1 = path[0]; r2, c2 = path[-1]
# # # # #         wire_start_img = (int(x1 + c1), int(y1 + r1))
# # # # #         wire_end_img = (int(x1 + c2), int(y1 + r2))

# # # # #     # Draw special deviation line (A to B) (Marked in Yellow)
# # # # #     cv2.drawMarker(out, A_img, (0,200,0), markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2)
# # # # #     cv2.line(out, A_img, B_img, (0,255,255), 2)
    
# # # # #     # Draw FULL WIRE ENDPOINTS (GREEN Star / RED Diamond)
# # # # #     if wire_start_img and wire_end_img:
# # # # #         cv2.drawMarker(out, wire_start_img, (0,255,0), markerType=cv2.MARKER_STAR, markerSize=8, thickness=2)
# # # # #         cv2.putText(out, "START", (wire_start_img[0] + 5, wire_start_img[1] - 5), 
# # # # #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
# # # # #         cv2.drawMarker(out, wire_end_img, (0,0,255), markerType=cv2.MARKER_DIAMOND, markerSize=8, thickness=2)
# # # # #         cv2.putText(out, "END", (wire_end_img[0] + 5, wire_end_img[1] + 15), 
# # # # #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

# # # # #     # Draw MOST TILTED SEGMENT (HIGHLIGHTED MAGENTA/RED line)
# # # # #     if tilt_start_img and tilt_end_img:
# # # # #         cv2.line(out, tilt_start_img, tilt_end_img, (255,0,255), 5)
# # # # #         cv2.circle(out, tilt_start_img, 6, (255,0,255), -1)
# # # # #         cv2.circle(out, tilt_end_img, 6, (255,0,255), -1)
        
# # # # #         mid_x = (tilt_start_img[0] + tilt_end_img[0]) // 2
# # # # #         mid_y = (tilt_start_img[1] + tilt_end_img[1]) // 2
# # # # #         cv2.putText(out, "MAX TILT SEGMENT", (mid_x - 50, mid_y - 15), 
# # # # #                         cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,255), 1)

# # # # #     # Draw Text
# # # # #     text_main = f"Len:{length_cm:.3f}cm | PCA Tilt:{angle_deg_pca:.2f}deg | Dev:{special_dev_cm:.4f}cm | Dir:{direction}"
# # # # #     cv2.putText(out, text_main, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)
    
# # # # #     text_tilt_seg = f"Max Seg Tilt:{tilt_seg_deg:.2f}deg | Seg Len:{tilt_seg_len_cm:.3f}cm"
# # # # #     cv2.putText(out, text_tilt_seg, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2, cv2.LINE_AA)
    
# # # # #     cv2.putText(out, f"pix/mm:{pixels_per_mm:.3f}", (10, out.shape[0]-12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

# # # # #     return out, pixels_per_mm, measurement

# # # # # # -----------------------------
# # # # # # Single Image Analysis Function (New entry point for Flask)
# # # # # # -----------------------------
# # # # # def analyze_single_image(img_path: str, output_dir: str, filename: str):
# # # # #     """
# # # # #     Analyzes a single image file and saves the result to the specified directory.
    
# # # # #     :param img_path: Full path to the input image file.
# # # # #     :param output_dir: Directory where the output image and data will be saved.
# # # # #     :param filename: Original filename for naming the output file.
# # # # #     :return: A dictionary containing 'status' and 'measurement' data, or an error dict.
# # # # #     """
# # # # #     frame = cv2.imread(img_path)
    
# # # # #     if frame is None:
# # # # #         return {"status": "error", "message": f"Could not read image from path: {img_path}"}
        
# # # # #     try:
# # # # #         # Use fallback scale for initial run
# # # # #         pixels_per_mm = FALLBACK_PIXELS_PER_MM 
        
# # # # #         # Pass the frame to the core analysis function
# # # # #         out, used_pix_per_mm, meas = analyze_frame(frame, pixels_per_mm)
        
# # # # #         # Determine output file path (using PNG for annotated image)
# # # # #         base_name, _ = os.path.splitext(filename)
# # # # #         result_path = os.path.join(output_dir, f"{base_name}_result.png") 
        
# # # # #         # Save the annotated image
# # # # #         cv2.imwrite(result_path, out)
        
# # # # #         print(f"🖼️ Analysis result saved to: {result_path}")

# # # # #         if meas:
# # # # #             return {
# # # # #                 "status": "success",
# # # # #                 "measurement": meas
# # # # #             }
# # # # #         else:
# # # # #             return {"status": "error", "message": "Analysis failed to produce measurements (No wire/skeleton found)."}

# # # # #     except Exception as e:
# # # # #         error_message = f"An unexpected error occurred during analysis: {e}"
# # # # #         print(f"❌ {error_message}")
# # # # #         print(traceback.format_exc()) # Print stack trace for debugging
# # # # #         return {"status": "error", "message": error_message}


# # # # #changes for the trigger increase 4/12/2025

# # # # import cv2
# # # # import numpy as np
# # # # import math
# # # # import os
# # # # import traceback 
# # # # from ultralytics import YOLO
# # # # from skimage.morphology import skeletonize
# # # # from datetime import datetime
# # # # import csv
# # # # import glob

# # # # # -----------------------------
# # # # # CONFIG
# # # # # -----------------------------
# # # # MODEL_PATH = "models/yolov8n-seg.pt" 
# # # # UPLOAD_FOLDER = "uploads"
# # # # RESULTS_DIR = "results"

# # # # os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# # # # os.makedirs(RESULTS_DIR, exist_ok=True)

# # # # # Calibration Constants
# # # # FALLBACK_PIXELS_PER_MM = 9.727
# # # # MM_TO_CM = 0.1

# # # # # ArUco settings
# # # # ARUCO_SIDE_MM = 35.0
# # # # ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
# # # # ARUCO_PARAMS = cv2.aruco.DetectorParameters()
# # # # ARUCO_DETECTOR = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)

# # # # # -----------------------------
# # # # # Load YOLO Model (Global Instance)
# # # # # -----------------------------
# # # # print(f"Loading YOLO model from: {MODEL_PATH}")
# # # # try:
# # # #     # Attempt to load the model
# # # #     model = YOLO(MODEL_PATH)
# # # #     print("YOLO model loaded successfully.")

# # # #     # 🔥 CRITICAL FIX: YOLO Model Pre-Warmup 🔥
# # # #     # Run a dummy inference to initialize the model's internal graph (fusing layers, etc.).
# # # #     # This runs once when the worker starts, preventing the first user request from timing out.
# # # #     print("Pre-warming up YOLO model for faster inference...")
# # # #     dummy_input = np.zeros((64, 64, 3), dtype=np.uint8) # Small dummy input
# # # #     model.predict(dummy_input, verbose=False)
# # # #     print("YOLO model successfully pre-warmed.")

# # # # except Exception as e:
# # # #     print(f"❌ Error loading or warming up YOLO model: {e}. Check MODEL_PATH and dependencies.")
# # # #     # Use a dummy class to prevent errors if model loading fails
# # # #     class DummyYOLO:
# # # #         def __call__(self, *args, **kwargs):
# # # #             return []
# # # #     model = DummyYOLO()


# # # # # -----------------------------
# # # # # Utilities
# # # # # -----------------------------
# # # # def clamp(v, a, b):
# # # #     return max(a, min(b, v))

# # # # def normalize_angle(angle):
# # # #     # Constrains angle to the range (-90, 90]
# # # #     while angle > 180:
# # # #         angle -= 360
# # # #     while angle <= -180:
# # # #         angle += 360
# # # #     if angle > 90:
# # # #         angle -= 180
# # # #     if angle < -90:
# # # #         angle += 180
# # # #     return angle

# # # # # -----------------------------
# # # # # Skeleton helpers
# # # # # -----------------------------
# # # # def find_skeleton_endpoints(skel):
# # # #     endpoints = []
# # # #     h, w = skel.shape
# # # #     for r in range(h):
# # # #         for c in range(w):
# # # #             if not skel[r, c]:
# # # #                 continue
# # # #             r0 = max(0, r-1); r1 = min(h-1, r+1)
# # # #             c0 = max(0, c-1); c1 = min(w-1, c+1)
# # # #             neigh = skel[r0:r1+1, c0:c1+1]
# # # #             cnt = np.count_nonzero(neigh) - 1
# # # #             if cnt == 1:
# # # #                 endpoints.append((r, c))
# # # #     return endpoints

# # # # def trace_skeleton_path(skel):
# # # #     sk = (skel > 0).astype(np.uint8)
# # # #     num_labels, labels = cv2.connectedComponents(sk)
# # # #     if num_labels <= 1:
# # # #         return [], 0.0
    
# # # #     # Find the largest connected component (the main wire)
# # # #     best_label = 1
# # # #     best_count = 0
# # # #     for lab in range(1, num_labels):
# # # #         cnt = int(np.count_nonzero(labels == lab))
# # # #         if cnt > best_count:
# # # #             best_count = cnt
# # # #             best_label = lab
# # # #     comp = (labels == best_label).astype(np.uint8)
    
# # # #     coords_arr = np.column_stack(np.where(comp))
# # # #     coords = set((int(r), int(c)) for r, c in coords_arr)
# # # #     if not coords:
# # # #         return [], 0.0
    
# # # #     # Find endpoints for starting the trace
# # # #     endpoints = []
# # # #     for (r, c) in coords:
# # # #         cnt = 0
# # # #         for dr in (-1,0,1):
# # # #             for dc in (-1,0,1):
# # # #                 if dr == 0 and dc == 0: continue
# # # #                 if (r+dr, c+dc) in coords:
# # # #                     cnt += 1
# # # #         if cnt == 1:
# # # #             endpoints.append((r,c))
            
# # # #     start = endpoints[0] if endpoints else next(iter(coords))
    
# # # #     # Simple path tracing (Depth-First-Search style on skeleton)
# # # #     visited = {start}
# # # #     path = [start]
# # # #     cur = start
# # # #     while True:
# # # #         r, c = cur
# # # #         neighbors = []
# # # #         for dr in (-1,0,1):
# # # #             for dc in (-1,0,1):
# # # #                 if dr == 0 and dc == 0: continue
# # # #                 n = (r+dr, c+dc)
# # # #                 if n in coords and n not in visited:
# # # #                     neighbors.append(n)
        
# # # #         if not neighbors:
# # # #             # Backtrack if dead end
# # # #             found = False
# # # #             for node in reversed(path):
# # # #                 rr, cc = node
# # # #                 for dr in (-1,0,1):
# # # #                     for dc in (-1,0,1):
# # # #                         if dr == 0 and dc == 0: continue
# # # #                         n = (rr+dr, cc+dc)
# # # #                         if n in coords and n not in visited:
# # # #                             cur = node
# # # #                             found = True
# # # #                             break
# # # #                     if found: break
# # # #                 if found: break
# # # #             if not found:
# # # #                 break
# # # #             else:
# # # #                 continue
                
# # # #         # Move to the first available neighbor (simplest nearest neighbor)
# # # #         nxt = neighbors[0] 
# # # #         path.append(nxt)
# # # #         visited.add(nxt)
# # # #         cur = nxt
        
# # # #     return path, 0.0 

# # # # # -----------------------------
# # # # # ArUco detection & calibration
# # # # # -----------------------------
# # # # def detect_aruco_and_pixels_per_mm(frame):
# # # #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
# # # #     # Apply CLAHE for better ArUco detection contrast
# # # #     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# # # #     gray_enhanced = clahe.apply(gray) 
    
# # # #     corners, ids, rejected = ARUCO_DETECTOR.detectMarkers(gray_enhanced)
# # # #     vis = frame.copy()
    
# # # #     if ids is None or len(corners) == 0:
# # # #         return None, vis
        
# # # #     # Find the largest ArUco marker
# # # #     best_idx = 0
# # # #     best_area = 0.0
# # # #     for i, c in enumerate(corners):
# # # #         pts = c.reshape(-1,2).astype(np.float32)
# # # #         area = cv2.contourArea(pts)
# # # #         if area > best_area:
# # # #             best_area = area
# # # #             best_idx = i
            
# # # #     best_c = corners[best_idx].reshape(4,2)
# # # #     cv2.polylines(vis, [best_c.astype(np.int32)], True, (0,255,0), 2)
    
# # # #     # Calculate average side length in pixels
# # # #     side_lengths = []
# # # #     for k in range(4):
# # # #         p1 = best_c[k]; p2 = best_c[(k+1)%4]
# # # #         side_lengths.append(math.hypot(p2[0]-p1[0], p2[1]-p1[1]))
# # # #     avg_side_px = float(np.mean(side_lengths))
    
# # # #     if avg_side_px <= 0:
# # # #         return None, vis
        
# # # #     # Calculate pixels per real-world millimeter
# # # #     pixels_per_mm = avg_side_px / ARUCO_SIDE_MM
    
# # # #     try:
# # # #         id_val = int(ids[best_idx][0]) if ids is not None else -1
# # # #         cv2.putText(vis, f"ArUco ID:{id_val}", (int(best_c[0][0]), int(best_c[0][1]-10)),
# # # #                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
# # # #     except Exception:
# # # #         pass
        
# # # #     return pixels_per_mm, vis

# # # # # -----------------------------
# # # # # YOLO bbox helper
# # # # # -----------------------------
# # # # def get_largest_bbox_from_res(res):
# # # #     boxes = getattr(res, "boxes", None)
# # # #     if boxes is None or boxes.xyxy.numel() == 0:
# # # #         return None
    
# # # #     try:
# # # #         xyxy = boxes.xyxy.cpu().numpy()
# # # #     except Exception:
# # # #         return None
        
# # # #     if xyxy.size == 0:
# # # #         return None
        
# # # #     areas = (xyxy[:,2] - xyxy[:,0]) * (xyxy[:,3] - xyxy[:,1])
# # # #     idx = int(np.argmax(areas))
# # # #     x1,y1,x2,y2 = xyxy[idx].astype(int).tolist()
    
# # # #     return x1,y1,x2,y2, idx # Return index for mask extraction

# # # # # -----------------------------
# # # # # Segment Tilt Analysis
# # # # # -----------------------------
# # # # def calculate_segment_tilt(path, x_offset, y_offset, mm_per_pixel, segment_len_px=70):
# # # #     """
# # # #     Analyzes segments of the wire path to find the segment with the largest tilt.
# # # #     """
# # # #     max_tilt_deg = 0.0
# # # #     best_start_img, best_end_img = None, None

# # # #     # Use a sliding window, moving by half the segment length each step
# # # #     step_size = max(1, segment_len_px // 2)

# # # #     if len(path) < segment_len_px:
# # # #         # If the path is short, just use the endpoints
# # # #         if len(path) >= 2:
# # # #             r1, c1 = path[0]; r2, c2 = path[-1]
            
# # # #             dx = float(c2 - c1); dy = float(r2 - r1)
            
# # # #             if math.hypot(dx, dy) >= 5: # Minimum length check
# # # #                 angle_rad = math.atan2(dy, dx)
# # # #                 angle_deg = normalize_angle(math.degrees(angle_rad))
# # # #                 max_tilt_deg = abs(angle_deg)
                
# # # #                 best_start_img = (int(x_offset + c1), int(y_offset + r1))
# # # #                 best_end_img = (int(x_offset + c2), int(y_offset + r2))
                
# # # #     else:
# # # #         # Iterate over segments
# # # #         for i in range(0, len(path) - segment_len_px, step_size):
# # # #             start_point = path[i]; end_point = path[i + segment_len_px]
            
# # # #             r1, c1 = start_point; r2, c2 = end_point
            
# # # #             dx = float(c2 - c1); dy = float(r2 - r1)
            
# # # #             if math.hypot(dx, dy) < 5: continue # Skip very short segments
            
# # # #             angle_rad = math.atan2(dy, dx)
# # # #             angle_deg = normalize_angle(math.degrees(angle_rad))
            
# # # #             # Tilt is measured as the absolute angle from the horizontal/vertical axes
# # # #             current_tilt_deg = abs(angle_deg) 
            
# # # #             if current_tilt_deg > max_tilt_deg:
# # # #                 max_tilt_deg = current_tilt_deg
# # # #                 best_start_img = (int(x_offset + c1), int(y_offset + r1))
# # # #                 best_end_img = (int(x_offset + c2), int(y_offset + r2))

# # # #     if best_start_img and best_end_img:
# # # #         # Calculate the length of the most tilted segment
# # # #         dx = best_end_img[0] - best_start_img[0]
# # # #         dy = best_end_img[1] - best_start_img[1]
# # # #         length_px = math.hypot(dx, dy)
# # # #         length_cm = length_px * mm_per_pixel * MM_TO_CM
        
# # # #         return max_tilt_deg, length_cm, best_start_img, best_end_img
    
# # # #     return 0.0, 0.0, None, None


# # # # # -----------------------------
# # # # # Analyze frame (Uses the global 'model' instance)
# # # # # -----------------------------
# # # # def analyze_frame(frame, last_pixels_per_mm=FALLBACK_PIXELS_PER_MM):
# # # #     pixels_per_mm_frame, frame_with_aruco = detect_aruco_and_pixels_per_mm(frame)
    
# # # #     # Use detected scale, otherwise use the last successful scale or fallback
# # # #     pixels_per_mm = pixels_per_mm_frame if pixels_per_mm_frame is not None else last_pixels_per_mm
# # # #     mm_per_pixel = 1.0 / pixels_per_mm

# # # #     # YOLO detection
# # # #     results = model(frame, verbose=False)
# # # #     if len(results) == 0:
# # # #         return frame_with_aruco, pixels_per_mm, None
# # # #     res = results[0]
    
# # # #     # Get largest bbox and its index
# # # #     bbox_result = get_largest_bbox_from_res(res)
# # # #     if bbox_result is None:
# # # #         return frame_with_aruco, pixels_per_mm, None

# # # #     x1,y1,x2,y2, idx = bbox_result # Unpack the returned index
# # # #     x1 = clamp(x1, 0, frame.shape[1]-1); x2 = clamp(x2, 0, frame.shape[1]-1)
# # # #     y1 = clamp(y1, 0, frame.shape[0]-1); y2 = clamp(y2, 0, frame.shape[0]-1)
# # # #     roi = frame[y1:y2, x1:x2].copy()
# # # #     if roi.size == 0:
# # # #         return frame_with_aruco, pixels_per_mm, None

# # # #     # ----------------------------------------------------
# # # #     # CORE CHANGE: Image Processing (Use Segmentation Mask)
# # # #     # ----------------------------------------------------
# # # #     mask_bool = None
    
# # # #     # 1. Try to use the segmentation mask if available (more robust)
# # # #     if res.masks is not None and res.masks.xyxy.numel() > 0:
# # # #         try:
# # # #             # Get the mask corresponding to the largest detected object (idx)
# # # #             h_orig, w_orig = frame.shape[:2]
            
# # # #             mask_data = res.masks.data[idx].cpu().numpy()
# # # #             mask_resized = cv2.resize(mask_data, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
            
# # # #             # Crop the mask to the bounding box (ROI) area
# # # #             mask_roi = mask_resized[y1:y2, x1:x2]
            
# # # #             mask_bool = (mask_roi > 0.5) # Threshold mask to binary (True/False)
# # # #         except Exception as mask_e:
# # # #             print(f"Warning: Failed to process segmentation mask. Falling back to thresholding. Error: {mask_e}")
# # # #             pass # Fall through to thresholding logic

# # # #     # 2. Fallback to traditional thresholding if mask processing failed or wasn't available
# # # #     if mask_bool is None or np.count_nonzero(mask_bool) == 0:
# # # #         gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
# # # #         gray_eq = cv2.equalizeHist(gray)
# # # #         gray_blur = cv2.GaussianBlur(gray_eq, (5,5), 0)
# # # #         _, thr = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
# # # #         mask_bool = (thr > 0)
# # # #         # Handle cases where the wire might be darker than the background
# # # #         if np.count_nonzero(mask_bool) < 10:
# # # #             mask_bool = (~mask_bool) 

# # # #     # ----------------------------------------------------
    
# # # #     mask_bool = mask_bool.astype(bool)
# # # #     try:
# # # #         skel = skeletonize(mask_bool).astype(np.uint8)
# # # #     except Exception:
# # # #         skel = np.zeros_like(mask_bool, dtype=np.uint8)
        
# # # #     if np.count_nonzero(skel) == 0:
# # # #         out = frame_with_aruco.copy()
# # # #         cv2.rectangle(out, (x1,y1), (x2,y2), (0,128,255), 2)
# # # #         cv2.putText(out, "No skeleton/wire found", (x1+5,y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,128,255), 2)
# # # #         return out, pixels_per_mm, None

# # # #     # Wire Length and Path
# # # #     path, _ = trace_skeleton_path(skel) 
    
# # # #     # Use the robust method of counting non-zero skeleton pixels for length.
# # # #     length_px = float(np.count_nonzero(skel)) 
# # # #     length_cm = length_px * mm_per_pixel * MM_TO_CM

# # # #     # PCA Tilt (Overall tilt of the bounding box area)
# # # #     ys, xs = np.where(mask_bool)
# # # #     angle_deg_pca = 0.0
# # # #     if len(xs) >= 3:
# # # #         coords = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
# # # #         try:
# # # #             mean, eig = cv2.PCACompute(coords, mean=None)
# # # #             principal = eig[0]
# # # #             angle_deg_pca = normalize_angle(math.degrees(math.atan2(float(principal[1]), float(principal[0]))))
# # # #         except Exception:
# # # #             angle_deg_pca = 0.0

# # # #     # New: Tilt Segment Analysis (Finds max tilt segment and its length)
# # # #     tilt_seg_deg, tilt_seg_len_cm, tilt_start_img, tilt_end_img = calculate_segment_tilt(
# # # #         path, x1, y1, mm_per_pixel, segment_len_px=70)

# # # #     # Original Special Deviation (A to B logic)
# # # #     cx0 = roi.shape[1] / 2.0
# # # #     cy0 = roi.shape[0] / 2.0
# # # #     A_xy = (int(cx0), int(cy0)) # A is the center of the bounding box
    
# # # #     endpoints_xy = [(c, r) for (r, c) in find_skeleton_endpoints(skel)]
# # # #     if endpoints_xy:
# # # #         B_xy = max(endpoints_xy, key=lambda p: math.hypot(p[0]-A_xy[0], p[1]-A_xy[1]))
# # # #     elif len(path) >= 2:
# # # #         r1,c1 = path[0]; r2,c2 = path[-1]
# # # #         p1 = (c1, r1); p2 = (c2, r2)
# # # #         B_xy = p1 if math.hypot(p1[0]-A_xy[0], p1[1]-A_xy[1]) > math.hypot(p2[0]-A_xy[0], p2[1]-A_xy[1]) else p2
# # # #     else:
# # # #         B_xy = (A_xy[0] + 40, A_xy[1]) # Fallback

# # # #     A_img = (int(x1 + A_xy[0]), int(y1 + A_xy[1]))
# # # #     B_img = (int(x1 + B_xy[0]), int(y1 + B_xy[1]))

# # # #     a_px = math.hypot(B_xy[0] - A_xy[0], B_xy[1] - A_xy[1])
# # # #     a_mm = a_px * mm_per_pixel
# # # #     ABx = float(B_xy[0] - A_xy[0])
# # # #     ABy = float(B_xy[1] - A_xy[1])
    
# # # #     # Calculate Theta (Angle of vector AB relative to Y-axis)
# # # #     AB_angle_deg = math.degrees(math.atan2(ABy, ABx)) if a_px > 0 else 0.0
# # # #     theta_deg = abs(normalize_angle(AB_angle_deg - 90.0))
# # # #     theta_rad = math.radians(theta_deg)
    
# # # #     # Special deviation formula
# # # #     special_dev_cm = math.sqrt(max(0.0, 2.0 * (a_mm ** 2) * (1.0 - math.cos(theta_rad)))) * MM_TO_CM

# # # #     # Correct direction calculation
# # # #     vx, vy = ABx, ABy
# # # #     THRESH = 2
# # # #     if abs(vx) < THRESH: dir_x = ""
# # # #     elif vx > 0: dir_x = "Right"
# # # #     else: dir_x = "Left"

# # # #     if abs(vy) < THRESH: dir_y = ""
# # # #     elif vy > 0: dir_y = "Down"
# # # #     else: dir_y = "Up"

# # # #     if dir_x and dir_y: direction = f"{dir_y}-{dir_x}"
# # # #     elif dir_x: direction = dir_x
# # # #     elif dir_y: direction = dir_y
# # # #     else: direction = "None"

# # # #     measurement = {
# # # #         "length_cm": float(length_cm),
# # # #         "overall_tilt_pca_deg": float(angle_deg_pca),
# # # #         "theta_dev_deg": float(theta_deg),
# # # #         "special_dev_cm": float(special_dev_cm),
# # # #         "direction": direction,
# # # #         "max_segment_tilt_deg": float(tilt_seg_deg),
# # # #         "max_tilt_segment_len_cm": float(tilt_seg_len_cm),
# # # #         "pixels_per_mm": float(pixels_per_mm)
# # # #     }

# # # #     # Draw overlays
# # # #     out = frame_with_aruco.copy()
# # # #     cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,0), 2)
    
# # # #     # Draw the skeleton in Cyan over the ROI
# # # #     roi_out = roi.copy()
# # # #     roi_out[skel > 0] = (255, 255, 0) # Cyan (B,G,R)

# # # #     # Put the processed ROI back into the output image
# # # #     out[y1:y2, x1:x2] = roi_out

# # # #     # --- Full Wire Endpoint Calculation (Start/End of path) ---
# # # #     wire_start_img = None
# # # #     wire_end_img = None
# # # #     if len(path) >= 2:
# # # #         r1, c1 = path[0]; r2, c2 = path[-1]
# # # #         wire_start_img = (int(x1 + c1), int(y1 + r1))
# # # #         wire_end_img = (int(x1 + c2), int(y1 + r2))

# # # #     # Draw special deviation line (A to B) (Marked in Yellow)
# # # #     cv2.drawMarker(out, A_img, (0,200,0), markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2)
# # # #     cv2.line(out, A_img, B_img, (0,255,255), 2)
    
# # # #     # Draw FULL WIRE ENDPOINTS (GREEN Star / RED Diamond)
# # # #     if wire_start_img and wire_end_img:
# # # #         cv2.drawMarker(out, wire_start_img, (0,255,0), markerType=cv2.MARKER_STAR, markerSize=8, thickness=2)
# # # #         cv2.putText(out, "START", (wire_start_img[0] + 5, wire_start_img[1] - 5), 
# # # #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
# # # #         cv2.drawMarker(out, wire_end_img, (0,0,255), markerType=cv2.MARKER_DIAMOND, markerSize=8, thickness=2)
# # # #         cv2.putText(out, "END", (wire_end_img[0] + 5, wire_end_img[1] + 15), 
# # # #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

# # # #     # Draw MOST TILTED SEGMENT (HIGHLIGHTED MAGENTA/RED line)
# # # #     if tilt_start_img and tilt_end_img:
# # # #         cv2.line(out, tilt_start_img, tilt_end_img, (255,0,255), 5)
# # # #         cv2.circle(out, tilt_start_img, 6, (255,0,255), -1)
# # # #         cv2.circle(out, tilt_end_img, 6, (255,0,255), -1)
        
# # # #         mid_x = (tilt_start_img[0] + tilt_end_img[0]) // 2
# # # #         mid_y = (tilt_start_img[1] + tilt_end_img[1]) // 2
# # # #         cv2.putText(out, "MAX TILT SEGMENT", (mid_x - 50, mid_y - 15), 
# # # #                         cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,255), 1)

# # # #     # Draw Text
# # # #     text_main = f"Len:{length_cm:.3f}cm | PCA Tilt:{angle_deg_pca:.2f}deg | Dev:{special_dev_cm:.4f}cm | Dir:{direction}"
# # # #     cv2.putText(out, text_main, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)
    
# # # #     text_tilt_seg = f"Max Seg Tilt:{tilt_seg_deg:.2f}deg | Seg Len:{tilt_seg_len_cm:.3f}cm"
# # # #     cv2.putText(out, text_tilt_seg, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2, cv2.LINE_AA)
    
# # # #     cv2.putText(out, f"pix/mm:{pixels_per_mm:.3f}", (10, out.shape[0]-12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

# # # #     return out, pixels_per_mm, measurement

# # # # # -----------------------------
# # # # # Single Image Analysis Function (New entry point for Flask)
# # # # # -----------------------------
# # # # def analyze_single_image(img_path: str, output_dir: str, filename: str):
# # # #     """
# # # #     Analyzes a single image file and saves the result to the specified directory.
# # # #     """
# # # #     frame = cv2.imread(img_path)
    
# # # #     if frame is None:
# # # #         return {"status": "error", "message": f"Could not read image from path: {img_path}"}
        
# # # #     try:
# # # #         # Use fallback scale for initial run
# # # #         pixels_per_mm = FALLBACK_PIXELS_PER_MM 
        
# # # #         # Pass the frame to the core analysis function
# # # #         out, used_pix_per_mm, meas = analyze_frame(frame, pixels_per_mm)
        
# # # #         # Determine output file path (using PNG for annotated image)
# # # #         base_name, _ = os.path.splitext(filename)
# # # #         result_path = os.path.join(output_dir, f"{base_name}_result.png") 
        
# # # #         # Save the annotated image
# # # #         cv2.imwrite(result_path, out)
        
# # # #         print(f"🖼️ Analysis result saved to: {result_path}")

# # # #         if meas:
# # # #             return {
# # # #                 "status": "success",
# # # #                 "measurement": meas
# # # #             }
# # # #         else:
# # # #             return {"status": "error", "message": "Analysis failed to produce measurements (No wire/skeleton found)."}

# # # #     except Exception as e:
# # # #         error_message = f"An unexpected error occurred during analysis: {e}"
# # # #         print(f"❌ {error_message}")
# # # #         print(traceback.format_exc()) # Print stack trace for debugging
# # # #         return {"status": "error", "message": error_message}

# # # # # -----------------------------
# # # # # Batch Analysis Function (Called by Flask /analyze route)
# # # # # -----------------------------
# # # # def analyze_all():
# # # #     """
# # # #     1. Reads all images from UPLOAD_FOLDER.
# # # #     2. Runs analysis, calibration, and measurement for each.
# # # #     3. Calculates overall averages.
# # # #     4. Saves results (summary.txt, csv, annotated images) to a NEW timestamped folder.
# # # #     """
    
# # # #     # 1. Setup new results folder
# # # #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# # # #     current_results_dir = os.path.join(RESULTS_DIR, timestamp)
# # # #     os.makedirs(current_results_dir, exist_ok=True)
# # # #     print(f"Created new results directory: {current_results_dir}")

# # # #     # 2. Find all images to process
# # # #     allowed_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
# # # #     image_paths = [
# # # #         os.path.join(UPLOAD_FOLDER, f)
# # # #         for f in os.listdir(UPLOAD_FOLDER)
# # # #         if os.path.splitext(f)[1].lower() in allowed_extensions
# # # #     ]

# # # #     if not image_paths:
# # # #         print("No images found in the upload folder to analyze.")
# # # #         raise Exception("No images found in 'uploads' folder for batch analysis.")

# # # #     # 3. Process each image and collect results
# # # #     all_measurements = []
    
# # # #     # Track the last successful pixels_per_mm for continuity
# # # #     last_pixels_per_mm = FALLBACK_PIXELS_PER_MM
    
# # # #     for img_path in image_paths:
# # # #         filename = os.path.basename(img_path)
# # # #         print(f"Processing: {filename}")
        
# # # #         # Analyze the single image
# # # #         result = analyze_single_image(img_path, current_results_dir, filename)
        
# # # #         if result['status'] == 'success' and result.get('measurement'):
# # # #             meas = result['measurement']
            
# # # #             # Update scale factor for the next image
# # # #             if 'pixels_per_mm' in meas and meas['pixels_per_mm'] != FALLBACK_PIXELS_PER_MM:
# # # #                 last_pixels_per_mm = meas['pixels_per_mm']
                
# # # #             # Add filename and the scale factor to the measurement dictionary for the CSV
# # # #             meas['filename'] = filename
# # # #             all_measurements.append(meas)
# # # #         else:
# # # #             print(f"Skipped {filename}: {result['message']}")

# # # #     # 4. Aggregate results and create summary
# # # #     if not all_measurements:
# # # #         raise Exception("Analysis failed for all uploaded images.")

# # # #     # Calculate Averages for the final summary
# # # #     avg_data = {}
# # # #     total_count = len(all_measurements)
    
# # # #     # Find all numeric keys for averaging
# # # #     numeric_keys = [k for k in all_measurements[0].keys() if isinstance(all_measurements[0][k], (int, float)) and k != 'pixels_per_mm']
    
# # # #     for key in numeric_keys:
# # # #         values = [m[key] for m in all_measurements]
# # # #         avg_data[f"Average {key.replace('_', ' ').title()}"] = f"{np.mean(values):.4f}"

# # # #     # Add count and current timestamp
# # # #     avg_data['Image Count'] = str(total_count)
# # # #     avg_data['Timestamp'] = timestamp
    
# # # #     # Add other key stats
# # # #     # Example: max deviation
# # # #     max_dev = max(m.get('special_dev_cm', 0) for m in all_measurements)
# # # #     avg_data['Max Special Deviation (cm)'] = f"{max_dev:.4f}"
    
# # # #     # Example: max segment tilt
# # # #     max_tilt = max(m.get('max_segment_tilt_deg', 0) for m in all_measurements)
# # # #     avg_data['Max Segment Tilt (deg)'] = f"{max_tilt:.2f}"

# # # #     # 5. Save the final CSV report
# # # #     csv_path = os.path.join(current_results_dir, "wire_analysis_results.csv")
# # # #     fieldnames = ['filename'] + numeric_keys + ['direction'] # Specify order for clarity
    
# # # #     try:
# # # #         with open(csv_path, 'w', newline='') as csvfile:
# # # #             writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
# # # #             writer.writeheader()
# # # #             for meas in all_measurements:
# # # #                 # Filter out the 'pixels_per_mm' for the main CSV (it can be inconsistent)
# # # #                 csv_row = {k: v for k, v in meas.items() if k in fieldnames}
# # # #                 writer.writerow(csv_row)
# # # #         print(f"CSV report saved to: {csv_path}")
# # # #     except Exception as e:
# # # #         print(f"Error writing CSV file: {e}")

# # # #     # 6. Save the text summary
# # # #     summary_path = os.path.join(current_results_dir, "average_summary.txt")
# # # #     with open(summary_path, 'w') as f:
# # # #         for key, value in avg_data.items():
# # # #             f.write(f"{key}: {value}\n")
# # # #     print(f"Summary saved to: {summary_path}")
    
# # # #     # 7. Delete uploaded files after successful processing (optional but recommended for cleaning)
# # # #     for f in image_paths:
# # # #         try:
# # # #             os.remove(f)
# # # #         except Exception as e:
# # # #             print(f"Could not delete uploaded file {f}: {e}")
            
# # # #     print("Batch analysis complete.")
    
# # # #     return current_results_dir



# # # # ### 04/12/2025 changes for the accuracy 





# # import cv2
# # import numpy as np
# # import math
# # import os
# # import traceback 
# # from ultralytics import YOLO
# # from skimage.morphology import skeletonize
# # from datetime import datetime
# # import csv
# # import glob
# # import warnings

# # # Suppress scikit-image FutureWarning for skeletonize
# # warnings.filterwarnings("ignore", category=FutureWarning)

# # # -----------------------------
# # # CONFIGURATION ⚙️
# # # -----------------------------
# # # **FIX: Updated path to look in the 'models/' directory**
# # MODEL_PATH = "models/yolov8s-seg.pt" 
# # UPLOAD_FOLDER = "uploads"
# # RESULTS_DIR = "results" 

# # # Ensure directories exist
# # os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# # os.makedirs(RESULTS_DIR, exist_ok=True)

# # # Calibration Constants
# # FALLBACK_PIXELS_PER_MM = 9.727
# # MM_TO_CM = 0.1

# # # ArUco settings (4x4_1000 dictionary and 35.0 mm size)
# # ARUCO_SIDE_MM = 35.0
# # ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
# # ARUCO_PARAMS = cv2.aruco.DetectorParameters()
# # ARUCO_DETECTOR = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)

# # # Supported image extensions
# # IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

# # # --- HSV COLOR FILTERING FOR COPPER ISOLATION ---
# # # NOTE: These constants are now ignored/unused.
# # LOWER_COPPER = np.array([0, 10, 10])  
# # UPPER_COPPER = np.array([25, 255, 255]) 
# # # ------------------------------------------------

# # # -----------------------------
# # # Load YOLO Model (Global Instance) 🧠
# # # -----------------------------
# # print(f"Loading YOLO model from: {MODEL_PATH}")
# # try:
# #     if not os.path.exists(MODEL_PATH):
# #         # Fallback to the small segmentation model 'yolov8s-seg.pt' for robustness
# #         print(f"WARNING: Model file not found at: {MODEL_PATH}. Falling back to default 'yolov8s-seg.pt'.")
# #         model = YOLO("yolov8s-seg.pt") # Use a known segmentation model for robustness
# #     else:
# #         model = YOLO(MODEL_PATH)
# #     print("✅ YOLO model loaded successfully.")

# #     # 🔥 CRITICAL FIX: YOLO Model Pre-Warmup 🔥
# #     print("Pre-warming up YOLO model for faster inference...")
# #     dummy_input = np.zeros((64, 64, 3), dtype=np.uint8) # Small dummy input
# #     # Pre-warmup also runs on CPU for safety
# #     model.predict(dummy_input, verbose=False, device='cpu') 
# #     print("YOLO model successfully pre-warmed.")

# # except Exception as e:
# #     print(f"❌ Error loading or warming up YOLO model: {e}. Check MODEL_PATH and dependencies.")
# #     # Use a dummy class to prevent errors if model loading fails
# #     class DummyYOLO:
# #         def __call__(self, *args, **kwargs):
# #             return []
# #     model = DummyYOLO()


# # # -----------------------------
# # # Utilities
# # # -----------------------------
# # def clamp(v, a, b):
# #     return max(a, min(b, v))

# # def normalize_angle(angle):
# #     """Constrains angle to the range (-90, 90] for tilt analysis."""
# #     while angle > 180: angle -= 360
# #     while angle <= -180: angle += 360
# #     if angle > 90: angle -= 180
# #     if angle <= -90: angle += 180
# #     return angle

# # # -----------------------------
# # # Reporting Functions 📊
# # # -----------------------------

# # def generate_summary_file(filename, measurement, result_folder_path):
# #     """Writes the measurement results to a text file."""
# #     base_name, _ = os.path.splitext(filename)
# #     summary_path = os.path.join(result_folder_path, f"{base_name}_summary.txt") 
    
# #     with open(summary_path, 'w') as f:
# #         f.write(f"--- Analysis Summary for: {filename} ---\n")
        
# #         if measurement and measurement.get('length_cm') is not None:
# #             f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
# #             f.write("-" * 40 + "\n")
# #             f.write(f"Length (Total): {measurement['length_cm']:.3f} cm\n")
# #             f.write(f"Overall PCA Tilt: {measurement['overall_tilt_pca_deg']:.2f} deg\n")
# #             f.write(f"Max Segment Tilt: {measurement['max_segment_tilt_deg']:.2f} deg\n")
# #             f.write(f"Tilt Segment Length: {measurement['tilt_segment_length_cm']:.3f} cm\n")
# #             f.write(f"Special Deviation (Center-to-End): {measurement['special_dev_cm']:.4f} cm\n")
# #             f.write(f"Deviation Direction: {measurement['direction']}\n")
# #             f.write(f"Scale Used (pix/mm): {measurement['pixels_per_mm']:.3f}\n")
# #         else:
# #             reason = measurement.get('failure_reason', 'Could not detect or isolate the copper wire for measurement.') if isinstance(measurement, dict) else 'Unknown Failure'
# #             f.write(f"Analysis failed: {reason}\n")
            
# # def create_average_summary(all_measurements, result_folder_path):
# #     """Calculates and writes average/max metrics for the entire batch."""
    
# #     # This MUST be 'average_summary.txt' to match app.py
# #     avg_summary_path = os.path.join(result_folder_path, "average_summary.txt")
# #     successful_measurements = [m for m in all_measurements if m.get('length_cm') is not None]
    
# #     if not successful_measurements:
# #         with open(avg_summary_path, 'w') as f:
# #             f.write("Batch Analysis Failed: No successful measurements were recorded.\n")
# #         print("❌ Batch Analysis Failed: No successful measurements.")
# #         return

# #     all_lens = [m['length_cm'] for m in successful_measurements]
# #     all_pca_tilts = [abs(m['overall_tilt_pca_deg']) for m in successful_measurements]
# #     all_max_seg_tilts = [m['max_segment_tilt_deg'] for m in successful_measurements]
# #     all_devs = [m['special_dev_cm'] for m in successful_measurements]
# #     all_scales = [m['pixels_per_mm'] for m in successful_measurements]

# #     with open(avg_summary_path, 'w') as f:
# #         f.write("--- Batch Analysis Summary ---\n")
# #         f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
# #         f.write(f"Total Successful Images: {len(successful_measurements)} / {len(all_measurements)}\n")
# #         f.write("-" * 40 + "\n")
# #         # Keys here must be simple key: value pairs for app.py's parser
# #         f.write(f"Average Length (cm): {np.mean(all_lens):.3f}\n")
# #         f.write(f"Average PCA Tilt (deg): {np.mean(all_pca_tilts):.2f}\n")
# #         f.write(f"Max Segment Tilt (deg): {np.max(all_max_seg_tilts):.2f}\n")
# #         f.write(f"Maximum Special Deviation (cm): {np.max(all_devs):.4f}\n")
# #         f.write(f"Average Scale (pix/mm): {np.mean(all_scales):.3f}\n")
            
# #     print(f"⭐ Batch Summary created at: {avg_summary_path}")

# # def create_csv_report(all_measurements, result_folder_path):
# #     """Writes all individual measurements to a CSV file."""
# #     # This MUST be 'wire_analysis_results.csv' to match app.py
# #     csv_path = os.path.join(result_folder_path, "wire_analysis_results.csv")
    
# #     fieldnames = [
# #         'filename', 'length_cm', 'overall_tilt_pca_deg', 
# #         'max_segment_tilt_deg', 'tilt_segment_length_cm', 
# #         'special_dev_cm', 'direction', 'pixels_per_mm', 
# #         'theta_dev_deg', 'failure_reason'
# #     ]
    
# #     with open(csv_path, 'w', newline='') as csvfile:
# #         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
# #         writer.writeheader()
# #         for meas in all_measurements:
# #             # Prepare row, ensuring missing keys are handled gracefully
# #             row = {k: meas.get(k) for k in fieldnames} 
            
# #             # Format/clean up non-numeric fields
# #             for key in ['direction', 'failure_reason', 'filename']:
# #                 if row.get(key) is None:
# #                     row[key] = 'N/A'
            
# #             writer.writerow(row)
            
# #     print(f"📊 Detailed CSV report created at: {csv_path}")

# # # -----------------------------
# # # Skeleton helpers
# # # -----------------------------
# # def find_skeleton_endpoints(skel):
# #     """Identifies pixels with exactly one neighbor in the skeleton."""
# #     endpoints = []
# #     h, w = skel.shape
    
# #     # Fast iteration over only skeleton pixels
# #     sy, sx = np.where(skel)
    
# #     for r, c in zip(sy, sx):
# #         # Check 3x3 neighborhood, ensuring bounds
# #         r0, r1 = max(0, r-1), min(h, r+2)
# #         c0, c1 = max(0, c-1), min(w, c+2)
        
# #         # Count neighbors (subtract the center pixel itself)
# #         connections = np.sum(skel[r0:r1, c0:c1]) - skel[r, c]
        
# #         # An endpoint has exactly 1 connection 
# #         if connections == 1:
# #             endpoints.append((r, c))
            
# #     return endpoints

# # def trace_skeleton_path(skel):
# #     """
# #     Traces the main path of the skeleton to get an ordered list of coordinates 
# #     using the largest connected component and a simple greedy trace.
# #     """
# #     sk = (skel > 0).astype(np.uint8)
# #     num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(sk)
    
# #     if num_labels <= 1: return [], 0.0
    
# #     # 1. Find the largest connected component (excluding background)
# #     largest_label = 1
# #     largest_size = 0
# #     for lab in range(1, num_labels):
# #         if stats[lab, cv2.CC_STAT_AREA] > largest_size:
# #             largest_size = stats[lab, cv2.CC_STAT_AREA]
# #             largest_label = lab
            
# #     comp = (labels == largest_label).astype(np.uint8)
# #     coords = set((r, c) for r, c in np.column_stack(np.where(comp)))
# #     if not coords: return [], 0.0
    
# #     # 2. Find endpoints and select a starting point
# #     endpoints = find_skeleton_endpoints(comp)
    
# #     start = endpoints[0] if endpoints else next(iter(coords))
    
# #     # 3. Greedy path tracing
# #     visited = {start}
# #     path = [start]
# #     cur = start
    
# #     while True:
# #         r, c = cur
# #         neighbors = []
# #         for dr in (-1, 0, 1):
# #             for dc in (-1, 0, 1):
# #                 if dr == 0 and dc == 0: continue
# #                 n = (r + dr, c + dc)
# #                 if n in coords and n not in visited:
# #                     neighbors.append(n)
        
# #         if not neighbors: 
# #             # Simple Backtracking check (can be slow, but makes path robust)
# #             found = False
# #             for node in reversed(path):
# #                 rr, cc = node
# #                 for dr in (-1,0,1):
# #                     for dc in (-1,0,1):
# #                         if dr == 0 and dc == 0: continue
# #                         n = (rr+dr, cc+dc)
# #                         if n in coords and n not in visited:
# #                             cur = node
# #                             found = True
# #                             break
# #                     if found: break
# #                 if found: break
# #             if not found:
# #                 break
# #             else:
# #                 continue
                
# #         nxt = neighbors[0] # Pick the first available neighbor
# #         path.append(nxt)
# #         visited.add(nxt)
# #         cur = nxt
        
# #     return path, 0.0 # Keeping the original tuple structure

# # # -----------------------------
# # # ArUco detection & calibration
# # # -----------------------------
# # def detect_aruco_and_pixels_per_mm(frame):
# #     """Detects the ArUco marker and calculates the real-world scale (pixels/mm)."""
# #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
# #     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# #     gray_enhanced = clahe.apply(gray) 
    
# #     corners, ids, rejected = ARUCO_DETECTOR.detectMarkers(gray_enhanced)
# #     vis = frame.copy()
    
# #     if ids is None or len(corners) == 0:
# #         return None, vis
        
# #     best_idx = 0
# #     best_area = 0.0
# #     for i, c in enumerate(corners):
# #         pts = c.reshape(-1,2).astype(np.float32)
# #         area = cv2.contourArea(pts)
# #         if area > best_area:
# #             best_area = area
# #             best_idx = i
            
# #     best_c = corners[best_idx].reshape(4,2)
# #     cv2.polylines(vis, [best_c.astype(np.int32)], True, (0,255,0), 2)
    
# #     side_lengths = []
# #     for k in range(4):
# #         p1 = best_c[k]; p2 = best_c[(k+1)%4]
# #         side_lengths.append(math.hypot(p2[0]-p1[0], p2[1]-p1[1]))
# #     avg_side_px = float(np.mean(side_lengths))
    
# #     if avg_side_px <= 0:
# #         return None, vis
        
# #     pixels_per_mm = avg_side_px / ARUCO_SIDE_MM 
    
# #     try:
# #         id_val = int(ids[best_idx][0]) if ids is not None else -1
# #         cv2.putText(vis, f"ArUco ID:{id_val} Scale:{pixels_per_mm:.3f}", (int(best_c[0][0]), int(best_c[0][1]-10)),
# #                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
# #     except Exception:
# #         pass
        
# #     return pixels_per_mm, vis

# # # -----------------------------
# # # YOLO bbox helper
# # # -----------------------------
# # def get_largest_bbox_from_res(res):
# #     """Extracts the largest bounding box and its index from YOLO segmentation results."""
# #     boxes = getattr(res, "boxes", None)
# #     if boxes is None or boxes.xyxy.numel() == 0:
# #         return None
    
# #     try:
# #         xyxy = boxes.xyxy.cpu().numpy()
# #     except Exception:
# #         return None
        
# #     if xyxy.size == 0:
# #         return None
        
# #     areas = (xyxy[:,2] - xyxy[:,0]) * (xyxy[:,3] - xyxy[:,1])
# #     idx = int(np.argmax(areas))
# #     x1,y1,x2,y2 = xyxy[idx].astype(int).tolist()
    
# #     return x1,y1,x2,y2, idx # Return index for mask extraction

# # # -----------------------------
# # # Segment Tilt Analysis
# # # -----------------------------
# # def calculate_segment_tilt(path, x_offset, y_offset, mm_per_pixel, segment_len_px=70):
# #     """Analyzes segments of the wire path to find the segment with the largest tilt."""
# #     max_tilt_deg = 0.0
# #     best_start_img, best_end_img = None, None
# #     segment_length_cm = 0.0

# #     step_size = max(1, segment_len_px // 2)

# #     if len(path) < segment_len_px:
# #         if len(path) >= 2:
# #             start_point = path[0]; end_point = path[-1]
# #             r1, c1 = start_point; r2, c2 = end_point
            
# #             dx = float(c2 - c1); dy = float(r2 - r1)
            
# #             if math.hypot(dx, dy) >= 5:
# #                 angle_rad = math.atan2(dy, dx)
# #                 # Deviation from a purely horizontal wire (0 degrees)
# #                 angle_deg = normalize_angle(math.degrees(angle_rad)) 
# #                 max_tilt_deg = abs(angle_deg) 
                
# #                 best_start_img = (int(x_offset + c1), int(y_offset + r1))
# #                 best_end_img = (int(x_offset + c2), int(y_offset + r2))
# #                 length_px = math.hypot(dx, dy)
# #                 segment_length_cm = length_px * mm_per_pixel * MM_TO_CM
                
# #             else: # Path too short for meaningful tilt
# #                 return 0.0, 0.0, None, None
            
# #         else:
# #             return 0.0, 0.0, None, None
            
# #     else:
# #         # Iterate over segments
# #         for i in range(0, len(path) - segment_len_px, step_size):
# #             start_point = path[i]; end_point = path[i + segment_len_px]
            
# #             r1, c1 = start_point; r2, c2 = end_point
            
# #             dx = float(c2 - c1); dy = float(r2 - r1)
            
# #             if math.hypot(dx, dy) < 5: continue # Skip very short segments
            
# #             angle_rad = math.atan2(dy, dx)
# #             # Tilt is deviation from horizontal axis (0 degrees)
# #             angle_deg = normalize_angle(math.degrees(angle_rad)) 
# #             current_tilt_deg = abs(angle_deg) 
            
# #             if current_tilt_deg > max_tilt_deg:
# #                 max_tilt_deg = current_tilt_deg
# #                 best_start_img = (int(x_offset + c1), int(y_offset + r1))
# #                 best_end_img = (int(x_offset + c2), int(y_offset + r2))

# #         if best_start_img and best_end_img:
# #             # Calculate the length of the most tilted segment
# #             dx = best_end_img[0] - best_start_img[0]
# #             dy = best_end_img[1] - best_start_img[1]
# #             length_px = math.hypot(dx, dy)
# #             segment_length_cm = length_px * mm_per_pixel * MM_TO_CM
    
# #     return max_tilt_deg, segment_length_cm, best_start_img, best_end_img

# # # -----------------------------
# # # Core Analysis Function
# # # -----------------------------
# # def analyze_frame(frame, last_pixels_per_mm=FALLBACK_PIXELS_PER_MM):
# #     """Main function to detect, calibrate, and measure the wire."""
# #     pixels_per_mm_frame, frame_with_aruco = detect_aruco_and_pixels_per_mm(frame)
    
# #     pixels_per_mm = pixels_per_mm_frame if pixels_per_mm_frame is not None else last_pixels_per_mm
# #     mm_per_pixel = 1.0 / pixels_per_mm
    
# #     # 1. YOLO DETECTION & SEGMENTATION
# #     try:
# #         # **🔥 CRITICAL FIX: Explicitly set device='cpu' to prevent GPU/CUDA conflicts 🔥**
# #         results = model(frame, verbose=False, conf=0.1, device='cpu') 
# #     except Exception as e:
# #         print(f"DEBUG: YOLO Model Call Failed: {e}")
# #         return frame_with_aruco, pixels_per_mm, {'failure_reason': 'YOLO Model Call Failed'}

# #     if not isinstance(results, list) or len(results) == 0:
# #         return frame_with_aruco, pixels_per_mm, {'failure_reason': 'YOLO No Object Detected'}
        
# #     res = results[0]
# #     bbox_info = get_largest_bbox_from_res(res)
    
# #     if bbox_info is None or res.masks is None:
# #         # --- DEBUGGING ADDITION ---
# #         if res.masks is None:
# #             debug_reason = 'YOLO Mask Data Missing (Check Model Type)'
# #         elif res.boxes.xyxy.numel() == 0:
# #             debug_reason = 'YOLO Boxes Missing (No Detection)'
# #         else:
# #             debug_reason = 'YOLO BBox/Mask Extraction Failed (Internal Error)'
            
# #         print(f"DEBUG: YOLO Output Failure Reason: {debug_reason}")
# #         # --- END DEBUGGING ADDITION ---
# #         return frame_with_aruco, pixels_per_mm, {'failure_reason': debug_reason}

# #     x1,y1,x2,y2, idx = bbox_info
    
# #     # 2. Extract Mask for the largest detection
# #     try:
# #         mask_np = res.masks.data[idx].cpu().numpy().astype(np.uint8)
# #         mask_resized = cv2.resize(mask_np, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        
# #         # Crop mask to the ROI
# #         roi_mask = mask_resized[y1:y2, x1:x2]
        
# #     except Exception as e:
# #         print(f"DEBUG: Mask processing failed: {e}")
# #         return frame_with_aruco, pixels_per_mm, {'failure_reason': 'YOLO Mask Processing Failed'}
        
# #     x1 = clamp(x1, 0, frame.shape[1]); x2 = clamp(x2, 0, frame.shape[1])
# #     y1 = clamp(y1, 0, frame.shape[0]); y2 = clamp(y2, 0, frame.shape[0])
    
# #     if x2 <= x1 or y2 <= y1 or roi_mask.size == 0:
# #         return frame_with_aruco, pixels_per_mm, {'failure_reason': 'BBox/ROI Invalid Dimensions'}

# #     roi = frame[y1:y2, x1:x2].copy()

# #     # 3. REVISED MASK ISOLATION (Using YOLO Segmentation ONLY)
# #     # The YOLO segmentation mask is used directly for isolation.
# #     combined_mask = roi_mask.copy() 

# #     kernel = np.ones((3,3), np.uint8)
# #     combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
# #     combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)
    
# #     mask_bool = (combined_mask > 0).astype(bool)
    
# #     # Add debug drawing for clarity in case of failure
# #     out = frame_with_aruco.copy()
# #     cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,0), 2)
    
# #     if np.count_nonzero(mask_bool) < 50: 
# #         failure_msg = "Isolation FAILED (Insufficient Pixels after YOLO Cleanup)"
# #         cv2.putText(out, failure_msg, (x1+5,y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,128,255), 1)
# #         return out, pixels_per_mm, {'failure_reason': failure_msg}

# #     # 4. SKELETONIZATION & PATH TRACING
# #     try:
# #         skel = skeletonize(mask_bool).astype(np.uint8)
# #     except Exception:
# #         skel = np.zeros_like(mask_bool, dtype=np.uint8)
        
# #     if np.count_nonzero(skel) < 5:
# #         failure_msg = "Isolation FAILED (No Skeleton/Thinning Error)"
# #         cv2.putText(out, failure_msg, (x1+5,y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,128,255), 1)
# #         return out, pixels_per_mm, {'failure_reason': failure_msg}

# #     path, _ = trace_skeleton_path(skel) 
    
# #     if len(path) < 5:
# #         failure_msg = "Path Tracing Failed (Path too short/discontinuous)"
# #         cv2.putText(out, failure_msg, (x1+5,y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,128,255), 1)
# #         return out, pixels_per_mm, {'failure_reason': failure_msg}


# #     # --- Wire Length (Path Distance) ---
# #     length_px = 0.0
# #     for i in range(len(path) - 1):
# #         r1, c1 = path[i]; r2, c2 = path[i+1]
# #         length_px += math.hypot(c2 - c1, r2 - r1)
            
# #     length_cm = length_px * mm_per_pixel * MM_TO_CM

# #     # --- Overall PCA Tilt ---
# #     ys, xs = np.where(mask_bool)
# #     angle_deg_pca = 0.0
# #     if len(xs) >= 3:
# #         coords = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
# #         try:
# #             mean, eig = cv2.PCACompute(coords, mean=None)
# #             principal = eig[0]
# #             angle_from_horizontal = math.degrees(math.atan2(float(principal[1]), float(principal[0])))
# #             # Tilt is deviation from 0/180 (horizontal line)
# #             angle_deg_pca = abs(normalize_angle(angle_from_horizontal)) 
# #         except Exception:
# #             angle_deg_pca = 0.0

# #     # --- Max Tilt Segment Analysis ---
# #     tilt_seg_deg, tilt_seg_len_cm, tilt_start_img, tilt_end_img = calculate_segment_tilt(
# #         path, x1, y1, mm_per_pixel, segment_len_px=70)

# #     # --- Special Deviation (Center-to-End Logic) ---
# #     cx0 = roi.shape[1] / 2.0 
# #     cy0 = roi.shape[0] / 2.0 
# #     A_xy = (int(cx0), int(cy0)) # Center of the ROI/BBox (relative to ROI)
    
# #     endpoints_xy = [(c, r) for (r, c) in find_skeleton_endpoints(skel)]
# #     if endpoints_xy:
# #         B_xy = max(endpoints_xy, key=lambda p: math.hypot(p[0]-A_xy[0], p[1]-A_xy[1]))
# #     else:
# #         # Fallback to the farthest path point from the center
# #         p1 = (path[0][1], path[0][0])
# #         p2 = (path[-1][1], path[-1][0])
# #         B_xy = p1 if math.hypot(p1[0]-A_xy[0], p1[1]-A_xy[1]) > math.hypot(p2[0]-A_xy[0], p2[1]-A_xy[1]) else p2

# #     A_img = (int(x1 + A_xy[0]), int(y1 + A_xy[1])) 
# #     B_img = (int(x1 + B_xy[0]), int(y1 + B_xy[1])) 

# #     ABx = float(B_xy[0] - A_xy[0])
# #     ABy = float(B_xy[1] - A_xy[1])
# #     a_px = math.hypot(ABx, ABy)
# #     a_mm = a_px * mm_per_pixel
    
# #     # Angle of the vector A->B 
# #     AB_angle_deg = math.degrees(math.atan2(ABy, ABx)) if a_px > 0 else 0.0
# #     # Theta is the deviation from the expected horizontal axis (0 degrees)
# #     theta_deg = abs(normalize_angle(AB_angle_deg)) 
# #     theta_rad = math.radians(theta_deg)
    
# #     # Deviation formula
# #     special_dev_cm = math.sqrt(max(0.0, 2.0 * (a_mm ** 2) * (1.0 - math.cos(theta_rad)))) * MM_TO_CM

# #     # Direction calculation
# #     vx, vy = ABx, ABy
# #     THRESH = 2 
# #     if abs(vx) < THRESH: dir_x = ""
# #     elif vx > 0: dir_x = "Right"
# #     else: dir_x = "Left"

# #     if abs(vy) < THRESH: dir_y = ""
# #     elif vy > 0: dir_y = "Down"
# #     else: dir_y = "Up"

# #     if dir_x and dir_y: direction = f"{dir_y}-{dir_x}"
# #     elif dir_x: direction = dir_x
# #     elif dir_y: direction = dir_y
# #     else: direction = "None"

# #     measurement = {
# #         "length_cm": float(length_cm),
# #         "overall_tilt_pca_deg": float(angle_deg_pca),
# #         "theta_dev_deg": float(theta_deg),
# #         "special_dev_cm": float(special_dev_cm),
# #         "direction": direction,
# #         "max_segment_tilt_deg": float(tilt_seg_deg),
# #         "tilt_segment_length_cm": float(tilt_seg_len_cm), 
# #         "pixels_per_mm": float(pixels_per_mm),
# #         "failure_reason": None
# #     }

# #     # --- Draw Overlays ---
# #     roi_out = roi.copy()
# #     roi_out[combined_mask > 0] = (0, 0, 0) # Black for the isolated wire area
# #     roi_out[skel > 0] = (255, 255, 0) # Cyan for the skeleton
# #     out[y1:y2, x1:x2] = roi_out 
    
# #     # Draw special deviation line (A to B) (Yellow)
# #     cv2.drawMarker(out, A_img, (0,200,0), markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2)
# #     cv2.line(out, A_img, B_img, (0,255,255), 2)
# #     cv2.drawMarker(out, B_img, (0,0,255), markerType=cv2.MARKER_DIAMOND, markerSize=8, thickness=2)

# #     # Draw MOST TILTED SEGMENT (MAGENTA)
# #     if tilt_start_img and tilt_end_img:
# #         cv2.line(out, tilt_start_img, tilt_end_img, (255,0,255), 5)
# #         cv2.circle(out, tilt_start_img, 6, (255,0,255), -1)
# #         cv2.circle(out, tilt_end_img, 6, (255,0,255), -1)
        
# #         mid_x = (tilt_start_img[0] + tilt_end_img[0]) // 2
# #         mid_y = (tilt_start_img[1] + tilt_end_img[1]) // 2
# #         cv2.putText(out, f"MAX TILT {tilt_seg_deg:.2f} deg", (mid_x - 50, mid_y - 15), 
# #                         cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,255), 1)

# #     # Draw Text Overlays
# #     text_main = f"Len:{length_cm:.3f}cm | PCA Tilt:{angle_deg_pca:.2f}deg | Dev:{special_dev_cm:.4f}cm | Dir:{direction}"
# #     cv2.putText(out, text_main, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)
    
# #     text_tilt_seg = f"Max Seg Tilt:{tilt_seg_deg:.2f}deg | Seg Len:{tilt_seg_len_cm:.3f}cm" 
# #     cv2.putText(out, text_tilt_seg, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2, cv2.LINE_AA)
    
# #     cv2.putText(out, f"pix/mm:{pixels_per_mm:.3f}", (10, out.shape[0]-12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

# #     return out, pixels_per_mm, measurement

# # # -----------------------------
# # # Main Batch Processing 📦 (Named analyze_all to match app.py)
# # # -----------------------------
# # def analyze_all():
# #     """Batch processes images and generates reports. Called by app.py."""
    
# #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# #     result_folder_path = os.path.join(RESULTS_DIR, timestamp)
# #     os.makedirs(result_folder_path, exist_ok=True)
    
# #     # Use glob for robust file extension matching
# #     image_paths = glob.glob(os.path.join(UPLOAD_FOLDER, '*'))
# #     image_files = [f for f in image_paths if f.lower().endswith(IMAGE_EXTS)]
    
# #     if not image_files:
# #         print(f"❌ No images found in the '{UPLOAD_FOLDER}' folder. Please place your images there.")
# #         # Create a placeholder summary so app.py can still find the folder
# #         with open(os.path.join(result_folder_path, "average_summary.txt"), 'w') as f:
# #             f.write("Batch Analysis Failed: No input images found.\n")
# #         return

# #     print(f"✅ Found {len(image_files)} images to process. Results saved to: {result_folder_path}")
    
# #     pixels_per_mm = FALLBACK_PIXELS_PER_MM 
# #     all_measurements = []

# #     for image_path in image_files:
# #         filename = os.path.basename(image_path)
# #         print(f"-> Processing {filename}...")
        
# #         try:
# #             # Read the image
# #             frame = cv2.imread(image_path)
# #             if frame is None:
# #                 print(f"WARNING: Could not read image {filename}. Skipping.")
# #                 all_measurements.append({'filename': filename, 'failure_reason': 'Could not read image file.'})
# #                 continue

# #             # Analyze the frame
# #             out_frame, current_scale, measurement = analyze_frame(frame, pixels_per_mm)
            
# #             # Update the scale for the next image, if a marker was found and is not the fallback
# #             if current_scale != FALLBACK_PIXELS_PER_MM:
# #                 pixels_per_mm = current_scale
# #                 print(f"  -> Scale calibrated to {pixels_per_mm:.3f} pix/mm.")
            
# #             # Store the filename and measurement
# #             measurement['filename'] = filename
# #             all_measurements.append(measurement)

# #             # Save the annotated image
# #             out_path = os.path.join(result_folder_path, f"{os.path.splitext(filename)[0]}_result.png")
# #             cv2.imwrite(out_path, out_frame)
            
# #             # Generate individual summary
# #             generate_summary_file(filename, measurement, result_folder_path)
            
# #             print(f"  -> Analysis complete. Dev: {measurement.get('special_dev_cm', 'N/A')} cm")

# #         except Exception as e:
# #             print(f"❌ CRITICAL ERROR processing {filename}: {e}")
# #             traceback.print_exc()
# #             all_measurements.append({'filename': filename, 'failure_reason': f'CRITICAL EXCEPTION: {e}'})

# #     # Generate Batch Reports
# #     create_average_summary(all_measurements, result_folder_path)
# #     create_csv_report(all_measurements, result_folder_path)

# # if __name__ == "__main__":
# #     # This part is optional but useful for local testing of the script
# #     analyze_all()




# ####for the lenght measure accuracy the above code works but show more that 30cm 
# import cv2
# import numpy as np
# import math
# import os
# import traceback 
# from ultralytics import YOLO
# from skimage.morphology import skeletonize
# from datetime import datetime
# import csv
# import glob
# import warnings

# # Suppress scikit-image FutureWarning for skeletonize
# warnings.filterwarnings("ignore", category=FutureWarning)

# # -----------------------------
# # CONFIGURATION ⚙️
# # -----------------------------
# # Using a known segmentation model for robustness
# MODEL_PATH = "models/yolov8s-seg.pt" 
# UPLOAD_FOLDER = "uploads"
# RESULTS_DIR = "results" 

# # Ensure directories exist
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(RESULTS_DIR, exist_ok=True)

# # Calibration Constants
# FALLBACK_PIXELS_PER_MM = 9.727 # Fallback scale if ArUco marker is not detected
# MM_TO_CM = 0.1

# # ArUco settings (4x4_1000 dictionary and 35.0 mm size)
# ARUCO_SIDE_MM = 35.0
# ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
# ARUCO_PARAMS = cv2.aruco.DetectorParameters()
# ARUCO_DETECTOR = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)

# # Supported image extensions
# IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

# # -----------------------------
# # Load YOLO Model (Global Instance) 🧠
# # -----------------------------
# print(f"Loading YOLO model from: {MODEL_PATH}")
# try:
#     if not os.path.exists(MODEL_PATH):
#         # Fallback to the small segmentation model 'yolov8s-seg.pt' for robustness
#         print(f"WARNING: Model file not found at: {MODEL_PATH}. Falling back to default 'yolov8s-seg.pt'.")
#         model = YOLO("yolov8s-seg.pt") # Use a known segmentation model for robustness
#     else:
#         model = YOLO(MODEL_PATH)
#     print("✅ YOLO model loaded successfully.")

#     # CRITICAL FIX: YOLO Model Pre-Warmup on CPU
#     print("Pre-warming up YOLO model for faster inference...")
#     dummy_input = np.zeros((64, 64, 3), dtype=np.uint8) # Small dummy input
#     model.predict(dummy_input, verbose=False, device='cpu') 
#     print("YOLO model successfully pre-warmed.")

# except Exception as e:
#     print(f"❌ Error loading or warming up YOLO model: {e}. Check MODEL_PATH and dependencies.")
#     class DummyYOLO:
#         def __call__(self, *args, **kwargs):
#             return []
#     model = DummyYOLO()


# # -----------------------------
# # Utilities
# # -----------------------------
# def clamp(v, a, b):
#     """Clamps a value v between a and b."""
#     return max(a, min(b, v))

# def normalize_angle(angle):
#     """Constrains angle to the range (-90, 90] for tilt analysis relative to horizontal."""
#     while angle > 180: angle -= 360
#     while angle <= -180: angle += 360
#     if angle > 90: angle -= 180
#     if angle <= -90: angle += 180
#     return angle

# # -----------------------------
# # Reporting Functions 📊
# # -----------------------------

# def generate_summary_file(filename, measurement, result_folder_path):
#     """Writes the measurement results to a text file."""
#     base_name, _ = os.path.splitext(filename)
#     summary_path = os.path.join(result_folder_path, f"{base_name}_summary.txt") 
    
#     with open(summary_path, 'w') as f:
#         f.write(f"--- Analysis Summary for: {filename} ---\n")
        
#         if measurement and measurement.get('length_cm') is not None:
#             f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
#             f.write("-" * 40 + "\n")
#             f.write(f"Length (Total): {measurement['length_cm']:.3f} cm\n")
#             f.write(f"Overall PCA Tilt: {measurement['overall_tilt_pca_deg']:.2f} deg\n")
#             f.write(f"Max Segment Tilt: {measurement['max_segment_tilt_deg']:.2f} deg\n")
#             f.write(f"Tilt Segment Length: {measurement['tilt_segment_length_cm']:.3f} cm\n")
#             f.write(f"Special Deviation (Center-to-End): {measurement['special_dev_cm']:.4f} cm\n")
#             f.write(f"Deviation Direction: {measurement['direction']}\n")
#             f.write(f"Scale Used (pix/mm): {measurement['pixels_per_mm']:.3f}\n")
#         else:
#             reason = measurement.get('failure_reason', 'Could not detect or isolate the copper wire for measurement.') if isinstance(measurement, dict) else 'Unknown Failure'
#             f.write(f"Analysis failed: {reason}\n")
            
# def create_average_summary(all_measurements, result_folder_path):
#     """Calculates and writes average/max metrics for the entire batch."""
    
#     avg_summary_path = os.path.join(result_folder_path, "average_summary.txt")
#     successful_measurements = [m for m in all_measurements if m.get('length_cm') is not None]
    
#     if not successful_measurements:
#         with open(avg_summary_path, 'w') as f:
#             f.write("Batch Analysis Failed: No successful measurements were recorded.\n")
#         print("❌ Batch Analysis Failed: No successful measurements.")
#         return

#     all_lens = [m['length_cm'] for m in successful_measurements]
#     all_pca_tilts = [abs(m['overall_tilt_pca_deg']) for m in successful_measurements]
#     all_max_seg_tilts = [m['max_segment_tilt_deg'] for m in successful_measurements]
#     all_devs = [m['special_dev_cm'] for m in successful_measurements]
#     all_scales = [m['pixels_per_mm'] for m in successful_measurements]

#     with open(avg_summary_path, 'w') as f:
#         f.write("--- Batch Analysis Summary ---\n")
#         f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
#         f.write(f"Total Successful Images: {len(successful_measurements)} / {len(all_measurements)}\n")
#         f.write("-" * 40 + "\n")
#         f.write(f"Average Length (cm): {np.mean(all_lens):.3f}\n")
#         f.write(f"Average PCA Tilt (deg): {np.mean(all_pca_tilts):.2f}\n")
#         f.write(f"Max Segment Tilt (deg): {np.max(all_max_seg_tilts):.2f}\n")
#         f.write(f"Maximum Special Deviation (cm): {np.max(all_devs):.4f}\n")
#         f.write(f"Average Scale (pix/mm): {np.mean(all_scales):.3f}\n")
            
#     print(f"⭐ Batch Summary created at: {avg_summary_path}")

# def create_csv_report(all_measurements, result_folder_path):
#     """Writes all individual measurements to a CSV file."""
#     csv_path = os.path.join(result_folder_path, "wire_analysis_results.csv")
    
#     fieldnames = [
#         'filename', 'length_cm', 'overall_tilt_pca_deg', 
#         'max_segment_tilt_deg', 'tilt_segment_length_cm', 
#         'special_dev_cm', 'direction', 'pixels_per_mm', 
#         'theta_dev_deg', 'failure_reason'
#     ]
    
#     with open(csv_path, 'w', newline='') as csvfile:
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         writer.writeheader()
#         for meas in all_measurements:
#             row = {k: meas.get(k) for k in fieldnames} 
            
#             for key in ['direction', 'failure_reason', 'filename']:
#                 if row.get(key) is None:
#                     row[key] = 'N/A'
            
#             writer.writerow(row)
            
#     print(f"📊 Detailed CSV report created at: {csv_path}")

# # -----------------------------
# # Skeleton helpers
# # -----------------------------
# def find_skeleton_endpoints(skel):
#     """
#     Identifies pixels with exactly one neighbor in the skeleton.
#     Uses cv2.filter2D for fast neighborhood counting.
#     """
#     kernel = np.array([[1, 1, 1],
#                        [1, 0, 1],
#                        [1, 1, 1]], dtype=np.uint8)
                       
#     neighbor_count = cv2.filter2D(skel.astype(np.uint8), -1, kernel, borderType=cv2.BORDER_CONSTANT)
    
#     endpoints_mask = (skel > 0) & (neighbor_count == 1)
    
#     sy, sx = np.where(endpoints_mask)
#     endpoints = list(zip(sy, sx))
            
#     return endpoints

# def trace_skeleton_path(skel):
#     """
#     Traces the main path of the skeleton to get an ordered list of coordinates 
#     using the largest connected component and a simple greedy trace.
#     """
#     sk = (skel > 0).astype(np.uint8)
#     num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(sk)
    
#     if num_labels <= 1: return [], 0.0
    
#     # 1. Find the largest connected component
#     largest_label = 1
#     largest_size = 0
#     for lab in range(1, num_labels):
#         if stats[lab, cv2.CC_STAT_AREA] > largest_size:
#             largest_size = stats[lab, cv2.CC_STAT_AREA]
#             largest_label = lab
            
#     comp = (labels == largest_label).astype(np.uint8)
#     coords = set((r, c) for r, c in np.column_stack(np.where(comp)))
#     if not coords: return [], 0.0
    
#     # 2. Find endpoints and select a starting point
#     endpoints = find_skeleton_endpoints(comp)
    
#     start = endpoints[0] if endpoints else next(iter(coords))
    
#     # 3. Greedy path tracing
#     visited = {start}
#     path = [start]
#     cur = start
    
#     while True:
#         r, c = cur
#         neighbors = []
        
#         for dr in (-1, 0, 1):
#             for dc in (-1, 0, 1):
#                 if dr == 0 and dc == 0: continue
#                 n = (r + dr, c + dc)
#                 if n in coords and n not in visited:
#                     neighbors.append(n)
        
#         if not neighbors: 
#             break
            
#         nxt = neighbors[0] 
        
#         path.append(nxt)
#         visited.add(nxt)
#         cur = nxt
            
#     return path, 0.0

# # -----------------------------
# # ArUco detection & calibration
# # -----------------------------
# def detect_aruco_and_pixels_per_mm(frame):
#     """Detects the ArUco marker and calculates the real-world scale (pixels/mm)."""
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#     gray_enhanced = clahe.apply(gray) 
    
#     corners, ids, rejected = ARUCO_DETECTOR.detectMarkers(gray_enhanced)
#     vis = frame.copy()
    
#     if ids is None or len(corners) == 0:
#         return None, vis
        
#     best_idx = 0
#     best_area = 0.0
#     for i, c in enumerate(corners):
#         pts = c.reshape(-1,2).astype(np.float32)
#         area = cv2.contourArea(pts)
#         if area > best_area:
#             best_area = area
#             best_idx = i
            
#     best_c = corners[best_idx].reshape(4,2)
#     cv2.polylines(vis, [best_c.astype(np.int32)], True, (0,255,0), 2)
    
#     side_lengths = []
#     for k in range(4):
#         p1 = best_c[k]; p2 = best_c[(k+1)%4]
#         side_lengths.append(math.hypot(p2[0]-p1[0], p2[1]-p1[1]))
#     avg_side_px = float(np.mean(side_lengths))
    
#     if avg_side_px <= 0:
#         return None, vis
        
#     pixels_per_mm = avg_side_px / ARUCO_SIDE_MM 
    
#     try:
#         id_val = int(ids[best_idx][0]) if ids is not None else -1
#         cv2.putText(vis, f"ArUco ID:{id_val} Scale:{pixels_per_mm:.3f}", (int(best_c[0][0]), int(best_c[0][1]-10)),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
#     except Exception:
#         pass
        
#     return pixels_per_mm, vis

# # -----------------------------
# # YOLO bbox helper
# # -----------------------------
# def get_largest_bbox_from_res(res):
#     """Extracts the largest bounding box and its index from YOLO segmentation results."""
#     boxes = getattr(res, "boxes", None)
#     if boxes is None or boxes.xyxy.numel() == 0:
#         return None
    
#     try:
#         xyxy = boxes.xyxy.cpu().numpy()
#     except Exception:
#         return None
        
#     if xyxy.size == 0:
#         return None
        
#     areas = (xyxy[:,2] - xyxy[:,0]) * (xyxy[:,3] - xyxy[:,1])
#     idx = int(np.argmax(areas))
#     x1,y1,x2,y2 = xyxy[idx].astype(int).tolist()
    
#     return x1,y1,x2,y2, idx 

# # -----------------------------
# # Segment Tilt Analysis
# # -----------------------------
# def calculate_segment_tilt(path, x_offset, y_offset, mm_per_pixel, segment_len_px=70):
#     """
#     Analyzes segments of the wire path to find the segment with the largest tilt.
#     """
#     max_tilt_deg = 0.0
#     best_start_img, best_end_img = None, None
#     segment_length_cm = 0.0

#     step_size = max(1, segment_len_px // 2)

#     if len(path) < segment_len_px:
#         segment_len_px = len(path) - 1 
#         if segment_len_px < 2:
#             return 0.0, 0.0, None, None
            
#         i_range = [0]
#     else:
#         i_range = range(0, len(path) - segment_len_px, step_size)
    
#     for i in i_range:
#         start_point = path[i]; end_point = path[i + segment_len_px]
        
#         r1, c1 = start_point; r2, c2 = end_point
        
#         dx = float(c2 - c1); dy = float(r2 - r1)
        
#         if math.hypot(dx, dy) < 5: continue 
        
#         angle_rad = math.atan2(dy, dx)
#         angle_deg = normalize_angle(math.degrees(angle_rad)) 
#         current_tilt_deg = abs(angle_deg) 
        
#         if current_tilt_deg > max_tilt_deg:
#             max_tilt_deg = current_tilt_deg
#             best_start_img = (int(x_offset + c1), int(y_offset + r1))
#             best_end_img = (int(x_offset + c2), int(y_offset + r2))

#     if best_start_img and best_end_img:
#         dx = best_end_img[0] - best_start_img[0]
#         dy = best_end_img[1] - best_start_img[1]
#         length_px = math.hypot(dx, dy)
#         segment_length_cm = length_px * mm_per_pixel * MM_TO_CM
    
#     return max_tilt_deg, segment_length_cm, best_start_img, best_end_img

# # -----------------------------
# # Core Analysis Function
# # -----------------------------
# def analyze_frame(frame, last_pixels_per_mm=FALLBACK_PIXELS_PER_MM):
#     """Main function to detect, calibrate, and measure the wire."""
#     pixels_per_mm_frame, frame_with_aruco = detect_aruco_and_pixels_per_mm(frame)
    
#     pixels_per_mm = pixels_per_mm_frame if pixels_per_mm_frame is not None else last_pixels_per_mm
#     mm_per_pixel = 1.0 / pixels_per_mm
    
#     # 1. YOLO DETECTION & SEGMENTATION
#     try:
#         # CRITICAL FIX: Explicitly set device='cpu'
#         results = model(frame, verbose=False, conf=0.1, device='cpu') 
#     except Exception as e:
#         return frame_with_aruco, pixels_per_mm, {'failure_reason': 'YOLO Model Call Failed'}

#     if not isinstance(results, list) or len(results) == 0:
#         return frame_with_aruco, pixels_per_mm, {'failure_reason': 'YOLO No Object Detected'}
        
#     res = results[0]
#     bbox_info = get_largest_bbox_from_res(res)
    
#     if bbox_info is None or res.masks is None:
#         debug_reason = 'YOLO Mask Data Missing' if res.masks is None else 'YOLO Boxes Missing'
#         return frame_with_aruco, pixels_per_mm, {'failure_reason': debug_reason}

#     x1,y1,x2,y2, idx = bbox_info
    
#     # 2. Extract Mask
#     try:
#         mask_np = res.masks.data[idx].cpu().numpy().astype(np.uint8)
#         mask_resized = cv2.resize(mask_np, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
#         roi_mask = mask_resized[y1:y2, x1:x2]
        
#     except Exception as e:
#         return frame_with_aruco, pixels_per_mm, {'failure_reason': 'YOLO Mask Processing Failed'}
        
#     x1 = clamp(x1, 0, frame.shape[1]); x2 = clamp(x2, 0, frame.shape[1])
#     y1 = clamp(y1, 0, frame.shape[0]); y2 = clamp(y2, 0, frame.shape[0])
    
#     if x2 <= x1 or y2 <= y1 or roi_mask.size == 0:
#         return frame_with_aruco, pixels_per_mm, {'failure_reason': 'BBox/ROI Invalid Dimensions'}

#     roi = frame[y1:y2, x1:x2].copy()

#     # 3. MASK CLEANUP & ISOLATION
#     combined_mask = roi_mask.copy() 

#     # Morphological operations (Closing then Opening for robust cleanup)
#     kernel = np.ones((5,5), np.uint8) 
#     combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
#     combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
#     mask_bool = (combined_mask > 0).astype(bool)
    
#     out = frame_with_aruco.copy()
#     cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,0), 2) 

#     if np.count_nonzero(mask_bool) < 50: 
#         failure_msg = "Isolation FAILED (Insufficient Pixels after Cleanup)"
#         cv2.putText(out, failure_msg, (x1+5,y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,128,255), 1)
#         return out, pixels_per_mm, {'failure_reason': failure_msg}

#     # 4. SKELETONIZATION & PATH TRACING
#     try:
#         skel = skeletonize(mask_bool).astype(np.uint8)
#     except Exception:
#         skel = np.zeros_like(mask_bool, dtype=np.uint8)
        
#     if np.count_nonzero(skel) < 5:
#         failure_msg = "Isolation FAILED (No Skeleton/Thinning Error)"
#         cv2.putText(out, failure_msg, (x1+5,y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,128,255), 1)
#         return out, pixels_per_mm, {'failure_reason': failure_msg}

#     path, _ = trace_skeleton_path(skel) 
    
#     if len(path) < 5:
#         failure_msg = "Path Tracing Failed (Path too short/discontinuous)"
#         cv2.putText(out, failure_msg, (x1+5,y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,128,255), 1)
#         return out, pixels_per_mm, {'failure_reason': failure_msg}


#     # --- Wire Length (Path Distance) ---
#     length_px = 0.0
#     for i in range(len(path) - 1):
#         r1, c1 = path[i]; r2, c2 = path[i+1]
#         length_px += math.hypot(c2 - c1, r2 - r1)
            
#     length_cm = length_px * mm_per_pixel * MM_TO_CM

#     # --- Overall PCA Tilt ---
#     ys, xs = np.where(mask_bool)
#     angle_deg_pca = 0.0
#     if len(xs) >= 3:
#         coords = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
#         try:
#             mean, eig = cv2.PCACompute(coords, mean=None)
#             principal = eig[0]
#             angle_from_horizontal = math.degrees(math.atan2(float(principal[1]), float(principal[0])))
#             angle_deg_pca = abs(normalize_angle(angle_from_horizontal)) 
#         except Exception:
#             angle_deg_pca = 0.0

#     # --- Max Tilt Segment Analysis ---
#     tilt_seg_deg, tilt_seg_len_cm, tilt_start_img, tilt_end_img = calculate_segment_tilt(
#         path, x1, y1, mm_per_pixel, segment_len_px=70)

#     # --- Special Deviation (Center-to-End Logic) ---
#     cx0 = roi.shape[1] / 2.0 
#     cy0 = roi.shape[0] / 2.0 
#     A_xy = (int(cx0), int(cy0)) # Center of the ROI (relative to ROI)
    
#     # B_xy: Farthest skeleton endpoint from A_xy
#     endpoints_xy = [(c, r) for (r, c) in find_skeleton_endpoints(skel)]
#     if endpoints_xy:
#         B_xy = max(endpoints_xy, key=lambda p: math.hypot(p[0]-A_xy[0], p[1]-A_xy[1]))
#     else:
#         p1 = (path[0][1], path[0][0]) # (c, r)
#         p2 = (path[-1][1], path[-1][0]) # (c, r)
#         B_xy = p1 if math.hypot(p1[0]-A_xy[0], p1[1]-A_xy[1]) > math.hypot(p2[0]-A_xy[0], p2[1]-A_xy[1]) else p2

#     A_img = (int(x1 + A_xy[0]), int(y1 + A_xy[1])) 
#     B_img = (int(x1 + B_xy[0]), int(y1 + B_xy[1])) 

#     ABx = float(B_xy[0] - A_xy[0])
#     ABy = float(B_xy[1] - A_xy[1])
#     a_px = math.hypot(ABx, ABy)
#     a_mm = a_px * mm_per_pixel
    
#     AB_angle_deg = math.degrees(math.atan2(ABy, ABx)) if a_px > 0 else 0.0
#     theta_deg = abs(normalize_angle(AB_angle_deg)) 
#     theta_rad = math.radians(theta_deg)
    
#     # Deviation calculation
#     special_dev_cm = (a_mm * abs(math.sin(theta_rad))) * MM_TO_CM 

#     # Direction calculation
#     vx, vy = ABx, ABy
#     THRESH = 2 
#     if abs(vx) < THRESH: dir_x = ""
#     elif vx > 0: dir_x = "Right"
#     else: dir_x = "Left"

#     if abs(vy) < THRESH: dir_y = ""
#     elif vy > 0: dir_y = "Down"
#     else: dir_y = "Up"

#     if dir_x and dir_y: direction = f"{dir_y}-{dir_x}"
#     elif dir_x: direction = dir_x
#     elif dir_y: direction = dir_y
#     else: direction = "None"

#     measurement = {
#         "length_cm": float(length_cm),
#         "overall_tilt_pca_deg": float(angle_deg_pca),
#         "theta_dev_deg": float(theta_deg),
#         "special_dev_cm": float(special_dev_cm),
#         "direction": direction,
#         "max_segment_tilt_deg": float(tilt_seg_deg),
#         "tilt_segment_length_cm": float(tilt_seg_len_cm), 
#         "pixels_per_mm": float(pixels_per_mm),
#         "failure_reason": None
#     }

#     # --- Draw Overlays ---
#     roi_out = roi.copy()
#     roi_out[combined_mask > 0] = (0, 0, 0) # Black for the isolated wire area
#     roi_out[skel > 0] = (255, 255, 0) # Cyan for the skeleton
#     out[y1:y2, x1:x2] = roi_out 
    
#     # Draw special deviation line (Yellow)
#     cv2.drawMarker(out, A_img, (0,200,0), markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2) # Center A
#     cv2.line(out, A_img, B_img, (0,255,255), 2)
#     cv2.drawMarker(out, B_img, (0,0,255), markerType=cv2.MARKER_DIAMOND, markerSize=8, thickness=2) # End B

#     # Draw MOST TILTED SEGMENT (MAGENTA)
#     if tilt_start_img and tilt_end_img:
#         cv2.line(out, tilt_start_img, tilt_end_img, (255,0,255), 5)
#         cv2.circle(out, tilt_start_img, 6, (255,0,255), -1)
#         cv2.circle(out, tilt_end_img, 6, (255,0,255), -1)
        
#         mid_x = (tilt_start_img[0] + tilt_end_img[0]) // 2
#         mid_y = (tilt_start_img[1] + tilt_end_img[1]) // 2
#         cv2.putText(out, f"MAX TILT {tilt_seg_deg:.2f} deg", (mid_x - 50, mid_y - 15), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,255), 1)

#     # Draw Text Overlays
#     text_main = f"Len:{length_cm:.3f}cm | PCA Tilt:{angle_deg_pca:.2f}deg | Dev:{special_dev_cm:.4f}cm | Dir:{direction}"
#     cv2.putText(out, text_main, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)
    
#     text_tilt_seg = f"Max Seg Tilt:{tilt_seg_deg:.2f}deg | Seg Len:{tilt_seg_len_cm:.3f}cm" 
#     cv2.putText(out, text_tilt_seg, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2, cv2.LINE_AA)
    
#     cv2.putText(out, f"pix/mm:{pixels_per_mm:.3f}", (10, out.shape[0]-12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

#     return out, pixels_per_mm, measurement

# # -----------------------------
# # Main Batch Processing 📦
# # -----------------------------
# def analyze_all():
#     """Batch processes images and generates reports."""
    
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     result_folder_path = os.path.join(RESULTS_DIR, timestamp)
#     os.makedirs(result_folder_path, exist_ok=True)
    
#     image_paths = glob.glob(os.path.join(UPLOAD_FOLDER, '*'))
#     image_files = [f for f in image_paths if f.lower().endswith(IMAGE_EXTS)]
    
#     if not image_files:
#         print(f"❌ No images found in the '{UPLOAD_FOLDER}' folder.")
#         with open(os.path.join(result_folder_path, "average_summary.txt"), 'w') as f:
#             f.write("Batch Analysis Failed: No input images found.\n")
#         return

#     print(f"✅ Found {len(image_files)} images to process. Results saved to: {result_folder_path}")
    
#     pixels_per_mm = FALLBACK_PIXELS_PER_MM 
#     all_measurements = []

#     for image_path in image_files:
#         filename = os.path.basename(image_path)
#         print(f"-> Processing {filename}...")
        
#         try:
#             frame = cv2.imread(image_path)
#             if frame is None:
#                 print(f"WARNING: Could not read image {filename}. Skipping.")
#                 all_measurements.append({'filename': filename, 'failure_reason': 'Could not read image file.'})
#                 continue

#             out_frame, current_scale, measurement = analyze_frame(frame, pixels_per_mm)
            
#             if current_scale != FALLBACK_PIXELS_PER_MM:
#                 pixels_per_mm = current_scale
            
#             measurement['filename'] = filename
#             all_measurements.append(measurement)

#             out_path = os.path.join(result_folder_path, f"{os.path.splitext(filename)[0]}_result.png")
#             cv2.imwrite(out_path, out_frame)
            
#             generate_summary_file(filename, measurement, result_folder_path)
            
#             print(f"  -> Analysis complete. Dev: {measurement.get('special_dev_cm', 'N/A')} cm")

#         except Exception as e:
#             print(f"❌ CRITICAL ERROR processing {filename}: {e}")
#             traceback.print_exc()
#             all_measurements.append({'filename': filename, 'failure_reason': f'CRITICAL EXCEPTION: {e}'})

#     # Generate Batch Reports
#     create_average_summary(all_measurements, result_folder_path)
#     create_csv_report(all_measurements, result_folder_path)

# if __name__ == "__main__":
#     analyze_all()













# deviation_check.py

import cv2
import numpy as np
import math
import os
import time
import csv
from datetime import datetime
from ultralytics import YOLO
from skimage.morphology import skeletonize

# -----------------------------
# CONFIG
# -----------------------------
# 🎯 CHANGE APPLIED HERE: Using your specified absolute path for the YOLO model.
MODEL_PATH = "working_copper_model.pt" 
UPLOAD_FOLDER = "uploads"
RESULTS_DIR = "results"

FALLBACK_PIXELS_PER_MM = 9.727
MM_TO_CM = 0.1

# ArUco settings
ARUCO_SIDE_MM = 35.0
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()
ARUCO_DETECTOR = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)

# Supported image extensions
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

# -----------------------------
# Load YOLO Model
# -----------------------------
model = None
try:
    print(f"Loading YOLO model from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("YOLO loaded successfully.")
except Exception as e:
    print(f"Error loading YOLO model from '{MODEL_PATH}': {e}")
    # Set model to None if loading fails so subsequent functions can check
    model = None

# -----------------------------
# Utilities (Unchanged)
# -----------------------------
def clamp(v, a, b):
    return max(a, min(b, v))

def normalize_angle(angle):
    """Constrains angle to (-90, 90]."""
    while angle > 180: angle -= 360
    while angle <= -180: angle += 360
    if angle > 90: angle -= 180
    if angle < -90: angle += 180
    return angle

# -----------------------------
# Skeleton helpers (Unchanged)
# -----------------------------
def find_skeleton_endpoints(skel):
    endpoints = []
    h, w = skel.shape
    for r in range(h):
        for c in range(w):
            if not skel[r, c]:
                continue
            r0 = max(0, r-1); r1 = min(h-1, r+1)
            c0 = max(0, c-1); c1 = min(w-1, c+1)
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
    if not coords:
        return [], 0.0
    # endpoints
    endpoints = []
    for (r, c) in coords:
        cnt = 0
        for dr in (-1,0,1):
            for dc in (-1,0,1):
                if dr == 0 and dc == 0: continue
                if (r+dr, c+dc) in coords:
                    cnt += 1
        if cnt == 1:
            endpoints.append((r,c))
    start = endpoints[0] if endpoints else next(iter(coords))
    visited = {start}
    path = [start]
    cur = start
    while True:
        r, c = cur
        neighbors = []
        for dr in (-1,0,1):
            for dc in (-1,0,1):
                if dr == 0 and dc == 0: continue
                n = (r+dr, c+dc)
                if n in coords and n not in visited:
                    neighbors.append(n)
        if not neighbors:
            found = False
            for node in reversed(path):
                rr, cc = node
                for dr in (-1,0,1):
                    for dc in (-1,0,1):
                        if dr == 0 and dc == 0: continue
                        n = (rr+dr, cc+dc)
                        if n in coords and n not in visited:
                            cur = node
                            found = True
                            break
                    if found: break
                if found: break
            if not found:
                break
            else:
                continue
        neighbors_sorted = sorted(neighbors, key=lambda n: (abs(n[0]-r)+abs(n[1]-c), math.hypot(n[0]-r, n[1]-c)))
        nxt = neighbors_sorted[0]
        path.append(nxt)
        visited.add(nxt)
        cur = nxt
    total_px = 0.0
    for i in range(1, len(path)):
        r1,c1 = path[i-1]; r2,c2 = path[i]
        total_px += math.hypot(c2-c1, r2-r1)
    return path, total_px

# -----------------------------
# ArUco detection & calibration (Unchanged)
# -----------------------------
def detect_aruco_and_pixels_per_mm(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = ARUCO_DETECTOR.detectMarkers(gray)
    vis = frame.copy()
    if ids is None or len(corners) == 0:
        return None, vis
    best_idx = 0
    best_area = 0.0
    for i, c in enumerate(corners):
        pts = c.reshape(-1,2).astype(np.float32)
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
    if avg_side_px <= 0:
        return None, vis
    pixels_per_mm = avg_side_px / ARUCO_SIDE_MM
    try:
        id_val = int(ids[best_idx][0]) if ids is not None else -1
        cv2.putText(vis, f"ArUco ID:{id_val}", (int(best_c[0][0]), int(best_c[0][1]-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    except Exception:
        pass
    return pixels_per_mm, vis

# -----------------------------
# YOLO bbox helper (Unchanged)
# -----------------------------
def get_largest_bbox_from_res(res):
    boxes = getattr(res, "boxes", None)
    if boxes is None:
        return None
    try:
        xyxy = boxes.xyxy.cpu().numpy()
    except Exception:
        try:
            xyxy = np.array([b.xyxy[0].cpu().numpy() for b in boxes])
        except Exception:
            return None
    if xyxy.size == 0:
        return None
    areas = (xyxy[:,2] - xyxy[:,0]) * (xyxy[:,3] - xyxy[:,1])
    idx = int(np.argmax(areas))
    x1,y1,x2,y2 = xyxy[idx].astype(int).tolist()
    return x1,y1,x2,y2

# -----------------------------
# Tilt analysis function (Unchanged)
# -----------------------------
def calculate_segment_tilt(path, x_offset, y_offset, mm_per_pixel, segment_len_px=70):
    """
    Analyzes segments of the wire path to find the segment with the largest tilt.
    """
    max_tilt_deg = 0.0
    best_start_img, best_end_img = None, None

    if len(path) < segment_len_px:
        # Not enough points for segmentation, use endpoints of the full path
        if len(path) >= 2:
            r1, c1 = path[0]; r2, c2 = path[-1]
            p1_img = (int(x_offset + c1), int(y_offset + r1))
            p2_img = (int(x_offset + c2), int(y_offset + r2))
            
            dx = float(c2 - c1)
            dy = float(r2 - r1)
            angle_rad = math.atan2(dy, dx)
            angle_deg = normalize_angle(math.degrees(angle_rad))
            max_tilt_deg = abs(angle_deg) 
            best_start_img, best_end_img = p1_img, p2_img

    else:
        # Iterate over segments
        for i in range(0, len(path) - segment_len_px, segment_len_px // 2):
            start_point = path[i]
            end_point = path[i + segment_len_px]
            
            r1, c1 = start_point
            r2, c2 = end_point
            
            dx = float(c2 - c1)
            dy = float(r2 - r1)
            
            if math.hypot(dx, dy) < 5: continue # Skip very short/coincident segments
            
            angle_rad = math.atan2(dy, dx)
            angle_deg = normalize_angle(math.degrees(angle_rad))
            
            current_tilt_deg = abs(angle_deg) 
            
            if current_tilt_deg > max_tilt_deg:
                max_tilt_deg = current_tilt_deg
                best_start_img = (int(x_offset + c1), int(y_offset + r1))
                best_end_img = (int(x_offset + c2), int(y_offset + r2))

    if best_start_img and best_end_img:
        dx = best_end_img[0] - best_start_img[0]
        dy = best_end_img[1] - best_start_img[1]
        length_px = math.hypot(dx, dy)
        length_cm = length_px * mm_per_pixel * MM_TO_CM
        
        return max_tilt_deg, length_cm, best_start_img, best_end_img
    
    return 0.0, 0.0, None, None


# -----------------------------
# Analyze single frame
# -----------------------------
def analyze_frame(frame, last_pixels_per_mm=FALLBACK_PIXELS_PER_MM):
    """Processes a single image frame for wire measurements and annotations."""
    if model is None:
        # Fallback if model failed to load
        return frame.copy(), last_pixels_per_mm, None
        
    pixels_per_mm_frame, frame_with_aruco = detect_aruco_and_pixels_per_mm(frame)
    pixels_per_mm = pixels_per_mm_frame if pixels_per_mm_frame is not None else last_pixels_per_mm
    mm_per_pixel = 1.0 / pixels_per_mm

    # YOLO detection
    results = model(frame, verbose=False)
    if len(results) == 0:
        return frame_with_aruco, pixels_per_mm, None
    res = results[0]
    bbox = get_largest_bbox_from_res(res)
    if bbox is None:
        return frame_with_aruco, pixels_per_mm, None

    x1,y1,x2,y2 = bbox
    x1 = clamp(x1, 0, frame.shape[1]-1); x2 = clamp(x2, 0, frame.shape[1]-1)
    y1 = clamp(y1, 0, frame.shape[0]-1); y2 = clamp(y2, 0, frame.shape[0]-1)
    roi = frame[y1:y2, x1:x2].copy()
    if roi.size == 0:
        return frame_with_aruco, pixels_per_mm, None

    # Image Processing (Grayscale, Thresholding, Skeletonization)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)
    gray_blur = cv2.GaussianBlur(gray_eq, (5,5), 0)
    _, thr = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask_bool = (thr > 0)
    if np.count_nonzero(mask_bool) < 10:
        mask_bool = (~mask_bool) # Invert mask if object is dark on light background
    
    mask_bool = mask_bool.astype(bool)
    try:
        skel = skeletonize(mask_bool).astype(np.uint8)
    except Exception:
        skel = np.zeros_like(mask_bool, dtype=np.uint8)
        
    if np.count_nonzero(skel) == 0:
        out = frame_with_aruco.copy()
        cv2.rectangle(out, (x1,y1), (x2,y2), (0,128,255), 2)
        cv2.putText(out, "No skeleton/wire found", (x1+5,y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,128,255), 2)
        return out, pixels_per_mm, None

    # Wire Length and Path
    path, length_px = trace_skeleton_path(skel)
    if length_px <= 0:
        length_px = float(np.count_nonzero(skel))
    length_cm = length_px * mm_per_pixel * MM_TO_CM

    # PCA Tilt 
    ys, xs = np.where(mask_bool)
    angle_deg_pca = 0.0
    if len(xs) >= 3:
        coords = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
        try:
            mean, eig = cv2.PCACompute(coords, mean=None)
            principal = eig[0]
            angle_deg_pca = normalize_angle(math.degrees(math.atan2(float(principal[1]), float(principal[0]))))
        except Exception:
            angle_deg_pca = 0.0

    # New: Tilt Segment Analysis
    tilt_seg_deg, tilt_seg_len_cm, tilt_start_img, tilt_end_img = calculate_segment_tilt(
        path, x1, y1, mm_per_pixel, segment_len_px=70)

    # Special Deviation (A to B logic)
    cx0 = roi.shape[1] / 2.0
    cy0 = roi.shape[0] / 2.0
    A_xy = (int(cx0), int(cy0))
    endpoints_xy = [(c, r) for (r, c) in find_skeleton_endpoints(skel)]
    if endpoints_xy:
        B_xy = max(endpoints_xy, key=lambda p: math.hypot(p[0]-A_xy[0], p[1]-A_xy[1]))
    elif len(path) >= 2:
        r1,c1 = path[0]; r2,c2 = path[-1]
        p1 = (c1, r1); p2 = (c2, r2)
        B_xy = p1 if math.hypot(p1[0]-A_xy[0], p1[1]-A_xy[1]) > math.hypot(p2[0]-A_xy[0], p2[1]-A_xy[1]) else p2
    else:
        B_xy = (A_xy[0] + 40, A_xy[1])

    A_img = (int(x1 + A_xy[0]), int(y1 + A_xy[1]))
    B_img = (int(x1 + B_xy[0]), int(y1 + B_xy[1]))

    a_px = math.hypot(B_xy[0] - A_xy[0], B_xy[1] - A_xy[1])
    a_mm = a_px * mm_per_pixel
    ABx = float(B_xy[0] - A_xy[0])
    ABy = float(B_xy[1] - A_xy[1])
    AB_angle_deg = math.degrees(math.atan2(ABy, ABx)) if a_px > 0 else 0.0
    theta_deg = abs(normalize_angle(AB_angle_deg - 90.0))
    theta_rad = math.radians(theta_deg)
    special_dev_cm = math.sqrt(max(0.0, 2.0 * (a_mm ** 2) * (1.0 - math.cos(theta_rad)))) * MM_TO_CM

    # Correct direction calculation
    vx, vy = ABx, ABy
    THRESH = 2
    if abs(vx) < THRESH: dir_x = ""
    elif vx > 0: dir_x = "Right"
    else: dir_x = "Left"

    if abs(vy) < THRESH: dir_y = ""
    elif vy > 0: dir_y = "Down"
    else: dir_y = "Up"

    if dir_x and dir_y: direction = f"{dir_y}-{dir_x}"
    elif dir_x: direction = dir_x
    elif dir_y: direction = dir_y
    else: direction = "None"

    measurement = {
        "filename": None, 
        "length_cm": float(length_cm),
        "overall_tilt_pca_deg": float(angle_deg_pca),
        "theta_dev_deg": float(theta_deg),
        "special_dev_cm": float(special_dev_cm),
        "direction": direction,
        "max_segment_tilt_deg": float(tilt_seg_deg),
        "max_tilt_segment_len_cm": float(tilt_seg_len_cm),
        "pixels_per_mm": float(pixels_per_mm)
    }

    # Draw overlays
    out = frame_with_aruco.copy()
    cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,0), 2)
    
    # Draw skeleton
    try:
        sk_vis = (skel * 255).astype(np.uint8)
        sk_col = cv2.cvtColor(sk_vis, cv2.COLOR_GRAY2BGR)
        out[y1:y2, x1:x2] = cv2.addWeighted(out[y1:y2, x1:x2], 0.7, sk_col, 0.45, 0)
    except Exception:
        pass
        
    # Draw special deviation line (A to B)
    cv2.drawMarker(out, A_img, (0,200,0), markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2)
    cv2.circle(out, B_img, 6, (0,0,255), -1)
    cv2.line(out, A_img, B_img, (0,255,255), 2)

    # Draw most tilted segment (Blue)
    if tilt_start_img and tilt_end_img:
        cv2.line(out, tilt_start_img, tilt_end_img, (255,0,0), 3) 
        cv2.circle(out, tilt_start_img, 4, (255,0,0), -1)
        cv2.circle(out, tilt_end_img, 4, (255,0,0), -1)

    # Draw Text
    text_main = f"Len:{length_cm:.3f}cm  PCA Tilt:{angle_deg_pca:.2f}deg  Dev:{special_dev_cm:.4f}cm  Dir:{direction}"
    cv2.putText(out, text_main, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)
    
    text_tilt_seg = f"Max Seg Tilt:{tilt_seg_deg:.2f}deg  Seg Len:{tilt_seg_len_cm:.3f}cm"
    cv2.putText(out, text_tilt_seg, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2, cv2.LINE_AA)
    
    cv2.putText(out, f"pix/mm:{pixels_per_mm:.3f}", (10, out.shape[0]-12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

    return out, pixels_per_mm, measurement

# -----------------------------
# Main batch processing function for Flask server (Unchanged)
# -----------------------------
def analyze_all(upload_folder=UPLOAD_FOLDER, results_dir=RESULTS_DIR):
    """
    Analyzes all images in the upload folder, saves results to a new timestamped folder,
    and returns the path to the new results folder.
    """
    image_files = [f for f in os.listdir(upload_folder) if f.lower().endswith(IMAGE_EXTS)]
    
    if not image_files:
        raise FileNotFoundError(f"No images found in {upload_folder}.")
    
    # 1. Create a timestamped results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_results_dir = os.path.join(results_dir, timestamp)
    os.makedirs(current_results_dir, exist_ok=True)
    print(f"Saving results to: {current_results_dir}")
    
    pixels_per_mm = FALLBACK_PIXELS_PER_MM # Start with fallback
    all_measurements = []
    
    # 2. Process images
    for filename in image_files:
        print(f"Processing: {filename}")
        img_path = os.path.join(upload_folder, filename)
        frame = cv2.imread(img_path)
        
        if frame is None:
            print(f"Could not read image: {filename}. Skipping.")
            continue
            
        try:
            out, used_pix_per_mm, meas = analyze_frame(frame, pixels_per_mm)
            
            if abs(used_pix_per_mm - FALLBACK_PIXELS_PER_MM) > 1e-6:
                pixels_per_mm = used_pix_per_mm # Update calibration
            
            # Save annotated image
            base_name, ext = os.path.splitext(filename)
            result_path = os.path.join(current_results_dir, f"{base_name}_result{ext}")
            cv2.imwrite(result_path, out)
            
            if meas:
                meas["filename"] = filename
                all_measurements.append(meas)

        except Exception as e:
            print(f"An error occurred while processing {filename}: {e}")

    if not all_measurements:
        raise Exception("No successful wire measurements were recorded.")

    # 3. Calculate Averages
    avg_meas = {}
    total_count = len(all_measurements)
    if total_count > 0:
        for key in all_measurements[0].keys():
            if key in ["filename", "direction"]:
                continue
            values = [m[key] for m in all_measurements]
            # Ensure proper handling for list of values
            if values and isinstance(values[0], (int, float)):
                avg_meas[f"average_{key}"] = float(np.mean(values))
            
    # 4. Save CSV
    csv_path = os.path.join(current_results_dir, "wire_analysis_results.csv")
    fieldnames = list(all_measurements[0].keys()) if all_measurements else []
    
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_measurements)
    print(f"CSV saved to: {csv_path}")

    # 5. Save Summary Text
    summary_path = os.path.join(current_results_dir, "average_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Total Images Analyzed: {total_count}\n")
        f.write(f"Pixels Per MM (Final): {pixels_per_mm:.3f}\n")
        f.write("---\n")
        for k, v in avg_meas.items():
            unit = "cm" if "cm" in k else "deg" if "deg" in k else ""
            # Format the output for better readability/parsing
            f.write(f"{k.replace('average_', '').replace('_', ' ').title()}: {v:.4f} {unit}\n")
    print(f"Summary saved to: {summary_path}")

    # The Flask server relies on this to determine the result folder ID
    return current_results_dir 

if __name__ == "__main__":
    # Example usage for testing the module directly
    print("--- Running standalone deviation check ---")
    try:
        # NOTE: For standalone test, ensure you have an 'uploads' directory with images.
        analyze_all()
    except Exception as e:
        print(f"Error in standalone run: {e}")