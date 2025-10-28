import cv2
import numpy as np
import os
import csv
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
REFERENCE_IMAGE_PATH = os.path.join(BASE_DIR, "wire_reference.png")
AR_MODEL_PATH = os.path.join(BASE_DIR, "images", "ar_model.png")  # Optional

REFERENCE_LENGTH_MM = 1.20  # Actual wire length in mm
COLOR_LOWER = np.array([5, 80, 50])   # HSV lower bound (tweak for copper color)
COLOR_UPPER = np.array([25, 255, 255]) # HSV upper bound
os.makedirs(RESULTS_DIR, exist_ok=True)

PIXELS_PER_MM = None  # Global calibration ratio


# ============================================================
# CALIBRATION FUNCTION
# ============================================================
def calibrate_reference():
    """
    Calibrate using the known reference wire image (wire_reference.png)
    """
    global PIXELS_PER_MM

    img = cv2.imread(REFERENCE_IMAGE_PATH)
    if img is None:
        print("[❌] Missing wire_reference.png")
        return False

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, COLOR_LOWER, COLOR_UPPER)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("[❌] No contour found in reference image.")
        return False

    c = max(contours, key=cv2.contourArea)
    _, (w, h), _ = cv2.minAreaRect(c)
    PIXELS_PER_MM = max(w, h) / REFERENCE_LENGTH_MM

    print(f"[✅] Calibrated successfully: 1 mm = {PIXELS_PER_MM:.3f} pixels")
    print(f"      (1 px = {1 / PIXELS_PER_MM:.6f} mm)")
    return True


# ============================================================
# SINGLE IMAGE ANALYSIS
# ============================================================
def analyze_wire_image(path):
    """
    Analyze one uploaded image for wire length, orientation, and tilt angle
    """
    img = cv2.imread(path)
    if img is None:
        print(f"[❌] Cannot read {path}")
        return None

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, COLOR_LOWER, COLOR_UPPER)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print(f"[⚠️] No wire detected in {os.path.basename(path)}")
        return None

    # Largest contour = wire
    c = max(contours, key=cv2.contourArea)
    (cx, cy), (w, h), angle = cv2.minAreaRect(c)

    # Convert pixels to mm
    length_mm = max(w, h) / PIXELS_PER_MM
    diameter_mm = min(w, h) / PIXELS_PER_MM
    deviation = abs(length_mm - REFERENCE_LENGTH_MM)
    orientation = "Vertical" if 45 < abs(angle) < 135 else "Horizontal"
    direction = "Tilted Left" if angle < 0 else "Tilted Right"

    # Draw bounding box
    annotated = img.copy()
    box = np.intp(cv2.boxPoints(cv2.minAreaRect(c)))
    cv2.drawContours(annotated, [box], 0, (0, 255, 0), 2)

    # Add text annotation
    info = f"Len: {length_mm:.2f}mm | Dev: {deviation:.3f}mm | {orientation} | {direction} ({angle:.2f}°)"
    cv2.putText(annotated, info, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Save annotated image
    out_path = os.path.join(RESULTS_DIR, f"annotated_{os.path.basename(path)}")
    cv2.imwrite(out_path, annotated)

    print(f"[INFO] {os.path.basename(path)} | Length: {length_mm:.3f} mm | "
          f"Deviation: {deviation:.3f} mm | Angle: {angle:.2f}° | {orientation}")

    return {
        "file": os.path.basename(path),
        "length_mm": round(length_mm, 3),
        "diameter_mm": round(diameter_mm, 3),
        "angle_deg": round(angle, 2),
        "orientation": orientation,
        "direction": direction,
        "deviation_mm": round(deviation, 3)
    }


# ============================================================
# MAIN ANALYSIS LOOP
# ============================================================
def analyze_all():
    """
    Run full analysis on all uploaded images in /uploads
    """
    if not calibrate_reference():
        print("[❌] Calibration failed. Exiting.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = os.path.join(RESULTS_DIR, timestamp)
    os.makedirs(folder, exist_ok=True)

    files = sorted([f for f in os.listdir(UPLOAD_FOLDER) if f.endswith(".png")])
    results = []

    for f in files:
        file_path = os.path.join(UPLOAD_FOLDER, f)
        r = analyze_wire_image(file_path)
        if r:
            results.append(r)

    if not results:
        print("[❌] No valid images processed.")
        return

    # Save CSV report
    csv_path = os.path.join(folder, "wire_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"✅ Analysis complete! Results saved in: {folder}")
    print(f"📊 CSV: {csv_path}")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    analyze_all()



# import cv2
# import numpy as np
# import os
# import csv
# from datetime import datetime
# from scipy.spatial.distance import euclidean

# # ============================================================
# # CONFIGURATION
# # ============================================================
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# REFERENCE_LENGTH_MM = 1.20  # actual known wire length in millimeters

# REFERENCE_IMAGE_PATH = os.path.join(BASE_DIR, "wire_reference.png")
# AR_MODEL_PATH = os.path.join(BASE_DIR, "images", "ar_model.png")
# UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
# RESULTS_DIR = os.path.join(BASE_DIR, "results")

# COLOR_LOWER = np.array([5, 80, 50])      # HSV lower bound for copper wire
# COLOR_UPPER = np.array([25, 255, 255])   # HSV upper bound for copper wire

# os.makedirs(RESULTS_DIR, exist_ok=True)
# PIXELS_PER_MM = None


# # ============================================================
# # CALIBRATION using reference wire image
# # ============================================================
# def calibrate_reference():
#     """Calibrate pixels per millimeter using the reference image."""
#     global PIXELS_PER_MM
#     img = cv2.imread(REFERENCE_IMAGE_PATH)
#     if img is None:
#         print(f"[ERROR] Could not read reference image: {REFERENCE_IMAGE_PATH}")
#         return False

#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     mask = cv2.inRange(hsv, COLOR_LOWER, COLOR_UPPER)
#     mask = cv2.medianBlur(mask, 5)

#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if not contours:
#         print("[ERROR] No wire detected in reference image.")
#         return False

#     c = max(contours, key=cv2.contourArea)
#     rect = cv2.minAreaRect(c)
#     (cx, cy), (w, h), angle = rect
#     wire_length_px = max(w, h)

#     PIXELS_PER_MM = wire_length_px / REFERENCE_LENGTH_MM
#     print(f"[INFO] Calibration complete: 1 pixel = {1/PIXELS_PER_MM:.6f} mm")
#     return True


# # ============================================================
# # ANALYZE SINGLE WIRE IMAGE
# # ============================================================
# def analyze_wire_image(image_path):
#     """Analyze a single wire image and return all geometric data."""
#     global PIXELS_PER_MM
#     img = cv2.imread(image_path)
#     if img is None:
#         print(f"[ERROR] Could not read {image_path}")
#         return None

#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     mask = cv2.inRange(hsv, COLOR_LOWER, COLOR_UPPER)
#     mask = cv2.medianBlur(mask, 5)
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     if not contours:
#         print(f"[WARN] No wire detected in {os.path.basename(image_path)}")
#         return None

#     c = max(contours, key=cv2.contourArea)
#     rect = cv2.minAreaRect(c)
#     box = np.intp(cv2.boxPoints(rect))
#     (cx, cy), (w, h), angle = rect

#     # Measurements
#     wire_length_px = max(w, h)
#     wire_diameter_px = min(w, h)
#     wire_length_mm = wire_length_px / PIXELS_PER_MM
#     wire_diameter_mm = wire_diameter_px / PIXELS_PER_MM
#     deviation = abs(wire_length_mm - REFERENCE_LENGTH_MM)

#     # Orientation and direction
#     orientation = "Vertical" if 45 < abs(angle) < 135 else "Horizontal"
#     direction = "Tilted Left" if angle < 0 else "Tilted Right"

#     # Bend detection (approximation)
#     epsilon = 0.01 * cv2.arcLength(c, True)
#     approx = cv2.approxPolyDP(c, epsilon, True)
#     bends = len(approx)

#     # Draw results on image
#     annotated = img.copy()
#     cv2.drawContours(annotated, [box], 0, (0, 255, 0), 2)
#     text_lines = [
#         f"Length: {wire_length_mm:.3f} mm",
#         f"Diameter: {wire_diameter_mm:.3f} mm",
#         f"Tilt: {angle:.2f}° ({direction})",
#         f"Orientation: {orientation}",
#         f"Deviation: {deviation:.3f} mm",
#         f"Bends: {bends}"
#     ]
#     y = 30
#     for line in text_lines:
#         cv2.putText(annotated, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
#         y += 25

#     annotated_path = os.path.join(RESULTS_DIR, f"annotated_{os.path.basename(image_path)}")
#     cv2.imwrite(annotated_path, annotated)

#     print(f"[INFO] {os.path.basename(image_path)} → "
#           f"Length: {wire_length_mm:.3f} mm | Angle: {angle:.2f}° | Orientation: {orientation}")

#     return {
#         "file": os.path.basename(image_path),
#         "length_mm": wire_length_mm,
#         "diameter_mm": wire_diameter_mm,
#         "angle_deg": angle,
#         "orientation": orientation,
#         "direction": direction,
#         "deviation_mm": deviation,
#         "bends": bends
#     }


# # ============================================================
# # ANALYZE AR MODEL (OPTIONAL)
# # ============================================================
# def analyze_ar_model():
#     """Analyze AR model wire if available."""
#     img = cv2.imread(AR_MODEL_PATH)
#     if img is None:
#         print("[WARN] AR model not found.")
#         return None

#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     mask = cv2.inRange(hsv, COLOR_LOWER, COLOR_UPPER)
#     mask = cv2.medianBlur(mask, 5)
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     if not contours:
#         print("[WARN] No wire detected in AR model.")
#         return None

#     c = max(contours, key=cv2.contourArea)
#     _, (w, h), _ = cv2.minAreaRect(c)
#     ar_length_px = max(w, h)
#     ar_length_mm = ar_length_px / PIXELS_PER_MM
#     return ar_length_mm


# # ============================================================
# # MAIN ANALYSIS FUNCTION
# # ============================================================
# def analyze_all():
#     """Run full analysis, save CSV and text report."""
#     print("🔍 Starting Wire and AR Model Analysis...\n")

#     if not calibrate_reference():
#         print("[ERROR] Calibration failed.")
#         return

#     files = [f for f in os.listdir(UPLOAD_FOLDER)
#              if f.lower().endswith((".png", ".jpg", ".jpeg"))]
#     if not files:
#         print("[ERROR] No images found in uploads folder.")
#         return

#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     result_folder = os.path.join(RESULTS_DIR, timestamp)
#     os.makedirs(result_folder, exist_ok=True)

#     results = []
#     for f in sorted(files):
#         path = os.path.join(UPLOAD_FOLDER, f)
#         result = analyze_wire_image(path)
#         if result:
#             results.append(result)

#     if not results:
#         print("[ERROR] No valid wire data extracted.")
#         return

#     ar_length = analyze_ar_model()
#     if ar_length is None:
#         print("[WARN] AR model analysis failed. Using reference length as default.")
#         ar_length = REFERENCE_LENGTH_MM

#     # Compute deviation from AR model
#     for r in results:
#         deviation_ar = abs(r["length_mm"] - ar_length)
#         r["deviation_from_AR_mm"] = deviation_ar

#     # === Compute Mean Summary ===
#     mean_length = np.mean([r["length_mm"] for r in results])
#     mean_diameter = np.mean([r["diameter_mm"] for r in results])
#     mean_angle = np.mean([abs(r["angle_deg"]) for r in results])
#     mean_dev_ar = np.mean([r["deviation_from_AR_mm"] for r in results])

#     summary_row = {
#         "file": "MEAN_RESULT",
#         "length_mm": round(mean_length, 3),
#         "diameter_mm": round(mean_diameter, 3),
#         "angle_deg": round(mean_angle, 2),
#         "orientation": "—",
#         "direction": "—",
#         "deviation_mm": "—",
#         "bends": "—",
#         "deviation_from_AR_mm": round(mean_dev_ar, 3)
#     }
#     results.append(summary_row)

#     # Save CSV
#     csv_path = os.path.join(result_folder, "wire_results.csv")
#     with open(csv_path, "w", newline="") as f:
#         writer = csv.DictWriter(f, fieldnames=results[0].keys())
#         writer.writeheader()
#         writer.writerows(results)

#     # Save text report
#     txt_path = os.path.join(result_folder, "wire_results.txt")
#     with open(txt_path, "w") as f:
#         f.write("=== WIRE LENGTH AND ORIENTATION ANALYSIS REPORT ===\n\n")
#         f.write(f"Analysis Timestamp: {datetime.now()}\n")
#         f.write(f"Total Images Analyzed: {len(results)-1}\n\n")
#         f.write(f"Reference AR Model Length: {ar_length:.3f} mm\n\n")

#         for r in results[:-1]:
#             f.write(f"📸 Image File: {r['file']}\n")
#             f.write(f"→ Length: {r['length_mm']:.3f} mm\n")
#             f.write(f"→ Diameter: {r['diameter_mm']:.3f} mm\n")
#             f.write(f"→ Tilt Angle: {r['angle_deg']:.2f}° ({r['direction']})\n")
#             f.write(f"→ Orientation: {r['orientation']}\n")
#             f.write(f"→ Deviation from AR Model: {r['deviation_from_AR_mm']:.3f} mm\n")
#             f.write(f"→ Bends Detected: {r['bends']}\n\n")

#         f.write("\n=== OVERALL MEAN SUMMARY ===\n")
#         f.write(f"Average Wire Length: {mean_length:.3f} mm\n")
#         f.write(f"Average Diameter: {mean_diameter:.3f} mm\n")
#         f.write(f"Average Tilt Angle: {mean_angle:.2f}°\n")
#         f.write(f"Average Deviation from AR Model: {mean_dev_ar:.3f} mm\n")

#     print(f"\n✅ Analysis complete! Results saved in: {result_folder}")
#     print(f"📊 CSV Report: {csv_path}")
#     print(f"📝 Text Report: {txt_path}")


# # ============================================================
# # MAIN
# # ============================================================
# if __name__ == "__main__":
#     analyze_all()
