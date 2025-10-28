

import cv2
import numpy as np
import os
import math
import csv
from datetime import datetime

# ================================================================
# CONFIGURATION
# ================================================================
REFERENCE_LENGTH_MM = 1.20  # Known wire length (mm)
UPLOAD_FOLDER = "uploads"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)


# ================================================================
# SINGLE IMAGE ANALYSIS
# ================================================================
def analyze_single_image(image_path, results_dir, return_json=True):
    """Analyze a single wire image for length, angle, direction, and deviation."""

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"❌ Invalid image: {image_path}")

    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("⚠️ No wire detected")

    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    wire_length_px = max(w, h)

    # Fit line to find angle
    [vx, vy, x0, y0] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
    angle_rad = math.atan2(vy, vx)
    angle_deg = math.degrees(angle_rad)
    if angle_deg < 0:
        angle_deg += 180

    # Direction logic
    if 85 <= angle_deg <= 95:
        direction = "Vertical"
    elif angle_deg < 85:
        direction = f"Tilted Right ({round(angle_deg, 1)}°)"
    else:
        direction = f"Tilted Left ({round(180 - angle_deg, 1)}°)"

    # Wire deviation
    line_pts = cnt.reshape(-1, 2)
    distances = [abs(vy * px - vx * py + x0 * vy - y0 * vx) for (px, py) in line_pts]
    deviation_px = np.std(distances)
    deviation_mm = round(deviation_px / (wire_length_px / REFERENCE_LENGTH_MM), 4)

    # Convert to mm
    pixels_per_mm = wire_length_px / REFERENCE_LENGTH_MM
    wire_length_mm = round(wire_length_px / pixels_per_mm, 3)
    wire_diameter_mm = round(wire_length_mm * 0.22, 3)  # simple estimation

    # Annotate
    annotated = original.copy()
    cv2.drawContours(annotated, [cnt], -1, (0, 255, 0), 2)
    cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 0, 0), 2)
    height, width = annotated.shape[:2]
    lefty = int((-x0 * vy / vx) + y0)
    righty = int(((width - x0) * vy / vx) + y0)
    cv2.line(annotated, (width - 1, righty), (0, lefty), (0, 0, 255), 2)

    # Overlay text
    cv2.putText(annotated, f"Angle: {round(angle_deg, 2)}°", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(annotated, direction, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(annotated, f"Deviation: {deviation_mm} mm", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(annotated, f"Length: {wire_length_mm} mm", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    out_name = f"analyzed_{os.path.basename(image_path)}"
    out_path = os.path.join(results_dir, out_name)
    cv2.imwrite(out_path, annotated)

    result = {
        "filename": os.path.basename(image_path),
        "length_mm": wire_length_mm,
        "diameter_mm": wire_diameter_mm,
        "angle_deg": round(angle_deg, 2),
        "direction": direction,
        "deviation_mm": deviation_mm,
        "annotated_image": f"/results/{out_name}"
    }

    print(f"✅ Analysis for {os.path.basename(image_path)}: {result}")
    return result if return_json else annotated


# ================================================================
# MULTI-IMAGE SUMMARY + CSV
# ================================================================
def analyze_all_images():
    image_files = sorted(
        [f for f in os.listdir(UPLOAD_FOLDER) if f.lower().endswith((".png", ".jpg", ".jpeg"))],
        key=lambda f: os.path.getmtime(os.path.join(UPLOAD_FOLDER, f)),
        reverse=True
    )[:6]

    if not image_files:
        raise ValueError("No images found for analysis")

    all_results = []
    for img in image_files:
        path = os.path.join(UPLOAD_FOLDER, img)
        result = analyze_single_image(path, RESULTS_DIR)
        all_results.append(result)

    # Compute mean values
    avg_length = np.mean([r["length_mm"] for r in all_results])
    avg_diameter = np.mean([r["diameter_mm"] for r in all_results])
    avg_dev = np.mean([r["deviation_mm"] for r in all_results])

    summary_text = (
        "=== SUMMARY OF RESULTS ===\n"
        f"Average Wire Length Across All Views: {round(avg_length, 3)} mm\n"
        f"Average Wire Diameter: {round(avg_diameter, 3)} mm\n"
        f"Average Length Deviation from AR Model: {round(avg_dev, 3)} mm"
    )

    # Save to CSV
    csv_name = f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    csv_path = os.path.join(RESULTS_DIR, csv_name)

    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = list(all_results[0].keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
        writer.writerow({})
        writer.writerow({"filename": "AVERAGES", "length_mm": avg_length, "diameter_mm": avg_diameter, "deviation_mm": avg_dev})

    print(summary_text)
    print(f"✅ Detailed results saved to: {csv_path}")

    return {
        "summary": summary_text,
        "csv_path": f"/results/{csv_name}",
        "results": all_results
    }