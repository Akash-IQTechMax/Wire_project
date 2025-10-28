from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
from datetime import datetime
import csv
from deviation_check import analyze_single_image  # your OpenCV logic file

# --------------------------------------------
# CONFIG
# --------------------------------------------
UPLOAD_FOLDER = "uploads"
RESULTS_DIR = "results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

app = Flask(__name__)

@app.route('/getStatus')
def hello_world():
    return 'Server is running!'

# --------------------------------------------
# FILE UPLOAD ENDPOINT
# --------------------------------------------
@app.route('/upload', methods=['POST'])
def upload_image():
    """Accept image uploads from Unity."""
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded"}), 400

    file = request.files['file']
    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)

    print(f"✅ Uploaded: {filename}")
    return jsonify({"status": "success", "filename": filename}), 200

# --------------------------------------------
# ANALYSIS ENDPOINT
# --------------------------------------------
@app.route('/analyze', methods=['GET'])
def analyze_all_images():
    """Analyze the latest 6 uploaded images."""
    image_files = sorted(
        [f for f in os.listdir(UPLOAD_FOLDER) if f.lower().endswith(".png")],
        key=lambda x: os.path.getmtime(os.path.join(UPLOAD_FOLDER, x)),
        reverse=True
    )[:6]

    if not image_files:
        return jsonify({"status": "error", "message": "No images found"}), 404

    all_results = []
    total_length = 0.0
    total_diameter = 0.0
    total_deviation = 0.0

    for img_name in image_files:
        try:
            img_path = os.path.join(UPLOAD_FOLDER, img_name)
            result = analyze_single_image(img_path, RESULTS_DIR, return_json=True)
            all_results.append(result)

            total_length += float(result["length_mm"])
            total_diameter += float(result.get("diameter_mm", 0))
            total_deviation += float(result["deviation_mm"])

            print(f"✅ Analyzed {img_name}: {result}")
        except Exception as e:
            print(f"❌ Error analyzing {img_name}: {e}")

    # ----------------------------------------
    # CALCULATE AVERAGES
    # ----------------------------------------
    count = len(all_results)
    avg_length = total_length / count if count else 0
    avg_diameter = total_diameter / count if count else 0
    avg_deviation = total_deviation / count if count else 0

    summary_text = (
        "=== SUMMARY OF RESULTS ===\n"
        f"Average Wire Length Across All Views: {avg_length:.6f} mm\n"
        f"Average Wire Diameter: {avg_diameter:.6f} mm\n"
        f"Average Length Deviation from AR Model: {avg_deviation:.6f} mm"
    )

    # ----------------------------------------
    # SAVE CSV SUMMARY
    # ----------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    csv_path = os.path.join(RESULTS_DIR, f"summary_{timestamp}.csv")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["filename", "length_mm", "angle_deg", "direction", "deviation_mm"])
        writer.writeheader()
        for row in all_results:
            writer.writerow({
                "filename": row["filename"],
                "length_mm": row["length_mm"],
                "angle_deg": row["angle_deg"],
                "direction": row["direction"],
                "deviation_mm": row["deviation_mm"]
            })

    print("✅ Summary CSV saved:", csv_path)

    # ----------------------------------------
    # RETURN JSON TO UNITY
    # ----------------------------------------
    return jsonify({
        "status": "success",
        "summary": summary_text,
        "csv_path": f"/results/summary_{timestamp}.csv",
        "results": all_results
    }), 200

# --------------------------------------------
# STATIC FILES
# --------------------------------------------
@app.route('/results/<path:filename>')
def serve_results(filename):
    return send_from_directory(RESULTS_DIR, filename)

@app.route('/uploads/<path:filename>')
def serve_uploads(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# --------------------------------------------
# RUN SERVER
# --------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
