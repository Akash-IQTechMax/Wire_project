# from flask import Flask, request, jsonify, send_from_directory, render_template_string
# import os
# import time
# import subprocess

# # =============================================================
# # BASIC SETUP
# # =============================================================
# app = Flask(__name__)

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
# RESULTS_DIR = os.path.join(BASE_DIR, "results")

# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(RESULTS_DIR, exist_ok=True)

# # 🔹 Set your Mac’s local IP here (check using: ifconfig | grep "inet ")
# FIXED_IP = "192.168.0.103"
# PORT = 5000

# # =============================================================
# # DASHBOARD (for browser view)
# # =============================================================
# @app.route('/')
# def home():
#     files = sorted(os.listdir(UPLOAD_FOLDER))
#     reports = sorted(os.listdir(RESULTS_DIR))

#     html = """
#     <html>
#     <head><title>📊 Wire Capture Dashboard</title></head>
#     <body style='font-family:Arial; margin:40px'>
#     <h2>📸 Uploaded Images</h2>
#     """

#     if files:
#         for f in files:
#             html += f'<a href="/images/{f}" target="_blank">{f}</a><br>'
#     else:
#         html += "<p>No images uploaded yet.</p>"

#     html += "<hr><h2>📄 Analysis Reports</h2>"

#     if reports:
#         for r in reports:
#             html += f'<a href="/results/{r}" target="_blank">{r}</a><br>'
#     else:
#         html += "<p>No reports generated yet.</p>"

#     html += "</body></html>"
#     return render_template_string(html)


# # =============================================================
# # UPLOAD ENDPOINT (Unity sends images here)
# # =============================================================
# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return jsonify({"status": "error", "message": "No file received"}), 400

#     file = request.files['file']
#     filename = f"{int(time.time())}_{file.filename.replace(' ', '')}"
#     path = os.path.join(UPLOAD_FOLDER, filename)
#     file.save(path)

#     print(f"✅ Uploaded: {filename}")
#     return jsonify({"status": "success", "file": filename})


# # =============================================================
# # SERVE UPLOADED FILES
# # =============================================================
# @app.route('/images/<filename>')
# def uploaded_file(filename):
#     return send_from_directory(UPLOAD_FOLDER, filename)


# # =============================================================
# # SERVE ANALYSIS RESULTS
# # =============================================================
# @app.route('/results/<path:filename>')
# def get_results(filename):
#     return send_from_directory(RESULTS_DIR, filename)


# # =============================================================
# # ANALYZE ENDPOINT (Unity triggers this)
# # =============================================================
# @app.route('/analyze', methods=['GET'])
# def run_analysis():
#     try:
#         print("🧠 Starting wire analysis script...")
#         subprocess.Popen(["python3", os.path.join(BASE_DIR, "wire_analysis.py")])
#         return jsonify({"status": "started"})
#     except Exception as e:
#         print(f"❌ Analysis error: {e}")
#         return jsonify({"status": "error", "message": str(e)}), 500


# # =============================================================
# # MAIN ENTRY POINT
# # =============================================================
# if __name__ == "__main__":
#     print(f"🚀 Flask running at: http://{FIXED_IP}:{PORT}")
#     app.run(host="0.0.0.0", port=PORT, debug=True)



from flask import Flask, request, jsonify, send_from_directory, render_template_string
import os
import time
import shutil
from datetime import datetime
from wire_analysis import analyze_single_image  # import your OpenCV function

# =============================================================
# BASIC SETUP
# =============================================================
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# 🔹 Set your Mac/PC IP (find via ifconfig/ipconfig)
FIXED_IP = "192.168.0.103"  # 🔧 change to your machine IP
PORT = 5000


# =============================================================
# FUNCTION: CLEANUP OLD SESSION DATA
# =============================================================
def clear_previous_session():
    print("🧹 Clearing old session data...")

    # Delete uploaded files
    for f in os.listdir(UPLOAD_FOLDER):
        try:
            os.remove(os.path.join(UPLOAD_FOLDER, f))
        except Exception as e:
            print(f"⚠️ Could not delete {f}: {e}")

    print("✅ Old session cleared!")


# =============================================================
# MANUAL CLEAR ENDPOINT
# =============================================================
@app.route('/clear', methods=['GET'])
def clear_manual():
    clear_previous_session()
    return jsonify({"status": "cleared"})


# =============================================================
# DASHBOARD (Browser view)
# =============================================================
@app.route('/')
def home():
    files = sorted(os.listdir(UPLOAD_FOLDER))
    reports = sorted(os.listdir(RESULTS_DIR))

    html = """
    <html>
    <head><title>📊 Wire Capture Dashboard</title></head>
    <body style='font-family:Arial; margin:40px'>
    <h2>📸 Uploaded Images</h2>
    """

    if files:
        for f in files:
            html += f'<a href="/images/{f}" target="_blank">{f}</a><br>'
    else:
        html += "<p>No images uploaded yet.</p>"

    html += "<hr><h2>📄 Analysis Reports</h2>"

    if reports:
        for r in reports:
            html += f'<a href="/results/{r}" target="_blank">{r}</a><br>'
    else:
        html += "<p>No reports generated yet.</p>"

    html += "</body></html>"
    return render_template_string(html)


# =============================================================
# UPLOAD + AUTO ANALYZE ENDPOINT (FULL AUTO)
# =============================================================
@app.route('/upload', methods=['POST'])
def upload_and_analyze():
    """Uploads file and runs OpenCV analysis automatically."""
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file received"}), 400

    file = request.files['file']
    filename = f"{int(time.time())}_{file.filename.replace(' ', '')}"
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    print(f"✅ Uploaded: {filename}")

    # 🧠 Auto-run OpenCV analysis here
    try:
        results = analyze_single_image(path, RESULTS_DIR, return_json=True)
        return jsonify({"status": "success", "results": results})
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# =============================================================
# SERVE FILES
# =============================================================
@app.route('/images/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route('/results/<path:filename>')
def get_results(filename):
    return send_from_directory(RESULTS_DIR, filename)


# =============================================================
# MAIN
# =============================================================
if __name__ == "__main__":
    print(f"🚀 Flask running at: http://{FIXED_IP}:{PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=True)
