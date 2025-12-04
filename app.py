# # # # from flask import Flask, request, jsonify, send_from_directory, Response
# # # # from werkzeug.utils import secure_filename
# # # # import os
# # # # import time
# # # # from datetime import datetime
# # # # from deviation_check import analyze_all  # updated import
# # # # import json

# # # # # --------------------------------------------
# # # # # CONFIG
# # # # # --------------------------------------------
# # # # UPLOAD_FOLDER = "uploads"
# # # # RESULTS_DIR = "results"
# # # # os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# # # # os.makedirs(RESULTS_DIR, exist_ok=True)

# # # # app = Flask(__name__)

# # # # @app.route('/getStatus')
# # # # def hello_world():
# # # #     return 'Server is running!'

# # # # # --------------------------------------------
# # # # # FILE UPLOAD ENDPOINT
# # # # # --------------------------------------------
# # # # @app.route('/upload', methods=['POST'])
# # # # def upload_image():
# # # #     """Accept image uploads from Unity."""
# # # #     if 'file' not in request.files:
# # # #         return jsonify({"status": "error", "message": "No file uploaded"}), 400

# # # #     file = request.files['file']
# # # #     filename = secure_filename(file.filename)
# # # #     save_path = os.path.join(UPLOAD_FOLDER, filename)
# # # #     file.save(save_path)

# # # #     print(f"✅ Uploaded: {filename}")
# # # #     return jsonify({"status": "success", "filename": filename}), 200


# # # # def parse_summary_text_to_json(file_path):
# # # #     data = {}
# # # #     with open(file_path, "r", encoding="utf-8") as f:
# # # #         for line in f:
# # # #             if ":" in line:
# # # #                 key, value = line.strip().split(":", 1)
# # # #                 data[key.strip()] = value.strip()
# # # #     return data

# # # # # --------------------------------------------
# # # # # ANALYSIS ENDPOINT (with duration logs)
# # # # # --------------------------------------------
# # # # @app.route('/analyze', methods=['GET'])
# # # # def analyze_all_images():
# # # #     """Run the latest wire deviation analysis (6 images + 3D deviation)."""
# # # #     start_time = time.time()
# # # #     print("\n🚀 Starting wire deviation analysis...")
# # # #     print(f"🕒 Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# # # #     try:
# # # #         # Trigger analysis (auto-calibration + averaging)
# # # #         analyze_all()

# # # #         # Get the most recent results folder
# # # #         folders = sorted(
# # # #             [os.path.join(RESULTS_DIR, f) for f in os.listdir(RESULTS_DIR) if os.path.isdir(os.path.join(RESULTS_DIR, f))],
# # # #             key=os.path.getmtime,
# # # #             reverse=True
# # # #         )
# # # #         if not folders:
# # # #             print("⚠️ No result folders found after analysis.")
# # # #             return jsonify({"status": "error", "message": "No results found"}), 404

# # # #         latest_folder = folders[0]
# # # #         summary_path = os.path.join(latest_folder, "average_summary.txt")
# # # #         csv_path = os.path.join(latest_folder, "wire_analysis_results.csv")
            
# # # #         summary_json = parse_summary_text_to_json(summary_path)

# # # #         elapsed_time = time.time() - start_time
# # # #         print(f"✅ Analysis completed successfully in {elapsed_time:.2f} seconds.")
# # # #         print(f"📂 Results Folder: {latest_folder}")
# # # #         print(f"📄 Summary File: {summary_path}")
# # # #         print(f"📊 CSV File: {csv_path}")

# # # #         response_data = {
# # # #             "status": "success",
# # # #             "summary": summary_json,
# # # #             "processing_time_sec": round(elapsed_time, 2),
# # # #             "summary_file": f"/results/{os.path.basename(latest_folder)}/average_summary.txt",
# # # #             "csv_file": f"/results/{os.path.basename(latest_folder)}/wire_analysis_results.csv"
# # # #         }
        
# # # #         return Response(
# # # #             json.dumps(response_data, ensure_ascii=False, indent=4),
# # # #             mimetype="application/json"
# # # #         ), 200

# # # #     except Exception as e:
# # # #         elapsed_time = time.time() - start_time
# # # #         print(f"❌ Error during analysis after {elapsed_time:.2f} seconds: {e}")
# # # #         return jsonify({
# # # #             "status": "error",
# # # #             "message": str(e),
# # # #             "processing_time_sec": round(elapsed_time, 2)
# # # #         }), 500


# # # # # --------------------------------------------
# # # # # STATIC FILE ROUTES
# # # # # --------------------------------------------
# # # # @app.route('/results/<path:filename>')
# # # # def serve_results(filename):
# # # #     return send_from_directory(RESULTS_DIR, filename)

# # # # @app.route('/uploads/<path:filename>')
# # # # def serve_uploads(filename):
# # # #     return send_from_directory(UPLOAD_FOLDER, filename)

# # # # @app.route('/list_uploads', methods=['GET'])
# # # # def list_uploaded_images():
# # # #     """Return a JSON list of all image filenames in the uploads folder."""
# # # #     try:
# # # #         # Get all files from the UPLOAD_FOLDER
# # # #         files = [
# # # #             f for f in os.listdir(UPLOAD_FOLDER)
# # # #             if os.path.isfile(os.path.join(UPLOAD_FOLDER, f))
# # # #         ]

# # # #         # Optionally filter only image files (jpg, png, jpeg, bmp, etc.)
# # # #         image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
# # # #         image_files = [f for f in files if f.lower().endswith(image_extensions)]

# # # #         return jsonify({
# # # #             "status": "success",
# # # #             "count": len(image_files),
# # # #             "images": image_files
# # # #         }), 200

# # # #     except Exception as e:
# # # #         return jsonify({
# # # #             "status": "error",
# # # #             "message": str(e)
# # # #         }), 500


# # # # # --------------------------------------------
# # # # # RUN SERVER
# # # # # --------------------------------------------
# # # # if __name__ == "__main__":
# # # #     app.run(host="0.0.0.0", port=8080, debug=True)







# # # from flask import Flask, request, jsonify, send_from_directory, Response
# # # from werkzeug.utils import secure_filename
# # # import os
# # # import time
# # # from datetime import datetime
# # # # Import the new function for single file analysis
# # # from deviation_check import analyze_single_image 
# # # import json

# # # # --------------------------------------------
# # # # CONFIG
# # # # --------------------------------------------
# # # UPLOAD_FOLDER = "uploads"
# # # RESULTS_DIR = "results"
# # # os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# # # os.makedirs(RESULTS_DIR, exist_ok=True)

# # # app = Flask(__name__)

# # # @app.route('/getStatus')
# # # def get_status():
# # #     return 'Server is running!'

# # # # --------------------------------------------
# # # # FILE UPLOAD & ANALYSIS ENDPOINT
# # # # --------------------------------------------
# # # @app.route('/upload_and_analyze', methods=['POST'])
# # # def upload_and_analyze_image():
# # #     """Accept image, save it, run analysis on it, and return the results."""
# # #     start_time = time.time()
    
# # #     # 1. File Upload Check
# # #     if 'file' not in request.files:
# # #         return jsonify({"status": "error", "message": "No file part in the request"}), 400

# # #     file = request.files['file']
# # #     if file.filename == '':
# # #         return jsonify({"status": "error", "message": "No selected file"}), 400
        
# # #     filename = secure_filename(file.filename)
# # #     if not filename:
# # #         return jsonify({"status": "error", "message": "Invalid filename"}), 400
        
# # #     # Create a unique results folder for this run
# # #     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
# # #     current_results_folder = os.path.join(RESULTS_DIR, timestamp)
# # #     os.makedirs(current_results_folder, exist_ok=True)
    
# # #     # Save the uploaded file
# # #     save_path = os.path.join(UPLOAD_FOLDER, filename)
# # #     file.save(save_path)

# # #     print(f"\n✅ Uploaded: {filename}")
# # #     print(f"🚀 Starting analysis for: {filename}")

# # #     try:
# # #         # 2. Trigger analysis for the SINGLE uploaded image
# # #         # We pass the full path, output folder, and filename
# # #         analysis_result = analyze_single_image(
# # #             img_path=save_path, 
# # #             output_dir=current_results_folder, 
# # #             filename=filename
# # #         )
        
# # #         # 3. Handle Analysis Success or Failure
# # #         if analysis_result is None or analysis_result.get("status") == "error":
# # #             raise Exception(analysis_result.get("message", "Unknown analysis error"))
            
# # #         # 4. Format the Response
# # #         elapsed_time = time.time() - start_time
        
# # #         # Get the measurement dictionary directly from the analysis result
# # #         measurement_data = analysis_result.get("measurement", {})
        
# # #         response_data = {
# # #             "status": "success",
# # #             "filename": filename,
# # #             "results_folder": os.path.basename(current_results_folder),
# # #             "processing_time_sec": round(elapsed_time, 2),
# # #             "measurement": measurement_data,
# # #             "annotated_image_file": f"/results/{os.path.basename(current_results_folder)}/{filename}_result.png"
# # #         }
        
# # #         print(f"✅ Analysis completed successfully in {elapsed_time:.2f} seconds.")
# # #         return Response(
# # #             json.dumps(response_data, ensure_ascii=False, indent=4),
# # #             mimetype="application/json"
# # #         ), 200

# # #     except Exception as e:
# # #         elapsed_time = time.time() - start_time
# # #         print(f"❌ Error during analysis after {elapsed_time:.2f} seconds: {e}")
# # #         return jsonify({
# # #             "status": "error",
# # #             "message": str(e),
# # #             "processing_time_sec": round(elapsed_time, 2),
# # #             "filename": filename
# # #         }), 500


# # # # --------------------------------------------
# # # # STATIC FILE ROUTES (Kept for serving results/uploads)
# # # # --------------------------------------------
# # # @app.route('/results/<folder_name>/<filename>')
# # # def serve_results(folder_name, filename):
# # #     return send_from_directory(os.path.join(RESULTS_DIR, folder_name), filename)

# # # @app.route('/uploads/<path:filename>')
# # # def serve_uploads(filename):
# # #     return send_from_directory(UPLOAD_FOLDER, filename)

# # # # --------------------------------------------
# # # # RUN SERVER
# # # # --------------------------------------------
# # # if __name__ == "__main__":
# # #     app.run(host="0.0.0.0", port=8080, debug=True)


# # from flask import Flask, request, jsonify, send_from_directory, Response
# # from werkzeug.utils import secure_filename
# # import os
# # import time
# # from datetime import datetime
# # # Import the new function for single file analysis
# # from deviation_check import analyze_single_image 
# # import json

# # # --------------------------------------------
# # # CONFIG
# # # --------------------------------------------
# # UPLOAD_FOLDER = "uploads"
# # RESULTS_DIR = "results"
# # os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# # os.makedirs(RESULTS_DIR, exist_ok=True)

# # app = Flask(__name__)

# # # --------------------------------------------
# # # NEW: ROOT ROUTE FOR HEALTH CHECKS 🟢
# # # --------------------------------------------
# # @app.route('/', methods=['GET'])
# # def home_status():
# #     """Simple route for platform health checks."""
# #     return 'Wire project server is operational.', 200

# # @app.route('/getStatus')
# # def get_status():
# #     return 'Server is running!'

# # # --------------------------------------------
# # # FILE UPLOAD & ANALYSIS ENDPOINT
# # # --------------------------------------------
# # @app.route('/upload_and_analyze', methods=['POST'])
# # def upload_and_analyze_image():
# #     """Accept image, save it, run analysis on it, and return the results."""
# #     start_time = time.time()
    
# #     # 1. File Upload Check
# #     if 'file' not in request.files:
# #         return jsonify({"status": "error", "message": "No file part in the request"}), 400

# #     file = request.files['file']
# #     if file.filename == '':
# #         return jsonify({"status": "error", "message": "No selected file"}), 400
        
# #     filename = secure_filename(file.filename)
# #     if not filename:
# #         return jsonify({"status": "error", "message": "Invalid filename"}), 400
        
# #     # Create a unique results folder for this run
# #     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
# #     current_results_folder = os.path.join(RESULTS_DIR, timestamp)
# #     os.makedirs(current_results_folder, exist_ok=True)
    
# #     # Save the uploaded file
# #     save_path = os.path.join(UPLOAD_FOLDER, filename)
# #     file.save(save_path)

# #     print(f"\n✅ Uploaded: {filename}")
# #     print(f"🚀 Starting analysis for: {filename}")

# #     try:
# #         # 2. Trigger analysis for the SINGLE uploaded image
# #         analysis_result = analyze_single_image(
# #             img_path=save_path, 
# #             output_dir=current_results_folder, 
# #             filename=filename
# #         )
        
# #         # 3. Handle Analysis Success or Failure
# #         if analysis_result is None or analysis_result.get("status") == "error":
# #             raise Exception(analysis_result.get("message", "Unknown analysis error"))
            
# #         # 4. Format the Response
# #         elapsed_time = time.time() - start_time
        
# #         # Get the measurement dictionary directly from the analysis result
# #         measurement_data = analysis_result.get("measurement", {})
        
# #         response_data = {
# #             "status": "success",
# #             "filename": filename,
# #             "results_folder": os.path.basename(current_results_folder),
# #             "processing_time_sec": round(elapsed_time, 2),
# #             "measurement": measurement_data,
# #             "annotated_image_file": f"/results/{os.path.basename(current_results_folder)}/{filename}_result.png"
# #         }
        
# #         print(f"✅ Analysis completed successfully in {elapsed_time:.2f} seconds.")
# #         return Response(
# #             json.dumps(response_data, ensure_ascii=False, indent=4),
# #             mimetype="application/json"
# #         ), 200

# #     except Exception as e:
# #         elapsed_time = time.time() - start_time
# #         print(f"❌ Error during analysis after {elapsed_time:.2f} seconds: {e}")
# #         return jsonify({
# #             "status": "error",
# #             "message": str(e),
# #             "processing_time_sec": round(elapsed_time, 2),
# #             "filename": filename
# #         }), 500


# # # --------------------------------------------
# # # STATIC FILE ROUTES (Kept for serving results/uploads)
# # # --------------------------------------------
# # @app.route('/results/<folder_name>/<filename>')
# # def serve_results(folder_name, filename):
# #     return send_from_directory(os.path.join(RESULTS_DIR, folder_name), filename)

# # @app.route('/uploads/<path:filename>')
# # def serve_uploads(filename):
# #     return send_from_directory(UPLOAD_FOLDER, filename)



# #change done on 4/12/2025 for the triger duraiation increase 

# from flask import Flask, request, jsonify, send_from_directory, Response
# from werkzeug.utils import secure_filename
# import os
# import time
# from datetime import datetime
# from deviation_check import analyze_single_image 
# import json

# # --------------------------------------------
# # CONFIG
# # --------------------------------------------
# UPLOAD_FOLDER = "uploads"
# RESULTS_DIR = "results"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(RESULTS_DIR, exist_ok=True)

# app = Flask(__name__)

# # --------------------------------------------
# # NEW: ROOT ROUTE FOR HEALTH CHECKS 🟢
# # --------------------------------------------
# @app.route('/', methods=['GET'])
# def home_status():
#     """Simple route for platform health checks."""
#     return 'Wire project server is operational.', 200

# @app.route('/getStatus')
# def get_status():
#     return 'Server is running!'

# # --------------------------------------------
# # FILE UPLOAD & ANALYSIS ENDPOINT
# # --------------------------------------------
# @app.route('/upload_and_analyze', methods=['POST'])
# def upload_and_analyze_image():
#     """Accept image, save it, run analysis on it, and return the results."""
#     start_time = time.time()
    
#     # 1. File Upload Check
#     if 'file' not in request.files:
#         return jsonify({"status": "error", "message": "No file part in the request"}), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"status": "error", "message": "No selected file"}), 400
        
#     filename = secure_filename(file.filename)
#     if not filename:
#         return jsonify({"status": "error", "message": "Invalid filename"}), 400
        
#     # Create a unique results folder for this run
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
#     current_results_folder = os.path.join(RESULTS_DIR, timestamp)
#     os.makedirs(current_results_folder, exist_ok=True)
    
#     # Save the uploaded file
#     save_path = os.path.join(UPLOAD_FOLDER, filename)
#     file.save(save_path)

#     print(f"\n✅ Uploaded: {filename}")
#     print(f"🚀 Starting analysis for: {filename}")

#     try:
#         # 2. Trigger analysis for the SINGLE uploaded image
#         analysis_result = analyze_single_image(
#             img_path=save_path, 
#             output_dir=current_results_folder, 
#             filename=filename
#         )
        
#         # 3. Handle Analysis Success or Failure
#         if analysis_result is None or analysis_result.get("status") == "error":
#             raise Exception(analysis_result.get("message", "Unknown analysis error"))
            
#         # 4. Format the Response
#         elapsed_time = time.time() - start_time
        
#         # Get the measurement dictionary directly from the analysis result
#         measurement_data = analysis_result.get("measurement", {})
        
#         response_data = {
#             "status": "success",
#             "filename": filename,
#             "results_folder": os.path.basename(current_results_folder),
#             "processing_time_sec": round(elapsed_time, 2),
#             "measurement": measurement_data,
#             "annotated_image_file": f"/results/{os.path.basename(current_results_folder)}/{filename}_result.png"
#         }
        
#         print(f"✅ Analysis completed successfully in {elapsed_time:.2f} seconds.")
#         return Response(
#             json.dumps(response_data, ensure_ascii=False, indent=4),
#             mimetype="application/json"
#         ), 200

#     except Exception as e:
#         elapsed_time = time.time() - start_time
#         print(f"❌ Error during analysis after {elapsed_time:.2f} seconds: {e}")
#         return jsonify({
#             "status": "error",
#             "message": str(e),
#             "processing_time_sec": round(elapsed_time, 2),
#             "filename": filename
#         }), 500


# # --------------------------------------------
# # STATIC FILE ROUTES (Kept for serving results/uploads)
# # --------------------------------------------
# @app.route('/results/<folder_name>/<filename>')
# def serve_results(folder_name, filename):
#     return send_from_directory(os.path.join(RESULTS_DIR, folder_name), filename)

# @app.route('/uploads/<path:filename>')
# def serve_uploads(filename):
#     return send_from_directory(UPLOAD_FOLDER, filename)


#local host code checking 4/12/2025 after the lunch vignesh bro asked to run in the local itself



from flask import Flask, request, jsonify, send_from_directory, Response
from werkzeug.utils import secure_filename
import os
import time
from datetime import datetime
import json
from deviation_check import analyze_all

# --------------------------------------------
# CONFIG
# --------------------------------------------
UPLOAD_FOLDER = "uploads"
RESULTS_DIR = "results"
# Ensure directories exist and are ready
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Initialize the Flask application
app = Flask(__name__)

# --------------------------------------------
# UTILITY FUNCTIONS
# --------------------------------------------
def parse_summary_text_to_json(file_path):
    """Parses a simple key: value text file into a JSON dictionary."""
    data = {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if ":" in line:
                    key, value = line.split(":", 1)
                    # Clean up keys and values, attempting to convert numeric values later if needed
                    data[key.strip()] = value.strip()
    except FileNotFoundError:
        print(f"Error: Summary file not found at {file_path}")
    return data

def get_latest_results_folder():
    """Finds the path to the most recently created results folder."""
    folders = [
        os.path.join(RESULTS_DIR, f) 
        for f in os.listdir(RESULTS_DIR) 
        if os.path.isdir(os.path.join(RESULTS_DIR, f))
    ]
    if not folders:
        return None
    # Sort by creation time (or modification time) and get the latest
    latest_folder = max(folders, key=os.path.getmtime)
    return latest_folder

# --------------------------------------------
# HEALTH & STATUS ROUTES
# --------------------------------------------
@app.route('/', methods=['GET'])
def home_status():
    """Simple route for platform health checks."""
    return 'Wire project server is operational.', 200

@app.route('/getStatus')
def get_status():
    """Simple route to confirm server readiness."""
    return 'Server is running!'

# --------------------------------------------
# FILE UPLOAD ENDPOINT 📤
# --------------------------------------------
@app.route('/upload', methods=['POST'])
def upload_image():
    """Accept image uploads and save them."""
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400
        
    filename = secure_filename(file.filename)
    if not filename:
        return jsonify({"status": "error", "message": "Invalid filename"}), 400
        
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)

    print(f"✅ Uploaded: {filename} to {save_path}")
    return jsonify({
        "status": "success", 
        "filename": filename
    }), 200

# --------------------------------------------
# ANALYSIS ENDPOINT 🧪 (Multi-Image/Batch)
# --------------------------------------------
@app.route('/analyze', methods=['GET'])
def analyze_all_images():
    """
    Triggers the deviation_check.analyze_all() function which is expected to:
    1. Read all images from the UPLOAD_FOLDER.
    2. Run analysis, calibration, and averaging.
    3. Save results (summary.txt, csv) into a NEW timestamped folder in RESULTS_DIR.
    """
    start_time = time.time()
    print("\n🚀 Starting wire deviation analysis (Batch mode)...")
    print(f"🕒 Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # 1. Trigger analysis (assumes analyze_all handles file reading/writing)
        # NOTE: If analyze_all needs arguments (like which folder to check), adjust here.
        analyze_all() 

        # 2. Get the most recent results folder created by analyze_all()
        latest_folder_path = get_latest_results_folder()
        if not latest_folder_path:
            raise Exception("Analysis ran, but no results folder was created or found.")

        latest_folder_name = os.path.basename(latest_folder_path)
        summary_path = os.path.join(latest_folder_path, "average_summary.txt")
        csv_path = os.path.join(latest_folder_path, "wire_analysis_results.csv")
            
        # 3. Parse the summary file to include its content in the JSON response
        summary_json = parse_summary_text_to_json(summary_path)

        # 4. Format response
        elapsed_time = time.time() - start_time
        print(f"✅ Analysis completed successfully in {elapsed_time:.2f} seconds.")

        response_data = {
            "status": "success",
            "results_folder_id": latest_folder_name,
            "processing_time_sec": round(elapsed_time, 2),
            "summary_data": summary_json, # Include parsed data directly
            # Construct URL paths for the client to retrieve the files
            "summary_file_url": f"/results/{latest_folder_name}/average_summary.txt",
            "csv_file_url": f"/results/{latest_folder_name}/wire_analysis_results.csv"
        }
        
        # NOTE: If you save the annotated images, you will need to list them here.
        # Assuming you save images as 'image1_result.png', we can list them:
        annotated_images = [f for f in os.listdir(latest_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        response_data['annotated_image_urls'] = [f"/results/{latest_folder_name}/{img}" for img in annotated_images]

        return Response(
            json.dumps(response_data, ensure_ascii=False, indent=4),
            mimetype="application/json"
        ), 200

    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"❌ Error during analysis after {elapsed_time:.2f} seconds: {e}")
        return jsonify({
            "status": "error",
            "message": str(e),
            "processing_time_sec": round(elapsed_time, 2)
        }), 500


# --------------------------------------------
# STATIC FILE ROUTES (Handling the new structure)
# --------------------------------------------
@app.route('/results/<folder_name>/<filename>')
def serve_results(folder_name, filename):
    """Serves files from the time-stamped results sub-folders."""
    # Note: We join RESULTS_DIR with the folder_name to specify the sub-directory
    full_path = os.path.join(RESULTS_DIR, folder_name)
    if not os.path.isdir(full_path):
        return jsonify({"status": "error", "message": "Results folder not found"}), 404
    return send_from_directory(full_path, filename)

@app.route('/uploads/<path:filename>')
def serve_uploads(filename):
    """Serves the original uploaded image files."""
    return send_from_directory(UPLOAD_FOLDER, filename)


# --------------------------------------------
# RUN SERVER
# --------------------------------------------
if __name__ == "__main__":
    print("--- Starting Flask Development Server ---")
    # Using the port you specified (8080)
    app.run(host="0.0.0.0", port=8080, debug=True)