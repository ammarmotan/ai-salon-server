from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os, uuid, urllib.request
from processor import process_image

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Auto-download model if not present
MODEL_PATH = "face_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("Downloading face_landmarker.task...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        MODEL_PATH
    )
    print("Model downloaded.")

@app.route("/process", methods=["POST"])
def process():
    if "image" not in request.files:
        return jsonify({"status": "error", "message": "No image"}), 400

    file       = request.files["image"]
    hair_color      = request.form.get("hair_color",      "#503200")
    lip_color       = request.form.get("lip_color",       "#c73b3b")
    eyeshadow_color = request.form.get("eyeshadow_color", "#6B4F62")
    blush_color     = request.form.get("blush_color",     "#D4847A")
    smoothness      = float(request.form.get("smoothness", 0.5))
    intensity       = float(request.form.get("intensity",  0.6))
    bridal_style    = request.form.get("bridal_style",    "none")
    face_filter     = request.form.get("face_filter",     "none")
    do_hair         = request.form.get("do_hair",   "false").lower() == "true"
    do_makeup       = request.form.get("do_makeup", "false").lower() == "true"
    do_bridal       = request.form.get("do_bridal", "false").lower() == "true"
    do_filter       = request.form.get("do_filter", "false").lower() == "true"

    ext      = os.path.splitext(file.filename)[1] or ".jpg"
    uid      = uuid.uuid4().hex[:8]
    in_path  = os.path.join(UPLOAD_FOLDER, uid + ext)
    out_name = "out_" + uid + ext
    out_path = os.path.join(OUTPUT_FOLDER, out_name)
    file.save(in_path)

    process_image(in_path, out_path,
                  hair_color=hair_color,
                  lip_color=lip_color, eyeshadow_color=eyeshadow_color,
                  blush_color=blush_color, smoothness=smoothness,
                  intensity=intensity, bridal_style=bridal_style,
                  face_filter=face_filter,
                  do_hair=do_hair, do_makeup=do_makeup,
                  do_bridal=do_bridal, do_filter=do_filter)

    return jsonify({"status": "success", "image": "outputs/" + out_name})

@app.route("/outputs/<path:filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

@app.route("/ping")
def ping():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
