# ==============================
# CADI-AI Flask Backend (FINAL)
# ==============================

import io
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

# ------------------------------
# MODEL PATHS
# ------------------------------
MODIFIED_MODEL_PATH = Path("weights/modified_yolo_best.pt")
BASELINE_MODEL_PATH = Path("weights/yolov8n.pt")  # baseline

models = {}

CLASS_NAMES = ["abiotic", "insect", "disease"]

# ------------------------------
# LOAD MODEL
# ------------------------------
def load_model(model_type="modified"):
    global models

    if model_type in models:
        return models[model_type]

    if model_type == "baseline":
        print("[INFO] Loading baseline model...")
        models[model_type] = YOLO("yolov8n.pt")
    else:
        print("[INFO] Loading modified model...")
        if not MODIFIED_MODEL_PATH.exists():
            raise FileNotFoundError("Modified model not found!")
        models[model_type] = YOLO(str(MODIFIED_MODEL_PATH))

    return models[model_type]


# ------------------------------
# HOME ROUTE
# ------------------------------
@app.route("/")
def index():
    return send_from_directory(".", "06_interactive_demo.html")


# ------------------------------
# PREDICT ROUTE
# ------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]

    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
    except:
        return jsonify({"error": "Invalid image"}), 400

    # get params
    try:
        conf = float(request.form.get("conf", 0.25))
    except:
        conf = 0.25

    model_type = request.form.get("model", "modified")

    model = load_model(model_type)

    results = model.predict(img, conf=conf, verbose=False)

    detections = []

    if results[0].boxes is not None:
        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf_score = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "class": CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else str(cls),
                "confidence": round(conf_score, 4),
                "bbox": [round(x1,1), round(y1,1), round(x2,1), round(y2,1)]
            })

    detections.sort(key=lambda x: x["confidence"], reverse=True)

    return jsonify({
        "model": model_type,
        "detections": detections
    })


# ------------------------------
# RUN
# ------------------------------
if __name__ == "__main__":
    print("Server running at http://localhost:5000")
    load_model("modified")
    app.run(port=5000)