from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import easyocr
import cv2
import numpy as np
import re
import os

app = Flask(__name__)
CORS(app)

# ---------------- CONFIG ----------------
MODEL_PATH = "license-plate-finetune-v1n.pt"

# ---------------- LOAD MODELS ----------------
plate_detector = YOLO(MODEL_PATH)
ocr_reader = easyocr.Reader(['en'])

# ---------------- UTILS ----------------
def preprocess_plate(img, scale=3):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    gray = cv2.resize(gray, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    thresh = cv2.medianBlur(thresh, 3)
    return thresh

def clean_text(text):
    text = text.upper()
    return re.sub(r'[^A-Z0-9]', '', text)

# ---------------- API ROUTE ----------------
@app.route("/detect", methods=["POST"])
def detect_plate():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    img_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    results = plate_detector(image)
    detected_plates = []

    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)
        plate_crop = image[y1:y2, x1:x2]
        if plate_crop.size == 0:
            continue

        processed = preprocess_plate(plate_crop)
        ocr_result = ocr_reader.readtext(
            processed,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            detail=0
        )

        plate_text = clean_text("".join(ocr_result)) if ocr_result else "UNKNOWN"

        detected_plates.append({
            "plate": plate_text,
            "box": [x1, y1, x2, y2]
        })

    return jsonify({
        "count": len(detected_plates),
        "plates": detected_plates
    })

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(port=5000, debug=True)
