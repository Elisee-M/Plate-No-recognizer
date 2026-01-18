from ultralytics import YOLO
import easyocr
import cv2
import os
import re

# ---------------- CONFIG ----------------
PROJECT_DIR = r"C:\Users\Admin\Documents\Elisee\New folder\Plate number recognition"
MODEL_PATH = os.path.join(PROJECT_DIR, "license-plate-finetune-v1n.pt")
IMAGE_FILE = os.path.join(PROJECT_DIR, "test_car3.png")  # <-- your single photo
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- LOAD MODELS ----------------
plate_detector = YOLO(MODEL_PATH)
ocr_reader = easyocr.Reader(['en'])

# ---------------- UTILS ----------------
def preprocess_plate(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def clean_text(text):
    text = text.upper()
    text = re.sub(r'[^A-Z0-9]', '', text)
    return text

# ---------------- PROCESS SINGLE IMAGE ----------------
image = cv2.imread(IMAGE_FILE)

results = plate_detector(image)

for i, box in enumerate(results[0].boxes.xyxy):
    x1, y1, x2, y2 = map(int, box)
    plate_crop = image[y1:y2, x1:x2]

    if plate_crop.size == 0:
        continue

    processed = preprocess_plate(plate_crop)

    ocr_result = ocr_reader.readtext(processed, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', detail=0)
    plate_text = clean_text("".join(ocr_result)) if ocr_result else "UNKNOWN"

    # ---------------- SPLIT EACH CHARACTER ----------------
    plate_chars = list(plate_text)
    char_vars = {f"char_{idx+1}": char for idx, char in enumerate(plate_chars)}

    # print results
    print(f"Plate {i}: {plate_text}")
    print("Parsed characters:", char_vars)

    # save cropped plate
    out_file = os.path.join(OUTPUT_DIR, f"plate{i}.png")
    cv2.imwrite(out_file, plate_crop)
