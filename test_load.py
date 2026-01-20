import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

print("Importing YOLO...")
from ultralytics import YOLO
print("Importing EasyOCR...")
import easyocr
print("Importing NumPy...")
import numpy as np
print("Done imports ✅")

PROJECT_DIR = "."
MODEL_PATH = "license-plate-finetune-v1n.pt"

print("Loading YOLO model...")
model = YOLO(MODEL_PATH)
print("YOLO loaded ✅")

print("Loading EasyOCR reader...")
reader = easyocr.Reader(['en'])
print("EasyOCR loaded ✅")
