from ultralytics import YOLO

# Load YOLOv11 license plate detector
model = YOLO("license-plate-finetune-v1n.pt")

# Use a REAL car image with a visible plate
source = "test_car.jpg"   # put an image in the same folder

# Run detection
results = model.predict(
    source=source,
    save=True,
    conf=0.4
)

print("Done! Check runs/detect/predict/")
