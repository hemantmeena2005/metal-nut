"""
Experiment B: Train YOLOv8n at imgsz=640 (vs baseline 320)
Hypothesis: larger input = more pixel detail → better recall on subtle scratches/bends
"""
from pathlib import Path
from ultralytics import YOLO

BASE     = Path(__file__).parent
DS_YAML  = BASE / "dataset" / "dataset.yaml"

model = YOLO("yolov8n.pt")
results = model.train(
    data     = str(DS_YAML),
    epochs   = 80,
    imgsz    = 640,
    batch    = 8,          # smaller batch because images are 4x bigger
    patience = 20,
    project  = str(BASE / "runs"),
    name     = "yolov8n_640",
    exist_ok = True,
    device   = "cpu",
    workers  = 4,
    mosaic   = 1.0,
    fliplr   = 0.5,
    flipud   = 0.5,
    hsv_h    = 0.015,
    hsv_s    = 0.7,
    hsv_v    = 0.4,
    degrees  = 10.0,
    scale    = 0.5,
    verbose  = True,
)

print("\n640px training complete!")
print(f"Best weights: {BASE}/runs/yolov8n_640/weights/best.pt")
