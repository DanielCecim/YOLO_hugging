from ultralytics import YOLO
import sys

model = YOLO("best.pt")

video_path = sys.argv[1] if len(sys.argv) > 1 else 0  # 0 = webcam

results = model.predict(
    source=video_path,
    conf=0.6,       # higher threshold to reduce false positives on cars
    iou=0.45,
    imgsz=1280,     # higher res helps detect small guns
    show=True,
    save=True,
    line_width=2,
)
