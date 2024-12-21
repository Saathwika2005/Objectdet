from ultralytics import YOLO

# Load a pretrained YOLOv5 model with maximum weights
model = YOLO("yolov5x.pt")  # yolov5x is the largest YOLOv5 model


model.predict(0, save=True, imgsz=320, conf=0.5,vid_stride=1)
