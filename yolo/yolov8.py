# Import necessary libraries from YOLOv8 package
from ultralytics import YOLO

# Path to your custom data.yaml file
data_yaml_path = './data.yaml'  # Replace with the correct path

# Define model: Load a pre-trained YOLOv8 model (can be a YOLOv8s, YOLOv8m, or YOLOv8l variant)
# Alternatively, you can train from scratch by passing an empty model
model = YOLO('best.pt')  # Use yolov8s.pt for small model, or yolov8m.pt / yolov8l.pt for medium/large

# Start training
model.train(data=data_yaml_path, epochs=100, imgsz=640, batch=16)