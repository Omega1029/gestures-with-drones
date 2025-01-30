from ultralytics import YOLO

model = YOLO('custom_yolov8.yaml')  # Load custom model architecture
data_yaml_path = './data.yaml'
model.train(data=data_yaml_path, epochs=100, imgsz=640, batch=16)