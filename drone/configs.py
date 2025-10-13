import os


# Set the root path to the 'yolo' directory (assuming this script is running from the root of your project)
YOLO_MODEL = os.path.join("yolo", "best.pt")
DETECTOR_MODEL_PATH = os.path.join("models", "hand_detector.onnx")
CLASSIFIER_MODEL_PATH = os.path.join("models", "crops_classifier.onnx")

#print(YOLO_MODEL)
