import cv2
from ultralytics import YOLO

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "yolo11n.pt"   # change to your model (e.g., "best.pt")
CAMERA_INDEX = 0            # default webcam


def main():
    # Load model
    model = YOLO(MODEL_PATH)

    # Open webcam
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Optional: set resolution
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Run YOLOv11 inference
        # stream=True returns a generator for better performance
        results = model.predict(frame, imgsz=640, conf=0.5, verbose=False)

        # results is a list; we use the first item
        annotated_frame = results[0].plot()  # draws boxes, labels, scores

        # Show the frame
        cv2.imshow("YOLOv11 Live Inference", annotated_frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

