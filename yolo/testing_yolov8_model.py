from ultralytics import YOLO
import cv2
import streamlit as st

# Load a pretrained YOLO11n model
model = YOLO("best.pt")

# Define source as YouTube video URL
#source = "https://youtu.be/LNwODJXcvt4"
source = 0
# Run inference on the source
results = model(source, stream=True)

for orig in results:
    # frame is a frame from the video capture and the results object provides annotations
    frame = orig.plot()
    # Display the frame with annotations (bounding boxes, labels, etc.)
    st.image(frame, channels="BGR")
    #cv2.imshow("YOLOv8 Webcam Inference", frame)
    labels = orig.names  # This contains class names for each detection (mapping from class ids to class names)

    # Iterate over the detected boxes
    for i, box in enumerate(orig.boxes):
        class_id = int(box.cls)  # Get the class id
        label = labels[class_id]  # Get the class name (label)
        print(f"Detected: {label}")
    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources and close the window
cv2.destroyAllWindows()