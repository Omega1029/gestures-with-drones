from codrone_edu.drone import *

import os
import threading
import time
import cv2
import numpy as np
import subprocess
from ultralytics import YOLO

import configs

drone = Drone()
drone.pair() # pair automatically, may not always work
#drone.pair(portname = os.enivron[DRONE_PORT_NAME])    # pair with a specific port

SPEED = 4
gesture_to_cmd = {
    "peace": (0, 0, 0, SPEED), #Rotate Right
    "ok": (0, 0, 0, -SPEED), #Rotate Left
    "palm": (-SPEED, 0, 0, 0), #Backward
    "fist": (SPEED, 0, 0, 0), #Forward
    "three": (0, SPEED, 0, 0), #Right
    "two_up": (0, -SPEED, 0, 0), #Left
    "like": (0, 0, SPEED, 0), #Up
    "dislike": (0, 0, -SPEED, 0), #Down
    # "peace_inverted": (SPEED, 0, 0, 0), #Forward
    "mute": (0, 0, 0, 0), #Hover
    # "palm": (SPEED, 0, 0, 0), #Forward
    # "rock": (-SPEED, 0, 0, 0), #Backward
}

def execute_command(gesture):
    """Executes the drone movement based on gesture input."""
    if gesture in gesture_to_cmd:
        pitch, roll, throttle, yaw = gesture_to_cmd[gesture]

        drone.set_pitch(pitch)
        drone.set_roll(roll)
        drone.set_throttle(throttle)
        drone.set_yaw(yaw)

        drone.move(1)  # Execute movement
        time.sleep(1)  # Adjust for smooth movement
        drone.hover()  # Stop motion

    else:
        print("Gesture not recognized.")
# Function to pause execution
def rest(t=2):
    time.sleep(t)


# Preprocess the frame for model input
def preprocess_frame(frame):
    return np.expand_dims(cv2.resize(frame, (224, 224)), axis=0)

# Load model and labels
model = YOLO(os.path.join(os.path.abspath(".."), configs.YOLO_MODEL))

# Start video capture
cap = cv2.VideoCapture(0)
drone.takeoff()
# Run hover command initially
drone.hover()
try:
    # Run inference on the source

    #time.sleep(2)
    results = model(source=0, stream=True)

    for orig in results:
        # frame is a frame from the video capture and the results object provides annotations
        frame = orig.plot()

        # Display the frame with annotations
        cv2.imshow("YOLOv8 Webcam Inference", frame)

        # Get the labels for the detections
        labels = orig.names  # This contains class names for each detection (mapping from class IDs to class names)

        # Check if there are any detections in the frame
        if len(orig.boxes) > 0:
            # Only process the first detected box
            first_box = orig.boxes[0]  # Get the first detection

            # Get the class ID and label (name) for the first detection
            class_id = int(first_box.cls)
            label = labels[class_id]
            print(f"Detected (first object): {label}")

            # Map the gesture label to a command
            print(f"Detected gesture: {label}")

            if label in gesture_to_cmd:
                execute_command(label)
            else:
                print(f"No command mapped for: {label}")

        # Exit on 'q' key
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

except Exception as e:
    print(e)

finally:
    print("\nLanding due to interruption...")
    drone.land()
    drone.emergency_stop()
    drone.close()

    cap.release()
    cv2.destroyAllWindows()
    #execute_command())  # Ensure to stop movement if program ends
