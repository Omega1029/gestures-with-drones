import os
import threading
import time
import cv2
import numpy as np
import subprocess
from ultralytics import YOLO

import configs


SPEED = 0.5
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
# Helper function to run system commands
def run_command(command):
    try:
        result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        print("Command Output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error:", e.stderr)

# Function to build the move command based on gestures
def build_move_command(x=0, y=0, z=0, rotation=0):
    return f'''gz topic -t "/X3/gazebo/command/twist" -m gz.msgs.Twist -p "linear: {{x: {x}, y: {y}, z: {z}}} angular: {{z: {rotation}}}"'''

# Hover command
def build_hover_command():
    return '''gz topic -t "/X3/gazebo/command/twist" -m gz.msgs.Twist -p " "'''

# Function to pause execution
def rest(t=2):
    time.sleep(t)

# Gesture-to-command mapping (modify this dictionary based on your gestures)





# Function to get the gesture command
def gesture_move(gesture="ok"):

    return build_move_command(*gesture_to_cmd.get(gesture, (0, 0, 0, 0)))  # Default to (SPEED, 0, 0, 0)

# Preprocess the frame for model input
def preprocess_frame(frame):
    return np.expand_dims(cv2.resize(frame, (224, 224)), axis=0)

# Load model and labels
model = YOLO(os.path.join(os.path.abspath(".."), configs.YOLO_MODEL))

# Start video capture
cap = cv2.VideoCapture(0)

# Run hover command initially
run_command(build_hover_command())

try:
    # Run inference on the source
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
            if label in gesture_to_cmd:
                # Get the gesture command based on the detected label
                cmd = gesture_move(label)
                print(f"Running command: {cmd}")
                x = threading.Thread(target=run_command, args=(cmd,))
                x.start()
                #run_command(cmd)  # Replace `run_command(cmd)` with your actual command

                # Optional: Add a delay between commands
                #time.sleep(3)
            else:
                print(f"No command mapped for gesture: {label}")

        # Exit on 'q' key
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

finally:
    cap.release()
    cv2.destroyAllWindows()
    run_command(build_hover_command())  # Ensure to stop movement if program ends
