
from codrone_edu.drone import *
import os
import cv2
import numpy as np
import threading
from ultralytics import YOLO
import configs

drone = Drone()
drone.pair()  # pair automatically, may not always work

SPEED = 20
gesture_to_cmd = {
    "like": (0, 0, SPEED, 0),  # Up
    "dislike": (0, 0, -SPEED//2, 0),  # Down
    "palm": (-SPEED, 0, 0, 0),  # Backward
    "fist": (SPEED, 0, 0, 0),  # Forward
    "three": (0, SPEED, 0, 0),  # Right
    "two_up": (0, -SPEED, 0, 0),  # Left
    "peace": (0, 0, 0, SPEED),  # Rotate Right
    "ok": (0, 0, 0, -SPEED),  # Rotate Left
    "mute": (0, 0, 0, 0),  # Hover
}
gesture_to_string = {
    "peace": "Rotate Right",
    "ok": "Rotate Left",
    "palm": "Backward",
    "fist": "Forward",
    "three": "Right",
    "two_up": "Left",
    "like": "Up",
    "dislike":"Down",
    # "peace_inverted": (SPEED, 0, 0, 0), #Forward,
    "mute": "Hover",
    # "palm": (SPEED, 0, 0, 0), #Forward
    # "rock": (-SPEED, 0, 0, 0), #Backward
}

def execute_command(gesture):
    """Executes the drone movement based on gesture input asynchronously."""
    if gesture in gesture_to_cmd:
        pitch, roll, throttle, yaw = gesture_to_cmd[gesture]

        def run_movement():
            drone.set_pitch(pitch)
            drone.set_roll(roll)
            drone.set_throttle(throttle)
            drone.set_yaw(yaw)
            drone.move(2)  # Execute movement
            drone.hover(1)  # Stop motion

        movement_thread = threading.Thread(target=run_movement)
        movement_thread.start()
    else:
        print("Gesture not recognized.")

# Load model and labels
model = YOLO(os.path.join(os.path.abspath(".."), configs.YOLO_MODEL))

# Start video capture
cap = cv2.VideoCapture(0)
drone.takeoff()
drone.hover()

try:
    results = model(source=0, stream=True)

    for orig in results:
        frame = orig.plot()
        cv2.imshow("YOLOv8 Webcam Inference", frame)
        labels = orig.names

        if len(orig.boxes) > 0:
            first_box = orig.boxes[0]
            class_id = int(first_box.cls)
            label = labels[class_id]
            print(f"Detected (first object): {label}")

            if label in gesture_to_cmd:
                gesture_text = f"{label}: {gesture_to_string[label]}"
                print(f"Detected gesture: {gesture_text}")

                cv2.putText(
                    frame,
                    gesture_text,
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )
                execute_command(label)
            else:
                print(f"No command mapped for: {label}")

        cv2.imshow("Frame With Gestures Displayed", frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    drone.land()
    drone.emergency_stop()
except Exception as e:
    print(e)
finally:
    print("\nLanding due to interruption...")
    drone.land()
    drone.emergency_stop()
    drone.close()
    cap.release()
    cv2.destroyAllWindows()
