from codrone_edu.drone import *
import os
import cv2
import numpy as np
import threading
from ultralytics import YOLO
import configs

drone = Drone()
drone.pair()  # Pair automatically, may not always work

# Speed control
SPEED = 30

def adjust_speed(gesture):
    """Adjusts the drone's speed based on the detected gesture."""
    global SPEED
    if gesture == "four":
        SPEED = min(SPEED + 5, 90)
        print(f"Speed increased to {SPEED}")
    elif gesture == "one":
        SPEED = max(20, SPEED - 5)  # Prevents SPEED from going below 5
        print(f"Speed decreased to {SPEED}")


def get_gesture_command(gesture):
    """Dynamically returns the movement tuple based on the updated SPEED."""
    gesture_to_cmd = {
        "like": (0, 0, SPEED, 0),  # Up
        "dislike": (0, 0, -SPEED, 0),  # Down
        "fist": (-SPEED, 0, 0, 0),  # Backward
        "palm": (SPEED, 0, 0, 0),  # Forward
        "three": (0, SPEED, 0, 0),  # Right
        "two_up": (0, -SPEED, 0, 0),  # Left
        "peace": (0, 0, 0, SPEED),  # Rotate Right
        "ok": (0, 0, 0, -SPEED),  # Rotate Left
        "stop": (0, 0, 0, 0),  # Hover
    }
    return gesture_to_cmd.get(gesture, (0, 0, 0, 0))  # Default to hover

gesture_to_string = {
    "peace": "Rotate Right",
    "ok": "Rotate Left",
    "fist": "Backward",
    "palm": "Forward",
    "three": "Right",
    "two_up": "Left",
    "like": "Up",
    "dislike": "Down",
    "stop": "Hover",
    "four": "Increase Speed by 5",
    "one": "Decrease Speed by 5",
    "mute":"Sounding Buzzer",
    "call":"Ending..."
}

def execute_command(gesture):
    """Executes the drone movement based on the gesture asynchronously."""
    adjust_speed(gesture)  # Adjust speed before getting the command

    if gesture in gesture_to_string and gesture not in ["four", "one", "call", "mute"]:  # Ignore speed adjustment gestures
        pitch, roll, throttle, yaw = get_gesture_command(gesture)

        def run_movement():
            drone.set_pitch(pitch)
            drone.set_roll(roll)
            drone.set_throttle(throttle)
            drone.set_yaw(yaw)
            drone.move(2)  # Execute movement
            drone.hover(1)  # Stop motion

        movement_thread = threading.Thread(target=run_movement)
        movement_thread.start()
    elif gesture == "mute":
        drone.drone_buzzer(400, 300)

    elif gesture == "call":
        print("STOPPING")
        raise Exception
    else:
        print(f"Speed changed, no direct movement for gesture: {gesture}")

# Load YOLO model
model = YOLO(os.path.join(os.path.abspath(".."), configs.YOLO_MODEL))

# Start video capture
cap = cv2.VideoCapture(0)
drone.takeoff()
drone.hover()

try:
    results = model(source=0, stream=True)

    for orig in results:
        frame = orig.plot()
        labels = orig.names

        if len(orig.boxes) > 0:
            first_box = orig.boxes[0]
            class_id = int(first_box.cls)
            label = labels[class_id]
            print(f"Detected: {label}")

            if label in gesture_to_string:

                gesture_text = f"{label}: {gesture_to_string[label]}"
                print(f"Executing gesture: {gesture_text}")

                cv2.putText(frame, gesture_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 0, 0), 2, cv2.LINE_AA)
                execute_command(label)

        cv2.imshow("Gesture Detection", frame)
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
