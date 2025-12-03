"""
I)
    pip3 install the following:

    opencv-python
    numpy
    ultralytics

II) Change Lines:
    -MODEL_PATH: to your model path

    -if label == "Organic" or label == "Aluminum can":
        if label in ["CLass1", "Class2"]
    Or you can just use one class (recommended)

"""

import os
import threading
import time
import cv2
import numpy as np
import subprocess
from ultralytics import YOLO

#######################################
# SETTINGS
#######################################

SPEED = 0.5
MODEL_PATH = "best_model.pt"   # <-- CHANGE THIS TO YOUR MODEL PATH

# Linear_x, Angular_z for WHEELED ROBOT
gesture_to_cmd = {
    "Forward":  (SPEED, 0),
    "Backward": (-SPEED, 0),
    "Right":    (SPEED/2, -SPEED),   # turn clockwise
    "Left":     (SPEED/2, SPEED),    # turn counterclockwise
    "Hover":    (0, 0),        # stop
}

#######################################
# HELPER FUNCTIONS
#######################################

def run_command(command):
    """Run a system command asynchronously."""
    try:
        result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        print("Command Output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Command Error:", e.stderr)


def build_move_command(x=0, rotation=0, topic = "/X3/gazebo/command/twist"):
    """Build a Gazebo twist command for a ground robot (x + angular z)."""
    return (
        f'''gz topic -t "{topic}" -m gz.msgs.Twist '''
        f'''-p "linear: {{x: {x}, y: 0, z: 0}} angular: {{z: {rotation}}}"'''
    )
    #gz topic -t "/model/vehicle_blue/cmd_vel" -m gz.msgs.Twist -p "linear: {x: 0.5}, angular: {z: 0.1}"#


def hover_command(topic="/X3/gazebo/command/twist"):
    """Send hover/stop command."""

    return f'''gz topic -t "{topic}" -m gz.msgs.Twist -p " "'''


def draw_direction(frame, direction_name):
    """Overlay direction text + arrow on the frame."""
    h, w = frame.shape[:2]

    # Draw the text
    cv2.putText(frame, f"Direction: {direction_name}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    # Arrow start point (screen center)
    cx, cy = w // 2, h // 2

    # Default arrow endpoint
    end = (cx, cy)

    if direction_name == "Forward":
        end = (cx, cy - 120)
    elif direction_name == "Backward":
        end = (cx, cy + 120)
    elif direction_name == "Left":
        end = (cx - 120, cy)
    elif direction_name == "Right":
        end = (cx + 120, cy)

    # Draw arrow
    cv2.arrowedLine(frame, (cx, cy), end, (0, 255, 0), 4, tipLength=0.4)
    return frame

#######################################
# QUADRANT LOGIC
#######################################

def get_quadrant(x1, y1, x2, y2, frame_w, frame_h):
    """Return quadrant based on center point of bounding box."""
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)

    w1 = frame_w // 3
    w2 = 2 * frame_w // 3
    h1 = frame_h // 3
    h2 = 2 * frame_h // 3

    # Columns
    if cx < w1: col = "L"
    elif cx < w2: col = "C"
    else: col = "R"

    # Rows
    if cy < h1: row = "T"
    elif cy < h2: row = "M"
    else: row = "B"

    return row + col


def movement_from_quadrant(q):
    """Map quadrant to movement direction for a ground robot."""
    if q == "MC":
        return "Forward"

    if q == "ML": return "Left"
    if q == "MR": return "Right"

    if q.startswith("T"): return "Backward"
    if q.startswith("B"): return "Backward"

    if q.endswith("L"): return "Left"
    if q.endswith("R"): return "Right"

    return "Hover"


#######################################
# MAIN
#######################################

def main():
    print("Loading YOLO model...")
    model = YOLO(MODEL_PATH)

    print("Starting webcam...")
    cap = cv2.VideoCapture(0)

    # Hover initially
    run_command(hover_command())

    try:
        for result in model(source=0, stream=True):
            frame = result.plot()  # YOLO annotated image

            # No detections → stop
            if len(result.boxes) == 0:
                direction = "Hover"
                frame = draw_direction(frame, direction)
                run_command(hover_command(topic="/model/vehicle_blue/cmd_vel"))
            else:
                box = result.boxes[0]  # only track first detection
                cls_id = int(box.cls)
                label = result.names[cls_id]

                print(f"Detected: {label}")

                # Change this to YOUR CLASS
                if label:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    frame_h, frame_w = frame.shape[:2]

                    q = get_quadrant(x1, y1, x2, y2, frame_w, frame_h)
                    print("Quadrant:", q)

                    direction = movement_from_quadrant(q)
                    print("Direction:", direction)

                    # Draw arrow
                    frame = draw_direction(frame, direction)

                    # movement tuple
                    linear_x, angular_z = gesture_to_cmd[direction]

                    cmd = build_move_command(linear_x, angular_z, topic="/model/vehicle_blue/cmd_vel")
                    threading.Thread(target=run_command, args=(cmd,)).start()
                else:
                    direction = "Hover"
                    frame = draw_direction(frame, direction)
                    run_command(hover_command())

            cv2.imshow("Trash Tracking Robot", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass

    finally:
        cap.release()
        cv2.destroyAllWindows()
        run_command(hover_command())
        print("Program ended. Robot stopped.")


if __name__ == "__main__":
    main()
