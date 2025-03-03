import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from codrone_edu.drone import *
import os
import cv2
import numpy as np
from ultralytics import YOLO
import configs

class Talker(Node):
    def __init__(self):
        super().__init__('talker')
        self.selected_drone = "drone1"  # Default drone
        self.command_publisher_ = self.create_publisher(String, f'/{self.selected_drone}', 10)
        self.movement_publisher_ = self.create_publisher(String, f'/{self.selected_drone}', 30)

    def update_selected_drone(self, drone_id):
        """Updates the selected drone based on gesture."""
        self.selected_drone = f'{drone_id}'
        self.command_publisher_ = self.create_publisher(String, f'/{self.selected_drone}', 10)
        self.movement_publisher_ = self.create_publisher(String, f'/{self.selected_drone}', 30)
        self.get_logger().info(f'Selected {self.selected_drone}')

    def publish_command(self, command_value):
        command = String()
        command.data = command_value
        self.command_publisher_.publish(command)
        self.get_logger().info(f'Publishing: "{command.data}" to {self.selected_drone}')

    def publish_movement(self, movement_value):
        movement = String()
        movement.data = f"movement, yaw: {movement_value['yaw']}, pitch: {movement_value['pitch']}, roll: {movement_value['roll']}, " \
                        f"throttle: {movement_value['throttle']}"
        self.movement_publisher_.publish(movement)
        self.get_logger().info(f'Publishing: "{movement.data}" to {self.selected_drone}')

SPEED = 30

def adjust_speed(gesture):
    global SPEED
    if gesture == "four":
        SPEED = min(SPEED + 5, 90)
        print(f"Speed increased to {SPEED}")
    elif gesture == "one":
        SPEED = max(20, SPEED - 5)
        print(f"Speed decreased to {SPEED}")

def get_gesture_command(gesture):
    gesture_to_cmd = {
        "like": (0, 0, SPEED, 0),
        "dislike": (0, 0, -SPEED, 0),
        "fist": (-SPEED, 0, 0, 0),
        "palm": (SPEED, 0, 0, 0),
        "three": (0, SPEED, 0, 0),
        "two_up": (0, -SPEED, 0, 0),
        "peace": (0, 0, 0, SPEED),
        "ok": (0, 0, 0, -SPEED),
        "stop": (0, 0, 0, 0),
    }
    return gesture_to_cmd.get(gesture, (0, 0, 0, 0))


gesture_to_string = {
    "peace": "Rotate Left",
    "ok": "Rotate Right",
    "fist": "Backward",
    "palm": "Forward",
    "three": "Right",
    "two_up": "Left",
    "like": "Up",
    "dislike": "Down",
    "stop": "Ending...",
    "four": "Increase Speed by 5",
    "mute": "Sounding Buzzer",
    "call": "Hover",

    # Drone Selection
    "rock": "Select Drone 1",
    "peace_inverted": "Select Drone 2",
    "stop_inverted": "Select Drone 3"
}

gesture_to_drone = {
    "rock": "drone1",
    "peace_inverted": "drone2",
    "stop_inverted": "drone3"
}


def execute_command(gesture):
    if gesture == "stop":
        print("Stopping all drones...")
        node.publish_command("stop")  # Broadcast stop command to all drones
        return

    elif gesture in gesture_to_drone:
        node.update_selected_drone(gesture_to_drone[gesture])

    elif gesture in gesture_to_string:
        pitch, roll, throttle, yaw = get_gesture_command(gesture)
        direction = {"yaw": yaw, "pitch": pitch, "roll": roll, "throttle": throttle}
        node.publish_movement(direction)



model = YOLO(os.path.join(os.path.abspath(".."), configs.YOLO_MODEL))
rclpy.init(args=None)
node = Talker()
cap = cv2.VideoCapture(0)
#node.publish_command("takeoff")

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
                cv2.putText(frame, gesture_text, (50, 75), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 4, cv2.LINE_AA)
                execute_command(label)


        cv2.imshow("Gesture Detection", frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    rclpy.spin_once(node)

except KeyboardInterrupt:
    pass
except Exception as e:
    print(e)
finally:
    print("\nLanding due to interruption...")
    node.publish_command("stop")
    node.destroy_node()
    rclpy.shutdown()
    cap.release()
    cv2.destroyAllWindows()
