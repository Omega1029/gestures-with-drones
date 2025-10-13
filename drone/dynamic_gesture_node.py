import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import cv2
import numpy as np
import argparse
import time
import sys, os
import configs
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynamic_gestures.utils import Drawer, Event, targets
from dynamic_gestures.main_controller import MainController

# ============================================================
# ROS2 Talker Node
# ============================================================
class Talker(Node):
    def __init__(self):
        super().__init__('talker')
        self.selected_drone = "drone1"
        self.cmd_pub = self.create_publisher(String, f'/{self.selected_drone}', 10)
        self.move_pub = self.create_publisher(String, f'/{self.selected_drone}', 30)
        self.get_logger().info("Gesture Talker Node Initialized")

    def update_selected_drone(self, drone_id):
        self.selected_drone = f'{drone_id}'
        self.cmd_pub = self.create_publisher(String, f'/{self.selected_drone}', 10)
        self.move_pub = self.create_publisher(String, f'/{self.selected_drone}', 30)
        self.get_logger().info(f"Switched to {self.selected_drone}")

    def publish_command(self, command):
        msg = String()
        msg.data = command
        self.cmd_pub.publish(msg)
        self.get_logger().info(f'Command: "{command}" → {self.selected_drone}')

    def publish_movement(self, move_dict):
        msg = String()
        msg.data = f"movement, yaw: {move_dict['yaw']}, pitch:{ move_dict['pitch']}, roll: {move_dict['roll']}, throttle: {move_dict['throttle']}"


        self.move_pub.publish(msg)
        self.get_logger().info(f'Movement: "{msg.data}" → {self.selected_drone}')


# ============================================================
# Gesture/Action to Command Mappings
# ============================================================
SPEED = 30

# Static gestures (select drones or modes)
gesture_to_drone = {
    "rock": "drone1",
    "peace_inverted": "drone2",
    "stop_inverted": "drone3"
}

# Dynamic actions → drone flight behavior
action_to_movement = {
    Event.SWIPE_LEFT:      {"yaw": 0, "pitch": 0, "roll": -SPEED, "throttle": 0},   # Move Left
    Event.SWIPE_RIGHT:     {"yaw": 0, "pitch": 0, "roll": SPEED, "throttle": 0},    # Move Right
    Event.SWIPE_UP:        {"yaw": 0, "pitch": SPEED, "roll": 0, "throttle": 0},    # Forward
    Event.SWIPE_DOWN:      {"yaw": 0, "pitch": -SPEED, "roll": 0, "throttle": 0},   # Backward
    Event.ZOOM_IN:         {"yaw": 0, "pitch": 0, "roll": 0, "throttle": SPEED},    # Ascend
    Event.ZOOM_OUT:        {"yaw": 0, "pitch": 0, "roll": 0, "throttle": -SPEED},   # Descend
    #Event.CLOCKWISE:       {"yaw": SPEED, "pitch": 0, "roll": 0, "throttle": 0},    # Rotate CW
    #Event.COUNTERCLOCK:    {"yaw": -SPEED, "pitch": 0, "roll": 0, "throttle": 0},   # Rotate CCW
    Event.DOUBLE_TAP:      "takeoff",                                               # Takeoff
    Event.FAST_SWIPE_DOWN: "land",                                                  # Land
    Event.TAP:             "hover",                                                 # Hover / stop motion
}


# ============================================================
# Execute gesture or dynamic action
# ============================================================
def execute_event(node, action):
    """Map gesture event to drone command or motion."""
    if action is None:
        return

    if action in action_to_movement:
        mapping = action_to_movement[action]

        # Movement or Command?
        if isinstance(mapping, dict):
            node.publish_movement(mapping)
        else:
            node.publish_command(mapping)

    else:
        print(f"Unmapped action: {action}")


# ============================================================
# Main Function
# ============================================================
def run(args):
    rclpy.init(args=None)
    node = Talker()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    controller = MainController(os.path.join(os.path.abspath(".."), args.detector), os.path.join(os.path.abspath(".."), args.classifier))
    drawer = Drawer()
    print("Dynamic Gesture Drone Controller Started...")

    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if not ret:
            break

        start = time.time()
        bboxes, ids, labels = controller(frame)

        if bboxes is not None:
            for i in range(len(bboxes)):
                box = bboxes[i].astype(int)
                gesture = targets[labels[i]] if labels[i] is not None else None
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 2)
                if gesture:
                    cv2.putText(frame, f"{gesture}", (box[0], box[1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Check for recognized dynamic actions
        for trk in controller.tracks:
            if trk["tracker"].time_since_update < 1 and trk["hands"].action is not None:
                execute_event(node, trk["hands"].action)
                trk["hands"].action = None  # Reset after execution

        fps = 1.0 / (time.time() - start)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Dynamic Gesture Drone Control", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    node.publish_command("land")
    node.destroy_node()
    rclpy.shutdown()
    cap.release()
    cv2.destroyAllWindows()


# ============================================================
# Entry
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dynamic Gesture Drone Control")
    parser.add_argument("--detector", default=configs.DETECTOR_MODEL_PATH, type=str)
    parser.add_argument("--classifier", default=configs.CLASSIFIER_MODEL_PATH, type=str)
    parser.add_argument("--debug", action="store_true", default=True, required=False)
    args = parser.parse_args()
    run(args)
