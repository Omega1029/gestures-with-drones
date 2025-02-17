import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64
from codrone_edu.drone import Drone
import argparse


class CoDroneController(Node):
    def __init__(self, port = None):
        super().__init__('codrone_controller')
        self.command_subscription = self.create_subscription(
            String,
            'gesture_command',
            self.command_callback,
            1)
        self.command_subscription
        self.drone = Drone()
        self.port = port
        if self.port:
            self.drone.pair(portname=self.port)
        else:
            self.drone.pair()


    def command_callback(self, msg):
        command = msg.data.lower()
        self.get_logger().info(f'Received command: {command}')

        if command == "takeoff":
            self.drone.takeoff()
        elif command.startswith("movement"):

            movement = msg.data.split(',')[1:]

            movement_dict = {k.strip(): float(v.strip()) for k, v in (item.split(':') for item in movement)}
            # Now you can use this data to control the drone
            self.get_logger().info(f"Parsed movement data: {movement_dict}")
            yaw = movement_dict.get("yaw", 0)
            pitch = movement_dict.get("pitch", 0)
            roll = movement_dict.get("roll", 0)
            throttle = movement_dict.get("throttle", 0)
            self.drone.set_pitch(pitch)
            self.drone.set_roll(roll)
            self.drone.set_throttle(throttle)
            self.drone.set_yaw(yaw)
            self.drone.move(2)  # Execute movement
            self.drone.hover(1)


        elif command == "stop":
            self.drone.land()
            self.drone.emergency_stop()
            self.drone.close()

    def publish_stop(self):
        stop_command = String()
        stop_command.data = "stop"
        self.command_subscription.publish(stop_command)

def main(args=None, port = None):
    rclpy.init(args=args)
    node = CoDroneController(port)
    try:
        rclpy.spin(node)  # Will continue until an exception occurs or node shutdown
    except KeyboardInterrupt:  # This will handle Ctrl+C or any other interruption
        node.get_logger().info('KeyboardInterrupt: Emergency stop triggered.')
        node.drone.emergency_stop()  # Trigger emergency stop on interruption
    except Exception as e:
        node.get_logger().error(f'Exception occurred: {e}')
        node.drone.emergency_stop()  # Trigger emergency stop on any exception
    finally:
        node.drone.emergency_stop()  # Ensure the drone stops before shutdown
        node.destroy_node()  # Cleanup
        rclpy.shutdown()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', '-p', type=str, help='The serial port for the CoDrone')
    args = vars(ap.parse_args())

    main(port = args["port"])

