import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64
from codrone_edu.drone import Drone
import argparse


class CoDroneController(Node):
    def __init__(self, port = None):
        super().__init__('codrone_controller')
        self.stored_position = None
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
            time.sleep(1)
            self.set_start_position()
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

        elif command == "four":
            self.drone.flip("back")
        elif command == "one":
            self.drone.flip("right")

        elif command == "stop":
            print("Stored Position :{} \n Current Position: {}".format(self.stored_position[1:], self.drone.get_position_data()[1:]))
            self.drone.hover(2)
            self.return_home()
            self.drone.land()
            self.drone.emergency_stop()
            #self.drone.close()

    def publish_stop(self):
        stop_command = String()
        stop_command.data = "stop"
        self.command_subscription.publish(stop_command)

    def set_start_position(self):
        self.stored_position = list(self.drone.get_position_data())
        print("INFO Setting STORED POSITION: {}".format(self.stored_position))

    def return_home(self):
        if hasattr(self, 'stored_position') and self.stored_position:
            home_x, home_y, home_z = self.stored_position[1], self.stored_position[2], self.stored_position[3]
            self.get_logger().info(f'Returning home to: X={home_x}, Y={home_y}, Z={home_z}')

            tolerance = 0.05  # Allowable error in meters
            max_attempts = 200  # Safety limit to avoid infinite loop
            attempt = 0

            while attempt < max_attempts:
                current_position = self.drone.get_position_data()
                current_x, current_y, current_z = current_position[1], current_position[2], current_position[3]

                # Check if drone is close enough to home
                if (abs(current_x - home_x) <= tolerance and
                        abs(current_y - home_y) <= tolerance and
                        abs(current_z - home_z) <= tolerance):
                    self.get_logger().info("Drone has reached home position.")
                    break  # Exit loop when drone is home

                # Move towards home position
                self.drone.send_absolute_position(home_x, home_y, home_z, 0.5, 0, 0)
                time.sleep(1)  # Allow time for movement
                attempt += 1

        else:
            self.get_logger().warning("Home position not set. Cannot return home.")

        return


def main(args=None, port = None, takeoff = False):
    rclpy.init(args=args)
    node = CoDroneController(port)
    if takeoff:
        node.drone.takeoff()
        time.sleep(1)
        node.set_start_position()
    try:
        rclpy.spin(node)  # Will continue until an exception occurs or node shutdown

    except KeyboardInterrupt:  # This will handle Ctrl+C or any other interruption
        node.get_logger().info('KeyboardInterrupt: Emergency stop triggered.')
        node.drone.land()
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
    ap.add_argument('--takeoff', '-t', type=bool, default=False, required=False, help='Does this drone need to take off')
    args = vars(ap.parse_args())

    main(port = args["port"], takeoff = args['takeoff'])

