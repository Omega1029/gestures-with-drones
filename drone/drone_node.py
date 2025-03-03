import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from codrone_edu.drone import Drone
import argparse


class CoDroneController(Node):
    def __init__(self, drone_id, port=None):
        super().__init__(f'codrone_controller_{drone_id}')
        self.drone_id = drone_id
        self.stored_position = None
        self.topic_name = f'/{drone_id}'

        self.command_subscription = self.create_subscription(
            String,
            self.topic_name,
            self.command_callback,
            1
        )

        self.drone = Drone()
        self.port = port
        if self.port:
            self.drone.pair(portname=self.port)
        else:
            self.drone.pair()

    def command_callback(self, msg):
        command = msg.data.lower()
        self.get_logger().info(f'Drone {self.drone_id} received command: {command}')

        if command == "takeoff":
            self.drone.takeoff()
            time.sleep(1)
            self.set_start_position()

        elif command.startswith("movement"):
            movement = msg.data.split(',')[1:]
            movement_dict = {k.strip(): float(v.strip()) for k, v in (item.split(':') for item in movement)}

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
            print(
                f"Stored Position: {self.stored_position[1:]} \n Current Position: {self.drone.get_position_data()[1:]}")
            self.drone.hover(1)
            #self.return_home()
            self.drone.land()
            self.drone.emergency_stop()

    def set_start_position(self):
        self.stored_position = list(self.drone.get_position_data())
        print(f"INFO: Drone {self.drone_id} setting stored position: {self.stored_position}")

    def return_home(self):
        if hasattr(self, 'stored_position') and self.stored_position:
            home_x, home_y, home_z = self.stored_position[1], self.stored_position[2], self.stored_position[3]
            self.get_logger().info(f'Drone {self.drone_id} returning home to: X={home_x}, Y={home_y}, Z={home_z}')

            tolerance = 0.5
            max_attempts = 200
            attempt = 0

            while attempt < max_attempts:
                current_position = self.drone.get_position_data()
                current_x, current_y, current_z = current_position[1], current_position[2], current_position[3]

                if (abs(current_x - home_x) <= tolerance and
                        abs(current_y - home_y) <= tolerance and
                        abs(current_z - home_z) <= tolerance):
                    self.get_logger().info(f"Drone {self.drone_id} has reached home position.")
                    break

                self.drone.send_absolute_position(home_x, home_y, home_z, 0.5, 0, 0)
                time.sleep(1)
                attempt += 1

        else:
            self.get_logger().warning(f"Drone {self.drone_id} home position not set. Cannot return home.")

    def emergency_stop(self):
        self.drone.land()
        self.drone.emergency_stop()


def main(args=None, drone_id='drone1', port=None, takeoff=False):
    rclpy.init(args=args)
    node = CoDroneController(drone_id, port)

    if takeoff:
        node.drone.takeoff()
        time.sleep(1)
        node.set_start_position()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info(f'Drone {drone_id}: Emergency stop triggered.')
        node.emergency_stop()
    except Exception as e:
        node.get_logger().error(f'Exception occurred: {e}')
        node.emergency_stop()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--drone_id', '-d', type=str, required=True, help='Unique ID for the drone (e.g., drone1, drone2)')
    ap.add_argument('--port', '-p', type=str, help='The serial port for the CoDrone')
    ap.add_argument('--takeoff', '-t', type=bool, default=False, required=False, help='Does this drone need to take off')

    args = vars(ap.parse_args())
    main(drone_id=args['drone_id'], port=args["port"], takeoff=args['takeoff'])
