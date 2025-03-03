import random
import time
from codrone_edu.drone import Drone
drone = Drone()

drone.pair()



drone.emergency_stop()  # Stop the drone in case of an error
drone.land()  # Ensure the drone lands safely
drone.close()
