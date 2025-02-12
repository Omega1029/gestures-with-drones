# Gestures with Drones

## Overview

This project focuses on controlling drones using hand gestures, leveraging computer vision and machine learning techniques to interpret human hand movements for intuitive drone navigation.

## Features

- **Hand Gesture Recognition**: Utilizes advanced algorithms to detect and classify various hand gestures in real-time.
- **Drone Control Integration**: Translates recognized gestures into drone commands, enabling touchless operation.
- **Modular Design**: Allows for easy updates and integration with different drone models and gesture sets.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Omega1029/gestures-with-drones.git
   cd gestures-with-drones
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.x installed. Then, install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Settings**:
   Update the `configs.py` file with your specific settings, such as camera source, drone connection details, and gesture configurations.

## Usage

1. **Prepare the Environment**:
   - Ensure your drone is fully charged and connected.
   - Set up your camera to capture hand gestures.

2. **Run the Application**:

   ### A) Running Just the YOLO Model
   ```bash
   cd yolo
   python3 testing_yolov8_model.py
   ```
   
   ### B) Running the Drone in Gazebo
   **Note**: Gazebo 7 must be installed first.
   ```bash
   cd drone
   gz sim -v 4 -g
   gz sim -v 4 drone/worlds/multicoptercontrol.sdf -s
   python3 drone_with_gestures.py
   ```
   
   ### C) Running with CoDrone EDU
   **Note**: You must set up CoDrone EDU.
   ```bash
   cd drone
   python3 codrone_edu_with_gestures.py
   ```

3. **Perform Gestures**:
   Use predefined hand gestures to control the drone. For example:
   - **Open Palm**: Take off
   - **Closed Fist**: Land
   - **Swipe Left**: Move left
   - **Swipe Right**: Move right

   *(Customize these gestures in the configuration file as needed.)*

## Project Structure

- `drone/`: Contains drone control modules and interfaces.
- `models/`: Includes pre-trained models for gesture recognition.
- `rcnn/` and `yolo/`: Directories for different object detection algorithms.
- `configs.py`: Configuration settings for the project.
- `requirements.txt`: List of required Python packages.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your enhancements or bug fixes. Ensure that your code adheres to the project's coding standards and includes appropriate tests.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgments

- [hukenovs/hagrid](https://github.com/hukenovs/hagrid)
- [Faster R-CNNs - PyImageSearch](https://pyimagesearch.com/2023/11/13/faster-r-cnns/)
- [Object Tracking with YOLOv8 and Python - PyImageSearch](https://pyimagesearch.com/2024/06/17/object-tracking-with-yolov8-and-python/)
- [e2eet-skeleton-based-hgr-using-data-level-fusion](https://github.com/outsiders17711/e2eet-skeleton-based-hgr-using-data-level-fusion)

## Contact

For questions or suggestions, please open an issue in this repository or contact the maintainer at [your-email@example.com].
