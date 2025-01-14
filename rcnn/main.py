import time
import cv2
import numpy as np
import subprocess
import pickle
from tensorflow.keras.models import load_model

gesture_to_cmd = {
    "peace": (0, 0, 0, 0.1),
    "ok": (0.1, 0, 0, 0),
    "four": (-0.1, 0, 0, 0),
    "stop_inverted": (0.1, 0, 0, 0),
    "three2": (-0.1, 0, 0, 0),
    "two_up": (0.1, 0, 0, 0),
    "three": (-0.1, 0, 0, 0),
    "peace_inverted": (0.1, 0, 0, 0),
    "mute": (-0.1, 0, 0, 0),
    "palm": (0.1, 0, 0, 0),
    "rock": (-0.1, 0, 0, 0),
}
face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
PURPLE = (255, 0, 255)
# Helper function to run system commands
def run_command(command):
    try:
        result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        print("Command Output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error:", e.stderr)

# Function to build the move command based on gestures
def build_move_command(x=0, y=0, z=0, rotation=0):
    return f'''gz topic -t "/X3/gazebo/command/twist" -m gz.msgs.Twist -p "linear: {{x: {x}, y: {y}, z: {z}}} angular: {{z: {rotation}}}"'''

# Hover command
def build_hover_command():
    return '''gz topic -t "/X3/gazebo/command/twist" -m gz.msgs.Twist -p " "'''

# Function to pause execution
def rest(t=2):
    time.sleep(t)

#(x, y, z, rotation)
# Optimize gesture-to-command mapping with a dictionary

# Function to get the gesture command
def gesture_move(gesture="ok"):
    return build_move_command(*gesture_to_cmd.get(gesture, (0.1, 0, 0, 0)))  # Default to (0.1, 0, 0, 0)

# Preprocess the frame for model input
def preprocess_frame(frame):
    return np.expand_dims(cv2.resize(frame, (224, 224)), axis=0)

# Detect objects and return the label and confidence
def detect_objects(frame, model, labels):
    frame = cv2.flip(frame, 1)

    #Find Face
    faces = face_cascade.detectMultiScale(frame, 1.1, 5, 30)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), PURPLE, 2)
    #Draw Box Net Too Face
        #Grab These Coordinates
        #Detect/Predict and Process This Frame

    input_frame = preprocess_frame(frame)
    predictions = model.predict(input_frame)
    print(predictions)
    confidence_scores = predictions[0]
    class_id = np.argmax(confidence_scores)
    print(class_id)
    confidence = confidence_scores[class_id]
    label = labels.classes_[class_id]
    return label, confidence, frame

# Load model and labels
model = load_model("models/handsWithNoBoxes.keras")
with open("models/handsWithNoBoxesLabel.pickle", 'rb') as f:
    labels = pickle.load(f)

# Start video capture
cap = cv2.VideoCapture(0)

# Run hover command initially
run_command(build_hover_command())

try:
    while cap.isOpened():
        ret, frame = cap.read()
        time.sleep(0.5)
        if not ret:
            break

        # Measure the time taken for prediction
        start = time.time()
        faces = face_cascade.detectMultiScale(frame, 1.15, 5, 30)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), PURPLE, 2)
        label, confidence, frame = detect_objects(frame, model, labels)
        x, y, width, height = 100, 100, 50, 50
        cv2.putText(frame, f"{label}: {confidence:.2f}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 10)
        end = time.time()
        print(f"Time to detect Gesture: {end-start:.2f}")

        # Draw label and confidence on the frame


        # Show the frame
        cv2.imshow('Object Detection', frame)

        # Run the command for the detected gesture
        start = time.time()
        cmd = gesture_move(label)
        #run_command(cmd)
        end = time.time()
        print(f"Time to run command: {end-start:.2f}")

        # Exit on 'q' key
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

finally:
    cap.release()
    cv2.destroyAllWindows()
    run_command(build_hover_command())
