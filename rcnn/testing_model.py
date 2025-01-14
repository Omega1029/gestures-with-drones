import json
import os
import cv2
import numpy as np
import pickle
from keras.src.applications.mobilenet_v2 import preprocess_input
from keras.src.utils import load_img, img_to_array
from tensorflow.keras.models import load_model

# Load Model and Labels
model = load_model("models/handsClassifier.keras")
with open("models/handsWithNoBoxesLabel.pickle", 'rb') as f:
    labels = pickle.load(f)

# Constants
ANNOTATIONS_ROOT = "datasets/hands"
annotation_dir = "ann_train_val"
ann_sub_dir = os.path.join(ANNOTATIONS_ROOT, annotation_dir)
updated = total = correct = 0

# Process each gesture annotation file
for gesture in os.listdir(ann_sub_dir):
    gesture_file = os.path.join(ann_sub_dir, gesture)
    with open(gesture_file, 'r') as f:
        annotation_data = json.loads(f.read())
        filenames = list(annotation_data.keys())
        gesture_filename = gesture.split('.')[0]

        for image_filename in filenames[:50]:  # Limit to first 50 files
            total += 1
            updated += 1
            image_path = os.path.join("datasets/hands", gesture_filename, f"{image_filename}.jpg")

            if not os.path.exists(image_path):
                continue  # Skip if image doesn't exist

            img = cv2.imread(image_path)
            if img is None:
                print(f"Error reading image at {image_path}")
                continue  # Skip if the image can't be read

            height, width = img.shape[:2]
            bboxes_annotations = annotation_data[image_filename]['bboxes']
            labels_annotations = annotation_data[image_filename]['labels']

            for bbox, label in zip(bboxes_annotations, labels_annotations):
                if label == "no_gesture":  # Skip empty gesture labels
                    continue

                x_center, y_center, bbox_width, bbox_height = bbox
                x_min = int((x_center - bbox_width / 3) * width)
                y_min = int((y_center - bbox_height / 3) * height)
                x_max = int((x_center + bbox_width) * width)
                y_max = int((y_center + bbox_height) * height)

                # Crop and resize the image to the bounding box
                hand = img[y_min:y_max, x_min:x_max]

                try:
                    # Preprocess image for model prediction

                    hand = cv2.resize(hand, (224, 224))
                    hand = np.expand_dims(hand, axis=0)
                    hand = preprocess_input(hand)

                    predictions = model.predict(hand, verbose =0)
                    label_predicted = labels.classes_[np.argmax(predictions)]

                    #print(f"Actual: {label}, Predicted: {label_predicted}")

                    if label == label_predicted:
                        correct += 1

                except Exception as e:
                    print(f"Error processing image {image_filename}: {e}")
                    continue  # Skip if there's an error during prediction

    print(f"Finished processing {gesture_filename}")

# Print final results
print("[INFO] Results: {:.2f}% accuracy".format((correct / total) * 100))
