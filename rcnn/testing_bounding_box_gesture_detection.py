import json
import os
import time
import cv2
import numpy as np
import subprocess
import pickle

from keras.src.utils import load_img, img_to_array
from tensorflow.keras.models import load_model
from ious import compute_iou

#Load Model
#Load Dataset

#Pick A Random Image
#Crop Gesture
#Classify

model = load_model("models/handsWithBoxes.keras", custom_objects={'custom_classification_loss':'custom_classification_loss', 'custom_bbox_loss':'custom_bbox_loss'})
with open("models/handsWithBoxesLabel.pickle", 'rb') as f:
    labels = pickle.load(f)

print("[INFO] Model Loaded....")
ANNOTATIONS_ROOT = "datasets/hands"

annotations_dir = os.listdir(ANNOTATIONS_ROOT)
# for annotation_dir in annotations_dir:
# print(annotation_dir)
updated = 0
#print("[INFO] Loading Annotations...")
annotation_dir = "ann_train_val"
ann_sub_dir = os.path.join(ANNOTATIONS_ROOT, annotation_dir)
for gesture in os.listdir(ann_sub_dir):
    gesture_file = os.path.join(ann_sub_dir, gesture)
    with open(gesture_file, 'r') as f:
        annotation_data = json.loads(f.read())
        filenames = annotation_data.keys()
        gesture_filename = gesture.split('.')[0]
        for image_filename in list(filenames)[:5]:
            updated += 1
            # input(gesture_filename+"/"+image_filename+".jpg")
            # filename = filename + ".jpg"
            image_path = os.path.join("datasets/hands", gesture_filename, image_filename + ".jpg")
            #input(image_path)
            if not os.path.exists(image_path):
                continue
            img = cv2.imread(image_path)
            if img is not None:
                height, width = img.shape[:2]
                # if updated % 1000 == 0:# Get the height and width of the image
                #     print(f"Image dimensions: Height = {height}, Width = {width}")
                #     pprint(annotation_data[image_filename])
                # Get the bounding boxes and labels
                bboxes_annotations = annotation_data[image_filename]['bboxes']
                labels_annotations = annotation_data[image_filename]['labels']
                # input(labels)
                i = 0
                for bbox, label in zip(bboxes_annotations, labels_annotations):

                    i += 1
                    x_center, y_center, bbox_width, bbox_height = bbox

                    #Convert normalized coordinates to pixel values
                    x_min = int((x_center - bbox_width / 3) * width)
                    y_min = int((y_center - bbox_height / 3) * height)
                    x_max = int((x_center + bbox_width) * width)
                    y_max = int((y_center + bbox_height) * height)

                    if updated % 1000 == 0:
                        print(f"Bounding box coordinates: (x_min, y_min) = ({x_min}, {y_min}), (x_max, y_max) = ({x_max}, {y_max})")

                    #Draw the bounding box on the image
                    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                    #Put the label text above the bounding box
                    text_position = (x_min, y_min - 10)  # Position the text above the box
                    #cv2.putText(img, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3,
                    #           cv2.LINE_AA)
                    # cv2.put

                    #hand = img[y_min:y_max, x_min:x_max]


                    image = load_img(image_path, target_size=(224, 224))

                    image = img_to_array(image)
                    # update our list of data, class labels, bounding boxes, and
                    # image paths
                    #scale_x = 224 / width
                    #scale_y = 224 / height

                    # Adjust bounding box coordinates
                    # x_min_resized = int(x_min * scale_x)
                    # y_min_resized = int(y_min * scale_y)
                    # x_max_resized = int(x_max * scale_x)
                    # y_max_resized = int(y_max * scale_y)

                    # Update lists

                    # data.append(image)
                    # labels.append(label)
                    # bboxes.append((x_min, y_min, x_max, y_max))
                    # imagePaths.append(image_path)

                # Display the image with bounding boxes

                #cv2.imshow("Image", img)
                #try:
                    #cv2.imshow("Hand", hand)
                    #hand = cv2.resize(hand, (224, 224))
                    # hand_predictions = cv2.resize(hand, (224, 224))
                image = np.expand_dims(image, axis=0)
                predictions = model.predict(image)
                #input(predictions)
                coordinates = predictions[0][0]
                label_predicted = labels.classes_[np.argmax(predictions[-1][0])]
                print("Actual: {}, Predicted: {}".format(label, label_predicted))
                print(coordinates)
                iou = compute_iou(coordinates, [x_min, y_min, x_max, y_max])
                print("IOU: {}".format(iou))




                #except:
                #    pass
                #cv2.waitKey(0)
                time.sleep(2)

                cv2.destroyAllWindows()
                #print("Displayed:", image_path)


            else:
                pass
                # print(f"Error: Could not read image at {image_path}")
                # os.remove(image_path)
#quit()
print("[INFO] Annoatations Loaded...")