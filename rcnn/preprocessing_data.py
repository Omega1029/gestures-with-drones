# initialize the list of data (images), class labels, target bounding
# box coordinates, and image paths
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import os, json, cv2
import tensorflow as tf
from tensorflow.keras import layers

print("[INFO] loading dataset...")




def load_images_and_labels(limit=100):
    data = []
    labels = []
    # bboxes = []
    imagePaths = []
    ANNOTATIONS_ROOT = "datasets/hands"

    annotations_dir = os.listdir(ANNOTATIONS_ROOT)
    # for annotation_dir in annotations_dir:
    # print(annotation_dir)
    updated = 0
    # print("[INFO] Loading Annotations...")
    annotation_dir = "ann_train_val"
    ann_sub_dir = os.path.join(ANNOTATIONS_ROOT, annotation_dir)
    for gesture in os.listdir(ann_sub_dir):
        gesture_file = os.path.join(ann_sub_dir, gesture)
        with open(gesture_file, 'r') as f:
            annotation_data = json.loads(f.read())
            filenames = annotation_data.keys()
            gesture_filename = gesture.split('.')[0]
            for image_filename in list(filenames)[:limit]:
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
                    #bboxes_annotations = annotation_data[image_filename]['bboxes']
                    labels_annotations = annotation_data[image_filename]['labels']
                    # input(labels)
                    i = 0
                    for label in labels_annotations:
                        if label != "no_gesture":
                            i += 1
                            #x_center, y_center, bbox_width, bbox_height = bbox

                            # Convert normalized coordinates to pixel values
                            # x_min = int((x_center - bbox_width / 2) * width)
                            # y_min = int((y_center - bbox_height / 2) * height)
                            # x_max = int((x_center + bbox_width) * width)
                            # y_max = int((y_center + bbox_height) * height)

                            # if updated % 1000 == 0:
                            #     print(f"Bounding box coordinates: (x_min, y_min) = ({x_min}, {y_min}), (x_max, y_max) = ({x_max}, {y_max})")

                            # Draw the bounding box on the image
                            # cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                            # Put the label text above the bounding box
                            #text_position = (x_min, y_min - 10)  # Position the text above the box
                            # cv2.putText(img, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3,
                            #            cv2.LINE_AA)
                            # cv2.put

                            image = load_img(image_path, target_size=(224, 224))
                            image = img_to_array(image)
                            # update our list of data, class labels, bounding boxes, and
                            # image paths
                            scale_x = 224 / width
                            scale_y = 224 / height

                            # Adjust bounding box coordinates
                            # x_min_resized = int(x_min * scale_x)
                            # y_min_resized = int(y_min * scale_y)
                            # x_max_resized = int(x_max * scale_x)
                            # y_max_resized = int(y_max * scale_y)

                            # Update lists
                            data.append(image)
                            labels.append(label)
                            #bboxes.append((x_min_resized, y_min_resized, x_max_resized, y_max_resized))
                            imagePaths.append(image_path)
                            # data.append(image)
                            # labels.append(label)
                            # bboxes.append((x_min, y_min, x_max, y_max))
                            # imagePaths.append(image_path)

                        # Display the image with bounding boxes
                        # cv2.imshow("Image", img)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()
                        # print("Displayed:", image_path)

                        # Print the first bounding box coordinates as percentages
                        if 'bboxes' in annotation_data[image_filename]:
                            box1, box2, box3, box4 = annotation_data[image_filename]['bboxes'][0]
                            # if updated % 1000 == 0:
                            #     print(f"Bounding box (normalized): x_center = {box1 * 100:.2f}%, y_center = {box2 * 100:.2f}%")
                else:
                    pass
                    # print(f"Error: Could not read image at {image_path}")
                    # os.remove(image_path)

    data = np.array(data, dtype="float32") / 255.0
    labels = np.array(labels)
    # bboxes = np.array(bboxes, dtype="float32")
    imagePaths = np.array(imagePaths)
    print("[INFO] Annoatations Loaded...")
    return (data, labels, imagePaths)



def create_label_binarizer(lb, labels):


    return lb.fit_transform(labels)

# print(labels)

def create_data_split(data, labels, imagePaths):
    split = train_test_split(data, labels, imagePaths,
                             test_size=0.20, random_state=42)

    #print(split)
    # unpack the data split


    return split


if __name__ =='__main__':
    data, labels, imagePaths = load_images_and_labels()
    lb = LabelBinarizer()
    labels = create_label_binarizer(lb, labels)

    split = create_data_split(data, labels, imagePaths)

    (trainImages, testImages) = split[:2]
    (trainLabels, testLabels) = split[2:4]
    # (trainBBoxes, testBBoxes) = split[4:6]
    (trainPaths, testPaths) = split[4:]



    # Build the CNN model
    model = tf.keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(len(lb.classes_), activation='softmax')
    ])
    # Compile the model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])
    # Train the model
    model.fit(trainImages, trainLabels, epochs=10, batch_size=2, validation_data=(testImages, testLabels))

    print("[INFO] evaluating network...")
    predIdxs = model.predict(x=testImages.astype("float32"), batch_size=2)

    print(predIdxs)
    # for each image in the testing set we need to find the index of the
    # label with corresponding largest predicted probability

    # Ensure that predIdxs is consistent before applying np.argmax
    predIdxs = np.argmax(predIdxs, axis=1)
    # input("Unexpected output shape. Predicted indices not computed.")
    # input(predIdxs)

    print(testLabels.argmax(axis=1))
    print(predIdxs)
    print(lb.classes_)
    # show a nicely formatted classification report
    print(classification_report(testLabels.argmax(axis=1),
                                predIdxs,
                                labels=lb.classes_))

# only there are only two labels in the dataset, then we need to use
# Keras/TensorFlow's utility function as well


