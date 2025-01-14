# import the necessary packages
# from pyimagesearch import config
import json
from pprint import pprint
import tensorflow as tf
from tensorflow.keras.applications import VGG16, MobileNetV2, MobileNetV3Small, ResNet101V2, EfficientNetV2S
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import os
from sklearn.metrics import classification_report

# initialize the list of data (images), class labels, target bounding
# box coordinates, and image paths
print("[INFO] loading dataset...")
data = []
labels = []
bboxes = []
imagePaths = []

"""
# loop over all CSV files in the annotations directory
for csvPath in paths.list_files('annotations', validExts=(".txt")):
    # load the contents of the current CSV annotations file
    rows = open(csvPath).read().strip().split("\n")
    # loop over the rows
    for row in rows:
        # break the row into the filename, bounding box coordinates,
        # and class label
        row = row.split(" ")
        try:
            if len(row) <= 1:
                label = row
                startX, startY, endX, endY = 0
            else:
                (label, startX, startY, endX, endY) = row
            #print(row)
            imagePath = csvPath.split(".")[0]
            imagePath += ".png"
            #print(imagePath)
            #input(imagePath)
            #imagePath = os.path.sep.join(['human_detection_dataset/humanpresent',
            #                              imagePath])
            image = cv2.imread(imagePath)
            (h, w) = image.shape[:2]
            # scale the bounding box coordinates relative to the spatial
            # dimensions of the input image
            startX = float(startX) / w
            startY = float(startY) / h
            endX = float(endX) / w
            endY = float(endY) / h








            image = load_img(imagePath, target_size=(224, 224))
            image = img_to_array(image)
            # update our list of data, class labels, bounding boxes, and
            # image paths
            data.append(image)
            labels.append(label)
            bboxes.append((startX, startY, endX, endY))
            imagePaths.append(imagePath)
        except KeyboardInterrupt:
            quit()
        except Exception as e:
            pass
            #print("Error With {} is {}".format(csvPath, e))
"""

ANNOTATIONS_ROOT = "myDatasets/hands"

annotations_dir = os.listdir(ANNOTATIONS_ROOT)
# for annotation_dir in annotations_dir:
# print(annotation_dir)
updated = 0
print("[INFO] Loading Annotations...")
annotation_dir = "ann_train_val"
ann_sub_dir = os.path.join(ANNOTATIONS_ROOT, annotation_dir)
for gesture in os.listdir(ann_sub_dir):
    gesture_file = os.path.join(ann_sub_dir, gesture)
    with open(gesture_file, 'r') as f:
        annotation_data = json.loads(f.read())
        filenames = annotation_data.keys()
        gesture_filename = gesture.split('.')[0]
        for image_filename in list(filenames)[:1000]:
            updated += 1
            # input(gesture_filename+"/"+image_filename+".jpg")
            # filename = filename + ".jpg"
            image_path = os.path.join("myDatasets/hands", gesture_filename, image_filename + ".jpg")
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
                    if label != "no_gesture":
                        i += 1
                        x_center, y_center, bbox_width, bbox_height = bbox

                        # Convert normalized coordinates to pixel values


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


                        # Update lists
                        data.append(image)
                        labels.append(label)
                        bboxes.append((x_center, y_center, bbox_width, bbox_height))
                        imagePaths.append(image_path)

                    # Print the first bounding box coordinates as percentages
                    if 'bboxes' in annotation_data[image_filename]:
                        box1, box2, box3, box4 = annotation_data[image_filename]['bboxes'][0]
            else:
                pass
                # print(f"Error: Could not read image at {image_path}")
                # os.remove(image_path)

print("[INFO] Annoatations Loaded...")

# print(labels)
data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)
bboxes = np.array(bboxes, dtype="float32")
imagePaths = np.array(imagePaths)
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
#print(labels)
# only there are only two labels in the dataset, then we need to use
# Keras/TensorFlow's utility function as well


split = train_test_split(data, labels, bboxes, imagePaths,
                         test_size=0.20, random_state=42)
# unpack the data split
(trainImages, testImages) = split[:2]
(trainLabels, testLabels) = split[2:4]
(trainBBoxes, testBBoxes) = split[4:6]
(trainPaths, testPaths) = split[6:]

# load the VGG16 network, ensuring the head FC layers are left off
vgg = VGG16(weights="imagenet", include_top=False,input_tensor=Input(shape=(224, 224, 3)))
# vgg = MobileNetV3Small(weights="imagenet", include_top=False,input_tensor=Input(shape=(224, 224, 3)))

#vgg = ResNet101V2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# freeze all VGG layers so they will *not* be updated during the
# training process
vgg.trainable = False
# flatten the max-pooling output of VGG
flatten = vgg.output
flatten = Flatten()(flatten)

# construct a fully-connected layer header to output the predicted
# bounding box coordinates
bboxHead = Dense(128, activation="relu")(flatten)
bboxHead = Dense(64, activation="relu")(bboxHead)
bboxHead = Dense(32, activation="relu")(bboxHead)
bboxHead = Dense(4, activation="relu",
                 name="bounding_box")(bboxHead)
# construct a second fully-connected layer head, this one to predict
# the class label
softmaxHead = Dense(512, activation="relu")(flatten)
softmaxHead = Dropout(0.25)(softmaxHead)
softmaxHead = Dense(512, activation="relu")(softmaxHead)
softmaxHead = Dropout(0.25)(softmaxHead)
softmaxHead = Dense(len(lb.classes_), activation="softmax",
                    name="class_label")(softmaxHead)
# put together our model which accept an input image and then output
# bounding box coordinates and a class label
model = Model(
    inputs=vgg.input,
    outputs=(bboxHead, softmaxHead))

# define a dictionary to set the loss methods -- categorical
# cross-entropy for the class label head and mean absolute error
# for the bounding box head
def custom_classification_loss(y_true, y_pred):
    # Custom loss for class labels (e.g., weighted categorical cross-entropy)
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))

def custom_bbox_loss(y_true, y_pred):
    # Custom loss for bounding box regression (e.g., smooth L1 loss)
    return tf.reduce_mean(tf.losses.huber(y_true, y_pred))


losses = {
    "class_label": custom_classification_loss,
    "bounding_box": custom_bbox_loss,
}
# define a dictionary that specifies the weights per loss (both the
# class label and bounding box outputs will receive equal weight)
lossWeights = {
    "class_label": 1.0,
    "bounding_box": 1.0
}
# initialize the optimizer, compile the model, and show the model
# summary
opt = Adam(learning_rate=1e-3)

# opt = SGD(learning_rate=1e-5)


# Compile the model with multiple loss functions
model.compile(
    loss=losses,
    optimizer=opt,
    metrics=["accuracy","accuracy"],
    loss_weights=lossWeights  # Weighting each loss function
)
print(model.summary())

# from tensorflow.keras.utils import plot_model
# plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)


# construct a dictionary for our target training outputs
trainTargets = {
    "class_label": trainLabels,
    "bounding_box": trainBBoxes
}
# construct a second dictionary, this one for our target testing
# outputs
testTargets = {
    "class_label": testLabels,
    "bounding_box": testBBoxes
}

print("[INFO] training model...")
H = model.fit(
    trainImages, trainTargets,
    validation_data=(testImages, testTargets),
    batch_size=2,
    epochs=20,
    verbose=1)

print("[INFO] evaluating network...")
predIdxs = model.predict(x=testImages.astype("float32"), batch_size=2)

print(predIdxs)
# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability


# Ensure that predIdxs is consistent before applying np.argmax
predIdxs = np.argmax(predIdxs[0], axis=1)
print("Unexpected output shape. Predicted indices not computed.")

# show a nicely formatted classification report
print(classification_report(testLabels.argmax(axis=1), predIdxs,
                            target_names=lb.classes_))

# from sklearn.metrics import precision_recall_curve, average_precision_score
#
# num_classes = testLabels.shape[1]
#
# # Initialize a list to store average precision for each class
# average_precisions = []
#
# # Loop through each class
# for i in range(num_classes):
# 	# Get the true labels and predicted scores for the current class
# 	y_true = testLabels[:, i]
# 	y_scores = predIdxs[:, i]
#
# 	# Calculate the average precision for the current class
# 	ap = average_precision_score(y_true, y_scores)
# 	average_precisions.append(ap)
#
# # Calculate mean average precision
# mAP = np.mean(average_precisions)
#
# print(f"Mean Average Precision (mAP): {mAP:.4f}")
#

results = ""#f"Mean Average Precision (mAP): {mAP:.4f}\n"
results += classification_report(testLabels.argmax(axis=1), predIdxs,
                            target_names=lb.classes_)

with open('results/people_detection_results.txt','w') as f:
	f.write(results)
# serialize the model to disk
print("[INFO] saving object detector model...")
model.save("results/handswithBoxes.keras")

print("[INFO] saving label encoder...")
f = open("results/handsWithBoxesLabel.pickle", "wb")
f.write(pickle.dumps(lb))
f.close()


