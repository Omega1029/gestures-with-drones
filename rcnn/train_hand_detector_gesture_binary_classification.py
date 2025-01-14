# import the necessary packages
# from pyimagesearch import config
import json
import random
from pprint import pprint

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
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint
from collections import Counter
from collections import defaultdict

# initialize the list of data (images), class labels, target bounding
# box coordinates, and image paths
print("[INFO] loading dataset...")
data = []
labels = []
#bboxes = []
imagePaths = []

FIRST_CLASS_LIMIT = 1000  # For "fist"
OTHER_CLASSES_LIMIT = 1000  # Total combined for the other 17 classes
updated = 0

# Initialize the counters for each class (gesture)
label_counts = defaultdict(int)

# Define your first class and other classes
first_class = "fist"  # This is the first class
other_classes = [
    'dislike', 'one', 'two_up_inverted', 'two_up', 'palm', 'call', 
    'peace_inverted', 'rock', 'mute', 'ok', 'three2', 'like', 'four', 'stop_inverted', 
    'peace', 'three', 'stop'
]  # These are the remaining classes

# Calculate how many samples each of the remaining classes should have
other_classes_sample_limit = OTHER_CLASSES_LIMIT // len(other_classes)



ANNOTATIONS_ROOT = "datasets/hands"

# for annotation_dir in annotations_dir:
# print(annotation_dir)
updated = 0
#print("[INFO] Loading Annotations...")
annotation_dir = "ann_train_val"
ann_sub_dir = os.path.join(ANNOTATIONS_ROOT, annotation_dir)


# Initialize the data storage lists
data = []
labels = []
imagePaths = []

gesture_file = os.path.join(ann_sub_dir, f"{first_class}.json")
with open(gesture_file, 'r') as f:
    annotation_data = json.loads(f.read())
    filenames = annotation_data.keys()

    # Collect 1000 samples for the first class (fist)
    for image_filename in filenames:
        if label_counts[first_class] >= FIRST_CLASS_LIMIT:
            break  # Stop once we have 1000 samples for the first class

        image_path = os.path.join("datasets/hands", first_class, image_filename + ".jpg")
        if not os.path.exists(image_path):
            continue  # Skip if the image doesn't exist
        
        img = cv2.imread(image_path)
        if img is None:
            continue  # Skip if image can't be read
        
        height, width = img.shape[:2]
        bboxes_annotations = annotation_data[image_filename]['bboxes']
        labels_annotations = annotation_data[image_filename]['labels']

        for bbox, label in zip(bboxes_annotations, labels_annotations):
            if label == "no_gesture":  # Skip "no_gesture" labels
                continue
            
            x_center, y_center, bbox_width, bbox_height = bbox
            x_min = int((x_center - bbox_width / 2) * width)
            y_min = int((y_center - bbox_height / 2) * height)
            x_max = int((x_center + bbox_width) * width)
            y_max = int((y_center + bbox_height) * height)
            
            # Crop the hand from the image
            hand = img[y_min:y_max, x_min:x_max]
            
            try:
                hand = cv2.resize(hand, (224, 224), interpolation=cv2.INTER_AREA)
                data.append(hand)
                labels.append(first_class)  # Add the class label ("fist")
                imagePaths.append(image_path)

                label_counts[first_class] += 1
                updated += 1
            except Exception as e:
                continue  # Skip if there's an error in resizing or processing

    print(f"Finished First Class processing {first_class} with {label_counts[first_class]} samples")

# Now process the remaining classes with an equal distribution of samples
l = list()
for gesture in other_classes:
    if gesture == first_class:
        continue  # Skip the first class since we've already processed it
    label = "nofist"
    gesture_file = os.path.join(ann_sub_dir, f"{gesture}.json")
    with open(gesture_file, 'r') as f:
        annotation_data = json.loads(f.read())
        filenames = annotation_data.keys()

        # Collect samples for the current class (like, dislike, etc.)
        for image_filename in filenames:
            l.append(os.path.join(ANNOTATIONS_ROOT, gesture, image_filename)+".jpg")

random.shuffle(l)

for image_path in l:
    if label_counts['nofist'] >= FIRST_CLASS_LIMIT:
        break

    #image_path = os.path.join("datasets/hands", gesture, image_filename + ".jpg")

    #print(image_path)
    if not os.path.exists(image_path):
        #raise FileExistsError
        continue  # Skip if the image doesn't exist

    img = cv2.imread(image_path)
    if img is None:
        #raise cv2.Error
        continue  # Skip if image can't be read

    height, width = img.shape[:2]
    bboxes_annotations = annotation_data[image_filename]['bboxes']
    labels_annotations = annotation_data[image_filename]['labels']

    for bbox, label in zip(bboxes_annotations, labels_annotations):
        if label == "no_gesture":  # Skip "no_gesture" labels
            continue
        label = "nofist"
        x_center, y_center, bbox_width, bbox_height = bbox
        x_min = int((x_center - bbox_width / 2) * width)
        y_min = int((y_center - bbox_height / 2) * height)
        x_max = int((x_center + bbox_width) * width)
        y_max = int((y_center + bbox_height) * height)

        # Crop the hand from the image
        hand = img[y_min:y_max, x_min:x_max]

        try:
            hand = cv2.resize(hand, (224, 224), interpolation=cv2.INTER_AREA)
            data.append(hand)
            labels.append(label)  # Add the class label (like, dislike, etc.)
            imagePaths.append(image_path)

            label_counts[label] += 1
            updated += 1
        except Exception as e:
            print(e)
            continue  # Skip if there's an error in resizing or processing


# After processing all gestures, print the label distribution
print("[INFO] Dataset Loaded...")
print(f"Total samples processed: {updated}")
print(f"Label distribution: {dict(label_counts)}")
input()
# Ensure the first class has exactly 1000 samples, and others have their share of 1000
# if label_counts[first_class] != FIRST_CLASS_LIMIT:
#     print(f"Warning: {first_class} has {label_counts[first_class]} samples, expected {FIRST_CLASS_LIMIT}")
#
# for gesture in other_classes:
#     if gesture != first_class and label_counts[gesture] != other_classes_sample_limit:
#         print(f"Warning: {gesture} has {label_counts[gesture]} samples, expected {other_classes_sample_limit}")
data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)
#bboxes = np.array(bboxes, dtype="float32")
imagePaths = np.array(imagePaths)
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
print(labels)
# only there are only two labels in the dataset, then we need to use
# Keras/TensorFlow's utility function as well


split = train_test_split(data, labels,
                         test_size=0.20, random_state=42)

print(split)
# unpack the data split
(trainImages, testImages) = split[:2]
(trainLabels, testLabels) = split[2:4]
#(trainBBoxes, testBBoxes) = split[4:6]
#(trainPaths, testPaths) = split[4:]

# load the VGG16 network, ensuring the head FC layers are left off
# vgg = VGG16(weights="imagenet", include_top=False,input_tensor=Input(shape=(224, 224, 3)))
# vgg = MobileNetV3Small(weights="imagenet", include_top=False,input_tensor=Input(shape=(224, 224, 3)))

vgg = EfficientNetV2S(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# freeze all VGG layers so they will *not* be updated during the
# training process
vgg.trainable = False
# flatten the max-pooling output of VGG
flatten = vgg.output
flatten = Flatten()(flatten)

# construct a fully-connected layer header to output the predicted
# bounding box coordinates
# bboxHead = Dense(128, activation="relu")(flatten)
# bboxHead = Dense(64, activation="relu")(bboxHead)
# bboxHead = Dense(32, activation="relu")(bboxHead)
# bboxHead = Dense(4, activation="relu",
#                  name="bounding_box")(bboxHead)
# construct a second fully-connected layer head, this one to predict
# the class label
softmaxHead = Dense(512, activation="relu")(flatten)
softmaxHead = Dropout(0.5)(softmaxHead)
softmaxHead = Dense(512, activation="relu")(softmaxHead)
softmaxHead = Dropout(0.5)(softmaxHead)
softmaxHead = Dense(1, activation="sigmoid",
                    name="class_label")(softmaxHead)
# put together our model which accept an input image and then output
# bounding box coordinates and a class label
model = Model(
    inputs=vgg.input,
    outputs=softmaxHead
    #outputs=(bboxHead, softmaxHead)
 )

# define a dictionary to set the loss methods -- categorical
# cross-entropy for the class label head and mean absolute error
# for the bounding box head
losses = {
    "class_label": "binary_crossentropy",
    #"bounding_box": "mean_squared_error",
}
# define a dictionary that specifies the weights per loss (both the
# class label and bounding box outputs will receive equal weight)
lossWeights = {
    "class_label": 1.0,
    #"bounding_box": 1.0
}
# initialize the optimizer, compile the model, and show the model
# summary
opt = Adam(learning_rate=1e-4)

# opt = SGD(learning_rate=1e-5)
model.compile(loss=losses, optimizer=opt, metrics=["accuracy"], loss_weights=lossWeights)
#print(model.summary())

# from tensorflow.keras.utils import plot_model
# plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)


# construct a dictionary for our target training outputs
trainTargets = {
    "class_label": trainLabels
    #"bounding_box": trainBBoxes
}
# construct a second dictionary, this one for our target testing
# outputs
testTargets = {
    "class_label": testLabels
    #"bounding_box": testBBoxes
}

print("[INFO] training model...")
H = model.fit(
    trainImages, trainTargets,
    validation_data=(testImages, testTargets),
    batch_size=32,
    epochs=10,
    verbose=1)
print("[INFO] saving object detector model...")
model.save("models/binaryClassification.keras")

print("[INFO] saving label encoder...")
f = open("models/binaryClassification.pickle", "wb")
f.write(pickle.dumps(lb))
f.close()

print("[INFO] evaluating network...")
predIdxs = model.predict(x=testImages.astype("float32"), batch_size=32)

print(predIdxs)
# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability


# Ensure that predIdxs is consistent before applying np.argmax
predIdxs = np.argmax(predIdxs, axis=1)
#input("Unexpected output shape. Predicted indices not computed.")
#input(predIdxs)



print(testLabels.argmax(axis=1))
print(predIdxs)
print(lb.classes_)
# show a nicely formatted classification report
print(classification_report(testLabels.argmax(axis=1),
                            predIdxs,
                            labels=lb.classes_))

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
# 	y_true = testLabels[i]
# 	y_scores = predIdxs[i]
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
#
# results = f"Mean Average Precision (mAP): {mAP:.4f}\n"
results = ""
results += classification_report(testLabels.argmax(axis=1), predIdxs,
                            labels=lb.classes_)

with open('results/hand_gesture_no_boxes_results.txt','w') as f:
    print(results)
    f.write(results)

# serialize the model to disk
print("[INFO] saving object detector model...")
model.save("models/handsBinaryClassify.keras")

print("[INFO] saving label encoder...")
f = open("models/handsBinaryClassify.pickle", "wb")
f.write(pickle.dumps(lb))
f.close()


