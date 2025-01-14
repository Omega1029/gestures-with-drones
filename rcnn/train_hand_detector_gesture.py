# import the necessary packages
# from pyimagesearch import config
import json
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
from tensorflow.python.keras.callbacks import ModelCheckpoint

# initialize the list of data (images), class labels, target bounding
# box coordinates, and image paths
print("[INFO] loading dataset...")
data = []
labels = []
#bboxes = []
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
        for image_filename in list(filenames)[:7]:

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
                    if label == "no_gesture":
                        continue
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
                    cv2.putText(img, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3,
                               cv2.LINE_AA)
                    # cv2.put

                    hand = img[y_min:y_max, x_min:x_max]
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
                cv2.imshow("Image", img)
                try:

                    hand = cv2.resize(hand, (224, 224), interpolation=cv2.INTER_AREA)
                    cv2.imshow("Hand", hand)
                    #data.append(image)
                    data.append(hand)
                    labels.append(label)
                    # bboxes.append((x_min_resized, y_min_resized, x_max_resized, y_max_resized))
                    imagePaths.append(image_path)
                except:
                    pass
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                #print("Displayed:", image_path)


            else:
                pass
                # print(f"Error: Could not read image at {image_path}")
                # os.remove(image_path)
#quit()
print("[INFO] Annoatations Loaded...")

# print(labels)
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
                         test_size=0.26, random_state=42)

print(split)
# unpack the data split
(trainImages, testImages) = split[:2]
(trainLabels, testLabels) = split[2:4]
#(trainBBoxes, testBBoxes) = split[4:6]
#(trainPaths, testPaths) = split[4:]

# load the VGG16 network, ensuring the head FC layers are left off
# vgg = VGG16(weights="imagenet", include_top=False,input_tensor=Input(shape=(224, 224, 3)))
# vgg = MobileNetV3Small(weights="imagenet", include_top=False,input_tensor=Input(shape=(224, 224, 3)))

vgg = ResNet101V2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

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
softmaxHead = Dense(len(lb.classes_), activation="softmax",
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
    "class_label": "categorical_crossentropy",
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
print(model.summary())

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
fname = os.path.sep.join("models",
	"weights-{epoch:03d}-{val_loss:.4f}.keras")
checkpoint = ModelCheckpoint(fname, monitor="val_loss", mode="min",
	save_best_only=True, verbose=1)
callbacks = [checkpoint]
print("[INFO] training model...")
H = model.fit(
    trainImages, trainTargets,
    validation_data=(testImages, testTargets),
    batch_size=32,
    epochs=10,
    verbose=2,
   callbacks=callbacks)
print("[INFO] saving object detector model...")
model.save("models/handsWithNoBoxes.keras")

print("[INFO] saving label encoder...")
f = open("models/handsWithNoBoxesLabel.pickle", "wb")
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
model.save("models/handsWithNoBoxes.keras")

print("[INFO] saving label encoder...")
f = open("models/handsWithNoBoxesLabel.pickle", "wb")
f.write(pickle.dumps(lb))
f.close()


