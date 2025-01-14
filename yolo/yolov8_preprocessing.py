import json
import os, shutil
import random

DATASET = "myDatasets/hands"
ANNOTATIONS_TRAIN_VAL = os.path.join(DATASET, "ann_train_val")



YOLO_DATASET_DIR = "yolodataset"
YOLO_DATASET_IMAGES = os.path.join(YOLO_DATASET_DIR, "images")
TRAIN_IMAGES_YOLO = os.path.join(YOLO_DATASET_IMAGES, "train")
VAL_IMAGES_YOLO = os.path.join(YOLO_DATASET_IMAGES, "val")

YOLO_DATASET_LABELS = os.path.join(YOLO_DATASET_DIR, "labels")
TRAIN_LABELS_YOLO = os.path.join(YOLO_DATASET_LABELS, "train")
VAL_LABELS_YOLO = os.path.join(YOLO_DATASET_LABELS, "val")

CLASS_LABELS = {
    'call': 0,
    'dislike': 1,
    'fist': 2,
    'four': 3,
    'like': 4,
    'mute': 5,
    'ok': 6,
    'one': 7,
    'palm': 8,
    'peace': 9,
    'peace_inverted': 10,
    'rock': 11,
    'stop': 12,
    'stop_inverted': 13,
    'three': 14,
    'three2': 15,
    'two_up': 16,
    'two_up_inverted': 17,
    'no_gesture': 18
}


directory = os.listdir(ANNOTATIONS_TRAIN_VAL)
directory.sort()
directory = directory
print(directory)

if not os.path.exists("yolodataset"):
    os.makedirs("yolodataset")
    os.makedirs("yolodataset/images")
    os.makedirs("yolodataset/labels")
    os.makedirs("yolodataset/images/train")
    os.makedirs("yolodataset/images/val")
    os.makedirs("yolodataset/labels/train")
    os.makedirs("yolodataset/labels/val")

for gesture_annotation_file in directory:
    #print(gesture_annotation_file)
    annotation = gesture_annotation_file.split('.')[0]
    #print(annotation)
    with open(os.path.join(ANNOTATIONS_TRAIN_VAL, gesture_annotation_file)) as f:
        annotation_data = json.load(f)
        filenames = list(annotation_data.keys())
        gesture_filename = gesture_annotation_file.split('.')[0]
        filenames.sort()
        for image_filename in filenames[:700]:
            bboxes_annotations = annotation_data[image_filename]['bboxes']
            labels_annotations = annotation_data[image_filename]['labels']

            #print("{} {} {}".format(image_filename, bboxes_annotations, labels_annotations))
            try:
                shutil.copy(os.path.join(DATASET, annotation, image_filename+".jpg"), TRAIN_IMAGES_YOLO)

                with open('{}.txt'.format(os.path.join(TRAIN_LABELS_YOLO, image_filename)), 'a') as f:
                    #f.write(bboxes_annotations)
                    for bbox, label in zip(bboxes_annotations, labels_annotations):
                        #print(bbox)
                        #print(label, CLASS_LABELS[label])
                        s = '{} '.format(CLASS_LABELS[label])
                        #f.write(str(CLASS_LABELS[label]))
                        for coordinate in bbox:
                            #f.write(str(coordinate))
                            s = s + '{} '.format(coordinate)
                        #s += '\n'
                        f.write(s)

            except FileNotFoundError:
                continue

for gesture_annotation_file in directory:
    print(gesture_annotation_file)
    annotation = gesture_annotation_file.split('.')[0]
    #print(annotation)
    with open(os.path.join(ANNOTATIONS_TRAIN_VAL, gesture_annotation_file)) as f:
        annotation_data = json.load(f)
        filenames = list(annotation_data.keys())
        gesture_filename = gesture_annotation_file.split('.')[0]
        filenames.sort()
        for image_filename in filenames[700:1000]:
            bboxes_annotations = annotation_data[image_filename]['bboxes']
            labels_annotations = annotation_data[image_filename]['labels']

            #print("{} {} {}".format(image_filename, bboxes_annotations, labels_annotations))
            try:
                shutil.copy(os.path.join(DATASET, annotation, image_filename+".jpg"), VAL_IMAGES_YOLO)

                with open('{}.txt'.format(os.path.join(VAL_LABELS_YOLO, image_filename)), 'a') as f:
                    #f.write(bboxes_annotations)
                    for bbox, label in zip(bboxes_annotations, labels_annotations):
                        print(bbox)
                        print(label, CLASS_LABELS[label])
                        s = '{} '.format(CLASS_LABELS[label])
                        #f.write(str(CLASS_LABELS[label]))
                        for coordinate in bbox:
                            #f.write(str(coordinate))
                            s = s + '{} '.format(coordinate)
                        #s += '\n'
                        f.write(s)

            except FileNotFoundError:
                continue