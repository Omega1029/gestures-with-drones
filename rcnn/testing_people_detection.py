import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt

# Load CIFAR-10 dataset
print("[INFO] loading CIFAR-10 dataset...")
(trainX, trainY), (testX, testY) = cifar10.load_data()

# Normalize the data to the range [0, 1]
trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0

# Apply ImageNet-specific preprocessing for MobileNetV2
trainX = preprocess_input(trainX)
testX = preprocess_input(testX)

# Convert labels to one-hot encoding
trainY = to_categorical(trainY, 10)
testY = to_categorical(testY, 10)

# Define the MobileNetV2 model (without the top layers)
print("[INFO] creating MobileNetV2 model...")
baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(32, 32, 3)))

# Add custom layers on top
x = baseModel.output
x = Flatten()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(10, activation="softmax")(x)  # 10 classes for CIFAR-10

# Create the model
model = Model(inputs=baseModel.input, outputs=x)

# Freeze the base layers (pre-trained weights)
for layer in baseModel.layers:
    layer.trainable = False

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=1e-4), metrics=["accuracy"])

# Data augmentation
print("[INFO] compiling image data generator...")
train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest"
)

# Train generator
train_generator = train_datagen.flow(trainX, trainY, batch_size=32)

# Test generator (no augmentation, only normalization)
test_datagen = ImageDataGenerator()
test_generator = test_datagen.flow(testX, testY, batch_size=32)

# Train the model
print("[INFO] training the model...")
H = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=10,  # Train for 10 epochs
    steps_per_epoch=len(trainX) // 32,  # Number of batches per epoch
    validation_steps=len(testX) // 32,  # Number of validation batches
    verbose=1
)

# Unfreeze some layers of the base model for fine-tuning
for layer in baseModel.layers[-10:]:  # Unfreeze the last 10 layers
    layer.trainable = True

# Recompile the model (important after unfreezing layers)
model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=1e-5), metrics=["accuracy"])

# Fine-tune the model
print("[INFO] fine-tuning the model...")
H_finetune = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=10,  # Continue training for another 10 epochs
    steps_per_epoch=len(trainX) // 32,
    validation_steps=len(testX) // 32,
    verbose=1
)

# Evaluate the model on the test set
print("[INFO] evaluating the model...")
loss, accuracy = model.evaluate(testX, testY)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Plot training loss and accuracy
plt.style.use("ggplot")
plt.figure()

# Plot loss during training
plt.subplot(1, 2, 1)
plt.plot(H.history["loss"], label="train_loss")
plt.plot(H.history["val_loss"], label="val_loss")
plt.title("Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# Plot accuracy during training
plt.subplot(1, 2, 2)
plt.plot(H.history["accuracy"], label="train_acc")
plt.plot(H.history["val_accuracy"], label="val_acc")
plt.title("Training Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()

# Save the trained model
print("[INFO] saving the model...")
model.save("mobilenetv2_cifar10.keras")
