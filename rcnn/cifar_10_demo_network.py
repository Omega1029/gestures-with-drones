#This Script will be used to test My Netowkrs
#Agaisnt Cifar - 10
import os

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

# Load CIFAR-10 data
print("Loading CIFAR-10 data...")
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Preprocess CIFAR-10 data
print("Preprocessing CIFAR-10 data...")
X_train = X_train.astype('float32') / 255.0  # Normalize to [0, 1]
X_test = X_test.astype('float32') / 255.0  # Normalize to [0, 1]

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Initialize MobileNetV2 with ImageNet weights (exclude top layers)
print("Initializing MobileNetV2 model...")
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Freeze the layers of MobileNetV2 (optional, if you want to fine-tune later, unfreeze some layers)
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification head
flatten = Flatten()(base_model.output)
dense_1 = Dense(256, activation='relu')(flatten)
dropout_1 = Dropout(0.5)(dense_1)
dense_2 = Dense(64, activation='relu')(dropout_1)
dropout_2 = Dropout(0.5)(dense_2)
output = Dense(10, activation='softmax')(dropout_2)
fname = os.path.sep.join("models",
	"weights-{epoch:03d}-{val_loss:.4f}.keras")
checkpoint = ModelCheckpoint(fname, monitor="val_loss", mode="min",
	save_best_only=True, verbose=1)
callbacks = [checkpoint]
# Create the model
model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# Early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
print("Training the model...")
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=64,
    validation_data=(X_test, y_test),
    callbacks=callbacks
)

# Evaluate the model
print("Evaluating the model...")
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Optionally, save the model
# model.save('mobilenet_cifar10.h5')
