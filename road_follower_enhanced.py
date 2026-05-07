from packaging import version
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard
from imutils import paths
from sklearn.model_selection import train_test_split
import numpy as np
import random
import cv2
import os
from datetime import datetime


class EnhancedLeNet:
    @staticmethod
    def build(width, height, depth, classes):
        # Initialize the model architecture
        model = Sequential()
        inputShape = (height, width, depth)

        # First block: convolutional, activation, pooling, and dropout layers
        model.add(Conv2D(24, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(tf.keras.layers.Dropout(0.25))

        # Second block: deeper convolutional layer with activation, pooling, and dropout
        model.add(Conv2D(48, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(tf.keras.layers.Dropout(0.25))

        # Third block: additional convolutional layer for more feature extraction
        model.add(Conv2D(96, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(tf.keras.layers.Dropout(0.25))

        # Fully connected layer for high-level reasoning
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(tf.keras.layers.Dropout(0.5))

        # Output layer with softmax for multi-class classification
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # Return the complete model architecture
        return model


# Define dataset path
dataset = '/home/common/CPMR3/ros2_ws/src/cpmr_ch6/cpmr_ch6/output'

# Prepare lists for image data and labels
print("[INFO] Loading images from dataset...")
data = []
labels = []

# Retrieve all image paths and shuffle them randomly for data variety
imagePaths = sorted(list(paths.list_images(dataset)))
random.seed(42)
random.shuffle(imagePaths)

# Process each image
for imagePath in imagePaths:
    # Load, resize, and format each image for the model
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (28, 28))
    image = img_to_array(image)
    data.append(image)

    # Assign a numeric label based on the folder name
    label = imagePath.split(os.path.sep)[-2]
    if label == 'left':
        label = 0
    elif label == 'forward':
        label = 1
    else:
        label = 2
    labels.append(label)

# Normalize pixel values to [0, 1] for efficient learning
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# Split data into training and testing sets (75% training, 25% testing)
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

# Convert labels to one-hot encoded format for categorical classification
trainY = to_categorical(trainY, num_classes=3)
testY = to_categorical(testY, num_classes=3)

# Set up TensorBoard log directory for tracking model training
logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=logdir)

# Training configuration: number of epochs, learning rate, and batch size
EPOCHS = 100
INIT_LR = 1e-3
BS = 32

# Build and compile the enhanced model
print("[INFO] Compiling model with updated architecture...")
model = EnhancedLeNet.build(width=28, height=28, depth=3, classes=3)

# Configure the optimizer with the defined learning rate
opt = Adam(learning_rate=INIT_LR)

# Compile model with categorical crossentropy for multi-class output
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Begin training with the specified parameters and TensorBoard tracking
print("[INFO] Training network with updated architecture...")
H = model.fit(
    trainX, trainY,
    batch_size=BS,
    validation_data=(testX, testY),
    epochs=EPOCHS,
    verbose=1,
    callbacks=[tensorboard_callback]
)

# Save the trained model to disk for future use
print("[INFO] Saving trained model to disk...")
model.save("model.keras")
