# Import libraries for building and training the model, as well as data handling
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


class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        # Define the LeNet architecture with specified input dimensions
        model = Sequential()
        inputShape = (height, width, depth)

        # First set of layers: convolution, activation, and pooling for basic feature extraction
        model.add(Conv2D(20, (5, 5), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Second set of layers: deeper convolution, activation, and pooling for enhanced features
        model.add(Conv2D(50, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Fully connected layer for integration of extracted features
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        # Output layer with softmax to categorize movement directions
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # Return the completed model structure
        return model


# Specify the path to the dataset directory
dataset = '/home/common/CPMR3/ros2_ws/src/cpmr_ch6/cpmr_ch6/output'
print("[INFO] Loading images from dataset...")

# Initialize lists for storing image data and corresponding labels
data = []
labels = []

# Collect all image paths and shuffle them for randomness
imagePaths = sorted(list(paths.list_images(dataset)))
random.seed(42)
random.shuffle(imagePaths)

# Process each image in the dataset
for imagePath in imagePaths:
    # Load, resize, and convert the image to an array format suitable for model input
    image = cv2.imread(imagePath)
    if image is None:
        print(f"[WARNING] Unable to read image: {imagePath}")
        continue  # Skip if the image can't be loaded
    image = cv2.resize(image, (28, 28))
    image = img_to_array(image)
    data.append(image)

    # Extract label from the folder name and convert it to a numerical category
    label = imagePath.split(os.path.sep)[-2]
    if label == 'left':
        label = 0
    elif label == 'forward':
        label = 1
    elif label == 'right':
        label = 2
    else:
        print(f"[WARNING] Unrecognized label '{label}' for image: {imagePath}")
        continue  # Skip if label is unrecognized
    labels.append(label)

# Exit if no images were successfully loaded
if len(data) == 0:
    print("[ERROR] No images found in the dataset path. Check the structure and path.")
    exit(1)

# Normalize image pixel values to fall between 0 and 1 for optimized model performance
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# Split data into training and testing sets with a 75/25 split
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

# Transform labels into one-hot encoding format, required for categorical classification
trainY = to_categorical(trainY, num_classes=3)
testY = to_categorical(testY, num_classes=3)

# Setup a directory for TensorBoard logs for tracking model performance
logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=logdir)

# Define training parameters
EPOCHS = 100
INIT_LR = 1e-3
BS = 32

# Initialize and compile the LeNet model
print("[INFO] Compiling the model...")
model = LeNet.build(width=28, height=28, depth=3, classes=3)
opt = Adam(learning_rate=INIT_LR)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Start the training process with specified parameters and TensorBoard logging
print("[INFO] Training model...")
H = model.fit(
    trainX, trainY,
    batch_size=BS,
    validation_data=(testX, testY),
    epochs=EPOCHS,
    verbose=1,
    callbacks=[tensorboard_callback]
)

# Save the trained model for deployment
print("[INFO] Saving the trained model...")
model.save("model.keras")
