# Autonomous Road-Following Robot Using TensorFlow (ROS2, Gazebo)

An autonomous road-following robot that learns from camera images to make real-time driving decisions, achieving ~90% accuracy in a ROS2 and Gazebo simulation environment.

## Overview
This repository contains:

- **AutoDriveByLine Node**  
  Handles real-time autonomous navigation using live camera input  

- **DriveByRoad Node**  
  Captures and labels training images through manual robot control  

- **Training Script**  
  Trains a model to classify driving actions from images  

- **Validation Script**  
  Evaluates model performance on unseen data  

---

## AutoDriveByLine Node

### Overview
Performs real-time autonomous navigation by continuously processing camera images and issuing velocity commands.

### Features

- **Image Input**  
  Subscribes to `/mycamera/image_raw` for live video feed  

- **Velocity Output**  
  Publishes movement commands to `/cmd_vel`  

- **Steering Logic**  
  Classifies each frame as `left`, `forward`, or `right` and adjusts the robot’s path  

- **Manual Override**  
  Keyboard controls for manual driving:
  - `j` → turn left  
  - `k` → go forward  
  - `l` → turn right  
  - `space` → stop  
  - `x` → resume autonomous mode  

---

## DriveByRoad Node

### Overview
Used to collect training data by manually controlling the robot while recording labeled images.

### Features

- **Manual Control**
  - `j` → turn left  
  - `k` → go forward  
  - `l` → turn right  

- **Recording Mode**
  - `s` → start recording  
  - `x` → stop recording  

- **Data Storage**  
  Images are automatically organized into:
  - `left/`  
  - `forward/`  
  - `right/`  

---

## Model

### Overview
A lightweight image classification model that maps camera input to driving actions.

- `0` → left  
- `1` → forward  
- `2` → right  

### Training

- Images are resized and normalized before training  
- Trained using manually labeled driving data  
- Achieves **~90% classification accuracy** on validation data  

---

## Results

The robot successfully navigates simulated road environments in real time, demonstrating reliable lane-following behavior driven entirely by camera-based predictions.

## Notes

Some file paths in this project are hardcoded to a specific ROS2 workspace:

/home/common/CPMR3/ros2_ws/src/cpmr_ch6/

Update these paths to match your own local ROS2 environment before running the code.
