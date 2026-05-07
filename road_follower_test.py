import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import os
import time


class RoadFollower(Node):
    def __init__(self):
        super().__init__('road_follower')
        self.get_logger().info(f'{self.get_name()} created')

        # Define parameters for topics, speed, model path, and data collection mode
        self.declare_parameter('image_topic', "/mycamera/image_raw")
        self.declare_parameter('cmd_topic', "/cmd_vel")
        self.declare_parameter('x_vel', 0.2)
        self.declare_parameter('theta_vel', 0.2)
        self.declare_parameter('model_path',
            "/home/common/CPMR3/ros2_ws/src/cpmr_ch6/model.keras")
        self.declare_parameter('data_collection_mode', False)
        self.declare_parameter('output_dir', "trainImages")

        # Retrieve parameter values
        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.cmd_topic = self.get_parameter('cmd_topic').get_parameter_value().string_value
        self.x_vel = self.get_parameter('x_vel').get_parameter_value().double_value
        self.theta_vel = self.get_parameter('theta_vel').get_parameter_value().double_value
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.data_collection_mode = self.get_parameter('data_collection_mode').get_parameter_value().bool_value
        self.output_dir = self.get_parameter('output_dir').get_parameter_value().string_value

        # Initialize publisher for movement commands and image converter
        self.publisher = self.create_publisher(Twist, self.cmd_topic, 1)
        self.bridge = CvBridge()

        # Load the CNN model
        self.model = load_model(self.model_path)

        # Subscribe to the image topic for receiving real-time images
        self.create_subscription(Image, self.image_topic, self.image_callback, 1)

        # Create output directory if data collection mode is active
        if self.data_collection_mode and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def image_callback(self, msg):
        # Convert ROS image to OpenCV format and resize for model input
        image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        image_resized = cv2.resize(image, (28, 28))
        image_resized = np.array(image_resized, dtype="float32") / 255.0
        image_resized = np.expand_dims(image_resized, axis=0)

        # Predict movement direction from model output
        prediction = self.model.predict(image_resized)
        direction = np.argmax(prediction)

        # Execute movement command based on model's output
        if direction == 0:
            self.turn_left()
            command_str = "left"
        elif direction == 1:
            self.go_straight()
            command_str = "straight"
        elif direction == 2:
            self.turn_right()
            command_str = "right"
        else:
            self.get_logger().warn("Received unrecognized prediction output.")
            return

        # Save the image with label if data collection mode is active
        if self.data_collection_mode:
            timestamp = int(time.time() * 1000)
            filename = f"{self.output_dir}/{command_str}_{timestamp}.png"
            image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(filename, cv2.resize(image_grayscale, (28, 28)))
            self.get_logger().info(f"Image saved as {filename} for movement '{command_str}'")

    def send_command(self, x_vel, theta_vel):
        twist = Twist()
        twist.linear.x = x_vel
        twist.angular.z = theta_vel
        self.publisher.publish(twist)

    def go_straight(self):
        self.send_command(self.x_vel, 0.0)

    def turn_left(self):
        self.send_command(self.x_vel, self.theta_vel)

    def turn_right(self):
        self.send_command(self.x_vel, -self.theta_vel)


def main(args=None):
    rclpy.init(args=args)
    node = RoadFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()


if __name__ == '__main__':
    main()
