# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pycuda.autoinit  # pylint: disable=unused-import
import pycuda.driver as cuda
import rclpy
import tensorrt as trt
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_geometry_msgs import do_transform_pose

IMAGE_TOPIC_NAME = '/front_stereo_camera/left/image_raw'
ODOM_TOPIC_NAME = '/chassis/odom'
CMD_TOPIC_NAME = '/cmd_vel'
ROUTE_TOPIC_NAME = '/route'
GOAL_TOPIC_NAME = '/goal_pose'
PATH_TOPIC_NAME = '/x_mobility_path'
RUNTIME_PATH = 'runtime_path'
MAPLESS_FLAG = 'is_mapless'

NUM_ROUTE_POINTS = 20
# Route vector with 4 values representing start and end positions
ROUTE_VECTOR_SIZE = 4
ROBOT_FRAME = 'base_link'


# Upsample the points between start and goal.
def upsample_points(start, goal, max_segment_length):
    x1, y1 = start
    x2, y2 = goal

    # Calculate the Euclidean distance between the two points
    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    # Handle the case where the start and goal are too close (or identical)
    if distance <= max_segment_length:
        return [start, goal]

    # Determine the number of segments based on the maximum segment length
    num_segments = max(1, int(np.ceil(distance / max_segment_length)))

    # Generate the interpolated points
    interpolated_points = [(x1 + (i / num_segments) * (x2 - x1),
                            y1 + (i / num_segments) * (y2 - y1))
                           for i in range(num_segments + 1)]

    return interpolated_points


class XMobilityNavigator(Node):
    '''X-Mobility Navigator ROS Node
    '''

    def __init__(self):
        super().__init__('x_mobility_navigator')
        # Parameters
        self.declare_parameter(RUNTIME_PATH, '/tmp/x_mobility.engine')
        self.declare_parameter(MAPLESS_FLAG, True)

        # Subscriber
        self.image_subscriber = self.create_subscription(
            Image, IMAGE_TOPIC_NAME, self.image_callback, 10)
        self.odom_subscriber = self.create_subscription(
            Odometry, ODOM_TOPIC_NAME, self.odom_callback, 10)
        self.route_subscriber = self.create_subscription(
            Path, ROUTE_TOPIC_NAME, self.route_callback, 10)
        self.goal_subscriber = self.create_subscription(
            PoseStamped, GOAL_TOPIC_NAME, self.goal_callback, 10)

        # Publisher
        self.cmd_publisher = self.create_publisher(Twist, CMD_TOPIC_NAME, 10)
        self.path_publisher = self.create_publisher(Path, PATH_TOPIC_NAME, 10)

        # Timer
        self.timer = self.create_timer(0.2, self.inference)

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Internal states
        self.action = np.zeros(6, dtype=np.float32)
        self.path = np.zeros(10, dtype=np.float32)
        self.history = np.zeros((1, 1024), dtype=np.float32)
        self.sample = np.zeros((1, 512), dtype=np.float32)
        self.camera_image = None
        self.route_vectors = None
        self.goal = None
        self.ego_speed = None
        self.runtime_context = None
        self.stream = cuda.Stream()
        self.cv_bridge = CvBridge()

    def load_model(self):
        self.get_logger().info('Loading model')
        runtime_path = self.get_parameter(
            RUNTIME_PATH).get_parameter_value().string_value
        with open(runtime_path, "rb") as f:
            engine_data = f.read()

        # Create a TensorRT runtime
        runtime = trt.Runtime(trt.Logger(trt.Logger.INFO))
        engine = runtime.deserialize_cuda_engine(engine_data)
        self.runtime_context = engine.create_execution_context()

    def image_callback(self, image_msg):
        self.camera_image = self.process_image_msg(image_msg)

    def odom_callback(self, odom_msg):
        self.ego_speed = np.array(odom_msg.twist.twist.linear.x,
                                  dtype=np.float32)

    def goal_callback(self, goal_msg):
        self.goal = goal_msg

    def route_callback(self, route_msg):
        # Get transform between route and robot.
        try:
            transform = self.tf_buffer.lookup_transform(
                ROBOT_FRAME, route_msg.header.frame_id, Time())
        except TransformException as ex:
            self.get_logger().error(
                f'Could not transform {ROBOT_FRAME} to {route_msg.header.frame_id}: {ex}'
            )
            return

        route_poses = route_msg.poses
        num_poses = min(len(route_poses), NUM_ROUTE_POINTS)
        # Return if route is empty.
        if num_poses == 0:
            return
        # Select the first NUM_ROUTE_POINTS and append the last route point as needed.
        indices = [idx for idx in range(num_poses)]
        indices.extend([num_poses - 1] * (NUM_ROUTE_POINTS - len(indices)))
        # Extract the x and y position in robot frame.
        selected_route_positions = []
        for idx in indices:
            transformed_pose = do_transform_pose(route_poses[idx].pose,
                                                 transform)
            selected_route_positions.append(
                [transformed_pose.position.x, transformed_pose.position.y])
        self.route_vectors = np.zeros(
            (NUM_ROUTE_POINTS - 1, ROUTE_VECTOR_SIZE), np.float32)
        for idx in range(NUM_ROUTE_POINTS - 1):
            self.route_vectors[idx] = np.concatenate(
                (selected_route_positions[idx],
                 selected_route_positions[idx + 1]),
                axis=0)

    def compose_mapless_route(self):
        if self.goal is None:
            return
        try:
            transform = self.tf_buffer.lookup_transform(
                ROBOT_FRAME, self.goal.header.frame_id, Time())
        except TransformException as ex:
            self.get_logger().error(
                f'Could not transform {ROBOT_FRAME} to {self.goal.header.frame_id}: {ex}'
            )
            return
        goal_in_robot_frame = do_transform_pose(self.goal.pose, transform)
        route_poses = upsample_points(
            [0.0, 0.0],
            [goal_in_robot_frame.position.x, goal_in_robot_frame.position.y],
            1.0)
        num_poses = min(len(route_poses), NUM_ROUTE_POINTS)
        # Return if route is empty.
        if num_poses == 0:
            return
        # Select the first NUM_ROUTE_POINTS and append the last route point as needed.
        indices = [idx for idx in range(num_poses)]
        indices.extend([num_poses - 1] * (NUM_ROUTE_POINTS - len(indices)))
        # Extract the x and y position in robot frame.
        selected_route_positions = []
        for idx in indices:
            selected_route_positions.append(route_poses[idx])
        self.route_vectors = np.zeros(
            (NUM_ROUTE_POINTS - 1, ROUTE_VECTOR_SIZE), np.float32)
        for idx in range(NUM_ROUTE_POINTS - 1):
            self.route_vectors[idx] = np.concatenate(
                (selected_route_positions[idx],
                 selected_route_positions[idx + 1]),
                axis=0)

    def inference(self):
        # Load model if not ready
        if not self.runtime_context:
            self.load_model()

        # Compose a simple route in mapless mode.
        if self.get_parameter(MAPLESS_FLAG).get_parameter_value().bool_value:
            self.compose_mapless_route()

        # Sanity checks of the inputs.
        # TODO: Sync the msgs.
        if self.camera_image is None or self.route_vectors is None or self.ego_speed is None:
            self.get_logger().info(f'Inputs are not ready.')
            return
        self._trt_inference()
        self.publish_action()
        self.publish_path()

    def _trt_inference(self):
        # Allocate device memory for inputs.
        image_input = cuda.mem_alloc(self.camera_image.nbytes)
        route_vec_input = cuda.mem_alloc(self.route_vectors.nbytes)
        speed_input = cuda.mem_alloc(self.ego_speed.nbytes)
        action_input = cuda.mem_alloc(self.action.nbytes)
        history_input = cuda.mem_alloc(self.history.nbytes)
        sample_input = cuda.mem_alloc(self.sample.nbytes)
        action_output = cuda.mem_alloc(self.action.nbytes)
        path_output = cuda.mem_alloc(self.path.nbytes)
        history_output = cuda.mem_alloc(self.history.nbytes)
        sample_ouput = cuda.mem_alloc(self.sample.nbytes)

        # Copy inputs to device.
        cuda.memcpy_htod(image_input, self.camera_image)
        cuda.memcpy_htod(route_vec_input, self.route_vectors)
        cuda.memcpy_htod(speed_input, self.ego_speed)
        cuda.memcpy_htod(action_input, self.action)
        cuda.memcpy_htod(history_input, self.history)
        cuda.memcpy_htod(sample_input, self.sample)

        # Order bindings based on sequence encoded in engine
        # Run engine.get_binding_name(binding_idx) to verify
        bindings = [
            int(image_input),
            int(route_vec_input),
            int(speed_input),
            int(action_input),
            int(history_input),
            int(sample_input),
            int(action_output),
            int(path_output),
            int(history_output),
            int(sample_ouput),
        ]

        # Run inference
        self.runtime_context.execute_v2(bindings)

        # Copy action back to host and publish
        cuda.memcpy_dtoh(self.action, action_output)
        cuda.memcpy_dtoh(self.path, path_output)
        cuda.memcpy_dtoh(self.history, history_output)
        cuda.memcpy_dtoh(self.sample, sample_ouput)

    def publish_action(self):
        cmd_vel = Twist()
        cmd_vel.linear.x = float(self.action[0])
        cmd_vel.angular.z = float(self.action[5])
        self.cmd_publisher.publish(cmd_vel)

    def publish_path(self):
        # print(self.path)
        path = Path()
        path.header.frame_id = ROBOT_FRAME
        path.header.stamp = self.get_clock().now().to_msg()
        for idx in range(5):
            path_pose = PoseStamped()
            path_pose.header = path.header
            path_pose.pose.position.x = float(self.path[2 * idx])
            path_pose.pose.position.y = float(self.path[2 * idx + 1])
            path.poses.append(path_pose)
        self.path_publisher.publish(path)

    def process_image_msg(self, image_msg):
        image_channels = int(image_msg.step / image_msg.width)
        image = np.array(image_msg.data).reshape(
            (image_msg.height, image_msg.width, image_channels))
        image = image.transpose(2, 0, 1).astype(np.float32) / 255.0
        return np.ascontiguousarray(image)


def main(args=None):
    rclpy.init(args=args)
    x_mobility_navigator = XMobilityNavigator()
    rclpy.spin(x_mobility_navigator)


if __name__ == '__main__':
    main()
