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

import os

import launch
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory("x_mobility_navigator")

    return launch.LaunchDescription([
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="Use simulation (Omniverse Isaac Sim) clock if true"),
        Node(
            package='x_mobility_navigator',
            executable='x_mobility_navigator',
            output='screen',
            parameters=[{
                'use_sim_time':
                LaunchConfiguration("use_sim_time", default="True"),
                'is_mapless':
                LaunchConfiguration("is_mapless", default="True"),
                'runtime_path':
                LaunchConfiguration("runtime_path",
                                    default="/tmp/compass_carter.engine"),
            }],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            arguments=["-d",
                       os.path.join(pkg_dir, "rviz", "basic_view.rviz")],
            output="screen",
        ),
    ])
