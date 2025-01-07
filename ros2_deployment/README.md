# E2E X-Mobility Navigator in ROS2

ROS2 package running E2E navigation using X-Mobility's TensorRT engine with Isaac Sim.

## Setup

### System requirements
1. OS: Ubuntu 22.04
2. ROS2 version: [humble](https://docs.ros.org/en/humble/Installation.html)
3. Isaac Sim version: [4.2](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_workstation.html)

### ROS2
First [create a ros workspace](https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Creating-A-Workspace/Creating-A-Workspace.html), e.g. `~/ros2_ws` and the requisite source folder, i.e `~/ros2_ws/src`

Then create symlink of this navigator folder to the ros workspace
```sh
$ ln -s <path_to_navigator_folder> <path_to_ros_workspace_source_folder>
```

Then compile the packages with colcon, e.g.
```sh
$ colcon build --symlink-install
```

### Isaac Sim ROS2 Bridge
Follow this [link](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_ros.html) to setup the ROS2 bridge.


### TensorRT Engine
For real-time TensorRT inference, install pycuda and python TensorRT inference package using the following commands:
```
pip3 install pycuda
sudo apt-get install python3-libnvinfer-dev
```

**Note:** TensorRT engines are specific to both the TensorRT version and the GPU on which they are created. Therefore, it's recommended to rebuild the engine on the target platform to run inference.

## Demo

Open the example ROS2 scenario `Carter Navigation` in Isaac Sim, then click the Play button to run the simulation. After that, Isaac Sim should start to publish ROS topics for sensor reading and odometry, and subscribe to the commands.

<p align="center">
    <img src="../images/carter_navigation.png" alt="Carter Navigation scenario" width="600" >
</p>

Launch the `x_mobility_navigator` node defined in the package `x_mobility_navigator` after setting the `runtime_path` to the TensorRT engine's path. Then the X-Mobility can run in mapless mode without localization, and the robot should start to navigate around after providing a `goal_pose` in odom frame using rviz.
```sh
$ source install/setup.bash
$ ros2 launch x_mobility_navigator x_mobility_navigator.launch.py
```

**Note:** This demo is in mapless mode for easier setup, and X-Mobility can also easily digest the global route with proper localization. To enable that, set the parameter `is_mapless` to False, and sent the route info via topic `/route`.