## Robot Assets Notes:

- [`robotiq_description/`](robotiq_description/) contains the URDF and meshes for the Robotiq grippers. Taken from:
  https://github.com/NVIDIA-ISAAC-ROS/ros2_robotiq_gripper
- `*_gripper_spheres.pt` contain serialized PyTorch tensors that represent the collision spheres for the gripper of
  different robots. We use this for approximate collision checking when perforing particle initialization.
- [`ur5e_robotiq_2f_85.urdf`](ur5e_robotiq_2f_85.urdf) is modified from cuMotion:
https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_cumotion/blob/main/isaac_ros_cumotion_robot_description/urdf/ur5e_robotiq_2f_85.urdf
- [`ur5e_robotiq_2f_85.yml`](ur5e_robotiq_2f_85.yml) contains the cuRobo config for the UR5e with Robotiq 2F-85 gripper
and modeling the camera mount as collision spheres in the `wrist_3` link.
- [`ur5e_robotiq_2f_85_wo_camera.yml`](ur5e_robotiq_2f_85_wo_camera.yml) contains the cuRobo config for the UR5e with Robotiq 2F-85 gripper
but without the camera mount collision spheres.