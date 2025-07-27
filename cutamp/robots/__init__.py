# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from dataclasses import dataclass
from typing import Sequence

import roma
import torch
from jaxtyping import Float

from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.geom.transform import quaternion_to_matrix
from curobo.types.base import TensorDeviceType
from .franka import (
    franka_curobo_cfg,
    get_franka_kinematics_model,
    get_franka_gripper_spheres,
    franka_neutral_joint_positions,
    load_franka_rerun,
)
from .ur5 import load_ur5_rerun, ur5_home, get_ur5_gripper_spheres, get_ur5_ik_solver, get_ur5_kinematics_model
from .utils import RerunRobot


@dataclass(frozen=True)
class RobotContainer:
    name: str
    kin_model: CudaRobotModel
    joint_limits: Float[torch.Tensor, "2 d"]
    # Note: in tool frame, not end-effector
    gripper_spheres: Float[torch.Tensor, "n 4"]
    # Transformation from tool pose to end-effector (defined in cuRobo config)
    tool_from_ee: Float[torch.Tensor, "4 4"]


def load_panda_container(tensor_args: TensorDeviceType) -> RobotContainer:
    kin_model = get_franka_kinematics_model()
    joint_limits = kin_model.kinematics_config.joint_limits.position
    assert joint_limits.shape == (2, 7), f"Invalid joint limits shape: {joint_limits.shape}"

    gripper_spheres = get_franka_gripper_spheres(tensor_args)
    tool_from_ee = torch.eye(4, device=tensor_args.device)
    gripper_down_quat = tensor_args.to_device([0.0, 1.0, 0.0, 0.0])
    tool_from_ee[:3, :3] = quaternion_to_matrix(gripper_down_quat[None])[0]
    tool_from_ee[:3, 3] = tensor_args.to_device([0.0, 0.0, 0.105])
    return RobotContainer("panda", kin_model, joint_limits, gripper_spheres, tool_from_ee)


def load_ur5_container(tensor_args: TensorDeviceType) -> RobotContainer:
    kin_model = get_ur5_kinematics_model()
    joint_limits = kin_model.kinematics_config.joint_limits.position
    assert joint_limits.shape == (2, 6), f"Invalid joint limits shape: {joint_limits.shape}"

    gripper_spheres = get_ur5_gripper_spheres(tensor_args)
    # See screenshot in assets/ur5_home.png to see gripper frame
    tool_from_ee = torch.eye(4, device=tensor_args.device)
    rpy = tensor_args.to_device([torch.pi, 0, torch.pi / 2])
    tool_from_ee[:3, :3] = roma.euler_to_rotmat("XYZ", rpy)
    # The Robotiq gripper goes down when closing, so we move the tool frame up by 1cm
    tool_from_ee[:3, 3] = tensor_args.to_device([0.0, 0.0, 0.01])
    return RobotContainer("ur5", kin_model, joint_limits, gripper_spheres, tool_from_ee)


robot_to_fns = {
    "panda": {
        "rerun": load_franka_rerun,
        "q_home": franka_neutral_joint_positions[:7],  # exclude gripper joint
        "container": load_panda_container,
    },
    "ur5": {
        "rerun": load_ur5_rerun,
        "q_home": ur5_home[:6],
        "container": load_ur5_container,
    },
}


def load_rerun_robot(robot: str, load_mesh: bool = True) -> RerunRobot:
    if robot not in robot_to_fns:
        raise ValueError(f"Unknown robot: {robot}. Supported robots: {list(robot_to_fns.keys())}")
    rerun_fn = robot_to_fns[robot]["rerun"]
    rerun_robot = rerun_fn(load_mesh)
    if not isinstance(rerun_robot, RerunRobot):
        raise TypeError(f"Expected RerunRobot, got {type(rerun_robot)}")
    return rerun_robot


def get_q_home(robot: str) -> Sequence[float]:
    """Get the home joint positions for the specified robot."""
    if robot not in robot_to_fns:
        raise ValueError(f"Unknown robot: {robot}. Supported robots: {list(robot_to_fns.keys())}")
    q_home = robot_to_fns[robot]["q_home"]
    return q_home


def load_robot_container(robot: str, tensor_args: TensorDeviceType) -> RobotContainer:
    """Load robot container which contains many helper classes and variables."""
    if robot not in robot_to_fns:
        raise ValueError(f"Unknown robot: {robot}. Supported robots: {list(robot_to_fns.keys())}")
    container_fn = robot_to_fns[robot]["container"]
    container = container_fn(tensor_args)
    if not isinstance(container, RobotContainer):
        raise TypeError(f"Expected RobotContainer, got {type(container)}")
    return container
