# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
from functools import lru_cache
from pathlib import Path

import torch
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.util_file import get_assets_path
from curobo.util_file import join_path, load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from jaxtyping import Float
from yourdfpy import URDF

from cutamp.robots.utils import RerunRobot

# or simplified: (0, -1.57, 1.57, -1.57, -1.57, 0)
ur5_home = (0, -torch.pi / 2, torch.pi / 2, -torch.pi / 2, -torch.pi / 2, 0)


@lru_cache(maxsize=1)
def ur5_curobo_cfg() -> dict:
    assets_dir = Path(__file__).parent / "assets"
    # Note: use ur5e_robotiq_2f_85.yml for UR5e with camera mount (on MIT setup)
    cfg = load_yaml(str(assets_dir / "ur5e_robotiq_2f_85_wo_camera.yml"))
    # Set some asset paths so cuRobo can load our URDF and meshes
    keys = ["external_asset_path", "external_robot_configs_path"]
    for key in keys:
        if key not in cfg["robot_cfg"]["kinematics"]:
            cfg["robot_cfg"]["kinematics"][key] = str(assets_dir)
    return cfg
    # return load_yaml(join_path(get_robot_configs_path(), "ur5e_robotiq_2f_140.yml"))


def _ur5_cfg_dict() -> dict:
    return ur5_curobo_cfg()["robot_cfg"]


def get_ur5_kinematics_model() -> CudaRobotModel:
    """cuRobo robot kinematics model."""
    robot_cfg = RobotConfig.from_dict(_ur5_cfg_dict())
    kinematics_model = CudaRobotModel(robot_cfg.kinematics)
    return kinematics_model


def get_ur5_ik_solver(
    world_cfg: WorldConfig,
    num_seeds: int = 12,
    self_collision_opt: bool = False,
    self_collision_check: bool = True,
    use_particle_opt: bool = False,
) -> IKSolver:
    """
    cuRobo IK solver for UR5e with Robotiq 2F-85 gripper.
    """
    robot_cfg = _ur5_cfg_dict()
    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        num_seeds=num_seeds,
        self_collision_opt=self_collision_opt,
        self_collision_check=self_collision_check,
        use_particle_opt=use_particle_opt,
    )
    ik_solver = IKSolver(ik_config)
    return ik_solver


def get_ur5_gripper_spheres(tensor_args: TensorDeviceType = TensorDeviceType()) -> Float[torch.Tensor, "num_spheres 4"]:
    """Collision spheres for UR5e with Robotiq 2F-85 gripper. Note: the spheres are in the origin frame with z-up."""
    assets_dir = Path(__file__).parent / "assets"
    spheres_pt = assets_dir / "ur5_robotiq_2f_85_gripper_spheres.pt"
    if not spheres_pt.exists():
        raise FileNotFoundError(f"File not found: {spheres_pt}")

    spheres = torch.load(spheres_pt, map_location=tensor_args.device)
    assert spheres.ndim == 2 and spheres.shape[1] == 4, f"Invalid shape for UR5 spheres: {spheres.shape}"
    return spheres


def load_ur5_rerun(load_mesh: bool = True) -> RerunRobot:
    robot_cfg = _ur5_cfg_dict()
    urdf_rel_path = robot_cfg["kinematics"]["urdf_path"]
    urdf_path = os.path.join(robot_cfg["kinematics"].get("external_asset_path", ""), urdf_rel_path) or join_path(
        get_assets_path(), urdf_rel_path
    )

    def _locate_curobo_asset(fname: str) -> str:
        if fname.startswith("package://"):
            return os.path.join(get_assets_path(), "robot", fname.replace("package://", ""))

        assets_dir = Path(__file__).parent
        return os.path.join(assets_dir, fname)

    urdf = URDF.load(urdf_path, filename_handler=_locate_curobo_asset)
    return RerunRobot("ur5", urdf, q_neutral=ur5_home, load_mesh=load_mesh)
