# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import logging
import warnings
from functools import cached_property
from typing import List, Literal, Dict, Union

import torch
from jaxtyping import Float

from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.geom.types import Obstacle
from curobo.types.base import TensorDeviceType
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig
from cutamp.envs import TAMPEnvironment
from cutamp.robots import RobotContainer, load_robot_container
from cutamp.robots.franka import franka_curobo_cfg, get_franka_ik_solver
from cutamp.robots.ur5 import ur5_curobo_cfg, get_ur5_ik_solver
from cutamp.tamp_domain import get_initial_state
from cutamp.task_planning import State
from cutamp.utils.collision import get_world_collision_cost
from cutamp.utils.common import approximate_goal_aabb, transform_spheres
from cutamp.utils.common import sample_between_bounds, get_world_cfg, pose_list_to_mat4x4
from cutamp.utils.shapes import sample_greedy_surface_spheres

_log = logging.getLogger(__name__)


class TAMPWorld:
    """
    Represents a TAMP world that wraps a static TAMPEnvironment with robot-specific logic,
    object indexing utilities, collision checking, IK solvers, and motion generation support.
    """

    def __init__(
        self,
        env: TAMPEnvironment,
        tensor_args: TensorDeviceType,
        robot: Union[Literal["panda", "ur5"], RobotContainer],
        q_init: Float[torch.Tensor, "dof"],
        collision_activation_distance: float = 0.0,
        coll_n_spheres: int = 50,
        coll_sphere_radius: float = 0.005,
    ):
        self.env = env
        self.tensor_args = tensor_args

        # Dicts and sets for indexing
        self._movable_names = {obj.name for obj in env.movables}
        self._name_to_obj = {obj.name: obj for obj in env.movables + env.statics}

        # Setup collision function
        self.world_cfg = get_world_cfg(env, include_movables=False)  # doesn't include movables
        self.collision_fn = get_world_collision_cost(self.world_cfg, tensor_args, collision_activation_distance)
        self.collision_activation_distance = collision_activation_distance

        # Setup robot container
        if isinstance(robot, str):
            warnings.warn(f"RobotContainer not provided, loading based on robot name {robot}")
            self.robot_container = load_robot_container(robot, tensor_args)
        else:
            self.robot_container = robot
        self.robot_name = self.robot_container.name
        self.q_init = q_init

        # Setup the IK solver, right now it needs WorldCfg and I don't know the behavior, can speed up later
        if self.robot_name == "panda":
            self.ik_solver = get_franka_ik_solver(self.world_cfg)
        elif self.robot_name == "ur5":
            self.ik_solver = get_ur5_ik_solver(self.world_cfg)
        else:
            raise ValueError(f"Unsupported robot: {self.robot_name}")

        # Sample collision spheres for all movables
        self._obj_to_spheres: Dict[str, Float[torch.Tensor, "n 4"]] = {}
        for obj in self.movables:
            spheres = sample_greedy_surface_spheres(obj, n_spheres=coll_n_spheres, sphere_radius=coll_sphere_radius)
            self._obj_to_spheres[obj.name] = spheres.to(tensor_args.device)

        # AABB cache
        self._obj_to_aabb = {}

    @property
    def movables(self) -> List[Obstacle]:
        return self.env.movables

    def is_movable(self, obj: Obstacle | str) -> bool:
        if isinstance(obj, Obstacle):
            obj = obj.name
        return obj in self._movable_names

    @property
    def statics(self) -> List[Obstacle]:
        return self.env.statics

    @property
    def kin_model(self) -> CudaRobotModel:
        return self.robot_container.kin_model

    @property
    def tool_from_ee(self) -> Float[torch.Tensor, "4 4"]:
        """Transformation from tool frame to end-effector frame used by kinematics model and IK solver."""
        return self.robot_container.tool_from_ee

    @property
    def device(self) -> torch.device:
        return self.tensor_args.device

    @property
    def initial_state(self) -> State:
        initial_state = get_initial_state(
            movables=self.get_objects_by_type("Movable", return_name=True),
            surfaces=self.get_objects_by_type("Surface", return_name=True),
            sticks=self.get_objects_by_type("Stick", return_name=True),
            buttons=self.get_objects_by_type("Button", return_name=True),
        )
        return initial_state

    @property
    def goal_state(self) -> State:
        return self.env.goal_state

    def get_objects_by_type(self, obj_type: str, return_name: bool = True) -> List[Union[Obstacle, str]]:
        if obj_type not in self.env.type_to_objects:
            return []
        objs = self.env.type_to_objects[obj_type]
        if return_name:
            objs = [obj.name for obj in objs]
        return objs

    def get_object(self, name: str) -> Obstacle:
        """Get cuRobo Obstacle for object with the given name."""
        if name not in self._name_to_obj:
            raise ValueError(f"Object '{name}' not found in environment")
        return self._name_to_obj[name]

    def has_object(self, name: str) -> bool:
        """Whether the object with the given name exists in the environment."""
        return name in self._name_to_obj

    def get_object_pose(self, obj: Union[Obstacle, str]) -> Float[torch.Tensor, "4 4"]:
        """Get the object initial pose."""
        obj = obj if isinstance(obj, Obstacle) else self.get_object(obj)
        mat4x4 = pose_list_to_mat4x4(obj.pose).to(self.device)
        return mat4x4

    def get_collision_spheres(self, obj: Union[Obstacle, str]) -> Float[torch.Tensor, "n 4"]:
        """Get the collision spheres for the object (by either name or the cuRobo Obstacle)."""
        obj_name = obj.name if isinstance(obj, Obstacle) else obj
        return self._obj_to_spheres[obj_name]

    def get_aabb(self, obj: Union[Obstacle, str]) -> Float[torch.Tensor, "2 3"]:
        """Get AABB for the given object."""
        obj_name = obj.name if isinstance(obj, Obstacle) else obj
        # Compute AABB if not cached
        if obj_name not in self._obj_to_aabb:
            obj = self.get_object(obj_name)
            aabb = approximate_goal_aabb(obj).to(self.device)
            self._obj_to_aabb[obj_name] = aabb
        return self._obj_to_aabb[obj_name]

    @cached_property
    def world_aabb(self) -> Float[torch.Tensor, "2 3"]:
        """Get AABB for the entire world (i.e., union of all objects)"""
        aabbs = [self.get_aabb(obj) for obj in self.movables] + [self.get_aabb(obj) for obj in self.statics]
        aabbs = torch.stack(aabbs)
        union_lower = aabbs[:, 0].min(dim=0).values
        union_upper = aabbs[:, 1].max(dim=0).values
        union_aabb = torch.stack([union_lower, union_upper])
        return union_aabb

    def warmup_ik_solver(self, num_particles: int):
        """Warmup cuRobo IK solver."""
        q = sample_between_bounds(num_particles, bounds=self.robot_container.joint_limits)
        goal_pose = self.kin_model.get_state(q).ee_pose
        _ = self.ik_solver.solve_batch(goal_pose)

    def get_motion_gen(self, collision_activation_distance: float, use_cuda_graph: bool = True) -> MotionGen:
        """
        Get the cuRobo motion generator for the robot. If you're debugging, you should set `use_cuda_graph=False`
        """
        if self.robot_name == "panda":
            robot_cfg = franka_curobo_cfg()
        elif self.robot_name == "ur5":
            robot_cfg = ur5_curobo_cfg()
        else:
            raise ValueError(f"Unsupported robot: {self.robot_name}")

        max_num_spheres = max([len(sphs) for sphs in self._obj_to_spheres.values()])
        robot_cfg["robot_cfg"]["kinematics"]["extra_collision_spheres"]["attached_object"] = max_num_spheres
        _log.info(f"Setting number of spheres for attachments to {max_num_spheres}")

        # World config needs to include movables for cuRobo
        world_cfg = get_world_cfg(self.env, include_movables=True)
        motion_gen_cfg = MotionGenConfig.load_from_robot_config(
            robot_cfg=robot_cfg,
            world_model=world_cfg,
            use_cuda_graph=use_cuda_graph,
            collision_activation_distance=collision_activation_distance,
        )
        motion_gen = MotionGen(motion_gen_cfg)
        return motion_gen


def check_tamp_world_not_in_collision(world: TAMPWorld, collision_tol: float = 1e-6):
    """Check that the initial state of the movable objects are not in collision."""
    for obj in world.movables:
        # Transform spheres to world frame
        mat4x4 = pose_list_to_mat4x4(obj.pose).to(world.device)
        spheres = transform_spheres(world.get_collision_spheres(obj), mat4x4)  # [n, 4]
        spheres = spheres[None, None].contiguous()  # [1, 1, n, 4]

        coll_cost = world.collision_fn(spheres).sum()
        if coll_cost > collision_tol:
            raise ValueError(f"Initial state in collision for object '{obj.name}' with cost {coll_cost}")

    # TODO: catch collisions between spheres for each movable objects here
