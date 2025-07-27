# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import rerun as rr
import torch
from curobo.geom.types import Mesh
from jaxtyping import Float

from cutamp.config import TAMPConfiguration
from cutamp.robots import load_rerun_robot
from cutamp.tamp_world import TAMPWorld
from cutamp.utils.rerun_utils import log_curobo_pose_to_rerun, curobo_to_rerun, log_curobo_mesh_to_rerun, AXIS_LENGTH


class Visualizer(ABC):
    def __init__(self, config: TAMPConfiguration, q_init: Float[torch.Tensor, "d"]):
        self.config = config
        self.set_joint_positions(q_init)

    @abstractmethod
    def set_time_sequence(self, timeline: str, val: int):
        raise NotImplementedError

    @abstractmethod
    def set_time_seconds(self, timeline: str, val: float):
        raise NotImplementedError

    @abstractmethod
    def set_joint_positions(self, q: Float[Union[torch.Tensor, np.ndarray, list], "d"]):
        raise NotImplementedError

    @abstractmethod
    def log_joint_trajectory(self, traj: Float[torch.Tensor, "n d"], timeline: str, start_time: float, dt: float):
        raise NotImplementedError

    @abstractmethod
    def log_joint_trajectory_with_mat4x4(
        self,
        traj: Float[torch.Tensor, "n d"],
        mat4x4_key: str,
        mat4x4: Float[torch.Tensor, "n 4 4"],
        timeline: str,
        start_time: float,
        dt: float,
    ):
        pass

    @abstractmethod
    def log_tamp_world(self, world: TAMPWorld):
        raise NotImplementedError

    @abstractmethod
    def log_mat4x4(self, name: str, mat4x4: Float[Union[torch.Tensor, np.ndarray], "4 4"]):
        raise NotImplementedError

    @abstractmethod
    def log_spheres(self, name: str, spheres: Float[torch.Tensor, "n 4"]):
        raise NotImplementedError

    @abstractmethod
    def log_scalar(self, name: str, value: float):
        raise NotImplementedError

    def log_cost_dict(self, cost_dict: dict):
        for cost_type, cost_info in cost_dict.items():
            for name, vals in cost_info["values"].items():
                self.log_scalar(f"{cost_type}/{name}", vals.mean().item())


class MockVisualizer(Visualizer):
    """Visualizer that does nothing."""

    def __init__(self):
        super().__init__(None, None)

    def set_time_sequence(self, timeline: str, val: int):
        pass

    def set_time_seconds(self, timeline: str, val: float):
        pass

    def set_joint_positions(self, q: Float[Union[torch.Tensor, np.ndarray, list], "d"]):
        pass

    def log_joint_trajectory(self, traj: Float[torch.Tensor, "n d"], timeline: str, start_time: float, dt: float):
        pass

    def log_joint_trajectory_with_mat4x4(
        self,
        traj: Float[torch.Tensor, "n d"],
        mat4x4_key: str,
        mat4x4: Float[torch.Tensor, "n 4 4"],
        timeline: str,
        start_time: float,
        dt: float,
    ):
        pass

    def log_tamp_world(self, world: TAMPWorld):
        pass

    def log_mat4x4(self, name: str, mat4x4: Float[Union[torch.Tensor, np.ndarray], "4 4"]):
        pass

    def log_spheres(self, name: str, spheres: Float[torch.Tensor, "n 4"]):
        pass

    def log_scalar(self, name: str, value: float):
        pass


class RerunVisualizer(Visualizer):
    """Wrapper around rerun for easier visualization, and switching in different visualizers."""

    def __init__(
        self,
        config: TAMPConfiguration,
        q_init: Float[torch.Tensor, "d"],
        application_id: str,
        recording_id: str,
        spawn: bool,
    ):
        rr.init(application_id, recording_id=recording_id, spawn=spawn)
        self.robot = load_rerun_robot(config.robot, load_mesh=config.viz_robot_mesh)
        super().__init__(config, q_init)

    def set_time_sequence(self, timeline: str, val: int):
        rr.set_time_sequence(timeline, val)

    def set_time_seconds(self, timeline: str, val: float):
        rr.set_time_seconds(timeline, val)

    def set_joint_positions(self, q: Float[Union[torch.Tensor, np.ndarray, list], "d"]):
        if isinstance(q, torch.Tensor):
            q = q.tolist()
        self.robot.set_joint_positions(q)

    def log_joint_trajectory(self, traj: Float[torch.Tensor, "n d"], timeline: str, start_time: float, dt: float):
        end_time = start_time + len(traj) * dt
        times = [rr.TimeColumn(timeline, duration=np.linspace(start_time, end_time, len(traj)))]
        key_to_columns = self.robot.get_rr_columns(traj)
        for key, columns in key_to_columns.items():
            rr.send_columns(key, indexes=times * len(columns), columns=columns)
        return end_time

    def log_joint_trajectory_with_mat4x4(
        self,
        traj: Float[torch.Tensor, "n d"],
        mat4x4_key: str,
        mat4x4: Float[torch.Tensor, "n 4 4"],
        timeline: str,
        start_time: float,
        dt: float,
    ):
        if traj.shape[0] != mat4x4.shape[0]:
            raise ValueError("Trajectory and mat4x4 must have the same length.")
        end_time = start_time + len(traj) * dt
        times = [rr.TimeColumn(timeline, duration=np.linspace(start_time, end_time, len(traj)))]
        key_to_columns = self.robot.get_rr_columns(traj)

        if mat4x4_key in key_to_columns:
            raise ValueError(f"Key {mat4x4_key} already exists in key_to_components.")
        mat4x4 = mat4x4.detach().cpu()
        key_to_columns[mat4x4_key] = rr.Transform3D.columns(mat3x3=mat4x4[:, :3, :3], translation=mat4x4[:, :3, 3])

        for key, columns in key_to_columns.items():
            rr.send_columns(key, indexes=times * len(columns), columns=columns)
        return end_time

    def log_tamp_world(self, world: TAMPWorld):
        rr_log_tamp_world(world)

    def log_mat4x4(self, name: str, mat4x4: Float[Union[torch.Tensor, np.ndarray], "4 4"]):
        if isinstance(mat4x4, torch.Tensor):
            mat4x4 = mat4x4.detach().cpu()
        rr.log(name, rr.Transform3D(translation=mat4x4[:3, 3], mat3x3=mat4x4[:3, :3], axis_length=AXIS_LENGTH))

    def log_spheres(self, name: str, spheres: Float[torch.Tensor, "n 4"]):
        if isinstance(spheres, torch.Tensor):
            spheres = spheres.detach().cpu()
        rr.log(name, rr.Points3D(positions=spheres[:, :3], radii=spheres[:, 3]))

    def log_scalar(self, name: str, value: float):
        rr.log(name, rr.Scalars(value))


def rr_log_tamp_world(world: TAMPWorld, log_spheres: bool = True, log_arrows: bool = True):
    """Log TAMPWorld in rerun."""
    # Log movables
    for obj in world.movables:
        log_curobo_pose_to_rerun(f"world/{obj.name}", obj, static_transform=False, log_arrows=log_arrows)
        # Log mesh under subkey and as static since it's not changing
        rr.log(f"world/{obj.name}/mesh", curobo_to_rerun(obj.get_mesh(), compute_vertex_normals=True), static=True)

        # Get average color for the object
        if isinstance(obj, Mesh):
            rgb = np.array(obj.vertex_colors).mean(0) if obj.vertex_colors else [0.5, 0.5, 0.5]
        else:
            rgb = obj.color

        # Log collision spheres under subkey
        if log_spheres:
            spheres = world.get_collision_spheres(obj)
            rr.log(
                f"world/{obj.name}/spheres",
                rr.Points3D(positions=spheres[:, :3].cpu(), radii=spheres[:, 3].cpu(), colors=rgb),
                static=True,
            )

    # Static objects
    for obj in world.statics:
        log_curobo_mesh_to_rerun(f"world/{obj.name}", obj.get_mesh(), static_transform=True)
