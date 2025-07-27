# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from collections import defaultdict
from typing import Dict

import roma
import torch
from curobo.geom.types import (
    Capsule,
    Cuboid,
    Cylinder,
    Mesh,
    Obstacle,
    Sphere,
    WorldConfig,
)
from einops import einsum
from jaxtyping import Float
from roma import quat_wxyz_to_xyzw, unitquat_to_rotmat

from cutamp.envs import TAMPEnvironment
from cutamp.utils.shapes import MultiSphere

Particles = Dict[str, Float[torch.Tensor, "num_particles *h d"]]


def pose_list_to_mat4x4(pose: list[float] | None) -> Float[torch.Tensor, "4 4"]:
    """cuRobo pose list to 4x4 transformation matrix."""
    mat4x4 = torch.eye(4)
    if pose is not None:
        mat4x4[:3, 3] = torch.tensor(pose[:3])
        # Curobo stores poses as [x y z qw qx qy qz]
        quat_wxyz = torch.tensor(pose[3:])
        quat_xyzw = quat_wxyz_to_xyzw(quat_wxyz)
        mat4x4[:3, :3] = unitquat_to_rotmat(quat_xyzw)
    return mat4x4


def action_4dof_to_mat4x4(action_4dof: Float[torch.Tensor, "*b 4"]) -> Float[torch.Tensor, "*b 4 4"]:
    """Convert 4-DOF action to 4x4 transformation matrix."""
    if action_4dof.shape[-1] != 4:
        raise ValueError(f"Expected last dimension to be 4, got shape {action_4dof.shape}")

    # Create 4x4 matrix and set rotation and translation
    mat4x4 = torch.eye(4, device=action_4dof.device, dtype=action_4dof.dtype)
    mat4x4 = mat4x4.repeat(*action_4dof.shape[:-1], 1, 1)

    # Set the rotation component
    yaw = action_4dof[..., 3]
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    mat4x4[..., 0, 0] = cos_yaw
    mat4x4[..., 0, 1] = -sin_yaw
    mat4x4[..., 1, 0] = sin_yaw
    mat4x4[..., 1, 1] = cos_yaw

    # Set the translation component
    mat4x4[..., :3, 3] = action_4dof[..., :3]
    return mat4x4


def action_6dof_to_mat4x4(action_6dof: Float[torch.Tensor, "*b 6"]) -> Float[torch.Tensor, "*b 4 4"]:
    """Convert 6-DOF action (xyz + rpy) to 4x4 transformation matrix."""
    translation = action_6dof[..., :3]
    rpy = action_6dof[..., 3:]

    # Create 4x4 matrix and set rotation and translation
    mat4x4 = torch.eye(4, device=action_6dof.device, dtype=action_6dof.dtype)
    mat4x4 = mat4x4.repeat(*action_6dof.shape[:-1], 1, 1)
    mat4x4[..., :3, :3] = roma.euler_to_rotmat("XYZ", rpy)
    mat4x4[..., :3, 3] = translation
    return mat4x4


def transform_spheres(
    spheres: Float[torch.Tensor, "num_spheres 4"], transform: Float[torch.Tensor, "*b 4 4"]
) -> Float[torch.Tensor, "*b num_spheres 4"]:
    """Transform spheres by the given transformation matrices"""
    centers = spheres[:, :3]
    centers_hom = torch.cat([centers, torch.ones_like(centers[:, :1])], dim=1)
    radii = spheres[:, 3]

    # Add a dimension to broadcast over num_spheres
    if transform.ndim > 2:
        # (*b, 4, 4) -> (*b, 1, 4, 4)
        transform = transform.unsqueeze(-3)

    out_spheres = einsum(transform, centers_hom, "... i j, ... j -> ... i")
    out_spheres[..., 3] = radii
    return out_spheres


def transform_points(
    points: Float[torch.Tensor, "num_points 3"], transform: Float[torch.Tensor, "*b 4 4"]
) -> Float[torch.Tensor, "*b num_points 3"]:
    """Transform points by the given transformation matrices"""
    points_hom = torch.cat([points, torch.ones_like(points[:, :1])], dim=1)

    # Add a dimension to broadcast over num_points
    if transform.ndim > 2:
        # (*b, 4, 4) -> (*b, 1, 4, 4)
        transform = transform.unsqueeze(-3)

    out_points_hom = einsum(transform, points_hom, "... i j, ... j -> ... i")
    out_points = out_points_hom[..., :3]
    return out_points


def sample_between_bounds(num_samples: int, bounds: Float[torch.Tensor, "2 d"]) -> Float[torch.Tensor, "num_samples d"]:
    """
    Sample uniformly between bounds.
    The bounds should be a (2, d) tensor where the first row is the lower bound and the second row is the upper bound.
    """
    lower, upper = bounds
    samples = torch.rand(num_samples, *lower.shape, device=bounds.device, dtype=bounds.dtype)
    samples = lower + samples * (upper - lower)
    return samples


def approximate_goal_aabb(goal: Obstacle) -> Float[torch.Tensor, "2 3"]:
    """Approximate the goal AABB from the goal object."""
    # Compute transformation matrix for object pose
    pose = torch.tensor(goal.pose)
    pos, wxyz = pose[:3], pose[3:]
    xyzw = roma.quat_wxyz_to_xyzw(wxyz)
    mat4x4 = torch.eye(4)
    mat4x4[:3, :3] = roma.unitquat_to_rotmat(xyzw)
    mat4x4[:3, 3] = pos
    if isinstance(goal, MultiSphere):
        goal = goal.get_mesh()

    if isinstance(goal, Mesh):
        vertices = torch.tensor(goal.vertices, dtype=mat4x4.dtype)
        vertices = transform_points(vertices, mat4x4)
        lower = vertices.min(dim=0).values
        upper = vertices.max(dim=0).values
        aabb = torch.stack([lower, upper])
    elif isinstance(goal, Cuboid):
        # TODO: handle cases when the goal is not axis-aligned. i.e., has rotation
        goal_xyz = torch.tensor(goal.dims)
        aabb = torch.stack([-goal_xyz / 2, goal_xyz / 2])
        aabb = transform_points(aabb, mat4x4)
    else:
        raise NotImplementedError(f"Goal type {type(goal)} not supported yet.")

    return aabb


def get_world_cfg(env: TAMPEnvironment, include_movables: bool = False) -> WorldConfig:
    """Get the cuRobo WorldConfig from the TAMP environment."""
    geoms = defaultdict(list)
    obstacles = env.movables if include_movables else []
    obstacles += env.statics
    for obj in obstacles:
        if isinstance(obj, Cuboid):
            geoms["cuboid"].append(obj)
        elif isinstance(obj, Sphere):
            geoms["sphere"].append(obj)
        elif isinstance(obj, Cylinder):
            geoms["cylinder"].append(obj)
        elif isinstance(obj, Capsule):
            geoms["capsule"].append(obj)
        elif isinstance(obj, (MultiSphere, Mesh)):
            # Need to use mesh for MultiSphere
            geoms["mesh"].append(obj)
        else:
            raise ValueError(f"Unknown object type: {type(obj)}")
    return WorldConfig(**geoms)
