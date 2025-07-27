# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Optional

import torch
from curobo.geom.types import Obstacle, Cuboid, Mesh
from jaxtyping import Float

from cutamp.utils.common import approximate_goal_aabb, pose_list_to_mat4x4, transform_points
from cutamp.utils.shapes import MultiSphere

Grasp4DOF = Place4DOF = Float[torch.Tensor, "n 4"]
Grasp6DOF = Place6DOF = Float[torch.Tensor, "n 6"]


def sample_yaw(num_samples: int, num_faces: Optional[int], device: torch.device):
    """Sample yaws. Continuous if num_faces is None, otherwise discrete."""
    if num_faces is None:
        yaw = torch.rand(num_samples, device=device) * 2 * torch.pi  # [0, 2pi)
    else:
        assert num_faces >= 1
        two_pi = 2 * torch.pi
        endpoint = two_pi - (two_pi / num_faces)
        yaw_choices = torch.linspace(0, endpoint, num_faces, device=device)
        yaw_idxs = torch.randint(0, num_faces, (num_samples,), device=device)
        yaw = yaw_choices[yaw_idxs]
    return yaw


def sample_stick_grasps(num_samples: int, stick: MultiSphere) -> Grasp4DOF:
    """Sample 4-DOF grasps for a stick."""
    spheres = stick.spheres
    if not (spheres[:, 1:3] == 0.0).all():
        raise ValueError(f"Expected stick spheres to have y and z positions of 0")

    # Randomly sample x-coordinate of the sphere
    sphere_x = spheres[:, 0]
    x_idxs = torch.randint(0, len(sphere_x), (num_samples,), device=spheres.device)
    sampled_x = sphere_x[x_idxs]

    # Sample yaws, use two faces since we just want original orientation and mirrored 180 degrees
    yaw = sample_yaw(num_samples, num_faces=2, device=spheres.device)

    # Create 4-DOF grasp!
    grasp_4dof = torch.zeros((num_samples, 4), device=spheres.device)
    grasp_4dof[:, 0] = sampled_x
    grasp_4dof[:, 3] = yaw
    return grasp_4dof


def grasp_4dof_sampler(
    num_samples: int, obj: Obstacle, obj_spheres: Float[torch.Tensor, "n 4"], num_faces: Optional[int] = None
) -> Grasp4DOF:
    """
    Sample 4-DOF grasps for the given object in the object's coordinate frame.
    This could be made a lot more sophisticated.
    """
    # Handle the stick as a special case
    if obj.name == "stick":
        obj: MultiSphere
        return sample_stick_grasps(num_samples, obj)

    # Determine point to grasp from the top of the object
    if isinstance(obj, Cuboid):
        obj_half_z = max(0.0, obj.dims[2] / 2 - 0.02)  # grasp 2cm from top of object
    elif isinstance(obj, MultiSphere):
        obj_half_z = 0.0
    else:
        max_z = obj_spheres[:, 2].max()
        obj_half_z = max_z - 0.02  # 2cm from top of obejct

    # Assume zero translation for now in x and y-axes
    translation = torch.zeros(num_samples, 3, device=obj.tensor_args.device)
    translation[:, 2] = obj_half_z

    # Sample yaw
    yaw = sample_yaw(num_samples, num_faces, obj.tensor_args.device)

    # Form full 4-DOF grasp
    grasp_4dof = torch.cat([translation, yaw.unsqueeze(-1)], dim=1)
    return grasp_4dof


def grasp_6dof_sampler(num_samples: int, obj: Obstacle, num_faces: Optional[int] = None) -> Grasp6DOF:
    """
    Sample 6-DOF grasps for the given object in the object's coordinate frame.
    Note: this is a very simple sampler which was written for the bookshelf domain and isn't general enough.
    """
    assert isinstance(obj, Cuboid), "only Cuboid objects supported for 6-dof grasps right now"
    # Sample roll from discrete choices
    roll_choices = torch.tensor(
        [-torch.pi / 2, -torch.pi / 3, -torch.pi / 4, torch.pi / 4, torch.pi / 3, torch.pi / 2],
        device=obj.tensor_args.device,
    )
    roll_idxs = torch.randint(0, len(roll_choices), (num_samples,), device=obj.tensor_args.device)
    roll = roll_choices[roll_idxs]

    # Let pitch be zero for now
    pitch = torch.zeros(num_samples, device=obj.tensor_args.device)

    # Sample yaw from discrete choices
    yaw_choices = torch.tensor([-torch.pi / 2, torch.pi / 2], device=obj.tensor_args.device)
    yaw_idxs = torch.randint(0, 2, (num_samples,), device=obj.tensor_args.device)
    yaw = yaw_choices[yaw_idxs]

    # Stack rpy
    rpy = torch.stack([roll, pitch, yaw], dim=1)

    # Compute offsets for gripper translation in object frame
    half_extents = obj.tensor_args.to_device([dim / 2 for dim in obj.dims])
    gripper_offset = 0.01
    upper = (half_extents - gripper_offset).clamp(min=0.0)
    lower = (obj.tensor_args.to_device(3 * [gripper_offset])).clamp(max=upper)
    lower[0] = upper[0] = 0.0  # remove translation in x-axis

    # Sample translation between bounds
    translation = torch.rand(num_samples, 3, device=obj.tensor_args.device)
    translation = lower + (upper - lower) * translation

    # Form 6-DOF grasps
    grasp_6dof = torch.cat([translation, rpy], dim=1)
    return grasp_6dof


def place_4dof_sampler(
    num_samples: int, obj: Obstacle, obj_spheres: Float[torch.Tensor, "n 4"], surface: Obstacle
) -> Place4DOF:
    """Sample 4-DOF placement poses in the world frame. This does not yet fully support surfaces with yaw."""
    if not isinstance(surface, (Cuboid, Mesh)):
        raise NotImplementedError(f"Only Cuboid or Mesh surfaces supported for now, not {type(surface)}")

    # Determine the z-position of the min of the object spheres (in object frame)
    sph_bottom = obj_spheres[:, 2] - obj_spheres[:, 3]
    obj_bottom = sph_bottom.min()  # used as a delta
    obj_z_delta = -obj_bottom

    # Assume the surface is a cuboid, sample xy positions within AABB in local frame
    if isinstance(surface, Cuboid):
        aabb_xy = surface.tensor_args.to_device(
            [[-surface.dims[0] / 2, -surface.dims[1] / 2], [surface.dims[0] / 2, surface.dims[1] / 2]]
        )
        surface_z = surface.dims[2] / 2
    else:
        # Note: this was only used for the Rummy demos in the past, so not tested extensively.
        aabb = approximate_goal_aabb(surface).to(obj.tensor_args.device)
        aabb_xy = aabb[:, :2]
        surface_z = aabb[1, 2]

    xy = torch.rand(num_samples, 2, device=obj.tensor_args.device)
    xy = aabb_xy[0] + xy * (aabb_xy[1] - aabb_xy[0])

    # TODO: consider collision activation distance
    # Sample z-offset and combine with xy
    z_lower, z_upper = 1e-3, 1e-2
    z = torch.rand(num_samples, 1, device=obj.tensor_args.device)
    z = z_lower + (z_upper - z_lower) * z
    z += obj_z_delta + surface_z
    xyz = torch.cat([xy, z], dim=1)

    # Transform to surface coordinate frame
    if isinstance(surface, Cuboid):
        surface_mat4x4 = pose_list_to_mat4x4(surface.pose).to(obj.tensor_args.device)
        xyz_surface = transform_points(xyz, surface_mat4x4)
    else:
        xyz_surface = xyz

    # Sample yaw
    yaw = sample_yaw(num_samples, None, obj.tensor_args.device)
    place_4dof = torch.cat([xyz_surface, yaw.unsqueeze(-1)], dim=1)
    return place_4dof
