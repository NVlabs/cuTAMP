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
from jaxtyping import Float

from curobo.types.math import Pose


def trajectory_length(
    confs: Float[torch.Tensor, "b *h d"], weights: Optional[Float[torch.Tensor, "d"]] = None
) -> Float[torch.Tensor, "b"]:
    """
    Compute the length of a trajectory by summing the distances between consecutive configurations.
    Optionally apply per-dimension weights before computing distances.
    """
    if weights is not None:
        assert weights.shape == (confs.shape[-1],), f"weights must be shape ({confs.shape[-1]},), got {weights.shape}"
        weights = weights.to(confs.device)
        confs = confs * weights

    diffs = confs[..., 1:, :] - confs[..., :-1, :]
    dists = diffs.norm(dim=-1)
    traj_lengths = dists.sum(-1)  # sum over horizon
    return traj_lengths


def dist_from_bounds(
    vals: Float[torch.Tensor, "b *h d"],
    lower: Float[torch.Tensor, "d"],
    upper: Float[torch.Tensor, "d"],
) -> Float[torch.Tensor, "b *h"]:
    """Euclidean distance of values from the given lower and upper bounds. If within the bounds, returns 0."""
    diff_lower = lower - vals
    diff_upper = vals - upper
    diff_max = torch.maximum(diff_lower, diff_upper)
    diff_max = diff_max.clamp(min=0.0)
    dists = diff_max.norm(p=2, dim=-1)
    return dists


@torch.jit.script
def dist_from_bounds_jit(vals, lower, upper):
    """Euclidean distance of values from the given lower and upper bounds. If within the bounds, returns 0."""
    diff_lower = lower - vals
    diff_upper = vals - upper
    diff_max = torch.maximum(diff_lower, diff_upper)
    diff_max = diff_max.clamp(min=0.0)
    dists = diff_max.norm(p=2, dim=-1)
    return dists


def get_aabb_from_spheres(spheres: Float[torch.Tensor, "*b n 4"]) -> Float[torch.Tensor, "*b 2 3"]:
    """Compute the axis-aligned bounding box (AABB) for a set of spheres that represent some object."""
    centers, radii = spheres[..., :3], spheres[..., 3]
    min_corners = centers - radii.unsqueeze(-1)
    min_corners = min_corners.min(-2).values  # take min over spheres in the batch

    max_corners = centers + radii.unsqueeze(-1)
    max_corners = max_corners.max(-2).values  # take max over spheres in the batch

    aabb = torch.stack([min_corners, max_corners], dim=-2)
    return aabb


def _sphere_to_sphere_overlap(
    spheres_1: Float[torch.Tensor, "b *h 4"],
    spheres_2: Float[torch.Tensor, "b *h 4"],
    activation_distance: float | None = None,
) -> Float[torch.Tensor, "b *h"]:
    """Compute the overlap volume between two sets of spheres. Can be used as a collision distance function."""
    centers_1, radii_1 = spheres_1[..., :3], spheres_1[..., 3]
    centers_2, radii_2 = spheres_2[..., :3], spheres_2[..., 3]

    dist_matrix = torch.cdist(centers_1, centers_2)
    radii_sum = radii_1.unsqueeze(-1) + radii_2.unsqueeze(-2)
    overlap = radii_sum - dist_matrix

    if activation_distance is not None:
        overlap += activation_distance
    return torch.relu(overlap).sum((-2, -1))


def sphere_to_sphere_overlap(
    spheres_1: Float[torch.Tensor, "b *h 4"],
    spheres_2: Float[torch.Tensor, "b *h 4"],
    activation_distance: float | None = None,
    aabb_1: Float[torch.Tensor, "b *h 2 3"] | None = None,
    aabb_2: Float[torch.Tensor, "b *h 2 3"] | None = None,
    use_aabb_check: bool = False,
) -> Float[torch.Tensor, "b *h"]:
    """
    Compute the overlap volume between two sets of spheres. Can be used as a collision distance function. If
    use_aabb_check=True, we compute the overlap only for batches of spheres that have intersecting AABBs.
    """
    if not use_aabb_check:
        return _sphere_to_sphere_overlap(spheres_1, spheres_2, activation_distance)

    # Compute AABB for each batch of spheres
    if aabb_1 is None:
        aabb_1 = get_aabb_from_spheres(spheres_1)  # [b, *h, 2, 3]
    if aabb_2 is None:
        aabb_2 = get_aabb_from_spheres(spheres_2)  # [b, *h, 2, 3]

    # Check intersection
    min_1, max_1 = aabb_1.unbind(-2)
    min_2, max_2 = aabb_2.unbind(-2)
    intersect = (min_1 <= max_2).all(-1) & (max_1 >= min_2).all(-1)  # [b, *h]

    output = torch.zeros_like(intersect, dtype=torch.float32)  # [b, *h]
    if intersect.any():
        # Only compute overlap for intersecting AABBs
        output[intersect] = _sphere_to_sphere_overlap(spheres_1[intersect], spheres_2[intersect], activation_distance)
    return output


def curobo_pose_error(
    pose_a_mat4x4: Float[torch.Tensor, "b *h 4 4"], pose_b_mat4x4: Float[torch.Tensor, "b *h 4 4"]
) -> (Float[torch.Tensor, "b *h"], Float[torch.Tensor, "b *h"]):
    """
    Compute the translational and rotational errors between two poses using curobo. Thanks Bala.
    """
    # Flatten
    pose_a_flat = pose_a_mat4x4.view(-1, 4, 4)
    pose_b_flat = pose_b_mat4x4.view(-1, 4, 4)

    # Create curobo pose
    pose_a = Pose.from_matrix(pose_a_flat)
    pose_b = Pose.from_matrix(pose_b_flat)

    # Compute distance and unflatten
    p_dist_flat, quat_dist_flat = pose_a.distance(pose_b)
    p_dist = p_dist_flat.view(pose_a_mat4x4.shape[:-2])
    quat_dist = quat_dist_flat.view(pose_b_mat4x4.shape[:-2])
    assert p_dist.shape == quat_dist.shape
    return p_dist, quat_dist
