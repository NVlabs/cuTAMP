# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Tuple, List, Optional

import numpy as np
import torch
import trimesh
from _heapq import heappush, heappop
from jaxtyping import Float

from curobo.geom.sphere_fit import SphereFitType
from curobo.geom.types import Obstacle, Sphere
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose

_log = logging.getLogger(__name__)


@dataclass
class MultiSphere(Obstacle):
    """Obstacle represented as a collection of spheres."""

    #: Spheres, (n, 4) where each row is (x, y, z, radius)
    spheres: Float[torch.Tensor, "n 4"] = field(default_factory=lambda: torch.tensor([]))

    def __post_init__(self):
        # Check spheres is not empty and (n, 4)
        self.spheres = self.spheres.to(self.tensor_args.device)
        if len(self.spheres) == 0:
            raise ValueError("Spheres must not be empty")
        if self.spheres.ndim != 2 or self.spheres.shape[1] != 4:
            raise ValueError(f"Spheres should be (n, 4) not {self.spheres.shape}")
        self.radius = self.spheres[0, 3].item()

    def get_trimesh_mesh(self, process: bool = True, process_color: bool = True) -> trimesh.Trimesh:
        """Create a trimesh instance from the obstacle representation.

        Args:
            process: Flag is not used.
            process_color: Flag is not used.

        Returns:
            trimesh.Trimesh: Instance of obstacle as a trimesh.
        """
        # Create base icosphere
        base_sphere = trimesh.creation.icosphere(radius=1.0)
        all_vertices = []
        all_faces = []

        # Vertex count so we can accumulate them for the faces
        vertex_count = 0
        for sphere in self.spheres.cpu().numpy():
            # Scale and translate the base sphere
            x, y, z, radius = sphere
            vertices = base_sphere.vertices * radius + [x, y, z]

            # Add vertices and faces for this sphere
            all_vertices.append(vertices)
            all_faces.append(base_sphere.faces + vertex_count)
            vertex_count += len(vertices)

        # Combine all vertices and faces to create the final mesh
        vertices = np.vstack(all_vertices)
        faces = np.vstack(all_faces)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        if self.color is not None:
            color_visual = trimesh.visual.color.ColorVisuals(
                mesh=mesh, face_colors=self.color, vertex_colors=self.color
            )
            mesh.visual = color_visual
        return mesh

    def get_mesh_data(self, process: bool = True) -> Tuple[List[List[float]], List[int]]:
        m = self.get_trimesh_mesh(process=process)
        verts = m.vertices.view(np.ndarray)
        faces = m.faces
        return verts, faces

    def get_bounding_spheres(
        self,
        n_spheres: int = 1,
        surface_sphere_radius: float = 0.002,
        fit_type: SphereFitType = SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE,
        voxelize_method: str = "ray",
        pre_transform_pose: Optional[Pose] = None,
        tensor_args: TensorDeviceType = TensorDeviceType(),
    ) -> List[Sphere]:
        """Use the ground truth spheres as the bounding spheres. Ignores most of the arguments."""
        pts = self.spheres[:, :3].cpu().numpy()
        n_radius = self.spheres[:, 3].cpu().numpy()

        obj_pose = Pose.from_list(self.pose, tensor_args)
        if pre_transform_pose is not None:
            obj_pose = pre_transform_pose.multiply(obj_pose)  # convert object pose to another frame

        if pts is None or len(pts) == 0:
            raise ValueError("No points found from the spheres")

        points_cuda = tensor_args.to_device(pts)
        pts = obj_pose.transform_points(points_cuda).cpu().view(-1, 3).numpy()

        new_spheres = [
            Sphere(
                name=f"{self.name}_sph_{i}",
                pose=[pts[i, 0], pts[i, 1], pts[i, 2], 1, 0, 0, 0],
                radius=n_radius[i],
            )
            for i in range(pts.shape[0])
        ]
        return new_spheres


def sample_collision_spheres(
    obj: Obstacle,
    n_spheres: int = 50,
    surface_sphere_radius: float = 0.005,
    fit_type: SphereFitType = SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE,
    voxelize_method: str = "subdivide",
) -> Float[torch.Tensor, "n 4"]:
    """Sample spheres for collision checking using cuRobo. Note the spheres will be in the object's frame."""
    # Need to temporarily override the pose, so the spheres are at the origin
    og_pose = obj.pose
    obj.pose = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]  # unit pose
    sph_objs = obj.get_bounding_spheres(
        n_spheres=n_spheres,
        fit_type=fit_type,
        voxelize_method=voxelize_method,
        surface_sphere_radius=surface_sphere_radius,
    )
    obj.pose = og_pose

    # Convert to (num_spheres, 4) tensor
    centers = torch.tensor([sph.pose[:3] for sph in sph_objs], dtype=torch.float32)
    radii = torch.tensor([sph.radius for sph in sph_objs], dtype=torch.float32)
    spheres = torch.cat([centers, radii[:, None]], dim=1)
    return spheres


def sample_greedy_surface_spheres(
    obj: Obstacle,
    n_spheres: int,
    sphere_radius: float,
    n_samples: int = 1000,
) -> Float[torch.Tensor, "n 4"]:
    """
    Samples approximately non-overlapping surface spheres on a 3D object using a greedy algorithm. Thanks Caelan!

    Samples points on the object's surface mesh, constructs a neighborhood graph using a KD-tree,
    and greedily selects points that are far apart (non-overlapping) based on radius.
    """
    if isinstance(obj, MultiSphere):
        _log.debug(f"{obj.name} is MultiSphere, using existing spheres of shape {obj.spheres.shape}")
        return obj.spheres

    # Need to temporarily override the pose, so the trimesh mesh is at the origin
    start_time = time.perf_counter()
    og_pose = obj.pose
    obj.pose = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]  # unit pose
    mesh = obj.get_trimesh_mesh()
    obj.pose = og_pose

    # Sample surface
    n_samples = max(n_spheres, n_samples)
    n_pts = trimesh.sample.sample_surface(mesh, n_samples)[0]
    cloud = trimesh.PointCloud(n_pts)

    # Build neighbor graph
    pairs = cloud.kdtree.query_pairs(sphere_radius, eps=1e-6)
    neighbors = defaultdict(set)
    for idx1, idx2 in pairs:
        neighbors[idx1].add(idx2)
        neighbors[idx2].add(idx1)
    # print(f"Neighbors) Samples: {n_samples} m | Elapsed: {time.perf_counter() - start_time:.3f}")

    # Greedy selection
    selected = []
    queue = []
    for idx in neighbors:
        heappush(queue, (-len(neighbors[idx]), idx))

    while queue and (len(selected) < n_spheres):
        num, idx = heappop(queue)
        if len(neighbors[idx]) != -num:
            heappush(queue, (-len(neighbors[idx]), idx))
            continue

        # Remove selected point and update neighbors
        for idx2 in list(neighbors[idx]):
            for idx3 in neighbors[idx2]:
                neighbors[idx3].discard(idx2)
            del neighbors[idx2]
        del neighbors[idx]
        selected.append(idx)

    # Convert selected points to tensor format
    selected_points = n_pts[selected]
    radii = torch.full((len(selected_points), 1), sphere_radius)

    # Combine points and radii into single tensor
    result = torch.cat([torch.tensor(selected_points, dtype=torch.float32), radii], dim=1)
    _log.debug(f"Sampled {len(result)} spheres for {obj.name} in {time.perf_counter() - start_time:.3f}s")
    return result
