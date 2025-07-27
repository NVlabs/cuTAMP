# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Partially modified from:
- https://github.com/rerun-io/rerun/blob/main/examples/python/ros_node/rerun_urdf.py
- https://github.com/williamshen-nz/rerun-robotics/blob/main/rerun_robotics/rerun_urdf.py
"""

import warnings
from typing import Union, cast

import numpy as np
import rerun as rr
import trimesh

from curobo.geom.types import Mesh as CuroboMesh, Obstacle

AXIS_LENGTH = 0.075


def clean_rerun_path(path: str) -> str:
    path = path.replace(".", "_")
    path = path.lstrip("/")
    return path


def trimesh_to_rerun(
    geometry: Union[trimesh.PointCloud, trimesh.Trimesh],
) -> Union[rr.Points3D, rr.Mesh3D]:
    if isinstance(geometry, trimesh.PointCloud):
        return rr.Points3D(positions=geometry.vertices, colors=geometry.colors)
    elif isinstance(geometry, trimesh.Trimesh):
        # If trimesh gives us a single vertex color for the entire mesh, we can interpret that
        # as an albedo factor for the whole primitive.
        mesh = geometry
        vertex_colors = None
        albedo_factor = None
        if hasattr(mesh.visual, "vertex_colors"):
            colors = mesh.visual.vertex_colors
            if len(colors) == 4:
                # If trimesh gives us a single vertex color for the entire mesh, we can interpret that
                # as an albedo factor for the whole primitive.
                albedo_factor = np.array(colors)
            else:
                vertex_colors = colors
        elif hasattr(mesh.visual, "material"):
            # There are other properties in trimesh material, but rerun only supports albedo
            trimesh_material = mesh.visual.material
            albedo_factor = trimesh_material.main_color
        else:
            raise NotImplementedError("Couldn't determine mesh color or material")

        return rr.Mesh3D(
            vertex_positions=mesh.vertices,
            vertex_colors=vertex_colors,
            vertex_normals=mesh.vertex_normals,
            triangle_indices=mesh.faces,
            albedo_factor=albedo_factor,
        )
    else:
        raise NotImplementedError(f"Unsupported trimesh geometry: {type(geometry)}")


def log_scene(
    scene: trimesh.Scene,
    node: str,
    path: Union[str, None] = None,
    static: bool = False,
    add_mesh: bool = True,
) -> None:
    """Log a trimesh scene to rerun."""
    path = path + "/" + node if path else node
    path = clean_rerun_path(path)

    parent = scene.graph.transforms.parents.get(node)
    children = scene.graph.transforms.children.get(node)

    node_data = scene.graph.get(frame_to=node, frame_from=parent)

    if node_data:
        # Log the transform between this node and its direct parent (if it has one!).
        if parent:
            # We assume 4x4 homogeneous transforms in column-vector (i.e., last column is translation + 1.0).
            world_from_mesh = node_data[0]
            rr.log(
                path,
                rr.Transform3D(
                    translation=world_from_mesh[:3, 3],
                    mat3x3=world_from_mesh[:3, :3],
                    axis_length=0.0,
                ),
                static=static,
            )

        # Log this node's mesh, if it has one.
        if add_mesh:
            mesh = cast(trimesh.Trimesh, scene.geometry.get(node_data[1]))
            # Log mesh as static so we can reuse it. Re-logging it has high costs
            if mesh:
                rr.log(path, trimesh_to_rerun(mesh), static=True)

    if children:
        for child in children:
            log_scene(scene, child, path, static, add_mesh)


def curobo_to_rerun(entity: CuroboMesh, compute_vertex_normals: bool = True) -> Union[rr.Mesh3D, rr.Asset3D]:
    """Convert a cuRobo entity to a rerun entity that can be logged."""
    if not isinstance(entity, CuroboMesh):
        raise NotImplementedError(f"Entity of {type(entity)} not supported yet")

    mesh = entity

    if mesh.file_path is not None:
        # Rerun needs to load the mesh from a file
        rr_mesh = rr.Asset3D(path=mesh.file_path)
    else:
        # The vertices and triangles are already in the mesh
        if not mesh.vertex_normals and compute_vertex_normals:
            try:
                # Use open3d to compute vertex normals, so we can get a prettier looking mesh
                import open3d as o3d

                o3d_mesh = o3d.geometry.TriangleMesh()
                o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
                o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
                o3d_mesh.compute_vertex_normals()
                vertex_normals = np.asarray(o3d_mesh.vertex_normals)
            except ImportError:
                warnings.warn("open3d not installed, setting vertex normals to None")
                vertex_normals = None
        else:
            vertex_normals = mesh.vertex_normals

        rr_mesh = rr.Mesh3D(
            vertex_positions=mesh.vertices,
            triangle_indices=mesh.faces,
            vertex_normals=vertex_normals,
            vertex_colors=mesh.vertex_colors,
        )
    return rr_mesh


def pose_list_to_transform3d(pose: Union[list[float], None], axis_length: Union[float, None] = None) -> rr.Transform3D:
    if pose is None:
        return rr.Transform3D()  # unit pose

    # Curobo stores poses as [x y z qw qx qy qz]
    quat_wxyz = pose[3:]
    quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
    return rr.Transform3D(translation=pose[:3], quaternion=quat_xyzw, axis_length=axis_length)


def log_curobo_pose_to_rerun(key: str, obj: Obstacle, static_transform: bool, log_arrows: bool = False):
    rr.log(
        key,
        pose_list_to_transform3d(obj.pose, axis_length=AXIS_LENGTH if log_arrows else None),
        static=static_transform,
    )


def log_curobo_mesh_to_rerun(
    key: str, mesh: CuroboMesh, static_transform: bool, static_mesh: bool = True, log_arrows: bool = False
):
    log_curobo_pose_to_rerun(key, mesh, static_transform, log_arrows)
    rr.log(key, curobo_to_rerun(mesh, compute_vertex_normals=True), static=static_mesh)
