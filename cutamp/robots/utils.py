# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Union, Sequence, Dict

import numpy as np
import rerun as rr
import torch
import trimesh
from jaxtyping import Float
from yourdfpy import URDF

from cutamp.utils.rerun_utils import clean_rerun_path, log_scene


def _get_scene_transforms(
    scene: trimesh.Scene, root_node: str, root_path: Union[str, None] = None
) -> Dict[str, Float[np.ndarray, "4 4"]]:
    """Get the transformations for each link in the scene in the expected rerun keys that we use."""
    stack = [(root_node, root_path)]
    path_to_transform = {}

    while stack:
        node, path = stack.pop()
        rerun_path = f"{path}/{node}" if path else node
        rerun_path = clean_rerun_path(rerun_path)

        # Get parent and children for this node
        parent = scene.graph.transforms.parents.get(node)
        children = scene.graph.transforms.children.get(node)
        node_data = scene.graph.get(frame_to=node, frame_from=parent)

        if node_data and parent:
            assert rerun_path not in path_to_transform
            path_to_transform[rerun_path] = node_data[0]

        # Add children to stack
        if children:
            for child in children:
                stack.append((child, rerun_path))

    return path_to_transform


class RerunRobot:
    """URDF robot for Rerun visualization."""

    def __init__(self, name: str, urdf: URDF, q_neutral: Union[Sequence[float], None] = None, load_mesh: bool = True):
        self.name = name
        self.urdf = urdf
        rr.log(self.urdf.base_link, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        log_scene(scene=self.urdf.scene, node=self.urdf.base_link, path=self.name, static=False, add_mesh=load_mesh)
        if q_neutral is not None:
            self.set_joint_positions(q_neutral)

    @property
    def joint_positions(self) -> np.ndarray:
        return self.urdf.cfg

    def set_joint_positions(self, joint_positions: Sequence[float]) -> None:
        # Append the remaining joint positions, this can happen if we're only set arm joints not gripper
        if len(joint_positions) != self.urdf.num_actuated_joints:
            start_idx = len(joint_positions)
            joint_positions = [*joint_positions, *self.urdf.cfg[start_idx:]]

        self.urdf.update_cfg(joint_positions)
        log_scene(scene=self.urdf.scene, node=self.urdf.base_link, path=self.name, static=False, add_mesh=False)

    def get_rr_columns(self, joint_positions: Float[torch.Tensor, "n d"]) -> Dict[str, rr.Transform3D.columns]:
        """
        Gets the Rerun columns for a batch of joint positions for use in columnar logging. This is especially helpful
        for logging joint trajectories, as it's much faster than individual row-oriented logging.

        See rerun documentation for more details: https://rerun.io/docs/howto/logging/send-columns

        Parameters
        ----------
        joint_positions: Float[torch.Tensor, "n d"]
           Batch of joint positions for which to get the rerun components.

        Returns
        -------
        Dict[str, rr.Transform3D.columns]
           Mapping of the rerun path to the columns for each joint position.
        """
        if joint_positions.ndim != 2:
            raise ValueError(f"joint_positions must be a 2D tensor of shape (n, d), got {joint_positions.shape}")

        # Detach and move to CPU to be safe
        joint_positions = joint_positions.detach().cpu()
        n, d = joint_positions.shape

        # Add remaining joint positions (e.g., for the gripper) if needed
        if d != self.urdf.num_actuated_joints:
            q_remaining = self.urdf.cfg[d:]
            q_remaining = torch.tensor(q_remaining, dtype=joint_positions.dtype)
            # Expand to batch size and concatenate with existing joint positions
            q_remaining = q_remaining.expand((n, len(q_remaining)))
            joint_positions = torch.cat([joint_positions, q_remaining], dim=1)

        path_to_tforms = {}  # map of path to list of transforms

        # Update the URDF with joint position and get the transformations
        for q in joint_positions:
            self.urdf.update_cfg(q.numpy())
            for path, tform in _get_scene_transforms(self.urdf.scene, self.urdf.base_link, self.name).items():
                if path not in path_to_tforms:
                    path_to_tforms[path] = []
                path_to_tforms[path].append(tform)
        path_to_tforms = {k: np.stack(v) for k, v in path_to_tforms.items()}

        # Convert transformations into rerun components
        path_to_components = {}
        for path, tforms in path_to_tforms.items():
            components = rr.Transform3D.columns(mat3x3=tforms[:, :3, :3], translation=tforms[:, :3, 3])
            path_to_components[path] = components
        return path_to_components
