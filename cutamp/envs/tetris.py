# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Literal, Optional

import torch
from curobo.geom.types import Cuboid
from curobo.types.base import TensorDeviceType
from jaxtyping import Float
from roma import euler_to_unitquat

from cutamp.envs import TAMPEnvironment
from cutamp.envs.utils import create_walls_for_cuboid, unit_quat
from cutamp.tamp_domain import HandEmpty, On
from cutamp.utils.shapes import MultiSphere

# Specified in terms of xyz coordinates (discrete)
_shape_coords = {
    "L": [
        (0, 0, 0),
        (0, 1, 0),  # up
        (0, -1, 0),  # down
        (1, -1, 0),  # right
    ],
    "O": [
        (0, 0, 0),
        (0, 1, 0),
        (1, 0, 0),
        (1, 1, 0),
    ],
}


def _create_tetris_spheres(
    shape: str,
    sph_radius: float,
    tensor_args: Optional[TensorDeviceType] = None,
) -> Float[torch.Tensor, "n 4"]:
    if shape not in _shape_coords:
        raise ValueError(f"Invalid shape: {shape}")

    coords = _shape_coords[shape]
    spheres = []
    for idx, (x, y, z) in enumerate(coords):
        spheres.append([x * sph_radius * 2, y * sph_radius * 2, z * sph_radius * 2, sph_radius])

    # Add stick to grasp onto
    spheres.append([0.0, 0.0, sph_radius * 1.25, sph_radius / 2])
    spheres.append([0.0, 0.0, sph_radius * 2, sph_radius / 2])

    if tensor_args is None:
        tensor_args = TensorDeviceType()
    spheres = tensor_args.to_device(spheres)

    # Shift the sphere z-positions
    z_offset = -spheres[-1][2]
    spheres[:, 2] += z_offset
    return spheres


def _sample_yaw():
    yaw = torch.rand(1) * 2 * torch.pi
    quat_xyzw = euler_to_unitquat("XYZ", torch.tensor([0.0, 0.0, yaw.item()]))
    quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
    return quat_wxyz.tolist()


def load_tetris_env(
    num_blocks: Literal[1, 2, 3, 5],
    sph_radius: float = 0.03,
    buffer_multiplier: float = 1.0,
    enable_walls: bool = True,
    wall_height: float = 0.045,
    random_yaws: bool = False,
    tensor_args: TensorDeviceType = TensorDeviceType(),
) -> TAMPEnvironment:
    """Tetris scenario with either 1, 3 or 5 blocks and the corresponding goal region."""
    if num_blocks not in {1, 2, 3, 5}:
        raise ValueError(f"Invalid number of blocks: {num_blocks}. Must be 1, 3 or 5.")

    # Table
    table = Cuboid(name="table", dims=[1.1, 1.5, 0.02], pose=[0.15, 0.0, -0.011, *unit_quat], color=[235, 196, 145])

    # Create spheres for the blocks
    L_sphs = _create_tetris_spheres("L", sph_radius, tensor_args)
    O_sphs = _create_tetris_spheres("O", sph_radius, tensor_args)
    L_block_z = -(L_sphs[:, 2] - L_sphs[:, 3]).min().item() + 1e-2
    O_block_z = -(O_sphs[:, 2] - O_sphs[:, 3]).min().item() + 1e-2

    # Now create the blocks
    quat_fn = _sample_yaw if random_yaws else lambda: unit_quat
    block_1 = MultiSphere(name="O_1", spheres=O_sphs, pose=[0.5, 0.45, O_block_z, *quat_fn()], color=[255, 255, 186])
    block_2 = MultiSphere(name="L_1", spheres=L_sphs, pose=[0.3, -0.5, L_block_z, *quat_fn()], color=[255, 186, 201])
    block_3 = MultiSphere(name="L_2", spheres=L_sphs, pose=[0.0, 0.45, L_block_z, *quat_fn()], color=[186, 201, 255])
    block_4 = MultiSphere(name="L_3", spheres=L_sphs, pose=[0.3, 0.5, L_block_z, *quat_fn()], color=[186, 255, 201])
    block_5 = MultiSphere(name="L_4", spheres=L_sphs, pose=[0.0, -0.45, L_block_z, *quat_fn()], color=[102, 186, 232])
    blocks = [block_1, block_2, block_3, block_4, block_5]
    if num_blocks == 2:
        blocks = [block_2, block_3]  # if two blocks just use the two L blocks
    else:
        blocks = blocks[:num_blocks]

    # Goal region
    buffer = sph_radius * buffer_multiplier
    diameter = sph_radius * 2
    goal_width = diameter * 2 + buffer
    if num_blocks == 1:
        goal_height = diameter * 2 + buffer
    elif num_blocks == 2:
        goal_height = diameter * 4 + buffer
    elif num_blocks == 3:
        goal_height = diameter * 6 + buffer
    else:
        goal_height = diameter * 10 + buffer
    goal_region = Cuboid(
        name="goal", dims=[goal_width, goal_height, 0.01], pose=[0.4, 0.0, -0.005, *unit_quat], color=[186, 255, 201]
    )

    # Walls for the goal region
    if enable_walls:
        goal_walls = create_walls_for_cuboid(goal_region, wall_height, wall_thickness=0.015, wall_color=[200, 200, 200])
    else:
        goal_walls = []

    # Goal is all blocks in the goal region
    goal_state = frozenset({HandEmpty.ground(), *[On.ground(block.name, goal_region.name) for block in blocks]})

    env = TAMPEnvironment(
        name=f"tetris_{num_blocks}_blocks",
        movables=blocks,
        statics=[table, goal_region, *goal_walls],
        type_to_objects={"Movable": blocks, "Surface": [table, goal_region]},
        goal_state=goal_state,
    )
    return env
