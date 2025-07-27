# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Tuple, List

from curobo.geom.types import Cuboid

from cutamp.envs.utils import unit_quat, TAMPEnvironment
from cutamp.tamp_domain import On, HandEmpty


def _create_shelf(
    shelf_width: float = 0.3,
    shelf_depth: float = 0.2,
    shelf_height: float = 0.3,
    shelf_thickness: float = 0.03,
    z_offset: float = 0.0,
) -> Tuple[Cuboid, List[Cuboid]]:
    """
    Create a shelf. Returns the shelf support (goal) and a list of the shelf walls (obstacles).
    This assumes it's on the left-side of the table (facing the robot).
    """
    shelf = Cuboid(
        name="shelf",
        dims=[shelf_width, shelf_depth, shelf_thickness],
        pose=[0.55, -0.425, z_offset + shelf_thickness / 2, *unit_quat],
        color=[186, 255, 201],
    )
    shelf_wall_left = Cuboid(
        name="shelf_wall_left",
        dims=[shelf_thickness, shelf_depth, shelf_height],
        pose=[
            shelf.pose[0] - shelf_width / 2 - shelf_thickness / 2,
            shelf.pose[1],
            z_offset + shelf_height / 2,
            *unit_quat,
        ],
        color=[255, 255, 255],
    )
    shelf_wall_right = Cuboid(
        name="shelf_wall_right",
        dims=[shelf_thickness, shelf_depth, shelf_height],
        pose=[
            shelf.pose[0] + shelf_width / 2 + shelf_thickness / 2,
            shelf.pose[1],
            z_offset + shelf_height / 2,
            *unit_quat,
        ],
        color=[255, 255, 255],
    )
    shelf_wall_back = Cuboid(
        name="shelf_wall_back",
        dims=[shelf_width + 2 * shelf_thickness, shelf_thickness, shelf_height],
        pose=[
            shelf.pose[0],
            shelf.pose[1] - shelf_depth / 2 - shelf_thickness / 2,
            z_offset + shelf_height / 2,
            *unit_quat,
        ],
        color=[255, 255, 255],
    )
    shelf_top = Cuboid(
        name="shelf_top",
        dims=[shelf_width + 2 * shelf_thickness, shelf_depth + shelf_thickness, shelf_thickness],
        pose=[
            shelf.pose[0],
            shelf.pose[1] - shelf_thickness / 2,
            z_offset + shelf_height + shelf_thickness / 2,
            *unit_quat,
        ],
        color=[255, 255, 255],
    )
    return shelf, [shelf_wall_left, shelf_wall_right, shelf_wall_back, shelf_top]


def load_book_shelf_env(include_obstacle: bool = False) -> TAMPEnvironment:
    """Book Shelf environment where we have some books and need to put into the shelf."""
    # Big table to make things pretty
    big_table = Cuboid(
        name="big_table", dims=[1.0, 1.3, 0.02], pose=[0.25, 0.0, -0.0111, *unit_quat], color=[202, 164, 114]
    )
    # The table is the active workspace
    table = Cuboid(name="table", dims=[0.8, 1.3, 0.02], pose=[0.5, 0.0, -0.011, *unit_quat], color=[235, 196, 145])
    platform = Cuboid(name="platform", dims=[0.5, 0.45, 0.15], pose=[0.55, -0.4, 0.075, *unit_quat], color=[99, 59, 59])

    # Shelf and its walls
    platform_bottom = platform.dims[2]
    shelf, shelf_walls = _create_shelf(z_offset=platform_bottom)

    # Books!
    buffer = 1e-2
    book_blue = Cuboid(
        name="book_blue", dims=[0.05, 0.13, 0.2], pose=[0.4, 0.3, 0.1 + buffer, *unit_quat], color=[0, 0, 255]
    )
    book_green = Cuboid(
        name="book_green", dims=[0.03, 0.15, 0.15], pose=[0.55, 0.2, 0.075 + buffer, *unit_quat], color=[0, 255, 0]
    )
    movables = [book_blue, book_green]

    # Obstacle in the shelf
    if include_obstacle:
        obstacle = Cuboid(
            name="obstacle",
            dims=[0.05, 0.05, 0.1],
            pose=[
                shelf.pose[0],
                shelf.pose[1] + shelf.dims[1] / 2 - 0.05,
                shelf.pose[2] + shelf.dims[2] / 2 + 0.05 + buffer,
                *unit_quat,
            ],
            color=[255, 0, 0],
        )
        movables.append(obstacle)

    # Goal
    goal_state = frozenset(
        {On.ground(book_blue.name, shelf.name), On.ground(book_green.name, shelf.name), HandEmpty.ground()}
    )

    env = TAMPEnvironment(
        name="book_shelf",
        movables=movables,
        statics=[big_table, table, platform, shelf] + shelf_walls,
        type_to_objects={
            "Movable": movables,
            "Surface": [table, shelf],
        },
        goal_state=goal_state,
    )
    return env
