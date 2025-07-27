# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import torch
from curobo.geom.types import Cuboid

from cutamp.envs.utils import unit_quat, TAMPEnvironment
from cutamp.tamp_domain import ButtonPushed, HandEmpty
from cutamp.utils.shapes import MultiSphere


def _create_stick(length: float = 0.3, radius: float = 0.015) -> MultiSphere:
    """Create the stick tool."""
    num_spheres = length / radius
    stick_spheres = []
    for i in range(int(num_spheres)):
        stick_spheres.append([i * radius, 0.0, 0.0, radius])
    stick = MultiSphere(
        spheres=torch.tensor(stick_spheres),
        name="stick",
        color=[255, 209, 220],
    )
    return stick


def load_stick_button_env(include_distractors: bool = True) -> TAMPEnvironment:
    """Stick Button environment."""
    # Big table to make things pretty (not a placement surface)
    big_table = Cuboid(
        name="big_table", dims=[1.4, 1.3, 0.02], pose=[0.5, 0.0, -0.0111, *unit_quat], color=[202, 164, 114]
    )
    # The table is the active workspace
    table = Cuboid(name="table", dims=[0.75, 1.15, 0.02], pose=[0.6, 0.0, -0.011, *unit_quat], color=[235, 196, 145])
    exterior_wall = Cuboid(
        name="exterior_wall", dims=[0.025, 1.0, 0.15], pose=[0.925, 0.0, 0.075, *unit_quat], color=[200, 200, 200]
    )
    obstacle_wall = Cuboid(
        name="obstacle_wall", dims=[0.1, 0.3, 0.15], pose=[0.65, -0.35, 0.075, *unit_quat], color=[200, 200, 200]
    )

    # Create stick, which is collection of spheres. Manually choose a reasonable pose.
    stick_sph_radius = 0.015
    stick = _create_stick(radius=stick_sph_radius)
    stick.pose = [0.35, -0.2, stick_sph_radius + 0.005, 0.951, 0.0, 0.0, 0.309]

    # Create button
    btn_red = Cuboid(name="btn_red", dims=[0.05, 0.05, 0.025], pose=[0.4, 0.2, 0.0125, *unit_quat], color=[255, 0, 0])
    btn_blue = Cuboid(
        name="btn_blue", dims=[0.05, 0.05, 0.025], pose=[0.85, -0.35, 0.0125, *unit_quat], color=[0, 0, 255]
    )
    btn_green = Cuboid(
        name="btn_green", dims=[0.05, 0.05, 0.025], pose=[0.85, 0.4, 0.0125, *unit_quat], color=[0, 255, 0]
    )
    buttons = [btn_red, btn_blue, btn_green]
    if include_distractors:
        distractor_buttons = [
            Cuboid(
                name="btn_white_1",
                dims=[0.05, 0.05, 0.025],
                pose=[0.85, 0.0, 0.0125, *unit_quat],
                color=[255, 255, 255],
            ),
            Cuboid(
                name="btn_white_2",
                dims=[0.05, 0.05, 0.025],
                pose=[0.65, 0.1, 0.0125, *unit_quat],
                color=[255, 255, 255],
            ),
        ]
        buttons.extend(distractor_buttons)

    # Goal is for the red, blue and green buttons to be pressed
    goal_buttons = [btn_red, btn_blue, btn_green]
    goal_state = frozenset({HandEmpty.ground(), *[ButtonPushed.ground(button.name) for button in goal_buttons]})

    # Create environment
    env = TAMPEnvironment(
        name="stick_button",
        movables=[stick],
        statics=[big_table, table, exterior_wall, obstacle_wall] + buttons,
        type_to_objects={
            "Movable": [stick],
            "Stick": [stick],
            "Surface": [table],
            "Button": buttons,
        },
        goal_state=goal_state,
    )
    return env
