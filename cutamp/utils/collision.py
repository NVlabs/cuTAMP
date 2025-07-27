# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import logging

from curobo.geom.sdf.utils import create_collision_checker
from curobo.geom.sdf.world import WorldPrimitiveCollision, WorldCollisionConfig
from curobo.geom.types import WorldConfig
from curobo.rollout.cost.primitive_collision_cost import PrimitiveCollisionCost, PrimitiveCollisionCostConfig
from curobo.types.base import TensorDeviceType


_log = logging.getLogger(__name__)


def get_collision_checker(
    world_config: WorldConfig, tensor_args: TensorDeviceType, max_distance: float | None = 0.1
) -> WorldPrimitiveCollision:
    """Create primitive cuRobo collision checker"""
    # We use cuRobo's cuboid collision checker which is a lot faster (i.e., sacrifice precision for speed)
    # This mean's we need to convert the world into oriented bounding boxes (obb)
    cuboid_world = WorldConfig.create_obb_world(world_config)
    _log.debug(f"Created cuboid world for WorldConfig with {len(world_config)} objects")
    coll_dict = {"checker_type": "PRIMITIVE"}
    if max_distance is not None:
        coll_dict["max_distance"] = max_distance

    collision_config = WorldCollisionConfig.load_from_dict(coll_dict, cuboid_world, tensor_args)
    collision_checker = create_collision_checker(collision_config)
    _log.debug(f"Created {collision_checker.__class__.__name__} collision checker")
    return collision_checker


def get_world_collision_cost(
    world_config: WorldConfig, tensor_args: TensorDeviceType, collision_activation_distance: float, weight: float = 1.0
) -> PrimitiveCollisionCost:
    """
    Get the PrimitiveCollisionCost for a given world config. The activation distance is the distance from which the SDF
    query will return a positive value. This can be used for "safer" trajectories.
    """
    if collision_activation_distance < 0.0:
        raise ValueError(f"Collision activation distance must be >= 0.0, not {collision_activation_distance}")

    world_collision_checker = get_collision_checker(world_config, tensor_args)
    collision_cost_config = PrimitiveCollisionCostConfig(
        tensor_args.to_device([weight]),
        tensor_args,
        return_loss=True,
        world_coll_checker=world_collision_checker,
        activation_distance=collision_activation_distance,
    )
    world_collision_cost = PrimitiveCollisionCost(collision_cost_config)
    _log.debug(f"Created {world_collision_cost} with activation distance {collision_activation_distance}")
    return world_collision_cost
