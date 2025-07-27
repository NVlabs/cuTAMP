# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import itertools
import warnings
from collections import defaultdict
from typing import Dict, Union

import roma
import torch
from curobo.rollout.cost.self_collision_cost import SelfCollisionCost, SelfCollisionCostConfig
from jaxtyping import Float

from cutamp.config import TAMPConfiguration
from cutamp.costs import curobo_pose_error, dist_from_bounds_jit, sphere_to_sphere_overlap, trajectory_length
from cutamp.rollout import Rollout
from cutamp.tamp_world import TAMPWorld
from cutamp.task_planning import PlanSkeleton
from cutamp.task_planning.constraints import (
    Collision,
    CollisionFree,
    CollisionFreeGrasp,
    CollisionFreeHolding,
    CollisionFreePlacement,
    KinematicConstraint,
    Motion,
    StablePlacement,
    ValidPush,
    ValidPushStick,
)
from cutamp.task_planning.costs import GraspCost, TrajectoryLength
from cutamp.utils.common import transform_spheres


class CostFunction:
    """
    Cost Function for a given plan skeleton. Given a rollout, we compute the constraints and costs.
    The __init__ function caches constraints and costs for quicker indexing during rollout evaluation.
    """

    def __init__(self, plan_skeleton: PlanSkeleton, world: TAMPWorld, config: TAMPConfiguration):
        if config.enable_traj:
            raise NotImplementedError("Trajectories not supported in cost function yet")

        self.plan_skeleton = plan_skeleton
        self.world = world
        self.config = config
        self._rollout_validated = False

        # Accumulate the constraints, so we can batch them up when computing the costs
        self.cfree_constraints = []
        self.kinematic_constraints = []
        self.motion_constraints = []
        self.stable_placement_constraints = []
        self.valid_push_constraints = []
        self.valid_push_stick_constraints = []
        self.traj_length_costs = []

        type_to_list = {
            KinematicConstraint.type: self.kinematic_constraints,
            Motion.type: self.motion_constraints,
            CollisionFree.type: self.cfree_constraints,
            CollisionFreeHolding.type: self.cfree_constraints,
            CollisionFreeGrasp.type: self.cfree_constraints,
            CollisionFreePlacement.type: self.cfree_constraints,
            StablePlacement.type: self.stable_placement_constraints,
            ValidPush.type: self.valid_push_constraints,
            ValidPushStick.type: self.valid_push_stick_constraints,
            TrajectoryLength.type: self.traj_length_costs,
        }
        warning_co = {GraspCost.type}
        for ground_op in plan_skeleton:
            for co in [*ground_op.constraints, *ground_op.costs]:
                if co.type not in type_to_list:
                    if co.type in warning_co:
                        warnings.warn(f"Cost {co} is not handled in the cost function")
                    else:
                        raise NotImplementedError(f"Unhandled constraint or cost: {co}")
                else:
                    type_to_list[co.type].append(co)

        # All conf parameters for motion constraints, we check rollout is subset
        self.motion_conf_params = set(
            iter(itertools.chain.from_iterable(con.params for con in self.motion_constraints))
        )

        # Setup self-collision cost
        self_collision_config = SelfCollisionCostConfig(
            self.world.tensor_args.to_device([1.0]),
            self.world.tensor_args,
            return_loss=True,
            self_collision_kin_config=self.world.kin_model.get_self_collision_config(),
        )
        self.self_collision_cost_fn = SelfCollisionCost(self_collision_config)
        # Should be using experimental kernel by default
        if not self.self_collision_cost_fn.self_collision_kin_config.experimental_kernel:
            raise ValueError("Expected self-collision cost to use experimental kernel")

        # Conf parameters for kinematic constraints, order in rollout should match
        self.kinematic_confs, self.kinematic_actions = zip(*(con.params for con in self.kinematic_constraints))
        self.kinematic_confs = list(self.kinematic_confs)
        self.kinematic_actions = list(self.kinematic_actions)

        # Compute the AABB and surface z-position for the placement surfaces
        self.surface_to_aabb = {}
        self.surface_to_target_z = {}
        self.surface_to_objs = defaultdict(list)
        for con in self.stable_placement_constraints:
            obj, _, _, surface = con.params
            self.surface_to_objs[surface].append(obj)
            if surface in self.surface_to_aabb:
                continue

            aabb = world.get_aabb(surface)  # includes xyz
            aabb_xy = aabb[:, :2]
            self.surface_to_aabb[surface] = aabb_xy

            # For the target z, need to take collision activation distance into account
            surface_z = aabb[1, 2]
            target_z = surface_z + world.collision_activation_distance + 2e-3  # add some buffer
            self.surface_to_target_z[surface] = target_z

        # Store the button AABBs for ValidPush
        self.button_to_action = {}
        self.button_aabbs = []
        for con in self.valid_push_constraints:
            button, action = con.params
            if not self.world.has_object(button):
                raise ValueError(f"{button=} not found in world")
            if button in self.button_to_action:
                raise NotImplementedError(f"We only support pushing a button once right now")

            # Set z to be 2cm buffer above surface of the button
            aabb = self.world.get_aabb(button).clone()
            aabb[0, 2] = aabb[1, 2] + world.collision_activation_distance + 2e-3
            aabb[1, 2] = aabb[1, 2] + world.collision_activation_distance + 0.02
            self.button_aabbs.append(aabb)
            self.button_to_action[button] = action
        self.button_aabbs = torch.stack(self.button_aabbs) if self.button_aabbs else None

        # Store the buttons and sticks for ValidPushStick
        self.button_stick_actions = {}
        self.button_stick_aabbs = []
        self.button_to_stick = {}
        for con in self.valid_push_stick_constraints:
            button, stick, action = con.params
            if not self.world.has_object(button):
                raise ValueError(f"{button=} not found in world")
            if not self.world.has_object(stick):
                raise ValueError(f"{stick=} not found in world")
            if button in self.button_to_stick:
                raise NotImplementedError(f"We only support pushing a button once right now")

            # Set z to be 2cm buffer above surface of the button
            button_aabb = self.world.get_aabb(button).clone()
            button_aabb[0, 2] = button_aabb[1, 2] + world.collision_activation_distance + 2e-3
            button_aabb[1, 2] = button_aabb[1, 2] + world.collision_activation_distance + 0.02
            self.button_stick_aabbs.append(button_aabb)

            self.button_to_stick[button] = stick
            self.button_stick_actions[button] = action
        self.button_stick_aabbs = torch.stack(self.button_stick_aabbs) if self.button_stick_aabbs else None

        # All conf parameters for trajectory length costs, we check rollout matches
        # No support for trajectories right now
        self.traj_length_confs = []
        for cost in self.traj_length_costs:
            q_start, traj, q_end = cost.params
            self.traj_length_confs.append(q_start)
            self.traj_length_confs.append(q_end)
        self.traj_length_confs = list(dict.fromkeys(self.traj_length_confs))  # remove duplicates
        if self.traj_length_confs[0] != "q0":
            raise ValueError("Expected q0 to be the first conf")
        self.traj_length_confs = self.traj_length_confs[1:]

    def _validate_rollout(self, rollout: Rollout):
        """Checks structure of the rollout conforms to the assumptions we make in the cost function implementation."""
        if self._rollout_validated:
            return

        # Configurations should be subset of parameters involved in motion constraints
        if not set(rollout["conf_params"]).issubset(self.motion_conf_params):
            raise RuntimeError(
                f"Missing conf params in motion constraints: {rollout['conf_params'] - self.motion_conf_params}"
            )

        # Kinematic configuration and actions (i.e., poses) should match
        if self.kinematic_confs != rollout["conf_params"]:
            raise RuntimeError(f"Expected conf params {self.kinematic_confs} but got {rollout['conf_params']}")
        if self.kinematic_actions != rollout["action_params"]:
            raise RuntimeError(f"Expected action params {self.kinematic_actions} but got {rollout['action_params']}")

        # Trajectory length parameters should match
        if self.traj_length_confs != rollout["conf_params"]:
            raise RuntimeError(f"Expected conf params {self.traj_length_confs} but got {rollout['conf_params']}")

        self._rollout_validated = True

    def kinematic_costs(self, rollout: Rollout) -> Union[dict, None]:
        """Kinematic constraints - i.e., pose error between actual and desired end-effector poses."""
        pos_errs, rot_errs = curobo_pose_error(rollout["world_from_ee"], rollout["world_from_ee_desired"])
        kinematic_cost = {
            "type": "constraint",
            "constraints": self.kinematic_constraints,
            "values": {"pos_err": pos_errs, "rot_err": rot_errs},
        }
        return kinematic_cost

    def motion_costs(self, rollout: Rollout) -> Union[dict, None]:
        """Motion constraints - valid motions don't exceed joint limits or self-collide."""
        # Joint limits
        confs = rollout["confs"]
        dist_from_joint_lims = dist_from_bounds_jit(
            confs, self.world.robot_container.joint_limits[0], self.world.robot_container.joint_limits[1]
        )

        # Self collisions
        robot_spheres = rollout["robot_spheres"]
        self_coll_vals = self.self_collision_cost_fn(robot_spheres)

        motion_cost = {
            "type": "constraint",
            "constraints": self.motion_constraints,
            "values": {"joint_limit": dist_from_joint_lims, "self_collision": self_coll_vals},
        }
        return motion_cost

    def valid_push_costs(
        self, rollout: Rollout, obj_to_spheres: Dict[str, Float[torch.Tensor, "b t n 4"]]
    ) -> Union[dict, None]:
        """Valid Push constraints - i.e., distance from the button for push actions."""
        if not (self.valid_push_constraints or self.valid_push_stick_constraints):
            return None

        valid_push_cost = {"type": "constraint", "constraints": [], "values": {}}

        if self.valid_push_constraints:
            ts_idxs = []  # get timestep for the button push actions
            for button, action in self.button_to_action.items():
                ts = rollout["action_to_ts"][action]
                ts_idxs.append(ts)
            tool_poses = rollout["world_from_tool_desired"][:, ts_idxs]
            tool_xyz = tool_poses[:, :, :3, 3]
            dist_from_button = dist_from_bounds_jit(tool_xyz, self.button_aabbs[:, 0], self.button_aabbs[:, 1])
            valid_push_cost["constraints"].extend(self.valid_push_constraints)
            valid_push_cost["values"]["dist_from_button"] = dist_from_button

        if self.valid_push_stick_constraints:
            # Get the stick spheres for each button push action
            all_stick_spheres = []
            for button, action in self.button_stick_actions.items():
                pose_ts = rollout["action_to_pose_ts"][action]
                stick_name = self.button_to_stick[button]
                stick_spheres = obj_to_spheres[stick_name][:, pose_ts]
                all_stick_spheres.append(stick_spheres)
            all_stick_spheres = torch.stack(all_stick_spheres, dim=1)

            # We'll consider bottom of spheres, so subtract the radius
            stick_xyz = all_stick_spheres[..., :3].clone()  # important! clone so we don't modify original tensor
            stick_xyz[..., 2] -= all_stick_spheres[..., 3]

            # Compute distance between all stick spheres and button AABB, take the minimum
            push_stick_cost = dist_from_bounds_jit(
                stick_xyz, self.button_stick_aabbs[:, None, 0], self.button_stick_aabbs[:, None, 1]
            )
            push_stick_cost = push_stick_cost.min(-1).values

            valid_push_cost["constraints"].extend(self.valid_push_stick_constraints)
            valid_push_cost["values"]["stick_dist_from_button"] = push_stick_cost

        return valid_push_cost

    def stable_placement_costs(
        self, rollout: Rollout, obj_to_spheres: Dict[str, Float[torch.Tensor, "b t n 4"]]
    ) -> Union[dict, None]:
        """
        Stable Placement constraints. Converted to two costs:
            1. Object sphere xy positions within surface AABB
            2. Minimum object sphere is supported by the surface (compute distance between surface and bottom of sphere)
        """
        if not self.stable_placement_constraints:
            return None

        # First collate the objects by placement surface.
        surface_to_obj = defaultdict(list)
        surface_to_spheres = defaultdict(list)
        for con in self.stable_placement_constraints:
            obj, _, placement, surface = con.params
            pose_ts = rollout["action_to_pose_ts"][placement]
            obj_spheres = obj_to_spheres[obj][:, pose_ts]
            surface_to_obj[surface].append(obj)
            surface_to_spheres[surface].append(obj_spheres)

        num_particles = rollout["num_particles"]
        support_vals = {}
        for surface, objs in surface_to_obj.items():
            # Create map of sphere index to object index
            sphere_idx_map = []
            for o_idx, (obj, spheres) in enumerate(zip(objs, surface_to_spheres[surface])):
                num_spheres = spheres.shape[1]
                sph_idxs = [o_idx] * num_spheres
                sphere_idx_map.extend(sph_idxs)
            sphere_idx_map = torch.tensor(sphere_idx_map, dtype=torch.int64, device=self.world.device)
            sphere_idx_map_expand = sphere_idx_map[None].expand(num_particles, -1)  # expand by batch size

            # Since objects can have different numbers of spheres, we need to concatenate instead of stack
            spheres = torch.cat(surface_to_spheres[surface], dim=1)
            spheres_xy = spheres[..., :2]

            # Within goal xy bounds, need to gather by the spheres for each object
            in_goal_xy = dist_from_bounds_jit(spheres_xy, *self.surface_to_aabb[surface])
            obj_in_goal_xy = torch.zeros((num_particles, len(objs)), dtype=in_goal_xy.dtype, device=in_goal_xy.device)
            obj_in_goal_xy.scatter_add_(1, sphere_idx_map_expand, in_goal_xy)
            support_vals[f"{surface}_in_xy"] = obj_in_goal_xy

            # Distance between bottom of spheres and z-position of the surface
            spheres_bottom = spheres[..., 2] - spheres[..., 3]
            obj_bottom = torch.full(
                (num_particles, len(objs)), float("inf"), dtype=spheres_bottom.dtype, device=spheres_bottom.device
            )
            obj_bottom.scatter_reduce_(1, sphere_idx_map_expand, spheres_bottom, reduce="amin")
            target_z = self.surface_to_target_z[surface]
            support_vals[f"{surface}_support"] = torch.abs(obj_bottom - target_z)

        stable_placement_cost = {
            "type": "constraint",
            "constraints": self.stable_placement_constraints,
            "values": support_vals,
        }
        return stable_placement_cost

    def trajectory_costs(self, rollout: Rollout) -> dict:
        """Trajectory costs, just joint space distance between configurations for now."""
        traj_cost = {
            "type": "cost",
            "costs": self.traj_length_costs,
            "values": {"traj_length": trajectory_length(rollout["confs"])},
        }
        return traj_cost

    def collision_costs(self, rollout: Rollout, obj_to_spheres: Dict[str, Float[torch.Tensor, "b t n 4"]]) -> dict:
        """Collision costs. This could be tied better to the constraints and sped up significantly."""
        # Robot to world
        robot_spheres = rollout["robot_spheres"]
        coll_values = {"robot_to_world": self.world.collision_fn(robot_spheres)}

        # Collision between movables and world
        all_movable_spheres = torch.cat(list(obj_to_spheres.values()), dim=2)
        coll_values["movable_to_world"] = self.world.collision_fn(all_movable_spheres)

        # Collision between robot and movables, need to expand poses to full timesteps
        all_pose_ts = list(rollout["ts_to_pose_ts"].values())
        coll_values["robot_to_movables"] = sphere_to_sphere_overlap(
            robot_spheres,
            all_movable_spheres[:, all_pose_ts],
            activation_distance=self.config.gripper_activation_distance,
        )

        # TODO: this is slow and a bottleneck, could consider using curobo's fast sphere-to-sphere kernel
        # Collision between movable objects
        for obj_1, obj_2 in itertools.combinations(self.world.movables, 2):
            obj_1_spheres = obj_to_spheres[obj_1.name]
            obj_2_spheres = obj_to_spheres[obj_2.name]
            coll_values[f"{obj_1.name}_to_{obj_2.name}"] = sphere_to_sphere_overlap(
                obj_1_spheres,
                obj_2_spheres,
                activation_distance=self.config.movable_activation_distance,
                use_aabb_check=True,
            )

        coll_cost = {
            "type": "constraint",
            "constraints": self.cfree_constraints,
            "values": coll_values,
        }
        return coll_cost

    def soft_costs(self, rollout: Rollout) -> dict:
        """Soft costs defined on the goal state."""
        # last object pose
        last_obj_position = [v[:, -1, :3, 3] for v in rollout["obj_to_pose"].values()]
        last_obj_position = torch.stack(last_obj_position, dim=1)

        if self.config.soft_cost == "dist_from_origin":
            dist_from_origin = last_obj_position.norm(dim=-1)
            dist_from_origin = -dist_from_origin.sum(dim=-1)
            values = {"dist_from_origin": dist_from_origin}
        elif self.config.soft_cost == "max_obj_dist" or self.config.soft_cost == "min_obj_dist":
            all_obj_dists = torch.cdist(last_obj_position, last_obj_position, p=2)  # (b, n, n)
            mask = torch.triu(torch.ones_like(all_obj_dists), diagonal=1) == 1
            obj_dists = all_obj_dists[mask].view(mask.shape[0], -1)  # reshape into num pairs
            dists_sum = obj_dists.sum(-1)
            if self.config.soft_cost == "max_obj_dist":
                values = {"max_obj_dist": -dists_sum}
            else:
                values = {"min_obj_dist": dists_sum}
        elif self.config.soft_cost == "min_y" or self.config.soft_cost == "max_y":
            last_obj_y = last_obj_position[..., 1]
            last_y = last_obj_y.sum(dim=-1)
            if self.config.soft_cost == "min_y":
                values = {"min_y": last_y}
            else:
                values = {"max_y": -last_y}
        elif self.config.soft_cost == "align_yaw":
            last_obj_mat3x3 = [v[:, -1, :3, :3] for v in rollout["obj_to_pose"].values()]
            last_obj_mat3x3 = torch.stack(last_obj_mat3x3, dim=1)
            last_obj_rpy = roma.rotmat_to_euler("XYZ", last_obj_mat3x3)
            last_obj_yaw = last_obj_rpy[..., 2]

            # Compute pairwise yaw differences and normalize to be between -pi and pi
            yaw_diffs = last_obj_yaw[:, :, None] - last_obj_yaw[:, None, :]
            yaw_diffs = torch.atan2(torch.sin(yaw_diffs), torch.cos(yaw_diffs)).abs()
            mask = torch.triu(torch.ones_like(yaw_diffs), diagonal=1) == 1
            yaw_diffs = yaw_diffs[mask].view(mask.shape[0], -1)  # reshape into num pairs
            yaw_diffs = yaw_diffs.sum(-1)
            values = {"align_yaw": yaw_diffs}
        else:
            raise ValueError(f"Unsupported soft cost: {self.config.soft_cost}")

        return {"type": "cost", "constraints": [], "values": values}

    def __call__(self, rollout: Rollout) -> Dict[str, dict]:
        self._validate_rollout(rollout)
        cost_dict = {}

        def add_cost(k_, v_):
            if v_ is not None:
                cost_dict[k_] = v_

        # Trajectory cost
        traj_cost = self.trajectory_costs(rollout)
        add_cost(TrajectoryLength.type, traj_cost)

        # Get collision spheres for movable objects
        obj_to_spheres = {}
        for idx, obj in enumerate(self.world.movables):
            if obj.name in obj_to_spheres:
                raise RuntimeError(f"Object {obj.name} already in obj_to_spheres")
            obj_pose = rollout["obj_to_pose"][obj.name]
            obj_spheres = transform_spheres(self.world.get_collision_spheres(obj), obj_pose)
            obj_to_spheres[obj.name] = obj_spheres

        # Collision costs
        collision_cost = self.collision_costs(rollout, obj_to_spheres)
        add_cost(Collision.type, collision_cost)

        # Valid Push constraints
        valid_push_cost = self.valid_push_costs(rollout, obj_to_spheres)
        add_cost(ValidPush.type, valid_push_cost)

        # Stable placement cost
        stable_placement_cost = self.stable_placement_costs(rollout, obj_to_spheres)
        add_cost(StablePlacement.type, stable_placement_cost)

        # Valid motions don't exceed joint limits
        motion_cost = self.motion_costs(rollout)
        add_cost(Motion.type, motion_cost)

        # Kinematic costs
        kinematic_cost = self.kinematic_costs(rollout)
        add_cost(KinematicConstraint.type, kinematic_cost)

        # Soft costs
        if self.config.soft_cost is not None:
            soft_cost = self.soft_costs(rollout)
            add_cost("soft", soft_cost)

        return cost_dict
