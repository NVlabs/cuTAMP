# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import logging
from typing import Optional

import roma
import torch
from curobo.geom.types import Cuboid
from curobo.types.math import Pose

from cutamp.config import TAMPConfiguration
from cutamp.costs import sphere_to_sphere_overlap
from cutamp.samplers import (
    grasp_4dof_sampler,
    grasp_6dof_sampler,
    place_4dof_sampler,
    sample_yaw,
)
from cutamp.tamp_domain import MoveFree, MoveHolding, Pick, Place, Push, PushStick
from cutamp.tamp_world import TAMPWorld
from cutamp.task_planning import PlanSkeleton
from cutamp.utils.common import (
    Particles,
    action_4dof_to_mat4x4,
    action_6dof_to_mat4x4,
    pose_list_to_mat4x4,
    sample_between_bounds,
    transform_spheres,
)
from cutamp.utils.shapes import MultiSphere

_log = logging.getLogger(__name__)


class ParticleInitializer:
    def __init__(self, world: TAMPWorld, config: TAMPConfiguration):
        if config.enable_traj:
            raise NotImplementedError("Trajectory initialization not yet supported")
        if config.place_dof != 4:
            raise NotImplementedError(f"Only 4-DOF grasp and placement supported for now, not {config.place_dof}")
        if config.grasp_dof != 4 and config.grasp_dof != 6:
            raise NotImplementedError(f"Only 4-DOF or 6-DOF grasp supported for now, not {config.grasp_dof}")
        self.world = world
        self.config = config
        self.q_init = world.q_init.repeat(config.num_particles, 1)

        # Sampler caching
        self.pick_cache = {}
        self.place_cache = {}
        self.push_button_cache = {}
        self.push_stick_cache = {}
        self.failed_push = set()

    def __call__(self, plan_skeleton: PlanSkeleton, verbose: bool = True) -> Optional[Particles]:
        config = self.config
        num_particles = self.config.num_particles
        world = self.world
        particles = {"q0": self.q_init.clone()}
        deferred_params = set()
        log_debug = _log.debug if verbose else lambda *args, **kwargs: None

        # Note: we don't consider state after executing earlier samples
        # Iterate through each ground operator in the plan skeleton and initialize and build up particles
        for idx, ground_op in enumerate(plan_skeleton):
            op_name = ground_op.operator.name
            params = ground_op.values
            header = f"{idx + 1}. {ground_op}"

            # MoveFree
            if op_name == MoveFree.name:
                q_start, _traj, q_end = params
                if q_start not in particles:
                    raise ValueError(f"{q_start=} should already be bound")
                deferred_params.add(q_end)
                log_debug(f"{header}. Deferred {q_end}")

            # MoveHolding
            elif op_name == MoveHolding.name:
                obj, grasp, q_start, _traj, q_end = params
                if not world.has_object(obj):
                    raise ValueError(f"{obj=} not found in world")
                if grasp not in particles:
                    raise ValueError(f"{grasp=} should already be bound")
                if q_start not in particles:
                    raise ValueError(f"{q_start=} should already be bound")
                deferred_params.add(q_end)
                log_debug(f"{header}. Deferred {q_end}")

            # Pick
            elif op_name == Pick.name:
                obj, grasp, q = params
                if not world.has_object(obj):
                    raise ValueError(f"{obj=} not found in world")
                if grasp in particles:
                    raise ValueError(f"{grasp=} shouldn't already be bound")
                if q in particles:
                    raise ValueError(f"{q=} shouldn't already be bound")

                # Note: pick cache currently assumes object is at same pose as when sampled
                if obj in self.pick_cache:
                    # important, we need to clone here
                    particles[grasp] = self.pick_cache[obj]["sampled_grasps"].clone()
                    ik_result = self.pick_cache[obj]["ik_result"]
                    particles[q] = ik_result.solution[:, 0].clone()
                    deferred_params.remove(q)
                    log_debug(
                        f"{header}. Using cached grasp poses for {obj}. {ik_result.success.sum()}/{num_particles} success"
                    )
                    continue

                # Sample grasps
                obj_curobo = world.get_object(obj)
                obj_spheres = world.get_collision_spheres(obj)
                num_faces = 4 if isinstance(obj_curobo, Cuboid) else None

                # Sample 4 times as many grasps as particles
                if config.grasp_dof == 4:
                    sampled_grasps = grasp_4dof_sampler(num_particles * 4, obj_curobo, obj_spheres, num_faces=num_faces)
                    obj_from_grasp = action_4dof_to_mat4x4(sampled_grasps)
                else:
                    sampled_grasps = grasp_6dof_sampler(num_particles * 4, obj_curobo, num_faces=num_faces)
                    obj_from_grasp = action_6dof_to_mat4x4(sampled_grasps)

                # Select the grasps that are not in collision with the object
                grasp_spheres = transform_spheres(world.robot_container.gripper_spheres, obj_from_grasp)
                grasp_coll = sphere_to_sphere_overlap(obj_spheres, grasp_spheres)
                good_idxs = grasp_coll.topk(num_particles, largest=False).indices
                particles[grasp] = sampled_grasps[good_idxs]

                # Transform grasps to hand frame
                if config.random_init:
                    q_sample = sample_between_bounds(num_particles, world.robot_container.joint_limits)
                    particles[q] = q_sample
                else:
                    obj_from_grasp = obj_from_grasp[good_idxs]
                    world_from_obj = pose_list_to_mat4x4(obj_curobo.pose).to(world.tensor_args.device)
                    world_from_grasp = world_from_obj @ obj_from_grasp
                    world_from_ee = world_from_grasp @ world.tool_from_ee

                    # Solve IK with cuRobo
                    world_from_ee = Pose.from_matrix(world_from_ee)
                    ik_result = world.ik_solver.solve_batch(world_from_ee, seed_config=None)  # TODO: seeding
                    log_debug(
                        f"{header}. IK success: {ik_result.success.sum()}/{num_particles}, took {ik_result.solve_time:.2f}s"
                    )
                    particles[q] = ik_result.solution[:, 0]
                deferred_params.remove(q)

                # Store in cache
                if config.cache_subgraphs:
                    self.pick_cache[obj] = {"sampled_grasps": particles[grasp], "ik_result": ik_result}

            # Place
            elif op_name == Place.name:
                obj, grasp, placement, surface, q = params
                if not world.has_object(obj):
                    raise ValueError(f"{obj=} not found in world")
                if grasp not in particles:
                    raise ValueError(f"{grasp=} should already be bound")
                if placement in particles:
                    raise ValueError(f"{placement=} shouldn't already be bound")
                if not world.has_object(surface):
                    raise ValueError(f"{surface=} not found in world")
                if q in particles:
                    raise ValueError(f"{q=} shouldn't already be bound")

                if (obj, surface) in self.place_cache:
                    # need to make sure the grasps match what is cached
                    actual_grasp = particles[grasp]
                    cached_grasp = self.place_cache[(obj, surface)]["grasp"]
                    if not (actual_grasp == cached_grasp).all():
                        raise RuntimeError(f"Grasps don't match for {obj} on {surface}")

                    # important, we need to clone here
                    sampled_placements = self.place_cache[(obj, surface)]["sampled_placements"].clone()
                    particles[placement] = sampled_placements
                    ik_result = self.place_cache[(obj, surface)]["ik_result"]
                    particles[q] = ik_result.solution[:, 0].clone()
                    deferred_params.remove(q)
                    log_debug(
                        f"{header}. Using cached placement poses for {obj}. {ik_result.success.sum()}/{num_particles} success"
                    )
                    continue

                # Sample placements pose of object (in world frame)
                obj_curobo = world.get_object(obj)
                obj_spheres = world.get_collision_spheres(obj)
                if config.random_init:
                    yaw = sample_yaw(num_particles * 4, None, self.world.tensor_args.device)
                    aabb = world.world_aabb.clone()
                    aabb[0, 2] = 0.0
                    aabb[1, 2] = max(aabb[1, 2], 0.2)
                    xyz = sample_between_bounds(num_particles * 4, aabb)
                    sampled_placements = torch.cat([xyz, yaw.unsqueeze(-1)], dim=1)
                else:
                    surface_curobo = world.get_object(surface)
                    sampled_placements = place_4dof_sampler(num_particles * 4, obj_curobo, obj_spheres, surface_curobo)

                # Select the placements that are not in collision with the object
                world_from_obj = action_4dof_to_mat4x4(sampled_placements)  # desired placement pose
                obj_place_spheres = transform_spheres(obj_spheres, world_from_obj)
                place_coll = world.collision_fn(obj_place_spheres[:, None].contiguous())[:, 0]
                best_idxs = place_coll.topk(num_particles, largest=False).indices
                sampled_placements = sampled_placements[best_idxs]
                world_from_obj = world_from_obj[best_idxs]

                # Set particles and then solve for robot configurations
                particles[placement] = sampled_placements
                if config.random_init:
                    q_sample = sample_between_bounds(num_particles, world.robot_container.joint_limits)
                    particles[q] = q_sample
                else:
                    # Get the hand pose given the placement pose in world frame.
                    # Need to take grasp into account to transform into hand frame.
                    if config.grasp_dof == 4:
                        obj_from_grasp = action_4dof_to_mat4x4(particles[grasp])
                    else:
                        obj_from_grasp = action_6dof_to_mat4x4(particles[grasp])
                    world_from_grasp = world_from_obj @ obj_from_grasp
                    world_from_ee = world_from_grasp @ world.tool_from_ee

                    # Solve IK
                    world_from_ee = Pose.from_matrix(world_from_ee)
                    ik_result = world.ik_solver.solve_batch(world_from_ee, seed_config=None)  # TODO: seeding?
                    log_debug(
                        f"{header}. IK success: {ik_result.success.sum()}/{num_particles}, took {ik_result.solve_time:.2f}s"
                    )
                    particles[q] = ik_result.solution[:, 0]
                deferred_params.remove(q)

                # Store in cache
                if config.cache_subgraphs:
                    self.place_cache[(obj, surface)] = {
                        "sampled_placements": sampled_placements,
                        "ik_result": ik_result,
                        "grasp": particles[grasp],
                    }

            # Push Button (without stick)
            elif op_name == Push.name:
                button, push_pose, q = params
                assert not config.random_init, "Random initialization not supported for pushing"
                if not world.has_object(button):
                    raise ValueError(f"{button=} not found in world")
                if push_pose in particles:
                    raise ValueError(f"{push_pose=} shouldn't already be bound")
                if q in particles:
                    raise ValueError(f"{q=} shouldn't already be bound")

                # Pruning failed subgraphs (i.e., we couldn't push this button at all)
                if button in self.failed_push and config.skip_failed_subgraphs:
                    return None

                if button in self.push_button_cache:
                    # important, we need to clone here
                    sampled_push = self.push_button_cache[button]["sampled_push"].clone()
                    particles[push_pose] = sampled_push
                    ik_result = self.push_button_cache[button]["ik_result"]
                    particles[q] = ik_result.solution[:, 0].clone()
                    deferred_params.remove(q)
                    log_debug(
                        f"{header}. Using cached push poses for {button}. {ik_result.success.sum()}/{num_particles} success"
                    )
                    continue

                # Sample 4-DOF push poses for the button
                aabb = world.get_aabb(button).clone()
                surface_z = aabb[1, 2]  # top of the button aabb
                # add 1cm buffer
                lower_xy, upper_xy = aabb[:, :2]
                lower_xy += 0.01
                upper_xy -= 0.01

                sampled_xy = lower_xy + torch.rand(num_particles, 2, device=world.tensor_args.device) * (
                    upper_xy - lower_xy
                )
                sampled_z = (
                    surface_z.expand(num_particles) + 0.02 + world.collision_activation_distance
                )  # 2cm above button for now
                sampled_yaw = sample_yaw(num_particles, num_faces=None, device=world.tensor_args.device)
                sampled_push = torch.cat([sampled_xy, sampled_z[:, None], sampled_yaw[:, None]], dim=1)
                particles[push_pose] = sampled_push

                # Transform from tool to hand frame
                world_from_push = action_4dof_to_mat4x4(sampled_push)
                world_from_ee = world_from_push @ world.tool_from_ee

                # Solve IK with cuRobo
                world_from_ee = Pose.from_matrix(world_from_ee)
                ik_result = world.ik_solver.solve_batch(world_from_ee, seed_config=None)  # TODO: seeding
                log_debug(
                    f"{header}. IK success: {ik_result.success.sum()}/{num_particles}, took {ik_result.solve_time:.2f}s"
                )
                particles[q] = ik_result.solution[:, 0]
                deferred_params.remove(q)

                # Failed subgraph!
                if not ik_result.success.any():
                    self.failed_push.add(button)

                # Cache the push poses
                if config.cache_subgraphs:
                    self.push_button_cache[button] = {"sampled_push": sampled_push, "ik_result": ik_result}

            # Push Button with Stick
            elif op_name == PushStick.name:
                button, stick_name, grasp, push_pose, q = params
                assert not config.random_init, "Random initialization not supported for pushing"
                if not world.has_object(button):
                    raise ValueError(f"{button=} not found in world")
                if not world.has_object(stick_name):
                    raise ValueError(f"{stick_name=} not found in world")
                if grasp not in particles:
                    raise ValueError(f"{grasp=} should already be binded")
                if push_pose in particles:
                    raise ValueError(f"{push_pose=} shouldn't already be binded")
                if q in particles:
                    raise ValueError(f"{q=} shouldn't already be binded")

                if (button, stick_name) in self.push_stick_cache:
                    # need to make sure the grasps match what is cached
                    actual_grasp = particles[grasp]
                    cached_grasp = self.push_stick_cache[(button, stick_name)]["grasp"]
                    if not (actual_grasp == cached_grasp).all():
                        raise RuntimeError(f"Grasps don't match for {button} with {stick_name}")

                    # important, we need to clone here
                    sampled_push = self.push_stick_cache[(button, stick_name)]["sampled_push"].clone()
                    particles[push_pose] = sampled_push
                    ik_result = self.push_stick_cache[(button, stick_name)]["ik_result"]
                    particles[q] = ik_result.solution[:, 0].clone()
                    deferred_params.remove(q)

                    log_debug(
                        f"{header}. Using cached push for {button} with {stick_name}. "
                        f"{ik_result.success.sum()}/{num_particles} success"
                    )
                    continue

                # Sample pushes for the button, this will be in the stick frame
                aabb = world.get_aabb(button).clone()
                surface_z = aabb[1, 2]  # top of the button aabb
                # add 1cm buffer
                lower_xy, upper_xy = aabb[:, :2]
                lower_xy += 0.01
                upper_xy -= 0.01

                sampled_xy = lower_xy + torch.rand(num_particles, 2, device=world.tensor_args.device) * (
                    upper_xy - lower_xy
                )
                sampled_z = (
                    surface_z.expand(num_particles) + 0.02 + world.collision_activation_distance
                )  # 2cm above button for now
                sampled_yaw = sample_yaw(num_particles, num_faces=None, device=world.tensor_args.device)
                sampled_push = torch.cat([sampled_xy, sampled_z[:, None], sampled_yaw[:, None]], dim=1)

                # Sample somewhere along the stick for the push
                stick: MultiSphere = world.get_object("stick")
                spheres = stick.spheres
                if not (spheres[:, 1:3] == 0.0).all():
                    raise ValueError("Expected stick spheres to have y and z positions of 0")
                sphere_x = spheres[:, 0]
                x_idxs = torch.randint(0, len(sphere_x), (num_particles,), device=spheres.device)
                sampled_x = sphere_x[x_idxs]

                stick_from_tip = torch.eye(4, device=world.tensor_args.device).repeat(num_particles, 1, 1)
                stick_from_tip[:, 0, 3] = -sampled_x

                # Where we are pushing the button with the stick - i.e., the pose of the stick
                world_from_push = action_4dof_to_mat4x4(sampled_push)
                world_from_stick = world_from_push @ stick_from_tip

                # Push pose is pose of stick in world frame
                rpy = roma.rotmat_to_euler("XYZ", world_from_stick[:, :3, :3])
                assert (rpy[:, :2] == 0.0).all(), "roll and pitch should be 0"
                pos = world_from_stick[:, :3, 3]
                yaw = rpy[:, 2]
                action_4dof = torch.cat([pos, yaw[:, None]], dim=1)
                particles[push_pose] = action_4dof

                # Convert to tool frame
                if config.grasp_dof == 4:
                    obj_from_grasp = action_4dof_to_mat4x4(particles[grasp])
                else:
                    obj_from_grasp = action_6dof_to_mat4x4(particles[grasp])
                world_from_grasp = world_from_stick @ obj_from_grasp
                world_from_ee = world_from_grasp @ world.tool_from_ee

                # Solve IK with cuRobo
                world_from_ee = Pose.from_matrix(world_from_ee)
                ik_result = world.ik_solver.solve_batch(world_from_ee, seed_config=None)  # TODO: seeding
                log_debug(
                    f"{header}. IK success: {ik_result.success.sum()}/{num_particles}, took {ik_result.solve_time:.2f}s"
                )
                particles[q] = ik_result.solution[:, 0]
                deferred_params.remove(q)

                # Store in cache
                if config.cache_subgraphs:
                    self.push_stick_cache[(button, stick_name)] = {
                        "sampled_push": sampled_push,
                        "ik_result": ik_result,
                        "grasp": particles[grasp],
                    }

            # Unknown
            else:
                raise NotImplementedError(f"Unsupported operator {op_name}")

        # There should not be any deferred parameters left
        if deferred_params:
            raise RuntimeError(f"Deferred parameters not resolved: {deferred_params}")

        return particles
