# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Solving motions with cuRobo."""

import logging
from typing import List

import torch
from curobo.geom.sphere_fit import SphereFitType
from curobo.geom.types import Sphere
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig

from cutamp.utils.common import Particles, action_6dof_to_mat4x4, action_4dof_to_mat4x4
from cutamp.config import TAMPConfiguration
from cutamp.optimize_plan import PlanContainer
from cutamp.tamp_domain import MoveHolding, Push, PushStick, MoveFree, Place, Pick
from cutamp.tamp_world import TAMPWorld
from cutamp.utils.timer import TorchTimer
from cutamp.utils.visualizer import Visualizer

_log = logging.getLogger(__name__)


def solve_curobo(
    plan_info: PlanContainer,
    best_particle: Particles,
    world: TAMPWorld,
    config: TAMPConfiguration,
    timer: TorchTimer,
    visualizer: Visualizer,
    timeline: str = "curobo",
):
    """
    Solve for full motion plan given a plan skeleton and optimized particles.
    Note that visualization adds non-trivial overhead.
    """
    plan_skeleton = plan_info["plan_skeleton"]
    motion_gen = world.get_motion_gen(collision_activation_distance=config.world_activation_distance)
    if config.warmup_motion_gen:
        with timer.time("curobo_motion_gen_warmup", log_callback=_log.debug):
            motion_gen.warmup()

    plan_config = MotionGenPlanConfig(
        timeout=0.5, enable_finetune_trajopt=False, time_dilation_factor=config.time_dilation_factor
    )

    # Log initial state
    ts = 0.0
    obj_to_current_pose = {obj.name: world.get_object_pose(obj) for obj in world.movables}
    visualizer.set_time_seconds(timeline, ts)
    visualizer.set_joint_positions(best_particle["q0"])
    for obj, pose in obj_to_current_pose.items():
        visualizer.log_mat4x4(f"world/{obj}", pose)

    last_js = JointState.from_position(best_particle["q0"][None].clone())
    last_q_name = "q0"

    # Fixed approach offset. This could be something we eventually optimize too
    approach_offset = torch.eye(4, device=world.device)
    approach_offset[2, 3] = -0.05

    # Accumulated plans we return that the real robot can actually execute
    accum_plans = []

    # Iterate through skeleton and motion plan
    for idx, ground_op in enumerate(plan_skeleton):
        op_name = ground_op.operator.name

        # MoveFree, defer motion planning to pick to use object pose instead of planning from q_start to q_end.
        # This works more reliably and gives higher quality motions.
        if op_name == MoveFree.name:
            q_start, traj, q_end = ground_op.values
            if traj in best_particle:
                raise NotImplementedError("Trajectories not supported yet")
            last_q_name = q_start

        # MoveHolding
        elif op_name == MoveHolding.name:
            obj, grasp, q_start, traj, q_end = ground_op.values
            if traj in best_particle:
                raise NotImplementedError("Trajectories not supported yet")
            last_q_name = q_start

        # Pick
        elif op_name == Pick.name:
            obj, grasp, q = ground_op.values
            assert last_js is not None

            with timer.time("curobo_planning"):
                start_js = last_js

                # Get the retract pose and plan to it if it's not q0
                if last_q_name != "q0":
                    world_from_ee = world.kin_model.get_state(start_js.position).ee_pose.get_matrix()[0]
                    world_from_retract = world_from_ee @ approach_offset
                    retract_result = motion_gen.plan_single(start_js, Pose.from_matrix(world_from_retract), plan_config)
                    if not retract_result.success:
                        raise RuntimeError(
                            f"Failed to plan for retract for {ground_op.name}. Status: {retract_result.status}"
                        )
                    retract_js = JointState.from_position(retract_result.get_interpolated_plan().position[-1:])
                else:
                    retract_result = None
                    retract_js = start_js

                # Get the approach pose and plan to it
                world_from_obj = obj_to_current_pose[obj]
                if config.grasp_dof == 4:
                    obj_from_grasp = action_4dof_to_mat4x4(best_particle[grasp].clone())
                else:
                    obj_from_grasp = action_6dof_to_mat4x4(best_particle[grasp].clone())
                world_from_grasp = world_from_obj @ obj_from_grasp
                world_from_ee = world_from_grasp @ world.tool_from_ee

                world_from_approach = world_from_ee @ approach_offset
                approach_result = motion_gen.plan_single(retract_js, Pose.from_matrix(world_from_approach), plan_config)
                if not approach_result.success:
                    raise RuntimeError(
                        f"Failed to plan for approach for {ground_op.name}. Status: {approach_result.status}"
                    )

                # Plan to from approach to end js
                approach_js = JointState.from_position(approach_result.get_interpolated_plan().position[-1:])
                end_result = motion_gen.plan_single(approach_js, Pose.from_matrix(world_from_ee), plan_config)
                if not end_result.success:
                    _log.error(
                        "Start state:",
                        motion_gen.check_start_state(approach_js),
                        motion_gen.check_constraints(approach_js),
                    )
                    _log.error("cuRobo result status:", end_result.status)
                    visualizer.set_joint_positions(approach_js.position[0])
                    raise RuntimeError(f"Failed to plan from approach to end for {ground_op.name}")

            for result in [retract_result, approach_result, end_result]:
                if result is None:
                    continue
                dt = result.interpolation_dt
                plan = result.get_interpolated_plan()
                accum_plans.append({"type": "trajectory", "plan": plan, "dt": dt})
                last_js = JointState.from_position(plan[-1:].position)
                ts = visualizer.log_joint_trajectory(plan.position, timeline=timeline, start_time=ts, dt=dt)

            # Temporarily monkey patch get_bounding_spheres to return the spheres we sampled
            obstacle = motion_gen.world_model.get_obstacle(obj)
            obstacle.old_get_bounding_spheres = obstacle.get_bounding_spheres

            def get_bounding_spheres(self, *args, **kwargs) -> List[Sphere]:
                spheres = world.get_collision_spheres(obj)
                pts = spheres[:, :3].cpu().numpy()
                n_radius = spheres[:, 3].cpu().numpy()

                obj_pose = Pose.from_list(self.pose, self.tensor_args)
                pre_transform_pose = kwargs["pre_transform_pose"]
                if pre_transform_pose is not None:
                    obj_pose = pre_transform_pose.multiply(obj_pose)  # convert object pose to another frame

                if pts is None or len(pts) == 0:
                    raise ValueError("No points found from the spheres")

                points_cuda = self.tensor_args.to_device(pts)
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

            obstacle.get_bounding_spheres = get_bounding_spheres.__get__(obstacle)

            # Attach the object to the robot
            with timer.time("curobo_planning"):
                motion_gen.attach_objects_to_robot(
                    last_js,
                    object_names=[obj],
                    surface_sphere_radius=0.005,
                    sphere_fit_type=SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE,
                    voxelize_method="subdivide",
                )

            obstacle.get_bounding_spheres = obstacle.old_get_bounding_spheres
            del obstacle.old_get_bounding_spheres

            # Close the gripper in the visualization
            if config.robot == "ur5":
                end_val = 0.4
                interp = torch.linspace(0.0, end_val, 20)
                interp = interp[:, None]
            else:
                end_val = 0.02
                interp = torch.linspace(0.04, end_val, 20)[:, None]
                interp = interp.repeat(1, 2)
            dt = 0.02
            accum_plans.append({"type": "gripper", "action": "close"})

            all_pos = last_js.position.expand(interp.shape[0], -1).cpu()
            all_pos = torch.cat([all_pos, interp], dim=1)
            ts = visualizer.log_joint_trajectory(all_pos, timeline=timeline, start_time=ts, dt=dt)

        # Place
        elif op_name == Place.name:
            obj, grasp, placement, surface, q = ground_op.values
            assert last_js is not None

            with timer.time("curobo_planning"):
                start_js = last_js

                # Plan to retract
                world_from_ee = world.kin_model.get_state(start_js.position).ee_pose.get_matrix()[0]
                world_from_ee_start = world_from_ee
                world_from_retract = world_from_ee @ approach_offset
                retract_result = motion_gen.plan_single(start_js, Pose.from_matrix(world_from_retract), plan_config)
                if not retract_result.success:
                    raise RuntimeError(
                        f"Failed to plan for retract for {ground_op.name}. Status: {retract_result.status}"
                    )

                # Plan from retract to approach
                retract_js = JointState.from_position(retract_result.get_interpolated_plan().position[-1:])
                world_from_obj = action_4dof_to_mat4x4(best_particle[placement].clone())
                if config.grasp_dof == 4:
                    obj_from_grasp = action_4dof_to_mat4x4(best_particle[grasp].clone())
                else:
                    obj_from_grasp = action_6dof_to_mat4x4(best_particle[grasp].clone())
                world_from_grasp = world_from_obj @ obj_from_grasp
                world_from_ee = world_from_grasp @ world.tool_from_ee
                world_from_approach = world_from_ee @ approach_offset
                approach_result = motion_gen.plan_single(retract_js, Pose.from_matrix(world_from_approach), plan_config)
                if not approach_result.success:
                    raise RuntimeError(
                        f"Failed to plan for approach for {ground_op.name}. Status: {approach_result.status}"
                    )

                # Plan from approach to end js
                approach_js = JointState.from_position(approach_result.get_interpolated_plan().position[-1:])
                end_result = motion_gen.plan_single(approach_js, Pose.from_matrix(world_from_ee), plan_config)
                if not end_result.success:
                    raise RuntimeError(
                        f"Failed to plan from approach to end for {ground_op.name}. Status: {end_result.status}"
                    )

            # Compute the offset between the object and end-effector at start of plan
            obj_from_ee = torch.inverse(obj_to_current_pose[obj]) @ world_from_ee_start
            ee_from_obj = torch.inverse(obj_from_ee)

            for result in [retract_result, approach_result, end_result]:
                dt = result.interpolation_dt
                plan = result.get_interpolated_plan()
                accum_plans.append({"type": "trajectory", "plan": plan, "dt": dt})
                last_js = JointState.from_position(plan[-1:].position)

                # Forward kinematics to get end-effector pose
                robot_state = world.kin_model.get_state(plan.position)
                world_from_ee = robot_state.ee_pose.get_matrix()
                world_from_obj = world_from_ee @ ee_from_obj
                ts = visualizer.log_joint_trajectory_with_mat4x4(
                    traj=plan.position,
                    mat4x4_key=f"world/{obj}",
                    mat4x4=world_from_obj,
                    timeline=timeline,
                    start_time=ts,
                    dt=dt,
                )

                # Updated pose is the last pose
                obj_to_current_pose[obj] = world_from_obj[-1]

            # Detach object from robot and enable it again
            with timer.time("curobo_planning"):
                motion_gen.detach_object_from_robot("attached_object")
                motion_gen.world_coll_checker.enable_obstacle(enable=True, name=obj)
                obj_pose = obj_to_current_pose[obj]
                motion_gen.world_collision.update_obstacle_pose(
                    obj, Pose.from_matrix(obj_pose), update_cpu_reference=True
                )

            # Open the gripper for visualization purposes
            if config.robot == "ur5":
                end_val = 0.0
                interp = torch.linspace(0.4, end_val, 20)
                interp = interp[:, None]
            else:
                end_val = 0.04
                interp = torch.linspace(0.02, end_val, 20)[:, None]
                interp = interp.repeat(1, 2)
            dt = 0.02
            accum_plans.append({"type": "gripper", "action": "open"})

            all_pos = last_js.position.expand(interp.shape[0], -1).cpu()
            all_pos = torch.cat([all_pos, interp], dim=1)
            ts = visualizer.log_joint_trajectory(all_pos, timeline=timeline, start_time=ts, dt=dt)

        # Push and PushStick
        elif op_name == Push.name or op_name == PushStick.name:
            # TODO: implement motion solving for these operators
            raise NotImplementedError("Push and PushStick operations are not yet supported in cuRobo motion planning.")

        # Unsupported
        else:
            raise NotImplementedError(f"Unsupported operator {op_name}")

        print(f"{idx + 1}. {ground_op.name}")

    start_js = last_js

    # Plan to retract
    world_from_ee = world.kin_model.get_state(start_js.position).ee_pose.get_matrix()[0]
    world_from_retract = world_from_ee @ approach_offset
    retract_result = motion_gen.plan_single(start_js, Pose.from_matrix(world_from_retract), plan_config)
    if not retract_result.success:
        raise RuntimeError(f"Failed to plan for retract. Status: {retract_result.status}")
    dt = retract_result.interpolation_dt
    plan = retract_result.get_interpolated_plan()
    accum_plans.append({"type": "trajectory", "plan": plan, "dt": dt})
    last_js = JointState.from_position(plan[-1:].position)
    ts = visualizer.log_joint_trajectory(plan.position, timeline=timeline, start_time=ts, dt=dt)

    # Plan to go home at the end which we'll assume is q0
    q_last = last_js.position[0]
    q_home = best_particle["q0"].clone()
    js_last = JointState.from_position(q_last[None])
    js_home = JointState.from_position(q_home[None])
    with timer.time("curobo_planning"):
        result = motion_gen.plan_single_js(js_last, js_home, plan_config)
    if not result.success:
        raise RuntimeError("Failed to plan for going home")

    dt = result.interpolation_dt
    plan = result.get_interpolated_plan()
    accum_plans.append({"type": "trajectory", "plan": plan, "dt": dt})
    _ = visualizer.log_joint_trajectory(plan.position, timeline=timeline, start_time=ts, dt=dt)
    _log.debug("Planned to go home")

    _log.info(f"Motion planning metrics: {timer.get_summary('curobo_planning')}")
    return accum_plans
