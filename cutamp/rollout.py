# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import List, Dict, TypedDict

import torch
from jaxtyping import Float

from cutamp.utils.common import Particles, action_6dof_to_mat4x4, action_4dof_to_mat4x4
from cutamp.config import TAMPConfiguration
from cutamp.tamp_domain import MoveFree, MoveHolding, Pick, Place, Push, PushStick, Conf
from cutamp.tamp_world import (
    TAMPWorld,
)
from cutamp.task_planning import PlanSkeleton


def get_conf_parameters(plan_skeleton: PlanSkeleton, ignore_initial: bool) -> List[str]:
    """Get the parameters of the plan skeletons that are of type Conf. Returns a list of unique parameter names."""
    conf_params = []
    for ground_op in plan_skeleton:
        conf_idxs = [idx for idx, param in enumerate(ground_op.operator.parameters) if param.type == Conf]
        op_conf_params = [ground_op.values[idx] for idx in conf_idxs]
        conf_params.extend(op_conf_params)
    conf_params = list(dict.fromkeys(conf_params))  # remove duplicates

    if ignore_initial:
        assert conf_params[0] == "q0", "Expected first configuration to be q0"
        conf_params = conf_params[1:]  # remove the initial configuration
    return conf_params


class Rollout(TypedDict):
    """Dict that stores results of a rollout."""

    num_particles: int
    confs: Float[torch.Tensor, "num_particles *h d"]
    conf_params: List[str]
    robot_spheres: Float[torch.Tensor, "num_particles *h 4"]
    world_from_ee: Float[torch.Tensor, "num_particles *h 4 4"]
    world_from_tool_desired: Float[torch.Tensor, "num_particles *h 4 4"]
    world_from_ee_desired: Float[torch.Tensor, "num_particles *h 4 4"]
    gripper_close: List[bool]
    action_params: List[str]
    obj_to_pose: Dict[str, Float[torch.Tensor, "num_particles *h 4 4"]]
    action_to_ts: Dict[str, int]
    action_to_pose_ts: Dict[str, int]
    ts_to_pose_ts: Dict[int, int]


class RolloutFunction:
    """Rollout function that rolls out the plan skeleton. Only supports robot and object kinematics right now."""

    def __init__(self, plan_skeleton: PlanSkeleton, world: TAMPWorld, config: TAMPConfiguration):
        if config.enable_traj:
            raise NotImplementedError("Trajectories are not supported in rollouts yet")
        self.plan_skeleton = plan_skeleton
        self.world = world
        self.config = config
        self.conf_params = get_conf_parameters(plan_skeleton, ignore_initial=True)
        self.obj_to_initial_pose = {obj.name: self.world.get_object_pose(obj) for obj in self.world.movables}

        # Grasp to 4x4 matrix function
        if config.grasp_dof == 4:
            self.grasp_to_mat4x4_fn = action_4dof_to_mat4x4
        elif config.grasp_dof == 6:
            self.grasp_to_mat4x4_fn = action_6dof_to_mat4x4
        else:
            raise ValueError(f"Unsupported {config.grasp_dof=}")

        # Assume placements are 4-DOF
        if config.place_dof != 4:
            raise ValueError(f"Unsupported {config.place_dof=}")

        # Flag for first rollout, used to apply a runtime check
        self._is_first_rollout = True

    def __call__(self, particles: Particles) -> Rollout:
        """
        Rollout particles given the plan skeleton through the world. We keep the rollout information minimal to avoid
        unnecessary computations and backward passes.
        """
        num_particles = particles["q0"].shape[0]

        # Forward kinematics, we use .view() as it's faster than rearrange
        confs = torch.stack([particles[conf] for conf in self.conf_params], dim=1)
        confs_flat = confs.view(-1, confs.shape[-1])
        robot_state = self.world.kin_model.get_state(confs_flat)
        world_from_ee_flat = robot_state.ee_pose.get_matrix()
        world_from_ee = world_from_ee_flat.view(num_particles, confs.shape[1], 4, 4)

        # Robot link spheres for collision checking from cuRobo
        robot_spheres_flat = robot_state.get_link_spheres()
        robot_spheres = robot_spheres_flat.view(num_particles, confs.shape[1], -1, 4)

        # Stores the desired actions
        world_from_tool_desired = []
        gripper_close: List[bool] = []
        action_params: List[str] = []
        action_to_ts: Dict[str, int] = {}

        # For pose timestamp (pose_ts), we only accumulate the poses if the operator causes a change in the object pose.
        # These dicts are used to map actions and timestamps to their corresponding pose timestamps.
        action_to_pose_ts: Dict[str, int] = {}
        ts_to_pose_ts: Dict[int, int] = {}

        # 4x4 transformation matrices for grasp parameters (if any)
        grasp_to_mat4x4: Dict[str, Float[torch.Tensor, "num_particles 4 4"]] = {}

        def get_grasp_mat4x4(grasp_name_: str) -> Float[torch.Tensor, "num_particles 4 4"]:
            if grasp_name_ not in grasp_to_mat4x4:
                grasp_ = particles[grasp_name_]
                grasp_to_mat4x4[grasp_name_] = self.grasp_to_mat4x4_fn(grasp_)
            return grasp_to_mat4x4[grasp_name_]

        # Object poses in world frame for every timestep
        obj_to_pose = {
            obj.name: [self.obj_to_initial_pose[obj.name].expand(num_particles, -1, -1)] for obj in self.world.movables
        }

        def current_pose(obj: str) -> Float[torch.Tensor, "num_particles 4 4"]:
            return obj_to_pose[obj][-1]

        # Timestep of all actionable operators, and the corresponding timestep in the obj_to_pose list
        ts, pose_ts = 0, 0

        # Rollout each ground operator in the plan skeleton
        for ground_op in self.plan_skeleton:
            op_name = ground_op.operator.name

            # Skip MoveFree and MoveHolding as we don't support trajectories yet
            if op_name == MoveFree.name or op_name == MoveHolding.name:
                continue

            # Pick
            elif op_name == Pick.name:
                obj_name, grasp_name, _ = ground_op.values
                # Grasp is in object frame
                obj_from_grasp = get_grasp_mat4x4(grasp_name)

                # Compute tool pose in world frame given object and grasp pose
                world_from_obj = current_pose(obj_name)
                world_from_grasp = world_from_obj @ obj_from_grasp

                world_from_tool_desired.append(world_from_grasp)
                gripper_close.append(True)  # closing gripper at Pick
                action_params.append(grasp_name)
                action_to_ts[grasp_name] = ts
                action_to_pose_ts[grasp_name] = pose_ts

            # Place
            elif op_name == Place.name:
                obj_name, grasp_name, place_name, _, _ = ground_op.values

                # Place is desired object pose in world frame
                place_4dof = particles[place_name]
                world_from_obj = action_4dof_to_mat4x4(place_4dof)

                # Apply the grasp offset to get the tool frame pose
                obj_from_grasp = get_grasp_mat4x4(grasp_name)
                world_from_tool = world_from_obj @ obj_from_grasp

                # Accumulate poses of all the movable objects as we've moved the object
                for obj in self.world.movables:
                    if obj.name == obj_name:
                        obj_to_pose[obj.name].append(world_from_obj)  # use new desired pose
                    else:
                        obj_to_pose[obj.name].append(current_pose(obj.name))
                pose_ts += 1

                world_from_tool_desired.append(world_from_tool)
                gripper_close.append(False)  # opening gripper at Place
                action_params.append(place_name)
                action_to_ts[place_name] = ts
                action_to_pose_ts[place_name] = pose_ts
                ts_to_pose_ts[ts] = pose_ts

            # Push
            elif op_name == Push.name:
                button_name, pose_name, _ = ground_op.values

                # Push pose is desired tool pose
                push_4dof = particles[pose_name]
                world_from_push = action_4dof_to_mat4x4(push_4dof)

                world_from_tool_desired.append(world_from_push)
                gripper_close.append(True)  # close gripper at Push
                action_params.append(pose_name)
                action_to_ts[pose_name] = ts
                action_to_pose_ts[pose_name] = pose_ts

            # PushStick
            elif op_name == PushStick.name:
                button_name, stick_name, grasp_name, pose_name, _ = ground_op.values

                # Pose is desired stick pose
                stick_4dof = particles[pose_name]
                world_from_stick = action_4dof_to_mat4x4(stick_4dof)

                # Apply the grasp offset to get the tool frame pose
                obj_from_grasp = get_grasp_mat4x4(grasp_name)
                world_from_tool = world_from_stick @ obj_from_grasp

                # Accumulate poses of all the movable objects as we've moved and pushing with stick
                for obj in self.world.movables:
                    if obj.name == stick_name:
                        obj_to_pose[obj.name].append(world_from_stick)
                    else:
                        obj_to_pose[obj.name].append(current_pose(obj.name))
                pose_ts += 1

                world_from_tool_desired.append(world_from_tool)
                gripper_close.append(True)  # close gripper at PushStick (it's still closed)
                action_params.append(pose_name)
                action_to_ts[pose_name] = ts
                action_to_pose_ts[pose_name] = pose_ts

            # Unknown
            else:
                raise ValueError(f"Unsupported operator {op_name}")

            # Increment time step
            ts_to_pose_ts[ts] = pose_ts
            ts += 1

        # Stack and store in rollout
        world_from_tool_desired = torch.stack(world_from_tool_desired, dim=1)
        world_from_ee_desired = world_from_tool_desired @ self.world.tool_from_ee

        # Object poses for each timestep
        obj_to_pose = {k: torch.stack(v, dim=1) for k, v in obj_to_pose.items()}

        # Sanity check
        if self._is_first_rollout:
            assert (
                confs.shape[1]
                == world_from_ee.shape[1]
                == world_from_ee_desired.shape[1]
                == world_from_tool_desired.shape[1]
                == ts
            ), "Number of timesteps do not match"
            for obj, poses in obj_to_pose.items():
                assert poses.shape[1] == pose_ts + 1, f"Number of pose timesteps do not match for {obj}"
            self._is_first_rollout = False

        rollout = Rollout(
            num_particles=num_particles,
            confs=confs,
            conf_params=self.conf_params,
            robot_spheres=robot_spheres,
            world_from_ee=world_from_ee,
            world_from_tool_desired=world_from_tool_desired,
            world_from_ee_desired=world_from_ee_desired,
            gripper_close=gripper_close,
            action_params=action_params,
            obj_to_pose=obj_to_pose,
            action_to_ts=action_to_ts,
            action_to_pose_ts=action_to_pose_ts,
            ts_to_pose_ts=ts_to_pose_ts,
        )
        return rollout
