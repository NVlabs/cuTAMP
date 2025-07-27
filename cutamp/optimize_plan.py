# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import logging
import time
from typing import Tuple, TypedDict

import torch
from torch.optim import Adam
from tqdm import tqdm

from cutamp.utils.common import Particles
from cutamp.config import TAMPConfiguration
from cutamp.constraint_checker import ConstraintChecker
from cutamp.cost_function import CostFunction
from cutamp.cost_reduction import CostReducer
from cutamp.rollout import RolloutFunction
from cutamp.tamp_domain import Conf, Grasp, Pose, Traj
from cutamp.task_planning import PlanSkeleton
from cutamp.utils.timer import TorchTimer
from cutamp.utils.visualizer import Visualizer

_log = logging.getLogger(__name__)
_known_types = {Conf, Grasp, Pose, Traj}


class PlanContainer(TypedDict):
    plan_skeleton: PlanSkeleton
    particles: Particles
    rollout_fn: RolloutFunction
    cost_fn: CostFunction
    heuristic: float


class ParticleOptimizer:
    """Attempts to optimize particles for a plan skeleton."""

    def __init__(self, config: TAMPConfiguration, cost_reducer: CostReducer, constraint_checker: ConstraintChecker):
        self.config = config
        self.cost_reducer = cost_reducer
        self.constraint_checker = constraint_checker
        self.get_satisfying_mask = self.constraint_checker.get_mask

        # Types to optimize
        self.types_to_optimize = {Pose, Conf}
        self.opt_counter = 0

    def __call__(self, plan_info: PlanContainer, timer: TorchTimer, visualizer: Visualizer) -> Tuple[bool, dict, bool]:
        """
        Optimize the particles for the given plan skeleton in the plan_info container.
        Returns a tuple of (whether a solution was found, optimization metrics, time exceeded flag).
        """
        plan_skeleton = plan_info["plan_skeleton"]
        particles = plan_info["particles"]
        rollout_fn = plan_info["rollout_fn"]
        cost_fn = plan_info["cost_fn"]
        opt_metrics = {
            "plan_skeleton": [str(op) for op in plan_skeleton],
            "num_particles": self.config.num_particles,
            "opt_params": {},
        }

        # Create parameter groups for the optimizer
        timer.start("setup_optimizer")
        param_to_type = {}
        for ground_op in plan_skeleton:
            param_names = ground_op.values
            types = [param.type for param in ground_op.operator.parameters]
            assert len(param_names) == len(types)
            for param_name, param_type in zip(param_names, types):
                param_to_type[param_name] = param_type

        param_groups = []
        param_msg = []
        for param, val in particles.items():
            param_type = param_to_type[param]
            if param_type not in self.types_to_optimize:
                continue
            if param == "q0":
                continue  # we don't optimize the initial configuration
            param_msg.append(f"(param: {param} = {tuple(val.shape)})")
            group = {"params": val}
            if param_type == Conf:
                group["lr"] = self.config.conf_lr
            param_groups.append(group)
            opt_metrics["opt_params"][param] = {"type": param_type, "shape": list(val.shape)}
            val.requires_grad = True

        if not param_groups:
            raise RuntimeError(f"No parameters to optimize! For plan skeleton: {plan_skeleton}")

        # Setup optimizer
        optimizer = Adam(params=param_groups, lr=self.config.lr)
        num_total_params = sum(p.numel() for group in optimizer.param_groups for p in group["params"])
        opt_metrics["optimizer"] = {
            "optimizer_cls": optimizer.__class__.__name__,
            "lr": self.config.lr,
            "conf_lr": self.config.conf_lr,
            "num_params": num_total_params,
        }
        _log.debug(f"Optimizing {num_total_params} parameters, {', '.join(param_msg)}")
        total_dof = sum(p.shape[-1] for group in optimizer.param_groups for p in group["params"])
        _log.debug(f"Total DOF for each particle: {total_dof}")

        indices = torch.arange(self.config.num_particles, device="cuda")
        consider_types = {"constraint"}
        if self.config.optimize_soft_costs:
            consider_types.add("cost")
        found_solution = False
        timer.stop("setup_optimizer")

        # Setup some more metrics
        opt_metrics["num_satisfying"] = []
        opt_metrics["loss"] = []
        if self.config.soft_cost is not None:
            opt_metrics["best_soft_costs"] = []
        opt_metrics["elapsed"] = []

        vis_every = self.config.opt_viz_interval
        num_steps = self.config.num_opt_steps
        opt_metrics["num_steps"] = num_steps

        # Now we optimize!
        pbar = tqdm(range(num_steps))
        start_time = time.perf_counter()
        time_exceeded = False
        for step in pbar:
            if self.config.max_loop_dur is not None and timer.elapsed("start_optimization") >= self.config.max_loop_dur:
                _log.warning(f"Optimization exceeded max time of {self.config.max_loop_dur:.2f}s")
                time_exceeded = True
                break
            timer.start("optimization_step")
            optimizer.zero_grad()

            rollout = rollout_fn(particles)
            cost_dict = cost_fn(rollout)
            costs = self.cost_reducer(cost_dict, consider_types=consider_types)
            satisfying_mask = self.get_satisfying_mask(cost_dict, verbose=False)
            num_satisfying = satisfying_mask.sum().item()
            opt_metrics["num_satisfying"].append(num_satisfying)

            if num_satisfying > 0 and not found_solution:
                if timer.has_timer("first_solution"):
                    time_to_first_sol = timer.stop("first_solution")
                    _log.info(f"Found first solution in {time_to_first_sol:.2f}s after sampling plans")

                torch.cuda.synchronize()
                opt_metrics["opt_start_to_first_sol"] = time.perf_counter() - start_time
                found_solution = True

            loss = costs.mean()
            opt_metrics["loss"].append(loss.item())

            # Compute soft costs if required (even if we don't optimize them)
            if self.config.soft_cost is not None:
                soft_costs = self.cost_reducer.soft_costs(cost_dict)
                best_soft_cost = soft_costs[satisfying_mask].min().item() if num_satisfying > 0 else None
                opt_metrics["best_soft_costs"].append(best_soft_cost)

            torch.cuda.synchronize()
            opt_metrics["elapsed"].append(time.perf_counter() - start_time)

            # Visualize the optimization progress. We do this before stepping the optimizer to see current state.
            if step % vis_every == 0 or step == num_steps - 1:
                timer.start("visualize_opt_rollout")
                # Visualize satisfying particles if we have any. Otherwise, just show the best particle so far.
                if num_satisfying > 0:
                    best_satisfying_idx = costs[satisfying_mask].argmin()
                    best_idx = indices[satisfying_mask][best_satisfying_idx]
                else:
                    best_idx = costs.argmin()

                # Show last state after rolling out the best particle
                visualizer.set_time_sequence(f"opt_{self.opt_counter}", step)
                q_last = rollout["confs"][best_idx, -1].tolist()
                visualizer.set_joint_positions(q_last)
                for obj in rollout["obj_to_pose"]:
                    mat4x4_last = rollout["obj_to_pose"][obj][best_idx, -1]
                    visualizer.log_mat4x4(f"world/{obj}", mat4x4_last)

                visualizer.log_cost_dict(cost_dict)
                visualizer.log_scalar("loss", loss.item())
                timer.stop("visualize_opt_rollout")

            # Compute gradients and step the optimizer
            loss.backward()
            optimizer.step()
            timer.stop("optimization_step")
            pbar.set_description(
                f"Loss: {loss:.3f}, Min: {costs.min():.3f}, {num_satisfying}/{self.config.num_particles} satisfying"
            )

        # Now we've finished the optimization loop. Check the satisfying particles and compute soft/hard costs.
        with torch.no_grad():
            rollout = rollout_fn(particles)
            cost_dict = cost_fn(rollout)
            costs = self.cost_reducer(cost_dict, consider_types=consider_types)
            soft_costs = self.cost_reducer.soft_costs(cost_dict)
            hard_costs = self.cost_reducer.hard_costs(cost_dict)
            satisfying_mask = self.get_satisfying_mask(cost_dict, verbose=True)
        num_satisfying = satisfying_mask.sum().item()
        _log.info(f"{num_satisfying}/{self.config.num_particles} satisfying after optimization")
        opt_metrics["num_satisfying_final"] = num_satisfying

        # Get the best costs from the satisfying particles. Note cost itself includes soft and hard (i.e., constraints)
        # costs if config.optimize_soft_costs is True, while it only includes hard costs otherwise.
        best_cost = costs[satisfying_mask].min().item() if num_satisfying > 0 else None
        best_soft_cost = soft_costs[satisfying_mask].min().item() if num_satisfying > 0 else None
        best_hard_cost = hard_costs[satisfying_mask].min().item() if num_satisfying > 0 else None
        if best_cost is not None:
            _log.info(
                f"Best particle cost: {best_cost:04f}, best soft cost: {best_soft_cost:04f}, "
                f"best hard cost: {best_hard_cost:04f}"
            )
        else:
            _log.info("No satisfying particles found at end of optimization")
        opt_metrics["best_cost"] = best_cost
        opt_metrics["best_soft_cost"] = best_soft_cost
        opt_metrics["best_hard_cost"] = best_hard_cost

        # Visualize rollout of the best particle in terms of soft costs, which is just trajectory length if no
        # other soft costs are specified. If no satisfying particles, just visualize lowest overall cost particle.
        if num_satisfying > 0:
            best_satisfying_idx = soft_costs[satisfying_mask].argmin()
            best_idx = indices[satisfying_mask][best_satisfying_idx]
        else:
            best_idx = costs.argmin()

        # Log initial state
        timer.start("visualize_rollout")
        world = rollout_fn.world
        visualizer.set_time_sequence(f"rollout_{self.opt_counter}", 0)
        visualizer.set_joint_positions(world.q_init.tolist())
        for obj in world.movables:
            obj_pose = world.get_object_pose(obj).cpu()
            visualizer.log_mat4x4(f"world/{obj.name}", obj_pose)

        for ts in range(len(rollout["conf_params"])):
            visualizer.set_time_sequence(f"rollout_{self.opt_counter}", ts + 1)
            q = rollout["confs"][best_idx, ts]

            gripper_close = rollout["gripper_close"][ts]
            if self.config.robot == "ur5":
                gripper_joints = [0.4] if gripper_close else [0.0]
            elif self.config.robot == "panda":
                gripper_joints = [0.01, 0.01] if gripper_close else [0.04, 0.04]
            else:
                gripper_joints = []
            visualizer.set_joint_positions(q.tolist() + gripper_joints)

            world_from_ee = rollout["world_from_ee"][best_idx, ts].cpu()
            visualizer.log_mat4x4("rollout/ee_pose", world_from_ee)

            robot_spheres = rollout["robot_spheres"][best_idx, ts].cpu()
            visualizer.log_spheres("rollout/robot_spheres", robot_spheres)

            pose_ts = rollout["ts_to_pose_ts"][ts]
            for obj in world.movables:
                obj_pose = rollout["obj_to_pose"][obj.name][best_idx, pose_ts].cpu()
                visualizer.log_mat4x4(f"world/{obj.name}", obj_pose)
        timer.stop("visualize_rollout")

        self.opt_counter += 1
        return num_satisfying > 0, opt_metrics, time_exceeded
