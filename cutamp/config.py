# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass(frozen=True)
class TAMPConfiguration:
    # Number of particles to initialize and optimize over
    num_particles: int = 1024

    # Robot embodiment to use
    robot: Literal["panda", "ur5"] = "panda"

    # Grasp and Placements
    grasp_dof: Literal[4, 6] = 4
    place_dof: Literal[4] = 4

    # Approach to use. Note: optimization includes particle initialization (i.e., sampling)
    approach: Literal["optimization", "sampling"] = "optimization"

    # Number of resampling attempts per plan skeleton if the approach is sampling
    num_resampling_attempts: int = 0

    # Optimization hyperparams
    num_opt_steps: int = 1_000
    lr: float = 7e-3  # default LR for optimizer
    conf_lr: float = 2.226e-2  # LR for robot configurations

    ## Advanced args - for soft cost experiments. Warning! Might cause unexpected behavior if not used correctly.
    # Maximum time for optimization or sampling in seconds before breaking
    max_loop_dur: Optional[float] = None
    # Whether to break upon finding a satisfying particle
    break_on_satisfying: bool = True
    # Whether we're running stick button experiment. Modifies heuristic for comparing baselines
    stick_button_experiment: bool = False

    ## Soft Costs
    optimize_soft_costs: bool = False
    # Supported: dist_from_origin, max_obj_dist, min_obj_dist, min_y, max_y, align_yaw
    soft_cost: Optional[str] = None

    ## Task Planning and subgraph caching
    # Number of initial plans to sample
    num_initial_plans: int = 30
    # Whether to check if state has been explored before adding to search tree for task planning
    explored_state_check: bool = True
    # Cache particles for reuse - this is coupled to our current domains so the implementation is not general
    # Warning: don't use this with the sampling approach, as the samplers will just return the same samples
    cache_subgraphs: bool = False
    skip_failed_subgraphs: bool = False
    # Random particle initialization for placements and robot configurations. Not supported for all domains.
    random_init: bool = False

    ## Collision Checking
    # Movable object collision sphere representation
    coll_n_spheres: int = 50
    coll_sphere_radius: float = 0.005
    # Distance at which collision checking is activated between the world (in cuRobo)
    world_activation_distance: float = 0.0
    # Distance at which collision checking is activated between gripper and movables
    gripper_activation_distance: float = 0.0
    # Distance at which collision checking is activated between movables and movables
    movable_activation_distance: float = 0.0

    ## Trajectories and cuRobo
    # Whether to also optimize full trajectories (not supported right now)
    enable_traj: bool = False
    # Motion plan with cuRobo after optimization
    curobo_plan: bool = False
    # For slowing down cuRobo motion plans (0.5 is safe on the real robot)
    time_dilation_factor: Optional[float] = None
    # Whether to warmup IK solver
    warmup_ik: bool = True
    # Whether to warmup motion generator
    warmup_motion_gen: bool = True

    ## Visualizer Args
    # Whether to use visualizer, if set to False a Mock is used
    enable_visualizer: bool = True
    # Number of steps between visualizations of optimization state (note: visualization takes non-trivial time)
    opt_viz_interval: int = 10
    # Whether to visualize the robot mesh, set to False if you want to save network bandwidth (~10MB)
    viz_robot_mesh: bool = True
    # Spawn the rerun visualizer
    rr_spawn: bool = True

    ## Logging Args
    enable_experiment_logging: bool = True
    # Root directory for logging experiments
    experiment_root: str = "/tmp/cutamp-experiments"


def validate_tamp_config(config: TAMPConfiguration):
    if config.num_particles <= 0:
        raise ValueError(f"num_particles must be positive, not {config.num_particles}")
    if config.robot not in {"panda", "ur5"}:
        raise ValueError(f"Invalid embodiment: {config.robot}")
    if config.grasp_dof not in {4, 6}:
        raise ValueError(f"Invalid grasp_dof: {config.grasp_dof}")
    if config.place_dof not in {4}:
        raise ValueError(f"Invalid place_dof: {config.place_dof}")
    if config.approach not in {"optimization", "sampling"}:
        raise ValueError(f"Invalid approach: {config.approach}")
    if config.num_resampling_attempts < 0:
        raise ValueError(f"num_resampling_attempts must be non-negative, not {config.num_resampling_attempts}")

    # Optimization hyperparams
    if config.num_opt_steps <= 0:
        raise ValueError(f"num_opt_steps must be positive, not {config.num_opt_steps}")
    if config.lr <= 0:
        raise ValueError(f"Learning rate (lr) must be positive, not {config.lr}")
    if config.conf_lr <= 0:
        raise ValueError(f"Configuration learning rate (conf_lr) must be positive, not {config.conf_lr}")

    # Advanced args
    if config.max_loop_dur is not None and config.max_loop_dur <= 0:
        raise ValueError(f"max_loop_dur must be positive or None, not {config.max_loop_dur}")

    # Task Planning and subgraph caching
    if config.num_initial_plans <= 0:
        raise ValueError(f"num_initial_plans must be positive, not {config.num_initial_plans}")
    if config.cache_subgraphs and config.approach == "sampling":
        raise ValueError("cache_subgraphs is not compatible with sampling approach")

    # Collision checking
    if config.coll_n_spheres <= 0:
        raise ValueError(f"coll_n_spheres must be positive, not {config.coll_n_spheres}")
    if config.coll_sphere_radius <= 0:
        raise ValueError(f"coll_sphere_radius must be positive, not {config.coll_sphere_radius}")
    if config.world_activation_distance < 0:
        raise ValueError(f"world_activation_distance must be non-negative, not {config.world_activation_distance}")
    if config.movable_activation_distance < 0:
        raise ValueError(f"movable_activation_distance must be non-negative, not {config.movable_activation_distance}")
