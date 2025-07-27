# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from cutamp.task_planning import Cost


class GraspCost(Cost):
    def __init__(self, obj, grasp):
        super().__init__(obj, grasp)


class TrajectoryLength(Cost):
    def __init__(self, q_start, traj, q_end):
        super().__init__(q_start, traj, q_end)
