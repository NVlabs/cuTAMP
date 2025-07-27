# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Sequence

from .base_structs import Atom, Fluent, GroundOperator, Operator, Parameter, State
from .tamp_structs import Constraint, Cost, GroundTAMPOperator, PlanSkeleton, TAMPOperator
from .search import breadth_first_search


def task_plan_generator(
    initial: State,
    goal: State,
    operators: Sequence[Operator],
    explored_state_check: bool = True,
    max_plan_skeletons: int = 99999,
) -> Sequence[PlanSkeleton]:
    """Iterator that yields task plans."""
    plan_iter = breadth_first_search(initial, goal, operators, explored_state_check=explored_state_check)
    for _ in range(max_plan_skeletons):
        try:
            plan = next(plan_iter)
            yield plan
        except StopIteration:
            break
