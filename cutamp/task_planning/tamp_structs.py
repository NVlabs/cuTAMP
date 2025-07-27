# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from abc import ABC
from dataclasses import dataclass, field
from typing import Sequence, Protocol, List

from cutamp.task_planning import Operator, GroundOperator


@dataclass(frozen=True)
class TAMPOperator(Operator):
    constraints: Sequence["Constraint"] = field(default_factory=list)
    costs: Sequence["Cost"] = field(default_factory=list)

    def ground(self, substitutions: dict[str, str]) -> "GroundTAMPOperator":
        ground_operator: GroundOperator = super().ground(substitutions)
        # Ground constraints and costs
        ground_constraints = [con.ground(substitutions) for con in self.constraints]
        ground_costs = [cost.ground(substitutions) for cost in self.costs]

        ground_tamp_operator = GroundTAMPOperator(
            **ground_operator.__dict__, constraints=ground_constraints, costs=ground_costs
        )
        return ground_tamp_operator


@dataclass(frozen=True)
class GroundTAMPOperator(GroundOperator):
    constraints: Sequence["Constraint"]
    costs: Sequence["Cost"]


PlanSkeleton = List[GroundTAMPOperator]


class Groundable(Protocol):
    def ground(self, substitutions: dict[str, str]) -> "Groundable": ...


class Constraint(ABC, Groundable):
    def __init__(self, *params):
        self.params = params

    @classmethod
    @property
    def type(cls) -> str:
        return cls.__name__

    def ground(self, substitutions: dict[str, str]) -> "Constraint":
        atoms = [substitutions[param.name] for param in self.params]
        return self.__class__(*atoms)

    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}({', '.join(map(str, self.params))})"


class Cost(ABC, Groundable):
    def __init__(self, *params):
        self.params = params

    @classmethod
    @property
    def type(cls) -> str:
        return cls.__name__

    def ground(self, substitutions: dict[str, str]) -> "Cost":
        atoms = [substitutions[param.name] for param in self.params]
        return self.__class__(*atoms)

    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}({', '.join(map(str, self.params))})"
