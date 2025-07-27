# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Domain file like in PDDL. The task planner can be improved, but it suffices for now.
"""

from typing import Sequence

from cutamp.task_planning import Fluent, Parameter, TAMPOperator, State
from cutamp.task_planning.constraints import (
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

# Types
Conf = "conf"
Traj = "traj"
Pose = "pose"
Grasp = "grasp"

Movable = "movable"
Surface = "surface"
Button = "button"

# Fluents (aka predicates)
At = Fluent("At", [Parameter("q", Conf)])
HandEmpty = Fluent("HandEmpty")
CanMove = Fluent("CanMove")
JustMoved = Fluent("JustMoved")
Holding = Fluent("Holding", [Parameter("obj", Movable)])
HoldingWithGrasp = Fluent("HoldingWithGrasp", [Parameter("obj", Movable), Parameter("grasp", Grasp)])
ButtonPushed = Fluent("ButtonPushed", [Parameter("button", "button")])
PushedWithStick = Fluent("PushedWithStick", [Parameter("button", "button"), Parameter("obj", Movable)])
CanPush = Fluent("CanPush", [Parameter("button", "button")])
IsMovable = Fluent("IsMovable", [Parameter("obj", Movable)])
IsButton = Fluent("IsButton", [Parameter("button", Button)])
IsSurface = Fluent("IsSurface", [Parameter("surface", Surface)])
IsStick = Fluent("IsStick", [Parameter("obj", Movable)])
HasNotPickedUp = Fluent("HasNotPickedUp", [Parameter("obj", Movable)])
On = Fluent("On", [Parameter("obj", Movable), Parameter("surface", Surface)])

all_tamp_fluents = [
    At,
    HandEmpty,
    CanMove,
    JustMoved,
    Holding,
    HoldingWithGrasp,
    ButtonPushed,
    PushedWithStick,
    CanPush,
    IsMovable,
    IsButton,
    IsSurface,
    IsStick,
    HasNotPickedUp,
    On,
]


# Parameters - used for naming in operators, so it's easier to read when debugging
q = Parameter("q", Conf)
q_start = Parameter("q_start", Conf)
q_end = Parameter("q_end", Conf)
traj = Parameter("traj", Traj)

obj = Parameter("obj", Movable)
button = Parameter("button", Button)
surface = Parameter("surface", Surface)

grasp = Parameter("grasp", Grasp)
pose = Parameter("pose", Pose)
placement = Parameter("placement", Pose)


# Operators - this is the important part!
MoveFree = TAMPOperator(
    "MoveFree",
    [q_start, traj, q_end],
    preconditions=[At(q_start), HandEmpty(), CanMove()],
    add_effects=[At(q_end), JustMoved()],
    del_effects=[At(q_start), CanMove()],
    constraints=[CollisionFree(q_start, traj, q_end), Motion(q_start, traj, q_end)],
    costs=[TrajectoryLength(q_start, traj, q_end)],
)


MoveHolding = TAMPOperator(
    "MoveHolding",
    [obj, grasp, q_start, traj, q_end],
    preconditions=[
        At(q_start),
        Holding(obj),
        HoldingWithGrasp(obj, grasp),
        CanMove(),
    ],
    add_effects=[At(q_end), JustMoved()],
    del_effects=[At(q_start), CanMove()],
    constraints=[CollisionFreeHolding(obj, grasp, q_start, traj, q_end), Motion(q_start, traj, q_end)],
    costs=[TrajectoryLength(q_start, traj, q_end)],
)


Pick = TAMPOperator(
    "Pick",
    [obj, grasp, q],
    preconditions=[
        At(q),
        HandEmpty(),
        IsMovable(obj),
        JustMoved(),
        HasNotPickedUp(obj),
    ],
    add_effects=[
        Holding(obj),
        HoldingWithGrasp(obj, grasp),
        CanMove(),
    ],
    del_effects=[HandEmpty(), JustMoved(), HasNotPickedUp(obj)],
    constraints=[KinematicConstraint(q, grasp), CollisionFreeGrasp(obj, grasp)],
    costs=[GraspCost(obj, grasp)],
)

Place = TAMPOperator(
    "Place",
    [obj, grasp, placement, surface, q],
    preconditions=[
        At(q),
        Holding(obj),
        HoldingWithGrasp(obj, grasp),
        IsSurface(surface),
        JustMoved(),
    ],
    add_effects=[HandEmpty(), CanMove(), On(obj, surface)],
    del_effects=[
        Holding(obj),
        HoldingWithGrasp(obj, grasp),
        JustMoved(),
    ],
    constraints=[
        KinematicConstraint(q, placement),
        StablePlacement(obj, grasp, placement, surface),
        CollisionFreePlacement(obj, placement, surface),
    ],
    costs=[],
)


Push = TAMPOperator(
    "Push",
    [button, pose, q],
    preconditions=[
        At(q),
        IsButton(button),
        HandEmpty(),
        CanPush(button),
        JustMoved(),
    ],
    add_effects=[ButtonPushed(button), CanMove()],
    del_effects=[JustMoved(), CanPush(button)],  # Note: CFree for Push already encoded in the Move operator
    constraints=[KinematicConstraint(q, pose), ValidPush(button, pose)],
    costs=[],
)


PushStick = TAMPOperator(
    "PushStick",
    [button, obj, grasp, pose, q],
    preconditions=[
        At(q),
        Holding(obj),
        HoldingWithGrasp(obj, grasp),
        IsButton(button),
        IsStick(obj),
        CanPush(button),
        JustMoved(),
    ],
    add_effects=[ButtonPushed(button), CanMove(), PushedWithStick(button, obj)],
    del_effects=[JustMoved(), CanPush(button)],
    # CFree is automatically handled right now within the operator
    constraints=[KinematicConstraint(q, pose), ValidPushStick(button, obj, pose)],
    costs=[],
)

all_tamp_operators = [MoveFree, MoveHolding, Pick, Place, Push, PushStick]


def get_initial_state(
    movables: Sequence[str] = (), surfaces: Sequence[str] = (), sticks: Sequence[str] = (), buttons: Sequence[str] = ()
) -> State:
    """Ground the initial state of the TAMP domain."""
    initial_state = {At.ground("q0"), HandEmpty.ground(), CanMove.ground()}
    for movable in movables:
        initial_state.add(IsMovable.ground(movable))
        initial_state.add(HasNotPickedUp.ground(movable))

    for surface in surfaces:
        initial_state.add(IsSurface.ground(surface))

    for stick in sticks:
        initial_state.add(IsStick.ground(stick))
        initial_state.add(IsMovable.ground(stick))
        initial_state.add(HasNotPickedUp.ground(stick))

    for button in buttons:
        initial_state.add(IsButton.ground(button))
        initial_state.add(CanPush.ground(button))

    initial_state = frozenset(initial_state)
    return initial_state
