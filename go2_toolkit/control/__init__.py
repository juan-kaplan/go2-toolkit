"""Turning a phase into a pose, split along its natural seams.

    phase.py        where each leg is in its cycle      <- your rhythm source plugs in
    trajectory.py   phase, or stance flag and progress -> foot position in the body frame
    kinematics.py   foot position <-> joint angles
    gait.py         the three composed, phase -> joints

The split follows the one in chvmp/libchamp, which separates `phase_generator`
from `trajectory_planner` from `kinematics` for the same reason: only the first
of them changes when the source of rhythm changes. Replace `phase.clock_phases`
with any phase source; the joint angles need nothing more, but the velocity
feedforward also assumes the clock's rate (see `phase.py`).

Everything here depends on numpy and on `go2` for the robot's dimensions, and on
nothing else. No transport, no clips, no plotting. `tests/test_imports.py`
enforces that, because it is one convenience import away from being false.

The names below are the pipeline's public surface. Reach into the submodules
when you want a stage rather than the whole thing.
"""

from __future__ import annotations

from ..go2.command import JointCommand
from . import gait, kinematics, phase, trajectory
from .gait import (
    DEFAULT_HEIGHT,
    DEFAULT_KD,
    DEFAULT_KP,
    DEFAULT_STEP_LENGTH,
    DEFAULT_SWING_HEIGHT,
    DEFAULT_WARMUP,
    clock_joint_command,
    clock_leg_state,
    commanded_joint_command,
    commanded_joints,
    default_gait_params,
    joint_trajectory,
    stand_command,
    stand_joints,
    state_joint_command,
    warmup_scale,
    warmup_scale_rate,
)
from .kinematics import (
    forward_kinematics,
    gravity_feedforward_torques,
    inverse_kinematics,
    joint_velocities,
    leg_jacobian,
    link_positions,
)
from .phase import (
    DEFAULT_DUTY,
    DEFAULT_PERIOD,
    GAITS,
    clock_phases,
    clock_state,
    phase_count,
    stance_spans,
    support_count,
)
from .trajectory import (
    GaitParams,
    amplitude_scaled,
    foot_offsets,
    foot_offsets_state,
    foot_positions,
    foot_positions_state,
    foot_velocities,
    foot_velocities_state,
    swing_control_points,
)

__all__ = [
    "DEFAULT_DUTY",
    "DEFAULT_HEIGHT",
    "DEFAULT_KD",
    "DEFAULT_KP",
    "DEFAULT_PERIOD",
    "DEFAULT_STEP_LENGTH",
    "DEFAULT_SWING_HEIGHT",
    "DEFAULT_WARMUP",
    "GAITS",
    "GaitParams",
    "JointCommand",
    "amplitude_scaled",
    "clock_joint_command",
    "clock_leg_state",
    "clock_phases",
    "clock_state",
    "commanded_joint_command",
    "commanded_joints",
    "default_gait_params",
    "foot_offsets",
    "foot_offsets_state",
    "foot_positions",
    "foot_positions_state",
    "foot_velocities",
    "foot_velocities_state",
    "forward_kinematics",
    "gait",
    "gravity_feedforward_torques",
    "inverse_kinematics",
    "joint_trajectory",
    "joint_velocities",
    "kinematics",
    "leg_jacobian",
    "link_positions",
    "phase",
    "phase_count",
    "stance_spans",
    "stand_command",
    "stand_joints",
    "state_joint_command",
    "support_count",
    "swing_control_points",
    "trajectory",
    "warmup_scale",
    "warmup_scale_rate",
]
