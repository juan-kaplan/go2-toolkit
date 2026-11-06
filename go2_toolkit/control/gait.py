"""The pipeline, composed. Phase in, joint angles out, and the command around them.

Everything difficult lives in the modules this one calls, and keeping the
composition visible in one place is the point: you can read the whole decoder
here and then go look at whichever stage you care about.

    phase  ->  foot position  ->  joint angles
    phase.py   trajectory.py      kinematics.py

`joint_trajectory` is that chain and nothing else. `state_joint_command` adds `dq`,
`tau_ff` and the gains at an amplitude the caller gives: it takes a stance flag,
progress and rate per leg instead of a phase, and is what the experiments call
through the trial session. Its state comes from `clock_leg_state` for a clock
(the one place the clock enters) or from the decoder for a network.
`clock_joint_command` is the two joined, and `commanded_joint_command` adds the
warmup ramp from the clock, with `commanded_joints` its joint angles alone; both
are kept for the golden tests. `stand_command` is the gait at amplitude zero, and
`stand_joints` its joint angles alone.
"""

from __future__ import annotations

import numpy as np

from ..go2.command import JointCommand
from ..go2.conventions import NUM_JOINTS
from .kinematics import gravity_feedforward_torques, inverse_kinematics, joint_velocities
from .phase import DEFAULT_DUTY, DEFAULT_PERIOD, clock_phases, clock_state
from .trajectory import (
    GaitParams,
    amplitude_scaled,
    foot_positions,
    foot_positions_state,
    foot_velocities_state,
)

# What the robot runs unless told otherwise. `GaitParams` keeps its older
# defaults, which the golden files were recorded with; these are the values the
# tuning harness promoted (6a9ae9e, fd36382): a 0.12 m stride with 0.035 m of
# clearance walks at 0.997 of its implied speed with no foot slip in simulation,
# and kp 180 with kd 6 holds it. Every command line reads them from here.
# fmt: off
DEFAULT_STEP_LENGTH = 0.12    # m per stride
DEFAULT_SWING_HEIGHT = 0.035  # m of peak foot clearance
DEFAULT_HEIGHT = 0.30         # m, hip above the foot at rest
# fmt: on
DEFAULT_KP = 180.0
DEFAULT_KD = 6.0
DEFAULT_WARMUP = 1.0  # s to grow the stride from zero


def default_gait_params(gait: str) -> GaitParams:
    """The promoted parameters for `gait`, with its own duty factor and period."""
    return GaitParams(
        step_length=DEFAULT_STEP_LENGTH,
        swing_height=DEFAULT_SWING_HEIGHT,
        stance_duration=DEFAULT_DUTY[gait],
        nominal_height=DEFAULT_HEIGHT,
        cycle_period=DEFAULT_PERIOD[gait],
    )


def joint_trajectory(phases: np.ndarray, params: GaitParams) -> np.ndarray:
    """Phase per leg straight through to joint angles. The decoder's whole job.

    It does not know or care that phase currently comes from a clock. The
    feedforward that `commanded_joint_command` adds on top is not so indifferent.
    """
    return inverse_kinematics(foot_positions(phases, params))


def stand_command(
    params: GaitParams,
    kp: float | np.ndarray = 100.0,
    kd: float | np.ndarray = 6.0,
    gravity_comp: bool = True,
) -> JointCommand:
    """The gait's own stand as a command: its zero-amplitude pose, gains and load.

    Whatever the phase, a gait at amplitude zero is every foot planted at the
    nominal height, so this needs no phase and no clock. It is what a trial stands
    in before it ramps the stride up and returns to when it ramps down, and it is
    exactly what `state_joint_command` returns at `scale = 0` and no growth, so the
    first loop command adds nothing to it: not a pose, not a gain, and not the
    gravity load, which a stand held with `tau_ff = 0` would pick up in one tick.
    """
    return state_joint_command(
        np.ones(4, dtype=bool),
        np.zeros(4),
        np.zeros(4),
        params,
        scale=0.0,
        kp=kp,
        kd=kd,
        gravity_comp=gravity_comp,
        velocity_feedforward=False,
    )


def stand_joints(params: GaitParams) -> np.ndarray:
    """The gait's own stand: its joint angles at zero amplitude, (12,)."""
    return stand_command(params, gravity_comp=False).q


def warmup_scale(t: float, warmup: float) -> float:
    """Stride amplitude at time `t`, ramping 0 to 1 over `warmup` seconds.

    Separate and named because it is a policy, not arithmetic: the robot is not
    asked to take a full length step out of a standstill. At scale 0 the
    generator reduces to standing still, so the ramp is continuous at both ends.
    """
    if warmup <= 0.0:
        return 1.0
    return min(1.0, max(0.0, t) / warmup)


def warmup_scale_rate(t: float, warmup: float) -> float:
    """Slope of `warmup_scale` at `t`, per second.

    Lives next to `warmup_scale` because it is that ramp's derivative: change the
    shape of one and this has to change with it. The velocity feedforward needs
    it, since a foot moving out along a growing stride has an extra velocity
    term from the growth itself.
    """
    return 1.0 / warmup if warmup > 0.0 and 0.0 <= t < warmup else 0.0


def commanded_joints(
    t: float, offsets: np.ndarray, params: GaitParams, warmup: float = 0.0
) -> np.ndarray:
    """Joint angles at wall clock time `t`, stride warmup included.

    Just the `q` of `commanded_joint_command`, so the two can never send
    different poses. That is the failure this exists to prevent: a driver that
    ramped into the full amplitude phase-zero pose and then began its loop at
    zero amplitude stepped RR by 0.276 rad and FL by 0.214 in one 5 ms period,
    since those two legs sit at the extremes of the stance sweep at phase zero
    under the walk offsets. At kp=100 that asks for 27.6 Nm against a joint
    limit near 23.7, and the leg snaps hard enough to strike the body. Anything
    that needs a pose to ramp towards must take it from here or from the command.
    """
    return commanded_joint_command(
        t, offsets, params, warmup=warmup, gravity_comp=False, velocity_feedforward=False
    ).q


def state_joint_command(
    is_stance: np.ndarray,
    progress: np.ndarray,
    rate: np.ndarray,
    params: GaitParams,
    scale: float = 1.0,
    scale_dot: float = 0.0,
    kp: float | np.ndarray = 100.0,
    kd: float | np.ndarray = 6.0,
    gravity_comp: bool = True,
    velocity_feedforward: bool = True,
) -> JointCommand:
    """Joint command for a foot state, with optional feedforward.

    The state entry point of the command: `is_stance` (bool), `progress` in [0, 1]
    within the current stance or swing, and `rate` in progress per second (0 while
    a foot waits), one value per leg. `dq` comes from that real `rate`, so a
    waiting foot is commanded no velocity, and the gravity mask comes from
    `is_stance`, so a foot parked at the end of its sweep is still loaded.

    `scale` is the amplitude of the stride and `scale_dot` its rate of change.
    When to ramp is the caller's policy (the trial session, or `warmup_scale` for
    the clock path); this only applies the numbers it is given. `params` is the
    full-amplitude gait. At amplitude zero the command is the gait's own stand,
    whatever the state: all four legs share the weight (see the gravity term in
    `gravity_feedforward_torques`) and `dq` is only the growth of the stride.
    The swing curve's touchdown match assumes a swing and stance rate ratio of
    `duty / (1 - duty)` (see `foot_velocities_state`).
    """
    stance = np.asarray(is_stance, dtype=bool)
    live = amplitude_scaled(params, scale)
    q = inverse_kinematics(foot_positions_state(stance, progress, live))

    if velocity_feedforward:
        v_feet = foot_velocities_state(
            stance, progress, rate, live, scale_dot=scale_dot, unscaled_params=params
        )
        dq = joint_velocities(q, v_feet)
    else:
        dq = np.zeros(NUM_JOINTS)

    if gravity_comp:
        # a foot "in swing" at a tiny amplitude has not left the ground
        swing_load = min(1.0, max(0.0, 1.0 - scale))
        tau_ff = gravity_feedforward_torques(q, stance, swing_load=swing_load)
    else:
        tau_ff = np.zeros(NUM_JOINTS)

    kp_arr = np.broadcast_to(np.asarray(kp, dtype=float), (NUM_JOINTS,)).copy()
    kd_arr = np.broadcast_to(np.asarray(kd, dtype=float), (NUM_JOINTS,)).copy()
    return JointCommand(q=q, dq=dq, kp=kp_arr, kd=kd_arr, tau_ff=tau_ff)


def clock_leg_state(
    t: float, offsets: np.ndarray, params: GaitParams
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The clock's `(is_stance, progress, rate)` per leg at time `t`, Unitree order.

    `clock_phases` is where the clock enters the gait and `clock_state` turns that
    phase into the state `state_joint_command` takes: the same three values the
    decoder gives, so a driver logs them whichever one made them.
    """
    phases = clock_phases(t, offsets, params.cycle_period)
    return clock_state(phases, params.stance_duration, params.cycle_period)


def clock_joint_command(
    t: float,
    offsets: np.ndarray,
    params: GaitParams,
    scale: float = 1.0,
    scale_dot: float = 0.0,
    kp: float | np.ndarray = 100.0,
    kd: float | np.ndarray = 6.0,
    gravity_comp: bool = True,
    velocity_feedforward: bool = True,
) -> JointCommand:
    """Joint command at wall clock time `t`, at the amplitude the caller gives.

    The clock path: `clock_leg_state` into `state_joint_command`. The amplitude
    `scale` and its rate `scale_dot` come from the caller, which is the trial
    session: it owns the ramp up at the start and the ramp down at the end,
    whatever produces the phase. The experiments compose the two themselves, to
    log the leg state; this is the same composition in one call.
    """
    stance, progress, rate = clock_leg_state(t, offsets, params)
    return state_joint_command(
        stance,
        progress,
        rate,
        params,
        scale=scale,
        scale_dot=scale_dot,
        kp=kp,
        kd=kd,
        gravity_comp=gravity_comp,
        velocity_feedforward=velocity_feedforward,
    )


def commanded_joint_command(
    t: float,
    offsets: np.ndarray,
    params: GaitParams,
    warmup: float = 0.0,
    kp: float | np.ndarray = 100.0,
    kd: float | np.ndarray = 6.0,
    gravity_comp: bool = True,
    velocity_feedforward: bool = True,
) -> JointCommand:
    """Joint command at wall clock time `t` with the stride warmed up from zero.

    `clock_joint_command` with the amplitude taken from the clock too: the warmup
    ramp over `warmup` seconds. The golden files pin this path bit for bit; the
    trial session ramps the amplitude itself, so the experiments do not call it.
    """
    return clock_joint_command(
        t,
        offsets,
        params,
        scale=warmup_scale(t, warmup),
        scale_dot=warmup_scale_rate(t, warmup),
        kp=kp,
        kd=kd,
        gravity_comp=gravity_comp,
        velocity_feedforward=velocity_feedforward,
    )
